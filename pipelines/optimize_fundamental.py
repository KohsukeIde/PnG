import os
import sys
import pickle
import torch
import numpy as np
import cv2

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

# pickle 内の 'twodgs' モジュールを src.primitive.twod_gaussians_rs にマッピング
sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

from utils.homography_pipeline_visualization import (
    visualize_point_matches,
    visualize_epipolar_lines,
    plot_epipolar_cost_change
    # save_warped_image  # Fundamental optimizationでは使わない
)

def get_top_correspondences_fundamental(solver, num_points=100):
    """
    Fundamental 行列 F を用いたコスト行列を計算し、
    Sinkhorn / Unbalanced Sinkhorn から上位の対応点を貪欲に取り出す。
    """
    with torch.no_grad():
        # Fundamental 用のコスト行列を計算
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
    
    T_np = transport_matrix.cpu().numpy()
    points1 = solver.means1.cpu().numpy()
    points2 = solver.means2.cpu().numpy()
    
    print(f"Transport matrix shape: {T_np.shape}")
    print(f"Transport matrix min value: {T_np.min()}")
    print(f"Transport matrix max value: {T_np.max()}")

    # Greedy 1:1 matching based on highest transport values
    used_rows = set()
    used_cols = set()
    matches = []

    flat_indices = np.argsort(-T_np.flatten())  # 降順ソート
    rows, cols = np.unravel_index(flat_indices, T_np.shape)
    
    for row, col in zip(rows, cols):
        if len(matches) >= num_points:
            break
        if row not in used_rows and col not in used_cols:
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
    
    if len(matches) == 0:
        print("No matches found.")
        return np.zeros((0, 2)), np.zeros((0, 2))

    matched_rows, matched_cols = zip(*matches)
    matched_rows = np.array(matched_rows)
    matched_cols = np.array(matched_cols)

    pts1 = points1[matched_rows]
    pts2 = points2[matched_cols]

    # Print matching statistics
    print(f"\nMatching statistics:")
    print(f"Number of matches found: {len(matches)}")
    print(f"Top 5 transport values for matches:")
    for i in range(min(5, len(matches))):
        row, col = matches[i]
        print(f"Match {i+1}: T[{row},{col}] = {T_np[row,col]}")

    return pts1, pts2

def compute_epipolar_cost_cv2(F, pts1, pts2):
    """Compute the average epipolar constraint residuals for given correspondences.
       epipolar residual = mean( |x2^T F x1| ).
    """
    if F is None or F.shape != (3, 3):
        return float('inf')
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    Fx1 = F @ pts1_h.T
    x2Fx1 = np.sum(pts2_h * Fx1.T, axis=1)
    residuals = np.abs(x2Fx1)
    return np.mean(residuals)

def main():
    # 1) Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Data paths
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/train_results_scan63_32_masked'
    gaussians1_path = os.path.join(data_dir_gmm, '0022_fitted_gaussians.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, '0023_fitted_gaussians.pkl')
    colmap_dir = os.path.join(data_dir, 'sparse/0')

    image1_name = '0022.png'
    image2_name = '0023.png'

    # 3) Load Gaussians
    _, gaussians1, _, _ = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, _ = load_gaussians_torch(gaussians2_path, device)

    # 4) Load camera information from COLMAP
    cameras = load_cameras_from_colmap(colmap_dir)
    images_data = load_images_from_colmap(colmap_dir)

    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
    image1_id = image_name_to_id[image1_name]
    image2_id = image_name_to_id[image2_name]

    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']
    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)

    K1 = camera1.K
    K2 = camera2.K

    # 5) Initialize Solver
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.01,
        lambda_mean=3.0,
        lambda_cov=1.0,
        lambda_color=0.0, # 一旦無効化
        lambda_epipolar=1e-4, 
        device=device
    )

    solver.f = torch.eye(3, device=device, dtype=torch.float32)
    def print_stats(tensor, name):
        print(f"\n{name} statistics:")
        print(f"  Min: {tensor.min().item()}")
        print(f"  Max: {tensor.max().item()}")
        print(f"  Mean: {tensor.mean().item()}")
        print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"  Has Inf: {torch.isinf(tensor).any().item()}")

    # 6) Before Optimization
    print("\n--- Optimization Before ---")
    
    # 輸送行列が最大のペアを抽出
    pts1_before, pts2_before = get_top_correspondences_fundamental(solver, num_points=1000)

    # RANSAC で F を推定
    F_before, mask_before = cv2.findFundamentalMat(
        pts1_before.astype(np.float32),
        pts2_before.astype(np.float32),
        cv2.FM_RANSAC
    )
    if F_before is not None and F_before.shape == (3, 3):
        print("\nEstimated Fundamental Matrix (RANSAC) Before Optimization:")
        print(F_before)
    else:
        print("Failed to estimate Fundamental Matrix before optimization.")
        F_before = np.eye(3, dtype=np.float32)

    epipolar_cost_before = compute_epipolar_cost_cv2(F_before, pts1_before, pts2_before)
    print(f"Epipolar Cost Before Optimization: {epipolar_cost_before:.6f}")

    # 7) Optimize with Fundamental
    print("\n--- Optimizing Fundamental Matrix ---")
    
    # 初期値としてRansacで推定したFを使用
    # F_torch = torch.from_numpy(F_before).float().to(device)
    # solver.f = F_torch

    solver.optimize_with_fundamental(max_iter=1000, tol=1e-4)

    # ここで最適化された solver.f を取得
    F_optimized = solver.f.detach().cpu().numpy()
    print("\nOptimized Fundamental matrix:")
    print(F_optimized)

    # コスト行列を確認
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        print_stats(cost_matrix, "Cost Matrix (after optimization)")
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        print_stats(transport_matrix, "Transport Matrix (after optimization)")

        cost_matrix = cost_matrix.cpu().numpy()
        transport_matrix = transport_matrix.cpu().numpy()

    # 8) After Optimization
    print("\n--- Optimization After ---")

    # 更新された輸送行列をもとに対応点を抽出
    pts1_after, pts2_after = get_top_correspondences_fundamental(solver, num_points=5000)

    F_after, mask_after = cv2.findFundamentalMat(
        pts1_after.astype(np.float32),
        pts2_after.astype(np.float32),
        cv2.FM_RANSAC
    )
    if F_after is not None and F_after.shape == (3, 3):
        print("\nEstimated Fundamental Matrix (RANSAC) After Optimization:")
        print(F_after)
    else:
        print("Failed to estimate Fundamental Matrix after optimization.")
        F_after = np.eye(3, dtype=np.float32)
        

    epipolar_cost_after = compute_epipolar_cost_cv2(F_optimized, pts1_after, pts2_after)
    print(f"Epipolar Cost After Optimization: {epipolar_cost_after:.6f}")

    # 9) Plot epipolar cost change
    plot_epipolar_cost_change(epipolar_cost_before, epipolar_cost_after, output_dir='results')

    # 10) Visualize epipolar lines and corresponding points
    image1_path = os.path.join(data_dir, 'images', image1_name)
    image2_path = os.path.join(data_dir, 'images', image2_name)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Failed to load images for visualization.")
        sys.exit(1)

    # - Before
    # Ransacで推定したFを可視化
    visualize_epipolar_lines(img1, img2, pts1_before, pts2_before, F_before, output_dir='results/epilines_before_ransac')
    visualize_point_matches(img1, img2, pts1_before, pts2_before, output_dir='results/matches_before_ransac')

    # - After
    # 最適化後のFを可視化
    visualize_epipolar_lines(img1, img2, pts1_after, pts2_after, F_optimized, output_dir='results/epilines_after_optimized')
    visualize_point_matches(img1, img2, pts1_after, pts2_after, output_dir='results/matches_after_optimized')
    # Ransacで推定したFを可視化
    visualize_epipolar_lines(img1, img2, pts1_after, pts2_after, F_after, output_dir='results/epilines_after_ransac')
    visualize_point_matches(img1, img2, pts1_after, pts2_after, output_dir='results/matches_after_ransac')
    

    # 11) Save results as pickle
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fundamental_optimization_results.pkl')

    results = {
        'fundamental_matrix_optimized': F_optimized,
        'fundamental_matrix_before_ransac': F_before,
        'fundamental_matrix_after_ransac': F_after,
        'epipolar_cost_before': epipolar_cost_before,
        'epipolar_cost_after': epipolar_cost_after,
        'cost_matrix': cost_matrix,
        'transport_matrix': transport_matrix,
        'camera1_K': K1,
        'camera2_K': K2,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_path}")
    print("Optimization Statistics:")
    print(f"  Final transport matrix shape: {transport_matrix.shape}")
    print(f"  Transport matrix sum: {transport_matrix.sum()}")
    print(f"  Final cost matrix mean: {cost_matrix.mean()}")

if __name__ == '__main__':
    main()
