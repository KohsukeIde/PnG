import os
import sys
import argparse
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
    visualize_epipolar_lines
)

def get_top_correspondences_fundamental(solver, num_points=100):
    """Fundamental 行列 F を用いたコスト行列を計算し、
    Sinkhorn / Unbalanced Sinkhorn から上位の対応点をGreedyで取り出す。
    """
    with torch.no_grad():
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

    flat_indices = np.argsort(-T_np.ravel())  # 降順ソート
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


def parse_args():
    """Parse command-line arguments for fundamental matrix optimization.
    """
    parser = argparse.ArgumentParser(description="Fundamental matrix optimization with Gaussian correspondences.")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63",
        help="Path to the base data directory."
    )
    parser.add_argument(
        "--data_dir_gmm",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked/",
        help="Path to the directory where the GMM data files are stored."
    )
    parser.add_argument(
        "--gaussians1_filename",
        type=str,
        default="0022_fitted_gaussians.pkl",
        help="Filename for the first set of Gaussians."
    )
    parser.add_argument(
        "--gaussians2_filename",
        type=str,
        default="0023_fitted_gaussians.pkl",
        help="Filename for the second set of Gaussians."
    )
    parser.add_argument(
        "--colmap_subdir",
        type=str,
        default="sparse/0",
        help="Subdirectory (relative to data_dir) where COLMAP files are located."
    )
    parser.add_argument(
        "--image1_name",
        type=str,
        default="0022.png",
        help="Name of the first image file."
    )
    parser.add_argument(
        "--image2_name",
        type=str,
        default="0023.png",
        help="Name of the second image file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory where results (images, pickles) will be saved."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1) Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2) Data paths (now derived from arguments)
    data_dir = args.data_dir
    
    data_dir_gmm = args.data_dir_gmm
    gaussians1_path = os.path.join(data_dir_gmm, args.gaussians1_filename)
    gaussians2_path = os.path.join(data_dir_gmm, args.gaussians2_filename)
    colmap_dir = os.path.join(data_dir, args.colmap_subdir)

    image1_name = args.image1_name
    image2_name = args.image2_name

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
        lambda_color=0.0,
        lambda_epipolar=1.0,
        device=device
    )

    # 画像読み込み（SIFT/ORBはグレースケール使用）
    image1_path = os.path.join(data_dir, 'images', image1_name)
    image2_path = os.path.join(data_dir, 'images', image2_name)
    img1_color = cv2.imread(image1_path)
    img2_color = cv2.imread(image2_path)
    if img1_color is None or img2_color is None:
        print(f"Failed to load images for visualization.")
        sys.exit(1)

    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # 特徴量検出 (SIFT)
    sift = cv2.SIFT_create()  # cv2.SIFT_create() はOpenCVのバージョンに依存
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    # BFMatcher でマッチング + ratio test
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    ratio_threshold = 0.7
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # キーポイントの座標を取得
    pts1_before = []
    pts2_before = []
    for m in good_matches:
        pts1_before.append(keypoints1[m.queryIdx].pt)
        pts2_before.append(keypoints2[m.trainIdx].pt)

    pts1_before = np.array(pts1_before, dtype=np.float32)
    pts2_before = np.array(pts2_before, dtype=np.float32)

    # 6) Use RANSAC to get initial F
    F_ransac, mask_before = cv2.findFundamentalMat(
        pts1_before,
        pts2_before,
        cv2.FM_RANSAC
    )
    print("\n--- Optimization Before ---")
    print("\nInitial F from ransac")
    print(F_ransac)

    # Helper function: debug stats
    def print_stats(tensor, name):
        print(f"\n{name} statistics:")
        print(f"  Min: {tensor.min().item()}")
        print(f"  Max: {tensor.max().item()}")
        print(f"  Mean: {tensor.mean().item()}")
        print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"  Has Inf: {torch.isinf(tensor).any().item()}")

    solver.f = None
    # 7) Optimize with Fundamental
    print("\n--- Optimizing Fundamental Matrix ---")
    solver.optimize_with_RT(max_iter=1000, tol=1e-6)
    # solver.optimize_with_fundamental(max_iter=1000, tol=1e-6)

    print("Initial F from ransac")
    print(F_ransac)
    
    # 最適化された solver.f を取得
    F_optimized = solver.f.detach().cpu().numpy()
    print("\nOptimized Fundamental matrix:")
    print(F_optimized)

    # コスト行列を確認
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        print_stats(cost_matrix, "Cost Matrix (after optimization)")

        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        print_stats(transport_matrix, "Transport Matrix (after optimization)")

        cost_matrix_np = cost_matrix.cpu().numpy()
        transport_matrix_np = transport_matrix.cpu().numpy()

    # 8) After Optimization
    print("\n--- Optimization After ---")

    # 再度対応点を抽出 (最大で 5000組)
    pts1_after, pts2_after = get_top_correspondences_fundamental(solver, num_points=5000)

    # 画像パス
    image1_path = os.path.join(data_dir, 'images', image1_name)
    image2_path = os.path.join(data_dir, 'images', image2_name)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print(f"Failed to load images for visualization.")
        sys.exit(1)

    # 9) Visualize epipolar lines and corresponding points
    # - Before
    visualize_epipolar_lines(img1, img2, pts1_before, pts2_before, F_ransac, output_dir=os.path.join(args.output_dir, 'epilines_before'))
    visualize_point_matches(img1, img2, pts1_before, pts2_before, output_dir=os.path.join(args.output_dir, 'matches_before'))

    # - After
    visualize_epipolar_lines(img1, img2, pts1_after, pts2_after, F_optimized, output_dir=os.path.join(args.output_dir, 'epilines_after'))
    visualize_point_matches(img1, img2, pts1_after, pts2_after, output_dir=os.path.join(args.output_dir, 'matches_after'))

    # 10) Save results as pickle
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'fundamental_optimization_results.pkl')

    results = {
        'fundamental_matrix_optimized': F_optimized,
        'cost_matrix': cost_matrix_np,
        'transport_matrix': transport_matrix_np,
        'camera1_K': K1,
        'camera2_K': K2,
        'pts1_before': pts1_before,
        'pts2_before': pts2_before,
        'pts1_after': pts1_after,
        'pts2_after': pts2_after,
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_path}")
    print("Optimization Statistics:")
    print(f"  Final transport matrix shape: {transport_matrix_np.shape}")
    print(f"  Transport matrix sum: {transport_matrix_np.sum()}")
    print(f"  Final cost matrix mean: {cost_matrix_np.mean()}")

if __name__ == '__main__':
    main()
