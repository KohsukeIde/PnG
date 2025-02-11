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

# from src.optimizer.POT import OptimalTransportSolver
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

# Import visualization functions from visualize_tools.py
from utils.homography_pipeline_visualization import (
    visualize_point_matches,
    visualize_epipolar_lines,
    plot_epipolar_cost_change,
    save_warped_image
)

def get_top_correspondences(solver, num_points=100):
    """Get top 1:1 correspondences based on transport matrix T using greedy matching.

    Args:
        solver (OptimalTransportSolver): Solver instance.
        num_points (int): Number of correspondences to retrieve.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of points from image 1 and image 2.
    """
    # Get transport matrix T
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    
    T_np = transport_matrix.cpu().numpy()
    points1 = solver.means1.cpu().numpy()
    points2 = solver.means2.cpu().numpy()
    
    print(f"Transport matrix shape: {T_np.shape}")
    print(f"Transport matrix min value: {T_np.min()}")
    print(f"Transport matrix max value: {T_np.max()}")

    # Greedy 1:1 matching based on transport values
    used_rows = set()
    used_cols = set()
    matches = []
    
    # Get all values and their indices sorted by transport value
    flat_indices = np.argsort(-T_np.flatten())
    rows, cols = np.unravel_index(flat_indices, T_np.shape)
    
    # Find matches greedily
    for row, col in zip(rows, cols):
        if len(matches) >= num_points:
            break
        if row not in used_rows and col not in used_cols:
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
    
    # Convert matches to arrays
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

    Args:
        F (np.ndarray): Fundamental matrix (3x3).
        pts1 (np.ndarray): Points from image 1 (N x 2).
        pts2 (np.ndarray): Points from image 2 (N x 2).

    Returns:
        float: Average epipolar cost.
    """
    # Convert points to homogeneous coordinates
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N, 3)
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N, 3)

    # Compute epipolar constraint residuals
    Fx1 = F @ pts1_h.T  # (3, N)
    x2Fx1 = np.sum(pts2_h * Fx1.T, axis=1)  # (N,)

    residuals = np.abs(x2Fx1)  # (N,)

    # Average residual as epipolar cost
    epipolar_cost = np.mean(residuals)

    return epipolar_cost

def evaluate_homography_residuals(solver, num_points=100):
    """Evaluate residuals between transformed points and their correspondences using optimized homography.

    Args:
        solver (OptimalTransportSolver): Solver instance.
        num_points (int): Number of correspondences to evaluate.
    """
    # Get transport matrix T
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    
    # Convert transport matrix to numpy array
    T_np = transport_matrix.cpu().numpy()
    
    # Get indices of top correspondences
    indices = np.unravel_index(np.argsort(-T_np, axis=None), T_np.shape)
    idx_pairs = list(zip(indices[0], indices[1]))
    
    # Select top N correspondences
    top_pairs = idx_pairs[:num_points]
    
    # Prepare points
    points1 = solver.means1.cpu().numpy()
    points2 = solver.means2.cpu().numpy()
    ones = np.ones((points1.shape[0], 1))
    points1_h = np.hstack([points1, ones])
    
    H_np = solver.h.detach().cpu().numpy()
    
    residuals = []
    for idx1, idx2 in top_pairs:
        x1_h = points1_h[idx1]
        x2 = points2[idx2]
        # Transform x1 using homography
        x1_mapped_h = H_np @ x1_h
        x1_mapped = x1_mapped_h[:2] / x1_mapped_h[2]
        # Calculate residual
        res = np.linalg.norm(x1_mapped - x2)
        residuals.append(res)
    
    residuals = np.array(residuals)
    print("\nHomography Residuals Evaluation:")
    print(f"Mean residual: {residuals.mean()}")
    print(f"Median residual: {np.median(residuals)}")
    print(f"Max residual: {residuals.max()}")
    print(f"Min residual: {residuals.min()}")

def evaluate_homography_matrix(H):
    """Evaluate characteristics of the homography matrix.

    Args:
        H (torch.Tensor): Optimized homography matrix.
    """
    H_np = H.detach().cpu().numpy()
    # Calculate determinant
    det = np.linalg.det(H_np)
    print("\nHomography matrix evaluation:")
    print(f"Determinant: {det}")
    # Check if H is invertible
    if np.abs(det) > 1e-6:
        print("Homography matrix is invertible.")
    else:
        print("Homography matrix is singular and not invertible.")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs'
    # gaussians1_path = os.path.join(data_dir_gmm, 'fitted_gaussians_22_1k.pkl')
    # gaussians2_path = os.path.join(data_dir_gmm, 'fitted_gaussians_23_1k.pkl')
    
    gaussians1_path = os.path.join(data_dir_gmm, 'fitted_gaussians_22_16.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, 'fitted_gaussians_23_16.pkl')
    colmap_dir = os.path.join(data_dir, 'sparse/0')
    
    image1_name = '0022.png'
    image2_name = '0023.png'

    # Load Gaussians
    _, gaussians1, _, _ = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, _ = load_gaussians_torch(gaussians2_path, device)

    # Load camera information from COLMAP
    cameras = load_cameras_from_colmap(colmap_dir)
    images_data = load_images_from_colmap(colmap_dir)

    # Get camera information
    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
    image1_id = image_name_to_id[image1_name]
    image2_id = image_name_to_id[image2_name]

    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']

    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)

    # Get intrinsic parameter matrices
    K1 = camera1.K
    K2 = camera2.K

    # Initialize OptimalTransportSolver
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.1,
        lambda_mean=1,
        lambda_cov=1,
        lambda_color=1,
        lambda_epipolar=0.0,
        device=device
    )

    # Initialize homography before optimization (for visualization purpose)
    solver.h = torch.eye(3, device=device, dtype=torch.float32)

    # Debug information function
    def print_stats(tensor, name):
        print(f"\n{name} statistics:")
        print(f"Min: {tensor.min().item()}")
        print(f"Max: {tensor.max().item()}")
        print(f"Mean: {tensor.mean().item()}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

    # --- Before Optimization ---
    
    print("\n--- Optimization Before ---")
    
    # Get top correspondences before optimization
    pts1_before, pts2_before = get_top_correspondences(solver, num_points=16)
    
    # Estimate Fundamental Matrix before optimization
    F_before, mask_before = cv2.findFundamentalMat(
        pts1_before.astype(np.float32), 
        pts2_before.astype(np.float32), 
        cv2.FM_RANSAC
    )
    
    if F_before is not None and F_before.shape == (3, 3):
        print("\nEstimated Fundamental Matrix Before Optimization:")
        print(F_before)
    else:
        print("Failed to estimate Fundamental Matrix before optimization.")
        sys.exit(1)
    
    # Compute epipolar cost before optimization
    epipolar_cost_before = compute_epipolar_cost_cv2(F_before, pts1_before, pts2_before)
    print(f"Epipolar Cost Before Optimization: {epipolar_cost_before:.6f}")
    
    # --- Optimizing Homography ---
    
    # Optimize homography
    print("\n--- Optimizing Homography ---")
    solver.optimize_with_homography(max_iter=1000, tol=1e-2)

    # Get optimized homography matrix
    H_optimized = solver.h.detach().cpu().numpy()
    print("Optimized Homography matrix:")
    print(H_optimized)

    # Calculate final cost matrix and transport plan
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        print_stats(cost_matrix, "Cost Matrix")
        
        # transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        print_stats(transport_matrix, "Transport Matrix")
        
        # Convert to numpy arrays for saving
        cost_matrix = cost_matrix.cpu().numpy()
        transport_matrix = transport_matrix.cpu().numpy()

    # --- Post-optimization Processing ---
    
    print("\n--- Optimization After ---")
    
    # Get top correspondences after optimization
    pts1_after, pts2_after = get_top_correspondences(solver, num_points=10000)

    # Estimate Fundamental Matrix after optimization
    F_cv2, mask_cv2 = cv2.findFundamentalMat(
        pts1_after.astype(np.float32), 
        pts2_after.astype(np.float32), 
        cv2.FM_RANSAC
    )

    if F_cv2 is not None and F_cv2.shape == (3, 3):
        print("\nEstimated Fundamental Matrix After Optimization:")
        print(F_cv2)
    else:
        print("Failed to estimate Fundamental Matrix after optimization.")
        F_cv2 = np.eye(3, dtype=np.float32)  # デフォルト値

    # Compute epipolar cost after optimization
    epipolar_cost_after = compute_epipolar_cost_cv2(F_cv2, pts1_after, pts2_after)
    print(f"Epipolar Cost After Optimization: {epipolar_cost_after:.6f}")

    # --- plot epipolar cost ---
    
    # Plot epipolar cost change
    plot_epipolar_cost_change(epipolar_cost_before, epipolar_cost_after, output_dir='results')

    # --- Visualize epipolar lines and corresponding points ---
    
    # Load images
    image1_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022.png'
    image2_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0023.png'
    
    # image1_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022_shifted.png'
    # image2_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022_shifted2.png'
    
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None:
        print(f"Failed to load image from {image1_path}")
        sys.exit(1)
    if img2 is None:
        print(f"Failed to load image from {image2_path}")
        sys.exit(1)

    # Visualize epipolar lines before optimization
    visualize_epipolar_lines(img1, img2, pts1_before, pts2_before, F_before, output_dir='results/epilines_before')
    
    # Visualize epipolar lines after optimization
    visualize_epipolar_lines(img1, img2, pts1_after, pts2_after, F_cv2, output_dir='results/epilines_after')

    # --- other evaluations ---
    
    # Evaluate homography matrix
    evaluate_homography_matrix(solver.h)
    
    # Evaluate homography residuals
    evaluate_homography_residuals(solver, num_points=100)
    
    # --- save visualizations ---
    
    # Visualize point matches before optimization
    visualize_point_matches(img1, img2, pts1_before, pts2_before, output_dir='results/matches_before')
    
    # Visualize point matches after optimization
    visualize_point_matches(img1, img2, pts1_after, pts2_after, output_dir='results/matches_after')

    # Save warped image after optimization
    save_warped_image(img1, img2, H_optimized, output_dir='results')

    # --- save results as pickle (Just in case) ---
    
    # Save results
    results = {
        'homography_matrix': H_optimized,
        'fundamental_matrix_before': F_before,
        'fundamental_matrix_after': F_cv2,
        'epipolar_cost_before': epipolar_cost_before,
        'epipolar_cost_after': epipolar_cost_after,
        'cost_matrix': cost_matrix,
        'transport_matrix': transport_matrix,
        'camera1_K': K1,
        'camera2_K': K2,
    }

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'homography_optimization_results.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_path}")

    print("\nOptimization Statistics:")
    print(f"Final transport matrix shape: {transport_matrix.shape}")
    print(f"Transport matrix sum: {transport_matrix.sum()}")
    print(f"Final cost matrix mean: {cost_matrix.mean()}")

if __name__ == '__main__':
    main()
