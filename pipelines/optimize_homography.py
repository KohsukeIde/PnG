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

def get_top_correspondences(solver, num_points=100):
    """Get top correspondences based on transport matrix T.

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
    
    pts1 = points1[[i for i, _ in top_pairs]]
    pts2 = points2[[j for _, j in top_pairs]]
    
    print(f"len of pts1: {len(pts1)}")
    print(f"len of pts2: {len(pts2)}")
    
    return pts1, pts2

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

def visualize_homography_on_images(img1, img2, pts1, pts2, H_np, output_dir='results'):
    """Visualize homography effect and save results.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        pts1 (np.ndarray): Points from first image.
        pts2 (np.ndarray): Points from second image.
        H_np (np.ndarray): Homography matrix.
        output_dir (str): Path to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Warp img1 using homography
    img1_warped = cv2.warpPerspective(img1, H_np, (img2.shape[1], img2.shape[0]))

    # Stack images horizontally
    combined_img = np.hstack((img2, img1_warped))

    # Define colors
    color_pt1 = (0, 0, 255)    # Red
    color_pt2 = (255, 0, 0)    # Blue
    color_line = (0, 255, 0)   # Green

    # Draw correspondences
    for pt1, pt2 in zip(pts1, pts2):
        # Point in img2
        pt2_int = (int(pt2[0]), int(pt2[1]))
        # Point in img1_warped (add img2 width to x-coordinate)
        pt1_int = (int(pt1[0] + img2.shape[1]), int(pt1[1]))
        
        # Draw green line connecting correspondences
        cv2.line(combined_img, pt2_int, pt1_int, color_line, 1)
        
        # Draw points
        cv2.circle(combined_img, pt2_int, 5, color_pt2, -1)  # Blue point in img2
        cv2.circle(combined_img, pt1_int, 5, color_pt1, -1)  # Red point in img1_warped

    # Save image
    combined_image_path = os.path.join(output_dir, 'combined_correspondences.png')
    cv2.imwrite(combined_image_path, combined_img)

    print(f"Visualization image saved to '{combined_image_path}'.")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs'
    gaussians1_path = os.path.join(data_dir_gmm, 'fitted_gaussians_22_1k.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, 'fitted_gaussians_23_1k.pkl')
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
        lambda_mean=0.4,
        lambda_cov=0.2,
        lambda_color=0.4,
        device=device
    )

    # Debug information function
    def print_stats(tensor, name):
        print(f"\n{name} statistics:")
        print(f"Min: {tensor.min().item()}")
        print(f"Max: {tensor.max().item()}")
        print(f"Mean: {tensor.mean().item()}")
        print(f"Has NaN: {torch.isnan(tensor).any().item()}")
        print(f"Has Inf: {torch.isinf(tensor).any().item()}")

    # Optimize homography
    solver.optimize_with_homography(max_iter=1000, tol=1e-6)

    # Get optimized homography matrix
    H_optimized = solver.h.detach().cpu().numpy()
    print("Optimized Homography matrix:")
    print(H_optimized)

    # Calculate final cost matrix and transport plan
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        print_stats(cost_matrix, "Cost Matrix")
        
        transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
        print_stats(transport_matrix, "Transport Matrix")
        
        # Convert to numpy arrays for saving
        cost_matrix = cost_matrix.cpu().numpy()
        transport_matrix = transport_matrix.cpu().numpy()

    # Evaluation
    evaluate_homography_matrix(solver.h)
    
    evaluate_homography_residuals(solver, num_points=100)
    
    image1_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022.png'
    image2_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0023.png'
    
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    # Extract correspondences (using top N correspondences)
    num_points = 100
    pts1, pts2 = get_top_correspondences(solver, num_points)

    # Get homography matrix
    H_np = solver.h.detach().cpu().numpy()

    # Visualize homography effect on images and save results
    visualize_homography_on_images(img1, img2, pts1, pts2, H_np, output_dir='results')

    # Save results
    results = {
        'homography_matrix': H_optimized,
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
