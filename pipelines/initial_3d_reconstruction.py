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

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']


from src.reconstructor.initial_3d_reconstructor import Initial3DReconstructor

def save_point_cloud_as_ply(points, filename):
    """
    3D点群をPLYファイルとして保存

    Args:
        points (np.ndarray): 3D点群 (N x 3)。
        filename (str): 保存先のファイルパス。
    """
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')
    print(f"Saved {points.shape[0]} points to {filename}")

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
    image1_id = image_name_to_id.get(image1_name)
    image2_id = image_name_to_id.get(image2_name)

    if image1_id is None or image2_id is None:
        print(f"Error: One of the images '{image1_name}' or '{image2_name}' not found in COLMAP data.")
        sys.exit(1)

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

    # Initialize homography before optimization (for initialization)
    solver.h = torch.eye(3, device=device, dtype=torch.float32)

    # --- Optimizing Homography ---
    print("\n--- Optimizing Homography ---")
    solver.optimize_with_homography(max_iter=1000, tol=1e-6)

    # Get optimized homography matrix
    H_optimized = solver.h.detach().cpu().numpy()
    print("\nOptimized Homography matrix:")
    print(H_optimized)

    # Calculate final cost matrix and transport plan
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()

    # --- Triangulation of 3D Points ---
    print("\n--- Triangulating 3D Points ---")

    # Initialize Initial3DReconstructor with optimized homography
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H_optimized)
    reconstructor.compute_camera_matrices_from_homography()

    # Define transport_matrix threshold
    threshold = 1e-6  # 適宜調整

    # Triangulate points using the transport_matrix and threshold
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold)

    # Retrieve triangulated 3D points
    points_3d = reconstructor.points_3d
    print(f"Triangulated 3D points shape: {points_3d.shape}")

    if points_3d.shape[0] == 0:
        print("No 3D points were triangulated.")
    else:
        # Save the 3D points to a PLY file for visualization
        save_path = os.path.join('results', 'point_cloud.ply')
        os.makedirs('results', exist_ok=True)
        save_point_cloud_as_ply(points_3d, save_path)
        print(f"3D point cloud saved to {save_path}")

    # --- Save Results as Pickle ---
    # Save results
    results = {
        'homography_matrix': H_optimized,
        'cost_matrix': cost_matrix.cpu().numpy(),
        'transport_matrix': transport_matrix_np,
        'camera1_K': K1,
        'camera2_K': K2,
        'points_3d': points_3d,
    }

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'homography_optimization_results.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nResults saved to {output_path}")

    print("\nOptimization Statistics:")
    print(f"Final transport matrix shape: {transport_matrix_np.shape}")
    print(f"Transport matrix sum: {transport_matrix_np.sum():.6f}")
    print(f"Final cost matrix mean: {cost_matrix.cpu().numpy().mean():.6f}")

if __name__ == '__main__':
    main()
