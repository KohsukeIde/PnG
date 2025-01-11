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

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

# Import your reconstructor that can compute 3D covariances
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

def sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta=12, n_phi=12):
    eigvals, eigvecs = np.linalg.eigh(Sigma_3)
    eigvals = np.clip(eigvals, 1e-12, None)
    scales = np.sqrt(eigvals)
    sqrtSigma = eigvecs @ np.diag(scales) @ eigvecs.T

    vertices = []
    faces = []
    for i in range(n_theta+1):
        theta = 2.0*np.pi*i/n_theta
        for j in range(n_phi+1):
            phi = np.pi*j/n_phi
            x_sph = np.sin(phi)*np.cos(theta)
            y_sph = np.sin(phi)*np.sin(theta)
            z_sph = np.cos(phi)
            unit_vec = np.array([x_sph, y_sph, z_sph])
            xyz_ellip = sqrtSigma @ unit_vec
            xyz_ellip += center
            vertices.append(xyz_ellip)

    def idx(i, j):
        return i*(n_phi+1) + j

    for i in range(n_theta):
        for j in range(n_phi):
            v1 = idx(i,j)
            v2 = idx(i+1,j)
            v3 = idx(i,j+1)
            v4 = idx(i+1,j+1)
            faces.append([v1,v2,v3])
            faces.append([v2,v4,v3])

    return vertices, faces

def save_point_cloud_as_ply(points, filename):
    """
    Save 3D points as a PLY file.
    Args:
        points (np.ndarray): 3D points (N, 3).
        filename (str): Output PLY file path.
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
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved {points.shape[0]} points to {filename}")
def save_ellipsoids_as_ply(all_vertices, all_faces, filename):
    merged_vertices = []
    merged_faces = []
    v_offset = 0
    
    # Add debug prints to check values
    print(f"First few vertices values:")
    for i, (vertices, faces) in enumerate(zip(all_vertices, all_faces)):
        if i < 3:  # Print first 3 gaussians
            print(f"Gaussian {i}:")
            print(f"  Center: {vertices[0][:3]}")
            print(f"  Color: {vertices[0][3:6]}")  # Only RGB, no alpha
            
        for v in vertices:
            # Ensure vertices are not at infinity or NaN
            if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                continue
                
            merged_vertices.append(v)
        for f in faces:
            merged_faces.append([f[0]+v_offset, f[1]+v_offset, f[2]+v_offset])
        v_offset += len(vertices)

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(merged_vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        # f.write('property uchar alpha\n')  # Commented out alpha
        f.write(f'element face {len(merged_faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        for mv in merged_vertices:
            x, y, z = mv[0], mv[1], mv[2]
            r_f, g_f, b_f = mv[3], mv[4], mv[5]  # Only RGB
            # a_f = mv[6]  # Commented out alpha

            # Convert color values to integers (0-255)
            r = int(np.clip(r_f*255, 0, 255))
            g = int(np.clip(g_f*255, 0, 255))
            b = int(np.clip(b_f*255, 0, 255))
            # a = int(np.clip(a_f*255, 0, 255))  # Commented out alpha
            
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")
            # f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b} {a}\n")  # Original line with alpha

        for face in merged_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"\nDebug info:")
    print(f"Number of valid vertices: {len(merged_vertices)}")
    print(f"Number of faces: {len(merged_faces)}")
    print(f"Vertex offset at end: {v_offset}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/train_results'
    gaussians1_path = os.path.join(data_dir_gmm, '0022_fitted_gaussians.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, '0023_fitted_gaussians.pkl')
    colmap_dir = os.path.join(data_dir, 'sparse/0')

    image1_name = '0022.png'
    image2_name = '0023.png'

    # 1) Load Gaussians
    _, gaussians1, _, K1 = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, K2 = load_gaussians_torch(gaussians2_path, device)

    # 2) Calculate dynamic target volume based on image properties
    W1, H1 = K1[0, 2] * 2, K1[1, 2] * 2  # Image width and height for camera 1
    W2, H2 = K2[0, 2] * 2, K2[1, 2] * 2  # Image width and height for camera 2
    avg_pixel_area = (W1 * H1 + W2 * H2) / 2
    num_gaussians = max(len(gaussians1.means), len(gaussians2.means))
    target_volume = avg_pixel_area / num_gaussians
    print(f"Calculated target volume: {target_volume:.2f}")

    # 3) Load camera + COLMAP
    cameras = load_cameras_from_colmap(colmap_dir)
    images_data = load_images_from_colmap(colmap_dir)

    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
    image1_id = image_name_to_id.get(image1_name)
    image2_id = image_name_to_id.get(image2_name)
    if image1_id is None or image2_id is None:
        print(f"Error: {image1_name} or {image2_name} not found in COLMAP.")
        sys.exit(1)

    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']

    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)
    K1 = camera1.K
    K2 = camera2.K

    # 4) Setup OptimalTransportSolver
    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.1,
        lambda_mean=1.0,
        lambda_cov=1.0,
        lambda_color=1.0,
        lambda_alpha=1.0,
        device=device
    )

    # 5) Homography optimization
    solver.h = torch.eye(3, dtype=torch.float32, device=device)
    print("\n--- Optimizing Homography ---")
    solver.optimize_with_homography(max_iter=500, tol=1e-6)

    H_optimized = solver.h.detach().cpu().numpy()
    print("\nOptimized Homography matrix:\n", H_optimized)

    # 6) Final cost & transport
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        # transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()

    # 7) Triangulate
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H_optimized)
    reconstructor.compute_camera_matrices_from_homography()
    threshold = 1e-6
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold, top_k=1000)
    # reconstructor.normalize_reconstruction()
    points_3d = reconstructor.points_3d
    print(f"\nTriangulated {points_3d.shape[0]} 3D points")

    # 8) Compute Covariances with Volume Prior
    print("\n--- Computing 3D Gaussian Covariances with Volume Prior ---")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=1.0, target_volume=target_volume)

    ply_points_out = os.path.join('results', 'triangulated_points.ply')
    save_point_cloud_as_ply(points_3d, ply_points_out)
    
    # 9) Compute color & alpha
    print("\n--- Computing 3D Gaussian Colors & Alphas ---")
    reconstructor.compute_3d_gaussian_colors(color_mode="average")
    reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")

    # 10) Build ellipsoids => PLY
    all_vertices = []
    all_faces = []
    n_theta, n_phi = 12, 12
    for i in range(points_3d.shape[0]):
        center = points_3d[i]
        Sigma_3 = reconstructor.covariances_3d[i]
        
        # Scale up covariance if ellipsoids are too small
        # scale_factor = 500.0  # Adjust this value if ellipsoids are too small/large
        # Sigma_3 = Sigma_3 * scale_factor
        
        c3 = reconstructor.color_3d[i]
        a3 = reconstructor.alpha_3d[i]  # Commented out alpha
        
        # Ensure minimum alpha
        # a3 = max(a3, 0.5)  # Commented out alpha adjustment

        raw_vertices, faces = sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta, n_phi)
        extended_vertices = []
        for vert in raw_vertices:
            # Only include position and RGB (no alpha)
            combo = np.concatenate([vert, c3[:3]])
            # combo = np.concatenate([vert, c3, [a3]])  # Original line with alpha
            extended_vertices.append(combo)

        all_vertices.append(extended_vertices)
        all_faces.append(faces)

    # 10) Save them as PLY
    ply_out = os.path.join('results', '3d_gaussians_ellipsoids.ply')
    os.makedirs('results', exist_ok=True)
    save_ellipsoids_as_ply(all_vertices, all_faces, ply_out)

    # 11) Save final results
    results = {
        'homography_matrix': H_optimized,
        'cost_matrix': cost_matrix.cpu().numpy(),
        'transport_matrix': transport_matrix_np,
        'camera1_K': K1,
        'camera2_K': K2,
        'points_3d': points_3d,
        'covariances_3d': reconstructor.covariances_3d,
        'color_3d': reconstructor.color_3d,
        'alpha_3d': reconstructor.alpha_3d,
    }
    out_pkl = os.path.join('results', 'homography_optimization_results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nSaved pipeline results to {out_pkl}")
    print("\nDone.")


if __name__ == '__main__':
    main()