import os
import sys
import pickle
import argparse 
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

########################
# 0) Import modules
########################
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

# Import your reconstructor that can compute 3D covariances (with volume prior, color, alpha)
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

########################
# Helper function: sample_ellipsoid_vertices_and_faces
########################
def sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta=12, n_phi=12):
    """
    Approximate the surface of a 3D ellipsoid (Gaussian) by sampling a sphere
    and applying sqrt(Sigma_3).

    Args:
        Sigma_3 (np.ndarray): 3x3 positive semi-definite covariance matrix
        center (np.ndarray): shape (3,), 3D center
        n_theta (int): number subdivisions for azimuth
        n_phi (int): number subdivisions for polar angle

    Returns:
        vertices (list of [x,y,z]): approximated ellipsoid vertices in 3D
        faces (list of [v1,v2,v3]): triangular faces (indices into vertices)
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma_3)
    eigvals = np.clip(eigvals, 1e-12, None)
    scales = np.sqrt(eigvals)
    sqrtSigma = eigvecs @ np.diag(scales) @ eigvecs.T

    vertices = []
    faces = []
    for i in range(n_theta+1):
        theta = 2.0 * np.pi * i / n_theta
        for j in range(n_phi+1):
            phi = np.pi * j / n_phi
            x_sph = np.sin(phi) * np.cos(theta)
            y_sph = np.sin(phi) * np.sin(theta)
            z_sph = np.cos(phi)
            unit_vec = np.array([x_sph, y_sph, z_sph])
            xyz_ellip = sqrtSigma @ unit_vec
            xyz_ellip += center
            vertices.append(xyz_ellip)

    def idx(ii, jj):
        return ii*(n_phi+1) + jj

    for i in range(n_theta):
        for j in range(n_phi):
            v1 = idx(i,   j)
            v2 = idx(i+1, j)
            v3 = idx(i,   j+1)
            v4 = idx(i+1, j+1)
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    return vertices, faces


########################
# Helper function: Save ellipsoids as PLY (with color, alpha optional)
########################
def save_ellipsoids_as_ply(all_vertices, all_faces, filename, use_alpha=True):
    """
    Merge all ellipsoids' geometry and save as a single PLY.

    Args:
        all_vertices: list of lists of shape (num_gaussians, [N_verts, 7]) if including color+alpha
        all_faces: list of lists of triangle indices
        filename: output ply path
        use_alpha (bool): if True, interpret the 7th dimension as alpha
    """
    merged_vertices = []
    merged_faces = []
    v_offset = 0

    for (vertices, faces) in zip(all_vertices, all_faces):
        for v in vertices:
            merged_vertices.append(v)
        for f in faces:
            merged_faces.append([f[0] + v_offset, f[1] + v_offset, f[2] + v_offset])
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
        if use_alpha:
            f.write('property uchar alpha\n')
        f.write(f'element face {len(merged_faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        for mv in merged_vertices:
            x, y, z = mv[0], mv[1], mv[2]
            r_f, g_f, b_f = mv[3], mv[4], mv[5]
            r = int(np.clip(r_f * 255, 0, 255))
            g = int(np.clip(g_f * 255, 0, 255))
            b = int(np.clip(b_f * 255, 0, 255))

            if use_alpha:
                a_f = mv[6]
                a = int(np.clip(a_f * 255, 0, 255))
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b} {a}\n")
            else:
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")

        for face in merged_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved {len(all_vertices)} ellipsoids => {len(merged_vertices)} vertices, {len(merged_faces)} faces to {filename}")


########################
# Helper function: Save 3D points as PLY
########################
def save_point_cloud_as_ply(points, filename):
    """
    Save 3D points as a PLY file.

    Args:
        points (np.ndarray): shape (N,3).
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


########################
#(WIP): Project 3D Gaussians back to 2D
########################
def render_gaussians_pure_mixture(
    points_3d,
    covariances_3d,
    color_3d,
    alpha_3d,
    R_cam,
    t_cam,
    K,
    out_width,
    out_height,
    splat_radius_factor=3.0
):
    """
    Render 3D Gaussians into a 2D image with a 'pure mixture' approach.
    Instead of alpha compositing, we accumulate:
       weight_buffer[py, px] += gauss_val
       color_buffer[py, px] += gauss_val * color_i
    Then final_pixel_color = color_buffer / weight_buffer (if weight_buffer>0).
    
    Args:
        points_3d      : shape (N,3)
        covariances_3d: shape (N,3,3)
        color_3d       : shape (N,3) in [0..1]
        alpha_3d       : shape (N,) in [0..1] – you can also incorporate alpha if desired
        R_cam, t_cam   : extrinsic transform from world->camera
        K             : intrinsics
        out_width, out_height: image size
        splat_radius_factor  : # std dev radius for bounding region
    Returns:
        mixture_img (H,W,3) float32 in [0..1]  # pure mixture of colors
        coverage_img (H,W)  float32 in [0..something]  # sum of Gauss weights
    """

    # Buffers for accumulation
    weight_buffer = np.zeros((out_height, out_width), dtype=np.float32)
    color_buffer  = np.zeros((out_height, out_width, 3), dtype=np.float32)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    for i in tqdm(range(points_3d.shape[0]), desc="Rendering Gaussians (pure mixture)"):
        X_w = points_3d[i]
        Sigma_3 = covariances_3d[i]
        rgb = color_3d[i]  # [r,g,b] in [0..1]
        # alpha can be used if you want to scale the amplitude or ignore it.

        # Transform to camera coords
        X_c = R_cam @ X_w + t_cam
        if X_c[2] <= 1e-6:
            continue

        # Project center
        u = fx*(X_c[0]/X_c[2]) + cx
        v = fy*(X_c[1]/X_c[2]) + cy

        px_center = int(np.round(u))
        py_center = int(np.round(v))
        if px_center<0 or px_center>=out_width or py_center<0 or py_center>=out_height:
            continue

        # Sigma_3 -> Sigma_cam
        Sigma_cam = R_cam @ Sigma_3 @ R_cam.T

        # local Jacobian
        X, Y, Z = X_c
        J = np.array([
            [fx/Z,     0.0,   -fx*X/(Z**2)],
            [0.0,     fy/Z,   -fy*Y/(Z**2)]
        ], dtype=np.float32)

        Sigma_2D = J @ Sigma_cam @ J.T
        e_vals, e_vecs = np.linalg.eig(Sigma_2D)
        e_vals = np.clip(e_vals, 1e-12, None)
        std_x = np.sqrt(e_vals[0])
        std_y = np.sqrt(e_vals[1])

        radius_x = int(np.ceil(std_x * splat_radius_factor))
        radius_y = int(np.ceil(std_y * splat_radius_factor))

        min_x = max(px_center - radius_x, 0)
        max_x = min(px_center + radius_x, out_width-1)
        min_y = max(py_center - radius_y, 0)
        max_y = min(py_center + radius_y, out_height-1)

        inv_Sigma_2D = np.linalg.inv(Sigma_2D)

        for py in range(min_y, max_y+1):
            for px in range(min_x, max_x+1):
                dx = px - u
                dy = py - v
                disp = np.array([dx, dy], dtype=np.float32)
                val = disp @ inv_Sigma_2D @ disp
                gauss_val = np.exp(-0.5*val)
                # If you'd like to incorporate alpha as amplitude, do gauss_val *= alpha_3d[i]

                # Accumulate in the mixture sense
                weight_buffer[py, px] += gauss_val
                color_buffer[py, px]  += (gauss_val * rgb)

    # finalize
    mixture_img = np.zeros((out_height, out_width, 3), dtype=np.float32)
    mask = (weight_buffer > 1e-12)
    mixture_img[mask] = color_buffer[mask] / weight_buffer[mask][...,None]  # broadcast

    return mixture_img, weight_buffer


def create_camera_frustum_mesh(
    K,
    R_world2cam,
    t_world2cam,
    color=(1.0, 0.0, 0.0),
    alpha=1.0,
    near_z=0.1,
    far_z=0.5,
    scale_fov=1.0
):
    """
    Create a simple triangular mesh representing the camera frustum (pyramid).
    - K: Intrinsic (3x3), typically [fx, 0, cx; 0, fy, cy; 0,0,1]
    - R_world2cam, t_world2cam: The extrinsic that maps world->camera. 
      If you have camera1.R, camera1.t as world->cam, 
      then the camera center in world coords is C = -R^T * t.
    - color, alpha: color in [0..1], alpha in [0..1]
    - near_z, far_z: position of near-plane and far-plane in camera coords (z>0)
    - scale_fov: to scale the pyramid size if you want bigger/smaller frustum

    Returns:
        frustum_vertices: list of np.array([x,y,z,r,g,b,a]) shape=(N,)
        frustum_faces   : list of [v1,v2,v3] index triplets
    """

    # 1) Invert (R,t) to get camera pose as cam->world
    R_cam2world = R_world2cam.T
    t_cam2world = -R_world2cam.T @ t_world2cam

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 2) Define corners in camera coords
    corners_cam = []
    for zval in [near_z, far_z]:
        # 4 corners in pixel coords (u,v)
        uvs = [
            (0, 0),
            (2*cx, 0),
            (2*cx, 2*cy),
            (0, 2*cy),
        ]
        for (u,v) in uvs:
            x = (u - cx)/fx * zval
            y = (v - cy)/fy * zval
            corners_cam.append(np.array([x, y, zval], dtype=np.float32))

    corners_cam = np.array(corners_cam)
    # optionally scale the FOV:
    corners_cam[:, :2] *= scale_fov

    # 3) Transform corners_cam to world coords
    corners_world = []
    for cc in corners_cam:
        cw = R_cam2world @ cc + t_cam2world
        corners_world.append(cw)

    corners_world = np.array(corners_world)  # shape (8,3)

    # Also define the camera center itself
    camera_center = t_cam2world  # shape (3,)

    def make_vert_xyzrgba(xyz):
        return np.array([xyz[0], xyz[1], xyz[2], color[0], color[1], color[2], alpha], dtype=np.float32)

    frustum_vertices = []
    frustum_vertices.append(make_vert_xyzrgba(camera_center))  # index 0
    for i in range(8):
        frustum_vertices.append(make_vert_xyzrgba(corners_world[i]))  # index i+1

    # 5) Build faces
    frustum_faces = []
    # center -> near-plane
    for i in range(4):
        i0 = 0
        i1 = 1 + i
        i2 = 1 + ((i+1) % 4)
        frustum_faces.append([i0, i1, i2])

    # center -> far-plane
    for i in range(4):
        i0 = 0
        i1 = 5 + i
        i2 = 5 + ((i+1) % 4)
        frustum_faces.append([i0, i1, i2])

    # near-plane ring => 2 triangles
    frustum_faces.append([1,2,3])
    frustum_faces.append([1,3,4])

    # far-plane ring => 2 triangles
    frustum_faces.append([5,6,7])
    frustum_faces.append([5,7,8])

    return frustum_vertices, frustum_faces



def parse_args():
    """Parse command-line arguments for path configuration.
    """
    parser = argparse.ArgumentParser(description="Pipeline to reconstruct 3D ellipsoids from 2D Gaussian data.")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63",
        help="Path to the main data directory (e.g. DTU scan folder)."
    )
    parser.add_argument(
        "--data_dir_gmm",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked",
        help="Path to the directory that contains fitted Gaussian pkls."
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default="sparse/0",
        help="Relative or absolute path to the COLMAP sparse folder."
    )
    parser.add_argument(
        "--image1_name",
        type=str,
        default="0022.png",
        help="Filename of the first image."
    )
    parser.add_argument(
        "--image2_name",
        type=str,
        default="0023.png",
        help="Filename of the second image."
    )
    parser.add_argument(
        "--gaussians1_filename",
        type=str,
        default="0022_fitted_gaussians.pkl",
        help="Filename of the first fitted Gaussians pickle."
    )
    parser.add_argument(
        "--gaussians2_filename",
        type=str,
        default="0023_fitted_gaussians.pkl",
        help="Filename of the second fitted Gaussians pickle."
    )

    return parser.parse_args()

########################
# Main pipeline function
########################
def main():
    args = parse_args()  # 追加

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ##############################
    # Data / paths 
    ##############################
    data_dir = args.data_dir
    data_dir_gmm = args.data_dir_gmm
    colmap_dir = os.path.join(data_dir, args.colmap_dir)
    image1_name = args.image1_name
    image2_name = args.image2_name

    gaussians1_path = os.path.join(data_dir_gmm, args.gaussians1_filename)
    gaussians2_path = os.path.join(data_dir_gmm, args.gaussians2_filename)

    ##############################
    # 1) Load Gaussians
    ##############################
    _, gaussians1, _, K1 = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, K2 = load_gaussians_torch(gaussians2_path, device)

    # 2) Calculate target volume based on image properties
    W1, H1 = K1[0, 2]*2, K1[1, 2]*2  # image1 width, height
    W2, H2 = K2[0, 2]*2, K2[1, 2]*2  # image2 width, height
    avg_pixel_area = (W1 * H1 + W2 * H2) / 2
    num_gaussians = max(len(gaussians1.means), len(gaussians2.means))
    target_volume = avg_pixel_area / num_gaussians
    print(f"Calculated target volume: {target_volume:.2f}")

    ##############################
    # 3) Load camera + COLMAP info
    ##############################
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

    ##############################
    # 4) Setup OptimalTransportSolver (unbalanced version)
    ##############################
    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.01,
        lambda_mean=0.0,
        lambda_cov=0.0,
        lambda_color=0.0,
        lambda_epipolar=1e-3,
        device=device
    )

    ##############################
    # 5) (Homography optimization is commented out)
    ##############################
    # solver.h = torch.eye(3, dtype=torch.float32, device=device)
    # print("\n--- Optimizing Homography ---")
    # solver.optimize_with_homography(max_iter=500, tol=1e-6)
    # H_optimized = solver.h.detach().cpu().numpy()
    # print("\nOptimized Homography matrix:\n", H_optimized)
    
    ##############################
    # 5) Fundamental matrix optimization (using R,t)
    ##############################
    print("\n--- Optimizing Fundamental Matrix ---")
    solver.optimize_with_RT(max_iter=1000, tol=1e-6)
    F_optimized = solver.f.detach().cpu().numpy()
    print("\nOptimized Fundamental matrix (from R,t):\n", F_optimized)

    ##############################
    # 6) Final cost & unbalanced transport
    ##############################
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()

    ##############################
    # 7) Triangulate
    ##############################
    from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor
    h_dummy = np.eye(3)
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, h_dummy)

    # Get R,t from solver
    r_optimized = solver.rvec.detach().cpu().numpy()
    t_optimized = solver.tvec.detach().cpu().numpy()
    R_est = solver.rodrigues(solver.rvec).detach().cpu().numpy()
    print("R_est:\n", R_est)
    print("t_est:\n", t_optimized)
    
    reconstructor.set_camera_matrices_explicitly(
        r1=np.eye(3), 
        t1=np.zeros(3), 
        r2=R_est, 
        t2=t_optimized
    )

    threshold = 1e-6
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold, top_k=100000)
    points_3d = reconstructor.points_3d
    print(f"\nTriangulated {points_3d.shape[0]} 3D points")

    ##############################
    # 8) Compute Covariances with Volume Prior
    ##############################
    print("\n--- Computing 3D Gaussian Covariances with Volume Prior ---")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=1.0, target_volume=target_volume)

    ply_points_out = os.path.join('results', 'triangulated_points.ply')
    os.makedirs('results', exist_ok=True)
    save_point_cloud_as_ply(points_3d, ply_points_out)

    ##############################
    # 9) Compute color & alpha
    ##############################
    print("\n--- Computing 3D Gaussian Colors & Alphas ---")
    reconstructor.compute_3d_gaussian_colors(color_mode="average")
    reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")

    ##############################
    # 10) Build ellipsoids => PLY
    ##############################
    all_vertices = []
    all_faces = []
    n_theta, n_phi = 12, 12

    for i in range(points_3d.shape[0]):
        center = points_3d[i]
        Sigma_3 = reconstructor.covariances_3d[i]
        c3 = reconstructor.color_3d[i]
        a3 = reconstructor.alpha_3d[i]

        raw_vertices, faces = sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta, n_phi)
        extended_vertices = []
        for vert in raw_vertices:
            combo = np.concatenate([vert, c3, [a3]])
            extended_vertices.append(combo)

        all_vertices.append(extended_vertices)
        all_faces.append(faces)

    ply_out = os.path.join('results', '3d_gaussians_ellipsoids.ply')
    save_ellipsoids_as_ply(all_vertices, all_faces, ply_out, use_alpha=True)
    
    ##############################
    # 10) Build ellipsoids => PLY (with camera frustums)
    ##############################
    # === ADDED for camera frustum ===
    R1 = camera1.R_wc  # world->camera
    t1 = camera1.t_wc
    frustum_color = (0.0, 1.0, 0.0)
    frustum_alpha = 1.0
    camera1_frustum_verts, camera1_frustum_faces = create_camera_frustum_mesh(
        K=camera1.K,
        R_world2cam=R1,
        t_world2cam=t1,
        color=frustum_color,
        alpha=frustum_alpha,
        near_z=0.1,
        far_z=0.4,
        scale_fov=100.0  
    )
    all_vertices.append(camera1_frustum_verts)
    all_faces.append(camera1_frustum_faces)

    R2 = camera2.R_wc
    t2 = camera2.t_wc
    frustum_color2 = (0.0, 0.0, 1.0)
    camera2_frustum_verts, camera2_frustum_faces = create_camera_frustum_mesh(
        K=camera2.K,
        R_world2cam=R2,
        t_world2cam=t2,
        color=frustum_color2,
        alpha=frustum_alpha,
        near_z=0.1,
        far_z=0.4,
        scale_fov=100.0
    )
    all_vertices.append(camera2_frustum_verts)
    all_faces.append(camera2_frustum_faces)

    ply_out = os.path.join('results', '3d_gaussians_ellipsoids_withCams.ply')
    save_ellipsoids_as_ply(all_vertices, all_faces, ply_out, use_alpha=True)

    ##############################
    # 11) (Optional) Project 3D Gaussians back to 2D for debug
    ##############################
    if True:
        print("\n--- Rendering 3D Gaussians back into camera1's 2D image (alpha-blend) ---")

        R_cam = np.eye(3)
        t_cam = np.zeros(3)

        out_width  = int(camera1.K[0,2]*2)
        out_height = int(camera1.K[1,2]*2)

        mixture_img, coverage_img = render_gaussians_pure_mixture(
            points_3d=reconstructor.points_3d,
            covariances_3d=reconstructor.covariances_3d,
            color_3d=reconstructor.color_3d,
            alpha_3d=reconstructor.alpha_3d,
            R_cam=R_cam,
            t_cam=t_cam,
            K=camera1.K,
            out_width=out_width,
            out_height=out_height,
        )
        
        rendered_rgba = np.zeros((out_height, out_width, 4), dtype=np.float32)
        rendered_rgba[..., :3] = mixture_img
        rendered_rgba[..., 3] = coverage_img

        rendered_8u = np.clip(rendered_rgba*255.0, 0, 255).astype(np.uint8)
        
        rendered_8u_bgra = rendered_8u.copy()
        rendered_8u_bgra[...,0] = rendered_8u[...,2]
        rendered_8u_bgra[...,2] = rendered_8u[...,0]

        cv2.imwrite("results/rendered_splats.png", rendered_8u_bgra)
        print("Saved alpha-blended splatting to results/rendered_splats.png")
        print("\nDone.")

    ##############################
    # 12) Save final results
    ##############################
    results = {
        'fundamental_matrix': F_optimized,
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


########################
# 6) Run main if needed
########################
if __name__ == '__main__':
    main()
