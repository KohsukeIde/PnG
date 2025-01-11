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

########################
# 0) Import your modules
########################
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

# Import your reconstructor that can compute 3D covariances (with volume prior, color, alpha)
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

########################
# 1) Helper function: sample_ellipsoid_vertices_and_faces
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
# 2) Helper function: Save ellipsoids as PLY (with color, alpha optional)
########################
def save_ellipsoids_as_ply(all_vertices, all_faces, filename, use_alpha=False):
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
# 3) Helper function: Save 3D points as PLY
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
# 4) OPTIONAL: Project 3D Gaussians back to 2D
########################
def render_gaussians_to_2d_splat(
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
    Render 3D Gaussians into a 2D image with alpha blending (splatting).
    - points_3d (N,3): 3D means in world coords
    - covariances_3d (N,3,3): 3D cov in world coords
    - color_3d (N,3): color in [0..1]
    - alpha_3d (N,): alpha in [0..1]
    - R_cam, t_cam: extrinsic transform from world->camera
    - K (3,3): intrinsics
    - out_width, out_height: size of output image
    - splat_radius_factor: #std dev radii in 2D bounding region

    Returns:
        A float32 RGBA image (H,W,4) in [0..1].
    """
    # Create an RGBA buffer: shape (H, W, 4)
    rendered_rgba = np.zeros((out_height, out_width, 4), dtype=np.float32)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    for i in range(points_3d.shape[0]):
        X_w = points_3d[i]
        Sigma_3 = covariances_3d[i]
        rgb = color_3d[i]     # e.g. [r, g, b]
        a3 = alpha_3d[i]      # single alpha

        # World->camera
        X_c = R_cam @ X_w + t_cam
        if X_c[2] < 1e-6:
            continue  # behind camera or degenerate

        # Project to 2D
        u = fx * (X_c[0]/X_c[2]) + cx
        v = fy * (X_c[1]/X_c[2]) + cy

        px_center = int(np.round(u))
        py_center = int(np.round(v))

        # Skip if center is fully out of the image
        if px_center<0 or px_center>=out_width or py_center<0 or py_center>=out_height:
            continue

        # Cov in camera coords => Sigma_cam = R_cam * Sigma_3 * R_cam^T
        Sigma_cam = R_cam @ Sigma_3 @ R_cam.T

        # Local Jacobian J (2x3)
        X, Y, Z = X_c
        J = np.array([
            [fx/Z,    0.0,  -fx*X/(Z**2)],
            [0.0,    fy/Z,  -fy*Y/(Z**2)]
        ], dtype=np.float32)

        # 2D covariance
        Sigma_2D = J @ Sigma_cam @ J.T
        e_vals, e_vecs = np.linalg.eig(Sigma_2D)
        e_vals = np.clip(e_vals, 1e-12, None)
        std_x = np.sqrt(e_vals[0])
        std_y = np.sqrt(e_vals[1])

        # bounding region in pixel coords
        radius_x = int(np.ceil(std_x*splat_radius_factor))
        radius_y = int(np.ceil(std_y*splat_radius_factor))

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
                gauss_val = np.exp(-0.5*val)  # unnormalized, but good enough for alpha splat

                alpha_local = gauss_val * a3  # alpha coverage

                old_rgba = rendered_rgba[py, px]
                old_rgb = old_rgba[:3]
                old_a   = old_rgba[3]

                new_rgb = rgb * gauss_val
                new_a   = alpha_local

                out_a = new_a + old_a*(1.0 - new_a)
                if out_a < 1e-8:
                    continue

                out_rgb = (new_rgb*new_a + old_rgb*old_a*(1.0-new_a)) / out_a

                rendered_rgba[py, px, :3] = out_rgb
                rendered_rgba[py, px, 3]  = out_a

    return rendered_rgba



########################
# 5) Main pipeline function
########################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ##############################
    # Data / paths
    ##############################
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/train_results'
    gaussians1_path = os.path.join(data_dir_gmm, '0022_fitted_gaussians.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, '0023_fitted_gaussians.pkl')
    colmap_dir = os.path.join(data_dir, 'sparse/0')

    image1_name = '0022.png'
    image2_name = '0023.png'

    ##############################
    # 1) Load Gaussians
    ##############################
    _, gaussians1, _, K1 = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, K2 = load_gaussians_torch(gaussians2_path, device)

    # 2) Calculate dynamic target volume based on image properties
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
        epsilon=0.1,
        lambda_mean=1.0,
        lambda_cov=1.0,
        lambda_color=1.0,
        lambda_alpha=1.0,
        device=device
    )

    ##############################
    # 5) Homography optimization
    ##############################
    solver.h = torch.eye(3, dtype=torch.float32, device=device)
    print("\n--- Optimizing Homography ---")
    solver.optimize_with_homography(max_iter=500, tol=1e-6)

    H_optimized = solver.h.detach().cpu().numpy()
    print("\nOptimized Homography matrix:\n", H_optimized)

    ##############################
    # 6) Final cost & unbalanced transport
    ##############################
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix(solver.h)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()

    ##############################
    # 7) Triangulate
    ##############################
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H_optimized)
    reconstructor.compute_camera_matrices_from_homography()
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
        c3 = reconstructor.color_3d[i]   # [r,g,b] in [0..1]
        a3 = reconstructor.alpha_3d[i]   # alpha in [0..1]

        raw_vertices, faces = sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta, n_phi)
        extended_vertices = []
        for vert in raw_vertices:
            # (x, y, z, r, g, b, a)
            combo = np.concatenate([vert, c3, [a3]])
            extended_vertices.append(combo)

        all_vertices.append(extended_vertices)
        all_faces.append(faces)

    # Save ellipsoids with alpha channel
    ply_out = os.path.join('results', '3d_gaussians_ellipsoids.ply')
    # save_ellipsoids_as_ply(all_vertices, all_faces, ply_out, use_alpha=True)

    ##############################
    # 11) (Optional) Project 3D Gaussians back to 2D for debug
    ##############################
    # e.g. project onto camera1's view
    if True:  # set to False if you don't want to do it
        # 11) Optionally render 3D Gaussians to 2D
        print("\n--- Rendering 3D Gaussians back into camera1's 2D image (alpha-blend) ---")

        # Suppose camera1 has R, t => If not, fallback to identity or from colmap extrinsics
        if hasattr(camera1, 'R') and hasattr(camera1, 't'):
            R_cam = camera1.R
            t_cam = camera1.t
        else:
            R_cam = np.eye(3)
            t_cam = np.zeros(3)

        # Output image resolution => e.g. match camera1's size
        out_width  = int(camera1.K[0,2]*2)  # if center is at K[0,2]
        out_height = int(camera1.K[1,2]*2)

        # Now call the splat function

        rendered_rgba = render_gaussians_to_2d_splat(
            points_3d     = reconstructor.points_3d,
            covariances_3d= reconstructor.covariances_3d,
            color_3d      = reconstructor.color_3d,
            alpha_3d      = reconstructor.alpha_3d,
            R_cam         = R_cam,
            t_cam         = t_cam,
            K             = camera1.K,
            out_width     = out_width,
            out_height    = out_height,
            splat_radius_factor=3.0
        )

        # Convert float RGBA [0..1] => 8-bit BGRA or RGBA
        rendered_8u = np.clip(rendered_rgba*255.0, 0, 255).astype(np.uint8)
        # If using OpenCV, typically BGR or BGRA => let's do RGBA->BGRA for saving
        # Make sure we have 4 channels => shape(H,W,4)
        # Then convert RGBA->BGRA so cv2 will not mix color
        rendered_8u_bgra = rendered_8u.copy()
        rendered_8u_bgra[...,0] = rendered_8u[...,2]  # swap R,B
        rendered_8u_bgra[...,2] = rendered_8u[...,0]

        cv2.imwrite("results/rendered_splats.png", rendered_8u_bgra)
        print("Saved alpha-blended splatting to results/rendered_splats.png")

        # done
        print("\nDone.")

    ##############################
    # 12) Save final results
    ##############################
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


########################
# 6) Run main if needed
########################
if __name__ == '__main__':
    main()
