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

# インポート部分を更新: 共通ユーティリティ関数を使用
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch
from utils.saving.geometry_utils import save_ellipsoids_as_ply


sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

# Import your reconstructor that can compute 3D covariances (with volume prior, color, alpha)
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

# 元のsample_ellipsoid_vertices_and_faces, create_camera_frustum_mesh, save_ellipsoids_as_plyの実装を削除し、
# 代わりにutilsからインポートしたものを使用

# 以下の関数は残す（ジオメトリ保存とは無関係のため）
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

# メイン部分は基本的に変更なし
def main():
    args = parse_args()

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
    ply_out = os.path.join('results', '3d_gaussians_ellipsoids.ply')
    save_ellipsoids_as_ply(
        points_3d=reconstructor.points_3d,
        covariances_3d=reconstructor.covariances_3d,
        colors_3d=reconstructor.color_3d,
        alphas_3d=reconstructor.alpha_3d,
        filename=ply_out,
        use_alpha=True
    )
    
    ##############################
    # 10) Build ellipsoids => PLY (with camera frustums)
    ##############################
    # カメラフラスタム付きのPLYを生成 - 共通ユーティリティ関数を使用
    R1 = camera1.R_wc  # world->camera
    t1 = camera1.t_wc
    R2 = camera2.R_wc
    t2 = camera2.t_wc
    
    # カメラパラメータリスト
    camera_params = [(R1, t1), (R2, t2)]
    
    ply_out = os.path.join('results', '3d_gaussians_ellipsoids_withCams.ply')
    save_ellipsoids_as_ply(
        points_3d=reconstructor.points_3d,
        covariances_3d=reconstructor.covariances_3d,
        colors_3d=reconstructor.color_3d,
        alphas_3d=reconstructor.alpha_3d,
        filename=ply_out,
        camera_params=camera_params,
        use_alpha=True
    )

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


if __name__ == '__main__':
    main()