import os
import sys
import pickle
import argparse 
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch
from utils.saving.geometry_utils import save_ellipsoids_as_ply, save_point_cloud_as_ply


sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

def render_gaussians_alpha_blend(
    points_3d,
    covariances_3d,
    color_3d,
    alpha_3d,
    R_cam,
    t_cam,
    K,
    out_width,
    out_height,
    splat_radius_factor=3.0,
    transport=None
):
    """3Dガウスをアルファブレンド(Over)でレンダリングする関数
    
    手順:
        1) ガウスの中心深度 Z_c (カメラ座標系) が大きい順に並び替え (遠い->近い)
        2) 後ろから順にガウスをレンダリングし、アルファブレンドする
        alpha_composite: 
                C_out = C_new * A_new + C_in * (1 - A_new)
                A_out = A_in + A_new * (1 - A_in)
        3) 結果を (H,W,3) の color_img と (H,W) の alpha_img にして返す

    Args:
        points_3d (N,3)          : 3Dガウスの中心 (world座標)
        covariances_3d (N,3,3)   : 3Dガウスの共分散行列 (world座標)
        color_3d (N,3)           : 各ガウスの色 (0~1)
        alpha_3d (N,)            : 各ガウスの基準アルファ (0~1)
        R_cam, t_cam             : ワールド->カメラ変換 (3x3, (3,))
        K                        : カメラ内部パラメータ (3x3)
        out_width, out_height    : 出力画像サイズ
        splat_radius_factor (float):
            ガウス投影時の描画範囲を標準偏差の何倍にするか
        transport (N,) or None:
            各3Dガウスの "輸送量" や "重み"。
            Noneでない場合は alpha_3d に乗算してアルファを決定する。
            例: final_alpha[i] = clip( alpha_3d[i] * transport[i], 0, 1 )
    
    Returns:
        color_img (H,W,3): 最終的なカラー画像 (float32, 0~1)
        alpha_img (H,W)  : 最終的なアルファ画像 (float32, 0~1)
    """
    # 出力バッファ（カラー+アルファ）
    color_buffer = np.zeros((out_height, out_width, 3), dtype=np.float32)
    alpha_buffer = np.zeros((out_height, out_width),     dtype=np.float32)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    N = points_3d.shape[0]

    #---------- (1) ガウスを「奥(Z大) -> 手前(Z小)」の順にソート ----------
    z_list = []
    for i in range(N):
        X_w = points_3d[i]
        X_c = R_cam @ X_w + t_cam
        z_list.append((X_c[2], i))
    z_list.sort(key=lambda x: x[0], reverse=True)  # Z降順(奥->手前)

    # transportが与えられたら alpha_3d に乗算しておく
    # (クリップで [0,1] に収まるようにする)
    if transport is not None:
        alpha_final = np.minimum(alpha_3d * transport, 1.0)  # shape(N,)
        print("transport matrix used!")
    else:
        print("no transport!")
        alpha_final = alpha_3d.copy()

    #---------- (2) ソート順にガウスを描画(アルファブレンド) ----------
    for _, i in tqdm(z_list, desc="Rendering Gaussians (alpha blend)"):
        X_w = points_3d[i]
        Sigma_3 = covariances_3d[i]
        rgb     = color_3d[i]
        alpha_i = alpha_final[i]  # 輸送量を掛けたアルファ

        # カメラ座標に変換
        X_c = R_cam @ X_w + t_cam
        z_c = X_c[2]
        # Zが正でない(背面)はスキップ
        if z_c <= 1e-8:
            continue

        # 2D投影座標 (u,v)
        u = fx*(X_c[0]/z_c) + cx
        v = fy*(X_c[1]/z_c) + cy
        
        px_center = int(np.round(u))
        py_center = int(np.round(v))

        # 画面外かどうかチェック
        if not (0 <= px_center < out_width and 0 <= py_center < out_height):
            # bounding boxの一部が可視領域に入るかもしれないので、ここでは一応続行
            pass

        # カメラ座標系でのガウス共分散
        Sigma_cam = R_cam @ Sigma_3 @ R_cam.T

        # ヤコビアンで 2D共分散行列 Sigma_2D を算出
        X, Y, Z = X_c
        J = np.array([
            [fx/Z,   0.0,    -fx*X/(Z**2)],
            [0.0,    fy/Z,   -fy*Y/(Z**2)]
        ], dtype=np.float32)
        
        Sigma_2D = J @ Sigma_cam @ J.T
        e_vals, _ = np.linalg.eig(Sigma_2D)
        e_vals = np.clip(e_vals, 1e-12, None)
        std_x = np.sqrt(e_vals[0])
        std_y = np.sqrt(e_vals[1])

        # スプラット描画範囲
        radius_x = int(np.ceil(std_x * splat_radius_factor))
        radius_y = int(np.ceil(std_y * splat_radius_factor))

        min_x = max(px_center - radius_x, 0)
        max_x = min(px_center + radius_x, out_width  - 1)
        min_y = max(py_center - radius_y, 0)
        max_y = min(py_center + radius_y, out_height - 1)

        inv_Sigma_2D = np.linalg.inv(Sigma_2D)

        # (min_x..max_x, min_y..max_y) のピクセルに対してガウス値を計算して Overブレンド
        for py in range(min_y, max_y + 1):
            dy = py - v
            for px in range(min_x, max_x + 1):
                dx = px - u
                disp = np.array([dx, dy], dtype=np.float32)
                val = disp @ inv_Sigma_2D @ disp
                gauss_val = np.exp(-0.5 * val)

                # blend_alpha = gauss_val * (輸送量を掛けたα_i)
                blend_alpha = gauss_val * alpha_i
                # 最大1にクリップ
                if blend_alpha > 1.0:
                    blend_alpha = 1.0
                # ほとんど寄与しない場合はスキップ（高速化）
                if blend_alpha <= 1e-8:
                    continue

                # 現状バッファの色(A_in, C_in)を取り出す
                C_in = color_buffer[py, px]
                A_in = alpha_buffer[py, px]

                # Overブレンド
                A_new = blend_alpha
                C_new = rgb
                A_out = A_in + A_new * (1.0 - A_in)
                if A_out > 1e-8:
                    # C_out = (C_new*A_new + C_in*A_in*(1 - A_new)) / A_out
                    C_out = (C_new * A_new + C_in * A_in * (1 - A_new)) / A_out
                else:
                    C_out = C_in

                # 書き戻し
                color_buffer[py, px] = C_out
                alpha_buffer[py, px] = A_out

    return color_buffer, alpha_buffer

def load_nerf_intrinsics(data_dir: str) -> np.ndarray:
    """NeRF形式のカメラ内部パラメータを読み込む関数
    
    Args:
        data_dir: NeRFデータディレクトリのパス
        
    Returns:
        K: 3x3カメラ内部パラメータ行列
    """
    transforms_file = os.path.join(data_dir, 'transforms.json')
    
    # transformsファイルが存在しない場合はエラー
    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f"NeRF transforms file not found at {transforms_file}")
    
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    
    # カメラパラメータを抽出
    H = transforms.get('h', 800)
    W = transforms.get('w', 800)
    
    # 焦点距離を取得（angle_xから計算することもある）
    if 'fl_x' in transforms and 'fl_y' in transforms:
        fx = transforms['fl_x']
        fy = transforms['fl_y']
    elif 'camera_angle_x' in transforms:
        # camera_angle_xから焦点距離を計算
        angle_x = transforms['camera_angle_x']
        fx = 0.5 * W / np.tan(0.5 * angle_x)
        fy = fx
    else:
        raise ValueError("Could not find camera focal length information in transforms.json")
    
    # 主点座標（通常は画像中心）
    cx = transforms.get('cx', W/2)
    cy = transforms.get('cy', H/2)
    
    # カメラ内部パラメータ行列
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"Loaded NeRF camera intrinsics: K=\n{K}")
    return K

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
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/textureless_32gs_5kiter",
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
        default="0026_fitted_gaussians.pkl",
        help="Filename of the first fitted Gaussians pickle."
    )
    parser.add_argument(
        "--gaussians2_filename",
        type=str,
        default="0095_fitted_gaussians.pkl",
        help="Filename of the second fitted Gaussians pickle."
    )
    parser.add_argument(
        "--use_nerf_intrinsics",
        action="store_true",
        default=False,
        help="Use NeRF format camera intrinsics instead of COLMAP intrinsics."
    )
    parser.add_argument(
        "--nerf_transforms_dir",
        type=str,
        default=None,
        help="Path to the directory containing NeRF transforms.json file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save output files."
    )

    return parser.parse_args()

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
    # 3) Load camera + COLMAP/NeRF(内部パラメータ) info
    ##############################
    
    # NeRF形式のカメラパラメータが指定されている場合は、それを使用
    if args.use_nerf_intrinsics:
        nerf_transforms_dir = args.nerf_transforms_dir or data_dir
        K_nerf = load_nerf_intrinsics(nerf_transforms_dir)
        print("Using NeRF intrinsics instead of COLMAP intrinsics")
        # NeRF intrinsicsでGaussians fitted時のK1, K2を上書き
        K1 = K_nerf
        K2 = K_nerf
        
        # カメラ外部パラメータはCOLMAPから読み込む（まだ必要）
        cameras = load_cameras_from_colmap(colmap_dir)
        images_data = load_images_from_colmap(colmap_dir)
        
        image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
        image1_id = image_name_to_id.get(image1_name)
        image2_id = image_name_to_id.get(image2_name)
        if image1_id is None or image2_id is None:
            print(f"Error: {image1_name} or {image2_name} not found in COLMAP.")
            sys.exit(1)
            
        camera1 = CameraModel(cameras[images_data[image1_id]['camera_id']], image1_id, images_data)
        camera2 = CameraModel(cameras[images_data[image2_id]['camera_id']], image2_id, images_data)
        
        # カメラ内部パラメータをNeRFのものに置き換え
        camera1.K = K1
        camera2.K = K2
    else:
        # 従来通りCOLMAPからカメラパラメータを読み込む
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
        lambda_epipolar=1.0,
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
        
    h_dummy = np.eye(3)
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, h_dummy)
    
    print("\n--- Identifying Source Gaussians ---")
    reconstructor.identify_source_gaussians(
        transport_matrix=transport_matrix_np,
        auto_threshold=False
    )
    ##############################
    # 7) Triangulate
    ##############################
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
    R1=np.eye(3), 
    t1=np.zeros(3),
    R2=R_est,
    t2=t_optimized
    #dont delete any gaussian (UOTの枠組みでthresholdは必要なくなったので)
    threshold = 0.0
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold)
    points_3d = reconstructor.points_3d
    print(f"\nTriangulated {points_3d.shape[0]} 3D points")

    ##############################
    # 8) Compute Covariances with Volume Prior
    ##############################
    print("\n--- Computing 3D Gaussian Covariances with Volume Prior ---")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=1.0, target_volume=target_volume)

    # 結果保存ディレクトリの作成
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    ply_points_out = os.path.join(output_dir, 'triangulated_points.ply')
    os.makedirs(output_dir, exist_ok=True)
    
    camera_params_list = [(np.eye(3), np.zeros(3)), (R_est, t_optimized)]
    save_point_cloud_as_ply(points_3d, ply_points_out, camera_params=camera_params_list)

    ##############################
    # 9) Compute color & alpha
    ##############################
    print("\n--- Computing 3D Gaussian Colors & Alphas ---")
    reconstructor.compute_3d_gaussian_colors(color_mode="average")
    reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")

    ##############################
    # 10) Build ellipsoids => PLY
    ##############################
    ply_out = os.path.join(output_dir, '3d_gaussians_ellipsoids.ply')
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
    # camera frustum from colmap 
    # R1 = camera1.R_wc  # world->camera
    # t1 = camera1.t_wc
    # R2 = camera2.R_wc
    # t2 = camera2.t_wc
    
    # actual camera frustum coord
    R1 = np.eye(3)
    t1= np.zeros(3)
    R2 = R_est
    t2 = t_optimized

    camera_params_list = [(R1, t1), (R2, t2)]
    
    ply_out = os.path.join(output_dir, '3d_gaussians_ellipsoids_withCams.ply')
    save_ellipsoids_as_ply(
        points_3d=reconstructor.points_3d,
        covariances_3d=reconstructor.covariances_3d,
        colors_3d=reconstructor.color_3d,
        alphas_3d=reconstructor.alpha_3d,
        filename=ply_out,
        camera_params=camera_params_list,
        use_alpha=True
    )

    ##############################
    # 11) (Optional) Project 3D Gaussians back to 2D for debug
    ##############################
    if True:
        print("\n--- Rendering 3D Gaussians back into both camera views (alpha-blend) ---")

        transport_values = None
        if hasattr(reconstructor, 'transport_values') and len(reconstructor.transport_values) > 0:
            transport_values = reconstructor.transport_values
            print(f"Using transport values from triangulation: min={transport_values.min():.4f}, "
                f"max={transport_values.max():.4f}, mean={transport_values.mean():.4f}")
        else:
            print("No transport values available, using default alpha values only.")
            sys.exit(1)

        # First camera rendering
        print("Rendering from camera 1 viewpoint...")
        R_cam1 = np.eye(3)  # Camera 1 is our reference frame
        t_cam1 = np.zeros(3)

        out_width1 = int(camera1.K[0,2]*2)
        out_height1 = int(camera1.K[1,2]*2)

        mixture_img1, coverage_img1 = render_gaussians_alpha_blend(
            points_3d=reconstructor.points_3d,
            covariances_3d=reconstructor.covariances_3d,
            color_3d=reconstructor.color_3d,
            alpha_3d=reconstructor.alpha_3d,
            R_cam=R_cam1,
            t_cam=t_cam1,
            K=camera1.K,
            out_width=out_width1,
            out_height=out_height1,
            transport=transport_values 
        )
            
        rendered_rgba1 = np.zeros((out_height1, out_width1, 4), dtype=np.float32)
        rendered_rgba1[..., :3] = mixture_img1
        rendered_rgba1[..., 3] = coverage_img1

        rendered_8u1 = np.clip(rendered_rgba1*255.0, 0, 255).astype(np.uint8)
        
        rendered_8u_bgra1 = rendered_8u1.copy()
        rendered_8u_bgra1[...,0] = rendered_8u1[...,2]
        rendered_8u_bgra1[...,2] = rendered_8u1[...,0]

        cv2.imwrite(os.path.join(output_dir, "rendered_splats_cam1.png"), rendered_8u_bgra1)
        print(f"Saved alpha-blended splatting for camera 1 to {os.path.join(output_dir, 'rendered_splats_cam1.png')}")

        # Second camera rendering
        print("Rendering from camera 2 viewpoint...")
        R_cam2 = R2  # Camera 2's rotation relative to world
        t_cam2 = t2  # Camera 2's translation relative to world

        out_width2 = int(camera2.K[0,2]*2)
        out_height2 = int(camera2.K[1,2]*2)

        mixture_img2, coverage_img2 = render_gaussians_alpha_blend(
            points_3d=reconstructor.points_3d,
            covariances_3d=reconstructor.covariances_3d,
            color_3d=reconstructor.color_3d,
            alpha_3d=reconstructor.alpha_3d,
            R_cam=R_cam2,
            t_cam=t_cam2,
            K=camera2.K,
            out_width=out_width2,
            out_height=out_height2,
            transport=transport_values 
        )
            
        rendered_rgba2 = np.zeros((out_height2, out_width2, 4), dtype=np.float32)
        rendered_rgba2[..., :3] = mixture_img2
        rendered_rgba2[..., 3] = coverage_img2

        rendered_8u2 = np.clip(rendered_rgba2*255.0, 0, 255).astype(np.uint8)
        
        rendered_8u_bgra2 = rendered_8u2.copy()
        rendered_8u_bgra2[...,0] = rendered_8u2[...,2]
        rendered_8u_bgra2[...,2] = rendered_8u2[...,0]

        cv2.imwrite(os.path.join(output_dir, "rendered_splats_cam2.png"), rendered_8u_bgra2)
        print(f"Saved alpha-blended splatting for camera 2 to {os.path.join(output_dir, 'rendered_splats_cam2.png')}")
        print("\nDone.")

    # ##############################
    # # 12) Save final results
    # ##############################
    # results = {
    #     'fundamental_matrix': F_optimized,
    #     'cost_matrix': cost_matrix.cpu().numpy(),
    #     'transport_matrix': transport_matrix_np,
    #     'camera1_K': K1,
    #     'camera2_K': K2,
    #     'points_3d': points_3d,
    #     'covariances_3d': reconstructor.covariances_3d,
    #     'color_3d': reconstructor.color_3d,
    #     'alpha_3d': reconstructor.alpha_3d,
    #     'transport_values': getattr(reconstructor, 'transport_values', None), 
    #     'source_gaussians1': reconstructor.source_gaussians1,
    #     'source_gaussians2': reconstructor.source_gaussians2,
    #     'source_gaussians1_data': getattr(reconstructor, 'source_gaussians1_data', None),
    #     'source_gaussians2_data': getattr(reconstructor, 'source_gaussians2_data', None),
    #     'camera_params_list': camera_params_list,  
    #     'R1': R1,
    #     't1': t1,
    #     'R2': R2,
    #     't2': t2,
    #     'camera1_R': R1,
    #     'camera1_t': t1,
    #     'camera2_R': R2,
    #     'camera2_t': t2,
    #     'existing_3d_gaussians': [
    #         {
    #             "center": points_3d[i],
    #             "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # Default unit quaternion (これいらないかも)
    #             "scale3d": np.sqrt(np.maximum(np.linalg.eigvalsh(reconstructor.covariances_3d[i]), 1e-10)),
    #             "color": reconstructor.color_3d[i],
    #             "alpha": reconstructor.alpha_3d[i]
    #         }
    #         for i in range(len(points_3d))
    #     ]
    # }
    
    # out_pkl = os.path.join(output_dir, 'initial_3dgs_results.pkl')
    # with open(out_pkl, 'wb') as f:
    #     pickle.dump(results, f)

    # print(f"\nSaved pipeline results to {out_pkl}")
    # print("\nDone.")


if __name__ == '__main__':
    main()