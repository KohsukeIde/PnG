#!/usr/bin/env python

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstructor.viewpoint_extender import ViewpointExtender
from utils.gs_pkl_loader import load_gaussians_torch
from src.reconstructor.initial_3d_non_linear import build_covariance_3d
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.saving.geometry_utils import save_ellipsoids_as_ply


sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']
def extract_3d_gaussians_from_dict(gaussians_3d):
    """3DGS辞書リストからNumPy配列を抽出"""
    points_3d = np.array([g["center"] for g in gaussians_3d])
    
    covariances_3d = []
    for g in gaussians_3d:
        quat = g["quat"]
        scale_3d = g["scale3d"]
        cov = build_covariance_3d(quat, scale_3d)
        covariances_3d.append(cov)
    covariances_3d = np.array(covariances_3d)
    
    colors_3d = np.array([g["color"] for g in gaussians_3d])
    alphas_3d = np.array([g["alpha"] for g in gaussians_3d])
    
    return points_3d, covariances_3d, colors_3d, alphas_3d


def parse_args():
    """Parse command-line arguments for path configuration.
    """
    parser = argparse.ArgumentParser(description="Add a new viewpoint to an existing 3D Gaussian scene.")
    
    parser.add_argument(
        "--initial_3d_path",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/initial_3dgs_results.pkl",
        help="Path to the initial 3D reconstruction results pickle."
    )
    parser.add_argument(
        "--new_view_2d_path",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked/0024_fitted_gaussians.pkl",
        help="Path to the new view's fitted 2D Gaussians pickle."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results."
    )
    parser.add_argument(
        "--reference_idx",
        type=int,
        default=0,
        help="Index of the reference camera for projection."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum iterations for camera pose optimization."
    )
    parser.add_argument(
        "--transport_threshold",
        type=float,
        default=1e-6,
        help="Threshold for transport values in optimal transport."
    )
    parser.add_argument(
        "--target_volume",
        type=float,
        default=1.0,
        help="Target volume for new 3D Gaussians."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63",
        help="Path to the dataset directory (for COLMAP data)."
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default="sparse/0",
        help="Relative path to COLMAP sparse directory."
    )
    parser.add_argument(
        "--image_name",
        type=str,
        default="0024.png",
        help="Name of the new image to add (must be in COLMAP data)."
    )
    
    parser.add_argument("--auto_threshold", action="store_true",
                   help="Automatically determine optimal transport threshold")

    return parser.parse_args()


def add_new_viewpoint():
    """Main function to add a new viewpoint to the scene.
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load initial 3D reconstruction
    print(f"Loading initial 3D reconstruction from {args.initial_3d_path}")
    with open(args.initial_3d_path, 'rb') as f:
        initial_data = pickle.load(f)
    
    # 2. COLMAP情報を読み込む
    K_new = None
    if args.data_dir is not None:
        colmap_dir = os.path.join(args.data_dir, args.colmap_dir)
        cameras = load_cameras_from_colmap(colmap_dir)
        images_data = load_images_from_colmap(colmap_dir)
        
        # 新しい画像のカメラ情報を取得
        image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
        new_image_id = image_name_to_id.get(args.image_name)
        if new_image_id is None:
            print(f"Error: {args.image_name} not found in COLMAP data.")
            sys.exit(1)
        
        camera_id = images_data[new_image_id]['camera_id']
        camera_model = CameraModel(cameras[camera_id], new_image_id, images_data)
        K_new = camera_model.K
        print(f"Loaded camera intrinsics from COLMAP for {args.image_name}")
    
    # 3. 新しい視点の2Dガウスを読み込む
    print(f"Loading new view 2D Gaussians from {args.new_view_2d_path}")
    _, new_2d_gaussians, _, K_from_pickle = load_gaussians_torch(args.new_view_2d_path, device)
    
    # カメラ内部パラメータの設定 (優先順位: COLMAP > pickle > initial_data)
    if K_new is None:
        if K_from_pickle is not None:
            K_new = K_from_pickle
            print("Using camera intrinsics from pickle file")
        elif 'camera1_K' in initial_data:
            K_new = initial_data['camera1_K']
            print("Using camera intrinsics from initial data")
        else:
            # ダミーの内部パラメータ
            K_new = np.array([
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0]
            ])
            print("Warning: Using default camera intrinsics")
    
    # 4. 既存の3Dガウスとカメラパラメータを抽出
    camera_params_list = []
    
    # 既存の3Dガウスを抽出
    if 'existing_3d_gaussians' in initial_data:
        gaussians_3d = initial_data['existing_3d_gaussians']
    else:
        # 3Dガウスパラメータが別々に保存されている場合
        points_3d = initial_data.get('points_3d', [])
        covariances_3d = initial_data.get('covariances_3d', [])
        colors_3d = initial_data.get('color_3d', initial_data.get('colors_3d', []))
        alphas_3d = initial_data.get('alpha_3d', initial_data.get('alphas_3d', []))
        
        # 3Dガウスの辞書リストを作成
        gaussians_3d = []
        for i in range(len(points_3d)):
            # 共分散行列から四元数とスケールを推定
            eigvals, eigvecs = np.linalg.eigh(covariances_3d[i])
            eigvals = np.maximum(eigvals, 1e-10)
            scales = np.sqrt(eigvals)
            
            # 簡易的にするために単位四元数を使用
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            gauss = {
                "center": points_3d[i],
                "quat": quat,
                "scale3d": scales,
                "color": colors_3d[i],
                "alpha": alphas_3d[i]
            }
            gaussians_3d.append(gauss)
    
    # カメラパラメータリストを取得
    if 'camera_params_list' in initial_data:
        camera_params_list = initial_data['camera_params_list']
    elif 'R1' in initial_data and 'R2' in initial_data:
        # R1, t1, R2, t2が個別に保存されている場合
        R1 = initial_data['R1']
        t1 = initial_data['t1']
        R2 = initial_data['R2']
        t2 = initial_data['t2']
        camera_params_list = [(R1, t1), (R2, t2)]
    elif 'camera1_R' in initial_data and 'camera2_R' in initial_data:
        # 別の命名規則
        R1 = initial_data['camera1_R']
        t1 = initial_data['camera1_t']
        R2 = initial_data['camera2_R']
        t2 = initial_data['camera2_t']
        camera_params_list = [(R1, t1), (R2, t2)]
    else:
        # デフォルト
        print("Warning: No camera parameters found in initial data. Using default.")
        R1 = np.eye(3)
        t1 = np.zeros(3)
        camera_params_list = [(R1, t1)]
    
    # 5. ViewpointExtenderを初期化
    print("Initializing ViewpointExtender...")
    extender = ViewpointExtender(
        existing_3d_gaussians=gaussians_3d,
        camera_params_list=camera_params_list,
        K_new=K_new,
        reference_camera_idx=args.reference_idx,
        threshold_reprojection=args.transport_threshold,
        device=device
    )
    
    # 6. 一連の処理を実行: カメラ姿勢推定 + 新規3Dガウス追加
    print("Running camera estimation and adding new Gaussians...")
    R_new, t_new = extender.integrate_new_view_and_gaussians(
        new_image_2d_gaussians=new_2d_gaussians,
        max_iterations=args.max_iterations,
        transport_threshold=args.transport_threshold,
        target_volume=args.target_volume,
        auto_threshold=args.auto_threshold  # 自動閾値フラグを追加
    )
    print(f"Estimated new camera parameters: R=\n{R_new}\nt={t_new}")
    print(f"Total 3D Gaussians after adding view: {len(extender.existing_3d_gaussians)}")
    
    # 7. 結果をPLYとして保存
    print("Saving results as PLY...")
    points_3d, covariances_3d, colors_3d, alphas_3d = extract_3d_gaussians_from_dict(extender.existing_3d_gaussians)
    ply_path = os.path.join(args.output_dir, "updated_3d_gaussians.ply")
    
    # カメラフラスタムも保存
    save_ellipsoids_as_ply(
        points_3d, 
        covariances_3d, 
        colors_3d, 
        alphas_3d, 
        ply_path,
        camera_params=extender.camera_params_list,
        use_alpha=True
    )
    
    # 8. 結果を保存
    print("Saving results...")
    results = {
        "existing_3d_gaussians": extender.existing_3d_gaussians,
        "camera_params_list": extender.camera_params_list,
        "K": K_new,
        "new_camera_R": R_new,
        "new_camera_t": t_new,
        "points_3d": points_3d,  # 互換性のために保持
        "covariances_3d": covariances_3d,
        "color_3d": colors_3d,
        "alpha_3d": alphas_3d
    }
    
    output_path = os.path.join(args.output_dir, "updated_3d_gaussians.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    add_new_viewpoint()