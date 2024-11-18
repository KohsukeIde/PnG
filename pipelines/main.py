# src/main.py

import os
import sys
import pickle
import cv2
import numpy as np

from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
from src.reconstruction.initial_reconstruction_naive import perform_initial_reconstruction
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from src.reconstruction.visualization import visualize_reconstruction, save_points_to_ply, visualize_new_view_integration
from src.reconstruction.view_integration import integrate_new_view

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

def load_gaussians(pickle_path: str) -> tuple:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        original_gaussians = data["original_gaussians"]
        projected_gaussians = data["projected_gaussians"]
        viewmat = data["viewmat"]
        K = data["K"]
    return original_gaussians, projected_gaussians, viewmat, K

def main():
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs'
    gaussians1_path = os.path.join(data_dir_gmm, 'fitted_gaussians_22_1k.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, 'fitted_gaussians_23_1k.pkl')
    colmap_dir = os.path.join(data_dir, 'sparse/0')  # COLMAPのスパースディレクトリ
    images_dir = os.path.join(data_dir, 'images')  # 画像ディレクトリ

    image1_name = '0022.png'  
    image2_name = '0023.png' 

    # Gaussiansの読み込み
    _, gaussians1, _, _ = load_gaussians(gaussians1_path)
    _, gaussians2, _, _ = load_gaussians(gaussians2_path)
    # _, gaussians3,
    print(f"{gaussians1.means=}")

    # COLMAPからカメラと画像の情報を読み込み
    cameras = load_cameras_from_colmap(colmap_dir)
    images_data = load_images_from_colmap(colmap_dir)

    # 画像名から画像IDを取得
    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
    image1_id = image_name_to_id[image1_name]
    image2_id = image_name_to_id[image2_name]

    # 対応するカメラIDを取得
    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']

    # CameraModelのインスタンスを作成
    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)
    
    print(f"Camera 1 Intrinsic Matrix K:\n{camera1.K}")
    print(f"Camera 1 Rotation Matrix R_wc:\n{camera1.R_wc}")
    print(f"Camera 1 Translation Vector t_wc:\n{camera1.t_wc}")
    print(f"Camera 1 Rotation Matrix R_cw:\n{camera1.R_cw}")
    print(f"Camera 1 Translation Vector t_cw:\n{camera1.t_cw}")
    print(f"Camera 1 Projection Matrix P:\n{camera1.P}")
    
    # Optimal Transport Solver
    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # 初期再構成
    points_3d, inlier_matches, pts1_inliers, pts2_inliers = perform_initial_reconstruction(
        gaussians1, gaussians2, camera1, camera2, transport_matrix
    )

    if points_3d.size > 0:
        # インライアマッチングのインデックスを取得
        indices_gaussians1 = [i for i, _ in inlier_matches]
        colors = gaussians1.rgb[indices_gaussians1]  # shape: (N, 3), 値は0-255の範囲
        save_points_to_ply(points_3d, filename='reconstructed_points.ply', colors=colors)
        print("Reconstructed 3D points saved to 'reconstructed_points.ply'.")
    else:
        print("No 3D points to save.")
    
    img1_path = os.path.join(images_dir, image1_name)
    img2_path = os.path.join(images_dir, image2_name)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 可視化の実行
    visualize_reconstruction(points_3d, img1, img2, pts1_inliers, pts2_inliers)
    
    # add new view
    updated_3d_points, new_matches = integrate_new_view(
        points_3d,  # 既存の3D点群
        gaussians2,  # 新しい画像のGaussians
        camera2,     # 新しいカメラ
        [camera1]    # 既存のカメラリスト
    )

    # visualization
    img2_path = os.path.join(images_dir, image2_name)
    visualize_new_view_integration(
        img2_path,
        points_3d,
        updated_3d_points,
        new_matches,
        gaussians2,
        camera2
    )

    # save updated pcl
    if updated_3d_points.size > 0:
        save_points_to_ply(updated_3d_points, filename='updated_points.ply')
        print("Updated 3D points saved to 'updated_points.ply'.")
    else:
        print("No updated 3D points to save.")

if __name__ == '__main__':
    main()
