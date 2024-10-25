# src/main.py

import os
import sys
import pickle
import cv2


# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
from src.reconstruction.initial_reconstruction import perform_initial_reconstruction, visualize_reconstruction, save_points_to_ply
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap

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
    
    # print(f"Camera 2 Intrinsic Matrix K:\n{camera2.K}")
    # print(f"Camera 2 Rotation Matrix R:\n{camera2.R}")
    # print(f"Camera 2 Translation Vector t:\n{camera2.t}")
    # print(f"Camera 2 Projection Matrix P:\n{camera2.P}")

    # Optimal Transport Solver
    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # Perform initial reconstruction
    points_3d, inlier_matches, pts1_inliers, pts2_inliers = perform_initial_reconstruction(
        gaussians1, gaussians2, camera1, camera2, transport_matrix
    )

    # Check if points_3d is non-empty
    if points_3d.size > 0:
        # Extract colors from the first image's Gaussians using the indices from inlier_matches
        indices_gaussians1 = [i for i, _ in inlier_matches]
        colors = gaussians1.rgb[indices_gaussians1]  # Assuming gaussians1.rgb exists
        save_points_to_ply(points_3d, filename='reconstructed_points.ply', colors=colors)
        print("Reconstructed 3D points saved to 'reconstructed_points.ply'.")
    else:
        print("No 3D points to save.")
    
    img1_path = os.path.join(images_dir, image1_name)
    img2_path = os.path.join(images_dir, image2_name)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    visualize_reconstruction(points_3d, inlier_matches, img1, img2, pts1_inliers, pts2_inliers)
    
    

if __name__ == '__main__':
    main()
