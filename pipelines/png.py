import os
import sys
import argparse
import pickle
import glob
import json
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from src.reconstructor.view_selector import ViewSelector
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor
from src.reconstructor.viewpoint_extender import ViewpointExtender
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from utils.gs_pkl_loader import load_gaussians_torch
from utils.saving.geometry_utils import save_ellipsoids_as_ply, save_gaussians_as_ply
from utils.export.export_utils import (
    export_colmap_format,
    export_gaussians_to_colmap_dir
)
from src.optimizer.bundle_adjuster import BundleAdjuster
from src.optimizer.observation_builder import ObservationBuilder

# Fix module import issues
sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

def parse_args():
    """Parse command-line arguments for the complete pipeline."""
    parser = argparse.ArgumentParser(description="Complete Perspective-n-Gaussian Pipeline")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/textureless",
        help="Path to the data directory containing images and COLMAP data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--fitted_gaussians_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/textureless_32gs_5kiter",
        help="Directory containing fitted 2D Gaussians"
    )
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default="sparse/0",
        help="Relative path to COLMAP sparse directory"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=200,
        help="Size of the visual vocabulary for BoVW"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="sift",
        choices=["sift", "orb", "clip"],
        help="Type of features to extract for view selection (sift, orb, or clip)"
    )
    parser.add_argument(
        "--min_overlap",
        type=float,
        default=0.2,
        help="Minimum overlap ratio between views"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=0.7,
        help="Maximum overlap ratio between views (for diversity)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000,
        help="Maximum iterations for optimization"
    )
    parser.add_argument(
        "--transport_threshold",
        type=float,
        default=1e-6,
        help="Threshold for transport values"
    )
    parser.add_argument(
        "--target_volume",
        type=float,
        default=None,
        help="Target volume for 3D Gaussians (None for automatic calculation)"
    )
    parser.add_argument(
        "--auto_target_volume",
        action="store_true",
        default=True,
        help="Automatically calculate target volume based on image properties"
    )
    parser.add_argument(
        "--auto_threshold", 
        action="store_true", 
        default=True,
        help="Automatically determine optimal transport threshold"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip processing if output files already exist"
    )
    parser.add_argument(
        "--enable_ba",
        action="store_true",
        default=True,
        help="Enable Bundle Adjustment"
    )
    parser.add_argument(
        "--ba_iterations",
        type=int,
        default=20,
        help="Maximum iterations for Bundle Adjustment (must be more than 10)"
    )
    parser.add_argument(
        "--ba_skip_initial",
        action="store_true",
        default=False,
        help="Skip Bundle Adjustment after initial reconstruction"
    )
    parser.add_argument(
        "--ba_skip_incremental",
        action="store_true",
        default=False,
        help="Skip incremental Bundle Adjustment after adding each view"
    )
    parser.add_argument(
        "--ba_every_n_views",
        type=int,
        default=1,
        help="Perform incremental Bundle Adjustment every N views (1 = after every view)"
    )
    parser.add_argument(
        "--force_single_intrinsic",
        action="store_true",
        default=False,
        help="Force using a single intrinsic matrix for all cameras (useful when COLMAP data is incomplete)"
    )
    parser.add_argument(
        "--use_nerf_intrinsics",
        action="store_true",
        default=True,
        help="Use camera intrinsics from NeRF dataset's transforms_train.json"
    )
    parser.add_argument(
        "--ba_skip_final",
        action="store_true",
        default=False,
        help="Skip final global Bundle Adjustment"
    )
    parser.add_argument(
        "--use_sparse_set",
        action="store_true",
        default=False,
        help="Use a sparse subset of available images" # (ランダムではないので注意)
    )
    parser.add_argument(
        "--sparse_interval",
        type=int,
        default=2,
        help="Interval for sparse image set (e.g., 2 means use every 2nd image)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to use (None means use all available)" #(これも修正必須)
    )
    parser.add_argument(
        "--max_views_to_add",
        type=int,
        default=8,
        help="Maximum number of views to add after initial pair (None means no limit)"
    )
    
    return parser.parse_args()

def get_image_names(directory: str) -> List[str]:
    """Get image file names from directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_names = []
    
    for ext in image_extensions:
        image_names.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    
    # Extract only file names from paths
    image_names = [os.path.basename(path) for path in image_names]
    
    return sorted(image_names)

def load_nerf_intrinsics(data_dir: str) -> np.ndarray:
    """
    Load camera intrinsics from NeRF dataset's transforms_train.json file.
    
    Args:
        data_dir: Directory containing the transforms_train.json file
        
    Returns:
        np.ndarray: The camera intrinsic matrix K
    """
    json_path = os.path.join(data_dir, "transforms_train.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"NeRF transforms file not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            transforms = json.load(f)
        
        # NeRF datasets provide camera_angle_x which is the horizontal FOV in radians
        fov_x = transforms.get("camera_angle_x")
        
        if fov_x is None:
            raise ValueError("transforms_train.json does not contain camera_angle_x")
        
        # Get image dimensions - assume square images
        frame_path = transforms["frames"][0]["file_path"]
        frame_path = frame_path.replace("./train/", "")
        
        # Check both train and images directories
        img_path = os.path.join(data_dir, "train", f"{frame_path}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, "images", f"{frame_path}.png")
        
        if not os.path.exists(img_path):
            # Assume default NeRF resolution of 800x800
            width = height = 800
            print(f"Image not found, assuming default NeRF resolution of {width}x{height}")
        else:
            from PIL import Image
            img = Image.open(img_path)
            width, height = img.size
            print(f"Found image with dimensions {width}x{height}")
        
        # Calculate focal length from FOV
        # focal_length = (width / 2) / tan(fov_x / 2)
        focal_length = (width / 2) / np.tan(fov_x / 2)
        
        # Construct intrinsic matrix K
        K = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ])
        
        print(f"Loaded NeRF intrinsics with focal length: {focal_length:.2f}")
        print(f"K = \n{K}")
        
        return K
        
    except Exception as e:
        print(f"Error loading NeRF intrinsics: {e}")
        raise

def get_fitted_gaussians_info(directory: str) -> Dict[str, str]:
    """Get mappings between image names and their fitted Gaussian pkl files."""
    pkl_files = glob.glob(os.path.join(directory, "*.pkl"))
    result = {}
    
    for pkl_path in pkl_files:
        basename = os.path.basename(pkl_path)
        # Naming convention: "0001_fitted_gaussians.pkl" -> "0001.png"
        image_name = basename.replace("_fitted_gaussians.pkl", ".png")
        # Also handle other common naming conventions
        if not os.path.exists(os.path.join(os.path.dirname(directory), "images", image_name)):
            # Try alternative naming patterns
            for ext in ['.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                alt_name = image_name.replace('.png', ext)
                if os.path.exists(os.path.join(os.path.dirname(directory), "images", alt_name)):
                    image_name = alt_name
                    break
        
        result[image_name] = pkl_path
    
    return result

def select_initial_pair(
    image_dir: str,
    gaussian_files: Dict[str, str],
    vocab_size: int = 200,
    feature_type: str = "sift",
    min_overlap: float = 0.3,
    max_overlap: float = 0.7,
    available_images: List[str] = None
) -> Tuple[str, str]:
    """Select the best initial pair of images using Bag of Visual Words.
    
    The best pair should have:
    1. Good feature overlap (within min/max range)
    2. Rich features in both images
    3. Good spatial distribution of features
    
    Args:
        image_dir: Directory containing images
        gaussian_files: Dictionary mapping image names to their fitted Gaussian files
        vocab_size: Size of the visual vocabulary
        feature_type: Type of features to extract
        min_overlap: Minimum overlap ratio
        max_overlap: Maximum overlap ratio
        available_images: Optional list of available images to consider
    
    Returns:
        Tuple containing the names of the two selected images and the ViewSelector
    """
    print("\n--- Selecting Initial Image Pair ---")
    
    # Get available images that have fitted Gaussians
    available_images = [img for img in available_images if img in gaussian_files]
    
    if len(available_images) < 2:
        raise ValueError(f"Need at least 2 images with fitted Gaussians, found {len(available_images)}")
    
    # Initialize ViewSelector with specified feature type
    selector = ViewSelector(
        image_dir=image_dir,
        vocab_size=vocab_size,
        feature_type=feature_type,
        min_overlap_ratio=min_overlap,
        max_overlap_ratio=max_overlap
    )
    
    # Process available images to extract features
    image_paths = [os.path.join(image_dir, img) for img in available_images]
    selector.process_images(image_paths)
    
    # Calculate similarity matrix between all pairs
    similarity_matrix = np.zeros((len(available_images), len(available_images)))
    
    # Compute similarities based on the feature type
    if selector.feature_type == 'clip' and selector.clip_model is not None:
        # Using CLIP features for similarity
        valid_paths = [p for p in image_paths if p in selector.clip_features]
        
        if len(valid_paths) >= 2:
            clip_features = np.vstack([selector.clip_features[p] for p in valid_paths])
            clip_similarity = cosine_similarity(clip_features, clip_features)
            
            # Map to original indices
            for i, path1 in enumerate(valid_paths):
                idx1 = image_paths.index(path1)
                for j, path2 in enumerate(valid_paths):
                    if i >= j:  # Avoid redundant computation and self-comparison
                        continue
                    idx2 = image_paths.index(path2)
                    similarity = clip_similarity[i, j]
                    similarity_matrix[idx1, idx2] = similarity
                    similarity_matrix[idx2, idx1] = similarity
        else:
            print("Warning: Not enough images with valid CLIP features")
            print("Using random similarity values")
            similarity_matrix = np.random.rand(len(available_images), len(available_images))
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            np.fill_diagonal(similarity_matrix, 1.0)
    else:
        # Using traditional feature histograms for similarity
        valid_paths = [p for p in image_paths if p in selector.image_histograms]
        
        if len(valid_paths) >= 2:
            # Construct histogram vectors for valid paths
            histograms = np.vstack([selector.image_histograms[p] for p in valid_paths])
            hist_similarity = cosine_similarity(histograms, histograms)
            
            # Map to original indices
            for i, path1 in enumerate(valid_paths):
                idx1 = image_paths.index(path1)
                for j, path2 in enumerate(valid_paths):
                    if i >= j:
                        continue
                    idx2 = image_paths.index(path2)
                    similarity = hist_similarity[i, j]
                    similarity_matrix[idx1, idx2] = similarity
                    similarity_matrix[idx2, idx1] = similarity
        else:
            print("Warning: Not enough images with valid feature histograms")
            print("Using random similarity values")
            similarity_matrix = np.random.rand(len(available_images), len(available_images))
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
            np.fill_diagonal(similarity_matrix, 1.0)
    
    # Assign feature richness scores (default to 1.0 for all images)
    feature_scores = np.ones(len(available_images))

    best_pair = None
    best_score = -1
    
    for i in range(len(available_images)):
        for j in range(i+1, len(available_images)):
            similarity = similarity_matrix[i, j]
            
            # 類似度が範囲内にあるか確認
            if min_overlap <= similarity <= max_overlap:
                # スコア = 類似度 + 両方の画像の特徴点のrichness
                feature_richness = (feature_scores[i] + feature_scores[j]) / 2
                score = similarity * 0.5 + feature_richness * 0.5
                
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
    
    # 適切な範囲内のペアが見つからなかった場合、特徴点が多いペアを選択
    if best_pair is None:
        print("No pairs within specified overlap range, selecting based on feature richness")
        for i in range(len(available_images)):
            for j in range(i+1, len(available_images)):
                similarity = similarity_matrix[i, j]
                # 最低限の類似度を確保
                if similarity > 0.1:
                    feature_richness = (feature_scores[i] + feature_scores[j]) / 2
                    score = feature_richness
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)
    
    # それでもペアが見つからない場合は最初の2つを使用（テキスチャレスの場合はあり得る→局所特徴に依存しないように修正）
    if best_pair is None:
        print("No suitable pairs found, using first two images")
        best_pair = (0, 1)

    img1 = available_images[best_pair[0]]
    img2 = available_images[best_pair[1]]
    
    print(f"Selected initial pair: {img1} and {img2}")
    print(f"Similarity: {similarity_matrix[best_pair[0], best_pair[1]]:.4f}")
    
    # With the CLIP-based ViewSelector, we don't have keypoint counts anymore
    # Instead, we're using semantic similarity from CLIP embeddings
    
    return img1, img2, selector

def perform_bundle_adjustment(
    reconstruction_data: Dict,
    ba_iterations: int = 100, 
    device: torch.device = None,
    verbose: bool = True,
    save_dir: Optional[str] = None,
    use_staged: bool = True  # Use staged optimization like COLMAP
) -> Dict:
    """BA実行,カメラパラメータとGS中心位置を最適化 (COLMAP like approach)
    
    Args:
        reconstruction_data: 再構成データ辞書
        ba_iterations: BAの最大イテレーション数
        device: 計算デバイス
        verbose: 詳細な出力を表示するかどうか
        save_dir: 結果を保存するディレクトリ
        use_staged: COLMAP風の段階的最適化を使用するか
        
    Returns:
        Dict: 更新された再構成データ
    """
    print("\n--- Performing Bundle Adjustment (COLMAP-like) ---")
    
    # 観測データ構築 - point3D_observations 構造を使用
    observation_map = ObservationBuilder.build_observation_map(reconstruction_data)
    
    if not observation_map:
        print("No valid observations found for BA. Skipping.")
        return reconstruction_data
    
    # 観測データをBundleAdjusterの形式に変換
    num_cameras = len(reconstruction_data["camera_params_list"])
    match_points_2d = ObservationBuilder.convert_to_match_points_2d(
        observation_map, num_cameras
    )
    
    # 最低限必要な観測数をチェック
    total_obs = sum(len(obs) for obs in match_points_2d)
    if total_obs < 10:
        print(f"Not enough observations ({total_obs}) for meaningful Bundle Adjustment. Skipping.")
        return reconstruction_data
    
    # 各カメラの内部パラメータリスト
    intrinsics_list = []
    for cam_idx in range(num_cameras):
        # カメラ固有のKがあればそれを使用
        cam_key = f"camera{cam_idx+1}_K"
        if cam_key in reconstruction_data:
            intrinsics_list.append(reconstruction_data[cam_key])
        elif "K" in reconstruction_data:
            intrinsics_list.append(reconstruction_data["K"])
        else:
            # Fallback to first camera's K
            intrinsics_list.append(reconstruction_data.get("camera1_K", np.eye(3)))
    
    image_names = reconstruction_data.get("used_images", None)
    
    # COLMAP風のロバスト損失スケールを初期値として使用
    initial_loss_scale = 2.0
    
    ba = BundleAdjuster(
        points_3d=reconstruction_data["points_3d"],
        camera_params_list=reconstruction_data["camera_params_list"],
        match_points_2d=match_points_2d,
        intrinsics_list=intrinsics_list,
        image_names=image_names,
        use_robust_loss=True,
        loss_scale=initial_loss_scale  # COLMAP standard scale
    )
    
    # BA最適化実行 - COLMAPスタイルの段階的最適化サポート
    ba_results = ba.optimize(
        n_iterations=ba_iterations, 
        verbose=verbose,
        use_staged=use_staged  # 段階的最適化を使用
    )
    
    if ba_results["success"]:
        # 段階的BAの場合は結果メッセージを調整
        if use_staged:
            print(f"Staged Bundle Adjustment completed successfully.")
        else:
            print(f"Bundle Adjustment completed successfully.")
            
        # 結果のRMSE情報を表示    
        if "initial_rmse" in ba_results:
            print(f"Initial RMSE: {ba_results['initial_rmse']:.4f} pixels")
        print(f"Final RMSE: {ba_results.get('final_rmse', 'N/A'):.4f} pixels")
        
        # 再構成データを更新
        reconstruction_data["camera_params_list"] = ba_results["optimized_cameras"]
        reconstruction_data["points_3d"] = ba_results["optimized_points"]
        
        # 最適化されたGS中心を反映
        for i, point in enumerate(ba_results["optimized_points"]):
            if i < len(reconstruction_data["existing_3d_gaussians"]):
                reconstruction_data["existing_3d_gaussians"][i]["center"] = point
            
        # 結果保存
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # COLMAPフォーマットで出力
            colmap_ba_dir = os.path.join(save_dir, "colmap_ba")
            os.makedirs(colmap_ba_dir, exist_ok=True)
            
            # Get valid point indices (points with at least 2 observations)
            valid_point_indices = ba.get_valid_point_indices(min_observations=2)
            
            # 観測数情報を出力
            if verbose:
                point_obs_counts = ba.get_point_observation_counts()
                points_with_2plus = sum(1 for count in point_obs_counts if count >= 2)
                print(f"Points with 2+ observations: {points_with_2plus} out of {len(point_obs_counts)}")
                
            # Use existing export functions
            export_colmap_format(
                output_dir=colmap_ba_dir,
                points_3d=reconstruction_data["points_3d"],
                camera_params_list=reconstruction_data["camera_params_list"],
                match_points_2d=ba.match_points_2d,
                intrinsics_list=ba.intrinsics_list,
                image_names=ba.image_names
            )
            
            # Filter points for PLY export - COLMAPスタイルで観測数2以上のみ使用
            valid_points = [reconstruction_data["points_3d"][i] for i in valid_point_indices]
            valid_colors = [reconstruction_data["color_3d"][i] for i in valid_point_indices] if "color_3d" in reconstruction_data else None
            valid_covariances = [reconstruction_data["covariances_3d"][i] for i in valid_point_indices] if "covariances_3d" in reconstruction_data else None
            valid_alphas = [reconstruction_data["alpha_3d"][i] for i in valid_point_indices] if "alpha_3d" in reconstruction_data else None
            
            # PLYとして保存
            ply_path = os.path.join(save_dir, "ba_optimized.ply")
            if len(valid_points) > 0:  # Only save if we have valid points
                save_ellipsoids_as_ply(
                    points_3d=np.array(valid_points),
                    covariances_3d=np.array(valid_covariances) if valid_covariances else None,
                    colors_3d=np.array(valid_colors) if valid_colors else None,
                    alphas_3d=np.array(valid_alphas) if valid_alphas else None,
                    filename=ply_path,
                    camera_params=reconstruction_data["camera_params_list"],
                    use_alpha=True
                )
            
            print(f"BA results saved to {save_dir}")
    else:
        print(f"Bundle Adjustment failed: {ba_results.get('message', 'Unknown error')}")
    
    return reconstruction_data

def perform_initial_reconstruction(
    img1_name: str,
    img2_name: str,
    data_dir: str,
    colmap_dir: str,
    fitted_gaussians_dir: str,
    output_dir: str,
    max_iterations: int = 1000,
    target_volume: float = None,
    auto_target_volume: bool = True,
    device: torch.device = None,
    enable_ba: bool = True,
    ba_iterations: int = 10,
    nerf_K: Optional[np.ndarray] = None
) -> Dict:
    """Perform initial 3D reconstruction from two views.
    
    Args:
        img1_name: First image name
        img2_name: Second image name
        data_dir: Data directory path
        colmap_dir: COLMAP directory path
        fitted_gaussians_dir: Directory with fitted Gaussians
        output_dir: Output directory
        max_iterations: Maximum optimization iterations
        target_volume: Target volume for 3D Gaussians
        device: Computation device
        enable_ba: Whether to perform Bundle Adjustment
        ba_iterations: Maximum BA iterations
    
    Returns:
        Dictionary with reconstruction results
    """
    print("\n--- Performing Initial 3D Reconstruction ---")
    

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to fitted Gaussians
    gaussians1_path = os.path.join(fitted_gaussians_dir, f"{img1_name.split('.')[0]}_fitted_gaussians.pkl")
    gaussians2_path = os.path.join(fitted_gaussians_dir, f"{img2_name.split('.')[0]}_fitted_gaussians.pkl")
    
    # Load fitted Gaussians
    _, gaussians1, _, K1 = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, K2 = load_gaussians_torch(gaussians2_path, device)
    
    # Attempt to load COLMAP data
    colmap_path = os.path.join(data_dir, colmap_dir)
    has_colmap_data = os.path.exists(colmap_path) and os.path.isdir(colmap_path)
    
    if has_colmap_data:
        try:
            cameras = load_cameras_from_colmap(colmap_path)
            images_data = load_images_from_colmap(colmap_path)
            print(f"Successfully loaded COLMAP data with {len(cameras)} cameras and {len(images_data)} images")
        except Exception as e:
            print(f"Error loading COLMAP data: {e}")
            has_colmap_data = False
            cameras = {}
            images_data = {}
    else:
        print(f"COLMAP directory {colmap_path} not found or not valid")
        cameras = {}
        images_data = {}
    
    # Priority for camera intrinsics:
    # 1. NeRF intrinsics (if provided)
    # 2. COLMAP camera data
    # 3. Intrinsics from Gaussian fitting
    # 4. Default intrinsic matrix
    
    # First check for NeRF intrinsics
    if nerf_K is not None:
        K1 = nerf_K
        K2 = nerf_K
        print("Using intrinsic matrix from NeRF dataset's transforms_train.json")
    else:
        # Next try COLMAP data
        use_colmap_cameras = has_colmap_data and bool(cameras) and bool(images_data)
        
        if use_colmap_cameras:
            # Get image IDs from COLMAP data
            image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
            print(f"Available images in COLMAP data: {image_name_to_id}")
    
            image1_id = image_name_to_id.get(img1_name)
            image2_id = image_name_to_id.get(img2_name)
            
            # Double check that both images are in COLMAP data
            if image1_id is None or image2_id is None:
                print(f"Warning: Image {img1_name} or {img2_name} not found in COLMAP data.")
                print("Will use a single generic intrinsic matrix for all cameras.")
                use_colmap_cameras = False
        else:
            print("No valid COLMAP data found. Will use a single generic intrinsic matrix for all cameras.")
        
        if use_colmap_cameras:
            # Create camera models from COLMAP data
            camera1_id = images_data[image1_id]['camera_id']
            camera2_id = images_data[image2_id]['camera_id']
            
            camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
            camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)
            
            K1 = camera1.K
            K2 = camera2.K
        else:
            # Use a common intrinsic matrix
            # Get information from 2D Gaussians if available
            _, _, _, K_from_gs1 = load_gaussians_torch(gaussians1_path, device)
            _, _, _, K_from_gs2 = load_gaussians_torch(gaussians2_path, device)
            
            if K_from_gs1 is not None:
                K1 = K_from_gs1
                K2 = K_from_gs1
                print(f"Using intrinsic matrix from 2D Gaussians data.")
            else:
                # Use a default intrinsic matrix (centered principal point, focal length based on arbitrary values)
                H, W = 800, 800  # Random image size, adjust as needed
                fx, fy = 1.2*W, 1.2*W  # Random focal length (1.2x image width)
                cx, cy = W/2, H/2  # Principal point at center
                
                K1 = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
                K2 = K1
                print(f"Using default intrinsic matrix with focal length: {fx:.2f}")
                print(f"K = \n{K1}")
    
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.01,
        lambda_mean=0.0,
        lambda_cov=0.0,
        lambda_color=0.2,
        lambda_epipolar=1.0,
        device=device
    )
    
    # Optimize fundamental matrix
    print("Optimizing fundamental matrix...")
    solver.optimize_with_RT(max_iter=max_iterations, tol=1e-6)
    F_optimized = solver.f.detach().cpu().numpy()
    
    # Compute optimal transport
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()
    
    # Set up reconstructor (homographyの可能性あるからまだh_dummy捨てない)
    h_dummy = np.eye(3)
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, h_dummy)
    
    # Identify source Gaussians
    print("Identifying source Gaussians...")
    reconstructor.identify_source_gaussians(
        transport_matrix=transport_matrix_np,
        auto_threshold=True
    )
    
    # Get optimized R, t
    r_optimized = solver.rvec.detach().cpu().numpy()
    t_optimized = solver.tvec.detach().cpu().numpy()
    R_est = solver.rodrigues(solver.rvec).detach().cpu().numpy()
    
    # Set camera matrices
    reconstructor.set_camera_matrices_explicitly(
        r1=np.eye(3),
        t1=np.zeros(3),
        r2=R_est,
        t2=t_optimized
    )
    
    # Triangulate Gaussian centers
    print("Triangulating Gaussian centers...")
    threshold = 0.0  # Don't drop any Gaussians (to cover local optima)
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold)
    
    if len(reconstructor.points_3d) == 0:
        raise ValueError("Triangulation failed: no 3D points generated")
    
    # Compute 3D covariances, colors, and alphas 
    print("Computing 3D Gaussian properties...")
    
    # Calculate target volume dynamically if needed
    if auto_target_volume or target_volume is None:
        W1, H1 = K1[0, 2]*2, K1[1, 2]*2  # image1 width, height
        W2, H2 = K2[0, 2]*2, K2[1, 2]*2  # image2 width, height
        avg_pixel_area = (W1 * H1 + W2 * H2) / 2
        num_gaussians = max(len(gaussians1.means), len(gaussians2.means))
        final_target_volume = avg_pixel_area / num_gaussians
        print(f"Calculated target volume: {final_target_volume:.2f}")
    else:
        # Use the provided target volume
        final_target_volume = target_volume
        
    print(f"Using target volume: {final_target_volume:.2f}")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=0.0, target_volume=final_target_volume)
    
    # Save the target volume for future viewpoints
    reconstruction_data_target_volume = final_target_volume
    if len(reconstructor.points_3d) == 0:
        raise ValueError("No valid 3D Gaussians after covariance optimization. Try different initial images.")

    reconstructor.compute_3d_gaussian_colors(color_mode="average")
    reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")
    
    # Camera poses for PLY export
    R1 = np.eye(3)  # world->camera1
    t1 = np.zeros(3)
    R2 = R_est      # world->camera2 (R_estはworld->camera2の回転)
    t2 = t_optimized  # world->camera2 (t_optimizedはworld->camera2の並進)
    camera_params_list = [(R1, t1), (R2, t2)]

    # Save results as PLY
    ply_path = os.path.join(output_dir, "initial_3d_gaussians.ply")
    save_ellipsoids_as_ply(
        points_3d=reconstructor.points_3d,
        covariances_3d=reconstructor.covariances_3d,
        colors_3d=reconstructor.color_3d,
        alphas_3d=reconstructor.alpha_3d,
        filename=ply_path,
        camera_params=camera_params_list,
        use_alpha=True
    )
    # Prepare source Gaussians data for future use
    source_gaussians_data = {}
    source_gaussians_data['source_gaussians1_data'] = reconstructor.source_gaussians1_data
    source_gaussians_data['source_gaussians1_data']['image_name'] = img1_name
    source_gaussians_data['source_gaussians2_data'] = reconstructor.source_gaussians2_data
    source_gaussians_data['source_gaussians2_data']['image_name'] = img2_name
    
    # Create 3D Gaussians in the expected format for ViewpointExtender
    existing_3d_gaussians = []
    for i in range(len(reconstructor.points_3d)):
        point_3d = reconstructor.points_3d[i]
        cov_3d = reconstructor.covariances_3d[i]
        color = reconstructor.color_3d[i]
        alpha = reconstructor.alpha_3d[i]
        quaternion = reconstructor.quaternions[i]
        scale = reconstructor.scales[i]
        
        gauss = {
            "center": point_3d,
            "covariance": cov_3d,  
            "color": color,
            "alpha": alpha,
            "quaternion": quaternion,
            "scale": scale
        }
        existing_3d_gaussians.append(gauss)
    
    # quaternionsとscalesはGaussにも追加してあるが，全体としても保持しておく
    quaternions = reconstructor.quaternions
    scales = reconstructor.scales
    
    # Save essential results, removing redundant fields
    results = {
        # Core 3D Gaussian data
        "existing_3d_gaussians": existing_3d_gaussians,
        "points_3d": reconstructor.points_3d,
        "covariances_3d": reconstructor.covariances_3d,
        "color_3d": reconstructor.color_3d,
        "alpha_3d": reconstructor.alpha_3d,
        "quaternions": quaternions,
        "scales": scales,
        
        # Camera parameters
        "camera_params_list": camera_params_list,
        "K": K1,  # (need to modify if the intrinsic matrix is not shared)
        
        # for adding viewpoint
        "source_gaussians_data": source_gaussians_data,
        
        # Metadata
        "used_images": [img1_name, img2_name],
        "total_3d_gaussians": len(reconstructor.points_3d),
        "all_matches": [],  # Initialize for future BA
        
        # Store the calculated target volume for future viewpoints
        "target_volume": reconstruction_data_target_volume
    }
    
    # Track observations for BA
    observations1 = []
    observations2 = []
    

    for i, (idx1, idx2) in enumerate(reconstructor.match_pairs):
        if i < len(reconstructor.points_3d):
            observations1.append((i, gaussians1.means[idx1]))
            observations2.append((i, gaussians2.means[idx2]))
    
    results["all_matches"] = [observations1, observations2]
    
    # Perform initial bundle adjustment if enabled
    if enable_ba and len(reconstructor.points_3d) >= 3:
        ba_dir = os.path.join(output_dir, "ba_initial")
        results = perform_bundle_adjustment(
            reconstruction_data=results,
            ba_iterations=ba_iterations,
            device=device,
            save_dir=ba_dir
        )
    
    # Save to pickle
    results_path = os.path.join(output_dir, "initial_3d_reconstruction.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Initial reconstruction completed with {len(existing_3d_gaussians)} 3D Gaussians")
    print(f"Results saved to {results_path}")
    
    return results

def process_remaining_source_gaussians(
    reconstruction_data: Dict, 
    args,
    device: torch.device = None
) -> int:
    """最後に残った未処理の湧出ガウスを処理
    
    主に初期ペアのうち、referenceとして使われなかった方の画像の湧出ガウスを処理する
    →湧出ガウスとして残っているものは使用されていないカメラとの輸送で潰されるべき（理想的には）
    
    Args:
        reconstruction_data: 再構成データ辞書
        args: コマンドライン引数
        device: 計算デバイス
        
    Returns:
        int: 処理されたガウスの数
    """
    if "source_gaussians_data" not in reconstruction_data:
        print("No source gaussians data available.")
        return 0
    
    # 未処理の湧出ガウスをカウント
    unprocessed_count = 0
    source_gaussians_data = reconstruction_data["source_gaussians_data"]
    
    # 各カメラごとの未処理湧出ガウス数を確認
    cam_unprocessed = {}
    for key, data in source_gaussians_data.items():
        if not key.startswith('source_gaussians') or 'indices' not in data:
            continue
        
        cam_idx = int(key.replace('source_gaussians', '').replace('_data', '')) - 1
        
        if 'processed' in data:
            unproc_count = np.sum(~data['processed'])
        else:
            unproc_count = len(data['indices'])
            
        if unproc_count > 0:
            cam_unprocessed[cam_idx] = unproc_count
            unprocessed_count += unproc_count
    
    if unprocessed_count == 0:
        print("No unprocessed source gaussians found.")
        return 0
    
    print(f"Found {unprocessed_count} unprocessed source gaussians from cameras: {list(cam_unprocessed.keys())}")
    
    # 初期ペアの情報を取得
    if len(reconstruction_data.get('used_images', [])) < 2:
        print("Not enough images to identify initial pair.")
        return 0
    
    initial_pair = reconstruction_data['used_images'][:2]
    print(f"Initial image pair: {initial_pair}")
    
    # カメラパラメータ情報を取得
    camera_params_list = reconstruction_data["camera_params_list"]
    if len(camera_params_list) < 2:
        print("Not enough camera parameters available.")
        return 0
    
    # カメラ内部パラメータを取得
    if "K" in reconstruction_data:
        K = reconstruction_data["K"]
    elif "camera1_K" in reconstruction_data:
        K = reconstruction_data["camera1_K"]
    else:
        assert False, "Camera intrinsics not found in reconstruction_data"
    
    # 最初の視点拡張で使われたreferenceカメラを特定
    # 通常0番のカメラ（初期ペアの1枚目）
    reference_cam_idx = 0
    
    # もう片方のカメラインデックス
    other_cam_idx = 1
    
    # 処理するカメラを選択→主に初期ペアのうち，referenceとして使われなかった方のカメラ
    target_cam_idx = other_cam_idx if other_cam_idx in cam_unprocessed else None
    
    if target_cam_idx is None:
        print("No unprocessed source gaussians from non-reference initial camera.")
        return 0
    
    print(f"Processing unprocessed source gaussians from camera {target_cam_idx} (initial pair)")
    
    # もう片方の画像のガウスをロード（ここも非効率だがとりまおけ）
    ref_image = initial_pair[reference_cam_idx]
    ref_gaussians_path = os.path.join(
        args.fitted_gaussians_dir, 
        f"{ref_image.split('.')[0]}_fitted_gaussians.pkl"
    )
    
    
    _, ref_2d_gaussians, _, _ = load_gaussians_torch(ref_gaussians_path, device=device)
    
    extender = ViewpointExtender(
        existing_3d_gaussians=reconstruction_data["existing_3d_gaussians"],
        camera_params_list=camera_params_list,
        K_new=K,
        reference_camera_idx=reference_cam_idx,
        device=device,
        source_gaussians_data=source_gaussians_data
    )
    
    # 湧出ガウスを対象とした三角測量
    processed = extender.triangulate_source_gaussians(
        source_camera_idx=target_cam_idx,
        new_image_2d_gaussians=ref_2d_gaussians,
        target_volume=args.target_volume,
        correspondence_threshold=1e-6 
    )
    
    # ジオメトリ/湧出ガウスデータを更新
    reconstruction_data["existing_3d_gaussians"] = extender.existing_3d_gaussians
    reconstruction_data["source_gaussians_data"] = extender.source_gaussians_data
    
    if processed > 0:
        points_3d, covariances_3d, colors_3d, alphas_3d = [], [], [], []
        for gauss in reconstruction_data["existing_3d_gaussians"]:
            points_3d.append(gauss["center"])
            covariances_3d.append(gauss["covariance"])
            colors_3d.append(gauss["color"])
            alphas_3d.append(gauss["alpha"])
        
        reconstruction_data["points_3d"] = np.array(points_3d)
        reconstruction_data["covariances_3d"] = np.array(covariances_3d)
        reconstruction_data["color_3d"] = np.array(colors_3d)
        reconstruction_data["alpha_3d"] = np.array(alphas_3d)
        
        print(f"Successfully processed {processed} source gaussians from non-reference initial camera")
        return processed
    else:
        print("No source gaussians could be processed")
        return 0

def select_reference_camera(camera_params_list, source_gaussians_data=None):
    """視点拡張時の参照カメラを選択する - 改良版
    
    湧出ガウスの数をベースにした参照カメラ選択。未処理湧出ガウスが最も多いカメラを優先する。
    
    Args:
        camera_params_list: カメラパラメータのリスト
        source_gaussians_data: 湧出ガウス情報（オプション）
        
    Returns:
        int: 参照カメラのインデックス
    """
    if len(camera_params_list) <= 1:
        # カメラが1つしかなければそれを使用
        return 0
    
    # 湧出ガウス情報がない場合、従来通り最新カメラを使用
    if source_gaussians_data is None:
        return len(camera_params_list) - 1
    
    # 各カメラの未処理湧出ガウス数をカウント
    unprocessed_counts = {}
    for key, data in source_gaussians_data.items():
        if not key.startswith('source_gaussians') or 'indices' not in data:
            continue
        
        # カメラインデックスを抽出（例: 'source_gaussians1_data' -> 1）
        try:
            cam_idx = int(key.replace('source_gaussians', '').replace('_data', '')) - 1
            if cam_idx < len(camera_params_list):  # 有効なインデックスか確認
                if 'processed' in data:
                    # 未処理の湧出ガウス数をカウント
                    unproc_count = np.sum(~data['processed'])
                    unprocessed_counts[cam_idx] = unproc_count
                else:
                    # processed フラグが無い場合は全て未処理と見なす
                    unprocessed_counts[cam_idx] = len(data['indices'])
        except (ValueError, IndexError):
            continue
    
    # デバッグ情報出力
    for cam_idx, count in unprocessed_counts.items():
        print(f"カメラ {cam_idx}: 未処理湧出ガウス {count}個")
    
    # 未処理湧出ガウスが最も多いカメラを選択
    if unprocessed_counts:
        best_cam_idx = max(unprocessed_counts.keys(), key=lambda k: unprocessed_counts[k])
        if unprocessed_counts[best_cam_idx] > 0:  # 未処理ガウスが存在する場合
            print(f"選択された参照カメラ {best_cam_idx}: 未処理湧出ガウス {unprocessed_counts[best_cam_idx]}個")
            return best_cam_idx
    
    # 未処理湧出ガウスが無いか、情報が不十分な場合は最新カメラを使用
    latest_cam_idx = len(camera_params_list) - 1
    print(f"未処理湧出ガウスが見つからなかったため、最新カメラ {latest_cam_idx} を使用")
    return latest_cam_idx

def add_new_viewpoint(
    reconstruction_data: Dict,
    new_image_name: str,
    fitted_gaussians_dir: str,
    output_dir: str,
    max_iterations: int = 1000,
    transport_threshold: float = 1e-6,
    target_volume: float = None,
    auto_target_volume: bool = True,
    auto_threshold: bool = True,
    device: torch.device = None,
    enable_ba: bool = True,
    ba_iterations: int = 10,
    force_single_intrinsic: bool = False
) -> Dict:
    print(f"\n--- Adding New Viewpoint: {new_image_name} ---")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    
    # Get path to fitted Gaussians for new image
    new_gaussians_path = os.path.join(
        fitted_gaussians_dir, 
        f"{new_image_name.split('.')[0]}_fitted_gaussians.pkl"
    )
    
    # Load new 2D Gaussians
    _, new_2d_gaussians, _, K_new = load_gaussians_torch(new_gaussians_path, device)
    
    # Extract existing 3D Gaussians and camera parameters
    existing_3d_gaussians = reconstruction_data["existing_3d_gaussians"]
    camera_params_list = reconstruction_data["camera_params_list"]
    
    # Determine which camera intrinsics to use
    if force_single_intrinsic and "K" in reconstruction_data:
        # Force using the shared K from reconstruction_data
        print(f"Using shared intrinsic matrix from initial reconstruction")
        K_new = reconstruction_data["K"]
    elif K_new is None:
        # If not provided by the 2D Gaussian loader, try to get from reconstruction data (this is redundant but keep it just in case)
        if "K" in reconstruction_data:
            K_new = reconstruction_data["K"]
            print(f"Using shared K from reconstruction_data")
        elif "camera1_K" in reconstruction_data:
            K_new = reconstruction_data["camera1_K"]
            print(f"Using first camera's K as fallback")
        else:
            # Create a default intrinsic matrix as last resort (not necessary as well, just keep it for now)
            assert False, "No intrinsic matrix available"
            H, W = 800, 800  # Random image size, adjust as needed
            fx, fy = 1.2*W, 1.2*W  # Random focal length (1.2x image width)
            cx, cy = W/2, H/2  # Principal point at center
            
            K_new = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            print(f"WARNING: No intrinsic matrix available. Using default with focal length: {fx:.2f}")
    else:
        print(f"Using intrinsic matrix from fitted Gaussians data")
    
    # Get source Gaussians data - initialize if not present(most likely does not happen→これが起きるということは初期ペア疑った方がいい)
    if "source_gaussians_data" in reconstruction_data:
        source_gaussians_data = reconstruction_data["source_gaussians_data"]
    else:
        source_gaussians_data = {}
    
    # Get used images 
    used_images = reconstruction_data["used_images"]
    reference_camera_idx = select_reference_camera(camera_params_list, source_gaussians_data)
    print(f"Using camera {reference_camera_idx} as reference for new viewpoint")
    
    extender = ViewpointExtender(
        existing_3d_gaussians=existing_3d_gaussians,
        camera_params_list=camera_params_list,
        K_new=K_new,
        reference_camera_idx=reference_camera_idx,  
        threshold_reprojection=transport_threshold,
        device=device,
        source_gaussians_data=source_gaussians_data
    )
    
    # 新視点のカメラパラメータ推定と3DGSの更新
    R_new, t_new = extender.integrate_new_view_and_gaussians(
        new_image_2d_gaussians=new_2d_gaussians,
        max_iterations=max_iterations,
        transport_threshold=transport_threshold,
        target_volume=target_volume,
        auto_threshold=auto_threshold
    )
    
    # 新しい視点との対応関係を抽出
    # ViewpointExtenderのtransport_solverを使用してcost matrixを計算
    # 輸送行列の計算
    with torch.no_grad():
        cost_matrix = extender.transport_solver.compute_cost_matrix_fundamental(
            extender.transport_solver.f
        )
        transport_matrix = extender.transport_solver.unbalanced_sinkhorn_algorithm(cost_matrix)
        transport_matrix_np = transport_matrix.cpu().numpy()
        
        # 輸送行列を観測情報として追跡
        observations = extender.track_observations(transport_matrix_np)
        
        # 新しいカメラの観測情報を追加
        reconstruction_data['all_matches'].append(observations)
        
        # # Ensure transport_matrices exists
        # if 'transport_matrices' not in reconstruction_data:
        #     reconstruction_data['transport_matrices'] = []
        
        # # 最適輸送行列を保存（後でBundle Adjustmentに使用するかも？の場合初期化はreconstructorで行うべき）
        # reconstruction_data['transport_matrices'].append(transport_matrix_np)
    
    # Extract updated data
    points_3d, covariances_3d, colors_3d, alphas_3d = [], [], [], []
    quaternions, scales = [], []
    
    for gauss in extender.existing_3d_gaussians:
        points_3d.append(gauss["center"])
        covariances_3d.append(gauss["covariance"])
        colors_3d.append(gauss["color"])
        alphas_3d.append(gauss["alpha"])
        quaternions.append(gauss["quaternion"])
        scales.append(gauss["scale"])
    
    # 配列に変換
    points_3d = np.array(points_3d)
    covariances_3d = np.array(covariances_3d)
    colors_3d = np.array(colors_3d)
    alphas_3d = np.array(alphas_3d)
    quaternions = np.array(quaternions)
    scales = np.array(scales)
    
    # Update used images list  its 
    used_images = reconstruction_data["used_images"].copy()
    used_images.append(new_image_name)
    
    
    # Create updated results
    updated_results = {
        # Core 3D Gaussian data
        "existing_3d_gaussians": extender.existing_3d_gaussians,
        "points_3d": points_3d,
        "covariances_3d": covariances_3d,
        "color_3d": colors_3d,
        "alpha_3d": alphas_3d,
        "quaternions": quaternions,
        "scales": scales,
        
        # Camera parameters
        "camera_params_list": extender.camera_params_list,
        "K": K_new,
        
        # Source Gaussian data
        "source_gaussians_data": source_gaussians_data,
        
        # Metadata
        "used_images": used_images,
        "total_3d_gaussians": len(points_3d),
        "all_matches": reconstruction_data.get('all_matches', [])
    }
    
    # Save as PLY
    ply_path = os.path.join(output_dir, f"updated_3d_gaussians_{new_image_name.split('.')[0]}.ply")
    save_ellipsoids_as_ply(
        points_3d=points_3d,
        covariances_3d=covariances_3d,
        colors_3d=colors_3d,
        alphas_3d=alphas_3d,
        filename=ply_path,
        camera_params=extender.camera_params_list,
        use_alpha=True
    )
    
    # 視点追加後にバンドル調整を実行(enable_baがTrueの場合)
    if enable_ba:
        ba_dir = os.path.join(output_dir, "ba_results")
        updated_results = perform_bundle_adjustment(
            reconstruction_data=updated_results,
            ba_iterations=ba_iterations,
            device=device,
            save_dir=ba_dir
        )
    
    # Save full results
    results_path = os.path.join(output_dir, f"updated_reconstruction_{new_image_name.split('.')[0]}.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(updated_results, f)
    
    print(f"Added viewpoint {new_image_name}")
    print(f"Total 3D Gaussians: {len(extender.existing_3d_gaussians)}")
    print(f"Results saved to {results_path}")
    
    return updated_results

def run_complete_pipeline(args):
    """Run the complete Perspective-n-Gaussian pipeline."""
    
    #####################################################
    # 0. Setup and validation
    #####################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    image_dir = os.path.join(args.data_dir, "images")
    colmap_dir = os.path.join(args.data_dir, args.colmap_dir)
    
    # We'll use the same target_volume throughout the pipeline for consistency
    global_target_volume = args.target_volume
    
    all_images = get_image_names(image_dir)
    gaussian_files = get_fitted_gaussians_info(args.fitted_gaussians_dir)
    
    # Check if we should use NeRF intrinsics
    nerf_K = None
    if args.use_nerf_intrinsics:
        try:
            print("Using camera intrinsics from NeRF dataset's transforms_train.json")
            nerf_K = load_nerf_intrinsics(args.data_dir)
        except Exception as e:
            print(f"Error loading NeRF intrinsics: {e}")
            print("Falling back to other intrinsics options")
    
    # Check if COLMAP directory exists (read camera intrinsics from colmap data)
    has_colmap_data = os.path.exists(colmap_dir) and os.path.isdir(colmap_dir)
    if not has_colmap_data or args.force_single_intrinsic:
        if not args.use_nerf_intrinsics:
            print(f"{'Warning: COLMAP directory not found' if not has_colmap_data else 'User requested single intrinsic matrix'}. Will use a single intrinsic matrix for all cameras.")
    
    # Filter to only images with fitted Gaussians 
    available_images = [img for img in all_images if img in gaussian_files]
    
    # Don't filter available images upfront anymore
    # We'll use all available images for initial pair selection
    
    # Apply maximum images limit if set - this still limits the total number of images in the dataset
    if args.max_images is not None and len(available_images) > args.max_images:
        print(f"Limiting to {args.max_images} images out of {len(available_images)} available")
        available_images = available_images[:args.max_images]
    
    print(f"Using {len(available_images)} images for reconstruction")
    
    # Validate input
    assert len(available_images) >= 2, f"Need at least 2 images with fitted Gaussians, found {len(available_images)}"
    print(f"Found {len(available_images)} images with fitted Gaussians")
    
    #####################################################
    # 2. Initial reconstruction - either load or compute
    #####################################################
    initial_results_path = os.path.join(args.output_dir, "initial_3d_reconstruction.pkl")
    
    # Determine whether to load existing results or create new ones(if initial_3d_reconstruction_non_linear.py is previsously conducted)
    if os.path.exists(initial_results_path) and args.skip_existing:
        # Load existing reconstruction
        with open(initial_results_path, 'rb') as f:
            reconstruction_data = pickle.load(f)
        
        used_images = reconstruction_data.get("used_images", [])
        print(f"Loaded reconstruction with {len(reconstruction_data['existing_3d_gaussians'])} 3D Gaussians")
        print(f"Used images: {used_images}")
    else:
        # Create new reconstruction
        # Select initial pair
        img1, img2, selector = select_initial_pair(
            image_dir=image_dir,
            gaussian_files=gaussian_files,
            vocab_size=args.vocab_size,
            feature_type=args.feature_type,
            min_overlap=args.min_overlap,
            max_overlap=args.max_overlap,
            available_images=available_images  # Pass the filtered available_images
        )
        
        # Conduct initial reconstruction
        reconstruction_data= perform_initial_reconstruction(
            img1_name=img1,
            img2_name=img2,
            data_dir=args.data_dir,
            colmap_dir=args.colmap_dir,
            fitted_gaussians_dir=args.fitted_gaussians_dir,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            target_volume=global_target_volume,
            auto_target_volume=args.auto_target_volume,
            device=device,
            enable_ba=args.enable_ba and not args.ba_skip_initial,
            ba_iterations=args.ba_iterations,
            nerf_K=nerf_K
        )
        
        # If we calculated the target volume automatically, store it for future use
        if global_target_volume is None and args.auto_target_volume:
            global_target_volume = reconstruction_data.get("target_volume", None)
        
        used_images = [img1, img2]

    # Set reference images (already used images)
    selector.add_reference_images(used_images)
    
    # Set source Gaussians data
    selector.set_source_gaussians_data(reconstruction_data["source_gaussians_data"])
    
    # Start timing
    start_time = time.time()
    
    #####################################################
    # 3. Incremental reconstruction
    #####################################################
    iteration = 1
    views_added = 0  # Counter for the number of views we've added
    remaining_images = [img for img in available_images if img not in used_images]
    total_remaining = len(remaining_images)
    
    # Calculate maximum views to add based on parameters
    if args.max_views_to_add is not None:
        max_views = args.max_views_to_add
        print(f"Will add at most {max_views} additional views")
    elif args.use_sparse_set:
        # If using sparse set, calculate max views from interval
        max_views = (len(available_images) - 2) // args.sparse_interval
        print(f"Using sparse interval {args.sparse_interval}, will add {max_views} views")
    else:
        max_views = None
        print("No limit on number of views to add")
    
    while remaining_images and (max_views is None or views_added < max_views):
        print(f"\n--- Iteration {iteration}/{total_remaining} ---")
        
        # Select next best view
        next_image = selector.select_next_view(remaining_images, n_select=1)[0]
        
        # Process the selected image
        iter_output_dir = os.path.join(args.output_dir, f"iteration_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Determine if we should run BA in this iteration
        run_ba_this_iteration = False
        
        if args.enable_ba and not args.ba_skip_incremental:
            # Run BA every N views as specified by ba_every_n_views
            if iteration % args.ba_every_n_views == 0:
                run_ba_this_iteration = True
                print(f"Will run Bundle Adjustment after adding view (iteration {iteration} is divisible by {args.ba_every_n_views})")
            else:
                print(f"Skipping Bundle Adjustment (iteration {iteration} is not divisible by {args.ba_every_n_views})")
        
        # Determine if we need to force single intrinsic matrix
        force_single_K = not has_colmap_data or args.force_single_intrinsic
        
        # Add the new viewpoint - use the consistent target volume from initial reconstruction
        updated_data = add_new_viewpoint(
            reconstruction_data=reconstruction_data,
            new_image_name=next_image,
            fitted_gaussians_dir=args.fitted_gaussians_dir,
            output_dir=iter_output_dir,
            max_iterations=args.max_iterations,
            transport_threshold=args.transport_threshold,
            target_volume=global_target_volume,
            auto_threshold=args.auto_threshold,
            device=device,
            enable_ba=run_ba_this_iteration,
            ba_iterations=args.ba_iterations,
            force_single_intrinsic=force_single_K
        )
        
        # Update reconstruction data for next iteration
        reconstruction_data = updated_data
        
        # Remove processed image from remaining images
        remaining_images.remove(next_image)
        
        # Add the new image to reference images for ViewSelector
        selector.add_reference_images([next_image])
        
        # Update source Gaussians data in selector
        if "source_gaussians_data" in updated_data:
            selector.set_source_gaussians_data(updated_data["source_gaussians_data"])
                
    
        # Increment counters
        iteration += 1
        views_added += 1
        
        # Calculate and print progress
        elapsed_time = time.time() - start_time
        processed_count = views_added
        
        # Adjust estimates based on view limits
        if max_views is not None:
            remaining_to_process = min(max_views - views_added, len(remaining_images))
        else:
            remaining_to_process = len(remaining_images)
            
        avg_time_per_image = elapsed_time / processed_count if processed_count > 0 else 0
        estimated_remaining = avg_time_per_image * remaining_to_process
        
        print(f"\nProgress: {processed_count} views added")
        if max_views is not None:
            print(f"Maximum views to add: {max_views}, Remaining: {max_views - views_added}")
        print(f"Remaining images to choose from: {len(remaining_images)}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {estimated_remaining:.2f} seconds")
    
    # Process remaining source Gaussians at the end 
    if "source_gaussians_data" in reconstruction_data and reconstruction_data["source_gaussians_data"]:
        print("\n--- Processing Remaining Source Gaussians ---")
        # Create a copy of args with our consistent target_volume
        class ArgsWithTargetVolume:
            pass
        args_copy = ArgsWithTargetVolume()
        for key, value in vars(args).items():
            setattr(args_copy, key, value)
        args_copy.target_volume = global_target_volume
        
        processed_count = process_remaining_source_gaussians(
            reconstruction_data, 
            args_copy,
            device=device
        )
        if processed_count > 0:
            print(f"Successfully processed {processed_count} remaining source gaussians")
        else:
            print("No remaining source gaussians to process or processing failed")
    
    # Final results directory
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Final global bundle adjustment
    if args.enable_ba and not args.ba_skip_final:
        print("\n--- Performing Final Global Bundle Adjustment ---")
        ba_dir = os.path.join(final_output_dir, "ba_final")
        reconstruction_data = perform_bundle_adjustment(
            reconstruction_data=reconstruction_data,
            ba_iterations=args.ba_iterations*2,  # more iterations for final BA
            device=device,
            verbose=True,
            save_dir=ba_dir
        )
    
    # Save final PLY
    ply_path = os.path.join(final_output_dir, "final_3d_gaussians.ply")
    save_ellipsoids_as_ply(
        points_3d=reconstruction_data["points_3d"],
        covariances_3d=reconstruction_data["covariances_3d"],
        colors_3d=reconstruction_data["color_3d"],
        alphas_3d=reconstruction_data["alpha_3d"],
        filename=ply_path,
        camera_params=reconstruction_data["camera_params_list"],
        use_alpha=True
    )
    
    # Gaussian Splattingの初期値としてのPLY保存
    gs_ply_path = os.path.join(final_output_dir, "gs_init_gaussians.ply")


    #####################################################
    # 4. Export to COLMAP format
    #####################################################
    # データを取得
    quaternions = reconstruction_data["quaternions"]
    scales = reconstruction_data["scales"]
    alphas_3d = reconstruction_data["alpha_3d"]
    
    # quaternionの正規性を検証 - 正規化されていないと後続処理で問題が起きる可能性あり
    quat_norms = np.linalg.norm(quaternions, axis=1)
    if not np.allclose(quat_norms, 1.0, rtol=1e-4):
        print("Warning: Quaternions are not normalized - normalizing now")
        quaternions = quaternions / quat_norms[:, np.newaxis]

    save_gaussians_as_ply(
        points_3d=reconstruction_data["points_3d"],
        quaternions=quaternions,
        scales=scales,
        colors_3d=reconstruction_data["color_3d"],
        alphas_3d=alphas_3d,
        filename=gs_ply_path
    )
    print(f"Saved Gaussian Splatting initialization data to {gs_ply_path}")

    # COLMAPフォーマットでのエクスポート
    colmap_output_dir = os.path.join(final_output_dir, "colmap")
    os.makedirs(colmap_output_dir, exist_ok=True)

    # 観測データの構築
    observation_map = ObservationBuilder.build_observation_map(reconstruction_data)
    match_points_2d = ObservationBuilder.convert_to_match_points_2d(
        observation_map, 
        len(reconstruction_data["camera_params_list"])
    )

    # 画像名の検証
    assert "used_images" in reconstruction_data, "Image names missing from reconstruction data"
    image_names = reconstruction_data["used_images"]
    assert len(image_names) == len(reconstruction_data["camera_params_list"]), "Number of image names must match number of cameras"
    print(f"Using image names: {image_names}")

    # カメラパラメータの存在チェック - 最低一つは必要
    assert "K" in reconstruction_data or "camera1_K" in reconstruction_data, "No camera intrinsics found in reconstruction_data"
    
    # 内部パラメータリストを構築
    num_cameras = len(reconstruction_data["camera_params_list"])
    intrinsics_list = []
    
    for cam_idx in range(num_cameras):
        # 各カメラの内部パラメータを明示的に決定
        cam_key = f"camera{cam_idx+1}_K"
        
        if cam_key in reconstruction_data:
            # このカメラ専用の内部パラメータを使用
            intrinsics_list.append(reconstruction_data[cam_key])
        elif "K" in reconstruction_data:
            # 共通の内部パラメータを使用
            intrinsics_list.append(reconstruction_data["K"])
        else:
            # camera1のパラメータを使用
            assert "camera1_K" in reconstruction_data, f"No intrinsics available for camera {cam_idx+1}"
            intrinsics_list.append(reconstruction_data["camera1_K"])
            
    # COLMAPフォーマットにエクスポート
    export_colmap_format(
        output_dir=colmap_output_dir,
        points_3d=reconstruction_data["points_3d"],
        camera_params_list=reconstruction_data["camera_params_list"],
        match_points_2d=match_points_2d,
        intrinsics_list=intrinsics_list,
        image_names=image_names
    )
    print(f"Exported reconstruction to COLMAP format in {colmap_output_dir}")

    # COLMAPフォーマットでのGaussian保存
    export_gaussians_to_colmap_dir(
        colmap_dir=colmap_output_dir,
        points_3d=reconstruction_data["points_3d"],
        covariances_3d=reconstruction_data["covariances_3d"],
        colors_3d=reconstruction_data["color_3d"],
        alphas_3d=reconstruction_data["alpha_3d"],
        quaternions=quaternions,
        scales=scales,
        save_ellipsoids_as_ply=save_ellipsoids_as_ply,
        save_gaussians_as_ply=save_gaussians_as_ply
    )

    # Save final results
    final_results_path = os.path.join(final_output_dir, "final_reconstruction.pkl")
    with open(final_results_path, 'wb') as f:
        pickle.dump(reconstruction_data, f)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"Total images used: {len(reconstruction_data['used_images'])}")
    if max_views is not None:
        print(f"Views added: {views_added} out of maximum {max_views}")
    print(f"Total 3D Gaussians: {len(reconstruction_data['existing_3d_gaussians'])}")
    print(f"Final results saved to {final_output_dir}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    args = parse_args()
    print("\n=== Perspective-n-Gaussian Pipeline ===")
    print(f"Data Directory: {args.data_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Fitted Gaussians Directory: {args.fitted_gaussians_dir}")
    
    # Print image selection settings
    if args.use_sparse_set:
        print(f"Using sparse interval: {args.sparse_interval}")
    if args.max_images:
        print(f"Maximum dataset images limit: {args.max_images}")
    if args.max_views_to_add:
        print(f"Maximum views to add: {args.max_views_to_add}")
        
    # Print bundle adjustment settings
    print(f"Bundle Adjustment: {'Enabled' if args.enable_ba else 'Disabled'}")
    if args.enable_ba:
        print(f"  - BA iterations: {args.ba_iterations}")
        print(f"  - Initial BA: {'Skip' if args.ba_skip_initial else 'Perform'}")
        print(f"  - Incremental BA: {'Skip' if args.ba_skip_incremental else 'Perform'}")
        if not args.ba_skip_incremental:
            print(f"  - BA frequency: Every {args.ba_every_n_views} view(s)")
        print(f"  - Final BA: {'Skip' if args.ba_skip_final else 'Perform'}")
    
    # Print camera settings
    camera_source = "NeRF transforms_train.json" if args.use_nerf_intrinsics else "Force single matrix" if args.force_single_intrinsic else "Use per-camera if available"
    print(f"Camera intrinsics: {camera_source}")
    
    print("=====================================\n")
    run_complete_pipeline(args)