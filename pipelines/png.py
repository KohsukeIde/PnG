import os
import sys
import traceback
import argparse
import pickle
import glob
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from src.reconstructor.view_selector import ViewSelector
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor, build_covariance_3d
from src.reconstructor.viewpoint_extender import ViewpointExtender
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from utils.gs_pkl_loader import load_gaussians_torch
from utils.saving.geometry_utils import save_ellipsoids_as_ply
from src.optimizer.bundle_adjuster import BundleAdjuster

# Fix module import issues
sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

def parse_args():
    """Parse command-line arguments for the complete pipeline."""
    parser = argparse.ArgumentParser(description="Complete Perspective-n-Gaussian Pipeline")
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63",
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
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked",
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
        choices=["sift", "orb"],
        help="Type of features to extract"
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
        default=1.0,
        help="Target volume for 3D Gaussians"
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
    max_overlap: float = 0.7
) -> Tuple[str, str]:
    """Select the best initial pair of images using Bag of Visual Words.
    
    The best pair should have:
    1. Good feature overlap (within min/max range)
    2. Rich features in both images
    3. Good spatial distribution of features → this is debatable (視差がありすぎると3D covが求められない可能性ありそう)
    
    Args:
        image_dir: Directory containing images
        gaussian_files: Dictionary mapping image names to their fitted Gaussian files
        vocab_size: Size of the visual vocabulary
        feature_type: Type of features to extract
        min_overlap: Minimum overlap ratio
        max_overlap: Maximum overlap ratio
    
    Returns:
        Tuple containing the names of the two selected images
    """
    print("\n--- Selecting Initial Image Pair ---")
    
    # Get available images that have fitted Gaussians (ここの処理はパイプライン最初と同じ)
    all_images = get_image_names(image_dir)
    available_images = [img for img in all_images if img in gaussian_files]
    
    if len(available_images) < 2:
        raise ValueError(f"Need at least 2 images with fitted Gaussians, found {len(available_images)}")
    
    # Initialize ViewSelector
    selector = ViewSelector(
        image_dir=image_dir,
        vocab_size=vocab_size,
        feature_type=feature_type,
        min_overlap_ratio=min_overlap,
        max_overlap_ratio=max_overlap
    )
    
    # Process all images to extract features and build codebook
    image_paths = [os.path.join(image_dir, img) for img in available_images]
    selector.initialize_from_images(image_paths)
    
    # Calculate similarity matrix between all pairs
    similarity_matrix = np.zeros((len(available_images), len(available_images)))
    
    for i, img1 in enumerate(available_images):
        # hist = ヒストグラムベクトル（dim = vocab_size）
        hist1 = selector.image_histograms[os.path.join(image_dir, img1)]
        for j, img2 in enumerate(available_images):
            if i >= j:  # Avoid redundant computation and self-comparison (対称行列なのでi < jのみでおけ)
                continue
            hist2 = selector.image_histograms[os.path.join(image_dir, img2)]
            # Compute cosine similarity
            similarity = np.sum(hist1 * hist2) / (np.sqrt(np.sum(hist1**2)) * np.sqrt(np.sum(hist2**2)) + 1e-10)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Calculate feature richness scores (number of features)
    feature_scores = np.array([
        len(selector.image_features[os.path.join(image_dir, img)]['keypoints'])
        for img in available_images
    ])
    
    # Normalize feature scores
    max_features = np.max(feature_scores)
    if max_features > 0:
        feature_scores = feature_scores / max_features
    
    # Find valid pairs (within overlap range)
    # valid_pairs = []
    # pair_scores = []
    
    # for i in range(len(available_images)):
    #     for j in range(i+1, len(available_images)):
    #         similarity = similarity_matrix[i, j]
            
    #         # Check if within desired overlap range
    #         if min_overlap <= similarity <= max_overlap:
    #             # Combined score: similarity + feature richness of both images
    #             score = similarity + 0.5 * (feature_scores[i] + feature_scores[j])
    #             valid_pairs.append((i, j))
    #             pair_scores.append(score)
    
    # if not valid_pairs:
    #     print("No pairs within specified overlap range, using best available pair")
    #     # Take pair with highest combined feature score
    #     best_pair = None
    #     best_score = -1
        
    #     for i in range(len(available_images)):
    #         for j in range(i+1, len(available_images)):
    #             combined_score = feature_scores[i] + feature_scores[j]
    #             if combined_score > best_score:
    #                 best_score = combined_score
    #                 best_pair = (i, j)
        
    #     if best_pair is None:
    #         # Fallback: just take the first two images
    #         best_pair = (0, 1)
            
    #     selected_idx = best_pair
    # else:
    #     # Select the best pair based on score
    #     best_idx = np.argmax(pair_scores)
    #     selected_idx = valid_pairs[best_idx]
    
    # Get the selected image names
    # img1 = available_images[selected_idx[0]]
    # img2 = available_images[selected_idx[1]]
    
    best_pair = None
    best_similarity = -1

    for i in range(len(available_images)):
        for j in range(i+1, len(available_images)):
            similarity = similarity_matrix[i, j]
            
            # 類似度が最大のペアを探す
            if similarity > best_similarity:
                best_similarity = similarity
                best_pair = (i, j)

    # best_pair が見つからなかった場合のフォールバック処理（全く特徴点が取れないケースとかのため）
    if best_pair is None:
        best_pair = (0, 1)

    img1 = available_images[best_pair[0]]
    img2 = available_images[best_pair[1]]
            

    
    print(f"Selected initial pair: {img1} and {img2}")
    print(f"Similarity: {similarity_matrix[best_pair[0], best_pair[1]]:.4f}")
    print(f"Feature counts: {len(selector.image_features[os.path.join(image_dir, img1)]['keypoints'])} and "f"{len(selector.image_features[os.path.join(image_dir, img2)]['keypoints'])}")
    
    return img1, img2, selector

def perform_initial_reconstruction(
    img1_name: str,
    img2_name: str,
    data_dir: str,
    colmap_dir: str,
    fitted_gaussians_dir: str,
    output_dir: str,
    max_iterations: int = 1000,
    target_volume: float = 1.0,
    device: torch.device = None
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
    
    Returns:
        Dictionary with reconstruction results
    """
    print("\n--- Performing Initial 3D Reconstruction ---")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths to fitted Gaussians
    gaussians1_path = os.path.join(fitted_gaussians_dir, f"{img1_name.split('.')[0]}_fitted_gaussians.pkl")
    gaussians2_path = os.path.join(fitted_gaussians_dir, f"{img2_name.split('.')[0]}_fitted_gaussians.pkl")
    
    # Load fitted Gaussians
    _, gaussians1, _, K1 = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, K2 = load_gaussians_torch(gaussians2_path, device)
    
    # Load COLMAP data
    colmap_path = os.path.join(data_dir, colmap_dir)
    cameras = load_cameras_from_colmap(colmap_path)
    images_data = load_images_from_colmap(colmap_path)
    
    # Get image IDs
    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}
    image1_id = image_name_to_id.get(img1_name)
    image2_id = image_name_to_id.get(img2_name)
    
    if image1_id is None or image2_id is None:
        raise ValueError(f"Image {img1_name} or {img2_name} not found in COLMAP data")
    
    # Create camera models
    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']
    
    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)
    
    K1 = camera1.K
    K2 = camera2.K
    
    # Initialize OptimalTransportSolver
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
    
    # Set up reconstructor
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
    
    # Set camera matrices and triangulate
    reconstructor.set_camera_matrices_explicitly(
        r1=np.eye(3),
        t1=np.zeros(3),
        r2=R_est,
        t2=t_optimized
    )
    
    # Triangulate Gaussian centers
    print("Triangulating Gaussian centers...")
    threshold = 0.0  # Don't drop any Gaussians at this stage
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold)
    
    if len(reconstructor.points_3d) == 0:
        raise ValueError("Triangulation failed: no 3D points generated")
    
    # Compute 3D covariances, colors, and alphas
    print("Computing 3D Gaussian properties...")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=10.0, target_volume=target_volume)
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
    print(f"camera_params_list {camera_params_list}")

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
    if hasattr(reconstructor, 'source_gaussians1_data'):
        source_gaussians_data['source_gaussians1_data'] = reconstructor.source_gaussians1_data
        # Add the image name to the data
        source_gaussians_data['source_gaussians1_data']['image_name'] = img1_name
    
    if hasattr(reconstructor, 'source_gaussians2_data'):
        source_gaussians_data['source_gaussians2_data'] = reconstructor.source_gaussians2_data
        source_gaussians_data['source_gaussians2_data']['image_name'] = img2_name
    
    # Create 3D Gaussians in the expected format for ViewpointExtender
    existing_3d_gaussians = []
    for i in range(len(reconstructor.points_3d)):
        point_3d = reconstructor.points_3d[i]
        cov_3d = reconstructor.covariances_3d[i]
        color = reconstructor.color_3d[i]
        alpha = reconstructor.alpha_3d[i]
        
        gauss = {
            "center": point_3d,
            "covariance": cov_3d,  
            "color": color,
            "alpha": alpha
        }
        existing_3d_gaussians.append(gauss)
    
    # Save full results
    results = {
        "fundamental_matrix": F_optimized,
        "transport_matrix": transport_matrix_np,
        "camera1_K": K1,
        "camera2_K": K2,
        "points_3d": reconstructor.points_3d,
        "covariances_3d": reconstructor.covariances_3d,
        "color_3d": reconstructor.color_3d,
        "alpha_3d": reconstructor.alpha_3d,
        'source_gaussians1': reconstructor.source_gaussians1,
        'source_gaussians2': reconstructor.source_gaussians2,
        'source_gaussians1_data': getattr(reconstructor, 'source_gaussians1_data', None),
        'source_gaussians2_data': getattr(reconstructor, 'source_gaussians2_data', None),
        'cov_failed_gaussians_included': True,  # Cov最適化に失敗したガウスが湧出ガウスに含まれていることを示すフラグ
        "transport_values": getattr(reconstructor, 'transport_values', None),
        "source_gaussians_data": source_gaussians_data,
        "camera_params_list": camera_params_list,
        "R1": R1,
        "t1": t1,
        "R2": R2,
        "t2": t2,
        "existing_3d_gaussians": existing_3d_gaussians,
        "used_images": [img1_name, img2_name]
    }
    
    # Save to pickle
    results_path = os.path.join(output_dir, "initial_3d_reconstruction.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Initial reconstruction completed with {len(existing_3d_gaussians)} 3D Gaussians")
    print(f"Results saved to {results_path}")
    
    return results

def add_new_viewpoint(
    reconstruction_data: Dict,
    new_image_name: str,
    fitted_gaussians_dir: str,
    output_dir: str,
    max_iterations: int = 1000,
    transport_threshold: float = 1e-6,
    target_volume: float = 1.0,
    auto_threshold: bool = True,
    device: torch.device = None
) -> Dict:
    print(f"\n--- Adding New Viewpoint: {new_image_name} ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure output directory exists
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
    
    # Get camera intrinsics from reconstruction data
    if K_new is None:
        if "K" in reconstruction_data:
            K_new = reconstruction_data["K"]
        elif "camera1_K" in reconstruction_data:
            # Fall back to using K1 if no K provided
            K_new = reconstruction_data["camera1_K"]
    
    # Get source Gaussians data
    source_gaussians_data = reconstruction_data.get("source_gaussians_data", {})
    
    # Initialize ViewpointExtender
    extender = ViewpointExtender(
        existing_3d_gaussians=existing_3d_gaussians,
        camera_params_list=camera_params_list,
        K_new=K_new,
        reference_camera_idx=0,  # Use first camera as reference
        threshold_reprojection=transport_threshold,
        device=device,
        source_gaussians_data=source_gaussians_data
    )
    
    # 新視点のカメラパラメータ推定と3Dガウス分布の更新
    R_new, t_new = extender.integrate_new_view_and_gaussians(
        new_image_2d_gaussians=new_2d_gaussians,
        max_iterations=max_iterations,
        transport_threshold=transport_threshold,
        target_volume=target_volume,
        auto_threshold=auto_threshold
    )
    
    # 新しい視点との対応関係を抽出
    # ViewpointExtender内のtransport_solverから輸送行列を取得
    if extender.transport_solver is not None and hasattr(extender.transport_solver, 'f'):
        with torch.no_grad():
            cost_matrix = extender.transport_solver.compute_cost_matrix_fundamental(
                extender.transport_solver.f
            )
            transport_matrix = extender.transport_solver.unbalanced_sinkhorn_algorithm(cost_matrix)
            transport_matrix_np = transport_matrix.cpu().numpy()
            
            # 観測情報を追跡
            observations = extender.track_observations(transport_matrix_np)
            
            # reconstruction_dataに保存
            if 'all_matches' not in reconstruction_data:
                reconstruction_data['all_matches'] = [[] for _ in range(len(camera_params_list))]
            
            # 新しいカメラの観測情報を追加
            reconstruction_data['all_matches'].append(observations)
            
            # 最適輸送行列自体も保存（後でBundle Adjustmentに使うため）
            if 'transport_matrices' not in reconstruction_data:
                reconstruction_data['transport_matrices'] = []
            reconstruction_data['transport_matrices'].append(transport_matrix_np)
    
    # Extract updated data
    points_3d, covariances_3d, colors_3d, alphas_3d = [], [], [], []
    for gauss in extender.existing_3d_gaussians:
        points_3d.append(gauss["center"])
        covariances_3d.append(gauss["covariance"])
        colors_3d.append(gauss["color"])
        alphas_3d.append(gauss["alpha"])
    
    points_3d = np.array(points_3d)
    covariances_3d = np.array(covariances_3d)
    colors_3d = np.array(colors_3d)
    alphas_3d = np.array(alphas_3d)
    
    # Update used images list
    used_images = reconstruction_data.get("used_images", []).copy()
    used_images.append(new_image_name)
    
    # Create updated results
    updated_results = {
        "existing_3d_gaussians": extender.existing_3d_gaussians,
        "camera_params_list": extender.camera_params_list,
        "points_3d": points_3d,
        "covariances_3d": covariances_3d,
        "color_3d": colors_3d,
        "alpha_3d": alphas_3d,
        "K": K_new,
        "used_images": used_images,
        "source_gaussians_data": source_gaussians_data,  # ViewpointExtender内で更新される
        "new_camera_R": R_new,
        "new_camera_t": t_new
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get image and fitted Gaussian info
    image_dir = os.path.join(args.data_dir, "images")
    colmap_dir = os.path.join(args.data_dir, args.colmap_dir)
    
    all_images = get_image_names(image_dir)
    gaussian_files = get_fitted_gaussians_info(args.fitted_gaussians_dir)
    
    # Filter to only images with fitted Gaussians (まあいらんかもしれない)
    available_images = [img for img in all_images if img in gaussian_files]
    
    if len(available_images) < 2:
        raise ValueError(f"Need at least 2 images with fitted Gaussians, found {len(available_images)}")
    
    print(f"Found {len(available_images)} images with fitted Gaussians")
    
    # Check for existing results
    initial_results_path = os.path.join(args.output_dir, "initial_3d_reconstruction.pkl")
    if os.path.exists(initial_results_path) and args.skip_existing:
        print(f"Loading existing initial reconstruction from {initial_results_path}")
        with open(initial_results_path, 'rb') as f:
            reconstruction_data = pickle.load(f)
        # Get used images
        used_images = reconstruction_data.get("used_images", [])
        print(f"Loaded reconstruction with {len(reconstruction_data['existing_3d_gaussians'])} 3D Gaussians")
        print(f"Used images: {used_images}")
    else:
        # Select initial pair
        img1, img2, selector = select_initial_pair(
        image_dir=image_dir,
        gaussian_files=gaussian_files,
        vocab_size=args.vocab_size,
        feature_type=args.feature_type,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        # device="cuda" if torch.cuda.is_available() else "cpu" 
    )
        
        # Perform initial reconstruction
        reconstruction_data = perform_initial_reconstruction(
            img1_name=img1,
            img2_name=img2,
            data_dir=args.data_dir,
            colmap_dir=args.colmap_dir,
            fitted_gaussians_dir=args.fitted_gaussians_dir,
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            target_volume=args.target_volume,
            device=device
        )
        
        # Get used images
        used_images = [img1, img2]

    # Set reference images (already used images)
    selector.add_reference_images(used_images)
    
    # Set source Gaussians data if available
    if "source_gaussians_data" in reconstruction_data:
        selector.set_source_gaussians_data(reconstruction_data["source_gaussians_data"])
    
    # Get remaining images to process
    remaining_images = [img for img in available_images if img not in used_images]
    
    # Start timing
    start_time = time.time()
    
    # Process each remaining image
    iteration = 1
    total_remaining = len(remaining_images)
    
    while remaining_images:
        print(f"\n--- Iteration {iteration}/{total_remaining} ---")
        
        # Select next best view
        next_image = selector.select_next_view_simple(remaining_images, n_select=1)[0]
        
        # Process the selected image
        iter_output_dir = os.path.join(args.output_dir, f"iteration_{iteration}")
        os.makedirs(iter_output_dir, exist_ok=True)
        
        # Add the new viewpoint
        # try:
        updated_data = add_new_viewpoint(
            reconstruction_data=reconstruction_data,
            new_image_name=next_image,
            fitted_gaussians_dir=args.fitted_gaussians_dir,
            output_dir=iter_output_dir,
            max_iterations=args.max_iterations,
            transport_threshold=args.transport_threshold,
            target_volume=args.target_volume,
            auto_threshold=args.auto_threshold,
            device=device
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
            
        # except Exception as e:
        #     print(f"Error processing {next_image}: {e}")

        #     traceback.print_exc()
            
        #     # Remove problematic image and continue
        #     remaining_images.remove(next_image)
        #     print(f"Skipping problematic image {next_image}")
        
        # Increment iteration counter
        iteration += 1
        
        # Calculate and print progress
        elapsed_time = time.time() - start_time
        processed_count = total_remaining - len(remaining_images)
        avg_time_per_image = elapsed_time / processed_count if processed_count > 0 else 0
        estimated_remaining = avg_time_per_image * len(remaining_images)
        
        print(f"\nProgress: {processed_count}/{total_remaining} images processed")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {estimated_remaining:.2f} seconds")
    
    # Final results
    final_output_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_output_dir, exist_ok=True)
    
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
    
    # Save final results
    final_results_path = os.path.join(final_output_dir, "final_reconstruction.pkl")
    with open(final_results_path, 'wb') as f:
        pickle.dump(reconstruction_data, f)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"Total images processed: {len(reconstruction_data['used_images'])}")
    print(f"Total 3D Gaussians: {len(reconstruction_data['existing_3d_gaussians'])}")
    print(f"Final results saved to {final_output_dir}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    
    # # Final Bundle Adjustment
    # if len(reconstruction_data["points_3d"]) > 0:
    #     print("\n--- Performing Bundle Adjustment ---")
        
    #     # 1. Build observation map from all accumulated data
    #     from src.optimizer.observation_builder import ObservationBuilder
        
    #     # ObservationBuilderはall_matchesまたはtransport_matricesから観測情報を構築
    #     observation_map = ObservationBuilder.build_observation_map(reconstruction_data)
    #     match_points_2d = ObservationBuilder.convert_to_match_points_2d(
    #         observation_map, 
    #         len(reconstruction_data["camera_params_list"])
    #     )
        
    #     # 最低限必要な観測数をチェック
    #     total_obs = sum(len(obs) for obs in match_points_2d)
    #     if total_obs < 10:
    #         print(f"Not enough observations ({total_obs}) for meaningful Bundle Adjustment. Skipping.")
    #     else:
    #         # Initialize Bundle Adjuster
            
            
    #         # 各カメラの内部パラメータリストを構築
    #         intrinsics_list = []
    #         for cam_idx in range(len(reconstruction_data["camera_params_list"])):
    #             # カメラ固有のKがあればそれを使用
    #             cam_key = f"camera{cam_idx+1}_K"
    #             if cam_key in reconstruction_data:
    #                 intrinsics_list.append(reconstruction_data[cam_key])
    #             elif "K" in reconstruction_data:
    #                 intrinsics_list.append(reconstruction_data["K"])
    #             else:
    #                 # Fallback to first camera's K
    #                 intrinsics_list.append(reconstruction_data.get("camera1_K", np.eye(3)))
            
    #         # BundleAdjuster初期化/最適化
    #         ba = BundleAdjuster(
    #             points_3d=reconstruction_data["points_3d"],
    #             camera_params_list=reconstruction_data["camera_params_list"],
    #             match_points_2d=match_points_2d,
    #             intrinsics_list=intrinsics_list,
    #             use_robust_loss=True,
    #             loss_scale=1.0
    #         )
            
    #         ba_results = ba.optimize(n_iterations=1000, verbose=True)
            
    #         if ba_results["success"]:
    #             reconstruction_data["camera_params_list"] = ba_results["optimized_cameras"]
    #             reconstruction_data["points_3d"] = ba_results["optimized_points"]
                
    #             # 更新されたカメラパラメータと3D点をViewpointExtenderの既存3Dガウスにも反映
    #             for i, point in enumerate(ba_results["optimized_points"]):
    #                 if i < len(reconstruction_data["existing_3d_gaussians"]):
    #                     reconstruction_data["existing_3d_gaussians"][i]["center"] = point
                
    #             # Export in COLMAP format
    #             colmap_dir = os.path.join(args.output_dir, "colmap_ba")
    #             ba.export_colmap_format(colmap_dir)
                
    #             print(f"Bundle Adjustment completed successfully. Results saved to {colmap_dir}")
    #         else:
    #             print(f"Bundle Adjustment failed: {ba_results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    args = parse_args()
    run_complete_pipeline(args)