import os
import sys
import pickle
import argparse 
import torch
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch
from utils.saving.geometry_utils import save_ellipsoids_as_ply, save_point_cloud_as_ply
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from utils.export.export_utils import export_points_as_ply
from utils.saving.ba_utils import render_gaussians_alpha_blend


sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor

def parse_args():
    """Parse command-line arguments for path configuration.
    """
    parser = argparse.ArgumentParser(description="Pipeline to reconstruct 3D ellipsoids from 2D Gaussian data.")

    parser.add_argument(
        "--data_dir",
        type=str,
        # default="/Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/textureless",
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63",
        help="Path to the main data directory (e.g. DTU scan folder)."
    )
    parser.add_argument(
        "--data_dir_gmm",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked",
        # default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_200gs_5kiter_masked_sequential",
        # default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/house_100gs_10kiter_masked",
        
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
        default="0009.png",
        help="Filename of the first image."
    )
    parser.add_argument(
        "--image2_name",
        type=str,
        default="0012.png",
        help="Filename of the second image."
    )
    parser.add_argument(
        "--gaussians1_filename",
        type=str,
        default="0009_fitted_gaussians.pkl",
        # default="fitted_gaussians_0022.pkl",
        help="Filename of the first fitted Gaussians pickle."
    )
    parser.add_argument(
        "--gaussians2_filename",
        type=str,
        default="0012_fitted_gaussians.pkl",
        # default="fitted_gaussians_0023.pkl",
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
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/nerf_synthetic/materials/",
        help="Path to the directory containing NeRF transforms.json file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save output files."
    )
    parser.add_argument(
        "--render_gaussians",
        action="store_true",
        default=True,
        help="Render Gaussians for OT optimized/SIFT base camera pose."
    )

    return parser.parse_args()



def load_nerf_intrinsics(data_dir: str) -> np.ndarray:
    """NeRF形式のカメラ内部パラメータを読み込む関数
    
    Args:
        data_dir: NeRFデータディレクトリのパス
        
    Returns:
        K: 3x3カメラ内部パラメータ行列
    """
    transforms_file = os.path.join(data_dir, 'transforms_train.json')
    
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

def analyze_transport_matrices(
    solver, 
    F_optimized, 
    F_sift, 
    gaussians1, 
    gaussians2, 
    image1_path, 
    image2_path, 
    output_dir
):
    """最適化されたF行列とSIFTベースのF行列による輸送行列を分析・比較する
    
    Args:
        solver: OptimalTransportSolverインスタンス
        F_optimized: 最適化で得られた基本行列
        F_sift: SIFTで得られた基本行列
        gaussians1: 1枚目の画像の2Dガウス
        gaussians2: 2枚目の画像の2Dガウス
        image1_path: 1枚目の画像パス
        image2_path: 2枚目の画像パス
        output_dir: 出力ディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 両方のF行列から輸送行列を計算
    device = solver.device
    F_opt_tensor = torch.tensor(F_optimized, dtype=torch.float32, device=device)
    F_sift_tensor = torch.tensor(F_sift, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # コスト行列計算
        cost_matrix_opt = solver.compute_cost_matrix_fundamental(F_opt_tensor)
        cost_matrix_sift = solver.compute_cost_matrix_fundamental(F_sift_tensor)
        
        # 輸送行列計算
        transport_opt = solver.unbalanced_sinkhorn_algorithm(cost_matrix_opt)
        transport_sift = solver.unbalanced_sinkhorn_algorithm(cost_matrix_sift)
        
        # NumPy配列に変換
        T_opt = transport_opt.cpu().numpy()
        T_sift = transport_sift.cpu().numpy()
    
    # 1. 輸送行列のヒートマップ可視化
    visualize_transport_matrices(T_opt, T_sift, output_dir)
    
    # 2. 高輸送量ペアの抽出と可視化
    visualize_high_transport_pairs(
        T_opt, T_sift, gaussians1, gaussians2, 
        image1_path, image2_path, output_dir, 
        top_n=10  # 上位100ペアを可視化
    )
    
    # 3. 対応点の性質分析
    analyze_correspondence_properties(
        T_opt, T_sift, gaussians1, gaussians2, output_dir
    )
    
def visualize_transport_matrices(T_opt, T_sift, output_dir):
    """輸送行列をヒートマップとして可視化
    
    Args:
        T_opt: 最適化で得られた輸送行列
        T_sift: SIFTで得られた輸送行列
        output_dir: 出力ディレクトリ
    """
    plt.figure(figsize=(20, 10))
    
    # 1. 最適化F行列の輸送行列
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(T_opt, cmap='hot', interpolation='nearest')
    plt.colorbar(im1)
    plt.title('Transport Matrix (Optimized F)')
    plt.xlabel('Image 2 Gaussians')
    plt.ylabel('Image 1 Gaussians')
    
    # 2. SIFT F行列の輸送行列
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(T_sift, cmap='hot', interpolation='nearest')
    plt.colorbar(im2)
    plt.title('Transport Matrix (SIFT F)')
    plt.xlabel('Image 2 Gaussians')
    plt.ylabel('Image 1 Gaussians')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transport_matrices_comparison.png'), dpi=300)
    plt.close()
    
    # 輸送行列の統計情報を分析
    analyze_transport_statistics(T_opt, T_sift, output_dir)
    
def analyze_transport_statistics(T_opt, T_sift, output_dir):
    """輸送行列の統計情報を分析し、テキストファイルに出力
    
    Args:
        T_opt: 最適化で得られた輸送行列
        T_sift: SIFTで得られた輸送行列
        output_dir: 出力ディレクトリ
    """
    stats_file = os.path.join(output_dir, 'transport_statistics.txt')
    
    with open(stats_file, 'w') as f:
        f.write("TRANSPORT MATRIX STATISTICS COMPARISON\n")
        f.write("=====================================\n\n")
        
        # 基本統計量
        f.write("Basic Statistics:\n")
        f.write(f"Optimized F - Min: {T_opt.min():.6f}, Max: {T_opt.max():.6f}, Mean: {T_opt.mean():.6f}, Sum: {T_opt.sum():.6f}\n")
        f.write(f"SIFT F      - Min: {T_sift.min():.6f}, Max: {T_sift.max():.6f}, Mean: {T_sift.mean():.6f}, Sum: {T_sift.sum():.6f}\n\n")
        
        # スパース性の分析
        nonzero_opt = np.count_nonzero(T_opt > 1e-5)
        nonzero_sift = np.count_nonzero(T_sift > 1e-5)
        total_elements = T_opt.size
        
        f.write("Sparsity Analysis (threshold = 1e-5):\n")
        f.write(f"Optimized F - Nonzero: {nonzero_opt}/{total_elements} ({nonzero_opt/total_elements*100:.2f}%)\n")
        f.write(f"SIFT F      - Nonzero: {nonzero_sift}/{total_elements} ({nonzero_sift/total_elements*100:.2f}%)\n\n")
        
        # 集中度の分析（上位N%の要素が全体の何%を占めるか）
        for top_percent in [1, 5, 10, 20]:
            num_elements = int(total_elements * top_percent / 100)
            
            # 要素をソートし、上位N%の合計を計算
            sorted_opt = np.sort(T_opt.flatten())[::-1]
            sorted_sift = np.sort(T_sift.flatten())[::-1]
            
            sum_top_opt = np.sum(sorted_opt[:num_elements])
            sum_top_sift = np.sum(sorted_sift[:num_elements])
            
            percent_opt = sum_top_opt / T_opt.sum() * 100
            percent_sift = sum_top_sift / T_sift.sum() * 100
            
            f.write(f"Concentration (top {top_percent}% elements):\n")
            f.write(f"Optimized F - Sum: {sum_top_opt:.6f}, Percentage of total: {percent_opt:.2f}%\n")
            f.write(f"SIFT F      - Sum: {sum_top_sift:.6f}, Percentage of total: {percent_sift:.2f}%\n\n")
        
        # 行/列の合計の分布
        row_sums_opt = np.sum(T_opt, axis=1)
        row_sums_sift = np.sum(T_sift, axis=1)
        col_sums_opt = np.sum(T_opt, axis=0)
        col_sums_sift = np.sum(T_sift, axis=0)
        
        f.write("Row Sums (Image 1 Gaussians):\n")
        f.write(f"Optimized F - Min: {row_sums_opt.min():.6f}, Max: {row_sums_opt.max():.6f}, Mean: {row_sums_opt.mean():.6f}\n")
        f.write(f"SIFT F      - Min: {row_sums_sift.min():.6f}, Max: {row_sums_sift.max():.6f}, Mean: {row_sums_sift.mean():.6f}\n\n")
        
        f.write("Column Sums (Image 2 Gaussians):\n")
        f.write(f"Optimized F - Min: {col_sums_opt.min():.6f}, Max: {col_sums_opt.max():.6f}, Mean: {col_sums_opt.mean():.6f}\n")
        f.write(f"SIFT F      - Min: {col_sums_sift.min():.6f}, Max: {col_sums_sift.max():.6f}, Mean: {col_sums_sift.mean():.6f}\n\n")
    
    print(f"Transport statistics saved to {stats_file}")
    
def visualize_high_transport_pairs(
    T_opt, T_sift, gaussians1, gaussians2, 
    image1_path, image2_path, output_dir, 
    top_n=100
):
    """高輸送量のペアを抽出し、画像上に可視化
    
    Args:
        T_opt: 最適化で得られた輸送行列
        T_sift: SIFTで得られた輸送行列
        gaussians1: 1枚目の画像の2Dガウス
        gaussians2: 2枚目の画像の2Dガウス
        image1_path: 1枚目の画像パス
        image2_path: 2枚目の画像パス
        output_dir: 出力ディレクトリ
        top_n: 可視化する上位ペアの数
    """
    # 画像読み込み
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Failed to load images from {image1_path} or {image2_path}")
        return
    
    # 結合画像を作成（左が画像1、右が画像2）
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 高さをそろえる
    h_max = max(h1, h2)
    img1_resized = cv2.copyMakeBorder(img1, 0, h_max - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img2_resized = cv2.copyMakeBorder(img2, 0, h_max - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # 高輸送量ペアの抽出
    # Optimized F
    opt_values = T_opt.flatten()
    opt_indices = np.argsort(opt_values)[::-1][:top_n]
    opt_pairs = [np.unravel_index(idx, T_opt.shape) for idx in opt_indices]
    
    # SIFT F
    sift_values = T_sift.flatten()
    sift_indices = np.argsort(sift_values)[::-1][:top_n]
    sift_pairs = [np.unravel_index(idx, T_sift.shape) for idx in sift_indices]
    
    # gaussians1とgaussians2の座標を取得
    means1 = gaussians1.means.cpu().numpy() if isinstance(gaussians1.means, torch.Tensor) else gaussians1.means
    means2 = gaussians2.means.cpu().numpy() if isinstance(gaussians2.means, torch.Tensor) else gaussians2.means
    
    # Optimized F のペアを可視化
    combined_opt = np.hstack([img1_resized, img2_resized])
    
    for (i, j), idx in zip(opt_pairs, opt_indices):
        # 画像1上の点
        pt1 = (int(means1[i, 0]), int(means1[i, 1]))
        # 画像2上の点（x座標をオフセット）
        pt2 = (int(means2[j, 0]) + w1, int(means2[j, 1]))
        
        # 輸送量が大きいほど線を太く、色を濃く
        transport_value = opt_values[idx]
        thickness = max(1, min(5, int(transport_value * 20)))
        color_intensity = min(255, int(transport_value * 1000))
        color = (0, color_intensity, 255 - color_intensity)  # 輸送量によって色が変化
        
        # 対応線の描画
        cv2.line(combined_opt, pt1, pt2, color, thickness)
        # 点も描画
        cv2.circle(combined_opt, pt1, 5, (0, 255, 0), -1)
        cv2.circle(combined_opt, pt2, 5, (0, 255, 0), -1)
    
    cv2.imwrite(os.path.join(output_dir, 'high_transport_pairs_optimized.png'), combined_opt)
    
    # SIFT F のペアを可視化
    combined_sift = np.hstack([img1_resized, img2_resized])
    
    for (i, j), idx in zip(sift_pairs, sift_indices):
        # 画像1上の点
        pt1 = (int(means1[i, 0]), int(means1[i, 1]))
        # 画像2上の点（x座標をオフセット）
        pt2 = (int(means2[j, 0]) + w1, int(means2[j, 1]))
        
        # 輸送量が大きいほど線を太く、色を濃く
        transport_value = sift_values[idx]
        thickness = max(1, min(5, int(transport_value * 20)))
        color_intensity = min(255, int(transport_value * 1000))
        color = (0, color_intensity, 255 - color_intensity)  # 輸送量によって色が変化
        
        # 対応線の描画
        cv2.line(combined_sift, pt1, pt2, color, thickness)
        # 点も描画
        cv2.circle(combined_sift, pt1, 5, (0, 255, 0), -1)
        cv2.circle(combined_sift, pt2, 5, (0, 255, 0), -1)
    
    cv2.imwrite(os.path.join(output_dir, 'high_transport_pairs_sift.png'), combined_sift)
    
    print(f"High transport pairs visualization saved to {output_dir}")
    
    # 上位ペアの対応関係をクロスチェックして分析
    analyze_correspondence_overlap(opt_pairs, sift_pairs, output_dir)
    
def analyze_correspondence_overlap(opt_pairs, sift_pairs, output_dir):
    """最適化FとSIFT Fから得られた対応点の重複を分析
    
    Args:
        opt_pairs: 最適化Fの上位対応ペア
        sift_pairs: SIFT Fの上位対応ペア
        output_dir: 出力ディレクトリ
    """
    # 対応ペアをセットに変換
    opt_set = set((i, j) for i, j in opt_pairs)
    sift_set = set((i, j) for i, j in sift_pairs)
    
    # 共通するペアを見つける
    common_pairs = opt_set.intersection(sift_set)
    
    # 分析結果を出力
    with open(os.path.join(output_dir, 'correspondence_overlap.txt'), 'w') as f:
        f.write("CORRESPONDENCE OVERLAP ANALYSIS\n")
        f.write("===============================\n\n")
        
        f.write(f"Number of top pairs analyzed: {len(opt_pairs)}\n")
        f.write(f"Common pairs between Optimized F and SIFT F: {len(common_pairs)}\n")
        f.write(f"Overlap percentage: {len(common_pairs)/len(opt_pairs)*100:.2f}%\n\n")
        
        # 完全一致しなくても、同じガウスが関与しているケースの分析
        # Image 1側のガウスの重複
        opt_img1_gaussians = set(i for i, _ in opt_pairs)
        sift_img1_gaussians = set(i for i, _ in sift_pairs)
        common_img1 = opt_img1_gaussians.intersection(sift_img1_gaussians)
        
        # Image 2側のガウスの重複
        opt_img2_gaussians = set(j for _, j in opt_pairs)
        sift_img2_gaussians = set(j for _, j in sift_pairs)
        common_img2 = opt_img2_gaussians.intersection(sift_img2_gaussians)
        
        f.write("Partial overlap analysis:\n")
        f.write(f"Image 1 Gaussians - Optimized F: {len(opt_img1_gaussians)}, SIFT F: {len(sift_img1_gaussians)}, Common: {len(common_img1)} ({len(common_img1)/len(opt_img1_gaussians)*100:.2f}%)\n")
        f.write(f"Image 2 Gaussians - Optimized F: {len(opt_img2_gaussians)}, SIFT F: {len(sift_img2_gaussians)}, Common: {len(common_img2)} ({len(common_img2)/len(opt_img2_gaussians)*100:.2f}%)\n\n")
        
        # 対応関係の詳細分析
        # 各Image 1ガウスがいくつのImage 2ガウスに対応しているか
        opt_img1_count = {}
        sift_img1_count = {}
        
        for i, _ in opt_pairs:
            opt_img1_count[i] = opt_img1_count.get(i, 0) + 1
        
        for i, _ in sift_pairs:
            sift_img1_count[i] = sift_img1_count.get(i, 0) + 1
        
        opt_img1_counts = list(opt_img1_count.values())
        sift_img1_counts = list(sift_img1_count.values())
        
        f.write("Correspondence distribution (Image 1 Gaussians):\n")
        f.write(f"Optimized F - Min: {min(opt_img1_counts)}, Max: {max(opt_img1_counts)}, Mean: {sum(opt_img1_counts)/len(opt_img1_counts):.2f}\n")
        f.write(f"SIFT F      - Min: {min(sift_img1_counts)}, Max: {max(sift_img1_counts)}, Mean: {sum(sift_img1_counts)/len(sift_img1_counts):.2f}\n\n")
        
        # 各Image 2ガウスがいくつのImage 1ガウスに対応しているか
        opt_img2_count = {}
        sift_img2_count = {}
        
        for _, j in opt_pairs:
            opt_img2_count[j] = opt_img2_count.get(j, 0) + 1
        
        for _, j in sift_pairs:
            sift_img2_count[j] = sift_img2_count.get(j, 0) + 1
        
        opt_img2_counts = list(opt_img2_count.values())
        sift_img2_counts = list(sift_img2_count.values())
        
        f.write("Correspondence distribution (Image 2 Gaussians):\n")
        f.write(f"Optimized F - Min: {min(opt_img2_counts)}, Max: {max(opt_img2_counts)}, Mean: {sum(opt_img2_counts)/len(opt_img2_counts):.2f}\n")
        f.write(f"SIFT F      - Min: {min(sift_img2_counts)}, Max: {max(sift_img2_counts)}, Mean: {sum(sift_img2_counts)/len(sift_img2_counts):.2f}\n")

def analyze_correspondence_overlap(opt_pairs, sift_pairs, output_dir):
    """最適化FとSIFT Fから得られた対応点の重複を分析
    
    Args:
        opt_pairs: 最適化Fの上位対応ペア
        sift_pairs: SIFT Fの上位対応ペア
        output_dir: 出力ディレクトリ
    """
    # 対応ペアをセットに変換
    opt_set = set((i, j) for i, j in opt_pairs)
    sift_set = set((i, j) for i, j in sift_pairs)
    
    # 共通するペアを見つける
    common_pairs = opt_set.intersection(sift_set)
    
    # 分析結果を出力
    with open(os.path.join(output_dir, 'correspondence_overlap.txt'), 'w') as f:
        f.write("CORRESPONDENCE OVERLAP ANALYSIS\n")
        f.write("===============================\n\n")
        
        f.write(f"Number of top pairs analyzed: {len(opt_pairs)}\n")
        f.write(f"Common pairs between Optimized F and SIFT F: {len(common_pairs)}\n")
        f.write(f"Overlap percentage: {len(common_pairs)/len(opt_pairs)*100:.2f}%\n\n")
        
        # 完全一致しなくても、同じガウスが関与しているケースの分析
        # Image 1側のガウスの重複
        opt_img1_gaussians = set(i for i, _ in opt_pairs)
        sift_img1_gaussians = set(i for i, _ in sift_pairs)
        common_img1 = opt_img1_gaussians.intersection(sift_img1_gaussians)
        
        # Image 2側のガウスの重複
        opt_img2_gaussians = set(j for _, j in opt_pairs)
        sift_img2_gaussians = set(j for _, j in sift_pairs)
        common_img2 = opt_img2_gaussians.intersection(sift_img2_gaussians)
        
        f.write("Partial overlap analysis:\n")
        f.write(f"Image 1 Gaussians - Optimized F: {len(opt_img1_gaussians)}, SIFT F: {len(sift_img1_gaussians)}, Common: {len(common_img1)} ({len(common_img1)/len(opt_img1_gaussians)*100:.2f}%)\n")
        f.write(f"Image 2 Gaussians - Optimized F: {len(opt_img2_gaussians)}, SIFT F: {len(sift_img2_gaussians)}, Common: {len(common_img2)} ({len(common_img2)/len(opt_img2_gaussians)*100:.2f}%)\n\n")
        
        # 対応関係の詳細分析
        # 各Image 1ガウスがいくつのImage 2ガウスに対応しているか
        opt_img1_count = {}
        sift_img1_count = {}
        
        for i, _ in opt_pairs:
            opt_img1_count[i] = opt_img1_count.get(i, 0) + 1
        
        for i, _ in sift_pairs:
            sift_img1_count[i] = sift_img1_count.get(i, 0) + 1
        
        opt_img1_counts = list(opt_img1_count.values())
        sift_img1_counts = list(sift_img1_count.values())
        
        f.write("Correspondence distribution (Image 1 Gaussians):\n")
        f.write(f"Optimized F - Min: {min(opt_img1_counts)}, Max: {max(opt_img1_counts)}, Mean: {sum(opt_img1_counts)/len(opt_img1_counts):.2f}\n")
        f.write(f"SIFT F      - Min: {min(sift_img1_counts)}, Max: {max(sift_img1_counts)}, Mean: {sum(sift_img1_counts)/len(sift_img1_counts):.2f}\n\n")
        
        # 各Image 2ガウスがいくつのImage 1ガウスに対応しているか
        opt_img2_count = {}
        sift_img2_count = {}
        
        for _, j in opt_pairs:
            opt_img2_count[j] = opt_img2_count.get(j, 0) + 1
        
        for _, j in sift_pairs:
            sift_img2_count[j] = sift_img2_count.get(j, 0) + 1
        
        opt_img2_counts = list(opt_img2_count.values())
        sift_img2_counts = list(sift_img2_count.values())
        
        f.write("Correspondence distribution (Image 2 Gaussians):\n")
        f.write(f"Optimized F - Min: {min(opt_img2_counts)}, Max: {max(opt_img2_counts)}, Mean: {sum(opt_img2_counts)/len(opt_img2_counts):.2f}\n")
        f.write(f"SIFT F      - Min: {min(sift_img2_counts)}, Max: {max(sift_img2_counts)}, Mean: {sum(sift_img2_counts)/len(sift_img2_counts):.2f}\n")

def analyze_correspondence_properties(T_opt, T_sift, gaussians1, gaussians2, output_dir, top_n=500):
    """対応点の性質（色、スケール、回転）を分析
    
    Args:
        T_opt: 最適化で得られた輸送行列
        T_sift: SIFTで得られた輸送行列
        gaussians1: 1枚目の画像の2Dガウス
        gaussians2: 2枚目の画像の2Dガウス
        output_dir: 出力ディレクトリ
        top_n: 分析する上位ペアの数
    """
    # 高輸送量ペアの抽出
    # Optimized F
    opt_values = T_opt.flatten()
    opt_indices = np.argsort(opt_values)[::-1][:top_n]
    opt_pairs = [np.unravel_index(idx, T_opt.shape) for idx in opt_indices]
    
    # SIFT F
    sift_values = T_sift.flatten()
    sift_indices = np.argsort(sift_values)[::-1][:top_n]
    sift_pairs = [np.unravel_index(idx, T_sift.shape) for idx in sift_indices]
    
    # RGB、スケール、回転の差異を計算
    def compute_property_differences(pairs, gaussians1, gaussians2):
        rgb_diffs = []
        scale_diffs = []
        rotation_diffs = []
        spatial_dists = []
        
        # tensorからnumpyに変換
        rgb1 = gaussians1.rgb.cpu().numpy() if isinstance(gaussians1.rgb, torch.Tensor) else gaussians1.rgb
        rgb2 = gaussians2.rgb.cpu().numpy() if isinstance(gaussians2.rgb, torch.Tensor) else gaussians2.rgb
        
        scales1 = gaussians1.scales.cpu().numpy() if isinstance(gaussians1.scales, torch.Tensor) else gaussians1.scales
        scales2 = gaussians2.scales.cpu().numpy() if isinstance(gaussians2.scales, torch.Tensor) else gaussians2.scales
        
        rot1 = gaussians1.rotations.cpu().numpy() if isinstance(gaussians1.rotations, torch.Tensor) else gaussians1.rotations
        rot2 = gaussians2.rotations.cpu().numpy() if isinstance(gaussians2.rotations, torch.Tensor) else gaussians2.rotations
        
        means1 = gaussians1.means.cpu().numpy() if isinstance(gaussians1.means, torch.Tensor) else gaussians1.means
        means2 = gaussians2.means.cpu().numpy() if isinstance(gaussians2.means, torch.Tensor) else gaussians2.means
        
        for i, j in pairs:
            # RGB差異（L2ノルム）
            rgb_diff = np.linalg.norm(rgb1[i] - rgb2[j])
            rgb_diffs.append(rgb_diff)
            
            # スケール差異（相対比）
            scale_ratio = np.zeros_like(scales1[i])
            nonzero_mask = (scales1[i] > 1e-10) & (scales2[j] > 1e-10)
            
            if np.any(nonzero_mask):
                ratio1 = np.zeros_like(scales1[i])
                ratio2 = np.zeros_like(scales1[i])
                
                valid_indices = np.where(scales2[j] > 1e-10)[0]
                if len(valid_indices) > 0:
                    ratio1[valid_indices] = scales1[i][valid_indices] / scales2[j][valid_indices]
                    
                valid_indices = np.where(scales1[i] > 1e-10)[0]
                if len(valid_indices) > 0:
                    ratio2[valid_indices] = scales2[j][valid_indices] / scales1[i][valid_indices]
                    
                scale_ratio[nonzero_mask] = np.maximum(ratio1[nonzero_mask], ratio2[nonzero_mask])
            
            # 無限大の値を除外
            scale_ratio = np.clip(scale_ratio, 0, 1e6)  # 適切な上限値に制限
            
            scale_diff = np.mean(scale_ratio)
            scale_diffs.append(scale_diff)
            
            # 回転差異（角度の最小差）
            rot_diff = min(abs(rot1[i] - rot2[j]) % (2*np.pi), (2*np.pi - abs(rot1[i] - rot2[j]) % (2*np.pi)))
            rotation_diffs.append(rot_diff)
            
            # 空間的な距離（簡易的なチェック）
            spatial_dist = np.linalg.norm(means1[i] - means2[j])
            spatial_dists.append(spatial_dist)
        
        return {
            'rgb_diffs': rgb_diffs,
            'scale_diffs': scale_diffs,
            'rotation_diffs': rotation_diffs,
            'spatial_dists': spatial_dists
        }
    
    # 各輸送行列の対応点性質の分析
    opt_props = compute_property_differences(opt_pairs, gaussians1, gaussians2)
    sift_props = compute_property_differences(sift_pairs, gaussians1, gaussians2)
    
    # 分析結果を保存
    with open(os.path.join(output_dir, 'correspondence_properties.txt'), 'w') as f:
        f.write("CORRESPONDENCE PROPERTIES ANALYSIS\n")
        f.write("=================================\n\n")
        
        f.write(f"Number of top pairs analyzed: {top_n}\n\n")
        
        # RGB差異
        f.write("RGB color differences (L2 norm):\n")
        f.write(f"Optimized F - Min: {min(opt_props['rgb_diffs']):.4f}, Max: {max(opt_props['rgb_diffs']):.4f}, Mean: {sum(opt_props['rgb_diffs'])/len(opt_props['rgb_diffs']):.4f}\n")
        f.write(f"SIFT F      - Min: {min(sift_props['rgb_diffs']):.4f}, Max: {max(sift_props['rgb_diffs']):.4f}, Mean: {sum(sift_props['rgb_diffs'])/len(sift_props['rgb_diffs']):.4f}\n\n")
        
        # スケール差異
        f.write("Scale differences (relative ratio):\n")
        f.write(f"Optimized F - Min: {min(opt_props['scale_diffs']):.4f}, Max: {max(opt_props['scale_diffs']):.4f}, Mean: {sum(opt_props['scale_diffs'])/len(opt_props['scale_diffs']):.4f}\n")
        f.write(f"SIFT F      - Min: {min(sift_props['scale_diffs']):.4f}, Max: {max(sift_props['scale_diffs']):.4f}, Mean: {sum(sift_props['scale_diffs'])/len(sift_props['scale_diffs']):.4f}\n\n")
        
        # 回転差異
        f.write("Rotation differences (radians):\n")
        f.write(f"Optimized F - Min: {min(opt_props['rotation_diffs']):.4f}, Max: {max(opt_props['rotation_diffs']):.4f}, Mean: {sum(opt_props['rotation_diffs'])/len(opt_props['rotation_diffs']):.4f}\n")
        f.write(f"SIFT F      - Min: {min(sift_props['rotation_diffs']):.4f}, Max: {max(sift_props['rotation_diffs']):.4f}, Mean: {sum(sift_props['rotation_diffs'])/len(sift_props['rotation_diffs']):.4f}\n\n")
        
        # 空間的距離
        f.write("Spatial distances (Euclidean distance):\n")
        f.write(f"Optimized F - Min: {min(opt_props['spatial_dists']):.4f}, Max: {max(opt_props['spatial_dists']):.4f}, Mean: {sum(opt_props['spatial_dists'])/len(opt_props['spatial_dists']):.4f}\n")
        f.write(f"SIFT F      - Min: {min(sift_props['spatial_dists']):.4f}, Max: {max(sift_props['spatial_dists']):.4f}, Mean: {sum(sift_props['spatial_dists'])/len(sift_props['spatial_dists']):.4f}\n\n")
        
        # 差異の総合分析
        f.write("Overall property differences analysis:\n")
        
        # RGB差異の比較
        rgb_diff_ratio = sum(opt_props['rgb_diffs'])/len(opt_props['rgb_diffs']) / (sum(sift_props['rgb_diffs'])/len(sift_props['rgb_diffs']))
        f.write(f"RGB difference ratio (Optimized F / SIFT F): {rgb_diff_ratio:.4f}\n")
        
        # スケール差異の比較
        scale_diff_ratio = sum(opt_props['scale_diffs'])/len(opt_props['scale_diffs']) / (sum(sift_props['scale_diffs'])/len(sift_props['scale_diffs']))
        f.write(f"Scale difference ratio (Optimized F / SIFT F): {scale_diff_ratio:.4f}\n")
        
        # 回転差異の比較
        rotation_diff_ratio = sum(opt_props['rotation_diffs'])/len(opt_props['rotation_diffs']) / (sum(sift_props['rotation_diffs'])/len(sift_props['rotation_diffs']))
        f.write(f"Rotation difference ratio (Optimized F / SIFT F): {rotation_diff_ratio:.4f}\n")
        
        # 空間的距離の比較
        spatial_dist_ratio = sum(opt_props['spatial_dists'])/len(opt_props['spatial_dists']) / (sum(sift_props['spatial_dists'])/len(sift_props['spatial_dists']))
        f.write(f"Spatial distance ratio (Optimized F / SIFT F): {spatial_dist_ratio:.4f}\n\n")
    
    print(f"Correspondence properties analysis saved to {os.path.join(output_dir, 'correspondence_properties.txt')}")
    
    # 視覚化のためのヒストグラム
    plt.figure(figsize=(20, 16))
    
    # RGB差異のヒストグラム
    plt.subplot(2, 2, 1)
    plt.hist(opt_props['rgb_diffs'], alpha=0.5, bins=20, label='Optimized F')
    plt.hist(sift_props['rgb_diffs'], alpha=0.5, bins=20, label='SIFT F')
    plt.xlabel('RGB Difference (L2 norm)')
    plt.ylabel('Frequency')
    plt.title('RGB Color Differences')
    plt.legend()
    
    # スケール差異のヒストグラム
    plt.subplot(2, 2, 2)
    plt.hist(opt_props['scale_diffs'], alpha=0.5, bins=20, label='Optimized F')
    plt.hist(sift_props['scale_diffs'], alpha=0.5, bins=20, label='SIFT F')
    plt.xlabel('Scale Difference (relative ratio)')
    plt.ylabel('Frequency')
    plt.title('Scale Differences')
    plt.legend()
    
    # 回転差異のヒストグラム
    plt.subplot(2, 2, 3)
    plt.hist(opt_props['rotation_diffs'], alpha=0.5, bins=20, label='Optimized F')
    plt.hist(sift_props['rotation_diffs'], alpha=0.5, bins=20, label='SIFT F')
    plt.xlabel('Rotation Difference (radians)')
    plt.ylabel('Frequency')
    plt.title('Rotation Differences')
    plt.legend()
    
    # 空間的距離のヒストグラム
    plt.subplot(2, 2, 4)
    plt.hist(opt_props['spatial_dists'], alpha=0.5, bins=20, label='Optimized F')
    plt.hist(sift_props['spatial_dists'], alpha=0.5, bins=20, label='SIFT F')
    plt.xlabel('Spatial Distance (pixels)')
    plt.ylabel('Frequency')
    plt.title('Spatial Distances')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correspondence_properties_histograms.png'), dpi=300)
    plt.close()
    
    print(f"Correspondence properties histograms saved to {os.path.join(output_dir, 'correspondence_properties_histograms.png')}")

def estimate_camera_pose_from_features(image1_path, image2_path, K1, K2, debug_dir=None):
    """特徴点ベースでカメラ姿勢推定を行う関数
    
    Args:
        image1_path: 1枚目の画像パス
        image2_path: 2枚目の画像パス
        K1: 1枚目のカメラ内部パラメータ
        K2: 2枚目のカメラ内部パラメータ
        debug_dir: デバッグ情報を保存するディレクトリ（省略可）
        
    Returns:
        R: カメラ2の回転行列（カメラ1基準）
        t: カメラ2の並進ベクトル（カメラ1基準）
        F: 基礎行列
        inlier_matches: インライアーとなったマッチング点
    """
    # 画像読み込み
    img1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Failed to load images: {image1_path} or {image2_path}")
    
    # デバッグ情報の出力
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "debug_image1.png"), img1)
        cv2.imwrite(os.path.join(debug_dir, "debug_image2.png"), img2)
        
        print(f"Image1 exists: {os.path.exists(image1_path)}")
        print(f"Image2 exists: {os.path.exists(image2_path)}")
        print(f"Image1 size: {img1.shape if img1 is not None else 'None'}")
        print(f"Image2 size: {img2.shape if img2 is not None else 'None'}")
    
    # グレースケール変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # SIFT特徴点検出
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    print(f"Detected {len(kp1)} keypoints in image1 and {len(kp2)} keypoints in image2")
    
    # 特徴点マッチング
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    print(f"特徴点マッチング数: {len(good_matches)}")
    
    # 十分なマッチングがない場合のチェック
    if len(good_matches) < 5:
        raise ValueError(f"Not enough good matches for fundamental matrix estimation: {len(good_matches)} < 5")
    
    # マッチした点の座標を取得
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    print(f"pts1.shape: {pts1.shape}, pts2.shape: {pts2.shape}")
    
    # マッチング結果を可視化（デバッグ用）
    if debug_dir:
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
        cv2.imwrite(os.path.join(debug_dir, "debug_matches.png"), match_img)
    
    # 基礎行列の計算（RANSAC）
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None or mask is None:
        raise ValueError("findFundamentalMat failed to find a solution")
    
    # インライアーのみ残す
    mask = mask.ravel().astype(bool)
    pts1_inliers = pts1[mask]
    pts2_inliers = pts2[mask]
    
    print(f"Found {np.sum(mask)} inliers for fundamental matrix")
    
    # 必要点数チェック
    if np.sum(mask) < 8:
        raise ValueError("Not enough inliers for reliable pose estimation")
    
    # 基礎行列からカメラ姿勢を復元
    E = K2.T @ F @ K1  # 基本行列の計算
    _, R, t, _ = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K1)
    
    # インライアーとなったマッチング
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    
    return R, t, F, inlier_matches

def compute_point_cloud_statistics(points_3d):
    """点群の統計情報を計算する
    
    Args:
        points_3d: 点群座標 (N, 3)
        
    Returns:
        stats: 統計情報を含む辞書
    """
    if len(points_3d) == 0:
        return {
            'num_points': 0,
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
            'median': None,
            'bounding_box': None
        }
    
    stats = {
        'num_points': len(points_3d),
        'min': points_3d.min(axis=0),
        'max': points_3d.max(axis=0),
        'mean': points_3d.mean(axis=0),
        'std': points_3d.std(axis=0),
        'median': np.median(points_3d, axis=0),
        'bounding_box': {
            'size': points_3d.max(axis=0) - points_3d.min(axis=0),
            'center': (points_3d.max(axis=0) + points_3d.min(axis=0)) / 2
        }
    }
    return stats

def save_statistics_to_file(stats, filename):
    """統計情報をテキストファイルに保存する
    
    Args:
        stats: 統計情報を含む辞書
        filename: 保存先ファイル名
    """
    with open(filename, 'w') as f:
        f.write(f"Number of points: {stats['num_points']}\n\n")
        
        if stats['num_points'] > 0:
            f.write(f"Min (x,y,z): {stats['min']}\n")
            f.write(f"Max (x,y,z): {stats['max']}\n")
            f.write(f"Mean (x,y,z): {stats['mean']}\n")
            f.write(f"Std (x,y,z): {stats['std']}\n")
            f.write(f"Median (x,y,z): {stats['median']}\n\n")
            
            f.write("Bounding Box:\n")
            f.write(f"  Size (x,y,z): {stats['bounding_box']['size']}\n")
            f.write(f"  Center (x,y,z): {stats['bounding_box']['center']}\n")

def compute_rotation_difference(R1, R2):
    """2つの回転行列間の差異を計算する
    
    Args:
        R1, R2: 3x3回転行列
        
    Returns:
        angle_deg: 回転差異の角度（度）
        frobenius_norm: 行列間のフロベニウスノルム
    """
    # R1とR2がどれだけ違うかを計算 (R_diff = R1 @ R2.T)
    R_diff = R1 @ R2.T
    
    # 回転行列から角度を計算
    trace = np.trace(R_diff)
    trace = min(3.0, max(-1.0, trace))  # 数値誤差対策
    angle_rad = np.arccos((trace - 1) / 2)
    angle_deg = angle_rad * 180 / np.pi
    
    # フロベニウスノルム（行列要素の二乗和の平方根）
    frobenius_norm = np.linalg.norm(R1 - R2, 'fro')
    
    return angle_deg, frobenius_norm

def compare_camera_poses(R1, t1, R2, t2, filename):
    """2つのカメラ姿勢を比較してファイルに出力
    
    Args:
        R1, t1: 1つ目のカメラの回転行列と平行移動ベクトル
        R2, t2: 2つ目のカメラの回転行列と平行移動ベクトル
        filename: 出力ファイル名
    """
    with open(filename, 'w') as f:
        f.write("CAMERA POSE COMPARISON\n")
        f.write("=====================\n\n")
        
        # 回転行列の出力
        f.write("Rotation Matrix 1 (optimize_with_RT):\n")
        for row in R1:
            f.write(f"  {row}\n")
        f.write("\n")
        
        f.write("Rotation Matrix 2 (SIFT):\n")
        for row in R2:
            f.write(f"  {row}\n")
        f.write("\n")
        
        # 平行移動ベクトルの出力
        f.write(f"Translation Vector 1 (optimize_with_RT): {t1}\n")
        f.write(f"Translation Vector 2 (SIFT): {t2}\n")
        f.write(f"Translation Vector 1 Norm: {np.linalg.norm(t1):.6f}\n")
        f.write(f"Translation Vector 2 Norm: {np.linalg.norm(t2):.6f}\n\n")
        
        # 方向の比較（コサイン類似度）
        if np.linalg.norm(t1) > 1e-8 and np.linalg.norm(t2) > 1e-8:
            cos_sim = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
            angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            angle_deg = angle_rad * 180 / np.pi
            f.write(f"Translation Direction Cosine Similarity: {cos_sim:.6f}\n")
            f.write(f"Translation Direction Angle Difference: {angle_deg:.2f} degrees\n\n")
        else:
            f.write("Cannot compute direction similarity (zero translation)\n\n")
        
        # 回転の差異
        angle_diff, frobenius_norm = compute_rotation_difference(R1, R2)
        f.write(f"Rotation Difference Angle: {angle_diff:.2f} degrees\n")
        f.write(f"Rotation Matrix Frobenius Norm Difference: {frobenius_norm:.6f}\n\n")
        
        # カメラ姿勢の差が再構成に与える影響
        f.write("ANALYSIS OF RECONSTRUCTION DIFFERENCES\n")
        f.write("====================================\n\n")
        
        # スケール差の原因分析
        if np.linalg.norm(t1) > 1e-8 and np.linalg.norm(t2) > 1e-8:
            t_scale_ratio = np.linalg.norm(t1) / np.linalg.norm(t2)
            f.write(f"Translation Scale Ratio (optimize_with_RT/SIFT): {t_scale_ratio:.6f}\n")
            if t_scale_ratio < 0.1:
                f.write("  NOTE: optimize_with_RT translation is much smaller than SIFT translation.\n")
                f.write("  This likely causes the large depth values in triangulation results.\n")
            elif t_scale_ratio > 10:
                f.write("  NOTE: optimize_with_RT translation is much larger than SIFT translation.\n")
        
        # 回転差の影響
        if angle_diff > 10:
            f.write(f"  WARNING: Large rotation difference ({angle_diff:.2f} degrees) between methods.\n")
            f.write("  This can cause significant differences in triangulation results.\n")

def compare_losses(solver, F_optimized, F_sift, filename):
    """最適化とSIFTによる基本行列のロスを比較する
    
    Args:
        solver: OptimalTransportSolverインスタンス
        F_optimized: optimize_with_RTで得られた基本行列（numpy array）
        F_sift: SIFTで得られた基本行列（numpy array）
        filename: 出力ファイル名
    """
    # numpy -> torch tensor変換
    device = solver.device
    F_optimized_tensor = torch.from_numpy(F_optimized).float().to(device)
    F_sift_tensor = torch.from_numpy(F_sift).float().to(device)
    
    # 各Fに対するコスト行列計算
    with torch.no_grad():
        cost_matrix_optimized = solver.compute_cost_matrix_fundamental(F_optimized_tensor)
        cost_matrix_sift = solver.compute_cost_matrix_fundamental(F_sift_tensor)
        
        # 各コスト行列に対する輸送行列計算
        transport_optimized = solver.unbalanced_sinkhorn_algorithm(cost_matrix_optimized)
        transport_sift = solver.unbalanced_sinkhorn_algorithm(cost_matrix_sift)
        
        # ロス計算
        loss_optimized = torch.sum(transport_optimized * cost_matrix_optimized).item()
        loss_sift = torch.sum(transport_sift * cost_matrix_sift).item()
        
        # コスト行列と輸送行列の統計情報
        opt_cost_stats = {
            'min': cost_matrix_optimized.min().item(),
            'max': cost_matrix_optimized.max().item(),
            'mean': cost_matrix_optimized.mean().item()
        }
        sift_cost_stats = {
            'min': cost_matrix_sift.min().item(),
            'max': cost_matrix_sift.max().item(),
            'mean': cost_matrix_sift.mean().item()
        }
        opt_transport_stats = {
            'min': transport_optimized.min().item(),
            'max': transport_optimized.max().item(),
            'mean': transport_optimized.mean().item(),
            'sum': transport_optimized.sum().item()
        }
        sift_transport_stats = {
            'min': transport_sift.min().item(),
            'max': transport_sift.max().item(),
            'mean': transport_sift.mean().item(),
            'sum': transport_sift.sum().item()
        }
    
    # 結果をファイルに書き出し
    with open(filename, 'w') as f:
        f.write("LOSS COMPARISON BETWEEN OPTIMIZE_WITH_RT AND SIFT\n")
        f.write("===============================================\n\n")
        
        f.write(f"Optimize_with_RT Loss: {loss_optimized:.6f}\n")
        f.write(f"SIFT Loss: {loss_sift:.6f}\n\n")
        
        # lossの比率
        if loss_sift > 0:
            ratio = loss_optimized / loss_sift
            f.write(f"Loss Ratio (Optimize_with_RT / SIFT): {ratio:.6f}\n\n")
        
        # 各行列の統計情報
        f.write("Cost Matrix Statistics:\n")
        f.write(f"  Optimize_with_RT: min={opt_cost_stats['min']:.6f}, max={opt_cost_stats['max']:.6f}, mean={opt_cost_stats['mean']:.6f}\n")
        f.write(f"  SIFT: min={sift_cost_stats['min']:.6f}, max={sift_cost_stats['max']:.6f}, mean={sift_cost_stats['mean']:.6f}\n\n")
        
        f.write("Transport Matrix Statistics:\n")
        f.write(f"  Optimize_with_RT: min={opt_transport_stats['min']:.6f}, max={opt_transport_stats['max']:.6f}, mean={opt_transport_stats['mean']:.6f}, sum={opt_transport_stats['sum']:.6f}\n")
        f.write(f"  SIFT: min={sift_transport_stats['min']:.6f}, max={sift_transport_stats['max']:.6f}, mean={sift_transport_stats['mean']:.6f}, sum={sift_transport_stats['sum']:.6f}\n\n")
        
        # 解析
        f.write("ANALYSIS:\n")
        if loss_optimized < loss_sift:
            f.write("  - Optimize_with_RT achieves lower loss as expected from optimization.\n")
        else:
            f.write("  - Unexpected result: SIFT solution has lower loss despite not being directly optimized for this cost function.\n")
        
        if opt_transport_stats['sum'] != sift_transport_stats['sum']:
            f.write(f"  - Transport matrices have different total mass (sum), which may affect loss comparison.\n")

def export_to_colmap_format(
    output_dir: str,
    points_3d: np.ndarray,
    camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
    intrinsics_list: List[np.ndarray],
    colors_3d: Optional[np.ndarray] = None,
    image_names: Optional[List[str]] = None
) -> None:
    """再構成データをCOLMAP形式でエクスポート（extrinsics_visualizer.py用に最適化）
    
    Args:
        output_dir: 出力ディレクトリ
        points_3d: 3D点の座標 [N, 3]
        camera_params_list: カメラごとの (R, t) のリスト
        intrinsics_list: カメラ内部パラメータ行列のリスト
        colors_3d: 3D点のRGB色（省略可）[N, 3]
        image_names: 画像名のリスト（省略可）
    """
    import scipy.spatial.transform as transform
    
    os.makedirs(output_dir, exist_ok=True)
    num_cameras = len(camera_params_list)
    
    # 1. 内部パラメータを保存 (cameras.txt)
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        # 各カメラの内部パラメータ（ここではすべて同じと仮定）
        for cam_id, K in enumerate(intrinsics_list):
            camera_id = cam_id + 1  # カメラIDは1から始まる
            width = int(K[0, 2] * 2)  # 主点座標から幅を推定
            height = int(K[1, 2] * 2)  # 主点座標から高さを推定
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # PINHOLE モデルを使用
            f.write(f"{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
    
    # 2. 3D点群を保存 (points3D.txt) - TRACKデータ付き
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for i, point in enumerate(points_3d):
            point_id = i + 1
            
            # 色情報
            if colors_3d is not None and i < len(colors_3d):
                color = colors_3d[i]
                if color.max() <= 1.0:
                    r, g, b = (color * 255).astype(np.uint8)
                else:
                    r, g, b = color.astype(np.uint8)
            else:
                r, g, b = 255, 255, 255
            
            # TRACK情報（各カメラからの観測）
            track_str = ""
            for cam_idx in range(num_cameras):
                # 各カメラが各点を観測
                image_id = cam_idx + 1
                point2d_idx = i  # 点のインデックスをそのまま使用
                track_str += f"{image_id} {point2d_idx} "
            
            # 点の座標、色情報、TRACK情報を書き込み
            f.write(f"{point_id} {point[0]} {point[1]} {point[2]} {r} {g} {b} 0.0 {track_str}\n")
    
    # 3. カメラ姿勢を保存 (images.txt) - 2D点情報付き
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (R, t) in enumerate(camera_params_list):
            image_id = i + 1
            camera_id = i + 1  # 各画像は対応するカメラIDを使用
            
            # 回転行列からクォータニオンに変換（COLMAPは[w,x,y,z]の順）
            rot = transform.Rotation.from_matrix(R)
            quat = rot.as_quat()  # scipy: [x,y,z,w]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]  # COLMAPは[w,x,y,z]
            
            # 画像名
            name = image_names[i] if image_names and i < len(image_names) else f"image_{i:08d}.png"
            
            # カメラパラメータ行
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n")
            
            # 2D点情報行 - すべての3D点がこのカメラから見えると仮定
            K = intrinsics_list[i] if i < len(intrinsics_list) else intrinsics_list[0]
            points2d_line = ""
            
            for j in range(len(points_3d)):
                # 投影点の座標（単純化のため画像中心付近に配置）
                x, y = K[0, 2], K[1, 2]
                point3d_id = j + 1
                points2d_line += f"{x} {y} {point3d_id} "
            
            f.write(f"{points2d_line}\n")
    
    # PLY形式の点群も出力
    export_points_as_ply(
        os.path.join(output_dir, 'points3D.ply'),
        points_3d,
        colors=colors_3d
    )
    
    # 使用方法のヒントを書き出す
    with open(os.path.join(output_dir, 'visualization_commands.txt'), 'w') as f:
        f.write("extrinsics_visualizer.pyでの可視化コマンド例:\n\n")
        f.write(f"# 実際のカメラ内部パラメータを使用（推奨）:\n")
        f.write(f"python extrinsics_visualizer.py --images {output_dir}/images.txt --points {output_dir}/points3D.txt --cameras {output_dir}/cameras.txt --use_true_intrinsics\n\n")
        f.write(f"# 近似のフラスタムを使用:\n")
        f.write(f"python extrinsics_visualizer.py --images {output_dir}/images.txt --points {output_dir}/points3D.txt\n")
    
    print(f"COLMAP形式でデータをエクスポートしました: {output_dir}")
    print(f"extrinsics_visualizer.pyでの可視化コマンド例:")
    print(f"python extrinsics_visualizer.py --images {output_dir}/images.txt --points {output_dir}/points3D.txt --cameras {output_dir}/cameras.txt --use_true_intrinsics")

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
    
    images_dir = os.path.join(data_dir, "images")
    image1_path = os.path.join(images_dir, image1_name)
    image2_path = os.path.join(images_dir, image2_name)
    print(image1_path)

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
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.01,
        lambda_mean=0.0,
        lambda_cov=0.0,
        lambda_color=0.5,
        lambda_epipolar=0.5,
        device=device
    )

    ##############################
    # 5.A) Fundamental matrix optimization (using R,t)
    ##############################
    print("\n--- Optimizing Fundamental Matrix ---")
    solver.optimize_with_RT(max_iter=1000, tol=1e-6)
    F_optimized = solver.f.detach().cpu().numpy()
    print("\nOptimized Fundamental matrix (from R,t):\n", F_optimized)

    ##############################
    # 5.B) SIFT-based camera pose estimation
    ##############################
    print("\n--- Estimating camera pose from SIFT features ---")
    try:
        R_sift, t_sift, F_sift, inlier_matches = estimate_camera_pose_from_features(
            image1_path, image2_path, K1, K2, debug_dir=args.output_dir)
        print(f"SIFT-based camera pose estimation successful with {len(inlier_matches)} inliers")
        print("R_sift:\n", R_sift)
        print("t_sift:\n", t_sift)
        print("F_sift:\n", F_sift)
        
        # SIFTとoptimize_with_RTのロスを比較
        losses_file = os.path.join(args.output_dir, 'loss_comparison.txt')
        compare_losses(solver, F_optimized, F_sift, losses_file)
        print(f"Saved loss comparison to {losses_file}")
        
    except Exception as e:
        print(f"Failed to estimate camera pose from SIFT features: {e}")
        print("Skipping SIFT-based triangulation")
        R_sift = None
        t_sift = None
        F_sift = None

    ##############################
    # 6.A) Final cost & unbalanced transport (original)
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
    # 7.A) Triangulate (original)
    ##############################
    # Get R,t from solver
    r_optimized = solver.rvec.detach().cpu().numpy()
    t_optimized = solver.tvec.detach().cpu().numpy()
    r_est, _ = cv2.Rodrigues(r_optimized) 
    
    t_norm = np.linalg.norm(t_optimized)
    if t_norm > 1e-10:
        t_optimized = t_optimized / t_norm
        
    print("r_est:\n", r_est)
    print("t_est:\n", t_optimized)
    
    reconstructor.set_camera_matrices_explicitly(
        r1=np.eye(3), 
        t1=np.zeros(3), 
        r2=r_est, 
        t2=t_optimized
    )
    R1=np.eye(3)
    t1=np.zeros(3)
    R2=r_est
    t2=t_optimized
    
    #dont delete any gaussian (UOTの枠組みでthresholdは必要なくなったので)
    threshold = 0.0
    reconstructor.triangulate_gaussian_centers(transport_matrix_np, threshold=threshold)
    points_3d_opt_rt = reconstructor.points_3d
    print(f"\nTriangulated {points_3d_opt_rt.shape[0]} 3D points")

    # 三角測量直後の統計情報を計算・保存（optimize_with_RT）
    stats_opt_rt_before_cov = compute_point_cloud_statistics(points_3d_opt_rt)
    stats_file_opt_rt_before_cov = os.path.join(args.output_dir, 'triangulated_points_opt_rt_before_cov_stats.txt')
    save_statistics_to_file(stats_opt_rt_before_cov, stats_file_opt_rt_before_cov)
    print(f"Saved optimize_with_RT point cloud statistics (before covariance estimation) to {stats_file_opt_rt_before_cov}")

    ##############################
    # 8) Compute Covariances with Volume Prior
    ##############################
    print("\n--- Computing 3D Gaussian Covariances with Volume Prior ---")
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=1.0, target_volume=target_volume)

    # 共分散最適化後の統計情報を計算・保存（optimize_with_RT）
    stats_opt_rt_after_cov = compute_point_cloud_statistics(reconstructor.points_3d)
    stats_file_opt_rt_after_cov = os.path.join(args.output_dir, 'triangulated_points_opt_rt_after_cov_stats.txt')
    save_statistics_to_file(stats_opt_rt_after_cov, stats_file_opt_rt_after_cov)
    print(f"Saved optimize_with_RT point cloud statistics (after covariance estimation) to {stats_file_opt_rt_after_cov}")

    # 結果保存ディレクトリの作成
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    ply_points_out = os.path.join(output_dir, 'triangulated_points.ply')
    os.makedirs(output_dir, exist_ok=True)
    
    camera_params_list = [(np.eye(3), np.zeros(3)), (r_est, t_optimized)]
    save_point_cloud_as_ply(points_3d_opt_rt, ply_points_out, camera_params=camera_params_list)

    ##############################
    # 6.B & 7.B) SIFT-based transport and triangulation
    ##############################
    if R_sift is not None and t_sift is not None and F_sift is not None:
        print("\n--- Using SIFT-based F for transport and triangulation ---")
        
        # SIFTから得られたFundamental matrixをtorch tensorに変換
        F_sift_tensor = torch.from_numpy(F_sift).float().to(device)
        
        # SIFTベースの輸送行列計算
        with torch.no_grad():
            cost_matrix_sift = solver.compute_cost_matrix_fundamental(F_sift_tensor)
            transport_matrix_sift = solver.unbalanced_sinkhorn_algorithm(cost_matrix_sift)
            transport_matrix_sift_np = transport_matrix_sift.cpu().numpy()
        
        # 新しいreconstructorインスタンスを作成
        reconstructor_sift = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, h_dummy)
        
        print("\n--- Identifying Source Gaussians (SIFT-based) ---")
        reconstructor_sift.identify_source_gaussians(
            transport_matrix=transport_matrix_sift_np,
            auto_threshold=False
        )
        
        # SIFTベースのR,tを設定
        reconstructor_sift.set_camera_matrices_explicitly(
            r1=np.eye(3), 
            t1=np.zeros(3), 
            r2=R_sift, 
            t2=t_sift.flatten()  # 形状を(3,)に変換
        )
        
        # TriangulateしてPLY保存
        reconstructor_sift.triangulate_gaussian_centers(transport_matrix_sift_np, threshold=threshold)
        points_3d_sift = reconstructor_sift.points_3d
        print(f"\nTriangulated {points_3d_sift.shape[0]} 3D points using SIFT-based pose")
        
        # SIFT点群の統計情報を計算・保存
        stats_sift_before_cov = compute_point_cloud_statistics(points_3d_sift)
        stats_file_sift_before_cov = os.path.join(output_dir, 'triangulated_points_sift_before_cov_stats.txt')
        save_statistics_to_file(stats_sift_before_cov, stats_file_sift_before_cov)
        print(f"Saved SIFT-based point cloud statistics (before covariance estimation) to {stats_file_sift_before_cov}")
        
        # SIFT方式の共分散計算
        print("\n--- Computing 3D Gaussian Covariances with Volume Prior (SIFT-based) ---")
        reconstructor_sift.compute_3d_gaussian_covariances(lambda_volume=1.0, target_volume=target_volume)

        # 共分散最適化後の統計情報を計算・保存（SIFT）
        stats_sift_after_cov = compute_point_cloud_statistics(reconstructor_sift.points_3d)
        stats_file_sift_after_cov = os.path.join(output_dir, 'triangulated_points_sift_after_cov_stats.txt')
        save_statistics_to_file(stats_sift_after_cov, stats_file_sift_after_cov)
        print(f"Saved SIFT-based point cloud statistics (after covariance estimation) to {stats_file_sift_after_cov}")

        # SIFT方式の色とアルファの計算
        print("\n--- Computing 3D Gaussian Colors (SIFT-based) ---")
        reconstructor_sift.compute_3d_gaussian_colors(color_mode="average")

        print("\n--- Computing 3D Gaussian Alphas (SIFT-based) ---")
        reconstructor_sift.compute_3d_gaussian_alphas(alpha_mode="max")

        # 比較ファイルの作成（三角測量直後）
        comparison_file_before_cov = os.path.join(output_dir, 'point_clouds_comparison_before_cov.txt')
        with open(comparison_file_before_cov, 'w') as f:
            f.write("COMPARISON BETWEEN OPTIMIZE_WITH_RT AND SIFT POINT CLOUDS (BEFORE COVARIANCE OPTIMIZATION)\n")
            f.write("===================================================================================\n\n")
            
            f.write(f"Points count - Optimize_with_RT: {stats_opt_rt_before_cov['num_points']}, SIFT: {stats_sift_before_cov['num_points']}\n\n")
            
            if stats_opt_rt_before_cov['num_points'] > 0 and stats_sift_before_cov['num_points'] > 0:
                f.write("Bounding Box Size Comparison:\n")
                f.write(f"  Optimize_with_RT: {stats_opt_rt_before_cov['bounding_box']['size']}\n")
                f.write(f"  SIFT: {stats_sift_before_cov['bounding_box']['size']}\n\n")
                
                f.write("Bounding Box Center Comparison:\n")
                f.write(f"  Optimize_with_RT: {stats_opt_rt_before_cov['bounding_box']['center']}\n")
                f.write(f"  SIFT: {stats_sift_before_cov['bounding_box']['center']}\n\n")
                
                # スケール比
                opt_rt_size = stats_opt_rt_before_cov['bounding_box']['size']
                sift_size = stats_sift_before_cov['bounding_box']['size']
                opt_rt_max_dim = max(opt_rt_size)
                sift_max_dim = max(sift_size)
                
                if sift_max_dim > 0:
                    scale_ratio = opt_rt_max_dim / sift_max_dim
                    f.write(f"Scale ratio (Optimize_with_RT / SIFT): {scale_ratio:.6f}\n")

        print(f"Saved point clouds comparison (before covariance optimization) to {comparison_file_before_cov}")

        # 比較ファイルの作成（共分散最適化後）
        comparison_file_after_cov = os.path.join(output_dir, 'point_clouds_comparison_after_cov.txt')
        with open(comparison_file_after_cov, 'w') as f:
            f.write("COMPARISON BETWEEN OPTIMIZE_WITH_RT AND SIFT POINT CLOUDS (AFTER COVARIANCE OPTIMIZATION)\n")
            f.write("===================================================================================\n\n")
            
            f.write(f"Points count - Optimize_with_RT: {stats_opt_rt_after_cov['num_points']}, SIFT: {stats_sift_after_cov['num_points']}\n\n")
            
            if stats_opt_rt_after_cov['num_points'] > 0 and stats_sift_after_cov['num_points'] > 0:
                f.write("Bounding Box Size Comparison:\n")
                f.write(f"  Optimize_with_RT: {stats_opt_rt_after_cov['bounding_box']['size']}\n")
                f.write(f"  SIFT: {stats_sift_after_cov['bounding_box']['size']}\n\n")
                
                f.write("Bounding Box Center Comparison:\n")
                f.write(f"  Optimize_with_RT: {stats_opt_rt_after_cov['bounding_box']['center']}\n")
                f.write(f"  SIFT: {stats_sift_after_cov['bounding_box']['center']}\n\n")
                
                # スケール比
                opt_rt_size = stats_opt_rt_after_cov['bounding_box']['size']
                sift_size = stats_sift_after_cov['bounding_box']['size']
                opt_rt_max_dim = max(opt_rt_size)
                sift_max_dim = max(sift_size)
                
                if sift_max_dim > 0:
                    scale_ratio = opt_rt_max_dim / sift_max_dim
                    f.write(f"Scale ratio (Optimize_with_RT / SIFT): {scale_ratio:.6f}\n")
                
                # 共分散最適化の成功率
                opt_rt_orig_count = stats_opt_rt_before_cov['num_points']
                sift_orig_count = stats_sift_before_cov['num_points']
                opt_rt_final_count = stats_opt_rt_after_cov['num_points']
                sift_final_count = stats_sift_after_cov['num_points']
                
                opt_rt_success_rate = opt_rt_final_count / opt_rt_orig_count * 100 if opt_rt_orig_count > 0 else 0
                sift_success_rate = sift_final_count / sift_orig_count * 100 if sift_orig_count > 0 else 0
                
                f.write("\nCovariance Optimization Success Rate:\n")
                f.write(f"  Optimize_with_RT: {opt_rt_success_rate:.1f}% ({opt_rt_final_count}/{opt_rt_orig_count})\n")
                f.write(f"  SIFT: {sift_success_rate:.1f}% ({sift_final_count}/{sift_orig_count})\n")

        print(f"Saved point clouds comparison (after covariance optimization) to {comparison_file_after_cov}")

        ply_points_sift_out = os.path.join(output_dir, 'triangulated_points_sift.ply')
        # t_siftもflatten()して正しい形状に変換
        camera_params_list_sift = [(np.eye(3), np.zeros(3)), (R_sift, t_sift.flatten())]
        save_point_cloud_as_ply(points_3d_sift, ply_points_sift_out, camera_params=camera_params_list_sift)
        print(f"Saved SIFT-based triangulated points to {ply_points_sift_out}")

        # カメラ姿勢の比較情報を出力
        camera_poses_file = os.path.join(output_dir, 'camera_poses_comparison.txt')
        compare_camera_poses(r_est, t_optimized, R_sift, t_sift.flatten(), camera_poses_file)
        print(f"Saved camera poses comparison to {camera_poses_file}")

    ##############################
    # 9) Compute color & alpha
    ##############################
    print("\n--- Computing 3D Gaussian Colors & Alphas ---")
    reconstructor.compute_3d_gaussian_colors(color_mode="average")
    reconstructor.compute_3d_gaussian_alphas(alpha_mode="max")

    ##############################
    # 10) Build ellipsoids => PLY
    ##############################
    # ply_out = os.path.join(output_dir, '3d_gaussians_ellipsoids.ply')
    # save_ellipsoids_as_ply(
    #     points_3d=reconstructor.points_3d,
    #     covariances_3d=reconstructor.covariances_3d,
    #     colors_3d=reconstructor.color_3d,
    #     alphas_3d=reconstructor.alpha_3d,
    #     filename=ply_out,
    #     use_alpha=True
    # )
    
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
    R2 = r_est
    t2 = t_optimized

    camera_params_list = [(R1, t1), (R2, t2)]
    
    # ply_out = os.path.join(output_dir, '3d_gaussians_ellipsoids_withCams.ply')
    # save_ellipsoids_as_ply(
    #     points_3d=reconstructor.points_3d,
    #     covariances_3d=reconstructor.covariances_3d,
    #     colors_3d=reconstructor.color_3d,
    #     alphas_3d=reconstructor.alpha_3d,
    #     filename=ply_out,
    #     camera_params=camera_params_list,
    #     use_alpha=True
    # )

    ##############################
    # 11) (Optional) Project 3D Gaussians back to 2D for debug
    ##############################
    if args.render_gaussians:
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

        # カメラオブジェクトがあればそれを使用、なければK1から直接サイズを計算
        if 'camera1' in locals() and camera1 is not None:
            out_width1 = int(camera1.K[0,2]*2)
            out_height1 = int(camera1.K[1,2]*2)
            K_render1 = camera1.K
        else:
            out_width1 = int(K1[0,2]*2)
            out_height1 = int(K1[1,2]*2)
            K_render1 = K1

        mixture_img1, coverage_img1 = render_gaussians_alpha_blend(
            points_3d=reconstructor.points_3d,
            covariances_3d=reconstructor.covariances_3d,
            color_3d=reconstructor.color_3d,
            alpha_3d=reconstructor.alpha_3d,
            R_cam=R_cam1,
            t_cam=t_cam1,
            K=K_render1,
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

        # カメラオブジェクトがあればそれを使用、なければK2から直接サイズを計算
        if 'camera2' in locals() and camera2 is not None:
            out_width2 = int(camera2.K[0,2]*2)
            out_height2 = int(camera2.K[1,2]*2)
            K_render2 = camera2.K
        else:
            out_width2 = int(K2[0,2]*2)
            out_height2 = int(K2[1,2]*2)
            K_render2 = K2

        mixture_img2, coverage_img2 = render_gaussians_alpha_blend(
            points_3d=reconstructor.points_3d,
            covariances_3d=reconstructor.covariances_3d,
            color_3d=reconstructor.color_3d,
            alpha_3d=reconstructor.alpha_3d,
            R_cam=R_cam2,
            t_cam=t_cam2,
            K=K_render2,  # K_render2を使用
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

        # SIFT-based rendering
        if R_sift is not None and t_sift is not None:
            print("Rendering from SIFT-based camera 1 viewpoint...")
            R_cam1_sift = np.eye(3)  # Camera 1 is our reference frame
            t_cam1_sift = np.zeros(3)

            out_width1_sift = int(K1[0,2]*2)
            out_height1_sift = int(K1[1,2]*2)
            K_render1_sift = K1

            # SIFTのtransportを使用
            transport_values_sift = None
            if hasattr(reconstructor_sift, 'transport_values') and len(reconstructor_sift.transport_values) > 0:
                transport_values_sift = reconstructor_sift.transport_values
                print(f"Using SIFT transport values from triangulation: min={transport_values_sift.min():.4f}, "
                      f"max={transport_values_sift.max():.4f}, mean={transport_values_sift.mean():.4f}")
            else:
                print("No SIFT transport values available")
                sys.exit(1)

            mixture_img1_sift, coverage_img1_sift = render_gaussians_alpha_blend(
                points_3d=reconstructor_sift.points_3d,
                covariances_3d=reconstructor_sift.covariances_3d,
                color_3d=reconstructor_sift.color_3d,
                alpha_3d=reconstructor_sift.alpha_3d,
                R_cam=R_cam1_sift,
                t_cam=t_cam1_sift,
                K=K_render1_sift,
                out_width=out_width1_sift,
                out_height=out_height1_sift,
                transport=transport_values_sift 
            )
            
            rendered_rgba1_sift = np.zeros((out_height1_sift, out_width1_sift, 4), dtype=np.float32)
            rendered_rgba1_sift[..., :3] = mixture_img1_sift
            rendered_rgba1_sift[..., 3] = coverage_img1_sift

            rendered_8u1_sift = np.clip(rendered_rgba1_sift*255.0, 0, 255).astype(np.uint8)
            
            rendered_8u_bgra1_sift = rendered_8u1_sift.copy()
            rendered_8u_bgra1_sift[...,0] = rendered_8u1_sift[...,2]
            rendered_8u_bgra1_sift[...,2] = rendered_8u1_sift[...,0]

            cv2.imwrite(os.path.join(output_dir, "rendered_splats_sift_cam1.png"), rendered_8u_bgra1_sift)
            print(f"Saved SIFT-based alpha-blended splatting for camera 1 to {os.path.join(output_dir, 'rendered_splats_sift_cam1.png')}")

            print("Rendering from SIFT-based camera 2 viewpoint...")
            R_cam2_sift = R_sift  # Camera 2's rotation relative to world
            t_cam2_sift = t_sift.flatten()  # Camera 2's translation relative to world

            out_width2_sift = int(K2[0,2]*2)
            out_height2_sift = int(K2[1,2]*2)
            K_render2_sift = K2

            mixture_img2_sift, coverage_img2_sift = render_gaussians_alpha_blend(
                points_3d=reconstructor_sift.points_3d,
                covariances_3d=reconstructor_sift.covariances_3d,
                color_3d=reconstructor_sift.color_3d,
                alpha_3d=reconstructor_sift.alpha_3d,
                R_cam=R_cam2_sift,
                t_cam=t_cam2_sift,
                K=K_render2_sift,
                out_width=out_width2_sift,
                out_height=out_height2_sift,
                transport=transport_values_sift 
            )
            
            rendered_rgba2_sift = np.zeros((out_height2_sift, out_width2_sift, 4), dtype=np.float32)
            rendered_rgba2_sift[..., :3] = mixture_img2_sift
            rendered_rgba2_sift[..., 3] = coverage_img2_sift

            rendered_8u2_sift = np.clip(rendered_rgba2_sift*255.0, 0, 255).astype(np.uint8)
            
            rendered_8u_bgra2_sift = rendered_8u2_sift.copy()
            rendered_8u_bgra2_sift[...,0] = rendered_8u2_sift[...,2]
            rendered_8u_bgra2_sift[...,2] = rendered_8u2_sift[...,0]

            cv2.imwrite(os.path.join(output_dir, "rendered_splats_sift_cam2.png"), rendered_8u_bgra2_sift)
            print(f"Saved SIFT-based alpha-blended splatting for camera 2 to {os.path.join(output_dir, 'rendered_splats_sift_cam2.png')}")
            print("\nDone.")

    ##############################
    # 12) Save final results　(NOT USING ANYMORE but keep for future reference)
    ##############################
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
    
    # SIFT方式の処理が完了した後、かつ最終的な結果保存の前
    if R_sift is not None and t_sift is not None and F_sift is not None:
        print("\n--- Analyzing Transport Matrices Comparison ---")
        analyze_transport_matrices(
            solver=solver,
            F_optimized=F_optimized,
            F_sift=F_sift,
            gaussians1=gaussians1,
            gaussians2=gaussians2,
            image1_path=image1_path,
            image2_path=image2_path,
            output_dir=os.path.join(output_dir, 'transport_analysis')
        )

    # COLMAP形式でエクスポート
    print("\n--- Exporting to COLMAP format for visualization ---")
    colmap_export_dir = os.path.join(output_dir, 'colmap')
    os.makedirs(colmap_export_dir, exist_ok=True)
    
    # 内部パラメータリスト
    intrinsics_list = [K1, K2]
    # 画像名リスト
    image_names = [image1_name, image2_name]
    
    export_to_colmap_format(
        output_dir=colmap_export_dir,
        points_3d=reconstructor.points_3d,
        camera_params_list=camera_params_list,
        intrinsics_list=intrinsics_list,
        colors_3d=reconstructor.color_3d,
        image_names=image_names
    )
    
    # SIFTベースの処理
    if R_sift is not None and t_sift is not None and F_sift is not None:
        # SIFTベースのCOLMAP形式でのエクスポート
        colmap_sift_dir = os.path.join(output_dir, 'colmap_sift')
        os.makedirs(colmap_sift_dir, exist_ok=True)
        
        camera_params_list_sift = [(np.eye(3), np.zeros(3)), (R_sift, t_sift.flatten())]
        
        export_to_colmap_format(
            output_dir=colmap_sift_dir,
            points_3d=points_3d_sift,
            camera_params_list=camera_params_list_sift,
            intrinsics_list=intrinsics_list,
            colors_3d=reconstructor_sift.color_3d if hasattr(reconstructor_sift, 'color_3d') else None,
            image_names=image_names
        )
        
        # 両方の結果を比較するためのヒントを表示
        print("\n両方の再構成結果を比較するためのコマンド例:")
        print(f"1. optimize_with_RT結果: python extrinsics_visualizer.py --images {colmap_export_dir}/images.txt --points {colmap_export_dir}/points3D.txt")
        print(f"2. SIFT結果: python extrinsics_visualizer.py --images {colmap_sift_dir}/images.txt --points {colmap_sift_dir}/points3D.txt")

if __name__ == '__main__':
    main()