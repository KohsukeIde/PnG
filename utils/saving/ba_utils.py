import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians
from utils.gs_pkl_loader import load_gaussians_torch

def visualize_ba_comparison(
    image_path: str,
    initial_2d_gaussians: TwoDGaussians,
    final_2d_gaussians: TwoDGaussians,
    original_2d_gaussians: TwoDGaussians,
    save_path: str
) -> None:
    """BAの前後での2D Gaussianの位置変化を可視化

    Args:
        image_path: 元画像のパス
        initial_2d_gaussians: BA前の3D Gaussianの投影
        final_2d_gaussians: BA後の3D Gaussianの投影
        original_2d_gaussians: フィッティングされた元の2D Gaussian
        save_path: 保存先パス
    """
    # 画像読み込み
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # プロット準備
    plt.figure(figsize=(15, 5))
    
    # 1. 元画像 + BA前の投影（赤）+ 元の2D Gaussian（青）
    plt.subplot(131)
    plt.imshow(img)
    plt.scatter(initial_2d_gaussians.means[:, 0], 
                initial_2d_gaussians.means[:, 1], 
                c='red', alpha=0.5, label='Before BA')
    plt.scatter(original_2d_gaussians.means[:, 0], 
                original_2d_gaussians.means[:, 1], 
                c='blue', alpha=0.5, label='Original 2D')
    plt.title('Before BA vs Original')
    plt.legend()
    
    # 2. 元画像 + BA後の投影（緑）+ 元の2D Gaussian（青）
    plt.subplot(132)
    plt.imshow(img)
    plt.scatter(final_2d_gaussians.means[:, 0], 
                final_2d_gaussians.means[:, 1], 
                c='green', alpha=0.5, label='After BA')
    plt.scatter(original_2d_gaussians.means[:, 0], 
                original_2d_gaussians.means[:, 1], 
                c='blue', alpha=0.5, label='Original 2D')
    plt.title('After BA vs Original')
    plt.legend()
    
    # 3. 対応関係を線で表示
    plt.subplot(133)
    plt.imshow(img)
    
    # BA前後の対応を線で結ぶ
    for i in range(len(initial_2d_gaussians.means)):
        plt.plot([initial_2d_gaussians.means[i, 0], final_2d_gaussians.means[i, 0]],
                 [initial_2d_gaussians.means[i, 1], final_2d_gaussians.means[i, 1]],
                 'r-', alpha=0.3)
    
    plt.scatter(initial_2d_gaussians.means[:, 0], 
                initial_2d_gaussians.means[:, 1], 
                c='red', alpha=0.5, label='Before BA')
    plt.scatter(final_2d_gaussians.means[:, 0], 
                final_2d_gaussians.means[:, 1], 
                c='green', alpha=0.5, label='After BA')
    plt.title('BA Movement')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rmse_progress(
    ba_results: Dict,
    save_path: str
) -> None:
    """BAの各イテレーションでのRMSEの推移をプロット

    Args:
        ba_results: BAの結果辞書（rmse_historyを含む）
        save_path: グラフの保存先パス
    """
    plt.figure(figsize=(10, 5))
    
    # RMSEの推移をプロット
    if 'rmse_history' in ba_results:
        iterations = range(len(ba_results['rmse_history']))
        plt.plot(iterations, ba_results['rmse_history'], 'b-', label='RMSE')
        
        # 初期値と最終値を強調
        if 'initial_rmse' in ba_results and not np.isnan(ba_results['initial_rmse']):
            plt.scatter([0], [ba_results['initial_rmse']], 
                       c='red', s=100, label='Initial RMSE')
        
        if 'final_rmse' in ba_results and not np.isnan(ba_results['final_rmse']):
            plt.scatter([len(iterations)-1], [ba_results['final_rmse']], 
                       c='green', s=100, label='Final RMSE')
            
            # 改善率を計算・表示
            if 'initial_rmse' in ba_results and not np.isnan(ba_results['initial_rmse']):
                improvement = ((ba_results['initial_rmse'] - ba_results['final_rmse']) 
                             / ba_results['initial_rmse'] * 100)
                plt.text(0.02, 0.98, f'Improvement: {improvement:.2f}%',
                        transform=plt.gca().transAxes,
                        verticalalignment='top')
        
        plt.xlabel('Iteration')
        plt.ylabel('RMSE (pixels)')
        plt.title('Bundle Adjustment RMSE Progress')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(save_path)
        plt.close()

def project_3d_gaussians(
    points_3d: np.ndarray,
    camera_params: Tuple[np.ndarray, np.ndarray],
    K: np.ndarray
) -> TwoDGaussians:
    """3D GaussianをカメラビューへProjection

    Args:
        points_3d: 3D点群 (N, 3)
        camera_params: (R, t) カメラの外部パラメータ
        K: カメラの内部パラメータ

    Returns:
        TwoDGaussians: 投影された2D Gaussians
    """
    R, t = camera_params
    
    # 3D点をカメラ座標系に変換
    points_cam = R @ points_3d.T + t[:, np.newaxis]
    
    # カメラ座標系の点を画像平面に投影
    points_2d = K @ points_cam
    points_2d = points_2d[:2] / points_2d[2]
    points_2d = points_2d.T
    
    # ダミーの共分散行列を生成
    num_points = points_2d.shape[0]
    dummy_covs = np.tile(np.eye(2)[np.newaxis, :, :], (num_points, 1, 1)).astype(np.float32)
    dummy_rgb = np.ones((num_points, 3), dtype=np.float32)  # 白色
    dummy_alpha = np.ones(num_points, dtype=np.float32)  # 完全不透明
    dummy_rotations = np.zeros(num_points, dtype=np.float32)  # 回転なし
    dummy_scales = np.ones((num_points, 2), dtype=np.float32)  # 等方的なスケール
    
    # TwoDGaussiansオブジェクトを作成して返す
    return TwoDGaussians(
        means=points_2d,
        covs=dummy_covs,       
        rgb=dummy_rgb,         
        alpha=dummy_alpha,     
        rotations=dummy_rotations,  
        scales=dummy_scales 
    )

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
                    C_out = (C_new * A_new + C_in * A_in * (1 - A_new)) / A_out
                else:
                    C_out = C_in

                # 書き戻し
                color_buffer[py, px] = C_out
                alpha_buffer[py, px] = A_out

    return color_buffer, alpha_buffer


def visualize_rendered_comparison(
    reconstruction_data: Dict,
    ba_results: Dict,
    image_name: str,
    original_image_path: str,
    save_path: str
) -> None:
    """BAの前後での3Dガウスのレンダリング結果を比較

    Args:
        reconstruction_data: 再構成データ
        ba_results: BAの結果
        image_name: 画像名
        original_image_path: 元画像のパス
        save_path: 保存先パス
    """
    # 画像読み込み
    orig_img = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
    if orig_img.shape[2] == 4:  # アルファチャンネルがある場合
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2RGBA)
    else:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    H, W = orig_img.shape[:2]
    
    # カメラパラメータ取得
    # 画像名からカメラインデックスを特定
    camera_idx = None
    for i, img in enumerate(reconstruction_data.get('used_images', [])):
        if img == image_name:
            camera_idx = i
            break
    
    if camera_idx is None:
        print(f"Warning: Camera index for {image_name} not found")
        return
    
    # BA前後のカメラパラメータ取得
    if 'initial_cameras' in ba_results:
        R_before, t_before = ba_results['initial_cameras'][camera_idx]
    else:
        # 初期カメラパラメータがない場合は既存のものを使用
        R_before, t_before = reconstruction_data['camera_params_list'][camera_idx]
    
    if 'optimized_cameras' in ba_results:
        R_after, t_after = ba_results['optimized_cameras'][camera_idx]
    else:
        print(f"Warning: Optimized camera for {image_name} not found")
        return
    
    # 内部パラメータ
    if 'K' in reconstruction_data:
        K = reconstruction_data['K']
    elif f'camera{camera_idx+1}_K' in reconstruction_data:
        K = reconstruction_data[f'camera{camera_idx+1}_K']
    else:
        print(f"Warning: Intrinsic matrix for {image_name} not found")
        return
    
    # BA前後の3D Gaussian
    if 'initial_points' in ba_results and 'optimized_points' in ba_results:
        points_before = ba_results['initial_points']
        points_after = ba_results['optimized_points']
    else:
        # BA結果に含まれない場合は既存のものを使用
        points_before = reconstruction_data['points_3d']
        points_after = reconstruction_data['points_3d']
    
    # 共分散行列、色、アルファは変わらないと仮定
    covariances = reconstruction_data['covariances_3d']
    colors = reconstruction_data['color_3d']
    alphas = reconstruction_data['alpha_3d']
    
    # BA前のレンダリング
    color_before, alpha_before = render_gaussians_alpha_blend(
        points_3d=points_before,
        covariances_3d=covariances,
        color_3d=colors,
        alpha_3d=alphas,
        R_cam=R_before,
        t_cam=t_before,
        K=K,
        out_width=W,
        out_height=H
    )
    
    # BA後のレンダリング
    color_after, alpha_after = render_gaussians_alpha_blend(
        points_3d=points_after,
        covariances_3d=covariances,
        color_3d=colors,
        alpha_3d=alphas,
        R_cam=R_after,
        t_cam=t_after,
        K=K,
        out_width=W,
        out_height=H
    )
    
    # 比較可視化
    plt.figure(figsize=(15, 5))
    
    # 元画像
    plt.subplot(131)
    plt.imshow(orig_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # BA前のレンダリング
    plt.subplot(132)
    plt.imshow(color_before)
    plt.title("Before BA")
    plt.axis('off')
    
    # BA後のレンダリング
    plt.subplot(133)
    plt.imshow(color_after)
    plt.title("After BA")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # エラー可視化（オリジナルとのピクセル差分）
    plt.figure(figsize=(15, 5))
    
    # BA前の誤差
    error_before = np.abs(color_before - orig_img/255.0).mean(axis=2)
    plt.subplot(131)
    plt.imshow(error_before, cmap='hot')
    plt.title(f"Error Before BA (Mean: {error_before.mean():.4f})")
    plt.colorbar()
    
    # BA後の誤差
    error_after = np.abs(color_after - orig_img/255.0).mean(axis=2)
    plt.subplot(132)
    plt.imshow(error_after, cmap='hot')
    plt.title(f"Error After BA (Mean: {error_after.mean():.4f})")
    plt.colorbar()
    
    # 誤差の改善
    error_diff = error_before - error_after
    plt.subplot(133)
    plt.imshow(error_diff, cmap='coolwarm')
    plt.title(f"Error Improvement (Mean: {error_diff.mean():.4f})")
    plt.colorbar()
    
    error_save_path = save_path.replace('.png', '_error.png')
    plt.tight_layout()
    plt.savefig(error_save_path)
    plt.close()

def visualize_ba_results(
    reconstruction_data: Dict,
    ba_results: Dict,
    data_dir: str,
    fitted_gaussians_dir: str,
    save_dir: str,
    device: Optional[torch.device] = None
) -> None:
    """Bundle Adjustmentの結果を包括的に可視化

    Args:
        reconstruction_data: 再構成データ
        ba_results: BAの結果
        data_dir: データディレクトリ
        fitted_gaussians_dir: フィッティングされたGaussianのディレクトリ
        save_dir: 保存先ディレクトリ
        device: 計算デバイス
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Visualizing BA results to {save_dir}...")
    
    # 1. RMSEの推移をプロット
    if 'rmse_history' in ba_results:
        rmse_plot_path = os.path.join(save_dir, "ba_rmse_progress.png")
        plot_rmse_progress(ba_results, rmse_plot_path)
        print(f"Saved RMSE plot to {rmse_plot_path}")
    
    # 2. 各画像での投影点比較とレンダリング比較
    for img_name in tqdm(reconstruction_data.get('used_images', []), desc="Generating visualizations"):
        img_path = os.path.join(data_dir, "images", img_name)
        
        # 2.1 投影点の比較
        camera_idx = None
        for i, img in enumerate(reconstruction_data.get('used_images', [])):
            if img == img_name:
                camera_idx = i
                break
        
        if camera_idx is not None:
            # カメラパラメータ取得
            if 'initial_cameras' in ba_results and camera_idx < len(ba_results['initial_cameras']):
                camera_before = ba_results['initial_cameras'][camera_idx]
            else:
                camera_before = reconstruction_data['camera_params_list'][camera_idx]
                
            if 'optimized_cameras' in ba_results and camera_idx < len(ba_results['optimized_cameras']):
                camera_after = ba_results['optimized_cameras'][camera_idx]
            else:
                camera_after = camera_before
            
            # BA前後の点群
            points_before = ba_results.get('initial_points', reconstruction_data['points_3d'])
            points_after = ba_results.get('optimized_points', reconstruction_data['points_3d'])
            
            # カメラの内部パラメータ
            if 'K' in reconstruction_data:
                K = reconstruction_data['K']
            elif f'camera{camera_idx+1}_K' in reconstruction_data:
                K = reconstruction_data[f'camera{camera_idx+1}_K']
            else:
                print(f"Warning: Intrinsic matrix for {img_name} not found")
                continue
            
            # BA前後の3D Gaussianを2Dに投影
            initial_2d = project_3d_gaussians(points_before, camera_before, K)
            final_2d = project_3d_gaussians(points_after, camera_after, K)
            
            # 元の2D Gaussianをロード
            gaussians_file = os.path.join(fitted_gaussians_dir, f"{img_name.split('.')[0]}_fitted_gaussians.pkl")
            if os.path.exists(gaussians_file):
                _, original_2d, _, _ = load_gaussians_torch(gaussians_file, device=device)
                
                # 2.1.1 2D投影点の比較可視化
                vis_path = os.path.join(save_dir, f"ba_comparison_{img_name.split('.')[0]}.png")
                visualize_ba_comparison(img_path, initial_2d, final_2d, original_2d, vis_path)
                
                # 2.1.2 3Dレンダリング比較の可視化
                render_path = os.path.join(save_dir, f"ba_render_{img_name.split('.')[0]}.png")
                visualize_rendered_comparison(
                    reconstruction_data=reconstruction_data,
                    ba_results=ba_results,
                    image_name=img_name,
                    original_image_path=img_path,
                    save_path=render_path
                )
            else:
                print(f"Warning: Fitted gaussians for {img_name} not found at {gaussians_file}")
    
    print("Visualization completed.")
    
    
def visualize_initial_pair_renders(
    reconstruction_data: Dict,
    ba_results: Dict,
    data_dir: str,
    save_dir: str
) -> None:
    """初期ペア（最初の2つのカメラ）についてBA前後のガウシアンレンダリング結果を比較視覚化する
    
    主な出力:
    - rendered_splats_camX_before_ba.png: BA前のレンダリング結果(BGRAフォーマット)
    - rendered_splats_camX_after_ba.png: BA後のレンダリング結果(BGRAフォーマット)
    - comparison_camX.png: 元画像とBA前後の結果を並べた比較画像
    - comparison_error_camX.png: 誤差解析画像
    
    Args:
        reconstruction_data: 再構成データ
        ba_results: BAの結果
        data_dir: データディレクトリ（オリジナル画像があるディレクトリ）
        save_dir: 保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n--- Visualizing BA Results for Initial Camera Pair ---")
    
    # 初期ペア（最初の2つのカメラ）のインデックスを取得
    if len(reconstruction_data.get('used_images', [])) < 2:
        print("Warning: Not enough cameras for initial pair comparison")
        return
    
    # 初期ペアのカメラインデックス
    camera_indices = [0, 1]  # 最初の2つのカメラ
    
    # 画像ディレクトリ
    image_dir = os.path.join(data_dir, "images")
    
    # Transport valuesの取得
    transport_values = None
    if 'transport_values' in reconstruction_data:
        transport_values = reconstruction_data['transport_values']
        print(f"Using transport values from reconstruction data: min={transport_values.min():.4f}, "
              f"max={transport_values.max():.4f}, mean={transport_values.mean():.4f}")
    else:
        print("Warning: No transport values found in reconstruction data. Using default alpha values only.")
    
    # 各カメラに対して処理
    for idx in camera_indices:
        image_name = reconstruction_data['used_images'][idx]
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found")
            continue
        
        print(f"Rendering comparison for camera {idx+1} ({image_name})...")
        
        # カメラパラメータ取得
        if 'initial_cameras' in ba_results and idx < len(ba_results['initial_cameras']):
            R_before, t_before = ba_results['initial_cameras'][idx]
        else:
            R_before, t_before = reconstruction_data['camera_params_list'][idx]
            
        if 'optimized_cameras' in ba_results and idx < len(ba_results['optimized_cameras']):
            R_after, t_after = ba_results['optimized_cameras'][idx]
        else:
            R_after, t_after = R_before, t_before
        
        # 内部パラメータ
        if 'K' in reconstruction_data:
            K = reconstruction_data['K']
        elif f'camera{idx+1}_K' in reconstruction_data:
            K = reconstruction_data[f'camera{idx+1}_K']
        else:
            print(f"Warning: Intrinsic matrix for {image_name} not found")
            continue
        
        # 画像読み込み
        orig_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if orig_img.shape[2] == 4:  # アルファチャンネルがある場合
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGRA2RGBA)
        else:
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        H, W = orig_img.shape[:2]
        
        # BA前後の3D Gaussian
        if 'initial_points' in ba_results and 'optimized_points' in ba_results:
            points_before = ba_results['initial_points']
            points_after = ba_results['optimized_points']
        else:
            points_before = reconstruction_data['points_3d']
            points_after = reconstruction_data['points_3d']
        
        # 共分散行列、色、アルファ
        covariances = reconstruction_data['covariances_3d']
        colors = reconstruction_data['color_3d']
        alphas = reconstruction_data['alpha_3d']
        
        # BA前のレンダリング
        color_before, alpha_before = render_gaussians_alpha_blend(
            points_3d=points_before,
            covariances_3d=covariances,
            color_3d=colors,
            alpha_3d=alphas,
            R_cam=R_before,
            t_cam=t_before,
            K=K,
            out_width=W,
            out_height=H,
            transport=transport_values  # Apply transport values here
        )
        
        # BA後のレンダリング
        color_after, alpha_after = render_gaussians_alpha_blend(
            points_3d=points_after,
            covariances_3d=covariances,
            color_3d=colors,
            alpha_3d=alphas,
            R_cam=R_after,
            t_cam=t_after,
            K=K,
            out_width=W,
            out_height=H,
            transport=transport_values  # Apply transport values here
        )
        
        # [1] BGRA形式のPNGとして保存（primary output - BGRA format with alpha channel）
        # BA前のレンダリング結果
        rendered_rgba_before = np.zeros((H, W, 4), dtype=np.float32)
        rendered_rgba_before[..., :3] = color_before
        rendered_rgba_before[..., 3] = alpha_before
        rendered_8u_before = np.clip(rendered_rgba_before*255.0, 0, 255).astype(np.uint8)
        rendered_8u_bgra_before = rendered_8u_before.copy()
        rendered_8u_bgra_before[..., 0] = rendered_8u_before[..., 2]  # R -> B
        rendered_8u_bgra_before[..., 2] = rendered_8u_before[..., 0]  # B -> R
        
        # BA後のレンダリング結果
        rendered_rgba_after = np.zeros((H, W, 4), dtype=np.float32)
        rendered_rgba_after[..., :3] = color_after
        rendered_rgba_after[..., 3] = alpha_after
        rendered_8u_after = np.clip(rendered_rgba_after*255.0, 0, 255).astype(np.uint8)
        rendered_8u_bgra_after = rendered_8u_after.copy()
        rendered_8u_bgra_after[..., 0] = rendered_8u_after[..., 2]  # R -> B
        rendered_8u_bgra_after[..., 2] = rendered_8u_after[..., 0]  # B -> R
        
        # PNG形式で保存（主要な出力）
        png_before_path = os.path.join(save_dir, f"rendered_splats_cam{idx+1}_before_ba.png")
        png_after_path = os.path.join(save_dir, f"rendered_splats_cam{idx+1}_after_ba.png")
        cv2.imwrite(png_before_path, rendered_8u_bgra_before)
        cv2.imwrite(png_after_path, rendered_8u_bgra_after)
        print(f"Saved raw renderings to {png_before_path} and {png_after_path}")
        
        # [2] 比較可視化（secondary output - for visual comparison）
        # 保存したファイルを読み込み直して使用（描画スタイルを統一）
        rendered_before_img = cv2.imread(png_before_path, cv2.IMREAD_UNCHANGED)
        rendered_after_img = cv2.imread(png_after_path, cv2.IMREAD_UNCHANGED)
        
        # BGRからRGBに変換
        rendered_before_rgb = cv2.cvtColor(rendered_before_img, cv2.COLOR_BGRA2RGBA)
        rendered_after_rgb = cv2.cvtColor(rendered_after_img, cv2.COLOR_BGRA2RGBA)
        
        plt.figure(figsize=(15, 5))
        
        # 元画像
        plt.subplot(131)
        plt.imshow(orig_img)
        plt.title(f"Original: Camera {idx+1}")
        plt.axis('off')
        
        # BA前のレンダリング
        plt.subplot(132)
        plt.imshow(rendered_before_rgb)
        plt.title("Before BA")
        plt.axis('off')
        
        # BA後のレンダリング
        plt.subplot(133)
        plt.imshow(rendered_after_rgb)
        plt.title("After BA")
        plt.axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(save_dir, f"comparison_cam{idx+1}.png")
        plt.savefig(comparison_path)
        plt.close()
        
        # [3] エラー可視化（secondary output - for error analysis）
        # アルファチャンネルを考慮した誤差計算
        alpha_mask_before = rendered_before_rgb[..., 3:4] / 255.0
        alpha_mask_after = rendered_after_rgb[..., 3:4] / 255.0
        
        # アルファをRGBに適用
        rendered_before_premult = rendered_before_rgb[..., :3] / 255.0 * alpha_mask_before
        rendered_after_premult = rendered_after_rgb[..., :3] / 255.0 * alpha_mask_after
        
        # 元画像を[0,1]範囲に正規化
        orig_norm = orig_img[..., :3] / 255.0  # RGB部分のみを使用
        
        # マスクされた領域のみで誤差計算
        error_before = np.abs(rendered_before_premult - orig_norm).mean(axis=2)
        error_after = np.abs(rendered_after_premult - orig_norm).mean(axis=2)
        
        plt.figure(figsize=(15, 5))
        
        # BA前の誤差
        plt.subplot(131)
        plt.imshow(error_before, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f"Error Before BA (Mean: {error_before.mean():.4f})")
        plt.colorbar()
        
        # BA後の誤差
        plt.subplot(132)
        plt.imshow(error_after, cmap='hot', vmin=0, vmax=0.5)
        plt.title(f"Error After BA (Mean: {error_after.mean():.4f})")
        plt.colorbar()
        
        # 誤差の改善
        error_diff = error_before - error_after
        plt.subplot(133)
        plt.imshow(error_diff, cmap='coolwarm', vmin=-0.2, vmax=0.2)
        plt.title(f"Error Improvement (Mean: {error_diff.mean():.4f})")
        plt.colorbar()
        
        error_comparison_path = os.path.join(save_dir, f"comparison_error_cam{idx+1}.png")
        plt.tight_layout()
        plt.savefig(error_comparison_path)
        plt.close()
        
        print(f"Saved comparison visualizations to {comparison_path} and {error_comparison_path}")
    
    print(f"Initial pair renderings saved to {save_dir}")