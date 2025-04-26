#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cost‑function sanity‑check for OptimalTransportSolver
====================================================

✓  SIFT で得た (擬似)Ground‑truth Fundamental‑Matrix が
   ランダム行列より常に低 Loss になるかを検証するユニットテスト。

使い方:
$ python test_ot_cost.py \
      --data_dir        /path/to/DTU/scan63 \
      --data_dir_gmm    /path/to/fitted_gs/apple_32gs_10kiter_masked \
      --image1_name     0009.png \
      --image2_name     0012.png \
      --gaussians1_filename 0009_fitted_gaussians.pkl \
      --gaussians2_filename 0012_fitted_gaussians.pkl
"""
# -------------------------------------------------------------
import os, sys, argparse, json
import numpy as np
import torch
import cv2
from typing import Tuple, List

# ---- 外部モジュール（元パイプラインと同じ場所） ----------------
from utils.gs_pkl_loader           import load_gaussians_torch
from src.utils.colmap_utils        import load_cameras_from_colmap, load_images_from_colmap
from src.camera.camera_model       import CameraModel
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']


# ---- ここで SIFT‑based F を取得する最小関数 ----------------------
def estimate_F_sift(img1_path:str, img2_path:str,
                    K1:np.ndarray, K2:np.ndarray,
                    debug:bool=False)->Tuple[np.ndarray,np.ndarray]:
    """return: (F_sift, inlier_mask)"""
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        raise FileNotFoundError("failed to load images")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), None)
    bf  = cv2.BFMatcher()
    raw = bf.knnMatch(des1, des2, k=2)

    good=[]
    for m,n in raw:
        if m.distance<0.7*n.distance: good.append(m)
    if len(good)<8: raise RuntimeError("not enough matches")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None: raise RuntimeError("findFundamentalMat failed")
    inlier = mask.ravel().astype(bool)
    if debug:
        print(f"SIFT‑F inliers {inlier.sum()}/{len(inlier)}")
    return F, inlier

# ---- ランダム rank‑2 Fundamental 行列生成 ------------------------
def random_F(K1: np.ndarray = None, K2: np.ndarray = None, rank2: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OptimalTransportSolverのアプローチに倣った、ランダムな基本行列Fの生成

    Args:
        K1: 1番目のカメラの内部パラメータ行列
        K2: 2番目のカメラの内部パラメータ行列
        rank2: rank-2の行列を保証するかどうか（常にTrue）

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (F, R, t)
            F: 基本行列
            R: カメラ2からカメラ1への回転行列
            t: カメラ2からカメラ1への並進ベクトル
    """
    # ランダムな回転ベクトル生成
    rvec = np.random.randn(3)
    # 長さを調整（大きすぎる回転を避けるため、π以下にする）
    norm = np.linalg.norm(rvec)
    if norm > np.pi:
        rvec = rvec * (np.pi / norm)
    
    # Rodriguesの公式で回転行列に変換
    R, _ = cv2.Rodrigues(rvec)
    
    # ランダムな並進ベクトル生成（単位長さに正規化）
    t = np.random.randn(3)
    t = t / np.linalg.norm(t)
    
    # tから反対称行列（skew-symmetric matrix）を作成
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    
    # 本質行列 E = [t]_× R
    E = tx @ R
    
    # 内部パラメータがある場合は基本行列F=K2^-T E K1^-1に変換
    if K1 is not None and K2 is not None:
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        F = K2_inv.T @ E @ K1_inv
    else:
        # 内部パラメータがない場合はE=F（正規化座標系）
        F = E
    
    # 正規化して返す
    return F / np.linalg.norm(F), R, t

# ---- OT loss を計算する util -------------------------------------
def compute_ot_loss(solver:OptimalTransportSolver, F:np.ndarray, return_transport=False)->float:
    """
    輸送行列を用いてOTロスを計算する
    
    Args:
        solver: 最適輸送ソルバー
        F: 基本行列
        return_transport: 輸送行列も返すかどうか
    
    Returns:
        float: ロス値
        torch.Tensor (オプション): 輸送行列τ
    """
    with torch.no_grad():
        F_t  = torch.tensor(F, dtype=torch.float32, device=solver.device)
        C    = solver.compute_cost_matrix_fundamental(F_t)
        τ    = solver.unbalanced_sinkhorn_algorithm(C)
        loss = float( (τ*C).sum().item() )
        
        if return_transport:
            return loss, τ
        else:
            return loss

# -----------------------------------------------------------------
def parse_args()->argparse.Namespace:
    pa = argparse.ArgumentParser(description="OT cost sanity‑check")
    pa.add_argument("--data_dir",        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63")
    pa.add_argument("--data_dir_gmm",    default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_32gs_10kiter_masked")
    pa.add_argument("--colmap_dir",      default="sparse/0")
    pa.add_argument("--image1_name",     default="0009.png")
    pa.add_argument("--image2_name",     default="0012.png")
    pa.add_argument("--gaussians1_filename", default="0009_fitted_gaussians.pkl")
    pa.add_argument("--gaussians2_filename", default="0012_fitted_gaussians.pkl")
    pa.add_argument("-n","--n_random",   type=int, default=500,
                    help="number of random F matrices")
    pa.add_argument("--lambda_color",    type=float, default=0.0)
    pa.add_argument("--lambda_epi",      type=float, default=0.5)
    return pa.parse_args()

# -----------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    # ---- load Gaussians -----------------------------------------
    g1_path = os.path.join(args.data_dir_gmm, args.gaussians1_filename)
    g2_path = os.path.join(args.data_dir_gmm, args.gaussians2_filename)
    _, g1, _, K1 = load_gaussians_torch(g1_path, device)
    _, g2, _, K2 = load_gaussians_torch(g2_path, device)

    # ---- COLMAP intrinsics overwrite (same手順) -----------------
    colmap_dir = os.path.join(args.data_dir, args.colmap_dir)
    cams   = load_cameras_from_colmap(colmap_dir)
    imgs   = load_images_from_colmap(colmap_dir)
    id1    = {d['name']:i for i,d in imgs.items()}[args.image1_name]
    id2    = {d['name']:i for i,d in imgs.items()}[args.image2_name]
    K1 = CameraModel(cams[ imgs[id1]['camera_id'] ], id1, imgs).K
    K2 = CameraModel(cams[ imgs[id2]['camera_id'] ], id2, imgs).K

    # ---- build solver (no optimisation) -------------------------
    solver = OptimalTransportSolver(
        g1, g2, K1, K2,
        epsilon=0.01,
        lambda_color=args.lambda_color,
        lambda_epipolar=args.lambda_epi,
        lambda_mean=0.0, lambda_cov=0.0,
        device=device
    )

    # ---- SIFT (pseudo ground‑truth) -----------------------------
    img_dir   = os.path.join(args.data_dir,"images")
    img1_path = os.path.join(img_dir, args.image1_name)
    img2_path = os.path.join(img_dir, args.image2_name)
    F_gt,_ = estimate_F_sift(img1_path, img2_path, K1, K2, debug=True)
    print("F_sift:\n",F_gt)

    # ---- compute losses ----------------------------------------
    loss_gt, T_gt = compute_ot_loss(solver, F_gt, return_transport=True)
    print(f"\nLoss(GT) = {loss_gt:.6f}")

    # 輸送行列の保存ディレクトリを作成
    transport_dir = os.path.join("results", "transport_matrices")
    os.makedirs(transport_dir, exist_ok=True)
    
    # SIFT-Fの輸送行列を保存
    import matplotlib.pyplot as plt
    T_gt_np = T_gt.cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(T_gt_np, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Transport Value")
    plt.title(f"SIFT-F Transport Matrix (Loss={loss_gt:.6f})")
    plt.xlabel("Image 2 Gaussians")
    plt.ylabel("Image 1 Gaussians")
    plt.tight_layout()
    plt.savefig(os.path.join(transport_dir, "transport_sift.png"), dpi=150)
    plt.close()
    
    np.save(os.path.join(transport_dir, "transport_sift.npy"), T_gt_np)

    losses_rand = []
    Rs = []
    ts = []
    Fs = []
    Ts = []  # 輸送行列を保存するリスト
    
    for i in range(args.n_random):
        # K1とK2を渡して、より適切なランダムF行列を生成
        F_r, R_r, t_r = random_F(K1, K2)
        l, T_r = compute_ot_loss(solver, F_r, return_transport=True)
        losses_rand.append(l)
        Rs.append(R_r)
        ts.append(t_r)
        Fs.append(F_r)
        Ts.append(T_r.cpu().numpy())  # 輸送行列を保存
        print(f"  random[{i:02d}]  loss={l:.6f}")

    # ---- ranking assertion -------------------------------------
    best_rand = min(losses_rand)
    best_rand_index = losses_rand.index(best_rand)
    if loss_gt < best_rand:
        print("\n✅  Cost function passes the sanity‑check "
              "(GT loss is the smallest).")
    else:
        print("\n❌  Cost function FAILED – some random F got lower loss!")
        print(f"SIFT-F loss: {loss_gt:.6f}")
        print(f"Best random F loss: {best_rand:.6f} (random[{best_rand_index:02d}])")
        print(f"Loss improvement: {(loss_gt - best_rand):.6f} ({((loss_gt - best_rand) / loss_gt * 100):.2f}%)")

        # SIFT-Fより良いR,tをすべて記録
        better_indices = [i for i, loss in enumerate(losses_rand) if loss < loss_gt]
        print(f"Found {len(better_indices)} solutions with lower loss than SIFT-F:")
        
        # 各解のロスを詳細表示（上位5つまで）
        better_losses = [(i, losses_rand[i]) for i in better_indices]
        better_losses.sort(key=lambda x: x[1])  # ロスの昇順にソート
        
        for idx, (i, loss) in enumerate(better_losses[:5]):  # 上位5つまで表示
            print(f"  Top {idx+1}: random[{i:02d}] loss={loss:.6f} (improvement: {(loss_gt - loss):.6f}, {((loss_gt - loss) / loss_gt * 100):.2f}%)")
            
            # 上位5つの解の輸送行列を保存
            plt.figure(figsize=(10, 8))
            plt.imshow(Ts[i], cmap="hot", interpolation="nearest")
            plt.colorbar(label="Transport Value")
            plt.title(f"Random[{i:02d}] Transport Matrix (Loss={loss:.6f})")
            plt.xlabel("Image 2 Gaussians")
            plt.ylabel("Image 1 Gaussians")
            plt.tight_layout()
            plt.savefig(os.path.join(transport_dir, f"transport_random_{i:02d}.png"), dpi=150)
            plt.close()
            
            # さらに、SIFT-Fとの差分も可視化
            diff = Ts[i] - T_gt_np
            plt.figure(figsize=(10, 8))
            plt.imshow(diff, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(label="Transport Difference")
            plt.title(f"Difference: Random[{i:02d}] - SIFT-F")
            plt.xlabel("Image 2 Gaussians")
            plt.ylabel("Image 1 Gaussians")
            plt.tight_layout()
            plt.savefig(os.path.join(transport_dir, f"transport_diff_{i:02d}.png"), dpi=150)
            plt.close()
        
        if len(better_losses) > 5:
            print(f"  ... and {len(better_losses) - 5} more solutions")
        
        # 輸送行列の解析：大きな違いがある部分をハイライト
        if len(better_indices) > 0:
            best_i = better_losses[0][0]
            best_transport = Ts[best_i]
            
            # 輸送値の大きな差がある上位N個の対応を見つける
            N = 20
            diff = best_transport - T_gt_np
            flat_indices = np.argsort(np.abs(diff).flatten())[-N:]
            
            rows, cols = np.unravel_index(flat_indices, diff.shape)
            print(f"\nTop {N} transport differences between best solution and SIFT-F:")
            print(f"{'Index':<10}{'SIFT-F':<15}{'Best Random':<15}{'Difference':<15}")
            print("-" * 55)
            
            for r, c in zip(rows, cols):
                sift_val = T_gt_np[r, c]
                rand_val = best_transport[r, c]
                diff_val = rand_val - sift_val
                print(f"({r},{c}):{sift_val:12.6f}  {rand_val:12.6f}  {diff_val:12.6f}")
            
            # 輸送の統計も保存
            stats = {
                "sift_transport_sum": float(T_gt_np.sum()),
                "sift_transport_mean": float(T_gt_np.mean()),
                "sift_transport_nonzero": int((T_gt_np > 1e-5).sum()),
                "best_transport_sum": float(best_transport.sum()),
                "best_transport_mean": float(best_transport.mean()),
                "best_transport_nonzero": int((best_transport > 1e-5).sum()),
            }
            
            import json
            with open(os.path.join(transport_dir, "transport_stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
            
            print("\nTransport statistics:")
            print(f"SIFT-F: sum={stats['sift_transport_sum']:.4f}, "
                  f"mean={stats['sift_transport_mean']:.6f}, "
                  f"nonzero={stats['sift_transport_nonzero']}")
            print(f"Best Random: sum={stats['best_transport_sum']:.4f}, "
                  f"mean={stats['best_transport_mean']:.6f}, "
                  f"nonzero={stats['best_transport_nonzero']}")
            
        # 可視化を行う
        if better_indices:
            from extrinsics_visualizer import CameraPoseVisualizer
            import matplotlib.pyplot as plt
            
            # 可視化範囲を定義
            xlim = [-5, 5]
            ylim = [-5, 5]
            zlim = [-1, 9]
            
            visualizer = CameraPoseVisualizer(xlim, ylim, zlim)
            
            # カメラ1（基準）の外部パラメータ（単位行列） - camera-to-world変換
            extrinsic1 = np.eye(4)
            
            # カメラ1を描画（緑色）
            visualizer.extrinsic2pyramid(extrinsic1, color_map='green', focal_len_scaled=2, aspect_ratio=0.3)
            
            # 各良解を描画（青色）
            for idx in better_indices:
                R = Rs[idx]
                t = ts[idx]
                
                # カメラ2のextrinsicマトリクスを計算
                # png.pyでは、R,tを使って[R|t]の形でcamera-to-worldの変換行列を作成
                # F = [t]× Rにおける、RとtからPの相対姿勢を計算
                # 基本行列は2番目のカメラから1番目のカメラへの射影を表す
                
                # 本来必要なのは基準カメラからのPの相対姿勢なので、
                # R,tをそのままcamera-to-worldの変換としては使えない
                
                # 必要なのは、カメラ2→カメラ1への変換R,tから
                # カメラ2のworld-to-camera変換を求めること
                
                # E = K2.T @ F @ K1における本質行列Eは
                # E = t× Rの形で、RはカメラB→カメラAの回転、tは同じく並進
                
                # camera-to-worldの変換行列を作成
                # R_c2wは第2カメラからワールドへの回転
                R_c2w = R.T  # カメラBからワールド（=カメラA）への回転
                
                # tはカメラB座標系でのカメラAの位置ベクトル（符号反転）
                # それをワールド座標系に変換
                C = -R.T @ t  # カメラ中心のワールド座標
                
                extrinsic2 = np.eye(4)
                extrinsic2[:3, :3] = R_c2w
                extrinsic2[:3, 3] = C
                
                # 青色でカメラ2を描画
                visualizer.extrinsic2pyramid(extrinsic2, color_map='blue', focal_len_scaled=2, aspect_ratio=0.3)
            
            # SIFT-Fの解も描画（赤色）
            try:
                # SIFT-FからE行列を計算
                E_sift = K2.T @ F_gt @ K1
                
                # SVD分解
                U, S, Vt = np.linalg.svd(E_sift)
                
                # 回転と並進を抽出
                W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
                R1 = U @ W @ Vt
                R2 = U @ W.T @ Vt
                t1 = U[:, 2]
                t2 = -U[:, 2]
                
                # 4つの解を作成（カメラB→カメラAの変換）
                possible_solutions = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
                
                # チェイラリティテストを実行して正しい解を選択
                # 簡単のため、少なくとも一点だけでテスト
                def triangulate_point(K1, K2, R, t, pt1, pt2):
                    # 射影行列を構築
                    P1 = K1 @ np.eye(3, 4)  # 最初のカメラはワールド座標の原点
                    P2 = K2 @ np.hstack([R, t.reshape(-1, 1)])
                    
                    # DLTアルゴリズムによる三角測量
                    A = np.zeros((4, 4))
                    A[0] = pt1[0] * P1[2] - P1[0]
                    A[1] = pt1[1] * P1[2] - P1[1]
                    A[2] = pt2[0] * P2[2] - P2[0]
                    A[3] = pt2[1] * P2[2] - P2[1]
                    
                    _, _, Vt = np.linalg.svd(A)
                    X = Vt[-1]
                    X = X / X[3]  # 同次座標から非同次座標へ
                    return X[:3]

                # 対応点を取得（簡易的に1点だけで検証）
                # 通常はSIFTのmatchesから取得すべき
                img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
                img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
                sift = cv2.SIFT_create()
                kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), None)
                kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), None)
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)
                good = []
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append(m)
                
                if len(good) > 0:
                    # 最初の対応点を使用
                    pt1 = np.array(kp1[good[0].queryIdx].pt)
                    pt2 = np.array(kp2[good[0].trainIdx].pt)
                    
                    # 正規化座標系に変換
                    pt1_norm = np.linalg.inv(K1) @ np.array([pt1[0], pt1[1], 1])
                    pt2_norm = np.linalg.inv(K2) @ np.array([pt2[0], pt2[1], 1])
                    
                    # 各解でチェイラリティテスト
                    valid_solution = None
                    for i, (R, t) in enumerate(possible_solutions):
                        # 3D点を再構成
                        X = triangulate_point(K1, K2, R, t, pt1, pt2)
                        
                        # カメラ座標系での3D点の位置
                        X1 = X  # カメラ1は原点
                        X2 = R @ X + t
                        
                        # 両方のカメラからの視点で3D点が前にあるか確認
                        if X1[2] > 0 and X2[2] > 0:
                            valid_solution = (R, t)
                            print(f"Valid solution found: {i}")
                            break
                    
                    if valid_solution:
                        R_sift, t_sift = valid_solution
                    else:
                        print("No valid solution found, using first solution as fallback")
                        R_sift, t_sift = possible_solutions[0]
                else:
                    print("Not enough matches for cheirality test, using first solution")
                    R_sift, t_sift = possible_solutions[0]

                # 必要に応じて符号を調整
                if np.linalg.det(R_sift) < 0:
                    R_sift = -R_sift
                
                # カメラ2のcamera-to-world変換行列を構築
                R_c2w_sift = R_sift.T
                C_sift = -R_sift.T @ t_sift
                
                extrinsic_sift = np.eye(4)
                extrinsic_sift[:3, :3] = R_c2w_sift
                extrinsic_sift[:3, 3] = C_sift
                
                # 赤色でSIFT解を描画
                visualizer.extrinsic2pyramid(extrinsic_sift, color_map='red', focal_len_scaled=2, aspect_ratio=0.3)
                print("Added SIFT-F camera pose (red)")
            except Exception as e:
                print(f"Could not visualize SIFT camera pose: {e}")
            
            # 可視化を表示
            plt.title('Camera Poses: SIFT(red), Better Random Solutions(blue), Reference(green)')
            plt.show()
            
    print(f"GT / best‑random ratio = {loss_gt/best_rand:.3f}")

    # ---- λ sensitivity -----------------------------------------
    for fac in [0.1,10]:
        solver.lambda_color   = args.lambda_color*fac
        solver.lambda_epipolar= args.lambda_epi   # fix epi
        l_fac = compute_ot_loss(solver,F_gt)
        rand_fac = [compute_ot_loss(solver,random_F(K1, K2)[0]) for _ in range(5)]
        msg = "OK" if l_fac < min(rand_fac) else "NG"
        print(f"λ_color×{fac:>4}:  loss={l_fac:.6f}  -> {msg}")

if __name__ == "__main__":
    main()
