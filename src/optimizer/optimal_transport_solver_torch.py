import copy
import os
import sys
import math
from typing import Optional, Tuple
import cv2

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.primitive.camera import Lie
from src.primitive.camera import Quaternion
from utils.optimizers.SAM import SAM

class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians with Homography Optimization."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        k1: Optional[np.ndarray] = None,
        k2: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
        lambda_mean: float = 1.0,
        lambda_cov: float = 0.3,
        lambda_color: float = 1.0,
        lambda_epipolar: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            k1 (Optional[np.ndarray]): Intrinsic parameters for camera 1 (3x3).
            k2 (Optional[np.ndarray]): Intrinsic parameters for camera 2 (3x3).
            epsilon (float): Entropy regularization parameter.
            lambda_mean (float): Weight for mean difference term.
            lambda_cov (float): Weight for covariance difference term.
            lambda_color (float): Weight for color difference term.
            device (torch.device): Device to perform computations on.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.k1 = (
            torch.tensor(k1, dtype=torch.float32, device=device)
            if k1 is not None
            else None
        )
        self.k2 = (
            torch.tensor(k2, dtype=torch.float32, device=device)
            if k2 is not None
            else None
        )
        self.h: Optional[torch.Tensor] = None
        # Added for Fundamental Matrix
        self.f: Optional[torch.Tensor] = None

        self.epsilon = epsilon
        self.lambda_mean = lambda_mean
        self.lambda_cov = lambda_cov
        self.lambda_color = lambda_color
        self.lambda_epipolar = lambda_epipolar

        # Convert Gaussian parameters to torch tensors
        self._prepare_gaussians()
        # Lieクラスのインスタンス化
        self.lie = Lie()
        self.quaternion = Quaternion()

    def _prepare_gaussians(self) -> None:
        """Prepare Gaussian parameters as torch tensors."""
        self.means1 = torch.as_tensor(
            self.gaussians1.means, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.scales1 = torch.as_tensor(
            self.gaussians1.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.rotations1 = torch.as_tensor(
            self.gaussians1.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)
        self.rgb1 = torch.as_tensor(
            self.gaussians1.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 3)
        self.alpha1 = torch.as_tensor(
            self.gaussians1.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)

        self.means2 = torch.as_tensor(
            self.gaussians2.means, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.scales2 = torch.as_tensor(
            self.gaussians2.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.rotations2 = torch.as_tensor(
            self.gaussians2.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)
        self.rgb2 = torch.as_tensor(
            self.gaussians2.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 3)
        self.alpha2 = torch.as_tensor(
            self.gaussians2.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)


    def _build_F_from_wc(self, R_wc: torch.Tensor, t_wc: torch.Tensor) -> torch.Tensor:
        """R_wc, t_wc から F を構築。Lieクラスのskew_symmetricを使用。"""
        tx = self.lie.skew_symmetric(t_wc)
        E = tx @ R_wc
        K1_inv = torch.inverse(self.k1)
        K2_inv = torch.inverse(self.k2)
        K2_inv_T = K2_inv.transpose(0,1)
        F = K2_inv_T @ E @ K1_inv
        assert not torch.isnan(F).any(), "NaN in fundamental matrix"
        return F

    def unbalanced_sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        rho: float = 0.1, 
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Implement the unbalanced optimal transport Sinkhorn algorithm.

        Args:
            cost_matrix (torch.Tensor): The cost matrix of shape (K1, K2).
            rho (float): Degree to which mass discrepancy is allowed (smaller means more freedom to create or remove mass).
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence threshold.

        Returns:
            torch.Tensor: The unbalanced OT transport plan of shape (K1, K2).
        """
        # alpha and beta represent the total mass (or approximated mass) of each Gaussian
        alpha = self.alpha1  # shape (K1,)
        beta = self.alpha2  # shape (K2,)
        print("cost min/max", cost_matrix.min(), cost_matrix.max())
        kernel = torch.exp(-cost_matrix / self.epsilon)  # shape (K1, K2)

        # Initialize u, v to 1
        u = torch.ones_like(alpha)  # (K1,)
        v = torch.ones_like(beta)  # (K2,)

        # if exponent == 1, it's the balanced Sinkhorn algorithm
        exponent = rho / (rho + self.epsilon)
        # print(f"rho: {rho}, epsilon: {self.epsilon}, exponent: {exponent}")

        stabilization_const = 1e-16
        for iteration in range(max_iter):
            kv = kernel @ v
            kv = kv + stabilization_const
            u_new = (alpha / kv).pow(exponent)

            ktu = kernel.t() @ u_new
            ktu = ktu + stabilization_const
            v_new = (beta / ktu).pow(exponent)

            # Check convergence
            if (
                torch.max(torch.abs(u_new - u)) < tol
                and torch.max(torch.abs(v_new - v)) < tol
            ):
                # print(f"[Unbalanced] iteration {iteration} -> converged.")
                break

            u, v = u_new, v_new

        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return transport

    def _make_cov_matrices(self, scales: torch.Tensor,
                        rotations: torch.Tensor) -> torch.Tensor:
        """"""
        cos_r, sin_r = torch.cos(rotations), torch.sin(rotations)
        rot = torch.stack(
            [torch.stack([cos_r, -sin_r], 1),
            torch.stack([sin_r,  cos_r], 1)],
            2)                                   # (K,2,2)
        scale_mat = torch.stack([torch.diag(s**2) for s in scales])
        return rot @ scale_mat @ rot.transpose(1, 2)   # (K,2,2)
    

    def compute_cost_matrix_fundamental(self, F: torch.Tensor) -> torch.Tensor:
        """
        期待二乗距離版  "対称エピポーラ距離" + 色差
        ------------------------------------------------
        L_ij = (d_12^2 + u1_i) + (d_21^2 + u2_j)
        ------------------------------------------------
        """

        k1, k2 = self.means1.size(0), self.means2.size(0)

        # -------- ① 同次座標 --------
        ones1 = torch.ones(k1, 1, device=self.device)
        ones2 = torch.ones(k2, 1, device=self.device)
        p1_h = torch.cat([self.means1, ones1], 1)   # (K1,3)
        p2_h = torch.cat([self.means2, ones2], 1)   # (K2,3)

        # -------- ② エピポーラ線 --------
        l1 = (F   @ p2_h.T).T            # image-1 上 (K2,3)
        l2 = (F.T @ p1_h.T).T            # image-2 上 (K1,3)

        n1 = l1[:, :2]                              # (K2,2)
        n1_norm = n1.norm(dim=1, keepdim=True) + 1e-12
        n1_unit = n1 / n1_norm

        n2 = l2[:, :2]                              # (K1,2)
        n2_norm = n2.norm(dim=1, keepdim=True) + 1e-12
        n2_unit = n2 / n2_norm

        # -------- ③ 点⇔線  "符号付き" 距離 --------
        #     d_12(i,j):  p1_i → ℓ1_j
        #     d_21(i,j):  p2_j → ℓ2_i
        # ※ abs を外して「符号付き」にしても結果は d^2 なので同じですが、そのまま再利用します
        dist_12 = torch.abs(p1_h @ l1.T) / n1_norm.T      # (K1,K2)
        dist_21 = torch.abs(p2_h @ l2.T).T / n2_norm      # (K1,K2)

        # -------- ④ "距離²＋分散" へ置換 --------
        # CHANGE 1:  割り算 → 2乗して足し算
        dist_sq_sum = dist_12.pow(2) + dist_21.pow(2)     # d_12² + d_21²  (K1,K2)

        #   共分散由来の分散 u = nᵀ Σ n         （計算方法はそのまま再利用）
        cov1 = self._make_cov_matrices(self.scales1, self.rotations1)   # (K1,2,2)
        cov2 = self._make_cov_matrices(self.scales2, self.rotations2)   # (K2,2,2)

        v1 = torch.bmm(cov1, n2_unit.unsqueeze(-1)).squeeze(-1)         # (K1,2)
        u1 = (v1 * n2_unit).sum(1)                                      # (K1,)

        v2 = torch.bmm(cov2, n1_unit.unsqueeze(-1)).squeeze(-1)         # (K2,2)
        u2 = (v2 * n1_unit).sum(1)                                      # (K2,)

        # CHANGE 2: 以前は `/ (1+u1+u2)`
        epi_with_shape = dist_sq_sum + u1.view(-1,1) + u2.view(1,-1)    # (K1,K2)

        # -------- ⑤ 色差（ --------
        rgb_diff   = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)    # (K1,K2,3)
        color_dist = (rgb_diff ** 2).sum(2)                             # (K1,K2)

        # -------- ⑥ 正規化 --------
        with torch.no_grad():
            p95_epi   = torch.quantile(epi_with_shape, 0.95)
            p95_color = torch.quantile(color_dist,    0.95)

        epi_norm   = torch.clamp(epi_with_shape, max=p95_epi) / p95_epi
        color_norm = torch.clamp(color_dist,    max=p95_color) / p95_color

        # -------- ⑦ コスト合成 --------
        cost = ( self.lambda_epipolar * epi_norm
            + self.lambda_color    * color_norm )
        
        cost = cost / cost.max().detach()  

        return cost

        
    def compute_cost_matrix_fundamental_sampson(self, F: torch.Tensor) -> torch.Tensor:
        """ガウス分布の形状を考慮した分布版サンプソン距離によるコスト行列計算
        
        サンプソン距離の優れた特性を活かしつつ、ガウス分布の形状情報も
        考慮することで、特に大きなガウスや楕円形ガウスの対応付けを改善する。
        完全にベクトル化された実装により、大量のガウスでも高速に計算可能。
        
        Args:
            F (torch.Tensor): 基礎行列 (3x3)
            
        Returns:
            torch.Tensor: コスト行列 (K1, K2)
        """
        # 画像1,2のガウス数を取得
        k1, k2 = self.means1.size(0), self.means2.size(0)
        
        # === 1. サンプソン距離の基本形 ===
        # 同次座標変換
        ones1 = torch.ones(k1, 1, device=self.device)
        ones2 = torch.ones(k2, 1, device=self.device)
        p1 = torch.cat([self.means1, ones1], 1)  # (K1,3)
        p2 = torch.cat([self.means2, ones2], 1)  # (K2,3)
        
        # サンプソン距離の主要成分計算
        Fp1 = F @ p1.T    # 画像2上のエピポーラ線 (3,K1)
        FTp2 = F.T @ p2.T  # 画像1上のエピポーラ線 (3,K2)
        
        # エピポーラ制約の値: (p2^T·F·p1)^2
        num = (p2 @ Fp1)      # (K2,K1)
        num = num.T ** 2      # (K1,K2)
        
        # サンプソン距離の分母: (F·p1)_xy^2 + (F^T·p2)_xy^2
        denom = (Fp1[:2]**2).sum(0).view(-1, 1) + (FTp2[:2]**2).sum(0).view(1, -1)
        
        # サンプソン距離計算
        sampson = num / (denom + 1e-12)  # (K1,K2)
        
        # === 2. ガウス分布の形状を考慮 ===
        # 2.1 共分散行列の作成
        # 共分散行列の計算
        cov1 = self._make_cov_matrices(self.scales1, self.rotations1)  # (K1,2,2)
        cov2 = self._make_cov_matrices(self.scales2, self.rotations2)  # (K2,2,2)
        
        # 2.2 エピポーラ線の法線ベクトル計算（正規化）
        # ℓ = (a,b,c) の法線は n = (a,b)/||(a,b)||
        n1 = Fp1[:2].T  # 画像2上のエピポーラ線の法線方向 (K1,2)
        n1 = n1 / (n1.norm(dim=1, keepdim=True) + 1e-12)
        
        n2 = FTp2[:2].T  # 画像1上のエピポーラ線の法線方向 (K2,2)
        n2 = n2 / (n2.norm(dim=1, keepdim=True) + 1e-12)
        
        # 2.3 共分散行列とエピポーラ線法線の積 (n^T·Σ·n)
        def shape_uncertainty(covs, normals):
            """共分散行列と法線ベクトルからエピポーラ線方向の不確かさを計算"""
            # covs: (K,2,2), normals: (K,2) -> returns: (K,)
            v = torch.bmm(covs, normals.unsqueeze(-1)).squeeze(-1)  # (K,2)
            return (v * normals).sum(dim=1)  # (K,) n^T·Σ·n を各ガウスごとに計算
        
        # 各ガウスのエピポーラ線方向への不確かさ
        u1 = shape_uncertainty(cov1, n2)  # 画像1のガウスの不確かさ (K1,)
        u2 = shape_uncertainty(cov2, n1)  # 画像2のガウスの不確かさ (K2,)
        
        # 2.4 サンプソン距離を不確かさで割引（形状を考慮した分布版サンプソン距離）
        # 不確かさが大きいほど（エピポーラ線に垂直な方向に広いガウスほど）、
        # サンプソン距離を小さく評価する
        uncertainty_factor = 1.0 + u1.view(-1, 1) + u2.view(1, -1)  # ブロードキャスト (K1,K2)
        sampson_with_shape = sampson / uncertainty_factor
        
        # === 3. 色差分の計算 ===
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = (color_diff ** 2).sum(dim=2)  # (K1,K2)
        
        # === 4. 正規化と最終コスト計算 ===
        # 4.1 95パーセンタイルでの正規化（外れ値の影響を抑制）
        with torch.no_grad():
            p95_sampson = torch.quantile(sampson_with_shape, 0.95)
            p95_color = torch.quantile(d_color, 0.95)
        
        # 4.2 正規化と重み付け
        sampson_norm = torch.clamp(sampson_with_shape, max=p95_sampson) / p95_sampson
        color_norm = torch.clamp(d_color, max=p95_color) / p95_color
        
        # 4.3 最終コスト行列の計算
        cost = (
            self.lambda_epipolar * sampson_norm + 
            self.lambda_color * color_norm
        )
        
        return cost

    def compute_cost_matrix_fundamental_original(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using a Fundamental Matrix.

        This replaces the Homography-based distance with an epipolar distance.
        Also includes color difference and covariance difference terms.

        Args:
            f (torch.Tensor): The Fundamental matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
        # 1) 画像1,2 それぞれのGaussians数
        k1 = self.means1.shape[0]
        k2 = self.means2.shape[0]

        # 2) 平均点を同次座標化 (x,y,1)
        ones1 = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
        p1_homo = torch.cat([self.means1, ones1], dim=1)  # (K1,3)

        ones2 = torch.ones((k2, 1), dtype=torch.float32, device=self.device)
        p2_homo = torch.cat([self.means2, ones2], dim=1)  # (K2,3)

        # 3) p2_j に対応するエピポーラ線を画像1上で計算: l1[j] = F * p2[j]
        #    p1_i に対応するエピポーラ線を画像2上で計算: l2[i] = F^T * p1[i]
        #    (shape: l1 -> (K2,3), l2 -> (K1,3))
        l1 = (f @ p2_homo.T).T  # (K2,3)
        l2 = (f.t() @ p1_homo.T).T  # (K1,3)

        # 4) p1[i] と l1[j] の距離をペアごとに計算
        #    l1[j] = (a_j, b_j, c_j), p1[i] = (x_i, y_i, 1)
        #    dist_12(i,j) = | p1[i]·l1[j] | / sqrt(a_j^2 + b_j^2)
        #
        #   - numerator_12(i,j) = | p1[i]·l1[j] |
        #   - denominator_12(j) = sqrt(a_j^2 + b_j^2)
        #   → shape はそれぞれ (K1,K2), (K2,) になり
        #     dist_12 = numerator_12 / denom_12(ブロードキャスト)
        numerator_12 = torch.abs(p1_homo @ l1.T)    # => (K1, K2)
        denom_12 = torch.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)  # (K2,)
        denom_12 = denom_12.view(1, -1)  # (1,K2) for broadcast
        dist_12 = numerator_12 / denom_12  # (K1,K2)

        # 5) p2[j] と l2[i] の距離をペアごとに計算
        #    l2[i] = (a_i, b_i, c_i), p2[j] = (x_j, y_j, 1)
        #    dist_21(i,j) = | p2[j]·l2[i] | / sqrt(a_i^2 + b_i^2)
        #
        #   - numerator_21(i,j) = | p2[j]·l2[i] |
        #   - denominator_21(i)  = sqrt(a_i^2 + b_i^2)
        #   → shape はそれぞれ (K2,K1) (K1,) となるのを転置して (K1,K2) など
        numerator_21 = torch.abs(p2_homo @ l2.T)    # => (K2, K1)
        numerator_21 = numerator_21.T              # => (K1, K2)
        denom_21 = torch.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)  # (K1,)
        denom_21 = denom_21.view(-1, 1)            # (K1,1)
        dist_21 = numerator_21 / denom_21          # (K1,K2)

        # 6) 対称エピポーラ距離を合計
        #    epipolar_dist(i,j) = dist_12(i,j) + dist_21(i,j)
        epipolar_dist = dist_12 + dist_21

        # 7) カラー差分 (RGB) を計算
        #    color_diff(i,j) = sum_k ( rgb1[i][k] - rgb2[j][k] )^2
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = torch.sum(color_diff ** 2, dim=2)  # (K1,K2)
        
        # 8) 共分散制約の計算
        #    R,tから予測される共分散の変換を計算し、実際の共分散との差を求める
        
        # 8.2) 共分散差分を初期化（すべてのペアに対して）
        cov_diff = torch.zeros((k1, k2), device=self.device)
        
        # 8.3) 各ガウシアンペアの共分散差分を計算
        for i in range(k1):
            # 画像1のガウシアンの共分散行列を取得
            scale1 = self.scales1[i]
            rotation1 = self.rotations1[i]
            
            # 回転行列の計算
            cos_r = torch.cos(rotation1)
            sin_r = torch.sin(rotation1)
            R_2d = torch.tensor([
                [cos_r, -sin_r],
                [sin_r, cos_r]
            ], device=self.device)
            
            # スケール行列
            S_diag = torch.diag(scale1.pow(2))
            
            # 共分散行列: R * S * R^T
            cov1 = R_2d @ S_diag @ R_2d.t()
            
            # エピポーラ線ベクトルを計算（F * x1）
            p1 = p1_homo[i]  # (3,)
            epipolar_lines = f @ p1  # (3,)：画像2上のエピポーラ線
            
            # エピポーラ線の方向ベクトル [normalized(-b, a)]
            line_dir = torch.tensor([-epipolar_lines[1], epipolar_lines[0]], 
                                device=self.device)
            line_dir = line_dir / (torch.norm(line_dir) + 1e-10)  # 正規化
            
            # R,tに基づく共分散の変換（エピポーラ線方向に伸長）
            # 奥行きの不確かさがエピポーラ線方向の不確かさとして現れる
            scale_factor = 2.0  # エピポーラ線方向の伸長係数（調整可能）
            line_outer = torch.outer(line_dir, line_dir)
            transform = torch.eye(2, device=self.device) + (scale_factor - 1.0) * line_outer
            
            # 変換された共分散行列
            transformed_cov1 = transform @ cov1 @ transform.t()
            
            for j in range(k2):
                # 画像2のガウシアンの共分散行列を取得
                scale2 = self.scales2[j]
                rotation2 = self.rotations2[j]
                
                cos_r2 = torch.cos(rotation2)
                sin_r2 = torch.sin(rotation2)
                R_2d2 = torch.tensor([
                    [cos_r2, -sin_r2],
                    [sin_r2, cos_r2]
                ], device=self.device)
                
                S_diag2 = torch.diag(scale2.pow(2))
                cov2 = R_2d2 @ S_diag2 @ R_2d2.t()
                
                # 共分散の差をフロベニウスノルムで計算
                diff = torch.norm(transformed_cov1 - cov2, 'fro')
                cov_diff[i, j] = diff
        
        # 9) 各コスト要素の正規化
        with torch.no_grad():
            p95_epipolar = torch.quantile(epipolar_dist, 0.95)
            p95_color = torch.quantile(d_color, 0.95)
            p95_cov = torch.quantile(cov_diff, 0.95)

        epipolar_dist = torch.clamp(epipolar_dist, max=p95_epipolar) / p95_epipolar
        d_color = torch.clamp(d_color, max=p95_color) / p95_color
        cov_diff = torch.clamp(cov_diff, max=p95_cov) / p95_cov

        cost_matrix = (
            self.lambda_epipolar * epipolar_dist + 
            self.lambda_color * d_color + 
            self.lambda_cov * cov_diff
        )

        return cost_matrix



    def _init_se3_like_cam1(self, rot_noise=0.05, trans_noise=0.05, seed=42):
        """回転と並進を別々のパラメータとして初期化"""
        # シード設定（再現性のため）
        if seed is not None:
            torch.manual_seed(seed)
        
        # 回転初期化（微小なランダム軸角）
        axis = torch.randn(3, device=self.device)
        axis /= axis.norm() + 1e-8
        omega = axis * rot_noise * torch.randn(1, device=self.device)
        
        # 並進初期化
        rho = trans_noise * torch.randn(3, device=self.device)
        
        # 別々のパラメータとして定義
        self.rot_vec = nn.Parameter(omega.clone())
        self.trans_vec = nn.Parameter(rho.clone())

    def optimize_with_SE3(self, max_iter=1000, tol=1e-6,
                        save_diagnostics=True, diagnostics_dir=None,
                        rot_lr=5e-3, trans_lr=5e-4, momentum=0.9,
                        grad_clip=0.1, seed=None):
        """Optimize camera pose using Lie algebra SE(3) representation with separate rotation/translation."""
        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_SE3")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_se3")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        param_history = {'rot_vec': [], 'trans_vec': []}
        grad_history = {'rot_vec': [], 'trans_vec': []}

        # ------------------------- パラメータ初期化 ----------------------- #
        if not hasattr(self, "rot_vec") or not hasattr(self, "trans_vec"):
            self._init_se3_like_cam1(rot_noise=0.05, trans_noise=0.05, seed=seed)

        # ------------------------- オプティマイザ設定 ----------------------- #
        optimizer = torch.optim.SGD([
            {'params': self.rot_vec, 'lr': rot_lr, 'momentum': momentum, 'nesterov': True},
            {'params': self.trans_vec, 'lr': trans_lr, 'momentum': momentum, 'nesterov': True}
        ])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))

        prev_loss_val = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing SE(3)", leave=True)

        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration, Loss, Rot_Grad_Norm, Trans_Grad_Norm, Rot_x, Rot_y, Rot_z, Trans_x, Trans_y, Trans_z\n")

        for iteration in pbar:
            optimizer.zero_grad()
            
            # --- SE(3)指数写像（分離したパラメータを結合して渡す） ---
            se3_vec = torch.cat([self.rot_vec, self.trans_vec])
            T_cw = self.lie.se3_to_SE3(se3_vec)  # (3,4) or (4,4)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]

            # 世界→カメラ変換
            R_wc = R_cw.t()
            t_wc = -R_cw.t() @ t_cw

            # 基礎行列と損失の計算
            F = self._build_F_from_wc(R_wc, t_wc)
            cost_matrix = self.compute_cost_matrix_fundamental(F)
            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix)
            loss = torch.sum(transport * cost_matrix)
            
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  Rot params: {self.rot_vec.data}")
                print(f"  Trans params: {self.trans_vec.data}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Rot LR: {rot_lr:.6e}, Trans LR: {trans_lr:.6e}")
            
            loss.backward()
            
            # 勾配クリッピング
            # torch.nn.utils.clip_grad_norm_([self.rot_vec, self.trans_vec], max_norm=grad_clip)
            
            if self.rot_vec.grad is not None and self.trans_vec.grad is not None:
                rot_grad = self.rot_vec.grad
                trans_grad = self.trans_vec.grad
                rot_grad_norm = rot_grad.norm().item()
                trans_grad_norm = trans_grad.norm().item()
                
                with open(debug_log_path, 'a') as f:
                    rot_vals = rot_grad.detach().cpu().numpy()
                    trans_vals = trans_grad.detach().cpu().numpy()
                    f.write(f"{iteration}, {loss.item():.6f}, {rot_grad_norm:.6f}, {trans_grad_norm:.6f}, " + 
                        f"{rot_vals[0]:.6f}, {rot_vals[1]:.6f}, {rot_vals[2]:.6f}, " +
                        f"{trans_vals[0]:.6f}, {trans_vals[1]:.6f}, {trans_vals[2]:.6f}\n")
                
                if iteration % 10 == 0:
                    print(f"  Rot gradient norm: {rot_grad_norm:.6f}")
                    print(f"  Trans gradient norm: {trans_grad_norm:.6f}")
                    print(f"  Rot/Trans gradient norm ratio: {rot_grad_norm/max(trans_grad_norm, 1e-10):.6f}")
            else:
                print("Warning: No gradient computed!")

            current_loss = loss.item()
            loss_history.append(current_loss)
            param_history['rot_vec'].append(self.rot_vec.clone())
            param_history['trans_vec'].append(self.trans_vec.clone())
            grad_history['rot_vec'].append(self.rot_vec.grad.clone() if self.rot_vec.grad is not None else None)
            grad_history['trans_vec'].append(self.trans_vec.grad.clone() if self.trans_vec.grad is not None else None)

            optimizer.step()
            # t の正規化
            with torch.no_grad():
                norm_t = self.trans_vec.data.norm() + 1e-9   # 0 除け
                self.trans_vec.data.div_(norm_t)
            
            # リトラクション: 回転ベクトルを基本領域（|θ| ≤ π）に投影
            with torch.no_grad():
                theta = self.rot_vec.data.norm()
                if theta > math.pi:
                    self.rot_vec.data.mul_(math.pi / theta)
            
            scheduler.step()
            
            if iteration % 10 == 0:
                rot_change = torch.norm(self.rot_vec.data - param_history['rot_vec'][-2].data) if iteration > 0 else torch.tensor(0)
                trans_change = torch.norm(self.trans_vec.data - param_history['trans_vec'][-2].data) if iteration > 0 else torch.tensor(0)
                print(f"  Rot parameter change: {rot_change.item():.6f}")
                print(f"  Trans parameter change: {trans_change.item():.6f}")
                print(f"  Updated Rot params: {self.rot_vec.data}")
                print(f"  Updated Trans params: {self.trans_vec.data}")

            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss

            if iteration % 10 == 0:
                rot_grad_norm = self.rot_vec.grad.norm().item() if self.rot_vec.grad is not None else 0
                trans_grad_norm = self.trans_vec.grad.norm().item() if self.trans_vec.grad is not None else 0
                ratio = rot_grad_norm/max(trans_grad_norm, 1e-10) 
                pbar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'r_grad': f"{rot_grad_norm:.4f}",
                    'r_t': f"{ratio:.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            # トランスポートプラン可視化（元のコードと同じ）
            if iteration % 10 == 0 or iteration == max_iter - 1:
                with torch.no_grad():
                    t_np = transport.detach().cpu().numpy()
                rows, cols = t_np.shape
                aspect_ratio = cols / rows
                if rows > cols:
                    fig_width = 8
                    fig_height = min(20, fig_width / aspect_ratio)
                else:
                    fig_height = 6
                    fig_width = min(20, fig_height * aspect_ratio)
                plt.figure(figsize=(fig_width, fig_height))
                if rows > 1000 or cols > 1000:
                    downsample_factor = max(1, int(max(rows, cols) / 1000))
                    t_np_display = t_np[::downsample_factor, ::downsample_factor]
                    plt.imshow(t_np_display, cmap="hot", interpolation="nearest", aspect="auto")
                    plt.title(f"Transport Plan at Iteration {iteration} (Downsampled {downsample_factor}x)")
                else:
                    plt.imshow(t_np, cmap="hot", interpolation="nearest", aspect="auto")
                    plt.title(f"Transport Plan at Iteration {iteration}")
                plt.colorbar(label="Transport Plan Value")
                plt.xlabel("Image 2 Gaussians")
                plt.ylabel("Image 1 Gaussians")
                plt.tight_layout()
                plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
                plt.savefig(plt_path, dpi=150)
                plt.close()

        # ------------------------- 最終パラメータ保存 --------------------------- #
        with torch.no_grad():
            # 最終SE(3)パラメータから変換結果を保存
            se3_vec = torch.cat([self.rot_vec, self.trans_vec])
            T_cw = self.lie.se3_to_SE3(se3_vec)
            self.R_cw = T_cw[:3, :3]
            self.t_cw = T_cw[:3, 3]
            self.R_wc = self.R_cw.t()
            self.t_wc = -self.R_wc @ self.t_cw
            final_F = self._build_F_from_wc(self.R_wc, self.t_wc)
            self.f = final_F
            if cv2 is not None:
                rvec_numpy, _ = cv2.Rodrigues(self.R_wc.cpu().numpy())
                self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
                self.tvec = nn.Parameter(self.t_cw)
                rvec_cw_numpy, _ = cv2.Rodrigues(self.R_cw.cpu().numpy())
                self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
                self.center = nn.Parameter(self.t_cw)

        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_SE3)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_SE3.png"))
        plt.close()

        if save_diagnostics:
            self.save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )

    def save_optimization_diagnostics_SE3(self, 
                                output_dir: str,
                                loss_history: list,
                                param_history: dict,
                                grad_history: dict) -> None:
        """分離したSE(3)パラメータの最適化過程に関する詳細な診断情報を保存する
        
        回転と並進を分離して最適化したSE(3)パラメータについて、以下の診断情報を生成・保存します：
        - 損失軌跡の分析
        - パラメータ進化の分析（回転と並進の各成分）
        - 勾配挙動の分析
        - 収束性分析
        - 3D軌跡可視化
        - テキスト形式のサマリーレポート
        
        Args:
            output_dir: 診断ファイルを保存するディレクトリ
            loss_history: イテレーションごとの損失値リスト
            param_history: パラメータ履歴の辞書（'rot_vec'と'trans_vec'を含む）
            grad_history: 勾配履歴の辞書（'rot_vec'と'trans_vec'を含む）
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 履歴をNumPy配列に変換
        param_history_np = {}
        grad_history_np = {}
        
        for param_name, history in param_history.items():
            param_history_np[param_name] = np.array([p.detach().cpu().numpy() for p in history])
            
        for param_name, history in grad_history.items():
            grad_history_np[param_name] = np.array([g.detach().cpu().numpy() if g is not None 
                                                else np.zeros_like(param_history_np[param_name][0]) 
                                                for g in history])
        
        # イテレーション数
        iterations = range(len(loss_history))
        
        # ======================= 1. 損失軌跡の分析 =======================
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 損失の変化（微分）をプロット
        plt.subplot(212)
        loss_changes = np.array([loss_history[i+1] - loss_history[i] 
                                for i in range(len(loss_history)-1)])
        plt.plot(iterations[:-1], loss_changes, 'r-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Loss Change Between Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Difference')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_analysis.png'), dpi=150)
        plt.close()
        
        # =============== 2. SE(3)パラメータ軌跡の分析 ===============
        rot_data = param_history_np['rot_vec']
        trans_data = param_history_np['trans_vec']
        
        # 全6成分をプロット
        fig = plt.figure(figsize=(15, 8))
        
        # 回転成分
        plt.subplot(211)
        plt.plot(iterations, rot_data[:, 0], 'r-', label='wx')
        plt.plot(iterations, rot_data[:, 1], 'g-', label='wy')
        plt.plot(iterations, rot_data[:, 2], 'b-', label='wz')
        plt.title('SE(3) Rotation Components (w) Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # 並進成分
        plt.subplot(212)
        plt.plot(iterations, trans_data[:, 0], 'r-', label='tx')
        plt.plot(iterations, trans_data[:, 1], 'g-', label='ty')
        plt.plot(iterations, trans_data[:, 2], 'b-', label='tz')
        plt.title('SE(3) Translation Components (t) Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_components_trajectory.png'), dpi=150)
        plt.close()
        
        # =============== 3. 3D軌跡可視化 ===============
        fig = plt.figure(figsize=(15, 7))
        
        # 回転成分の3D軌跡
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(rot_data[:, 0], rot_data[:, 1], rot_data[:, 2], 'r-', linewidth=2)
        ax1.scatter(rot_data[0, 0], rot_data[0, 1], rot_data[0, 2], c='g', s=100, label='Initial')
        ax1.scatter(rot_data[-1, 0], rot_data[-1, 1], rot_data[-1, 2], c='b', s=100, label='Final')
        ax1.set_title('Rotation Components (w) Trajectory in 3D')
        ax1.set_xlabel('wx')
        ax1.set_ylabel('wy')
        ax1.set_zlabel('wz')
        ax1.legend()
        
        # 並進成分の3D軌跡
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(trans_data[:, 0], trans_data[:, 1], trans_data[:, 2], 'r-', linewidth=2)
        ax2.scatter(trans_data[0, 0], trans_data[0, 1], trans_data[0, 2], c='g', s=100, label='Initial')
        ax2.scatter(trans_data[-1, 0], trans_data[-1, 1], trans_data[-1, 2], c='b', s=100, label='Final')
        ax2.set_title('Translation Components (t) Trajectory in 3D')
        ax2.set_xlabel('tx')
        ax2.set_ylabel('ty')
        ax2.set_zlabel('tz')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_3d_trajectory.png'), dpi=150)
        plt.close()
        
        # =============== 4. 勾配分析 ===============
        rot_grad_data = grad_history_np['rot_vec']
        trans_grad_data = grad_history_np['trans_vec']
        
        # 勾配の大きさ（ノルム）
        rot_grad_magnitude = np.linalg.norm(rot_grad_data, axis=1)
        trans_grad_magnitude = np.linalg.norm(trans_grad_data, axis=1)
        total_grad_magnitude = np.sqrt(rot_grad_magnitude**2 + trans_grad_magnitude**2)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # 総合勾配の大きさプロット
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, total_grad_magnitude, 'k-', linewidth=2, label='Total')
        ax1.plot(iterations, rot_grad_magnitude, 'r-', linewidth=1.5, label='Rotation')
        ax1.plot(iterations, trans_grad_magnitude, 'b-', linewidth=1.5, label='Translation')
        ax1.set_title('SE(3) Gradient Magnitude')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')  # Log scale to better see changes
        ax1.grid(True)
        ax1.legend()
        
        # 回転勾配成分プロット
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, rot_grad_data[:, 0], 'r-', label='grad_wx')
        ax2.plot(iterations, rot_grad_data[:, 1], 'g-', label='grad_wy')
        ax2.plot(iterations, rot_grad_data[:, 2], 'b-', label='grad_wz')
        ax2.set_title('Rotation Gradient Components')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # 並進勾配成分プロット
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, trans_grad_data[:, 0], 'r-', label='grad_tx')
        ax3.plot(iterations, trans_grad_data[:, 1], 'g-', label='grad_ty')
        ax3.plot(iterations, trans_grad_data[:, 2], 'b-', label='grad_tz')
        ax3.set_title('Translation Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_gradient_analysis.png'), dpi=150)
        plt.close()
        
        # =============== 5. 回転/並進勾配比率分析 ===============
        plt.figure(figsize=(12, 6))
        # ゼロ除算防止のためのイプシロン
        ratio = rot_grad_magnitude / (trans_grad_magnitude + 1e-10)
        plt.plot(iterations, ratio, 'b-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Balanced ratio (1.0)')
        plt.title('Ratio of Rotation to Translation Gradient Norms')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.yscale('log')  # Log scaleで変化を見やすく
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_ratio.png'), dpi=150)
        plt.close()
        
        # =============== 6. パラメータ変化量の分析 ===============
        plt.figure(figsize=(12, 6))
        # 各イテレーションでのパラメータの変化量
        rot_changes = np.array([np.linalg.norm(rot_data[i+1] - rot_data[i]) for i in range(len(rot_data)-1)])
        trans_changes = np.array([np.linalg.norm(trans_data[i+1] - trans_data[i]) for i in range(len(trans_data)-1)])
        
        plt.plot(iterations[:-1], rot_changes, 'r-', label='Rotation Change')
        plt.plot(iterations[:-1], trans_changes, 'b-', label='Translation Change')
        plt.title('Parameter Change Magnitude per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Change Magnitude')
        plt.yscale('log')  # Log scaleで変化を見やすく
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_changes.png'), dpi=150)
        plt.close()
        
        # =============== 7. テキスト形式のサマリーレポート ===============
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("SE(3) OPTIMIZATION PROCESS ANALYSIS (SEPARATED PARAMETERS)\n")
            f.write("=====================================================\n\n")
            
            # 損失分析
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # 単調減少性チェック
            is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
            f.write(f"Loss decreases monotonically: {is_monotonic}\n")
            
            # 振動とプラトー（平坦部）の検出
            oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                    if (loss_history[i] > loss_history[i+1] and 
                                        loss_history[i+1] < loss_history[i+2]))
            
            plateau_threshold = 1e-6  # プラトー判定の閾値
            plateau_count = sum(1 for i in range(len(loss_history)-1) 
                            if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
            
            f.write(f"Number of oscillations: {oscillation_count}\n")
            f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # パラメータ分析
            f.write("2. SE(3) PARAMETER BEHAVIOR\n")
            f.write("-------------------------\n")
            
            # 回転成分
            f.write(f"Rotation component (w) initial: " + np.array2string(rot_data[0], precision=6) + "\n")
            f.write(f"Rotation component (w) final: " + np.array2string(rot_data[-1], precision=6) + "\n")
            rot_change = np.linalg.norm(rot_data[-1] - rot_data[0])
            f.write(f"Total rotation change magnitude: {rot_change:.6f}\n\n")
            
            # 並進成分
            f.write(f"Translation component (t) initial: " + np.array2string(trans_data[0], precision=6) + "\n")
            f.write(f"Translation component (t) final: " + np.array2string(trans_data[-1], precision=6) + "\n")
            trans_change = np.linalg.norm(trans_data[-1] - trans_data[0])
            f.write(f"Total translation change magnitude: {trans_change:.6f}\n\n")
            
            # 勾配分析
            f.write("3. GRADIENT BEHAVIOR\n")
            f.write("-------------------\n")
            
            # 回転勾配の統計
            max_rot_grad = np.max(rot_grad_magnitude)
            min_rot_grad = np.min(rot_grad_magnitude)
            avg_rot_grad = np.mean(rot_grad_magnitude)
            f.write(f"Rotation gradient - Max: {max_rot_grad:.6f}, " 
                    f"Min: {min_rot_grad:.6f}, Avg: {avg_rot_grad:.6f}\n")
            
            # 並進勾配の統計
            max_trans_grad = np.max(trans_grad_magnitude)
            min_trans_grad = np.min(trans_grad_magnitude)
            avg_trans_grad = np.mean(trans_grad_magnitude)
            f.write(f"Translation gradient - Max: {max_trans_grad:.6f}, "
                    f"Min: {min_trans_grad:.6f}, Avg: {avg_trans_grad:.6f}\n")
            
            # 総合勾配の統計
            max_grad = np.max(total_grad_magnitude)
            min_grad = np.min(total_grad_magnitude)
            avg_grad = np.mean(total_grad_magnitude)
            f.write(f"Total gradient - Max: {max_grad:.6f}, Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n\n")
            
            # 勾配消失/爆発チェック
            vanishing_threshold = 1e-6
            exploding_threshold = 1e2
            
            vanishing_rot_grad = any(grad < vanishing_threshold for grad in rot_grad_magnitude)
            exploding_rot_grad = any(grad > exploding_threshold for grad in rot_grad_magnitude)
            
            vanishing_trans_grad = any(grad < vanishing_threshold for grad in trans_grad_magnitude)
            exploding_trans_grad = any(grad > exploding_threshold for grad in trans_grad_magnitude)
            
            f.write(f"Rotation gradient vanishing detected: {vanishing_rot_grad}\n")
            f.write(f"Rotation gradient exploding detected: {exploding_rot_grad}\n")
            f.write(f"Translation gradient vanishing detected: {vanishing_trans_grad}\n")
            f.write(f"Translation gradient exploding detected: {exploding_trans_grad}\n\n")
            
            # 回転/並進勾配の比率 - 最適化バランスの重要な指標
            avg_ratio = np.mean(ratio)
            f.write(f"Average rotation/translation gradient ratio: {avg_ratio:.4f}\n")
            f.write("Ideal ratio should be close to 1.0 for balanced optimization\n")
            f.write("Higher values mean rotation is dominating, lower values mean translation is dominating\n\n")
            
            # 結論
            f.write("4. CONCLUSION\n")
            f.write("-------------\n")
            
            # 最適化の成功判定
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # 潜在的な問題点のレポート
            issues = []
            if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                
            if plateau_count > len(loss_history) * 0.3:
                issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
            if vanishing_rot_grad or vanishing_trans_grad:
                issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
            if exploding_rot_grad or exploding_trans_grad:
                issues.append("- Gradients are very large, suggesting exploding gradient issues.")
                
            if avg_ratio > 10.0:
                issues.append(f"- Rotation/translation gradient ratio ({avg_ratio:.2f}) is much higher than 1.0, "
                            "which means rotation updates are dominating the optimization.")
            elif avg_ratio < 0.1:
                issues.append(f"- Rotation/translation gradient ratio ({avg_ratio:.2f}) is much lower than 1.0, "
                            "which means translation updates are dominating the optimization.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("No significant optimization issues detected.\n")
                
            # 最適化改善のための提案
            f.write("\n5. SUGGESTIONS FOR IMPROVEMENT\n")
            f.write("------------------------------\n")
            
            suggestions = []
            
            if avg_ratio > 5.0:
                suggestions.append("- Consider decreasing rotation learning rate or increasing translation learning rate.")
            elif avg_ratio < 0.2:
                suggestions.append("- Consider increasing rotation learning rate or decreasing translation learning rate.")
                
            if exploding_rot_grad or exploding_trans_grad:
                suggestions.append("- Consider using a smaller learning rate or increasing gradient clipping threshold.")
                
            if vanishing_rot_grad or vanishing_trans_grad:
                suggestions.append("- Consider using a larger learning rate or different optimizer (e.g. Adam).")
                
            if oscillation_count > len(loss_history) * 0.2:
                suggestions.append("- Increase momentum or add decay to learning rate to stabilize optimization.")
                
            if plateau_count > len(loss_history) * 0.4:
                suggestions.append("- Try different learning rate schedule or optimizer to escape plateaus.")
                
            if len(suggestions) > 0:
                for suggestion in suggestions:
                    f.write(suggestion + "\n")
            else:
                f.write("No specific improvements needed. The optimization appears to be well-configured.\n")
                
        # ============== 8. 成分ごとの勾配履歴の詳細分析 ==============
        # 回転成分（wx, wy, wz）のグラフ
        plt.figure(figsize=(15, 10))
        for i, component in enumerate(['wx', 'wy', 'wz']):
            plt.subplot(3, 1, i+1)
            plt.plot(iterations, rot_grad_data[:, i], 'b-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Rotation Gradient - {component} Component')
            plt.xlabel('Iteration')
            plt.ylabel(f'Gradient Value (d/d{component})')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rot_gradient_components_detail.png'), dpi=150)
        plt.close()

        # 並進成分（tx, ty, tz）のグラフ
        plt.figure(figsize=(15, 10))
        for i, component in enumerate(['tx', 'ty', 'tz']):
            plt.subplot(3, 1, i+1)
            plt.plot(iterations, trans_grad_data[:, i], 'r-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Translation Gradient - {component} Component')
            plt.xlabel('Iteration')
            plt.ylabel(f'Gradient Value (d/d{component})')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trans_gradient_components_detail.png'), dpi=150)
        plt.close()
        
        print(f"Saved SE(3) optimization diagnostics to {output_dir}")


    def optimize_with_essential_geoopt(
        self,
        max_iter: int = 5000,
        tol: float = 1e-6,
        lr: float = 3e-3,
        grad_clip: Optional[float] = None,
        save_diagnostics: bool = True,
        diagnostics_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Camera-2 の姿勢 (R_wc, t̂_wc) を
            (q, t̂) ∈ S³ × S²
        の製品多様体上で内在的に最適化します。四元数表現を使うことでSO(3)制約を
        自然に満たし、数値的安定性を向上させます。
        
        コア最適化ロジックに加え、詳細な診断情報とビジュアライゼーションを提供します。
        
        Args:
            max_iter: 最大イテレーション数
            tol: 収束閾値
            lr: 学習率（回転と並進で共通）
            grad_clip: 勾配クリップの閾値（Noneの場合はクリップしない）
            save_diagnostics: 詳細な診断情報を保存するかどうか
            diagnostics_dir: 診断ファイルを保存するディレクトリ
            results_dir: 結果ファイルを保存するディレクトリ
            seed: 乱数シード
            
        Returns:
            loss_history: 損失値の履歴
        """
        import os
        import torch
        import geoopt
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        # -------------------- 出力セットアップ -------------------- #
        results_dir = results_dir or os.path.join("results", "transport_geoopt_diagnose")
        os.makedirs(results_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_geoopt_diagnose")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # -------------------- 乱数シード設定 -------------------- #
        if seed is not None:
            torch.manual_seed(seed)

        # -------------------- 球面多様体を設定 -------------------- #
        sphere4 = geoopt.manifolds.Sphere()  # S³ for quaternion
        sphere3 = geoopt.manifolds.Sphere()  # S² for unit translation
        
        # ProductManifoldを正しく構築 - 各多様体とそのシェイプをタプルで指定
        manifold = geoopt.ProductManifold((sphere4, 4), (sphere3, 3))
        
        # -------------------- パラメータ初期化 -------------------- #
        if not hasattr(self, "theta_param"):
            # 初期回転行列から四元数へ変換
            if hasattr(self, "R_wc_param"):
                # 既存の回転行列から四元数に変換
                R0 = self.R_wc_param.detach().clone()
                q0 = self.lie.SO3_to_quat(R0)
            else:
                # 単位四元数 (恒等回転)
                q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            
            # 初期並進
            if hasattr(self, "t_hat_param"):
                t0 = self.t_hat_param.detach().clone()
            else:
                t0 = torch.tensor([1.0, 0.0, 0.0], device=self.device)  # +X方向
                t0 = t0 / t0.norm()
            
            # ProductManifold用の初期パラメータを正しく構築
            init_param = manifold.pack_point(q0.to(self.device), t0.to(self.device))
            self.theta_param = geoopt.ManifoldParameter(init_param, manifold=manifold)

        # -------------------- オプティマイザとスケジューラ -------------------- #
        optimizer = geoopt.optim.RiemannianAdam([self.theta_param], lr=lr)
        
        # 毎ステップでスケジューラを呼ぶ場合の減衰率
        # 「max_iterステップで0.25倍」になるように調整
        gamma = 0.999 ** (1 / max_iter)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # -------------------- ログ・診断用変数 -------------------- #
        loss_history = []
        q_param_history = []
        t_param_history = []
        q_grad_history = []
        t_grad_history = []
        R_history = []
        E_history = []  # Essential matrix履歴
        F_history = []  # Fundamental matrix履歴
        
        # デバッグログ設定
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration,Loss,q_Grad_Norm,t_Grad_Norm,q_t_Grad_Ratio,Quaternion_Norm\n")

        # -------------------- メインループ -------------------- #
        pbar = tqdm(range(max_iter), desc="Optimizing (S³×S²)", leave=True)
        prev_loss = float("inf")
        
        for it in pbar:
            optimizer.zero_grad()
            
            # パラメータが多様体上にあることを定期的に確認（デバッグ用）
            if it % 50 == 0:
                ok, reason = manifold.check_point_on_manifold(
                self.theta_param, explain=True)
                assert ok, f"Theta went off-manifold: {reason}"
            
            # 四元数と並進をProductManifoldから取得（cloneなしで軽量化）
            q, t_hat = manifold.unpack_tensor(self.theta_param)
            
            # 四元数から回転行列へ変換 - 数値安定性のため明示的な正規化
            q_normalized = q / q.norm()
            R_wc = self.quaternion.q_to_R(q_normalized)
            
            # Fundamental matrix → Cost → Transport → Loss
            F = self._build_F_from_wc(R_wc, t_hat)
            C = self.compute_cost_matrix_fundamental(F)
            T = self.unbalanced_sinkhorn_algorithm(C)
            loss = (T * C).sum()

            loss.backward()

            # 勾配クリッピング - NaN検知付き
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [self.theta_param], 
                    grad_clip, 
                    error_if_nonfinite=True
                )

            # -------------------- 勾配ノルムと診断情報の記録 -------------------- #
            with torch.no_grad():
                # 勾配ノルムの計算
                grad = self.theta_param.grad
                q_grad, t_grad = manifold.unpack_tensor(grad)
                
                gq = q_grad.norm().item() if q_grad is not None else 0
                gt = t_grad.norm().item() if t_grad is not None else 0
                q_t_ratio = gq / (gt + 1e-10)
                
                # Essential matrix計算（診断用）
                E = self.lie.skew_symmetric(t_hat) @ R_wc
                
                # 履歴の記録
                curr_loss = loss.item()
                loss_history.append(curr_loss)
                q_param_history.append(q.detach().clone())
                t_param_history.append(t_hat.detach().clone())
                q_grad_history.append(q_grad.detach().clone() if q_grad is not None else None)
                t_grad_history.append(t_grad.detach().clone() if t_grad is not None else None)
                R_history.append(R_wc.detach().clone())
                E_history.append(E.detach().clone())
                F_history.append(F.detach().clone())
                
                # デバッグログに記録
                with open(debug_log_path, 'a') as f:
                    f.write(f"{it},{curr_loss:.8f},{gq:.8f},{gt:.8f},{q_t_ratio:.8f},{q.norm().item():.8f}\n")
                
                # 定期的な詳細ログ出力
                if it % 20 == 0:
                    # Essential matrixの特異値分析
                    U, S, Vh = torch.linalg.svd(E, full_matrices=False)
                    print(f"\nIteration {it}")
                    print(f"  Loss: {curr_loss:.8f}")
                    print(f"  q grad norm: {gq:.8f}, t grad norm: {gt:.8f}, ratio: {q_t_ratio:.3f}")
                    print(f"  quaternion norm: {q.norm().item():.6f}")
                    print(f"  E singular values: {S.cpu().numpy()}")
                    print(f"  σ1/σ2 ratio: {(S[0]/S[1]).item():.3f} (ideal: 1.0)")
                    print(f"  σ3 value: {S[2].item():.6f} (ideal: 0.0)")

            # 最適化ステップ
            optimizer.step()
            
            # スケジューラを毎ステップ更新
            scheduler.step()

            # 収束チェック
            pbar.set_postfix(
                loss=f"{curr_loss:.6f}",
                gq=f"{gq:.4f}",
                gt=f"{gt:.4f}",
                q_t=f"{q_t_ratio:.2f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )
            if it > 5 and abs(prev_loss - curr_loss) < tol:
                pbar.set_description("Converged")
                break
            prev_loss = curr_loss

            # --------- 途中のトランスポートプラン可視化 (20iter毎) --------- #
            if it % 20 == 0 or it == max_iter - 1:
                with torch.no_grad():
                    T_np = T.detach().cpu().numpy()
                    rows, cols = T_np.shape
                    aspect_ratio = cols / rows
                    if rows > cols:
                        fig_width = 8
                        fig_height = min(20, fig_width / aspect_ratio)
                    else:
                        fig_height = 6
                        fig_width = min(20, fig_height * aspect_ratio)
                        
                    plt.figure(figsize=(fig_width, fig_height))
                    
                    if rows > 1000 or cols > 1000:
                        downsample_factor = max(1, int(max(rows, cols) / 1000))
                        t_np_display = T_np[::downsample_factor, ::downsample_factor]
                        plt.imshow(t_np_display, cmap="hot", interpolation="nearest", aspect="auto")
                        plt.title(f"Transport Plan at Iteration {it} (Downsampled {downsample_factor}x)")
                    else:
                        plt.imshow(T_np, cmap="hot", interpolation="nearest", aspect="auto")
                        plt.title(f"Transport Plan at Iteration {it}")
                        
                    plt.colorbar(label="Transport")
                    plt.xlabel("Image 2 Gaussians")
                    plt.ylabel("Image 1 Gaussians")
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"transport_iter_{it:05d}.png"), dpi=150)
                    plt.close()

        # -------------------- 最終パラメータの保存 -------------------- #
        with torch.no_grad():
            # 最終四元数と並進ベクトルを取得
            q_final, t_final = manifold.unpack_tensor(self.theta_param)
            
            # 四元数の符号を統一（連続性のため）
            if q_final[0] < 0:
                q_final = -q_final
                # 符号修正をtheta_paramにも反映（再学習時の一貫性のため）
                self.theta_param.copy_(manifold.pack_point((q_final, t_final)))
            
            # 最終回転行列と並進ベクトル
            q_final_normalized = q_final / q_final.norm()  # 最終的な数値安定性を確保
            self.R_wc = self.quaternion.q_to_R(q_final_normalized).detach()
            self.t_wc = t_final.detach()
            
            # 基礎行列を計算
            self.f = self._build_F_from_wc(self.R_wc, self.t_wc)
            
            # 解析用にSE(3)形式(camera→world)も保持
            self.R_cw = self.R_wc.t()
            self.t_cw = -self.R_cw @ self.t_wc
            
            # 最終Essential matrixの保存
            self.E_raw = self.lie.skew_symmetric(self.t_wc) @ self.R_wc
            
            # 内部パラメータも更新しておく
            self.q_final = q_final.clone()  # 明示的にcloneして安全に保存
            
            # OpenCV形式のパラメータも保存（cv2が利用可能な場合）
            if 'cv2' in globals() and cv2 is not None:
                rvec_numpy, _ = cv2.Rodrigues(self.R_wc.cpu().numpy())
                self.rvec = torch.from_numpy(rvec_numpy).to(self.device)
                self.tvec = self.t_cw
                rvec_cw_numpy, _ = cv2.Rodrigues(self.R_cw.cpu().numpy())
                self.rvec_cw = torch.from_numpy(rvec_cw_numpy).to(self.device)
                self.center = self.t_cw

        # --------- 損失曲線 --------- #
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, "-o", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss (S³×S² Optimization)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=150)
        plt.close()

        # --------- 診断情報の保存 --------- #
        if save_diagnostics:
            # 簡易グラフ：勾配ノルム
            plt.figure(figsize=(10, 6))
            q_grad_norms = [g.norm().item() if g is not None else 0 for g in q_grad_history]
            t_grad_norms = [g.norm().item() if g is not None else 0 for g in t_grad_history]
            plt.semilogy(q_grad_norms, 'r-', label="‖grad q‖")
            plt.semilogy(t_grad_norms, 'b-', label="‖grad t̂‖")
            plt.legend(); plt.grid(True)
            plt.title("Gradient norms")
            plt.xlabel("Iteration")
            plt.ylabel("Norm (log scale)")
            plt.tight_layout()
            plt.savefig(os.path.join(diagnostics_dir, "grad_norm.png"), dpi=150)
            plt.close()
            
            # 詳細な診断情報
            self.save_optimization_diagnostics_quaternion(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                q_param_history=q_param_history,
                t_param_history=t_param_history,
                q_grad_history=q_grad_history,
                t_grad_history=t_grad_history,
                R_history=R_history,
                E_history=E_history
            )

        print(f"[Geoopt-S³×S²] finished after {it+1} iterations, final loss = {curr_loss:.6f}")
        print(f"Final quaternion norm: {q_final_normalized.norm().item():.6f}")
        print(f"Final R_wc det: {torch.linalg.det(self.R_wc).item():.6f}")
        return loss_history

    def save_optimization_diagnostics_quaternion(self, 
                                   output_dir: str,
                                   loss_history: list,
                               q_param_history: list,
                               t_param_history: list,
                               q_grad_history: list,
                               t_grad_history: list,
                               R_history: list,
                               E_history: list) -> None:
        """
        S³×S²リーマン多様体上での最適化の詳細な診断情報を保存します。
        
        Args:
            output_dir: 診断ファイルを保存するディレクトリ
            loss_history: 各イテレーションでの損失値
            q_param_history: 各イテレーションでの四元数パラメータ (S³)
            t_param_history: 各イテレーションでの並進単位ベクトル (S²)
            q_grad_history: 各イテレーションでのqの勾配
            t_grad_history: 各イテレーションでのtの勾配
            R_history: 各イテレーションでの回転行列
            E_history: 各イテレーションでのEssential matrix
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.mplot3d import Axes3D
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # リスト→NumPy配列への変換
        iterations = range(len(loss_history))
        
        # 勾配ノルム計算
        q_grad_norms = np.array([g.norm().item() if g is not None else 0 for g in q_grad_history])
        t_grad_norms = np.array([g.norm().item() if g is not None else 0 for g in t_grad_history])
        total_grad_norms = np.sqrt(q_grad_norms**2 + t_grad_norms**2)
        
        # 勾配比率 (q/t)
        q_t_ratios = q_grad_norms / (t_grad_norms + 1e-10)
        
        # パラメータをNumPy配列に変換
        q_params_np = np.array([q.cpu().numpy() for q in q_param_history])
        t_params_np = np.array([t.cpu().numpy() for t in t_param_history])
        R_params_np = np.array([R.cpu().numpy() for R in R_history])
        
        # 回転行列の行列式履歴
        det_history = np.array([np.linalg.det(R) for R in R_params_np])
        
        # 四元数のノルム履歴
        q_norms = np.array([np.linalg.norm(q) for q in q_params_np])
        
        # Essential matrixの特異値履歴
        singular_values = []
        for E in E_history:
            U, S, Vh = torch.linalg.svd(E, full_matrices=False)
            singular_values.append(S.cpu().numpy())
        
        singular_values = np.array(singular_values)
        
        # ======================= 1. 損失軌跡の分析 =======================
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization (S³×S²)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 損失の変化率
        if len(loss_history) > 1:
            plt.subplot(212)
            loss_changes = np.array([loss_history[i+1] - loss_history[i] 
                                    for i in range(len(loss_history)-1)])
            plt.plot(iterations[:-1], loss_changes, 'r-')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title('Loss Change Between Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Loss Difference')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_analysis.png'), dpi=150)
        plt.close()
        
        # ======================= 2. 勾配分析 =======================
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # 総合勾配ノルム
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, total_grad_norms, 'k-', linewidth=2, label='Total')
        ax1.plot(iterations, q_grad_norms, 'r-', linewidth=1.5, label='S³')
        ax1.plot(iterations, t_grad_norms, 'b-', linewidth=1.5, label='S²')
        ax1.set_title('Riemannian Gradient Magnitude')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')
        ax1.grid(True)
        ax1.legend()
        
        # q/t勾配比率
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, q_t_ratios, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Balanced ratio (1.0)')
        ax2.set_title('Ratio of S³ to S² Gradient Norms')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Ratio')
        ax2.set_yscale('log')
        ax2.grid(True)
        ax2.legend()
        
        # 四元数と並進の勾配成分
        ax3 = fig.add_subplot(gs[2, 0])
        for i in range(4):
            ax3.plot(iterations, [g[i].item() if g is not None else 0 for g in q_grad_history], 
                    '-', label=f'q_grad[{i}]')
        
        for i in range(3):
            ax3.plot(iterations, [g[i].item() if g is not None else 0 for g in t_grad_history], 
                    '--', label=f't_grad[{i}]')
        
        ax3.set_title('Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        ax3.legend(ncol=3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_analysis.png'), dpi=150)
        plt.close()
        
        # ======================= 3. Essential matrix特異値分析 =======================
        plt.figure(figsize=(12, 8))
        
        # 特異値
        plt.subplot(211)
        plt.plot(iterations, singular_values[:, 0], 'r-', label='σ₁')
        plt.plot(iterations, singular_values[:, 1], 'g-', label='σ₂')
        plt.plot(iterations, singular_values[:, 2], 'b-', label='σ₃')
        plt.title('Singular Values of Essential Matrix')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # σ₁/σ₂比率と σ₃
        plt.subplot(212)
        sigma_ratios = singular_values[:, 0] / (singular_values[:, 1] + 1e-10)
        plt.plot(iterations, sigma_ratios, 'r-', label='σ₁/σ₂ (ideal: 1.0)')
        plt.plot(iterations, singular_values[:, 2], 'b-', label='σ₃ (ideal: 0.0)')
        plt.title('Essential Matrix Constraints')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'essential_matrix_analysis.png'), dpi=150)
        plt.close()
        
        # ======================= 4. パラメータ状態分析 =======================
        fig = plt.figure(figsize=(12, 12))
        
        # 四元数成分
        ax1 = fig.add_subplot(311)
        for i in range(4):
            ax1.plot(iterations, q_params_np[:, i], '-', label=f'q[{i}]')
            ax1.set_title('Quaternion Components')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Value')
            ax1.grid(True)
            ax1.legend()
            
        # 四元数ノルム
        ax2 = fig.add_subplot(312)
        ax2.plot(iterations, q_norms, 'b-', linewidth=2)
        ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Unit norm (1.0)')
        ax2.set_title('Quaternion Norm')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Norm')
        ax2.set_ylim([0.98, 1.02])  # ノルムが1付近に集中しているため
        ax2.grid(True)
        ax2.legend()
        
        # 回転行列の行列式
        ax3 = fig.add_subplot(313)
        ax3.plot(iterations, det_history, 'g-', linewidth=2)
        ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='det = +1 (SO(3))')
        ax3.set_title('Determinant of Rotation Matrix')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Determinant')
        ax3.set_ylim([0.98, 1.02])  # 行列式が1付近に集中しているため
        ax3.grid(True)
        ax3.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_state.png'), dpi=150)
        plt.close()
            
        # ======================= 5. 3D軌跡の可視化 =======================
        # 並進ベクトルの軌跡（S²上）
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 単位球面の表示
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = 0.98 * np.outer(np.cos(u), np.sin(v))
        y = 0.98 * np.outer(np.sin(u), np.sin(v))
        z = 0.98 * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, color='lightgray', alpha=0.2)
        
        # t_hatの軌跡プロット
        ax.plot(t_params_np[:, 0], t_params_np[:, 1], t_params_np[:, 2], 'r-', linewidth=2)
        ax.scatter(t_params_np[0, 0], t_params_np[0, 1], t_params_np[0, 2], 
                   c='g', s=100, label='Initial')
        ax.scatter(t_params_np[-1, 0], t_params_np[-1, 1], t_params_np[-1, 2], 
                   c='b', s=100, label='Final')
        
        ax.set_title('Translation Direction on Unit Sphere (S²)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 't_trajectory_3d.png'), dpi=150)
        plt.close()

        # ======================= 6. テキスト形式のサマリーレポート =======================
        with open(os.path.join(output_dir, 'optimization_summary.txt'), 'w') as f:
            f.write("S³×S² RIEMANNIAN OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("==============================================\n\n")
            
            # 損失分析
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # 単調減少性チェック
            if len(loss_history) > 1:
                is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
                f.write(f"Loss decreases monotonically: {is_monotonic}\n")
                
                # 振動とプラトーの検出
                oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                        if (loss_history[i] > loss_history[i+1] and 
                                            loss_history[i+1] < loss_history[i+2]))
                
                plateau_threshold = 1e-6
                plateau_count = sum(1 for i in range(len(loss_history)-1) 
                                if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
                
                f.write(f"Number of oscillations: {oscillation_count}\n")
                f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # パラメータ分析
            f.write("2. QUATERNION AND TRANSLATION PARAMETERS\n")
            f.write("-------------------------------------\n")
            
            # 四元数
            f.write(f"Initial quaternion: [{q_params_np[0, 0]:.6f}, {q_params_np[0, 1]:.6f}, {q_params_np[0, 2]:.6f}, {q_params_np[0, 3]:.6f}]\n")
            f.write(f"Final quaternion: [{q_params_np[-1, 0]:.6f}, {q_params_np[-1, 1]:.6f}, {q_params_np[-1, 2]:.6f}, {q_params_np[-1, 3]:.6f}]\n")
            f.write(f"Initial quaternion norm: {q_norms[0]:.6f}\n")
            f.write(f"Final quaternion norm: {q_norms[-1]:.6f}\n\n")
            
            # 並進
            f.write(f"Initial translation: [{t_params_np[0, 0]:.6f}, {t_params_np[0, 1]:.6f}, {t_params_np[0, 2]:.6f}]\n")
            f.write(f"Final translation: [{t_params_np[-1, 0]:.6f}, {t_params_np[-1, 1]:.6f}, {t_params_np[-1, 2]:.6f}]\n\n")
            
            # 回転行列の行列式
            f.write(f"Initial rotation determinant: {det_history[0]:.6f}\n")
            f.write(f"Final rotation determinant: {det_history[-1]:.6f}\n\n")
            
            # Essential matrix分析
            f.write("3. ESSENTIAL MATRIX PROPERTIES\n")
            f.write("----------------------------\n")
            
            # 最終特異値
            final_sv = singular_values[-1]
            final_ratio = final_sv[0] / final_sv[1]
            f.write(f"Final singular values: {final_sv[0]:.6f}, {final_sv[1]:.6f}, {final_sv[2]:.6f}\n")
            f.write(f"σ₁/σ₂ ratio: {final_ratio:.6f} (ideal: 1.0)\n")
            f.write(f"σ₃ value: {final_sv[2]:.6e} (ideal: 0.0)\n\n")
            
            # 理論的な制約チェック
            sv_constraint_satisfied = final_ratio < 1.1 and final_sv[2] < 0.01
            f.write(f"Essential matrix constraints are {'satisfied' if sv_constraint_satisfied else 'NOT fully satisfied'}.\n\n")
            
            # 勾配分析
            f.write("4. GRADIENT BEHAVIOR\n")
            f.write("-------------------\n")
            
            # 勾配の統計
            max_grad = np.max(total_grad_norms)
            min_grad = np.min(total_grad_norms)
            avg_grad = np.mean(total_grad_norms)
            f.write(f"Gradient norm - Max: {max_grad:.6f}, Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n")
                
            # q/t勾配比率
            avg_ratio = np.mean(q_t_ratios)
            max_ratio = np.max(q_t_ratios)
            min_ratio = np.min(q_t_ratios)
            f.write(f"S³/S² gradient ratio - Avg: {avg_ratio:.3f}, Max: {max_ratio:.3f}, Min: {min_ratio:.3f}\n")
            f.write(f"This ratio shows the balance between rotation and translation optimization.\n\n")
            
            # 勾配消失/爆発チェック
            vanishing_threshold = 1e-6
            exploding_threshold = 1e2
            
            vanishing_q_grad = any(grad < vanishing_threshold for grad in q_grad_norms)
            exploding_q_grad = any(grad > exploding_threshold for grad in q_grad_norms)
            
            vanishing_t_grad = any(grad < vanishing_threshold for grad in t_grad_norms)
            exploding_t_grad = any(grad > exploding_threshold for grad in t_grad_norms)
            
            f.write(f"Quaternion gradient vanishing detected: {vanishing_q_grad}\n")
            f.write(f"Quaternion gradient exploding detected: {exploding_q_grad}\n")
            f.write(f"Translation gradient vanishing detected: {vanishing_t_grad}\n")
            f.write(f"Translation gradient exploding detected: {exploding_t_grad}\n\n")
            
            # 結論と最終評価
            f.write("5. CONCLUSION AND EVALUATION\n")
            f.write("---------------------------\n")
            
            # 最適化の成功判定
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # 潜在的な問題点のレポート
            issues = []
            if len(loss_history) > 2:
                if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                    issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                    
                if plateau_count > len(loss_history) * 0.3:
                    issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
                if vanishing_q_grad or vanishing_t_grad:
                    issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
                if exploding_q_grad or exploding_t_grad:
                    issues.append("- Gradients are very large, suggesting exploding gradient issues.")
                    
                if avg_ratio > 10.0:
                    issues.append(f"- S³/S² gradient ratio ({avg_ratio:.2f}) is much higher than 1.0, "
                                "which means quaternion updates are dominating the optimization.")
                elif avg_ratio < 0.1:
                    issues.append(f"- S³/S² gradient ratio ({avg_ratio:.2f}) is much lower than 1.0, "
                                "which means translation updates are dominating the optimization.")
            
                if final_ratio > 1.2:
                    issues.append(f"- Final σ₁/σ₂ ratio ({final_ratio:.2f}) is higher than ideal (1.0), "
                                "which may indicate an issue with the Essential matrix.")
                
                if final_sv[2] > 0.05:
                    issues.append(f"- Final σ₃ value ({final_sv[2]:.2e}) is not close enough to 0.0, "
                                "which may indicate an issue with the Essential matrix rank constraint.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
                f.write("\n")
            else:
                f.write("No significant optimization issues detected.\n\n")
                
            # 改善のための提案
            f.write("6. SUGGESTIONS FOR IMPROVEMENT\n")
            f.write("------------------------------\n")
            
            suggestions = []
            
            if avg_ratio > 5.0:
                suggestions.append("- Consider decreasing quaternion learning rate or increasing translation learning rate.")
            elif avg_ratio < 0.2:
                suggestions.append("- Consider increasing quaternion learning rate or decreasing translation learning rate.")
                
            if exploding_q_grad or exploding_t_grad:
                suggestions.append("- Consider using a smaller learning rate or increasing gradient clipping threshold.")
                
            if vanishing_q_grad or vanishing_t_grad:
                suggestions.append("- Consider using a larger learning rate or different optimizer parameters.")
                
                if oscillation_count > len(loss_history) * 0.2:
                    suggestions.append("- Increase momentum or add decay to learning rate to stabilize optimization.")
                    
                if plateau_count > len(loss_history) * 0.4:
                    suggestions.append("- Try different learning rate schedule or optimizer to escape plateaus.")
                
            if final_ratio > 1.2 or final_sv[2] > 0.05:
                suggestions.append("- Consider adding an explicit Essential matrix constraint or regularization term.")
            
            if len(suggestions) > 0:
                for suggestion in suggestions:
                    f.write(suggestion + "\n")
            else:
                f.write("No specific improvements needed. The optimization appears to be well-configured.\n")
        
        # ======================= 7. 成分ごとの勾配履歴の詳細分析 =======================
        # 四元数成分(w,x,y,z)の勾配グラフ
        plt.figure(figsize=(15, 10))
        components = ['w', 'x', 'y', 'z']
        for i, component in enumerate(components):
            plt.subplot(4, 1, i+1)
            plt.plot(iterations, [g[i].item() if g is not None else 0 for g in q_grad_history], 
                    'b-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Quaternion Gradient - {component} Component')
            plt.xlabel('Iteration')
            plt.ylabel(f'Gradient Value (d/d{component})')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quaternion_gradient_components.png'), dpi=150)
            plt.close()
        
        # 並進成分（tx, ty, tz）のグラフ
        plt.figure(figsize=(15, 10))
        for i, component in enumerate(['x', 'y', 'z']):
            plt.subplot(3, 1, i+1)
            plt.plot(iterations, [g[i].item() if g is not None else 0 for g in t_grad_history], 
                    'r-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Translation Gradient - {component} Component')
            plt.xlabel('Iteration')
            plt.ylabel(f'Gradient Value (d/dt_{component})')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'translation_gradient_components.png'), dpi=150)
            plt.close()
            
        # ======================= 8. パラメータ変化の分析 =======================
        # 各イテレーションでのパラメータの変化量
        plt.figure(figsize=(12, 6))
        
        # 四元数と並進の変化量
        q_changes = np.array([np.linalg.norm(q_params_np[i+1] - q_params_np[i]) 
                              for i in range(len(q_params_np)-1)])
        t_changes = np.array([np.linalg.norm(t_params_np[i+1] - t_params_np[i]) 
                              for i in range(len(t_params_np)-1)])
        
        plt.plot(iterations[:-1], q_changes, 'r-', label='Quaternion Change')
        plt.plot(iterations[:-1], t_changes, 'b-', label='Translation Change')
        plt.title('Parameter Change Magnitude per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Change Magnitude')
        plt.yscale('log')  # Log scaleで変化を見やすく
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_changes.png'), dpi=150)
        plt.close()
            
        # ======================= 9. 四元数空間での3D軌跡 =======================
        # 四元数の最初の3成分(x,y,z)を使った3D軌跡
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 単位四元数の純虚部は単位球面上の点を表す
        # そのため、四元数の虚部(x,y,z)を3D空間にプロット
        ax.plot(q_params_np[:, 1], q_params_np[:, 2], q_params_np[:, 3], 'r-', linewidth=2)
        ax.scatter(q_params_np[0, 1], q_params_np[0, 2], q_params_np[0, 3], 
                   c='g', s=100, label='Initial')
        ax.scatter(q_params_np[-1, 1], q_params_np[-1, 2], q_params_np[-1, 3], 
                   c='b', s=100, label='Final')
        
        ax.set_title('Quaternion Imaginary Part Trajectory (x,y,z)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quaternion_trajectory_3d.png'), dpi=150)
        plt.close()
        
        print(f"Saved detailed S³×S² optimization diagnostics to {output_dir}")

