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
        分布版 "対称エピポーラ距離" + 色差  
        Sampson ではなく  |p×ℓ|/‖ℓ‖  形式を左右合計し，
        ガウス形状の不確かさ u = nᵀΣn で割引する。
        """
        k1, k2 = self.means1.size(0), self.means2.size(0)

        # ------------------- ① 同次座標 -------------------
        ones1 = torch.ones(k1, 1, device=self.device)
        ones2 = torch.ones(k2, 1, device=self.device)
        p1_h = torch.cat([self.means1, ones1], 1)   # (K1,3)
        p2_h = torch.cat([self.means2, ones2], 1)   # (K2,3)

        # ------------------- ② エピポーラ線 ----------------
        l1 = (F   @ p2_h.T).T            # image-1 上 (K2,3)
        l2 = (F.T @ p1_h.T).T            # image-2 上 (K1,3)

        #  法線ベクトルとノルム
        n1 = l1[:, :2]                                   # (K2,2)
        n1_norm = n1.norm(dim=1, keepdim=True) + 1e-12
        n1_unit = n1 / n1_norm                           # (K2,2)

        n2 = l2[:, :2]                                   # (K1,2)
        n2_norm = n2.norm(dim=1, keepdim=True) + 1e-12
        n2_unit = n2 / n2_norm                           # (K1,2)

        # ------------------- ③ 点⇔線距離 -------------------
        #   dist_12(i,j) : p1_i → ℓ1_j
        numer_12 = torch.abs(p1_h @ l1.T)          # (K1,K2)
        dist_12  = numer_12 / n1_norm.T            # /‖ℓ1‖  (K1,K2)

        #   dist_21(i,j) : p2_j → ℓ2_i
        numer_21 = torch.abs(p2_h @ l2.T).T        # (K1,K2)
        dist_21  = numer_21 / n2_norm              # (K1,K2)

        symmetric_dist = dist_12 + dist_21         # (K1,K2)

        # ------------------- ④ 形状による割引 ---------------
        cov1 = self._make_cov_matrices(self.scales1, self.rotations1)   # (K1,2,2)
        cov2 = self._make_cov_matrices(self.scales2, self.rotations2)   # (K2,2,2)

        #   u1_i = n2_i^T Σ1_i n2_i   （n2_i は image-2 上の法線）
        v1 = torch.bmm(cov1, n2_unit.unsqueeze(-1)).squeeze(-1)    # (K1,2)
        u1 = (v1 * n2_unit).sum(1)                                 # (K1,)

        #   u2_j = n1_j^T Σ2_j n1_j
        v2 = torch.bmm(cov2, n1_unit.unsqueeze(-1)).squeeze(-1)    # (K2,2)
        u2 = (v2 * n1_unit).sum(1)                                 # (K2,)

        uncertainty = 1.0 + u1.view(-1, 1) + u2.view(1, -1)        # (K1,K2)
        epi_with_shape = symmetric_dist / uncertainty              # (K1,K2)

        # ------------------- ⑤ 色差 -------------------------
        rgb_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)   # (K1,K2,3)
        color_dist = (rgb_diff ** 2).sum(2)                          # (K1,K2)

        # ------------------- ⑥ 正規化 -----------------------
        with torch.no_grad():
            p95_epi   = torch.quantile(epi_with_shape, 0.95)
            p95_color = torch.quantile(color_dist, 0.95)

        epi_norm   = torch.clamp(epi_with_shape, max=p95_epi)   / p95_epi
        color_norm = torch.clamp(color_dist,    max=p95_color) / p95_color

        # ------------------- ⑦ コスト合成 -------------------
        cost = ( self.lambda_epipolar * epi_norm
            + self.lambda_color    * color_norm )
        
        # cost = (self.lambda_epipolar * epi_with_shape
        #     + self.lambda_color * color_dist)
        
        # cost = cost / cost.max().detach()

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
            {'params': self.rot_vec, 'lr': rot_lr, 'momentum': momentum, 'nesterov': False},
            {'params': self.trans_vec, 'lr': trans_lr, 'momentum': momentum, 'nesterov': False}
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
            t_wc = -R_wc @ t_cw

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
            
            # 勾配クリッピング（高周波ノイズ抑制）
            torch.nn.utils.clip_grad_norm_([self.rot_vec, self.trans_vec], max_norm=grad_clip)
            
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
                    'r/t': f"{ratio:.2f}",
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
                
        print(f"Saved SE(3) optimization diagnostics to {output_dir}")
 