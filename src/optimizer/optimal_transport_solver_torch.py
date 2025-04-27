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


class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians with Homography Optimization."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        k1: Optional[np.ndarray] = None,
        k2: Optional[np.ndarray] = None,
        epsilon: float = 0.01,
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

    

    def rodrigues(self, rvec: torch.Tensor) -> torch.Tensor:
        """SO(3) exponential map with small-angle safeguard and autograd support.
        
        rvec: (3,) -> rotation vector
        Returns: (3,3) rotation matrix using Rodrigues formula
        """
        # Calculate norm (rotation angle)
        theta = torch.linalg.norm(rvec)
        eps = 1e-6  # 小角近似の閾値
        
        # skew-symmetric matrix
        K = self.skew(rvec)
        
        # 小角度近似（θ ≈ 0の場合）
        R_small = torch.eye(3, dtype=torch.float32, device=rvec.device) + K
        
        # 通常の計算
        r_axis = rvec / torch.max(theta, torch.tensor(eps, device=rvec.device))
        K_unit = self.skew(r_axis)
        R_normal = (
            torch.eye(3, dtype=torch.float32, device=rvec.device)  # Identity matrix
            + torch.sin(theta) * K_unit                           # sin(θ)K term
            + (1.0 - torch.cos(theta)) * (K_unit @ K_unit)        # (1-cos(θ))K² term
        )
        
        # 条件に応じて値を選択（勾配は両経路に流れる）
        is_small = theta < eps
        R = torch.where(is_small, R_small, R_normal)
        
        return R

    def _build_F_from_wc(self, R_wc: torch.Tensor, t_wc: torch.Tensor) -> torch.Tensor:
        """R_wc, t_wc から F を構築。(build_f_from_rtの代替関数)
        F = K2^-T [t]_x R K1^-1
        """
        # [t]_x (外積行列)
        # tx = torch.zeros((3,3), device=self.device)
        # tx[0,1] = -t_wc[2]
        # tx[0,2] =  t_wc[1]
        # tx[1,0] =  t_wc[2]
        # tx[1,2] = -t_wc[0]
        # tx[2,0] = -t_wc[1]
        # tx[2,1] =  t_wc[0]
        tx = self.skew(t_wc)

        E = tx @ R_wc  # Essential matrix

        K1_inv = torch.inverse(self.k1)
        K2_inv = torch.inverse(self.k2)
        K2_inv_T = K2_inv.transpose(0,1)

        F = K2_inv_T @ E @ K1_inv
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
    
    def optimize_with_RT(self, max_iter=1000, tol=1e-6,
                            save_diagnostics=True, diagnostics_dir=None,
                            rot_scale: float = 1.0):
        """Optimize camera-to-world parameters (R_cw, c_w) using optimal transport loss.
        
        This method optimizes the camera-to-world transformation parameters (rotation and camera center)
        to find the best matching between two sets of 2D Gaussians. It uses SGD optimization
        with momentum to minimize the optimal transport cost based on the fundamental matrix constraint.
        
        The optimized parameters represent the camera-to-world transformation:
        - rvec_cw: Rotation vector (axis-angle) for camera-to-world rotation
        - center: Camera center position in world coordinates
        
        These parameters are then converted to world-to-camera (R_wc, t_wc) for computing the 
        fundamental matrix. The method also tracks optimization history and provides
        visualization of the transport plan and optimization progress.
        
        Args:
            max_iter (int): Maximum number of optimization iterations. Default: 1000.
            tol (float): Convergence tolerance for loss change. Default: 1e-6.
            save_diagnostics (bool): Whether to save detailed diagnostic information about 
                                    the optimization process. Default: True.
            diagnostics_dir (str, optional): Directory path to save diagnostic information.
                                            If None, uses "results/diagnostics_c2w". Default: None.
            rot_scale (float): Scaling factor for rotation gradients, which helps balance
                            the optimization between rotation and translation parameters. Default: 5.0.
        
        Returns:
            None: The optimized transformation parameters are stored as instance attributes:
                  - self.rvec_cw: Camera-to-world rotation vector
                  - self.center: Camera center in world coordinates
                  - self.f: The resulting fundamental matrix
                  - self.rvec, self.tvec: World-to-camera parameters (for compatibility)
        """

        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_RT_C2W")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_c2w")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history, param_history, grad_history = [], {'rvec_cw': [], 'center': []}, {'rvec_cw': [], 'center': []}

        # ------------------------- パラメータ初期化 ----------------------- #
        if not hasattr(self, 'rvec_cw'):
            self.rvec_cw = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
            # self.rvec_cw = nn.Parameter(torch.tensor([0.1, 0.05, 0.02], dtype=torch.float32, device=self.device))
        if not hasattr(self, 'center'):
            self.center = nn.Parameter(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device))

        # ----------- Adam → SGD  -------------- #
        optimizer = torch.optim.SGD(
            [{'params': self.rvec_cw, 'lr': 5e-3},      # 回転を速め
            {'params': self.center,  'lr': 5e-4}],    # 並進を遅め
        )

        prev_loss_val = float('inf')

        pbar = tqdm(range(max_iter), desc="Optimizing C2W", leave=True)

        # -------------------------  ループ  ------------------------------- #
        for iteration in pbar:
            optimizer.zero_grad()

            # C2W→W2C 変換
            R_cw = self.rodrigues(self.rvec_cw)
            R_wc = R_cw.t()
            t_wc = -R_wc @ self.center

            F = self._build_F_from_wc(R_wc, t_wc)

            # OT 損失
            cost_matrix = self.compute_cost_matrix_fundamental(F)
            transport   = self.unbalanced_sinkhorn_algorithm(cost_matrix)
            loss = torch.sum(transport * cost_matrix)
            loss.backward()

            # ---- 回転勾配を rot_scale 倍 ----
            if self.rvec_cw.grad is not None:
                self.rvec_cw.grad.mul_(rot_scale)

            # ---- ログ ----
            current_loss = loss.item()
            loss_history.append(current_loss)
            param_history['rvec_cw'].append(self.rvec_cw.clone())
            param_history['center'].append(self.center.clone())
            grad_history['rvec_cw'].append(self.rvec_cw.grad.clone() if self.rvec_cw.grad is not None else None)
            grad_history['center'].append(self.center.grad.clone() if self.center.grad is not None else None)

            # ---- 更新 ----
            optimizer.step()

            # π クランプ/tvecの正規化
            # with torch.no_grad():
            #     theta = torch.linalg.norm(self.rvec_cw)
            #     if theta > np.pi:
            #         self.rvec_cw.mul_(np.pi / theta)
            
            #     norm_t = torch.linalg.norm(self.center)
            #     if norm_t > 1e-8:
            #         self.center /= norm_t  

            # ---- 収束判定 ----
            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss

            if iteration % 10 == 0:
                grad_r = self.rvec_cw.grad.norm().item() if self.rvec_cw.grad is not None else 0
                grad_c = self.center.grad.norm().item() if self.center.grad is not None else 0
                pbar.set_postfix(loss=f"{current_loss:.6f}", grad_r=f"{grad_r:.6f}", grad_c=f"{grad_c:.6f}")

                # ---- Transport matrix visualization (元コードと同一) ----
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

        # ------------------------- 最終 F を保存 --------------------------- #
        with torch.no_grad():
            R_cw = self.rodrigues(self.rvec_cw)
            R_wc = R_cw.t()
            t_wc = -R_wc @ self.center
            final_F = self._build_F_from_wc(R_wc, t_wc)
            self.f = final_F

            rvec_numpy, _ = cv2.Rodrigues(R_wc.cpu().numpy())
            self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
            self.tvec = nn.Parameter(t_wc)

        # ------------------------- 損失プロット (元コードと同一) ------------ #
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_RT_C2W)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_RT_C2W.png"))
        plt.close()

        # ------------------------- 最適化過程描画 ------------- #
        if save_diagnostics:
            self.save_optimization_diagnostics_C2W(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )

    def save_optimization_diagnostics_C2W(self, 
                                    output_dir: str,
                                    loss_history: list,
                                    param_history: dict,
                                    grad_history: dict) -> None:
        """Save detailed diagnostics about the optimization process.
        
        Analyzes and visualizes the optimization process, including:
        - Loss trajectory
        - Parameter evolution
        - Gradient behavior
        - Convergence analysis
        
        Args:
            output_dir: Directory to save diagnostic files
            loss_history: List of loss values at each iteration
            param_history: Dictionary of parameter histories (e.g. {'rvec': [...], 'tvec': [...]})
            grad_history: Dictionary of gradient histories corresponding to parameters
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert histories to numpy arrays
        param_history_np = {}
        grad_history_np = {}
        
        for param_name, history in param_history.items():
            param_history_np[param_name] = np.array([p.detach().cpu().numpy() for p in history])
            
        for param_name, history in grad_history.items():
            grad_history_np[param_name] = np.array([g.detach().cpu().numpy() if g is not None 
                                                else np.zeros_like(param_history_np[param_name][0]) 
                                                for g in history])
        
        # Number of iterations
        iterations = range(len(loss_history))
        
        # 1. Loss Trajectory Analysis
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot loss changes (derivative) to see stability
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
        
        # 2. Parameter Trajectory Analysis for each parameter
        for param_name, param_data in param_history_np.items():
            if param_data.shape[1] == 3:  # For 3D vectors like rvec or center
                fig = plt.figure(figsize=(15, 5))
                plt.plot(iterations, param_data[:, 0], 'r-', label=f'{param_name}[0]')
                plt.plot(iterations, param_data[:, 1], 'g-', label=f'{param_name}[1]')
                plt.plot(iterations, param_data[:, 2], 'b-', label=f'{param_name}[2]')
                plt.title(f'{param_name} Components Over Time')
                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{param_name}_trajectory.png'), dpi=150)
                plt.close()
                
                # 3D Visualization of Parameter Trajectory
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(param_data[:, 0], param_data[:, 1], param_data[:, 2], 'r-', linewidth=2)
                ax.scatter(param_data[0, 0], param_data[0, 1], param_data[0, 2], c='g', s=100, label='Initial')
                ax.scatter(param_data[-1, 0], param_data[-1, 1], param_data[-1, 2], c='b', s=100, label='Final')
                ax.set_title(f'{param_name} Trajectory in 3D')
                ax.set_xlabel(f'{param_name}[0]')
                ax.set_ylabel(f'{param_name}[1]')
                ax.set_zlabel(f'{param_name}[2]')
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{param_name}_3d_trajectory.png'), dpi=150)
                plt.close()
        
        # 3. Gradient Analysis for each parameter
        for param_name, grad_data in grad_history_np.items():
            if grad_data.shape[1] == 3:  # For 3D vectors
                # Gradient magnitude
                grad_magnitude = np.linalg.norm(grad_data, axis=1)
                
                fig = plt.figure(figsize=(15, 10))
                gs = GridSpec(2, 2, figure=fig)
                
                # Plot gradient magnitude
                ax1 = fig.add_subplot(gs[0, :])
                ax1.plot(iterations, grad_magnitude, 'r-', linewidth=2)
                ax1.set_title(f'{param_name} Gradient Magnitude')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Gradient Norm')
                ax1.set_yscale('log')  # Log scale to better see changes
                ax1.grid(True)
                
                # Plot gradient components
                ax2 = fig.add_subplot(gs[1, :])
                ax2.plot(iterations, grad_data[:, 0], 'r-', label=f'grad_{param_name}[0]')
                ax2.plot(iterations, grad_data[:, 1], 'g-', label=f'grad_{param_name}[1]')
                ax2.plot(iterations, grad_data[:, 2], 'b-', label=f'grad_{param_name}[2]')
                ax2.set_title(f'{param_name} Gradient Components')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Gradient Value')
                ax2.grid(True)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{param_name}_gradient_analysis.png'), dpi=150)
                plt.close()
        
        # 4. Generate a text report with analysis
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("===========================\n\n")
            
            # Loss analysis
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # Monotonicity check
            is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
            f.write(f"Loss decreases monotonically: {is_monotonic}\n")
            
            # Find oscillations or plateaus
            oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                if (loss_history[i] > loss_history[i+1] and 
                                    loss_history[i+1] < loss_history[i+2]))
            
            plateau_threshold = 1e-6  # Define what constitutes a plateau
            plateau_count = sum(1 for i in range(len(loss_history)-1) 
                            if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
            
            f.write(f"Number of oscillations: {oscillation_count}\n")
            f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # Parameter analysis for each parameter
            f.write("2. PARAMETER BEHAVIOR\n")
            f.write("---------------------\n")
            for param_name, param_data in param_history_np.items():
                f.write(f"{param_name} (initial): " + np.array2string(param_data[0], precision=6) + "\n")
                f.write(f"{param_name} (final): " + np.array2string(param_data[-1], precision=6) + "\n")
                param_change = np.linalg.norm(param_data[-1] - param_data[0])
                f.write(f"Total {param_name} change magnitude: {param_change:.6f}\n\n")
            
            # Gradient analysis for each parameter
            f.write("3. GRADIENT BEHAVIOR\n")
            f.write("-------------------\n")
            for param_name, grad_data in grad_history_np.items():
                grad_magnitude = np.linalg.norm(grad_data, axis=1)
                max_grad = np.max(grad_magnitude)
                min_grad = np.min(grad_magnitude)
                avg_grad = np.mean(grad_magnitude)
                
                f.write(f"{param_name} gradient - Max: {max_grad:.6f}, Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n")
                
                # Check for vanishing/exploding gradients
                vanishing_threshold = 1e-6
                exploding_threshold = 1e2
                
                vanishing_grad = any(grad < vanishing_threshold for grad in grad_magnitude)
                exploding_grad = any(grad > exploding_threshold for grad in grad_magnitude)
                
                f.write(f"{param_name} gradient vanishing: {vanishing_grad}\n")
                f.write(f"{param_name} gradient exploding: {exploding_grad}\n\n")
            
            # Conclusion
            f.write("4. CONCLUSION\n")
            f.write("-------------\n")
            
            # Determine if the optimization was successful
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # Report potential issues
            issues = []
            if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                
            if plateau_count > len(loss_history) * 0.3:
                issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
                
            # Check for vanishing or exploding gradients across all parameters
            any_vanishing = any(
                np.any(np.linalg.norm(grad_history_np[param_name], axis=1) < vanishing_threshold)
                for param_name in grad_history_np
            )
            
            any_exploding = any(
                np.any(np.linalg.norm(grad_history_np[param_name], axis=1) > exploding_threshold)
                for param_name in grad_history_np
            )
            
            if any_vanishing:
                issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
            if any_exploding:
                issues.append("- Gradients are very large, suggesting exploding gradient issues.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("No significant optimization issues detected.\n")
            
        print(f"Saved optimization diagnostics to {output_dir}")

    def se3_exp(self, xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xi: (..., 6)  axis-angle と並進を連結した se(3) ベクトル
        return: R_cw (3×3), t_cw (3,)
        """
        omega, rho = xi[..., :3], xi[..., 3:]  # 回転と並進成分に分ける
        theta2 = torch.linalg.norm(omega, dim=-1, keepdim=True)**2  # 回転の大きさの二乗
        eps = 1e-8  # iNeRF Appendix Eq.(26)の閾値

        # JAX lax.condと同様に、torch.whereを使って勾配連続性を確保
        # 小角近似（theta≃0の場合）のテイラー展開係数
        A_small = 1.0 - theta2/6.0
        B_small = 0.5 - theta2/24.0
        C_small = 1.0/6.0 - theta2/120.0
        
        # 通常の計算（theta≠0の場合）
        theta = torch.sqrt(theta2)
        A_normal = torch.sin(theta) / theta
        B_normal = (1.0 - torch.cos(theta)) / theta**2
        C_normal = (1.0 - A_normal) / theta**2
        
        # 条件に応じて値を選択（勾配は両経路に流れる）
        is_small = theta2 < eps
        A = torch.where(is_small, A_small, A_normal)
        B = torch.where(is_small, B_small, B_normal)
        C = torch.where(is_small, C_small, C_normal)

        # Rodrigues公式による回転行列計算
        K = torch.zeros((*xi.shape[:-1], 3, 3), device=xi.device, dtype=xi.dtype)
        K[..., 0, 1] = -omega[..., 2]; K[..., 0, 2] =  omega[..., 1]
        K[..., 1, 0] =  omega[..., 2]; K[..., 1, 2] = -omega[..., 0]
        K[..., 2, 0] = -omega[..., 1]; K[..., 2, 1] =  omega[..., 0]

        eye = torch.eye(3, device=xi.device, dtype=xi.dtype)
        eye = eye.expand_as(K)

        # 回転行列計算: R = I + A*K + B*K^2
        R = eye + A[..., None] * K + B[..., None] * (K @ K)     # (3×3)

        # 並進ベクトル計算: t = V * rho  (Vは並進ヤコビアン)
        V = eye + B[..., None] * K + C[..., None] * (K @ K)
        t = (V @ rho[..., None]).squeeze(-1)                    # (3,)

        return R, t

    def optimize_with_SE3(self, max_iter=1000, tol=1e-6,
                        save_diagnostics=True, diagnostics_dir=None):
        """Optimize camera pose using Lie algebra SE(3) representation.
        
        This method uses a unified 6-DoF representation from the Lie algebra se(3) for
        camera pose optimization, combining rotation and translation into a single parameter vector.
        This provides better gradient behavior compared to separate optimization parameters.
        
        Args:
            max_iter (int): Maximum number of optimization iterations. Default: 1000.
            tol (float): Convergence tolerance for loss change. Default: 1e-6.
            save_diagnostics (bool): Whether to save diagnostic information. Default: True.
            diagnostics_dir (str, optional): Directory path for diagnostics. Default: None.
        
        Returns:
            None: The optimized transformation parameters are stored as instance attributes.
        """
        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_SE3")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_se3")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        param_history = {'se3_vec': []}
        grad_history = {'se3_vec': []}

        # ------------------------- パラメータ初期化 ----------------------- #
        if not hasattr(self, "se3_vec"):
            self._init_se3_like_cam1(rot_noise=0.05, trans_noise=0.05)

        # ------------------------- オプティマイザ設定 ----------------------- #
        # iNeRFと同様にAdamを使用し、weight_decayを0に設定
        optimizer = torch.optim.Adam([self.se3_vec], lr=3e-3)
        # iNeRFと同じ指数関数的学習率減衰: 0.8^(t/100)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))

        prev_loss_val = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing SE(3)", leave=True)

        # ---- 勾配デバッグ用ログファイル ----
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration, Loss, Grad_Norm, Grad_Rot_x, Grad_Rot_y, Grad_Rot_z, Grad_Trans_x, Grad_Trans_y, Grad_Trans_z\n")

        # -------------------------  ループ  ------------------------------- #
        for iteration in pbar:
            optimizer.zero_grad()
            
            # SE(3) Lie algebra → 回転行列と並進ベクトル
            R_cw, t_cw = self.se3_exp(self.se3_vec)
            
            # 世界→カメラ変換
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw

            # 基礎行列と損失の計算
            F = self._build_F_from_wc(R_wc, t_wc)
            cost_matrix = self.compute_cost_matrix_fundamental(F)
            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix)
            loss = torch.sum(transport * cost_matrix)
            
            # バックワードパス前のデバッグ出力
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  SE3 params: {self.se3_vec.data}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6e}")
            
            # 勾配計算
            loss.backward()
            
            # 勾配デバッグ - 値とノルムを表示
            if self.se3_vec.grad is not None:
                grad = self.se3_vec.grad
                grad_norm = grad.norm().item()
                
                # 勾配情報をログに記録
                with open(debug_log_path, 'a') as f:
                    grad_vals = grad.detach().cpu().numpy()
                    f.write(f"{iteration}, {loss.item():.6f}, {grad_norm:.6f}, " + 
                        f"{grad_vals[0]:.6f}, {grad_vals[1]:.6f}, {grad_vals[2]:.6f}, " +
                        f"{grad_vals[3]:.6f}, {grad_vals[4]:.6f}, {grad_vals[5]:.6f}\n")
                
                # 定期的に詳細な勾配情報を表示
                if iteration % 10 == 0:
                    print(f"  Gradient norm: {grad_norm:.6f}")
                    print(f"  Rot gradient: {grad[:3].detach().cpu().numpy()}")
                    print(f"  Trans gradient: {grad[3:].detach().cpu().numpy()}")
                    rot_grad_norm = grad[:3].norm().item()
                    trans_grad_norm = grad[3:].norm().item()
                    print(f"  Rot/Trans gradient norm ratio: {rot_grad_norm/max(trans_grad_norm, 1e-10):.6f}")
            else:
                print("Warning: No gradient computed!")

            # ---- ログ ----
            current_loss = loss.item()
            loss_history.append(current_loss)
            param_history['se3_vec'].append(self.se3_vec.clone())
            grad_history['se3_vec'].append(self.se3_vec.grad.clone() if self.se3_vec.grad is not None else None)

            # ---- 更新 ----
            optimizer.step()
            
            # iNeRFと同様の-π~πクリッピング（数値安定性のため）
            with torch.no_grad():
                self.se3_vec.data[:3].clamp_(-math.pi, math.pi)
            
            # 学習率更新
            scheduler.step()
            
            # 更新後のパラメータをデバッグ表示
            if iteration % 10 == 0:
                param_change = torch.norm(self.se3_vec.data - param_history['se3_vec'][-1].data)
                print(f"  Parameter change: {param_change.item():.6f}")
                print(f"  Updated SE3 params: {self.se3_vec.data}")

            # ---- 収束判定 ----
            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss

            # ---- プログレス更新 ----
            if iteration % 10 == 0:
                grad_norm = self.se3_vec.grad.norm().item() 
                rot_grad_norm = self.se3_vec.grad[:3].norm().item() 
                trans_grad_norm = self.se3_vec.grad[3:].norm().item()
                ratio = rot_grad_norm/max(trans_grad_norm, 1e-10) 
                pbar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'grad': f"{grad_norm:.4f}",
                    'r/t': f"{ratio:.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            # ---- 可視化（既存コードと同じ） ----
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
            # 最終的なSE(3)パラメータから変換結果を保存
            R_cw, t_cw = self.se3_exp(self.se3_vec)
            self.R_cw = R_cw
            self.t_cw = t_cw
            
            # 世界→カメラ変換も保存
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw
            self.R_wc = R_wc
            self.t_wc = t_wc
            
            # 基礎行列を計算して保存
            final_F = self._build_F_from_wc(R_wc, t_wc)
            self.f = final_F
            
            # 既存APIとの互換性のために従来のパラメータも更新
            rvec_numpy, _ = cv2.Rodrigues(R_wc.cpu().numpy())
            self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
            self.tvec = nn.Parameter(t_wc)
            
            # camera-to-world パラメータも更新
            rvec_cw_numpy, _ = cv2.Rodrigues(R_cw.cpu().numpy())
            self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
            self.center = nn.Parameter(t_cw)

        # ------------------------- 損失プロット -------------------------- #
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_SE3)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_SE3.png"))
        plt.close()

        # ------------------------- 最適化過程描画 -------------------------- #
        if save_diagnostics:
            self.save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )
            
    def _init_se3_like_cam1(self, rot_noise=0.05, trans_noise=0.05):
        """Cam-1 の姿勢 (I,0) から微小ノイズを加えて se3_vec を初期化
        
        カメラ1の姿勢（単位行列の回転と原点）から微小なランダムノイズを加えて
        SE(3)パラメータを初期化します。シーンが未知の場合の良い初期値になります。
        
        Args:
            rot_noise (float): 回転ノイズの大きさ [rad]。Default: 0.05
            trans_noise (float): 並進ノイズの大きさ [m]。Default: 0.05
        """
        # ① 回転：ランダム軸に ±rot_noise [rad] だけ回す
        axis = torch.randn(3, device=self.device)
        axis /= axis.norm() + 1e-8
        omega0 = axis * rot_noise * torch.randn(1, device=self.device)

        # ② 並進：cam-1 原点 (0,0,0) から trans_noise [m] だけずらす
        rho0 = trans_noise * torch.randn(3, device=self.device)

        self.se3_vec = nn.Parameter(torch.cat([omega0, rho0], dim=0))

    def save_optimization_diagnostics_SE3(self, 
                                    output_dir: str,
                                    loss_history: list,
                                    param_history: dict,
                                    grad_history: dict) -> None:
        """Save detailed diagnostics about the SE(3) optimization process.
        
        Analyzes and visualizes the optimization process of the SE(3) parameters, including:
        - Loss trajectory
        - SE(3) parameter evolution (rotation and translation components)
        - Gradient behavior
        - Convergence analysis
        
        Args:
            output_dir: Directory to save diagnostic files
            loss_history: List of loss values at each iteration
            param_history: Dictionary of parameter histories (contains 'se3_vec')
            grad_history: Dictionary of gradient histories corresponding to parameters
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert histories to numpy arrays
        param_history_np = {}
        grad_history_np = {}
        
        for param_name, history in param_history.items():
            param_history_np[param_name] = np.array([p.detach().cpu().numpy() for p in history])
            
        for param_name, history in grad_history.items():
            grad_history_np[param_name] = np.array([g.detach().cpu().numpy() if g is not None 
                                                else np.zeros_like(param_history_np[param_name][0]) 
                                                for g in history])
        
        # Number of iterations
        iterations = range(len(loss_history))
        
        # 1. Loss Trajectory Analysis
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot loss changes (derivative) to see stability
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
        
        # 2. SE(3) Parameter Trajectory Analysis
        se3_data = param_history_np['se3_vec']
        
        # Plot all 6 components
        fig = plt.figure(figsize=(15, 8))
        
        # Rotation components
        plt.subplot(211)
        plt.plot(iterations, se3_data[:, 0], 'r-', label='ωx')
        plt.plot(iterations, se3_data[:, 1], 'g-', label='ωy')
        plt.plot(iterations, se3_data[:, 2], 'b-', label='ωz')
        plt.title('SE(3) Rotation Components (ω) Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # Translation components
        plt.subplot(212)
        plt.plot(iterations, se3_data[:, 3], 'r-', label='ρx')
        plt.plot(iterations, se3_data[:, 4], 'g-', label='ρy')
        plt.plot(iterations, se3_data[:, 5], 'b-', label='ρz')
        plt.title('SE(3) Translation Components (ρ) Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_components_trajectory.png'), dpi=150)
        plt.close()
        
        # 3D Visualization of rotation and translation trajectories
        fig = plt.figure(figsize=(15, 7))
        
        # 3D plot of rotation components
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(se3_data[:, 0], se3_data[:, 1], se3_data[:, 2], 'r-', linewidth=2)
        ax1.scatter(se3_data[0, 0], se3_data[0, 1], se3_data[0, 2], c='g', s=100, label='Initial')
        ax1.scatter(se3_data[-1, 0], se3_data[-1, 1], se3_data[-1, 2], c='b', s=100, label='Final')
        ax1.set_title('Rotation Components (ω) Trajectory in 3D')
        ax1.set_xlabel('ωx')
        ax1.set_ylabel('ωy')
        ax1.set_zlabel('ωz')
        ax1.legend()
        
        # 3D plot of translation components
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(se3_data[:, 3], se3_data[:, 4], se3_data[:, 5], 'r-', linewidth=2)
        ax2.scatter(se3_data[0, 3], se3_data[0, 4], se3_data[0, 5], c='g', s=100, label='Initial')
        ax2.scatter(se3_data[-1, 3], se3_data[-1, 4], se3_data[-1, 5], c='b', s=100, label='Final')
        ax2.set_title('Translation Components (ρ) Trajectory in 3D')
        ax2.set_xlabel('ρx')
        ax2.set_ylabel('ρy')
        ax2.set_zlabel('ρz')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_3d_trajectory.png'), dpi=150)
        plt.close()
        
        # 3. Gradient Analysis
        grad_data = grad_history_np['se3_vec']
        
        # Gradient magnitude
        grad_magnitude = np.linalg.norm(grad_data, axis=1)
        rot_grad_magnitude = np.linalg.norm(grad_data[:, :3], axis=1)
        trans_grad_magnitude = np.linalg.norm(grad_data[:, 3:], axis=1)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Plot total gradient magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, grad_magnitude, 'k-', linewidth=2, label='Total')
        ax1.plot(iterations, rot_grad_magnitude, 'r-', linewidth=1.5, label='Rotation')
        ax1.plot(iterations, trans_grad_magnitude, 'b-', linewidth=1.5, label='Translation')
        ax1.set_title('SE(3) Gradient Magnitude')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')  # Log scale to better see changes
        ax1.grid(True)
        ax1.legend()
        
        # Plot rotation gradient components
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, grad_data[:, 0], 'r-', label='grad_ωx')
        ax2.plot(iterations, grad_data[:, 1], 'g-', label='grad_ωy')
        ax2.plot(iterations, grad_data[:, 2], 'b-', label='grad_ωz')
        ax2.set_title('Rotation Gradient Components')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot translation gradient components
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, grad_data[:, 3], 'r-', label='grad_ρx')
        ax3.plot(iterations, grad_data[:, 4], 'g-', label='grad_ρy')
        ax3.plot(iterations, grad_data[:, 5], 'b-', label='grad_ρz')
        ax3.set_title('Translation Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'se3_gradient_analysis.png'), dpi=150)
        plt.close()
        
        # 4. Plot ratio of rotation to translation gradient norms
        plt.figure(figsize=(12, 6))
        # Add small epsilon to avoid division by zero
        ratio = rot_grad_magnitude / (trans_grad_magnitude + 1e-10)
        plt.plot(iterations, ratio, 'b-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Balanced ratio (1.0)')
        plt.title('Ratio of Rotation to Translation Gradient Norms')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.yscale('log')  # Log scale to better see changes
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_ratio.png'), dpi=150)
        plt.close()
        
        # 5. Generate a text report with analysis
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("SE(3) OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("==================================\n\n")
            
            # Loss analysis
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # Monotonicity check
            is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
            f.write(f"Loss decreases monotonically: {is_monotonic}\n")
            
            # Find oscillations or plateaus
            oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                    if (loss_history[i] > loss_history[i+1] and 
                                        loss_history[i+1] < loss_history[i+2]))
            
            plateau_threshold = 1e-6  # Define what constitutes a plateau
            plateau_count = sum(1 for i in range(len(loss_history)-1) 
                            if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
            
            f.write(f"Number of oscillations: {oscillation_count}\n")
            f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # Parameter analysis
            f.write("2. SE(3) PARAMETER BEHAVIOR\n")
            f.write("-------------------------\n")
            f.write(f"SE(3) vector (initial): " + np.array2string(se3_data[0], precision=6) + "\n")
            f.write(f"SE(3) vector (final): " + np.array2string(se3_data[-1], precision=6) + "\n")
            
            # Split into rotation and translation
            f.write(f"Rotation component (ω) initial: " + np.array2string(se3_data[0, :3], precision=6) + "\n")
            f.write(f"Rotation component (ω) final: " + np.array2string(se3_data[-1, :3], precision=6) + "\n")
            rot_change = np.linalg.norm(se3_data[-1, :3] - se3_data[0, :3])
            f.write(f"Total rotation change magnitude: {rot_change:.6f}\n\n")
            
            f.write(f"Translation component (ρ) initial: " + np.array2string(se3_data[0, 3:], precision=6) + "\n")
            f.write(f"Translation component (ρ) final: " + np.array2string(se3_data[-1, 3:], precision=6) + "\n")
            trans_change = np.linalg.norm(se3_data[-1, 3:] - se3_data[0, 3:])
            f.write(f"Total translation change magnitude: {trans_change:.6f}\n\n")
            
            # Gradient analysis
            f.write("3. GRADIENT BEHAVIOR\n")
            f.write("-------------------\n")
            max_grad = np.max(grad_magnitude)
            min_grad = np.min(grad_magnitude)
            avg_grad = np.mean(grad_magnitude)
            
            f.write(f"Total gradient - Max: {max_grad:.6f}, Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n")
            f.write(f"Rotation gradient - Max: {np.max(rot_grad_magnitude):.6f}, " 
                    f"Min: {np.min(rot_grad_magnitude):.6f}, Avg: {np.mean(rot_grad_magnitude):.6f}\n")
            f.write(f"Translation gradient - Max: {np.max(trans_grad_magnitude):.6f}, "
                    f"Min: {np.min(trans_grad_magnitude):.6f}, Avg: {np.mean(trans_grad_magnitude):.6f}\n")
            
            # Check for vanishing/exploding gradients
            vanishing_threshold = 1e-6
            exploding_threshold = 1e2
            
            vanishing_grad = any(grad < vanishing_threshold for grad in grad_magnitude)
            exploding_grad = any(grad > exploding_threshold for grad in grad_magnitude)
            
            f.write(f"Gradient vanishing detected: {vanishing_grad}\n")
            f.write(f"Gradient exploding detected: {exploding_grad}\n\n")
            
            # Ratio of rotation/translation gradients - good indicator of balance
            avg_ratio = np.mean(ratio)
            f.write(f"Average rotation/translation gradient ratio: {avg_ratio:.4f}\n")
            f.write(f"Ideal ratio should be close to 1.0 for balanced optimization\n\n")
            
            # Conclusion
            f.write("4. CONCLUSION\n")
            f.write("-------------\n")
            
            # Determine if the optimization was successful
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # Report potential issues
            issues = []
            if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                
            if plateau_count > len(loss_history) * 0.3:
                issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
            if vanishing_grad:
                issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
            if exploding_grad:
                issues.append("- Gradients are very large, suggesting exploding gradient issues.")
                
            if avg_ratio > 10.0 or avg_ratio < 0.1:
                issues.append(f"- Rotation/translation gradient ratio ({avg_ratio:.2f}) is far from balanced, "
                            "which may cause biased optimization.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("No significant optimization issues detected.\n")
                
        print(f"Saved SE(3) optimization diagnostics to {output_dir}")

    def skew(self, v: torch.Tensor) -> torch.Tensor:
        """(…,3) → (…,3,3)  skew-sym. matrix"""
        K = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
        K[..., 0, 1] = -v[..., 2];  K[..., 0, 2] =  v[..., 1]
        K[..., 1, 0] =  v[..., 2];  K[..., 1, 2] = -v[..., 0]
        K[..., 2, 0] = -v[..., 1];  K[..., 2, 1] =  v[..., 0]
        return K

    def se3_exp_T(self, xi: torch.Tensor) -> torch.Tensor:
        """
        xi: (6,) → 4×4  SE(3)   ― iNeRF Eq.(6)
        """
        omega, v = xi[:3], xi[3:]  # 回転と並進成分に分ける
        theta2 = (omega * omega).sum()  # 回転ベクトルの大きさの二乗
        eps = 1e-8  # iNeRF Appendix Eq.(26)の閾値

        # JAX lax.condと同様に、torch.whereを使って勾配連続性を確保
        # 小角近似（theta≃0の場合）のテイラー展開係数
        A_small = 1.0 - theta2/6.0
        B_small = 0.5 - theta2/24.0
        C_small = 1.0/6.0 - theta2/120.0
        
        # 通常の計算（theta≠0の場合）
        theta = torch.sqrt(theta2)
        A_normal = torch.sin(theta) / theta
        B_normal = (1.0 - torch.cos(theta)) / theta2
        C_normal = (1.0 - A_normal) / theta2
        
        # 条件に応じて値を選択（勾配は両経路に流れる）
        is_small = theta2 < eps
        A = torch.where(is_small, A_small, A_normal)
        B = torch.where(is_small, B_small, B_normal)
        C = torch.where(is_small, C_small, C_normal)

        # 歪対称行列を計算
        K = self.skew(omega)
        
        # 回転行列と並進ヤコビアンを計算
        eye = torch.eye(3, device=xi.device)
        R = eye + A * K + B * K @ K
        V = eye + B * K + C * K @ K
        
        # 並進ベクトルを計算
        t = V @ v

        # 4x4変換行列を作成
        T = torch.eye(4, device=xi.device)
        T[:3, :3], T[:3, 3] = R, t
        return T

    def optimize_with_inerf(self, 
                        max_iter: int = 1000, 
                        tol: float = 1e-5, 
                        save_diagnostics: bool = True, 
                        diagnostics_dir: Optional[str] = None,
                        viz_every: int = 10,
                        learning_rate: float = 1e-3):
        """
        iNeRF風最適化により、カメラポーズ（R & t）を最適化する。
        """
        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_inerf")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_inerf")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        param_history = {'T': []} if save_diagnostics else None
        grad_history = {'delta': []} if save_diagnostics else None

        # ------------------------- パラメータ初期化 ----------------------- #
        if not hasattr(self, 'T'):
            # 初期パラメータを作成
            if hasattr(self, 'se3_vec'):
                # SE3パラメータが既に存在する場合はそれを使う
                R_cw, t_cw = self.se3_exp(self.se3_vec)
            else:
                # なければランダム初期化
                self._init_se3_like_cam1(rot_noise=0.05, trans_noise=0.05)
                R_cw, t_cw = self.se3_exp(self.se3_vec)
                
            # 4x4の同次変換行列を作成
            self.T = torch.eye(4, device=self.device)
            self.T[:3, :3] = R_cw
            self.T[:3, 3] = t_cw

        # --- iNeRFと同じ: ループ外で1度だけパラメータとoptimizerを生成 ---
        self.delta = nn.Parameter(torch.zeros(6, device=self.device))
        # iNeRFと同様、Adamを使用（重み減衰なし）
        optimizer = torch.optim.Adam([self.delta], lr=learning_rate, weight_decay=0.0)
        # iNeRFと同じ指数関数的学習率減衰: 0.8^(t/100)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))

        prev_loss_val = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing iNeRF", leave=True)

        # ---- 勾配デバッグ用ログファイル ----
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug_inerf.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration, Loss, Delta_Norm, Grad_Norm, Grad_Rot_x, Grad_Rot_y, Grad_Rot_z, Grad_Trans_x, Grad_Trans_y, Grad_Trans_z\n")

        # -------------------------  ループ  ------------------------------- #
        for iteration in pbar:
            # 1. Forward pass: 勾配計算のリセット
            optimizer.zero_grad()
            
            # 2. Δξからexp(Δξ)を計算
            T_delta = self.se3_exp_T(self.delta)
            
            # 3. 更新: T_new = T_delta * T (左から掛ける - iNeRFと同じ)
            T_next = T_delta @ self.T
            
            # 4. カメラ→ワールド変換行列から回転と並進を抽出
            R_cw = T_next[:3, :3]
            t_cw = T_next[:3, 3]
            
            # 5. ワールド→カメラ変換に変更
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw
            
            # 6. 基礎行列計算
            F = self._build_F_from_wc(R_wc, t_wc)
            
            # 7. コスト行列と最適輸送計算
            cost_matrix = self.compute_cost_matrix_fundamental(F)
            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix)
            
            # 8. 損失計算
            loss = torch.sum(transport * cost_matrix)
            
            # 9. バックワード前のデバッグ情報
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  Delta SE3: {self.delta.data}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6e}")
                with torch.no_grad():
                    print(f"  Cost matrix min/max: {cost_matrix.min().item():.6f}/{cost_matrix.max().item():.6f}")
            
            # 10. バックワード計算
            loss.backward()
            
            # 11. 勾配チェック
            if self.delta.grad is not None:
                # 勾配情報取得
                grad = self.delta.grad
                grad_norm = grad.norm().item()
                delta_norm = self.delta.norm().item()
                
                # 勾配情報をログに記録
                with open(debug_log_path, 'a') as f:
                    grad_vals = grad.detach().cpu().numpy()
                    f.write(f"{iteration}, {loss.item():.6f}, {delta_norm:.6f}, {grad_norm:.6f}, " + 
                        f"{grad_vals[0]:.6f}, {grad_vals[1]:.6f}, {grad_vals[2]:.6f}, " +
                        f"{grad_vals[3]:.6f}, {grad_vals[4]:.6f}, {grad_vals[5]:.6f}\n")
                
                # 詳細な勾配情報を表示
                if iteration % 10 == 0:
                    print(f"  Gradient norm: {grad_norm:.6f}")
                    print(f"  Rot gradient: {grad[:3].detach().cpu().numpy()}")
                    print(f"  Trans gradient: {grad[3:].detach().cpu().numpy()}")
                    rot_grad_norm = grad[:3].norm().item()
                    trans_grad_norm = grad[3:].norm().item()
                    print(f"  Rot/Trans gradient norm ratio: {rot_grad_norm/max(trans_grad_norm, 1e-10):.6f}")
            else:
                print("Warning: No gradient computed!")
            
            # 12. 最適化ステップと学習率の更新
            optimizer.step()
            scheduler.step()
            
            # 13. 履歴の保存（メモリ効率化）
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if save_diagnostics:
                # メモリ効率化: GPUテンソルではなくCPUの浮動小数点値を保存
                if param_history is not None:
                    param_history['T'].append(self.T.detach().cpu().clone())
                
                if grad_history is not None and self.delta.grad is not None:
                    grad_history['delta'].append(self.delta.grad.detach().cpu().clone())
            
            # 14. 更新されたdeltaを元のポーズに適用し、deltaをリセット（値だけ、モーメンタムは保持）
            with torch.no_grad():
                # 回転成分をπ範囲にクランプ（数値安定性のため）
                self.delta.data[:3].clamp_(-math.pi, math.pi)
                
                # T_new = exp(δ) * T
                self.T = self.se3_exp_T(self.delta) @ self.T
                
                # iNeRFスタイル: deltaパラメータを0にリセットするが、Adamのモーメンタムは保持
                self.delta.zero_()
                
                # 数値安定性のためのチェック
                if torch.isnan(self.T).any():
                    print("NaN detected in transformation matrix. Stopping optimization.")
                    break
            
            # 15. 収束判定
            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss
            
            # 16. プログレスバー更新
            if iteration % 10 == 0:
                # self.deltaを使用（deltaではなく）
                delta_norm = self.delta.norm().item()
                rot_delta_norm = self.delta[:3].norm().item()
                trans_delta_norm = self.delta[3:].norm().item()
                ratio = rot_delta_norm / max(trans_delta_norm, 1e-10)
                pbar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'delta': f"{delta_norm:.4f}",
                    'r/t': f"{ratio:.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 輸送行列の可視化（頻度を下げる）
                if iteration % 50 == 0 or iteration == max_iter - 1:
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
            # 4x4行列から回転と並進を抽出
            R_cw = self.T[:3, :3]
            t_cw = self.T[:3, 3]
            
            # 保存
            self.R_cw = R_cw
            self.t_cw = t_cw
            
            # 世界→カメラ変換も保存
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw
            self.R_wc = R_wc
            self.t_wc = t_wc
            
            # 基礎行列を計算して保存
            final_F = self._build_F_from_wc(R_wc, t_wc)
            self.f = final_F
            
            # 既存APIとの互換性のために従来のパラメータも更新
            rvec_numpy, _ = cv2.Rodrigues(R_wc.cpu().numpy())
            self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
            self.tvec = nn.Parameter(t_wc)
            
            # camera-to-world パラメータも更新
            rvec_cw_numpy, _ = cv2.Rodrigues(R_cw.cpu().numpy())
            self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
            self.center = nn.Parameter(t_cw)
            
            # SE3ベクトルとしても保存 (他の最適化器との互換性のため)
            se3_vec = torch.zeros(6, device=self.device)
            rvec_cw = torch.from_numpy(rvec_cw_numpy).to(self.device).float().flatten()
            se3_vec[:3] = rvec_cw
            se3_vec[3:] = t_cw
            self.se3_vec = nn.Parameter(se3_vec)
        
        # ------------------------- 損失プロット -------------------------- #
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_inerf)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_inerf.png"))
        plt.close()

        # ------------------------- 最適化過程描画 -------------------------- #
        if save_diagnostics and param_history is not None and grad_history is not None:
            self.save_optimization_diagnostics_inerf(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )
        
        return loss_history

    def save_optimization_diagnostics_inerf(self, 
                               output_dir: str,
                               loss_history: list,
                               param_history: dict,
                               grad_history: dict) -> None:
        """Save detailed diagnostics about the iNeRF optimization process.
        
        Analyzes and visualizes the optimization process of the SE(3) parameters, including:
        - Loss trajectory
        - Parameter evolution
        - Gradient behavior
        - Convergence analysis
        
        Args:
            output_dir: Directory to save diagnostic files
            loss_history: List of loss values at each iteration
            param_history: Dictionary of parameter histories (contains 'T')
            grad_history: Dictionary of gradient histories corresponding to parameters
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute rotation and translation components from T matrices
        T_history = [T.detach().cpu().numpy() for T in param_history['T']]
        R_history = [T[:3, :3] for T in T_history]
        t_history = [T[:3, 3] for T in T_history]
        
        # Convert rotation matrices to axis-angle representation
        rvec_history = []
        for R in R_history:
            rvec, _ = cv2.Rodrigues(R)
            rvec_history.append(rvec.flatten())
        rvec_history = np.array(rvec_history)
        t_history = np.array(t_history)
        
        # Combine into a parameter history that matches the SE3 format
        param_history_np = {}
        param_history_np['se3_vec'] = np.hstack([rvec_history, t_history])
        
        # Convert delta gradients to numpy arrays
        grad_history_np = {}
        grad_history_np['delta'] = np.array([g.detach().cpu().numpy() if g is not None 
                                            else np.zeros(6) 
                                            for g in grad_history['delta']])
        
        # Number of iterations
        iterations = range(len(loss_history))
        
        # 1. Loss Trajectory Analysis
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot loss changes (derivative) to see stability
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
        
        # 2. SE(3) Parameter Trajectory Analysis
        se3_data = param_history_np['se3_vec']
        
        # Plot all 6 components
        fig = plt.figure(figsize=(15, 8))
        
        # Rotation components
        plt.subplot(211)
        plt.plot(iterations, se3_data[:, 0], 'r-', label='ωx')
        plt.plot(iterations, se3_data[:, 1], 'g-', label='ωy')
        plt.plot(iterations, se3_data[:, 2], 'b-', label='ωz')
        plt.title('Camera Rotation Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value (rad)')
        plt.grid(True)
        plt.legend()
        
        # Translation components
        plt.subplot(212)
        plt.plot(iterations, se3_data[:, 3], 'r-', label='tx')
        plt.plot(iterations, se3_data[:, 4], 'g-', label='ty')
        plt.plot(iterations, se3_data[:, 5], 'b-', label='tz')
        plt.title('Camera Translation Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'camera_trajectory.png'), dpi=150)
        plt.close()
        
        # 3. Delta Gradient Analysis
        grad_data = grad_history_np['delta']
        
        # Gradient magnitude
        grad_magnitude = np.linalg.norm(grad_data, axis=1)
        rot_grad_magnitude = np.linalg.norm(grad_data[:, :3], axis=1)
        trans_grad_magnitude = np.linalg.norm(grad_data[:, 3:], axis=1)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Plot total gradient magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, grad_magnitude, 'k-', linewidth=2, label='Total')
        ax1.plot(iterations, rot_grad_magnitude, 'r-', linewidth=1.5, label='Rotation')
        ax1.plot(iterations, trans_grad_magnitude, 'b-', linewidth=1.5, label='Translation')
        ax1.set_title('Delta Gradient Magnitudes')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')  # Log scale to better see changes
        ax1.grid(True)
        ax1.legend()
        
        # Plot rotation gradient components
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, grad_data[:, 0], 'r-', label='grad_ωx')
        ax2.plot(iterations, grad_data[:, 1], 'g-', label='grad_ωy')
        ax2.plot(iterations, grad_data[:, 2], 'b-', label='grad_ωz')
        ax2.set_title('Rotation Gradient Components')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot translation gradient components
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, grad_data[:, 3], 'r-', label='grad_tx')
        ax3.plot(iterations, grad_data[:, 4], 'g-', label='grad_ty')
        ax3.plot(iterations, grad_data[:, 5], 'b-', label='grad_tz')
        ax3.set_title('Translation Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delta_gradient_analysis.png'), dpi=150)
        plt.close()
        
        # 4. Generate a text report with analysis
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("iNeRF OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("====================================\n\n")
            
            # Loss analysis
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n")
            f.write(f"Number of iterations: {len(loss_history)}\n\n")
            
            # Parameter analysis
            f.write("2. PARAMETER ANALYSIS\n")
            f.write("---------------------\n")
            initial_params = param_history_np['se3_vec'][0]
            final_params = param_history_np['se3_vec'][-1]
            
            f.write("Initial camera parameters (rvec, t):\n")
            f.write(f"  Rotation: [{initial_params[0]:.4f}, {initial_params[1]:.4f}, {initial_params[2]:.4f}]\n")
            f.write(f"  Translation: [{initial_params[3]:.4f}, {initial_params[4]:.4f}, {initial_params[5]:.4f}]\n\n")
            
            f.write("Final camera parameters (rvec, t):\n")
            f.write(f"  Rotation: [{final_params[0]:.4f}, {final_params[1]:.4f}, {final_params[2]:.4f}]\n")
            f.write(f"  Translation: [{final_params[3]:.4f}, {final_params[4]:.4f}, {final_params[5]:.4f}]\n\n")
            
            # Gradient analysis
            f.write("3. GRADIENT ANALYSIS\n")
            f.write("--------------------\n")
            
            avg_grad_magnitude = np.mean(grad_magnitude)
            max_grad_magnitude = np.max(grad_magnitude)
            min_grad_magnitude = np.min(grad_magnitude)
            
            f.write(f"Average gradient magnitude: {avg_grad_magnitude:.6f}\n")
            f.write(f"Maximum gradient magnitude: {max_grad_magnitude:.6f}\n")
            f.write(f"Minimum gradient magnitude: {min_grad_magnitude:.6f}\n\n")
            
            avg_rot_grad = np.mean(rot_grad_magnitude)
            avg_trans_grad = np.mean(trans_grad_magnitude)
            avg_ratio = avg_rot_grad / max(avg_trans_grad, 1e-10)
            
            f.write(f"Average rotation gradient: {avg_rot_grad:.6f}\n")
            f.write(f"Average translation gradient: {avg_trans_grad:.6f}\n")
            f.write(f"Average rotation/translation ratio: {avg_ratio:.6f}\n\n")
            
            # Check for potential issues
            issues = []
            
            # Gradient vanishing check
            if min_grad_magnitude < 1e-6:
                issues.append("Potential gradient vanishing detected.")
                
            # Gradient explosion check    
            if max_grad_magnitude > 1e3:
                issues.append("Potential gradient explosion detected.")
                
            # Rotation/translation ratio imbalance
            if avg_ratio > 100 or avg_ratio < 0.01:
                issues.append(f"Imbalanced rotation/translation gradients (ratio: {avg_ratio:.2f}).")
                
            # Oscillations check
            if len(loss_history) >= 3:
                oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                       if (loss_history[i] > loss_history[i+1] and 
                                           loss_history[i+1] < loss_history[i+2]))
                oscillation_ratio = oscillation_count / (len(loss_history) - 2)
                
                if oscillation_ratio > 0.3:
                    issues.append(f"High oscillation detected ({oscillation_ratio:.2%} of iterations).")
            
            if issues:
                f.write("4. POTENTIAL ISSUES\n")
                f.write("-------------------\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("4. OPTIMIZATION APPEARS STABLE\n")
                f.write("------------------------------\n")
                f.write("No significant optimization issues detected.\n")
        
        print(f"Saved iNeRF optimization diagnostics to {output_dir}")
