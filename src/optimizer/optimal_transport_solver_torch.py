import copy
import os
import sys
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

    
    def _hat(self, v: torch.Tensor) -> torch.Tensor:
        """Skew‑symmetric matrix (hat operator) for a 3‑vector."""
        h = torch.zeros((3, 3), dtype=torch.float32, device=v.device)
        h[0, 1], h[0, 2] = -v[2],  v[1]
        h[1, 0], h[1, 2] =  v[2], -v[0]
        h[2, 0], h[2, 1] = -v[1],  v[0]
        return h

    def rodrigues(self, rvec: torch.Tensor) -> torch.Tensor:
        """SO(3) exponential map with small‑angle safeguard and autograd support.
        
        rvec: (3,) -> rotation vector
        Returns: (3,3) rotation matrix using Rodrigues formula
        """
        # Calculate norm (rotation angle)
        theta = torch.linalg.norm(rvec)
        
        # Small angle approximation for numerical stability
        if theta < 1.0e-6:                               # 1st‑order Taylor
            return torch.eye(3, dtype=torch.float32, device=rvec.device) + self._hat(rvec)
        
        # Normalize to get rotation axis
        r_axis = rvec / theta
        
        # Get skew-symmetric matrix from rotation axis
        K = self._hat(r_axis)
        
        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        return (
            torch.eye(3, dtype=torch.float32, device=rvec.device)  # Identity matrix
            + torch.sin(theta) * K                                 # sin(θ)K term
            + (1.0 - torch.cos(theta)) * (K @ K)                   # (1-cos(θ))K² term
        )

    def _build_F_from_wc(self, R_wc: torch.Tensor, t_wc: torch.Tensor) -> torch.Tensor:
        """R_wc, t_wc から F を構築。(build_f_from_rtの代替関数)
        F = K2^-T [t]_x R K1^-1
        """
        # [t]_x (外積行列)
        tx = torch.zeros((3,3), device=self.device)
        tx[0,1] = -t_wc[2]
        tx[0,2] =  t_wc[1]
        tx[1,0] =  t_wc[2]
        tx[1,2] = -t_wc[0]
        tx[2,0] = -t_wc[1]
        tx[2,1] =  t_wc[0]

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
        cost_matrix = (
            self.lambda_epipolar * sampson_norm + 
            self.lambda_color * color_norm
        )
        
        return cost_matrix

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

        # ----------- Adam → SGD (momentum0.9, weight_decay0) -------------- #
        optimizer = torch.optim.SGD(
            [{'params': self.rvec_cw, 'lr': 1e-3},      # 回転を速め
            {'params': self.center,  'lr': 0.0}],    # 並進を遅め
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
            transport   = self.unbalanced_sinkhorn_algorithm(cost_matrix, rho=0.5,
                                                            max_iter=10000, tol=1e-6)
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

            # π クランプ
            with torch.no_grad():
                theta = torch.linalg.norm(self.rvec_cw)
                if theta > np.pi:
                    self.rvec_cw.mul_(np.pi / theta)

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
