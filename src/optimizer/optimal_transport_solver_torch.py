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
        epsilon: Optional[float] = None,
        rho: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """Simplified Unbalanced Optimal Transport with log-domain stability.
        
        Uses fixed epsilon scaling (2 steps) and log-domain computation only.
        Eliminates complex parameter tuning while maintaining numerical stability.

        Args:
            cost_matrix: Cost matrix of shape (K1, K2)
            epsilon: Entropy regularization (None for auto: 0.08 * median)
            rho: KL regularization (None for auto: 10.0 * epsilon)
            max_iter: Max iterations (fixed)
            tol: Convergence tolerance

        Returns:
            Transport matrix of shape (K1, K2)
        """
        
        # 1. Simple parameter auto-setting
        with torch.no_grad():
            c_median = torch.median(cost_matrix).item()
            if epsilon is None:
                epsilon = max(0.08 * c_median, 1e-3)  # Fixed: 8% of median with minimum
            if rho is None:
                rho = 10.0 * epsilon  # Fixed ratio
        
        # 2. Fixed epsilon scaling (2 steps only)
        epsilons = [epsilon, 0.5 * epsilon]  # Simple 2-step scaling
        
        # 3. Simple iteration with warmstart
        transport = None
        log_u, log_v = None, None
        
        for eps_current in epsilons:
            rho_current = 10.0 * eps_current  # Always proportional
            transport, log_u, log_v = self._sinkhorn_log_simple(
                cost_matrix, eps_current, rho_current, max_iter, tol, log_u, log_v
            )
        
        return transport

    def _sinkhorn_log_simple(
        self,
        cost_matrix: torch.Tensor,
        epsilon: float,
        rho: float,
        max_iter: int,
        tol: float,
        log_u_init: Optional[torch.Tensor] = None,
        log_v_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simplified log-domain Sinkhorn with minimal overhead."""
        
        # Masses with small epsilon for numerical stability
        eps_mass = 1e-8  # Increased from 1e-16 for float32 stability
        alpha = self.alpha1 + eps_mass
        beta = self.alpha2 + eps_mass
        
        # Log kernel
        log_K = -cost_matrix / epsilon
        
        # Initialize dual variables (warmstart support)
        if log_u_init is not None and log_v_init is not None:
            log_u = log_u_init.clone()
            log_v = log_v_init.clone()
        else:
            log_u = torch.zeros_like(alpha)
            log_v = torch.zeros_like(beta)
        
        # Log masses
        log_alpha = torch.log(alpha)
        log_beta = torch.log(beta)
        
        # Tau parameter
        tau = rho / (rho + epsilon)
        
        # Fixed stabilization frequency
        stabilize_freq = 20
        
        for iteration in range(max_iter):
            # Update log(u)
            log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_u = tau * (log_alpha - log_Kv)
            
            # Update log(v)
            log_KTu = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
            log_v = tau * (log_beta - log_KTu)
            
            # Fixed stabilization every 20 iterations
            if iteration > 0 and iteration % stabilize_freq == 0:
                center = (log_u.mean() + log_v.mean()) / 2
                log_u = log_u - center
                log_v = log_v - center
                log_K = log_K + 2 * center  # Factor 2 from exp distribution
            
            # Simple convergence check every 10 iterations
            if iteration % 10 == 0:
                log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
                T = torch.exp(log_T)
                
                # L∞ marginal error only
                row_sums = T.sum(dim=1)
                col_sums = T.sum(dim=0)
                err = max(
                    torch.abs(row_sums - alpha).max().item(),
                    torch.abs(col_sums - beta).max().item()
                )
                
                if err < tol:
                    break
        
        # Final transport matrix
        log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        transport = torch.exp(log_T)
        transport = torch.nan_to_num(transport, nan=0.0, posinf=1e6, neginf=0.0)  # Reduced from 1e10
        
        return transport, log_u, log_v

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
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_SE3
            save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )



    def optimize_with_essential_geoopt(
        self,
        max_iter: int = 5000,
        tol: float = 1e-6,
        lr: float = 3e-3,
        grad_clip: Optional[float] = None,
        save_diagnostics: bool = True,
        diagnostics_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        seed: Optional[int] = None,
        # 新しいSinkhornパラメータ
        sinkhorn_epsilon: Optional[float] = None,
        sinkhorn_rho: Optional[float] = None,
        sinkhorn_max_iter: int = 1000,
        sinkhorn_tol: float = 1e-6,
        use_log_domain: bool = True,
        epsilon_scaling: bool = True,
        scaling_steps: int = 4,
        scaling_factor: float = 0.25,
        k_rho: float = 10.0,
        warmstart: bool = True,
        early_stop: bool = True,
        sinkhorn_verbose: bool = False
    ):
        """
        Camera-2 の姿勢 (R_wc, t̂_wc) を
            (q, t̂) ∈ S³ × S²
        の製品多様体上で内在的に最適化します。四元数表現を使うことでSO(3)制約を
        自然に満たし、数値的安定性を向上させます。
        
        コア最適化ロジックに加え、詳細な診断情報とビジュアライゼーションを提供します。
        新しいシーン適応型Sinkhornアルゴリズムを使用して、段階的ε-スケーリングにより
        粗い対応から精密な対応へと収束させます。
        
        Args:
            max_iter: 最大イテレーション数
            tol: 収束閾値
            lr: 学習率（回転と並進で共通）
            grad_clip: 勾配クリップの閾値（Noneの場合はクリップしない）
            save_diagnostics: 詳細な診断情報を保存するかどうか
            diagnostics_dir: 診断ファイルを保存するディレクトリ
            results_dir: 結果ファイルを保存するディレクトリ
            seed: 乱数シード
            
            # シーン適応型Sinkhornアルゴリズムパラメータ
            sinkhorn_epsilon: エントロピー正則化パラメータ（Noneで自動設定）
            sinkhorn_rho: KL正則化パラメータ（Noneで自動設定）
            sinkhorn_max_iter: 各εレベルでの最大反復数
            sinkhorn_tol: Sinkhorn収束閾値
            use_log_domain: log-domain計算を使用（数値安定性向上）
            epsilon_scaling: ε-スケーリングを有効化（段階的細化）
            scaling_steps: εスケーリングのステップ数
            scaling_factor: 最小ε = scaling_factor * 初期ε
            k_rho: rho = k_rho * epsilon（自動設定時）
            warmstart: スケール間で双対変数を再利用
            early_stop: 品質劣化時の早期停止
            sinkhorn_verbose: Sinkhorn進捗表示
            
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
            T = self.unbalanced_sinkhorn_algorithm(
                cost_matrix=C,
                epsilon=sinkhorn_epsilon,
                rho=sinkhorn_rho,
                max_iter=sinkhorn_max_iter,
                tol=sinkhorn_tol
            )
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
            
            # 詳細な診断情報 - unified diagnostics
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_unified
            save_optimization_diagnostics_unified(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                q_param_history=q_param_history,
                t_param_history=t_param_history,
                s_history=[torch.tensor(1.0) for _ in range(len(loss_history))],  # placeholder scale
                q_grad_history=q_grad_history,
                t_grad_history=t_grad_history,
                s_grad_history=[torch.tensor(0.0) for _ in range(len(loss_history))],  # placeholder
                R_history=R_history,
                E_history=E_history
            )

        print(f"[Geoopt-S³×S²] finished after {it+1} iterations, final loss = {curr_loss:.6f}")
        print(f"Final quaternion norm: {q_final_normalized.norm().item():.6f}")
        print(f"Final R_wc det: {torch.linalg.det(self.R_wc).item():.6f}")
        return loss_history


