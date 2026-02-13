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
    """Optimal Transport Solver for 2D Gaussians"""

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
        # Noise-scale parameters for NLL interpretation
        sigma_epipolar: float = 1.0,
        sigma_color: float = 1.0,
        sigma_cov: float = 1.0,
        noise_model: str = "gaussian",  # one of {"gaussian", "cauchy", "huber"}
        huber_delta: float = 1.0,       # used if noise_model == "huber"
        cauchy_c: float = 1.0,          # used if noise_model == "cauchy"
        lambda_cheirality: float = 0.0,
        cheirality_topk: Optional[int] = 3,
        epipolar_mode: str = "sed",  # one of {"sed", "sampson", "hybrid"}
        hybrid_alpha: float = 0.5,    # when epipolar_mode == "hybrid": alpha in [0,1]
        epi_clip: Optional[float] = None,  # if set, add large cost when Sampson residual exceeds this px threshold
        device: Optional[torch.device] = None,
        normalize_ot_mass: bool = True,
        ot_mass_eps: float = 1e-8,
        # OT marginal mass (visible mass in image region = alpha * S_k)
        ot_mass1: Optional[np.ndarray] = None,  # If None, use gaussians1.alpha
        ot_mass2: Optional[np.ndarray] = None,  # If None, use gaussians2.alpha
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
            normalize_ot_mass (bool): If True, normalize OT masses to sum to 1.
            ot_mass_eps (float): Epsilon for mass normalization safety checks.
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
        self.k1_inv = torch.linalg.inv(self.k1) if self.k1 is not None else None
        self.k2_inv = torch.linalg.inv(self.k2) if self.k2 is not None else None
        self.h: Optional[torch.Tensor] = None
        # Added for Fundamental Matrix
        self.f: Optional[torch.Tensor] = None

        self.epsilon = epsilon
        self.lambda_mean = lambda_mean
        self.lambda_cov = lambda_cov
        self.lambda_color = lambda_color
        self.lambda_epipolar = lambda_epipolar
        self.lambda_cheirality = float(lambda_cheirality)
        self.cheirality_topk = cheirality_topk
        # NLL parameters
        self.sigma_epipolar = float(sigma_epipolar)
        self.sigma_color = float(sigma_color)
        self.sigma_cov = float(sigma_cov)
        self.normalize_ot_mass = bool(normalize_ot_mass)
        self.ot_mass_eps = float(ot_mass_eps)
        self.noise_model = noise_model.lower()
        if self.noise_model not in {"gaussian", "cauchy", "huber"}:
            raise ValueError("noise_model must be one of {'gaussian','cauchy','huber'}")
        self.huber_delta = float(huber_delta)
        self.cauchy_c = float(cauchy_c)
        # Epipolar cost selection
        self.epipolar_mode = epipolar_mode.lower()
        if self.epipolar_mode not in {"sed", "sampson", "hybrid"}:
            raise ValueError(f"Invalid epipolar_mode: {epipolar_mode}. Choose from 'sed', 'sampson', 'hybrid'.")
        self.hybrid_alpha = float(hybrid_alpha)
        if not (0.0 <= self.hybrid_alpha <= 1.0):
            raise ValueError("hybrid_alpha must be in [0, 1].")
        self.epi_clip = epi_clip

        # OT marginal mass (visible mass = alpha * S_k, or alpha if not provided)
        self._ot_mass1_input = ot_mass1  # Store for use in _prepare_gaussians
        self._ot_mass2_input = ot_mass2

        # Gate mask for epsilon calculation (set by compute_cost_matrix_*)
        self._last_gate_mask: Optional[torch.Tensor] = None

        # Actual ε/ρ used in last Sinkhorn call (for full_uot consistency)
        # These are set by unbalanced_sinkhorn_algorithm after final iteration
        self._last_sinkhorn_epsilon: Optional[float] = None
        self._last_sinkhorn_rho: Optional[float] = None

        # Convert Gaussian parameters to torch tensors
        self._prepare_gaussians()
        # Lieクラスのインスタンス化
        self.lie = Lie()
        self.quaternion = Quaternion()

    def _normalize_mass(self, mass: torch.Tensor) -> torch.Tensor:
        """Normalize mass to sum to 1 with safety fallback."""
        mass = torch.clamp(mass, min=0.0)
        total = mass.sum()
        if not torch.isfinite(total) or total <= self.ot_mass_eps:
            return torch.full_like(mass, 1.0 / mass.numel())
        return mass / total

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
        # Use ot_mass if provided (= alpha * S_k), otherwise fall back to alpha
        if self._ot_mass1_input is not None:
            self.alpha1 = torch.as_tensor(
                self._ot_mass1_input, dtype=torch.float32, device=self.device
            )  # Shape: (K1,) - visible mass for OT marginal
        else:
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
        # Use ot_mass if provided (= alpha * S_k), otherwise fall back to alpha
        if self._ot_mass2_input is not None:
            self.alpha2 = torch.as_tensor(
                self._ot_mass2_input, dtype=torch.float32, device=self.device
            )  # Shape: (K2,) - visible mass for OT marginal
        else:
            self.alpha2 = torch.as_tensor(
                self.gaussians2.alpha, dtype=torch.float32, device=self.device
            )  # Shape: (K2,)

    def _build_F_from_wc(self, R_wc: torch.Tensor, t_wc: torch.Tensor) -> torch.Tensor:
        """R_wc, t_wc から F を構築。Lieクラスのskew_symmetricを使用。"""
        if self.k1_inv is None or self.k2_inv is None:
            raise ValueError("Camera intrinsics must be provided to build the fundamental matrix.")
        tx = self.lie.skew_symmetric(t_wc)
        E = tx @ R_wc
        K2_inv_T = self.k2_inv.transpose(0, 1)
        F = K2_inv_T @ E @ self.k1_inv
        assert not torch.isnan(F).any(), "NaN in fundamental matrix"
        return F

    def unbalanced_sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        epsilon: Optional[float] = None,
        rho: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        record_mass: bool = False,
        dustbin_cost: Optional[float] = None,
        dustbin_mass: float = 1.0,
        gate_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unbalanced OT (log-domain) with optional mass diagnostics and dustbin.

        Args:
            cost_matrix: Cost matrix (K1,K2)
            epsilon: Entropy weight (None → 0.08 * median of valid entries)
            rho: KL weight (None → 10 * epsilon)  ※渡した rho をそのまま使う
            record_mass: If True, also return (row_sum, col_sum)
            dustbin_cost: If set, append a no-match row/col with this constant cost.
            dustbin_mass: Relative mass assigned to each dustbin marginal.
            gate_mask: Boolean mask (True=valid) to exclude gated entries from median.
        """
        # Reset actual ε/ρ to avoid stale values on failed calls.
        self._last_sinkhorn_epsilon = None
        self._last_sinkhorn_rho = None
        with torch.no_grad():
            if dustbin_cost is not None:
                total_mass = max(float(self.alpha1.sum().item()), float(self.alpha2.sum().item()))
                db_mass = float(dustbin_mass) * total_mass
                a = torch.cat(
                    [
                        self.alpha1.clone(),
                        torch.tensor([db_mass], device=self.device, dtype=self.alpha1.dtype),
                    ]
                )
                b = torch.cat(
                    [
                        self.alpha2.clone(),
                        torch.tensor([db_mass], device=self.device, dtype=self.alpha2.dtype),
                    ]
                )
                pad_row = torch.full(
                    (cost_matrix.shape[0], 1),
                    dustbin_cost,
                    device=self.device,
                    dtype=cost_matrix.dtype,
                )
                pad_col = torch.full(
                    (1, cost_matrix.shape[1] + 1),
                    dustbin_cost,
                    device=self.device,
                    dtype=cost_matrix.dtype,
                )
                cost_matrix = torch.cat([torch.cat([cost_matrix, pad_row], dim=1), pad_col], dim=0)
            else:
                a = self.alpha1.clone()
                b = self.alpha2.clone()

            # Compute median excluding gated (invalid) entries if mask is provided
            if gate_mask is not None and epsilon is None:
                valid_costs = cost_matrix[gate_mask]
                if valid_costs.numel() > 0:
                    c_median = torch.median(valid_costs).item()
                else:
                    # Fallback to full median if no valid entries
                    c_median = torch.median(cost_matrix).item()
            else:
                c_median = torch.median(cost_matrix).item()
            if epsilon is None:
                epsilon = max(0.08 * c_median, 1e-3)
            if rho is None:
                rho = 10.0 * epsilon
            base_eps = epsilon
            base_rho = rho

        epsilons = [epsilon, 0.5 * epsilon]
        transport = None
        log_u, log_v = None, None

        for eps_current in epsilons:
            rho_current = base_rho * (eps_current / base_eps)
            transport, log_u, log_v = self._sinkhorn_log_simple(
                cost_matrix, eps_current, rho_current, max_iter, tol, log_u, log_v, a, b
            )

        # Store actual ε/ρ used in final iteration for full_uot calculation
        # This is critical: the outer optimization must use the same ε/ρ as Sinkhorn
        self._last_sinkhorn_epsilon = eps_current
        self._last_sinkhorn_rho = rho_current

        if record_mass:
            row_sum = transport.sum(dim=1)
            col_sum = transport.sum(dim=0)
            return transport, (row_sum, col_sum)
        return transport, None

    def _apply_gate_mask(
        self, cost: torch.Tensor, gate: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a boolean gate (allowed=True) to cost.
        Ensures each row/col has at least one allowed entry by reopening the min-cost spot if needed.

        Returns:
            gated_cost: Cost matrix with invalid entries set to high penalty
            updated_gate: Gate mask after fallback corrections (True=valid)
        """
        gate = gate.clone()
        # row fallback
        row_all_false = gate.sum(dim=1) == 0
        if row_all_false.any():
            idx = torch.argmin(cost[row_all_false], dim=1)
            gate[row_all_false, idx] = True
        # col fallback
        col_all_false = gate.sum(dim=0) == 0
        if col_all_false.any():
            idx = torch.argmin(cost[:, col_all_false], dim=0)
            gate[idx, col_all_false] = True
        gated_cost = torch.where(gate, cost, cost.max().detach() + 1e6)
        return gated_cost, gate

    def _sinkhorn_log_simple(
        self,
        cost_matrix: torch.Tensor,
        epsilon: float,
        rho: float,
        max_iter: int,
        tol: float,
        log_u_init: Optional[torch.Tensor] = None,
        log_v_init: Optional[torch.Tensor] = None,
        alpha_override: Optional[torch.Tensor] = None,
        beta_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simplified log-domain Sinkhorn with minimal overhead."""
        
        # Masses with small epsilon for numerical stability
        eps_mass = 1e-8  # Increased from 1e-16 for float32 stability
        if alpha_override is None:
            alpha = self.alpha1
        else:
            alpha = alpha_override
        if beta_override is None:
            beta = self.alpha2
        else:
            beta = beta_override
        if self.normalize_ot_mass:
            alpha = self._normalize_mass(alpha)
            beta = self._normalize_mass(beta)
        alpha = alpha + eps_mass
        beta = beta + eps_mass
        
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
        log_u_prev: Optional[torch.Tensor] = None
        log_v_prev: Optional[torch.Tensor] = None

        for iteration in range(max_iter):
            # Update log(u)
            log_Kv = torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_u = tau * (log_alpha - log_Kv)
            
            # Update log(v)
            log_KTu = torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
            log_v = tau * (log_beta - log_KTu)
            
            # Fixed stabilization every 20 iterations
            if iteration > 0 and iteration % stabilize_freq == 0:
                offset = log_u.mean()
                log_u = log_u - offset
                log_v = log_v + offset
            
            # Simple convergence check every 10 iterations
            if iteration % 10 == 0:
                if log_u_prev is not None and log_v_prev is not None:
                    max_du = (log_u - log_u_prev).abs().max()
                    max_dv = (log_v - log_v_prev).abs().max()
                    if torch.max(max_du, max_dv).item() < tol:
                        break
                log_u_prev = log_u.clone().detach()
                log_v_prev = log_v.clone().detach()

        # Final transport matrix
        log_T = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        transport = torch.exp(log_T)
        transport = torch.nan_to_num(transport, nan=0.0, posinf=0.0, neginf=0.0)

        return transport, log_u, log_v

    def _make_cov_matrices(self, scales: torch.Tensor,
                        rotations: torch.Tensor) -> torch.Tensor:
        """"""
        cos_r, sin_r = torch.cos(rotations), torch.sin(rotations)
        rot = torch.stack(
            [torch.stack([cos_r, -sin_r], 1),
            torch.stack([sin_r,  cos_r], 1)],
            2)                                   # (K,2,2)
        scale_mat = torch.diag_embed(scales * scales)
        return rot @ scale_mat @ rot.transpose(1, 2)   # (K,2,2)
    
    def _matrix_sqrt_spd2x2(self, matrices: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Batch square-root for 2x2 SPD matrices with symmetrization and jitter."""
        # Ensure symmetric
        matrices = 0.5 * (matrices + matrices.transpose(-2, -1))
        # Add small jitter for numerical stability
        eye = torch.eye(2, device=matrices.device, dtype=matrices.dtype)
        matrices = matrices + eps * eye
        # Flatten batch dims
        batch_shape = matrices.shape[:-2]
        mats2 = matrices.reshape(-1, 2, 2)
        # EVD
        evals, evecs = torch.linalg.eigh(mats2)
        evals = torch.clamp(evals, min=eps).sqrt()
        sqrt_diag = torch.diag_embed(evals)
        sqrt_mats = evecs @ sqrt_diag @ evecs.transpose(-2, -1)
        return sqrt_mats.reshape(*batch_shape, 2, 2)

    def _bures_wasserstein_cov_dist(self, cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
        """Pairwise Bures (W2^2) distance between 2x2 SPD covariances.

        Args:
            cov1: (K1,2,2)
            cov2: (K2,2,2)
        Returns:
            (K1,K2) matrix with d_Bures^2(cov1[i], cov2[j])
        """
        # Precompute sqrt of cov1
        sqrt1 = self._matrix_sqrt_spd2x2(cov1)                  # (K1,2,2)
        # Broadcast to pairwise
        cov1_exp = cov1.unsqueeze(1)                             # (K1,1,2,2)
        cov2_exp = cov2.unsqueeze(0)                             # (1,K2,2,2)
        sqrt1_exp = sqrt1.unsqueeze(1)                           # (K1,1,2,2)
        # Intermediate and its sqrt
        inter = sqrt1_exp @ cov2_exp @ sqrt1_exp                 # (K1,K2,2,2)
        sqrt_inter = self._matrix_sqrt_spd2x2(inter)             # (K1,K2,2,2)
        # Trace of A + B - 2*sqrt(sqrt(A) B sqrt(A))
        diff = cov1_exp + cov2_exp - 2.0 * sqrt_inter
        # Enforce symmetry before diagonal extraction
        diff = 0.5 * (diff + diff.transpose(-2, -1))
        tr = torch.diagonal(diff, dim1=-2, dim2=-1).sum(-1)
        tr = torch.clamp(tr, min=0.0)
        return tr  # (K1,K2)

    def _nll_from_squared(self, sq: torch.Tensor, sigma: float, *, model: Optional[str] = None) -> torch.Tensor:
        """Convert squared residuals into a negative log-likelihood-like penalty.

        - gaussian: rho(s) = s / (sigma^2)
        - cauchy:  rho(s) = c^2 * log(1 + s / (c^2 * sigma^2))
        - huber:   residual r = sqrt(s); rho(r) as Huber with delta; scale by sigma
        """
        eps = 1e-12
        model = (model or self.noise_model).lower()
        if model == "gaussian":
            return sq / (sigma * sigma + eps)
        if model == "cauchy":
            c2 = self.cauchy_c * self.cauchy_c + eps
            return c2 * torch.log1p(sq / (c2 * (sigma * sigma + eps)))
        # huber
        r = torch.sqrt(torch.clamp(sq, min=0.0)) / (sigma + eps)
        delta = self.huber_delta
        mask = r <= delta
        quad = 0.5 * (r * r)
        lin = delta * (r - 0.5 * delta)
        return torch.where(mask, quad, lin)

    def _cheirality_loss(
        self,
        R_wc: torch.Tensor,
        t_wc: torch.Tensor,
        transport: torch.Tensor,
        topk: Optional[int] = 3,
    ) -> torch.Tensor:
        if self.k1_inv is None or self.k2_inv is None:
            return torch.tensor(0.0, device=self.device)

        T = transport
        if T.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        if topk is not None and topk > 0:
            k = min(topk, T.size(1))
            idx = torch.topk(T, k=k, dim=1).indices
            rows = torch.arange(T.size(0), device=self.device).unsqueeze(1).expand_as(idx)
            pairs = torch.stack([rows.reshape(-1), idx.reshape(-1)], dim=1)
        else:
            pairs = torch.nonzero(T > 0, as_tuple=False)

        if pairs.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        ones1 = torch.ones(self.means1.size(0), 1, device=self.device)
        ones2 = torch.ones(self.means2.size(0), 1, device=self.device)
        x1 = torch.cat([self.means1, ones1], dim=1)
        x2 = torch.cat([self.means2, ones2], dim=1)

        x1n = (self.k1_inv @ x1.T).T
        x2n = (self.k2_inv @ x2.T).T

        x1p = x1n[pairs[:, 0]]
        x2p = x2n[pairs[:, 1]]

        I = torch.eye(3, device=self.device, dtype=torch.float32)
        P1 = torch.cat([I, torch.zeros(3, 1, device=self.device)], dim=1)
        P2 = torch.cat([R_wc, t_wc.view(3, 1)], dim=1)

        num_pairs = x1p.size(0)
        A = torch.zeros(num_pairs, 4, 4, device=self.device)
        A[:, 0, :] = x1p[:, 0:1] * P1[2, :] - P1[0, :]
        A[:, 1, :] = x1p[:, 1:2] * P1[2, :] - P1[1, :]
        A[:, 2, :] = x2p[:, 0:1] * P2[2, :] - P2[0, :]
        A[:, 3, :] = x2p[:, 1:2] * P2[2, :] - P2[1, :]

        _, _, Vh = torch.linalg.svd(A)
        X_h = Vh[:, -1, :]
        X = X_h[:, :3] / (X_h[:, 3:4] + 1e-12)

        Z1 = X[:, 2]
        X_cam2 = (R_wc @ X.T + t_wc.view(3, 1)).T
        Z2 = X_cam2[:, 2]

        weights = T[pairs[:, 0], pairs[:, 1]].detach()
        loss = (torch.nn.functional.relu(-Z1) + torch.nn.functional.relu(-Z2)) * weights
        return loss.mean()

    def compute_cost_matrix(self, F: torch.Tensor) -> torch.Tensor:
        """Dispatch to selected epipolar cost.

        Modes:
        - "sed":     Symmetric epipolar distance (+ shape) + color
        - "sampson": Sampson distance (+ shape) + color
        - "hybrid":  alpha * Sampson + (1-alpha) * SED (same lambda weights)
        """
        if self.epipolar_mode == "sed":
            return self.compute_cost_matrix_fundamental(F)
        elif self.epipolar_mode == "sampson":
            return self.compute_cost_matrix_fundamental_sampson(F)
        else:
            c_samp = self.compute_cost_matrix_fundamental_sampson(F)
            c_sed = self.compute_cost_matrix_fundamental(F)
            return self.hybrid_alpha * c_samp + (1.0 - self.hybrid_alpha) * c_sed


    def compute_cost_matrix_fundamental(self, F: torch.Tensor) -> torch.Tensor:
        """
        期待二乗距離版  "対称エピポーラ距離" + 色差
        ------------------------------------------------
        L_ij = (d_12^2 + u1_i) + (d_21^2 + u2_j)
        ------------------------------------------------
        """

        k1, k2 = self.means1.size(0), self.means2.size(0)

        # -------- ① 同次座標 --------
        # means は (x,y) 保存を前提
        ones1 = torch.ones(k1, 1, device=self.device)
        ones2 = torch.ones(k2, 1, device=self.device)
        p1_h = torch.cat([self.means1, ones1], 1)   # (K1,3)
        p2_h = torch.cat([self.means2, ones2], 1)   # (K2,3)

        # -------- ② エピポーラ線 --------
        # Epipolar constraint: x2^T F x1 = 0
        # Line in image1 (from x2): l1 = F^T @ x2
        # Line in image2 (from x1): l2 = F @ x1
        l1 = (F.T @ p2_h.T).T            # (K2,3) lines in image1 for points in image2
        l2 = (F   @ p1_h.T).T            # (K1,3) lines in image2 for points in image1

        n1 = l1[:, :2]                              # (K2,2)
        eps = 1e-9
        n1_norm = n1.norm(dim=1, keepdim=True) + eps
        n1_unit = n1 / n1_norm

        n2 = l2[:, :2]                              # (K1,2)
        n2_norm = n2.norm(dim=1, keepdim=True) + eps
        n2_unit = n2 / n2_norm

        # -------- ③ 点⇔線  "符号付き" 距離 --------
        # d_12(i,j): p1_i → l1_j (point i in image1 to line j in image1, line from p2_j)
        # d_21(i,j): p2_j → l2_i (point j in image2 to line i in image2, line from p1_i)
        # For correct correspondence (i,j), both distances should be ~0
        num_12 = p1_h @ l1.T                                           # (K1,K2)
        num_21 = (p2_h @ l2.T).T                                       # (K1,K2)

        # -------- ④ "距離²＋分散" へ置換 --------
        dist_sq_sum = (
            (num_12 ** 2) / (n1_norm.T ** 2)
            + (num_21 ** 2) / (n2_norm ** 2)
        )

        cov1 = self._make_cov_matrices(self.scales1, self.rotations1)   # (K1,2,2)
        cov2 = self._make_cov_matrices(self.scales2, self.rotations2)   # (K2,2,2)

        cov1_exp = cov1.unsqueeze(1).expand(-1, k2, -1, -1)             # (K1,K2,2,2)
        cov2_exp = cov2.unsqueeze(0).expand(k1, -1, -1, -1)             # (K1,K2,2,2)
        n1_exp = n1_unit.unsqueeze(0).expand(k1, -1, -1)                # (K1,K2,2)
        n2_exp = n2_unit.unsqueeze(1).expand(-1, k2, -1)                # (K1,K2,2)

        u1 = torch.einsum('...i,...ij,...j->...', n1_exp, cov1_exp, n1_exp)
        u2 = torch.einsum('...i,...ij,...j->...', n2_exp, cov2_exp, n2_exp)

        epi_with_shape = dist_sq_sum + u1 + u2                          # (K1,K2)
        if self.epi_clip is not None:
            tau_sq = float(self.epi_clip) ** 2
            gate = dist_sq_sum <= tau_sq  # ゲートは純粋な幾何残差に対して適用
            epi_with_shape, updated_gate = self._apply_gate_mask(epi_with_shape, gate)
            self._last_gate_mask = updated_gate  # Store updated gate for epsilon calculation
        else:
            self._last_gate_mask = None

        # -------- ⑤ 形状の類似度（Bures/Wasserstein距離） --------
        cov_dist = self._bures_wasserstein_cov_dist(cov1, cov2)    # (K1,K2)

        # -------- ⑥ 色差 --------
        rgb_diff   = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)    # (K1,K2,3)
        color_dist = (rgb_diff ** 2).sum(2)                             # (K1,K2)

        # -------- ⑦ 正規化 --------
        # Negative log-likelihood style penalties with noise scales
        epi_nll   = self._nll_from_squared(epi_with_shape, self.sigma_epipolar)
        color_nll = self._nll_from_squared(color_dist,   self.sigma_color)
        cov_nll   = self._nll_from_squared(cov_dist,     self.sigma_cov)

        # -------- ⑧ コスト合成 --------
        cost = (
            self.lambda_epipolar * epi_nll
            + self.lambda_color  * color_nll
            + (self.lambda_cov   * cov_nll if self.lambda_cov > 0 else 0.0)
        )

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
        # 同次座標変換（means は (x,y) 保存を前提）
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
        eps = 1e-9
        denom = (Fp1[:2]**2).sum(0).view(-1, 1) + (FTp2[:2]**2).sum(0).view(1, -1) 
        
        # サンプソン距離計算
        sampson = num / (denom + eps)  # (K1,K2)
        if self.epi_clip is not None:
            tau_sq = float(self.epi_clip) ** 2
            gate = sampson <= tau_sq
            sampson, updated_gate = self._apply_gate_mask(sampson, gate)
            self._last_gate_mask = updated_gate  # Store updated gate for epsilon calculation
        else:
            self._last_gate_mask = None
        
        # === 2. ガウス分布の形状を考慮 ===
        # 2.1 共分散行列の作成
        # 共分散行列の計算
        cov1 = self._make_cov_matrices(self.scales1, self.rotations1)  # (K1,2,2)
        cov2 = self._make_cov_matrices(self.scales2, self.rotations2)  # (K2,2,2)
        
        # 2.2 エピポーラ線の法線ベクトル計算（正規化）
        # ℓ = (a,b,c) の法線は n = (a,b)/||(a,b)||
        n1 = Fp1[:2].T  # 画像2上のエピポーラ線の法線方向 (K1,2)
        n1 = n1 / (n1.norm(dim=1, keepdim=True) + eps)

        n2 = FTp2[:2].T  # 画像1上のエピポーラ線の法線方向 (K2,2)
        n2 = n2 / (n2.norm(dim=1, keepdim=True) + eps)
        
        cov1_exp = cov1.unsqueeze(1).expand(-1, k2, -1, -1)  # (K1,K2,2,2)
        cov2_exp = cov2.unsqueeze(0).expand(k1, -1, -1, -1)  # (K1,K2,2,2)
        n2_exp = n2.unsqueeze(0).expand(k1, -1, -1)          # (K1,K2,2)
        n1_exp = n1.unsqueeze(1).expand(-1, k2, -1)          # (K1,K2,2)
        
        u1 = torch.einsum('...i,...ij,...j->...', n2_exp, cov1_exp, n2_exp)
        u2 = torch.einsum('...i,...ij,...j->...', n1_exp, cov2_exp, n1_exp)
        
        # uncertainty_factor = 1.0 + u1 + u2
        uncertainty_factor = 1.0 # DEBUG: Disable shape uncertainty
        sampson_with_shape = sampson / (uncertainty_factor + 1e-12)
        
        # === 3. 形状の類似度（Bures/Wasserstein距離） ===
        cov_dist = self._bures_wasserstein_cov_dist(cov1, cov2)        # (K1,K2)

        # === 4. 色差分の計算 ===
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = (color_diff ** 2).sum(dim=2)  # (K1,K2)
        
        # === 5. NLL化と最終コスト計算 ===
        sampson_nll = self._nll_from_squared(sampson_with_shape, self.sigma_epipolar)
        color_nll   = self._nll_from_squared(d_color,            self.sigma_color)
        cov_nll     = self._nll_from_squared(cov_dist,           self.sigma_cov)

        cost = (
            self.lambda_epipolar * sampson_nll + 
            self.lambda_color * color_nll +
            (self.lambda_cov * cov_nll if self.lambda_cov > 0 else 0.0)
        )
        
        return cost

    # (Legacy) compute_cost_matrix_fundamental_original removed: Frobenius-based covariance term deprecated.



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
                        grad_clip=0.1, seed=None,
                        differentiable_transport=True,
                        log_cheirality: bool = False,
                        # Fixed Sinkhorn parameters (Step E compatibility)
                        sinkhorn_epsilon: Optional[float] = None,
                        sinkhorn_rho: Optional[float] = None,
                        # Score function selection
                        score_type: str = "loss",  # "loss", "avg_cost", "mass_aware", "full_uot"
                        lambda_kl: float = 0.1,  # for mass_aware score
                        # Epsilon annealing (Step G: collapse prevention)
                        epsilon_annealing: bool = False,
                        epsilon_start: float = 0.2,  # Initial high epsilon
                        epsilon_end: float = 0.05,   # Final low epsilon
                        anneal_steps: int = 100,     # Steps to anneal from start to end
                        # Step II: R/t separation for drift diagnosis
                        optimize_mode: str = "both",  # "both", "rotation_only", "translation_only"
                        ):
        """Optimize camera pose using Lie algebra SE(3) representation with separate rotation/translation.

        Args:
            optimize_mode: Which parameters to optimize:
                - "both": Optimize rotation and translation (default)
                - "rotation_only": Only optimize rotation, keep translation fixed
                - "translation_only": Only optimize translation, keep rotation fixed
            log_cheirality: If True, log raw cheirality loss even when lambda_cheirality is 0.
        """
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
        # Step II: Configure optimizer based on optimize_mode
        if optimize_mode == "rotation_only":
            # Only optimize rotation, fix translation
            self.rot_vec.requires_grad = True
            self.trans_vec.requires_grad = False
            optimizer = torch.optim.SGD([
                {'params': self.rot_vec, 'lr': rot_lr, 'momentum': momentum, 'nesterov': True},
            ])
        elif optimize_mode == "translation_only":
            # Only optimize translation, fix rotation
            self.rot_vec.requires_grad = False
            self.trans_vec.requires_grad = True
            optimizer = torch.optim.SGD([
                {'params': self.trans_vec, 'lr': trans_lr, 'momentum': momentum, 'nesterov': True},
            ])
        elif optimize_mode == "both":
            # Default: optimize both
            self.rot_vec.requires_grad = True
            self.trans_vec.requires_grad = True
            optimizer = torch.optim.SGD([
                {'params': self.rot_vec, 'lr': rot_lr, 'momentum': momentum, 'nesterov': True},
                {'params': self.trans_vec, 'lr': trans_lr, 'momentum': momentum, 'nesterov': True}
            ])
        else:
            raise ValueError(f"Unknown optimize_mode: {optimize_mode}. Choose from 'both', 'rotation_only', 'translation_only'.")
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))

        prev_loss_val = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing SE(3)", leave=True)

        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug.log")
        with open(debug_log_path, 'w') as f:
            f.write(
                "Iteration, Loss, Transport_Cost, Cheirality_Loss, T_sum, top1_mean, "
                "Rot_Grad_Norm, Trans_Grad_Norm, Rot_x, Rot_y, Rot_z, "
                "Trans_x, Trans_y, Trans_z\n"
            )

        for iteration in pbar:
            optimizer.zero_grad()

            # --- Epsilon annealing (Step G: collapse prevention) ---
            if epsilon_annealing:
                if iteration < anneal_steps:
                    # Linear annealing from epsilon_start to epsilon_end
                    t = iteration / anneal_steps
                    current_epsilon = epsilon_start * (1 - t) + epsilon_end * t
                else:
                    current_epsilon = epsilon_end
                # Keep rho proportional to epsilon
                current_rho = (sinkhorn_rho / sinkhorn_epsilon * current_epsilon
                              if sinkhorn_epsilon and sinkhorn_rho else 10.0 * current_epsilon)
            else:
                current_epsilon = sinkhorn_epsilon
                current_rho = sinkhorn_rho

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
            cost_matrix = self.compute_cost_matrix(F)

            # Differentiable transport context
            context = torch.enable_grad() if differentiable_transport else torch.no_grad()
            with context:
                transport, _ = self.unbalanced_sinkhorn_algorithm(
                    cost_matrix,
                    epsilon=current_epsilon,
                    rho=current_rho,
                    gate_mask=self._last_gate_mask
                )
                if not differentiable_transport:
                    transport = transport.detach()

            # Compute score based on selected score_type
            transport_cost = torch.sum(transport * cost_matrix)
            T_sum = transport.sum()

            if score_type == "loss":
                # Original loss (vulnerable to collapse)
                loss = transport_cost
            elif score_type == "avg_cost":
                # Normalized by mass
                loss = transport_cost / (T_sum + 1e-10)
            elif score_type == "mass_aware":
                # avg_cost + lambda * KL
                avg_cost = transport_cost / (T_sum + 1e-10)
                a = self.alpha1 / self.alpha1.sum()
                b = self.alpha2 / self.alpha2.sum()
                row_sum = transport.sum(dim=1)
                col_sum = transport.sum(dim=0)
                # KL divergences (using safe log)
                eps_kl = 1e-10
                KL_row = (row_sum * torch.log((row_sum + eps_kl) / (a + eps_kl)) - row_sum + a).sum()
                KL_col = (col_sum * torch.log((col_sum + eps_kl) / (b + eps_kl)) - col_sum + b).sum()
                loss = avg_cost + lambda_kl * (KL_row + KL_col)
            elif score_type == "full_uot":
                # Full UOT objective (matches Sinkhorn)
                # CRITICAL: Use the ACTUAL ε/ρ from Sinkhorn's final iteration
                # The Sinkhorn algorithm internally uses ε-scaling (e.g., [ε, 0.5ε])
                # so we must use the stored values, not the input parameters
                a = self.alpha1 / self.alpha1.sum()
                b = self.alpha2 / self.alpha2.sum()
                row_sum = transport.sum(dim=1)
                col_sum = transport.sum(dim=0)
                eps_kl = 1e-10
                KL_row = (row_sum * torch.log((row_sum + eps_kl) / (a + eps_kl)) - row_sum + a).sum()
                KL_col = (col_sum * torch.log((col_sum + eps_kl) / (b + eps_kl)) - col_sum + b).sum()
                entropy = -(transport * torch.log(transport + eps_kl)).sum()
                # Use ACTUAL ε/ρ from Sinkhorn (stored after final iteration)
                eps_actual = self._last_sinkhorn_epsilon
                rho_actual = self._last_sinkhorn_rho
                if eps_actual is None or rho_actual is None:
                    raise RuntimeError(
                        "full_uot requires unbalanced_sinkhorn_algorithm() to "
                        "run successfully before computing the score."
                    )
                # UOT objective: <T,C> + ρ*(KL_row + KL_col) + ε*sum(T*log(T) - T)
                entropic = -entropy - T_sum  # = sum(T*log(T) - T)
                loss = transport_cost + rho_actual * (KL_row + KL_col) + eps_actual * entropic
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

            if self.lambda_cheirality > 0.0:
                cheirality_raw = self._cheirality_loss(
                    R_wc, t_wc, transport.detach(), self.cheirality_topk
                )
                loss = loss + self.lambda_cheirality * cheirality_raw
            elif log_cheirality:
                with torch.no_grad():
                    cheirality_raw = self._cheirality_loss(
                        R_wc, t_wc, transport.detach(), self.cheirality_topk
                    )
            else:
                cheirality_raw = torch.tensor(0.0, device=self.device)

            # Transport diagnostics for collapse detection
            with torch.no_grad():
                T_sum = transport.sum().item()
                row_sum = transport.sum(dim=1)
                col_sum = transport.sum(dim=0)
                top1_mean = transport.max(dim=1).values.mean().item()

            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  Rot params: {self.rot_vec.data}")
                print(f"  Trans params: {self.trans_vec.data}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  T.sum(): {T_sum:.4f} (collapse if decreasing!)")
                print(f"  row_sum: mean={row_sum.mean().item():.4f}, min={row_sum.min().item():.4f}, max={row_sum.max().item():.4f}")
                print(f"  col_sum: mean={col_sum.mean().item():.4f}, min={col_sum.min().item():.4f}, max={col_sum.max().item():.4f}")
                print(f"  top1_mean: {top1_mean:.4f}")
                print(f"  Rot LR: {rot_lr:.6e}, Trans LR: {trans_lr:.6e}")
            
            loss.backward()
            
            # 勾配クリッピング
            # torch.nn.utils.clip_grad_norm_([self.rot_vec, self.trans_vec], max_norm=grad_clip)
            
            rot_grad = self.rot_vec.grad
            trans_grad = self.trans_vec.grad
            rot_grad_norm = rot_grad.norm().item() if rot_grad is not None else 0.0
            trans_grad_norm = trans_grad.norm().item() if trans_grad is not None else 0.0

            with open(debug_log_path, 'a') as f:
                rot_vals = rot_grad.detach().cpu().numpy() if rot_grad is not None else np.zeros(3)
                trans_vals = trans_grad.detach().cpu().numpy() if trans_grad is not None else np.zeros(3)
                f.write(
                    f"{iteration}, {loss.item():.6f}, {transport_cost.item():.6f}, "
                    f"{cheirality_raw.item():.6f}, {T_sum:.6f}, {top1_mean:.6f}, "
                    f"{rot_grad_norm:.6f}, {trans_grad_norm:.6f}, "
                    f"{rot_vals[0]:.6f}, {rot_vals[1]:.6f}, {rot_vals[2]:.6f}, "
                    f"{trans_vals[0]:.6f}, {trans_vals[1]:.6f}, {trans_vals[2]:.6f}\n"
                )

            if iteration % 10 == 0:
                print(f"  Rot gradient norm: {rot_grad_norm:.6f}")
                print(f"  Trans gradient norm: {trans_grad_norm:.6f}")
                print(f"  Rot/Trans gradient norm ratio: {rot_grad_norm/max(trans_grad_norm, 1e-10):.6f}")
                if rot_grad is None or trans_grad is None:
                    print("  Note: One gradient is None due to optimize_mode.")

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
        
        return loss_history



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
        sinkhorn_verbose: bool = False,
        differentiable_transport: bool = True,
        # Step III: Score type support (same as SE3 optimizer)
        score_type: str = "avg_cost",  # "loss", "avg_cost", "mass_aware", "full_uot"
        lambda_kl: float = 0.1,  # for mass_aware score
        # Step III: R/t separation for comparison with SE3
        optimize_mode: str = "both",  # "both", "rotation_only", "translation_only"
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
            differentiable_transport: Sinkhornの勾配を計算するかどうか
            score_type: スコア関数の種類
                - "loss": 単純な <T,C>
                - "avg_cost": <T,C> / T.sum() (collapse-robust)
                - "mass_aware": avg_cost + λ * KL
                - "full_uot": UOT目的関数全体
            lambda_kl: mass_aware スコアの KL 重み
            optimize_mode: 最適化モード (Step III: R/t 分離テスト用)
                - "both": R と t を同時に最適化
                - "rotation_only": R のみ最適化（t は固定）
                - "translation_only": t のみ最適化（R は固定）

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
        gamma = (0.25) ** (1.0 / max_iter)
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
            f.write("Iteration,Loss,T_sum,top1_mean,q_Grad_Norm,t_Grad_Norm,q_t_Grad_Ratio,Quaternion_Norm\n")

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
            C = self.compute_cost_matrix(F)

            # Differentiable transport context
            context = torch.enable_grad() if differentiable_transport else torch.no_grad()
            with context:
                T, _ = self.unbalanced_sinkhorn_algorithm(
                    cost_matrix=C if differentiable_transport else C.detach(),
                    epsilon=sinkhorn_epsilon,
                    rho=sinkhorn_rho,
                    max_iter=sinkhorn_max_iter,
                    tol=sinkhorn_tol,
                    gate_mask=self._last_gate_mask,
                )
                if not differentiable_transport:
                    T = T.detach()

            # Step III: Score-type based loss computation (same logic as SE3 optimizer)
            transport_cost = (T * C).sum()
            T_sum = T.sum()

            if score_type == "loss":
                loss = transport_cost
            elif score_type == "avg_cost":
                loss = transport_cost / (T_sum + 1e-10)
            elif score_type == "mass_aware":
                avg_cost = transport_cost / (T_sum + 1e-10)
                a = self.ot_mass1 if self.ot_mass1 is not None else torch.ones(T.shape[0], device=T.device) / T.shape[0]
                b = self.ot_mass2 if self.ot_mass2 is not None else torch.ones(T.shape[1], device=T.device) / T.shape[1]
                row_marginal = T.sum(dim=1)
                col_marginal = T.sum(dim=0)
                KL_row = (row_marginal * (torch.log(row_marginal + 1e-10) - torch.log(a + 1e-10)) - row_marginal + a).sum()
                KL_col = (col_marginal * (torch.log(col_marginal + 1e-10) - torch.log(b + 1e-10)) - col_marginal + b).sum()
                loss = avg_cost + lambda_kl * (KL_row + KL_col)
            elif score_type == "full_uot":
                # Use ACTUAL ε/ρ from Sinkhorn
                eps_actual = self._last_sinkhorn_epsilon
                rho_actual = self._last_sinkhorn_rho
                if eps_actual is None or rho_actual is None:
                    raise RuntimeError("full_uot requires Sinkhorn to run first")
                a = self.ot_mass1 if self.ot_mass1 is not None else torch.ones(T.shape[0], device=T.device) / T.shape[0]
                b = self.ot_mass2 if self.ot_mass2 is not None else torch.ones(T.shape[1], device=T.device) / T.shape[1]
                row_marginal = T.sum(dim=1)
                col_marginal = T.sum(dim=0)
                KL_row = (row_marginal * (torch.log(row_marginal + 1e-10) - torch.log(a + 1e-10)) - row_marginal + a).sum()
                KL_col = (col_marginal * (torch.log(col_marginal + 1e-10) - torch.log(b + 1e-10)) - col_marginal + b).sum()
                entropy = -(T * torch.log(T + 1e-10)).sum()
                entropic = -entropy - T_sum  # = sum(T*log(T) - T)
                loss = transport_cost + rho_actual * (KL_row + KL_col) + eps_actual * entropic
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

            if self.lambda_cheirality > 0.0:
                loss = loss + self.lambda_cheirality * self._cheirality_loss(R_wc, t_hat, T.detach(), self.cheirality_topk)

            # Transport diagnostics for collapse detection
            with torch.no_grad():
                T_sum = T.sum().item()
                row_sum = T.sum(dim=1)
                col_sum = T.sum(dim=0)
                top1_mean = T.max(dim=1).values.mean().item()

            if it % 20 == 0:
                print(f"  T.sum(): {T_sum:.4f} (collapse if decreasing!)")
                print(f"  row_sum: mean={row_sum.mean().item():.4f}, min={row_sum.min().item():.4f}, max={row_sum.max().item():.4f}")
                print(f"  top1_mean: {top1_mean:.4f}")

            loss.backward()

            # Step III: optimize_mode - zero out gradients for fixed components
            if optimize_mode != "both" and self.theta_param.grad is not None:
                grad = self.theta_param.grad
                q_grad, t_grad = manifold.unpack_tensor(grad)
                if optimize_mode == "rotation_only":
                    # Zero out translation gradient
                    t_grad.zero_()
                elif optimize_mode == "translation_only":
                    # Zero out rotation gradient
                    q_grad.zero_()
                # Pack back into theta_param.grad
                self.theta_param.grad = manifold.pack_point(q_grad, t_grad)

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
                    f.write(f"{it},{curr_loss:.8f},{T_sum:.6f},{top1_mean:.6f},{gq:.8f},{gt:.8f},{q_t_ratio:.8f},{q.norm().item():.8f}\n")
                
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
                self.theta_param.copy_(manifold.pack_point(q_final, t_final))
            
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

        final_loss = loss_history[-1] if loss_history else float('nan')
        print(f"[Geoopt-S³×S²] finished after {it+1} iterations, final loss = {final_loss:.6f}")
        print(f"Final quaternion norm: {q_final_normalized.norm().item():.6f}")
        print(f"Final R_wc det: {torch.linalg.det(self.R_wc).item():.6f}")
        return loss_history
    
