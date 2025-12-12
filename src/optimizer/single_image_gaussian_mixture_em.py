import os
from typing import List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer


class SingleImageGaussianMixtureEM:
    """conduct Gaussian Mixture Model optimization on a single image using the EM algorithm."""

    def __init__(self, image_path: str) -> None:
        """Initialize the SingleImageGaussianMixtureEM with an image file.

        Args:
            image_path (str): Path to the input image file.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be opened or processed.
        """
        try:
            with Image.open(image_path) as img:
                self.image = np.array(img).astype(float) / 255.0
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {image_path}") from e
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}") from e

        # if self.image.ndim != 3 or self.image.shape[2] != 3:
        #     raise ValueError("Input image must be a 3-channel color image")

    def initialize_gaussians(
        self, n_gaussians: int, mode: str = "grid", seed: int = 0
    ) -> TwoDGaussians:
        """Initialize Gaussians with a deterministic, image-covering layout.

        Args:
            n_gaussians (int): Number of Gaussians to initialize.
            mode (str): Initialization mode. Currently supports {"grid", "random"}.
            seed (int): RNG seed used for reproducibility in non-grid paths.

        Returns:
            TwoDGaussians: Initialized Gaussians.
        """
        height, width = self.image.shape[:2]
        eps = 1e-12
        rng = np.random.default_rng(seed)

        if mode == "grid":
            n_side = int(np.ceil(np.sqrt(n_gaussians)))
            ys = np.linspace(0.5, height - 0.5, n_side)
            xs = np.linspace(0.5, width - 0.5, n_side)
            Y, X = np.meshgrid(ys, xs, indexing="ij")
            means = np.stack([Y.ravel(), X.ravel()], axis=1)
            if means.shape[0] > n_gaussians:
                means = means[:n_gaussians]
            elif means.shape[0] < n_gaussians:
                pad = n_gaussians - means.shape[0]
                means = np.concatenate([means, means[:pad]], axis=0)
        else:
            means = np.column_stack(
                [
                    rng.uniform(0, height, size=n_gaussians),
                    rng.uniform(0, width, size=n_gaussians),
                ]
            ).astype(np.float64)

        # Covariance: start from cell-sized diagonal
        cell_h = height / np.sqrt(n_gaussians)
        cell_w = width / np.sqrt(n_gaussians)
        # 少し広めにとり、初期レンダの暗さを避ける
        sigma_y = (0.5 * cell_h) ** 2
        sigma_x = (0.5 * cell_w) ** 2
        covs = np.tile(np.diag([sigma_y, sigma_x]), (n_gaussians, 1, 1)).astype(
            np.float64
        )

        # Colors: local patch mean around each grid point
        rgb = []
        patch = int(max(cell_h, cell_w) // 2)
        for y, x in means:
            yy0 = int(np.clip(y - patch, 0, height - 1))
            yy1 = int(np.clip(y + patch, 0, height))
            xx0 = int(np.clip(x - patch, 0, width - 1))
            xx1 = int(np.clip(x + patch, 0, width))
            patch_img = self.image[yy0:yy1, xx0:xx1]
            if patch_img.size == 0:
                rgb.append(self.image[int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))])
            else:
                rgb.append(patch_img.mean(axis=(0, 1)))
        rgb = np.maximum(np.array(rgb, dtype=np.float64), eps)

        alpha = np.full(n_gaussians, 1.0 / n_gaussians, dtype=np.float64)
        rotations = np.zeros(n_gaussians, dtype=np.float64)
        eigenvalues, _ = np.linalg.eigh(covs)
        scales = np.sqrt(np.maximum(eigenvalues, eps))
        return TwoDGaussians(means, covs, rgb, alpha, rotations, scales)

    def gaussian_pdf(
        self, mean: np.ndarray, cov: np.ndarray, height: int, width: int, normalize_per_k: bool = False
    ) -> np.ndarray:
        """Compute the Gaussian PDF for multiple points and multiple Gaussians.

        Args:
            mean (np.ndarray): Means of Gaussians, shape (K, 2)
            cov (np.ndarray): Covariance matrices, shape (K, 2, 2)
            height (int): Height of the image
            width (int): Width of the image
            normalize_per_k (bool): Whether to normalize each Gaussian on discrete grid (for Multinomial)

        Returns:
            np.ndarray: Gaussian PDF values, shape (height, width, K)
        """
        eps = 1e-12
        
        # Create grid coordinates
        y, x = np.mgrid[0:height, 0:width]
        xy = np.stack([y, x], axis=-1).astype(np.float64)  # (H, W, 2)

        K = mean.shape[0]
        
        # Regularize covariance matrices before inversion
        cov_reg = cov + eps * np.eye(2)[None, :, :]  # (K, 2, 2)
        inv = np.linalg.inv(cov_reg)  # (K, 2, 2)
        det = np.clip(np.linalg.det(cov_reg), eps, None)  # (K,)

        # Compute Mahalanobis distance: (x-μ)^T Σ^{-1} (x-μ)
        diff = xy[:, :, None, :] - mean[None, None, :, :]  # (H, W, K, 2)
        maha = np.einsum('hwki,kij,hwkj->hwk', diff, inv, diff)  # (H, W, K)

        # 2D Gaussian PDF with continuous normalization constant
        phi = np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(det))[None, None, :]  # (H, W, K)

        # Optional discrete normalization for Multinomial model (disabled for Poisson)
        if normalize_per_k:
            Z = phi.sum(axis=(0, 1), keepdims=True) + eps
            phi = phi / Z

        return phi

    def e_step(self, gaussians: TwoDGaussians, k_chunk_size: int = 256) -> np.ndarray:
        """Compute the responsibilities (gamma) for each pixel, each color channel, and each Gaussian.
        
        Args:
            gaussians: The current Gaussian mixture model
            k_chunk_size: Process K Gaussians in chunks to save memory
        
        Returns:
            np.ndarray: Responsibilities, shape (height, width, K, 3)
        """
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = 1e-12

        # For large K, use chunked processing to save memory
        if K > k_chunk_size:
            return self._e_step_chunked(gaussians, k_chunk_size)

        # Standard processing for small K
        # Spatial distribution φ_k(x,y) = continuous 2D Gaussian (no discrete normalization)
        phi = self.gaussian_pdf(gaussians.means, gaussians.covs, H, W, normalize_per_k=False)  # (H, W, K)

        # Color intensities ρ_{k,i} (non-negative, no normalization constraint)
        rho = np.clip(gaussians.rgb, eps, None)  # (K, 3)

        # Log responsibilities: log α + log φ + log ρ
        log_alpha = np.log(np.clip(gaussians.alpha, eps, None))  # (K,)
        log_phi = np.log(np.clip(phi, eps, None))  # (H, W, K)
        log_rho = np.log(rho)  # (K, 3)

        # Broadcast and combine: (H, W, K, 3)
        log_r = (log_alpha[None, None, :, None] + 
                 log_phi[:, :, :, None] + 
                 log_rho[None, None, :, :])  # (H, W, K, 3)

        # Normalize over k using log-sum-exp for numerical stability
        max_log_r = np.max(log_r, axis=2, keepdims=True)  # (H, W, 1, 3)
        exp_r = np.exp(log_r - max_log_r)  # (H, W, K, 3)
        gamma = exp_r / (np.sum(exp_r, axis=2, keepdims=True) + eps)  # (H, W, K, 3)

        return gamma

    def _e_step_chunked(self, gaussians: TwoDGaussians, k_chunk_size: int) -> np.ndarray:
        """Memory-efficient E-step for large K using chunked processing with stable log-sum-exp."""
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = 1e-12
        
        # Running max and sum for stable log-sum-exp accumulation
        m = np.full((H, W, 3), -np.inf, dtype=np.float32)
        s = np.zeros((H, W, 3), dtype=np.float32)

        # ---------- 1st pass: accumulate denominator ----------
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            phi = self.gaussian_pdf(gaussians.means[k_start:k_end],
                                    gaussians.covs[k_start:k_end],
                                    H, W, normalize_per_k=False).astype(np.float32)
            rho = np.clip(gaussians.rgb[k_start:k_end], eps, None).astype(np.float32)
            log_alpha = np.log(np.clip(gaussians.alpha[k_start:k_end], eps, None)).astype(np.float32)
            
            log_r = (log_alpha[None, None, :, None]
                     + np.log(np.clip(phi, eps, None))[:, :, :, None]
                     + np.log(rho)[None, None, :, :])                            # (H,W,kc,3)

            chunk_max = np.max(log_r, axis=2)                                    # (H,W,3)
            new_m = np.maximum(m, chunk_max)                                     # (H,W,3)
            # Base change: s_new = s*exp(m-new_m) + sum(exp(log_r - new_m))
            s = s * np.exp(m - new_m) + np.sum(np.exp(log_r - new_m[:, :, None, :]), axis=2)
            m = new_m

        log_denom = m[:, :, None, :] + np.log(s[:, :, None, :] + eps)            # (H,W,1,3)

        # ---------- 2nd pass: output gamma ----------
        gamma = np.zeros((H, W, K, 3), dtype=np.float32)
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            phi = self.gaussian_pdf(gaussians.means[k_start:k_end],
                                    gaussians.covs[k_start:k_end],
                                    H, W, normalize_per_k=False).astype(np.float32)
            rho = np.clip(gaussians.rgb[k_start:k_end], eps, None).astype(np.float32)
            log_alpha = np.log(np.clip(gaussians.alpha[k_start:k_end], eps, None)).astype(np.float32)
            
            log_r = (log_alpha[None, None, :, None]
                     + np.log(np.clip(phi, eps, None))[:, :, :, None]
                     + np.log(rho)[None, None, :, :])                            # (H,W,kc,3)
            gamma[:, :, k_start:k_end, :] = np.exp(log_r - log_denom)
        
        return gamma.astype(np.float64)

    def _sum_Sk_chunked(self, gaussians: TwoDGaussians, H: int, W: int, k_chunk_size: int = 256) -> np.ndarray:
        """Compute S_k = sum of phi_k values in chunked manner to save memory."""
        S_k = np.zeros(gaussians.k, dtype=np.float64)
        for k_start in range(0, gaussians.k, k_chunk_size):
            k_end = min(k_start + k_chunk_size, gaussians.k)
            phi = self.gaussian_pdf(gaussians.means[k_start:k_end],
                                    gaussians.covs[k_start:k_end],
                                    H, W, normalize_per_k=False)      # (H,W,kc)
            S_k[k_start:k_end] = phi.sum(axis=(0, 1))
        return S_k

    def m_step(self, gamma: np.ndarray, gaussians: TwoDGaussians) -> TwoDGaussians:
        """Update the parameters of the Poisson rate model.

        Args:
            gamma (ndarray): Responsibilities, shape (height, width, K, 3).
            gaussians (TwoDGaussians): Current Gaussian mixture model.

        Returns:
            TwoDGaussians: Updated Gaussian mixture model.
        """
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = 1e-12

        # Expected counts: n_hat = I * gamma
        I = np.clip(self.image, 0.0, None)  # (H, W, 3) - no upper clipping for Poisson
        n_hat = I[:, :, None, :] * gamma  # (H, W, K, 3)
        
        # Compute N_ki and N_k
        N_ki = n_hat.sum(axis=(0, 1))  # (K, 3)
        N_k = N_ki.sum(axis=1) + eps  # (K,)
        N_total = N_k.sum() + eps

        # Compute S_k for Poisson rate model using chunked computation
        S_k = self._sum_Sk_chunked(gaussians, H, W, k_chunk_size=256) + eps  # (K,) - discrete sum of continuous Gaussian

        # Update mixing coefficients (MLE, simple normalization)
        new_alpha = N_k / N_total
        new_alpha = np.clip(new_alpha, eps, None)
        new_alpha /= new_alpha.sum()
        alpha_thresh = 1e-4
        dead = new_alpha < alpha_thresh  # mark for potential reinit

        # Update color intensities: ρ_{k,i} = N_{k,i} / (α_k * S_k)
        new_colors = N_ki / (new_alpha[:, None] * S_k[:, None])  # (K, 3) - no normalization constraint

        # Spatial weights (sum over color channels)
        w_xyk = n_hat.sum(axis=3)  # (H, W, K)

        # Create coordinate grid
        y, x = np.mgrid[0:H, 0:W]
        xy = np.stack([y, x], axis=-1).astype(np.float64)  # (H, W, 2)

        # Update means: μ_k = Σ_{x,y,i} n̂_{x,y,i,k} * (x,y) / N_k
        new_means = (w_xyk[..., None] * xy[:, :, None, :]).sum(axis=(0, 1)) / N_k[:, None]  # (K, 2)

        # Update covariances: Σ_k = Σ_{x,y,i} n̂_{x,y,i,k} * (x,y-μ_k)(x,y-μ_k)^T / N_k
        diff = xy[:, :, None, :] - new_means[None, None, :, :]  # (H, W, K, 2)
        new_covs = np.einsum('hwk,hwki,hwkj->kij', w_xyk, diff, diff) / N_k[:, None, None]  # (K, 2, 2)

        # Ensure covariance matrices are positive definite
        for k in range(K):
            new_covs[k] = self.ensure_positive_definite(new_covs[k])

        # Handle Gaussians with very low responsibility using residual-based reinit
        responsibility_threshold = 1e-6
        small_responsibility_indices = np.where((N_k < responsibility_threshold) | dead)[0]
        if len(small_responsibility_indices) > 0:
            rasterizer = Vanilla2DRasterizer(H, W)
            # build rotations/scales consistent with current covs for rendering
            tmp_rot = np.zeros(K, dtype=np.float64)
            tmp_scales = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                ev, evec = np.linalg.eigh(new_covs[k])
                tmp_rot[k] = np.arctan2(evec[1, 0], evec[0, 0])
                tmp_scales[k] = np.sqrt(np.clip(ev, 1e-6, None))
            tmp_gauss = TwoDGaussians(
                new_means.copy(),
                new_covs.copy(),
                new_colors.copy(),
                new_alpha.copy(),
                tmp_rot,
                tmp_scales,
            )
            rates = rasterizer.render_rates(tmp_gauss)
            res = (np.clip(self.image, 0.0, None) - rates).sum(axis=2)
            flat_idx = np.argsort(res.ravel())[::-1]
            ys, xs = np.unravel_index(flat_idx, res.shape)
            boost_alpha = float(new_alpha.mean() * 2.0)
            for j, idx in enumerate(small_responsibility_indices):
                if j >= len(flat_idx):
                    break
                y0, x0 = ys[j], xs[j]
                new_means[idx] = np.array([y0, x0], dtype=np.float64)
                sigma0 = (0.05 * min(H, W)) ** 2
                new_covs[idx] = np.array([[sigma0, 0.0], [0.0, sigma0]], dtype=np.float64)
                sampled_color = self.image[int(y0), int(x0)]
                new_colors[idx] = np.maximum(sampled_color, eps)
                # give a bit more mass so reinit components can compete in next E-step
                new_alpha[idx] = boost_alpha
            # redistribute alpha: keep alive ratios, spread dead_mass evenly
            orig_alpha = new_alpha.copy()
            dead_mass = float(orig_alpha[small_responsibility_indices].sum())
            if dead_mass < eps:
                dead_mass = eps * len(small_responsibility_indices)
            alive_mask = np.ones(K, dtype=bool)
            alive_mask[small_responsibility_indices] = False
            alive_mass = float(orig_alpha[alive_mask].sum())
            if alive_mass <= eps:
                new_alpha[:] = 1.0 / K
            else:
                new_alpha[alive_mask] = orig_alpha[alive_mask] * max(
                    1.0 - dead_mass, eps
                ) / max(alive_mass, eps)
                new_alpha[small_responsibility_indices] = dead_mass / len(
                    small_responsibility_indices
                )
                new_alpha = np.maximum(new_alpha, eps)
                new_alpha /= new_alpha.sum()

        # Update rotations and scales from the new covariance matrices
        new_rotations = np.zeros(K, dtype=np.float64)
        new_scales = np.zeros((K, 2), dtype=np.float64)
        
        max_eig = (0.5 * max(H, W)) ** 2
        for k in range(K):
            new_covs[k] = self.ensure_positive_definite(
                new_covs[k], min_eigenvalue=1e-3, max_eigenvalue=max_eig
            )
            eigenvalues, eigenvectors = np.linalg.eigh(new_covs[k])
            new_rotations[k] = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            new_scales[k] = np.sqrt(np.clip(eigenvalues, 1e-6, None))

        # Clamp means inside image bounds to avoid OOB Gaussians downstream
        new_means[:, 0] = np.clip(new_means[:, 0], 0, H - 1)
        new_means[:, 1] = np.clip(new_means[:, 1], 0, W - 1)

        return TwoDGaussians(new_means, new_covs, new_colors, new_alpha, new_rotations, new_scales)

        return TwoDGaussians(new_means, new_covs, new_colors, new_alpha, new_rotations, new_scales)

    def ensure_positive_definite(
        self,
        cov: np.ndarray,
        min_eigenvalue: float = 1e-3,
        max_eigenvalue: Optional[float] = None,
    ) -> np.ndarray:
        """Ensure the covariance matrix is positive definite.

        Args:
            cov (np.ndarray): The covariance matrix to check and adjust.
            min_eigenvalue (float): The minimum eigenvalue threshold.

        Returns:
            np.ndarray: The adjusted positive definite covariance matrix.
        """
        # Symmetrize the matrix first
        cov = 0.5 * (cov + cov.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Clip eigenvalues to ensure positive definiteness
        if max_eigenvalue is None:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
        else:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, max_eigenvalue)
        
        # Reconstruct the matrix
        return (eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T).astype(cov.dtype)

    def check_positive_definite(self, cov: np.ndarray) -> bool:
        """Check if the covariance matrix is positive definite.

        Args:
            cov (np.ndarray): The covariance matrix to check.

        Returns:
            bool: True if the matrix is positive definite, False otherwise.
        """
        eigenvalues = np.linalg.eigvals(cov)
        return bool(np.all(eigenvalues > 0))

    def print_covariance_stats(self, new_covs: List[np.ndarray]) -> None:
        """Print statistics about the covariance matrices.

        Args:
            new_covs (List[np.ndarray]): List of new covariance matrices.
        """
        det_values = np.array([np.linalg.det(cov) for cov in new_covs])
        min_eigenvalues = np.array([np.min(np.linalg.eigvals(cov)) for cov in new_covs])

        print(
            f"New covs determinant min-max: {np.min(det_values)}, {np.max(det_values)}"
        )
        print(
            f"New covs min eigenvalue min-max: {np.min(min_eigenvalues)}, {np.max(min_eigenvalues)}"
        )
        print(
            f"Positive definite covariances: {np.sum([self.check_positive_definite(cov) for cov in new_covs])}/{len(new_covs)}"
        )

    def poisson_nll(self, gaussians: TwoDGaussians, eps: float = 1e-12) -> float:
        """Compute Poisson negative log-likelihood for convergence monitoring.
        
        Args:
            gaussians: Current Gaussian mixture model
            eps: Small constant for numerical stability
            
        Returns:
            float: Negative log-likelihood value (should decrease monotonically in EM)
        """
        from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer
        
        # Get Poisson rates λ = Σ_k α_k ρ_k φ_k(x,y)
        rasterizer = Vanilla2DRasterizer(self.image.shape[0], self.image.shape[1])
        rates = rasterizer.render_rates(gaussians)  # (H, W, 3)
        
        # Observed intensities
        I = np.clip(self.image, 0.0, None)
        
        # Poisson NLL = Σ(λ - I*log(λ)) + constants
        nll = (rates - I * np.log(rates + eps)).sum()
        return float(nll)
