"""Single-image Gaussian Mixture EM optimizer with PyTorch/MPS acceleration.

This module implements EM optimization for fitting a Gaussian Mixture Model to an image,
designed for the perspective-n-gaussian pipeline where 2D Gaussians are matched via OT
to 3D points.

Key design decisions:
- alpha = Poisson rate mass/intensity (NOT opacity 0-1, can be >> 1)
- rgb = normalized color ratios (sum to 1 per Gaussian)
- Torch-first implementation for speed, NumPy reference for testing
- Streaming EM (em_step_streaming) to avoid holding large gamma tensors in memory
- 2x2 analytical determinant/inverse for MPS compatibility

IMPORTANT: For K > k_chunk_size, always use em_step_streaming() instead of
separate e_step() + m_step() to avoid OOM. The e_step() function will warn
or raise an error for large K.

Heuristics (optional, can be disabled):
- Dead component reinitialization
- Covariance SPD enforcement
These may cause NLL to be non-monotonic in some iterations.
"""
import os
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

# PyTorch/MPS support (required for efficient operation)
try:
    import torch
    HAS_TORCH = True
    HAS_MPS = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    HAS_TORCH = False
    HAS_MPS = False

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer


# =============================================================================
# Epsilon constants - split by purpose for numerical stability
# =============================================================================
EPS_FLOAT64 = 1e-12  # For NumPy float64 operations
EPS_COV32 = 1e-6     # For covariance regularization (det clamp, inverse stability)
EPS_LOG32 = 1e-8     # For log operations (alpha, rgb clamp before log)

# Default chunk sizes (smaller for MPS due to memory constraints)
DEFAULT_K_CHUNK_CUDA = 256
DEFAULT_K_CHUNK_MPS = 64
DEFAULT_K_CHUNK_CPU = 128

# Memory threshold for auto chunk size reduction (in pixels)
LARGE_IMAGE_THRESHOLD = 400 * 400  # 160,000 pixels


def _get_default_k_chunk(device: str, h: int = 0, w: int = 0) -> int:
    """Get default k_chunk_size based on device and image size.

    For large images, reduces chunk size to avoid OOM.
    """
    if "mps" in device:
        base_chunk = DEFAULT_K_CHUNK_MPS
    elif "cuda" in device:
        base_chunk = DEFAULT_K_CHUNK_CUDA
    else:
        base_chunk = DEFAULT_K_CHUNK_CPU

    # Auto-reduce for large images
    if h > 0 and w > 0:
        pixels = h * w
        if pixels > LARGE_IMAGE_THRESHOLD * 4:  # > 640k pixels
            base_chunk = max(16, base_chunk // 4)
        elif pixels > LARGE_IMAGE_THRESHOLD * 2:  # > 320k pixels
            base_chunk = max(16, base_chunk // 2)
        elif pixels > LARGE_IMAGE_THRESHOLD:  # > 160k pixels
            base_chunk = max(32, base_chunk * 3 // 4)

    return base_chunk


class SingleImageGaussianMixtureEM:
    """Gaussian Mixture Model optimization on a single image using EM algorithm.

    Uses streaming M-step to handle large K without memory explosion.
    Torch-first implementation with NumPy reference for testing.

    IMPORTANT: For large K (> k_chunk_size), always use em_step_streaming()
    instead of e_step() + m_step() to avoid OOM.
    """

    def __init__(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        mask: Optional[np.ndarray] = None,
        mask_threshold: float = 0.5,
    ) -> None:
        """Initialize with an image file.

        Args:
            image_path: Path to input image file.
            mask_path: Optional path to mask image.
            mask: Optional mask array (H,W). Overrides mask_path.
            mask_threshold: Threshold to binarize mask.
        """
        alpha_mask = None
        try:
            with Image.open(image_path) as img:
                if img.mode == "RGBA":
                    rgb, a = img.convert("RGBA").split()[:3], img.convert("RGBA").split()[3]
                    img_rgb = Image.merge("RGB", rgb)
                    self.image = np.asarray(img_rgb, dtype=np.float64) / 255.0
                    alpha_arr = np.asarray(a, dtype=np.float64) / 255.0
                    alpha_mask = alpha_arr
                else:
                    img = img.convert("RGB")
                    self.image = np.asarray(img, dtype=np.float64) / 255.0
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {image_path}") from e
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}") from e

        self.mask_threshold = float(mask_threshold)
        self.mask: Optional[np.ndarray] = None
        self.valid_pixels: Optional[np.ndarray] = None
        self._rng: np.random.Generator = np.random.default_rng(0)

        # Device setup
        self.use_torch = HAS_TORCH
        self.device = "mps" if HAS_MPS else ("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu")

        # Default chunk size based on device and image size
        H, W = self.image.shape[:2]
        self.default_k_chunk = _get_default_k_chunk(self.device, H, W)

        # Grid caches
        self._grid_cache_np: Optional[Tuple[int, int, np.ndarray]] = None
        self._grid_cache_torch: Optional[Tuple[int, int, str, "torch.Tensor"]] = None

        # Image/mask tensor cache (kept on device)
        self._image_tensor: Optional["torch.Tensor"] = None
        self._mask_tensor: Optional["torch.Tensor"] = None

        # Set mask
        if mask is not None:
            self.set_mask(mask, threshold=self.mask_threshold)
        elif alpha_mask is not None:
            self.set_mask(alpha_mask, threshold=self.mask_threshold)
        elif mask_path is not None:
            try:
                m = np.array(Image.open(mask_path)).astype(np.float64)
                if m.max() > 1.0:
                    m = m / 255.0
                self.set_mask(m, threshold=self.mask_threshold)
            except Exception as e:
                raise ValueError(f"Error processing mask: {str(e)}") from e

    # =========================================================================
    # Mask utilities
    # =========================================================================
    def set_mask(self, mask: Optional[np.ndarray], threshold: float = 0.5) -> None:
        """Set or clear mask."""
        if mask is None:
            self.mask = None
            self.valid_pixels = None
            self._mask_tensor = None
            return
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        m = m.astype(np.float64)
        if m.max() > 1.0:
            m = m / 255.0
        H, W = self.image.shape[:2]
        if m.shape[0] != H or m.shape[1] != W:
            raise ValueError(f"Mask shape {m.shape} does not match image shape {(H, W)}")
        mb = (m > float(threshold)).astype(np.float64)
        vp = np.argwhere(mb > 0.5)
        if vp.size == 0:
            self.mask = None
            self.valid_pixels = None
            self._mask_tensor = None
            return
        self.mask = mb
        self.valid_pixels = vp
        self._mask_tensor = None  # Invalidate cache

    # =========================================================================
    # Grid and tensor caching
    # =========================================================================
    def _get_grid_numpy(self, H: int, W: int) -> np.ndarray:
        """Get cached NumPy grid coordinates (H, W, 2)."""
        if self._grid_cache_np is not None and self._grid_cache_np[0] == H and self._grid_cache_np[1] == W:
            return self._grid_cache_np[2]
        y, x = np.mgrid[0:H, 0:W]
        xy = np.stack([y, x], axis=-1).astype(np.float32)
        self._grid_cache_np = (H, W, xy)
        return xy

    def _get_grid_torch(self, H: int, W: int) -> "torch.Tensor":
        """Get cached PyTorch grid coordinates on device (H, W, 2)."""
        if (self._grid_cache_torch is not None and
            self._grid_cache_torch[0] == H and
            self._grid_cache_torch[1] == W and
            self._grid_cache_torch[2] == self.device):
            return self._grid_cache_torch[3]
        with torch.no_grad():
            y = torch.arange(H, dtype=torch.float32, device=self.device)
            x = torch.arange(W, dtype=torch.float32, device=self.device)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            xy = torch.stack([yy, xx], dim=-1)
        self._grid_cache_torch = (H, W, self.device, xy)
        return xy

    def _get_image_tensor(self) -> "torch.Tensor":
        """Get cached image tensor on device (H, W, 3)."""
        if self._image_tensor is None:
            self._image_tensor = torch.from_numpy(
                np.clip(self.image, 0.0, None).astype(np.float32)
            ).to(self.device)
        return self._image_tensor

    def _get_mask_tensor(self) -> Optional["torch.Tensor"]:
        """Get cached mask tensor on device (H, W)."""
        if self.mask is None:
            return None
        if self._mask_tensor is None:
            self._mask_tensor = torch.from_numpy(
                self.mask.astype(np.float32)
            ).to(self.device)
        return self._mask_tensor

    # =========================================================================
    # Analytical 2x2 computation helpers (MPS-compatible, no torch.linalg)
    # =========================================================================
    def _compute_maha_and_logdet_inv_torch(
        self, xy: "torch.Tensor", mean_t: "torch.Tensor", cov_t: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Compute Mahalanobis distance and log-determinant using analytical 2x2 formulas.

        Uses analytical 2x2 determinant and inverse to avoid torch.linalg.det/inv
        (det not supported on MPS, inv can be slow).
        Symmetrizes covariance to prevent numerical drift.

        For 2x2 matrix [[a,b],[c,d]]:
            det = a*d - b*c
            inv = (1/det) * [[d,-b],[-c,a]]
        """
        eps_cov = EPS_COV32
        device = mean_t.device

        # Symmetrize covariance to prevent numerical drift
        cov_t = 0.5 * (cov_t + cov_t.transpose(-1, -2))

        eye = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0)
        cov_reg = cov_t + eps_cov * eye

        # Analytical 2x2 determinant: det([[a,b],[c,d]]) = a*d - b*c
        a = cov_reg[:, 0, 0]
        b = cov_reg[:, 0, 1]
        c = cov_reg[:, 1, 0]
        d = cov_reg[:, 1, 1]

        det_t = a * d - b * c
        det_t = torch.clamp(det_t, min=eps_cov)
        logdet = torch.log(det_t)

        # Analytical 2x2 inverse: inv = (1/det) * [[d,-b],[-c,a]]
        inv_det = 1.0 / det_t
        inv_t = torch.zeros_like(cov_reg)
        inv_t[:, 0, 0] = d * inv_det
        inv_t[:, 0, 1] = -b * inv_det
        inv_t[:, 1, 0] = -c * inv_det
        inv_t[:, 1, 1] = a * inv_det

        diff = xy.unsqueeze(2) - mean_t.unsqueeze(0).unsqueeze(0)
        tmp = torch.einsum('hwki,kij->hwkj', diff, inv_t)
        maha = (tmp * diff).sum(dim=-1)

        return maha, logdet

    # =========================================================================
    # Initialization
    # =========================================================================
    def initialize_gaussians(
        self,
        n_gaussians: int,
        mode: str = "grid",
        seed: int = 0,
        mask: Optional[np.ndarray] = None,
    ) -> TwoDGaussians:
        """Initialize Gaussians with deterministic layout.

        Args:
            n_gaussians: Number of Gaussians.
            mode: "grid" or "random".
            seed: RNG seed.
            mask: Optional mask override.

        Returns:
            Initialized TwoDGaussians.
        """
        if mask is not None:
            self.set_mask(mask, threshold=self.mask_threshold)

        height, width = self.image.shape[:2]
        eps = EPS_FLOAT64
        rng = np.random.default_rng(seed)
        self._rng = rng

        mask_w = self.mask
        vp = self.valid_pixels

        # Compute means
        if mask_w is not None and vp is not None:
            y_min, x_min = vp.min(axis=0)
            y_max, x_max = vp.max(axis=0)
            bbox_h = int(y_max - y_min + 1)
            bbox_w = int(x_max - x_min + 1)
            bbox_area = float(max(bbox_h * bbox_w, 1))
            fill_ratio = float(vp.shape[0]) / bbox_area
            if fill_ratio < 0.15:
                vp_sorted = vp[np.lexsort((vp[:, 1], vp[:, 0]))]
                idx = np.linspace(0, vp_sorted.shape[0] - 1, n_gaussians, dtype=int)
                means = vp_sorted[idx].astype(np.float64)
            else:
                n_side = int(np.ceil(np.sqrt(n_gaussians / max(fill_ratio, 1e-6))))
                n_side = max(n_side, 1)
                ys = np.linspace(y_min + 0.5, y_max - 0.5, n_side)
                xs = np.linspace(x_min + 0.5, x_max - 0.5, n_side)
                Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
                grid = np.stack([Yg.ravel(), Xg.ravel()], axis=1)
                yi = np.clip(grid[:, 0].astype(int), 0, height - 1)
                xi = np.clip(grid[:, 1].astype(int), 0, width - 1)
                keep = mask_w[yi, xi] > 0.5
                means = grid[keep]
                if means.shape[0] >= n_gaussians:
                    means = means[:n_gaussians]
                else:
                    pad = n_gaussians - means.shape[0]
                    vp_sorted = vp[np.lexsort((vp[:, 1], vp[:, 0]))]
                    if vp_sorted.shape[0] == 1:
                        extra = np.repeat(vp_sorted.astype(np.float64), pad, axis=0)
                    else:
                        idx = np.linspace(0, vp_sorted.shape[0] - 1, pad, dtype=int)
                        extra = vp_sorted[idx].astype(np.float64)
                    means = np.concatenate([means.astype(np.float64), extra], axis=0)
        else:
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
                means = np.column_stack([
                    rng.uniform(0, height, size=n_gaussians),
                    rng.uniform(0, width, size=n_gaussians),
                ]).astype(np.float64)

        # Covariance
        if mask_w is not None and vp is not None:
            y_min, x_min = vp.min(axis=0)
            y_max, x_max = vp.max(axis=0)
            eff_h = float(max(y_max - y_min + 1, 1))
            eff_w = float(max(x_max - x_min + 1, 1))
        else:
            eff_h = float(height)
            eff_w = float(width)

        cell_h = eff_h / np.sqrt(n_gaussians)
        cell_w = eff_w / np.sqrt(n_gaussians)
        sigma_y = (0.5 * cell_h) ** 2
        sigma_x = (0.5 * cell_w) ** 2
        covs = np.tile(np.diag([sigma_y, sigma_x]), (n_gaussians, 1, 1)).astype(np.float64)

        # Colors from local patches
        rgb = []
        patch = int(max(cell_h, cell_w) // 2)
        patch = max(patch, 1)
        for y, x in means:
            yy0 = int(np.clip(y - patch, 0, height - 1))
            yy1 = int(np.clip(y + patch, 0, height))
            xx0 = int(np.clip(x - patch, 0, width - 1))
            xx1 = int(np.clip(x + patch, 0, width))
            patch_img = self.image[yy0:yy1, xx0:xx1]
            if patch_img.size == 0:
                rgb.append(self.image[int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))])
                continue
            if mask_w is not None:
                mp = mask_w[yy0:yy1, xx0:xx1]
                wsum = float(mp.sum())
                if wsum <= 0.0:
                    rgb.append(self.image[int(np.clip(y, 0, height - 1)), int(np.clip(x, 0, width - 1))])
                else:
                    rgb.append((patch_img * mp[:, :, None]).sum(axis=(0, 1)) / wsum)
            else:
                rgb.append(patch_img.mean(axis=(0, 1)))
        rgb = np.maximum(np.array(rgb, dtype=np.float64), eps)

        # Reparameterize: alpha = mass, rgb = normalized ratio
        alpha = rgb.sum(axis=1) + eps
        rgb = rgb / alpha[:, None]
        rgb = np.maximum(rgb, eps)
        rgb = rgb / (rgb.sum(axis=1, keepdims=True) + eps)
        alpha = alpha / (alpha.mean() + eps)

        rotations = np.zeros(n_gaussians, dtype=np.float64)
        eigenvalues, _ = np.linalg.eigh(covs)
        scales = np.sqrt(np.maximum(eigenvalues, eps))
        return TwoDGaussians(means, covs, rgb, alpha, rotations, scales)

    # =========================================================================
    # Gaussian PDF computation
    # =========================================================================
    def gaussian_pdf(
        self, mean: np.ndarray, cov: np.ndarray, height: int, width: int, normalize_per_k: bool = False
    ) -> np.ndarray:
        """Compute Gaussian PDF values.

        Args:
            mean: (K, 2) means
            cov: (K, 2, 2) covariances
            height, width: image dimensions
            normalize_per_k: whether to normalize each Gaussian on discrete grid

        Returns:
            (H, W, K) PDF values
        """
        if self.use_torch and HAS_TORCH:
            return self._gaussian_pdf_torch(mean, cov, height, width, normalize_per_k)
        return self._gaussian_pdf_numpy(mean, cov, height, width, normalize_per_k)

    def _gaussian_pdf_numpy(
        self, mean: np.ndarray, cov: np.ndarray, height: int, width: int, normalize_per_k: bool = False
    ) -> np.ndarray:
        """NumPy implementation of Gaussian PDF."""
        eps = EPS_FLOAT64
        xy = self._get_grid_numpy(height, width)

        cov_reg = cov + eps * np.eye(2)[None, :, :]
        inv = np.linalg.inv(cov_reg)
        det = np.clip(np.linalg.det(cov_reg), eps, None)

        mean_f32 = mean.astype(np.float32)
        inv_f32 = inv.astype(np.float32)
        diff = xy[:, :, None, :] - mean_f32[None, None, :, :]
        maha = np.einsum('hwki,kij,hwkj->hwk', diff, inv_f32, diff)

        phi = np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(det).astype(np.float32))[None, None, :]

        if normalize_per_k:
            Z = phi.sum(axis=(0, 1), keepdims=True) + eps
            phi = phi / Z

        return phi.astype(np.float64)

    def _gaussian_pdf_torch(
        self, mean: np.ndarray, cov: np.ndarray, height: int, width: int, normalize_per_k: bool = False
    ) -> np.ndarray:
        """PyTorch implementation of Gaussian PDF."""
        with torch.no_grad():
            device = self.device
            xy = self._get_grid_torch(height, width)

            mean_t = torch.from_numpy(mean.astype(np.float32)).to(device)
            cov_t = torch.from_numpy(cov.astype(np.float32)).to(device)

            maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)

            # phi = exp(-0.5 * maha) / (2π * sqrt(det))
            # log(phi) = -0.5 * maha - log(2π) - 0.5 * logdet
            log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)
            phi = torch.exp(log_phi)

            if normalize_per_k:
                Z = phi.sum(dim=(0, 1), keepdim=True) + EPS_COV32
                phi = phi / Z

            return phi.cpu().numpy().astype(np.float64)

    # =========================================================================
    # Rates computation (torch version for speed)
    # =========================================================================
    def render_rates_torch(
        self, gaussians: TwoDGaussians, k_chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute Poisson rates λ(x,y,i) = Σ_k α_k ρ_{k,i} φ_k(x,y) using torch.

        This is much faster than Vanilla2DRasterizer for large K.

        Args:
            gaussians: The Gaussian mixture model
            k_chunk_size: Chunk size for processing. If None, uses device default.

        Returns:
            rates: (H, W, 3) Poisson rate image
        """
        if k_chunk_size is None:
            k_chunk_size = self.default_k_chunk

        H, W = self.image.shape[:2]
        K = gaussians.k

        if not (self.use_torch and HAS_TORCH):
            # Fallback to vanilla rasterizer
            rasterizer = Vanilla2DRasterizer(H, W)
            return rasterizer.render_rates(gaussians)

        with torch.no_grad():
            device = self.device
            xy = self._get_grid_torch(H, W)

            rates = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

            for k_start in range(0, K, k_chunk_size):
                k_end = min(k_start + k_chunk_size, K)

                mean_t = torch.from_numpy(gaussians.means[k_start:k_end].astype(np.float32)).to(device)
                cov_t = torch.from_numpy(gaussians.covs[k_start:k_end].astype(np.float32)).to(device)
                alpha_t = torch.from_numpy(gaussians.alpha[k_start:k_end].astype(np.float32)).to(device)
                rgb_t = torch.from_numpy(gaussians.rgb[k_start:k_end].astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)
                phi = torch.exp(log_phi)  # (H, W, kc)

                # rates += Σ_k α_k * ρ_k * φ_k
                # (H, W, kc) * (kc,) * (kc, 3) -> (H, W, 3)
                amp = alpha_t.unsqueeze(0).unsqueeze(0) * phi  # (H, W, kc)
                rates += torch.einsum('hwk,ki->hwi', amp, rgb_t)

            return rates.cpu().numpy().astype(np.float64)

    # =========================================================================
    # E-step: Compute responsibilities
    # =========================================================================
    def e_step(
        self, gaussians: TwoDGaussians, k_chunk_size: Optional[int] = None,
        return_tensor: bool = False, allow_large_k: bool = False
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """Compute responsibilities gamma.

        WARNING: For K > k_chunk_size, this creates a large gamma array that may cause OOM.
        For large K, use em_step_streaming() instead.

        Args:
            gaussians: Current GMM
            k_chunk_size: Chunk size. If None, uses device default.
            return_tensor: If True and using torch with small K, return torch.Tensor
            allow_large_k: If False (default), raises error for K > k_chunk_size.
                          Set to True only if you know you have enough memory.

        Returns:
            gamma: (H, W, K, 3) responsibilities

        Raises:
            MemoryError: If K > k_chunk_size and allow_large_k is False
        """
        if k_chunk_size is None:
            k_chunk_size = self.default_k_chunk

        K = gaussians.k
        H, W = self.image.shape[:2]

        if K > k_chunk_size:
            # Estimate memory usage for gamma array
            gamma_bytes = H * W * K * 3 * 8  # float64
            gamma_gb = gamma_bytes / (1024 ** 3)

            if not allow_large_k:
                raise MemoryError(
                    f"e_step() would create a {gamma_gb:.2f} GB gamma array "
                    f"(H={H}, W={W}, K={K}). Use em_step_streaming() instead, "
                    f"or set allow_large_k=True if you have enough memory."
                )
            else:
                import warnings
                warnings.warn(
                    f"e_step() creating large gamma array ({gamma_gb:.2f} GB). "
                    f"Consider using em_step_streaming() instead.",
                    ResourceWarning
                )
            # For large K, always return numpy (chunked processing)
            return self._e_step_chunked(gaussians, k_chunk_size)

        if self.use_torch and HAS_TORCH:
            gamma_t = self._e_step_torch_tensor(gaussians)
            if return_tensor:
                return gamma_t
            return gamma_t.cpu().numpy().astype(np.float64)

        return self._e_step_numpy(gaussians)

    def _e_step_numpy(self, gaussians: TwoDGaussians) -> np.ndarray:
        """NumPy E-step implementation (reference)."""
        H, W = self.image.shape[:2]
        eps = EPS_FLOAT64

        xy = self._get_grid_numpy(H, W)
        mean = gaussians.means
        cov = gaussians.covs

        cov_reg = cov + eps * np.eye(2)[None, :, :]
        inv_cov = np.linalg.inv(cov_reg)
        det_cov = np.clip(np.linalg.det(cov_reg), eps, None)

        mean_f32 = mean.astype(np.float32)
        inv_f32 = inv_cov.astype(np.float32)
        diff = xy[:, :, None, :] - mean_f32[None, None, :, :]
        maha = np.einsum('hwki,kij,hwkj->hwk', diff, inv_f32, diff)

        log_phi = -0.5 * maha - np.log(2.0 * np.pi * np.sqrt(det_cov))[None, None, :]

        rho = np.clip(gaussians.rgb, eps, None)
        log_alpha = np.log(np.clip(gaussians.alpha, eps, None))
        log_rho = np.log(rho)

        log_r = (log_alpha[None, None, :, None] +
                 log_phi[:, :, :, None] +
                 log_rho[None, None, :, :])

        max_log_r = np.max(log_r, axis=2, keepdims=True)
        exp_r = np.exp(log_r - max_log_r)
        gamma = exp_r / (np.sum(exp_r, axis=2, keepdims=True) + eps)

        return gamma

    def _e_step_torch_tensor(self, gaussians: TwoDGaussians) -> "torch.Tensor":
        """PyTorch E-step returning tensor (stays on device)."""
        with torch.no_grad():
            H, W = self.image.shape[:2]
            eps_log = EPS_LOG32
            device = self.device

            xy = self._get_grid_torch(H, W)

            mean_t = torch.from_numpy(gaussians.means.astype(np.float32)).to(device)
            cov_t = torch.from_numpy(gaussians.covs.astype(np.float32)).to(device)
            alpha_t = torch.from_numpy(gaussians.alpha.astype(np.float32)).to(device)
            rgb_t = torch.from_numpy(gaussians.rgb.astype(np.float32)).to(device)

            maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
            log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)

            log_alpha = torch.log(torch.clamp(alpha_t, min=eps_log))
            rho_t = torch.clamp(rgb_t, min=eps_log)
            log_rho = torch.log(rho_t)

            log_r = (log_alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1) +
                     log_phi.unsqueeze(-1) +
                     log_rho.unsqueeze(0).unsqueeze(0))

            max_log_r = torch.max(log_r, dim=2, keepdim=True)[0]
            exp_r = torch.exp(log_r - max_log_r)
            gamma = exp_r / (torch.sum(exp_r, dim=2, keepdim=True) + eps_log)

            return gamma

    def _e_step_chunked(self, gaussians: TwoDGaussians, k_chunk_size: int) -> np.ndarray:
        """Memory-efficient E-step for large K using 2-pass chunked processing."""
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = EPS_LOG32 if self.use_torch else EPS_FLOAT64

        m = np.full((H, W, 3), -np.inf, dtype=np.float32)
        s = np.zeros((H, W, 3), dtype=np.float32)

        # 1st pass: accumulate denominator
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            log_phi = self._compute_log_phi_chunk(
                gaussians.means[k_start:k_end],
                gaussians.covs[k_start:k_end],
                H, W
            )

            rho = np.clip(gaussians.rgb[k_start:k_end], eps, None).astype(np.float32)
            log_alpha = np.log(np.clip(gaussians.alpha[k_start:k_end], eps, None)).astype(np.float32)

            log_r = (log_alpha[None, None, :, None]
                     + log_phi[:, :, :, None]
                     + np.log(rho)[None, None, :, :])

            chunk_max = np.max(log_r, axis=2)
            new_m = np.maximum(m, chunk_max)
            s = s * np.exp(m - new_m) + np.sum(np.exp(log_r - new_m[:, :, None, :]), axis=2)
            m = new_m

        log_denom = m[:, :, None, :] + np.log(s[:, :, None, :] + eps)

        # 2nd pass: output gamma
        gamma = np.zeros((H, W, K, 3), dtype=np.float32)
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            log_phi = self._compute_log_phi_chunk(
                gaussians.means[k_start:k_end],
                gaussians.covs[k_start:k_end],
                H, W
            )

            rho = np.clip(gaussians.rgb[k_start:k_end], eps, None).astype(np.float32)
            log_alpha = np.log(np.clip(gaussians.alpha[k_start:k_end], eps, None)).astype(np.float32)

            log_r = (log_alpha[None, None, :, None]
                     + log_phi[:, :, :, None]
                     + np.log(rho)[None, None, :, :])
            gamma[:, :, k_start:k_end, :] = np.exp(log_r - log_denom)

        return gamma.astype(np.float64)

    def _compute_log_phi_chunk(self, means: np.ndarray, covs: np.ndarray, H: int, W: int) -> np.ndarray:
        """Compute log_phi for a chunk using torch if available."""
        if self.use_torch and HAS_TORCH:
            with torch.no_grad():
                device = self.device
                xy = self._get_grid_torch(H, W)

                mean_t = torch.from_numpy(means.astype(np.float32)).to(device)
                cov_t = torch.from_numpy(covs.astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)

                return log_phi.cpu().numpy().astype(np.float32)
        else:
            eps = EPS_FLOAT64
            xy = self._get_grid_numpy(H, W)

            cov_reg = covs + eps * np.eye(2)[None, :, :]
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.clip(np.linalg.det(cov_reg), eps, None)

            mean_f32 = means.astype(np.float32)
            inv_f32 = inv_cov.astype(np.float32)
            diff = xy[:, :, None, :] - mean_f32[None, None, :, :]
            maha = np.einsum('hwki,kij,hwkj->hwk', diff, inv_f32, diff)

            log_phi = -0.5 * maha - np.log(2.0 * np.pi * np.sqrt(det_cov))[None, None, :]
            return log_phi.astype(np.float32)

    # =========================================================================
    # M-step: Update parameters
    # =========================================================================
    def m_step(
        self,
        gamma: Union[np.ndarray, "torch.Tensor"],
        gaussians: TwoDGaussians,
        reinit_dead: bool = True,
        enforce_spd: bool = True,
    ) -> TwoDGaussians:
        """Update GMM parameters.

        IMPORTANT: Only uses torch when gamma is already a torch.Tensor.
        If gamma is numpy (e.g., from chunked E-step), uses numpy path to avoid OOM.

        Args:
            gamma: Responsibilities (H, W, K, 3) - numpy or torch.Tensor
            gaussians: Current GMM
            reinit_dead: If True (default), reinitialize dead components at high-residual pixels.
                        Set to False for strict EM (monotonic NLL decrease).
            enforce_spd: If True (default), enforce positive definite covariance.
                        Set to False for strict EM (may cause numerical issues).

        Returns:
            Updated TwoDGaussians

        Note:
            Heuristics (reinit_dead, enforce_spd) may cause non-monotonic NLL.
            For strict EM behavior, set both to False.
        """
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = EPS_FLOAT64

        # CRITICAL: Only use torch when gamma is already a tensor
        # This prevents OOM when gamma is a large numpy array
        if self.use_torch and HAS_TORCH and isinstance(gamma, torch.Tensor):
            N_ki, N_k, new_means, new_covs = self._m_step_core_torch(gamma, H, W, K)
            N_total = N_k.sum() + eps
        else:
            # NumPy path - safe for large numpy gamma
            if HAS_TORCH and isinstance(gamma, torch.Tensor):
                gamma = gamma.cpu().numpy()
            N_ki, N_k, new_means, new_covs, N_total = self._m_step_core_numpy(gamma, H, W, K)

        # Compute S_k using torch (chunked, safe)
        S_k = self._sum_Sk_chunked(gaussians, H, W) + eps

        # Update colors
        amp = N_ki / S_k[:, None]
        amp = np.maximum(amp, eps)
        alpha_mass = amp.sum(axis=1) + eps
        rgb_norm = amp / alpha_mass[:, None]
        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)

        # Dead component detection
        rel_mass = N_k / N_total
        alpha_thresh = 1e-4
        dead = rel_mass < alpha_thresh

        # Ensure positive definite (optional heuristic)
        if enforce_spd:
            new_covs, _, _ = self.ensure_positive_definite_batch(new_covs)

        # Reinit dead components (optional heuristic)
        reinit_rgb = {}
        reinit_alpha = {}
        if reinit_dead:
            responsibility_threshold = 1e-6
            small_responsibility_indices = np.where((N_k < responsibility_threshold) | dead)[0]
        else:
            small_responsibility_indices = np.array([], dtype=np.int64)

        if len(small_responsibility_indices) > 0:
            # Use torch render_rates for speed
            _, tmp_rot, tmp_scales = self.ensure_positive_definite_batch(new_covs)
            tmp_gauss = TwoDGaussians(
                new_means.copy(), new_covs.copy(), rgb_norm.copy(),
                alpha_mass.copy(), tmp_rot, tmp_scales
            )
            rates = self.render_rates_torch(tmp_gauss)
            res = (np.clip(self.image, 0.0, None) - rates).sum(axis=2)
            if self.mask is not None:
                res = np.where(self.mask > 0.5, res, -np.inf)
            flat = res.ravel()
            finite_idx = np.flatnonzero(np.isfinite(flat))
            if finite_idx.size == 0:
                finite_idx = np.arange(flat.size)
            sorted_idx = finite_idx[np.argsort(flat[finite_idx])[::-1]]
            ys, xs = np.unravel_index(sorted_idx, res.shape)
            boost_alpha = float(alpha_mass.mean() * 2.0)
            for j, idx in enumerate(small_responsibility_indices):
                if j >= len(sorted_idx):
                    break
                y0, x0 = ys[j], xs[j]
                new_means[idx] = np.array([y0, x0], dtype=np.float64)
                sigma0 = (0.05 * min(H, W)) ** 2
                new_covs[idx] = np.array([[sigma0, 0.0], [0.0, sigma0]], dtype=np.float64)
                sampled_color = self.image[int(y0), int(x0)]
                c = np.maximum(sampled_color, eps)
                s = float(c.sum()) + eps
                reinit_rgb[idx] = c / s
                reinit_alpha[idx] = max(boost_alpha, s)

        # Final cleanup - decompose to rotation/scale
        max_eig = (0.5 * max(H, W)) ** 2
        if enforce_spd:
            new_covs, new_rotations, new_scales = self.ensure_positive_definite_batch(
                new_covs, min_eigenvalue=1e-3, max_eigenvalue=max_eig
            )
        else:
            # Just decompose without enforcement
            new_rotations = np.zeros(K, dtype=np.float64)
            new_scales = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                eigvals, eigvecs = np.linalg.eigh(new_covs[k])
                new_rotations[k] = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                new_scales[k] = np.sqrt(np.maximum(eigvals, 1e-12))

        new_means[:, 0] = np.clip(new_means[:, 0], 0, H - 1)
        new_means[:, 1] = np.clip(new_means[:, 1], 0, W - 1)

        if self.mask is not None and self.valid_pixels is not None and self.valid_pixels.size > 0:
            mask_bool = self.mask > 0.5
            rng = self._rng
            for k in range(K):
                yi = int(np.clip(round(new_means[k, 0]), 0, H - 1))
                xi = int(np.clip(round(new_means[k, 1]), 0, W - 1))
                if not mask_bool[yi, xi]:
                    ridx = int(rng.integers(0, self.valid_pixels.shape[0]))
                    yx = self.valid_pixels[ridx]
                    new_means[k] = np.array([float(yx[0]), float(yx[1])], dtype=np.float64)

        # Recompute S_k with updated geometry
        dummy_gauss = TwoDGaussians(
            new_means, new_covs, gaussians.rgb, alpha_mass, new_rotations, new_scales
        )
        S_k_new = self._sum_Sk_chunked(dummy_gauss, H, W) + eps
        amp = N_ki / S_k_new[:, None]
        amp = np.maximum(amp, eps)
        alpha_mass = amp.sum(axis=1) + eps
        rgb_norm = amp / alpha_mass[:, None]
        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)

        for idx, col in reinit_rgb.items():
            rgb_norm[idx] = col
        for idx, val in reinit_alpha.items():
            alpha_mass[idx] = val

        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)
        total_pred = float(np.sum(alpha_mass * S_k_new))
        target = float(N_total)
        if np.isfinite(total_pred) and total_pred > eps:
            alpha_mass *= target / total_pred

        return TwoDGaussians(new_means, new_covs, rgb_norm, alpha_mass, new_rotations, new_scales)

    def _m_step_core_torch(
        self, gamma_t: "torch.Tensor", H: int, W: int, K: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """PyTorch M-step core (gamma already on device)."""
        with torch.no_grad():
            eps = EPS_COV32

            I_t = self._get_image_tensor()
            mask_t = self._get_mask_tensor()

            if mask_t is not None:
                I_masked = I_t * mask_t.unsqueeze(2)
                gamma_masked = gamma_t * mask_t.unsqueeze(2).unsqueeze(3)
            else:
                I_masked = I_t
                gamma_masked = gamma_t

            n_hat = I_masked.unsqueeze(2) * gamma_masked

            N_ki = n_hat.sum(dim=(0, 1))
            N_k = N_ki.sum(dim=1) + eps

            w_xyk = n_hat.sum(dim=3)

            xy = self._get_grid_torch(H, W)

            new_means = (w_xyk.unsqueeze(-1) * xy.unsqueeze(2)).sum(dim=(0, 1)) / N_k.unsqueeze(-1)

            diff = xy.unsqueeze(2) - new_means.unsqueeze(0).unsqueeze(0)
            new_covs = torch.einsum('hwk,hwki,hwkj->kij', w_xyk, diff, diff) / N_k.unsqueeze(-1).unsqueeze(-1)

            return (
                N_ki.cpu().numpy().astype(np.float64),
                N_k.cpu().numpy().astype(np.float64),
                new_means.cpu().numpy().astype(np.float64),
                new_covs.cpu().numpy().astype(np.float64),
            )

    def _m_step_core_numpy(
        self, gamma: np.ndarray, H: int, W: int, K: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """NumPy M-step core (reference implementation)."""
        eps = EPS_FLOAT64
        mask_w = self.mask

        I = np.clip(self.image, 0.0, None)
        if mask_w is not None:
            I = I * mask_w[:, :, None]
            gamma = gamma * mask_w[:, :, None, None]

        n_hat = I[:, :, None, :] * gamma

        N_ki = n_hat.sum(axis=(0, 1))
        N_k = N_ki.sum(axis=1) + eps
        N_total = N_k.sum() + eps

        w_xyk = n_hat.sum(axis=3)
        xy = self._get_grid_numpy(H, W).astype(np.float64)

        new_means = (w_xyk[..., None] * xy[:, :, None, :]).sum(axis=(0, 1)) / N_k[:, None]
        diff = xy[:, :, None, :] - new_means[None, None, :, :]
        new_covs = np.einsum('hwk,hwki,hwkj->kij', w_xyk, diff, diff) / N_k[:, None, None]

        return N_ki, N_k, new_means, new_covs, N_total

    # =========================================================================
    # Streaming M-step: Avoids holding gamma for large K
    # =========================================================================
    def em_step_streaming(
        self, gaussians: TwoDGaussians, k_chunk_size: Optional[int] = None,
        reinit_dead: bool = True, enforce_spd: bool = True,
    ) -> Tuple[TwoDGaussians, float]:
        """Combined E-step + M-step without holding full gamma tensor.

        For large K, this accumulates sufficient statistics during E-step
        instead of storing gamma. ALWAYS use this for K > k_chunk_size.

        Args:
            gaussians: Current GMM
            k_chunk_size: Chunk size. If None, uses device default.
            reinit_dead: If True, reinitialize dead components (may cause non-monotonic NLL)
            enforce_spd: If True, enforce positive definite covariance

        Returns:
            (updated_gaussians, nll)
        """
        if k_chunk_size is None:
            k_chunk_size = self.default_k_chunk

        H, W = self.image.shape[:2]
        K = gaussians.k

        # For small K, use standard approach with tensor
        if K <= k_chunk_size:
            gamma = self.e_step(
                gaussians, k_chunk_size=k_chunk_size, return_tensor=True, allow_large_k=True
            )
            new_gaussians = self.m_step(gamma, gaussians, reinit_dead=reinit_dead, enforce_spd=enforce_spd)
            nll = self.poisson_nll(new_gaussians)
            return new_gaussians, nll

        # For large K, use streaming sufficient statistics
        if self.use_torch and HAS_TORCH:
            return self._em_step_streaming_torch(gaussians, k_chunk_size, reinit_dead, enforce_spd)
        else:
            return self._em_step_streaming_numpy(gaussians, k_chunk_size, reinit_dead, enforce_spd)

    def _em_step_streaming_torch(
        self, gaussians: TwoDGaussians, k_chunk_size: int,
        reinit_dead: bool = True, enforce_spd: bool = True,
    ) -> Tuple[TwoDGaussians, float]:
        """Streaming EM step using torch."""
        with torch.no_grad():
            H, W = self.image.shape[:2]
            K = gaussians.k
            eps_log = EPS_LOG32
            eps_cov = EPS_COV32
            device = self.device

            I_t = self._get_image_tensor()
            mask_t = self._get_mask_tensor()
            xy = self._get_grid_torch(H, W)

            if mask_t is not None:
                I_masked = I_t * mask_t.unsqueeze(2)
            else:
                I_masked = I_t

            # 1st pass: compute log-sum-exp denominator
            m = torch.full((H, W, 3), float('-inf'), dtype=torch.float32, device=device)
            s = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

            for k_start in range(0, K, k_chunk_size):
                k_end = min(k_start + k_chunk_size, K)

                mean_t = torch.from_numpy(gaussians.means[k_start:k_end].astype(np.float32)).to(device)
                cov_t = torch.from_numpy(gaussians.covs[k_start:k_end].astype(np.float32)).to(device)
                alpha_t = torch.from_numpy(gaussians.alpha[k_start:k_end].astype(np.float32)).to(device)
                rgb_t = torch.from_numpy(gaussians.rgb[k_start:k_end].astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)

                log_alpha = torch.log(torch.clamp(alpha_t, min=eps_log))
                log_rho = torch.log(torch.clamp(rgb_t, min=eps_log))

                log_r = (log_alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1) +
                         log_phi.unsqueeze(-1) +
                         log_rho.unsqueeze(0).unsqueeze(0))

                chunk_max = torch.max(log_r, dim=2)[0]
                new_m = torch.maximum(m, chunk_max)
                s = s * torch.exp(m - new_m) + torch.sum(torch.exp(log_r - new_m.unsqueeze(2)), dim=2)
                m = new_m

            log_denom = m.unsqueeze(2) + torch.log(s.unsqueeze(2) + eps_log)

            # 2nd pass: accumulate sufficient statistics
            N_ki = torch.zeros((K, 3), dtype=torch.float32, device=device)
            sum_w_xy = torch.zeros((K, 2), dtype=torch.float32, device=device)

            for k_start in range(0, K, k_chunk_size):
                k_end = min(k_start + k_chunk_size, K)

                mean_t = torch.from_numpy(gaussians.means[k_start:k_end].astype(np.float32)).to(device)
                cov_t = torch.from_numpy(gaussians.covs[k_start:k_end].astype(np.float32)).to(device)
                alpha_t = torch.from_numpy(gaussians.alpha[k_start:k_end].astype(np.float32)).to(device)
                rgb_t = torch.from_numpy(gaussians.rgb[k_start:k_end].astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)

                log_alpha = torch.log(torch.clamp(alpha_t, min=eps_log))
                log_rho = torch.log(torch.clamp(rgb_t, min=eps_log))

                log_r = (log_alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1) +
                         log_phi.unsqueeze(-1) +
                         log_rho.unsqueeze(0).unsqueeze(0))

                gamma_chunk = torch.exp(log_r - log_denom)

                if mask_t is not None:
                    gamma_chunk = gamma_chunk * mask_t.unsqueeze(2).unsqueeze(3)

                n_hat = I_masked.unsqueeze(2) * gamma_chunk
                N_ki[k_start:k_end] = n_hat.sum(dim=(0, 1))

                w_xyk = n_hat.sum(dim=3)
                sum_w_xy[k_start:k_end] = (w_xyk.unsqueeze(-1) * xy.unsqueeze(2)).sum(dim=(0, 1))

            # Compute new means
            N_k = N_ki.sum(dim=1) + eps_cov
            new_means = sum_w_xy / N_k.unsqueeze(-1)

            # 3rd pass: compute covariances
            sum_cov = torch.zeros((K, 2, 2), dtype=torch.float32, device=device)

            for k_start in range(0, K, k_chunk_size):
                k_end = min(k_start + k_chunk_size, K)

                mean_t = torch.from_numpy(gaussians.means[k_start:k_end].astype(np.float32)).to(device)
                cov_t = torch.from_numpy(gaussians.covs[k_start:k_end].astype(np.float32)).to(device)
                alpha_t = torch.from_numpy(gaussians.alpha[k_start:k_end].astype(np.float32)).to(device)
                rgb_t = torch.from_numpy(gaussians.rgb[k_start:k_end].astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)

                log_alpha = torch.log(torch.clamp(alpha_t, min=eps_log))
                log_rho = torch.log(torch.clamp(rgb_t, min=eps_log))

                log_r = (log_alpha.unsqueeze(0).unsqueeze(0).unsqueeze(-1) +
                         log_phi.unsqueeze(-1) +
                         log_rho.unsqueeze(0).unsqueeze(0))

                gamma_chunk = torch.exp(log_r - log_denom)

                if mask_t is not None:
                    gamma_chunk = gamma_chunk * mask_t.unsqueeze(2).unsqueeze(3)

                n_hat = I_masked.unsqueeze(2) * gamma_chunk
                w_xyk = n_hat.sum(dim=3)

                diff_new = xy.unsqueeze(2) - new_means[k_start:k_end].unsqueeze(0).unsqueeze(0)
                sum_cov[k_start:k_end] = torch.einsum('hwk,hwki,hwkj->kij', w_xyk, diff_new, diff_new)

            new_covs = sum_cov / N_k.unsqueeze(-1).unsqueeze(-1)

            # Convert to numpy
            N_ki_np = N_ki.cpu().numpy().astype(np.float64)
            N_k_np = N_k.cpu().numpy().astype(np.float64)
            new_means_np = new_means.cpu().numpy().astype(np.float64)
            new_covs_np = new_covs.cpu().numpy().astype(np.float64)

        return self._finish_m_step(
            gaussians, N_ki_np, N_k_np, new_means_np, new_covs_np,
            reinit_dead=reinit_dead, enforce_spd=enforce_spd
        )

    def _em_step_streaming_numpy(
        self, gaussians: TwoDGaussians, k_chunk_size: int,
        reinit_dead: bool = True, enforce_spd: bool = True,
    ) -> Tuple[TwoDGaussians, float]:
        """Streaming EM step using numpy (fallback)."""
        gamma = self._e_step_chunked(gaussians, k_chunk_size)
        new_gaussians = self.m_step(gamma, gaussians, reinit_dead=reinit_dead, enforce_spd=enforce_spd)
        nll = self.poisson_nll(new_gaussians)
        return new_gaussians, nll

    def _finish_m_step(
        self, gaussians: TwoDGaussians,
        N_ki: np.ndarray, N_k: np.ndarray,
        new_means: np.ndarray, new_covs: np.ndarray,
        reinit_dead: bool = True, enforce_spd: bool = True,
    ) -> Tuple[TwoDGaussians, float]:
        """Finish M-step from sufficient statistics."""
        H, W = self.image.shape[:2]
        K = gaussians.k
        eps = EPS_FLOAT64
        N_total = N_k.sum() + eps

        S_k = self._sum_Sk_chunked(gaussians, H, W) + eps

        amp = N_ki / S_k[:, None]
        amp = np.maximum(amp, eps)
        alpha_mass = amp.sum(axis=1) + eps
        rgb_norm = amp / alpha_mass[:, None]
        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)

        rel_mass = N_k / N_total
        dead = rel_mass < 1e-4

        # Ensure positive definite (optional heuristic)
        if enforce_spd:
            new_covs, _, _ = self.ensure_positive_definite_batch(new_covs)

        # Reinit dead components (optional heuristic)
        reinit_rgb = {}
        reinit_alpha = {}
        if reinit_dead:
            small_resp = np.where((N_k < 1e-6) | dead)[0]
        else:
            small_resp = np.array([], dtype=np.int64)

        if len(small_resp) > 0:
            _, tmp_rot, tmp_scales = self.ensure_positive_definite_batch(new_covs)
            tmp_gauss = TwoDGaussians(
                new_means.copy(), new_covs.copy(), rgb_norm.copy(),
                alpha_mass.copy(), tmp_rot, tmp_scales
            )
            rates = self.render_rates_torch(tmp_gauss)
            res = (np.clip(self.image, 0.0, None) - rates).sum(axis=2)
            if self.mask is not None:
                res = np.where(self.mask > 0.5, res, -np.inf)
            flat = res.ravel()
            finite_idx = np.flatnonzero(np.isfinite(flat))
            if finite_idx.size == 0:
                finite_idx = np.arange(flat.size)
            sorted_idx = finite_idx[np.argsort(flat[finite_idx])[::-1]]
            ys, xs = np.unravel_index(sorted_idx, res.shape)
            boost = float(alpha_mass.mean() * 2.0)
            for j, idx in enumerate(small_resp):
                if j >= len(sorted_idx):
                    break
                y0, x0 = ys[j], xs[j]
                new_means[idx] = np.array([y0, x0], dtype=np.float64)
                sig = (0.05 * min(H, W)) ** 2
                new_covs[idx] = np.array([[sig, 0.0], [0.0, sig]], dtype=np.float64)
                c = np.maximum(self.image[int(y0), int(x0)], eps)
                s = float(c.sum()) + eps
                reinit_rgb[idx] = c / s
                reinit_alpha[idx] = max(boost, s)

        # Final cleanup - decompose to rotation/scale
        max_eig = (0.5 * max(H, W)) ** 2
        if enforce_spd:
            new_covs, new_rot, new_scales = self.ensure_positive_definite_batch(
                new_covs, min_eigenvalue=1e-3, max_eigenvalue=max_eig
            )
        else:
            # Just decompose without enforcement
            new_rot = np.zeros(K, dtype=np.float64)
            new_scales = np.zeros((K, 2), dtype=np.float64)
            for k in range(K):
                eigvals, eigvecs = np.linalg.eigh(new_covs[k])
                new_rot[k] = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
                new_scales[k] = np.sqrt(np.maximum(eigvals, 1e-12))

        new_means[:, 0] = np.clip(new_means[:, 0], 0, H - 1)
        new_means[:, 1] = np.clip(new_means[:, 1], 0, W - 1)

        if self.mask is not None and self.valid_pixels is not None and self.valid_pixels.size > 0:
            mask_bool = self.mask > 0.5
            for k in range(K):
                yi = int(np.clip(round(new_means[k, 0]), 0, H - 1))
                xi = int(np.clip(round(new_means[k, 1]), 0, W - 1))
                if not mask_bool[yi, xi]:
                    ridx = int(self._rng.integers(0, self.valid_pixels.shape[0]))
                    yx = self.valid_pixels[ridx]
                    new_means[k] = np.array([float(yx[0]), float(yx[1])], dtype=np.float64)

        dummy = TwoDGaussians(new_means, new_covs, gaussians.rgb, alpha_mass, new_rot, new_scales)
        S_k_new = self._sum_Sk_chunked(dummy, H, W) + eps
        amp = N_ki / S_k_new[:, None]
        amp = np.maximum(amp, eps)
        alpha_mass = amp.sum(axis=1) + eps
        rgb_norm = amp / alpha_mass[:, None]
        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)

        for idx, col in reinit_rgb.items():
            rgb_norm[idx] = col
        for idx, val in reinit_alpha.items():
            alpha_mass[idx] = val

        rgb_norm = np.maximum(rgb_norm, eps)
        rgb_norm = rgb_norm / (rgb_norm.sum(axis=1, keepdims=True) + eps)
        total_pred = float(np.sum(alpha_mass * S_k_new))
        if np.isfinite(total_pred) and total_pred > eps:
            alpha_mass *= N_total / total_pred

        new_g = TwoDGaussians(new_means, new_covs, rgb_norm, alpha_mass, new_rot, new_scales)
        nll = self.poisson_nll(new_g)
        return new_g, nll

    # =========================================================================
    # S_k computation (sum of phi values)
    # =========================================================================
    def _sum_Sk_chunked(
        self, gaussians: TwoDGaussians, H: int, W: int,
        k_chunk_size: Optional[int] = None
    ) -> np.ndarray:
        """Compute S_k = sum of phi_k values (mask-weighted)."""
        if k_chunk_size is None:
            k_chunk_size = self.default_k_chunk

        K = gaussians.k

        if self.use_torch and HAS_TORCH:
            return self._sum_Sk_torch_chunked(gaussians, H, W, k_chunk_size)

        mask_w = self.mask
        S_k = np.zeros(K, dtype=np.float64)
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            phi = self._gaussian_pdf_numpy(
                gaussians.means[k_start:k_end],
                gaussians.covs[k_start:k_end],
                H, W, normalize_per_k=False
            )
            if mask_w is None:
                S_k[k_start:k_end] = phi.sum(axis=(0, 1))
            else:
                S_k[k_start:k_end] = (phi * mask_w[:, :, None]).sum(axis=(0, 1))
        return S_k

    def _sum_Sk_torch_chunked(
        self, gaussians: TwoDGaussians, H: int, W: int, k_chunk_size: int
    ) -> np.ndarray:
        """PyTorch S_k computation with chunking."""
        with torch.no_grad():
            device = self.device
            K = gaussians.k

            xy = self._get_grid_torch(H, W)
            mask_t = self._get_mask_tensor()

            S_k = np.zeros(K, dtype=np.float64)

            for k_start in range(0, K, k_chunk_size):
                k_end = min(k_start + k_chunk_size, K)

                mean_t = torch.from_numpy(gaussians.means[k_start:k_end].astype(np.float32)).to(device)
                cov_t = torch.from_numpy(gaussians.covs[k_start:k_end].astype(np.float32)).to(device)

                maha, logdet = self._compute_maha_and_logdet_inv_torch(xy, mean_t, cov_t)
                log_phi = -0.5 * maha - np.log(2.0 * np.pi) - 0.5 * logdet.unsqueeze(0).unsqueeze(0)
                phi = torch.exp(log_phi)

                if mask_t is not None:
                    S_k_chunk = (phi * mask_t.unsqueeze(2)).sum(dim=(0, 1))
                else:
                    S_k_chunk = phi.sum(dim=(0, 1))

                S_k[k_start:k_end] = S_k_chunk.cpu().numpy().astype(np.float64)

            return S_k

    # =========================================================================
    # Covariance utilities
    # =========================================================================
    def ensure_positive_definite(
        self, cov: np.ndarray,
        min_eigenvalue: float = 1e-3,
        max_eigenvalue: Optional[float] = None
    ) -> np.ndarray:
        """Ensure covariance is positive definite."""
        cov = 0.5 * (cov + cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        if max_eigenvalue is None:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
        else:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, max_eigenvalue)
        return (eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T).astype(cov.dtype)

    def ensure_positive_definite_batch(
        self, covs: np.ndarray,
        min_eigenvalue: float = 1e-3,
        max_eigenvalue: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch ensure positive definite. Returns (covs, rotations, scales)."""
        covs = 0.5 * (covs + covs.transpose(0, 2, 1))
        eigenvalues, eigenvectors = np.linalg.eigh(covs)

        if max_eigenvalue is None:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
        else:
            eigenvalues = np.clip(eigenvalues, min_eigenvalue, max_eigenvalue)

        new_covs = np.einsum('kij,kj,klj->kil', eigenvectors, eigenvalues, eigenvectors)
        rotations = np.arctan2(eigenvectors[:, 1, 0], eigenvectors[:, 0, 0])
        scales = np.sqrt(np.maximum(eigenvalues, 1e-6))

        return new_covs.astype(covs.dtype), rotations, scales

    def check_positive_definite(self, cov: np.ndarray) -> bool:
        """Check if covariance is positive definite."""
        eigenvalues = np.linalg.eigvals(cov)
        return bool(np.all(eigenvalues > 0))

    def print_covariance_stats(self, new_covs: List[np.ndarray]) -> None:
        """Print covariance statistics."""
        det_values = np.array([np.linalg.det(cov) for cov in new_covs])
        min_eigenvalues = np.array([np.min(np.linalg.eigvals(cov)) for cov in new_covs])
        print(f"Determinant min-max: {np.min(det_values)}, {np.max(det_values)}")
        print(f"Min eigenvalue min-max: {np.min(min_eigenvalues)}, {np.max(min_eigenvalues)}")
        print(f"Positive definite: {np.sum([self.check_positive_definite(cov) for cov in new_covs])}/{len(new_covs)}")

    # =========================================================================
    # Loss computation
    # =========================================================================
    def poisson_nll(self, gaussians: TwoDGaussians, eps: float = 1e-12) -> float:
        """Compute Poisson negative log-likelihood using torch render_rates."""
        rates = self.render_rates_torch(gaussians)

        I = np.clip(self.image, 0.0, None)
        mask_w = self.mask
        if mask_w is not None:
            nll = ((rates - I * np.log(rates + eps)) * mask_w[:, :, None]).sum()
        else:
            nll = (rates - I * np.log(rates + eps)).sum()
        return float(nll)
