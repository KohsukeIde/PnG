import cv2
import numpy as np

from src.primitive.twod_gaussians_rs import TwoDGaussians


class Vanilla2DRasterizer:
    """Rasterize 2D Gaussians to rgb image.
    This class is naive(slow) implementation of GS rasterizer.

    Attributes:
        width (int): rasterize target image width
        height (int): rasterize target image height
    """

    def __init__(self, height: int, width: int) -> None:
        """Initialize the Vanilla2DRasterizer with rasterize target image size.

        Args:
            height (int): rasterize target image height
            width (int): rasterize target image width
        """
        self.height = height
        self.width = width

    def render_rates(self, gaussians: TwoDGaussians) -> np.ndarray:
        """Render Poisson rate image: Σ_k α_k ρ_k φ_k(x,y).
        
        Args:
            gaussians (TwoDGaussians): rasterize target gaussians
            
        Returns:
            np.ndarray: rate image with shape [height, width, 3]
        """
        H, W = self.height, sef.width
        xy = (np.mgrid[0:H, 0:W].astype(np.float64)
              .reshape(2, -1).T)  # (H*W, 2)
        rates = np.zeros((H * W, 3), np.float64)
        eps = 1e-12

        for k in range(gaussians.k):
            cov = gaussians.covs[k] + eps * np.eye(2)  # Add regularization
            cov_det = np.linalg.det(cov)
            if cov_det < eps:
                raise ValueError("Covariance determinant should be positive.")
            cov_inv = np.linalg.inv(cov)
            
            # Compute spatial distribution φ_k(x,y) = continuous 2D Gaussian
            xy_k = xy - gaussians.means[k, None, :]  # (H*W, 2)
            maha = np.sum((xy_k @ cov_inv) * xy_k, axis=1, keepdims=True)  # (H*W, 1)
            
            # φ_k(x,y) = 1/(2π√|Σ|) * exp(-0.5 * maha) - continuous normalization
            phi_k = np.exp(-0.5 * maha) / (2.0 * np.pi * np.sqrt(cov_det))  # (H*W, 1)
            
            # rates += α_k * ρ_k * φ_k
            rates += (gaussians.alpha[k] * phi_k) * gaussians.rgb[k, None, :]

        return rates.reshape(H, W, 3)

    def rasterize(
        self, gaussians: TwoDGaussians, save_to_file: bool = False
    ) -> np.ndarray:
        """Rasterize image from input 2D Gaussians using Poisson rate model.

        Args:
            gaussians (TwoDGaussians): rasterize target gaussians
            save_to_file (bool): Flag for debug rasterized image.

        Returns:
            np.ndarray: rendered image with shape [height, width, 3], values in [0, 255]

        Raises:
            ValueError: If covariance determinant is non-positive
        """
        # Get Poisson rate image λ(x,y,i) = Σ_k α_k ρ_k φ_k(x,y)
        rates = self.render_rates(gaussians)  # (H, W, 3)
        
        # Clip rates to [0, 1] for display (no external scaling needed)
        img = np.clip(rates, 0.0, 1.0)
        
        # Convert to 8-bit image
        img_cv = (img * 255).astype(np.uint8)
        
        if save_to_file:
            cv2.imwrite("outputs/tmp.png", img_cv)

        return img_cv
