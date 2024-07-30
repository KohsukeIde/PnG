import cv2
import numpy as np

from src.primitive.twod_gaussians import TwoDGaussians


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

    def rasterize(
        self, gaussians: TwoDGaussians, save_to_file: bool = False
    ) -> np.ndarray:
        """Rasterize image from input 2D gaussians.

        Args:
            gaussians (TwoDGaussians): rasterize target gaussians
            save_to_file (bool): Flag for debug rasterized image.
                                 If true, raisterized image will write to `output/tmp.png`

        Returns:
            np.ndarray: rendered image with shape [height, width, 3]

        Raises:
            ValueError: If covariance take negative
        """
        # Position of each pixel[height * width, 2]
        xy = (
            np.mgrid[0 : self.height, 0 : self.width]
            .astype(np.float64)
            .reshape(2, -1)
            .transpose(1, 0)
        )
        # Initialize image with 0
        img = np.zeros((self.height * self.width, 3), np.float64)

        # print(f"Number of Gaussians: {gaussians.k}")
        # print(f"Means shape: {gaussians.means.shape}")
        # print(f"Covs shape: {gaussians.covs.shape}")
        # print(f"RGB shape: {gaussians.rgb.shape}")
        # print(f"Alpha shape: {gaussians.alpha.shape}")

        for k in range(gaussians.k):
            cov_det = np.linalg.det(gaussians.covs[k, :, :])
            if cov_det < 0:
                print(f"Warning: Negative covariance determinant for Gaussian {k}")
                continue

            cov_inv = np.linalg.inv(gaussians.covs[k, :, :])
            # Scaled Color (ndarray[1, 3])
            scaled_color = (
                0.5
                / (np.pi * np.sqrt(cov_det))
                * gaussians.alpha[k]
                * gaussians.rgb[k, None, :]
            )
            # scaled_color = (
            #     gaussians.alpha[k]
            #     * gaussians.rgb[k, None, :]
            # )
            # print(f"Gaussian {k}:")
            # print(f"  Mean: {gaussians.means[k]}")
            # print(f"  Cov: {gaussians.covs[k]}")
            # print(f"  RGB: {gaussians.rgb[k]}")
            # print(f"  Alpha: {gaussians.alpha[k]}")
            # print(f"  Scaled color: {scaled_color}")

            # Coordinates with gaussian's mean as origin (ndarray[height * width, 2])
            xy_k = xy - gaussians.means[k, None, :]
            # Normalized coordinates (ndarray[height*width, 2])
            xy_n = np.sum(np.matmul(xy_k, cov_inv) * xy_k, axis=1, keepdims=True)
            img += scaled_color * np.exp(-xy_n)

        # Reshape and convert to save image
        img = img * 255 # 0-255 scaling
        img_cv = img.reshape(self.height, self.width, 3).clip(0, 255).astype(np.uint8)
        if save_to_file:
            cv2.imwrite("outputs/tmp.png", img_cv)

        return img_cv
