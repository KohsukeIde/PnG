import numpy as np
import cv2

from src.primitive.twod_gaussians import TwoDGaussians

class Vanilla2DRasterizer:
    """Rasterize 2D Gaussians to rgb image
    This class is naive(slow) implementation of GS rasterizer

    Attributes:
        width (int): rasterize target image width
        height (int): rasterize target image height
    """
    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def rasterize(self, gaussians: TwoDGaussians, save_to_file: bool = False) -> np.ndarray:
        """Rasterize image from input 2D gaussians

        Args:
            gaussians (TwoDGaussians): rasterize target gaussians
            save_to_file (bool): Flag for debug rasterized image.
                                 If true, raisterized image will write to `output/tmp.png`

        Returns:
            np.ndarray: rendered image with shape [height, width, 3]
        """
        # position of each pixel[height * width, 2]
        xy = np.mgrid[0:self.height, 0:self.width].astype(np.float64).reshape(2, -1).transpose(1, 0)
        img = np.zeros((self.height * self.width, 3), np.float64)
        for k in range(gaussians.k):
            cov_det = np.linalg.det(gaussians.covs[k, :, :])
            cov_inv = np.linalg.pinv(gaussians.covs[k, :, :])
            img += 0.5 / np.pi / np.sqrt(cov_det) * gaussians.rgb[k, None, :] * np.exp(- 0.5 * (np.matmul(xy - gaussians.means[k, None, :], cov_inv)*(xy - gaussians.means[k, None, :])).sum(1)[:, None])
        img_cv = img.reshape(self.height, self.width, 3).astype(np.uint8)
        cv2.imwrite("outputs/tmp.png", img_cv)


if __name__ == "__main__":
    rasterizer = Vanilla2DRasterizer(100, 100)
    k = 512
    obj = TwoDGaussians(
        np.random.rand(k, 2) * 100,
        np.random.rand(k, 2, 2) * 4,
        np.random.rand(k, 3) * 255,
        np.random.rand(k)*10,
    )
    obj.covs[:, 0, 0] = np.abs(obj.covs[:, 0, 0])
    obj.covs[:, 1, 1] = np.abs(obj.covs[:, 1, 1])
    obj.covs[:, 0, 1] = -obj.covs[:, 1, 0]
    rasterizer.rasterize(obj, True)