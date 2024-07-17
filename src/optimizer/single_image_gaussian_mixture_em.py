import numpy as np
from PIL import Image

from src.primitive.twod_gaussians import TwoDGaussians


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

    def initialize_gaussians(self, n_gauss: int, alpha_0: float = 0.4) -> TwoDGaussians:
        """Initialize Gaussians with naive settings.

        Args:
            n_gauss (int): Number of Gaussians to initialize.
            alpha_0 (float, optional): Initial alpha value. Defaults to 0.4.

        Returns:
            TwoDGaussians: Initialized Gaussians.
        """
        height, width = self.image.shape[:2]

        # Initialize positions randomly
        means = np.random.rand(n_gauss, 2) * [height, width]

        # Initialize covariances with const
        cov_const = min(height, width) / 10  # You can adjust this constant
        covs = np.array([np.eye(2) * cov_const for _ in range(n_gauss)])

        # Initialize RGB values from the image
        rgb = np.array([self.image[int(y), int(x)] for y, x in means])

        # Initialize alpha values
        alpha = np.full(n_gauss, alpha_0)

        return TwoDGaussians(means, covs, rgb, alpha)
