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

        # Initialize covariances with constant for now
        cov_const = min(height, width) / 10  # You can adjust this constant
        covs = np.array([np.eye(2) * cov_const for _ in range(n_gauss)])

        # Initialize RGB values from the image
        rgb = np.array([self.image[int(y), int(x)] for y, x in means])

        # Initialize alpha values
        alpha = np.full(n_gauss, alpha_0)

        return TwoDGaussians(means, covs, rgb, alpha)


    def gaussian_pdf(self, mean, cov, height, width) -> np.ndarray:
        """
        Compute the Gaussian PDF for multiple points and multiple Gaussians.

        Args:
            mean (np.ndarray): Means of Gaussians, shape (K, 2)
            cov (np.ndarray): Covariance matrices, shape (K, 2, 2)
            height (int): Height of the image
            width (int): Width of the image

        Returns:
            np.ndarray: Gaussian PDF values, shape (height, width, K)
        """
        K = mean.shape[0]
        cov_inv = np.linalg.inv(cov)  # (K, 2, 2)
        cov_det = np.linalg.det(cov)  # (K,)

        y, x = np.mgrid[0:height, 0:width]
        xy = np.stack([x, y], axis=-1)  # (height, width, 2)

        N = np.zeros((height, width, K))

        for i in range(height):
            for j in range(width):
                xy_m = xy[i, j] - mean  # (K, 2)
                N[i, j, :] = 1.0 / np.sqrt(2 * np.pi * cov_det) * np.exp(
                    np.matmul(np.matmul(xy_m[:, None, :], cov_inv), xy_m[:, :, None])[:, 0, 0]
                )

        return N
    
    def e_step(self, gaussians: TwoDGaussians) -> np.ndarray:
        """
        Compute the responsibilities (gamma) for each pixel and each Gaussian.

        Args:
            gaussians (TwoDGaussians): The current Gaussian mixture model.

        Returns:
            np.ndarray: Responsibilities with shape (height, width, k).
        """
        height, width = self.image.shape[:2]

        # Compute spatial probabilities: N(x,y|μ_k,Σ_k)
        spatial_probs = self.gaussian_pdf(gaussians.means, gaussians.covs, height, width)

        # Compute color probabilities: ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
        color_probs = np.prod(gaussians.rgb[:, np.newaxis, np.newaxis, :] ** self.image, axis=-1)

        # Compute joint probabilities: α_k N(x,y|μ_k,Σ_k) ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
        responsibilities = gaussians.alpha[:, np.newaxis, np.newaxis] * spatial_probs * color_probs

        # Normalize: γ_{x,y,k} = (joint probability) / (sum of joint probabilities over all k)
        responsibilities /= np.sum(responsibilities, axis=-1, keepdims=True)

        return responsibilities