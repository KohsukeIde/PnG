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

    def gaussian_pdf(
        self, mean: np.ndarray, cov: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Compute the Gaussian PDF for multiple points and multiple Gaussians.

        Args:
            mean (np.ndarray): Means of Gaussians, shape (K, 2)
            cov (np.ndarray): Covariance matrices, shape (K, 2, 2)
            height (int): Height of the image
            width (int): Width of the image

        Returns:
            np.ndarray: Gaussian PDF values, shape (height, width, K)
        """
        k = mean.shape[0]
        cov_inv = np.linalg.inv(cov)  # (K, 2, 2)
        cov_det = np.linalg.det(cov)  # (K,)

        # y, x = np.mgrid[0:height, 0:width]
        n = np.zeros((height, width, k))

        for i in range(height):
            for j in range(width):
                # xy = np.array([[j, i]])  # Note: x corresponds to j, y to i
                # xy_m = xy - mean[:, None, :]  # (K, 1, 2)

                # print(f"xy shape: {xy.shape}")
                # print(f"xy_m shape: {xy_m.shape}")
                # print(f"cov_inv shape: {cov_inv.shape}")

                # temp1 = np.matmul(xy_m, cov_inv)  # (K, 1, 2)
                # print(f"temp1 shape: {temp1.shape}")

                # temp2 = np.array([[[j], [i]]]) - mean[:, :, None]  # (K, 2, 1)
                # print(f"temp2 shape: {temp2.shape}")

                # maha = np.matmul(temp1, temp2)[:, 0, 0]  # (K,)
                # print(f"maha shape: {maha.shape}")

                n[i, j, :] = (
                    1.0
                    / np.sqrt(2 * np.pi * cov_det)
                    * np.exp(
                        -0.5
                        * np.matmul(
                            np.matmul(np.array([[j, i]]) - mean[:, None, :], cov_inv),
                            np.array([[[j], [i]]]) - mean[:, :, None],
                        )[:, 0, 0]
                    )
                )

        return n

    def e_step(self, gaussians: TwoDGaussians) -> np.ndarray:
        """Compute the responsibilities (gamma) for each pixel and each Gaussian.

        Args:
            gaussians (TwoDGaussians): The current Gaussian mixture model.

        Returns:
            ndarray: Responsibilities, shape (height, width, K).
        """
        height, width = self.image.shape[:2]

        # spatial probabilities: N(x,y|μ_k,Σ_k)
        n = self.gaussian_pdf(gaussians.means, gaussians.covs, height, width)

        # color probabilities: ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
        c_sum = np.sum(gaussians.rgb ** self.image[:, :, None, :], axis=-1)

        # concatenated probabilities: α_k N(x,y|μ_k,Σ_k) ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
        responsibilities = gaussians.alpha[None, None, :] * n * c_sum

        # Normalize: γ_{x,y,k} = (joint probability) / (sum of concatenated probabilities over all k)
        sum_reciprocal = np.reciprocal(np.sum(responsibilities, axis=-1, keepdims=True))
        responsibilities = responsibilities * sum_reciprocal

        # responsibilities = responsibilities.astype(np.float64)
        assert isinstance(responsibilities, np.ndarray)
        return responsibilities

    def m_step(self, gamma: np.ndarray, gaussians: TwoDGaussians) -> TwoDGaussians:
        """Update the parameters of the Gaussian mixture model.

        Args:
            gamma (ndarray): Responsibilities, shape (height, width, K).
            gaussians (TwoDGaussians): Current Gaussian mixture model.

        Returns:
            TwoDGaussians: Updated Gaussian mixture model.
        """
        height, width = self.image.shape[:2]
        # k = gaussians.k

        # Compute sum of responsibilities for each Gaussian
        n_k = np.sum(gamma, axis=(0, 1))  # shape: (k,)
        n_k_reciprocal = np.reciprocal(n_k)

        # Create meshgrid for x and y coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xy = np.stack([x, y], axis=-1)  # shape: (height, width, 2)

        # Update means
        # μ_k' = Σ_{x,y} γ_{x,y,k} * (x,y) / Σ_{x,y} γ_{x,y,k}

        new_means = np.sum(gamma[:, :, :, None] * xy[:, :, None, :], axis=(0, 1))
        new_means = new_means * n_k_reciprocal[:, None]

        # new_means = np.zeros((k, 2))
        # for i in range(k):
        #     for y in range(height):
        #         for x in range(width):
        #             new_means[i] = new_means[i] +gamma[y, x, i] * np.array(
        #                 [x, y]
        #             )
        # new_means = new_means * n_k_reciprocal[:, None]

        # Update covariances
        # Σ_k' = Σ_{x,y} γ_{x,y,k} * ((x,y) - μ_k')((x,y) - μ_k')^T / Σ_{x,y} γ_{x,y,k}
        diff = xy[:, :, None, :] - new_means[None, None, :, :]
        new_covs = np.einsum("ijkl,ijkm,ijk->klm", diff, diff, gamma)
        new_covs = new_covs * n_k_reciprocal[:, None, None]

        # new_covs = np.zeros((k, 2, 2))
        # for i in range(k):
        #     for y in range(height):
        #         for x in range(width):
        #             diff = np.array([x, y]) - new_means[i]
        #             new_covs[i] = new_covs[i] + gamma[y, x, i] * np.outer(
        #                 diff, diff
        #             )
        # new_covs = new_covs * n_k_reciprocal[:, None, None]

        # Update mixing coefficients (alpha)
        # α_k' = Σ_{x,y,i} I_{x,y,i} * γ_{x,y,k} / Σ_{x,y,i} I_{x,y,i}
        pixel_sum = np.sum(self.image)
        pixel_sum_reciprocal = np.reciprocal(pixel_sum)
        new_alpha = np.sum(np.sum(self.image, axis=2)[:, :, None] * gamma, axis=(0, 1))
        new_alpha = new_alpha * pixel_sum_reciprocal

        # pixel_sum = np.sum(self.image)
        # pixel_sum_reciprocal = np.reciprocal(pixel_sum)
        # new_alpha = np.zeros(k)
        # for i in range(k):
        #     for y in range(height):
        #         for x in range(width):
        #             new_alpha[i] = (
        #                 new_alpha[i]
        #                 + np.sum(self.image[y, x]) * gamma[y, x, i]
        #             )
        # new_alpha = new_alpha * pixel_sum_reciprocal

        # Update colors
        # c_k' = Σ_{x,y} γ_{x,y,k} * I_{x,y} / Σ_{x,y} γ_{x,y,k}
        new_colors = np.sum(
            self.image[:, :, None, :] * gamma[:, :, :, None], axis=(0, 1)
        )
        new_colors = new_colors * n_k_reciprocal[:, None]

        # new_colors = np.zeros((k, 3))
        # for i in range(k):
        #     for y in range(height):
        #         for x in range(width):
        #             new_colors[i] = (
        #                 new_colors[i] + gamma[y, x, i] * self.image[y, x]
        #             )
        # new_colors = new_colors * n_k_reciprocal[:, None]

        return TwoDGaussians(new_means, new_covs, new_colors, new_alpha)
