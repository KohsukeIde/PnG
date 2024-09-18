import os

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

    def initialize_gaussians(
        self, n_gaussians: int, alpha_0: float = 0.4
    ) -> TwoDGaussians:
        """Initialize Gaussians with naive settings.

        Args:
            n_gaussians (int): Number of Gaussians to initialize.
            alpha_0 (float, optional): Initial alpha value. Defaults to 0.4.

        Returns:
            TwoDGaussians: Initialized Gaussians.
        """
        height, width = self.image.shape[:2]

        # Initialize positions randomly
        means = np.random.rand(n_gaussians, 2) * [height, width]

        # Initialize rotation angles randomly
        # rotation_angles = np.random.uniform(0, 2*np.pi, n_gaussians)

        # Initialize scales with constant for now
        # scale_const = np.sqrt(min(height, width)/10)
        # scale_x = np.full(n_gaussians, scale_const) #[n_gaussian, 1]
        # scale_y = np.full(n_gaussians, scale_const) #[n_gaussian, 1]

        # Initialize covariances with constant for now
        cov_const = min(height, width) / 10  # adjustable  constant
        covs = np.array([np.eye(2) * cov_const for _ in range(n_gaussians)])
        # covs = np.array([TwoDGaussians.params_to_cov(angle, sx, sy) for angle, sx, sy in zip(rotation_angles, scale_x, scale_y)])

        # Initialize RGB values from the image
        rgb = np.array([self.image[int(y), int(x)] for y, x in means])
        print(f"{rgb=}")
        print(f"np.max{rgb=}")
        # Initialize alpha values
        alpha = np.full(n_gaussians, alpha_0)
        alpha /= np.sum(alpha)

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
        cov_inv = np.linalg.pinv(cov)  # (K, 2, 2)
        cov_inv += np.eye(2)[None, :, :] * 1e-6

        cov_det = np.linalg.det(cov)  # (K,)

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
                            np.matmul(np.array([[i, j]]) - mean[:, None, :], cov_inv),
                            np.array([[[i], [j]]]) - mean[:, :, None],
                        )[:, 0, 0]
                    )
                )
        return n

    def e_step(self, gaussians: TwoDGaussians) -> np.ndarray:
        """Compute the responsibilities (gamma) for each pixel and each Gaussian."""
        os.makedirs("debug_output", exist_ok=True)
        height, width = self.image.shape[:2]

        image_pixels = self.image.reshape(-1, 3)

        with open("debug_output/e_step_debug.txt", "w") as f:
            f.write(f"Image shape: {self.image.shape}\n")
            f.write(f"Gaussians: {gaussians}\n\n")
            f.write(
                f"Image pixels shape: {image_pixels.shape}\n\n"
            )  

            # spatial probabilities: N(x,y|μ_k,Σ_k)
            n = self.gaussian_pdf(gaussians.means, gaussians.covs, height, width)
            f.write(f"n shape: {n.shape}\n")
            f.write(f"n min: {n.min()}, max: {n.max()}, mean: {n.mean()}\n\n")

            # color probabilities: ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
            color_prob = np.prod(
                gaussians.rgb[None, None, :, :] ** self.image[:, :, None, :], axis=3
            )
            f.write(f"color_prob shape: {color_prob.shape}\n")
            f.write(
                f"color_prob min: {color_prob.min()}, max: {color_prob.max()}, mean: {color_prob.mean()}\n\n"
            )

            # concatenated probabilities: α_k N(x,y|μ_k,Σ_k) ∏_{i ∈ {r,g,b}} c_{k,i}^{I_{x,y,i}}
            responsibilities = gaussians.alpha[None, None, :] * n * color_prob
            f.write(f"responsibilities shape: {responsibilities.shape}\n")
            f.write(
                f"responsibilities min: {responsibilities.min()}, max: {responsibilities.max()}, mean: {responsibilities.mean()}\n\n"
            )

            # Normalize: γ_{x,y,k} = (concatenated probability) / (sum of concatenated probabilities over all k)
            responsibilities_sum = np.sum(responsibilities, axis=-1, keepdims=True)
            responsibilities_sum = np.maximum(
                responsibilities_sum, 1e-10
            )  # 数値的安定性のため
            responsibilities_reciprocal = np.reciprocal(responsibilities_sum)
            responsibilities = responsibilities * responsibilities_reciprocal

            f.write(
                f"Normalized responsibilities min: {responsibilities.min()}, max: {responsibilities.max()}, mean: {responsibilities.mean()}\n\n"
            )

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
        k = gaussians.k

        # Compute sum of responsibilities for each Gaussian
        n_k = np.sum(gamma, axis=(0, 1))  # shape: (k,)
        # processing Gaussians with low responsibility
        responsibility_threshold = 1e-6
        small_responsibility_indices = np.where(n_k < responsibility_threshold)[0]

        if len(small_responsibility_indices) > 0:
            print(
                f"Resetting {len(small_responsibility_indices)} Gaussians with small responsibilities"
            )
            for idx in small_responsibility_indices:
                # assign random means and covariances
                gaussians.means[idx] = np.random.rand(2) * [height, width]
                gaussians.covs[idx] = np.eye(2) * min(height, width) / 10
                gaussians.rgb[idx] = self.image[
                    int(gaussians.means[idx, 0]), int(gaussians.means[idx, 1])
                ]
            # recompute responsibility
            gamma = self.e_step(gaussians)

        n_k = np.sum(gamma, axis=(0, 1))
        n_k = np.maximum(n_k, 1e-10)  # avoid division by zero
        n_k_reciprocal = np.reciprocal(n_k)
        # print(f"nk min: {n_k.min()}, max: {n_k.max()}, mean: {n_k.mean()}")
        # small_nk_count = np.sum(n_k < 1e-6)
        # print(f"Number of Gaussians with nk < 1e-6: {small_nk_count}")

        # Create meshgrid for x and y coordinates
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        xy = np.stack([y, x], axis=-1)  # shape: (height, width, 2)

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

        # Apply constraints to covariance matrices
        # params = [TwoDGaussians.cov_to_params(cov) for cov in new_covs]
        # new_rotation_angles, new_scale_x, new_scale_y = zip(*params)

        # Ensure covariance matrices are positive definite
        epsilon = 1e-6
        for i in range(k):
            min_eig = np.min(np.real(np.linalg.eigvals(new_covs[i])))
            if min_eig < epsilon:
                new_covs[i] += (epsilon - min_eig) * np.eye(2)

        # Update mixing coefficients (alpha)
        # α_k' = Σ_{x,y,i} I_{x,y,i} * γ_{x,y,k} / Σ_{x,y,i} I_{x,y,i}
        pixel_sum = np.sum(self.image)
        pixel_sum_reciprocal = np.reciprocal(pixel_sum)
        new_alpha = np.sum(
            np.sum(self.image[:, :, :, None] * gamma[:, :, None, :], axis=2),
            axis=(0, 1),
        )
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
        # new_colors = new_colors * n_k_reciprocal[:, None]

        # new_colors = np.zeros((k, 3))
        # for i in range(k):
        #     for y in range(height):
        #         for x in range(width):
        #             new_colors[i] = (
        #                 new_colors[i] + gamma[y, x, i] * self.image[y, x]
        #             )
        # new_colors = new_colors * n_k_reciprocal[:, None]

        print(f"New means min-max: {np.min(new_means)}, {np.max(new_means)}")
        print(f"New covs min-max: {np.min(new_covs)}, {np.max(new_covs)}")
        print(f"New colors min-max: {np.min(new_colors)}, {np.max(new_colors)}")
        print(f"New alpha min-max: {np.min(new_alpha)}, {np.max(new_alpha)}")

        return TwoDGaussians(new_means, new_covs, new_colors, new_alpha)
