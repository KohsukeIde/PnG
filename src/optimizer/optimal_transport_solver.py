import copy

import numpy as np
from scipy.linalg import sqrtm

from src.primitive.twod_gaussians import TwoDGaussians


class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        epsilon: float = 0.1,
        lambda_color: float = 1.0,
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            epsilon (float): Entropy regularization parameter. Defaults to 0.1.
            lambda_color (float): Weight for color difference in cost calculation. Defaults to 1.0.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)
        self.epsilon = epsilon
        self.lambda_color = lambda_color

    def compute_cost_matrix(self) -> np.ndarray:
        """Compute the cost matrix between two sets of 2D Gaussians.

        Returns:
            np.ndarray: Cost matrix of shape (K1, K2) where K1 and K2 are the number of Gaussians in each set.
        """
        k1, k2 = self.gaussians1.k, self.gaussians2.k
        cost_matrix = np.zeros((k1, k2))

        for i in range(k1):
            for j in range(k2):
                wasserstein_sq = self._wasserstein_distance(
                    self.gaussians1.means[i],
                    self.gaussians1.covs[i],
                    self.gaussians2.means[j],
                    self.gaussians2.covs[j],
                )
                color_diff_sq = np.sum(
                    (self.gaussians1.rgb[i] - self.gaussians2.rgb[j]) ** 2
                )
                cost = wasserstein_sq + self.lambda_color * color_diff_sq
                cost_matrix[i, j] = cost

                reverse_cost = cost_matrix[j, i] if j < k1 else "N/A"
                print(
                    f"Cost[{i},{j}]: {cost:.6f}, Reverse Cost[{j},{i}]: {reverse_cost}"
                )
                # print(f"Cost[{i},{j}]:")
                # print(f"  Means1: {self.gaussians1.means[i]}, Means2: {self.gaussians2.means[j]}")
                # print(f"  Covs1: {self.gaussians1.covs[i]}, Covs2: {self.gaussians2.covs[j]}")
                # print(f"  RGB1: {self.gaussians1.rgb[i]}, RGB2: {self.gaussians2.rgb[j]}")
                # print(f"  Wasserstein_sq: {wasserstein_sq}")
                # print(f"  Color_diff_sq: {color_diff_sq}")
                # print(f"  Total cost: {cost_matrix[i,j]}")
                # print()
        return cost_matrix

    def _wasserstein_distance(
        self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    ) -> float:
        """Compute the squared 2-Wasserstein distance between two Gaussian distributions.

        Args:
            mu1 (np.ndarray): Mean of the first Gaussian.
            sigma1 (np.ndarray): Covariance matrix of the first Gaussian.
            mu2 (np.ndarray): Mean of the second Gaussian.
            sigma2 (np.ndarray): Covariance matrix of the second Gaussian.

        Returns:
            float: Squared 2-Wasserstein distance between the two Gaussians.
        """
        # print(f"mu1 shape: {mu1.shape}, mu2 shape: {mu2.shape}")
        # print(f"sigma1 shape: {sigma1.shape}, sigma2 shape: {sigma2.shape}")

        # Compute the squared norm of the mean difference
        diff_means = np.sum((mu1 - mu2) ** 2)

        # Compute the matrix square root of sigma2
        sqrt_sigma2 = self._matrix_sqrt(sigma2)

        # Compute the intermediate matrix
        intermediate = sqrt_sigma2 @ sigma1 @ sqrt_sigma2

        # Compute the matrix square root of the intermediate matrix
        sqrt_intermediate = self._matrix_sqrt(intermediate)

        # Compute the trace term
        trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_intermediate)

        return float(diff_means + trace_term)

    # def _matrix_sqrt(self, matrix):
    #     """
    #     Compute the square root of a matrix.

    #     Args:
    #         matrix (np.ndarray): Input matrix.

    #     Returns:
    #         np.ndarray: Square root of the input matrix.
    #     """
    #     eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    #     sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))  # Ensure non-negative eigenvalues
    #     return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T

    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the square root of a matrix using scipy's sqrtm for better numerical stability.

        Args:
            matrix (np.ndarray): Input matrix.

        Returns:
            np.ndarray: Square root of the input matrix.
        """
        sqrt_matrix = sqrtm(matrix)
        # sqrtm may return complex numbers; take the real part if the imaginary part is negligible
        if np.iscomplexobj(sqrt_matrix):
            # if not np.allclose(np.imag(sqrt_matrix), 0, atol=1e-10):
            #     raise ValueError("Matrix square root resulted in significant imaginary components.")
            sqrt_matrix = np.real(sqrt_matrix)
        return np.array(sqrt_matrix)

    def sinkhorn_algorithm(self) -> np.ndarray:
        """Implement the Sinkhorn algorithm for optimal transport.

        Returns:
            np.ndarray: The transport plan matrix.
        """
        # TODO: Implement Sinkhorn algorithm
        k1, k2 = self.gaussians1.k, self.gaussians2.k
        return np.zeros((k1, k2))  # Placeholder
