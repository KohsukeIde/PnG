import copy
from typing import Tuple

import numpy as np
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians


class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        epsilon: float = 0.1,
        lambda_pos: float = 0.3,
        lambda_color: float = 1.0,
        lambda_shape: float = 1.0,
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            epsilon (float): Entropy regularization parameter. Defaults to 0.1.
            lambda_pos (float): Weight for position difference in cost calculation. Defaults to 1.0.
            lambda_color (float): Weight for color difference in cost calculation. Defaults to 1.0.
            lambda_shape (float): Weight for shape difference in cost calculation. Defaults to 1.0.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)
        self.epsilon = epsilon
        self.lambda_pos = lambda_pos
        self.lambda_color = lambda_color
        self.lambda_shape = lambda_shape

    def compute_cost_matrix(self) -> np.ndarray:
        """Compute the cost matrix between two sets of 2D Gaussians using scales and rotations.

        Returns:
            np.ndarray: Cost matrix of shape (K1, K2), where K1 and K2 are the number of Gaussians in each set.
        """
        k1, k2 = self.gaussians1.k, self.gaussians2.k
        cost_matrix = np.zeros((k1, k2))

        total_iterations = k1 * k2

        with tqdm(
            total=total_iterations, desc="Computing cost matrix", unit="pair"
        ) as pbar:
            for i in range(k1):
                mu1 = self.gaussians1.means[i]
                s1 = self.gaussians1.scales[i]
                theta1 = self.gaussians1.rotations[i]

                for j in range(k2):
                    mu2 = self.gaussians2.means[j]
                    s2 = self.gaussians2.scales[j]
                    theta2 = self.gaussians2.rotations[j]

                    # Compute the squared 2-Wasserstein distance components
                    D_pos, D_shape = self._wasserstein_distance(
                        mu1, s1, theta1, mu2, s2, theta2
                    )

                    # Compute the squared color difference
                    D_color = np.sum(
                        (self.gaussians1.rgb[i] - self.gaussians2.rgb[j]) ** 2
                    )

                    # Total cost with weights
                    cost = (
                        self.lambda_pos * D_pos
                        + self.lambda_shape * D_shape
                        + self.lambda_color * D_color
                    )
                    cost_matrix[i, j] = cost
                    pbar.update(1)

        return cost_matrix

    def _construct_covariance(self, scales: np.ndarray, rotation: float) -> np.ndarray:
        """Construct the covariance matrix from scales and rotation.

        Args:
            scales (np.ndarray): Scales [s_x, s_y].
            rotation (float): Rotation angle in radians.

        Returns:
            np.ndarray: Covariance matrix of shape (2, 2).
        """
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        # Rotation matrix
        r = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        # Scale matrix
        s = np.diag(scales**2)
        # Covariance matrix
        covariance = r @ s @ r.T
        return np.array(covariance)

    def _matrix_sqrt_2x2(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the square root of a 2x2 symmetric positive-definite matrix analytically.

        Args:
            matrix (np.ndarray): Input 2x2 symmetric positive-definite matrix.

        Returns:
            np.ndarray: Square root of the input matrix.
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # Ensure non-negative eigenvalues
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0))
        # Reconstruct the square root matrix
        sqrt_matrix = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
        return np.array(sqrt_matrix)

    def _wasserstein_distance(
        self,
        mu1: np.ndarray,
        s1: np.ndarray,
        theta1: float,
        mu2: np.ndarray,
        s2: np.ndarray,
        theta2: float,
    ) -> Tuple[float, float]:
        """Compute the squared 2-Wasserstein distance components between two Gaussians.

        Args:
            mu1 (np.ndarray): Mean of the first Gaussian (shape: (2,)).
            s1 (np.ndarray): Scales of the first Gaussian [s_x, s_y].
            theta1 (float): Rotation angle of the first Gaussian in radians.
            mu2 (np.ndarray): Mean of the second Gaussian (shape: (2,)).
            s2 (np.ndarray): Scales of the second Gaussian [s_x, s_y].
            theta2 (float): Rotation angle of the second Gaussian in radians.

        Returns:
            Tuple[float, float]: Tuple containing squared mean difference and trace term (shape difference).
        """
        # Compute the squared norm of the mean difference
        diff_means = np.sum((mu1 - mu2) ** 2)

        # Construct covariance matrices
        sigma1 = self._construct_covariance(s1, theta1)
        sigma2 = self._construct_covariance(s2, theta2)

        # Compute the square root of sigma1
        sqrt_sigma1 = self._matrix_sqrt_2x2(sigma1)

        # Compute the intermediate matrix
        intermediate = sqrt_sigma1 @ sigma2 @ sqrt_sigma1

        # Compute the square root of the intermediate matrix
        sqrt_intermediate = self._matrix_sqrt_2x2(intermediate)

        # Compute the trace term
        trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_intermediate)

        # Return the components separately
        return diff_means, trace_term

    def sinkhorn_algorithm(
        self,
        cost_matrix: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Implement the Sinkhorn algorithm for optimal transport.

        Args:
            cost_matrix (np.ndarray): The cost matrix.
            max_iter (int): Maximum number of iterations. Defaults to 100.
            tol (float): Convergence tolerance. Defaults to 1e-6.

        Returns:
            np.ndarray: The transport plan matrix.
        """
        # Initialize the Gibbs kernel
        k = np.exp(-cost_matrix / self.epsilon)

        # Initialize scaling vectors
        u = np.ones(self.gaussians1.k)
        v = np.ones(self.gaussians2.k)

        # Normalize alpha and beta
        alpha = self.gaussians1.alpha / np.sum(self.gaussians1.alpha)
        beta = self.gaussians2.alpha / np.sum(self.gaussians2.alpha)

        for _ in tqdm(range(max_iter), desc="Sinkhorn iterations"):
            # Update u
            denominator = k @ v
            denominator = np.maximum(denominator, 1e-16)  # Prevent division by zero
            u_new = alpha / denominator

            # Update v
            denominator = k.T @ u_new
            denominator = np.maximum(denominator, 1e-16)  # Prevent division by zero
            v_new = beta / denominator

            # Check for convergence
            if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
                break

            u, v = u_new, v_new

        # Compute the transport plan
        p = np.diag(u) @ k @ np.diag(v)
        p = np.asarray(p, dtype=np.float64)
        return np.array(p)
