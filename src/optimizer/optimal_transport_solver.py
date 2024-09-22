import numpy as np

from src.primitive.twod_gaussians import TwoDGaussians


class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians."""

    def __init__(
        self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians, epsilon: float = 0.1
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            epsilon (float): Entropy regularization parameter. Defaults to 0.1.
        """
        self.gaussians1 = gaussians1
        self.gaussians2 = gaussians2
        self.epsilon = epsilon

    def compute_cost_matrix(self) -> np.ndarray:
        """Compute the cost matrix for optimal transport.

        Returns:
            np.ndarray: The computed cost matrix.
        """
        # TODO: Implement cost matrix computation
        k1, k2 = self.gaussians1.k, self.gaussians2.k
        return np.zeros((k1, k2))  # Placeholder

    def sinkhorn_algorithm(self) -> np.ndarray:
        """Implement the Sinkhorn algorithm for optimal transport.

        Returns:
            np.ndarray: The transport plan matrix.
        """
        # TODO: Implement Sinkhorn algorithm
        k1, k2 = self.gaussians1.k, self.gaussians2.k
        return np.zeros((k1, k2))  # Placeholder
