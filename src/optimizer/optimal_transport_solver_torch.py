import copy
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.primitive.twod_gaussians import TwoDGaussians
from utils.adopt import ADOPT



class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians with Homography Optimization."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        K1: Optional[np.ndarray] = None,
        K2: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
        lambda_pos: float = 0.3,
        lambda_color: float = 1.0,
        lambda_shape: float = 1.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            K1 (Optional[np.ndarray]): Intrinsic parameters of the first camera (3x3).
            K2 (Optional[np.ndarray]): Intrinsic parameters of the second camera (3x3).
            epsilon (float): Entropy regularization parameter. Defaults to 0.1.
            lambda_pos (float): Weight for position difference in cost calculation. Defaults to 0.3.
            lambda_color (float): Weight for color difference in cost calculation. Defaults to 1.0.
            lambda_shape (float): Weight for shape difference in cost calculation. Defaults to 1.0.
            device (torch.device): Device to perform computations on.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)

        self.K1 = torch.tensor(K1, dtype=torch.float32, device=device) if K1 is not None else None
        self.K2 = torch.tensor(K2, dtype=torch.float32, device=device) if K2 is not None else None
        self.F = None  # Will be initialized in the optimization process

        self.epsilon = epsilon
        self.lambda_pos = lambda_pos
        self.lambda_color = lambda_color
        self.lambda_shape = lambda_shape

        self.device = device
        print(f"Using {device} device")


        # Convert Gaussian parameters to torch tensors
        self._prepare_gaussians()

    def _prepare_gaussians(self):
        """Prepare Gaussian parameters as torch tensors."""
        self.means1 = torch.tensor(
            self.gaussians1.means, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.scales1 = torch.tensor(
            self.gaussians1.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.rotations1 = torch.tensor(
            self.gaussians1.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)
        self.rgb1 = torch.tensor(
            self.gaussians1.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 3)
        self.alpha1 = torch.tensor(
            self.gaussians1.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)

        self.means2 = torch.tensor(
            self.gaussians2.means, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.scales2 = torch.tensor(
            self.gaussians2.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.rotations2 = torch.tensor(
            self.gaussians2.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)
        self.rgb2 = torch.tensor(
            self.gaussians2.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 3)
        self.alpha2 = torch.tensor(
            self.gaussians2.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)

    def compute_cost_matrix(self, F: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using homography.

        Args:
            F (torch.Tensor): The projection transformation matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
        K1, K2 = self.means1.shape[0], self.means2.shape[0]

        # Prepare homogeneous coordinates for means1
        ones = torch.ones((K1, 1), dtype=torch.float32, device=self.device)
        means1_h = torch.cat([self.means1, ones], dim=1)  # Shape: (K1, 3)

        # Apply transformation F to means1
        transformed_means1_h = (F @ means1_h.T).T  # Shape: (K1, 3)

        # Convert back to inhomogeneous coordinates
        transformed_means1 = transformed_means1_h[:, :2] / transformed_means1_h[:, 2].unsqueeze(1)  # Shape: (K1, 2)

        # Compute pairwise distances between transformed_means1 and means2
        diff = transformed_means1.unsqueeze(1) - self.means2.unsqueeze(0)  # Shape: (K1, K2, 2)
        D_pos = torch.sum(diff ** 2, dim=2)  # Shape: (K1, K2)

        # Compute shape differences using Wasserstein distance
        D_shape = self._wasserstein_distance(transformed_means1, self.scales1, self.rotations1,
                                             self.means2, self.scales2, self.rotations2)  # Shape: (K1, K2)

        # Compute color differences
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # Shape: (K1, K2, 3)
        D_color = torch.sum(color_diff ** 2, dim=2)  # Shape: (K1, K2)

        # Total cost with weights
        cost_matrix = (
            self.lambda_pos * D_pos
            + self.lambda_shape * D_shape
            + self.lambda_color * D_color
        )  # Shape: (K1, K2)

        return cost_matrix

    def _construct_covariance(self, scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        """Construct covariance matrices from scales and rotations.

        Args:
            scales (torch.Tensor): Scales (K, 2).
            rotations (torch.Tensor): Rotations (K,).

        Returns:
            torch.Tensor: Covariance matrices (K, 2, 2).
        """
        cos_r = torch.cos(rotations)
        sin_r = torch.sin(rotations)

        # Rotation matrices
        R = torch.stack([
            torch.stack([cos_r, -sin_r], dim=1),
            torch.stack([sin_r, cos_r], dim=1)
        ], dim=2)  # Shape: (K, 2, 2)

        # Scale matrices
        S = torch.zeros((scales.shape[0], 2, 2), dtype=torch.float32, device=self.device)
        S[:, 0, 0] = scales[:, 0] ** 2
        S[:, 1, 1] = scales[:, 1] ** 2

        # Covariance matrices
        covariance = R @ S @ R.transpose(1, 2)  # Shape: (K, 2, 2)

        return covariance

    def _matrix_sqrt(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute the square root of symmetric positive-definite matrices.

        Args:
            matrix (torch.Tensor): Input matrices (K, 2, 2).

        Returns:
            torch.Tensor: Square roots of input matrices (K, 2, 2).
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)  # Shapes: (K, 2), (K, 2, 2)
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
        sqrt_matrix = eigenvectors @ torch.diag_embed(sqrt_eigenvalues) @ eigenvectors.transpose(1, 2)
        return sqrt_matrix

    def _wasserstein_distance(
        self,
        means1: torch.Tensor,
        scales1: torch.Tensor,
        rotations1: torch.Tensor,
        means2: torch.Tensor,
        scales2: torch.Tensor,
        rotations2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the squared 2-Wasserstein distance components between two sets of Gaussians.

        Args:
            means1 (torch.Tensor): Transformed means of the first set (K1, 2).
            scales1 (torch.Tensor): Scales of the first set (K1, 2).
            rotations1 (torch.Tensor): Rotations of the first set (K1,).
            means2 (torch.Tensor): Means of the second set (K2, 2).
            scales2 (torch.Tensor): Scales of the second set (K2, 2).
            rotations2 (torch.Tensor): Rotations of the second set (K2,).

        Returns:
            torch.Tensor: Shape differences (K1, K2).
        """
        K1 = means1.shape[0]
        K2 = means2.shape[0]

        # Construct covariance matrices
        sigma1 = self._construct_covariance(scales1, rotations1)  # Shape: (K1, 2, 2)
        sigma2 = self._construct_covariance(scales2, rotations2)  # Shape: (K2, 2, 2)

        # Expand dimensions for broadcasting
        sigma1_exp = sigma1.unsqueeze(1).expand(-1, K2, -1, -1)  # Shape: (K1, K2, 2, 2)
        sigma2_exp = sigma2.unsqueeze(0).expand(K1, -1, -1, -1)  # Shape: (K1, K2, 2, 2)

        # Compute the square roots
        sqrt_sigma1 = self._matrix_sqrt(sigma1)  # Shape: (K1, 2, 2)
        sqrt_sigma1_exp = sqrt_sigma1.unsqueeze(1).expand(-1, K2, -1, -1)  # Shape: (K1, K2, 2, 2)

        # Compute the intermediate matrices
        intermediate = sqrt_sigma1_exp @ sigma2_exp @ sqrt_sigma1_exp  # Shape: (K1, K2, 2, 2)

        # Compute the square roots of intermediate matrices
        sqrt_intermediate = self._matrix_sqrt(intermediate)  # Shape: (K1, K2, 2, 2)

        # Compute the trace terms
        trace_term = torch.diagonal(sigma1_exp + sigma2_exp - 2 * sqrt_intermediate, dim1=-2, dim2=-1).sum(-1)  # Shape: (K1, K2)

        return trace_term

    def sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Implement the Sinkhorn algorithm for optimal transport.

        Args:
            cost_matrix (torch.Tensor): The cost matrix (K1, K2).
            max_iter (int): Maximum number of iterations. Defaults to 100.
            tol (float): Convergence tolerance. Defaults to 1e-6.

        Returns:
            torch.Tensor: The transport plan matrix (K1, K2).
        """
        K1, K2 = cost_matrix.shape

        # Initialize the Gibbs kernel
        K = torch.exp(-cost_matrix / self.epsilon)  # Shape: (K1, K2)

        # Initialize scaling vectors
        u = torch.ones(K1, dtype=torch.float32, device=self.device)
        v = torch.ones(K2, dtype=torch.float32, device=self.device)

        # Normalize alpha and beta
        alpha = self.alpha1 / self.alpha1.sum()
        beta = self.alpha2 / self.alpha2.sum()

        for _ in range(max_iter):
            # Update u
            Kv = torch.matmul(K, v)  # Shape: (K1,)
            u_new = alpha / (Kv + 1e-16)

            # Update v
            KTu = torch.matmul(K.t(), u_new)  # Shape: (K2,)
            v_new = beta / (KTu + 1e-16)

            # Check for convergence
            if torch.max(torch.abs(u_new - u)) < tol and torch.max(torch.abs(v_new - v)) < tol:
                break

            u, v = u_new, v_new

        # Compute the transport plan
        T = torch.diag(u) @ K @ torch.diag(v)  # Shape: (K1, K2)

        return T

    def optimize_with_homography(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        """Optimize the projection matrix F using homography transformation.

        Args:
            max_iter (int): Maximum number of iterations. Defaults to 100.
            tol (float): Convergence tolerance. Defaults to 1e-6.
        """
        # Initialize F as an identity matrix if not provided
        if self.F is None:
            self.F = torch.eye(3, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            self.F = torch.tensor(self.F, dtype=torch.float32, device=self.device, requires_grad=True)

        # optimizer = torch.optim.Adam([self.F], lr=1e-3)
        optimizer = ADOPT([self.F], lr=1e-3)


        prev_loss = float('inf')

        for iteration in tqdm(range(max_iter), desc="Optimization with Homography"):
            optimizer.zero_grad()

            # Compute cost matrix with current F
            cost_matrix = self.compute_cost_matrix(self.F)  # Shape: (K1, K2)

            # Compute transport plan
            T = self.sinkhorn_algorithm(cost_matrix)

            # Compute objective function (total cost)
            loss = torch.sum(T * cost_matrix)

            # Backpropagation
            loss.backward()

            # Update F
            optimizer.step()

            # Projection step (if necessary)
            # For homography, we can normalize F to prevent scaling issues
            with torch.no_grad():
                self.F /= self.F.norm()

            # Check for convergence
            if torch.abs(prev_loss - loss.item()) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_loss = loss.item()

        # Detach F from the computation graph
        self.F = self.F.detach()

