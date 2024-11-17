import copy
from typing import Optional, Tuple
import sys

import numpy as np
import torch
from tqdm import tqdm

from src.primitive.twod_gaussians import TwoDGaussians
# from utils.adopt import ADOPT



class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians with Homography Optimization."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        K1: Optional[np.ndarray] = None,
        K2: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
        lambda_mean: float = 1.0,
        lambda_cov: float = 1.0,
        lambda_color: float = 1.0,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            K1 (Optional[np.ndarray]): Intrinsic parameters of the first camera (3x3).
            K2 (Optional[np.ndarray]): Intrinsic parameters of the second camera (3x3).
            epsilon (float): Entropy regularization parameter. Defaults to 0.1.
            lambda_mean (float): Weight for mean difference term. Defaults to 0.3.
            lambda_cov (float): Weight for covariance difference term. Defaults to 1.0.
            lambda_color (float): Weight for color difference. Defaults to 1.0.
            device (torch.device): Device to perform computations on.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)

        self.K1 = torch.tensor(K1, dtype=torch.float32, device=device) if K1 is not None else None
        self.K2 = torch.tensor(K2, dtype=torch.float32, device=device) if K2 is not None else None
        self.H = None  # Will be initialized in the optimization process

        self.epsilon = epsilon
        self.lambda_mean = lambda_mean
        self.lambda_cov = lambda_cov
        self.lambda_color = lambda_color

        self.device = device
        # print(f"Using {device} device")


        # Convert Gaussian parameters to torch tensors
        self._prepare_gaussians()

    def _prepare_gaussians(self):
        """Prepare Gaussian parameters as torch tensors."""
        self.means1 = self.gaussians1.means.clone().detach().to(self.device)  # Shape: (K1, 2)
        self.scales1 = self.gaussians1.scales.clone().detach().to(self.device)  # Shape: (K1, 2)
        self.rotations1 = self.gaussians1.rotations.clone().detach().to(self.device)  # Shape: (K1,)
        self.rgb1 = self.gaussians1.rgb.clone().detach().to(self.device)  # Shape: (K1, 3)
        self.alpha1 = self.gaussians1.alpha.clone().detach().to(self.device)  # Shape: (K1,)

        self.means2 = self.gaussians2.means.clone().detach().to(self.device)  # Shape: (K2, 2)
        self.scales2 = self.gaussians2.scales.clone().detach().to(self.device)  # Shape: (K2, 2)
        self.rotations2 = self.gaussians2.rotations.clone().detach().to(self.device)  # Shape: (K2,)
        self.rgb2 = self.gaussians2.rgb.clone().detach().to(self.device)  # Shape: (K2, 3)
        self.alpha2 = self.gaussians2.alpha.clone().detach().to(self.device)  # Shape: (K2,)

    def compute_cost_matrix(self, H: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using homography.

        Args:
            H (torch.Tensor): The homography transformation matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
        K1, K2 = self.means1.shape[0], self.means2.shape[0]

        # Prepare homogeneous coordinates for means1
        ones = torch.ones((K1, 1), dtype=torch.float32, device=self.device)
        means1_h = torch.cat([self.means1, ones], dim=1)  # Shape: (K1, 3)

        # Apply homography H to means1
        transformed_means1_h = (H @ means1_h.T).T  # Shape: (K1, 3)

        # Convert back to inhomogeneous coordinates
        transformed_means1 = transformed_means1_h[:, :2] / transformed_means1_h[:, 2].unsqueeze(1)  # Shape: (K1, 2)

        # Compute pairwise distances between transformed_means1 and means2
        diff = transformed_means1.unsqueeze(1) - self.means2.unsqueeze(0)  # Shape: (K1, K2, 2)
        D_pos = torch.sum(diff ** 2, dim=2)  # Shape: (K1, K2)

        # Get separate Wasserstein distance components
        mean_term, cov_term = self._wasserstein_distance(
            transformed_means1, self.scales1, self.rotations1,
            self.means2, self.scales2, self.rotations2
        )  # Each shape: (K1, K2)

        # Compute color differences
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)
        D_color = torch.sum(color_diff ** 2, dim=2)

        # Normalize components before combining
        mean_term = mean_term / mean_term.max()
        cov_term = cov_term / cov_term.max()
        D_color = D_color / D_color.max()

        # Combine with weights
        cost_matrix = (
            self.lambda_mean * mean_term
            + self.lambda_cov * cov_term
            + self.lambda_color * D_color
        )

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

    def _matrix_sqrt(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the square root of batched 2x2 symmetric positive-definite matrices.

        Args:
            matrices (torch.Tensor): Input matrices of shape (..., 2, 2)

        Returns:
            torch.Tensor: Square root of the input matrices with same shape
        """
        # Reshape to (-1, 2, 2) to handle all batches
        batch_shape = matrices.shape[:-2]
        matrices_2d = matrices.reshape(-1, 2, 2)
        
        # Compute eigenvalues and eigenvectors for all matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices_2d)
        
        # Ensure non-negative eigenvalues and compute sqrt
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
        
        # Create diagonal matrices for each sqrt eigenvalue
        sqrt_eigenvalues = torch.diag_embed(sqrt_eigenvalues)
        
        # Compute square root for all matrices
        sqrt_matrices = eigenvectors @ sqrt_eigenvalues @ eigenvectors.transpose(-2, -1)
        
        # Reshape back to original batch shape
        sqrt_matrices = sqrt_matrices.reshape(*batch_shape, 2, 2)
        
        return sqrt_matrices

    def _wasserstein_distance(
        self,
        means1: torch.Tensor,
        scales1: torch.Tensor,
        rotations1: torch.Tensor,
        means2: torch.Tensor,
        scales2: torch.Tensor,
        rotations2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the squared 2-Wasserstein distance components between two sets of Gaussians.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean term and covariance term, each of shape (K1, K2)
        """
        K1, K2 = means1.shape[0], means2.shape[0]

        # Compute mean differences
        diff = means1.unsqueeze(1) - means2.unsqueeze(0)  # Shape: (K1, K2, 2)
        mean_term = torch.sum(diff ** 2, dim=2)  # Shape: (K1, K2)

        # Construct covariance matrices
        sigma1 = self._construct_covariance(scales1, rotations1)  # Shape: (K1, 2, 2)
        sigma2 = self._construct_covariance(scales2, rotations2)  # Shape: (K2, 2, 2)

        # Expand dimensions for broadcasting
        sigma1_exp = sigma1.unsqueeze(1).expand(-1, K2, -1, -1)  # Shape: (K1, K2, 2, 2)
        sigma2_exp = sigma2.unsqueeze(0).expand(K1, -1, -1, -1)  # Shape: (K1, K2, 2, 2)

        # Compute the square roots
        sqrt_sigma1 = self._matrix_sqrt(sigma1)  # Shape: (K1, 2, 2)
        sqrt_sigma1_exp = sqrt_sigma1.unsqueeze(1).expand(-1, K2, -1, -1)

        # Compute the intermediate matrices
        intermediate = sqrt_sigma1_exp @ sigma2_exp @ sqrt_sigma1_exp
        sqrt_intermediate = self._matrix_sqrt(intermediate)

        # Compute the trace term
        trace_term = torch.diagonal(
            sigma1_exp + sigma2_exp - 2 * sqrt_intermediate, 
            dim1=-2, dim2=-1
        ).sum(-1)  # Shape: (K1, K2)

        return mean_term, trace_term

    def sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        max_iter: int = 1000,
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

        # Normalize cost matrix to prevent numerical issues
        # cost_max = torch.max(cost_matrix)
        # cost_matrix = cost_matrix / cost_max

        # Initialize the Gibbs kernel with numerical stability
        K = torch.exp(-cost_matrix / self.epsilon)  # Shape: (K1, K2)
        
        # Check if K has any valid values
        if torch.any(torch.isnan(K)) or torch.all(K == 0):
            print("Warning: Numerical issues in kernel computation")
            print(f"Cost matrix stats - Min: {cost_matrix.min()}, Max: {cost_matrix.max()}, Mean: {cost_matrix.mean()}")
        
        # Initialize scaling vectors
        u = torch.ones(K1, dtype=torch.float32, device=self.device)
        v = torch.ones(K2, dtype=torch.float32, device=self.device)

        # Normalize alpha and beta
        alpha = self.alpha1 / self.alpha1.sum()
        beta = self.alpha2 / self.alpha2.sum()

        for i in range(max_iter):
            # Update u
            Kv = torch.matmul(K, v)  # Shape: (K1,)
            u_new = alpha / (Kv + 1e-16)

            # Update v
            KTu = torch.matmul(K.t(), u_new)  # Shape: (K2,)
            v_new = beta / (KTu + 1e-16)

            # Check for convergence
            if (torch.max(torch.abs(u_new - u)) < tol and 
                torch.max(torch.abs(v_new - v)) < tol):
                break

            u, v = u_new, v_new

            # Add debugging information
            # if i % 10 == 0:
            #     current_sum = (torch.diag(u) @ K @ torch.diag(v)).sum().item()
            #     print(f"Iteration {i}, Transport sum: {current_sum}")

        # Compute the transport plan
        T = torch.diag(u) @ K @ torch.diag(v)

        # Rescale back
        # T = T / T.sum() 
        return T

    def optimize_with_homography(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        """Optimize the homography matrix H.

        Args:
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        """
        # Initialize H as identity if not provided
        if self.H is None:
            self.H = torch.eye(3, dtype=torch.float32, device=self.device, requires_grad=True)
        else:
            self.H = torch.tensor(self.H, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([self.H], lr=1e-3)
        # optimizer = ADOPT([self.H], lr=1e-3)


        prev_loss = torch.tensor(float('inf'), device=self.device)

        for iteration in tqdm(range(max_iter), desc="Optimization with Homography"):
            optimizer.zero_grad()

            # Compute cost matrix with current H
            cost_matrix = self.compute_cost_matrix(self.H)  # Shape: (K1, K2)

            # Compute transport plan
            T = self.sinkhorn_algorithm(cost_matrix)

            # Compute objective function (total cost)
            loss = torch.sum(T * cost_matrix)
            print(f"Iteration {iteration}, Loss: {loss}")

            # Backpropagation
            loss.backward()

            # Update H
            optimizer.step()

            # Projection step (if necessary)
            # For homography, we can normalize H to prevent scaling issues
            with torch.no_grad():
                self.H /= self.H.norm()

            # Check for convergence using torch.abs()
            if torch.abs(prev_loss - loss) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_loss = loss

        # Detach H from the computation graph
        self.H = self.H.detach()

