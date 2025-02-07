import copy
import os
from typing import Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.primitive.twod_gaussians_rs import TwoDGaussians


class OptimalTransportSolver:
    """Optimal Transport Solver for 2D Gaussians with Homography Optimization."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        k1: Optional[np.ndarray] = None,
        k2: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
        lambda_mean: float = 1.0,
        lambda_cov: float = 0.3,
        lambda_color: float = 1.0,
        lambda_alpha: float = 1.0,  # <--- NEW for alpha weighting
        device: Optional[torch.device] = None,
    ):
        """Initialize the OptimalTransportSolver.

        Args:
            gaussians1 (TwoDGaussians): First set of 2D Gaussians.
            gaussians2 (TwoDGaussians): Second set of 2D Gaussians.
            k1 (Optional[np.ndarray]): Intrinsic parameters for camera 1 (3x3).
            k2 (Optional[np.ndarray]): Intrinsic parameters for camera 2 (3x3).
            epsilon (float): Entropy regularization parameter.
            lambda_mean (float): Weight for mean difference term.
            lambda_cov (float): Weight for covariance difference term.
            lambda_color (float): Weight for color difference term.
            lambda_alpha (float): Weight for alpha difference term.
            device (torch.device): Device to perform computations on.
        """
        self.gaussians1 = copy.deepcopy(gaussians1)
        self.gaussians2 = copy.deepcopy(gaussians2)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.k1 = (
            torch.tensor(k1, dtype=torch.float32, device=device)
            if k1 is not None
            else None
        )
        self.k2 = (
            torch.tensor(k2, dtype=torch.float32, device=device)
            if k2 is not None
            else None
        )
        self.h: Optional[torch.Tensor] = None

        self.epsilon = epsilon
        self.lambda_mean = lambda_mean
        self.lambda_cov = lambda_cov
        self.lambda_color = lambda_color
        self.lambda_alpha = lambda_alpha

        # Convert Gaussian parameters to torch tensors
        self._prepare_gaussians()

    def _prepare_gaussians(self) -> None:
        """Prepare Gaussian parameters as torch tensors."""
        self.means1 = torch.as_tensor(
            self.gaussians1.means, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.scales1 = torch.as_tensor(
            self.gaussians1.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 2)
        self.rotations1 = torch.as_tensor(
            self.gaussians1.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)
        self.rgb1 = torch.as_tensor(
            self.gaussians1.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K1, 3)
        self.alpha1 = torch.as_tensor(
            self.gaussians1.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K1,)

        self.means2 = torch.as_tensor(
            self.gaussians2.means, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.scales2 = torch.as_tensor(
            self.gaussians2.scales, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 2)
        self.rotations2 = torch.as_tensor(
            self.gaussians2.rotations, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)
        self.rgb2 = torch.as_tensor(
            self.gaussians2.rgb, dtype=torch.float32, device=self.device
        )  # Shape: (K2, 3)
        self.alpha2 = torch.as_tensor(
            self.gaussians2.alpha, dtype=torch.float32, device=self.device
        )  # Shape: (K2,)

    def compute_cost_matrix(self, h: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using a homography.

        Args:
            h (torch.Tensor): The homography transformation matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
        k1 = self.means1.shape[0]

        # 1) Transform means1 by homography h
        ones = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
        means1_h = torch.cat([self.means1, ones], dim=1)  # shape (K1, 3)
        transformed_means1_h = (h @ means1_h.T).T  # shape (K1, 3)
        transformed_means1 = transformed_means1_h[:, :2] / transformed_means1_h[
            :, 2
        ].unsqueeze(1)

        # 2) Compute Wasserstein distance components
        mean_term, cov_term = self._wasserstein_distance(
            transformed_means1,
            self.scales1,
            self.rotations1,
            self.means2,
            self.scales2,
            self.rotations2,
        )  # each shape (K1, K2)

        # 3) Compute color differences
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = torch.sum(color_diff**2, dim=2)  # (K1,K2)

        # Debug prints (optional)
        print("=== Before Normalization ===")
        print(
            f"mean_term: min={mean_term.min():.4f}, max={mean_term.max():.4f}, mean={mean_term.mean():.4f}"
        )
        print(
            f"cov_term: min={cov_term.min():.4f}, max={cov_term.max():.4f}, mean={cov_term.mean():.4f}"
        )
        print(
            f"d_color: min={d_color.min():.4f}, max={d_color.max():.4f}, mean={d_color.mean():.4f}"
        )

        # 6) Normalize each component (example approach)
        max_dim = 1554  # largest image dimension
        mean_term = mean_term / (max_dim**2)
        cov_term = cov_term / (max_dim**2 / 100000)
        d_color = d_color / 3.0

        print("=== After Normalization ===")
        print(
            f"mean_term: min={mean_term.min():.6f}, max={mean_term.max():.6f}, mean={mean_term.mean():.6f}"
        )
        print(
            f"cov_term: min={cov_term.min():.6f}, max={cov_term.max():.6f}, mean={cov_term.mean():.6f}"
        )
        print(
            f"d_color: min={d_color.min():.6f}, max={d_color.max():.6f}, mean={d_color.mean():.6f}"
        )


        cost_matrix = (
            self.lambda_mean * mean_term
            + self.lambda_cov * cov_term
            + self.lambda_color * d_color
        )

        print("=== Final Cost Matrix ===")
        print(
            f"cost_matrix: min={cost_matrix.min():.6f}, max={cost_matrix.max():.6f}, mean={cost_matrix.mean():.6f}"
        )

        return torch.as_tensor(cost_matrix)

    def _construct_covariance(
        self, scales: torch.Tensor, rotations: torch.Tensor
    ) -> torch.Tensor:
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
        r = torch.stack(
            [torch.stack([cos_r, -sin_r], dim=1), torch.stack([sin_r, cos_r], dim=1)],
            dim=2,
        )  # shape (K, 2, 2)

        # Scale matrices
        s = torch.zeros(
            (scales.shape[0], 2, 2), dtype=torch.float32, device=self.device
        )
        s[:, 0, 0] = scales[:, 0] ** 2
        s[:, 1, 1] = scales[:, 1] ** 2

        # Covariance matrices
        covariance = r @ s @ r.transpose(1, 2)  # shape (K, 2, 2)
        return covariance

    def _matrix_sqrt(self, matrices: torch.Tensor) -> torch.Tensor:
        """Compute the square root of batched 2x2 symmetric positive-definite matrices.

        Args:
            matrices (torch.Tensor): Input matrices of shape (..., 2, 2).

        Returns:
            torch.Tensor: Square root of the input matrices with same shape.
        """
        batch_shape = matrices.shape[:-2]
        matrices_2d = matrices.reshape(-1, 2, 2)

        # Compute eigenvalues and eigenvectors for all matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices_2d)
        sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
        sqrt_eigenvalues = torch.diag_embed(sqrt_eigenvalues)

        sqrt_matrices = eigenvectors @ sqrt_eigenvalues @ eigenvectors.transpose(-2, -1)
        sqrt_matrices = sqrt_matrices.reshape(*batch_shape, 2, 2)
        return torch.Tensor(sqrt_matrices)

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
            Tuple[torch.Tensor, torch.Tensor]: Mean term and covariance term (K1, K2).
        """
        k1, k2 = means1.shape[0], means2.shape[0]

        # Mean difference
        diff = means1.unsqueeze(1) - means2.unsqueeze(0)  # (K1,K2,2)
        mean_term = torch.sum(diff**2, dim=2)  # (K1,K2)

        # 2D covariance
        sigma1 = self._construct_covariance(scales1, rotations1)  # (K1,2,2)
        sigma2 = self._construct_covariance(scales2, rotations2)  # (K2,2,2)

        sigma1_exp = sigma1.unsqueeze(1).expand(-1, k2, -1, -1)  # (K1,K2,2,2)
        sigma2_exp = sigma2.unsqueeze(0).expand(k1, -1, -1, -1)  # (K1,K2,2,2)

        sqrt_sigma1 = self._matrix_sqrt(sigma1)  # (K1,2,2)
        sqrt_sigma1_exp = sqrt_sigma1.unsqueeze(1).expand(-1, k2, -1, -1)  # (K1,K2,2,2)

        intermediate = sqrt_sigma1_exp @ sigma2_exp @ sqrt_sigma1_exp
        sqrt_intermediate = self._matrix_sqrt(intermediate)

        trace_term = torch.diagonal(
            sigma1_exp + sigma2_exp - 2 * sqrt_intermediate, dim1=-2, dim2=-1
        ).sum(-1)  # (K1,K2)

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
            max_iter (int): Maximum number of iterations. Defaults to 1000.
            tol (float): Convergence tolerance. Defaults to 1e-6.

        Returns:
            torch.Tensor: The transport plan matrix (K1, K2).
        """
        k1, k2 = cost_matrix.shape
        kernel = torch.exp(-cost_matrix / self.epsilon)

        alpha = self.alpha1 / self.alpha1.sum()
        beta = self.alpha2 / self.alpha2.sum()

        # Initialize scaling vectors
        u = torch.ones(k1, dtype=torch.float32, device=self.device)
        v = torch.ones(k2, dtype=torch.float32, device=self.device)

        for i in range(max_iter):
            # Update u
            kv = torch.matmul(kernel, v)
            u_new = alpha / (kv + 1e-16)

            # Update v
            ktu = torch.matmul(kernel.t(), u_new)
            v_new = beta / (ktu + 1e-16)

            # Check for convergence
            if (
                torch.max(torch.abs(u_new - u)) < tol
                and torch.max(torch.abs(v_new - v)) < tol
            ):
                print(f"Sinkhorn converged at iteration {i}")
                break

            u, v = u_new, v_new

            if i % 100 == 0:
                cur_sum = (u.unsqueeze(1) * kernel * v.unsqueeze(0)).sum().item()
                print(f"Sinkhorn Iteration {i}, Transport sum: {cur_sum}")

        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return transport

    def unbalanced_sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        rho: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Implement the unbalanced optimal transport Sinkhorn algorithm.

        Args:
            cost_matrix (torch.Tensor): The cost matrix of shape (K1, K2).
            rho (float): Degree to which mass discrepancy is allowed (smaller means more freedom to create or remove mass).
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence threshold.

        Returns:
            torch.Tensor: The unbalanced OT transport plan of shape (K1, K2).
        """
        # alpha and beta represent the total mass (or approximated mass) of each Gaussian
        alpha = self.alpha1  # shape (K1,)
        beta = self.alpha2  # shape (K2,)

        kernel = torch.exp(-cost_matrix / self.epsilon)  # shape (K1, K2)

        # Initialize u, v to 1
        u = torch.ones_like(alpha)  # (K1,)
        v = torch.ones_like(beta)  # (K2,)

        exponent = rho / (rho + self.epsilon)

        for iteration in range(max_iter):
            kv = kernel @ v
            kv = kv + 1e-16
            u_new = (alpha / kv).pow(exponent)

            ktu = kernel.t() @ u_new
            ktu = ktu + 1e-16
            v_new = (beta / ktu).pow(exponent)

            # Check convergence
            if (
                torch.max(torch.abs(u_new - u)) < tol
                and torch.max(torch.abs(v_new - v)) < tol
            ):
                print(f"[Unbalanced] iteration {iteration} -> converged.")
                break

            u, v = u_new, v_new

        transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
        return transport

    def optimize_with_homography(self, max_iter: int = 1000, tol: float = 1e-3) -> None:
        """Optimize the homography matrix H.

        Args:
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        """
        transport_dir = os.path.join("results", "transport")
        os.makedirs(transport_dir, exist_ok=True)

        self.h = torch.eye(
            3, dtype=torch.float32, device=self.device, requires_grad=True
        )
        optimizer = torch.optim.Adam([self.h], lr=1e-4)
        prev_loss = torch.tensor(float("inf"), device=self.device)

        loss_history = []

        for iteration in tqdm(range(max_iter), desc="Optimization with Homography"):
            optimizer.zero_grad()

            # Compute cost matrix with current h
            cost_matrix = self.compute_cost_matrix(self.h)
            # Compute transport plan using unbalanced Sinkhorn
            transport = self.unbalanced_sinkhorn_algorithm(
                cost_matrix, rho=1.0, max_iter=10000, tol=1e-6
            )

            # Compute objective function
            loss = torch.sum(transport * cost_matrix)
            loss.backward()  # type: ignore

            if iteration % 5 == 0 or iteration == max_iter - 1:
                grad_norm = (
                    self.h.grad.norm().item() if self.h.grad is not None else 0.0
                )
                print(
                    f"Iteration {iteration}, Loss: {loss.item():.6f}, H.grad norm: {grad_norm:.6f}"
                )

            optimizer.step()
            loss_history.append(loss.item())

            if iteration % 5 == 0 or iteration == max_iter - 1:
                print(
                    f"Iteration {iteration}, Loss={loss.item():.6f}, GradNorm={grad_norm:.6f}"
                )

            if iteration % 10 == 0 or iteration == max_iter - 1:
                with torch.no_grad():
                    t_np = transport.cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    plt.imshow(t_np, cmap="hot", interpolation="nearest")
                    plt.colorbar(label="Transport Plan Value")
                    plt.title(f"Transport Plan at Iteration {iteration}")
                    plt.xlabel("Image 2 Gaussians")
                    plt.ylabel("Image 1 Gaussians")
                    plt.tight_layout()
                    plt_path = os.path.join(
                        transport_dir, f"transport_iter_{iteration}.png"
                    )
                    plt.savefig(plt_path)
                    plt.close()
                    print(f"Transport matrix heatmap saved to '{plt_path}'")

            if abs(prev_loss.item() - loss.item()) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_loss = loss

        with torch.no_grad():
            print(f"Before normalization, H[2,2]: {self.h[2,2].item()}")
            if self.h[2, 2] != 0:
                h_norm = self.h / self.h[2, 2].clone()
            else:
                h_norm = self.h / torch.max(torch.abs(self.h)).clone()
            self.h.copy_(h_norm)
            print(f"After normalization, H[2,2]: {self.h[2,2].item()}")

        # Detach final homography
        self.h = self.h.detach()

        # Visualize loss
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Optimization Loss over Iterations")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt_path = os.path.join("results", "optimization_loss.png")
        plt.savefig(plt_path)
        plt.close()
        print(f"Optimization loss plot saved to '{plt_path}'")
