import copy
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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
        lambda_epipolar: float = 0.0,
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
        # Added for Fundamental Matrix
        self.f: Optional[torch.Tensor] = None

        self.epsilon = epsilon
        self.lambda_mean = lambda_mean
        self.lambda_cov = lambda_cov
        self.lambda_color = lambda_color
        self.lambda_epipolar = lambda_epipolar

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

    # def compute_cost_matrix(self, h: torch.Tensor) -> torch.Tensor:
    #     """Compute the cost matrix between two sets of 2D Gaussians using a homography.

    #     Args:
    #         h (torch.Tensor): The homography transformation matrix (3x3).

    #     Returns:
    #         torch.Tensor: Cost matrix of shape (K1, K2).
    #     """
    #     k1 = self.means1.shape[0]

    #     # 1) Transform means1 by homography h
    #     ones = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
    #     means1_h = torch.cat([self.means1, ones], dim=1)  # shape (K1, 3)
    #     transformed_means1_h = (h @ means1_h.T).T  # shape (K1, 3)
    #     transformed_means1 = transformed_means1_h[:, :2] / transformed_means1_h[
    #         :, 2
    #     ].unsqueeze(1)

    #     # 2) Compute Wasserstein distance components
    #     mean_term, cov_term = self._wasserstein_distance(
    #         transformed_means1,
    #         self.scales1,
    #         self.rotations1,
    #         self.means2,
    #         self.scales2,
    #         self.rotations2,
    #     )  # each shape (K1, K2)

    #     # 3) Compute color differences
    #     color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
    #     d_color = torch.sum(color_diff**2, dim=2)  # (K1,K2)

    #     # Debug prints (optional)
    #     print("=== Before Normalization ===")
    #     print(
    #         f"mean_term: min={mean_term.min():.4f}, max={mean_term.max():.4f}, mean={mean_term.mean():.4f}"
    #     )
    #     print(
    #         f"cov_term: min={cov_term.min():.4f}, max={cov_term.max():.4f}, mean={cov_term.mean():.4f}"
    #     )
    #     print(
    #         f"d_color: min={d_color.min():.4f}, max={d_color.max():.4f}, mean={d_color.mean():.4f}"
    #     )

    #     # 6) Normalize each component (example approach)
    #     max_dim = 1554  # largest image dimension
    #     mean_term = mean_term / (max_dim**2)
    #     cov_term = cov_term / (max_dim**2 / 100000)
    #     d_color = d_color / 3.0

    #     print("=== After Normalization ===")
    #     print(
    #         f"mean_term: min={mean_term.min():.6f}, max={mean_term.max():.6f}, mean={mean_term.mean():.6f}"
    #     )
    #     print(
    #         f"cov_term: min={cov_term.min():.6f}, max={cov_term.max():.6f}, mean={cov_term.mean():.6f}"
    #     )
    #     print(
    #         f"d_color: min={d_color.min():.6f}, max={d_color.max():.6f}, mean={d_color.mean():.6f}"
    #     )

    #     cost_matrix = (
    #         self.lambda_mean * mean_term
    #         + self.lambda_cov * cov_term
    #         + self.lambda_color * d_color
    #     )

    #     print("=== Final Cost Matrix ===")
    #     print(
    #         f"cost_matrix: min={cost_matrix.min():.6f}, max={cost_matrix.max():.6f}, mean={cost_matrix.mean():.6f}"
    #     )

    #     return torch.as_tensor(cost_matrix)

    # def _construct_covariance(
    #     self, scales: torch.Tensor, rotations: torch.Tensor
    # ) -> torch.Tensor:
    #     """Construct covariance matrices from scales and rotations.

    #     Args:
    #         scales (torch.Tensor): Scales (K, 2).
    #         rotations (torch.Tensor): Rotations (K,).

    #     Returns:
    #         torch.Tensor: Covariance matrices (K, 2, 2).
    #     """
    #     cos_r = torch.cos(rotations)
    #     sin_r = torch.sin(rotations)

    #     # Rotation matrices
    #     r = torch.stack(
    #         [torch.stack([cos_r, -sin_r], dim=1), torch.stack([sin_r, cos_r], dim=1)],
    #         dim=2,
    #     )  # shape (K, 2, 2)

    #     # Scale matrices
    #     s = torch.zeros(
    #         (scales.shape[0], 2, 2), dtype=torch.float32, device=self.device
    #     )
    #     s[:, 0, 0] = scales[:, 0] ** 2
    #     s[:, 1, 1] = scales[:, 1] ** 2

    #     # Covariance matrices
    #     covariance = r @ s @ r.transpose(1, 2)  # shape (K, 2, 2)
    #     return covariance

    # def _matrix_sqrt(self, matrices: torch.Tensor) -> torch.Tensor:
    #     """Compute the square root of batched 2x2 symmetric positive-definite matrices.

    #     Args:
    #         matrices (torch.Tensor): Input matrices of shape (..., 2, 2).

    #     Returns:
    #         torch.Tensor: Square root of the input matrices with same shape.
    #     """
    #     batch_shape = matrices.shape[:-2]
    #     matrices_2d = matrices.reshape(-1, 2, 2)

    #     # Compute eigenvalues and eigenvectors for all matrices
    #     eigenvalues, eigenvectors = torch.linalg.eigh(matrices_2d)
    #     sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
    #     sqrt_eigenvalues = torch.diag_embed(sqrt_eigenvalues)

    #     sqrt_matrices = eigenvectors @ sqrt_eigenvalues @ eigenvectors.transpose(-2, -1)
    #     sqrt_matrices = sqrt_matrices.reshape(*batch_shape, 2, 2)
    #     return torch.Tensor(sqrt_matrices)

    # def _wasserstein_distance(
    #     self,
    #     means1: torch.Tensor,
    #     scales1: torch.Tensor,
    #     rotations1: torch.Tensor,
    #     means2: torch.Tensor,
    #     scales2: torch.Tensor,
    #     rotations2: torch.Tensor,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """Compute the squared 2-Wasserstein distance components between two sets of Gaussians.

    #     Returns:
    #         Tuple[torch.Tensor, torch.Tensor]: Mean term and covariance term (K1, K2).
    #     """
    #     k1, k2 = means1.shape[0], means2.shape[0]

    #     # Mean difference
    #     diff = means1.unsqueeze(1) - means2.unsqueeze(0)  # (K1,K2,2)
    #     mean_term = torch.sum(diff**2, dim=2)  # (K1,K2)

    #     # 2D covariance
    #     sigma1 = self._construct_covariance(scales1, rotations1)  # (K1,2,2)
    #     sigma2 = self._construct_covariance(scales2, rotations2)  # (K2,2,2)

    #     sigma1_exp = sigma1.unsqueeze(1).expand(-1, k2, -1, -1)  # (K1,K2,2,2)
    #     sigma2_exp = sigma2.unsqueeze(0).expand(k1, -1, -1, -1)  # (K1,K2,2,2)

    #     sqrt_sigma1 = self._matrix_sqrt(sigma1)  # (K1,2,2)
    #     sqrt_sigma1_exp = sqrt_sigma1.unsqueeze(1).expand(-1, k2, -1, -1)  # (K1,K2,2,2)

    #     intermediate = sqrt_sigma1_exp @ sigma2_exp @ sqrt_sigma1_exp
    #     sqrt_intermediate = self._matrix_sqrt(intermediate)

    #     trace_term = torch.diagonal(
    #         sigma1_exp + sigma2_exp - 2 * sqrt_intermediate, dim1=-2, dim2=-1
    #     ).sum(-1)  # (K1,K2)

    #     return mean_term, trace_term

    # def sinkhorn_algorithm(
    #     self,
    #     cost_matrix: torch.Tensor,
    #     max_iter: int = 1000,
    #     tol: float = 1e-6,
    # ) -> torch.Tensor:
    #     """Implement the Sinkhorn algorithm for optimal transport.

    #     Args:
    #         cost_matrix (torch.Tensor): The cost matrix (K1, K2).
    #         max_iter (int): Maximum number of iterations. Defaults to 1000.
    #         tol (float): Convergence tolerance. Defaults to 1e-6.

    #     Returns:
    #         torch.Tensor: The transport plan matrix (K1, K2).
    #     """
    #     k1, k2 = cost_matrix.shape
    #     kernel = torch.exp(-cost_matrix / self.epsilon)

    #     alpha = self.alpha1 / self.alpha1.sum()
    #     beta = self.alpha2 / self.alpha2.sum()

    #     # Initialize scaling vectors
    #     u = torch.ones(k1, dtype=torch.float32, device=self.device)
    #     v = torch.ones(k2, dtype=torch.float32, device=self.device)

    #     for i in range(max_iter):
    #         # Update u
    #         kv = torch.matmul(kernel, v)
    #         u_new = alpha / (kv + 1e-16)

    #         # Update v
    #         ktu = torch.matmul(kernel.t(), u_new)
    #         v_new = beta / (ktu + 1e-16)

    #         # Check for convergence
    #         if (
    #             torch.max(torch.abs(u_new - u)) < tol
    #             and torch.max(torch.abs(v_new - v)) < tol
    #         ):
    #             print(f"Sinkhorn converged at iteration {i}")
    #             break

    #         u, v = u_new, v_new

    #         if i % 100 == 0:
    #             cur_sum = (u.unsqueeze(1) * kernel * v.unsqueeze(0)).sum().item()
    #             print(f"Sinkhorn Iteration {i}, Transport sum: {cur_sum}")

    #     transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    #     return transport
    
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

        # if exponent == 1, it's the balanced Sinkhorn algorithm
        exponent = rho / (rho + self.epsilon)
        print(f"rho: {rho}, epsilon: {self.epsilon}, exponent: {exponent}")

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

    # def optimize_with_homography(self, max_iter: int = 1000, tol: float = 1e-3) -> None:
    #     """Optimize the homography matrix H.

    #     Args:
    #         max_iter (int): Maximum number of iterations.
    #         tol (float): Convergence tolerance.
    #     """
    #     transport_dir = os.path.join("results", "transport_homography")
    #     os.makedirs(transport_dir, exist_ok=True)

    #     self.h = torch.eye(
    #         3, dtype=torch.float32, device=self.device, requires_grad=True
    #     )
    #     optimizer = torch.optim.Adam([self.h], lr=1e-4)
    #     prev_loss = torch.tensor(float("inf"), device=self.device)

    #     loss_history = []

    #     for iteration in tqdm(range(max_iter), desc="Optimization with Homography"):
    #         optimizer.zero_grad()

    #         # Compute cost matrix with current h
    #         cost_matrix = self.compute_cost_matrix(self.h)
    #         # Compute transport plan using unbalanced Sinkhorn
    #         transport = self.unbalanced_sinkhorn_algorithm(
    #             cost_matrix, rho=1.0, max_iter=10000, tol=1e-6
    #         )

    #         # Compute objective function
    #         loss = torch.sum(transport * cost_matrix)
    #         loss.backward()  # type: ignore

    #         if iteration % 5 == 0 or iteration == max_iter - 1:
    #             grad_norm = (
    #                 self.h.grad.norm().item() if self.h.grad is not None else 0.0
    #             )
    #             print(
    #                 f"Iteration {iteration}, Loss: {loss.item():.6f}, H.grad norm: {grad_norm:.6f}"
    #             )

    #         optimizer.step()
    #         loss_history.append(loss.item())

    #         if iteration % 5 == 0 or iteration == max_iter - 1:
    #             print(
    #                 f"Iteration {iteration}, Loss={loss.item():.6f}, GradNorm={grad_norm:.6f}"
    #             )

    #         if iteration % 10 == 0 or iteration == max_iter - 1:
    #             with torch.no_grad():
    #                 t_np = transport.cpu().numpy()
    #                 plt.figure(figsize=(8, 6))
    #                 plt.imshow(t_np, cmap="hot", interpolation="nearest")
    #                 plt.colorbar(label="Transport Plan Value")
    #                 plt.title(f"Transport Plan at Iteration {iteration}")
    #                 plt.xlabel("Image 2 Gaussians")
    #                 plt.ylabel("Image 1 Gaussians")
    #                 plt.tight_layout()
    #                 plt_path = os.path.join(
    #                     transport_dir, f"transport_iter_{iteration}.png"
    #                 )
    #                 plt.savefig(plt_path)
    #                 plt.close()
    #                 print(f"Transport matrix heatmap saved to '{plt_path}'")

    #         if abs(prev_loss.item() - loss.item()) < tol:
    #             print(f"Converged at iteration {iteration}")
    #             break
    #         prev_loss = loss

    #     with torch.no_grad():
    #         print(f"Before normalization, H[2,2]: {self.h[2,2].item()}")
    #         if self.h[2, 2] != 0:
    #             h_norm = self.h / self.h[2, 2].clone()
    #         else:
    #             h_norm = self.h / torch.max(torch.abs(self.h)).clone()
    #         self.h.copy_(h_norm)
    #         print(f"After normalization, H[2,2]: {self.h[2,2].item()}")

    #     # Detach final homography
    #     self.h = self.h.detach()

    #     # Visualize loss
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(loss_history, label="Loss")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Loss")
    #     plt.title("Optimization Loss over Iterations")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt_path = os.path.join("results", "optimization_loss.png")
    #     plt.savefig(plt_path)
    #     plt.close()
    #     print(f"Optimization loss plot saved to '{plt_path}'")


    # === Added below for Fundamental Matrix version ===

    #　ピクセル座標でのエピポーラコストを計算
    # def compute_cost_matrix_fundamental_direct(self, f: torch.Tensor) -> torch.Tensor:
    #     """Compute the cost matrix as the absolute value of x2^T F x1 in pixel space."""
        
    
    #     k1 = self.means1.shape[0]
    #     k2 = self.means2.shape[0]

    #     # Homogeneous coords in pixel space
    #     ones1 = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
    #     p1_homo = torch.cat([self.means1, ones1], dim=1)  # (K1,3)

    #     ones2 = torch.ones((k2, 1), dtype=torch.float32, device=self.device)
    #     p2_homo = torch.cat([self.means2, ones2], dim=1)  # (K2,3)

    #     # Compute x2^T F x1 for all i,j
    #     # p1_homo: (K1,3)
    #     # p2_homo: (K2,3)
    #     # We want a cost_matrix of shape (K1,K2).
    #     # cost[i,j] = | p2_homo[j] @ f @ p1_homo[i] 

    #     # (K2,3) x (3,3) -> (K2,3)
    #     Fx1 = (f @ p1_homo.T).T  # shape (K1,3)
    #     # Then x2^T Fx1: shape (K2, K1)
    #     # But we want (K1, K2). So we can do:
    #     cost_matrix = torch.abs(
    #         (p2_homo.unsqueeze(1) * Fx1.unsqueeze(0)).sum(dim=2)
    #     )
    #     # => cost_matrix: (K2, K1). make it (K1,K2)
    #     cost_matrix = cost_matrix.transpose(0,1)  # shape (K1,K2)

    #     # color difference　
    #     if self.lambda_color > 0:
    #         color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
    #         d_color = torch.sum(color_diff ** 2, dim=2)  # (K1,K2)
    #         cost_matrix = self.lambda_epipolar * cost_matrix + self.lambda_color * d_color 
    #     else:
    #         cost_matrix = self.lambda_epipolar * cost_matrix 
        
    #     print("=== Cost Matrix ===")
    #     print(f"cost_matrix: min={cost_matrix.min():.6f}, max={cost_matrix.max():.6f}, mean={cost_matrix.mean():.6f}")

    #     return cost_matrix

    def compute_cost_matrix_fundamental(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using a Fundamental Matrix.

        This replaces the Homography-based distance with an epipolar distance.
        Also includes color difference term as an example.

        Args:
            f (torch.Tensor): The Fundamental matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
        # 内部パラメータ行列 K の逆行列を使って画像座標を正規化平面（理想ピンホール）に変換も考えられる
        # w, h = 1554, 1162
        
        # means1_norm = self.means1 / torch.tensor([w, h], dtype=torch.float32, device=self.device)
        # means2_norm = self.means2 / torch.tensor([w, h], dtype=torch.float32, device=self.device)
        # Get number of gaussians for each image
        k1 = self.means1.shape[0]
        k2 = self.means2.shape[0]

        # Create homogeneous coords
        ones1 = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
        p1_homo = torch.cat([self.means1, ones1], dim=1)  # (K1,3)
        # p1_homo_norm = torch.cat([means1_norm, ones1], dim=1)  # (K1,3)

        ones2 = torch.ones((k2, 1), dtype=torch.float32, device=self.device)
        p2_homo = torch.cat([self.means2, ones2], dim=1)  # (K2,3)
        # p2_homo_norm = torch.cat([means2_norm, ones2], dim=1)  # (K2,3)

        # Compute epipolar lines
        # line in image1 for each p2: l1 = F * p2
        l1 = (f @ p2_homo.T).T  # (K2,3)
        # line in image2 for each p1: l2 = F^T * p1
        l2 = (f.t() @ p1_homo.T).T  # (K1,3)
        
        # Computeepipolar lines in normalized space
        # l1 = (f @ p2_homo_norm.T).T  # (K2,3)
        # l2 = (f.t() @ p1_homo_norm.T).T  # (K1,3)

        # Define the distance p1-l1 + p2-l2 as a symmetrical epipolar cost
        # Dist from p1[i] to line l1[j]: |p1[i].dot(l1[j])| / sqrt(a^2 + b^2) where l1[j] = [a,b,c]
        # shape: (K1,K2)

        # shape (K1,1,3) * (1,K2,3,1) => (K1,K2,1,1)
        numerator_12 = torch.abs(
            p1_homo.unsqueeze(1) @ l1.unsqueeze(2)
        ).squeeze(-1).squeeze(-1)  # (K1,K2)
        denom_12 = torch.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)  # (K2,)
        denom_12 = denom_12.view(1, -1)  # (1,K2)
        dist_12 = numerator_12 / denom_12  # (K1,K2)
        
        # numerator_12_norm = torch.abs(
        #     p1_homo_norm.unsqueeze(1) @ l1.unsqueeze(2)
        # ).squeeze(-1).squeeze(-1)  # (K1,K2)
        # denom_12 = torch.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)  # (K2,)
        # denom_12 = denom_12.view(1, -1)  # (1,K2)
        # dist_12_norm = numerator_12_norm / denom_12  # (K1,K2)

        numerator_21 = torch.abs(
            p2_homo.unsqueeze(0) @ l2.unsqueeze(2)
        ).squeeze(-1).squeeze(-1)  # (K1,K2)
        denom_21 = torch.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)  # (K1,)
        denom_21 = denom_21.view(-1, 1)  # (K1,1)
        dist_21 = numerator_21 / denom_21  # (K1,K2)
        
        # numerator_21_norm = torch.abs(
        #     p2_homo_norm.unsqueeze(0) @ l2.unsqueeze(2)
        # ).squeeze(-1).squeeze(-1)  # (K1,K2)
        # denom_21_norm = torch.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)  # (K1,)
        # denom_21_norm = denom_21_norm.view(-1, 1)  # (K1,1)
        # dist_21_norm = numerator_21_norm / denom_21_norm  # (K1,K2)

        epipolar_dist = dist_12 + dist_21
        # epipolar_dist_norm = dist_12_norm + dist_21_norm
        # color difference
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = torch.sum(color_diff**2, dim=2)  # (K1,K2)
        print("=== Before Normalization ===")
        print(f"epipolar_dist: min={epipolar_dist.min():.6f}, max={epipolar_dist.max():.6f}, mean={epipolar_dist.mean():.6f}")
        # print(f"epipolar_dist: min={epipolar_dist_norm.min():.6f}, max={epipolar_dist_norm.max():.6f}, mean={epipolar_dist_norm.mean():.6f}")
        print(f"d_color: min={d_color.min():.6f}, max={d_color.max():.6f}, mean={d_color.mean():.6f}")

        # max_dim = 1554  # largest image dimension
        
        epipolar_dist = epipolar_dist /  2
        # epipolar_dist_norm = epipolar_dist_norm / 2 # オーダー調整
        d_color = d_color / 3.0
        
        print("=== After Normalization ===")
        print(f"epipolar_dist: min={self.lambda_epipolar * epipolar_dist.min():.6f}, max={self.lambda_epipolar * epipolar_dist.max():.6f}, mean={self.lambda_epipolar * epipolar_dist.mean():.6f}")
        # print(f"epipolar_dist_norm: min={epipolar_dist_norm.min():.6f}, max={epipolar_dist_norm.max():.6f}, mean={epipolar_dist_norm.mean():.6f}")
        print(f"d_color: min={d_color.min():.6f}, max={d_color.max():.6f}, mean={d_color.mean():.6f}")
        
        # combine with weights, epipolar_dist can be scaled if needed
        cost_matrix = self.lambda_epipolar * epipolar_dist + self.lambda_color * d_color
        # cost_matrix = self.lambda_epipolar * epipolar_dist_norm + self.lambda_color * d_color
        return cost_matrix
    

    def optimize_with_fundamental(self, max_iter: int = 1000, tol: float = 1e-3) -> None:
        """Optimize the Fundamental matrix F using epipolar distance + color difference. Enforce rank-2 during forward to avoid broken momentum of Adam.

        Args:
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.
        """
        transport_dir = os.path.join("results", "transport_fundamental")
        os.makedirs(transport_dir, exist_ok=True)

        def rank2_enforce(mat: torch.Tensor) -> torch.Tensor:
            """Perform rank-2 projection in a no_grad block,
            then do a 'straight-through' approach so that
            the returned tensor still requires grad.
            """
            with torch.no_grad():
                u, s, vt = torch.linalg.svd(mat, full_matrices=False)
                s[-1] = 0.0
                mat_rank2 = u @ torch.diag(s) @ vt
                
                # スケール正規化: Frobeniusノルムが小さすぎないよう対策
                norm_val = torch.norm(mat_rank2, p="fro")  # Frobeniusノルム
                if norm_val > 1e-12:
                    mat_rank2 = mat_rank2 / norm_val
            
            # Use mat_rank2 for forward computation, but backpropagate to mat (original parameter)
            return mat + (mat_rank2 - mat).detach()
        
        if self.f is None:
            self.f = nn.Parameter(torch.eye(3, dtype=torch.float32, device=self.device))
        else:
            # 初期値がある場合はその値を使う（パイプライン側でsolver.fを設定）
            self.f = nn.Parameter(self.f.clone().detach())
        optimizer = torch.optim.Adam([self.f], lr=1e-2)
        
        prev_loss_val = torch.tensor(float('inf'), device=self.device)
        loss_history = []

        for iteration in range(max_iter):
            optimizer.zero_grad()

            # rank 2 enforce for forward computation
            f_enforced = rank2_enforce(self.f)  

            cost_matrix = self.compute_cost_matrix_fundamental(f_enforced)
            # print(f"cost_matrix: min={cost_matrix.min():.6f}, max={cost_matrix.max():.6f}, mean={cost_matrix.mean():.6f}")
            # print("=== cost_matrix ===")
            # sys.exit()
            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix, rho=0.5, max_iter=10000, tol=1e-6)
            loss = torch.sum(transport * cost_matrix)

            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            loss_history.append(current_loss)

            # check convergence　(add grad norm check as well?)
            if abs(prev_loss_val - current_loss) < tol:
                print(f"Converged at iteration {iteration}")
                break
            prev_loss_val = current_loss

            # debug print
            if iteration % 5 == 0 or iteration == max_iter - 1:
                grad_norm = 0.0
                if self.f.grad is not None:
                    grad_norm = self.f.grad.norm().item()
                print(f"Iteration {iteration}, Loss={current_loss:.6f}, F.grad norm={grad_norm:.6f}")

            # show transport matrix
            if iteration % 10 == 0 or iteration == max_iter - 1:
                with torch.no_grad():
                    t_np = transport.detach().cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    plt.imshow(t_np, cmap="hot", interpolation="nearest")
                    plt.colorbar(label="Transport Plan Value")
                    plt.title(f"Transport Plan at Iteration {iteration}")
                    plt.xlabel("Image 2 Gaussians")
                    plt.ylabel("Image 1 Gaussians")
                    plt.tight_layout()
                    plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
                    plt.savefig(plt_path)
                    plt.close()
                    print(f"Transport matrix heatmap saved to '{plt_path}'")

        with torch.no_grad():
            final_rank2 = rank2_enforce(self.f)
            self.f.copy_(final_rank2)
            print("Final rank-2 enforcement on fundamental matrix.")

        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Fundamental Optimization Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "fundamental_loss.png"))
        plt.close()
        print(f"Saved fundamental loss plot to '{transport_dir}'")

        # detach
        self.f = self.f.detach()
        print("Done optimizing fundamental.")


    #計算遅いから一旦使わない
    # def optimize_with_fundamental_svd(self, max_iter: int = 1000, tol: float = 1e-3) -> None:
    #     """
    #     Optimize the Fundamental matrix F using epipolar distance + color difference.
    #     F is parameterized as F = U @ S @ V.T with S = diag(sigma1, sigma2, 0) so that F is always rank-2.

    #     Args:
    #         max_iter (int): Maximum number of iterations.
    #         tol (float): Convergence tolerance.
    #     """
    #     transport_dir = os.path.join("results", "transport_fundamental")
    #     os.makedirs(transport_dir, exist_ok=True)

    #     # Initialize learnable parameters:
    #     # - U and V are 3x3 matrices (initialized to identity) and will be re-orthonormalized.
    #     # - raw_singular will be optimized and then converted to singular_vals using softplus (non-negative).
    #     U_init = torch.eye(3, dtype=torch.float32, device=self.device)
    #     V_init = torch.eye(3, dtype=torch.float32, device=self.device)
    #     self.U = torch.nn.Parameter(U_init.clone())
    #     self.V = torch.nn.Parameter(V_init.clone())
    #     # Optimize raw singular values; softplus will ensure they remain non-negative.
    #     self.raw_singular = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32, device=self.device))

    #     # Adam now optimizes U, V, and raw_singular.
    #     optimizer = torch.optim.Adam([self.U, self.V, self.raw_singular], lr=1e-4)
    #     prev_loss = float("inf")
    #     loss_history = []

    #     for iteration in tqdm(range(max_iter), desc="Optimization with Fundamental"):
    #         optimizer.zero_grad()

    #         # Reconstruct F from the learnable parameters:
    #         # 1. Orthonormalize U and V using QR decomposition.
    #         U_q, _ = torch.linalg.qr(self.U)
    #         # Create a new tensor if the determinant is negative. (No in-place modification)
    #         if torch.det(U_q) < 0:
    #             U_q = torch.cat([-U_q[:, :1], U_q[:, 1:]], dim=1)

    #         V_q, _ = torch.linalg.qr(self.V)
    #         if torch.det(V_q) < 0:
    #             V_q = torch.cat([-V_q[:, :1], V_q[:, 1:]], dim=1)

    #         # Convert raw_singular to non-negative singular values.
    #         singular_vals = torch.nn.functional.softplus(self.raw_singular)
    #         S_diag = torch.cat([singular_vals, torch.tensor([0.0], device=self.device)])
    #         S = torch.diag(S_diag)
    #         F = U_q @ S @ V_q.T

    #         # Compute cost matrix using the current F.
    #         cost_matrix = self.compute_cost_matrix_fundamental(F)
    #         # Compute transport plan using unbalanced Sinkhorn.
    #         transport = self.unbalanced_sinkhorn_algorithm(
    #             cost_matrix, rho=1.0, max_iter=10000, tol=1e-6
    #         )
    #         # Define the loss as the inner product between transport and cost_matrix.
    #         loss = torch.sum(transport * cost_matrix)
    #         loss.backward()
    #         optimizer.step()

    #         if iteration % 5 == 0 or iteration == max_iter - 1:
    #             print(f"Iteration {iteration}, Loss={loss.item():.6f}")
    #         loss_history.append(loss.item())

    #         if abs(prev_loss - loss.item()) < tol:
    #             print(f"Converged at iteration {iteration}")
    #             break
    #         prev_loss = loss.item()

    #         if iteration % 100 == 0 or iteration == max_iter - 1:
    #             with torch.no_grad():
    #                 t_np = transport.cpu().numpy()
    #                 plt.figure(figsize=(8, 6))
    #                 plt.imshow(t_np, cmap="hot", interpolation="nearest")
    #                 plt.colorbar(label="Transport Plan Value")
    #                 plt.title(f"Transport Plan at Iteration {iteration}")
    #                 plt.xlabel("Image 2 Gaussians")
    #                 plt.ylabel("Image 1 Gaussians")
    #                 plt.tight_layout()
    #                 plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
    #                 plt.savefig(plt_path)
    #                 plt.close()
    #                 print(f"Transport matrix heatmap saved to '{plt_path}'")

    #     # After optimization, reconstruct and save the final F.
    #     with torch.no_grad():
    #         U_q, _ = torch.linalg.qr(self.U)
    #         if torch.det(U_q) < 0:
    #             U_q = torch.cat([-U_q[:, :1], U_q[:, 1:]], dim=1)

    #         V_q, _ = torch.linalg.qr(self.V)
    #         if torch.det(V_q) < 0:
    #             V_q = torch.cat([-V_q[:, :1], V_q[:, 1:]], dim=1)

    #         singular_vals = torch.nn.functional.softplus(self.raw_singular)
    #         S_diag = torch.cat([singular_vals, torch.tensor([0.0], device=self.device)])
    #         S = torch.diag(S_diag)
    #         self.f = U_q @ S @ V_q.T
    #         print("Final Fundamental matrix computed from SVD parameters.")

    #     self.f = self.f.detach()


    # def optimize_with_fundamental_visualize_momentum(self, max_iter: int = 1000, tol: float = 1e-3) -> None:
    #     """Optimize the Fundamental matrix F using epipolar distance + color difference. 
    #     Enforce rank-2 during forward to avoid broken momentum of Adam.

    #     Args:
    #         max_iter (int): Maximum number of iterations.
    #         tol (float): Convergence tolerance.
    #     """
    #     transport_dir = os.path.join("results", "transport_fundamental")
    #     os.makedirs(transport_dir, exist_ok=True)

    #     def rank2_enforce(mat: torch.Tensor) -> torch.Tensor:
    #         """Perform rank-2 projection in a no_grad block,
    #         then do a 'straight-through' approach so that
    #         the returned tensor still requires grad.
    #         """
    #         with torch.no_grad():
    #             u, s, vt = torch.linalg.svd(mat, full_matrices=False)
    #             s[-1] = 0.0
    #             mat_rank2 = u @ torch.diag(s) @ vt
            
    #         # Use mat_rank2 for forward computation, but backpropagate to mat (original parameter)
    #         return mat + (mat_rank2 - mat).detach()
        
    #     # F の初期化
    #     if self.f is None:
    #         self.f = nn.Parameter(torch.eye(3, dtype=torch.float32, device=self.device))
    #     else:
    #         # 初期値がある場合はその値を使う（パイプライン側でsolver.fを設定）
    #         self.f = nn.Parameter(self.f.clone().detach())

    #     # Adam オプティマイザ
    #     optimizer = torch.optim.Adam([self.f], lr=1e-4)
        
    #     # 収束判定用
    #     prev_loss_val = torch.tensor(float('inf'), device=self.device)
    #     loss_history = []

    #     # ★ 追加: Adamのモーメントと rank2 の差分ノルムを可視化するための配列
    #     m1_norm_history = []
    #     m2_norm_history = []
    #     rank2_diff_history = []

    #     for iteration in range(max_iter):
    #         optimizer.zero_grad()

    #         # rank 2 enforce for forward computation
    #         f_enforced = rank2_enforce(self.f)

    #         # コスト行列の計算
    #         cost_matrix = self.compute_cost_matrix_fundamental(f_enforced)
            
    #         # アンバランスドSinkhornでtransportを計算
    #         transport = self.unbalanced_sinkhorn_algorithm(cost_matrix, rho=1.0, max_iter=10000, tol=1e-6)
            
    #         # ロス計算
    #         loss = torch.sum(transport * cost_matrix)
    #         loss.backward()

    #         optimizer.step()

    #         current_loss = loss.item()
    #         loss_history.append(current_loss)

    #         # ★ 追加: rank2_enforceでの差分ノルムを測る
    #         #         f_enforced (ランク2投影後) と self.f (オリジナル) のノルム差
    #         with torch.no_grad():
    #             rank2_diff = (f_enforced - self.f).norm().item()
    #             rank2_diff_history.append(rank2_diff)
            
    #         # ★ 追加: Adam のモーメントを取得してノルムを記録
    #         with torch.no_grad():
    #             # optimizer.state[self.f] に 'exp_avg' (m1) と 'exp_avg_sq' (m2) が入っている
    #             m1 = optimizer.state[self.f]["exp_avg"]
    #             m2 = optimizer.state[self.f]["exp_avg_sq"]
    #             m1_norm_history.append(m1.norm().item())
    #             m2_norm_history.append(m2.norm().item())

    #         # 収束判定
    #         if abs(prev_loss_val - current_loss) < tol:
    #             print(f"Converged at iteration {iteration}")
    #             break
    #         prev_loss_val = current_loss

    #         # debug print
    #         if iteration % 5 == 0 or iteration == max_iter - 1:
    #             grad_norm = 0.0
    #             if self.f.grad is not None:
    #                 grad_norm = self.f.grad.norm().item()
    #             print(f"Iteration {iteration}, Loss={current_loss:.6f}, "
    #                   f"F.grad norm={grad_norm:.6f}, rank2_diff={rank2_diff:.6f}")

    #         # show transport matrix 
    #         if iteration % 10 == 0 or iteration == max_iter - 1:
    #             with torch.no_grad():
    #                 t_np = transport.detach().cpu().numpy()
    #                 plt.figure(figsize=(8, 6))
    #                 plt.imshow(t_np, cmap="hot", interpolation="nearest")
    #                 plt.colorbar(label="Transport Plan Value")
    #                 plt.title(f"Transport Plan at Iteration {iteration}")
    #                 plt.xlabel("Image 2 Gaussians")
    #                 plt.ylabel("Image 1 Gaussians")
    #                 plt.tight_layout()
    #                 plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
    #                 plt.savefig(plt_path)
    #                 plt.close()
    #                 print(f"Transport matrix heatmap saved to '{plt_path}'")

    #     # 最終的にランク2 enforce をかけて self.f に反映
    #     with torch.no_grad():
    #         final_rank2 = rank2_enforce(self.f)
    #         self.f.copy_(final_rank2)
    #         print("Final rank-2 enforcement on fundamental matrix.")

    #     # ====== visualization ======

    #     # 1) Loss
    #     plt.figure()
    #     plt.plot(loss_history, label="loss")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Loss")
    #     plt.title("Fundamental Optimization Loss")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(transport_dir, "fundamental_loss.png"))
    #     plt.close()
    #     print(f"Saved fundamental loss plot to '{transport_dir}'")

    #     # 2) Rank-2差分 
    #     plt.figure()
    #     plt.plot(rank2_diff_history, label="rank2_diff")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("||F_enforced - F||")
    #     plt.title("Rank2 Difference per Iteration")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(transport_dir, "rank2_diff.png"))
    #     plt.close()
    #     print(f"Saved rank2 difference plot to '{transport_dir}'")

    #     # 3) Adam のモーメント ノルム
    #     plt.figure()
    #     plt.plot(m1_norm_history, label="exp_avg (1st moment) norm")
    #     plt.plot(m2_norm_history, label="exp_avg_sq (2nd moment) norm")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Norm value")
    #     plt.title("Adam Moments Norm")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.savefig(os.path.join(transport_dir, "adam_moments.png"))
    #     plt.close()
    #     print(f"Saved Adam moments plot to '{transport_dir}'")

    #     # detach
    #     self.f = self.f.detach()
    #     print("Done optimizing fundamental.")