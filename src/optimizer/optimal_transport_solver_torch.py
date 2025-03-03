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
        epsilon: float = 0.01,
        lambda_mean: float = 1.0,
        lambda_cov: float = 0.3,
        lambda_color: float = 1.0,
        lambda_epipolar: float = 1.0,
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
    
    def rodrigues(self, rvec: torch.Tensor) -> torch.Tensor:
        """Rodrigues変換 (OpenCVのcv2.Rodrigues相当)。
        rvec: (3,) -> 回転ベクトル
        Returns : (3,3) 回転行列
        """
        # ノルム(回転角)
        theta = torch.norm(rvec) + 1e-12
        # 単位方向
        r_axis = rvec / theta

        # 外積行列Kを「定数tensor([...])」ではなく，zeros + 代入で組み立て(じゃないと勾配流れない...)
        K = torch.zeros((3,3), dtype=torch.float32, device=rvec.device)
        K[0,1] = -r_axis[2]
        K[0,2] =  r_axis[1]
        K[1,0] =  r_axis[2]
        K[1,2] = -r_axis[0]
        K[2,0] = -r_axis[1]
        K[2,1] =  r_axis[0]

        # Rodrigues formula
        I = torch.eye(3, dtype=torch.float32, device=rvec.device)
        R = I + torch.sin(theta)*K + (1.0 - torch.cos(theta))*(K @ K)
        return R

    def build_f_from_rt(self, rvec: torch.Tensor, tvec: torch.Tensor) -> torch.Tensor:
        """rvec, tvec から F を構築。
        F = K2^-T [t]_x R K1^-1
        """
        R = self.rodrigues(rvec)

        # [t]_x も同様に zeros + 代入
        tx = torch.zeros((3,3), device=self.device)
        tx[0,1] = -tvec[2]
        tx[0,2] =  tvec[1]
        tx[1,0] =  tvec[2]
        tx[1,2] = -tvec[0]
        tx[2,0] = -tvec[1]
        tx[2,1] =  tvec[0]

        E = tx @ R  # (3,3)

        K1_inv = torch.inverse(self.k1)
        K2_inv = torch.inverse(self.k2)
        K2_inv_T = K2_inv.transpose(0,1)

        F = K2_inv_T @ E @ K1_inv
        return F


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
        rho: float = 0.1,
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
        # print(f"rho: {rho}, epsilon: {self.epsilon}, exponent: {exponent}")

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
                # print(f"[Unbalanced] iteration {iteration} -> converged.")
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


    def compute_cost_matrix_fundamental_sampson(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using the Sampson error
        with a Fundamental Matrix F. Also includes color difference term as an example.

        Sampson error for a pair of correspondences (p1, p2):
            d_sampson(p1, p2) = (p2^T * F * p1)^2
                                --------------------------------
                                (F * p1)[0]^2 + (F * p1)[1]^2 + (F^T * p2)[0]^2 + (F^T * p2)[1]^2

        The resulting cost is then combined with a color difference term.

        Args:
            f (torch.Tensor): The Fundamental matrix (3 x 3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
                        cost_matrix[i,j] = lambda_epipolar * SampsonError(i,j) + lambda_color * colorDiff(i,j)
        """
        # ----------------------------------------
        # Optional: Normalize image points by image width/height or by Hartley normalization
        # w, h = 1554, 1162
        # means1_norm = self.means1 / torch.tensor([w, h], dtype=torch.float32, device=self.device)
        # means2_norm = self.means2 / torch.tensor([w, h], dtype=torch.float32, device=self.device)
        # (Or use a custom function that does full Hartley normalization.)
        #
        # For now, we assume self.means1, self.means2 are in raw image coordinates.
        # ----------------------------------------

        # Number of Gaussians in each image
        k1 = self.means1.shape[0]
        k2 = self.means2.shape[0]

        # Create homogeneous coords
        ones1 = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
        p1_homo = torch.cat([self.means1, ones1], dim=1)  # (K1,3)

        ones2 = torch.ones((k2, 1), dtype=torch.float32, device=self.device)
        p2_homo = torch.cat([self.means2, ones2], dim=1)  # (K2,3)

        # --------------------------
        #  Sampson error calculation
        # --------------------------
        # F * p1 (shape: (3, K1))
        Fx1 = f @ p1_homo.T
        # F^T * p2 (shape: (3, K2))
        Ftx2 = f.t() @ p2_homo.T

        # 分子: (p2^T * F * p1)^2
        # -> p2_homo (K2,3) dot Fx1 (3, K1) -> shape (K2, K1)
        # -> transpose to (K1, K2)
        dot_vals = p2_homo @ Fx1  # (K2, K1)
        numerator = dot_vals.T.pow(2)  # (K1, K2)

        # 分母: (F p1)_x^2 + (F p1)_y^2 + (F^T p2)_x^2 + (F^T p2)_y^2
        # (F p1) -> shape (3, K1), take first 2 rows => (2, K1), sum of squares over row => (K1,)
        Fx1_sq = Fx1[:2, :].pow(2).sum(dim=0)  # shape: (K1,)
        Ftx2_sq = Ftx2[:2, :].pow(2).sum(dim=0)  # shape: (K2,)

        denominator = Fx1_sq.unsqueeze(1) + Ftx2_sq.unsqueeze(0) + 1e-12  # shape: (K1, K2)

        # Sampson error (K1, K2)
        sampson_error = numerator / denominator
        sampson_error = sampson_error / 1e5  

        # ---------------
        # Color difference
        # ---------------
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1, K2, 3)
        d_color = torch.sum(color_diff ** 2, dim=2)  # (K1, K2)

        # ------------
        # Debug prints
        # ------------
        print("=== Sampson Error Stats ===")
        print(f"  min={sampson_error.min():.6f}, max={sampson_error.max():.6f}, mean={sampson_error.mean():.6f}")
        print("=== Color Diff Stats ===")
        print(f"  min={d_color.min():.6f}, max={d_color.max():.6f}, mean={d_color.mean():.6f}")

        # ------------------------------------------
        # Example: Simple normalization or scaling
        # ------------------------------------------
        # e.g. Optional small scaling to keep values in a good range
        # sampson_error = sampson_error / 2.0
        # d_color       = d_color / 3.0

        # -------------
        # Combine costs
        # -------------
        cost_matrix = self.lambda_epipolar * sampson_error + self.lambda_color * d_color

        return cost_matrix

    def compute_cost_matrix_fundamental(self, f: torch.Tensor) -> torch.Tensor:
        """Compute the cost matrix between two sets of 2D Gaussians using a Fundamental Matrix.

        This replaces the Homography-based distance with an epipolar distance.
        Also includes color difference term as an example.

        Args:
            f (torch.Tensor): The Fundamental matrix (3x3).

        Returns:
            torch.Tensor: Cost matrix of shape (K1, K2).
        """
         # 1) 画像1,2 それぞれのGaussians数
        k1 = self.means1.shape[0]
        k2 = self.means2.shape[0]

        # 2) 平均点を同次座標化 (x,y,1)
        ones1 = torch.ones((k1, 1), dtype=torch.float32, device=self.device)
        p1_homo = torch.cat([self.means1, ones1], dim=1)  # (K1,3)

        ones2 = torch.ones((k2, 1), dtype=torch.float32, device=self.device)
        p2_homo = torch.cat([self.means2, ones2], dim=1)  # (K2,3)

        # 3) p2_j に対応するエピポーラ線を画像1上で計算: l1[j] = F * p2[j]
        #    p1_i に対応するエピポーラ線を画像2上で計算: l2[i] = F^T * p1[i]
        #    (shape: l1 -> (K2,3), l2 -> (K1,3))
        l1 = (f @ p2_homo.T).T  # (K2,3)
        l2 = (f.t() @ p1_homo.T).T  # (K1,3)

        # 4) p1[i] と l1[j] の距離をペアごとに計算
        #    l1[j] = (a_j, b_j, c_j), p1[i] = (x_i, y_i, 1)
        #    dist_12(i,j) = | p1[i]·l1[j] | / sqrt(a_j^2 + b_j^2)
        #
        #   - numerator_12(i,j) = | p1[i]·l1[j] |
        #   - denominator_12(j) = sqrt(a_j^2 + b_j^2)
        #   → shape はそれぞれ (K1,K2), (K2,) になり
        #     dist_12 = numerator_12 / denom_12(ブロードキャスト)
        numerator_12 = torch.abs(p1_homo @ l1.T)    # => (K1, K2)
        denom_12 = torch.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2 + 1e-12)  # (K2,)
        denom_12 = denom_12.view(1, -1)  # (1,K2) for broadcast
        dist_12 = numerator_12 / denom_12  # (K1,K2)

        # 5) p2[j] と l2[i] の距離をペアごとに計算
        #    l2[i] = (a_i, b_i, c_i), p2[j] = (x_j, y_j, 1)
        #    dist_21(i,j) = | p2[j]·l2[i] | / sqrt(a_i^2 + b_i^2)
        #
        #   - numerator_21(i,j) = | p2[j]·l2[i] |
        #   - denominator_21(i)  = sqrt(a_i^2 + b_i^2)
        #   → shape はそれぞれ (K2,K1) (K1,) となるのを転置して (K1,K2) など
        numerator_21 = torch.abs(p2_homo @ l2.T)    # => (K2, K1)
        numerator_21 = numerator_21.T              # => (K1, K2)
        denom_21 = torch.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2 + 1e-12)  # (K1,)
        denom_21 = denom_21.view(-1, 1)            # (K1,1)
        dist_21 = numerator_21 / denom_21          # (K1,K2)

        # 6) 対称エピポーラ距離を合計
        #    epipolar_dist(i,j) = dist_12(i,j) + dist_21(i,j)
        epipolar_dist = dist_12 + dist_21

        # 7) カラー差分 (RGB) を計算
        #    color_diff(i,j) = sum_k ( rgb1[i][k] - rgb2[j][k] )^2
        color_diff = self.rgb1.unsqueeze(1) - self.rgb2.unsqueeze(0)  # (K1,K2,3)
        d_color = torch.sum(color_diff ** 2, dim=2)  # (K1,K2)

        # - デバッグ出力（正規化前）
        # print("=== Before Normalization ===")
        # print(f"epipolar_dist: min={epipolar_dist.min():.6f}, "
        #     f"max={epipolar_dist.max():.6f}, mean={epipolar_dist.mean():.6f}")
        # print(f"d_color:       min={d_color.min():.6f}, "
        #     f"max={d_color.max():.6f}, mean={d_color.mean():.6f}")

        # 8) （任意のスケーリング・調整）
        # epipolar_max = epipolar_dist.max() + 1e-12
        # epipolar_dist = epipolar_dist / epipolar_max
        # d_color = d_color / 3.0             # 例: 色差も軽く正規化
        with torch.no_grad():
            p95_epipolar = torch.quantile(epipolar_dist, 0.95)
            p95_color = torch.quantile(d_color, 0.95)

        epipolar_dist = torch.clamp(epipolar_dist, max=p95_epipolar) / p95_epipolar
        d_color = torch.clamp(d_color, max=p95_color) / p95_color

        # print("=== After Normalization ===")
        # print(f"epipolar_dist: min={self.lambda_epipolar * epipolar_dist.min():.6f}, "
        #     f"max={self.lambda_epipolar * epipolar_dist.max():.6f}, "
        #     f"mean={self.lambda_epipolar * epipolar_dist.mean():.6f}")
        # print(f"d_color:       min={d_color.min():.6f}, "
        #     f"max={d_color.max():.6f}, mean={d_color.mean():.6f}")

        # 9) 上記2種のコストを重み付けして合成
        #    cost_matrix(i,j) = lambda_epipolar * epipolar_dist(i,j)
        #                     + lambda_color   * d_color(i,j)
        cost_matrix = self.lambda_epipolar * epipolar_dist + self.lambda_color * d_color

        return cost_matrix
        


    
    
    def optimize_with_RT(self, max_iter=1000, tol=1e-6):
        """R,tを直接最適化してFを構築して self.f に反映させる。
        最終的に得られた rvec,tvec を「カメラ姿勢(R,t)」として利用する想定。
        """

        # R,tを最適化パラメータ設定  (これは相対変換)
        if not hasattr(self, 'rvec'):
            self.rvec = nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
        if not hasattr(self, 'tvec'):
            self.tvec = nn.Parameter(torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=self.device))
        
        optimizer = torch.optim.Adam([self.rvec, self.tvec], lr=1e-3)
        prev_loss_val = float('inf')
        loss_history = []

        transport_dir = os.path.join("results", "transport_RT")
        os.makedirs(transport_dir, exist_ok=True)

        pbar = tqdm(range(max_iter), desc="Optimizing R,t", leave=True)
        
        for iteration in pbar:
            optimizer.zero_grad()

            F = self.build_f_from_rt(self.rvec, self.tvec)

            cost_matrix = self.compute_cost_matrix_fundamental(F)

            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix, rho=0.5, max_iter=10000, tol=1e-6)

            # ロス計算 + 逆伝播
            loss = torch.sum(transport * cost_matrix)
            loss.backward()

            # パラメータ更新
            optimizer.step()

            current_loss = loss.item()
            loss_history.append(current_loss)

            # 収束判定
            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss

            if iteration % 10 == 0:
                grad_r = self.rvec.grad.norm().item()
                grad_t = self.tvec.grad.norm().item()
                pbar.set_postfix(loss=f"{current_loss:.6f}", grad_r=f"{grad_r:.6f}", grad_t=f"{grad_t:.6f}")

                # Transport matrix 可視化
                if iteration % 10 == 0 or iteration == max_iter - 1:
                    with torch.no_grad():
                        t_np = transport.detach().cpu().numpy()
                        
                        # 行列の次元に応じてサイズとアスペクト比を調整
                        rows, cols = t_np.shape
                        aspect_ratio = cols / rows
                        
                        if rows > cols:
                            fig_width = 8  # 基本幅
                            fig_height = min(20, fig_width / aspect_ratio)  # 高さは幅/アスペクト比（最大20に制限）
                        else:
                            fig_height = 6  # 基本高さ
                            fig_width = min(20, fig_height * aspect_ratio)  # 幅は高さ*アスペクト比（最大20に制限）
                        
                        plt.figure(figsize=(fig_width, fig_height))
                        
                        # 大きな行列の場合はダウンサンプリング
                        if rows > 1000 or cols > 1000:
                            downsample_factor = max(1, int(max(rows, cols) / 1000))
                            t_np_display = t_np[::downsample_factor, ::downsample_factor]
                            plt.imshow(t_np_display, cmap="hot", interpolation="nearest", aspect="auto")
                            plt.title(f"Transport Plan at Iteration {iteration} (Downsampled {downsample_factor}x)")
                        else:
                            plt.imshow(t_np, cmap="hot", interpolation="nearest", aspect="auto")
                            plt.title(f"Transport Plan at Iteration {iteration}")
                        
                        plt.colorbar(label="Transport Plan Value")
                        plt.xlabel("Image 2 Gaussians")
                        plt.ylabel("Image 1 Gaussians")
                        
                        # 軸ラベルの位置調整
                        plt.tight_layout()
                        plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
                        plt.savefig(plt_path, dpi=150)
                        plt.close()

        print("Optimized rvec:", self.rvec)
        print("Optimized tvec:", self.tvec)

        # build_f_from_rt から最終Fを取り出す
        final_F = self.build_f_from_rt(self.rvec, self.tvec).detach()
        print("Final F:\n", final_F.cpu().numpy())

        # solver.f にコピー
        with torch.no_grad():
            self.f = final_F

        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_RT)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_RT.png"))
        plt.close()
