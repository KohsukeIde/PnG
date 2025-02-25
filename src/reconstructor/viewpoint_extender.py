# src/reconstructor/viewpoint_extender.py
import numpy as np
import torch
import os
from typing import List, Optional, Tuple, Dict, Any

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.reconstructor.initial_3d_non_linear import build_covariance_3d



class ViewpointExtender:
    """
    Extends an existing 3D Gaussian scene by adding a new viewpoint.
    
    Takes an existing 3D Gaussian distribution, projects it onto a reference viewpoint,
    matches it to a new image's 2D Gaussians, and solves for the new viewpoint's camera
    parameters using optimal transport.
    """

    def __init__(
        self,
        existing_3d_gaussians: np.ndarray,
        camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
        K_new: np.ndarray,
        reference_camera_idx: int,
        threshold_reprojection: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the ViewpointExtender.

        Args:
            existing_3d_gaussians: Array of 3D Gaussian parameters (center, covariance, etc.)
            camera_params_list: List of (R, t) for each registered camera
            K_new: Intrinsic matrix for the new viewpoint
            reference_camera_idx: Index of the reference camera for projection
            threshold_reprojection: Threshold for filtering outliers
            device: Device to run computations on (CPU/GPU)
        """
        self.existing_3d_gaussians = existing_3d_gaussians
        self.camera_params_list = camera_params_list
        self.K_new = K_new
        self.reference_camera_idx = reference_camera_idx
        self.threshold_reprojection = threshold_reprojection
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Transport solver will be initialized later when needed
        self.transport_solver = None
        
        # Camera parameters for the new viewpoint (to be optimized)
        self.rvec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
        self.tvec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
        
        # Validate camera parameters
        if len(self.camera_params_list) <= reference_camera_idx:
            raise ValueError(f"Reference camera index {reference_camera_idx} is out of bounds")
        
        # Convert K_new to torch tensor if it's not already
        if not isinstance(self.K_new, torch.Tensor):
            self.K_new = torch.tensor(self.K_new, dtype=torch.float32, device=self.device)

    def project_3d_gaussians(self) -> TwoDGaussians:
        """
        Project existing 3D Gaussians onto 2D using the reference camera.
        
        Returns:
            TwoDGaussians: Projected 2D Gaussians
        """
        # Get reference camera parameters
        R_ref, t_ref = self.camera_params_list[self.reference_camera_idx]
        
        # Convert to torch tensors if they're not already
        if not isinstance(R_ref, torch.Tensor):
            R_ref = torch.tensor(R_ref, dtype=torch.float32, device=self.device)
        if not isinstance(t_ref, torch.Tensor):
            t_ref = torch.tensor(t_ref, dtype=torch.float32, device=self.device)
            
        # Lists to collect 2D Gaussian parameters
        means_2d = []
        covs_2d = []
        rotations_2d = []
        scales_2d = []
        rgb_values = []
        alpha_values = []
        
        # For each 3D Gaussian
        for idx, gauss in enumerate(self.existing_3d_gaussians):
            # Extract 3D Gaussian parameters
            center_3d = gauss["center"]
            quat = gauss["quat"]
            scale_3d = gauss["scale3d"]
            color = gauss["color"]
            alpha = gauss["alpha"]
            
            # Convert to torch tensors if needed
            if not isinstance(center_3d, torch.Tensor):
                center_3d = torch.tensor(center_3d, dtype=torch.float32, device=self.device)
            
            # Build 3D covariance matrix from quaternion and scale
            sigma_3d = build_covariance_3d(quat, scale_3d)
            
            # Project center to camera coordinates
            x_cam = R_ref @ center_3d + t_ref
            
            # Check if point is in front of camera
            if x_cam[2] <= 1e-6:
                continue  # Skip points behind the camera
                
            # Project to image coordinates
            px_hom = self.K_new @ x_cam
            center_2d = px_hom[:2] / px_hom[2]
            
            # Project 3D covariance to 2D
            sigma_2d = self._project_covariance_3d_to_2d(
                sigma_3d, center_3d, self.K_new, R_ref, t_ref
            )
            
            # Get rotation angle and scales from 2D covariance
            rot_angle, scale_xy = self._convert_cov2d_to_params(sigma_2d)
            
            # Append to lists
            means_2d.append(center_2d.cpu().numpy() if isinstance(center_2d, torch.Tensor) else center_2d)
            covs_2d.append(sigma_2d.cpu().numpy() if isinstance(sigma_2d, torch.Tensor) else sigma_2d)
            rotations_2d.append(rot_angle)
            scales_2d.append(scale_xy)
            rgb_values.append(color)
            alpha_values.append(alpha)
        
        # Convert lists to arrays
        means_2d_arr = np.array(means_2d, dtype=np.float32)
        covs_2d_arr = np.array(covs_2d, dtype=np.float32)
        rotations_2d_arr = np.array(rotations_2d, dtype=np.float32)
        scales_2d_arr = np.array(scales_2d, dtype=np.float32)
        rgb_arr = np.array(rgb_values, dtype=np.float32)
        alpha_arr = np.array(alpha_values, dtype=np.float32)
        
        # Create TwoDGaussians object
        projected_gaussians = TwoDGaussians(
            means=means_2d_arr,
            covs=covs_2d_arr,
            rgb=rgb_arr,
            alpha=alpha_arr,
            rotations=rotations_2d_arr,
            scales=scales_2d_arr
        )
        
        return projected_gaussians

    def _project_covariance_3d_to_2d(
        self,
        sigma_3: np.ndarray,
        point_3d: np.ndarray,
        K: np.ndarray,
        R_cam: np.ndarray,
        t_cam: np.ndarray,
    ) -> np.ndarray:
        """
        Project a 3D covariance to 2D using the local Jacobian approximation.
        
        Args:
            sigma_3: 3D covariance matrix (3x3)
            point_3d: 3D point in world coordinates (3,)
            K: Camera intrinsic matrix (3x3)
            R_cam: Camera rotation matrix (3x3)
            t_cam: Camera translation vector (3,)
            
        Returns:
            2D covariance matrix (2x2)
        """
        # Convert to numpy if tensors
        if isinstance(sigma_3, torch.Tensor):
            sigma_3 = sigma_3.cpu().numpy()
        if isinstance(point_3d, torch.Tensor):
            point_3d = point_3d.cpu().numpy()
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        if isinstance(R_cam, torch.Tensor):
            R_cam = R_cam.cpu().numpy()
        if isinstance(t_cam, torch.Tensor):
            t_cam = t_cam.cpu().numpy()
        
        # Project 3D point to camera coordinates
        x_c = R_cam @ point_3d + t_cam
        x, y, z = x_c
        
        # Avoid division by zero
        if abs(z) < 1e-12:
            z = 1e-12
            
        # Get focal lengths
        fx, fy = K[0, 0], K[1, 1]
        
        # Compute Jacobian of projection
        J = np.array([
            [fx / z, 0.0, -fx * x / (z**2)],
            [0.0, fy / z, -fy * y / (z**2)]
        ], dtype=np.float64)
        
        # Project covariance to camera coordinates
        sigma_cam = R_cam @ sigma_3 @ R_cam.T
        
        # Project to 2D using Jacobian
        sigma_2d = J @ sigma_cam @ J.T
        
        return sigma_2d

    def _convert_cov2d_to_params(self, cov_2d: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert a 2D covariance matrix to rotation angle and scales.
        
        Args:
            cov_2d: 2D covariance matrix (2x2)
            
        Returns:
            Tuple of (rotation_angle, [scale_x, scale_y])
        """
        # Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        
        # Sort by descending eigenvalue
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Ensure positive eigenvalues (should already be, but just in case)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Compute scales as sqrt of eigenvalues
        scales = np.sqrt(eigvals)
        
        # Compute rotation angle from principal eigenvector
        principal_axis = eigvecs[:, 0]
        angle = np.arctan2(principal_axis[1], principal_axis[0])
        
        return float(angle), scales

    def match_2d_gaussians(
        self,
        projected_2d_gaussians: TwoDGaussians,
        new_2d_gaussians: TwoDGaussians,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match projected 2D Gaussians with new 2D Gaussians using optimal transport.
        
        Args:
            projected_2d_gaussians: Projected 2D Gaussians from existing 3D model
            new_2d_gaussians: 2D Gaussians from the new viewpoint
            
        Returns:
            Tuple of (projected_indices, new_indices) for matching Gaussians
        """
        if self.transport_solver is None:
            raise ValueError("Transport solver not initialized. Call initialize_transport_solver first.")
        
        # Make sure the F matrix is set in the transport solver
        if self.transport_solver.f is None:
            # Initialize F from rvec and tvec
            with torch.no_grad():
                F = self.transport_solver.build_f_from_rt(self.rvec, self.tvec)
                self.transport_solver.f = F
        
        # Compute cost matrix
        cost_matrix = self.transport_solver.compute_cost_matrix_fundamental(
            self.transport_solver.f
        )
        
        # Compute transport plan
        transport = self.transport_solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, rho=0.5, max_iter=10000, tol=1e-6
        )
        
        # Convert to numpy
        transport_np = transport.detach().cpu().numpy()
        
        # Find top matches
        matches = []
        threshold = self.threshold_reprojection
        
        # Flatten and sort transport values in descending order
        transport_flat = transport_np.flatten()
        sorted_indices = np.argsort(-transport_flat)
        
        # Get dimensions
        k1, k2 = transport_np.shape
        
        # Track which Gaussians have been matched
        matched_proj = set()
        matched_new = set()
        
        # Extract matches
        for idx in sorted_indices:
            if transport_flat[idx] < threshold:
                break
                
            i, j = np.unravel_index(idx, transport_np.shape)
            
            # Ensure one-to-one matching (greedy approach)
            if i not in matched_proj and j not in matched_new:
                matches.append((i, j))
                matched_proj.add(i)
                matched_new.add(j)
        
        # Convert to arrays
        if matches:
            matched_proj_indices, matched_new_indices = zip(*matches)
            return np.array(matched_proj_indices), np.array(matched_new_indices)
        else:
            return np.array([], dtype=int), np.array([], dtype=int)

    def initialize_transport_solver(
        self,
        projected_2d_gaussians: TwoDGaussians,
        new_2d_gaussians: TwoDGaussians,
        epsilon: float = 0.01,
        lambda_epipolar: float = 1e-3,
        lambda_color: float = 1.0,
    ) -> None:
        """
        Initialize the transport solver for 2D Gaussian matching.
        
        Args:
            projected_2d_gaussians: Projected 2D Gaussians from existing 3D model
            new_2d_gaussians: 2D Gaussians from the new viewpoint
            epsilon: Entropy regularization parameter
            lambda_epipolar: Weight for epipolar constraint
            lambda_color: Weight for color difference
        """
        self.transport_solver = OptimalTransportSolver(
            gaussians1=projected_2d_gaussians,
            gaussians2=new_2d_gaussians,
            k1=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            k2=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            epsilon=epsilon,
            lambda_mean=0.0,
            lambda_cov=0.0,
            lambda_color=lambda_color,
            lambda_epipolar=lambda_epipolar,
            device=self.device
        )
        
        # rvecとtvecを明示的に初期化
        self.transport_solver.rvec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=self.device))
        self.transport_solver.tvec = torch.nn.Parameter(torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=self.device))
        
        # ViewpointExtenderのrvecとtvecを設定
        self.rvec = self.transport_solver.rvec
        self.tvec = self.transport_solver.tvec
    
    def select_new_viewpoint(self, candidates: List[TwoDGaussians]) -> int:
        """
        Select the best new viewpoint from a list of candidates.
        
        This implementation selects the viewpoint with the highest number of potential
        matches to the projected 2D Gaussians.
        
        Args:
            candidates: List of 2D Gaussians for candidate viewpoints
            
        Returns:
            Index of the selected viewpoint
        """
        if not candidates:
            raise ValueError("No candidate viewpoints provided")
        
        # Project existing 3D Gaussians
        projected_2d = self.project_3d_gaussians()
        
        # Evaluate each candidate
        best_score = -1
        best_idx = 0
        
        for i, candidate in enumerate(candidates):
            # Initialize a temporary transport solver
            temp_solver = OptimalTransportSolver(
                gaussians1=projected_2d,
                gaussians2=candidate,
                k1=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
                k2=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
                epsilon=0.01,
                lambda_mean=0.0,
                lambda_cov=0.0,
                lambda_color=1.0,
                lambda_epipolar=1e-3,
                device=self.device
            )
            
            # Initialize F from identity rotation and zero translation
            with torch.no_grad():
                temp_solver.rvec.copy_(torch.zeros(3, device=self.device))
                temp_solver.tvec.copy_(torch.tensor([1.0, 0.0, 0.0], device=self.device))
                F = temp_solver.build_f_from_rt(temp_solver.rvec, temp_solver.tvec)
                temp_solver.f = F
            
            # Compute cost matrix
            cost_matrix = temp_solver.compute_cost_matrix_fundamental(temp_solver.f)
            
            # Compute transport plan
            transport = temp_solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, rho=0.5, max_iter=5000, tol=1e-5
            )
            
            # Score based on number of good matches
            transport_np = transport.detach().cpu().numpy()
            score = np.sum(transport_np > self.threshold_reprojection)
            
            # Update best if better
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx

    def integrate_new_view(self, new_image_2d_gaussians: TwoDGaussians, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate a new viewpoint into the 3D reconstruction.
        
        Args:
            new_image_2d_gaussians: 2D Gaussians from the new viewpoint
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (R_new, t_new) for the new camera viewpoint
        """
        # 1. Project existing 3D Gaussians to 2D
        projected_2d = self.project_3d_gaussians()
        
        # 2. Initialize transport solver
        self.initialize_transport_solver(
            projected_2d_gaussians=projected_2d,
            new_2d_gaussians=new_image_2d_gaussians
        )
        
        # 3. Optimize camera pose (R,t)
        self.transport_solver.optimize_with_RT(max_iter=max_iterations, tol=1e-6)
        
        # 4. Extract optimized R, t
        with torch.no_grad():
            R_est = self.transport_solver.rodrigues(self.rvec).detach().cpu().numpy()
            t_est = self.tvec.detach().cpu().numpy()
            
        # 5. Add to camera_params_list
        self.camera_params_list.append((R_est, t_est))
        
        # 6. Return the new camera parameters
        return R_est, t_est