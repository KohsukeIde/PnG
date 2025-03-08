# src/optimizer/bundle_adjuster.py

import numpy as np
import cv2
import os
from typing import List, Dict, Tuple, Optional
import scipy.sparse as sp
from scipy.optimize import least_squares

class BundleAdjuster:
    """
    Bundle Adjustment for 3D Gaussian reconstruction.
    Optimizes camera poses and 3D point positions simultaneously.
    """
    
    def __init__(
        self,
        points_3d: np.ndarray,
        camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
        match_points_2d: List[List[Tuple[int, np.ndarray]]],
        intrinsics_list: List[np.ndarray],
        image_names: Optional[List[str]] = None,
        use_robust_loss: bool = True,
        loss_scale: float = 1.0
    ):
        """
        Initialize the Bundle Adjuster.
        
        Args:
            points_3d: 3D point coordinates with shape [N, 3]
            camera_params_list: List of (R, t) camera parameters
            match_points_2d: For each camera, list of (point_idx, [x, y]) observations
            intrinsics_list: List of camera intrinsic matrices K
            image_names: Optional list of image names for COLMAP export
            use_robust_loss: Whether to use robust loss function
            loss_scale: Scale parameter for robust loss
        """
        self.points_3d = points_3d.copy()
        self.camera_params_list = [(R.copy(), t.copy()) for R, t in camera_params_list]
        self.match_points_2d = match_points_2d
        self.intrinsics_list = intrinsics_list
        self.image_names = image_names
        self.use_robust_loss = use_robust_loss
        self.loss_scale = loss_scale
        
        # Convert camera rotations to rodrigues vectors for optimization
        self.rvecs = []
        for R, _ in self.camera_params_list:
            rvec, _ = cv2.Rodrigues(R)
            self.rvecs.append(rvec.flatten())
        
        # Extract translation vectors
        self.tvecs = [t.flatten() for _, t in self.camera_params_list]
        
        # Number of cameras and points
        self.n_cameras = len(self.camera_params_list)
        self.n_points = len(self.points_3d)
        
        # Count total observations for statistics
        self.n_observations = sum(len(obs) for obs in self.match_points_2d)
        print(f"Bundle Adjustment: {self.n_observations} observations across {self.n_cameras} cameras for {self.n_points} points")
        
    def _pack_parameters(self) -> np.ndarray:
        """Pack parameters into a 1D array for optimization."""
        params = []
        
        # Camera parameters (rvec & tvec for each camera)
        for i in range(self.n_cameras):
            params.extend(self.rvecs[i])
            params.extend(self.tvecs[i])
            
        # 3D points
        for i in range(self.n_points):
            params.extend(self.points_3d[i])
            
        return np.array(params)
    
    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Unpack parameters from 1D array after optimization."""
        offset = 0
        
        # Camera parameters
        for i in range(self.n_cameras):
            self.rvecs[i] = params[offset:offset+3]
            offset += 3
            self.tvecs[i] = params[offset:offset+3]
            offset += 3
            
            # Update R from rvec
            R, _ = cv2.Rodrigues(self.rvecs[i])
            self.camera_params_list[i] = (R, self.tvecs[i])
            
        # 3D points
        for i in range(self.n_points):
            self.points_3d[i] = params[offset:offset+3]
            offset += 3
    
    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """Compute reprojection error residuals."""
        self._unpack_parameters(params)
        
        residuals = []
        
        # For each camera
        for cam_idx in range(self.n_cameras):
            R, t = self.camera_params_list[cam_idx]
            K = self.intrinsics_list[cam_idx]
            
            # For each 2D observation in this camera
            for point_idx, point_2d in self.match_points_2d[cam_idx]:
                # Project 3D point
                point_3d = self.points_3d[point_idx]
                point_cam = R @ point_3d + t
                
                # Skip points behind the camera
                if point_cam[2] <= 0:
                    residuals.extend([0.0, 0.0])  # Dummy residual to maintain structure
                    continue
                
                # Project to image coordinates
                point_img = K @ point_cam
                point_img = point_img[:2] / point_img[2]
                
                # Compute residual
                residual = point_img - point_2d
                residuals.extend(residual)
        
        return np.array(residuals)
    
    def optimize(self, n_iterations: int = 100, verbose: bool = True) -> Dict:
        """
        Run bundle adjustment optimization.
        
        Args:
            n_iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Dict with optimization results
        """
        # Check if we have enough data
        if self.n_points < 3 or self.n_cameras < 2 or self.n_observations < 10:
            print(f"Warning: Insufficient data for meaningful Bundle Adjustment.")
            print(f"  Points: {self.n_points}, Cameras: {self.n_cameras}, Observations: {self.n_observations}")
            return {
                'success': False,
                'message': 'Insufficient data',
                'optimized_cameras': self.camera_params_list,
                'optimized_points': self.points_3d
            }
        
        # Pack initial parameters
        params_initial = self._pack_parameters()
        
        if verbose:
            print(f"Starting Bundle Adjustment with {self.n_cameras} cameras and {self.n_points} points")
            print(f"Total parameters: {len(params_initial)}")
            
            # Compute initial error
            initial_residuals = self._compute_residuals(params_initial)
            initial_rmse = np.sqrt(np.mean(initial_residuals**2))
            print(f"Initial RMSE: {initial_rmse:.4f} pixels")
        
        # Define loss function
        loss_fn = 'cauchy' if self.use_robust_loss else None
        
        # Run optimization
        try:
            result = least_squares(
                self._compute_residuals,
                params_initial,
                method='trf',
                loss=loss_fn,
                f_scale=self.loss_scale,
                max_nfev=n_iterations,
                verbose=2 if verbose else 0
            )
            
            # Unpack final parameters
            self._unpack_parameters(result.x)
            
            # Final error
            final_residuals = self._compute_residuals(result.x)
            final_rmse = np.sqrt(np.mean(final_residuals**2))
            
            if verbose:
                print(f"Bundle Adjustment completed:")
                print(f"  Initial RMSE: {initial_rmse:.4f} pixels")
                print(f"  Final RMSE: {final_rmse:.4f} pixels")
                print(f"  Optimization success: {result.success}")
                
            return {
                'success': result.success,
                'initial_rmse': initial_rmse,
                'final_rmse': final_rmse,
                'n_iterations': result.nfev,
                'optimized_cameras': self.camera_params_list,
                'optimized_points': self.points_3d
            }
        except Exception as e:
            print(f"Bundle Adjustment failed with error: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'optimized_cameras': self.camera_params_list,
                'optimized_points': self.points_3d
            }