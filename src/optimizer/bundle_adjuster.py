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
    Uses a COLMAP-like approach with staged optimization and point-centered observations.
    """
    
    def __init__(
        self,
        points_3d: np.ndarray,
        camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
        match_points_2d: List[List[Tuple[int, np.ndarray]]],
        intrinsics_list: List[np.ndarray],
        image_names: Optional[List[str]] = None,
        use_robust_loss: bool = True,
        loss_scale: float = 2.0  # Default scale similar to COLMAP
    ):
        """
        Initialize the Bundle Adjuster with COLMAP-like structure.
        
        Args:
            points_3d: 3D point coordinates with shape [N, 3]
            camera_params_list: List of (R, t) camera parameters
            match_points_2d: For each camera, list of (point_idx, [x, y]) observations
            intrinsics_list: List of camera intrinsic matrices K
            image_names: Optional list of image names for COLMAP export
            use_robust_loss: Whether to use robust loss function
            loss_scale: Scale parameter for robust loss (default 2.0 like COLMAP)
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
        
        # Build 3D point-centered observation structure (COLMAP style)
        self.point3D_observations = {}  # point_id -> [(camera_id, point2D), ...]
        
        # Construct from existing correspondences
        for cam_idx, observations in enumerate(self.match_points_2d):
            for point_idx, point_2d in observations:
                if point_idx not in self.point3D_observations:
                    self.point3D_observations[point_idx] = []
                self.point3D_observations[point_idx].append((cam_idx, point_2d))
        
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
        """
        Compute reprojection error residuals using COLMAP-style point-centered approach.
        
        This implementation:
        1. Iterates over 3D points and their observations
        2. Only computes residuals for valid observations
        3. Properly handles points that might be behind cameras
        """
        if params is not None:
            self._unpack_parameters(params)
        
        residuals = []
        
        # Iterate over each 3D point and its observations (COLMAP style)
        for point_idx, observations in self.point3D_observations.items():
            if point_idx >= len(self.points_3d):
                continue
                
            point_3d = self.points_3d[point_idx]
            
            # For each camera observing this point
            for cam_idx, point_2d in observations:
                # Get camera parameters
                if cam_idx >= len(self.camera_params_list):
                    continue
                    
                R, t = self.camera_params_list[cam_idx]
                K = self.intrinsics_list[cam_idx]
                
                # Project 3D point to camera coordinates
                point_cam = R @ point_3d + t
                
                # Skip points behind the camera
                if point_cam[2] <= 0:
                    continue
                
                # Project to image coordinates
                point_img = K @ point_cam
                point_img = point_img[:2] / point_img[2]
                
                # Skip if NaN
                if np.isnan(point_2d).any() or np.isnan(point_img).any():
                    continue
                
                # Compute residual
                residual = point_img - point_2d
                residuals.extend(residual)
        
        # Return empty array if no valid residuals
        if not residuals:
            return np.array([])
            
        return np.array(residuals)
    
    def get_point_observation_counts(self) -> List[int]:
        """
        Count observations for each 3D point.
        
        Returns:
            List[int]: Number of observations per 3D point
        """
        point_observation_counts = [0] * len(self.points_3d)
        
        for observations in self.match_points_2d:
            for point_idx, _ in observations:
                if point_idx < len(self.points_3d):
                    point_observation_counts[point_idx] += 1
                    
        return point_observation_counts
        
    def get_valid_point_indices(self, min_observations: int = 2) -> List[int]:
        """
        Get indices of points with sufficient observations.
        
        Args:
            min_observations: Minimum number of observations required
            
        Returns:
            List[int]: Indices of valid points
        """
        observation_counts = self.get_point_observation_counts()
        return [i for i, count in enumerate(observation_counts) if count >= min_observations]
        
    def _optimize_standard(self, n_iterations: int, verbose: bool, loss_fn: str, 
                       method: str, initial_rmse: float = float('nan')) -> Dict:
        """Standard single-stage optimization."""
        # Pack parameters
        params_initial = self._pack_parameters()
        
        # Run optimization
        result = least_squares(
            self._compute_residuals,
            params_initial,
            method=method,
            loss=loss_fn,
            f_scale=self.loss_scale,
            max_nfev=n_iterations,
            verbose=2 if verbose else 0
        )
        
        # Unpack final parameters
        self._unpack_parameters(result.x)
        
        # Final error
        final_residuals = self._compute_residuals(result.x)
        if len(final_residuals) > 0:
            final_rmse = np.sqrt(np.mean(final_residuals**2))
        else:
            print("Warning: No valid residuals computed for final parameters")
            final_rmse = float('nan')
        
        if verbose:
            print(f"Bundle Adjustment completed:")
            print(f"  Initial RMSE: {initial_rmse:.4f} pixels")
            print(f"  Final RMSE: {final_rmse:.4f} pixels")
            print(f"  Optimization success: {result.success}")
            if not np.isnan(initial_rmse) and not np.isnan(final_rmse):
                improvement = (initial_rmse - final_rmse) / initial_rmse * 100
                print(f"  Improvement: {improvement:.2f}%")
            
        return {
            'success': result.success,
            'initial_rmse': initial_rmse,
            'final_rmse': final_rmse,
            'n_iterations': result.nfev,
            'optimized_cameras': self.camera_params_list,
            'optimized_points': self.points_3d
        }
            
    def _optimize_staged(self, n_iterations: int, verbose: bool, loss_fn: str, method: str) -> Dict:
        """Staged optimization like COLMAP."""
        # Backup original parameters in case optimization fails
        original_points = self.points_3d.copy()
        original_rvecs = [rvec.copy() for rvec in self.rvecs]
        original_tvecs = [tvec.copy() for tvec in self.tvecs]
        
        # Stage 1: Optimize only 3D points with fixed cameras
        if verbose:
            print("Stage 1: Optimizing 3D points with fixed cameras...")
        
        # For Stage 1, we'll only optimize points that have observations in the point3D_observations dict
        visible_point_indices = list(self.point3D_observations.keys())
        
        # Ensure indices are valid by filtering out any that exceed the array size
        valid_indices = [idx for idx in visible_point_indices if idx < len(self.points_3d)]
        
        if verbose:
            print(f"Optimizing {len(valid_indices)} points that have observations (out of {len(self.points_3d)} total)")
        
        # Create a mapping from original indices to parameter array indices
        index_mapping = {idx: i for i, idx in enumerate(valid_indices)}
        
        # Extract only the points that have observations
        selected_points = np.array([self.points_3d[idx] for idx in valid_indices])
        
        # Flatten for optimization
        params_points = selected_points.flatten()
        
        # Create a function that only updates visible points
        def compute_residuals_fixed_cameras(points_params):
            try:
                # Reshape flat array back to points
                reshaped_points = points_params.reshape(-1, 3)
                
                # Update only the points in our mapping
                for i, point_idx in enumerate(valid_indices):
                    if i < len(reshaped_points):
                        self.points_3d[point_idx] = reshaped_points[i]
                
                # Compute residuals without changing camera parameters
                return self._compute_residuals(None)
            except Exception as e:
                print(f"Error in compute_residuals_fixed_cameras: {e}")
                # Return a small dummy residual to prevent optimization failure
                return np.array([1e-6])
        
        # Optimize points only - we need at least 10 iterations for this to be meaningful
        min_iterations = max(10, n_iterations // 3)
        result_points = least_squares(
            compute_residuals_fixed_cameras,
            params_points,
            method=method,
            loss=loss_fn,
            f_scale=self.loss_scale,
            max_nfev=min_iterations,
            verbose=1 if verbose else 0
        )
        
        if verbose:
            if result_points.success:
                points_rmse = np.sqrt(np.mean(result_points.fun**2))
                print(f"  Stage 1 RMSE: {points_rmse:.4f} pixels")
            else:
                print("  Stage 1 optimization failed.")
        
        # Stage 2: Optimize only cameras with fixed 3D points
        if verbose:
            print("Stage 2: Optimizing cameras with fixed 3D points...")
            
        # Find cameras that have observations
        cameras_with_obs = set()
        for observations in self.point3D_observations.values():
            for cam_idx, _ in observations:
                if cam_idx < self.n_cameras:  # Ensure valid index
                    cameras_with_obs.add(cam_idx)
                    
        # Convert to sorted list
        valid_cameras = sorted(list(cameras_with_obs))
        
        if verbose:
            print(f"Optimizing {len(valid_cameras)} cameras that have observations (out of {self.n_cameras} total)")
            
        # Extract only the camera parameters that have observations
        rvecs_to_optimize = [self.rvecs[idx] for idx in valid_cameras]
        tvecs_to_optimize = [self.tvecs[idx] for idx in valid_cameras]
        
        # Pack all camera parameters into a single array
        # First all rotation vectors, then all translation vectors
        camera_params = np.concatenate(
            [np.concatenate(rvecs_to_optimize), np.concatenate(tvecs_to_optimize)]
        )
        
        # Create a function that only updates cameras
        def compute_residuals_fixed_points(camera_params):
            try:
                # Get number of cameras to update
                num_cameras = len(valid_cameras)
                
                # First update all rotation vectors
                for i, cam_idx in enumerate(valid_cameras):
                    # Extract rvec (first part of parameters)
                    start_idx = i * 3
                    self.rvecs[cam_idx] = camera_params[start_idx:start_idx+3]
                
                # Then update all translation vectors
                for i, cam_idx in enumerate(valid_cameras):
                    # Extract tvec (second part of parameters, after all rvecs)
                    start_idx = num_cameras * 3 + i * 3
                    self.tvecs[cam_idx] = camera_params[start_idx:start_idx+3]
                    
                    # Update camera_params_list
                    R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                    self.camera_params_list[cam_idx] = (R, self.tvecs[cam_idx])
                
                # Compute residuals without changing 3D points
                return self._compute_residuals(None)
            except Exception as e:
                print(f"Error in compute_residuals_fixed_points: {e}")
                # Return a small dummy residual to prevent optimization failure
                return np.array([1e-6])
        
        # Optimize cameras only - ensure minimum iterations
        min_iterations = max(10, n_iterations // 3)
        result_cameras = least_squares(
            compute_residuals_fixed_points,
            camera_params,
            method=method,
            loss=loss_fn,
            f_scale=self.loss_scale,
            max_nfev=min_iterations,
            verbose=1 if verbose else 0
        )
        
        if verbose:
            if result_cameras.success:
                cameras_rmse = np.sqrt(np.mean(result_cameras.fun**2))
                print(f"  Stage 2 RMSE: {cameras_rmse:.4f} pixels")
            else:
                print("  Stage 2 optimization failed.")
        
        # Stage 3: Optimize all parameters together
        if verbose:
            print("Stage 3: Optimizing all parameters...")
        
        try:    
            # Pack camera parameters - all rvecs first, then all tvecs
            rvec_params = []
            tvec_params = []
            for i in valid_cameras:
                rvec_params.extend(self.rvecs[i])
                tvec_params.extend(self.tvecs[i])
                
            # Combine camera params - all rvecs followed by all tvecs
            camera_params = np.concatenate([np.array(rvec_params), np.array(tvec_params)])
                
            # Pack 3D points for points with observations
            point_params = []
            for i in valid_indices:
                point_params.extend(self.points_3d[i])
                
            # Combine all parameters
            params_all = np.concatenate([camera_params, np.array(point_params)])
            
            if verbose:
                print(f"Stage 3: Optimizing {len(valid_cameras)} cameras and {len(valid_indices)} points")
                print(f"Total parameters: {len(params_all)}")
                print(f"Camera params: {len(camera_params)}, Point params: {len(point_params)}")
            
            # Create a custom residual function to handle our reduced parameter set
            def compute_residuals_all(all_params):
                try:
                    # Split parameters into camera and point parts
                    camera_count = len(valid_cameras)
                    camera_param_count = camera_count * 6  # 6 params per camera
                    
                    camera_params = all_params[:camera_param_count]
                    point_params = all_params[camera_param_count:]
                    
                    # First get all the rvecs
                    rvec_start = 0
                    for i, cam_idx in enumerate(valid_cameras):
                        self.rvecs[cam_idx] = camera_params[rvec_start + i*3:rvec_start + (i+1)*3]
                    
                    # Then get all the tvecs
                    tvec_start = camera_count * 3  # Skip all rvecs
                    for i, cam_idx in enumerate(valid_cameras):
                        self.tvecs[cam_idx] = camera_params[tvec_start + i*3:tvec_start + (i+1)*3]
                        
                        # Update camera_params_list
                        R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                        self.camera_params_list[cam_idx] = (R, self.tvecs[cam_idx])
                    
                    # Check the number of points to update matches the expected size
                    point_count = len(point_params) // 3
                    if point_count != len(valid_indices):
                        # If there's a mismatch, ensure we only update as many as are valid
                        point_count = min(point_count, len(valid_indices))
                        
                    # Reshape points and update them
                    points_reshaped = point_params.reshape(-1, 3)
                    
                    for i in range(point_count):
                        if i < len(valid_indices):
                            point_idx = valid_indices[i]
                            self.points_3d[point_idx] = points_reshaped[i]
                    
                    # Compute residuals
                    return self._compute_residuals(None)
                except Exception as e:
                    print(f"Error in compute_residuals_all: {e}")
                    # Return a small dummy residual to prevent optimization failure
                    return np.array([1e-6])
            
            # Optimize everything - ensure minimum iterations
            min_iterations = max(20, n_iterations // 3)
            result_all = least_squares(
                compute_residuals_all,
                params_all,
                method=method,
                loss=loss_fn,
                f_scale=self.loss_scale,
                max_nfev=min_iterations,
                verbose=2 if verbose else 0
            )
            
            # Compute error metrics
            final_residuals = self._compute_residuals(None)  # Use current state
            if len(final_residuals) > 0:
                final_rmse = np.sqrt(np.mean(final_residuals**2))
            else:
                final_rmse = float('nan')
                
            # Consider optimization successful if it completes without error
            optimization_success = result_all.success
            
        except Exception as e:
            print(f"Stage 3 optimization failed with error: {e}")
            optimization_success = False
            final_rmse = float('nan')
        
        # Check if the optimization failed
        if not optimization_success:
            print("Stage 3 optimization failed. Reverting to original parameters.")
            # Restore original values
            self.points_3d = original_points.copy()
            self.rvecs = [rvec.copy() for rvec in original_rvecs]
            self.tvecs = [tvec.copy() for tvec in original_tvecs]
            
            # Update camera_params_list from rvecs and tvecs
            for i in range(self.n_cameras):
                try:
                    R, _ = cv2.Rodrigues(self.rvecs[i])
                    self.camera_params_list[i] = (R, self.tvecs[i])
                except Exception as e:
                    print(f"Error restoring camera {i}: {e}")
        
        if verbose:
            print(f"Staged Bundle Adjustment completed:")
            print(f"  Final RMSE: {final_rmse:.4f} pixels")
            print(f"  Success: {optimization_success}")
        
        # Calculate total iterations (handle the case where result_all might not exist)
        total_iterations = result_points.nfev + result_cameras.nfev
        if 'result_all' in locals() and hasattr(result_all, 'nfev'):
            total_iterations += result_all.nfev
        
        return {
            'success': optimization_success,
            'final_rmse': final_rmse,
            'n_iterations': total_iterations,
            'optimized_cameras': self.camera_params_list,
            'optimized_points': self.points_3d
        }
    
    def optimize(self, n_iterations: int = 100, verbose: bool = True, use_staged: bool = True) -> Dict:
        """
        Run bundle adjustment optimization with COLMAP-like approach.
        
        Args:
            n_iterations: Maximum number of iterations
            verbose: Whether to print progress
            use_staged: Whether to use staged optimization (COLMAP style)
            
        Returns:
            Dict with optimization results
        """
        # Analyze the observation model to count actual valid observations
        total_valid_observations = 0
        points_with_observations = set()
        cameras_with_observations = set()
        
        # Use point-centered observation structure for analysis
        for point_idx, observations in self.point3D_observations.items():
            if point_idx >= len(self.points_3d):
                continue
                
            valid_point_observations = 0
            for cam_idx, point_2d in observations:
                if cam_idx >= len(self.camera_params_list):
                    continue
                    
                # Check if point is in front of camera and not NaN
                R, t = self.camera_params_list[cam_idx]
                point_3d = self.points_3d[point_idx]
                point_cam = R @ point_3d + t
                
                if point_cam[2] > 0 and not np.isnan(point_2d).any():
                    total_valid_observations += 1
                    valid_point_observations += 1
                    cameras_with_observations.add(cam_idx)
            
            # Only count points with at least 2 observations (COLMAP requirement)
            if valid_point_observations >= 2:
                points_with_observations.add(point_idx)
        
        if verbose:
            print(f"Valid observations: {total_valid_observations}")
            print(f"Points with 2+ observations: {len(points_with_observations)}")
            print(f"Cameras with observations: {len(cameras_with_observations)}")
        
        # Check if we have enough data for a meaningful BA
        # Each point must be observed by at least 2 cameras to be constrained
        # We need at least 3 points with observations for a well-posed problem
        if (len(points_with_observations) < 3 or 
            len(cameras_with_observations) < 2 or 
            total_valid_observations < 10):
            
            print(f"Warning: Insufficient data for meaningful Bundle Adjustment:")
            print(f"  Points with observations: {len(points_with_observations)} (need at least 3)")
            print(f"  Cameras with observations: {len(cameras_with_observations)} (need at least 2)")
            print(f"  Valid observations: {total_valid_observations} (need at least 10)")
            
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
            if len(initial_residuals) > 0:
                initial_rmse = np.sqrt(np.mean(initial_residuals**2))
                print(f"Initial RMSE: {initial_rmse:.4f} pixels")
                
                # Adaptively adjust scale if RMSE is too large (COLMAP-like)
                if self.use_robust_loss and initial_rmse > 50:
                    adaptive_scale = initial_rmse / 25  # Example: RMSE / 25 as scale
                    original_scale = self.loss_scale
                    self.loss_scale = max(2.0, min(adaptive_scale, 10.0))  # Limit range
                    print(f"Adjusting robust loss scale from {original_scale:.2f} to {self.loss_scale:.2f} based on RMSE")
            else:
                print("Warning: No valid residuals computed for initial parameters")
                initial_rmse = float('nan')
        
        # Define loss function - Cauchy loss function is robust against outliers
        loss_fn = 'cauchy' if self.use_robust_loss else None
        
        # Method selection - 'lm' doesn't support robust loss
        if self.use_robust_loss:
            method = 'trf'  # Trust Region Reflective
        else:
            method = 'lm' if self.n_points < 500 else 'trf'
        
        # Run optimization - choose between standard and staged approaches
        try:
            if use_staged and self.n_points > 20 and self.n_cameras > 2:
                return self._optimize_staged(n_iterations, verbose, loss_fn, method)
            else:
                return self._optimize_standard(n_iterations, verbose, loss_fn, method, initial_rmse)
        except Exception as e:
            print(f"Bundle Adjustment failed with error: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'optimized_cameras': self.camera_params_list,
                'optimized_points': self.points_3d
            }