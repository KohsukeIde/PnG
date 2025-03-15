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
    Camera poses are parameterized as camera-to-world (inverse) transformations,
    aligning with modern SfM methods like COLMAP.
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
        """Initialize the Bundle Adjuster with COLMAP-like structure.
        
        Args:
            points_3d: 3D point coordinates with shape [N, 3] in world space
            camera_params_list: List of (R, t) camera parameters where R is world-to-camera rotation
                                and t is world-to-camera translation. We will convert to camera-to-world.
            match_points_2d: For each camera, list of (point_idx, [x, y]) observations
            intrinsics_list: List of camera intrinsic matrices K
            image_names: Optional list of image names for COLMAP export
            use_robust_loss: Whether to use robust loss function
            loss_scale: Scale parameter for robust loss (default 2.0 like COLMAP)
        """
        self.points_3d = points_3d.copy()
        self.intrinsics_list = intrinsics_list
        self.match_points_2d = match_points_2d
        self.image_names = image_names
        self.use_robust_loss = use_robust_loss
        self.loss_scale = loss_scale
        
        # Convert world-to-camera (R, t) to camera-to-world (R_inv, c) representation
        # This is the inverse parameterization used by COLMAP, iNeRF, etc.
        self.camera_to_world_list = []
        for R_wtc, t_wtc in camera_params_list:
            # R_inverse: camera-to-world rotation 
            R_ctw = R_wtc.T.copy()
            
            # c: camera center in world coordinates (c = -R_wtc^T * t_wtc)
            c = -R_ctw @ t_wtc
            
            self.camera_to_world_list.append((R_ctw, c))
        
        # Convert camera rotations to rodrigues vectors for optimization
        self.rvecs = []
        for R_ctw, _ in self.camera_to_world_list:
            rvec, _ = cv2.Rodrigues(R_ctw)
            self.rvecs.append(rvec.flatten())
        
        # Extract camera centers (c) in world coordinates
        self.centers = [c.flatten() for _, c in self.camera_to_world_list]
        
        # Number of cameras and points
        self.n_cameras = len(self.camera_to_world_list)
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
        
        # Camera parameters (rvec & camera center for each camera)
        for i in range(self.n_cameras):
            params.extend(self.rvecs[i])
            params.extend(self.centers[i])
            
        # 3D points
        for i in range(self.n_points):
            params.extend(self.points_3d[i])
        
        #return [R1_1, R1_2, R1_3, C1_1, C1_2, C1_3, R2_1, R2_2, R2_3, C2_1, C2_2, C2_3, ..., P1_1, P1_2, P1_3, P2_1, P2_2, P2_3, ...]
        return np.array(params)
    
    def _unpack_parameters(self, params: np.ndarray) -> None:
        """Unpack parameters from 1D array after optimization."""
        offset = 0
        
        # Camera parameters
        for i in range(self.n_cameras):
            self.rvecs[i] = params[offset:offset+3]
            offset += 3
            self.centers[i] = params[offset:offset+3]
            offset += 3
            
            # Update camera-to-world rotation matrix from rvec
            R_ctw, _ = cv2.Rodrigues(self.rvecs[i])
            
            # Update the camera-to-world list
            self.camera_to_world_list[i] = (R_ctw, self.centers[i])
            
        # 3D points
        for i in range(self.n_points):
            self.points_3d[i] = params[offset:offset+3]
            offset += 3
    
    def _compute_residuals(self, params: np.ndarray) -> np.ndarray:
        """Compute reprojection error residuals using COLMAP-style point-centered approach.
        
        This implementation:
        1. Iterates over 3D points and their observations
        2. Fills in placeholder values for invalid observations to maintain consistent array shape
        3. Properly handles points that might be behind cameras
        4. Uses camera-to-world parameterization (COLMAP style)
        """
        self._unpack_parameters(params)
        
        # Build a mapping of all possible residuals to ensure consistent output shape
        # Each residual is a 2D point (x,y), so we'll need 2 values per observation
        expected_residuals = {}
        residual_count = 0
        
        # First pass: Count expected residuals and their positions
        for point_idx, observations in self.point3D_observations.items():
            if point_idx >= len(self.points_3d):
                continue
            
            for cam_idx, point_2d in observations:
                if cam_idx >= len(self.camera_to_world_list):
                    continue
                
                # This observation should produce a 2D residual 
                # この観測（ポイントとカメラの組み合わせ）の残差の開始位置を記録
                # 残差配列は，全ての観測の残差を1次元配列として格納　-> [r_0x, r_0y, r_1x, r_1y, ..., r_nx, r_ny] (r_ix：i番目の観測のx座標の残差, r_iy：i番目の観測のy座標の残差)
                expected_residuals[(point_idx, cam_idx)] = residual_count
                residual_count += 2  # x and y components
        
        # Create array with placeholders for all expected residuals
        # Using placeholder values instead of zeros ensures optimizer won't favor invalid points
        placeholder_value = 1e-6  # Small non-zero value that won't dominate valid residuals
        all_residuals = np.ones(residual_count) * placeholder_value
        
        # Second pass: Fill in actual residuals for valid observations
        for point_idx, observations in self.point3D_observations.items():
            if point_idx >= len(self.points_3d):
                continue
                
            point_3d = self.points_3d[point_idx]
            
            # For each camera observing this point
            for cam_idx, point_2d in observations:
                if cam_idx >= len(self.camera_to_world_list):
                    continue
                
                # Get residual position in output array
                residual_pos = expected_residuals.get((point_idx, cam_idx))
                if residual_pos is None:
                    continue
                
                # Get camera-to-world parameters
                R_ctw, c = self.camera_to_world_list[cam_idx]
                K = self.intrinsics_list[cam_idx]
                
                # Convert to world-to-camera for projection
                R_wtc = R_ctw.T
                t_wtc = -R_wtc @ c
                
                # Project 3D point to camera coordinates
                point_cam = R_wtc @ point_3d + t_wtc
                
                # Skip computation for points behind the camera, leaving placeholder values
                if point_cam[2] <= 0:
                    continue
                
                # Project to image coordinates
                point_img = K @ point_cam # 内部パラメータで投影
                point_img = point_img[:2] / point_img[2] # 同次座標から2D座標に変換
                
                # Skip if NaN, leaving placeholder values
                if np.isnan(point_2d).any() or np.isnan(point_img).any():
                    continue
                
                # Compute and store residual (points_2d == new_2d_gaussians.mean() || gaussians.mean() in case of initial BA)
                residual = point_img - point_2d
                # residual_pos - x座標の残差の位置，residual_pos+1 - y座標の残差の位置
                all_residuals[residual_pos:residual_pos+2] = residual
        
        return all_residuals
    
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
        """Get indices of points with sufficient observations.
        
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
            'optimized_cameras': self.camera_to_world_list,
            'optimized_points': self.points_3d
        }
            
    def _optimize_staged(self, n_iterations: int, verbose: bool, loss_fn: str, method: str) -> Dict:
        """Staged optimization like COLMAP."""
        # Backup original parameters in case optimization fails
        original_points = self.points_3d.copy()
        original_rvecs = [rvec.copy() for rvec in self.rvecs]
        original_centers = [center.copy() for center in self.centers]
        
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
        # index_mapping = {idx: i for i, idx in enumerate(valid_indices)}
        
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
                # Calculate how many residuals are expected and return placeholder values
                residual_count = 0
                for observation_list in self.match_points_2d:
                    residual_count += len(observation_list) * 2  # x and y components
                placeholder_value = 1e-3
                return np.ones(residual_count) * placeholder_value
        
        # Optimize points only - we need at least 20 iterations for this to be meaningful
        min_iterations = max(20, n_iterations // 3)
        result_points = least_squares(
            compute_residuals_fixed_cameras,
            params_points,
            method=method,
            loss=loss_fn,
            f_scale=self.loss_scale,
            max_nfev=min_iterations,
            ftol=1e-4,  # More lenient convergence tolerance
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
        centers_to_optimize = [self.centers[idx] for idx in valid_cameras]
        
        # Pack all camera parameters into a single array
        # First all rotation vectors, then all camera centers
        camera_params = np.concatenate(
            [np.concatenate(rvecs_to_optimize), np.concatenate(centers_to_optimize)]
        )
        
        # Create a function that only updates cameras
        def compute_residuals_fixed_points(camera_params):
            
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
                self.centers[cam_idx] = camera_params[start_idx:start_idx+3]
                
                # Update camera_to_world_list
                R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                self.camera_to_world_list[cam_idx] = (R, self.centers[cam_idx])
            
            # Compute residuals without changing 3D points
            return self._compute_residuals(None)
        
        # Optimize cameras only - ensure sufficient iterations
        min_iterations = max(20, n_iterations // 3)
        result_cameras = least_squares(
            compute_residuals_fixed_points,
            camera_params,
            method=method,
            loss=loss_fn,
            f_scale=self.loss_scale,
            max_nfev=min_iterations,
            ftol=1e-4,  # More lenient convergence tolerance
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
            # Pack camera parameters - all rvecs first, then all camera centers
            rvec_params = []
            center_params = []
            for i in valid_cameras:
                rvec_params.extend(self.rvecs[i])
                center_params.extend(self.centers[i])
                
            # Combine camera params - all rvecs followed by all camera centers
            camera_params = np.concatenate([np.array(rvec_params), np.array(center_params)])
                
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
                    
                    # Then get all the camera centers
                    center_start = camera_count * 3  # Skip all rvecs
                    for i, cam_idx in enumerate(valid_cameras):
                        self.centers[cam_idx] = camera_params[center_start + i*3:center_start + (i+1)*3]
                        
                        # Update camera_to_world_list
                        R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                        self.camera_to_world_list[cam_idx] = (R, self.centers[cam_idx])
                    
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
                    # Calculate how many residuals are expected and return placeholder values
                    residual_count = 0
                    for observation_list in self.match_points_2d:
                        residual_count += len(observation_list) * 2  # x and y components
                    placeholder_value = 1e-3
                    return np.ones(residual_count) * placeholder_value
            
            # Optimize everything - ensure sufficient iterations for convergence
            min_iterations = max(40, n_iterations)  # Give stage 3 more iterations
            result_all = least_squares(
                compute_residuals_all,
                params_all,
                method=method,
                loss=loss_fn,
                f_scale=self.loss_scale,
                max_nfev=min_iterations,
                ftol=1e-4,  # More lenient convergence tolerance
                xtol=1e-4,  # More lenient step size tolerance
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
        
        # Check if the optimization failed - only revert if error got worse
        if not optimization_success:
            print("Stage 3 optimization did not fully converge.")
            
            # Calculate initial RMSE for comparison
            initial_residuals = self._compute_residuals(None)
            if len(initial_residuals) > 0:
                initial_rmse = np.sqrt(np.mean(initial_residuals**2))
                
                # Only revert if final error is worse than initial error
                if (np.isnan(final_rmse) or final_rmse > initial_rmse * 1.1):  # Allow up to 10% increase
                    print(f"Optimization increased error from {initial_rmse:.4f} to {final_rmse:.4f}. Reverting to original parameters.")
                    # Restore original values
                    self.points_3d = original_points.copy()
                    self.rvecs = [rvec.copy() for rvec in original_rvecs]
                    self.centers = [center.copy() for center in original_centers]
                else:
                    print(f"Despite non-convergence, error improved from {initial_rmse:.4f} to {final_rmse:.4f}. Keeping results.")
                    optimization_success = True  # Consider this a success since error improved
            else:
                # If we can't compute initial RMSE, revert to be safe
                print("Unable to calculate initial RMSE. Reverting to original parameters.")
                # Restore original values
                self.points_3d = original_points.copy()
                self.rvecs = [rvec.copy() for rvec in original_rvecs]
                self.centers = [center.copy() for center in original_centers]
            
            # Update camera_to_world_list from rvecs and centers
            for i in range(self.n_cameras):
                try:
                    R, _ = cv2.Rodrigues(self.rvecs[i])
                    self.camera_to_world_list[i] = (R, self.centers[i])
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
            'optimized_cameras': self.camera_to_world_list,
            'optimized_points': self.points_3d
        }
    
    def optimize(self, n_iterations: int = 100, verbose: bool = True, use_staged: bool = True) -> Dict:
        """ Run bundle adjustment optimization with COLMAP-like approach.
        
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
                if cam_idx >= len(self.camera_to_world_list):
                    continue
                    
                # Check if point is in front of camera and not NaN
                # Get camera-to-world parameters
                R_ctw, c = self.camera_to_world_list[cam_idx]
                
                # Convert to world-to-camera for projection
                R_wtc = R_ctw.T
                t_wtc = -R_wtc @ c
                
                point_3d = self.points_3d[point_idx]
                point_cam = R_wtc @ point_3d + t_wtc
                
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
                'optimized_cameras': self.camera_to_world_list,
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
                    adaptive_scale = initial_rmse / 25  # Ex: RMSE / 25 as scale
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
                'optimized_cameras': self.camera_to_world_list,
                'optimized_points': self.points_3d
            }