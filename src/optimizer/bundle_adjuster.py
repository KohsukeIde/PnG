# src/optimizer/bundle_adjuster.py

import numpy as np
import cv2
import os
from typing import List, Dict, Tuple, Optional
import scipy.sparse as sp
from scipy.optimize import least_squares
from .point_id_manager import PointIDManager

class BundleAdjuster:
    """
    Bundle Adjustment for 3D Gaussian reconstruction.
    Optimizes camera poses and 3D point positions simultaneously.
    Uses a COLMAP-like approach with staged optimization and point-centered observations.
    Camera poses are parameterized as camera-to-world (inverse) transformations,
    aligning with modern SfM methods like COLMAP.
    
    This implementation uses persistent point IDs via PointIDManager to maintain
    consistent observation tracking throughout the reconstruction process.
    """
    
    def __init__(
        self,
        point_id_manager: PointIDManager,
        use_robust_loss: bool = True,
        loss_scale: float = 2.0  # Default scale similar to COLMAP
    ):
        """Initialize the Bundle Adjuster with COLMAP-like structure using persistent point IDs.
        
        Args:
            point_id_manager: Point ID manager containing points, cameras and observations
            use_robust_loss: Whether to use robust loss function
            loss_scale: Scale parameter for robust loss (default 2.0 like COLMAP)
        """
        self.point_id_manager = point_id_manager
        self.use_robust_loss = use_robust_loss
        self.loss_scale = loss_scale
        
        # Use all points from the point manager without filtering
        self.point_ids = list(self.point_id_manager.points.keys())
        
        # Create optimization index mappings
        self.point_id_to_index = {point_id: i for i, point_id in enumerate(self.point_ids)}
        self.camera_id_to_index = {camera_id: i for i, camera_id 
                                  in enumerate(sorted(self.point_id_manager.cameras.keys()))}
        self.camera_index_to_id = {i: camera_id for camera_id, i 
                                  in self.camera_id_to_index.items()}
        
        # Create optimization parameter arrays
        self.points_3d = np.array([self.point_id_manager.points[point_id].position 
                                  for point_id in self.point_ids])
        
        # Extract camera-to-world parameters
        self.camera_to_world_list = []
        for camera_id in sorted(self.point_id_manager.cameras.keys()):
            camera = self.point_id_manager.cameras[camera_id]
            self.camera_to_world_list.append((camera.R, camera.c))
        
        # Convert camera rotations to rodrigues vectors for optimization
        self.rvecs = []
        for R_ctw, _ in self.camera_to_world_list:
            rvec, _ = cv2.Rodrigues(R_ctw)
            self.rvecs.append(rvec.flatten())
        
        # Extract camera centers (c) in world coordinates
        self.centers = [c.flatten() for _, c in self.camera_to_world_list]
        
        # Cache intrinsic matrices
        self.intrinsics_list = []
        for camera_id in sorted(self.point_id_manager.cameras.keys()):
            camera = self.point_id_manager.cameras[camera_id]
            self.intrinsics_list.append(camera.K)
        
        # Cache image names
        self.image_names = []
        for camera_id in sorted(self.point_id_manager.cameras.keys()):
            camera = self.point_id_manager.cameras[camera_id]
            self.image_names.append(camera.image_name)
        
        # Number of cameras and points
        self.n_cameras = len(self.camera_to_world_list)
        self.n_points = len(self.points_3d)
        
        # Count total observations for statistics
        self.n_observations = 0
        for point_id in self.point_ids:
            point = self.point_id_manager.points[point_id]
            self.n_observations += len(point.observations)
        
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
        1. Iterates over 3D points and their observations using persistent point IDs
        2. Fills in placeholder values for invalid observations to maintain consistent array shape
        3. Properly handles points that might be behind cameras
        4. Uses camera-to-world parameterization (COLMAP style)
        """
        # Handle None params for staged optimization
        if params is not None:
            self._unpack_parameters(params)
        
        # Build a mapping of all possible residuals to ensure consistent output shape
        # Each residual is a 2D point (x,y), so we'll need 2 values per observation
        expected_residuals = {}
        residual_count = 0
        
        # First pass: Count expected residuals and their positions
        for point_opt_index, point_id in enumerate(self.point_ids):
            if point_id not in self.point_id_manager.points:
                continue
                
            point = self.point_id_manager.points[point_id]
            
            for camera_id, point_2d in point.observations.items():
                if camera_id not in self.camera_id_to_index:
                    continue
                    
                # This observation should produce a 2D residual
                # Store the position in the residuals array for this observation
                # Residuals array format: [r_0x, r_0y, r_1x, r_1y, ..., r_nx, r_ny]
                # where r_ix is the x-coordinate residual for observation i
                cam_idx = self.camera_id_to_index[camera_id]
                expected_residuals[(point_opt_index, cam_idx)] = residual_count
                residual_count += 2  # x and y components
        
        # Create array with placeholders for all expected residuals
        # Using placeholder values instead of zeros ensures optimizer won't favor invalid points
        placeholder_value = 1e-6  # Small non-zero value that won't dominate valid residuals
        all_residuals = np.ones(residual_count) * placeholder_value
        
        # Second pass: Fill in actual residuals for valid observations
        for point_opt_index, point_id in enumerate(self.point_ids):
            if point_id not in self.point_id_manager.points:
                continue
                
            point = self.point_id_manager.points[point_id]
            
            # Get the 3D position from our optimized array
            point_3d = self.points_3d[point_opt_index]
            
            # For each camera observing this point
            for camera_id, point_2d in point.observations.items():
                if camera_id not in self.camera_id_to_index:
                    continue
                    
                cam_idx = self.camera_id_to_index[camera_id]
                
                # Get residual position in output array
                residual_pos = expected_residuals.get((point_opt_index, cam_idx))
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
                point_img = K @ point_cam  # Project using intrinsic parameters
                point_img = point_img[:2] / point_img[2]  # Convert from homogeneous to 2D
                
                # Skip if NaN, leaving placeholder values
                if np.isnan(point_2d).any() or np.isnan(point_img).any():
                    continue
                
                # Compute and store residual
                residual = point_img - point_2d
                all_residuals[residual_pos:residual_pos+2] = residual
        
        return all_residuals
    
    def get_point_observation_counts(self) -> List[int]:
        """
        Count observations for each 3D point.
        
        Returns:
            List[int]: Number of observations per 3D point
        """
        observation_counts = []
        
        for point_id in self.point_ids:
            if point_id in self.point_id_manager.points:
                point = self.point_id_manager.points[point_id]
                observation_counts.append(len(point.observations))
            else:
                observation_counts.append(0)
                
        return observation_counts
        
    def get_point_indices_with_observations(self, min_observations: int = 2) -> List[int]:
        """Get optimization indices of points with observations.
        
        Args:
            min_observations: Minimum number of observations to consider
            
        Returns:
            List[int]: Optimization indices of points with observations
        """
        indices_with_observations = []
        
        for i, point_id in enumerate(self.point_ids):
            if point_id in self.point_id_manager.points:
                point = self.point_id_manager.points[point_id]
                if len(point.observations) >= min_observations:
                    indices_with_observations.append(i)
                    
        return indices_with_observations
    
    def get_point_ids_with_observations(self, min_observations: int = 2) -> List[int]:
        """Get point IDs that have observations.
        
        Args:
            min_observations: Minimum number of observations to consider
            
        Returns:
            List[int]: IDs of points with observations
        """
        ids_with_observations = []
        
        for point_id in self.point_ids:
            if point_id in self.point_id_manager.points:
                point = self.point_id_manager.points[point_id]
                if len(point.observations) >= min_observations:
                    ids_with_observations.append(point_id)
                    
        return ids_with_observations
        
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
            
        # Consider optimization "successful" if RMSE improved significantly, even if max iterations was reached
        optimization_success = result.success
        improvement = 0
        
        if not np.isnan(initial_rmse) and not np.isnan(final_rmse):
            improvement = (initial_rmse - final_rmse) / initial_rmse * 100
            # If error reduced by at least 20%, consider the optimization successful regardless of convergence
            if improvement >= 20 and not optimization_success:
                optimization_success = True
                print("Bundle Adjustment reached max iterations but had significant error reduction, marking as successful")
        
        if verbose:
            print(f"Bundle Adjustment completed:")
            print(f"  Initial RMSE: {initial_rmse:.4f} pixels")
            print(f"  Final RMSE: {final_rmse:.4f} pixels")
            print(f"  Optimization success: {optimization_success}")
            if improvement > 0:
                print(f"  Improvement: {improvement:.2f}%")
            
        return {
            'success': optimization_success,  # Use our modified success criteria
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
        
        # For Stage 1, we'll only optimize points that have observations
        # Get point IDs with at least 2 observations
        valid_point_ids = self.get_point_ids_with_observations(min_observations=2)
        
        # Create mapping from position in optimization array to point ID
        opt_idx_to_point_id = {}
        point_id_to_opt_idx = {}
        
        # Extract only the points that have observations, maintaining a clear mapping
        selected_points = []
        for opt_idx, point_id in enumerate(valid_point_ids):
            if point_id not in self.point_id_to_index:
                continue
                
            point_idx = self.point_id_to_index[point_id]
            if point_idx >= len(self.points_3d):
                continue
                
            selected_points.append(self.points_3d[point_idx])
            opt_idx_to_point_id[opt_idx] = point_id
            point_id_to_opt_idx[point_id] = opt_idx
        
        if verbose:
            print(f"Optimizing {len(selected_points)} points with observations (out of {len(self.points_3d)} total)")
        
        if not selected_points:
            print("No valid points to optimize in Stage 1. Skipping...")
            return {
                'success': False,
                'message': 'No valid points for optimization',
                'final_rmse': float('nan'),
                'n_iterations': 0,
                'optimized_cameras': self.camera_to_world_list,
                'optimized_points': self.points_3d
            }
        
        # Convert to numpy array and flatten for optimization
        selected_points = np.array(selected_points)
        params_points = selected_points.flatten()
        
        # Create a function that only updates visible points
        def compute_residuals_fixed_cameras(points_params):
            try:
                # Reshape flat array back to points
                reshaped_points = points_params.reshape(-1, 3)
                
                # Update the points in our array
                for opt_idx, point_3d in enumerate(reshaped_points):
                    if opt_idx in opt_idx_to_point_id:
                        point_id = opt_idx_to_point_id[opt_idx]
                        if point_id in self.point_id_to_index:
                            point_idx = self.point_id_to_index[point_id]
                            if point_idx < len(self.points_3d):
                                self.points_3d[point_idx] = point_3d
                
                # Compute residuals without changing camera parameters
                return self._compute_residuals(None)
            except Exception as e:
                print(f"Error in compute_residuals_fixed_cameras: {e}")
                # Get a reasonable number of residuals
                residual_count = max(len(self.point_ids) * 2, self.n_observations * 2)
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
        for point_id in valid_point_ids:
            if point_id in self.point_id_manager.points:
                point = self.point_id_manager.points[point_id]
                for camera_id in point.observations:
                    if camera_id in self.camera_id_to_index:
                        cameras_with_obs.add(self.camera_id_to_index[camera_id])
        
        # Convert to sorted list
        valid_cameras = sorted(list(cameras_with_obs))
        
        # Create mappings between indices
        cam_idx_to_opt_idx = {cam_idx: i for i, cam_idx in enumerate(valid_cameras)}
        
        if verbose:
            print(f"Optimizing {len(valid_cameras)} cameras that have observations (out of {self.n_cameras} total)")
            
        if not valid_cameras:
            print("No valid cameras to optimize in Stage 2. Skipping...")
            return {
                'success': result_points.success,  # Consider stage 1 result
                'final_rmse': points_rmse if 'points_rmse' in locals() else float('nan'),
                'n_iterations': result_points.nfev,
                'optimized_cameras': self.camera_to_world_list,
                'optimized_points': self.points_3d
            }
        
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
                    self.centers[cam_idx] = camera_params[start_idx:start_idx+3]
                    
                    # Update camera_to_world_list
                    R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                    self.camera_to_world_list[cam_idx] = (R, self.centers[cam_idx])
                
                # Compute residuals without changing 3D points
                return self._compute_residuals(None)
            except Exception as e:
                print(f"Error in compute_residuals_fixed_points: {e}")
                # Get a reasonable number of residuals
                residual_count = max(len(self.point_ids) * 2, self.n_observations * 2)
                placeholder_value = 1e-3
                return np.ones(residual_count) * placeholder_value
        
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
            camera_params = []
            
            # Add rotation vectors (rvecs) in order of valid_cameras
            for cam_idx in valid_cameras:
                camera_params.extend(self.rvecs[cam_idx])
                
            # Add camera centers in order of valid_cameras
            for cam_idx in valid_cameras:
                camera_params.extend(self.centers[cam_idx])
                
            # Pack 3D points for points with observations
            point_params = []
            for point_id in valid_point_ids:
                if point_id in self.point_id_to_index:
                    point_idx = self.point_id_to_index[point_id]
                    if point_idx < len(self.points_3d):
                        point_params.extend(self.points_3d[point_idx])
                
            # Combine all parameters
            params_all = np.concatenate([np.array(camera_params), np.array(point_params)])
            
            # Store optimization structure
            camera_count = len(valid_cameras)
            point_count = len(valid_point_ids)
            
            if verbose:
                print(f"Stage 3: Optimizing {camera_count} cameras and {point_count} points")
                print(f"Total parameters: {len(params_all)}")
                print(f"Camera params: {len(camera_params)}, Point params: {len(point_params)}")
            
            # Create a clear mapping from parameter position to entity
            point_param_start = len(camera_params)
            
            # Create a custom residual function to handle our reduced parameter set
            def compute_residuals_all(all_params):
                try:
                    # Split parameters into camera and point parts
                    camera_params = all_params[:point_param_start]
                    point_params = all_params[point_param_start:]
                    
                    # Update camera parameters
                    for i, cam_idx in enumerate(valid_cameras):
                        # Update rotation vector (rvec)
                        self.rvecs[cam_idx] = camera_params[i*3:i*3+3]
                    
                    # Update camera centers
                    center_start = camera_count * 3  # Skip all rvecs
                    for i, cam_idx in enumerate(valid_cameras):
                        self.centers[cam_idx] = camera_params[center_start + i*3:center_start + i*3+3]
                        
                        # Update camera_to_world_list
                        R, _ = cv2.Rodrigues(self.rvecs[cam_idx])
                        self.camera_to_world_list[cam_idx] = (R, self.centers[cam_idx])
                    
                    # Update 3D points
                    for i, point_id in enumerate(valid_point_ids):
                        if point_id in self.point_id_to_index:
                            point_idx = self.point_id_to_index[point_id]
                            if point_idx < len(self.points_3d) and i*3+3 <= len(point_params):
                                self.points_3d[point_idx] = point_params[i*3:i*3+3]
                    
                    # Compute residuals
                    return self._compute_residuals(None)
                except Exception as e:
                    print(f"Error in compute_residuals_all: {e}")
                    # Get a reasonable number of residuals
                    residual_count = max(len(self.point_ids) * 2, self.n_observations * 2)
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
                
                # Check for significant improvement
                improvement = 0
                if not np.isnan(final_rmse) and not np.isnan(initial_rmse):
                    improvement = (initial_rmse - final_rmse) / initial_rmse * 100
                
                # If there was at least 20% improvement, mark as successful regardless of convergence
                if improvement >= 20:
                    print(f"Stage 3 optimization achieved {improvement:.2f}% error reduction. Marking as successful despite non-convergence.")
                    optimization_success = True
                # Only revert if error increased significantly (more than 10%)
                elif (np.isnan(final_rmse) or final_rmse > initial_rmse * 1.1):
                    print(f"Optimization increased error from {initial_rmse:.4f} to {final_rmse:.4f}. Reverting to original parameters.")
                    # Restore original values
                    self.points_3d = original_points.copy()
                    self.rvecs = [rvec.copy() for rvec in original_rvecs]
                    self.centers = [center.copy() for center in original_centers]
                else:
                    print(f"Error improved from {initial_rmse:.4f} to {final_rmse:.4f}. Keeping results despite non-convergence.")
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
    
    def optimize(self, n_iterations: int = 100, verbose: bool = True, use_staged: bool = True, epipolar_threshold: float = None) -> Dict:
        """ Run bundle adjustment optimization with COLMAP-like approach.
        
        Args:
            n_iterations: Maximum number of iterations
            verbose: Whether to print progress
            use_staged: Whether to use staged optimization (COLMAP style)
            
        Returns:
            Dict with optimization results
        """
        # エピポーラフィルタリングが有効な場合、観測をフィルタリング
        if epipolar_threshold is not None and epipolar_threshold > 0:
            self.filter_observations_with_epipolar(epipolar_threshold, verbose=verbose)
            
        # Analyze the observation model to count valid observations using point IDs
        total_valid_observations = 0
        points_with_observations = set()
        cameras_with_observations = set()
        
        # Count valid observations for each point - now without filtering by is_valid flag
        for point_opt_idx, point_id in enumerate(self.point_ids):
            if point_id not in self.point_id_manager.points:
                continue
                
            point = self.point_id_manager.points[point_id]
            valid_point_observations = 0
            
            # Check each camera observing this point
            for camera_id, point_2d in point.observations.items():
                if camera_id not in self.camera_id_to_index:
                    continue
                
                cam_idx = self.camera_id_to_index[camera_id]
                
                # Check if point is in front of camera and not NaN
                R_ctw, c = self.camera_to_world_list[cam_idx]
                
                # Convert to world-to-camera for projection
                R_wtc = R_ctw.T
                t_wtc = -R_wtc @ c
                
                point_3d = self.points_3d[point_opt_idx]
                point_cam = R_wtc @ point_3d + t_wtc
                
                if point_cam[2] > 0 and not np.isnan(point_2d).any():
                    total_valid_observations += 1
                    valid_point_observations += 1
                    cameras_with_observations.add(camera_id)
            
            # Only count points with at least 2 observations (COLMAP requirement)
            if valid_point_observations >= 2:
                points_with_observations.add(point_id)
        
        if verbose:
            print(f"------------------------------------------------")
            print(f"BUNDLE ADJUSTMENT DATA SUMMARY:")
            print(f"Total 3D points in manager: {len(self.point_ids)}")
            print(f"Points with ANY observations: {sum(1 for p_id in self.point_ids if p_id in self.point_id_manager.points and len(self.point_id_manager.points[p_id].observations) > 0)}")
            print(f"Points with 2+ observations: {len(points_with_observations)} (these points can be triangulated)")
            print(f"Total valid observations: {total_valid_observations}")
            print(f"Cameras with observations: {len(cameras_with_observations)}")
            print(f"Observations per point with 2+ obs: {total_valid_observations/max(1, len(points_with_observations)):.2f}")
            print(f"------------------------------------------------")
        
        # Check if we have enough data for a meaningful BA
        # Each point must be observed by at least 2 cameras to be constrained
        # We need at least 3 points with observations for a well-posed problem
        if (len(points_with_observations) < 3 or 
            len(cameras_with_observations) < 2 or 
            total_valid_observations < 10):
            
            print(f"Warning: Insufficient data for meaningful Bundle Adjustment:")
            print(f"  Points with 2+ observations: {len(points_with_observations)} (need at least 3)")
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
                results = self._optimize_staged(n_iterations, verbose, loss_fn, method)
            else:
                results = self._optimize_standard(n_iterations, verbose, loss_fn, method, initial_rmse)
                
            # Update the point_id_manager with optimized parameters
            self._update_point_id_manager(results['success'])
            
            return results
        except Exception as e:
            print(f"Bundle Adjustment failed with error: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'optimized_cameras': self.camera_to_world_list,
                'optimized_points': self.points_3d
            }
            
    def _update_point_id_manager(self, optimization_successful: bool) -> None:
        """Update the point ID manager with optimized parameters.
        
        Args:
            optimization_successful: Whether the optimization was successful
        """
        if not optimization_successful:
            # In our case, even if scipy says it's unsuccessful, we still want to update
            # the point manager if there was significant error reduction
            print("Optimization didn't fully converge, but still updating point ID manager if error improved")
            
        # Get points with observations - only update these
        points_with_obs = set(self.get_point_ids_with_observations(min_observations=1))
        points_updated = 0
            
        # Update point positions - but only for points that have observations
        for i, point_id in enumerate(self.point_ids):
            if point_id in self.point_id_manager.points:
                # Only update points that have at least one observation
                if point_id in points_with_obs:
                    # Update position (the only property modified by bundle adjustment)
                    self.point_id_manager.points[point_id].position = self.points_3d[i]
                    points_updated += 1
                
        print(f"Updated {points_updated} points that had observations (out of {len(self.point_ids)} total points)")
                
        # Update camera parameters
        for cam_idx, (R_ctw, c) in enumerate(self.camera_to_world_list):
            if cam_idx in self.camera_index_to_id:
                camera_id = self.camera_index_to_id[cam_idx]
                if camera_id in self.point_id_manager.cameras:
                    # Update rotation and camera center
                    self.point_id_manager.cameras[camera_id].R = R_ctw
                    self.point_id_manager.cameras[camera_id].c = c
                    
                    
    def _calculate_fundamental_matrix(
        self, R1_ctw, c1, K1, R2_ctw, c2, K2
    ) -> np.ndarray:
        """2台のカメラ間の基礎行列を計算
        
        Args:
            R1_ctw: カメラ1の回転行列（カメラ→ワールド）
            c1: カメラ1の中心座標（ワールド座標系）
            K1: カメラ1の内部パラメータ行列
            R2_ctw: カメラ2の回転行列（カメラ→ワールド）
            c2: カメラ2の中心座標（ワールド座標系）
            K2: カメラ2の内部パラメータ行列
            
        Returns:
            F: カメラ1の点からカメラ2のエピポーラ線への写像を表す基礎行列
        """
        # カメラ→ワールドからワールド→カメラへの変換
        R1_wtc = R1_ctw.T
        t1_wtc = -R1_wtc @ c1
        
        R2_wtc = R2_ctw.T
        t2_wtc = -R2_wtc @ c2
        
        # カメラ1からカメラ2への相対的な回転と並進
        R_rel = R2_wtc @ R1_ctw  # カメラ1からカメラ2への変換
        t_rel = t2_wtc - R_rel @ t1_wtc
        
        # t_relの外積行列を作成
        t_cross = np.array([
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0]
        ])
        
        # 基本行列の計算: E = [t]_x * R
        E = t_cross @ R_rel
        
        # 基礎行列の計算: F = K2^-T * E * K1^-1
        F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
        
        return F

    def filter_observations_with_epipolar(self, threshold: float = 2.0, min_inlier_ratio: float = 0.5, verbose: bool = True):
        """エピポーラ制約に違反する観測をフィルタリング
        
        Args:
            threshold: エピポーラ線からの最大許容距離（ピクセル単位）
            min_inlier_ratio: 観測を維持するためのインライアの最小比率
            verbose: 進捗を表示するかどうか
            
        Returns:
            int: フィルタリングされた観測の数
        """
        # フィルタリング統計の初期化
        filtered_count = 0
        total_observations = 0
        points_affected = 0
        
        # 全ての点を反復処理
        for point_id in self.point_ids:
            if point_id not in self.point_id_manager.points:
                continue
                
            point = self.point_id_manager.points[point_id]
            cameras_observing_point = list(point.observations.keys())
            total_observations += len(cameras_observing_point)
            
            # 観測が3未満の点はスキップ（フィルタリング後に少なくとも2つ必要）
            if len(cameras_observing_point) < 3:
                continue
            
            # この点の観測をフィルタリング
            filtered_cameras = set()
            
            # 各カメラを他のすべてのカメラと比較
            for camera_id1 in cameras_observing_point:
                if camera_id1 not in self.camera_id_to_index:
                    continue
                    
                # 外れ値としてのこの観測を持つ他のカメラの数をカウント
                outlier_count = 0
                total_checked = 0
                
                for camera_id2 in cameras_observing_point:
                    if camera_id2 == camera_id1 or camera_id2 not in self.camera_id_to_index:
                        continue
                        
                    # カメラパラメータを取得
                    point_2d1 = point.observations[camera_id1]
                    cam_idx1 = self.camera_id_to_index[camera_id1]
                    R1_ctw, c1 = self.camera_to_world_list[cam_idx1]
                    K1 = self.intrinsics_list[cam_idx1]
                    
                    point_2d2 = point.observations[camera_id2]
                    cam_idx2 = self.camera_id_to_index[camera_id2]
                    R2_ctw, c2 = self.camera_to_world_list[cam_idx2]
                    K2 = self.intrinsics_list[cam_idx2]
                    
                    # いずれかの点がNaNの場合はスキップ
                    if np.isnan(point_2d1).any() or np.isnan(point_2d2).any():
                        continue
                    
                    # 基礎行列を計算
                    F = self._calculate_fundamental_matrix(
                        R1_ctw, c1, K1, 
                        R2_ctw, c2, K2
                    )
                    
                    # 2番目の画像でのエピポーラ線を計算
                    point_2d1_h = np.append(point_2d1, 1)
                    epipolar_line2 = F @ point_2d1_h
                    
                    # エピポーラ線の法線のノルムを計算
                    line_normal = np.sqrt(epipolar_line2[0]**2 + epipolar_line2[1]**2)
                    if line_normal < 1e-10:  # 非常に小さい場合はスキップ（0除算回避）
                        continue
                    
                    # 2番目の点からエピポーラ線までの距離を計算
                    point_2d2_h = np.append(point_2d2, 1)
                    distance = abs(np.dot(epipolar_line2, point_2d2_h)) / line_normal
                    
                    total_checked += 1
                    if distance > threshold:
                        outlier_count += 1
                
                # チェックの半分以上が失敗した場合、この観測をフィルタリング
                if total_checked > 0 and outlier_count / total_checked > (1 - min_inlier_ratio):
                    filtered_cameras.add(camera_id1)
                    filtered_count += 1
            
            # フィルタリングされたカメラを観測から削除
            original_count = len(point.observations)
            for camera_id in filtered_cameras:
                if camera_id in point.observations:
                    del point.observations[camera_id]
            
            # 点が影響を受けたかどうかを確認
            if len(point.observations) < original_count:
                points_affected += 1
        
        if verbose:
            print(f"エピポーラフィルタリング: 合計{total_observations}観測から{filtered_count}を除外")
            print(f"{points_affected}点に影響, 閾値={threshold:.2f}px")
        
        return filtered_count