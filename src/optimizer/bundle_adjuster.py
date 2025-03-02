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
            use_robust_loss: Whether to use robust loss function
            loss_scale: Scale parameter for robust loss
        """
        self.points_3d = points_3d.copy()
        self.camera_params_list = [(R.copy(), t.copy()) for R, t in camera_params_list]
        self.match_points_2d = match_points_2d
        self.intrinsics_list = intrinsics_list
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
    
    def export_colmap_format(self, output_dir: str) -> None:
        """
        Export the bundle adjustment results in COLMAP format.
        
        Args:
            output_dir: Directory to save the COLMAP files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export cameras (cameras.txt)
        with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
            # Header
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            # Simple pinhole camera model for all cameras
            for i, K in enumerate(self.intrinsics_list):
                width = int(K[0, 2] * 2)  # Approximate from principal point
                height = int(K[1, 2] * 2)
                f.write(f"{i+1} SIMPLE_PINHOLE {width} {height} {K[0, 0]} {K[0, 2]} {K[1, 2]}\n")
        
        # Export images (images.txt)
        with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
            # Header
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for i, (R, t) in enumerate(self.camera_params_list):
                # Convert R to quaternion
                rot_mat = np.eye(4)
                rot_mat[:3, :3] = R
                rot_mat[:3, 3] = t
                
                # OpenCV to COLMAP transformation (right-handed to left-handed)
                rot_mat[1, :] = -rot_mat[1, :]
                rot_mat[2, :] = -rot_mat[2, :]
                
                # Extract quaternion
                trace = np.trace(rot_mat[:3, :3])
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    qw = 0.25 / s
                    qx = (rot_mat[2, 1] - rot_mat[1, 2]) * s
                    qy = (rot_mat[0, 2] - rot_mat[2, 0]) * s
                    qz = (rot_mat[1, 0] - rot_mat[0, 1]) * s
                else:
                    if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
                        qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                        qx = 0.25 * s
                        qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                        qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                    elif rot_mat[1, 1] > rot_mat[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
                        qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                        qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                        qy = 0.25 * s
                        qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
                        qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
                        qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                        qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                        qz = 0.25 * s
                
                # Image name
                image_name = f"image_{i+1:06d}.jpg"
                
                # Write image info
                f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i+1} {image_name}\n")
                
                # Write point observations
                points_str = ""
                for point_idx, point_2d in self.match_points_2d[i]:
                    points_str += f"{point_2d[0]:.6f} {point_2d[1]:.6f} {point_idx+1} "
                
                f.write(f"{points_str}\n")
        
        # Export points3D (points3d.txt)
        with open(os.path.join(output_dir, 'points3d.txt'), 'w') as f:
            # Header
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            for i, point in enumerate(self.points_3d):
                # Default color (white)
                r, g, b = 255, 255, 255
                
                # Find all track elements
                track = []
                for cam_idx, observations in enumerate(self.match_points_2d):
                    for j, (point_idx, _) in enumerate(observations):
                        if point_idx == i:
                            track.append((cam_idx + 1, j))
                
                track_str = " ".join([f"{cam_id} {point2d_idx}" for cam_id, point2d_idx in track])
                
                # Write 3D point
                f.write(f"{i+1} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {r} {g} {b} 1.0 {track_str}\n")
        
        # Also export as PLY
        self.export_ply(os.path.join(output_dir, 'points3d.ply'))
        
        print(f"COLMAP format data exported to {output_dir}")
    
    def export_ply(self, filepath: str) -> None:
        """
        Export 3D points as PLY file.
        
        Args:
            filepath: Path to save the PLY file
        """
        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # Points (with default white color)
            for point in self.points_3d:
                f.write(f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f} 255 255 255\n")
        
        print(f"PLY file exported to {filepath}")