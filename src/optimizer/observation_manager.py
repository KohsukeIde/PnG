# src/optimizer/observation_manager.py

import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set, Union
from .point_id_manager import PointIDManager

class ObservationManager:
    """Manages observations between 3D points and cameras.
    
    This class provides a unified interface for tracking and managing
    correspondences between 3D points and cameras using persistent point IDs.
    """
    
    def __init__(self, point_manager: Optional[PointIDManager] = None):
        """Initialize the observation manager.
        
        Args:
            point_manager: Optional existing PointIDManager to use
        """
        self.point_manager = point_manager if point_manager is not None else PointIDManager()
    
    def track_observations_from_transport(
        self,
        transport_matrix: np.ndarray,
        means_2d: Union[np.ndarray, torch.Tensor],
        point_ids: List[int],
        camera_id: int,
    ) -> List[int]:
        """Track correspondences from a transport matrix and add to the point manager.
        
        This method handles converting tensors to numpy arrays, validating inputs, and
        applying different filtering strategies based on parameters.
        
        Args:
            transport_matrix: Optimal transport matrix between 3D points and 2D Gaussians
            means_2d: 2D means of the Gaussians in the target view
            point_ids: List of point IDs corresponding to rows in transport_matrix
            camera_id: ID of the camera observing the 2D points
        Returns:
            List of point IDs for which observations were added
        """
        updated_point_ids = []

       # PyTorchテンソルの場合はnumpy配列に変換
        if isinstance(means_2d, torch.Tensor):
            means_2d = means_2d.detach().cpu().numpy()
            
        # カメラの存在確認
        if camera_id not in self.point_manager.cameras:
            sys.exit(f"Warning: Camera ID {camera_id} not found in point manager")
        
        # 輸送行列から観測を追跡
        # 各ポイントIDに対して処理
        for i, point_id in enumerate(point_ids):
            # point_idが無効な場合はスキップ
            if point_id not in self.point_manager.points:
                sys.exit(f"Error: Point ID {point_id} not found in point manager")
                
            # 非ゼロの輸送値を持つ全てのインデックスを取得
            nonzero_indices = np.where(transport_matrix[i] > 0)[0]
            # 非ゼロの輸送を持つ全ての観測を追加
            for idx in nonzero_indices:
                    point_2d = means_2d[idx]
                    self.point_manager.add_observation(point_id, camera_id, point_2d)
                    if point_id not in updated_point_ids:
                        updated_point_ids.append(point_id)
        
        print(f"Added {len(updated_point_ids)} observations to camera {camera_id}")
        return updated_point_ids
        
    def track_observations_from_match_pairs(
        self,
        match_pairs: List[Tuple[int, int]],
        gaussians1_means: Union[np.ndarray, torch.Tensor],
        gaussians2_means: Union[np.ndarray, torch.Tensor],
        point_ids: List[int],
        camera1_id: int,
        camera2_id: int
    ) -> None:
        """Track correspondences from match pairs that were used in triangulation.
        
        This is a more direct and accurate way to track observations when the exact
        2D Gaussian indices used in triangulation are known.
        
        Args:
            match_pairs: List of (gaussian1_idx, gaussian2_idx) tuples from triangulation
            gaussians1_means: 2D means of the Gaussians in the first view
            gaussians2_means: 2D means of the Gaussians in the second view
            point_ids: List of point IDs corresponding to triangulated 3D points
            camera1_id: ID of the first camera
            camera2_id: ID of the second camera
        Returns:
            List of point IDs for which observations were added
        """
        
        # Convert tensors to numpy arrays if needed
        if isinstance(gaussians1_means, torch.Tensor):
            gaussians1_means = gaussians1_means.detach().cpu().numpy()
        if isinstance(gaussians2_means, torch.Tensor):
            gaussians2_means = gaussians2_means.detach().cpu().numpy()
            
        # Check if cameras exist
        if camera1_id not in self.point_manager.cameras:
            sys.exit(f"Error: Camera ID {camera1_id} not found in point manager")
        if camera2_id not in self.point_manager.cameras:
            sys.exit(f"Error: Camera ID {camera2_id} not found in point manager")
            
        # Verify that the lengths match
        if len(point_ids) != len(match_pairs):
            sys.exit(f"Error: Number of point IDs ({len(point_ids)}) does not match number of match pairs ({len(match_pairs)})")
            
        # Add observations for each 3D point using the exact match pairs
        for i, point_id in enumerate(point_ids):
            # Skip invalid point IDs
            if point_id not in self.point_manager.points:
                sys.exit(f"Error: Point ID {point_id} not found in point manager")
                
            # Get the corresponding 2D Gaussian indices for this point
            gaussian1_idx, gaussian2_idx = match_pairs[i]
            
            # Add observation for camera 1
            point_2d_camera1 = gaussians1_means[gaussian1_idx]
            self.point_manager.add_observation(point_id, camera1_id, point_2d_camera1)
                
            # Add observation for camera 2

            point_2d_camera2 = gaussians2_means[gaussian2_idx]
            self.point_manager.add_observation(point_id, camera2_id, point_2d_camera2)
            
                
        print(f"Added observations for {len(point_ids)} points using exact match pairs")
        
    
    def count_points_by_observation(self, min_observations: int = 2) -> Dict[str, int]:
        """Count points based on observation count.
        
        Unlike the previous filter_points_by_observation_count, this method DOES NOT
        mark points as valid/invalid. It only returns statistics about point observations.
        All points are included in the BA regardless of observation count, with appropriate
        filtering only happening during visibility checks.
        
        Args:
            min_observations: Threshold for counting points with sufficient observations
            
        Returns:
            Dict with counts of points by observation category
        """
        # Count observations by category
        with_sufficient_obs = 0
        single_observation = 0
        no_observations = 0
        
        for point_id, point in self.point_manager.points.items():
            obs_count = point.observation_count()
            
            if obs_count >= min_observations:
                with_sufficient_obs += 1
            elif obs_count == 1:
                single_observation += 1
            else:
                no_observations += 1
                
        print(f"Point statistics:")
        print(f"  {min_observations}+ observations: {with_sufficient_obs}")
        print(f"  Single observation: {single_observation}")
        print(f"  No observations: {no_observations}")
        print(f"  Total points: {len(self.point_manager.points)}")
              
        return {
            "with_sufficient_obs": with_sufficient_obs,
            "single_observation": single_observation,
            "no_observations": no_observations,
            "total": len(self.point_manager.points)
        }
    
    def build_observation_map(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Build observation map from the point manager.
        
        Returns:
            Dict mapping point_id -> {camera_id -> 2D point}
        """
        return self.point_manager.build_observation_map()
    
    def convert_to_match_points_2d(self) -> List[List[Tuple[int, np.ndarray]]]:
        """Convert observations to the match_points_2d format required by bundle adjustment.
        
        Returns:
            List of lists, where each inner list contains (point_id, [x, y]) pairs for one camera
        """
        return self.point_manager.convert_to_match_points_2d()
    
    
    def from_legacy_reconstruction_data(self, reconstruction_data: Dict) -> None:
        """Initialize the observation manager from legacy reconstruction data.
        
        Args:
            reconstruction_data: Dictionary containing reconstruction data
        """
        # Extract data from reconstruction_data
        points_3d = reconstruction_data.get("points_3d", None)
        camera_params_list = reconstruction_data.get("camera_params_list", None)
        all_matches = reconstruction_data.get("all_matches", None)
        
        # Determine if we have intrinsics for each camera or a shared one
        intrinsics_list = []
        if "intrinsics_list" in reconstruction_data:
            intrinsics_list = reconstruction_data["intrinsics_list"]
        else:
            # Try to build intrinsics list from individual K matrices or shared K
            for i in range(len(camera_params_list)):
                cam_key = f"camera{i+1}_K"
                if cam_key in reconstruction_data:
                    intrinsics_list.append(reconstruction_data[cam_key])
                elif "K" in reconstruction_data:
                    intrinsics_list.append(reconstruction_data["K"])
                else:
                    # Default to identity if no intrinsics found
                    intrinsics_list.append(np.eye(3))
        
        # Get image names if available
        image_names = reconstruction_data.get("used_images", None)
        
        # Get additional data for 3D Gaussians
        covariances_3d = reconstruction_data.get("covariances_3d", None)
        colors_3d = reconstruction_data.get("color_3d", None)
        alphas_3d = reconstruction_data.get("alpha_3d", None)
        quaternions = reconstruction_data.get("quaternions", None)
        scales = reconstruction_data.get("scales", None)
        
        # Only proceed if we have the minimum required data
        if points_3d is None or camera_params_list is None:
            raise ValueError("Missing required data in reconstruction_data")
            
        # Convert to persistent ID format
        self.point_manager.convert_from_legacy_format(
            points_3d=points_3d,
            camera_params_list=camera_params_list,
            match_points_2d=all_matches if all_matches is not None else [],
            intrinsics_list=intrinsics_list,
            image_names=image_names,
            covariances_3d=covariances_3d,
            colors_3d=colors_3d,
            alphas_3d=alphas_3d,
            quaternions=quaternions,
            scales=scales
        )
    
    def to_legacy_reconstruction_data(self, reconstruction_data: Dict) -> Dict:
        """Convert the observation manager data to legacy reconstruction data.
        
        Args:
            reconstruction_data: Dictionary containing original reconstruction data to be updated
            
        Returns:
            Updated reconstruction_data with new arrays and observations
        """
        # Get legacy format data
        (
            points_3d,
            camera_params_list,
            match_points_2d,
            intrinsics_list,
            image_names,
            covariances_3d,
            colors_3d,
            alphas_3d,
            quaternions,
            scales
        ) = self.point_manager.convert_to_legacy_format()
        
        # Update reconstruction_data with new arrays
        reconstruction_data["points_3d"] = points_3d
        reconstruction_data["camera_params_list"] = camera_params_list
        reconstruction_data["all_matches"] = match_points_2d
        
        # Update other arrays if they were in the original data
        if covariances_3d is not None:
            reconstruction_data["covariances_3d"] = covariances_3d
        if colors_3d is not None:
            reconstruction_data["color_3d"] = colors_3d
        if alphas_3d is not None:
            reconstruction_data["alpha_3d"] = alphas_3d
        if quaternions is not None:
            reconstruction_data["quaternions"] = quaternions
        if scales is not None:
            reconstruction_data["scales"] = scales
            
        # Update intrinsics
        if "intrinsics_list" in reconstruction_data:
            reconstruction_data["intrinsics_list"] = intrinsics_list
        elif len(intrinsics_list) > 0:
            # If we only had a shared K before, update it
            if "K" in reconstruction_data and len(intrinsics_list) > 0:
                reconstruction_data["K"] = intrinsics_list[0]
                
            # Add individual camera intrinsics
            for i, K in enumerate(intrinsics_list):
                cam_key = f"camera{i+1}_K"
                reconstruction_data[cam_key] = K
                
        # Update image names
        if image_names is not None:
            reconstruction_data["used_images"] = image_names
            
        # Update total count
        reconstruction_data["total_3d_gaussians"] = len(points_3d)
        
        # Store the observation data directly
        reconstruction_data["existing_3d_gaussians"] = []
        for i in range(len(points_3d)):
            gauss = {
                "center": points_3d[i],
                "covariance": covariances_3d[i] if covariances_3d is not None and i < len(covariances_3d) else None,
                "color": colors_3d[i] if colors_3d is not None and i < len(colors_3d) else None,
                "alpha": alphas_3d[i] if alphas_3d is not None and i < len(alphas_3d) else None,
                "quaternion": quaternions[i] if quaternions is not None and i < len(quaternions) else None,
                "scale": scales[i] if scales is not None and i < len(scales) else None
            }
            reconstruction_data["existing_3d_gaussians"].append(gauss)
            
        # Store the point ID manager directly for future use
        reconstruction_data["point_id_manager"] = self.point_manager
        reconstruction_data["observation_manager"] = self
            
        return reconstruction_data
        
    @staticmethod
    def create_from_reconstruction(reconstruction_data: Dict) -> 'ObservationManager':
        """Create an ObservationManager from reconstruction data.
        
        Args:
            reconstruction_data: Dictionary containing reconstruction data
            
        Returns:
            ObservationManager initialized with data from reconstruction_data
        """
        # Check if we already have an observation manager
        if "observation_manager" in reconstruction_data:
            return reconstruction_data["observation_manager"]
            
        # Check if we have a point manager to use
        if "point_id_manager" in reconstruction_data:
            manager = ObservationManager(reconstruction_data["point_id_manager"])
        else:
            # Create a new manager and populate it from legacy data
            manager = ObservationManager()
            manager.from_legacy_reconstruction_data(reconstruction_data)
            
        return manager