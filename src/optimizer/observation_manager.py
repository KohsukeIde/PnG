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
    