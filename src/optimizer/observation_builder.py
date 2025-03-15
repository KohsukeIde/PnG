# src/optimizer/observation_builder.py

import numpy as np
from typing import Dict, List, Tuple, Optional

class ObservationBuilder:
    """Builds observation maps for 3D Gaussian Bundle Adjustment.
    
    This class creates the necessary data structures that track which 3D points
    are visible in which cameras, and at what 2D positions.
    """
    
    @staticmethod
    def track_observations(transport_matrix: np.ndarray, means_2d: np.ndarray, 
                           confidence_threshold: float = 0.1, 
                           top_k: Optional[int] = None,
                           use_combined_filtering: bool = False) -> List[Tuple[int, np.ndarray]]:
        """Extract point correspondences from a transport matrix.
        
        Args:
            transport_matrix: Optimal transport matrix between 3D points and 2D Gaussians
            means_2d: 2D means of the Gaussians in the target view
            confidence_threshold: Minimum transport value to consider a valid correspondence
            top_k: If provided, select only the top-k matches with highest transport values
                  (defaults to None, which means take all matches above threshold)
            use_combined_filtering: If True, apply both top-k and threshold filtering
                                   (i.e., take the top-k matches that also exceed the threshold)
            
        Returns:
            List of (point_idx, [x, y]) tuples for each valid correspondence
        """
        observations = []
        
        if top_k is not None:
            # Adjust top_k to not exceed the number of 2D Gaussians
            effective_top_k = min(top_k, means_2d.shape[0])
            
            if use_combined_filtering:
                # Combined approach: select top-k matches that also exceed threshold
                for point_idx in range(transport_matrix.shape[0]):
                    # Get indices sorted by transport value (descending)
                    sorted_indices = np.argsort(-transport_matrix[point_idx])
                    
                    # Take top-k indices that also exceed threshold
                    matches_added = 0
                    for idx in sorted_indices[:effective_top_k]:
                        transport_value = transport_matrix[point_idx, idx]
                        if transport_value > confidence_threshold:
                            point_2d = means_2d[idx]
                            observations.append((point_idx, point_2d))
                            matches_added += 1
            else:
                # Top-k only approach: select the top-k matches regardless of threshold
                for point_idx in range(transport_matrix.shape[0]):
                    # Get indices of top-k matches
                    top_indices = np.argsort(-transport_matrix[point_idx])[:effective_top_k]
                    
                    for idx in top_indices:
                        transport_value = transport_matrix[point_idx, idx]
                        # Skip entries with zero transport value
                        if transport_value > 0:
                            point_2d = means_2d[idx]
                            observations.append((point_idx, point_2d))
        else:
            # Original threshold-only approach
            for point_idx in range(transport_matrix.shape[0]):
                best_idx = np.argmax(transport_matrix[point_idx])
                best_transport_value = transport_matrix[point_idx, best_idx]
                
                # Only keep correspondences with sufficient confidence
                if best_transport_value > confidence_threshold:
                    point_2d = means_2d[best_idx]
                    observations.append((point_idx, point_2d))
        
        print(f"Found {len(observations)} observations from transport matrix")
        return observations
    
    @staticmethod
    def build_observation_map(reconstruction_data: Dict) -> Dict[int, Dict[int, np.ndarray]]:
        """Build observation map from reconstruction data.
        
        This function uses a single, explicit approach to build observations
        using the 'all_matches' data, which should be the primary source of
        correspondence information.
        
        Args:
            reconstruction_data: Dictionary containing:
                - 'points_3d': 3D point coordinates
                - 'all_matches': List of correspondences for each camera
            
        Returns:
            Dict mapping 3D point idx -> {camera_idx -> 2D point}
        """
        observation_map = {}
        
        # Check for required data
        if 'points_3d' not in reconstruction_data:
            print("ERROR: 'points_3d' not found in reconstruction data")
            return observation_map
            
        # Process correspondences from all_matches
        if 'all_matches' in reconstruction_data:
            all_matches = reconstruction_data['all_matches']
            print(f"Building observations from all_matches ({len(all_matches)} cameras)")
            
            for cam_idx, point_matches in enumerate(all_matches):
                for point_idx, point_2d in point_matches:
                    if point_idx not in observation_map:
                        observation_map[point_idx] = {}
                    observation_map[point_idx][cam_idx] = point_2d
                    
            if observation_map:
                return observation_map
                
        # If no observations found, issue a warning
        assert False, "WARNING: No observations could be built. Make sure 'all_matches' is available in reconstruction_data."
        
        # #observation_map = {
        #     0: {0: np.array([100, 200]), 1: np.array([110, 210])},  # ポイント0はカメラ0と1で見える
        #     1: {0: np.array([150, 250]), 2: np.array([170, 270])}   # ポイント1はカメラ0と2で見える
        # }

    
    @staticmethod
    def convert_to_match_points_2d(
        observation_map: Dict[int, Dict[int, np.ndarray]], 
        num_cameras: int
    ) -> List[List[Tuple[int, np.ndarray]]]:
        """Convert observation map to match_points_2d format required by BA.
        
        Args:
            observation_map: Dict mapping 3D point idx -> {camera_idx -> 2D point}
            num_cameras: Number of cameras in the scene
            
        Returns:
            List of lists, where each inner list contains (point_idx, [x, y]) pairs for one camera
        """
        match_points_2d = [[] for _ in range(num_cameras)]
        
        for point_idx, camera_observations in observation_map.items():
            for cam_idx, point_2d in camera_observations.items():
                # Convert to numpy array if it's not already
                if not isinstance(point_2d, np.ndarray):
                    point_2d = np.array(point_2d)
                
                # Ensure it's the right shape
                if point_2d.shape != (2,):
                    # If it's a vector with extra dimensions, reshape it
                    point_2d = point_2d.flatten()[:2]
                
                if cam_idx < num_cameras:
                    match_points_2d[cam_idx].append((point_idx, point_2d))
        
        # Print statistics
        total_obs = sum(len(cam_obs) for cam_obs in match_points_2d)
        print(f"Total observations: {total_obs} across {num_cameras} cameras")
        for i, obs in enumerate(match_points_2d):
            print(f"  Camera {i}: {len(obs)} observations")
        
        return match_points_2d
    
        #     match_points_2d = [
        #     [(0, np.array([100, 200])), (1, np.array([150, 250]))],  # カメラ0の観測
        #     [(0, np.array([110, 210]))],                             # カメラ1の観測
        #     [(1, np.array([170, 270]))]                              # カメラ2の観測
        # ]