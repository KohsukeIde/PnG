# src/optimizer/observation_builder.py

import numpy as np
from typing import Dict, List, Tuple, Optional

class ObservationBuilder:
    """
    Builds observation maps for 3D Gaussian Bundle Adjustment.
    
    This class creates the necessary data structures that track which 3D points
    are visible in which cameras, and at what 2D positions.
    """
    
    @staticmethod
    def build_observation_map(reconstruction_data: Dict) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Build observation map from reconstruction data.
        
        Args:
            reconstruction_data: Dictionary containing reconstruction information
            
        Returns:
            Dict mapping 3D point idx -> {camera_idx -> 2D point}
        """
        observation_map = {}
        
        # Step 1: Process the initial pair of images
        if 'match_pairs' in reconstruction_data and reconstruction_data['match_pairs']:
            # Initial match pairs for first two cameras
            match_pairs = reconstruction_data['match_pairs']
            points_3d = reconstruction_data['points_3d']
            
            # Get back-projection of match pairs to 2D coordinates
            gaussians1 = reconstruction_data.get('gaussians1', None)
            gaussians2 = reconstruction_data.get('gaussians2', None)
            
            if gaussians1 is not None and gaussians2 is not None:
                for point_idx, (idx1, idx2) in enumerate(match_pairs):
                    # Create entry for this 3D point if it doesn't exist
                    if point_idx not in observation_map:
                        observation_map[point_idx] = {}
                    
                    # Add observations from camera 0 and 1
                    observation_map[point_idx][0] = gaussians1.means[idx1]
                    observation_map[point_idx][1] = gaussians2.means[idx2]
        
        # Step 2: Process additional cameras from viewpoint extension
        if 'all_matches' in reconstruction_data:
            # This would be a custom field we add to track all matches
            all_matches = reconstruction_data['all_matches']
            for cam_idx, point_matches in enumerate(all_matches):
                for point_idx, point_2d in point_matches:
                    if point_idx not in observation_map:
                        observation_map[point_idx] = {}
                    observation_map[point_idx][cam_idx] = point_2d
        
        # If 'all_matches' wasn't found, try to rebuild from transport matrices
        elif 'transport_matrices' in reconstruction_data:
            # This assumes we've stored all transport matrices
            transport_matrices = reconstruction_data['transport_matrices']
            
            # Process each transport matrix
            for cam_idx, transport_mat in enumerate(transport_matrices):
                if cam_idx < 2:  # Skip first two cameras which we already processed
                    continue
                
                # Get 2D Gaussians for this camera
                cam_gaussians = reconstruction_data.get(f'gaussians{cam_idx+1}', None)
                if cam_gaussians is None:
                    continue
                
                # Find best matches in the transport matrix
                for point_idx in range(len(reconstruction_data['points_3d'])):
                    best_idx = np.argmax(transport_mat[point_idx])
                    if transport_mat[point_idx, best_idx] > 0.1:  # Threshold
                        if point_idx not in observation_map:
                            observation_map[point_idx] = {}
                        observation_map[point_idx][cam_idx] = cam_gaussians.means[best_idx]
        
        # Step 3: If neither method worked, try a simpler approach
        # For each camera, project 3D points and check if they're in view
        if not observation_map and 'camera_params_list' in reconstruction_data:
            camera_params_list = reconstruction_data['camera_params_list']
            points_3d = reconstruction_data['points_3d']
            
            for cam_idx, (R, t) in enumerate(camera_params_list):
                K = reconstruction_data.get('K', None)
                if K is None:
                    K = reconstruction_data.get(f'camera{cam_idx+1}_K', None)
                if K is None:
                    continue
                
                # Project each 3D point to this camera
                for point_idx, point_3d in enumerate(points_3d):
                    # Convert to camera coordinates
                    point_cam = R @ point_3d + t
                    
                    # Check if point is in front of camera
                    if point_cam[2] <= 0:
                        continue
                    
                    # Project to image coordinates
                    point_img = K @ point_cam
                    point_2d = point_img[:2] / point_img[2]
                    
                    # Check if point is within image bounds
                    width = K[0, 2] * 2
                    height = K[1, 2] * 2
                    if 0 <= point_2d[0] < width and 0 <= point_2d[1] < height:
                        if point_idx not in observation_map:
                            observation_map[point_idx] = {}
                        observation_map[point_idx][cam_idx] = point_2d
        
        return observation_map
    
    @staticmethod
    def convert_to_match_points_2d(
        observation_map: Dict[int, Dict[int, np.ndarray]], 
        num_cameras: int
    ) -> List[List[Tuple[int, np.ndarray]]]:
        """
        Convert observation map to match_points_2d format required by BA.
        
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