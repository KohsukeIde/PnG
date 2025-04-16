# src/optimizer/point_id_manager.py

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys

class Point3D:
    """A 3D point with persistent ID and observations.
    
    This class represents a 3D point with a unique identifier that remains
    consistent regardless of deletions or additions of other points.
    """
    
    def __init__(
        self, 
        point_id: int, 
        position: np.ndarray, 
        covariance: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None, 
        alpha: Optional[float] = None,
        quaternion: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None
    ):
        """Initialize a 3D point with a unique ID.
        
        Args:
            point_id: Unique identifier for this point
            position: 3D position vector [x, y, z]
            covariance: 3x3 covariance matrix for 3D Gaussian
            color: RGB color vector
            alpha: Alpha/opacity value
            quaternion: Quaternion for Gaussian orientation
            scale: Scale factors for Gaussian
        """
        self.id = point_id
        self.position = position
        self.covariance = covariance
        self.color = color
        self.alpha = alpha
        self.quaternion = quaternion
        self.scale = scale
        self.observations = {}  # camera_id -> 2D coordinate
        # Track which view the point was created from
        self.created_from_view_id = None
        self.is_valid = True  # Flag to mark if point should be filtered out
        
    def add_observation(self, camera_id: int, point_2d: np.ndarray) -> None:
        """Add a 2D observation of this point from a specific camera.
        
        Args:
            camera_id: ID of the observing camera
            point_2d: 2D coordinates [x, y] in the camera's image plane
        """
        self.observations[camera_id] = point_2d
        
    def remove_observation(self, camera_id: int) -> None:
        """Remove an observation from a specific camera.
        
        Args:
            camera_id: ID of the camera to remove observation for
        """
        if camera_id in self.observations:
            del self.observations[camera_id]
            
    def observation_count(self) -> int:
        """Get the number of cameras observing this point.
        
        Returns:
            Number of observations
        """
        return len(self.observations)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of this point
        """
        return {
            "id": self.id,
            "position": self.position.tolist() if self.position is not None else None,
            "covariance": self.covariance.tolist() if self.covariance is not None else None,
            "color": self.color.tolist() if self.color is not None else None,
            "alpha": float(self.alpha) if self.alpha is not None else None,
            "quaternion": self.quaternion.tolist() if self.quaternion is not None else None,
            "scale": self.scale.tolist() if self.scale is not None else None,
            "observations": {str(k): v.tolist() for k, v in self.observations.items()},
            "created_from_view_id": self.created_from_view_id,
            "is_valid": self.is_valid
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Point3D':
        """Create a Point3D instance from a dictionary.
        
        Args:
            data: Dictionary containing point data
            
        Returns:
            New Point3D instance
        """
        point = cls(
            point_id=data["id"],
            position=np.array(data["position"]) if data["position"] is not None else None,
            covariance=np.array(data["covariance"]) if data["covariance"] is not None else None,
            color=np.array(data["color"]) if data["color"] is not None else None,
            alpha=data["alpha"],
            quaternion=np.array(data["quaternion"]) if data["quaternion"] is not None else None,
            scale=np.array(data["scale"]) if data["scale"] is not None else None,
        )
        
        if "observations" in data:
            for k, v in data["observations"].items():
                point.observations[int(k)] = np.array(v)
                
        if "created_from_view_id" in data:
            point.created_from_view_id = data["created_from_view_id"]
            
        if "is_valid" in data:
            point.is_valid = data["is_valid"]
            
        return point


class Camera:
    """Camera with persistent ID and intrinsic/extrinsic parameters.
    
    This class represents a camera with a unique identifier, intrinsic
    parameters (K matrix), and extrinsic parameters (R and t).
    """
    
    def __init__(
        self, 
        camera_id: int, 
        R: np.ndarray,  # Camera-to-world rotation
        c: np.ndarray,  # Camera center in world coordinates
        K: np.ndarray,  # Intrinsic matrix
        image_name: Optional[str] = None
    ):
        """Initialize a camera with unique ID and parameters.
        
        Args:
            camera_id: Unique identifier for this camera
            R: Rotation matrix (camera-to-world)
            c: Camera center in world coordinates
            K: Intrinsic matrix
            image_name: Name of the image for this camera view
        """
        self.id = camera_id
        self.R = R  # camera-to-world rotation
        self.c = c  # camera center in world coordinates
        self.K = K  # intrinsic matrix
        self.image_name = image_name
        self.observations = {}  # point3d_id -> 2D coordinate
        
    def add_observation(self, point_id: int, point_2d: np.ndarray) -> None:
        """Add a 2D observation of a 3D point.
        
        Args:
            point_id: ID of the observed 3D point
            point_2d: 2D coordinates [x, y] in the camera's image plane
        """
        self.observations[point_id] = point_2d
        
    def remove_observation(self, point_id: int) -> None:
        """Remove an observation of a specific 3D point.
        
        Args:
            point_id: ID of the 3D point to remove observation for
        """
        if point_id in self.observations:
            del self.observations[point_id]
            
    def world_to_camera_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the world-to-camera rotation and translation.
        
        Returns:
            Tuple of (R_wtc, t_wtc) where R_wtc is the world-to-camera rotation
            and t_wtc is the world-to-camera translation
        """
        R_wtc = self.R.T  # Inverse rotation
        t_wtc = -R_wtc @ self.c  # -R_wtc * c
        return R_wtc, t_wtc
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of this camera
        """
        return {
            "id": self.id,
            "R": self.R.tolist() if self.R is not None else None,
            "c": self.c.tolist() if self.c is not None else None,
            "K": self.K.tolist() if self.K is not None else None,
            "image_name": self.image_name,
            "observations": {str(k): v.tolist() for k, v in self.observations.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Camera':
        """Create a Camera instance from a dictionary.
        
        Args:
            data: Dictionary containing camera data
            
        Returns:
            New Camera instance
        """
        camera = cls(
            camera_id=data["id"],
            R=np.array(data["R"]) if data["R"] is not None else None,
            c=np.array(data["c"]) if data["c"] is not None else None,
            K=np.array(data["K"]) if data["K"] is not None else None,
            image_name=data["image_name"],
        )
        
        if "observations" in data:
            for k, v in data["observations"].items():
                camera.observations[int(k)] = np.array(v)
                
        return camera


class PointIDManager:
    """Manager for persistent point IDs in a 3D reconstruction.
    
    This class manages 3D points with persistent IDs, allowing points to be
    added, removed, or modified while maintaining consistent IDs and observations.
    """
    
    def __init__(self):
        """Initialize the point ID manager."""
        self.points = {}  # point_id -> Point3D
        self.cameras = {}  # camera_id -> Camera
        self.next_point_id = 0
        self.next_camera_id = 0
        
    def add_point(self, 
                  position: np.ndarray, 
                  covariance: Optional[np.ndarray] = None,
                  color: Optional[np.ndarray] = None, 
                  alpha: Optional[float] = None,
                  quaternion: Optional[np.ndarray] = None,
                  scale: Optional[np.ndarray] = None,
                  created_from_view_id: Optional[int] = None) -> int:
        """Add a new 3D point and assign a unique ID.
        
        Args:
            position: 3D position vector [x, y, z]
            covariance: 3x3 covariance matrix for 3D Gaussian
            color: RGB color vector
            alpha: Alpha/opacity value
            quaternion: Quaternion for Gaussian orientation
            scale: Scale factors for Gaussian
            created_from_view_id: ID of the view this point was triangulated from
            
        Returns:
            ID of the new point
        """
        point_id = self.next_point_id
        self.next_point_id += 1
        
        point = Point3D(
            point_id=point_id,
            position=position,
            covariance=covariance,
            color=color,
            alpha=alpha,
            quaternion=quaternion,
            scale=scale
        )
        
        if created_from_view_id is not None:
            point.created_from_view_id = created_from_view_id
            
        self.points[point_id] = point
        return point_id
    
    def add_camera(self, 
                   R: np.ndarray, 
                   c: np.ndarray, 
                   K: np.ndarray,
                   image_name: Optional[str] = None) -> int:
        """Add a new camera and assign a unique ID.
        
        Args:
            R: Rotation matrix (camera-to-world)
            c: Camera center in world coordinates
            K: Intrinsic matrix
            image_name: Name of the image for this camera view
            
        Returns:
            ID of the new camera
        """
        camera_id = self.next_camera_id
        self.next_camera_id += 1
        
        camera = Camera(
            camera_id=camera_id,
            R=R,
            c=c,
            K=K,
            image_name=image_name
        )
        
        self.cameras[camera_id] = camera
        return camera_id
    
    def add_observation(self, point_id: int, camera_id: int, point_2d: np.ndarray) -> bool:
        """Add an observation of a 3D point from a camera.
        
        This updates both the point's observations and the camera's observations.
        
        Args:
            point_id: ID of the 3D point
            camera_id: ID of the observing camera
            point_2d: 2D coordinates [x, y] in the camera's image plane
            
        Returns:
            True if observation was added successfully, False otherwise
        """
        if point_id not in self.points or camera_id not in self.cameras:
            sys.exit(f"Error: Point ID {point_id} or camera ID {camera_id} not found in point manager")
        
        self.points[point_id].add_observation(camera_id, point_2d)
        self.cameras[camera_id].add_observation(point_id, point_2d)
    
    def remove_observation(self, point_id: int, camera_id: int) -> bool:
        """Remove an observation between a 3D point and a camera.
        
        This updates both the point's observations and the camera's observations.
        
        Args:
            point_id: ID of the 3D point
            camera_id: ID of the observing camera
            
        Returns:
            True if observation was removed successfully, False otherwise
        """
        if point_id not in self.points or camera_id not in self.cameras:
            return False
        
        self.points[point_id].remove_observation(camera_id)
        self.cameras[camera_id].remove_observation(point_id)
        return True
    
    def remove_point(self, point_id: int) -> bool:
        """Remove a 3D point and all its observations.
        
        Args:
            point_id: ID of the point to remove
            
        Returns:
            True if point was removed successfully, False otherwise
        """
        if point_id not in self.points:
            return False
        
        # Remove observations from cameras
        point = self.points[point_id]
        for camera_id in point.observations.keys():
            if camera_id in self.cameras:
                self.cameras[camera_id].remove_observation(point_id)
        
        # Remove the point
        del self.points[point_id]
        return True
    
    def remove_camera(self, camera_id: int) -> bool:
        """Remove a camera and all its observations.
        
        Args:
            camera_id: ID of the camera to remove
            
        Returns:
            True if camera was removed successfully, False otherwise
        """
        if camera_id not in self.cameras:
            return False
        
        # Remove observations from points
        camera = self.cameras[camera_id]
        for point_id in camera.observations.keys():
            if point_id in self.points:
                self.points[point_id].remove_observation(camera_id)
        
        # Remove the camera
        del self.cameras[camera_id]
        return True
    
    def update_point_position(self, point_id: int, position: np.ndarray) -> bool:
        """Update the position of a 3D point.
        
        Args:
            point_id: ID of the point to update
            position: New 3D position
            
        Returns:
            True if position was updated successfully, False otherwise
        """
        if point_id not in self.points:
            return False
        
        self.points[point_id].position = position
        return True
    
    def update_camera_params(self, camera_id: int, R: np.ndarray, c: np.ndarray) -> bool:
        """Update the extrinsic parameters of a camera.
        
        Args:
            camera_id: ID of the camera to update
            R: New rotation matrix (camera-to-world)
            c: New camera center in world coordinates
            
        Returns:
            True if parameters were updated successfully, False otherwise
        """
        if camera_id not in self.cameras:
            return False
        
        self.cameras[camera_id].R = R
        self.cameras[camera_id].c = c
        return True
    
    def get_valid_points(self) -> List[Point3D]:
        """Get all valid points in the reconstruction.
        
        Returns:
            List of valid Point3D objects
        """
        return [p for p in self.points.values() if p.is_valid]
    
    def get_points_with_min_observations(self, min_observations: int = 2) -> List[Point3D]:
        """Get points with at least the specified number of observations.
        
        Args:
            min_observations: Minimum number of required observations
            
        Returns:
            List of Point3D objects with sufficient observations
        """
        return [p for p in self.points.values() 
                if p.is_valid and p.observation_count() >= min_observations]
    
    def get_all_point_arrays(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], 
                                    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Get all point arrays (positions, covariances, colors, alphas, quaternions, scales) at once.
        
        Returns:
            Tuple of:
            - Nx3 array of point positions
            - Nx3x3 array of covariance matrices or None
            - Nx3 array of RGB colors or None
            - N array of alpha values or None
            - Nx4 array of quaternions or None
            - Nx3 array of scales or None
        """
        valid_points = self.get_valid_points()
        
        # Positions are mandatory
        positions = np.array([p.position for p in valid_points])
        
        # Optional attributes - only create arrays if all points have the attribute
        has_covariance = all(p.covariance is not None for p in valid_points)
        has_color = all(p.color is not None for p in valid_points)
        has_alpha = all(p.alpha is not None for p in valid_points)
        has_quaternion = all(p.quaternion is not None for p in valid_points)
        has_scale = all(p.scale is not None for p in valid_points)
        
        # Create arrays for attributes that all points have
        covariances = np.array([p.covariance for p in valid_points]) if has_covariance else None
        colors = np.array([p.color for p in valid_points]) if has_color else None
        alphas = np.array([p.alpha for p in valid_points]) if has_alpha else None
        quaternions = np.array([p.quaternion for p in valid_points]) if has_quaternion else None
        scales = np.array([p.scale for p in valid_points]) if has_scale else None
        
        return positions, covariances, colors, alphas, quaternions, scales
    
    def get_all_points_array(self) -> np.ndarray:
        """Get positions of all valid points as a numpy array.
        
        Returns:
            Nx3 array of point positions
        """
        valid_points = self.get_valid_points()
        return np.array([p.position for p in valid_points])
    
    def get_all_covariances_array(self) -> np.ndarray:
        """Get covariances of all valid points as a numpy array.
        
        Returns:
            Nx3x3 array of covariance matrices
        """
        valid_points = self.get_valid_points()
        return np.array([p.covariance for p in valid_points if p.covariance is not None])
    
    def get_all_colors_array(self) -> np.ndarray:
        """Get colors of all valid points as a numpy array.
        
        Returns:
            Nx3 array of RGB colors
        """
        valid_points = self.get_valid_points()
        return np.array([p.color for p in valid_points if p.color is not None])
    
    def get_all_alphas_array(self) -> np.ndarray:
        """Get alpha values of all valid points as a numpy array.
        
        Returns:
            N array of alpha values
        """
        valid_points = self.get_valid_points()
        return np.array([p.alpha for p in valid_points if p.alpha is not None])
    
    def get_camera_parameters_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get extrinsic parameters of all cameras in world-to-camera format.
        
        This format is consistent with the existing system.
        
        Returns:
            List of (R_wtc, t_wtc) tuples for each camera
        """
        camera_params = []
        for camera_id in sorted(self.cameras.keys()):
            camera = self.cameras[camera_id]
            R_wtc, t_wtc = camera.world_to_camera_transform()
            camera_params.append((R_wtc, t_wtc))
        return camera_params
    
    def get_camera_to_world_list(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get extrinsic parameters of all cameras in camera-to-world format.
        
        Returns:
            List of (R_ctw, c) tuples for each camera
        """
        camera_params = []
        for camera_id in sorted(self.cameras.keys()):
            camera = self.cameras[camera_id]
            camera_params.append((camera.R, camera.c))
        return camera_params
    
    def get_intrinsics_list(self) -> List[np.ndarray]:
        """Get intrinsic matrices of all cameras.
        
        Returns:
            List of K matrices
        """
        return [self.cameras[camera_id].K for camera_id in sorted(self.cameras.keys())]
    
    def get_image_names(self) -> List[str]:
        """Get image names of all cameras.
        
        Returns:
            List of image names
        """
        return [self.cameras[camera_id].image_name for camera_id in sorted(self.cameras.keys())]
    
    def build_observation_map(self) -> Dict[int, Dict[int, np.ndarray]]:
        """Build an observation map for compatibility with existing code.
        
        Returns:
            Dict mapping point_id -> {camera_id -> 2D point}
        """
        observation_map = {}
        
        for point_id, point in self.points.items():
            if not point.is_valid:
                continue
                
            if point.observations:
                observation_map[point_id] = {}
                for camera_id, point_2d in point.observations.items():
                    observation_map[point_id][camera_id] = point_2d
                    
        return observation_map
    
    def convert_to_match_points_2d(self) -> List[List[Tuple[int, np.ndarray]]]:
        """Convert observations to the match_points_2d format for bundle adjustment.
        
        Returns:
            List of lists, where each inner list contains (point_id, [x, y]) pairs for one camera
        """
        match_points_2d = [[] for _ in range(len(self.cameras))]
        
        camera_id_to_idx = {camera_id: idx for idx, camera_id in enumerate(sorted(self.cameras.keys()))}
        
        for point_id, point in self.points.items():
            if not point.is_valid:
                continue
                
            for camera_id, point_2d in point.observations.items():
                if camera_id in camera_id_to_idx:
                    cam_idx = camera_id_to_idx[camera_id]
                    match_points_2d[cam_idx].append((point_id, point_2d))
        
        return match_points_2d
    
    def convert_from_legacy_format(self, 
                                  points_3d: np.ndarray,
                                  camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
                                  match_points_2d: List[List[Tuple[int, np.ndarray]]],
                                  intrinsics_list: List[np.ndarray],
                                  image_names: Optional[List[str]] = None,
                                  covariances_3d: Optional[np.ndarray] = None,
                                  colors_3d: Optional[np.ndarray] = None,
                                  alphas_3d: Optional[np.ndarray] = None,
                                  quaternions: Optional[np.ndarray] = None,
                                  scales: Optional[np.ndarray] = None) -> None:
        """Convert from legacy format to the new persistent ID format.
        
        Args:
            points_3d: Nx3 array of 3D point positions
            camera_params_list: List of (R_wtc, t_wtc) tuples for each camera
            match_points_2d: For each camera, list of (point_idx, [x, y]) observations
            intrinsics_list: List of K matrices
            image_names: List of image names (optional)
            covariances_3d: Nx3x3 array of covariance matrices (optional)
            colors_3d: Nx3 array of RGB colors (optional)
            alphas_3d: N array of alpha values (optional)
            quaternions: Nx4 array of quaternions (optional)
            scales: Nx3 array of scale factors (optional)
        """
        # Clear existing data
        self.points = {}
        self.cameras = {}
        self.next_point_id = 0
        self.next_camera_id = 0
        
        # Add points
        point_idx_to_id = {}
        for i in range(len(points_3d)):
            point_position = points_3d[i]
            point_covariance = covariances_3d[i] if covariances_3d is not None and i < len(covariances_3d) else None
            point_color = colors_3d[i] if colors_3d is not None and i < len(colors_3d) else None
            point_alpha = alphas_3d[i] if alphas_3d is not None and i < len(alphas_3d) else None
            point_quaternion = quaternions[i] if quaternions is not None and i < len(quaternions) else None
            point_scale = scales[i] if scales is not None and i < len(scales) else None
            
            point_id = self.add_point(
                position=point_position,
                covariance=point_covariance,
                color=point_color,
                alpha=point_alpha,
                quaternion=point_quaternion,
                scale=point_scale
            )
            
            point_idx_to_id[i] = point_id
        
        # Add cameras
        for i, (R_wtc, t_wtc) in enumerate(camera_params_list):
            # Convert world-to-camera to camera-to-world
            R_ctw = R_wtc.T
            c = -R_ctw @ t_wtc
            
            K = intrinsics_list[i] if i < len(intrinsics_list) else np.eye(3)
            
            img_name = None
            if image_names is not None and i < len(image_names):
                img_name = image_names[i]
                
            camera_id = self.add_camera(
                R=R_ctw,
                c=c,
                K=K,
                image_name=img_name
            )
            
            # Add observations
            if i < len(match_points_2d):
                for point_idx, point_2d in match_points_2d[i]:
                    if point_idx in point_idx_to_id:
                        point_id = point_idx_to_id[point_idx]
                        self.add_observation(point_id, camera_id, point_2d)
    
    def convert_to_legacy_format(self) -> Tuple:
        """Convert to legacy format for compatibility with existing code.
        
        Returns:
            Tuple of (points_3d, camera_params_list, match_points_2d, intrinsics_list, image_names,
                     covariances_3d, colors_3d, alphas_3d, quaternions, scales)
        """
        # Get valid points
        valid_points = self.get_valid_points()
        
        # Create mapping from point ID to index
        point_id_to_idx = {point.id: i for i, point in enumerate(valid_points)}
        
        # Extract point arrays
        points_3d = np.array([point.position for point in valid_points])
        
        covariances_3d = None
        if any(point.covariance is not None for point in valid_points):
            covariances_3d = np.array([
                point.covariance if point.covariance is not None else np.eye(3)
                for point in valid_points
            ])
            
        colors_3d = None
        if any(point.color is not None for point in valid_points):
            colors_3d = np.array([
                point.color if point.color is not None else np.zeros(3)
                for point in valid_points
            ])
            
        alphas_3d = None
        if any(point.alpha is not None for point in valid_points):
            alphas_3d = np.array([
                point.alpha if point.alpha is not None else 1.0
                for point in valid_points
            ])
            
        quaternions = None
        if any(point.quaternion is not None for point in valid_points):
            quaternions = np.array([
                point.quaternion if point.quaternion is not None else np.array([1, 0, 0, 0])
                for point in valid_points
            ])
            
        scales = None
        if any(point.scale is not None for point in valid_points):
            scales = np.array([
                point.scale if point.scale is not None else np.ones(3)
                for point in valid_points
            ])
        
        # Extract camera parameters
        camera_params_list = self.get_camera_parameters_list()
        intrinsics_list = self.get_intrinsics_list()
        image_names = self.get_image_names()
        
        # Create match_points_2d with remapped indices
        match_points_2d = []
        
        for camera_id in sorted(self.cameras.keys()):
            camera = self.cameras[camera_id]
            camera_observations = []
            
            for point_id, point_2d in camera.observations.items():
                if point_id in point_id_to_idx:
                    point_idx = point_id_to_idx[point_id]
                    camera_observations.append((point_idx, point_2d))
                    
            match_points_2d.append(camera_observations)
            
        return (
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
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the point ID manager
        """
        return {
            "points": {str(k): v.to_dict() for k, v in self.points.items()},
            "cameras": {str(k): v.to_dict() for k, v in self.cameras.items()},
            "next_point_id": self.next_point_id,
            "next_camera_id": self.next_camera_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PointIDManager':
        """Create a PointIDManager instance from a dictionary.
        
        Args:
            data: Dictionary containing manager data
            
        Returns:
            New PointIDManager instance
        """
        manager = cls()
        
        manager.next_point_id = data["next_point_id"]
        manager.next_camera_id = data["next_camera_id"]
        
        for k, v in data["points"].items():
            point_id = int(k)
            manager.points[point_id] = Point3D.from_dict(v)
            
        for k, v in data["cameras"].items():
            camera_id = int(k)
            manager.cameras[camera_id] = Camera.from_dict(v)
            
        return manager