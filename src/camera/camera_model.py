# src/camera/camera_model.py

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from src.camera.colmap_camera_utils import Camera


class CameraModel:
    """A class representing a camera model with intrinsic and extrinsic parameters. Based off of gsplat CameraModel class."""

    def __init__(self, camera: Camera, image_id: int, images_data: dict):
        """Initialize a camera model with given parameters.

        Args:
            camera: Camera, an instance of the existing Camera class
            image_id: int, ID of the image corresponding to this camera
            images_data: dict, external parameter information for the image (obtained from COLMAP's images.bin or images.txt)
        """
        self.camera = camera
        self.image_id = image_id
        self.images_data = images_data

        # Camera intrinsic parameter matrix K
        self.K = self.camera.get_camera_matrix()
        self.K_inv = self.camera.get_inverse_camera_matrix()

        # Camera extrinsic parameters (rotation matrix R_wc, translation vector t_wc)
        self.R_wc, self.t_wc = self.get_extrinsics()

        # Transformation from camera coordinate system to world coordinate system
        self.R_cw = self.R_wc.T
        self.t_cw = -self.R_wc.T @ self.t_wc

        # Projection matrix P
        self.P = self.get_projection_matrix()
        # print(f"Camera {image_id} Projection Matrix P:\n{self.P}")

    def get_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the camera's extrinsic parameters (rotation matrix and translation vector).

        Returns:
            r_wc: np.ndarray, rotation matrix (3x3)
            t_wc: np.ndarray, translation vector (3,)
        """
        # Get quaternion and camera position from images_data
        image_info = self.images_data[self.image_id]
        qw, qx, qy, qz = (
            image_info["qw"],
            image_info["qx"],
            image_info["qy"],
            image_info["qz"],
        )
        tx, ty, tz = image_info["tx"], image_info["ty"], image_info["tz"]

        # Convert quaternion to rotation matrix
        # Note: In scipy, the quaternion order is (qx, qy, qz, qw)
        rotation = Rotation.from_quat([qx, qy, qz, qw])
        r_wc = rotation.as_matrix()

        # Translation vector
        t_wc = np.array([tx, ty, tz])

        return r_wc, t_wc

    def get_projection_matrix(self) -> np.ndarray:
        """Get the camera's projection matrix.

        Returns:
            p: np.ndarray, projection matrix (3x4)
        """
        p = self.K @ np.hstack((self.R_cw, self.t_cw.reshape(-1, 1)))
        return np.array(p, dtype=float)

    def get_position(self) -> np.ndarray:
        """Get the camera's position in world coordinates.

        Returns:
            position: np.ndarray, camera position (3,)
        """
        # Calculate the position of the camera center
        position = -self.R_wc.T @ self.t_wc
        print(f"{position=}")
        return np.array(position, dtype=float)

    def undistort_points(self, x: np.ndarray) -> np.ndarray:
        """Perform distortion correction.

        Args:
            x: np.ndarray, pixel coordinates with distortion (N, 2)

        Returns:
            xu: np.ndarray, pixel coordinates after distortion correction (N, 2)
        """
        return self.camera.undistort_points(x, normalized=False, denormalize=True)

    def distort_points(self, x: np.ndarray) -> np.ndarray:
        """Apply distortion.

        Args:
            x: np.ndarray, pixel coordinates without distortion (N, 2)

        Returns:
            xd: np.ndarray, pixel coordinates after applying distortion (N, 2)
        """
        return self.camera.distort_points(x, normalized=False, denormalize=True)
