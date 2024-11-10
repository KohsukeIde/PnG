# Author: True Price <jtprice at cs.unc.edu>

from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import root

# -------------------------------------------------------------------------------
#
# camera distortion functions for arrays of size (..., 2)
#
# -------------------------------------------------------------------------------


def simple_radial_distortion(camera: "Camera", x: np.ndarray) -> np.ndarray:
    """Apply simple radial distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    if camera.k1 is None:
        raise ValueError("k1 cannot be None for simple_radial_distortion")
    k1 = float(camera.k1)
    return np.ndarray(x * (1.0 + k1 * np.square(x).sum(axis=-1, keepdims=True)))


def radial_distortion(camera: "Camera", x: np.ndarray) -> np.ndarray:
    """Apply radial distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    if camera.k1 is None or camera.k2 is None:
        raise ValueError("k1 and k2 cannot be None for radial_distortion")
    k1 = float(camera.k1)
    k2 = float(camera.k2)
    r_sq = np.square(x).sum(axis=-1, keepdims=True)
    return np.ndarray(x * (1.0 + r_sq * (k1 + k2 * r_sq)))


def opencv_distortion(camera: "Camera", x: np.ndarray) -> np.ndarray:
    """Apply OpenCV-style distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    if camera.k1 is None or camera.k2 is None:
        raise ValueError("k1 and k2 cannot be None for opencv_distortion")
    k1 = float(camera.k1)
    k2 = float(camera.k2)
    p1 = float(camera.p1) if camera.p1 is not None else 0.0
    p2 = float(camera.p2) if camera.p2 is not None else 0.0

    x_sq = np.square(x)
    xy = np.prod(x, axis=-1, keepdims=True)
    r_sq = x_sq.sum(axis=-1, keepdims=True)
    y_sq = x_sq[..., 1:]  # Get y-squared component

    distorted = x * (1.0 + r_sq * (k1 + k2 * r_sq))
    distort_xy = 2.0 * p1 * xy + p2 * (r_sq + 2.0 * x_sq[..., :1])
    distort_yx = p1 * (r_sq + 2.0 * y_sq) + 2.0 * p2 * xy
    return np.ndarray(distorted + np.concatenate((distort_xy, distort_yx), axis=-1))


# -------------------------------------------------------------------------------
#
# Camera
#
# -------------------------------------------------------------------------------


class Camera:
    """A class representing a camera with various distortion models. Based off of gsplat Camera class."""

    fx: float
    fy: float
    cx: float
    cy: float
    k1: Optional[float]
    k2: Optional[float]
    p1: Optional[float]
    p2: Optional[float]
    k3: Optional[float]
    k4: Optional[float]
    width: int
    height: int
    distortion_func: Optional[Callable[["Camera", np.ndarray], np.ndarray]]
    camera_type: int

    @staticmethod
    def get_num_params(type_: Union[int, str]) -> int:
        """Get the number of parameters for a given camera type.

        Args:
            type_: int or str, camera type identifier

        Returns:
            int: Number of parameters for the camera type
        """
        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            return 3
        if type_ == 1 or type_ == "PINHOLE":
            return 4
        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            return 4
        if type_ == 3 or type_ == "RADIAL":
            return 5
        if type_ == 4 or type_ == "OPENCV":
            return 8
        if type_ == 5 or type_ == "OPENCV_FISHEYE":
            return 8
        raise Exception("Camera type not supported")

    @staticmethod
    def get_name_from_type(type_: int) -> str:
        """Get the camera type name from its identifier.

        Args:
            type_: int, camera type identifier

        Returns:
            str: Name of the camera type
        """
        if type_ == 0:
            return "SIMPLE_PINHOLE"
        if type_ == 1:
            return "PINHOLE"
        if type_ == 2:
            return "SIMPLE_RADIAL"
        if type_ == 3:
            return "RADIAL"
        if type_ == 4:
            return "OPENCV"
        if type_ == 5:
            return "OPENCV_FISHEYE"
        raise Exception("Camera type not supported")

    def __init__(
        self,
        type_: Union[int, str],
        width_: int,
        height_: int,
        params: Union[np.ndarray, Tuple[float, ...]],
    ) -> None:
        """Initialize a camera instance.

        Args:
            type_: int or str, camera type identifier
            width_: int, image width
            height_: int, image height
            params: array-like, camera parameters
        """
        self.width: int = width_
        self.height: int = height_

        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            if len(params) != 3:
                raise ValueError("SIMPLE_PINHOLE requires 3 parameters.")
            self.fx, self.cx, self.cy = map(float, params)
            self.fy: float = self.fx
            self.distortion_func = None
            self.camera_type: int = 0
            self.k1 = self.k2 = self.p1 = self.p2 = self.k3 = self.k4 = None

        elif type_ == 1 or type_ == "PINHOLE":
            if len(params) != 4:
                raise ValueError("PINHOLE requires 4 parameters.")
            self.fx, self.fy, self.cx, self.cy = map(float, params)
            self.distortion_func = None
            self.camera_type = 1
            self.k1 = self.k2 = self.p1 = self.p2 = self.k3 = self.k4 = None

        elif type_ == 2 or type_ == "SIMPLE_RADIAL":
            if len(params) != 4:
                raise ValueError("SIMPLE_RADIAL requires 4 parameters.")
            self.fx, self.cx, self.cy, self.k1 = map(float, params)
            self.fy = self.fx
            self.distortion_func = simple_radial_distortion
            self.camera_type = 2
            self.k2 = self.p1 = self.p2 = self.k3 = self.k4 = None

        elif type_ == 3 or type_ == "RADIAL":
            if len(params) != 5:
                raise ValueError("RADIAL requires 5 parameters.")
            self.fx, self.cx, self.cy, self.k1, self.k2 = map(float, params)
            self.fy = self.fx
            self.distortion_func = radial_distortion
            self.camera_type = 3
            self.p1 = self.p2 = self.k3 = self.k4 = None

        elif type_ == 4 or type_ == "OPENCV":
            if len(params) != 8:
                raise ValueError("OPENCV requires 8 parameters.")
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2 = (
                map(float, params)
            )
            self.distortion_func = opencv_distortion
            self.camera_type = 4
            self.k3 = self.k4 = None

        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            if len(params) != 8:
                raise ValueError("OPENCV_FISHEYE requires 8 parameters.")
            self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4 = (
                map(float, params)
            )

            def fn(camera: "Camera", x: np.ndarray) -> np.ndarray:
                raise Exception("Fisheye distortion not supported")

            self.distortion_func = fn
            self.camera_type = 5

        else:
            raise Exception("Camera type not supported")

    def __str__(self) -> str:
        """Return a string representation of the camera.

        Returns:
            str: String representation of the camera parameters
        """
        s = self.get_name_from_type(self.camera_type) + " {} {} {}".format(
            self.width, self.height, self.fx
        )

        if self.camera_type in (1, 4):  # PINHOLE, OPENCV
            s += " {}".format(self.fy)

        s += " {} {}".format(self.cx, self.cy)

        if self.camera_type == 2:  # SIMPLE_RADIAL
            s += " {}".format(self.k1)

        elif self.camera_type == 3:  # RADIAL
            s += " {} {}".format(self.k1, self.k2)

        elif self.camera_type == 4:  # OPENCV
            s += " {} {} {} {}".format(self.k1, self.k2, self.p1, self.p2)

        elif self.camera_type == 5:  # OPENCV_FISHEYE
            s += " {} {} {} {}".format(self.k1, self.k2, self.k3, self.k4)

        return s

    def get_params(self) -> np.ndarray:
        """Get the camera parameters in COLMAP format.

        Returns:
            np.ndarray: Array of camera parameters
        """
        if self.camera_type == 0:
            return np.array([self.fx, self.cx, self.cy], dtype=float)
        if self.camera_type == 1:
            return np.array([self.fx, self.fy, self.cx, self.cy], dtype=float)
        if self.camera_type == 2:
            return np.array([self.fx, self.cx, self.cy, self.k1], dtype=float)
        if self.camera_type == 3:
            return np.array([self.fx, self.cx, self.cy, self.k1, self.k2], dtype=float)
        if self.camera_type == 4:
            return np.array(
                [
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    self.k1,
                    self.k2,
                    self.p1,
                    self.p2,
                ],
                dtype=float,
            )
        if self.camera_type == 5:
            return np.array(
                [
                    self.fx,
                    self.fy,
                    self.cx,
                    self.cy,
                    self.k1,
                    self.k2,
                    self.k3,
                    self.k4,
                ],
                dtype=float,
            )
        raise Exception("Camera type not supported")

    def get_camera_matrix(self) -> np.ndarray:
        """Get the camera intrinsic matrix.

        Returns:
            np.ndarray: 3x3 camera intrinsic matrix
        """
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    def get_inverse_camera_matrix(self) -> np.ndarray:
        """Get the inverse of the camera intrinsic matrix.

        Returns:
            np.ndarray: 3x3 inverse camera intrinsic matrix
        """
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @property
    def k(self) -> np.ndarray:
        """Get the camera matrix.

        Returns:
            np.ndarray: Camera matrix
        """
        return self.get_camera_matrix()

    @property
    def k_inv(self) -> np.ndarray:
        """Get the inverse camera matrix.

        Returns:
            np.ndarray: Inverse camera matrix
        """
        return self.get_inverse_camera_matrix()

    def get_inv_camera_matrix(self) -> np.ndarray:
        """Get the inverse camera matrix (deprecated).

        Returns:
            np.ndarray: 3x3 inverse camera matrix
        """
        inv_fx = 1.0 / self.fx
        inv_fy = 1.0 / self.fy
        return np.array(
            [
                [inv_fx, 0.0, -inv_fx * self.cx],
                [0.0, inv_fy, -inv_fy * self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def get_image_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get an (x, y) pixel coordinate grid for this camera.

        Returns:
            tuple: Arrays of x and y coordinates
        """
        xmin = (0.5 - self.cx) / self.fx
        xmax = (self.width - 0.5 - self.cx) / self.fx
        ymin = (0.5 - self.cy) / self.fy
        ymax = (self.height - 0.5 - self.cy) / self.fy
        x_grid = np.linspace(xmin, xmax, self.width, dtype=float)
        y_grid = np.linspace(ymin, ymax, self.height, dtype=float)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        return np.array(grid_x, dtype=float), np.array(grid_y, dtype=float)

    def distort_points(
        self, x: np.ndarray, normalized: bool = True, denormalize: bool = True
    ) -> np.ndarray:
        """Apply distortion to points.

        Args:
            x: np.ndarray, points to distort
            normalized: bool, whether points are normalized
            denormalize: bool, whether to denormalize output

        Returns:
            np.ndarray: Distorted points
        """
        x = np.array(x, dtype=float)

        if not normalized:
            x -= np.array([[self.cx, self.cy]], dtype=float)
            x /= np.array([[self.fx, self.fy]], dtype=float)

        if self.distortion_func is not None:
            x = np.array(self.distortion_func(self, x), dtype=float)

        if denormalize:
            x *= np.array([[self.fx, self.fy]], dtype=float)
            x += np.array([[self.cx, self.cy]], dtype=float)

        return np.array(x, dtype=float)

    def undistort_points(
        self, x: np.ndarray, normalized: bool = False, denormalize: bool = True
    ) -> np.ndarray:
        """Perform distortion correction on points.

        Args:
            x: np.ndarray, points to undistort
            normalized: bool, whether points are normalized
            denormalize: bool, whether to denormalize output

        Returns:
            np.ndarray: Undistorted points
        """
        x = np.array(x, dtype=float)

        if not normalized:
            x = x - np.array([self.cx, self.cy], dtype=float)  # creates a copy
            x /= np.array([self.fx, self.fy], dtype=float)

        xu = x  # Default value if no distortion
        if self.distortion_func is not None:
            distort_fn = self.distortion_func  # Local copy to satisfy mypy

            def objective(xu: np.ndarray) -> np.ndarray:
                return np.ndarray(
                    x - np.array(distort_fn(self, xu.reshape(*x.shape)), dtype=float)
                ).ravel()

            solution = root(objective, x)
            if solution.success:
                xu = np.array(solution.x.reshape(*x.shape), dtype=float)

        if denormalize:
            xu *= np.array([[self.fx, self.fy]], dtype=float)
            xu += np.array([[self.cx, self.cy]], dtype=float)

        return np.array(xu, dtype=float)
