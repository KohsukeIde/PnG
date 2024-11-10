# Author: True Price <jtprice at cs.unc.edu>

import numpy as np
from scipy.optimize import root

# -------------------------------------------------------------------------------
#
# camera distortion functions for arrays of size (..., 2)
#
# -------------------------------------------------------------------------------


def simple_radial_distortion(camera, x):
    """Apply simple radial distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    return x * (1.0 + camera.k1 * np.square(x).sum(axis=-1, keepdims=True))


def radial_distortion(camera, x):
    """Apply radial distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    r_sq = np.square(x).sum(axis=-1, keepdims=True)
    return x * (1.0 + r_sq * (camera.k1 + camera.k2 * r_sq))


def opencv_distortion(camera, x):
    """Apply OpenCV-style distortion to points.

    Args:
        camera: Camera object containing distortion parameters
        x: np.ndarray of shape (..., 2) containing points to distort

    Returns:
        np.ndarray: Distorted points
    """
    x_sq = np.square(x)
    xy = np.prod(x, axis=-1, keepdims=True)
    r_sq = x_sq.sum(axis=-1, keepdims=True)
    y_sq = x_sq[..., 1:]  # Get y-squared component

    return x * (1.0 + r_sq * (camera.k1 + camera.k2 * r_sq)) + np.concatenate(
        (
            2.0 * camera.p1 * xy + camera.p2 * (r_sq + 2.0 * x_sq[..., :1]),
            camera.p1 * (r_sq + 2.0 * y_sq) + 2.0 * camera.p2 * xy,
        ),
        axis=-1,
    )


# -------------------------------------------------------------------------------
#
# Camera
#
# -------------------------------------------------------------------------------


class Camera:
    """A class representing a camera with various distortion models. Based off of gsplat Camera class."""

    @staticmethod
    def get_num_params(type_):
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
    def get_name_from_type(type_):
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

    def __init__(self, type_, width_, height_, params):
        """Initialize a camera instance.

        Args:
            type_: int or str, camera type identifier
            width_: int, image width
            height_: int, image height
            params: array-like, camera parameters
        """
        self.width = width_
        self.height = height_

        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            self.fx, self.cx, self.cy = params
            self.fy = self.fx
            self.distortion_func = None
            self.camera_type = 0

        elif type_ == 1 or type_ == "PINHOLE":
            self.fx, self.fy, self.cx, self.cy = params
            self.distortion_func = None
            self.camera_type = 1

        elif type_ == 2 or type_ == "SIMPLE_RADIAL":
            self.fx, self.cx, self.cy, self.k1 = params
            self.fy = self.fx
            self.distortion_func = simple_radial_distortion
            self.camera_type = 2

        elif type_ == 3 or type_ == "RADIAL":
            self.fx, self.cx, self.cy, self.k1, self.k2 = params
            self.fy = self.fx
            self.distortion_func = radial_distortion
            self.camera_type = 3

        elif type_ == 4 or type_ == "OPENCV":
            self.fx, self.fy, self.cx, self.cy = params[:4]
            self.k1, self.k2, self.p1, self.p2 = params[4:]
            self.distortion_func = opencv_distortion
            self.camera_type = 4

        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            self.fx, self.fy, self.cx, self.cy = params[:4]
            self.k1, self.k2, self.k3, self.k4 = params[4:]

            def fn(camera, x):
                raise Exception("Fisheye distortion not supported")

            self.distortion_func = fn
            self.camera_type = 5

        else:
            raise Exception("Camera type not supported")

    def __str__(self):
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

    def get_params(self):
        """Get the camera parameters in COLMAP format.

        Returns:
            np.ndarray: Array of camera parameters
        """
        if self.camera_type == 0:
            return np.array((self.fx, self.cx, self.cy))
        if self.camera_type == 1:
            return np.array((self.fx, self.fy, self.cx, self.cy))
        if self.camera_type == 2:
            return np.array((self.fx, self.cx, self.cy, self.k1))
        if self.camera_type == 3:
            return np.array((self.fx, self.cx, self.cy, self.k1, self.k2))
        if self.camera_type == 4:
            return np.array(
                (self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2)
            )
        if self.camera_type == 5:
            return np.array(
                (self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.k3, self.k4)
            )

    def get_camera_matrix(self):
        """Get the camera intrinsic matrix.

        Returns:
            np.ndarray: 3x3 camera intrinsic matrix
        """
        return np.array(((self.fx, 0, self.cx), (0, self.fy, self.cy), (0, 0, 1)))

    def get_inverse_camera_matrix(self):
        """Get the inverse of the camera intrinsic matrix.

        Returns:
            np.ndarray: 3x3 inverse camera intrinsic matrix
        """
        return np.array(
            (
                (1.0 / self.fx, 0, -self.cx / self.fx),
                (0, 1.0 / self.fy, -self.cy / self.fy),
                (0, 0, 1),
            )
        )

    @property
    def k(self):
        """Get the camera matrix.

        Returns:
            np.ndarray: Camera matrix
        """
        return self.get_camera_matrix()

    @property
    def k_inv(self):
        """Get the inverse camera matrix.

        Returns:
            np.ndarray: Inverse camera matrix
        """
        return self.get_inverse_camera_matrix()

    def get_inv_camera_matrix(self):
        """Get the inverse camera matrix (deprecated).

        Returns:
            np.ndarray: 3x3 inverse camera matrix
        """
        inv_fx, inv_fy = 1.0 / self.fx, 1.0 / self.fy
        return np.array(
            ((inv_fx, 0, -inv_fx * self.cx), (0, inv_fy, -inv_fy * self.cy), (0, 0, 1))
        )

    def get_image_grid(self):
        """Get an (x, y) pixel coordinate grid for this camera.

        Returns:
            tuple: Arrays of x and y coordinates
        """
        xmin = (0.5 - self.cx) / self.fx
        xmax = (self.width - 0.5 - self.cx) / self.fx
        ymin = (0.5 - self.cy) / self.fy
        ymax = (self.height - 0.5 - self.cy) / self.fy
        return np.meshgrid(
            np.linspace(xmin, xmax, self.width), np.linspace(ymin, ymax, self.height)
        )

    def distort_points(self, x, normalized=True, denormalize=True):
        """Apply distortion to points.

        Args:
            x: np.ndarray, points to distort
            normalized: bool, whether points are normalized
            denormalize: bool, whether to denormalize output

        Returns:
            np.ndarray: Distorted points
        """
        x = np.atleast_2d(x)

        if not normalized:
            x -= np.array([[self.cx, self.cy]])
            x /= np.array([[self.fx, self.fy]])

        if self.distortion_func is not None:
            x = self.distortion_func(self, x)

        if denormalize:
            x *= np.array([[self.fx, self.fy]])
            x += np.array([[self.cx, self.cy]])

        return x

    def undistort_points(self, x, normalized=False, denormalize=True):
        """Perform distortion correction on points.

        Args:
            x: np.ndarray, points to undistort
            normalized: bool, whether points are normalized
            denormalize: bool, whether to denormalize output

        Returns:
            np.ndarray: Undistorted points
        """
        x = np.atleast_2d(x)

        if not normalized:
            x = x - np.array([self.cx, self.cy])  # creates a copy
            x /= np.array([self.fx, self.fy])

        if self.distortion_func is not None:

            def objective(xu):
                return (x - self.distortion_func(self, xu.reshape(*x.shape))).ravel()

            xu = root(objective, x).x.reshape(*x.shape)
        else:
            xu = x

        if denormalize:
            xu *= np.array([[self.fx, self.fy]])
            xu += np.array([[self.cx, self.cy]])

        return xu
