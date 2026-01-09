import os
import struct
from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from src.camera.colmap_camera_utils import Camera


def load_cameras_from_colmap(colmap_dir: str) -> OrderedDict[int, Camera]:
    """Load camera parameters from COLMAP directory.

    Args:
        colmap_dir: Path to COLMAP directory containing cameras.bin/txt

    Returns:
        OrderedDict: Dictionary mapping camera_id to Camera object
    """
    cameras: OrderedDict[int, Camera] = OrderedDict()
    cameras_file = os.path.join(colmap_dir, "cameras.bin")
    if not os.path.exists(cameras_file):
        cameras_file = os.path.join(colmap_dir, "cameras.txt")
        if not os.path.exists(cameras_file):
            raise FileNotFoundError("No cameras file found in COLMAP directory.")
    if cameras_file.endswith(".bin"):
        cameras = OrderedDict(read_cameras_binary(cameras_file))
    else:
        cameras = OrderedDict(read_cameras_text(cameras_file))
    return cameras


def load_images_from_colmap(colmap_dir: str) -> OrderedDict[int, Dict[str, Any]]:
    """Load image parameters from COLMAP directory.

    Args:
        colmap_dir: Path to COLMAP directory containing images.bin/txt

    Returns:
        OrderedDict: Dictionary mapping image_id to image parameters
    """
    images: OrderedDict[int, Dict[str, Any]] = OrderedDict()
    images_file = os.path.join(colmap_dir, "images.bin")
    if not os.path.exists(images_file):
        images_file = os.path.join(colmap_dir, "images.txt")
        if not os.path.exists(images_file):
            raise FileNotFoundError("No images file found in COLMAP directory.")
    if images_file.endswith(".bin"):
        images = OrderedDict(read_images_binary(images_file))
    else:
        images = OrderedDict(read_images_text(images_file))
    return images


def read_cameras_text(path: str) -> dict:
    """Read camera parameters from COLMAP text file.

    Args:
        path: Path to cameras.txt file

    Returns:
        dict: Dictionary mapping camera_id to Camera object
    """
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            if len(elems) < 4:
                continue
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array([float(p) for p in elems[4:]])
            cameras[camera_id] = Camera(model, width, height, params)
    return cameras


def read_cameras_binary(path_to_model_file: str) -> dict:
    """Read camera parameters from COLMAP binary file.

    Args:
        path_to_model_file: Path to cameras.bin file

    Returns:
        dict: Dictionary mapping camera_id to Camera object
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack("<IIQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            model_name = Camera.get_name_from_type(model_id)
            num_params = Camera.get_num_params(model_id)
            params = struct.unpack("<" + "d" * num_params, fid.read(8 * num_params))
            cameras[camera_id] = Camera(model_name, width, height, params)
    return cameras


def read_images_text(path: str) -> dict:
    """Read image parameters from COLMAP text file.

    Args:
        path: Path to images.txt file

    Returns:
        dict: Dictionary mapping image_id to image parameters
    """
    images = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            if len(elems) < 9:
                continue
            image_id = int(elems[0])
            qw, qx, qy, qz = map(float, elems[1:5])
            tx, ty, tz = map(float, elems[5:8])
            camera_id = int(elems[8])
            image_name = elems[9]
            images[image_id] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": image_name,
            }
            # Skip the 2D points
            f.readline()
    return images


def read_images_binary(path_to_model_file: str) -> dict:
    """Read image parameters from COLMAP binary file.

    Args:
        path_to_model_file: Path to images.bin file

    Returns:
        dict: Dictionary mapping image_id to image parameters
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<Idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qw = binary_image_properties[1]
            qx = binary_image_properties[2]
            qy = binary_image_properties[3]
            qz = binary_image_properties[4]
            tx = binary_image_properties[5]
            ty = binary_image_properties[6]
            tz = binary_image_properties[7]
            camera_id = binary_image_properties[8]

            image_name = ""
            while True:
                current_char = fid.read(1).decode("utf-8")
                if current_char == "\x00":
                    break
                image_name += current_char
            images[image_id] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": image_name,
            }
            # Skip 2D points data
            num_points_2d = struct.unpack("<Q", fid.read(8))[0]
            fid.seek(num_points_2d * (8 * 2 + 8), os.SEEK_CUR)
    return images


def read_images_with_points2d(path_to_model_file: str) -> dict:
    """Read image parameters from COLMAP binary file including 2D points.

    Args:
        path_to_model_file: Path to images.bin file

    Returns:
        dict: Dictionary mapping image_id to image parameters with 2D points
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack("<Idddddddi", fid.read(64))
            image_id = binary_image_properties[0]
            qw = binary_image_properties[1]
            qx = binary_image_properties[2]
            qy = binary_image_properties[3]
            qz = binary_image_properties[4]
            tx = binary_image_properties[5]
            ty = binary_image_properties[6]
            tz = binary_image_properties[7]
            camera_id = binary_image_properties[8]

            image_name = ""
            while True:
                current_char = fid.read(1).decode("utf-8")
                if current_char == "\x00":
                    break
                image_name += current_char

            # Read 2D points
            num_points_2d = struct.unpack("<Q", fid.read(8))[0]
            points2d = []
            for _ in range(num_points_2d):
                x, y = struct.unpack("<dd", fid.read(16))
                point3d_id = struct.unpack("<q", fid.read(8))[0]  # signed
                points2d.append({
                    "x": x,
                    "y": y,
                    "point3d_id": point3d_id  # -1 if not triangulated
                })

            images[image_id] = {
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "camera_id": camera_id,
                "name": image_name,
                "points2d": points2d,
            }
    return images


def get_corresponding_points(images: dict, image_name1: str, image_name2: str) -> tuple:
    """Get 2D points that are observed in both images (same 3D point).

    Args:
        images: Dictionary from read_images_with_points2d
        image_name1: Name of first image
        image_name2: Name of second image

    Returns:
        tuple: (pts1, pts2) where each is (N, 2) array of 2D coordinates
    """
    # Find image data by name
    img1_data = None
    img2_data = None
    for img_data in images.values():
        if img_data["name"] == image_name1:
            img1_data = img_data
        if img_data["name"] == image_name2:
            img2_data = img_data

    if img1_data is None or img2_data is None:
        raise ValueError(f"Images not found: {image_name1}, {image_name2}")

    # Build point3d_id -> (x, y) mapping for each image
    pts1_by_id = {}
    for pt in img1_data["points2d"]:
        if pt["point3d_id"] != -1:
            pts1_by_id[pt["point3d_id"]] = (pt["x"], pt["y"])

    pts2_by_id = {}
    for pt in img2_data["points2d"]:
        if pt["point3d_id"] != -1:
            pts2_by_id[pt["point3d_id"]] = (pt["x"], pt["y"])

    # Find common point3d_ids
    common_ids = set(pts1_by_id.keys()) & set(pts2_by_id.keys())

    pts1 = []
    pts2 = []
    for pid in sorted(common_ids):
        pts1.append(pts1_by_id[pid])
        pts2.append(pts2_by_id[pid])

    return np.array(pts1), np.array(pts2)


def quaternion_to_rotation_matrix(
    qw: float, qx: float, qy: float, qz: float
) -> np.ndarray:
    """Convert quaternion to rotation matrix.

    Args:
        qw: Scalar component of quaternion
        qx: X component of quaternion
        qy: Y component of quaternion
        qz: Z component of quaternion

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )
    return rot_matrix
