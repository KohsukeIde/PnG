import os
from collections import OrderedDict
import numpy as np
from src.camera.colmap_camera_utils import Camera

def load_cameras_from_colmap(colmap_dir: str):
    cameras = OrderedDict()
    cameras_file = os.path.join(colmap_dir, 'cameras.bin')
    if not os.path.exists(cameras_file):
        cameras_file = os.path.join(colmap_dir, 'cameras.txt')
        if not os.path.exists(cameras_file):
            raise FileNotFoundError('No cameras file found in COLMAP directory.')
    if cameras_file.endswith('.bin'):
        cameras = read_cameras_binary(cameras_file)
    else:
        cameras = read_cameras_text(cameras_file)
    return cameras

def load_images_from_colmap(colmap_dir: str):
    images = OrderedDict()
    images_file = os.path.join(colmap_dir, 'images.bin')
    if not os.path.exists(images_file):
        images_file = os.path.join(colmap_dir, 'images.txt')
        if not os.path.exists(images_file):
            raise FileNotFoundError('No images file found in COLMAP directory.')
    if images_file.endswith('.bin'):
        images = read_images_binary(images_file)
    else:
        images = read_images_text(images_file)
    return images

def read_cameras_text(path):
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
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

def read_cameras_binary(path_to_model_file):
    import struct
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_cameras):
            camera_properties = struct.unpack('<IIQQ', fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            model_name = Camera.GetNameFromType(model_id)
            num_params = Camera.GetNumParams(model_id)
            params = struct.unpack('<' + 'd' * num_params, fid.read(8 * num_params))
            cameras[camera_id] = Camera(model_name, width, height, params)
    return cameras

def read_images_text(path):
    images = {}
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('#'):
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
                'qw': qw,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'tx': tx,
                'ty': ty,
                'tz': tz,
                'camera_id': camera_id,
                'name': image_name,
            }
            # Skip the 2D points
            f.readline()
    return images

def read_images_binary(path_to_model_file):
    import struct
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_reg_images):
            binary_image_properties = struct.unpack('<Idddddddi', fid.read(64))
            image_id = binary_image_properties[0]
            qw = binary_image_properties[1]
            qx = binary_image_properties[2]
            qy = binary_image_properties[3]
            qz = binary_image_properties[4]
            tx = binary_image_properties[5]
            ty = binary_image_properties[6]
            tz = binary_image_properties[7]
            camera_id = binary_image_properties[8]
            # Read image name
            image_name = ''
            while True:
                current_char = fid.read(1).decode('utf-8')
                if current_char == '\x00':
                    break
                image_name += current_char
            images[image_id] = {
                'qw': qw,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'tx': tx,
                'ty': ty,
                'tz': tz,
                'camera_id': camera_id,
                'name': image_name,
            }
            # Skip 2D points data
            num_points2D = struct.unpack('<Q', fid.read(8))[0]
            fid.seek(num_points2D * (8 * 2 + 8), os.SEEK_CUR)  # Each point: x, y (double), point3D_id (uint64)
    return images

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    R = np.array([
        [1.0 - q[2, 2] - q[3, 3],       q[1, 2] - q[3, 0],       q[1, 3] + q[2, 0]],
        [      q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3],       q[2, 3] - q[1, 0]],
        [      q[1, 3] - q[2, 0],       q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]
    ])
    return R
