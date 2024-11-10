import numpy as np

from src.camera.camera_model import CameraModel
from src.camera.colmap_camera_utils import Camera


def create_test_camera_data():
    """Create test camera and image data for testing.

    Returns:
        tuple: (Camera, dict) containing test camera and image data
    """
    # Create test camera data similar to what load_cameras_from_colmap returns
    camera = Camera(
        "SIMPLE_PINHOLE",  # model_name
        1000,  # width
        1000,  # height
        np.array([1000.0, 500.0, 500.0], dtype=np.float64),  # params: fx, cx, cy
    )

    # Create test image data similar to what load_images_from_colmap returns
    images_data = {
        1: {
            "name": "test_image.png",
            "camera_id": 1,
            "qw": 1.0,  # Identity rotation
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "tx": 0.0,  # Zero translation
            "ty": 0.0,
            "tz": 0.0,
        }
    }

    return camera, images_data


def test_camera_model_initialization():
    """Test initialization of CameraModel class."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    assert isinstance(camera_model, CameraModel)
    assert camera_model.K.shape == (3, 3)
    assert camera_model.R_wc.shape == (3, 3)
    assert camera_model.t_wc.shape == (3,)


def test_camera_model_initialization_invalid_input():
    """Test initialization with invalid image_id raises KeyError."""
    camera, images_data = create_test_camera_data()

    # Test with invalid image_id
    try:
        CameraModel(camera, image_id=999, images_data=images_data)
        raise AssertionError("KeyError was not raised")
    except KeyError:
        pass


def test_get_extrinsics():
    """Test getting extrinsic parameters from camera model."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    r_wc, t_wc = camera_model.get_extrinsics()
    assert r_wc.shape == (3, 3)
    assert t_wc.shape == (3,)
    # For identity quaternion, R should be identity matrix
    np.testing.assert_allclose(r_wc, np.eye(3))
    # For zero translation, t should be zero vector
    np.testing.assert_allclose(t_wc, np.zeros(3))


def test_get_position():
    """Test getting camera position in world coordinates."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    # For identity rotation and zero translation, camera position should be at origin
    position = camera_model.get_position()
    np.testing.assert_allclose(position, np.zeros(3))


def test_get_projection_matrix():
    """Test getting camera projection matrix."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    proj_matrix = camera_model.get_projection_matrix()
    assert proj_matrix.shape == (3, 4)

    # For identity rotation and zero translation, P should be [K|0]
    expected_proj = np.hstack((camera_model.K, np.zeros((3, 1))))
    np.testing.assert_allclose(proj_matrix, expected_proj)


def test_undistort_points():
    """Test undistortion of image points."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    points = np.array(
        [[500.0, 500.0], [600.0, 600.0], [400.0, 400.0]], dtype=np.float64
    )
    undistorted_points = camera_model.undistort_points(points)
    assert undistorted_points.shape == (3, 2)


def test_distort_points():
    """Test distortion of image points."""
    camera, images_data = create_test_camera_data()
    camera_model = CameraModel(camera, image_id=1, images_data=images_data)

    points = np.array(
        [[500.0, 500.0], [600.0, 600.0], [400.0, 400.0]], dtype=np.float64
    )
    distorted_points = camera_model.distort_points(points)
    assert distorted_points.shape == (3, 2)
