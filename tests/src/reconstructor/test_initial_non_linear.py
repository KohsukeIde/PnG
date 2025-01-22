import numpy as np
from scipy.spatial.transform import Rotation

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor


def generate_covariances_from_rotations_and_scales(rotations, scales):
    """Generate covariance matrices from rotations and scales.

    Args:
        rotations (np.ndarray): Array of rotation angles in radians, shape (k,).
        scales (np.ndarray): Array of scales, shape (k, 2).

    Returns:
        covs (np.ndarray): Array of covariance matrices, shape (k, 2, 2).
    """
    k_val = rotations.shape[0]
    covs = np.zeros((k_val, 2, 2))
    for i in range(k_val):
        theta = rotations[i]
        s_val = scales[i]
        cos_r = np.cos(theta)
        sin_r = np.sin(theta)
        r_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        s_matrix = np.diag(s_val**2)
        covs[i] = r_mat @ s_matrix @ r_mat.T
    return covs


########################################
# Tests excluding 3D->2D->3D round-trip
########################################


def test_initial_3d_reconstructor_creation():
    """Test initialization of Initial3DReconstructor."""
    k_val = 5
    means1 = np.random.rand(k_val, 2)
    rotations1 = np.random.uniform(0, 2 * np.pi, k_val)
    scales1 = np.random.rand(k_val, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k_val, 3)
    alpha1 = np.random.rand(k_val)

    means2 = np.random.rand(k_val, 2)
    rotations2 = np.random.uniform(0, 2 * np.pi, k_val)
    scales2 = np.random.rand(k_val, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.random.rand(k_val, 3)
    alpha2 = np.random.rand(k_val)

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        rgb=rgb1,
        alpha=alpha1,
        rotations=rotations1,
        scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        rgb=rgb2,
        alpha=alpha2,
        rotations=rotations2,
        scales=scales2,
    )

    k1 = np.eye(3)
    k2 = np.eye(3)
    h_mat = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_mat)

    assert isinstance(reconstructor, Initial3DReconstructor)
    assert reconstructor.k1.shape == (3, 3)
    assert reconstructor.k2.shape == (3, 3)
    assert reconstructor.h.shape == (3, 3)
    assert reconstructor.p1 is None
    assert reconstructor.p2 is None
    assert reconstructor.points_3d is None
    assert reconstructor.covariances_3d is None


def test_compute_camera_matrices():
    """Test computation of camera matrices from homography."""
    theta = np.pi / 2
    r_mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h_mat = r_mat + (1 / d_val) * np.outer(t_translation, n_val)

    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.array([np.eye(2) for _ in range(5)]),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
        rotations=np.random.uniform(0, 2 * np.pi, 5),
        scales=np.random.rand(5, 2) + 0.1,
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.array([np.eye(2) for _ in range(5)]),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
        rotations=np.random.uniform(0, 2 * np.pi, 5),
        scales=np.random.rand(5, 2) + 0.1,
    )
    k1 = np.eye(3)
    k2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_mat)
    reconstructor.compute_camera_matrices_from_homography()

    assert reconstructor.p1.shape == (3, 4)
    assert reconstructor.p2.shape == (3, 4)

    expected_p1 = k1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    np.testing.assert_array_almost_equal(reconstructor.p1, expected_p1, decimal=6)

    expected_p2 = k2 @ np.hstack((r_mat, t_translation.reshape(3, 1)))
    np.testing.assert_array_almost_equal(reconstructor.p2, expected_p2, decimal=6)


def test_triangulate_gaussian_centers():
    """Test triangulation of Gaussian centers."""
    np.random.seed(0)
    k_val = 5
    means1 = np.random.rand(k_val, 2) * 100
    rotations1 = np.random.uniform(0, 2 * np.pi, k_val)
    scales1 = np.random.rand(k_val, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k_val, 3)
    alpha1 = np.random.rand(k_val)

    theta = np.pi / 2
    r_mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h_mat = r_mat + (1 / d_val) * np.outer(t_translation, n_val)

    means1_hom = np.hstack([means1, np.ones((k_val, 1))])
    means2_hom = (h_mat @ means1_hom.T).T
    means2 = means2_hom[:, :2] / means2_hom[:, 2].reshape(-1, 1)
    covs2 = covs1.copy()
    rgb2 = rgb1.copy()
    alpha2 = alpha1.copy()

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        rgb=rgb1,
        alpha=alpha1,
        rotations=rotations1,
        scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        rgb=rgb2,
        alpha=alpha2,
        rotations=rotations1,
        scales=scales1,
    )

    k1 = np.eye(3)
    k2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_mat)
    reconstructor.compute_camera_matrices_from_homography()

    transport_matrix = np.eye(k_val)
    reconstructor.triangulate_gaussian_centers(transport_matrix)

    # Assert that the shape of the reconstructed 3D points array is (k_val, 3),
    # meaning there are k_val points, each with 3 coordinates (x, y, z).
    assert reconstructor.points_3d.shape == (k_val, 3)

    # Loop over each Gaussian center to verify the triangulation results.
    for i in range(k_val):
        # Get the i-th 3D point from the reconstructed points.
        point_3d = reconstructor.points_3d[i]

        # Convert the 3D point to homogeneous coordinates by appending a 1.
        x_hom = np.hstack((point_3d, 1))

        # Project the 3D point into the first camera's 2D image plane using the camera matrix p1.
        x1_proj = reconstructor.p1 @ x_hom

        # Normalize the projected point by dividing by the third (homogeneous) coordinate.
        x1_proj /= x1_proj[2]

        # Project the 3D point into the second camera's 2D image plane using the camera matrix p2.
        x2_proj = reconstructor.p2 @ x_hom

        # Normalize the projected point by dividing by the third (homogeneous) coordinate.
        x2_proj /= x2_proj[2]

        # Assert that the projected 2D point in the first camera matches the original 2D Gaussian mean.
        np.testing.assert_allclose(x1_proj[:2], gaussians1.means[i], atol=1e-5)

        # Assert that the projected 2D point in the second camera matches the original 2D Gaussian mean.
        np.testing.assert_allclose(x2_proj[:2], gaussians2.means[i], atol=1e-5)


def test_compute_3d_gaussian_covariances():
    """Test computation of 3D Gaussian covariances using rotation+scale approach."""
    np.random.seed(0)
    k_val = 5
    means1 = np.random.rand(k_val, 2) * 100
    rotations1 = np.random.uniform(0, 2 * np.pi, k_val)
    scales1 = np.random.rand(k_val, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k_val, 3)
    alpha1 = np.random.rand(k_val)

    theta = np.pi / 2
    r_mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h_mat = r_mat + (1 / d_val) * np.outer(t_translation, n_val)

    means1_hom = np.hstack([means1, np.ones((k_val, 1))])
    means2_hom = (h_mat @ means1_hom.T).T
    means2 = means2_hom[:, :2] / means2_hom[:, 2].reshape(-1, 1)
    covs2 = covs1.copy()
    rgb2 = rgb1.copy()
    alpha2 = alpha1.copy()

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        rgb=rgb1,
        alpha=alpha1,
        rotations=rotations1,
        scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        rgb=rgb2,
        alpha=alpha2,
        rotations=rotations1,
        scales=scales1,
    )

    k1 = np.eye(3)
    k2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_mat)
    reconstructor.compute_camera_matrices_from_homography()

    transport_matrix = np.eye(k_val)
    reconstructor.triangulate_gaussian_centers(transport_matrix)
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.covariances_3d.shape == (k_val, 3, 3)

    for i in range(k_val):
        sigma_3d = reconstructor.covariances_3d[i]
        eigvals = np.linalg.eigvalsh(sigma_3d)
        assert np.all(eigvals >= -1e-6), f"Negative eigenvalues found: {eigvals}"
        assert np.allclose(sigma_3d, sigma_3d.T, atol=1e-5)


########################################
# Additional Test: Checking 3D->2D->3D Round-Trip
########################################


def quaternion_to_rotation(q_val):
    """Convert quaternion [qw, qx, qy, qz] -> 3x3 rotation."""
    qw, qx, qy, qz = q_val
    norm_q = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm_q < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q

    r_array = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )
    return r_array


def build_covariance_3d(q_val, s_val):
    """Build a 3D covariance: Sigma_3 = R diag(s^2) R^T."""
    r_ = quaternion_to_rotation(q_val)
    s_diag = np.diag(s_val**2)
    return r_ @ s_diag @ r_.T


def project_covariance_3d_to_2d(sigma_3, point_3d, k_val, r_cam, t_cam):
    """Project a 3D covariance sigma_3 to 2D using local Jacobian approximation."""
    x_c = r_cam @ point_3d + t_cam
    x_val, y_val, z_val = x_c
    fx, fy = k_val[0, 0], k_val[1, 1]

    j_mat = np.array(
        [
            [fx / z_val, 0.0, -fx * x_val / (z_val**2)],
            [0.0, fy / z_val, -fy * y_val / (z_val**2)],
        ]
    )
    sigma_cam = r_cam @ sigma_3 @ r_cam.T
    sigma_2d_model = j_mat @ sigma_cam @ j_mat.T
    return sigma_2d_model


def test_3d_to_2d_and_back_non_linear():
    """Test the complete pipeline of projecting a 3D Gaussian to 2D views and reconstructing back to 3D."""
    rng = np.random.default_rng(seed=42)

    # Generate a random rotation using quaternions
    # scipy.spatial.transform.Rotation.random() returns [qx, qy, qz, qw]
    random_quat = Rotation.random(random_state=rng).as_quat()
    qx, qy, qz, qw = random_quat

    # Reorder quaternion components to [qw, qx, qy, qz] format for our quaternion_to_rotation function
    q_val = np.array([qw, qx, qy, qz])

    # Generate random scale values between 1.0 and 4.0 for X, Y, Z dimensions
    s_val = rng.uniform(1.0, 4.0, size=3)

    # Build the true 3D covariance matrix using rotation and scale
    sigma_3_true = build_covariance_3d(q_val, s_val)

    # Set up first camera (camera 1) at origin
    r1 = np.eye(3)  # Identity rotation matrix (no rotation)
    t1 = np.zeros(3)  # No translation (at origin)

    # Set up second camera (camera 2) with rotation and translation
    angle = np.radians(50.0)  # Convert 50 degrees to radians
    # Create rotation matrix for camera 2 (rotation around Y axis)
    r2 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],  # First row
            [0, 1, 0],  # Second row (Y axis unchanged)
            [-np.sin(angle), 0, np.cos(angle)],  # Third row
        ]
    )
    # Translate camera 2 along X axis
    t2 = np.array([2.0, 0.0, 0.0])

    # Set up camera intrinsic parameters
    fx = 800.0  # Focal length in x direction
    fy = 800.0  # Focal length in y direction
    # Create camera calibration matrices (same for both cameras)
    k1 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    k2 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    # Define 3D point location
    point_3d = np.array([3.0, 0.0, 8.0])  # Point is 3 units right, 8 units forward

    # Project 3D covariance to 2D in both camera views
    sigma_2d_1_obs = project_covariance_3d_to_2d(sigma_3_true, point_3d, k1, r1, t1)
    sigma_2d_2_obs = project_covariance_3d_to_2d(sigma_3_true, point_3d, k2, r2, t2)

    def project_point(x_val, k_val_local, r_cam_local, t_cam_local):
        x_c_local = r_cam_local @ x_val + t_cam_local
        px_local = k_val_local @ x_c_local
        if px_local[2] != 0:
            px_local[:2] /= px_local[2]
        return px_local[:2]

    mean2d_1 = project_point(point_3d, k1, r1, t1)
    mean2d_2 = project_point(point_3d, k2, r2, t2)

    # Create 2D Gaussian for first camera view (red color)
    gaussians1 = TwoDGaussians(
        means=np.array([mean2d_1]),  # 2D projected point
        covs=np.array([sigma_2d_1_obs]),  # 2D projected covariance
        rgb=np.array([[1.0, 0.0, 0.0]]),  # Red color
        alpha=np.array([1.0]),  # Full opacity
        rotations=np.array([0.0]),  # No additional rotation
        scales=np.array([[1.0, 1.0]]),  # Unit scale
    )

    # Create 2D Gaussian for second camera view (green color)
    gaussians2 = TwoDGaussians(
        means=np.array([mean2d_2]),  # 2D projected point
        covs=np.array([sigma_2d_2_obs]),  # 2D projected covariance
        rgb=np.array([[1.0, 0.0, 0.0]]),  # Red color
        alpha=np.array([1.0]),  # Full opacity
        rotations=np.array([0.0]),  # No additional rotation
        scales=np.array([[1.0, 1.0]]),  # Unit scale
    )

    # Create identity homography (not used in this test)
    h_fake = np.eye(3)

    # Initialize the 3D reconstructor with our 2D Gaussians
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_fake)

    # Set the camera projection matrices
    # Format is K[R|t] for each camera
    reconstructor.p1 = k1 @ np.hstack((r1, t1.reshape(3, 1)))
    reconstructor.p2 = k2 @ np.hstack((r2, t2.reshape(3, 1)))

    # Set the known 3D point (in real use, this would be computed via triangulation)
    reconstructor.points_3d = np.array([point_3d])

    # Set matching pairs between views (in this case, just one pair: point 0 matches point 0)
    reconstructor.match_pairs = [(0, 0)]

    # Compute 3D Gaussian covariances
    # lambda_volume=0.0: no volume regularization
    # target_volume=1.0: target volume constraint (not used when lambda=0)
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=0.0, target_volume=1.0)

    # Get the reconstructed 3D covariance
    sigma_3_reconstructed = reconstructor.covariances_3d[0]

    # Verify that reconstructed covariance matches the original
    # Allow for small numerical differences (atol=1e-2)
    assert np.allclose(
        sigma_3_reconstructed, sigma_3_true, atol=1e-2
    ), "Reconstructed covariance does not match true covariance."
