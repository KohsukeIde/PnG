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
# Existing Tests
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
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.points_3d.shape == (k_val, 3)

    for i in range(k_val):
        point_3d = reconstructor.points_3d[i]
        x_hom = np.hstack((point_3d, 1))
        x1_proj = reconstructor.p1 @ x_hom
        x1_proj /= x1_proj[2]
        x2_proj = reconstructor.p2 @ x_hom
        x2_proj /= x2_proj[2]

        np.testing.assert_allclose(x1_proj[:2], gaussians1.means[i], atol=1e-5)
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
    """Check that if we start with a known 3D Gaussian, project it into 2D for two cameras,
    and run the inverse approach, we can recover a 3D Gaussian close to the original.
    """
    rng = np.random.default_rng(seed=42)
    random_quat = Rotation.random(random_state=rng).as_quat()  # [qx, qy, qz, qw]
    qx, qy, qz, qw = random_quat
    q_val = np.array([qw, qx, qy, qz])
    s_val = rng.uniform(1.0, 4.0, size=3)
    sigma_3_true = build_covariance_3d(q_val, s_val)

    r1 = np.eye(3)
    t1 = np.zeros(3)
    angle = np.radians(50.0)
    r2 = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    t2 = np.array([2.0, 0.0, 0.0])

    fx = 800.0
    fy = 800.0
    k1 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    k2 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    point_3d = np.array([3.0, 0.0, 8.0])
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

    gaussians1 = TwoDGaussians(
        means=np.array([mean2d_1]),
        covs=np.array([sigma_2d_1_obs]),
        rgb=np.array([[1.0, 0.0, 0.0]]),
        alpha=np.array([1.0]),
        rotations=np.array([0.0]),
        scales=np.array([[1.0, 1.0]]),
    )
    gaussians2 = TwoDGaussians(
        means=np.array([mean2d_2]),
        covs=np.array([sigma_2d_2_obs]),
        rgb=np.array([[0.0, 1.0, 0.0]]),
        alpha=np.array([1.0]),
        rotations=np.array([0.0]),
        scales=np.array([[1.0, 1.0]]),
    )

    h_fake = np.eye(3)
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h_fake)

    reconstructor.p1 = k1 @ np.hstack((r1, t1.reshape(3, 1)))
    reconstructor.p2 = k2 @ np.hstack((r2, t2.reshape(3, 1)))

    reconstructor.points_3d = np.array([point_3d])
    reconstructor.match_pairs = [(0, 0)]

    reconstructor.compute_3d_gaussian_covariances(lambda_volume=0.0, target_volume=1.0)
    sigma_3_reconstructed = reconstructor.covariances_3d[0]
    assert np.allclose(
        sigma_3_reconstructed, sigma_3_true, atol=1e-2
    ), "Reconstructed covariance does not match true covariance."
