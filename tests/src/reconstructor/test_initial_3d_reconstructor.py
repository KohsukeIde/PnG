import numpy as np

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstructor.initial_3d_reconstructor import Initial3DReconstructor


def generate_covariances_from_rotations_and_scales(rotations, scales):
    """Generate covariance matrices from rotations and scales.

    Args:
        rotations (np.ndarray): Array of rotation angles in radians, shape (k,).
        scales (np.ndarray): Array of scales, shape (k, 2).

    Returns:
        covs (np.ndarray): Array of covariance matrices, shape (k, 2, 2).
    """
    k = rotations.shape[0]
    covs = np.zeros((k, 2, 2))
    for i in range(k):
        theta = rotations[i]
        s_val = scales[i]
        cos_r = np.cos(theta)
        sin_r = np.sin(theta)
        r_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        s_matrix = np.diag(s_val**2)
        covs[i] = r_mat @ s_matrix @ r_mat.T
    return covs


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
    h = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h)

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
    r = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h = r + (1 / d_val) * np.outer(t_translation, n_val)

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

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h)
    reconstructor.compute_camera_matrices_from_homography()

    assert reconstructor.p1.shape == (3, 4)
    assert reconstructor.p2.shape == (3, 4)

    expected_p1 = k1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    np.testing.assert_array_almost_equal(reconstructor.p1, expected_p1, decimal=6)

    expected_p2 = k2 @ np.hstack((r, t_translation.reshape(3, 1)))
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
    r = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h = r + (1 / d_val) * np.outer(t_translation, n_val)

    means1_hom = np.hstack([means1, np.ones((k_val, 1))])
    means2_hom = (h @ means1_hom.T).T
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

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h)
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

        assert np.allclose(x1_proj[:2], gaussians1.means[i], atol=1e-5)
        assert np.allclose(x2_proj[:2], gaussians2.means[i], atol=1e-5)


def test_compute_3d_gaussian_covariances():
    """Test computation of 3D Gaussian covariances."""
    np.random.seed(0)
    k_val = 5
    means1 = np.random.rand(k_val, 2) * 100
    rotations1 = np.random.uniform(0, 2 * np.pi, k_val)
    scales1 = np.random.rand(k_val, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k_val, 3)
    alpha1 = np.random.rand(k_val)

    theta = np.pi / 2
    r = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h = r + (1 / d_val) * np.outer(t_translation, n_val)

    means1_hom = np.hstack([means1, np.ones((k_val, 1))])
    means2_hom = (h @ means1_hom.T).T
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

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, k1, k2, h)
    reconstructor.compute_camera_matrices_from_homography()

    transport_matrix = np.eye(k_val)
    reconstructor.triangulate_gaussian_centers(transport_matrix)
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.covariances_3d.shape == (k_val, 3, 3)

    for i in range(k_val):
        sigma_3d = reconstructor.covariances_3d[i]
        # Check if sigma_3d is positive definite
        assert np.all(np.linalg.eigvalsh(sigma_3d) >= -1e-5)
        # Check if sigma_3d is symmetric
        assert np.allclose(sigma_3d, sigma_3d.T, atol=1e-5)
