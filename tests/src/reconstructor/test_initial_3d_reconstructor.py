import numpy as np
from src.reconstructor.initial_3d_reconstructor import Initial3DReconstructor
from src.primitive.twod_gaussians_rs import TwoDGaussians


def generate_covariances_from_rotations_and_scales(rotations, scales):
    """
    Generate covariance matrices from rotations and scales.

    Args:
        rotations (np.ndarray): Array of rotation angles in radians, shape (k,)
        scales (np.ndarray): Array of scales, shape (k, 2)

    Returns:
        covs (np.ndarray): Array of covariance matrices, shape (k, 2, 2)
    """
    k = rotations.shape[0]
    covs = np.zeros((k, 2, 2))
    for i in range(k):
        theta = rotations[i]
        s = scales[i]
        cos_r = np.cos(theta)
        sin_r = np.sin(theta)
        r = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        s_matrix = np.diag(s**2)
        covs[i] = r @ s_matrix @ r.T
    return covs


def test_initial_3d_reconstructor_creation():
    """Test initialization of Initial3DReconstructor."""
    k = 5
    means1 = np.random.rand(k, 2)
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1  # Avoid zero scales
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)

    means2 = np.random.rand(k, 2)
    rotations2 = np.random.uniform(0, 2 * np.pi, k)
    scales2 = np.random.rand(k, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.random.rand(k, 3)
    alpha2 = np.random.rand(k)

    gaussians1 = TwoDGaussians(
        means=means1, covs=covs1, rgb=rgb1, alpha=alpha1,
        rotations=rotations1, scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2, covs=covs2, rgb=rgb2, alpha=alpha2,
        rotations=rotations2, scales=scales2,
    )

    K1 = np.eye(3)
    K2 = np.eye(3)
    H = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H)

    assert isinstance(reconstructor, Initial3DReconstructor)
    assert reconstructor.K1.shape == (3, 3)
    assert reconstructor.K2.shape == (3, 3)
    assert reconstructor.H.shape == (3, 3)
    assert reconstructor.P1 is None
    assert reconstructor.P2 is None
    assert reconstructor.points_3d is None
    assert reconstructor.covariances_3d is None


def test_compute_camera_matrices():
    """Test computation of camera matrices from homography."""
    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    t_translation = np.array([0.5, 0.0, 0.0])
    n = np.array([0, 0, 1])
    d = 1.0
    H = R + (1 / d) * np.outer(t_translation, n)

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
    K1 = np.eye(3)
    K2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H)
    reconstructor.compute_camera_matrices_from_homography()

    assert reconstructor.P1.shape == (3, 4)
    assert reconstructor.P2.shape == (3, 4)

    expected_P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    np.testing.assert_array_almost_equal(reconstructor.P1, expected_P1, decimal=6)

    expected_P2 = K2 @ np.hstack((R, t_translation.reshape(3, 1)))
    np.testing.assert_array_almost_equal(reconstructor.P2, expected_P2, decimal=6)


def test_triangulate_gaussian_centers():
    """Test triangulation of Gaussian centers."""
    np.random.seed(0)
    k = 5
    means1 = np.random.rand(k, 2) * 100
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)

    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    t_translation = np.array([0.5, 0.0, 0.0])
    n = np.array([0, 0, 1])
    d = 1.0
    H = R + (1 / d) * np.outer(t_translation, n)

    means1_hom = np.hstack([means1, np.ones((k, 1))])
    means2_hom = (H @ means1_hom.T).T
    means2 = means2_hom[:, :2] / means2_hom[:, 2].reshape(-1, 1)
    covs2 = covs1.copy()
    rgb2 = rgb1.copy()
    alpha2 = alpha1.copy()

    gaussians1 = TwoDGaussians(
        means=means1, covs=covs1, rgb=rgb1, alpha=alpha1,
        rotations=rotations1, scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2, covs=covs2, rgb=rgb2, alpha=alpha2,
        rotations=rotations1, scales=scales1,
    )

    K1 = np.eye(3)
    K2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H)
    reconstructor.compute_camera_matrices_from_homography()

    transport_matrix = np.eye(k)
    reconstructor.triangulate_gaussian_centers(transport_matrix)
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.points_3d.shape == (k, 3)

    for i in range(k):
        point_3d = reconstructor.points_3d[i]
        X_hom = np.hstack((point_3d, 1))
        x1_proj = reconstructor.P1 @ X_hom
        x1_proj /= x1_proj[2]
        x2_proj = reconstructor.P2 @ X_hom
        x2_proj /= x2_proj[2]

        assert np.allclose(x1_proj[:2], gaussians1.means[i], atol=1e-5)
        assert np.allclose(x2_proj[:2], gaussians2.means[i], atol=1e-5)


def test_compute_3d_gaussian_covariances():
    """Test computation of 3D Gaussian covariances."""
    np.random.seed(0)
    k = 5
    means1 = np.random.rand(k, 2) * 100
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)

    theta = np.pi / 2
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    t_translation = np.array([0.5, 0.0, 0.0])
    n = np.array([0, 0, 1])
    d = 1.0
    H = R + (1 / d) * np.outer(t_translation, n)

    means1_hom = np.hstack([means1, np.ones((k, 1))])
    means2_hom = (H @ means1_hom.T).T
    means2 = means2_hom[:, :2] / means2_hom[:, 2].reshape(-1, 1)
    covs2 = covs1.copy()
    rgb2 = rgb1.copy()
    alpha2 = alpha1.copy()

    gaussians1 = TwoDGaussians(
        means=means1, covs=covs1, rgb=rgb1, alpha=alpha1,
        rotations=rotations1, scales=scales1,
    )
    gaussians2 = TwoDGaussians(
        means=means2, covs=covs2, rgb=rgb2, alpha=alpha2,
        rotations=rotations1, scales=scales1,
    )

    K1 = np.eye(3)
    K2 = np.eye(3)

    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H)
    reconstructor.compute_camera_matrices_from_homography()

    transport_matrix = np.eye(k)
    reconstructor.triangulate_gaussian_centers(transport_matrix)
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.covariances_3d.shape == (k, 3, 3)

    for i in range(k):
        Sigma_3D = reconstructor.covariances_3d[i]
        #check if Sigma_3D is positive definite
        assert np.all(np.linalg.eigvalsh(Sigma_3D) >= -1e-5)
        #check if Sigma_3D is symmetric
        assert np.allclose(Sigma_3D, Sigma_3D.T, atol=1e-5)

