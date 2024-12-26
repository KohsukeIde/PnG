import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
# Update this import path to wherever you store your non-linear reconstructor class
from src.reconstructor.initial_3d_non_linear import Initial3DReconstructor
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
        r = np.array([[cos_r, -sin_r],
                      [sin_r,  cos_r]])
        s_matrix = np.diag(s**2)
        covs[i] = r @ s_matrix @ r.T
    return covs


########################################
# Existing Tests
########################################

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
    # Now the non-linear approach to compute 3D covariance
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.points_3d.shape == (k, 3)

    for i in range(k):
        point_3d = reconstructor.points_3d[i]
        X_hom = np.hstack((point_3d, 1))
        x1_proj = reconstructor.P1 @ X_hom
        x1_proj /= x1_proj[2]
        x2_proj = reconstructor.P2 @ X_hom
        x2_proj /= x2_proj[2]

        np.testing.assert_allclose(x1_proj[:2], gaussians1.means[i], atol=1e-5)
        np.testing.assert_allclose(x2_proj[:2], gaussians2.means[i], atol=1e-5)


def test_compute_3d_gaussian_covariances():
    """Test computation of 3D Gaussian covariances using rotation+scale approach."""
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

    # Non-linear approach
    reconstructor.compute_3d_gaussian_covariances()

    assert reconstructor.covariances_3d.shape == (k, 3, 3)

    for i in range(k):
        Sigma_3D = reconstructor.covariances_3d[i]
        eigvals = np.linalg.eigvalsh(Sigma_3D)
        assert np.all(eigvals >= -1e-6), f"Negative eigenvalues found: {eigvals}"
        assert np.allclose(Sigma_3D, Sigma_3D.T, atol=1e-5)


########################################
# Additional Test:
# Checking 3D->2D->3D Round-Trip
########################################

def quaternion_to_rotation(q):
    """
    Convert quaternion [qw, qx, qy, qz] -> 3x3 rotation.
    """
    qw, qx, qy, qz = q
    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm_q < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw/norm_q, qx/norm_q, qy/norm_q, qz/norm_q

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

def build_covariance_3d(q, s):
    """
    Sigma_3 = R diag(s^2) R^T
    """
    R_ = quaternion_to_rotation(q)
    S_diag = np.diag(s**2)
    return R_ @ S_diag @ R_.T

def project_covariance_3d_to_2d(Sigma_3, point_3d, K, R_cam, t_cam):
    """
    Sigma_2D = J (R_cam Sigma_3 R_cam^T) J^T
    """
    X_c = R_cam @ point_3d + t_cam
    X, Y, Z = X_c
    fx, fy = K[0,0], K[1,1]

    J = np.array([
        [fx / Z,     0.,     -fx*X/(Z**2)],
        [0.,     fy / Z,     -fy*Y/(Z**2)]
    ])
    Sigma_cam = R_cam @ Sigma_3 @ R_cam.T
    Sigma_2D_model = J @ Sigma_cam @ J.T
    return Sigma_2D_model


def test_3d_to_2d_and_back_non_linear():
    """
    This test ensures that if we start with a known 3D Gaussian,
    project it into 2D Gaussians for two cameras, and then run the inverse approach,
    we recover a 3D Gaussian close to the original.
    """
    rng = np.random.default_rng(seed=42)
    # 1) Random quaternion + scale
    # we'll do a quick random approach
    from scipy.spatial.transform import Rotation as R_scipy
    random_quat = R_scipy.random(random_state=rng).as_quat()  # [qx,qy,qz,qw]
    qx, qy, qz, qw = random_quat
    q = np.array([qw, qx, qy, qz])
    s = rng.uniform(1.0, 4.0, size=3)  # random scales

    Sigma_3_true = build_covariance_3d(q, s)

    # 2) Two cameras
    R1 = np.eye(3)
    t1 = np.zeros(3)
    angle = np.radians(50.0)
    R2 = np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [ 0,            1, 0           ],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    t2 = np.array([2.,0.,0.])

    fx = 800.
    fy = 800.
    K1 = np.array([[fx,0,0],[0,fy,0],[0,0,1]])
    K2 = np.array([[fx,0,0],[0,fy,0],[0,0,1]])

    # 3) Single 3D point
    point_3d = np.array([3.,0.,8.])

    # Project Sigma_3_true into 2D
    Sigma_2D_1_obs = project_covariance_3d_to_2d(Sigma_3_true, point_3d, K1, R1, t1)
    Sigma_2D_2_obs = project_covariance_3d_to_2d(Sigma_3_true, point_3d, K2, R2, t2)

    # Also find 2D means
    def project_point(X, K, R_cam, t_cam):
        X_c = R_cam @ X + t_cam
        px = K @ X_c
        if px[2] != 0:
            px[:2] /= px[2]
        return px[:2]

    mean2d_1 = project_point(point_3d, K1, R1, t1)
    mean2d_2 = project_point(point_3d, K2, R2, t2)

    gaussians1 = TwoDGaussians(
        means=np.array([mean2d_1]),
        covs=np.array([Sigma_2D_1_obs]),
        rgb=np.array([[1.,0.,0.]]),
        alpha=np.array([1.]),
        rotations=np.array([0.]),
        scales=np.array([[1.,1.]])
    )
    gaussians2 = TwoDGaussians(
        means=np.array([mean2d_2]),
        covs=np.array([Sigma_2D_2_obs]),
        rgb=np.array([[0.,1.,0.]]),
        alpha=np.array([1.]),
        rotations=np.array([0.]),
        scales=np.array([[1.,1.]])
    )

    # Fake homography
    H_fake = np.eye(3)
    reconstructor = Initial3DReconstructor(gaussians1, gaussians2, K1, K2, H_fake)

    # Set P1, P2 manually
    reconstructor.P1 = K1 @ np.hstack((R1, t1.reshape(3,1)))
    reconstructor.P2 = K2 @ np.hstack((R2, t2.reshape(3,1)))

    # We know the 3D point, so just store it
    reconstructor.points_3d = np.array([point_3d])

    # 4) Non-linear approach that reconstructs 3D covariance
    reconstructor.compute_3d_gaussian_covariances()

    Sigma_3_est = reconstructor.covariances_3d[0]

    # Check closeness
    fro_diff = np.linalg.norm(Sigma_3_est - Sigma_3_true, ord='fro')
    print("Sigma_3_true:\n", Sigma_3_true)
    print("Sigma_3_est:\n", Sigma_3_est)
    print("Fro diff:", fro_diff)

    # Tolerance depends on your scene scale / solver quality
    assert fro_diff < 20.0, f"3D->2D->3D round trip is not accurate enough (FroDiff={fro_diff})"

    # Check positivity
    eigvals = np.linalg.eigvalsh(Sigma_3_est)
    assert np.all(eigvals >= 0.0), f"Negative eigenvalues: {eigvals}"
