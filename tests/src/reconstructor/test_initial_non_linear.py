import numpy as np
from scipy.spatial.transform import Rotation

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
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
    rng = np.random.default_rng(seed=42)
    k_val = 5
    means1 = rng.random((k_val, 2)) * 100
    rotations1 = rng.uniform(0, 2 * np.pi, k_val)
    scales1 = rng.random((k_val, 2)) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = rng.random((k_val, 3))
    alpha1 = rng.random(k_val)

    means2 = rng.random((k_val, 2))
    rotations2 = rng.uniform(0, 2 * np.pi, k_val)
    scales2 = rng.random((k_val, 2)) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = rng.random((k_val, 3))
    alpha2 = rng.random(k_val)

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
    rng = np.random.default_rng(seed=42)
    k_val = 5
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
    # if K is not identity, homography = K2 @ (R + (t @ n.T) / d) @ np.linalg.inv(K1)
    h_mat = r_mat + (1 / d_val) * np.outer(t_translation, n_val)

    gaussians1 = TwoDGaussians(
        means=rng.random((k_val, 2)),
        covs=np.array([np.eye(2) for _ in range(k_val)]),
        rgb=rng.random((k_val, 3)),
        alpha=rng.random(k_val),
        rotations=rng.uniform(0, 2 * np.pi, k_val),
        scales=rng.random((k_val, 2)) + 0.1,
    )
    gaussians2 = TwoDGaussians(
        means=rng.random((k_val, 2)),
        covs=np.array([np.eye(2) for _ in range(k_val)]),
        rgb=rng.random((k_val, 3)),
        alpha=rng.random(k_val),
        rotations=rng.uniform(0, 2 * np.pi, k_val),
        scales=rng.random((k_val, 2)) + 0.1,
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
    rng = np.random.default_rng(seed=42)
    k_val = 5
    means1 = rng.random((k_val, 2)) * 100
    rotations1 = rng.uniform(0, 2 * np.pi, k_val)
    scales1 = rng.random((k_val, 2)) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = rng.random((k_val, 3))
    alpha1 = rng.random(k_val)

    theta = np.pi / 2
    r_mat = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    #Plane induced homography
    t_translation = np.array([0.5, 0.0, 0.0])
    n_val = np.array([0, 0, 1])
    d_val = 1.0
    h_mat = r_mat + (1 / d_val) * np.outer(t_translation, n_val)

    means1_hom = np.hstack([means1, np.ones((k_val, 1))])
    means2_hom = (h_mat @ means1_hom.T).T
    
    #2D point projected onto the second camera image, when a 2D point on the first camera image is transformed by homography.
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

    # Loop over each Gaussian center
    for i in range(k_val):
        # Get the i-th 3D point.
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
    rng = np.random.default_rng(seed=42)
    k_val = 5
    means1 = rng.random((k_val, 2)) * 100
    rotations1 = rng.uniform(0, 2 * np.pi, k_val)
    scales1 = rng.random((k_val, 2)) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = rng.random((k_val, 3))
    alpha1 = rng.random(k_val)

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


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [qw, qx, qy, qz] into a 3x3 rotation matrix.

    Args:
        q (np.ndarray): A 4-element array representing the quaternion (qw, qx, qy, qz).

    Returns:
        np.ndarray: A 3x3 orthonormal rotation matrix.
    """
    qw, qx, qy, qz = q
    norm_q = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm_q < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q
    r_mat = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )
    return r_mat


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
    
    # Pinhole camera model Jacobian
    j_mat = np.array(
        [
            [fx / z_val, 0.0, -fx * x_val / (z_val**2)],
            [0.0, fy / z_val, -fy * y_val / (z_val**2)],
        ]
    )
    sigma_cam = r_cam @ sigma_3 @ r_cam.T
    sigma_2d_model = j_mat @ sigma_cam @ j_mat.T
    return sigma_2d_model


def project_point(x_val, k_val_local, r_cam_local, t_cam_local):
    """Project a 3D point to 2D using camera parameters.

    Args:
        x_val: 3D point, shape (3,)
        k_val_local: Camera intrinsic matrix, shape (3, 3)
        r_cam_local: Camera rotation matrix, shape (3, 3)
        t_cam_local: Camera translation vector, shape (3,)

    Returns:
        2D projected point, shape (2,)
    """
    # Transform to camera space: (3,3) @ (3,) + (3,) -> (3,)
    x_c_local = r_cam_local @ x_val + t_cam_local

    # Project to image plane: (3,3) @ (3,) -> (3,)
    px_local = k_val_local @ x_c_local

    # Perspective division if z != 0
    # Convert from homogeneous to euclidean: (3,) -> (2,)
    if px_local[2] != 0:
        px_local[:2] /= px_local[2]

    # Return x,y pixel coordinates: shape (2,)
    return px_local[:2]


def test_3d_to_2d_and_back_non_linear():
    """Test the complete pipeline of projecting a 3D Gaussian to 2D views and reconstructing back to 3D,
       using a Fundamental-based approach instead of Homography.
    """
    rng = np.random.default_rng(seed=42)

    # 1) Generate random 3D covariance (same as original)
    random_quat = Rotation.random(random_state=rng).as_quat()
    qx, qy, qz, qw = random_quat
    q_val = np.array([qw, qx, qy, qz])
    s_val = rng.uniform(1.0, 2.0, size=3)
    sigma_3_true = build_covariance_3d(q_val, s_val)

    # 2) Set up camera1, camera2
    r1 = np.eye(3)
    t1 = np.zeros(3)
    angle = np.radians(40.0)
    r2 = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t2 = np.array([2.0, 0.0, 0.0])

    fx = fy = 800.0
    k1 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    k2 = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    point_3d = np.array([3.0, 0.0, 30.0])

    # 3) Project that 3D covariance into each camera => 2D Gaussians
    sigma_2d_1_obs = project_covariance_3d_to_2d(sigma_3_true, point_3d, k1, r1, t1)
    sigma_2d_2_obs = project_covariance_3d_to_2d(sigma_3_true, point_3d, k2, r2, t2)
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

    # 4) Instead of homography, use Fundamental approach:
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, k1, k2,
        epsilon=0.1,
        lambda_mean=0.0,    # might set to 0 if you only want epipolar dist
        lambda_cov=0.0,     
        lambda_color=0.0,   
        device=None
    )
    # Optimize fundamental
    solver.optimize_with_fundamental(max_iter=1000, tol=1e-6)
    # Get transport
    cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
    transport = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
    transport_matrix = transport.detach().cpu().numpy()  # shape (1,1)

    # 5) Use reconstructor => pass dummy homography to constructor
    reconstructor = Initial3DReconstructor(
        gaussians1, gaussians2, k1, k2, np.eye(3)
    )

    # Explicitly define p1, p2 from (r1,t1), (r2,t2)
    reconstructor.set_camera_matrices_explicitly(r1, t1, r2, t2)

    # Triangulate => we have only 1 match => top_k=1
    reconstructor.triangulate_gaussian_centers(transport_matrix, threshold=0.0, top_k=1)

    # 6) Now compute 3D covariance => no volume regularization
    reconstructor.compute_3d_gaussian_covariances(lambda_volume=0.0, target_volume=1.0)

    sigma_3_reconstructed = reconstructor.covariances_3d[0]

    # 7) Compare
    diff_mat = sigma_3_reconstructed - sigma_3_true
    frob_diff = np.linalg.norm(diff_mat, ord='fro')  # Frobenius norm
    n_elements = diff_mat.size  # 3x3 => 9

    # Root Mean Square Error (RMS) per element
    rms = frob_diff / np.sqrt(n_elements)

    threshold_rms = 0.5
    assert rms < threshold_rms, f"RMS difference too large: {rms}"
    denom = np.linalg.norm(sigma_3_true, ord='fro') + 1e-12
    rel_diff = frob_diff / denom

    assert rel_diff < 0.3, f"Relative Frobenius difference too large: {rel_diff}"
