import numpy as np

from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
from src.primitive.twod_gaussians_rs import TwoDGaussians


def generate_covariances_from_rotations_and_scales(rotations, scales):
    """Generate covariance matrices from rotations and scales.

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
        s = np.diag(s**2)
        covs[i] = r @ s @ r.T
    return covs


def test_optimal_transport_solver_initialization():
    """Test the initialization of OptimalTransportSolver with valid input."""
    k1 = 5
    k2 = 7
    # Generate random means, rgb, alpha, rotations, scales
    means1 = np.random.rand(k1, 2)
    rgb1 = np.random.rand(k1, 3)
    alpha1 = np.random.rand(k1)
    rotations1 = np.random.uniform(0, 2 * np.pi, k1)
    scales1 = np.random.rand(k1, 2) + 0.1  # Avoid zero scales
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k2, 2)
    rgb2 = np.random.rand(k2, 3)
    alpha2 = np.random.rand(k2)
    rotations2 = np.random.uniform(0, 2 * np.pi, k2)
    scales2 = np.random.rand(k2, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

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

    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=0.1, lambda_color=0.5
    )
    assert isinstance(solver, OptimalTransportSolver)
    assert solver.epsilon == 0.1
    assert solver.lambda_color == 0.5


def test_compute_cost_matrix_shape():
    """Test if the computed cost matrix has the correct shape."""
    k1 = 5
    k2 = 7
    means1 = np.random.rand(k1, 2)
    rgb1 = np.random.rand(k1, 3)
    alpha1 = np.random.rand(k1)
    rotations1 = np.random.uniform(0, 2 * np.pi, k1)
    scales1 = np.random.rand(k1, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k2, 2)
    rgb2 = np.random.rand(k2, 3)
    alpha2 = np.random.rand(k2)
    rotations2 = np.random.uniform(0, 2 * np.pi, k2)
    scales2 = np.random.rand(k2, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

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

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    assert cost_matrix.shape == (k1, k2)


def test_compute_cost_matrix_simple_case():
    """Test the cost matrix computation with a simple case."""
    # Define Gaussians with identity covariance matrices
    means1 = np.array([[0, 0], [1, 1]])
    scales1 = np.array([[1, 1], [1, 1]])  # Scales corresponding to identity covariance
    rotations1 = np.array([0.0, 0.0])
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.array([[1, 0, 0], [0, 1, 0]])
    alpha1 = np.array([0.5, 0.5])

    means2 = np.array([[0, 0], [2, 2]])
    scales2 = np.array([[1, 1], [1, 1]])
    rotations2 = np.array([0.0, 0.0])
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.array([[1, 0, 0], [0, 0, 1]])
    alpha2 = np.array([0.5, 0.5])

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

    # Set lambda_pos explicitly to match the expected cost
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, lambda_pos=1.0, lambda_color=1.0
    )
    cost_matrix = solver.compute_cost_matrix()

    expected_cost = np.array(
        [
            [0, 10],  # (0,0) to (0,0) and (0,0) to (2,2)
            [4, 4],  # (1,1) to (0,0) and (1,1) to (2,2)
        ]
    )

    np.testing.assert_allclose(cost_matrix, expected_cost, atol=1e-6)


def test_compute_cost_matrix_identical_distribution():
    """Test if the cost matrix diagonal is zero for identical distributions."""
    k = 5
    means = np.random.rand(k, 2)
    rgb = np.random.rand(k, 3)
    alpha = np.random.rand(k)
    rotations = np.random.uniform(0, 2 * np.pi, k)
    scales = np.random.rand(k, 2) + 0.1
    covs = generate_covariances_from_rotations_and_scales(rotations, scales)

    gaussians1 = TwoDGaussians(means, covs, rgb, alpha, rotations, scales)
    gaussians2 = TwoDGaussians(
        means.copy(),
        covs.copy(),
        rgb.copy(),
        alpha.copy(),
        rotations.copy(),
        scales.copy(),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2, lambda_color=1.0)
    cost_matrix = solver.compute_cost_matrix()
    np.testing.assert_allclose(np.diag(cost_matrix), 0, atol=1e-6)


def test_compute_cost_matrix_similarity():
    """Test the similarity of the cost matrix when swapping input order."""
    np.random.seed(42)  # Set random seed for reproducibility
    k = 5
    means1 = np.random.rand(k, 2)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k, 2)
    rgb2 = np.random.rand(k, 3)
    alpha2 = np.random.rand(k)
    rotations2 = np.random.uniform(0, 2 * np.pi, k)
    scales2 = np.random.rand(k, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

    gaussians1 = TwoDGaussians(means1, covs1, rgb1, alpha1, rotations1, scales1)
    gaussians2 = TwoDGaussians(means2, covs2, rgb2, alpha2, rotations2, scales2)

    solver_ab = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix_ab = solver_ab.compute_cost_matrix()

    solver_ba = OptimalTransportSolver(gaussians2, gaussians1)
    cost_matrix_ba = solver_ba.compute_cost_matrix()

    # Transpose cost_matrix_ba to compare with cost_matrix_ab
    diff = np.abs(cost_matrix_ab - cost_matrix_ba.T)

    # Check that the maximum absolute difference is within a reasonable tolerance
    assert np.max(diff) < 0.2, f"Maximum difference {np.max(diff)} exceeds tolerance"

    # Check that the average absolute difference is small
    assert np.mean(diff) < 0.05, f"Mean difference {np.mean(diff)} exceeds tolerance"


def test_sinkhorn_algorithm_shape():
    """Test if the Sinkhorn algorithm returns a matrix of the correct shape."""
    k1 = 5
    k2 = 7
    means1 = np.random.rand(k1, 2)
    rgb1 = np.random.rand(k1, 3)
    alpha1 = np.random.rand(k1)
    rotations1 = np.random.uniform(0, 2 * np.pi, k1)
    scales1 = np.random.rand(k1, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k2, 2)
    rgb2 = np.random.rand(k2, 3)
    alpha2 = np.random.rand(k2)
    rotations2 = np.random.uniform(0, 2 * np.pi, k2)
    scales2 = np.random.rand(k2, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

    gaussians1 = TwoDGaussians(means1, covs1, rgb1, alpha1, rotations1, scales1)
    gaussians2 = TwoDGaussians(means2, covs2, rgb2, alpha2, rotations2, scales2)

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    assert transport_matrix.shape == (k1, k2)


def test_sinkhorn_algorithm_properties():
    """Test the properties of the Sinkhorn algorithm output."""
    np.random.seed(42)
    k = 5
    means1 = np.random.rand(k, 2)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k, 2)
    rgb2 = np.random.rand(k, 3)
    alpha2 = np.random.rand(k)
    rotations2 = np.random.uniform(0, 2 * np.pi, k)
    scales2 = np.random.rand(k, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

    gaussians1 = TwoDGaussians(means1, covs1, rgb1, alpha1, rotations1, scales1)
    gaussians2 = TwoDGaussians(means2, covs2, rgb2, alpha2, rotations2, scales2)

    solver = OptimalTransportSolver(gaussians1, gaussians2, epsilon=0.1)
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # Test non-negativity
    assert np.all(transport_matrix >= 0), "Transport matrix contains negative values"

    # Test row sum constraint
    row_sums = np.sum(transport_matrix, axis=1)
    alpha_normalized = gaussians1.alpha / np.sum(gaussians1.alpha)
    np.testing.assert_allclose(row_sums, alpha_normalized, rtol=1e-5)

    # Test column sum constraint
    col_sums = np.sum(transport_matrix, axis=0)
    beta_normalized = gaussians2.alpha / np.sum(gaussians2.alpha)
    np.testing.assert_allclose(col_sums, beta_normalized, rtol=1e-5)


def test_sinkhorn_algorithm_simple_case():
    """Test the Sinkhorn algorithm with a simple 2x2 case."""
    means1 = np.array([[0, 0], [1, 1]])
    scales1 = np.array([[1, 1], [1, 1]])
    rotations1 = np.array([0.0, 0.0])
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.array([[1, 0, 0], [0, 1, 0]])
    alpha1 = np.array([0.5, 0.5])

    means2 = np.array([[0, 0], [2, 2]])
    scales2 = np.array([[1, 1], [1, 1]])
    rotations2 = np.array([0.0, 0.0])
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.array([[1, 0, 0], [0, 0, 1]])
    alpha2 = np.array([0.5, 0.5])

    gaussians1 = TwoDGaussians(means1, covs1, rgb1, alpha1, rotations1, scales1)
    gaussians2 = TwoDGaussians(means2, covs2, rgb2, alpha2, rotations2, scales2)

    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=0.1, lambda_color=1.0
    )
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    expected_transport = np.array([[0.5, 0.0], [0.0, 0.5]])

    np.testing.assert_allclose(transport_matrix, expected_transport, atol=1e-2)


def test_sinkhorn_algorithm_convergence():
    """Test if the Sinkhorn algorithm converges within a reasonable number of iterations."""
    np.random.seed(42)
    k = 5
    means1 = np.random.rand(k, 2)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k, 2)
    rgb2 = np.random.rand(k, 3)
    alpha2 = np.random.rand(k)
    rotations2 = np.random.uniform(0, 2 * np.pi, k)
    scales2 = np.random.rand(k, 2) + 0.1
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)

    gaussians1 = TwoDGaussians(means1, covs1, rgb1, alpha1, rotations1, scales1)
    gaussians2 = TwoDGaussians(means2, covs2, rgb2, alpha2, rotations2, scales2)

    solver = OptimalTransportSolver(gaussians1, gaussians2, epsilon=0.1)
    cost_matrix = solver.compute_cost_matrix()

    # Compute transport matrix with a small number of iterations
    transport_matrix_few = solver.sinkhorn_algorithm(
        cost_matrix, max_iter=100, tol=1e-4
    )

    # Compute transport matrix with a large number of iterations
    transport_matrix_many = solver.sinkhorn_algorithm(
        cost_matrix, max_iter=1000, tol=1e-4
    )

    # Check if the results are close, indicating convergence
    np.testing.assert_allclose(transport_matrix_few, transport_matrix_many, atol=1e-2)
