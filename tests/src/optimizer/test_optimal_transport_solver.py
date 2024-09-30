import numpy as np

from src.optimizer.optimal_transport_solver import OptimalTransportSolver
from src.primitive.twod_gaussians import TwoDGaussians


def generate_positive_definite_covs(k, dim=2):
    """Generate k positive definite covariance matrices of dimension dim."""
    covs = []
    for _ in range(k):
        a = np.random.rand(dim, dim)
        cov = a @ a.T + np.eye(dim)  # Make it symmetric positive definite
        covs.append(cov)
    return np.array(covs)


def test_optimal_transport_solver_initialization():
    """Test the initialization of OptimalTransportSolver with valid input."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=generate_positive_definite_covs(7),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=0.1, lambda_color=0.5
    )
    assert isinstance(solver, OptimalTransportSolver)
    assert solver.epsilon == 0.1
    assert solver.lambda_color == 0.5


def test_compute_cost_matrix_shape():
    """Test if the computed cost matrix has the correct shape."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=generate_positive_definite_covs(7),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    assert cost_matrix.shape == (5, 7)


def test_compute_cost_matrix_simple_case():
    """Test the cost matrix computation with a simple case."""
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
        rgb=np.array([[1, 0, 0], [0, 1, 0]]),
        alpha=np.array([0.5, 0.5]),
    )
    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [2, 2]]),
        covs=np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]]),
        rgb=np.array([[1, 0, 0], [0, 0, 1]]),
        alpha=np.array([0.5, 0.5]),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2, lambda_color=1.0)
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
    # Generate positive definite covariance matrices
    means = np.random.rand(5, 2)
    covs = generate_positive_definite_covs(5)
    rgb = np.random.rand(5, 3)
    alpha = np.random.rand(5)

    gaussians1 = TwoDGaussians(means, covs, rgb, alpha)
    gaussians2 = TwoDGaussians(means.copy(), covs.copy(), rgb.copy(), alpha.copy())

    solver = OptimalTransportSolver(gaussians1, gaussians2, lambda_color=1.0)
    cost_matrix = solver.compute_cost_matrix()
    # print(cost_matrix)
    np.testing.assert_allclose(np.diag(cost_matrix), 0, atol=1e-6)


def test_compute_cost_matrix_similarity():
    """Test the similarity of the cost matrix when swapping input order."""
    np.random.seed(42)  # Set random seed for reproducibility
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )

    solver_ab = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix_ab = solver_ab.compute_cost_matrix()

    solver_ba = OptimalTransportSolver(gaussians2, gaussians1)
    cost_matrix_ba = solver_ba.compute_cost_matrix()

    print("Cost matrix AB:")
    print(cost_matrix_ab)
    print("\nCost matrix BA (transposed):")
    print(cost_matrix_ba.T)
    print("\nAbsolute Difference:")
    diff = np.abs(cost_matrix_ab - cost_matrix_ba.T)
    print(diff)

    # Check that the maximum absolute difference is within a reasonable tolerance
    assert np.max(diff) < 0.2, f"Maximum difference {np.max(diff)} exceeds tolerance"

    # Check that the average absolute difference is small
    assert np.mean(diff) < 0.05, f"Mean difference {np.mean(diff)} exceeds tolerance"


def test_sinkhorn_algorithm_shape():
    """Test if the Sinkhorn algorithm returns a matrix of the correct shape."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=np.random.rand(7, 2, 2),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    # Generate a cost matrix (this is just an example, adjust as needed)
    cost_matrix = solver.compute_cost_matrix()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    assert transport_matrix.shape == (5, 7)


def test_sinkhorn_algorithm_properties():
    """Test the properties of the Sinkhorn algorithm output."""
    np.random.seed(42)
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(5, 2),  # number of gaussians = 5
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )

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
    # Define two sets of 2 Gaussians each
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array(
            [
                [[1, 0], [0, 1]],  # Covariance for Gaussian 1
                [[1, 0], [0, 1]],  # Covariance for Gaussian 2
            ]
        ),
        rgb=np.array(
            [
                [1, 0, 0],  # Color for Gaussian 1
                [0, 1, 0],  # Color for Gaussian 2
            ]
        ),
        alpha=np.array([0.5, 0.5]),  # Mixing coefficients
    )

    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [2, 2]]),
        covs=np.array(
            [
                [[1, 0], [0, 1]],  # Covariance for Gaussian 1
                [[1, 0], [0, 1]],  # Covariance for Gaussian 2
            ]
        ),
        rgb=np.array(
            [
                [1, 0, 0],  # Color for Gaussian 1 (same as gaussians1)
                [0, 0, 1],  # Color for Gaussian 2 (different from gaussians1)
            ]
        ),
        alpha=np.array([0.5, 0.5]),
    )

    # Initialize the solver with lambda_color=1.0
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
    
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=generate_positive_definite_covs(5),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2, epsilon=0.1)

    cost_matrix = solver.compute_cost_matrix()

    # Compute transport matrix with a small number of iterations
    transport_matrix_few = solver.sinkhorn_algorithm(
        cost_matrix, max_iter=100, tol=1e-4
    )
    print("Transport Matrix Few Iterations:")
    print(transport_matrix_few)

    # Compute transport matrix with a large number of iterations
    transport_matrix_many = solver.sinkhorn_algorithm(
        cost_matrix, max_iter=1000, tol=1e-4
    )
    print("Transport Matrix Many Iterations:")
    print(transport_matrix_many)

    # Check if the results are close, indicating convergence
    np.testing.assert_allclose(transport_matrix_few, transport_matrix_many, atol=1e-2)
