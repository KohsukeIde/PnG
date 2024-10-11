import numpy as np
import os
import tempfile
from src.evaluator.matching_evaluator import MatchingEvaluator
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
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
        R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        S = np.diag(s ** 2)
        covs[i] = R @ S @ R.T
    return covs


def test_matching_evaluator_initialization():
    """Test if MatchingEvaluator can be initialized with random TwoDGaussians and transport_matrix."""
    np.random.seed(42)
    k = 5
    # Generate random parameters
    means1 = np.random.rand(k, 2)
    rgb1 = np.random.rand(k, 3)
    alpha1 = np.random.rand(k)
    rotations1 = np.random.uniform(0, 2 * np.pi, k)
    scales1 = np.random.rand(k, 2) + 0.1  # Avoid zero scales
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)

    means2 = np.random.rand(k, 2)
    rgb2 = np.random.rand(k, 3)
    alpha2 = np.random.rand(k)
    rotations2 = np.random.uniform(0, 2 * np.pi, k)
    scales2 = np.random.rand(k, 2) + 0.1
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

    # Initialize OptimalTransportSolver
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=0.01, lambda_color=0.0
    )

    # Compute cost matrix
    cost_matrix = solver.compute_cost_matrix()

    # Compute transport_matrix via Sinkhorn algorithm
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # Initialize MatchingEvaluator with computed transport_matrix
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)
    assert isinstance(evaluator, MatchingEvaluator), "Failed to create an instance of MatchingEvaluator."
    
    
def test_matching_evaluator_extract_matches():
    """Test if extract_matches() returns a list of tuples with correct indices."""
    # Define Gaussians with identity covariance matrices
    means1 = np.array([[0, 0], [1, 1]])
    scales1 = np.array([[1, 1], [1, 1]])
    rotations1 = np.array([0.0, 0.0])
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    alpha1 = np.array([0.5, 0.5])

    means2 = np.array([[0, 0], [2, 2]])
    scales2 = np.array([[1, 1], [1, 1]])
    rotations2 = np.array([0.0, 0.0])
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.array([
        [1, 0, 0],
        [0, 0, 1]
    ])
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

    # Initialize OptimalTransportSolver with lambda_color=0 and epsilon=1.0
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # Compute cost matrix
    cost_matrix = solver.compute_cost_matrix()
    print("Cost Matrix:")
    print(cost_matrix)

    # Compute transport_matrix via Sinkhorn algorithm
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    print("Transport Matrix:")
    print(transport_matrix)

    # Initialize MatchingEvaluator with computed transport_matrix
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    # Extract matches
    matches = evaluator.matches
    print("Extracted Matches:")
    print(matches)

    # Define expected_matches based on the input Gaussians and transport_matrix computation
    expected_matches = [(0, 0), (1, 1)]  # Assuming optimal transport aligns as such

    assert matches == expected_matches, f"Matching pairs {matches} differ from expected pairs {expected_matches}"


def test_matching_evaluator_evaluate_matches():
    """Test if evaluate_matches() returns correct metrics."""
    means1 = np.array([[0, 0], [1, 1]])
    scales1 = np.array([[1, 1], [1, 1]])
    rotations1 = np.array([0.0, 0.0])
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    alpha1 = np.array([0.5, 0.5])

    means2 = np.array([[0, 0], [2, 2]])
    scales2 = np.array([[1, 1], [1, 1]])
    rotations2 = np.array([0.0, 0.0])
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.array([
        [1, 0, 0],
        [0, 0, 1]
    ])
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

    # Initialize OptimalTransportSolver with lambda_color=0 and epsilon=1.0
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # Compute cost matrix
    cost_matrix = solver.compute_cost_matrix()
    print("Cost Matrix:")
    print(cost_matrix)

    # Compute transport_matrix via Sinkhorn algorithm
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    print("Transport Matrix:")
    print(transport_matrix)

    # Initialize MatchingEvaluator with computed transport_matrix
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)
    metrics = evaluator.evaluate_matches()

    # Calculate expected metrics
    # Matches are [(0,0), (1,1)]
    distance_0 = np.linalg.norm(gaussians1.means[0] - gaussians2.means[0])  # 0
    distance_1 = np.linalg.norm(gaussians1.means[1] - gaussians2.means[1])  # sqrt(2)
    expected_distance = (distance_0 + distance_1) / 2  # (0 + sqrt(2)) / 2

    color_diff_0 = np.linalg.norm(gaussians1.rgb[0] - gaussians2.rgb[0])  # 0
    color_diff_1 = np.linalg.norm(gaussians1.rgb[1] - gaussians2.rgb[1])  # sqrt(2)
    expected_color_diff = (color_diff_0 + color_diff_1) / 2  # (0 + sqrt(2)) / 2

    expected_matching_rate = 2 / ((gaussians1.k + gaussians2.k) / 2)  # 2 / 2 = 1.0

    print("Computed Metrics:")
    print(metrics)
    print("Expected Metrics:")
    print({
        "average_distance": expected_distance,
        "average_color_difference": expected_color_diff,
        "matching_rate": expected_matching_rate
    })

    # Assert metrics
    np.testing.assert_allclose(
        metrics["average_distance"], expected_distance, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["average_color_difference"], expected_color_diff, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["matching_rate"], expected_matching_rate, atol=1e-6
    )
def test_visualize_complete_match():
    """Test visualize_matches() with a complete match case."""
    # Define Gaussians with identical parameters
    means1 = np.array([[0, 0], [1, 1]])
    rotations1 = np.array([0.0, 0.0])  # No rotation
    scales1 = np.array([[1, 1], [1, 1]])  # Unit scales
    covs1 = generate_covariances_from_rotations_and_scales(rotations1, scales1)
    rgb1 = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0]   # Green
    ])
    alpha1 = np.array([0.5, 0.5])

    means2 = np.array([[0, 0], [1, 1]])
    rotations2 = np.array([0.0, 0.0])  # No rotation
    scales2 = np.array([[1, 1], [1, 1]])  # Unit scales
    covs2 = generate_covariances_from_rotations_and_scales(rotations2, scales2)
    rgb2 = np.array([
        [1, 0, 0],  # Red (same as gaussians1)
        [0, 1, 0]   # Green (same as gaussians1)
    ])
    alpha2 = np.array([0.5, 0.5])

    # Initialize TwoDGaussians instances
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

    # Initialize OptimalTransportSolver
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # Compute cost matrix
    cost_matrix = solver.compute_cost_matrix()
    print("Cost Matrix:")
    print(cost_matrix)

    # Compute transport_matrix via Sinkhorn algorithm
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    print("Transport Matrix:")
    print(transport_matrix)

    # Initialize MatchingEvaluator with computed transport_matrix
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    # Define output path (Use a temporary directory for testing)
    output_path = "/Users/kohsukeide/dev/perspective-n-gaussian/outputs/complete_match_visualization_rs.png"
    evaluator.visualize_matches(output_path)
    print(f"Visualization saved to {output_path}")

    # Assert that the image file was created
    assert os.path.exists(output_path), "Visualization image was not saved at the specified path."