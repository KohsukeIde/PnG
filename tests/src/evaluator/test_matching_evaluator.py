import numpy as np
import os
import tempfile
from src.evaluator.matching_evaluator import MatchingEvaluator
from src.primitive.twod_gaussians import TwoDGaussians
from src.optimizer.optimal_transport_solver import OptimalTransportSolver


def generate_positive_definite_covs(k, dim=2):
    """Generate k positive definite covariance matrices of dimension dim."""
    covs = []
    for _ in range(k):
        a = np.random.rand(dim, dim)
        cov = a @ a.T + np.eye(dim)  # Make it symmetric positive definite
        covs.append(cov)
    return np.array(covs)


def test_matching_evaluator_initialization():
    """Test if MatchingEvaluator can be initialized with random TwoDGaussians and transport_matrix."""
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
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [2, 2]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 0, 1]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    # Initialize OptimalTransportSolver with lambda_color=0 and epsilon=1.0
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0  # Changed epsilon to 1.0
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
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [2, 2]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 0, 1]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    # Initialize OptimalTransportSolver with lambda_color=0 and epsilon=1.0
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0  # Changed epsilon to 1.0
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


    
def test_matching_evaluator_evaluate_matches_complete_match():
    """Test evaluate_matches() with a complete match case."""
    # Complete match case
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )

    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]), 
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
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
    print("Extracted Matches:", matches)

    # Ensure matches are of type int
    matches = [(int(i), int(j)) for i, j in matches]

    # Define expected_matches based on the input Gaussians and transport_matrix computation
    expected_matches = [(0, 0), (1, 1)]  

    assert matches == expected_matches, f"Matching pairs {matches} differ from expected pairs {expected_matches}"

    # Evaluate matches
    metrics = evaluator.evaluate_matches()

    expected_distance = (0.0 + np.linalg.norm(np.array([1,1]) - np.array([1,1]))) / 2  # (0 + 0) / 2 = 0.0
    expected_color_diff = (0.0 + np.linalg.norm(np.array([0,1,0]) - np.array([0,1,0]))) / 2  # (0 + 0) / 2 = 0.0
    expected_matching_rate = 2 / ((2 + 2) / 2)  # 2 / 2 = 1.0

    np.testing.assert_allclose(
        metrics["average_distance"], expected_distance, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["average_color_difference"], expected_color_diff, atol=1e-6
    )
    np.testing.assert_allclose(
        metrics["matching_rate"], expected_matching_rate, atol=1e-6
    )

def test_matching_evaluator_evaluate_matches_complete_mismatch():
    """Test evaluate_matches() with a complete mismatch case."""
    # Complete mismatch case
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0]]),
        covs=np.array([[[1, 0], [0, 1]]]),
        rgb=np.array([[1, 0, 0]]),
        alpha=np.array([1.0]),
    )

    gaussians2 = TwoDGaussians(
        means=np.array([[10, 10]]),
        covs=np.array([[[1, 0], [0, 1]]]),
        rgb=np.array([[0, 1, 0]]),
        alpha=np.array([1.0]),
    )

    # Initialize OptimalTransportSolver with lambda_color=0 and epsilon=1.0
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # Compute cost matrix
    cost_matrix = solver.compute_cost_matrix()

    # Compute transport_matrix via Sinkhorn algorithm
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # Initialize MatchingEvaluator with computed transport_matrix
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)
    metrics = evaluator.evaluate_matches()

    expected_metrics = {
        "average_distance": float('inf'),
        "average_color_difference": float('inf'),
        "matching_rate": 0.0
    }

    np.testing.assert_equal(metrics, expected_metrics)

def test_matching_evaluator_visualize_matches():
    """Test if visualize_matches() generates an image at the specified path."""
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [2, 2]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 0, 1]
        ]),
        alpha=np.array([0.5, 0.5]),
    )
    
    transport_matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)
    
    # Create a temporary directory to store the output image
    with tempfile.TemporaryDirectory() as tmpdirname:
        output_path = os.path.join(tmpdirname, "matches.png")
        evaluator.visualize_matches(output_path)
        
        assert os.path.exists(output_path), "Visualization image was not saved at the specified path."
        
def test_visualize_complete_match():
    """Test visualize_matches() with a complete match case."""
    # 完全一致ケースの設定
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )

    gaussians2 = TwoDGaussians(
        means=np.array([[0, 0], [1, 1]]),
        covs=np.array([
            [[1, 0], [0, 1]],
            [[1, 0], [0, 1]]
        ]),
        rgb=np.array([
            [1, 0, 0],
            [0, 1, 0]
        ]),
        alpha=np.array([0.5, 0.5]),
    )

    # OptimalTransportSolverの初期化
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # コスト行列の計算
    cost_matrix = solver.compute_cost_matrix()

    # Sinkhornアルゴリズムによる輸送行列の計算
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # MatchingEvaluatorの初期化
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    # マッチングの視覚化
    output_path = "/Users/kohsukeide/dev/perspective-n-gaussian/outputs/complete_match_visualization.png"
    evaluator.visualize_matches(output_path)

    print(f"Visualization saved to {output_path}")

def test_visualize_complete_mismatch():
    """Test visualize_matches() with a complete mismatch case."""
    # 完全不一致ケースの設定
    gaussians1 = TwoDGaussians(
        means=np.array([[0, 0]]),
        covs=np.array([[[1, 0], [0, 1]]]),
        rgb=np.array([[1, 0, 0]]),
        alpha=np.array([1.0]),
    )

    gaussians2 = TwoDGaussians(
        means=np.array([[10, 10]]),
        covs=np.array([[[1, 0], [0, 1]]]),
        rgb=np.array([[0, 1, 0]]),
        alpha=np.array([1.0]),
    )

    # OptimalTransportSolverの初期化
    solver = OptimalTransportSolver(
        gaussians1, gaussians2, epsilon=1.0, lambda_color=0.0
    )

    # コスト行列の計算
    cost_matrix = solver.compute_cost_matrix()

    # Sinkhornアルゴリズムによる輸送行列の計算
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # MatchingEvaluatorの初期化
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    # マッチングの視覚化
    output_path = "/Users/kohsukeide/dev/perspective-n-gaussian/outputs/complete_mismatch_visualization.png"
    evaluator.visualize_matches(output_path)

    print(f"Visualization saved to {output_path}")
