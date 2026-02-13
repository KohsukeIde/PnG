"""
Step B: Sinkhorn (OT) Unit Test (without pose optimization)

Purpose: Verify that unbalanced Sinkhorn implementation has expected properties.

Test with small K (e.g., K1=K2=32) and arbitrary cost matrix.

Verification items:
1. Balanced approximation: large rho (e.g., rho=1e3 or 1e4), fixed epsilon
   → row_sum ≈ a, col_sum ≈ b
2. Unbalanced: rho = 10*epsilon (default)
   → row_sum/col_sum can deviate from a, b
3. Epsilon effect: increasing epsilon → T becomes more uniform,
   decreasing epsilon → T becomes more peaked
"""

import numpy as np
import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.primitive.twod_gaussians_rs import TwoDGaussians


def create_test_gaussians(K: int = 32, seed: int = 42) -> TwoDGaussians:
    """Create random 2D Gaussians for testing."""
    np.random.seed(seed)

    # Random means in [100, 540] x [100, 380] (image coordinates)
    means = np.random.uniform([100, 100], [540, 380], size=(K, 2))

    # Random scales (standard deviations)
    scales = np.random.uniform(5, 30, size=(K, 2))

    # Random rotations
    rotations = np.random.uniform(-np.pi, np.pi, size=(K,))

    # Random RGB
    rgb = np.random.uniform(0, 1, size=(K, 3))

    # Random alpha (intensities) - ensure positive
    alpha = np.random.uniform(0.5, 2.0, size=(K,))

    # Compute covariance matrices from scales and rotations
    covs = np.zeros((K, 2, 2), dtype=np.float64)
    for i in range(K):
        c, s = np.cos(rotations[i]), np.sin(rotations[i])
        R = np.array([[c, -s], [s, c]])
        S = np.diag(scales[i] ** 2)  # variance = scale^2
        covs[i] = R @ S @ R.T

    return TwoDGaussians(
        means=means.astype(np.float64),
        covs=covs,
        scales=scales.astype(np.float64),
        rotations=rotations.astype(np.float64),
        rgb=rgb.astype(np.float64),
        alpha=alpha.astype(np.float64)
    )


def create_simple_cost_matrix(K1: int, K2: int, seed: int = 42) -> torch.Tensor:
    """Create a simple positive cost matrix for testing."""
    np.random.seed(seed)
    cost = np.random.uniform(0.1, 10.0, size=(K1, K2))
    return torch.tensor(cost, dtype=torch.float32)


def test_balanced_approximation():
    """Test 1: Large rho should approximate balanced OT."""
    print("=" * 60)
    print("Test 1: Balanced Approximation (large rho)")
    print("=" * 60)

    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

    K = 32
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    # Create solver with dummy intrinsics (not used for this test)
    K_mat = np.eye(3) * 500
    K_mat[2, 2] = 1

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu"
    )

    # Create cost matrix
    cost = create_simple_cost_matrix(K, K)

    # Test with different rho values
    epsilon = 1.0
    rho_values = [10.0, 100.0, 1000.0, 10000.0]

    print(f"\nMarginals: a.sum()={solver.alpha1.sum().item():.4f}, b.sum()={solver.alpha2.sum().item():.4f}")
    print(f"epsilon = {epsilon}\n")

    for rho in rho_values:
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8
        )

        row_sum = T.sum(dim=1)
        col_sum = T.sum(dim=0)

        # Compare with target marginals
        a = solver.alpha1 / solver.alpha1.sum()  # normalized
        b = solver.alpha2 / solver.alpha2.sum()

        row_error = (row_sum - a).abs().mean().item()
        col_error = (col_sum - b).abs().mean().item()

        print(f"rho={rho:8.1f}: T.sum()={T.sum().item():.4f}, "
              f"row_err={row_error:.6f}, col_err={col_error:.6f}")

    print("\nExpected: As rho increases, row_err and col_err should decrease (→ balanced)")
    return True


def test_unbalanced_behavior():
    """Test 2: Small rho allows mass deviation."""
    print("\n" + "=" * 60)
    print("Test 2: Unbalanced Behavior (small rho)")
    print("=" * 60)

    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

    K = 32
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.eye(3) * 500
    K_mat[2, 2] = 1

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu"
    )

    # Create cost matrix with some very high costs (simulating bad matches)
    cost = create_simple_cost_matrix(K, K)
    # Make some entries very expensive
    cost[0:5, :] = 100.0  # First 5 rows are expensive

    epsilon = 1.0
    rho_small = 10.0 * epsilon  # default unbalanced
    rho_large = 1000.0 * epsilon  # nearly balanced

    print(f"\nCost matrix has high costs in first 5 rows (simulating bad matches)")
    print(f"epsilon = {epsilon}\n")

    for rho, label in [(rho_small, "unbalanced"), (rho_large, "balanced")]:
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8
        )

        row_sum = T.sum(dim=1)

        print(f"\n{label} (rho={rho:.1f}):")
        print(f"  T.sum() = {T.sum().item():.4f}")
        print(f"  First 5 rows (high cost): row_sum = {row_sum[:5].tolist()}")
        print(f"  Other rows: row_sum mean = {row_sum[5:].mean().item():.4f}")

    print("\nExpected: Unbalanced should have smaller row_sum for high-cost rows")
    return True


def test_epsilon_effect():
    """Test 3: Epsilon controls entropy/smoothness."""
    print("\n" + "=" * 60)
    print("Test 3: Epsilon Effect (entropy/smoothness)")
    print("=" * 60)

    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

    K = 32
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.eye(3) * 500
    K_mat[2, 2] = 1

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu"
    )

    cost = create_simple_cost_matrix(K, K)

    rho = 1000.0  # Large rho for balanced
    epsilon_values = [0.1, 1.0, 10.0, 100.0]

    print(f"\nrho = {rho} (balanced)\n")

    for epsilon in epsilon_values:
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8
        )

        # Compute entropy
        T_norm = T / (T.sum() + 1e-10)
        entropy = -(T_norm * torch.log(T_norm + 1e-10)).sum().item()

        # Compute sparsity (how peaked is T)
        top1_per_row = T.max(dim=1).values
        top1_mean = top1_per_row.mean().item()
        row_sum_mean = T.sum(dim=1).mean().item()
        concentration = top1_mean / (row_sum_mean + 1e-10)

        print(f"epsilon={epsilon:6.1f}: entropy={entropy:.4f}, "
              f"concentration={concentration:.4f} (top1/row_sum)")

    print("\nExpected: Higher epsilon → higher entropy, lower concentration")
    print("          Lower epsilon → lower entropy, higher concentration (more peaked)")
    return True


def test_transport_sum_stability():
    """Test 4: T.sum() should be stable and predictable."""
    print("\n" + "=" * 60)
    print("Test 4: Transport Sum Stability")
    print("=" * 60)

    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

    K = 32
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.eye(3) * 500
    K_mat[2, 2] = 1

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu"
    )

    # Different cost matrices
    costs = [
        ("uniform", torch.ones(K, K)),
        ("random", create_simple_cost_matrix(K, K)),
        ("high_cost", create_simple_cost_matrix(K, K) * 100),
    ]

    epsilon = 1.0
    rho_balanced = 10000.0

    print(f"\nBalanced OT (rho={rho_balanced}, epsilon={epsilon})")
    print(f"Expected T.sum() ≈ min(a.sum(), b.sum()) = "
          f"{min(solver.alpha1.sum().item(), solver.alpha2.sum().item()):.4f}\n")

    for name, cost in costs:
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho_balanced,
            max_iter=500,
            tol=1e-8
        )
        print(f"{name:12s}: T.sum() = {T.sum().item():.4f}")

    return True


def test_collapse_detection():
    """Test 5: Demonstrate collapse scenario."""
    print("\n" + "=" * 60)
    print("Test 5: Collapse Detection Scenario")
    print("=" * 60)

    from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

    K = 32
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.eye(3) * 500
    K_mat[2, 2] = 1

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu"
    )

    # Simulate increasing cost (as if pose is getting worse)
    base_cost = create_simple_cost_matrix(K, K)

    epsilon = 1.0
    rho_unbalanced = 10.0  # Small rho allows collapse

    print(f"\nUnbalanced OT (rho={rho_unbalanced}) with increasing cost")
    print("This simulates pose degradation where OT can 'escape' by reducing T.sum()\n")

    for scale in [1.0, 2.0, 5.0, 10.0, 50.0]:
        cost = base_cost * scale
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho_unbalanced,
            max_iter=500,
            tol=1e-8
        )

        loss = (T * cost).sum().item()
        print(f"cost_scale={scale:5.1f}: T.sum()={T.sum().item():.4f}, "
              f"loss=<T,C>={loss:.4f}")

    print("\nWARNING: If T.sum() decreases as cost increases, this is the collapse mode!")
    print("In pose optimization, this means the loss can decrease by 'giving up' matches.")
    return True


if __name__ == "__main__":
    print("Sinkhorn Properties Unit Test")
    print("=" * 60)

    test_balanced_approximation()
    test_unbalanced_behavior()
    test_epsilon_effect()
    test_transport_sum_stability()
    test_collapse_detection()

    print("\n" + "=" * 60)
    print("All Sinkhorn tests completed")
    print("=" * 60)
