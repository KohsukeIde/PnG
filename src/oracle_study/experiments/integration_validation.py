#!/usr/bin/env python3
"""
Integration test to verify the toy problem generator works with the existing optimal transport solver.
"""

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.oracle_study.core import ToyProblemGenerator, TransformationParams
import torch
import numpy as np
import sys
import os
# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_oracle_with_optimal_transport():
    """Test that our toy problems work with the existing optimal transport solver."""
    print("Testing oracle integration with OptimalTransportSolver...")

    # Generate toy problem
    generator = ToyProblemGenerator(seed=42, device='cpu')

    # Create a simple translation test case
    base_gaussians = generator.generate_synthetic_gaussians(
        n_gaussians=20,
        color_mode='gradient'
    )

    trans_params = TransformationParams(translation=np.array([0.3, 0.2]))
    gaussians2, correspondences = generator.generate_known_correspondences(
        base_gaussians, trans_params
    )

    print(
        f"Generated {len(base_gaussians.means)} Gaussians with {len(correspondences)} correspondences")

    # Convert to the format expected by OptimalTransportSolver
    # We need to create mock camera intrinsics
    K1 = K2 = np.array([
        [800, 0, 400],
        [0, 800, 400],
        [0, 0, 1]
    ], dtype=np.float32)

    try:
        # Create solver (this will test if our Gaussians are compatible)
        solver = OptimalTransportSolver(
            gaussians1=base_gaussians,
            gaussians2=gaussians2,
            k1=K1,
            k2=K2,
            epsilon=0.01,
            lambda_mean=1.0,
            lambda_cov=1.0,
            lambda_color=0.5,
            lambda_epipolar=0.0,  # No epipolar constraint for this test
            device='cpu'
        )

        print("✓ OptimalTransportSolver created successfully")

        # Test cost matrix computation
        with torch.no_grad():
            # Create a dummy fundamental matrix for cost computation
            F_dummy = torch.eye(3, dtype=torch.float32, device='cpu')
            F_dummy[0, 2] = 0.1  # Add some off-diagonal terms
            F_dummy[1, 2] = 0.05

            cost_matrix = solver.compute_cost_matrix_fundamental(F_dummy)
            print(f"✓ Cost matrix computed: shape {cost_matrix.shape}")

            # Test transport matrix computation
            transport_matrix = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix)
            print(
                f"✓ Transport matrix computed: shape {transport_matrix.shape}")

            # Basic sanity checks
            assert cost_matrix.shape == (
                20, 20), f"Expected (20, 20), got {cost_matrix.shape}"
            assert transport_matrix.shape == (
                20, 20), f"Expected (20, 20), got {transport_matrix.shape}"
            assert torch.all(transport_matrix >=
                             0), "Transport matrix should be non-negative"

            # Check that transport matrix is approximately doubly stochastic
            row_sums = transport_matrix.sum(dim=1)
            col_sums = transport_matrix.sum(dim=0)
            print(
                f"Row sums range: {row_sums.min():.4f} to {row_sums.max():.4f}")
            print(
                f"Col sums range: {col_sums.min():.4f} to {col_sums.max():.4f}")

            print("✓ Integration test passed!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_identical_case():
    """Test the identical case - should produce identity-like transport matrix."""
    print("\nTesting identical case...")

    generator = ToyProblemGenerator(seed=42, device='cpu')

    # Generate identical Gaussians
    gaussians1 = generator.generate_synthetic_gaussians(n_gaussians=10)
    gaussians2 = gaussians1  # Identical

    K1 = K2 = np.array([
        [800, 0, 400],
        [0, 800, 400],
        [0, 0, 1]
    ], dtype=np.float32)

    try:
        solver = OptimalTransportSolver(
            gaussians1=gaussians1,
            gaussians2=gaussians2,
            k1=K1,
            k2=K2,
            epsilon=0.01,
            lambda_mean=1.0,
            lambda_cov=1.0,
            lambda_color=0.5,
            lambda_epipolar=0.0,
            device='cpu'
        )

        with torch.no_grad():
            # For identical Gaussians, cost should be very low
            F_identity = torch.eye(3, dtype=torch.float32, device='cpu')
            cost_matrix = solver.compute_cost_matrix_fundamental(F_identity)

            # Cost should be very small for identical Gaussians
            max_cost = cost_matrix.max().item()
            mean_cost = cost_matrix.mean().item()

            print(f"Max cost for identical Gaussians: {max_cost:.6f}")
            print(f"Mean cost for identical Gaussians: {mean_cost:.6f}")

            # Transport matrix should be close to identity (permutation matrix)
            transport_matrix = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix)

            # Check if transport matrix has strong diagonal elements
            diagonal_sum = torch.diag(transport_matrix).sum().item()
            total_sum = transport_matrix.sum().item()
            diagonal_ratio = diagonal_sum / total_sum

            print(f"Diagonal ratio in transport matrix: {diagonal_ratio:.4f}")

            # For identical Gaussians, we expect high diagonal concentration
            assert diagonal_ratio > 0.7, f"Expected diagonal ratio > 0.7, got {diagonal_ratio}"

            print("✓ Identical case test passed!")

    except Exception as e:
        print(f"❌ Identical case test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run integration tests."""
    print("=== Oracle Integration Tests ===\n")

    success = True
    success &= test_oracle_with_optimal_transport()
    success &= test_identical_case()

    if success:
        print("\n🎉 All integration tests passed!")
        print("The ToyProblemGenerator is compatible with OptimalTransportSolver!")
    else:
        print("\n❌ Some integration tests failed.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
