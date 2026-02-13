"""
Step C: OT with Fixed Pose Experiment

Purpose: Verify that cost matrix → OT correspondence is reasonable.

Procedure:
1. Fix pose (e.g., initial value or known near-optimal)
2. Compute C = compute_cost_matrix(F)
3. Compute T = sinkhorn(C)
4. For each i, get j = argmax T[i,j] and check:
   - Epipolar residual (Sampson/SED)
   - Color difference
   - Statistics to see if correspondence is reasonable

If "top correspondence is epipolarically bad", then cost composition/scale/gate is wrong.
If "reasonably good", proceed to pose optimization.
"""

import numpy as np
import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver


def create_test_gaussians(K: int = 32, seed: int = 42) -> TwoDGaussians:
    """Create random 2D Gaussians for testing."""
    np.random.seed(seed)

    means = np.random.uniform([100, 100], [540, 380], size=(K, 2))
    scales = np.random.uniform(5, 30, size=(K, 2))
    rotations = np.random.uniform(-np.pi, np.pi, size=(K,))
    rgb = np.random.uniform(0, 1, size=(K, 3))
    alpha = np.random.uniform(0.5, 2.0, size=(K,))

    covs = np.zeros((K, 2, 2), dtype=np.float64)
    for i in range(K):
        c, s = np.cos(rotations[i]), np.sin(rotations[i])
        R = np.array([[c, -s], [s, c]])
        S = np.diag(scales[i] ** 2)
        covs[i] = R @ S @ R.T

    return TwoDGaussians(
        means=means.astype(np.float64),
        covs=covs,
        scales=scales.astype(np.float64),
        rotations=rotations.astype(np.float64),
        rgb=rgb.astype(np.float64),
        alpha=alpha.astype(np.float64)
    )


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def create_test_pose(angle: float = 0.1):
    """Create test R, t for camera pose."""
    axis = np.array([0, 1, 0], dtype=np.float64)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t = np.array([0.5, 0.0, 0.1], dtype=np.float64)
    t = t / np.linalg.norm(t)
    return R.astype(np.float64), t.astype(np.float64)


def compute_sampson_distance_batch(
    means1: torch.Tensor, means2: torch.Tensor, F: torch.Tensor
) -> torch.Tensor:
    """Compute Sampson distance for all pairs (K1, K2)."""
    K1, K2 = means1.shape[0], means2.shape[0]

    ones1 = torch.ones(K1, 1, device=means1.device)
    ones2 = torch.ones(K2, 1, device=means2.device)
    p1 = torch.cat([means1, ones1], dim=1)  # (K1, 3)
    p2 = torch.cat([means2, ones2], dim=1)  # (K2, 3)

    Fp1 = F @ p1.T  # (3, K1)
    FTp2 = F.T @ p2.T  # (3, K2)

    num = (p2 @ Fp1)  # (K2, K1)
    num = num.T ** 2  # (K1, K2)

    eps = 1e-9
    denom = (Fp1[:2]**2).sum(0).view(-1, 1) + (FTp2[:2]**2).sum(0).view(1, -1)

    sampson = num / (denom + eps)
    return sampson


def analyze_transport_quality(
    solver: OptimalTransportSolver,
    T: torch.Tensor,
    F: torch.Tensor,
    top_k: int = 3
):
    """Analyze the quality of transport plan."""
    K1, K2 = T.shape

    # Get top-k matches for each row
    top_vals, top_idx = torch.topk(T, k=min(top_k, K2), dim=1)

    # Compute Sampson distances
    sampson_all = compute_sampson_distance_batch(solver.means1, solver.means2, F)

    # Color distances
    rgb1 = solver.rgb1  # (K1, 3)
    rgb2 = solver.rgb2  # (K2, 3)
    color_diff_all = ((rgb1.unsqueeze(1) - rgb2.unsqueeze(0)) ** 2).sum(dim=2)  # (K1, K2)

    print("\n" + "-" * 60)
    print("Transport Quality Analysis")
    print("-" * 60)

    # Statistics for top-1 matches
    top1_idx = top_idx[:, 0]  # (K1,)
    top1_sampson = sampson_all[torch.arange(K1), top1_idx]
    top1_color = color_diff_all[torch.arange(K1), top1_idx]
    top1_weight = top_vals[:, 0]

    print(f"\nTop-1 match statistics (K1={K1}):")
    print(f"  Sampson distance: mean={top1_sampson.mean().item():.4f}, "
          f"median={top1_sampson.median().item():.4f}, "
          f"max={top1_sampson.max().item():.4f}")
    print(f"  Color distance: mean={top1_color.mean().item():.4f}, "
          f"median={top1_color.median().item():.4f}, "
          f"max={top1_color.max().item():.4f}")
    print(f"  Transport weight: mean={top1_weight.mean().item():.6f}, "
          f"median={top1_weight.median().item():.6f}")

    # Check how many top-1 matches are "good" (Sampson < threshold)
    thresholds = [1.0, 5.0, 10.0, 50.0]
    print(f"\n  Fraction of top-1 matches with Sampson < threshold:")
    for thresh in thresholds:
        frac = (top1_sampson < thresh).float().mean().item()
        print(f"    Sampson < {thresh:5.1f}: {frac*100:.1f}%")

    # Show some example matches
    print(f"\n  Example top-1 matches (first 5):")
    for i in range(min(5, K1)):
        j = top1_idx[i].item()
        s = top1_sampson[i].item()
        c = top1_color[i].item()
        w = top1_weight[i].item()
        print(f"    {i} -> {j}: Sampson={s:.4f}, color={c:.4f}, weight={w:.6f}")

    return {
        'top1_sampson_mean': top1_sampson.mean().item(),
        'top1_sampson_median': top1_sampson.median().item(),
        'top1_color_mean': top1_color.mean().item(),
        'top1_weight_mean': top1_weight.mean().item(),
    }


def test_ot_fixed_pose_sampson():
    """Test OT with fixed pose using Sampson mode."""
    print("=" * 60)
    print("Test: OT with Fixed Pose (Sampson mode)")
    print("=" * 60)

    K = 64
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    # Camera intrinsics
    K_mat = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Test with different configurations
    configs = [
        {"epipolar_mode": "sampson", "lambda_color": 0.0, "lambda_cov": 0.0, "rho": 1000.0},
        {"epipolar_mode": "sampson", "lambda_color": 1.0, "lambda_cov": 0.0, "rho": 1000.0},
        {"epipolar_mode": "sed", "lambda_color": 0.0, "lambda_cov": 0.0, "rho": 1000.0},
        {"epipolar_mode": "sed", "lambda_color": 1.0, "lambda_cov": 0.0, "rho": 1000.0},
    ]

    R_wc, t_wc = create_test_pose(angle=0.1)
    print(f"\nFixed pose: angle=0.1 rad around Y axis")

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config}")

        solver = OptimalTransportSolver(
            gaussians1=g1,
            gaussians2=g2,
            k1=K_mat,
            k2=K_mat,
            device="cpu",
            epipolar_mode=config["epipolar_mode"],
            lambda_color=config["lambda_color"],
            lambda_cov=config["lambda_cov"],
            lambda_epipolar=1.0,
            sigma_epipolar=1.0,
            sigma_color=0.1,
        )

        # Build F from pose
        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
        t_wc_t = torch.tensor(t_wc, dtype=torch.float32)
        F = solver._build_F_from_wc(R_wc_t, t_wc_t)

        # Compute cost matrix
        cost = solver.compute_cost_matrix(F)

        print(f"\nCost matrix stats:")
        print(f"  shape: {cost.shape}")
        print(f"  min: {cost.min().item():.4f}, max: {cost.max().item():.4f}")
        print(f"  mean: {cost.mean().item():.4f}, median: {cost.median().item():.4f}")

        # Compute OT with balanced (large rho)
        epsilon = 1.0
        rho = config["rho"]
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8,
            gate_mask=solver._last_gate_mask,
        )

        print(f"\nTransport stats:")
        print(f"  T.sum(): {T.sum().item():.4f}")
        print(f"  row_sum: mean={T.sum(dim=1).mean().item():.4f}")

        # Analyze quality
        analyze_transport_quality(solver, T, F)


def test_ot_varying_pose():
    """Test OT with varying pose quality."""
    print("\n" + "=" * 60)
    print("Test: OT with Varying Pose Quality")
    print("=" * 60)

    K = 64
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
    )

    # Test with different rotation angles (representing pose quality)
    angles = [0.0, 0.1, 0.5, 1.0, 2.0]  # radians

    print(f"\nVarying rotation angle (larger = further from identity):")
    print(f"{'Angle':>8} {'T.sum()':>10} {'Loss':>12} {'Sampson_mean':>14}")

    epsilon = 1.0
    rho = 1000.0  # Balanced

    for angle in angles:
        R_wc, t_wc = create_test_pose(angle=angle)

        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
        t_wc_t = torch.tensor(t_wc, dtype=torch.float32)
        F = solver._build_F_from_wc(R_wc_t, t_wc_t)

        cost = solver.compute_cost_matrix(F)

        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8,
            gate_mask=solver._last_gate_mask,
        )

        loss = (T * cost).sum().item()
        T_sum = T.sum().item()

        # Compute Sampson for top-1 matches
        sampson_all = compute_sampson_distance_batch(solver.means1, solver.means2, F)
        top1_idx = T.argmax(dim=1)
        top1_sampson = sampson_all[torch.arange(K), top1_idx]
        sampson_mean = top1_sampson.mean().item()

        print(f"{angle:8.2f} {T_sum:10.4f} {loss:12.4f} {sampson_mean:14.4f}")


def test_balanced_vs_unbalanced():
    """Compare balanced vs unbalanced OT behavior."""
    print("\n" + "=" * 60)
    print("Test: Balanced vs Unbalanced OT")
    print("=" * 60)

    K = 64
    g1 = create_test_gaussians(K, seed=42)
    g2 = create_test_gaussians(K, seed=43)

    K_mat = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K_mat,
        k2=K_mat,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
    )

    R_wc, t_wc = create_test_pose(angle=0.5)
    R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_wc, dtype=torch.float32)
    F = solver._build_F_from_wc(R_wc_t, t_wc_t)

    cost = solver.compute_cost_matrix(F)

    epsilon = 1.0
    rho_configs = [
        ("unbalanced", 10.0),
        ("semi-balanced", 100.0),
        ("balanced", 1000.0),
        ("very-balanced", 10000.0),
    ]

    print(f"\nFixed moderate pose (angle=0.5 rad)")
    print(f"\n{'Config':>15} {'rho':>10} {'T.sum()':>10} {'Loss':>12} {'row_err':>10}")

    for name, rho in rho_configs:
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8,
            gate_mask=solver._last_gate_mask,
        )

        loss = (T * cost).sum().item()
        T_sum = T.sum().item()

        a_norm = solver.alpha1 / solver.alpha1.sum()
        row_err = (T.sum(dim=1) - a_norm).abs().mean().item()

        print(f"{name:>15} {rho:10.1f} {T_sum:10.4f} {loss:12.4f} {row_err:10.6f}")


if __name__ == "__main__":
    print("OT with Fixed Pose Experiments")
    print("=" * 60)

    test_ot_fixed_pose_sampson()
    test_ot_varying_pose()
    test_balanced_vs_unbalanced()

    print("\n" + "=" * 60)
    print("All fixed pose tests completed")
    print("=" * 60)
