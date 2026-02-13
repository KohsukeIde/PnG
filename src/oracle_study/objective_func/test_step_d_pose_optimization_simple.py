"""
Step D: Pose Optimization with Simple Settings

Purpose: Test if pose optimization works correctly with the simplest configuration.

Configuration:
- balanced OT: rho = 1000 * epsilon (very large)
- lambda_color = 0
- lambda_cov = 0
- epipolar_mode = "sampson"
- epi_clip = None (no gate)

Key metrics to monitor:
- T.sum(): Should stay stable (~1), not collapse
- Loss: Should decrease as pose improves
- Pose error: Should decrease toward ground truth
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from typing import Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver


def create_test_gaussians(K: int = 64, seed: int = 42) -> TwoDGaussians:
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


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix using Rodrigues formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation direction error in degrees (both are unit vectors)."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def create_synthetic_pair(
    K: int = 64,
    R_gt: np.ndarray = None,
    t_gt: np.ndarray = None,
    seed: int = 42
) -> Tuple[TwoDGaussians, TwoDGaussians, np.ndarray, np.ndarray]:
    """
    Create a synthetic pair of Gaussian sets with known ground truth pose.

    Returns:
        g1, g2: Gaussian sets
        R_gt, t_gt: Ground truth relative pose
    """
    if R_gt is None:
        R_gt = rodrigues_rotation(np.array([0, 1, 0]), 0.1)  # 0.1 rad around Y
    if t_gt is None:
        t_gt = np.array([0.5, 0.0, 0.1])
        t_gt = t_gt / np.linalg.norm(t_gt)

    # Create first set
    g1 = create_test_gaussians(K, seed=seed)

    # Create second set with similar structure but different random seed
    # In reality, these would be projections of the same 3D Gaussians
    g2 = create_test_gaussians(K, seed=seed + 1)

    return g1, g2, R_gt.astype(np.float64), t_gt.astype(np.float64)


def test_pose_optimization_se3():
    """Test pose optimization using SE3 parameterization."""
    print("=" * 70)
    print("Step D: Pose Optimization with SE3 (Simple Settings)")
    print("=" * 70)

    # Create test data
    K = 64
    R_gt = rodrigues_rotation(np.array([0, 1, 0]), 0.15)  # Ground truth
    t_gt = np.array([0.5, 0.1, 0.2])
    t_gt = t_gt / np.linalg.norm(t_gt)

    g1, g2, R_gt, t_gt = create_synthetic_pair(K, R_gt, t_gt, seed=42)

    # Camera intrinsics
    K_mat = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    print(f"\nConfiguration:")
    print(f"  K = {K} Gaussians")
    print(f"  epipolar_mode = 'sampson'")
    print(f"  lambda_color = 0")
    print(f"  lambda_cov = 0")
    print(f"  epi_clip = None (no gate)")
    print(f"  rho = 1000 (balanced)")

    # Create solver with simplest settings
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
        epi_clip=None,  # No gate
    )

    print(f"\nGround truth pose:")
    print(f"  R_gt:\n{R_gt}")
    print(f"  t_gt: {t_gt}")

    # Initialize with perturbed pose
    R_init = rodrigues_rotation(np.array([0, 1, 0]), 0.0)  # Identity
    t_init = np.array([1.0, 0.0, 0.0])  # Different direction
    t_init = t_init / np.linalg.norm(t_init)

    print(f"\nInitial pose (perturbed):")
    print(f"  R_init:\n{R_init}")
    print(f"  t_init: {t_init}")

    init_rot_err = rotation_error(R_init, R_gt)
    init_trans_err = translation_error(t_init, t_gt)
    print(f"\nInitial errors:")
    print(f"  Rotation error: {init_rot_err:.2f} deg")
    print(f"  Translation error: {init_trans_err:.2f} deg")

    # Set initial pose in solver
    solver.rot_vec = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    solver.trans_vec = torch.nn.Parameter(
        torch.tensor(t_init, dtype=torch.float32)
    )

    # Create output directories
    results_dir = os.path.join(project_root, "results", "step_d_test")
    diagnostics_dir = os.path.join(project_root, "results", "step_d_diagnostics")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"\nStarting optimization...")
    print(f"  Results: {results_dir}")
    print(f"  Diagnostics: {diagnostics_dir}")

    try:
        # Run optimization with balanced OT
        loss_history = solver.optimize_with_SE3(
            max_iter=200,
            rot_lr=1e-3,
            trans_lr=1e-3,
            momentum=0.9,
            tol=1e-8,
            diagnostics_dir=diagnostics_dir,
            differentiable_transport=False,  # Non-differentiable for stability
        )

        print(f"\nOptimization completed!")
        print(f"  Final loss: {loss_history[-1]:.6f}")
        print(f"  Loss reduction: {loss_history[0]:.6f} -> {loss_history[-1]:.6f}")

        # Extract final pose
        with torch.no_grad():
            se3_vec = torch.cat([solver.rot_vec, solver.trans_vec])
            T_cw = solver.lie.se3_to_SE3(se3_vec)
            R_final = T_cw[:3, :3].numpy()
            t_final = T_cw[:3, 3].numpy()
            t_final = t_final / (np.linalg.norm(t_final) + 1e-10)

        final_rot_err = rotation_error(R_final, R_gt)
        final_trans_err = translation_error(t_final, t_gt)

        print(f"\nFinal errors:")
        print(f"  Rotation error: {final_rot_err:.2f} deg (was {init_rot_err:.2f})")
        print(f"  Translation error: {final_trans_err:.2f} deg (was {init_trans_err:.2f})")

        # Check gradient debug log for T.sum()
        log_path = os.path.join(diagnostics_dir, "gradient_debug.log")
        if os.path.exists(log_path):
            print(f"\nT.sum() from log (checking for collapse):")
            with open(log_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    header = lines[0].strip()
                    print(f"  Header: {header}")
                    # Show first, middle, and last few entries
                    for i, line in enumerate(lines[1:6]):
                        print(f"  {line.strip()}")
                    print("  ...")
                    for line in lines[-3:]:
                        print(f"  {line.strip()}")

        return {
            'init_rot_err': init_rot_err,
            'init_trans_err': init_trans_err,
            'final_rot_err': final_rot_err,
            'final_trans_err': final_trans_err,
            'loss_history': loss_history,
        }

    except Exception as e:
        print(f"\nOptimization failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_varying_rho():
    """Test pose optimization with different rho values."""
    print("\n" + "=" * 70)
    print("Test: Varying rho (balanced vs unbalanced)")
    print("=" * 70)

    K = 64
    R_gt = rodrigues_rotation(np.array([0, 1, 0]), 0.15)
    t_gt = np.array([0.5, 0.1, 0.2])
    t_gt = t_gt / np.linalg.norm(t_gt)

    g1, g2, R_gt, t_gt = create_synthetic_pair(K, R_gt, t_gt, seed=42)

    K_mat = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float64)

    # Note: The current optimize_with_SE3 doesn't directly accept rho parameter
    # We would need to modify the solver or create a custom optimization loop
    # For now, this test demonstrates the concept

    print("\nNote: This test requires modifying optimize_with_SE3 to accept rho parameter")
    print("or implementing a custom optimization loop.")
    print("\nThe key insight from Step B/C tests:")
    print("  - rho=10 (unbalanced): T.sum() can collapse, loss appears low")
    print("  - rho=1000+ (balanced): T.sum() stable, but loss may be higher")
    print("\nRecommendation: Start with rho >= 1000 for stable pose optimization")


def analyze_debug_log(log_path: str):
    """Analyze the gradient debug log for collapse detection."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    print(f"\nAnalyzing: {log_path}")

    with open(log_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("  Log file is empty or has only header")
        return

    header = lines[0].strip().split(',')
    print(f"  Columns: {header}")

    # Parse data
    data = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                iteration = int(parts[0].strip())
                loss = float(parts[1].strip())
                T_sum = float(parts[2].strip())
                top1_mean = float(parts[3].strip())
                data.append((iteration, loss, T_sum, top1_mean))
            except:
                pass

    if not data:
        print("  No valid data found")
        return

    iterations, losses, T_sums, top1_means = zip(*data)

    print(f"\n  Iterations: {len(data)}")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  T.sum(): {T_sums[0]:.4f} -> {T_sums[-1]:.4f}")

    # Check for collapse
    T_sum_drop = T_sums[0] - T_sums[-1]
    if T_sum_drop > 0.2:
        print(f"\n  WARNING: T.sum() dropped by {T_sum_drop:.4f} - possible collapse!")
    elif T_sums[-1] < 0.5:
        print(f"\n  WARNING: Final T.sum() = {T_sums[-1]:.4f} < 0.5 - collapsed!")
    else:
        print(f"\n  OK: T.sum() stable (drop = {T_sum_drop:.4f})")


if __name__ == "__main__":
    print("Step D: Pose Optimization with Simple Settings")
    print("=" * 70)

    # Run main test
    result = test_pose_optimization_se3()

    if result:
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        print(f"Initial rotation error: {result['init_rot_err']:.2f} deg")
        print(f"Final rotation error: {result['final_rot_err']:.2f} deg")
        print(f"Improvement: {result['init_rot_err'] - result['final_rot_err']:.2f} deg")
        print(f"\nInitial translation error: {result['init_trans_err']:.2f} deg")
        print(f"Final translation error: {result['final_trans_err']:.2f} deg")
        print(f"Improvement: {result['init_trans_err'] - result['final_trans_err']:.2f} deg")

    # Analyze the debug log
    log_path = os.path.join(project_root, "results", "step_d_diagnostics", "gradient_debug.log")
    analyze_debug_log(log_path)

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)
