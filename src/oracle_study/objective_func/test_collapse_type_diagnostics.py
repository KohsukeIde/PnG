#!/usr/bin/env python3
"""
Step B: Collapse Type Diagnostics

Analyzes OT transport collapse to distinguish between:
1. Numerical collapse: Sinkhorn divergence due to numerical instability
2. UOT collapse: Mass correctly vanishing because poses are poor

Key metrics:
- Iteration count at collapse
- Log-sum-exp scaling factors (log_u, log_v)
- Marginal sums evolution
- Cost matrix statistics

Usage:
    python test_collapse_type_diagnostics.py --n-samples 50
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
)
from src.utils.gaussian_utils import load_gaussians


def load_colmap_cameras(scan_name: str = "scan63"):
    """Load camera data from COLMAP sparse reconstruction."""
    colmap_dir = os.path.join(PROJECT_ROOT, f"data/DTU/{scan_name}/sparse/0")
    cameras = load_cameras_from_colmap(colmap_dir)
    images = load_images_from_colmap(colmap_dir)
    return cameras, images


def get_colmap_camera_params(cameras, images, image_name: str):
    """Get camera parameters for a given image from COLMAP data."""
    image_data = None
    for img_id, img in images.items():
        if img['name'] == image_name:
            image_data = img
            break
    if image_data is None:
        raise ValueError(f"Image {image_name} not found in COLMAP data")
    camera = cameras[image_data['camera_id']]
    K = camera.get_camera_matrix()
    R = quaternion_to_rotation_matrix(
        image_data['qw'], image_data['qx'],
        image_data['qy'], image_data['qz']
    )
    t = np.array([image_data['tx'], image_data['ty'], image_data['tz']])
    return {'K': K, 'R': R, 't': t, 'name': image_name}


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (world-to-camera) from cam1 to cam2."""
    R1 = cam1['R']
    t1 = cam1['t']
    R2 = cam2['R']
    t2 = cam2['t']
    R_rel = R2 @ R1.T
    t_rel = t2 - R2 @ R1.T @ t1
    t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-10)
    return R_rel, t_rel


def random_rotation_matrix(max_angle_deg: float = 90.0) -> np.ndarray:
    """Generate a random rotation matrix with angle up to max_angle_deg."""
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    angle = np.random.uniform(0, np.deg2rad(max_angle_deg))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def random_unit_vector() -> np.ndarray:
    """Generate a random unit vector on the sphere."""
    v = np.random.randn(3)
    return v / (np.linalg.norm(v) + 1e-10)


@dataclass
class CollapseInfo:
    """Detailed collapse diagnostic information."""
    # Basic info
    R_err: float
    t_err: float
    collapsed: bool

    # Final transport stats
    T_sum: float
    T_max: float
    T_min_nonzero: float

    # Cost matrix stats
    C_mean: float
    C_std: float
    C_min: float
    C_max: float

    # Sinkhorn diagnostics
    n_iterations: int
    final_log_u_range: float  # max - min of log(u)
    final_log_v_range: float  # max - min of log(v)
    final_row_sum_range: Tuple[float, float]  # (min, max) of row sums
    final_col_sum_range: Tuple[float, float]  # (min, max) of col sums

    # Collapse classification
    collapse_type: str  # "none", "numerical", "uot"


def sinkhorn_with_diagnostics(
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float = 0.05,
    rho: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, Dict]:
    """
    Unbalanced Sinkhorn with detailed diagnostics.

    Returns transport matrix and diagnostic info dict.
    """
    n, m = cost_matrix.shape
    device = cost_matrix.device
    dtype = cost_matrix.dtype

    # Initialize log-domain scaling factors
    log_u = torch.zeros(n, device=device, dtype=dtype)
    log_v = torch.zeros(m, device=device, dtype=dtype)

    # Kernel
    K = torch.exp(-cost_matrix / epsilon)

    # UOT scaling
    tau = rho / (rho + epsilon)

    diagnostics = {
        'iterations': 0,
        'converged': False,
        'log_u_history': [],
        'log_v_history': [],
        'row_sum_history': [],
        'col_sum_history': [],
        'T_sum_history': [],
    }

    for it in range(max_iter):
        # Store previous for convergence check
        log_u_prev = log_u.clone()

        # Update log_u
        log_Kv = torch.logsumexp(torch.log(K + 1e-300) + log_v.unsqueeze(0), dim=1)
        log_u = tau * (torch.log(a + 1e-300) - log_Kv)

        # Update log_v
        log_Ku = torch.logsumexp(torch.log(K + 1e-300) + log_u.unsqueeze(1), dim=0)
        log_v = tau * (torch.log(b + 1e-300) - log_Ku)

        # Compute transport for diagnostics
        T = torch.exp(log_u.unsqueeze(1) + torch.log(K + 1e-300) + log_v.unsqueeze(0))

        # Store history (every 10 iterations to save memory)
        if it % 10 == 0 or it < 10:
            diagnostics['log_u_history'].append(log_u.clone())
            diagnostics['log_v_history'].append(log_v.clone())
            diagnostics['row_sum_history'].append(T.sum(dim=1).clone())
            diagnostics['col_sum_history'].append(T.sum(dim=0).clone())
            diagnostics['T_sum_history'].append(T.sum().item())

        # Convergence check
        diff = torch.abs(log_u - log_u_prev).max().item()
        if diff < tol:
            diagnostics['converged'] = True
            diagnostics['iterations'] = it + 1
            break
    else:
        diagnostics['iterations'] = max_iter

    # Final transport
    T = torch.exp(log_u.unsqueeze(1) + torch.log(K + 1e-300) + log_v.unsqueeze(0))

    # Store final diagnostics
    diagnostics['final_log_u'] = log_u
    diagnostics['final_log_v'] = log_v
    diagnostics['final_T'] = T

    return T, diagnostics


def classify_collapse(
    T_sum: float,
    log_u_range: float,
    log_v_range: float,
    C_mean: float,
    threshold_mass: float = 0.1,
    threshold_log_range: float = 50.0,  # Log scaling factor range indicating numerical issues
) -> str:
    """
    Classify collapse type.

    Returns:
        "none": No collapse (T_sum >= threshold)
        "numerical": Collapse due to extreme log-scaling (numerical instability)
        "uot": Collapse due to UOT correctly rejecting matches (high cost)
    """
    if T_sum >= threshold_mass:
        return "none"

    # Check for numerical instability: very large log-scaling ranges
    if log_u_range > threshold_log_range or log_v_range > threshold_log_range:
        return "numerical"

    # Otherwise, UOT is correctly rejecting due to high cost
    return "uot"


def run_collapse_diagnostics(
    solver: OptimalTransportSolver,
    R: np.ndarray,
    t: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    epsilon: float = 0.05,
    rho: float = 0.5,
) -> CollapseInfo:
    """Run Sinkhorn with diagnostics and classify collapse.

    Uses solver's built-in Sinkhorn for stability, then analyzes results.
    """

    R_t = torch.tensor(R, dtype=torch.float32)
    t_t = torch.tensor(t, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    # Build cost matrix
    F = solver._build_F_from_wc(R_t, t_t)
    C = solver.compute_cost_matrix(F)

    # Use solver's built-in Sinkhorn (more numerically stable)
    with torch.no_grad():
        T, sinkhorn_info = solver.unbalanced_sinkhorn_algorithm(
            C, epsilon=epsilon, rho=rho,
            gate_mask=solver._last_gate_mask
        )

    # Compute stats
    T_sum = T.sum().item()
    T_max = T.max().item() if not torch.isnan(T).all() else 0.0
    T_nonzero = T[T > 0]
    T_min_nonzero = T_nonzero.min().item() if len(T_nonzero) > 0 else 0.0

    C_np = C.detach().numpy()
    C_mean = C_np.mean()
    C_std = C_np.std()
    C_min = C_np.min()
    C_max = C_np.max()

    row_sums = T.sum(dim=1)
    col_sums = T.sum(dim=0)

    # Collapse classification based on T_sum and cost statistics
    # Numerical collapse: T_sum is NaN or C has extreme values
    # UOT collapse: T_sum < threshold with normal C values
    if np.isnan(T_sum) or np.isinf(T_sum):
        collapse_type = "numerical"
        collapsed = True
        T_sum = 0.0  # Replace NaN for analysis
    elif T_sum < 0.1:
        # Check if collapse is due to high cost (UOT) or numerical issues
        if C_mean > 2.0 or C_max > 10.0:  # High cost -> UOT rejecting
            collapse_type = "uot"
        else:
            collapse_type = "numerical"  # Low cost but still collapsed -> numerical
        collapsed = True
    else:
        collapse_type = "none"
        collapsed = False

    # GT errors
    R_err = rotation_error(R, R_gt)
    t_err_pos = rotation_error_t(t, t_gt)
    t_err_neg = rotation_error_t(-t, t_gt)
    t_err = min(t_err_pos, t_err_neg)

    return CollapseInfo(
        R_err=R_err,
        t_err=t_err,
        collapsed=collapsed,
        T_sum=T_sum,
        T_max=T_max,
        T_min_nonzero=T_min_nonzero,
        C_mean=C_mean,
        C_std=C_std,
        C_min=C_min,
        C_max=C_max,
        n_iterations=sinkhorn_info.get('iterations', -1) if isinstance(sinkhorn_info, dict) else -1,
        final_log_u_range=0.0,  # Not available from solver's Sinkhorn
        final_log_v_range=0.0,
        final_row_sum_range=(row_sums.min().item(), row_sums.max().item()) if not torch.isnan(row_sums).all() else (0, 0),
        final_col_sum_range=(col_sums.min().item(), col_sums.max().item()) if not torch.isnan(col_sums).all() else (0, 0),
        collapse_type=collapse_type,
    )


def rotation_error_t(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation direction error in degrees."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def run_step_b_analysis(
    idx1: int = 0,
    idx2: int = 10,
    n_samples: int = 50,
    epsilon: float = 0.05,
    rho: float = 0.5,
):
    """Run Step B collapse diagnostics."""

    print("\n" + "=" * 70)
    print("Step B: Collapse Type Diagnostics")
    print(f"  Image pair: ({idx1}, {idx2})")
    print(f"  n_samples: {n_samples}")
    print(f"  epsilon: {epsilon}, rho: {rho}")
    print("=" * 70)

    # Load data
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']

    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Generate random samples
    print(f"\nGenerating {n_samples} random poses...")

    results: List[CollapseInfo] = []

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  Processing {i}/{n_samples}...")

        R = random_rotation_matrix(90.0)
        t = random_unit_vector()

        info = run_collapse_diagnostics(solver, R, t, R_gt, t_gt, epsilon, rho)
        results.append(info)

    # Also test GT pose
    print("  Testing GT pose...")
    gt_info = run_collapse_diagnostics(solver, R_gt, t_gt, R_gt, t_gt, epsilon, rho)
    results.append(gt_info)

    # Analysis
    print("\n" + "-" * 50)
    print("Collapse Statistics")
    print("-" * 50)

    n_total = len(results)
    n_collapsed = sum(1 for r in results if r.collapsed)
    n_numerical = sum(1 for r in results if r.collapse_type == "numerical")
    n_uot = sum(1 for r in results if r.collapse_type == "uot")
    n_none = sum(1 for r in results if r.collapse_type == "none")

    print(f"\nTotal samples: {n_total}")
    print(f"  Collapsed (T_sum < 0.1): {n_collapsed} ({100*n_collapsed/n_total:.1f}%)")
    print(f"    - Numerical (NaN or low-cost collapse): {n_numerical} ({100*n_numerical/n_total:.1f}%)")
    print(f"    - UOT (high-cost rejection): {n_uot} ({100*n_uot/n_total:.1f}%)")
    print(f"  Not collapsed: {n_none} ({100*n_none/n_total:.1f}%)")

    # T_sum distribution
    T_sums = np.array([r.T_sum for r in results])
    print(f"\nT_sum distribution:")
    print(f"  min={T_sums.min():.3f}, max={T_sums.max():.3f}, mean={T_sums.mean():.3f}, std={T_sums.std():.3f}")
    print(f"  T_sum < 0.5: {sum(T_sums < 0.5)}")
    print(f"  T_sum < 1.0: {sum(T_sums < 1.0)}")
    print(f"  T_sum >= 1.0: {sum(T_sums >= 1.0)}")

    # GT pose result
    print(f"\nGT pose: collapse_type={gt_info.collapse_type}, T_sum={gt_info.T_sum:.4f}")

    # Analyze collapsed vs non-collapsed
    collapsed = [r for r in results if r.collapsed]
    non_collapsed = [r for r in results if not r.collapsed]

    print("\n" + "-" * 50)
    print("Collapsed vs Non-Collapsed Comparison")
    print("-" * 50)

    if collapsed and non_collapsed:
        print(f"\n{'Metric':<25} {'Collapsed (mean±std)':<25} {'Non-collapsed (mean±std)':<25}")
        print("-" * 75)

        metrics = [
            ('R_err', 'R_err'),
            ('C_mean', 'C_mean'),
            ('C_std', 'C_std'),
            ('log_u_range', 'final_log_u_range'),
            ('log_v_range', 'final_log_v_range'),
            ('n_iterations', 'n_iterations'),
        ]

        for name, attr in metrics:
            c_vals = [getattr(r, attr) for r in collapsed]
            nc_vals = [getattr(r, attr) for r in non_collapsed]
            c_mean, c_std = np.mean(c_vals), np.std(c_vals)
            nc_mean, nc_std = np.mean(nc_vals), np.std(nc_vals)
            print(f"{name:<25} {c_mean:>10.2f} ± {c_std:<10.2f} {nc_mean:>10.2f} ± {nc_std:<10.2f}")

    # Analyze numerical vs UOT collapse
    numerical = [r for r in results if r.collapse_type == "numerical"]
    uot = [r for r in results if r.collapse_type == "uot"]

    if numerical and uot:
        print("\n" + "-" * 50)
        print("Numerical vs UOT Collapse Comparison")
        print("-" * 50)

        print(f"\n{'Metric':<25} {'Numerical (mean±std)':<25} {'UOT (mean±std)':<25}")
        print("-" * 75)

        for name, attr in metrics:
            n_vals = [getattr(r, attr) for r in numerical]
            u_vals = [getattr(r, attr) for r in uot]
            n_mean, n_std = np.mean(n_vals), np.std(n_vals)
            u_mean, u_std = np.mean(u_vals), np.std(u_vals)
            print(f"{name:<25} {n_mean:>10.2f} ± {n_std:<10.2f} {u_mean:>10.2f} ± {u_std:<10.2f}")

    # R_err distribution by collapse type
    print("\n" + "-" * 50)
    print("R_err Distribution by Collapse Type")
    print("-" * 50)

    for ctype in ["none", "numerical", "uot"]:
        subset = [r for r in results if r.collapse_type == ctype]
        if subset:
            r_errs = [r.R_err for r in subset]
            print(f"\n{ctype}: n={len(subset)}")
            print(f"  R_err: min={min(r_errs):.1f}, max={max(r_errs):.1f}, "
                  f"mean={np.mean(r_errs):.1f}, std={np.std(r_errs):.1f}")

    # T_sum vs R_err correlation
    from scipy import stats
    R_errs = np.array([r.R_err for r in results])
    C_means = np.array([r.C_mean for r in results])
    valid_mask = ~(np.isnan(T_sums) | np.isnan(R_errs))
    if valid_mask.sum() > 10:
        corr_T_R, p_T_R = stats.spearmanr(T_sums[valid_mask], R_errs[valid_mask])
        corr_C_R, p_C_R = stats.spearmanr(C_means[valid_mask], R_errs[valid_mask])
        print(f"\nCorrelations:")
        print(f"  Spearman(T_sum, R_err): {corr_T_R:.3f} (p={p_T_R:.2e})")
        print(f"  Spearman(C_mean, R_err): {corr_C_R:.3f} (p={p_C_R:.2e})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step B: Collapse Type Diagnostics")
    parser.add_argument("--idx1", type=int, default=0, help="First image index")
    parser.add_argument("--idx2", type=int, default=10, help="Second image index")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of random samples")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--rho", type=float, default=0.5, help="Unbalanced OT rho")
    args = parser.parse_args()

    run_step_b_analysis(
        idx1=args.idx1,
        idx2=args.idx2,
        n_samples=args.n_samples,
        epsilon=args.epsilon,
        rho=args.rho,
    )
