#!/usr/bin/env python3
"""
Step A: Score vs GT Error Correlation Analysis

This script analyzes the correlation between various OT-based scores and
ground truth pose errors (R_err, t_err).

Key questions:
1. Do any scores correlate well with R_err? (Selection feasibility)
2. Do scores correlate with t_err? (t identifiability)
3. What's the best score for non-oracle selection?

Usage:
    python test_score_correlation_analysis.py --n-samples 200
    python test_score_correlation_analysis.py --n-samples 500 --t-mode all
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from scipy import stats
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.primitive.camera import Lie
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


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation direction error in degrees."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


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
class ScoreResult:
    """Container for all computed scores at a given pose."""
    # Basic OT scores
    avg_cost: float           # <T,C> / T.sum
    mass_aware: float         # <T,C> / T.sum + λ * (1 - T.sum)
    full_uot: float           # Full UOT objective with entropy

    # Transport statistics
    T_sum: float              # Total mass
    concentration: float      # mean(row_max / row_sum)
    entropy: float            # -sum(T * log(T))

    # Eigenvalue-based (from closed-form t)
    eigen_gap: float          # (λ2 - λ1) / λ3

    # Top-K statistics
    topk_sum: float           # Sum of top-K transport values
    topk_cost: float          # Avg cost of top-K correspondences

    # GT errors (for correlation)
    R_err: float
    t_err: float


def compute_all_scores(
    solver: OptimalTransportSolver,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    epsilon: float = 0.05,
    rho: float = 0.5,
    top_k: int = 50,
) -> ScoreResult:
    """Compute all scores for a given pose."""

    R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_wc, dtype=torch.float32)
    t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

    # Build F and cost matrix
    F = solver._build_F_from_wc(R_wc_t, t_wc_t)
    cost_matrix = solver.compute_cost_matrix(F)

    # Compute transport
    with torch.no_grad():
        transport, sinkhorn_info = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, epsilon=epsilon, rho=rho,
            gate_mask=solver._last_gate_mask
        )

    T = transport.detach()
    C = cost_matrix.detach()

    # Basic scores
    T_sum = T.sum().item()
    if T_sum < 1e-10:
        # Collapsed - return sentinel values
        return ScoreResult(
            avg_cost=float('inf'),
            mass_aware=float('inf'),
            full_uot=float('inf'),
            T_sum=0.0,
            concentration=0.0,
            entropy=0.0,
            eigen_gap=0.0,
            topk_sum=0.0,
            topk_cost=float('inf'),
            R_err=rotation_error(R_wc, R_gt),
            t_err=min(translation_error(t_wc, t_gt), translation_error(-t_wc, t_gt)),
        )

    # avg_cost = <T,C> / T.sum
    transport_cost = torch.sum(T * C).item()
    avg_cost = transport_cost / T_sum

    # mass_aware = avg_cost + λ * (1 - T.sum)  [penalize low mass]
    lambda_mass = 0.1
    mass_aware = avg_cost + lambda_mass * max(0, 1.0 - T_sum)

    # full_uot = <T,C> + ρ*KL + ε*entropy
    # Approximate: use sinkhorn_info if available, else compute
    T_log_T = T * torch.log(T + 1e-10)
    entropy = -torch.sum(T_log_T).item()

    # KL divergence terms (approximate)
    a = solver._last_a if hasattr(solver, '_last_a') else torch.ones(T.shape[0]) / T.shape[0]
    b = solver._last_b if hasattr(solver, '_last_b') else torch.ones(T.shape[1]) / T.shape[1]
    if isinstance(a, np.ndarray):
        a = torch.tensor(a, dtype=torch.float32)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float32)

    row_sums = T.sum(dim=1)
    col_sums = T.sum(dim=0)

    # KL(row_sums || a)
    kl_row = torch.sum(row_sums * torch.log(row_sums / (a + 1e-10) + 1e-10) - row_sums + a).item()
    kl_col = torch.sum(col_sums * torch.log(col_sums / (b + 1e-10) + 1e-10) - col_sums + b).item()

    full_uot = transport_cost + rho * (kl_row + kl_col) + epsilon * entropy

    # Concentration: mean(row_max / row_sum)
    row_maxs = T.max(dim=1).values
    valid_mask = row_sums > 1e-10
    if valid_mask.sum() > 0:
        concentrations = row_maxs[valid_mask] / row_sums[valid_mask]
        concentration = concentrations.mean().item()
    else:
        concentration = 0.0

    # Top-K statistics
    T_np = T.cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]
    topk_values = T_flat[top_indices]
    topk_sum = topk_values.sum()

    C_np = C.cpu().numpy()
    C_flat = C_np.flatten()
    topk_costs = C_flat[top_indices]
    topk_cost = np.average(topk_costs, weights=topk_values) if topk_values.sum() > 0 else float('inf')

    # Eigenvalue gap from closed-form t (need K for normalized coords)
    K_np = solver.k1.cpu().numpy() if isinstance(solver.k1, torch.Tensor) else solver.k1
    eigen_gap = compute_eigen_gap(solver, R_wc_t, T, K_np, top_k=top_k)

    # GT errors
    R_err = rotation_error(R_wc, R_gt)
    t_err = min(translation_error(t_wc, t_gt), translation_error(-t_wc, t_gt))

    return ScoreResult(
        avg_cost=avg_cost,
        mass_aware=mass_aware,
        full_uot=full_uot,
        T_sum=T_sum,
        concentration=concentration,
        entropy=entropy,
        eigen_gap=eigen_gap,
        topk_sum=topk_sum,
        topk_cost=topk_cost,
        R_err=R_err,
        t_err=t_err,
    )


def compute_eigen_gap(
    solver: OptimalTransportSolver,
    R_wc: torch.Tensor,
    transport: torch.Tensor,
    K: np.ndarray,
    top_k: int = 50,
    min_mass: float = 0.001,
) -> float:
    """Compute eigenvalue gap from closed-form t estimation."""
    means1 = solver.gaussians1.means
    means2 = solver.gaussians2.means
    K_inv = np.linalg.inv(K)

    # Get top-K correspondences
    T_np = transport.detach().cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]

    n1, n2 = T_np.shape
    weights = []
    a_vectors = []

    for idx in top_indices:
        i = idx // n2
        j = idx % n2
        w = T_flat[idx]
        if w < min_mass:
            continue

        # Get normalized coordinates
        p1 = means1[i]
        p2 = means2[j]

        if isinstance(p1, torch.Tensor):
            p1 = p1.numpy()
            p2 = p2.numpy()

        x1_hom = np.array([p1[0], p1[1], 1.0])
        x2_hom = np.array([p2[0], p2[1], 1.0])

        x1_norm = K_inv @ x1_hom
        x2_norm = K_inv @ x2_hom

        # a_ij = x2 × (R @ x1)
        R_np = R_wc.numpy() if isinstance(R_wc, torch.Tensor) else R_wc
        Rx1 = R_np @ x1_norm
        a = np.cross(x2_norm, Rx1)

        weights.append(w)
        a_vectors.append(a)

    if len(weights) < 3:
        return 0.0

    weights = np.array(weights)
    a_vectors = np.array(a_vectors)

    # Build M = Σ w * a @ a^T
    M = np.zeros((3, 3))
    for w, a in zip(weights, a_vectors):
        M += w * np.outer(a, a)

    # Eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals)
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        return gap
    except:
        return 0.0


def compute_closed_form_t(
    solver: OptimalTransportSolver,
    R_wc: np.ndarray,
    transport: torch.Tensor,
    K: np.ndarray,
    top_k: int = 50,
) -> Optional[np.ndarray]:
    """Compute closed-form translation direction."""
    means1 = solver.gaussians1.means
    means2 = solver.gaussians2.means
    K_inv = np.linalg.inv(K)

    T_np = transport.detach().cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]

    n1, n2 = T_np.shape
    weights = []
    a_vectors = []

    for idx in top_indices:
        i = idx // n2
        j = idx % n2
        w = T_flat[idx]
        if w < 0.001:
            continue

        p1 = means1[i]
        p2 = means2[j]

        if isinstance(p1, torch.Tensor):
            p1 = p1.numpy()
            p2 = p2.numpy()

        x1_hom = np.array([p1[0], p1[1], 1.0])
        x2_hom = np.array([p2[0], p2[1], 1.0])

        x1_norm = K_inv @ x1_hom
        x2_norm = K_inv @ x2_hom

        Rx1 = R_wc @ x1_norm
        a = np.cross(x2_norm, Rx1)

        weights.append(w)
        a_vectors.append(a)

    if len(weights) < 3:
        return None

    weights = np.array(weights)
    a_vectors = np.array(a_vectors)

    M = np.zeros((3, 3))
    for w, a in zip(weights, a_vectors):
        M += w * np.outer(a, a)

    try:
        eigvals, eigvecs = np.linalg.eigh(M)
        t_opt = eigvecs[:, 0]  # Minimum eigenvalue
        t_opt = t_opt / (np.linalg.norm(t_opt) + 1e-10)
        return t_opt
    except:
        return None


def run_correlation_analysis(
    idx1: int = 0,
    idx2: int = 10,
    n_samples: int = 200,
    t_modes: List[str] = ["fixed", "random", "closed_form"],
    epsilon: float = 0.05,
    rho: float = 0.5,
    max_R_angle: float = 90.0,
    verbose: bool = True,
):
    """
    Run correlation analysis between scores and GT errors.

    Args:
        idx1, idx2: Image pair indices
        n_samples: Number of random R samples
        t_modes: List of t initialization modes
        epsilon, rho: OT parameters
        max_R_angle: Maximum angle for random R
        verbose: Print progress
    """
    print("\n" + "=" * 70)
    print(f"Step A: Score vs GT Error Correlation Analysis")
    print(f"  Image pair: ({idx1}, {idx2})")
    print(f"  n_samples: {n_samples}")
    print(f"  t_modes: {t_modes}")
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

    print(f"\nGT relative pose:")
    print(f"  R angle: {rotation_error(R_gt, np.eye(3)):.1f}deg from identity")
    print(f"  t direction: {t_gt}")

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Generate random R samples
    print(f"\nGenerating {n_samples} random R samples...")
    R_samples = []
    for i in range(n_samples):
        R_rand = random_rotation_matrix(max_R_angle)
        R_samples.append(R_rand)

    # Add some R near GT (for better coverage of low-error region)
    n_near_gt = n_samples // 10
    for i in range(n_near_gt):
        # Small perturbation from GT
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.random.uniform(0, np.deg2rad(15))  # Small angle
        K_mat = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_perturb = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
        R_samples.append(R_perturb @ R_gt)

    print(f"  Total samples: {len(R_samples)} ({n_samples} random + {n_near_gt} near GT)")

    all_results = {}

    for t_mode in t_modes:
        print(f"\n--- t_mode: {t_mode} ---")

        results = []
        n_collapsed = 0

        for i, R in enumerate(R_samples):
            if verbose and i % 50 == 0:
                print(f"  Processing sample {i}/{len(R_samples)}...")

            # Determine t based on mode
            if t_mode == "fixed":
                # Fixed random t (same for all samples)
                if i == 0:
                    t_fixed = random_unit_vector()
                t = t_fixed
            elif t_mode == "random":
                # Different random t for each sample
                t = random_unit_vector()
            elif t_mode == "closed_form":
                # Compute closed-form t from R
                # First compute transport at current R with some t
                t_init = random_unit_vector()
                R_t = torch.tensor(R, dtype=torch.float32)
                t_t = torch.tensor(t_init, dtype=torch.float32)
                t_t = t_t / (t_t.norm() + 1e-10)

                F = solver._build_F_from_wc(R_t, t_t)
                C = solver.compute_cost_matrix(F)

                with torch.no_grad():
                    transport, _ = solver.unbalanced_sinkhorn_algorithm(
                        C, epsilon=epsilon, rho=rho,
                        gate_mask=solver._last_gate_mask
                    )

                t_closed = compute_closed_form_t(solver, R, transport, K)
                t = t_closed if t_closed is not None else t_init
            else:
                raise ValueError(f"Unknown t_mode: {t_mode}")

            # Compute scores
            score = compute_all_scores(
                solver, R, t, R_gt, t_gt,
                epsilon=epsilon, rho=rho,
            )

            if score.T_sum < 0.1:
                n_collapsed += 1

            results.append(score)

        print(f"  Collapsed samples: {n_collapsed}/{len(R_samples)}")

        # Filter out collapsed samples for correlation
        valid_results = [r for r in results if r.T_sum > 0.1]
        print(f"  Valid samples: {len(valid_results)}")

        if len(valid_results) < 10:
            print(f"  WARNING: Too few valid samples for correlation analysis")
            continue

        # Compute correlations
        print(f"\n  Correlations (Spearman):")

        score_names = [
            'avg_cost', 'mass_aware', 'full_uot', 'T_sum',
            'concentration', 'entropy', 'eigen_gap', 'topk_sum', 'topk_cost'
        ]

        R_errs = np.array([r.R_err for r in valid_results])
        t_errs = np.array([r.t_err for r in valid_results])

        print(f"\n  {'Score':<15} {'corr(R_err)':>12} {'p-value':>10} {'corr(t_err)':>12} {'p-value':>10}")
        print("  " + "-" * 62)

        correlations = {}
        for name in score_names:
            values = np.array([getattr(r, name) for r in valid_results])

            # Handle inf values
            finite_mask = np.isfinite(values)
            if finite_mask.sum() < 10:
                print(f"  {name:<15} {'N/A (too many inf)':>50}")
                continue

            values_f = values[finite_mask]
            R_errs_f = R_errs[finite_mask]
            t_errs_f = t_errs[finite_mask]

            # Spearman correlation
            corr_R, p_R = stats.spearmanr(values_f, R_errs_f)
            corr_t, p_t = stats.spearmanr(values_f, t_errs_f)

            correlations[name] = {
                'corr_R': corr_R, 'p_R': p_R,
                'corr_t': corr_t, 'p_t': p_t,
            }

            # Highlight strong correlations
            R_marker = "***" if abs(corr_R) > 0.5 else ("**" if abs(corr_R) > 0.3 else "")
            t_marker = "***" if abs(corr_t) > 0.5 else ("**" if abs(corr_t) > 0.3 else "")

            print(f"  {name:<15} {corr_R:>10.3f}{R_marker:<2} {p_R:>10.2e} {corr_t:>10.3f}{t_marker:<2} {p_t:>10.2e}")

        # Statistics
        print(f"\n  R_err statistics: min={R_errs.min():.1f}, max={R_errs.max():.1f}, "
              f"mean={R_errs.mean():.1f}, std={R_errs.std():.1f}")
        print(f"  t_err statistics: min={t_errs.min():.1f}, max={t_errs.max():.1f}, "
              f"mean={t_errs.mean():.1f}, std={t_errs.std():.1f}")

        all_results[t_mode] = {
            'results': valid_results,
            'correlations': correlations,
            'n_collapsed': n_collapsed,
        }

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nBest scores for R_err prediction (by |correlation|):")
    for t_mode in t_modes:
        if t_mode not in all_results:
            continue
        corrs = all_results[t_mode]['correlations']
        if not corrs:
            continue

        sorted_by_R = sorted(corrs.items(), key=lambda x: abs(x[1]['corr_R']), reverse=True)
        print(f"\n  t_mode={t_mode}:")
        for name, c in sorted_by_R[:3]:
            print(f"    {name}: corr={c['corr_R']:.3f} (p={c['p_R']:.2e})")

    print("\nBest scores for t_err prediction (by |correlation|):")
    for t_mode in t_modes:
        if t_mode not in all_results:
            continue
        corrs = all_results[t_mode]['correlations']
        if not corrs:
            continue

        sorted_by_t = sorted(corrs.items(), key=lambda x: abs(x[1]['corr_t']), reverse=True)
        print(f"\n  t_mode={t_mode}:")
        for name, c in sorted_by_t[:3]:
            print(f"    {name}: corr={c['corr_t']:.3f} (p={c['p_t']:.2e})")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score vs GT Error Correlation Analysis")
    parser.add_argument("--idx1", type=int, default=0, help="First image index")
    parser.add_argument("--idx2", type=int, default=10, help="Second image index")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of random R samples")
    parser.add_argument("--t-mode", type=str, default="all",
                       choices=["fixed", "random", "closed_form", "all"],
                       help="Translation initialization mode")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--rho", type=float, default=0.5, help="Unbalanced OT rho")
    parser.add_argument("--max-R-angle", type=float, default=90.0, help="Max random R angle")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.t_mode == "all":
        t_modes = ["fixed", "random", "closed_form"]
    else:
        t_modes = [args.t_mode]

    run_correlation_analysis(
        idx1=args.idx1,
        idx2=args.idx2,
        n_samples=args.n_samples,
        t_modes=t_modes,
        epsilon=args.epsilon,
        rho=args.rho,
        max_R_angle=args.max_R_angle,
        verbose=not args.quiet,
    )
