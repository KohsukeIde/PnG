"""
Step F: Score Function Comparison Experiment

Purpose: Compare different score functions to find one that:
1. Is robust to mass collapse
2. Has GT pose as the minimum

Score functions to test:
1. loss = <T,C> (current - vulnerable to collapse)
2. primal = -T.sum() (collapse-robust but may ignore cost quality)
3. avg_cost = <T,C> / (T.sum() + eps) (normalized by mass)
4. mass_aware = avg_cost + lambda * (KL_row + KL_col)
5. full_uot = <T,C> + rho*(KL_row + KL_col) + epsilon*sum(T*log T - T) (Sinkhorn objective)
"""

import numpy as np
import torch
import sys
import os
from typing import Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
)
from src.utils.gaussian_utils import load_gaussians


def load_colmap_cameras(scan_name: str = "scan63"):
    """Load camera data from COLMAP sparse reconstruction."""
    colmap_dir = os.path.join(project_root, f"data/DTU/{scan_name}/sparse/0")
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


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (world-to-camera) from camera 1 to camera 2."""
    R1, t1 = cam1['R'], cam1['t']
    R2, t2 = cam2['R'], cam2['t']

    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    t_rel_norm = t_rel / (np.linalg.norm(t_rel) + 1e-10)

    return R_rel, t_rel_norm


def invert_pose(R_wc: np.ndarray, t_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert world-to-camera pose to camera-to-world pose."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def compute_relative_pose_cw(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (camera-to-world) from camera 1 to camera 2."""
    R_wc, t_wc = compute_relative_pose_wc(cam1, cam2)
    return invert_pose(R_wc, t_wc)


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute KL(p || q) = sum(p * log(p/q) - p + q)."""
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (p * torch.log(p / q) - p + q).sum()


def compute_entropy(T: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Compute entropy H(T) = -sum(T * log(T))."""
    T_safe = torch.clamp(T, min=eps)
    return -(T * torch.log(T_safe)).sum()


def compute_all_scores(
    T: torch.Tensor,
    C: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    epsilon: float,
    rho: float,
    lambda_kl: float = 0.1,
) -> dict:
    """Compute all score functions for a given transport plan.

    IMPORTANT: epsilon and rho must be the ACTUAL values used in the final
    Sinkhorn iteration, not the input values. The Sinkhorn algorithm may
    internally scale these values (e.g., [ε, 0.5ε] schedule). Use
    solver._last_sinkhorn_epsilon and solver._last_sinkhorn_rho after
    calling unbalanced_sinkhorn_algorithm().

    Args:
        T: Transport plan (K1, K2)
        C: Cost matrix (K1, K2)
        a: Source marginal (K1,)
        b: Target marginal (K2,)
        epsilon: Entropy regularization (ACTUAL value from Sinkhorn)
        rho: KL regularization (ACTUAL value from Sinkhorn)
        lambda_kl: Weight for KL in mass-aware score

    Returns:
        dict with all scores
    """
    eps_safe = 1e-10

    # Basic quantities
    T_sum = T.sum()
    transport_cost = (T * C).sum()

    # Marginals
    row_sum = T.sum(dim=1)
    col_sum = T.sum(dim=0)

    # KL divergences
    KL_row = compute_kl_divergence(row_sum, a)
    KL_col = compute_kl_divergence(col_sum, b)
    KL_total = KL_row + KL_col

    # Entropy
    entropy = compute_entropy(T)

    # Score 1: Current loss (vulnerable to collapse)
    loss = transport_cost.item()

    # Score 2: Primal = -T.sum() (robust to collapse)
    primal = -T_sum.item()

    # Score 3: Average cost (normalized by mass)
    avg_cost = (transport_cost / (T_sum + eps_safe)).item()

    # Score 4: Mass-aware = avg_cost + lambda * KL
    mass_aware = avg_cost + lambda_kl * KL_total.item()

    # Score 5: Full UOT objective (matches Sinkhorn)
    entropic = -entropy - T_sum  # sum(T * log T - T)
    uot_cost_term = transport_cost
    uot_kl_term = rho * KL_total
    uot_entropic_term = epsilon * entropic
    full_uot = (uot_cost_term + uot_kl_term + uot_entropic_term).item()

    return {
        'loss': loss,
        'primal': primal,
        'avg_cost': avg_cost,
        'mass_aware': mass_aware,
        'full_uot': full_uot,
        'T_sum': T_sum.item(),
        'KL_row': KL_row.item(),
        'KL_col': KL_col.item(),
        'entropy': entropy.item(),
        'epsilon': float(epsilon),
        'rho': float(rho),
        'uot_cost_term': uot_cost_term.item(),
        'uot_kl_term': uot_kl_term.item(),
        'uot_entropic_term': uot_entropic_term.item(),
    }


def test_score_functions(idx1: int = 0, idx2: int = 10):
    """Test all score functions with GT and perturbed poses."""
    print("=" * 70)
    print(f"Step F: Score Function Comparison (images {idx1} and {idx2})")
    print("=" * 70)

    # Load Gaussians
    print("\nLoading fitted Gaussians...")
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)

    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)

    print(f"  Image {idx1}: K={g1.means.shape[0]} Gaussians")
    print(f"  Image {idx2}: K={g2.means.shape[0]} Gaussians")

    # Load COLMAP camera poses
    print("\nLoading COLMAP camera poses...")
    cameras, images = load_colmap_cameras()

    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")

    # Compute ground truth relative pose (camera-to-world)
    R_gt, t_gt = compute_relative_pose_cw(cam1, cam2)
    print(f"\nGround truth relative pose (cam1 -> cam2, camera-to-world):")
    print(f"  R_gt:\n{R_gt}")
    print(f"  t_gt: {t_gt}")

    K = cam1['K']

    # OT settings from LOG.md
    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5
    lambda_kl = 0.1  # for mass-aware score

    print(f"\nOT Settings: sigma_epipolar={sigma_epipolar}, epsilon={epsilon}, rho={rho}")

    # Create OT solver
    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=sigma_epipolar,
        epi_clip=None,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
    )

    # Get normalized marginals
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    # Test poses: GT and perturbations
    angles_deg = [0, 10, 30, 60, -10, -30, -60]
    results = []

    for angle_deg in angles_deg:
        if angle_deg == 0:
            R_test = R_gt
            label = "GT"
        else:
            angle_rad = np.radians(angle_deg)
            axis = np.array([0, 1, 0])
            K_skew = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_perturb = np.eye(3) + np.sin(angle_rad) * K_skew + (1 - np.cos(angle_rad)) * (K_skew @ K_skew)
            R_test = R_perturb @ R_gt
            label = f"{angle_deg:+d}deg"

        R_test_wc, t_test_wc = invert_pose(R_test, t_gt)
        R_test_t = torch.tensor(R_test_wc, dtype=torch.float32)
        t_test_t = torch.tensor(t_test_wc, dtype=torch.float32)
        F = solver._build_F_from_wc(R_test_t, t_test_t)

        cost = solver.compute_cost_matrix(F)

        T, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix=cost,
            epsilon=epsilon,
            rho=rho,
            max_iter=500,
            tol=1e-8,
            gate_mask=solver._last_gate_mask,
        )

        # CRITICAL: Use ACTUAL ε/ρ from Sinkhorn's final iteration
        # Sinkhorn internally uses ε-scaling (e.g., [ε, 0.5ε]), so the final
        # transport plan corresponds to the scaled values, not the input values.
        eps_actual = solver._last_sinkhorn_epsilon
        rho_actual = solver._last_sinkhorn_rho
        if eps_actual is None or rho_actual is None:
            raise RuntimeError("Sinkhorn did not record actual ε/ρ values.")
        scores = compute_all_scores(T, cost, a, b, eps_actual, rho_actual, lambda_kl)
        scores['label'] = label
        scores['angle_deg'] = angle_deg
        results.append(scores)

    # Print results table
    print("\n" + "-" * 100)
    print("Score Comparison Table")
    print("-" * 100)
    print(f"{'Pose':>8} {'T.sum()':>10} {'loss':>12} {'primal':>12} {'avg_cost':>12} {'mass_aware':>12} {'full_uot':>12}")
    print("-" * 100)

    gt_result = results[0]
    for r in results:
        print(f"{r['label']:>8} {r['T_sum']:>10.4f} {r['loss']:>12.4f} {r['primal']:>12.4f} "
              f"{r['avg_cost']:>12.4f} {r['mass_aware']:>12.4f} {r['full_uot']:>12.4f}")

    # Report actual epsilon/rho used (unique values across poses)
    eps_values = sorted({r['epsilon'] for r in results})
    rho_values = sorted({r['rho'] for r in results})
    print(f"\nActual ε values used: {eps_values}")
    print(f"Actual ρ values used: {rho_values}")

    # UOT term decomposition
    print("\n" + "-" * 100)
    print("UOT Term Decomposition (lower is better)")
    print("-" * 100)
    print(f"{'Pose':>8} {'<T,C>':>12} {'rho*KL':>12} {'eps*Ent':>12} {'full_uot':>12}")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:>8} {r['uot_cost_term']:>12.4f} {r['uot_kl_term']:>12.4f} "
              f"{r['uot_entropic_term']:>12.4f} {r['full_uot']:>12.4f}")

    # Analyze which score has GT as minimum
    print("\n" + "-" * 100)
    print("GT Wins Analysis (lower is better for all scores)")
    print("-" * 100)

    score_names = ['loss', 'primal', 'avg_cost', 'mass_aware', 'full_uot']
    perturbed = [r for r in results if r['angle_deg'] != 0]

    for score_name in score_names:
        gt_val = gt_result[score_name]
        wins = sum(1 for r in perturbed if gt_val < r[score_name])
        total = len(perturbed)

        # Find which poses beat GT
        beaten_by = [r['label'] for r in perturbed if gt_val >= r[score_name]]

        status = "GOOD" if wins == total else "BAD"
        print(f"{score_name:>12}: GT wins {wins}/{total} - {status}")
        if beaten_by:
            print(f"             GT beaten by: {', '.join(beaten_by)}")

    # Additional analysis: correlation with pose error
    print("\n" + "-" * 100)
    print("Score vs Pose Error Correlation")
    print("-" * 100)

    for score_name in score_names:
        values = [r[score_name] for r in results]
        angles = [abs(r['angle_deg']) for r in results]

        # Check if score increases with angle (good behavior)
        monotonic = all(values[i] <= values[i+1] for i in range(len(angles)-1)
                       if angles[i] <= angles[i+1])

        if monotonic:
            print(f"{score_name:>12}: Monotonic with error (GOOD)")
        else:
            print(f"{score_name:>12}: Non-monotonic (may have local minima)")

    return results


def test_multiple_pairs():
    """Test score functions on multiple image pairs."""
    print("\n" + "=" * 70)
    print("Testing Multiple Pairs")
    print("=" * 70)

    pairs = [(0, 1), (0, 2), (0, 10), (11, 14)]

    all_results = {}
    for idx1, idx2 in pairs:
        print(f"\n{'='*70}")
        try:
            results = test_score_functions(idx1, idx2)
            all_results[(idx1, idx2)] = results
        except Exception as e:
            print(f"Error processing pair ({idx1}, {idx2}): {e}")
            import traceback
            traceback.print_exc()

    # Summary across all pairs
    print("\n" + "=" * 70)
    print("Summary: GT Wins Across All Pairs")
    print("=" * 70)

    score_names = ['loss', 'primal', 'avg_cost', 'mass_aware', 'full_uot']

    print(f"{'Pair':>12} " + " ".join(f"{s:>12}" for s in score_names))
    print("-" * 80)

    total_wins = {s: 0 for s in score_names}
    total_tests = 0

    for pair, results in all_results.items():
        gt_result = results[0]
        perturbed = [r for r in results if r['angle_deg'] != 0]

        row = f"{str(pair):>12}"
        for score_name in score_names:
            gt_val = gt_result[score_name]
            wins = sum(1 for r in perturbed if gt_val < r[score_name])
            total = len(perturbed)
            total_wins[score_name] += wins
            row += f" {wins:>5}/{total:<5}"
        print(row)
        total_tests += len(perturbed)

    print("-" * 80)
    row = f"{'TOTAL':>12}"
    for score_name in score_names:
        row += f" {total_wins[score_name]:>5}/{total_tests:<5}"
    print(row)

    # Best score
    best_score = max(score_names, key=lambda s: total_wins[s])
    print(f"\nBest performing score: {best_score} ({total_wins[best_score]}/{total_tests} wins)")


if __name__ == "__main__":
    # Test single pair first
    results = test_score_functions(0, 10)

    # Test multiple pairs
    test_multiple_pairs()

    print("\n" + "=" * 70)
    print("Step F completed")
    print("=" * 70)
