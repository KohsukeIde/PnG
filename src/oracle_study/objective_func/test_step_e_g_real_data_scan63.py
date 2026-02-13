"""
Step E: Test with Real Fitted Gaussian Data (scan63)

Uses:
- Fitted Gaussians from: data/fitted_gs/scan63_images_resized_em_200/
- Camera poses from: COLMAP (data/DTU/scan63/sparse/0/)

Per LOG.md findings:
- DTU cameras.npz has incorrect K (principal point offset due to asymmetric crop)
- COLMAP has correct K (bundle adjustment optimized)
- R, t match exactly between DTU and COLMAP

Recommended pairs (baseline < 1.5): 0000-0001, 0000-0002, 0000-0010
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

    # Load camera intrinsics (single shared camera for DTU)
    cameras = load_cameras_from_colmap(colmap_dir)

    # Load image poses
    images = load_images_from_colmap(colmap_dir)

    return cameras, images


def get_colmap_camera_params(cameras, images, image_name: str):
    """Get camera parameters for a given image from COLMAP data.

    Returns:
        dict with K (3x3 intrinsics), R (3x3 rotation), t (3 translation)
    """
    # Find image by name
    image_data = None
    for img_id, img in images.items():
        if img['name'] == image_name:
            image_data = img
            break

    if image_data is None:
        raise ValueError(f"Image {image_name} not found in COLMAP data")

    # Get camera intrinsics
    camera = cameras[image_data['camera_id']]
    K = camera.get_camera_matrix()

    # Get rotation from quaternion
    R = quaternion_to_rotation_matrix(
        image_data['qw'], image_data['qx'],
        image_data['qy'], image_data['qz']
    )

    # Translation
    t = np.array([image_data['tx'], image_data['ty'], image_data['tz']])

    return {
        'K': K,
        'R': R,
        't': t,
        'name': image_name,
    }


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (world-to-camera) from camera 1 to camera 2."""
    R1, t1 = cam1['R'], cam1['t']
    R2, t2 = cam2['R'], cam2['t']

    # Relative rotation: R_21 = R2 @ R1.T
    R_rel = R2 @ R1.T

    # Relative translation: t_21 = t2 - R_rel @ t1
    t_rel = t2 - R_rel @ t1

    # Normalize translation to unit vector
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


def compute_baseline(cam1: dict, cam2: dict) -> float:
    """Compute baseline (camera center distance) between two cameras."""
    # Camera center = -R^T @ t
    c1 = -cam1['R'].T @ cam1['t']
    c2 = -cam2['R'].T @ cam2['t']
    return np.linalg.norm(c2 - c1)


def test_ot_with_real_data(idx1: int = 0, idx2: int = 1):
    """Test OT matching with real fitted Gaussians using COLMAP cameras."""
    print("=" * 70)
    print(f"Step E: OT with Real Data (images {idx1} and {idx2})")
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

    print(f"  Camera {idx1} intrinsics K1:\n{cam1['K']}")
    print(f"  Camera {idx2} intrinsics K2:\n{cam2['K']}")

    # Compute baseline
    baseline = compute_baseline(cam1, cam2)
    print(f"\n  Baseline (camera distance): {baseline:.3f}")
    if baseline > 1.5:
        print("  WARNING: Baseline > 1.5, SIFT matches may be unreliable")

    # Compute ground truth relative pose (camera-to-world)
    R_gt, t_gt = compute_relative_pose_cw(cam1, cam2)
    print(f"\nGround truth relative pose (cam1 -> cam2, camera-to-world):")
    print(f"  R_gt:\n{R_gt}")
    print(f"  t_gt: {t_gt}")

    # Use the same K for both (they should be identical in DTU)
    K = cam1['K']
    print(f"\nUsing intrinsics:\n{K}")

    # Create OT solver with settings from LOG.md
    # sigma_epipolar normalizes the cost matrix
    sigma_epipolar = 400.0  # LOG.md recommendation
    print("\nCreating OT solver...")
    print(f"  sigma_epipolar={sigma_epipolar}")
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

    # Test with GT pose first
    print("\n" + "-" * 60)
    print("Test 1: OT with Ground Truth pose")
    print("-" * 60)

    R_gt_wc, t_gt_wc = invert_pose(R_gt, t_gt)
    R_gt_t = torch.tensor(R_gt_wc, dtype=torch.float32)
    t_gt_t = torch.tensor(t_gt_wc, dtype=torch.float32)
    F_gt = solver._build_F_from_wc(R_gt_t, t_gt_t)

    cost_gt = solver.compute_cost_matrix(F_gt)
    print(f"Cost matrix (GT pose):")
    print(f"  min: {cost_gt.min().item():.4f}, max: {cost_gt.max().item():.4f}")
    print(f"  mean: {cost_gt.mean().item():.4f}, median: {cost_gt.median().item():.4f}")

    # OT settings from LOG.md (epi-only, ε=0.05, ρ=0.5)
    epsilon = 0.05
    rho = 0.5  # Unbalanced (allows mass to adjust)
    print(f"\nSinkhorn params: epsilon={epsilon:.4f}, rho={rho:.1f}")

    T_gt, _ = solver.unbalanced_sinkhorn_algorithm(
        cost_matrix=cost_gt,
        epsilon=epsilon,
        rho=rho,
        max_iter=500,
        tol=1e-8,
        gate_mask=solver._last_gate_mask,
    )

    loss_gt = (T_gt * cost_gt).sum().item()
    primal_gt = -T_gt.sum().item()  # primal score (mass-aware)

    print(f"\nResults with GT pose:")
    print(f"  T.sum(): {T_gt.sum().item():.4f}")
    print(f"  Loss: {loss_gt:.4f}")
    print(f"  Primal score: {primal_gt:.4f}")

    # Analyze top matches
    top1_weights = T_gt.max(dim=1).values
    print(f"  top1 weight: mean={top1_weights.mean().item():.6f}")

    # Test with perturbed poses at various angles
    print("\n" + "-" * 60)
    print("Test 2: OT with Perturbed poses")
    print("-" * 60)

    angles_deg = [10, 30, 60]
    results_bad = []

    for sign in [1, -1]:
        for angle_deg in angles_deg:
            angle_rad = sign * np.radians(angle_deg)

            # Create perturbed pose (rotation around Y axis)
            axis = np.array([0, 1, 0])
            K_perturb = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R_perturb = np.eye(3) + np.sin(angle_rad) * K_perturb + (1 - np.cos(angle_rad)) * (K_perturb @ K_perturb)
            R_bad_cw = R_perturb @ R_gt

            R_bad_wc, t_bad_wc = invert_pose(R_bad_cw, t_gt)
            R_bad_t = torch.tensor(R_bad_wc, dtype=torch.float32)
            t_bad_t = torch.tensor(t_bad_wc, dtype=torch.float32)
            F_bad = solver._build_F_from_wc(R_bad_t, t_bad_t)

            cost_bad = solver.compute_cost_matrix(F_bad)

            T_bad, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix=cost_bad,
                epsilon=epsilon,
                rho=rho,
                max_iter=500,
                tol=1e-8,
                gate_mask=solver._last_gate_mask,
            )

            loss_bad = (T_bad * cost_bad).sum().item()
            primal_bad = -T_bad.sum().item()

            label = f"{sign*angle_deg:+d}deg"
            results_bad.append({
                'label': label,
                'loss': loss_bad,
                'primal': primal_bad,
                'T_sum': T_bad.sum().item(),
            })

            gt_better_loss = "GT better" if loss_gt < loss_bad else "BAD better"
            gt_better_primal = "GT better" if primal_gt < primal_bad else "BAD better"

            print(f"\n  {label}:")
            print(f"    Loss: {loss_bad:.4f} ({gt_better_loss})")
            print(f"    Primal: {primal_bad:.4f} ({gt_better_primal})")
            print(f"    T.sum(): {T_bad.sum().item():.4f}")

    # Summary
    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"GT pose:   Loss={loss_gt:.4f}, Primal={primal_gt:.4f}, T.sum={T_gt.sum().item():.4f}")

    gt_wins_loss = sum(1 for r in results_bad if loss_gt < r['loss'])
    gt_wins_primal = sum(1 for r in results_bad if primal_gt < r['primal'])

    print(f"\nGT wins (Loss):   {gt_wins_loss}/{len(results_bad)}")
    print(f"GT wins (Primal): {gt_wins_primal}/{len(results_bad)}")

    if gt_wins_loss == len(results_bad):
        print("\nGOOD: GT pose has lowest loss!")
    elif gt_wins_primal == len(results_bad):
        print("\nOK: GT pose has lowest primal (but not always lowest loss)")
    else:
        print("\nBAD: GT pose does NOT consistently win")

    return {
        'loss_gt': loss_gt,
        'primal_gt': primal_gt,
        'T_sum_gt': T_gt.sum().item(),
        'results_bad': results_bad,
        'gt_wins_loss': gt_wins_loss,
        'gt_wins_primal': gt_wins_primal,
        'baseline': baseline,
    }


def test_recommended_pairs():
    """Test with recommended pairs (baseline < 1.5)."""
    print("\n" + "=" * 70)
    print("Testing Recommended Pairs (baseline < 1.5)")
    print("=" * 70)

    # Recommended pairs from LOG.md
    pairs = [(0, 1), (0, 2), (0, 10), (11, 14)]

    results = []
    for idx1, idx2 in pairs:
        print(f"\n{'='*70}")
        try:
            result = test_ot_with_real_data(idx1, idx2)
            result['pair'] = (idx1, idx2)
            results.append(result)
        except Exception as e:
            print(f"Error processing pair ({idx1}, {idx2}): {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Summary of All Pairs")
    print("=" * 70)
    print(f"{'Pair':>10} {'Baseline':>10} {'Loss_GT':>12} {'Primal_GT':>12} {'GT wins (L)':>12} {'GT wins (P)':>12}")
    for r in results:
        print(f"{str(r['pair']):>10} {r['baseline']:>10.3f} {r['loss_gt']:>12.4f} {r['primal_gt']:>12.4f} "
              f"{r['gt_wins_loss']:>12} {r['gt_wins_primal']:>12}")


if __name__ == "__main__":
    # Test single pair first (0, 1 has small baseline)
    result = test_ot_with_real_data(0, 1)

    # Test multiple recommended pairs
    test_recommended_pairs()

    print("\n" + "=" * 70)
    print("Test completed")
    print("=" * 70)
