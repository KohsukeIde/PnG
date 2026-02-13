#!/usr/bin/env python3
"""
Step D: Structural Descriptors for Improved OT Cost

Add structural features beyond point positions to the OT cost matrix:
1. Covariance shape (eigenvalue ratio, orientation)
2. Color similarity (if available)
3. Local context (neighbor statistics)
4. Scale consistency

Usage:
    python test_structural_descriptors_analysis.py
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

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


def extract_cov_features(covs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract structural features from covariance matrices.

    Args:
        covs: (N, 2, 2) array of covariance matrices

    Returns:
        Dict with:
        - 'eigenratio': λ_max / λ_min (elongation)
        - 'scale': sqrt(λ_max * λ_min) (geometric mean scale)
        - 'orientation': angle of major axis
    """
    n = len(covs)
    eigenratios = np.zeros(n)
    scales = np.zeros(n)
    orientations = np.zeros(n)

    for i, cov in enumerate(covs):
        if isinstance(cov, torch.Tensor):
            cov = cov.numpy()

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)  # Ensure positive

        lmax, lmin = max(eigvals), min(eigvals)
        eigenratios[i] = lmax / (lmin + 1e-10)
        scales[i] = np.sqrt(lmax * lmin)

        # Orientation: angle of major eigenvector
        major_vec = eigvecs[:, np.argmax(eigvals)]
        orientations[i] = np.arctan2(major_vec[1], major_vec[0])

    return {
        'eigenratio': eigenratios,
        'scale': scales,
        'orientation': orientations,
    }


def compute_structural_cost_matrix(
    means1: np.ndarray,
    means2: np.ndarray,
    cov_features1: Dict[str, np.ndarray],
    cov_features2: Dict[str, np.ndarray],
    lambda_shape: float = 0.1,
    lambda_scale: float = 0.1,
) -> np.ndarray:
    """
    Compute structural cost matrix based on covariance features.

    Cost = λ_shape * |log(ratio1) - log(ratio2)| + λ_scale * |log(scale1/scale2)|

    Returns:
        (n1, n2) cost matrix
    """
    n1, n2 = len(means1), len(means2)

    # Log eigenratio difference
    log_ratio1 = np.log(cov_features1['eigenratio'] + 1e-10)
    log_ratio2 = np.log(cov_features2['eigenratio'] + 1e-10)
    shape_cost = np.abs(log_ratio1[:, None] - log_ratio2[None, :])

    # Scale ratio
    log_scale1 = np.log(cov_features1['scale'] + 1e-10)
    log_scale2 = np.log(cov_features2['scale'] + 1e-10)
    scale_cost = np.abs(log_scale1[:, None] - log_scale2[None, :])

    return lambda_shape * shape_cost + lambda_scale * scale_cost


def run_structural_descriptor_test(
    idx1: int = 0,
    idx2: int = 10,
    epsilon: float = 0.05,
    rho: float = 0.5,
    lambda_cov_list: Optional[List[float]] = None,
):
    """Test structural descriptors in OT cost."""

    print("\n" + "=" * 70)
    print("Step D: Structural Descriptors Analysis")
    print(f"  Image pair: ({idx1}, {idx2})")
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

    # Extract covariance features
    print("\n--- Covariance Feature Statistics ---")

    covs1 = g1.covs.numpy() if isinstance(g1.covs, torch.Tensor) else g1.covs
    covs2 = g2.covs.numpy() if isinstance(g2.covs, torch.Tensor) else g2.covs

    features1 = extract_cov_features(covs1)
    features2 = extract_cov_features(covs2)

    for name in ['eigenratio', 'scale', 'orientation']:
        f1, f2 = features1[name], features2[name]
        print(f"\n{name}:")
        print(f"  Image 1: mean={f1.mean():.3f}, std={f1.std():.3f}, range=[{f1.min():.3f}, {f1.max():.3f}]")
        print(f"  Image 2: mean={f2.mean():.3f}, std={f2.std():.3f}, range=[{f2.min():.3f}, {f2.max():.3f}]")

    # Test different cost configurations
    print("\n--- Cost Matrix Comparison ---")

    means1 = g1.means.numpy() if isinstance(g1.means, torch.Tensor) else g1.means
    means2 = g2.means.numpy() if isinstance(g2.means, torch.Tensor) else g2.means

    # Baseline: epipolar-only
    solver_baseline = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Cov-only (for scale normalization)
    solver_cov_only = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=1.0, lambda_epipolar=0.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    R_t = torch.tensor(R_gt, dtype=torch.float32)
    t_t = torch.tensor(t_gt, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    # Compute costs at GT pose
    F_baseline = solver_baseline._build_F_from_wc(R_t, t_t)
    C_baseline = solver_baseline.compute_cost_matrix(F_baseline)

    F_cov_only = solver_cov_only._build_F_from_wc(R_t, t_t)
    C_cov_only = solver_cov_only.compute_cost_matrix(F_cov_only)

    C_baseline_np = C_baseline.detach().numpy()
    C_cov_only_np = C_cov_only.detach().numpy()

    print(f"\nBaseline cost (epipolar only):")
    print(f"  mean={C_baseline_np.mean():.4f}, std={C_baseline_np.std():.4f}")
    print(f"  min={C_baseline_np.min():.4f}, max={C_baseline_np.max():.4f}")

    print(f"\nCov-only cost (λ_cov=1.0, epipolar=0):")
    print(f"  mean={C_cov_only_np.mean():.4f}, std={C_cov_only_np.std():.4f}")
    print(f"  min={C_cov_only_np.min():.4f}, max={C_cov_only_np.max():.4f}")

    # Compute transports
    with torch.no_grad():
        T_baseline, _ = solver_baseline.unbalanced_sinkhorn_algorithm(
            C_baseline, epsilon=epsilon, rho=rho,
            gate_mask=solver_baseline._last_gate_mask
        )

    print(f"\nTransport comparison at GT:")
    print(f"  Baseline T_sum: {T_baseline.sum().item():.3f}")

    # Compare top correspondences
    print("\n--- Top Correspondence Analysis ---")

    def get_top_correspondences(T, means1, means2, covs1, covs2, k=10):
        T_np = T.detach().cpu().numpy()
        T_flat = T_np.flatten()
        top_idx = np.argsort(T_flat)[-k:]

        n2 = T_np.shape[1]
        results = []
        for idx in reversed(top_idx):
            i = idx // n2
            j = idx % n2
            mass = T_flat[idx]

            # Get features
            c1, c2 = covs1[i], covs2[j]
            if isinstance(c1, torch.Tensor):
                c1, c2 = c1.numpy(), c2.numpy()

            e1 = np.linalg.eigvalsh(c1)
            e2 = np.linalg.eigvalsh(c2)
            ratio1 = max(e1) / (min(e1) + 1e-10)
            ratio2 = max(e2) / (min(e2) + 1e-10)

            results.append({
                'i': i, 'j': j, 'mass': mass,
                'ratio1': ratio1, 'ratio2': ratio2,
                'ratio_diff': abs(np.log(ratio1) - np.log(ratio2)),
            })
        return results

    print("\nBaseline top-10:")
    top_baseline = get_top_correspondences(T_baseline, means1, means2, covs1, covs2)
    print(f"  {'i':>4} {'j':>4} {'mass':>8} {'ratio1':>8} {'ratio2':>8} {'log_diff':>8}")
    for r in top_baseline:
        print(f"  {r['i']:>4} {r['j']:>4} {r['mass']:>8.4f} {r['ratio1']:>8.2f} {r['ratio2']:>8.2f} {r['ratio_diff']:>8.2f}")

    avg_ratio_diff_baseline = np.mean([r['ratio_diff'] for r in top_baseline])
    print(f"\n  Average log(ratio) diff: {avg_ratio_diff_baseline:.3f}")

    # === New: λ_cov sweep with normalization ===
    if lambda_cov_list is None:
        lambda_cov_list = [0.02, 0.05, 0.1, 0.2, 0.5]

    # Normalize cov scale to roughly match baseline median
    median_epi = float(np.median(C_baseline_np))
    median_cov = float(np.median(C_cov_only_np))
    scale_ratio = median_epi / (median_cov + 1e-10)
    print(f"\n--- λ_cov sweep (normalized) ---")
    print(f"  median_epi={median_epi:.4f}, median_cov={median_cov:.4f}, scale_ratio={scale_ratio:.6f}")
    print(f"  {'λ_cov(raw)':>10} {'λ_cov(eff)':>12} {'T_sum':>8} {'conc':>6} {'avg_cost':>9} {'log_diff':>9}")
    print("  " + "-" * 58)

    for lambda_raw in lambda_cov_list:
        lambda_eff = lambda_raw * scale_ratio
        solver_cov = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2,
            k1=K, k2=K, device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=lambda_eff, lambda_epipolar=1.0,
            sigma_epipolar=400.0,
            ot_mass1=ot_mass1, ot_mass2=ot_mass2,
        )
        F_cov = solver_cov._build_F_from_wc(R_t, t_t)
        C_cov = solver_cov.compute_cost_matrix(F_cov)
        with torch.no_grad():
            T_cov, _ = solver_cov.unbalanced_sinkhorn_algorithm(
                C_cov, epsilon=epsilon, rho=rho,
                gate_mask=solver_cov._last_gate_mask
            )
        T_sum_cov = T_cov.sum().item()
        row_sums = T_cov.sum(dim=1)
        row_maxs = T_cov.max(dim=1).values
        valid_mask = row_sums > 1e-10
        conc = (row_maxs[valid_mask] / row_sums[valid_mask]).mean().item() if valid_mask.any() else 0.0
        avg_cost_cov = (T_cov * C_cov).sum().item() / (T_sum_cov + 1e-10)
        top_cov = get_top_correspondences(T_cov, means1, means2, covs1, covs2)
        avg_ratio_diff_cov = np.mean([r['ratio_diff'] for r in top_cov]) if top_cov else float('inf')
        print(f"  {lambda_raw:>10.3f} {lambda_eff:>12.6f} {T_sum_cov:>8.3f} "
              f"{conc:>6.3f} {avg_cost_cov:>9.4f} {avg_ratio_diff_cov:>9.3f}")

    # Test at perturbed poses
    print("\n--- Pose Perturbation Test ---")

    def random_rotation_matrix(max_angle_deg: float) -> np.ndarray:
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.random.uniform(0, np.deg2rad(max_angle_deg))
        K_mat = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)

    test_angles = [0, 5, 10, 15, 20, 30]
    results_comparison = []

    for angle in test_angles:
        if angle == 0:
            R_test = R_gt
        else:
            R_test = random_rotation_matrix(angle) @ R_gt

        R_t = torch.tensor(R_test, dtype=torch.float32)

        # Baseline
        F = solver_baseline._build_F_from_wc(R_t, t_t)
        C = solver_baseline.compute_cost_matrix(F)
        with torch.no_grad():
            T, _ = solver_baseline.unbalanced_sinkhorn_algorithm(
                C, epsilon=epsilon, rho=rho,
                gate_mask=solver_baseline._last_gate_mask
            )
        T_sum_baseline = T.sum().item()
        avg_cost_baseline = (T * C).sum().item() / (T_sum_baseline + 1e-10)

        # With normalized cov (use middle lambda by default)
        lambda_mid = lambda_cov_list[len(lambda_cov_list) // 2]
        lambda_eff = lambda_mid * scale_ratio
        solver_cov = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2,
            k1=K, k2=K, device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=lambda_eff, lambda_epipolar=1.0,
            sigma_epipolar=400.0,
            ot_mass1=ot_mass1, ot_mass2=ot_mass2,
        )
        F = solver_cov._build_F_from_wc(R_t, t_t)
        C = solver_cov.compute_cost_matrix(F)
        with torch.no_grad():
            T, _ = solver_cov.unbalanced_sinkhorn_algorithm(
                C, epsilon=epsilon, rho=rho,
                gate_mask=solver_cov._last_gate_mask
            )
        T_sum_cov = T.sum().item()
        avg_cost_cov = (T * C).sum().item() / (T_sum_cov + 1e-10)

        R_err = rotation_error(R_test, R_gt)
        results_comparison.append({
            'angle': angle,
            'R_err': R_err,
            'T_sum_baseline': T_sum_baseline,
            'T_sum_cov': T_sum_cov,
            'avg_cost_baseline': avg_cost_baseline,
            'avg_cost_cov': avg_cost_cov,
        })

    print(f"\n{'Perturb':>8} {'R_err':>8} {'T_base':>8} {'T_cov':>8} {'cost_base':>10} {'cost_cov':>10}")
    print("-" * 60)
    for r in results_comparison:
        print(f"{r['angle']:>8}° {r['R_err']:>8.1f} {r['T_sum_baseline']:>8.3f} {r['T_sum_cov']:>8.3f} "
              f"{r['avg_cost_baseline']:>10.4f} {r['avg_cost_cov']:>10.4f}")

    return results_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step D: Structural Descriptors")
    parser.add_argument("--idx1", type=int, default=0, help="First image index")
    parser.add_argument("--idx2", type=int, default=10, help="Second image index")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--rho", type=float, default=0.5, help="Unbalanced OT rho")
    args = parser.parse_args()

    run_structural_descriptor_test(
        idx1=args.idx1,
        idx2=args.idx2,
        epsilon=args.epsilon,
        rho=args.rho,
    )
