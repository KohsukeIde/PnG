"""
Step XIX-XX: Verification and Implementation

This script:
1. Verification ①: Check if 60deg failure is due to mass collapse
2. Verification ②: Check ε/ρ input vs actual values
3. Verification ③: Statistical test for translation landscape stability
4. Verification ④: Check if translation_only gradient is zero vs not progressing
5. Step XIX: Two-stage optimization (avg_cost coarse → full_uot selection)
6. Step XX: Closed-form translation update via minimum eigenvector
7. Step XXV: full_uot optimization with differentiable transport
"""

import csv
import os
import sys
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.primitive.camera import Lie
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_step_f_score_functions import compute_all_scores


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
    """Compute relative pose (world-to-camera)."""
    R1_w2c, t1_w2c = cam1['R'], cam1['t']
    R2_w2c, t2_w2c = cam2['R'], cam2['t']
    R_12 = R2_w2c @ R1_w2c.T
    t_12 = t2_w2c - R_12 @ t1_w2c
    t_12_norm = t_12 / (np.linalg.norm(t_12) + 1e-10)
    return R_12, t_12_norm


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
    """Compute translation direction error in degrees."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def perturb_pose_wc(R_gt_wc, t_gt_wc, rot_deg, trans_deg=0.0):
    """Perturb ground truth pose."""
    angle_rad = np.radians(rot_deg)
    R_perturb = rodrigues_rotation(np.array([0, 1, 0]), angle_rad)
    R_init_wc = R_perturb @ R_gt_wc
    if trans_deg > 0:
        t_perturb = rodrigues_rotation(np.array([1, 0, 0]), np.radians(trans_deg))
        t_init_wc = t_perturb @ t_gt_wc
        t_init_wc = t_init_wc / (np.linalg.norm(t_init_wc) + 1e-10)
    else:
        t_init_wc = t_gt_wc.copy()
    return R_init_wc, t_init_wc


def _subsample_gaussians(gaussians: TwoDGaussians, indices: np.ndarray) -> TwoDGaussians:
    """Subsample TwoDGaussians by indices."""
    return TwoDGaussians(
        means=gaussians.means[indices],
        covs=gaussians.covs[indices],
        rgb=gaussians.rgb[indices],
        alpha=gaussians.alpha[indices],
        rotations=gaussians.rotations[indices],
        scales=gaussians.scales[indices],
    )


def _subsample_optional_array(arr: Optional[np.ndarray], indices: np.ndarray, expected_len: int):
    if arr is None:
        return None
    if hasattr(arr, "shape") and arr.shape[0] == expected_len:
        return arr[indices]
    return arr


def _subsample_data(data: dict, indices: np.ndarray) -> dict:
    """Subsample a gaussian data dict returned by load_gaussians()."""
    g = data["original_gaussians"]
    subsampled = {"original_gaussians": _subsample_gaussians(g, indices)}
    if "ot_mass" in data:
        subsampled["ot_mass"] = _subsample_optional_array(data["ot_mass"], indices, g.k)
    if "S_k" in data:
        subsampled["S_k"] = _subsample_optional_array(data["S_k"], indices, g.k)
    return subsampled


def _sample_indices(k: int, subset_size: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if subset_size is None or subset_size >= k:
        return np.arange(k)
    return rng.choice(k, size=subset_size, replace=False)


def _parse_float_list(env_var: str, default_list: List[float]) -> List[float]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default_list
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values if values else default_list


def compute_cov_scale_ratio(
    gaussians1: TwoDGaussians,
    gaussians2: TwoDGaussians,
    K: np.ndarray,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    sigma_epipolar: float = 400.0,
    sigma_cov: float = 1.0,
    epipolar_mode: str = "sampson",
    ot_mass1: Optional[np.ndarray] = None,
    ot_mass2: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute scale ratio to normalize lambda_cov against epipolar cost."""
    solver_epi = OptimalTransportSolver(
        gaussians1=gaussians1, gaussians2=gaussians2,
        k1=K, k2=K, device="cpu",
        epipolar_mode=epipolar_mode,
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=sigma_epipolar, sigma_cov=sigma_cov,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )
    solver_cov = OptimalTransportSolver(
        gaussians1=gaussians1, gaussians2=gaussians2,
        k1=K, k2=K, device="cpu",
        epipolar_mode=epipolar_mode,
        lambda_color=0.0, lambda_cov=1.0, lambda_epipolar=0.0,
        sigma_epipolar=sigma_epipolar, sigma_cov=sigma_cov,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    R_t = torch.tensor(R_wc, dtype=torch.float32)
    t_t = torch.tensor(t_wc, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    F_epi = solver_epi._build_F_from_wc(R_t, t_t)
    C_epi = solver_epi.compute_cost_matrix(F_epi)

    F_cov = solver_cov._build_F_from_wc(R_t, t_t)
    C_cov = solver_cov.compute_cost_matrix(F_cov)

    median_epi = torch.median(C_epi).item()
    median_cov = torch.median(C_cov).item()
    scale_ratio = median_epi / (median_cov + 1e-10)
    return scale_ratio, median_epi, median_cov


# =============================================================================
# Verification ①: Check if 60deg failure is due to mass collapse
# =============================================================================

def verify_collapse_at_60deg(idx1: int = 0, idx2: int = 10, max_iter: int = 50):
    """
    Run optimization at 60deg and log T.sum, grad norms per iteration
    to verify if collapse is causing gradient death.
    """
    print("=" * 70)
    print("Verification ①: 60deg collapse check")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 60.0)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0, epi_clip=None,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    R_wc_t = torch.tensor(R_init_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_init_wc, dtype=torch.float32)
    rot_vec = torch.nn.Parameter(lie.SO3_to_so3(R_wc_t).clone())
    trans_vec = torch.nn.Parameter(t_wc_t.clone())

    optimizer = torch.optim.SGD([
        {'params': rot_vec, 'lr': 1e-3},
        {'params': trans_vec, 'lr': 1e-4},
    ], momentum=0.9, nesterov=True)

    epsilon = 0.05
    rho = 0.5
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    records = []

    print(f"\n{'Iter':>5} {'T.sum':>10} {'<T,C>':>12} {'avg_cost':>12} {'full_uot':>12} "
          f"{'rot_grad':>10} {'trans_grad':>10} {'cost_med':>10}")
    print("-" * 100)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        R_wc = lie.so3_to_SO3(rot_vec)
        t_wc = trans_vec / (trans_vec.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=epsilon, rho=rho,
                gate_mask=solver._last_gate_mask
            )
            transport = transport.detach()

        eps_actual = solver._last_sinkhorn_epsilon
        rho_actual = solver._last_sinkhorn_rho

        # Compute scores (for logging only - returns scalars)
        scores = compute_all_scores(transport, cost_matrix, a, b, eps_actual, rho_actual)

        # Compute avg_cost as tensor for gradient
        transport_cost = torch.sum(transport * cost_matrix)
        T_sum = transport.sum()
        loss = transport_cost / (T_sum + 1e-10)
        loss.backward()

        rot_grad_norm = rot_vec.grad.norm().item() if rot_vec.grad is not None else 0.0
        trans_grad_norm = trans_vec.grad.norm().item() if trans_vec.grad is not None else 0.0

        cost_median = torch.median(cost_matrix).item()

        records.append({
            'iter': iteration,
            'T_sum': scores['T_sum'],
            'transport_cost': scores['loss'],  # 'loss' = <T,C>
            'avg_cost': scores['avg_cost'],
            'full_uot': scores['full_uot'],
            'rot_grad_norm': rot_grad_norm,
            'trans_grad_norm': trans_grad_norm,
            'cost_median': cost_median,
            'eps_actual': eps_actual,
            'rho_actual': rho_actual,
        })

        if iteration % 5 == 0:
            print(f"{iteration:>5} {scores['T_sum']:>10.4f} {scores['loss']:>12.4f} "
                  f"{scores['avg_cost']:>12.4f} {scores['full_uot']:>12.4f} "
                  f"{rot_grad_norm:>10.4e} {trans_grad_norm:>10.4e} {cost_median:>10.2f}")

        optimizer.step()
        with torch.no_grad():
            trans_vec.div_(trans_vec.norm() + 1e-10)

    # Analysis
    print("\n" + "=" * 70)
    print("Analysis:")
    T_sums = [r['T_sum'] for r in records]
    grad_norms = [r['rot_grad_norm'] + r['trans_grad_norm'] for r in records]

    if T_sums[0] < 0.1:
        print("  ⚠️  Initial T.sum is very small - likely collapse from start")
    if min(T_sums) < 0.01:
        print("  ⚠️  T.sum drops below 0.01 - severe collapse")
    if max(grad_norms) < 1e-4:
        print("  ⚠️  Gradient norms are tiny - gradient death confirmed")

    # Correlation between T.sum and grad_norm
    corr = np.corrcoef(T_sums, grad_norms)[0, 1]
    print(f"  Correlation(T.sum, grad_norm) = {corr:.3f}")
    if corr > 0.5:
        print("  → Strong positive correlation: collapse causes gradient death")

    return records


# =============================================================================
# Verification ②: Check ε/ρ input vs actual values
# =============================================================================

def verify_epsilon_rho_consistency():
    """Check that input ε/ρ matches actual values used in Sinkhorn."""
    print("\n" + "=" * 70)
    print("Verification ②: ε/ρ input vs actual")
    print("=" * 70)

    idx1, idx2 = 0, 10
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']
    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
    )

    R_wc_t = torch.tensor(R_gt_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_gt_wc, dtype=torch.float32)
    F = solver._build_F_from_wc(R_wc_t, t_wc_t)
    cost_matrix = solver.compute_cost_matrix(F)

    test_cases = [
        (0.05, 0.5),
        (0.1, 1.0),
        (0.2, 2.0),
        (None, None),  # auto
    ]

    print(f"\n{'Input ε':>12} {'Input ρ':>12} {'Actual ε':>12} {'Actual ρ':>12} {'Match':>8}")
    print("-" * 60)

    for eps_in, rho_in in test_cases:
        with torch.no_grad():
            _, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=eps_in, rho=rho_in
            )
        eps_out = solver._last_sinkhorn_epsilon
        rho_out = solver._last_sinkhorn_rho

        # Note: Sinkhorn uses 2-stage ε-scaling: [ε, 0.5*ε]
        # So actual ε = 0.5 * input ε
        eps_in_str = f"{eps_in:.4f}" if eps_in else "auto"
        rho_in_str = f"{rho_in:.4f}" if rho_in else "auto"

        match = "✓" if eps_in is None or abs(eps_out - 0.5 * eps_in) < 1e-6 else "✗"

        print(f"{eps_in_str:>12} {rho_in_str:>12} {eps_out:>12.4f} {rho_out:>12.4f} {match:>8}")

    print("\nNote: Sinkhorn internally uses 2-stage ε-scaling: [ε, 0.5*ε]")
    print("      So actual ε = 0.5 * input ε (this is expected behavior)")


# =============================================================================
# Verification ③: Statistical test for translation landscape stability
# =============================================================================

def verify_translation_landscape_stability(
    n_seeds: int = 5,
    subset_size: Optional[int] = None,
):
    """
    Test translation landscape stability across different random seeds
    for Gaussian subsampling.
    """
    print("\n" + "=" * 70)
    print("Verification ③: Translation landscape stability (multi-seed)")
    print("=" * 70)

    idx1, idx2 = 0, 10

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']
    R_wc, t_wc_gt = compute_relative_pose_wc(cam1, cam2)

    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    angles = np.arange(-60, 61, 10)

    results_by_seed = []

    for seed in range(n_seeds):
        print(f"\nSeed {seed}:")
        rng = np.random.default_rng(seed)

        # Load with potential random subsampling (use full for now)
        data1 = load_gaussians(idx1)
        data2 = load_gaussians(idx2)
        g1_full = data1['original_gaussians']
        g2_full = data2['original_gaussians']
        subset_k = subset_size
        if subset_k is not None:
            subset_k = min(subset_k, g1_full.k, g2_full.k)
        idxs1 = _sample_indices(g1_full.k, subset_k, rng)
        idxs2 = _sample_indices(g2_full.k, subset_k, rng)
        data1 = _subsample_data(data1, idxs1)
        data2 = _subsample_data(data2, idxs2)

        g1 = data1['original_gaussians']
        g2 = data2['original_gaussians']
        ot_mass1 = data1.get('ot_mass', None)
        ot_mass2 = data2.get('ot_mass', None)
        print(f"  Subsample: K1={g1.k}, K2={g2.k}")

        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2,
            k1=K, k2=K, device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
            sigma_epipolar=400.0,
            ot_mass1=ot_mass1, ot_mass2=ot_mass2,
        )

        epsilon = 0.05
        rho = 0.5
        a = solver.alpha1 / solver.alpha1.sum()
        b = solver.alpha2 / solver.alpha2.sum()

        min_angles = {}

        with torch.no_grad():
            R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
            for axis_name, axis_vec in axes.items():
                scores_list = []
                for angle in angles:
                    R_delta = rodrigues_rotation(axis_vec, np.radians(angle))
                    t_wc = R_delta @ t_wc_gt
                    t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)
                    t_wc_t = torch.tensor(t_wc, dtype=torch.float32)

                    F = solver._build_F_from_wc(R_wc_t, t_wc_t)
                    cost = solver.compute_cost_matrix(F)
                    transport, _ = solver.unbalanced_sinkhorn_algorithm(
                        cost, epsilon=epsilon, rho=rho,
                        gate_mask=solver._last_gate_mask
                    )
                    eps_actual = solver._last_sinkhorn_epsilon
                    rho_actual = solver._last_sinkhorn_rho
                    scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)
                    scores_list.append(scores['full_uot'])

                min_idx = np.argmin(scores_list)
                min_angle = angles[min_idx]
                min_angles[axis_name] = min_angle
                print(f"  axis={axis_name}: min at {min_angle}deg")

        results_by_seed.append(min_angles)

    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary across seeds:")
    for axis_name in axes:
        min_angles_axis = [r[axis_name] for r in results_by_seed]
        mean_angle = np.mean(min_angles_axis)
        std_angle = np.std(min_angles_axis)
        print(f"  axis={axis_name}: mean={mean_angle:.1f}deg, std={std_angle:.1f}deg")
        if std_angle > 10:
            print(f"    ⚠️  High variance - landscape is unstable for this axis")
        elif abs(mean_angle) > 10:
            print(f"    ⚠️  Mean offset from GT - possible bias")
        else:
            print(f"    ✓  Stable and near GT")


# =============================================================================
# Verification ④: translation_only gradient analysis
# =============================================================================

def verify_translation_gradient(max_iter: int = 30):
    """
    Check if translation_only gradient is zero vs not progressing.
    """
    print("\n" + "=" * 70)
    print("Verification ④: translation_only gradient analysis")
    print("=" * 70)

    idx1, idx2 = 0, 10

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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    # Perturb translation by 30deg, keep R at GT
    _, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 0.0, 30.0)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    R_wc_t = torch.tensor(R_gt_wc, dtype=torch.float32)  # Fixed at GT
    trans_vec = torch.nn.Parameter(torch.tensor(t_init_wc, dtype=torch.float32))

    optimizer = torch.optim.SGD([{'params': trans_vec, 'lr': 1e-3}], momentum=0.9)

    epsilon = 0.05
    rho = 0.5

    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    print(f"\nInitial translation error: {init_trans_err:.2f}deg")
    print(f"R is fixed at GT")
    print(f"\n{'Iter':>5} {'trans_err':>12} {'loss':>12} {'grad_norm':>12} {'update_deg':>12}")
    print("-" * 60)

    prev_t = trans_vec.data.clone()

    for iteration in range(max_iter):
        optimizer.zero_grad()

        t_wc = trans_vec / (trans_vec.norm() + 1e-10)
        F = solver._build_F_from_wc(R_wc_t, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=epsilon, rho=rho,
                gate_mask=solver._last_gate_mask
            )
            transport = transport.detach()

        transport_cost = torch.sum(transport * cost_matrix)
        T_sum = transport.sum()
        loss = transport_cost / (T_sum + 1e-10)  # avg_cost

        loss.backward()

        grad_norm = trans_vec.grad.norm().item() if trans_vec.grad is not None else 0.0

        optimizer.step()
        with torch.no_grad():
            trans_vec.div_(trans_vec.norm() + 1e-10)

        # Compute update in degrees
        with torch.no_grad():
            t_curr = trans_vec.data
            cos_angle = torch.dot(prev_t / prev_t.norm(), t_curr / t_curr.norm())
            update_deg = np.degrees(np.arccos(np.clip(cos_angle.item(), -1, 1)))
            prev_t = t_curr.clone()

        t_np = trans_vec.detach().numpy()
        trans_err = min(
            translation_error(t_np, t_gt_wc),
            translation_error(-t_np, t_gt_wc)
        )

        if iteration % 3 == 0:
            print(f"{iteration:>5} {trans_err:>12.2f} {loss.item():>12.4f} "
                  f"{grad_norm:>12.4e} {update_deg:>12.4f}")

    final_trans_err = trans_err
    print(f"\nFinal translation error: {final_trans_err:.2f}deg")
    print(f"Improvement: {init_trans_err - final_trans_err:.2f}deg")

    if grad_norm < 1e-6:
        print("\n⚠️  Gradient is near zero - objective is insensitive to t")
    elif final_trans_err > init_trans_err - 1.0:
        print("\n⚠️  Translation barely moved despite gradient - check learning rate or landscape")
    else:
        print("\n✓  Translation optimization is working")


# =============================================================================
# Step XIX: Two-stage optimization (avg_cost coarse → full_uot selection)
# =============================================================================

def run_step_xix_two_stage_optimization(
    idx1: int = 0,
    idx2: int = 10,
    init_rot_error_deg: float = 60.0,
    stage1_iters: int = 100,
    stage2_iters: int = 50,
):
    """
    Step XIX: Two-stage optimization
    - Stage A (coarse): Optimize with avg_cost (scale-invariant, won't collapse gradient)
    - Stage B (refine): Continue with full_uot for model selection
    """
    print("\n" + "=" * 70)
    print(f"Step XIX: Two-stage optimization (init_rot_err={init_rot_error_deg}deg)")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, init_rot_error_deg)

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    print(f"Initial errors: R={init_rot_err:.2f}deg, t={init_trans_err:.2f}deg")

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    R_wc_t = torch.tensor(R_init_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_init_wc, dtype=torch.float32)
    rot_vec = torch.nn.Parameter(lie.SO3_to_so3(R_wc_t).clone())
    trans_vec = torch.nn.Parameter(t_wc_t.clone())

    optimizer = torch.optim.SGD([
        {'params': rot_vec, 'lr': 1e-3},
        {'params': trans_vec, 'lr': 1e-4},
    ], momentum=0.9, nesterov=True)

    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    # Epsilon annealing schedule
    epsilon_start = 0.2
    epsilon_end = 0.05
    rho = 0.5

    best_full_uot = float('inf')
    best_rot_vec = rot_vec.data.clone()
    best_trans_vec = trans_vec.data.clone()

    # ====================== Stage A: avg_cost (coarse) ======================
    print(f"\n--- Stage A: avg_cost optimization ({stage1_iters} iters) ---")
    print(f"{'Iter':>5} {'R_err':>8} {'t_err':>8} {'avg_cost':>12} {'full_uot':>12} {'T.sum':>10}")
    print("-" * 60)

    for iteration in range(stage1_iters):
        optimizer.zero_grad()

        # Epsilon annealing
        t = min(iteration / 50, 1.0)
        current_epsilon = epsilon_start * (1 - t) + epsilon_end * t
        current_rho = rho * (current_epsilon / epsilon_end)

        R_wc = lie.so3_to_SO3(rot_vec)
        t_wc = trans_vec / (trans_vec.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=current_epsilon, rho=current_rho,
                gate_mask=solver._last_gate_mask
            )
            transport = transport.detach()

        eps_actual = solver._last_sinkhorn_epsilon
        rho_actual = solver._last_sinkhorn_rho
        scores = compute_all_scores(transport, cost_matrix, a, b, eps_actual, rho_actual)

        # Compute avg_cost as tensor for gradient
        transport_cost_t = torch.sum(transport * cost_matrix)
        T_sum_t = transport.sum()
        loss = transport_cost_t / (T_sum_t + 1e-10)
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            trans_vec.div_(trans_vec.norm() + 1e-10)

        # Track best by full_uot
        if scores['full_uot'] < best_full_uot:
            best_full_uot = scores['full_uot']
            best_rot_vec = rot_vec.data.clone()
            best_trans_vec = trans_vec.data.clone()

        if iteration % 10 == 0:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
                t_wc_np = (trans_vec / trans_vec.norm()).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            print(f"{iteration:>5} {r_err:>8.2f} {t_err:>8.2f} {scores['avg_cost']:>12.4f} "
                  f"{scores['full_uot']:>12.4f} {scores['T_sum']:>10.4f}")

    # ====================== Stage B: full_uot (refine/select) ======================
    print(f"\n--- Stage B: full_uot refinement ({stage2_iters} iters) ---")

    for iteration in range(stage2_iters):
        optimizer.zero_grad()

        R_wc = lie.so3_to_SO3(rot_vec)
        t_wc = trans_vec / (trans_vec.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=epsilon_end, rho=rho,
                gate_mask=solver._last_gate_mask
            )
            transport = transport.detach()

        eps_actual = solver._last_sinkhorn_epsilon
        rho_actual = solver._last_sinkhorn_rho
        scores = compute_all_scores(transport, cost_matrix, a, b, eps_actual, rho_actual)

        # Compute avg_cost as tensor for gradient
        transport_cost_t = torch.sum(transport * cost_matrix)
        T_sum_t = transport.sum()
        loss = transport_cost_t / (T_sum_t + 1e-10)
        loss.backward()

        optimizer.step()
        with torch.no_grad():
            trans_vec.div_(trans_vec.norm() + 1e-10)

        # Track best by full_uot
        if scores['full_uot'] < best_full_uot:
            best_full_uot = scores['full_uot']
            best_rot_vec = rot_vec.data.clone()
            best_trans_vec = trans_vec.data.clone()

        if iteration % 10 == 0:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
                t_wc_np = (trans_vec / trans_vec.norm()).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            print(f"{stage1_iters + iteration:>5} {r_err:>8.2f} {t_err:>8.2f} "
                  f"{scores['avg_cost']:>12.4f} {scores['full_uot']:>12.4f} {scores['T_sum']:>10.4f}")

    # Final result using best by full_uot
    with torch.no_grad():
        R_best = lie.so3_to_SO3(best_rot_vec).numpy()
        t_best = (best_trans_vec / best_trans_vec.norm()).numpy()

    final_rot_err = rotation_error(R_best, R_gt_wc)
    final_trans_err = min(
        translation_error(t_best, t_gt_wc),
        translation_error(-t_best, t_gt_wc)
    )

    print(f"\n{'='*60}")
    print(f"Final result (best by full_uot):")
    print(f"  Rotation error: {final_rot_err:.2f}deg (was {init_rot_err:.2f}deg)")
    print(f"  Translation error: {final_trans_err:.2f}deg (was {init_trans_err:.2f}deg)")
    print(f"  Improvement: R={init_rot_err - final_rot_err:.2f}deg, t={init_trans_err - final_trans_err:.2f}deg")

    return {
        'init_rot_err': init_rot_err,
        'init_trans_err': init_trans_err,
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
    }


# =============================================================================
# Step XX: Closed-form translation update via minimum eigenvector
# =============================================================================

def compute_closed_form_translation(
    solver: OptimalTransportSolver,
    R_wc: torch.Tensor,
    transport: torch.Tensor,
    use_hard_assignment: bool = True,
    top_k: int = 50,
    min_transport_mass: float = 0.05,
    min_topk_weight: float = 0.0,
    matching_mode: str = "global_topk",
    row_top_k: Optional[int] = None,
    diversity_grid: Optional[Tuple[int, int]] = None,
    max_per_cell: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute optimal translation direction via closed-form solution.

    For each pair (i,j) with OT weight w_ij, the epipolar constraint gives:
        x2^T @ [t]_x @ R @ x1 = 0
    which can be rewritten as:
        (x2 × (R @ x1))^T @ t = 0

    We solve:
        min_{|t|=1} Σ_{i,j} w_ij * ((a_ij)^T @ t)^2
    where a_ij = x2_j × (R @ x1_i)

    This is: t* = min eigenvector of M = Σ w_ij * a_ij @ a_ij^T

    Args:
        use_hard_assignment: If True, use top-1 match per row instead of soft weights
        top_k: Number of top correspondences to use (by transport weight)
        min_transport_mass: Minimum T.sum to attempt the update (collapse guard)
        min_topk_weight: Minimum sum of top-K weights to attempt the update

    Returns:
        Tuple of (t_opt, eigenvalues) or (None, None) if computation failed.
        - t_opt: Unit translation direction (3,)
        - eigenvalues: Sorted eigenvalues of M (3,), for gap-based stability check
    """
    total_mass = float(transport.sum().item())
    if not np.isfinite(total_mass) or total_mass < min_transport_mass:
        return None, None

    K1_inv = solver.k1_inv
    K2_inv = solver.k2_inv

    # Get homogeneous coordinates
    means1 = solver.means1  # (K1, 2)
    means2 = solver.means2  # (K2, 2)

    ones1 = torch.ones(means1.shape[0], 1, device=means1.device)
    ones2 = torch.ones(means2.shape[0], 1, device=means2.device)

    # Normalized coordinates: x = K^{-1} @ [u, v, 1]^T
    p1_hom = torch.cat([means1, ones1], dim=1)  # (K1, 3)
    p2_hom = torch.cat([means2, ones2], dim=1)  # (K2, 3)

    x1 = (K1_inv @ p1_hom.T).T  # (K1, 3)
    x2 = (K2_inv @ p2_hom.T).T  # (K2, 3)

    # R @ x1 for all points
    Rx1 = (R_wc @ x1.T).T  # (K1, 3)

    T = transport  # (K1, K2)

    if use_hard_assignment:
        T_np = T.detach().cpu().numpy()
        K1, K2 = T_np.shape
        pairs = []

        if matching_mode == "global_topk":
            T_flat = T_np.reshape(-1)
            top_k_eff = min(top_k, T_flat.size)
            top_indices = np.argpartition(-T_flat, top_k_eff - 1)[:top_k_eff]
            for idx in top_indices:
                i = idx // K2
                j = idx % K2
                pairs.append((i, j, T_flat[idx]))
        elif matching_mode == "row_topk":
            k_per_row = row_top_k or max(1, top_k)
            for i in range(K1):
                row = T_np[i]
                k_eff = min(k_per_row, K2)
                idxs = np.argpartition(-row, k_eff - 1)[:k_eff]
                for j in idxs:
                    pairs.append((i, j, row[j]))
        elif matching_mode == "mutual_top1":
            row_argmax = np.argmax(T_np, axis=1)
            col_argmax = np.argmax(T_np, axis=0)
            for i in range(K1):
                j = row_argmax[i]
                if col_argmax[j] == i:
                    pairs.append((i, j, T_np[i, j]))
        elif matching_mode == "greedy_one_to_one":
            T_flat = T_np.reshape(-1)
            order = np.argsort(-T_flat)
            used_i = set()
            used_j = set()
            for idx in order:
                i = idx // K2
                j = idx % K2
                if i in used_i or j in used_j:
                    continue
                pairs.append((i, j, T_flat[idx]))
                used_i.add(i)
                used_j.add(j)
                if len(pairs) >= top_k:
                    break
        else:
            raise ValueError(f"Unknown matching_mode: {matching_mode}")

        if diversity_grid is not None and len(pairs) > 0:
            gx, gy = diversity_grid
            means1_np = means1.detach().cpu().numpy()
            min_xy = means1_np.min(axis=0)
            max_xy = means1_np.max(axis=0)
            span = np.maximum(max_xy - min_xy, 1e-6)
            counts = {}
            diverse_pairs = []
            for i, j, w in sorted(pairs, key=lambda x: -x[2]):
                rel = (means1_np[i] - min_xy) / span
                cell_x = min(gx - 1, max(0, int(rel[0] * gx)))
                cell_y = min(gy - 1, max(0, int(rel[1] * gy)))
                key = (cell_x, cell_y)
                if counts.get(key, 0) >= max_per_cell:
                    continue
                counts[key] = counts.get(key, 0) + 1
                diverse_pairs.append((i, j, w))
            pairs = diverse_pairs

        total_weight = sum(w for _, _, w in pairs)
        if total_weight <= 0.0:
            return None, None
        if min_topk_weight > 0.0 and total_weight < min_topk_weight:
            return None, None

        M = torch.zeros(3, 3, device=T.device, dtype=T.dtype)
        for i, j, w in pairs:
            a = torch.cross(Rx1[i], x2[j], dim=0)
            M = M + w * torch.outer(a, a)
    else:
        # Original soft assignment version
        Rx1_exp = Rx1.unsqueeze(1)  # (K1, 1, 3)
        x2_exp = x2.unsqueeze(0)    # (1, K2, 3)

        # Cross product: a_ij = Rx1_i × x2_j (corrected order)
        a = torch.cross(Rx1_exp.expand(-1, x2.shape[0], -1),
                        x2_exp.expand(Rx1.shape[0], -1, -1),
                        dim=2)  # (K1, K2, 3)

        a_flat = a.reshape(-1, 3)  # (K1*K2, 3)
        T_flat = T.reshape(-1)     # (K1*K2,)

        if T_flat.sum().item() <= 0.0:
            return None, None
        M = (a_flat * T_flat.unsqueeze(1)).T @ a_flat  # (3, 3)

    # Symmetrize
    M = 0.5 * (M + M.T)
    if not torch.isfinite(M).all():
        return None, None
    if torch.linalg.norm(M).item() < 1e-12:
        return None, None

    # Minimum eigenvector
    eigenvalues, eigenvectors = torch.linalg.eigh(M)
    t_opt = eigenvectors[:, 0]  # Smallest eigenvalue's eigenvector

    # Return eigenvalues for gap-based guard
    return t_opt.detach().cpu().numpy(), eigenvalues.detach().cpu().numpy()


def test_closed_form_translation_at_gt():
    """Test closed-form t update works correctly at GT pose."""
    print("\n" + "=" * 70)
    print("Step XX Debug: Test closed-form t at GT")
    print("=" * 70)

    idx1, idx2 = 0, 10

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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Compute OT at GT pose
    R_wc_t = torch.tensor(R_gt_wc, dtype=torch.float32)
    t_wc_t = torch.tensor(t_gt_wc, dtype=torch.float32)

    F = solver._build_F_from_wc(R_wc_t, t_wc_t)
    cost_matrix = solver.compute_cost_matrix(F)

    with torch.no_grad():
        transport, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, epsilon=0.05, rho=0.5,
            gate_mask=solver._last_gate_mask
        )

    print(f"Transport at GT: T.sum = {transport.sum().item():.4f}")

    results = {}
    for label, use_hard in [("soft", False), ("hard", True)]:
        result = compute_closed_form_translation(
            solver, R_wc_t, transport,
            use_hard_assignment=use_hard,
            top_k=50,
            min_transport_mass=0.05,
        )
        t_opt, eigvals = result if result[0] is not None else (None, None)
        if t_opt is None:
            print(f"{label}: skipped (collapse guard triggered)")
            results[label] = None
            continue

        # Eigenvalue gap: (λ2-λ1)/λ3
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)

        # Compare with GT
        t_err1 = translation_error(t_opt, t_gt_wc)
        t_err2 = translation_error(-t_opt, t_gt_wc)
        t_err = min(t_err1, t_err2)

        print(f"{label} t: {t_opt}")
        print(f"{label} translation error: {t_err:.2f}deg, gap: {gap:.6f}")
        results[label] = t_err

    return results


def run_step_xx_closed_form_translation(
    idx1: int = 0,
    idx2: int = 10,
    init_rot_error_deg: float = 30.0,
    r_converge_iters: int = 80,
    em_iters: int = 20,
    use_hard_assignment: bool = True,
    top_k: int = 50,
    min_transport_mass: float = 0.05,
    lambda_cov: float = 0.0,
    lambda_cov_raw: Optional[float] = None,
    cov_scale_ratio: Optional[float] = None,
    epsilon_start: float = None,  # Step XXVI: auto-determined from cost scale
    epsilon_end: float = 0.05,
):
    """
    Step XX: Two-phase optimization:
    Phase 1: Converge R using avg_cost (t fixed at GT initially)
    Phase 2: EM-style alternating - R gradient + t closed-form
    """
    print("\n" + "=" * 70)
    print(f"Step XX: Closed-form translation (init_rot_err={init_rot_error_deg}deg)")
    print(f"  Closed-form t: {'hard' if use_hard_assignment else 'soft'} "
          f"(top_k={top_k}, min_T_sum={min_transport_mass})")
    if lambda_cov_raw is not None and cov_scale_ratio is not None:
        print(f"  lambda_cov: raw={lambda_cov_raw:.3f}, scale_ratio={cov_scale_ratio:.6f}, "
              f"eff={lambda_cov:.6f}")
    else:
        print(f"  lambda_cov: {lambda_cov:.6f}")
    print(f"  Epsilon: {epsilon_start} -> {epsilon_end}")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, init_rot_error_deg, 20.0)

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    print(f"Initial errors: R={init_rot_err:.2f}deg, t={init_trans_err:.2f}deg")

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=lambda_cov, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    rot_vec = torch.nn.Parameter(lie.SO3_to_so3(torch.tensor(R_init_wc, dtype=torch.float32)).clone())
    # Start with perturbed t (not GT)
    t_wc_np = t_init_wc.copy()

    rho = 0.5

    # ====================== Step XXVI-B: Auto-determine epsilon_start ======================
    # Compute initial cost to determine appropriate epsilon_start
    # Goal: cost_median / eps_actual <= 50 to avoid exp underflow
    with torch.no_grad():
        R_wc_init = torch.tensor(R_init_wc, dtype=torch.float32)
        t_wc_init = torch.tensor(t_init_wc, dtype=torch.float32)
        t_wc_init = t_wc_init / (t_wc_init.norm() + 1e-10)
        F_init = solver._build_F_from_wc(R_wc_init, t_wc_init)
        cost_init = solver.compute_cost_matrix(F_init)
        cost_median = torch.median(cost_init).item()

    if epsilon_start is None:
        # Auto-determine: target cost_median / (0.5 * eps_start) <= 50
        # => eps_start >= cost_median / 25
        eps_auto = max(0.2, cost_median / 25.0)
        eps_auto = min(eps_auto, 2.0)  # Cap at 2.0
        epsilon_start = eps_auto
        print(f"  Auto ε_start: {epsilon_start:.3f} (cost_median={cost_median:.2f})")
    else:
        print(f"  ε_start: {epsilon_start:.3f} (cost_median={cost_median:.2f})")

    # ====================== Step XXVI-A: Multi-scale LR (always increasing) ======================
    # Key insight: Use increasing LR schedule for all cases
    # The schedule is deterministic and doesn't depend on initial angle
    lr_schedule = [
        (1e-3, 0.9, r_converge_iters // 4),   # Warm-up
        (5e-3, 0.95, r_converge_iters // 4),  # Moderate
        (1e-2, 0.95, r_converge_iters // 4),  # Aggressive
        (2e-2, 0.95, r_converge_iters // 4),  # Very aggressive
    ]
    print(f"  LR schedule: {[lr for lr, _, _ in lr_schedule]}")

    # ====================== Phase 1: Converge R with avg_cost ======================
    print(f"\n--- Phase 1: R convergence ({r_converge_iters} iters, t fixed) ---")
    print(f"{'Iter':>5} {'R_err':>8} {'t_err':>8} {'avg_cost':>12} {'T.sum':>10} {'LR':>8}")
    print("-" * 60)

    global_iter = 0
    for lr, momentum, phase_iters in lr_schedule:
        optimizer = torch.optim.SGD([rot_vec], lr=lr, momentum=momentum, nesterov=True)

        for iteration in range(phase_iters):
            optimizer.zero_grad()

            # Epsilon annealing based on global iteration
            t = min(global_iter / 50, 1.0)
            current_epsilon = epsilon_start * (1 - t) + epsilon_end * t

            R_wc = lie.so3_to_SO3(rot_vec)
            t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
            t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

            F = solver._build_F_from_wc(R_wc, t_wc_t)
            cost_matrix = solver.compute_cost_matrix(F)

            with torch.no_grad():
                transport, _ = solver.unbalanced_sinkhorn_algorithm(
                    cost_matrix, epsilon=current_epsilon, rho=rho,
                    gate_mask=solver._last_gate_mask
                )
                transport = transport.detach()

            # Compute loss with grad for R
            transport_cost = torch.sum(transport * cost_matrix)
            T_sum = transport.sum()
            loss = transport_cost / (T_sum + 1e-10)

            loss.backward()
            optimizer.step()

            if global_iter % 10 == 0:
                with torch.no_grad():
                    R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
                r_err = rotation_error(R_wc_np, R_gt_wc)
                t_err = min(
                    translation_error(t_wc_np, t_gt_wc),
                    translation_error(-t_wc_np, t_gt_wc)
                )
                print(f"{global_iter:>5} {r_err:>8.2f} {t_err:>8.2f} {loss.item():>12.4f} {T_sum.item():>10.4f} {lr:>8.0e}")

            global_iter += 1

    # ====================== Phase 2: EM alternating R/t ======================
    print(f"\n--- Phase 2: EM alternating ({em_iters} iters) ---")

    # Track best solution seen (by eigen-gap)
    best_t_wc = t_wc_np.copy()
    best_gap_val = -float('inf')
    prev_T_sum = None
    T_sum_stable_threshold = 0.1  # T.sum change threshold for stability

    for iteration in range(em_iters):
        # E-step: Compute OT
        with torch.no_grad():
            R_wc = lie.so3_to_SO3(rot_vec)
        t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
        t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

        cost_matrix, transport, _ = compute_transport_with_gate(
            solver,
            R_wc,
            t_wc_t,
            epsilon_end,
            rho,
            gate_solvers=gate_solvers,
            gate_top_m_epi=gate_top_m_epi or 0,
            gate_top_m_cov=gate_top_m_cov or 0,
        )

        # M-step for R (gradient)
        optimizer.zero_grad()
        R_wc = lie.so3_to_SO3(rot_vec)
        F = solver._build_F_from_wc(R_wc, t_wc_t.detach())
        cost_matrix = solver.compute_cost_matrix(F)

        transport_cost = torch.sum(transport.detach() * cost_matrix)
        T_sum = transport.sum()
        loss = transport_cost / (T_sum + 1e-10)

        loss.backward()
        optimizer.step()

        # Check T.sum stability before t update
        current_T_sum = T_sum.item()
        T_sum_stable = (prev_T_sum is None or
                        abs(current_T_sum - prev_T_sum) < T_sum_stable_threshold)
        prev_T_sum = current_T_sum

        # M-step for t (closed-form with gap-based guard)
        gap_val = None
        damping_val = None
        skip_reason = None

        with torch.no_grad():
            R_wc_updated = lie.so3_to_SO3(rot_vec)

            # Only update t if T.sum is stable
            if not T_sum_stable:
                skip_reason = "T.sum unstable"
            else:
                result = compute_closed_form_translation(
                    solver,
                    R_wc_updated,
                    transport,
                    use_hard_assignment=use_hard_assignment,
                    top_k=top_k,
                    min_transport_mass=min_transport_mass,
                )
                t_opt, eigvals = result if result[0] is not None else (None, None)

                if t_opt is None:
                    skip_reason = "collapse"
                else:
                    # Eigenvalue gap-based stability check: gap = (λ2-λ1)/λ3
                    # Small gap means the minimum eigenvector is unstable
                    gap_val = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
                    gap_threshold = 0.01  # Skip if gap too small

                    if gap_val < gap_threshold:
                        skip_reason = f"gap={gap_val:.4f}"
                    else:
                        # Handle sign ambiguity
                        if np.dot(t_opt, t_wc_np) < 0:
                            t_opt = -t_opt

                        # Damped update based on gap (higher gap = more confidence)
                        damping_val = min(1.0, gap_val / 0.05)  # Full update when gap >= 0.05
                        t_new = damping_val * t_opt + (1 - damping_val) * t_wc_np
                        t_new = t_new / (np.linalg.norm(t_new) + 1e-10)
                        t_wc_np = t_new

                        if gap_val > best_gap_val:
                            best_t_wc = t_wc_np.copy()
                            best_gap_val = gap_val

        if iteration % 5 == 0:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            status = ""
            if skip_reason is not None:
                status = f" [skip: {skip_reason}]"
            elif damping_val is not None and damping_val < 1.0:
                status = f" [damp={damping_val:.2f}]"
            print(f"{r_converge_iters + iteration:>5} {r_err:>8.2f} {t_err:>8.2f} {loss.item():>12.4f} {T_sum.item():>10.4f}{status}")

    # Final result - use best t seen during optimization
    with torch.no_grad():
        R_final = lie.so3_to_SO3(rot_vec).numpy()

    # Use best t observed during Phase 2
    final_t = best_t_wc

    final_rot_err = rotation_error(R_final, R_gt_wc)
    final_trans_err = min(
        translation_error(final_t, t_gt_wc),
        translation_error(-final_t, t_gt_wc)
    )

    # Evaluate geometry metrics at epsilon_end
    scores = compute_geometry_score(
        solver,
        torch.tensor(R_final, dtype=torch.float32),
        torch.tensor(final_t, dtype=torch.float32),
        epsilon_end,
        rho,
    )

    print(f"\n{'='*50}")
    print(f"Final result:")
    print(f"  Rotation error: {final_rot_err:.2f}deg (was {init_rot_err:.2f}deg)")
    print(f"  Translation error: {final_trans_err:.2f}deg (was {init_trans_err:.2f}deg)")
    print(f"  Improvement: R={init_rot_err - final_rot_err:.2f}deg, t={init_trans_err - final_trans_err:.2f}deg")
    print(f"  avg_cost@ε_end: {scores['avg_cost']:.4f}, conc: {scores['concentration']:.3f}, T_sum: {scores['T_sum']:.3f}")

    return {
        'init_rot_err': init_rot_err,
        'init_trans_err': init_trans_err,
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'final_avg_cost': scores['avg_cost'],
        'final_concentration': scores['concentration'],
        'final_T_sum': scores['T_sum'],
    }


# =============================================================================
# Step XXV: full_uot optimization with differentiable transport
# =============================================================================

def _compute_full_uot_loss(
    transport: torch.Tensor,
    cost_matrix: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps_actual: float,
    rho_actual: float,
) -> torch.Tensor:
    eps_safe = 1e-10
    transport_cost = torch.sum(transport * cost_matrix)
    row_sum = transport.sum(dim=1)
    col_sum = transport.sum(dim=0)
    KL_row = (row_sum * torch.log((row_sum + eps_safe) / (a + eps_safe)) - row_sum + a).sum()
    KL_col = (col_sum * torch.log((col_sum + eps_safe) / (b + eps_safe)) - col_sum + b).sum()
    entropy = -(transport * torch.log(transport + eps_safe)).sum()
    entropic = -entropy - transport.sum()
    return transport_cost + rho_actual * (KL_row + KL_col) + eps_actual * entropic


def run_step_xxv_full_uot_differentiable(
    idx1: int = 0,
    idx2: int = 10,
    init_rot_error_deg: float = 60.0,
    init_trans_error_deg: float = 0.0,
    max_iter: int = 50,
    differentiable_transport: bool = True,
    epsilon_start: float = 0.05,
    epsilon_end: float = 0.05,
    anneal_steps: int = 0,
    rho: float = 0.5,
    rot_lr: float = 1e-3,
    trans_lr: float = 1e-4,
    optimize_translation: bool = False,
):
    """
    Step XXV: Test whether full_uot can recover from collapse when transport is differentiable.
    """
    label = "diff" if differentiable_transport else "detach"
    print("\n" + "=" * 70)
    print(f"Step XXV: full_uot optimization ({label})")
    print(f"  init_rot_err={init_rot_error_deg}deg, init_trans_err={init_trans_error_deg}deg")
    print(f"  epsilon: {epsilon_start} -> {epsilon_end} (anneal_steps={anneal_steps})")
    print(f"  optimize_translation={optimize_translation}")
    print("=" * 70)

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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(
        R_gt_wc, t_gt_wc, init_rot_error_deg, init_trans_error_deg
    )

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )
    print(f"Initial errors: R={init_rot_err:.2f}deg, t={init_trans_err:.2f}deg")

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    rot_vec = torch.nn.Parameter(
        lie.SO3_to_so3(torch.tensor(R_init_wc, dtype=torch.float32)).clone()
    )

    if optimize_translation:
        trans_vec = torch.nn.Parameter(torch.tensor(t_init_wc, dtype=torch.float32))
        optimizer = torch.optim.SGD(
            [{'params': rot_vec, 'lr': rot_lr}, {'params': trans_vec, 'lr': trans_lr}],
            momentum=0.9,
            nesterov=True,
        )
    else:
        trans_vec = torch.tensor(t_init_wc, dtype=torch.float32)
        optimizer = torch.optim.SGD([{'params': rot_vec, 'lr': rot_lr}], momentum=0.9, nesterov=True)

    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    best_loss = float('inf')
    best_rot_vec = rot_vec.data.clone()
    best_trans_vec = trans_vec.data.clone() if optimize_translation else None
    best_r_err = float('inf')
    best_r_rot_vec = rot_vec.data.clone()
    best_r_trans_vec = trans_vec.data.clone() if optimize_translation else None
    best_r_iter = 0
    best_full_uot_iter = 0
    records = []
    collapse_underflow = 0
    collapse_uot_zero = 0
    collapse_both = 0
    collapse_t_sum_threshold = 1e-4

    print(f"\n{'Iter':>5} {'R_err':>8} {'t_err':>8} {'full_uot':>12} {'T.sum':>10} "
          f"{'rot_grad':>10} {'trans_grad':>11}")
    print("-" * 80)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        if anneal_steps > 0:
            t = min(iteration / anneal_steps, 1.0)
            current_epsilon = epsilon_start * (1 - t) + epsilon_end * t
        else:
            current_epsilon = epsilon_start
        current_rho = rho * (current_epsilon / max(epsilon_end, 1e-8))

        R_wc = lie.so3_to_SO3(rot_vec)
        if optimize_translation:
            t_wc = trans_vec / (trans_vec.norm() + 1e-10)
        else:
            t_wc = trans_vec / (trans_vec.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)

        if differentiable_transport:
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=current_epsilon, rho=current_rho,
                gate_mask=solver._last_gate_mask
            )
        else:
            with torch.no_grad():
                transport, _ = solver.unbalanced_sinkhorn_algorithm(
                    cost_matrix, epsilon=current_epsilon, rho=current_rho,
                    gate_mask=solver._last_gate_mask
                )
            transport = transport.detach()

        eps_actual = solver._last_sinkhorn_epsilon
        rho_actual = solver._last_sinkhorn_rho
        if eps_actual is None or rho_actual is None:
            raise RuntimeError("Sinkhorn did not set eps/rho; transport failed.")

        loss = _compute_full_uot_loss(transport, cost_matrix, a, b, eps_actual, rho_actual)
        loss.backward()

        rot_grad_norm = rot_vec.grad.norm().item() if rot_vec.grad is not None else 0.0
        trans_grad_norm = trans_vec.grad.norm().item() if optimize_translation and trans_vec.grad is not None else 0.0

        optimizer.step()
        if optimize_translation:
            with torch.no_grad():
                trans_vec.div_(trans_vec.norm() + 1e-10)

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_rot_vec = rot_vec.data.clone()
            if optimize_translation:
                best_trans_vec = trans_vec.data.clone()
            best_full_uot_iter = iteration
        with torch.no_grad():
            R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            if r_err < best_r_err:
                best_r_err = r_err
                best_r_rot_vec = rot_vec.data.clone()
                if optimize_translation:
                    best_r_trans_vec = trans_vec.data.clone()
                best_r_iter = iteration
            scores = compute_all_scores(transport, cost_matrix, a, b, eps_actual, rho_actual)
            min_log_kernel = (-cost_matrix / max(eps_actual, 1e-12)).min().item()
            if scores['T_sum'] < collapse_t_sum_threshold:
                underflow = min_log_kernel < -80.0
                uot_zero = abs(scores['full_uot'] - 2.0 * rho_actual) < 1e-3
                if underflow:
                    collapse_underflow += 1
                if uot_zero:
                    collapse_uot_zero += 1
                if underflow and uot_zero:
                    collapse_both += 1
            records.append({
                'iter': iteration,
                'r_err': r_err,
                'full_uot': scores['full_uot'],
                'T_sum': scores['T_sum'],
                'uot_cost_term': scores['uot_cost_term'],
                'uot_kl_term': scores['uot_kl_term'],
                'uot_entropic_term': scores['uot_entropic_term'],
                'min_log_kernel': min_log_kernel,
            })

        if iteration % 10 == 0 or iteration == max_iter - 1:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
                if optimize_translation:
                    t_wc_np = (trans_vec / trans_vec.norm()).numpy()
                else:
                    t_wc_np = (trans_vec / trans_vec.norm()).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            print(f"{iteration:>5} {r_err:>8.2f} {t_err:>8.2f} {loss_val:>12.4f} "
                  f"{transport.sum().item():>10.4f} {rot_grad_norm:>10.3e} {trans_grad_norm:>11.3e}")

    with torch.no_grad():
        R_best = lie.so3_to_SO3(best_rot_vec).numpy()
        if optimize_translation and best_trans_vec is not None:
            t_best = (best_trans_vec / best_trans_vec.norm()).numpy()
        else:
            t_best = (trans_vec / trans_vec.norm()).numpy()

    final_rot_err = rotation_error(R_best, R_gt_wc)
    final_trans_err = min(
        translation_error(t_best, t_gt_wc),
        translation_error(-t_best, t_gt_wc)
    )

    with torch.no_grad():
        R_best_r = lie.so3_to_SO3(best_r_rot_vec).numpy()
        if optimize_translation and best_r_trans_vec is not None:
            t_best_r = (best_r_trans_vec / best_r_trans_vec.norm()).numpy()
        else:
            t_best_r = (trans_vec / trans_vec.norm()).numpy()

    best_r_trans_err = min(
        translation_error(t_best_r, t_gt_wc),
        translation_error(-t_best_r, t_gt_wc)
    )

    print(f"\nFinal result (best by full_uot):")
    print(f"  Rotation error: {final_rot_err:.2f}deg (was {init_rot_err:.2f}deg)")
    print(f"  Translation error: {final_trans_err:.2f}deg (was {init_trans_err:.2f}deg)")
    print(f"Best by R_err (iter {best_r_iter}):")
    print(f"  Rotation error: {best_r_err:.2f}deg (t_err={best_r_trans_err:.2f}deg)")
    if records:
        best_full = records[best_full_uot_iter]
        best_r = records[best_r_iter]
        print("full_uot breakdown (best by full_uot):")
        print(f"  cost={best_full['uot_cost_term']:.4f}, KL={best_full['uot_kl_term']:.4f}, "
              f"entropy={best_full['uot_entropic_term']:.4f}, T.sum={best_full['T_sum']:.4f}")
        print("full_uot breakdown (best by R_err):")
        print(f"  cost={best_r['uot_cost_term']:.4f}, KL={best_r['uot_kl_term']:.4f}, "
              f"entropy={best_r['uot_entropic_term']:.4f}, T.sum={best_r['T_sum']:.4f}")
        print("collapse diagnostics:")
        print(f"  T.sum<{collapse_t_sum_threshold}: underflow={collapse_underflow}, "
              f"uot_zero={collapse_uot_zero}, both={collapse_both}")

    return {
        'init_rot_err': init_rot_err,
        'init_trans_err': init_trans_err,
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'best_full_uot': best_loss,
        'best_r_err': best_r_err,
        'best_r_iter': best_r_iter,
        'best_r_trans_err': best_r_trans_err,
        'best_full_uot_iter': best_full_uot_iter,
        'collapse_underflow': collapse_underflow,
        'collapse_uot_zero': collapse_uot_zero,
        'collapse_both': collapse_both,
    }


# =============================================================================
# Step XXVIII: Multi-start R optimization
# =============================================================================

def random_rotation_matrix(max_angle_deg: float = 180.0) -> np.ndarray:
    """Generate a random rotation matrix with angle uniformly distributed."""
    # Random axis on unit sphere
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    # Random angle
    angle_rad = np.random.uniform(0, np.radians(max_angle_deg))
    return rodrigues_rotation(axis, angle_rad)


def run_step_xxviii_multi_start(
    idx1: int = 0,
    idx2: int = 10,
    n_starts: int = 5,
    max_init_angle_deg: float = 90.0,
    r_converge_iters: int = 80,
    em_iters: int = 20,
    use_hard_assignment: bool = True,
    top_k: int = 50,
    min_transport_mass: float = 0.05,
    epsilon_start: float = None,
    epsilon_end: float = 0.05,
    verbose: bool = True,
):
    """
    Step XXVIII: Multi-start optimization to escape R local minima.

    Strategy:
    1. Generate N random initial rotations
    2. Run Phase 1 for each start (parallel if possible)
    3. Select best based on final avg_cost
    4. Continue with Phase 2 for the best

    Args:
        n_starts: Number of random starting points
        max_init_angle_deg: Maximum rotation angle for random starts
    """
    print("\n" + "=" * 70)
    print(f"Step XXVIII: Multi-start R optimization (n_starts={n_starts})")
    print(f"  Max init angle: {max_init_angle_deg}deg")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    rho = 0.5

    # Generate random starting rotations
    print(f"\nGenerating {n_starts} random starting rotations...")
    start_rotations = []
    for i in range(n_starts):
        R_rand = random_rotation_matrix(max_init_angle_deg)
        R_init_err = rotation_error(R_rand, R_gt_wc)
        start_rotations.append((R_rand, R_init_err))
        print(f"  Start {i+1}: init R_err = {R_init_err:.1f}deg")

    # Also include identity as one starting point (for cases where GT is close to identity)
    R_identity_err = rotation_error(np.eye(3), R_gt_wc)
    start_rotations.append((np.eye(3), R_identity_err))
    print(f"  Start {n_starts+1} (identity): init R_err = {R_identity_err:.1f}deg")

    # Use same initial t for all starts (random perturbation of GT)
    _, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 0.0, 20.0)  # Only perturb t
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    # Auto-determine epsilon from the max cost_median across starts
    cost_medians = []
    with torch.no_grad():
        t_wc_init = torch.tensor(t_init_wc, dtype=torch.float32)
        t_wc_init = t_wc_init / (t_wc_init.norm() + 1e-10)
        for R_init, _ in start_rotations:
            R_wc_init = torch.tensor(R_init, dtype=torch.float32)
            F_init = solver._build_F_from_wc(R_wc_init, t_wc_init)
            cost_init = solver.compute_cost_matrix(F_init)
            cost_medians.append(torch.median(cost_init).item())
    max_cost_median = max(cost_medians)

    if epsilon_start is None:
        eps_auto = max(0.2, max_cost_median / 25.0)
        eps_auto = min(eps_auto, 2.0)
        epsilon_start = eps_auto
        print(f"  Auto ε_start: {epsilon_start:.3f} (max_cost_median={max_cost_median:.2f})")

    # LR schedule (same as Step XXVI-A)
    lr_schedule = [
        (1e-3, 0.9, r_converge_iters // 4),
        (5e-3, 0.95, r_converge_iters // 4),
        (1e-2, 0.95, r_converge_iters // 4),
        (2e-2, 0.95, r_converge_iters // 4),
    ]

    # ====================== Phase 1 for all starts ======================
    print(f"\n--- Phase 1: Running {len(start_rotations)} starts ({r_converge_iters} iters each) ---")

    phase1_results = []

    for start_idx, (R_init, init_r_err) in enumerate(start_rotations):
        rot_vec = torch.nn.Parameter(
            lie.SO3_to_so3(torch.tensor(R_init, dtype=torch.float32)).clone()
        )
        t_wc_np = t_init_wc.copy()

        global_iter = 0
        final_loss = float('inf')
        final_T_sum = 0.0

        for lr, momentum, phase_iters in lr_schedule:
            optimizer = torch.optim.SGD([rot_vec], lr=lr, momentum=momentum, nesterov=True)

            for iteration in range(phase_iters):
                optimizer.zero_grad()

                t = min(global_iter / 50, 1.0)
                current_epsilon = epsilon_start * (1 - t) + epsilon_end * t

                R_wc = lie.so3_to_SO3(rot_vec)
                t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
                t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

                F = solver._build_F_from_wc(R_wc, t_wc_t)
                cost_matrix = solver.compute_cost_matrix(F)

                with torch.no_grad():
                    transport, _ = solver.unbalanced_sinkhorn_algorithm(
                        cost_matrix, epsilon=current_epsilon, rho=rho,
                        gate_mask=solver._last_gate_mask
                    )
                    transport = transport.detach()

                transport_cost = torch.sum(transport * cost_matrix)
                T_sum = transport.sum()
                loss = transport_cost / (T_sum + 1e-10)

                loss.backward()
                optimizer.step()

                final_loss = loss.item()
                final_T_sum = T_sum.item()
                global_iter += 1

        # Evaluate final R error
        with torch.no_grad():
            R_final = lie.so3_to_SO3(rot_vec).numpy()
        final_r_err = rotation_error(R_final, R_gt_wc)

        phase1_results.append({
            'start_idx': start_idx,
            'init_r_err': init_r_err,
            'final_r_err': final_r_err,
            'final_loss': final_loss,
            'final_T_sum': final_T_sum,
            'R_final': R_final.copy(),
            'rot_vec': rot_vec.detach().clone(),
        })

        improvement = init_r_err - final_r_err
        print(f"  Start {start_idx+1}: R {init_r_err:.1f}→{final_r_err:.1f}deg "
              f"(Δ={improvement:+.1f}), loss={final_loss:.4f}, T.sum={final_T_sum:.2f}")

    # Select best start by final loss with collapse guard
    collapse_threshold = 0.5
    valid_results = [r for r in phase1_results if r['final_T_sum'] > collapse_threshold]
    if not valid_results:
        raise RuntimeError("All starts collapsed (final_T_sum <= 0.5).")
    best_result = min(valid_results, key=lambda x: x['final_loss'])
    print(f"\n  Best start: {best_result['start_idx']+1} "
          f"(loss={best_result['final_loss']:.4f}, R_err={best_result['final_r_err']:.1f}deg)")

    # ====================== Phase 2: EM for best start ======================
    print(f"\n--- Phase 2: EM alternating for best start ({em_iters} iters) ---")

    rot_vec = torch.nn.Parameter(best_result['rot_vec'].clone())
    t_wc_np = t_init_wc.copy()

    best_t_wc = t_wc_np.copy()
    best_gap_val = -float('inf')
    prev_T_sum = None
    T_sum_stable_threshold = 0.1

    optimizer = torch.optim.SGD([rot_vec], lr=1e-2, momentum=0.95, nesterov=True)

    for iteration in range(em_iters):
        with torch.no_grad():
            R_wc = lie.so3_to_SO3(rot_vec)
        t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
        t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

        cost_matrix, transport, _ = compute_transport_with_gate(
            solver,
            R_wc,
            t_wc_t,
            epsilon_end,
            rho,
            gate_solvers=gate_solvers,
            gate_top_m_epi=gate_top_m_epi or 0,
            gate_top_m_cov=gate_top_m_cov or 0,
        )

        # R gradient step
        optimizer.zero_grad()
        R_wc = lie.so3_to_SO3(rot_vec)
        F = solver._build_F_from_wc(R_wc, t_wc_t.detach())
        cost_matrix = solver.compute_cost_matrix(F)

        transport_cost = torch.sum(transport.detach() * cost_matrix)
        T_sum = transport.sum()
        loss = transport_cost / (T_sum + 1e-10)

        loss.backward()
        optimizer.step()

        current_T_sum = T_sum.item()
        T_sum_stable = (prev_T_sum is None or
                        abs(current_T_sum - prev_T_sum) < T_sum_stable_threshold)
        prev_T_sum = current_T_sum

        # t closed-form update
        skip_reason = None
        with torch.no_grad():
            R_wc_updated = lie.so3_to_SO3(rot_vec)

            if not T_sum_stable:
                skip_reason = "T.sum unstable"
            else:
                result = compute_closed_form_translation(
                    solver,
                    R_wc_updated,
                    transport,
                    use_hard_assignment=use_hard_assignment,
                    top_k=top_k,
                    min_transport_mass=min_transport_mass,
                )
                t_opt, eigvals = result if result[0] is not None else (None, None)

                if t_opt is None:
                    skip_reason = "collapse"
                else:
                    gap_val = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
                    gap_threshold = 0.001  # Lower threshold to allow more updates

                    if gap_val < gap_threshold:
                        skip_reason = f"gap={gap_val:.4f}"
                    else:
                        if np.dot(t_opt, t_wc_np) < 0:
                            t_opt = -t_opt

                        # Stronger damping for small gaps
                        damping_val = min(1.0, gap_val / 0.02)
                        t_new = damping_val * t_opt + (1 - damping_val) * t_wc_np
                        t_new = t_new / (np.linalg.norm(t_new) + 1e-10)
                        t_wc_np = t_new
                        if gap_val > best_gap_val:
                            best_t_wc = t_wc_np.copy()
                            best_gap_val = gap_val

        if verbose and iteration % 5 == 0:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            status = f" [skip: {skip_reason}]" if skip_reason else ""
            print(f"{iteration:>5} {r_err:>8.2f} {t_err:>8.2f} {loss.item():>12.4f} {T_sum.item():>10.4f}{status}")

    # Final result
    with torch.no_grad():
        R_final = lie.so3_to_SO3(rot_vec).numpy()

    final_rot_err = rotation_error(R_final, R_gt_wc)
    final_trans_err = min(
        translation_error(best_t_wc, t_gt_wc),
        translation_error(-best_t_wc, t_gt_wc)
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Multi-start summary ({n_starts}+1 starts):")
    print(f"  Best initial R_err: {best_result['init_r_err']:.1f}deg")
    print(f"  After Phase 1: R_err = {best_result['final_r_err']:.1f}deg")
    print(f"  After Phase 2: R_err = {final_rot_err:.2f}deg, t_err = {final_trans_err:.2f}deg")

    # Compare with single random start baseline
    avg_init_r_err = np.mean([r['init_r_err'] for r in phase1_results])
    avg_final_r_err = np.mean([r['final_r_err'] for r in phase1_results])
    print(f"\n  Baseline comparison:")
    print(f"    Avg init R_err: {avg_init_r_err:.1f}deg")
    print(f"    Avg Phase1 R_err: {avg_final_r_err:.1f}deg")
    print(f"    Best selected R_err: {best_result['final_r_err']:.1f}deg")
    print(f"    Improvement over avg: {avg_final_r_err - final_rot_err:.1f}deg")

    return {
        'n_starts': n_starts + 1,
        'phase1_results': phase1_results,
        'best_start_idx': best_result['start_idx'],
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'avg_phase1_r_err': avg_final_r_err,
    }


# =============================================================================
# Step XXXII: Weighted 8-point Essential matrix estimation
# =============================================================================

def weighted_eight_point(
    pts1: np.ndarray,  # (N, 2) normalized camera coordinates
    pts2: np.ndarray,  # (N, 2) normalized camera coordinates
    weights: np.ndarray,  # (N,) weights from transport matrix
    min_weight_sum: float = 0.001,  # Very low for MNN which has fewer matches
) -> Optional[np.ndarray]:
    """Weighted 8-point algorithm for Essential matrix estimation.

    Args:
        pts1: Normalized camera coordinates from image 1, shape (N, 2)
        pts2: Normalized camera coordinates from image 2, shape (N, 2)
        weights: Weights for each correspondence (from OT), shape (N,)
        min_weight_sum: Minimum total weight required

    Returns:
        E: Essential matrix (3, 3) or None if degenerate
    """
    if weights.sum() < min_weight_sum:
        return None

    N = len(pts1)
    if N < 8:
        return None

    # Build constraint matrix A
    # For each correspondence: x2^T E x1 = 0
    # Vectorize: [x2_x*x1_x, x2_x*x1_y, x2_x, x2_y*x1_x, x2_y*x1_y, x2_y, x1_x, x1_y, 1] @ e = 0
    x1_x, x1_y = pts1[:, 0], pts1[:, 1]
    x2_x, x2_y = pts2[:, 0], pts2[:, 1]

    A = np.column_stack([
        x2_x * x1_x,
        x2_x * x1_y,
        x2_x,
        x2_y * x1_x,
        x2_y * x1_y,
        x2_y,
        x1_x,
        x1_y,
        np.ones(N),
    ])  # (N, 9)

    # Apply weights (sqrt for least squares)
    sqrt_w = np.sqrt(weights)[:, np.newaxis]
    A_weighted = A * sqrt_w

    # SVD to find minimum eigenvector
    try:
        _, S, Vt = np.linalg.svd(A_weighted)
    except np.linalg.LinAlgError:
        return None

    # E is the last row of Vt (minimum singular value)
    e = Vt[-1]
    E = e.reshape(3, 3)

    # Project onto Essential manifold: E = U @ diag(1, 1, 0) @ Vt
    try:
        U, S_e, Vt_e = np.linalg.svd(E)
    except np.linalg.LinAlgError:
        return None

    # Enforce singular values (1, 1, 0)
    E_projected = U @ np.diag([1, 1, 0]) @ Vt_e

    return E_projected


def decompose_essential_matrix(
    E: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    weights: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decompose Essential matrix into R and t with cheirality check.

    Args:
        E: Essential matrix (3, 3)
        pts1: Normalized camera coordinates from image 1, shape (N, 2)
        pts2: Normalized camera coordinates from image 2, shape (N, 2)
        weights: Weights for cheirality voting

    Returns:
        R: Rotation matrix (3, 3) or None
        t: Translation direction (3,) or None
    """
    # SVD of E
    U, S, Vt = np.linalg.svd(E)

    # Ensure proper rotation (det = 1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    # W matrix for decomposition
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Four possible solutions
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    candidates = [
        (R1, t1),
        (R1, t2),
        (R2, t1),
        (R2, t2),
    ]

    # Cheirality check: count points in front of both cameras
    best_R, best_t = None, None
    best_score = -1

    for R, t in candidates:
        # Check if det(R) = 1 (proper rotation)
        if np.linalg.det(R) < 0:
            R = -R

        score = 0
        for i in range(len(pts1)):
            # Triangulate point (simplified: check z > 0 in both cameras)
            x1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
            x2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])

            # Build linear system for triangulation
            # [x1]_x @ P1 @ X = 0, [x2]_x @ P2 @ X = 0
            # Simplified: check depth sign consistency

            # Depth in camera 1: z1
            # Depth in camera 2: z2 = R @ X + t
            # For epipolar: x2^T @ E @ x1 = 0 should hold

            # Simple cheirality: x2^T @ [t]_x @ R @ x1 should have consistent sign
            # with the triangulated depth

            # Use the constraint: if x1 and R^T @ x2 point to same half-space relative to t
            Rx1 = R @ x1
            # Check if Rx1 and x2 are on same side of epipolar plane
            cross_t_Rx1 = np.cross(t, Rx1)
            if np.dot(cross_t_Rx1, x2) > 0:
                score += weights[i]

        if score > best_score:
            best_score = score
            best_R = R
            best_t = t

    # Normalize t
    if best_t is not None:
        best_t = best_t / (np.linalg.norm(best_t) + 1e-10)

    return best_R, best_t


def compute_essential_from_transport_mnn(
    solver,
    R_current: torch.Tensor,
    transport: torch.Tensor,
    K: np.ndarray,
    min_transport_mass: float = 0.0001,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Compute Essential matrix using Mutual Nearest Neighbor hard assignment.

    Unlike the top-K soft version, this extracts one-to-one correspondences
    using mutual nearest neighbor matching, which is required for valid
    Essential matrix estimation.
    """
    # Get Gaussian means
    means1 = solver.gaussians1.means  # (N1, 2)
    means2 = solver.gaussians2.means  # (N2, 2)

    T_np = transport.detach().cpu().numpy()
    n1, n2 = T_np.shape

    # Mutual nearest neighbor: (i, j) is a match iff
    # j = argmax_j' T[i, j'] AND i = argmax_i' T[i', j]
    row_max_idx = T_np.argmax(axis=1)  # (N1,) best j for each i
    col_max_idx = T_np.argmax(axis=0)  # (N2,) best i for each j

    mnn_pairs = []
    mnn_weights = []
    for i in range(n1):
        j = row_max_idx[i]
        if col_max_idx[j] == i:  # Mutual match
            mass = T_np[i, j]
            if mass > min_transport_mass:
                mnn_pairs.append((i, j))
                mnn_weights.append(mass)

    n_matches = len(mnn_pairs)
    if n_matches < 8:
        return None, None, {
            'reason': 'insufficient_mnn_matches',
            'n_matches': n_matches,
            'threshold': min_transport_mass,
        }

    # Convert to arrays
    i_indices = np.array([p[0] for p in mnn_pairs])
    j_indices = np.array([p[1] for p in mnn_pairs])
    weights = np.array(mnn_weights)

    # Get corresponding 2D points
    pts1_px = means1[i_indices]  # (K, 2)
    pts2_px = means2[j_indices]  # (K, 2)

    # Convert to numpy if needed
    if hasattr(pts1_px, 'numpy'):
        pts1_px = pts1_px.numpy()
        pts2_px = pts2_px.numpy()
    pts1_px = np.array(pts1_px)
    pts2_px = np.array(pts2_px)

    # Normalized camera coordinates
    K_inv = np.linalg.inv(K)
    pts1_hom = np.column_stack([pts1_px, np.ones(len(pts1_px))])
    pts2_hom = np.column_stack([pts2_px, np.ones(len(pts2_px))])
    pts1_norm = (K_inv @ pts1_hom.T).T[:, :2]
    pts2_norm = (K_inv @ pts2_hom.T).T[:, :2]

    # Weighted 8-point
    E = weighted_eight_point(pts1_norm, pts2_norm, weights)
    if E is None:
        return None, None, {
            'reason': 'eight_point_failed',
            'n_points': len(weights),
            'weight_sum': weights.sum(),
        }

    # Decompose E
    R, t = decompose_essential_matrix(E, pts1_norm, pts2_norm, weights)
    if R is None:
        return None, None, {'reason': 'decomposition_failed'}

    return R, t, {
        'n_correspondences': n_matches,
        'total_weight': weights.sum(),
        'E': E,
        'mnn_pairs': mnn_pairs,
    }


def compute_essential_from_transport(
    solver,
    R_current: torch.Tensor,
    transport: torch.Tensor,
    K: np.ndarray,
    top_k: int = 50,
    min_transport_mass: float = 0.0005,
    use_mnn: bool = False,  # If True, use mutual nearest neighbor
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    """Compute Essential matrix from transport correspondences.

    Args:
        solver: OptimalTransportSolver instance
        R_current: Current rotation estimate (for reference)
        transport: Transport matrix from Sinkhorn
        K: Camera intrinsic matrix
        top_k: Number of top correspondences to use
        min_transport_mass: Minimum mass threshold
        use_mnn: If True, use mutual nearest neighbor instead of top-K

    Returns:
        R: Estimated rotation or None
        t: Estimated translation direction or None
        info: Debug information
    """
    # Dispatch to MNN version if requested
    if use_mnn:
        return compute_essential_from_transport_mnn(
            solver, R_current, transport, K, min_transport_mass
        )

    # Get Gaussian means
    means1 = solver.gaussians1.means  # (N1, 2) in (x, y) format
    means2 = solver.gaussians2.means  # (N2, 2)

    # Get top-K correspondences
    T_np = transport.detach().cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]

    # Filter by minimum mass
    top_masses = T_flat[top_indices]
    valid_mask = top_masses > min_transport_mass
    n_valid_before_filter = valid_mask.sum()
    top_indices = top_indices[valid_mask]

    if len(top_indices) < 8:
        return None, None, {
            'reason': 'insufficient_correspondences',
            'n_valid': len(top_indices),
            'n_valid_calc': n_valid_before_filter,
            'threshold': min_transport_mass,
            'top_k_masses': sorted(top_masses.tolist(), reverse=True)[:10],
        }

    # Convert flat indices to (i, j) pairs
    n1, n2 = T_np.shape
    i_indices = top_indices // n2
    j_indices = top_indices % n2
    weights = T_flat[top_indices]

    # Get corresponding 2D points
    pts1_px = means1[i_indices]  # (K, 2) pixel coordinates
    pts2_px = means2[j_indices]  # (K, 2)

    # Convert to normalized camera coordinates: x_norm = K^-1 @ x_px
    K_inv = np.linalg.inv(K)

    pts1_hom = np.column_stack([pts1_px, np.ones(len(pts1_px))])  # (K, 3)
    pts2_hom = np.column_stack([pts2_px, np.ones(len(pts2_px))])  # (K, 3)

    pts1_norm = (K_inv @ pts1_hom.T).T[:, :2]  # (K, 2)
    pts2_norm = (K_inv @ pts2_hom.T).T[:, :2]  # (K, 2)

    # Weighted 8-point
    E = weighted_eight_point(pts1_norm, pts2_norm, weights)
    if E is None:
        return None, None, {
            'reason': 'eight_point_failed',
            'n_points': len(weights),
            'weight_sum': weights.sum(),
        }

    # Decompose E
    R, t = decompose_essential_matrix(E, pts1_norm, pts2_norm, weights)
    if R is None:
        return None, None, {'reason': 'decomposition_failed'}

    return R, t, {
        'n_correspondences': len(top_indices),
        'total_weight': weights.sum(),
        'E': E,
    }


# =============================================================================
# Step XXIX-XXXI: Improved multi-start with non-oracle t selection
# =============================================================================

def compute_transport_concentration(transport: torch.Tensor) -> float:
    """Compute transport concentration: mean(row_max / row_sum).

    Higher value means more peaked/confident correspondences.
    """
    row_sums = transport.sum(dim=1)
    row_maxs = transport.max(dim=1).values
    # Avoid division by zero
    valid_mask = row_sums > 1e-10
    if valid_mask.sum() == 0:
        return 0.0
    concentrations = row_maxs[valid_mask] / row_sums[valid_mask]
    return concentrations.mean().item()


def compute_topk_weight_sum(transport: torch.Tensor, top_k: int) -> float:
    """Compute the sum of global top-K transport weights."""
    T_flat = transport.reshape(-1)
    if top_k >= T_flat.numel():
        return float(T_flat.sum().item())
    top_values, _ = torch.topk(T_flat, top_k)
    return float(top_values.sum().item())


def compute_topk_cost(
    transport: torch.Tensor,
    cost_matrix: torch.Tensor,
    top_k: int,
) -> float:
    """Compute weighted average cost over global top-K transport entries."""
    T_flat = transport.reshape(-1)
    C_flat = cost_matrix.reshape(-1)
    if T_flat.numel() == 0:
        return float("inf")
    if top_k >= T_flat.numel():
        top_values = T_flat
        top_indices = torch.arange(T_flat.numel(), device=T_flat.device)
    else:
        top_values, top_indices = torch.topk(T_flat, top_k)
    weight_sum = top_values.sum().item()
    if weight_sum <= 0.0:
        return float("inf")
    top_costs = C_flat[top_indices]
    return float((top_costs * top_values).sum().item() / (weight_sum + 1e-10))


def build_sequential_gate_mask(
    costs: List[Tuple[torch.Tensor, int]],
) -> torch.Tensor:
    """Build a gate mask by applying top-M filters sequentially per row."""
    if not costs:
        raise ValueError("No costs provided for gate mask.")
    k1, k2 = costs[0][0].shape
    gate = torch.zeros((k1, k2), dtype=torch.bool, device=costs[0][0].device)
    for i in range(k1):
        idx = torch.arange(k2, device=costs[0][0].device)
        for cost, top_m in costs:
            if idx.numel() == 0:
                break
            top_m_i = min(top_m, idx.numel())
            if top_m_i <= 0:
                idx = idx[:0]
                break
            row = cost[i, idx]
            _, sel = torch.topk(row, top_m_i, largest=False)
            idx = idx[sel]
        if idx.numel() > 0:
            gate[i, idx] = True
    return gate


def compute_color_cost_matrix(gaussians1, gaussians2) -> torch.Tensor:
    rgb1 = gaussians1.rgb
    rgb2 = gaussians2.rgb
    if isinstance(rgb1, torch.Tensor):
        rgb1 = rgb1.detach().cpu().numpy()
    if isinstance(rgb2, torch.Tensor):
        rgb2 = rgb2.detach().cpu().numpy()
    diff = rgb1[:, None, :] - rgb2[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    return torch.tensor(cost, dtype=torch.float32)


def compute_knn_descriptor(
    means: np.ndarray,
    scales: Optional[np.ndarray],
    k: int,
) -> np.ndarray:
    n = means.shape[0]
    if n == 0:
        return np.zeros((0, k), dtype=np.float32)
    if n == 1:
        return np.zeros((1, k), dtype=np.float32)
    k_eff = min(k, max(0, n - 1))
    diffs = means[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.partition(dists, k_eff, axis=1)[:, :k_eff]
    nearest.sort(axis=1)
    if k_eff < k:
        pad = np.full((n, k - k_eff), nearest.max(axis=1, keepdims=True), dtype=nearest.dtype)
        nearest = np.concatenate([nearest, pad], axis=1)
    if scales is not None and scales.shape[0] == n:
        scale_ref = np.mean(scales, axis=1, keepdims=True)
        scale_ref = np.clip(scale_ref, 1e-6, None)
        nearest = nearest / scale_ref
    mean_ref = np.mean(nearest, axis=1, keepdims=True)
    mean_ref = np.clip(mean_ref, 1e-6, None)
    return (nearest / mean_ref).astype(np.float32)


def compute_descriptor_cost_matrix(gaussians1, gaussians2, k: int) -> torch.Tensor:
    means1 = gaussians1.means
    means2 = gaussians2.means
    scales1 = gaussians1.scales
    scales2 = gaussians2.scales
    if isinstance(means1, torch.Tensor):
        means1 = means1.detach().cpu().numpy()
    if isinstance(means2, torch.Tensor):
        means2 = means2.detach().cpu().numpy()
    if isinstance(scales1, torch.Tensor):
        scales1 = scales1.detach().cpu().numpy()
    if isinstance(scales2, torch.Tensor):
        scales2 = scales2.detach().cpu().numpy()
    desc1 = compute_knn_descriptor(np.asarray(means1), np.asarray(scales1), k)
    desc2 = compute_knn_descriptor(np.asarray(means2), np.asarray(scales2), k)
    diff = desc1[:, None, :] - desc2[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=2))
    return torch.tensor(cost, dtype=torch.float32)


def compute_gap_metrics(eigvals: np.ndarray) -> Dict[str, float]:
    lam1, lam2, lam3 = float(eigvals[0]), float(eigvals[1]), float(eigvals[2])
    eps = 1e-10
    trace = lam1 + lam2 + lam3 + eps
    return {
        "gap_norm": (lam2 - lam1) / (lam3 + eps),
        "gap_ratio": lam2 / (lam1 + eps),
        "gap_rel": (lam2 - lam1) / (lam2 + eps),
        "gap_trace": lam2 / trace,
    }


def compute_transport_with_gate(
    solver,
    R_wc: torch.Tensor,
    t_wc: torch.Tensor,
    epsilon: float,
    rho: float,
    gate_solvers: Optional[Tuple[OptimalTransportSolver, OptimalTransportSolver]] = None,
    gate_top_m_epi: int = 50,
    gate_top_m_cov: int = 10,
    gate_top_m_color: int = 0,
    gate_top_m_desc: int = 0,
    gate_color_cost: Optional[torch.Tensor] = None,
    gate_desc_cost: Optional[torch.Tensor] = None,
):
    """Compute cost/transport, optionally applying gated candidates."""
    F = solver._build_F_from_wc(R_wc, t_wc)
    cost_matrix = solver.compute_cost_matrix(F)
    gate_mask = None

    gate_steps: List[Tuple[torch.Tensor, int]] = []
    if gate_solvers is not None:
        solver_epi, solver_cov = gate_solvers
        if gate_top_m_epi > 0:
            epi_cost = solver_epi.compute_cost_matrix(F)
            gate_steps.append((epi_cost, gate_top_m_epi))
        if gate_top_m_cov > 0:
            cov_cost = solver_cov.compute_cost_matrix(F)
            gate_steps.append((cov_cost, gate_top_m_cov))
    if gate_top_m_color > 0 and gate_color_cost is not None:
        gate_steps.append((gate_color_cost, gate_top_m_color))
    if gate_top_m_desc > 0 and gate_desc_cost is not None:
        gate_steps.append((gate_desc_cost, gate_top_m_desc))
    if gate_steps:
        gate_mask = build_sequential_gate_mask(gate_steps)
        if solver._last_gate_mask is not None:
            gate_mask = gate_mask & solver._last_gate_mask
        cost_matrix, gate_mask = solver._apply_gate_mask(cost_matrix, gate_mask)

    with torch.no_grad():
        transport, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, epsilon=epsilon, rho=rho,
            gate_mask=gate_mask if gate_mask is not None else solver._last_gate_mask,
        )

    return cost_matrix, transport, gate_mask


def compute_geometry_score(
    solver,
    R_wc: torch.Tensor,
    t_wc: torch.Tensor,
    epsilon: float,
    rho: float,
    top_k: int = 50,
    gate_solvers: Optional[Tuple[OptimalTransportSolver, OptimalTransportSolver]] = None,
    gate_top_m_epi: int = 50,
    gate_top_m_cov: int = 10,
    gate_top_m_color: int = 0,
    gate_top_m_desc: int = 0,
    gate_color_cost: Optional[torch.Tensor] = None,
    gate_desc_cost: Optional[torch.Tensor] = None,
    lambda_kl: float = 0.1,
) -> dict:
    """Compute geometry-based scores (entropy-free) for selection.

    Returns:
        dict with 'avg_cost', 'geom_uot', 'T_sum', 'concentration',
        'topk_cost', 'topk_sum'
    """
    cost_matrix, transport, _ = compute_transport_with_gate(
        solver,
        R_wc,
        t_wc,
        epsilon,
        rho,
        gate_solvers=gate_solvers,
        gate_top_m_epi=gate_top_m_epi,
        gate_top_m_cov=gate_top_m_cov,
        gate_top_m_color=gate_top_m_color,
        gate_top_m_desc=gate_top_m_desc,
        gate_color_cost=gate_color_cost,
        gate_desc_cost=gate_desc_cost,
    )

    T_sum = transport.sum().item()
    if T_sum < 1e-10:
        return {
            'avg_cost': float('inf'),
            'geom_uot': float('inf'),
            'T_sum': 0.0,
            'concentration': 0.0,
            'topk_cost': float('inf'),
            'topk_sum': 0.0,
        }

    eps_safe = 1e-10
    transport_cost = torch.sum(transport * cost_matrix).item()
    avg_cost = transport_cost / T_sum

    # Geometry-only UOT (no entropy): <T,C> + rho * (KL_row + KL_col)
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()
    row_sum = transport.sum(dim=1)
    col_sum = transport.sum(dim=0)
    KL_row = (row_sum * torch.log((row_sum + eps_safe) / (a + eps_safe)) - row_sum + a).sum()
    KL_col = (col_sum * torch.log((col_sum + eps_safe) / (b + eps_safe)) - col_sum + b).sum()
    geom_uot = transport_cost + rho * (KL_row + KL_col).item()
    mass_aware = avg_cost + lambda_kl * (KL_row + KL_col).item()

    concentration = compute_transport_concentration(transport)
    topk_sum = compute_topk_weight_sum(transport, top_k)
    topk_cost = compute_topk_cost(transport, cost_matrix, top_k)

    return {
        'avg_cost': avg_cost,
        'mass_aware': mass_aware,
        'geom_uot': geom_uot,
        'T_sum': T_sum,
        'concentration': concentration,
        'topk_cost': topk_cost,
        'topk_sum': topk_sum,
    }


def run_step_xxix_multi_start(
    idx1: int = 0,
    idx2: int = 10,
    n_starts: int = 5,
    max_init_angle_deg: float = 90.0,
    r_converge_iters: int = 80,
    em_iters: int = 20,
    use_hard_assignment: bool = True,
    top_k: int = 50,
    min_transport_mass: float = 0.05,
    min_topk_weight: float = 0.1,
    epsilon_end: float = 0.05,
    # Step XXX params: t update conditions
    concentration_threshold: float = 0.2,
    gap_threshold: float = 0.01,
    gap_metric: str = "gap_norm",
    gap_damp_scale: Optional[float] = None,
    selection_metric: str = "avg_cost",
    filter_metric: Optional[str] = None,
    filter_threshold: Optional[float] = None,
    t_accept_metric: str = "topk_cost",
    t_accept_margin: float = 1e-4,
    gate_top_m_epi: Optional[int] = None,
    gate_top_m_cov: Optional[int] = None,
    gate_top_m_color: Optional[int] = None,
    gate_top_m_desc: Optional[int] = None,
    desc_k: int = 8,
    top_l: int = 1,
    matching_mode: str = "global_topk",
    row_top_k: Optional[int] = None,
    diversity_grid: Optional[Tuple[int, int]] = None,
    max_per_cell: int = 2,
    lambda_cov: float = 0.0,
    lambda_cov_raw: Optional[float] = None,
    cov_scale_ratio: Optional[float] = None,
    verbose: bool = True,
    print_header: bool = True,
):
    """
    Step XXIX-XXXI: Improved multi-start with:
    1. Per-start ε_start (each start uses its own cost_median)
    2. Common ε_end evaluation for fair start selection
    3. Entropy-free geometry score for selection
    4. Conditional t update (concentration + gap guard)
    5. Best_t selection by non-oracle accept metric
    """
    if print_header:
        print("\n" + "=" * 70)
        print(f"Step XXIX: Improved multi-start (n_starts={n_starts})")
        print(f"  Per-start ε_start, common ε_end={epsilon_end} evaluation")
        print(f"  t update guards: concentration>{concentration_threshold}, gap>{gap_threshold}, "
              f"topk_sum>{min_topk_weight}")
        print(f"  gap metric: {gap_metric}")
        print(f"  selection metric: {selection_metric}")
        if filter_metric is not None and filter_threshold is not None:
            print(f"  filter metric: {filter_metric} (thr={filter_threshold})")
        print(f"  t accept metric: {t_accept_metric} (margin={t_accept_margin})")
        print(f"  top-L for Phase2: {top_l}")
        print(f"  matching mode: {matching_mode}")
        if row_top_k is not None:
            print(f"  row_top_k: {row_top_k}")
        if diversity_grid is not None:
            print(f"  diversity grid: {diversity_grid}, max_per_cell={max_per_cell}")
        if gate_top_m_epi is not None and gate_top_m_cov is not None:
            print(f"  cov gate: epi_top={gate_top_m_epi}, cov_top={gate_top_m_cov}")
        if gate_top_m_color:
            print(f"  color gate: top={gate_top_m_color}")
        if gate_top_m_desc:
            print(f"  desc gate: top={gate_top_m_desc}, k={desc_k}")
        if lambda_cov_raw is not None and cov_scale_ratio is not None:
            print(f"  lambda_cov: raw={lambda_cov_raw:.3f}, scale_ratio={cov_scale_ratio:.6f}, "
                  f"eff={lambda_cov:.6f}")
        else:
            print(f"  lambda_cov: {lambda_cov:.6f}")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=lambda_cov, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    gate_solvers = None
    if gate_top_m_epi is not None and gate_top_m_cov is not None:
        solver_epi = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2,
            k1=K, k2=K, device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
            sigma_epipolar=400.0,
            ot_mass1=ot_mass1, ot_mass2=ot_mass2,
        )
        solver_cov = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2,
            k1=K, k2=K, device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=1.0, lambda_epipolar=0.0,
            sigma_epipolar=400.0,
            ot_mass1=ot_mass1, ot_mass2=ot_mass2,
        )
        gate_solvers = (solver_epi, solver_cov)

    gate_color_cost = None
    if gate_top_m_color:
        gate_color_cost = compute_color_cost_matrix(g1, g2)

    gate_desc_cost = None
    if gate_top_m_desc:
        gate_desc_cost = compute_descriptor_cost_matrix(g1, g2, desc_k)

    lie = Lie()
    rho = 0.5

    seed_env = os.getenv("STEP_XXIX_SEED")
    if seed_env is not None:
        try:
            seed_val = int(seed_env)
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if print_header:
                print(f"  Seed: {seed_val}")
        except ValueError:
            pass

    # Generate random starting rotations
    if print_header:
        print(f"\nGenerating {n_starts} random starting rotations...")
    start_rotations = []
    for i in range(n_starts):
        R_rand = random_rotation_matrix(max_init_angle_deg)
        R_init_err = rotation_error(R_rand, R_gt_wc)
        start_rotations.append((R_rand, R_init_err))

    # Add identity
    R_identity_err = rotation_error(np.eye(3), R_gt_wc)
    start_rotations.append((np.eye(3), R_identity_err))

    # Random initial t
    _, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 0.0, 20.0)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    # ====================== Step XXIX: Per-start ε_start ======================
    if print_header:
        print(f"\nComputing per-start ε_start...")
    start_epsilon_starts = []
    with torch.no_grad():
        t_wc_init = torch.tensor(t_init_wc, dtype=torch.float32)
        t_wc_init = t_wc_init / (t_wc_init.norm() + 1e-10)
        for i, (R_init, init_r_err) in enumerate(start_rotations):
            R_wc_init = torch.tensor(R_init, dtype=torch.float32)
            F_init = solver._build_F_from_wc(R_wc_init, t_wc_init)
            cost_init = solver.compute_cost_matrix(F_init)
            cost_median = torch.median(cost_init).item()
            eps_start = max(0.2, cost_median / 25.0)
            eps_start = min(eps_start, 2.0)
            start_epsilon_starts.append(eps_start)
            start_name = "identity" if i == len(start_rotations) - 1 else f"{i+1}"
            if print_header:
                print(f"  Start {start_name}: R_err={init_r_err:.1f}deg, "
                      f"cost_median={cost_median:.2f}, ε_start={eps_start:.3f}")

    # LR schedule
    lr_schedule = [
        (1e-3, 0.9, r_converge_iters // 4),
        (5e-3, 0.95, r_converge_iters // 4),
        (1e-2, 0.95, r_converge_iters // 4),
        (2e-2, 0.95, r_converge_iters // 4),
    ]

    # ====================== Phase 1 for all starts ======================
    if print_header:
        print(f"\n--- Phase 1: Running {len(start_rotations)} starts ---")

    phase1_results = []

    for start_idx, (R_init, init_r_err) in enumerate(start_rotations):
        eps_start = start_epsilon_starts[start_idx]
        rot_vec = torch.nn.Parameter(
            lie.SO3_to_so3(torch.tensor(R_init, dtype=torch.float32)).clone()
        )
        t_wc_np = t_init_wc.copy()

        global_iter = 0

        for lr, momentum, phase_iters in lr_schedule:
            optimizer = torch.optim.SGD([rot_vec], lr=lr, momentum=momentum, nesterov=True)

            for iteration in range(phase_iters):
                optimizer.zero_grad()

                # Per-start epsilon annealing
                t = min(global_iter / 50, 1.0)
                current_epsilon = eps_start * (1 - t) + epsilon_end * t

                R_wc = lie.so3_to_SO3(rot_vec)
                t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
                t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

                F = solver._build_F_from_wc(R_wc, t_wc_t)
                cost_matrix = solver.compute_cost_matrix(F)

                with torch.no_grad():
                    transport, _ = solver.unbalanced_sinkhorn_algorithm(
                        cost_matrix, epsilon=current_epsilon, rho=rho,
                        gate_mask=solver._last_gate_mask
                    )
                    transport = transport.detach()

                transport_cost = torch.sum(transport * cost_matrix)
                T_sum = transport.sum()
                loss = transport_cost / (T_sum + 1e-10)

                loss.backward()
                optimizer.step()
                global_iter += 1

        # ====================== Common ε_end evaluation ======================
        with torch.no_grad():
            R_final = lie.so3_to_SO3(rot_vec)
            t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
            t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

            # Evaluate at common ε_end for fair comparison
            scores = compute_geometry_score(
                solver,
                R_final,
                t_wc_t,
                epsilon_end,
                rho,
                top_k=top_k,
                gate_solvers=gate_solvers,
                gate_top_m_epi=gate_top_m_epi or 0,
                gate_top_m_cov=gate_top_m_cov or 0,
                gate_top_m_color=gate_top_m_color or 0,
                gate_top_m_desc=gate_top_m_desc or 0,
                gate_color_cost=gate_color_cost,
                gate_desc_cost=gate_desc_cost,
            )

        final_r_err = rotation_error(R_final.numpy(), R_gt_wc)

        phase1_results.append({
            'start_idx': start_idx,
            'init_r_err': init_r_err,
            'final_r_err': final_r_err,
            'eps_start': eps_start,
            'avg_cost_end': scores['avg_cost'],
            'mass_aware_end': scores.get('mass_aware', float('inf')),
            'geom_uot_end': scores['geom_uot'],
            'T_sum_end': scores['T_sum'],
            'concentration': scores['concentration'],
            'topk_sum_end': scores.get('topk_sum', 0.0),
            'topk_cost_end': scores.get('topk_cost', float('inf')),
            'R_final': R_final.numpy().copy(),
            'rot_vec': rot_vec.detach().clone(),
        })

        improvement = init_r_err - final_r_err
        start_name = "identity" if start_idx == len(start_rotations) - 1 else f"{start_idx+1}"
        if print_header:
            print(f"  Start {start_name}: R {init_r_err:.1f}→{final_r_err:.1f}deg (Δ={improvement:+.1f}), "
                  f"avg_cost@ε_end={scores['avg_cost']:.4f}, geom@ε_end={scores['geom_uot']:.4f}, "
                  f"T.sum={scores['T_sum']:.2f}, "
                  f"conc={scores['concentration']:.3f}")

    # ====================== Select top-L by metric@ε_end (with T.sum guard) ======================
    valid_results = [r for r in phase1_results if r['T_sum_end'] > 0.5]
    if not valid_results:
        print("WARNING: All starts collapsed! Using best of collapsed.")
        valid_results = phase1_results

    higher_is_better = {"T_sum", "concentration", "topk_sum"}
    def _metric_key(name: str) -> str:
        if name == "geom_uot":
            return "geom_uot_end"
        if name == "mass_aware":
            return "mass_aware_end"
        if name == "T_sum":
            return "T_sum_end"
        if name == "concentration":
            return "concentration"
        if name == "topk_sum":
            return "topk_sum_end"
        if name == "topk_cost":
            return "topk_cost_end"
        return "avg_cost_end"

    filtered_results = valid_results
    if filter_metric is not None and filter_threshold is not None:
        filter_key = _metric_key(filter_metric)
        is_high = filter_metric in higher_is_better
        if is_high:
            filtered_results = [r for r in valid_results if r.get(filter_key, -float("inf")) >= filter_threshold]
        else:
            filtered_results = [r for r in valid_results if r.get(filter_key, float("inf")) <= filter_threshold]
        if not filtered_results:
            print(f"WARNING: filter '{filter_metric}' removed all starts; skipping filter.")
            filtered_results = valid_results
    metric_key = _metric_key(selection_metric)

    reverse = selection_metric in higher_is_better
    sorted_results = sorted(filtered_results, key=lambda x: x.get(metric_key, float("inf")), reverse=reverse)
    top_l = max(1, min(top_l, len(sorted_results)))
    top_results = sorted_results[:top_l]
    best_result = top_results[0]
    best_metric_val = best_result.get(metric_key, float("inf"))

    if print_header:
        print(f"\n  Top-{top_l} starts by {selection_metric}@ε_end:")
        for rank, r in enumerate(top_results, 1):
            metric_val = r.get(metric_key, float("inf"))
            print(f"    {rank}) start {r['start_idx']+1}: {selection_metric}={metric_val:.4f}, "
                  f"R_err={r['final_r_err']:.1f}deg")

    # ====================== Phase 2: EM with conditional t update (Step XXX-XXXI) ======================
    if print_header:
        print(f"\n--- Phase 2: EM with conditional t update ({em_iters} iters) ---")
        print(f"  t update conditions: concentration>{concentration_threshold}, gap>{gap_threshold}")

    phase2_results = []

    def _run_phase2_for_start(candidate, label: str):
        rot_vec = torch.nn.Parameter(candidate['rot_vec'].clone())
        t_wc_np = t_init_wc.copy()

        best_t_wc = t_wc_np.copy()
        higher_is_better = {"topk_sum", "T_sum", "concentration"}
        init_scores = compute_geometry_score(
            solver,
            lie.so3_to_SO3(rot_vec).detach(),
            torch.tensor(t_wc_np, dtype=torch.float32),
            epsilon_end,
            rho,
            top_k=top_k,
            gate_solvers=gate_solvers,
            gate_top_m_epi=gate_top_m_epi or 0,
            gate_top_m_cov=gate_top_m_cov or 0,
            gate_top_m_color=gate_top_m_color or 0,
            gate_top_m_desc=gate_top_m_desc or 0,
            gate_color_cost=gate_color_cost,
            gate_desc_cost=gate_desc_cost,
        )
        if t_accept_metric not in init_scores:
            raise ValueError(f"Unknown t_accept_metric: {t_accept_metric}")
        best_t_score = init_scores[t_accept_metric]
        t_update_count = 0

        optimizer = torch.optim.SGD([rot_vec], lr=1e-2, momentum=0.95, nesterov=True)

        for iteration in range(em_iters):
            with torch.no_grad():
                R_wc = lie.so3_to_SO3(rot_vec)
            t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
            t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

            cost_matrix, transport, _ = compute_transport_with_gate(
                solver,
                R_wc,
                t_wc_t,
                epsilon_end,
                rho,
                gate_solvers=gate_solvers,
                gate_top_m_epi=gate_top_m_epi or 0,
                gate_top_m_cov=gate_top_m_cov or 0,
                gate_top_m_color=gate_top_m_color or 0,
                gate_top_m_desc=gate_top_m_desc or 0,
                gate_color_cost=gate_color_cost,
                gate_desc_cost=gate_desc_cost,
            )

            # R gradient step
            optimizer.zero_grad()
            R_wc = lie.so3_to_SO3(rot_vec)
            F = solver._build_F_from_wc(R_wc, t_wc_t.detach())
            cost_matrix = solver.compute_cost_matrix(F)

            transport_cost = torch.sum(transport.detach() * cost_matrix)
            T_sum = transport.sum()
            loss = transport_cost / (T_sum + 1e-10)

            loss.backward()
            optimizer.step()

            # ====================== Step XXX: Conditional t update ======================
            skip_reason = None
            t_updated = False

            with torch.no_grad():
                R_wc_updated = lie.so3_to_SO3(rot_vec)

                concentration = compute_transport_concentration(transport)
                topk_sum = compute_topk_weight_sum(transport, top_k)

                if concentration < concentration_threshold:
                    skip_reason = f"conc={concentration:.3f}<{concentration_threshold}"
                elif topk_sum < min_topk_weight:
                    skip_reason = f"topk={topk_sum:.4f}<{min_topk_weight}"
                else:
                    result = compute_closed_form_translation(
                        solver,
                        R_wc_updated,
                        transport,
                        use_hard_assignment=use_hard_assignment,
                        top_k=top_k,
                        min_transport_mass=min_transport_mass,
                        min_topk_weight=min_topk_weight,
                        matching_mode=matching_mode,
                        row_top_k=row_top_k,
                        diversity_grid=diversity_grid,
                        max_per_cell=max_per_cell,
                    )
                    t_opt, eigvals = result if result[0] is not None else (None, None)

                    if t_opt is None:
                        skip_reason = "collapse"
                    else:
                        gap_metrics = compute_gap_metrics(eigvals)
                        gap_val = gap_metrics.get(gap_metric, gap_metrics["gap_norm"])
                        if gap_val < gap_threshold:
                            skip_reason = f"{gap_metric}={gap_val:.4f}<{gap_threshold}"
                        else:
                            if np.dot(t_opt, t_wc_np) < 0:
                                t_opt = -t_opt

                            damp_scale = gap_damp_scale if gap_damp_scale is not None else 0.02
                            damping_val = min(1.0, gap_val / damp_scale)
                            t_new = damping_val * t_opt + (1 - damping_val) * t_wc_np
                            t_new = t_new / (np.linalg.norm(t_new) + 1e-10)

                            t_new_t = torch.tensor(t_new, dtype=torch.float32)
                            t_new_t = t_new_t / (t_new_t.norm() + 1e-10)
                            new_scores = compute_geometry_score(
                                solver,
                                R_wc_updated,
                                t_new_t,
                                epsilon_end,
                                rho,
                                top_k=top_k,
                                gate_solvers=gate_solvers,
                                gate_top_m_epi=gate_top_m_epi or 0,
                                gate_top_m_cov=gate_top_m_cov or 0,
                                gate_top_m_color=gate_top_m_color or 0,
                                gate_top_m_desc=gate_top_m_desc or 0,
                                gate_color_cost=gate_color_cost,
                                gate_desc_cost=gate_desc_cost,
                            )

                            candidate_score = new_scores[t_accept_metric]
                            if t_accept_metric in higher_is_better:
                                is_better = candidate_score > best_t_score + t_accept_margin
                            else:
                                is_better = candidate_score < best_t_score - t_accept_margin

                            if is_better:
                                t_wc_np = t_new
                                best_t_wc = t_wc_np.copy()
                                best_t_score = candidate_score
                                t_updated = True
                                t_update_count += 1
                            else:
                                skip_reason = f"{t_accept_metric} {candidate_score:.4f} no-improve"

            if verbose and iteration % 5 == 0:
                with torch.no_grad():
                    R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
                r_err = rotation_error(R_wc_np, R_gt_wc)
                t_err = min(
                    translation_error(t_wc_np, t_gt_wc),
                    translation_error(-t_wc_np, t_gt_wc)
                )
                status = f" [skip: {skip_reason}]" if skip_reason else " [t updated]"
                print(f"{label} {iteration:>3} R={r_err:>6.2f} t={t_err:>6.2f} "
                      f"loss={loss.item():.4f} T.sum={T_sum.item():.2f} "
                      f"conc={concentration:.3f} topk={topk_sum:.4f}{status}")

        with torch.no_grad():
            R_final = lie.so3_to_SO3(rot_vec).numpy()

        final_rot_err = rotation_error(R_final, R_gt_wc)
        final_trans_err = min(
            translation_error(best_t_wc, t_gt_wc),
            translation_error(-best_t_wc, t_gt_wc)
        )

        final_scores = compute_geometry_score(
            solver,
            torch.tensor(R_final, dtype=torch.float32),
            torch.tensor(best_t_wc, dtype=torch.float32),
            epsilon_end,
            rho,
            top_k=top_k,
            gate_solvers=gate_solvers,
            gate_top_m_epi=gate_top_m_epi or 0,
            gate_top_m_cov=gate_top_m_cov or 0,
            gate_top_m_color=gate_top_m_color or 0,
            gate_top_m_desc=gate_top_m_desc or 0,
            gate_color_cost=gate_color_cost,
            gate_desc_cost=gate_desc_cost,
        )

        return {
            'start_idx': candidate['start_idx'],
            'final_rot_err': final_rot_err,
            'final_trans_err': final_trans_err,
            'best_t_score': best_t_score,
            't_update_count': t_update_count,
            'scores': final_scores,
            'R_final': R_final,
            't_final': best_t_wc,
        }

    for rank, candidate in enumerate(top_results, 1):
        label = f"[{rank}/{top_l}]"
        if print_header:
            print(f"\n  Phase2 start {rank}/{top_l}: start {candidate['start_idx']+1}")
        phase2_results.append(_run_phase2_for_start(candidate, label))

    # Select best from Phase2 results
    def _phase2_metric(res):
        if selection_metric == "geom_uot":
            return res['scores']['geom_uot']
        if selection_metric == "mass_aware":
            return res['scores']['mass_aware']
        if selection_metric == "T_sum":
            return res['scores']['T_sum']
        if selection_metric == "concentration":
            return res['scores']['concentration']
        if selection_metric == "topk_sum":
            return res['scores']['topk_sum']
        if selection_metric == "topk_cost":
            return res['scores']['topk_cost']
        return res['scores']['avg_cost']

    reverse = selection_metric in higher_is_better
    best_phase2 = sorted(phase2_results, key=_phase2_metric, reverse=reverse)[0]

    final_rot_err = best_phase2['final_rot_err']
    final_trans_err = best_phase2['final_trans_err']
    scores = best_phase2['scores']
    best_t_score = best_phase2['best_t_score']
    t_update_count = best_phase2['t_update_count']

    if print_header:
        print(f"\n{'='*60}")
        print(f"Step XXIX-XXXI summary ({n_starts}+1 starts):")
        print(f"  Best Phase1 R_err: {best_result['final_r_err']:.1f}deg")
        print(f"  Final R_err: {final_rot_err:.2f}deg")
        print(f"  Final t_err: {final_trans_err:.2f}deg (non-oracle selection)")
        print(f"  best_t {t_accept_metric}: {best_t_score:.4f}")
        print(f"  avg_cost@ε_end: {scores['avg_cost']:.4f}, conc: {scores['concentration']:.3f}, T_sum: {scores['T_sum']:.3f}")
        print(f"  t updates accepted: {t_update_count}/{em_iters}")

    avg_final_r_err = np.mean([r['final_r_err'] for r in phase1_results])
    if print_header:
        print(f"\n  Avg Phase1 R_err: {avg_final_r_err:.1f}deg")
        print(f"  Improvement over avg: {avg_final_r_err - final_rot_err:.1f}deg")

    return {
        'n_starts': n_starts + 1,
        'phase1_results': phase1_results,
        'top_phase1_start_indices': [r['start_idx'] for r in top_results],
        'best_start_idx': best_result['start_idx'],
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'best_t_score': best_t_score,
        't_update_count': t_update_count,
        'final_avg_cost': scores['avg_cost'],
        'final_concentration': scores['concentration'],
        'final_T_sum': scores['T_sum'],
        'R_final': best_phase2['R_final'],
        't_final': best_phase2['t_final'],
        'phase2_results': phase2_results,
    }


# =============================================================================
# Step XXXII: Test Essential matrix estimation
# =============================================================================

def verify_essential_at_gt_pose(
    idx1: int = 0,
    idx2: int = 10,
    top_k: int = 50,
    epsilon: float = 0.05,
    verbose: bool = True,
):
    """Verify Essential matrix estimation at GT pose.

    This tests whether the 8-point algorithm works correctly when
    the transport matrix is computed at the ground truth pose.
    """
    print("\n" + "=" * 70)
    print("Verifying Essential matrix estimation at GT pose")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    print(f"GT R (axis-angle): {Lie().SO3_to_so3(torch.tensor(R_gt_wc)).numpy()}")
    print(f"GT t: {t_gt_wc}")

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Compute transport at GT pose
    R_gt_t = torch.tensor(R_gt_wc, dtype=torch.float32)
    t_gt_t = torch.tensor(t_gt_wc, dtype=torch.float32)
    t_gt_t = t_gt_t / (t_gt_t.norm() + 1e-10)

    F = solver._build_F_from_wc(R_gt_t, t_gt_t)
    cost_matrix = solver.compute_cost_matrix(F)

    with torch.no_grad():
        transport, _ = solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, epsilon=epsilon, rho=0.5,
            gate_mask=solver._last_gate_mask
        )

    print(f"\nTransport stats at GT:")
    print(f"  T.sum = {transport.sum().item():.4f}")
    print(f"  T.max = {transport.max().item():.6f}")
    print(f"  Concentration = {compute_transport_concentration(transport):.4f}")

    # Estimate E from transport - Method 1: Top-K soft correspondences
    print("\n--- Method 1: Top-K soft correspondences ---")
    R_est, t_est, info = compute_essential_from_transport(
        solver, R_gt_t, transport, K,
        top_k=top_k, min_transport_mass=0.0001, use_mnn=False,
    )

    if R_est is not None:
        R_est_err = rotation_error(R_est, R_gt_wc)
        t_est_err = min(
            translation_error(t_est, t_gt_wc),
            translation_error(-t_est, t_gt_wc)
        )
        print(f"  R_err = {R_est_err:.2f}deg")
        print(f"  t_err = {t_est_err:.2f}deg")
        print(f"  n_correspondences = {info.get('n_correspondences', 0)}")
        print(f"  total_weight = {info.get('total_weight', 0):.4f}")
    else:
        print(f"  FAILED: {info}")

    # Estimate E from transport - Method 2: Mutual Nearest Neighbor
    print("\n--- Method 2: Mutual Nearest Neighbor (MNN) ---")
    R_est_mnn, t_est_mnn, info_mnn = compute_essential_from_transport(
        solver, R_gt_t, transport, K,
        min_transport_mass=0.0001, use_mnn=True,
    )

    if R_est_mnn is not None:
        R_est_err_mnn = rotation_error(R_est_mnn, R_gt_wc)
        t_est_err_mnn = min(
            translation_error(t_est_mnn, t_gt_wc),
            translation_error(-t_est_mnn, t_gt_wc)
        )
        print(f"  R_err = {R_est_err_mnn:.2f}deg")
        print(f"  t_err = {t_est_err_mnn:.2f}deg")
        print(f"  n_correspondences = {info_mnn.get('n_correspondences', 0)}")
        print(f"  total_weight = {info_mnn.get('total_weight', 0):.4f}")

        # Use MNN result for remaining analysis
        R_est = R_est_mnn
        t_est = t_est_mnn
        info = info_mnn
    else:
        print(f"  FAILED: {info_mnn}")

    if R_est is not None:
        R_est_err = rotation_error(R_est, R_gt_wc)
        t_est_err = min(
            translation_error(t_est, t_gt_wc),
            translation_error(-t_est, t_gt_wc)
        )
        print(f"\n--- Best Essential matrix estimation at GT pose ---")
        print(f"  R_err = {R_est_err:.2f}deg")
        print(f"  t_err = {t_est_err:.2f}deg")
        print(f"  n_correspondences = {info.get('n_correspondences', 0)}")
        print(f"  total_weight = {info.get('total_weight', 0):.4f}")

        # Check if E is close to GT E
        # E = [t]_x @ R where [t]_x is the skew-symmetric matrix
        t_norm = t_gt_wc / (np.linalg.norm(t_gt_wc) + 1e-10)
        t_skew = np.array([
            [0, -t_norm[2], t_norm[1]],
            [t_norm[2], 0, -t_norm[0]],
            [-t_norm[1], t_norm[0], 0]
        ])
        E_gt = t_skew @ R_gt_wc
        E_gt = E_gt / np.linalg.norm(E_gt, 'fro')  # Normalize

        E_est = info.get('E', None)
        if E_est is not None:
            E_est_norm = E_est / np.linalg.norm(E_est, 'fro')
            # E is defined up to scale, check both signs
            err1 = np.linalg.norm(E_est_norm - E_gt, 'fro')
            err2 = np.linalg.norm(E_est_norm + E_gt, 'fro')
            E_err = min(err1, err2)
            print(f"  E Frobenius error = {E_err:.4f} (0 = perfect)")

            # Debug: print E matrices
            if verbose:
                print(f"\n  GT E (normalized):\n{E_gt}")
                print(f"\n  Est E (normalized):\n{E_est_norm}")

        # Debug: Check top correspondences quality
        if verbose:
            print(f"\n  Checking top-10 correspondences epipolar residuals:")
            means1_np = solver.gaussians1.means
            means2_np = solver.gaussians2.means
            # Convert to numpy if tensor
            if hasattr(means1_np, 'numpy'):
                means1_np = means1_np.numpy()
            if hasattr(means2_np, 'numpy'):
                means2_np = means2_np.numpy()
            means1_np = np.array(means1_np)
            means2_np = np.array(means2_np)

            T_np = transport.detach().cpu().numpy()
            T_flat = T_np.flatten()
            top_indices = np.argsort(T_flat)[-10:][::-1]
            n1, n2 = T_np.shape

            K_inv = np.linalg.inv(K)
            residuals = []
            for idx in top_indices:
                i = idx // n2
                j = idx % n2
                mass = T_flat[idx]
                pt1_px = means1_np[i]
                pt2_px = means2_np[j]

                # Normalized coordinates
                pt1_norm = K_inv @ np.array([pt1_px[0], pt1_px[1], 1])
                pt2_norm = K_inv @ np.array([pt2_px[0], pt2_px[1], 1])

                # Epipolar residual: |x2^T E x1| (at GT E)
                residual_gt = abs(pt2_norm @ E_gt @ pt1_norm)
                residuals.append(residual_gt)
                print(f"    Corr ({i},{j}): mass={mass:.6f}, res@GT_E={residual_gt:.6f}")

            avg_res = np.mean(residuals)
            print(f"\n  Average epipolar residual at GT E: {avg_res:.6f}")
            print(f"  NOTE: For valid correspondences, residual should be ~0")

    else:
        print(f"\nEssential matrix estimation FAILED: {info}")

    # Also test with OpenCV's findEssentialMat (uses RANSAC internally)
    print("\n--- Method 3: OpenCV findEssentialMat (RANSAC) ---")
    try:
        import cv2
        # Get top-K correspondences in pixel coordinates
        means1_np = solver.gaussians1.means
        means2_np = solver.gaussians2.means
        if hasattr(means1_np, 'numpy'):
            means1_np = means1_np.numpy()
        if hasattr(means2_np, 'numpy'):
            means2_np = means2_np.numpy()
        means1_np = np.array(means1_np)
        means2_np = np.array(means2_np)

        T_np = transport.detach().cpu().numpy()
        T_flat = T_np.flatten()
        top_indices = np.argsort(T_flat)[-100:]  # Use more points for RANSAC
        n1, n2 = T_np.shape

        i_indices = top_indices // n2
        j_indices = top_indices % n2

        pts1_px = means1_np[i_indices].astype(np.float64)
        pts2_px = means2_np[j_indices].astype(np.float64)

        # OpenCV's findEssentialMat expects points and camera matrix
        E_cv, mask = cv2.findEssentialMat(
            pts1_px, pts2_px, K.astype(np.float64),
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        if E_cv is not None and mask is not None:
            n_inliers = mask.sum()
            print(f"  OpenCV found E with {n_inliers} inliers out of {len(mask)}")

            # Recover pose from E
            _, R_cv, t_cv, mask_pose = cv2.recoverPose(E_cv, pts1_px, pts2_px, K.astype(np.float64))

            R_cv_err = rotation_error(R_cv, R_gt_wc)
            t_cv_flat = t_cv.flatten()
            t_cv_err = min(
                translation_error(t_cv_flat, t_gt_wc),
                translation_error(-t_cv_flat, t_gt_wc)
            )
            print(f"  R_err = {R_cv_err:.2f}deg")
            print(f"  t_err = {t_cv_err:.2f}deg")

            # Compare E matrices
            E_cv_norm = E_cv / np.linalg.norm(E_cv, 'fro')
            err1 = np.linalg.norm(E_cv_norm - E_gt, 'fro')
            err2 = np.linalg.norm(E_cv_norm + E_gt, 'fro')
            E_cv_err = min(err1, err2)
            print(f"  E Frobenius error = {E_cv_err:.4f}")
        else:
            print(f"  OpenCV findEssentialMat failed")
    except Exception as e:
        print(f"  OpenCV test failed: {e}")

    return R_est, t_est, info


def run_step_xxxii_essential_matrix(
    idx1: int = 0,
    idx2: int = 10,
    n_starts: int = 5,
    max_init_angle_deg: float = 90.0,
    r_converge_iters: int = 80,
    em_iters: int = 30,
    top_k: int = 50,
    min_transport_mass: float = 0.0005,  # Lowered: transport values ~0.001-0.002
    epsilon_end: float = 0.05,
    use_essential_update: bool = True,
    essential_interval: int = 5,
    verbose: bool = True,
):
    """
    Step XXXII: Multi-start with Essential matrix M-step.

    Key change: Every `essential_interval` iterations in Phase 2,
    compute Essential matrix from correspondences and update R.
    """
    print("\n" + "=" * 70)
    print(f"Step XXXII: Essential matrix M-step (n_starts={n_starts})")
    print(f"  Essential update: every {essential_interval} iters in Phase 2")
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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    lie = Lie()
    rho = 0.5

    # Generate random starting rotations
    print(f"\nGenerating {n_starts} random starting rotations...")
    start_rotations = []
    for i in range(n_starts):
        R_rand = random_rotation_matrix(max_init_angle_deg)
        R_init_err = rotation_error(R_rand, R_gt_wc)
        start_rotations.append((R_rand, R_init_err))

    # Add identity
    R_identity_err = rotation_error(np.eye(3), R_gt_wc)
    start_rotations.append((np.eye(3), R_identity_err))

    # Random initial t
    _, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 0.0, 20.0)

    # Per-start epsilon
    print(f"\nComputing per-start ε_start...")
    start_epsilon_starts = []
    with torch.no_grad():
        t_wc_init = torch.tensor(t_init_wc, dtype=torch.float32)
        t_wc_init = t_wc_init / (t_wc_init.norm() + 1e-10)
        for i, (R_init, init_r_err) in enumerate(start_rotations):
            R_wc_init = torch.tensor(R_init, dtype=torch.float32)
            F_init = solver._build_F_from_wc(R_wc_init, t_wc_init)
            cost_init = solver.compute_cost_matrix(F_init)
            cost_median = torch.median(cost_init).item()
            eps_start = max(0.2, cost_median / 25.0)
            eps_start = min(eps_start, 2.0)
            start_epsilon_starts.append(eps_start)

    # LR schedule
    lr_schedule = [
        (1e-3, 0.9, r_converge_iters // 4),
        (5e-3, 0.95, r_converge_iters // 4),
        (1e-2, 0.95, r_converge_iters // 4),
        (2e-2, 0.95, r_converge_iters // 4),
    ]

    # Phase 1 for all starts
    print(f"\n--- Phase 1: Running {len(start_rotations)} starts ---")
    phase1_results = []

    for start_idx, (R_init, init_r_err) in enumerate(start_rotations):
        eps_start = start_epsilon_starts[start_idx]
        rot_vec = torch.nn.Parameter(
            lie.SO3_to_so3(torch.tensor(R_init, dtype=torch.float32)).clone()
        )
        t_wc_np = t_init_wc.copy()
        global_iter = 0

        for lr, momentum, phase_iters in lr_schedule:
            optimizer = torch.optim.SGD([rot_vec], lr=lr, momentum=momentum, nesterov=True)
            for iteration in range(phase_iters):
                optimizer.zero_grad()
                t = min(global_iter / 50, 1.0)
                current_epsilon = eps_start * (1 - t) + epsilon_end * t

                R_wc = lie.so3_to_SO3(rot_vec)
                t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
                t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

                F = solver._build_F_from_wc(R_wc, t_wc_t)
                cost_matrix = solver.compute_cost_matrix(F)

                with torch.no_grad():
                    transport, _ = solver.unbalanced_sinkhorn_algorithm(
                        cost_matrix, epsilon=current_epsilon, rho=rho,
                        gate_mask=solver._last_gate_mask
                    )
                    transport = transport.detach()

                transport_cost = torch.sum(transport * cost_matrix)
                T_sum = transport.sum()
                loss = transport_cost / (T_sum + 1e-10)

                loss.backward()
                optimizer.step()
                global_iter += 1

        with torch.no_grad():
            R_final = lie.so3_to_SO3(rot_vec)
            t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
            t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)
            scores = compute_geometry_score(solver, R_final, t_wc_t, epsilon_end, rho)

        final_r_err = rotation_error(R_final.numpy(), R_gt_wc)
        phase1_results.append({
            'start_idx': start_idx,
            'init_r_err': init_r_err,
            'final_r_err': final_r_err,
            'avg_cost_end': scores['avg_cost'],
            'T_sum_end': scores['T_sum'],
            'rot_vec': rot_vec.detach().clone(),
        })

        start_name = "identity" if start_idx == len(start_rotations) - 1 else f"{start_idx+1}"
        print(f"  Start {start_name}: R {init_r_err:.1f}→{final_r_err:.1f}deg, "
              f"avg_cost@ε_end={scores['avg_cost']:.4f}")

    # Select best
    valid_results = [r for r in phase1_results if r['T_sum_end'] > 0.5]
    if not valid_results:
        valid_results = phase1_results
    best_result = min(valid_results, key=lambda x: x['avg_cost_end'])
    print(f"\n  Best start: {best_result['start_idx']+1} (R_err={best_result['final_r_err']:.1f}deg)")

    # Phase 2: EM with Essential matrix M-step
    print(f"\n--- Phase 2: EM with Essential matrix M-step ({em_iters} iters) ---")

    rot_vec = torch.nn.Parameter(best_result['rot_vec'].clone())
    t_wc_np = t_init_wc.copy()
    best_t_wc = t_wc_np.copy()
    best_score = float('inf')

    optimizer = torch.optim.SGD([rot_vec], lr=1e-2, momentum=0.95, nesterov=True)

    for iteration in range(em_iters):
        with torch.no_grad():
            R_wc = lie.so3_to_SO3(rot_vec)
        t_wc_t = torch.tensor(t_wc_np, dtype=torch.float32)
        t_wc_t = t_wc_t / (t_wc_t.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc_t)
        cost_matrix = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix, epsilon=epsilon_end, rho=rho,
                gate_mask=solver._last_gate_mask
            )

        # Standard R gradient step
        optimizer.zero_grad()
        R_wc = lie.so3_to_SO3(rot_vec)
        F = solver._build_F_from_wc(R_wc, t_wc_t.detach())
        cost_matrix = solver.compute_cost_matrix(F)

        transport_cost = torch.sum(transport.detach() * cost_matrix)
        T_sum = transport.sum()
        loss = transport_cost / (T_sum + 1e-10)

        loss.backward()
        optimizer.step()

        # Essential matrix M-step
        essential_applied = False
        if use_essential_update and iteration % essential_interval == 0 and iteration > 0:
            with torch.no_grad():
                R_current = lie.so3_to_SO3(rot_vec)
                R_est, t_est, info = compute_essential_from_transport(
                    solver, R_current, transport, K,
                    top_k=top_k, min_transport_mass=min_transport_mass
                )

                if R_est is not None:
                    # Evaluate Essential estimate
                    R_est_t = torch.tensor(R_est, dtype=torch.float32)
                    t_est_t = torch.tensor(t_est, dtype=torch.float32)
                    t_est_t = t_est_t / (t_est_t.norm() + 1e-10)

                    est_scores = compute_geometry_score(
                        solver, R_est_t, t_est_t, epsilon_end, rho
                    )

                    # Debug: Check Essential estimate quality
                    R_est_err = rotation_error(R_est, R_gt_wc)
                    t_est_err = min(
                        translation_error(t_est, t_gt_wc),
                        translation_error(-t_est, t_gt_wc)
                    )
                    if verbose:
                        n_corr = info.get('n_correspondences', 0)
                        w_sum = info.get('total_weight', 0)
                        print(f"      [E] R_err={R_est_err:.1f}, t_err={t_est_err:.1f}, "
                              f"est_cost={est_scores['avg_cost']:.4f}, cur_loss={loss.item():.4f}, "
                              f"n_corr={n_corr}, w_sum={w_sum:.4f}")

                    # Accept if improves geometry score
                    if est_scores['avg_cost'] < loss.item() * 0.95:  # 5% improvement threshold
                        # Update rot_vec to Essential estimate
                        new_rot_vec = lie.SO3_to_so3(R_est_t)
                        rot_vec.data.copy_(new_rot_vec)

                        # Update t if Essential gives better t
                        if est_scores['avg_cost'] < best_score:
                            t_wc_np = t_est.copy()
                            best_t_wc = t_wc_np.copy()
                            best_score = est_scores['avg_cost']

                        essential_applied = True
                else:
                    if verbose:
                        reason = info.get('reason', 'unknown')
                        n_pts = info.get('n_points', info.get('n_valid', 0))
                        w_sum = info.get('weight_sum', info.get('n_valid_calc', 'N/A'))
                        if reason == 'insufficient_correspondences':
                            top_masses = info.get('top_k_masses', [])[:5]
                            top_str = ", ".join([f"{m:.5f}" for m in top_masses])
                            print(f"      [E] failed: {reason} (n_pts={n_pts}, top: {top_str})")
                        else:
                            print(f"      [E] failed: {reason} (n_pts={n_pts}, w_sum={w_sum:.4f})" if isinstance(w_sum, float) else f"      [E] failed: {reason}")

        if verbose and iteration % 5 == 0:
            with torch.no_grad():
                R_wc_np = lie.so3_to_SO3(rot_vec).numpy()
            r_err = rotation_error(R_wc_np, R_gt_wc)
            t_err = min(
                translation_error(t_wc_np, t_gt_wc),
                translation_error(-t_wc_np, t_gt_wc)
            )
            status = " [E applied]" if essential_applied else ""
            print(f"{iteration:>5} R={r_err:>6.2f} t={t_err:>6.2f} "
                  f"loss={loss.item():.4f} T.sum={T_sum.item():.2f}{status}")

    # Final result
    with torch.no_grad():
        R_final = lie.so3_to_SO3(rot_vec).numpy()

    final_rot_err = rotation_error(R_final, R_gt_wc)
    final_trans_err = min(
        translation_error(best_t_wc, t_gt_wc),
        translation_error(-best_t_wc, t_gt_wc)
    )

    print(f"\n{'='*60}")
    print(f"Step XXXII summary:")
    print(f"  Best Phase1 R_err: {best_result['final_r_err']:.1f}deg")
    print(f"  Final R_err: {final_rot_err:.2f}deg")
    print(f"  Final t_err: {final_trans_err:.2f}deg")

    avg_final_r_err = np.mean([r['final_r_err'] for r in phase1_results])
    print(f"\n  Avg Phase1 R_err: {avg_final_r_err:.1f}deg")
    print(f"  Improvement: Phase1→Final = {best_result['final_r_err'] - final_rot_err:.1f}deg")

    return {
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'phase1_best_r_err': best_result['final_r_err'],
    }


# =============================================================================
# Step XXXIII: Multi-pair benchmark
# =============================================================================

def run_single_pair_optimization(
    idx1: int,
    idx2: int,
    n_starts: int = 5,
    max_init_angle_deg: float = 90.0,
    epsilon_end: float = 0.05,
    concentration_threshold: float = 0.2,
    gap_threshold: float = 0.01,
    min_topk_weight: float = 0.1,
    selection_metric: str = "avg_cost",
    verbose: bool = False,
) -> dict:
    """Run multi-start optimization on a single image pair.

    Returns dict with 'success', 'final_r_err', 'final_t_err', etc.
    """
    try:
        result = run_step_xxix_multi_start(
            idx1=idx1,
            idx2=idx2,
            n_starts=n_starts,
            max_init_angle_deg=max_init_angle_deg,
            epsilon_end=epsilon_end,
            concentration_threshold=concentration_threshold,
            gap_threshold=gap_threshold,
            min_topk_weight=min_topk_weight,
            selection_metric=selection_metric,
            verbose=verbose,
            print_header=False,
        )

        final_rot_err = result['final_rot_err']
        final_trans_err = result['final_trans_err']
        phase1_results = result['phase1_results']
        best_phase1_r_err = min(r['final_r_err'] for r in phase1_results)
        collapse_threshold = 0.5
        collapsed_starts = sum(
            1 for r in phase1_results if r.get('T_sum_end', 0.0) <= collapse_threshold
        )

        return {
            'success': True,
            'idx1': idx1,
            'idx2': idx2,
            'final_r_err': final_rot_err,
            'final_t_err': final_trans_err,
            'phase1_r_err': best_phase1_r_err,
            'n_starts': n_starts + 1,
            'collapsed_starts': collapsed_starts,
        }

    except Exception as e:
        return {
            'success': False,
            'idx1': idx1,
            'idx2': idx2,
            'error': str(e),
        }


def run_step_xxxiii_benchmark(
    pairs: list = None,
    n_starts: int = 5,
    n_seeds: int = 3,
    max_init_angle_deg: float = 90.0,
    concentration_threshold: float = 0.2,
    gap_threshold: float = 0.01,
    min_topk_weight: float = 0.1,
    selection_metric: str = "avg_cost",
    success_rot_deg: float = 10.0,
    success_trans_deg: float = 5.0,
    verbose: bool = True,
):
    """
    Step XXXIII: Benchmark success rate on multiple image pairs.

    Args:
        pairs: List of (idx1, idx2) tuples. If None, uses default pairs.
        n_starts: Number of random starts per pair.
    """
    print("\n" + "=" * 70)
    print(f"Step XXXIII: Multi-pair benchmark (n_starts={n_starts}, n_seeds={n_seeds})")
    print(f"  selection_metric={selection_metric}, success thresholds: "
          f"R<{success_rot_deg}deg, t<{success_trans_deg}deg")
    print("=" * 70)

    if pairs is None:
        pairs_env = os.getenv("STEP_XXXIII_PAIRS", "").strip()
        if pairs_env:
            pairs = []
            for token in pairs_env.split(","):
                token = token.strip()
                if not token:
                    continue
                parts = token.split("-")
                if len(parts) != 2:
                    continue
                pairs.append((int(parts[0]), int(parts[1])))
        else:
            # Default: moderate number of pairs to keep runtime reasonable
            pairs = [
                (0, 10),
                (0, 20),
                (5, 15),
                (10, 20),
                (15, 25),
            ]

    results = []
    for i, (idx1, idx2) in enumerate(pairs):
        for seed in range(n_seeds):
            if verbose:
                print(f"\n[{i+1}/{len(pairs)}][seed={seed}] "
                      f"Processing pair ({idx1}, {idx2})...", end=" ", flush=True)

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            result = run_single_pair_optimization(
                idx1, idx2,
                n_starts=n_starts,
                max_init_angle_deg=max_init_angle_deg,
                concentration_threshold=concentration_threshold,
                gap_threshold=gap_threshold,
                min_topk_weight=min_topk_weight,
                selection_metric=selection_metric,
                verbose=False,
            )
            result['seed'] = seed

            if result['success']:
                if verbose:
                    print(f"R_err={result['final_r_err']:.1f}deg, "
                          f"t_err={result['final_t_err']:.1f}deg")
            else:
                if verbose:
                    print(f"FAILED: {result.get('error', 'unknown')}")

            results.append(result)

    # Compute statistics
    successful = [r for r in results if r['success']]
    n_total = len(results)
    n_success = len(successful)

    print(f"\n{'='*60}")
    print(f"Benchmark Results ({n_success}/{n_total} pairs completed)")
    print(f"{'='*60}")

    if n_success > 0:
        r_errs = [r['final_r_err'] for r in successful]
        t_errs = [r['final_t_err'] for r in successful]
        collapsed = [r.get('collapsed_starts', 0) for r in successful]

        # Success rate at various thresholds
        print(f"\nRotation error distribution:")
        print(f"  Mean: {np.mean(r_errs):.1f}deg")
        print(f"  Median: {np.median(r_errs):.1f}deg")
        print(f"  Std: {np.std(r_errs):.1f}deg")
        print(f"  Min: {np.min(r_errs):.1f}deg, Max: {np.max(r_errs):.1f}deg")

        print(f"\nSuccess rate (R_err thresholds):")
        for thresh in [5, 10, 15, 20, 30]:
            count = sum(1 for r in r_errs if r < thresh)
            print(f"  R_err < {thresh}deg: {count}/{n_success} ({100*count/n_success:.0f}%)")

        print(f"\nTranslation error distribution:")
        print(f"  Mean: {np.mean(t_errs):.1f}deg")
        print(f"  Median: {np.median(t_errs):.1f}deg")
        print(f"  Std: {np.std(t_errs):.1f}deg")

        # Success rate at target thresholds
        success_count = sum(
            1 for r, t in zip(r_errs, t_errs)
            if r < success_rot_deg and t < success_trans_deg
        )
        print(f"\nJoint success rate:")
        print(f"  R<{success_rot_deg}deg & t<{success_trans_deg}deg: "
              f"{success_count}/{n_success} ({100*success_count/n_success:.0f}%)")

        print(f"\nCollapsed starts (Phase1, T.sum<=0.5):")
        print(f"  Mean: {np.mean(collapsed):.2f}, Max: {np.max(collapsed):.0f}")

        # Per-pair details
        print(f"\nPer-pair results:")
        print(f"{'Pair':<12} {'R_err':>8} {'t_err':>8} {'Phase1':>8} {'Seed':>6}")
        print("-" * 40)
        for r in successful:
            print(f"({r['idx1']:02d},{r['idx2']:02d})     "
                  f"{r['final_r_err']:>7.1f}° {r['final_t_err']:>7.1f}° "
                  f"{r['phase1_r_err']:>7.1f}° {r['seed']:>6}")

    return {
        'results': results,
        'n_total': n_total,
        'n_success': n_success,
        'r_errs': r_errs if n_success > 0 else [],
        't_errs': t_errs if n_success > 0 else [],
    }


# =============================================================================
# Step D (Optimization): Cov-normalized lambda_cov in Step XX / Step XXIX
# =============================================================================

def run_step_d_cov_sweep_step_xx(
    idx1: int = 0,
    idx2: int = 10,
    init_rot_error_deg: float = 30.0,
    lambda_cov_list: Optional[List[float]] = None,
):
    """Step D-1: Sweep lambda_cov (raw) with normalized scaling in Step XX."""
    if lambda_cov_list is None:
        lambda_cov_list = [0.05, 0.10, 0.15, 0.20]

    print("\n" + "=" * 70)
    print("Step D-1: Step XX with normalized lambda_cov sweep")
    print(f"  Pair: ({idx1}, {idx2}), init_rot_error_deg={init_rot_error_deg}")
    print(f"  lambda_cov(raw) list: {lambda_cov_list}")
    print("=" * 70)

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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, init_rot_error_deg, 20.0)

    scale_ratio, median_epi, median_cov = compute_cov_scale_ratio(
        g1, g2, K, R_init_wc, t_init_wc,
        sigma_epipolar=400.0,
        sigma_cov=1.0,
        epipolar_mode="sampson",
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )
    print(f"  scale_ratio={scale_ratio:.6f} (median_epi={median_epi:.4f}, median_cov={median_cov:.4f})")

    results = []
    print(f"\n{'λ_raw':>6} {'λ_eff':>10} {'R_final':>8} {'t_final':>8} {'avg_cost':>9} {'conc':>6}")
    print("-" * 54)
    for lambda_raw in lambda_cov_list:
        lambda_eff = lambda_raw * scale_ratio
        out = run_step_xx_closed_form_translation(
            idx1=idx1,
            idx2=idx2,
            init_rot_error_deg=init_rot_error_deg,
            lambda_cov=lambda_eff,
            lambda_cov_raw=lambda_raw,
            cov_scale_ratio=scale_ratio,
            epsilon_start=None,
        )
        results.append({
            'lambda_raw': lambda_raw,
            'lambda_eff': lambda_eff,
            **out,
        })
        print(f"{lambda_raw:>6.2f} {lambda_eff:>10.6f} {out['final_rot_err']:>8.2f} "
              f"{out['final_trans_err']:>8.2f} {out['final_avg_cost']:>9.4f} "
              f"{out['final_concentration']:>6.3f}")

    return results


def run_step_d_cov_sweep_step_xxix(
    idx1: int = 0,
    idx2: int = 10,
    n_starts: int = 3,
    max_init_angle_deg: float = 90.0,
    lambda_cov_list: Optional[List[float]] = None,
    concentration_threshold: float = 0.05,
    gap_threshold: float = 0.01,
    min_topk_weight: float = 0.1,
    selection_metric: str = "avg_cost",
    seed: Optional[int] = 0,
):
    """Step D-2: Sweep lambda_cov (raw) with normalized scaling in Step XXIX."""
    if lambda_cov_list is None:
        lambda_cov_list = [0.05, 0.10, 0.15, 0.20]

    print("\n" + "=" * 70)
    print("Step D-2: Step XXIX with normalized lambda_cov sweep")
    print(f"  Pair: ({idx1}, {idx2}), n_starts={n_starts}")
    print(f"  lambda_cov(raw) list: {lambda_cov_list}")
    print("=" * 70)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"  Seed: {seed}")

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

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    _, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, 0.0, 20.0)

    scale_ratio, median_epi, median_cov = compute_cov_scale_ratio(
        g1, g2, K, np.eye(3), t_init_wc,
        sigma_epipolar=400.0,
        sigma_cov=1.0,
        epipolar_mode="sampson",
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )
    print(f"  scale_ratio={scale_ratio:.6f} (median_epi={median_epi:.4f}, median_cov={median_cov:.4f})")

    results = []
    print(f"\n{'λ_raw':>6} {'λ_eff':>10} {'R_final':>8} {'t_final':>8} {'avg_cost':>9} {'conc':>6}")
    print("-" * 54)
    for lambda_raw in lambda_cov_list:
        lambda_eff = lambda_raw * scale_ratio
        out = run_step_xxix_multi_start(
            idx1=idx1,
            idx2=idx2,
            n_starts=n_starts,
            max_init_angle_deg=max_init_angle_deg,
            concentration_threshold=concentration_threshold,
            gap_threshold=gap_threshold,
            min_topk_weight=min_topk_weight,
            selection_metric=selection_metric,
            lambda_cov=lambda_eff,
            lambda_cov_raw=lambda_raw,
            cov_scale_ratio=scale_ratio,
            verbose=False,
            print_header=True,
        )
        results.append({
            'lambda_raw': lambda_raw,
            'lambda_eff': lambda_eff,
            **out,
        })
        print(f"{lambda_raw:>6.2f} {lambda_eff:>10.6f} {out['final_rot_err']:>8.2f} "
              f"{out['final_trans_err']:>8.2f} {out['final_avg_cost']:>9.4f} "
              f"{out['final_concentration']:>6.3f}")

    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="Run verifications only")
    parser.add_argument("--step-xix", action="store_true", help="Run Step XIX")
    parser.add_argument("--step-xx", action="store_true", help="Run Step XX")
    parser.add_argument("--step-xxv", action="store_true", help="Run Step XXV")
    parser.add_argument("--step-xxviii", action="store_true", help="Run Step XXVIII (multi-start)")
    parser.add_argument("--step-xxix", action="store_true", help="Run Step XXIX (improved multi-start)")
    parser.add_argument("--step-xxxii", action="store_true", help="Run Step XXXII (Essential matrix)")
    parser.add_argument("--step-xxxiii", action="store_true", help="Run Step XXXIII (multi-pair benchmark)")
    parser.add_argument("--step-d-opt", action="store_true", help="Run Step D (cov-normalized optimization sweeps)")
    parser.add_argument("--verify-essential", action="store_true", help="Verify Essential matrix at GT pose")
    parser.add_argument("--all", action="store_true", help="Run all")
    args = parser.parse_args()

    if args.all or args.verify_only or (
        not args.step_xix and not args.step_xx and not args.step_xxv
        and not args.step_xxviii and not args.step_xxix and not args.step_xxxii
        and not args.step_xxxiii and not args.step_d_opt
    ):
        # Run verifications
        verify_collapse_at_60deg()
        verify_epsilon_rho_consistency()
        n_seeds = int(os.getenv("STEP_XIX_NUM_SEEDS", "3"))
        subset_k_env = int(os.getenv("STEP_XIX_SUBSAMPLE_K", "100"))
        subset_k = None if subset_k_env <= 0 else subset_k_env
        verify_translation_landscape_stability(n_seeds=n_seeds, subset_size=subset_k)
        verify_translation_gradient()

    if args.all or args.step_xix:
        # Step XIX: Two-stage optimization
        for init_err in [30, 60]:
            run_step_xix_two_stage_optimization(init_rot_error_deg=init_err)

    if args.all or args.step_xx:
        # First test closed-form at GT
        test_closed_form_translation_at_gt()
        # Step XX/XXVI: Closed-form translation with auto epsilon and increasing LR
        # epsilon_start=None enables auto-determination from cost scale
        for init_err in [30, 60]:
            run_step_xx_closed_form_translation(
                init_rot_error_deg=init_err,
                epsilon_start=None,  # Auto-determine from cost scale
            )

    if args.all or args.step_xxv:
        step_xxv_iters = int(os.getenv("STEP_XXV_ITERS", "50"))
        step_xxv_eps_start = float(os.getenv("STEP_XXV_EPS_START", "0.05"))
        step_xxv_eps_end = float(os.getenv("STEP_XXV_EPS_END", str(step_xxv_eps_start)))
        step_xxv_anneal = int(os.getenv("STEP_XXV_ANNEAL_STEPS", "0"))
        step_xxv_opt_trans = os.getenv("STEP_XXV_OPT_TRANSLATION", "0") == "1"
        diff_mode = os.getenv("STEP_XXV_DIFF_MODE", "both").lower()
        if diff_mode == "diff":
            diff_flags = [True]
        elif diff_mode == "detach":
            diff_flags = [False]
        else:
            diff_flags = [False, True]
        for diff_flag in diff_flags:
            run_step_xxv_full_uot_differentiable(
                init_rot_error_deg=60,
                max_iter=step_xxv_iters,
                differentiable_transport=diff_flag,
                epsilon_start=step_xxv_eps_start,
                epsilon_end=step_xxv_eps_end,
                anneal_steps=step_xxv_anneal,
                optimize_translation=step_xxv_opt_trans,
            )

    if args.all or args.step_xxviii:
        # Step XXVIII: Multi-start R optimization
        n_starts = int(os.getenv("STEP_XXVIII_N_STARTS", "5"))
        max_init_angle = float(os.getenv("STEP_XXVIII_MAX_ANGLE", "90"))
        run_step_xxviii_multi_start(
            n_starts=n_starts,
            max_init_angle_deg=max_init_angle,
            epsilon_start=None,  # Auto-determine
        )

    if args.step_xxix:
        # Step XXIX-XXXI: Improved multi-start with non-oracle t selection
        n_starts = int(os.getenv("STEP_XXIX_N_STARTS", "5"))
        max_init_angle = float(os.getenv("STEP_XXIX_MAX_ANGLE", "90"))
        conc_thresh = float(os.getenv("STEP_XXIX_CONC_THRESH", "0.2"))
        gap_thresh = float(os.getenv("STEP_XXIX_GAP_THRESH", "0.01"))
        gap_metric = os.getenv("STEP_XXIX_GAP_METRIC", "gap_norm")
        gap_damp_env = os.getenv("STEP_XXIX_GAP_DAMP_SCALE", "").strip()
        gap_damp_scale = float(gap_damp_env) if gap_damp_env else None
        topk_weight = float(os.getenv("STEP_XXIX_TOPK_WEIGHT", "0.1"))
        selection_metric = os.getenv("STEP_XXIX_SELECTION", "avg_cost")
        filter_metric_env = os.getenv("STEP_XXIX_FILTER_METRIC", "").strip()
        filter_metric = filter_metric_env if filter_metric_env else None
        filter_thresh_env = os.getenv("STEP_XXIX_FILTER_THRESH", "").strip()
        filter_threshold = float(filter_thresh_env) if filter_thresh_env else None
        t_accept_metric = os.getenv("STEP_XXIX_T_ACCEPT", "topk_cost")
        t_accept_margin = float(os.getenv("STEP_XXIX_T_ACCEPT_MARGIN", "1e-4"))
        top_l = int(os.getenv("STEP_XXIX_TOP_L", "1"))
        matching_mode = os.getenv("STEP_XXIX_MATCHING", "global_topk")
        row_top_k_env = os.getenv("STEP_XXIX_ROW_TOPK", "").strip()
        row_top_k = int(row_top_k_env) if row_top_k_env else None
        grid_env = os.getenv("STEP_XXIX_GRID", "").strip()
        diversity_grid = None
        if grid_env:
            if "x" in grid_env:
                parts = grid_env.lower().split("x")
            else:
                parts = grid_env.split(",")
            if len(parts) == 2:
                diversity_grid = (int(parts[0]), int(parts[1]))
        max_per_cell = int(os.getenv("STEP_XXIX_MAX_PER_CELL", "2"))
        gate_epi_env = os.getenv("STEP_XXIX_GATE_EPI", "").strip()
        gate_cov_env = os.getenv("STEP_XXIX_GATE_COV", "").strip()
        gate_epi = int(gate_epi_env) if gate_epi_env else None
        gate_cov = int(gate_cov_env) if gate_cov_env else None
        gate_color_env = os.getenv("STEP_XXIX_GATE_COLOR", "").strip()
        gate_desc_env = os.getenv("STEP_XXIX_GATE_DESC", "").strip()
        gate_color = int(gate_color_env) if gate_color_env else None
        gate_desc = int(gate_desc_env) if gate_desc_env else None
        desc_k = int(os.getenv("STEP_XXIX_DESC_K", "8"))
        run_step_xxix_multi_start(
            n_starts=n_starts,
            max_init_angle_deg=max_init_angle,
            concentration_threshold=conc_thresh,
            gap_threshold=gap_thresh,
            gap_metric=gap_metric,
            gap_damp_scale=gap_damp_scale,
            min_topk_weight=topk_weight,
            selection_metric=selection_metric,
            filter_metric=filter_metric,
            filter_threshold=filter_threshold,
            t_accept_metric=t_accept_metric,
            t_accept_margin=t_accept_margin,
            gate_top_m_epi=gate_epi,
            gate_top_m_cov=gate_cov,
            gate_top_m_color=gate_color,
            gate_top_m_desc=gate_desc,
            desc_k=desc_k,
            top_l=top_l,
            matching_mode=matching_mode,
            row_top_k=row_top_k,
            diversity_grid=diversity_grid,
            max_per_cell=max_per_cell,
        )

    if args.step_xxxii:
        # Step XXXII: Essential matrix M-step for R local minima escape
        n_starts = int(os.getenv("STEP_XXXII_N_STARTS", "5"))
        essential_interval = int(os.getenv("STEP_XXXII_INTERVAL", "5"))
        run_step_xxxii_essential_matrix(
            n_starts=n_starts,
            essential_interval=essential_interval,
        )

    if args.verify_essential:
        # Verify Essential matrix estimation at GT pose
        verify_essential_at_gt_pose()

    if args.step_xxxiii:
        # Step XXXIII: Multi-pair benchmark
        n_starts = int(os.getenv("STEP_XXXIII_N_STARTS", "5"))
        n_seeds = int(os.getenv("STEP_XXXIII_SEEDS", "3"))
        max_init_angle = float(os.getenv("STEP_XXXIII_MAX_ANGLE", "90"))
        conc_thresh = float(os.getenv("STEP_XXXIII_CONC_THRESH", "0.2"))
        gap_thresh = float(os.getenv("STEP_XXXIII_GAP_THRESH", "0.01"))
        topk_weight = float(os.getenv("STEP_XXXIII_TOPK_WEIGHT", "0.1"))
        selection_metric = os.getenv("STEP_XXXIII_SELECTION", "avg_cost")
        success_r = float(os.getenv("STEP_XXXIII_R_THRESH", "10.0"))
        success_t = float(os.getenv("STEP_XXXIII_T_THRESH", "5.0"))
        run_step_xxxiii_benchmark(
            n_starts=n_starts,
            n_seeds=n_seeds,
            max_init_angle_deg=max_init_angle,
            concentration_threshold=conc_thresh,
            gap_threshold=gap_thresh,
            min_topk_weight=topk_weight,
            selection_metric=selection_metric,
            success_rot_deg=success_r,
            success_trans_deg=success_t,
        )

    if args.step_d_opt:
        # Step D: Cov-normalized optimization sweeps
        lambda_cov_list = _parse_float_list("STEP_D_LAMBDA_COV_LIST", [0.05, 0.10, 0.15, 0.20])
        init_rot = float(os.getenv("STEP_D_INIT_ROT", "30"))
        run_step_d_cov_sweep_step_xx(
            init_rot_error_deg=init_rot,
            lambda_cov_list=lambda_cov_list,
        )
        n_starts = int(os.getenv("STEP_D_N_STARTS", "3"))
        max_init_angle = float(os.getenv("STEP_D_MAX_ANGLE", "90"))
        conc_thresh = float(os.getenv("STEP_D_CONC_THRESH", "0.05"))
        gap_thresh = float(os.getenv("STEP_D_GAP_THRESH", "0.01"))
        topk_weight = float(os.getenv("STEP_D_TOPK_WEIGHT", "0.1"))
        selection_metric = os.getenv("STEP_D_SELECTION", "avg_cost")
        seed = int(os.getenv("STEP_D_SEED", "0"))
        run_step_d_cov_sweep_step_xxix(
            n_starts=n_starts,
            max_init_angle_deg=max_init_angle,
            concentration_threshold=conc_thresh,
            gap_threshold=gap_thresh,
            min_topk_weight=topk_weight,
            selection_metric=selection_metric,
            lambda_cov_list=lambda_cov_list,
            seed=seed,
        )
