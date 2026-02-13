#!/usr/bin/env python3
"""
Step C: Affine Correspondence from 2D Gaussian Covariances

The covariance of a 2D Gaussian encodes local shape information.
If Gaussians correspond to the same 3D surface patch, their covariances
are related by the local affine transformation induced by the camera motion.

Affine Epipolar Constraint:
Given point correspondence (x1, x2) and local affine A (2x2 matrix),
in addition to x2^T E x1 = 0, we have:
    e2^T * A - λ * e1^T = 0
where e1, e2 are epipolar line directions and λ is the scale.

For 2D Gaussians with covariances Σ1, Σ2:
If they correspond, Σ2 ≈ A Σ1 A^T for some affine A.
This A encodes the local surface orientation and camera motion.

Usage:
    python test_affine_correspondence_analysis.py
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


def extract_affine_from_covariances(
    cov1: np.ndarray,  # 2x2 covariance of Gaussian 1
    cov2: np.ndarray,  # 2x2 covariance of Gaussian 2
) -> Optional[np.ndarray]:
    """
    Extract affine transformation A such that cov2 ≈ A @ cov1 @ A^T.

    Uses Cholesky decomposition: Σ = L L^T
    If Σ2 = A Σ1 A^T, then L2 = A L1 (up to rotation).

    Returns A (2x2) or None if decomposition fails.
    """
    try:
        # Cholesky decomposition
        L1 = np.linalg.cholesky(cov1 + 1e-6 * np.eye(2))
        L2 = np.linalg.cholesky(cov2 + 1e-6 * np.eye(2))

        # A = L2 @ L1^{-1}
        A = L2 @ np.linalg.inv(L1)
        return A
    except np.linalg.LinAlgError:
        return None


def compute_affine_epipolar_residual(
    x1: np.ndarray,    # Point in image 1 (3,) homogeneous
    x2: np.ndarray,    # Point in image 2 (3,) homogeneous
    A: np.ndarray,     # Affine transformation (2x2)
    F: np.ndarray,     # Fundamental matrix (3x3)
) -> float:
    """
    Compute affine epipolar residual.

    Standard epipolar: r_point = x2^T F x1
    Affine constraint: The local affine A relates tangent spaces.

    For affine correspondence, the epipolar line directions should be related by A:
        l2 = F x1  (epipolar line in image 2)
        l1 = F^T x2  (epipolar line in image 1)
        tangent(l2) ≈ A @ tangent(l1)

    Returns combined residual.
    """
    # Standard epipolar residual
    r_point = abs(x2.T @ F @ x1)

    # Epipolar line directions (perpendicular to epipolar lines)
    l2 = F @ x1  # (a, b, c) where ax + by + c = 0
    l1 = F.T @ x2

    # Tangent directions (normalized)
    t1 = np.array([-l1[1], l1[0]])  # perpendicular to line normal
    t2 = np.array([-l2[1], l2[0]])

    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)

    # Check if A @ t1 aligns with t2
    At1 = A @ t1_norm
    At1_norm = At1 / (np.linalg.norm(At1) + 1e-10)

    # Angular difference
    cos_angle = np.abs(np.dot(At1_norm, t2_norm))
    r_affine = 1 - cos_angle  # 0 if perfectly aligned

    return r_point, r_affine


def compute_essential_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute Essential matrix from R and t."""
    tx = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = tx @ R
    return E


def compute_fundamental_matrix(E: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Compute Fundamental matrix from Essential matrix and intrinsics."""
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    return F


@dataclass
class AffineCorrespondenceResult:
    """Result of affine correspondence analysis."""
    idx1: int
    idx2: int
    mass: float
    point_residual: float
    affine_residual: float
    affine_det: float  # |det(A)| - indicates scale change
    affine_cond: float  # condition number of A


def analyze_affine_correspondences(
    solver: OptimalTransportSolver,
    transport: torch.Tensor,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    top_k: int = 50,
    min_mass: float = 0.001,
) -> List[AffineCorrespondenceResult]:
    """
    Analyze affine correspondences from transport matrix.

    For each high-mass correspondence, extract affine from covariances
    and compute affine epipolar residual.
    """
    means1 = solver.gaussians1.means
    means2 = solver.gaussians2.means
    covs1 = solver.gaussians1.covs
    covs2 = solver.gaussians2.covs

    E = compute_essential_matrix(R, t)
    F = compute_fundamental_matrix(E, K)

    T_np = transport.detach().cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]

    n1, n2 = T_np.shape
    results = []

    for idx in top_indices:
        i = idx // n2
        j = idx % n2
        mass = T_flat[idx]

        if mass < min_mass:
            continue

        # Get points
        p1 = means1[i]
        p2 = means2[j]
        if isinstance(p1, torch.Tensor):
            p1 = p1.numpy()
            p2 = p2.numpy()

        x1 = np.array([p1[0], p1[1], 1.0])
        x2 = np.array([p2[0], p2[1], 1.0])

        # Get covariances
        c1 = covs1[i]
        c2 = covs2[j]
        if isinstance(c1, torch.Tensor):
            c1 = c1.numpy()
            c2 = c2.numpy()

        # Extract affine
        A = extract_affine_from_covariances(c1, c2)
        if A is None:
            continue

        # Compute residuals
        r_point, r_affine = compute_affine_epipolar_residual(x1, x2, A, F)

        # Affine statistics
        det_A = np.abs(np.linalg.det(A))
        cond_A = np.linalg.cond(A)

        results.append(AffineCorrespondenceResult(
            idx1=i, idx2=j, mass=mass,
            point_residual=r_point,
            affine_residual=r_affine,
            affine_det=det_A,
            affine_cond=cond_A,
        ))

    return results


def run_affine_analysis(
    idx1: int = 0,
    idx2: int = 10,
    epsilon: float = 0.05,
    rho: float = 0.5,
):
    """Run affine correspondence analysis."""

    print("\n" + "=" * 70)
    print("Step C: Affine Correspondence Analysis")
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

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    # Compute transport at GT pose
    R_t = torch.tensor(R_gt, dtype=torch.float32)
    t_t = torch.tensor(t_gt, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    F = solver._build_F_from_wc(R_t, t_t)
    C = solver.compute_cost_matrix(F)

    with torch.no_grad():
        transport, _ = solver.unbalanced_sinkhorn_algorithm(
            C, epsilon=epsilon, rho=rho,
            gate_mask=solver._last_gate_mask
        )

    print(f"\nTransport at GT pose: T_sum = {transport.sum().item():.3f}")

    # Analyze affine correspondences at GT
    print("\n--- Analysis at GT Pose ---")
    results_gt = analyze_affine_correspondences(
        solver, transport, R_gt, t_gt, K, top_k=100
    )

    if results_gt:
        point_residuals = [r.point_residual for r in results_gt]
        affine_residuals = [r.affine_residual for r in results_gt]
        det_As = [r.affine_det for r in results_gt]
        cond_As = [r.affine_cond for r in results_gt]

        print(f"\nAnalyzed {len(results_gt)} correspondences")
        print(f"\nPoint residual (x2^T F x1):")
        print(f"  mean={np.mean(point_residuals):.4f}, std={np.std(point_residuals):.4f}")
        print(f"  min={np.min(point_residuals):.4f}, max={np.max(point_residuals):.4f}")

        print(f"\nAffine residual (tangent alignment):")
        print(f"  mean={np.mean(affine_residuals):.4f}, std={np.std(affine_residuals):.4f}")
        print(f"  min={np.min(affine_residuals):.4f}, max={np.max(affine_residuals):.4f}")

        print(f"\nAffine |det(A)| (scale change):")
        print(f"  mean={np.mean(det_As):.4f}, std={np.std(det_As):.4f}")
        print(f"  min={np.min(det_As):.4f}, max={np.max(det_As):.4f}")

        print(f"\nAffine cond(A) (conditioning):")
        print(f"  mean={np.mean(cond_As):.2f}, std={np.std(cond_As):.2f}")
        print(f"  min={np.min(cond_As):.2f}, max={np.max(cond_As):.2f}")

    # Test at perturbed poses
    print("\n--- Comparison: GT vs Perturbed Poses ---")

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

    test_cases = [
        ("GT", R_gt, t_gt, 0.0),
        ("5deg perturb", random_rotation_matrix(5) @ R_gt, t_gt, 5.0),
        ("15deg perturb", random_rotation_matrix(15) @ R_gt, t_gt, 15.0),
        ("45deg perturb", random_rotation_matrix(45) @ R_gt, t_gt, 45.0),
        ("Random R", random_rotation_matrix(90), t_gt, None),
    ]

    print(f"\n{'Case':<20} {'R_err':>8} {'mean_pt_res':>12} {'mean_aff_res':>12} {'T_sum':>8}")
    print("-" * 65)

    for name, R, t, expected_err in test_cases:
        R_t = torch.tensor(R, dtype=torch.float32)
        t_t = torch.tensor(t, dtype=torch.float32)
        t_t = t_t / (t_t.norm() + 1e-10)

        F = solver._build_F_from_wc(R_t, t_t)
        C = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                C, epsilon=epsilon, rho=rho,
                gate_mask=solver._last_gate_mask
            )

        T_sum = transport.sum().item()
        results = analyze_affine_correspondences(solver, transport, R, t, K, top_k=50)

        R_err = rotation_error(R, R_gt)

        if results:
            mean_pt = np.mean([r.point_residual for r in results])
            mean_aff = np.mean([r.affine_residual for r in results])
        else:
            mean_pt = float('nan')
            mean_aff = float('nan')

        print(f"{name:<20} {R_err:>8.1f} {mean_pt:>12.4f} {mean_aff:>12.4f} {T_sum:>8.3f}")

    # Correlation analysis: Can affine residual predict R_err?
    print("\n--- Affine Residual vs R_err Correlation ---")

    n_samples = 50
    R_errs = []
    mean_pt_residuals = []
    mean_aff_residuals = []

    for i in range(n_samples):
        R_rand = random_rotation_matrix(90)
        t_rand = np.random.randn(3)
        t_rand = t_rand / (np.linalg.norm(t_rand) + 1e-10)

        R_t = torch.tensor(R_rand, dtype=torch.float32)
        t_t = torch.tensor(t_rand, dtype=torch.float32)
        t_t = t_t / (t_t.norm() + 1e-10)

        F = solver._build_F_from_wc(R_t, t_t)
        C = solver.compute_cost_matrix(F)

        with torch.no_grad():
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                C, epsilon=epsilon, rho=rho,
                gate_mask=solver._last_gate_mask
            )

        results = analyze_affine_correspondences(solver, transport, R_rand, t_rand, K, top_k=30)

        if results and len(results) >= 5:
            R_errs.append(rotation_error(R_rand, R_gt))
            mean_pt_residuals.append(np.mean([r.point_residual for r in results]))
            mean_aff_residuals.append(np.mean([r.affine_residual for r in results]))

    if len(R_errs) > 10:
        from scipy import stats
        corr_pt, p_pt = stats.spearmanr(mean_pt_residuals, R_errs)
        corr_aff, p_aff = stats.spearmanr(mean_aff_residuals, R_errs)

        print(f"\nSamples with valid correspondences: {len(R_errs)}/{n_samples}")
        print(f"Spearman(mean_point_res, R_err): {corr_pt:.3f} (p={p_pt:.2e})")
        print(f"Spearman(mean_affine_res, R_err): {corr_aff:.3f} (p={p_aff:.2e})")

    return results_gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step C: Affine Correspondence Analysis")
    parser.add_argument("--idx1", type=int, default=0, help="First image index")
    parser.add_argument("--idx2", type=int, default=10, help="Second image index")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--rho", type=float, default=0.5, help="Unbalanced OT rho")
    args = parser.parse_args()

    run_affine_analysis(
        idx1=args.idx1,
        idx2=args.idx2,
        epsilon=args.epsilon,
        rho=args.rho,
    )
