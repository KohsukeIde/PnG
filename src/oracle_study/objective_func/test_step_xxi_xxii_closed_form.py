"""
Step XXI-XXII: Closed-form translation regression test and hardening design.

Step XXI: Verify closed-form t is correct with true COLMAP correspondences.
Step XXII: Design and compare hardening strategies for OT-based t update.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import torch
from typing import Tuple, List, Dict
from scipy.spatial.transform import Rotation

from src.utils.gaussian_utils import load_gaussians, PROJECT_ROOT
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    read_images_with_points2d,
    get_corresponding_points as colmap_get_corresponding_points,
)


# =============================================================================
# COLMAP utilities
# =============================================================================

def load_colmap_data():
    """Load COLMAP cameras and images with 2D points."""
    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")
    cameras = load_cameras_from_colmap(colmap_dir)
    images = read_images_with_points2d(os.path.join(colmap_dir, "images.bin"))
    return cameras, images


def qvec2rotmat(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])


def get_colmap_camera_params(cameras, images, image_name: str) -> dict:
    """Get camera parameters for a specific image."""
    for img_id, img_data in images.items():
        if img_data['name'] == image_name:
            cam = cameras[img_data['camera_id']]
            # Camera class has individual attributes fx, fy, cx, cy
            K = np.array([
                [cam.fx, 0, cam.cx],
                [0, cam.fy, cam.cy],
                [0, 0, 1]
            ])
            R_cw = qvec2rotmat(img_data['qw'], img_data['qx'], img_data['qy'], img_data['qz'])
            t_cw = np.array([img_data['tx'], img_data['ty'], img_data['tz']])
            return {
                'K': K,
                'R_cw': R_cw,
                't_cw': t_cw,
                'image_name': image_name,
            }
    raise ValueError(f"Image {image_name} not found")


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose from cam1 to cam2.

    Returns R_rel, t_rel where the epipolar constraint is:
        x2^T [t_rel]_x R_rel x1 = 0

    This means x2 = R_rel * x1 + t_rel * z1 (in normalized camera coordinates).
    """
    R1_cw, t1_cw = cam1['R_cw'], cam1['t_cw']
    R2_cw, t2_cw = cam2['R_cw'], cam2['t_cw']

    R1_wc = R1_cw.T
    t1_wc = -R1_cw.T @ t1_cw

    # Relative pose: transforms points from cam1 frame to cam2 frame
    R_rel = R2_cw @ R1_wc
    t_rel = R2_cw @ t1_wc + t2_cw
    t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-10)

    return R_rel, t_rel


def get_corresponding_points_for_images(images, image_name1: str, image_name2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get corresponding 2D points between two images using COLMAP tracks.
    Uses the utility from colmap_utils.
    """
    return colmap_get_corresponding_points(images, image_name1, image_name2)


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R1 @ R2.T
    trace = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(trace))


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation error in degrees (angle between directions)."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(abs(cos_angle)))


def perturb_rotation(R: np.ndarray, angle_deg: float, axis: str = "y") -> np.ndarray:
    """Perturb rotation by angle around axis."""
    axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    r_perturb = Rotation.from_rotvec(np.radians(angle_deg) * np.array(axis_vec))
    return r_perturb.as_matrix() @ R


# =============================================================================
# Step XXI: Closed-form t with true correspondences
# =============================================================================

def compute_closed_form_t_from_correspondences(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R_wc: np.ndarray,
    weights: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal translation direction via closed-form solution.
    
    Given R and correspondences (x1_i, x2_i), find t that minimizes:
        sum_i w_i * (a_i^T t)^2
    where a_i = (R @ x1_i) × x2_i (cross product)
    
    Solution: t = min eigenvector of M = sum_i w_i * a_i @ a_i^T
    
    Args:
        pts1: (N, 2) 2D points in image 1
        pts2: (N, 2) 2D points in image 2  
        K: (3, 3) intrinsic matrix
        R_wc: (3, 3) rotation matrix (world-to-camera convention)
        weights: (N,) optional weights for each correspondence
        
    Returns:
        t_opt: (3,) optimal translation direction (unit vector)
        eigvals: (3,) eigenvalues of M [smallest to largest]
    """
    N = len(pts1)
    if weights is None:
        weights = np.ones(N)
    
    K_inv = np.linalg.inv(K)
    
    # Convert to normalized coordinates
    pts1_h = np.column_stack([pts1, np.ones(N)])  # (N, 3)
    pts2_h = np.column_stack([pts2, np.ones(N)])  # (N, 3)
    
    x1_norm = (K_inv @ pts1_h.T).T  # (N, 3)
    x2_norm = (K_inv @ pts2_h.T).T  # (N, 3)
    
    # Apply rotation: Rx1
    Rx1 = (R_wc @ x1_norm.T).T  # (N, 3)
    
    # Compute cross products: a_i = Rx1_i × x2_i
    # This gives the normal to the epipolar plane
    a = np.cross(Rx1, x2_norm)  # (N, 3)
    
    # Build M = sum_i w_i * a_i @ a_i^T
    M = np.zeros((3, 3))
    for i in range(N):
        M += weights[i] * np.outer(a[i], a[i])
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M)
    
    # Smallest eigenvalue corresponds to optimal t
    t_opt = eigvecs[:, 0]  # eigvecs sorted ascending by eigenvalue
    t_opt = t_opt / (np.linalg.norm(t_opt) + 1e-10)
    
    return t_opt, eigvals


def test_step_xxi_regression():
    """
    Step XXI: Regression test - closed-form t should give ~0 error with true correspondences.
    """
    print("\n" + "=" * 70)
    print("Step XXI: Closed-form t regression test with COLMAP correspondences")
    print("=" * 70)

    cameras, images = load_colmap_data()

    idx1, idx2 = 0, 10
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']

    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

    # Get true correspondences
    pts1, pts2 = get_corresponding_points_for_images(images, f"{idx1:04d}.png", f"{idx2:04d}.png")
    print(f"\nTrue correspondences: {len(pts1)} points")
    
    # Test 1: R = GT
    print("\n--- Test 1: R = GT ---")
    t_opt, eigvals = compute_closed_form_t_from_correspondences(pts1, pts2, K, R_gt)
    t_err = translation_error(t_opt, t_gt)
    gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
    
    print(f"  t_err: {t_err:.4f} deg")
    print(f"  Eigenvalues: [{eigvals[0]:.6f}, {eigvals[1]:.6f}, {eigvals[2]:.6f}]")
    print(f"  Eigenvalue gap (λ2-λ1)/λ3: {gap:.6f}")
    
    if t_err < 1.0:
        print(f"  ✓ PASS: t_err < 1deg")
    else:
        print(f"  ✗ FAIL: t_err >= 1deg")
    
    # Test 2: R with small perturbations
    print("\n--- Test 2: R with perturbations ---")
    print(f"{'R_err':>8} {'t_err':>10} {'gap':>10}")
    print("-" * 30)
    
    for r_perturb_deg in [0, 5, 10, 20, 30]:
        R_perturbed = perturb_rotation(R_gt, r_perturb_deg, "y")
        r_err = rotation_error(R_perturbed, R_gt)
        
        t_opt, eigvals = compute_closed_form_t_from_correspondences(pts1, pts2, K, R_perturbed)
        t_err = translation_error(t_opt, t_gt)
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        
        print(f"{r_err:>8.2f} {t_err:>10.4f} {gap:>10.6f}")
    
    # Test 3: Subsample to verify robustness
    print("\n--- Test 3: Subsampling robustness (R=GT) ---")
    print(f"{'N_pts':>8} {'t_err':>10} {'gap':>10}")
    print("-" * 30)
    
    np.random.seed(42)
    for n_pts in [50, 100, 200, 500, len(pts1)]:
        n_use = min(n_pts, len(pts1))
        indices = np.random.choice(len(pts1), n_use, replace=False)
        
        t_opt, eigvals = compute_closed_form_t_from_correspondences(
            pts1[indices], pts2[indices], K, R_gt
        )
        t_err = translation_error(t_opt, t_gt)
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        
        print(f"{n_use:>8} {t_err:>10.4f} {gap:>10.6f}")
    
    return t_err < 1.0  # Return True if passed


# =============================================================================
# Step XXII: Hardening strategies comparison
# =============================================================================

def extract_hard_correspondences_rowwise_topk(
    transport: torch.Tensor,
    top_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-wise argmax + top-k rows by weight.
    
    Returns:
        indices1: source indices
        indices2: target indices (best match per source)
        weights: transport weights
    """
    T = transport.detach().cpu().numpy()
    best_j = np.argmax(T, axis=1)  # (N1,)
    weights = np.max(T, axis=1)    # (N1,)
    
    # Select top-k by weight
    top_indices = np.argsort(weights)[-top_k:]
    
    return top_indices, best_j[top_indices], weights[top_indices]


def extract_hard_correspondences_global_topk(
    transport: torch.Tensor,
    top_k: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global top-K edges by transport weight.
    
    Returns:
        indices1: source indices
        indices2: target indices
        weights: transport weights
    """
    T = transport.detach().cpu().numpy()
    flat_indices = np.argsort(T.ravel())[-top_k:]
    
    indices1 = flat_indices // T.shape[1]
    indices2 = flat_indices % T.shape[1]
    weights = T.ravel()[flat_indices]
    
    return indices1, indices2, weights


def extract_hard_correspondences_mutual(
    transport: torch.Tensor,
    max_pairs: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mutual nearest neighbors: i→j AND j→i.
    
    Returns:
        indices1: source indices
        indices2: target indices  
        weights: transport weights
    """
    T = transport.detach().cpu().numpy()
    
    # Forward: best j for each i
    best_j_for_i = np.argmax(T, axis=1)  # (N1,)
    
    # Backward: best i for each j
    best_i_for_j = np.argmax(T, axis=0)  # (N2,)
    
    # Mutual: i→j and j→i
    mutual_pairs = []
    for i, j in enumerate(best_j_for_i):
        if best_i_for_j[j] == i:
            mutual_pairs.append((i, j, T[i, j]))
    
    if len(mutual_pairs) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by weight and take top
    mutual_pairs.sort(key=lambda x: -x[2])
    mutual_pairs = mutual_pairs[:max_pairs]
    
    indices1 = np.array([p[0] for p in mutual_pairs])
    indices2 = np.array([p[1] for p in mutual_pairs])
    weights = np.array([p[2] for p in mutual_pairs])
    
    return indices1, indices2, weights


def extract_hard_correspondences_cost_filtered(
    transport: torch.Tensor,
    cost_matrix: torch.Tensor,
    top_k: int = 100,
    cost_percentile: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Global top-K edges filtered by cost threshold.
    
    Returns:
        indices1, indices2, weights
    """
    T = transport.detach().cpu().numpy()
    C = cost_matrix.detach().cpu().numpy()
    
    # Cost threshold
    cost_thresh = np.percentile(C[T > 1e-8], cost_percentile)
    
    # Mask: high transport AND low cost
    mask = (T > 1e-8) & (C < cost_thresh)
    
    valid_indices = np.where(mask.ravel())[0]
    if len(valid_indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Sort by transport weight
    weights_valid = T.ravel()[valid_indices]
    sorted_order = np.argsort(weights_valid)[-top_k:]
    selected = valid_indices[sorted_order]
    
    indices1 = selected // T.shape[1]
    indices2 = selected % T.shape[1]
    weights = T.ravel()[selected]
    
    return indices1, indices2, weights


def compute_closed_form_t_from_gaussians(
    solver: OptimalTransportSolver,
    R_wc: torch.Tensor,
    transport: torch.Tensor,
    indices1: np.ndarray,
    indices2: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute closed-form t using extracted correspondences from Gaussians.
    """
    if len(indices1) == 0:
        return np.array([0, 0, 1.0]), np.array([0, 0, 0])
    
    K = solver.k1.cpu().numpy()
    K_inv = np.linalg.inv(K)
    R = R_wc.detach().cpu().numpy()
    
    means1 = solver.means1.cpu().numpy()  # (N1, 2)
    means2 = solver.means2.cpu().numpy()  # (N2, 2)
    
    pts1 = means1[indices1]  # (K, 2)
    pts2 = means2[indices2]  # (K, 2)
    
    N = len(pts1)
    pts1_h = np.column_stack([pts1, np.ones(N)])
    pts2_h = np.column_stack([pts2, np.ones(N)])
    
    x1_norm = (K_inv @ pts1_h.T).T
    x2_norm = (K_inv @ pts2_h.T).T
    
    Rx1 = (R @ x1_norm.T).T
    a = np.cross(Rx1, x2_norm)
    
    M = np.zeros((3, 3))
    for i in range(N):
        M += weights[i] * np.outer(a[i], a[i])
    
    eigvals, eigvecs = np.linalg.eigh(M)
    t_opt = eigvecs[:, 0]
    t_opt = t_opt / (np.linalg.norm(t_opt) + 1e-10)
    
    return t_opt, eigvals


def test_step_xxii_hardening_comparison():
    """
    Step XXII: Compare different hardening strategies for t update.
    """
    print("\n" + "=" * 70)
    print("Step XXII: Hardening strategies comparison")
    print("=" * 70)
    
    cameras, images = load_colmap_data()
    
    idx1, idx2 = 0, 10
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']
    
    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)
    
    # Load Gaussians
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)
    
    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )
    
    R_wc_t = torch.tensor(R_gt, dtype=torch.float32)
    t_wc_t = torch.tensor(t_gt, dtype=torch.float32)
    
    # Compute transport at GT
    F = solver._build_F_from_wc(R_wc_t, t_wc_t)
    cost_matrix = solver.compute_cost_matrix(F)
    
    transport, _ = solver.unbalanced_sinkhorn_algorithm(
        cost_matrix, epsilon=0.05, rho=0.5,
        gate_mask=solver._last_gate_mask
    )
    
    T_sum = transport.sum().item()
    print(f"\nTransport at GT: T.sum = {T_sum:.4f}")
    
    # Compare hardening strategies
    strategies = [
        ("Soft (all)", lambda T, C: (np.arange(T.shape[0]), 
                                     np.argmax(T.detach().cpu().numpy(), axis=1),
                                     np.max(T.detach().cpu().numpy(), axis=1))),
        ("Row-wise top-50", lambda T, C: extract_hard_correspondences_rowwise_topk(T, 50)),
        ("Row-wise top-100", lambda T, C: extract_hard_correspondences_rowwise_topk(T, 100)),
        ("Global top-50", lambda T, C: extract_hard_correspondences_global_topk(T, 50)),
        ("Global top-100", lambda T, C: extract_hard_correspondences_global_topk(T, 100)),
        ("Mutual", lambda T, C: extract_hard_correspondences_mutual(T, 200)),
        ("Cost-filtered top-50", lambda T, C: extract_hard_correspondences_cost_filtered(T, C, 50, 30)),
        ("Cost-filtered top-100", lambda T, C: extract_hard_correspondences_cost_filtered(T, C, 100, 50)),
    ]
    
    print(f"\n{'Strategy':<25} {'N_pairs':>8} {'t_err':>10} {'gap':>12}")
    print("-" * 60)
    
    results = []
    for name, extract_fn in strategies:
        indices1, indices2, weights = extract_fn(transport, cost_matrix)
        
        if len(indices1) == 0:
            print(f"{name:<25} {'0':>8} {'N/A':>10} {'N/A':>12}")
            continue
        
        t_opt, eigvals = compute_closed_form_t_from_gaussians(
            solver, R_wc_t, transport, indices1, indices2, weights
        )
        t_err = translation_error(t_opt, t_gt)
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        
        print(f"{name:<25} {len(indices1):>8} {t_err:>10.4f} {gap:>12.6f}")
        results.append((name, len(indices1), t_err, gap))
    
    # Best strategy
    if results:
        best = min(results, key=lambda x: x[2])
        print(f"\n✓ Best strategy: {best[0]} (t_err={best[2]:.4f}deg)")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Step XXI-XXII: Closed-form translation tests")
    print("=" * 70)
    
    # Step XXI: Regression test
    passed = test_step_xxi_regression()
    
    # Step XXII: Hardening comparison
    results = test_step_xxii_hardening_comparison()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Step XXI (COLMAP regression): {'PASS' if passed else 'FAIL'}")
    print(f"Step XXII (hardening comparison): {len(results)} strategies tested")
