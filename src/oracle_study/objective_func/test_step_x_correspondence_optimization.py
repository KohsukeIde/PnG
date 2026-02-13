"""
Step X: End-to-end optimization with corresponding points as Gaussians.

Purpose: Confirm that "optimization code works correctly with OT+Sampson for pose"
by using COLMAP corresponding points converted to TwoDGaussians.

This is a regression test - if this works, the optimization pipeline is correct.
"""

import os
import sys
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    read_images_with_points2d,
    get_corresponding_points,
    load_cameras_from_colmap,
    quaternion_to_rotation_matrix,
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
    """Compute translation direction error in degrees."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> tuple:
    """Compute relative pose (world-to-camera) from camera 1 to camera 2."""
    R1_w2c = quaternion_to_rotation_matrix(
        cam1['qw'], cam1['qx'], cam1['qy'], cam1['qz']
    )
    t1_w2c = np.array([cam1['tx'], cam1['ty'], cam1['tz']])

    R2_w2c = quaternion_to_rotation_matrix(
        cam2['qw'], cam2['qx'], cam2['qy'], cam2['qz']
    )
    t2_w2c = np.array([cam2['tx'], cam2['ty'], cam2['tz']])

    # Relative pose: cam1 -> cam2
    R_12 = R2_w2c @ R1_w2c.T
    t_12 = t2_w2c - R_12 @ t1_w2c
    t_12_norm = t_12 / (np.linalg.norm(t_12) + 1e-10)

    return R_12, t_12_norm


def points_to_gaussians(pts: np.ndarray, sigma: float = 5.0) -> TwoDGaussians:
    """Convert 2D points to TwoDGaussians with small isotropic covariance.

    Args:
        pts: (N, 2) array of 2D points
        sigma: Standard deviation for isotropic Gaussian

    Returns:
        TwoDGaussians object
    """
    N = len(pts)

    means = pts.astype(np.float32)
    scales = np.full((N, 2), sigma, dtype=np.float32)
    rotations = np.zeros(N, dtype=np.float32)

    # Covariance matrices: sigma^2 * I
    covs = np.zeros((N, 2, 2), dtype=np.float32)
    for i in range(N):
        covs[i] = np.eye(2) * (sigma ** 2)

    # Uniform alpha and dummy RGB
    alpha = np.ones(N, dtype=np.float32) / N
    rgb = np.ones((N, 3), dtype=np.float32) * 0.5  # Gray

    return TwoDGaussians(
        means=means,
        covs=covs,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha,
    )


def run_step_x_optimization(
    idx1: int = 0,
    idx2: int = 10,
    init_rot_error_deg: float = 30,
    max_iter: int = 200,
    score_type: str = "full_uot",
    max_points: int = 300,
):
    """Run Step X: End-to-end optimization with corresponding points."""
    print("=" * 70)
    print(f"Step X: Corresponding Points -> Gaussian Optimization")
    print(f"  Init rot error: {init_rot_error_deg}deg")
    print(f"  Score type: {score_type}")
    print(f"  Max points: {max_points}")
    print("=" * 70)

    colmap_dir = os.path.join(project_root, "data/DTU/scan63/sparse/0")

    # Load images with 2D point observations
    print("\nLoading COLMAP data...")
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)

    # Get corresponding points
    image_name1 = f"{idx1:04d}.png"
    image_name2 = f"{idx2:04d}.png"
    pts1, pts2 = get_corresponding_points(images_with_pts, image_name1, image_name2)
    print(f"Found {len(pts1)} corresponding points")

    # Subsample if too many
    if len(pts1) > max_points:
        np.random.seed(42)
        indices = np.random.choice(len(pts1), max_points, replace=False)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
        print(f"Subsampled to {len(pts1)} points")

    # Convert to Gaussians
    print("\nConverting points to Gaussians...")
    g1 = points_to_gaussians(pts1, sigma=5.0)
    g2 = points_to_gaussians(pts2, sigma=5.0)

    # Load camera intrinsics
    cameras = load_cameras_from_colmap(colmap_dir)
    cam1_data = None
    cam2_data = None
    for img_data in images_with_pts.values():
        if img_data["name"] == image_name1:
            cam1_data = img_data
        if img_data["name"] == image_name2:
            cam2_data = img_data

    K = cameras[cam1_data["camera_id"]].get_camera_matrix()

    # Get GT relative pose (world-to-camera)
    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1_data, cam2_data)

    # Create perturbed initial pose
    R_perturb = rodrigues_rotation(np.array([0, 1, 0]), np.radians(init_rot_error_deg))
    R_init_wc = R_perturb @ R_gt_wc
    t_init_wc = t_gt_wc.copy()  # Start with GT translation

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    print(f"\nInitial errors (w2c):")
    print(f"  Rotation: {init_rot_err:.2f} deg")
    print(f"  Translation: {init_trans_err:.2f} deg")

    # Create solver
    # Note: sigma_epipolar needs to be large enough to avoid numerical underflow
    # in exp(-C/ε). With Sampson cost and small epsilon, costs can be large.
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
        sigma_epipolar=100.0,  # Larger sigma to avoid underflow
        epi_clip=None,
    )

    # Initialize SE3 params using SE3_to_se3 (Step VIII fix)
    R_cw_init = R_init_wc.T
    t_cw_init = -R_init_wc.T @ t_init_wc
    Rt_cw = np.concatenate([R_cw_init, t_cw_init[:, None]], axis=1)
    Rt_cw_tensor = torch.tensor(Rt_cw, dtype=torch.float32)
    se3_vec = solver.lie.SE3_to_se3(Rt_cw_tensor)
    solver.rot_vec = torch.nn.Parameter(se3_vec[:3].clone())
    solver.trans_vec = torch.nn.Parameter(se3_vec[3:].clone())

    # Verify initialization
    with torch.no_grad():
        T_check = solver.lie.se3_to_SE3(se3_vec)
        R_check = T_check[:3, :3].numpy()
        t_check = T_check[:3, 3].numpy()
        R_wc_check = R_check.T
        t_wc_check = -R_check.T @ t_check
        t_wc_check /= (np.linalg.norm(t_wc_check) + 1e-10)
        init_verify_rot = rotation_error(R_wc_check, R_init_wc)
        init_verify_trans = translation_error(t_wc_check, t_init_wc)
        print(f"  Init verification: R_err={init_verify_rot:.4f}deg, t_err={init_verify_trans:.4f}deg")

    # Output directories
    results_dir = os.path.join(
        project_root, "results",
        f"step_x_{idx1}_{idx2}_{init_rot_error_deg}deg_{score_type}"
    )
    diagnostics_dir = os.path.join(results_dir, "diagnostics")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Run optimization
    print(f"\nStarting optimization...")
    print(f"  Results: {results_dir}")

    # Use larger epsilon to avoid mass collapse
    # Corresponding points have larger Sampson costs when pose is off
    epsilon = 0.1
    rho = 1.0

    loss_history = solver.optimize_with_SE3(
        max_iter=max_iter,
        rot_lr=1e-3,
        trans_lr=1e-4,
        momentum=0.9,
        tol=1e-8,
        diagnostics_dir=diagnostics_dir,
        differentiable_transport=False,
        sinkhorn_epsilon=epsilon,
        sinkhorn_rho=rho,
        score_type=score_type,
        lambda_kl=0.1,
        epsilon_annealing=True,
        epsilon_start=0.5,  # Higher start to avoid initial collapse
        epsilon_end=epsilon,
        anneal_steps=100,
    )

    # Extract final pose
    with torch.no_grad():
        se3_vec_final = torch.cat([solver.rot_vec, solver.trans_vec])
        T_cw_final = solver.lie.se3_to_SE3(se3_vec_final)
        R_cw_final = T_cw_final[:3, :3].numpy()
        t_cw_final = T_cw_final[:3, 3].numpy()

        R_wc_final = R_cw_final.T
        t_wc_final = -R_cw_final.T @ t_cw_final
        t_wc_final = t_wc_final / (np.linalg.norm(t_wc_final) + 1e-10)

    final_rot_err = rotation_error(R_wc_final, R_gt_wc)
    final_trans_err = min(
        translation_error(t_wc_final, t_gt_wc),
        translation_error(-t_wc_final, t_gt_wc)
    )

    print(f"\nFinal errors (w2c):")
    print(f"  Rotation: {final_rot_err:.2f} deg (was {init_rot_err:.2f})")
    print(f"  Translation: {final_trans_err:.2f} deg (was {init_trans_err:.2f})")

    rot_improved = init_rot_err - final_rot_err
    trans_change = init_trans_err - final_trans_err

    print(f"\nImprovement:")
    print(f"  Rotation: {rot_improved:.2f} deg {'(better)' if rot_improved > 0 else '(worse)'}")
    print(f"  Translation: {trans_change:.2f} deg")

    return {
        'init_rot_err': init_rot_err,
        'init_trans_err': init_trans_err,
        'final_rot_err': final_rot_err,
        'final_trans_err': final_trans_err,
        'rot_improved': rot_improved,
        'trans_change': trans_change,
        'success': rot_improved > 5,  # Significant improvement
    }


def run_comprehensive_step_x():
    """Run Step X with multiple configurations."""
    print("=" * 70)
    print("Step X: Comprehensive Corresponding Points Optimization Test")
    print("=" * 70)

    configs = [
        (30, "full_uot"),
        (30, "avg_cost"),
        (60, "full_uot"),
    ]

    results = []
    for init_err, score_type in configs:
        print(f"\n{'='*70}")
        result = run_step_x_optimization(
            init_rot_error_deg=init_err,
            score_type=score_type,
            max_iter=200,
        )
        result['config'] = {'init_err': init_err, 'score_type': score_type}
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Step X Summary")
    print("=" * 70)
    print(f"{'Init Err':>10} {'Score':>12} {'Init R':>10} {'Final R':>10} {'Init t':>10} {'Final t':>10} {'Status':>10}")
    print("-" * 80)

    for r in results:
        cfg = r['config']
        status = "OK" if r['success'] else "FAIL"
        print(f"{cfg['init_err']:>10} {cfg['score_type']:>12} "
              f"{r['init_rot_err']:>10.2f} {r['final_rot_err']:>10.2f} "
              f"{r['init_trans_err']:>10.2f} {r['final_trans_err']:>10.2f} {status:>10}")

    return results


if __name__ == "__main__":
    # Quick single test
    result = run_step_x_optimization(
        init_rot_error_deg=30,
        score_type="full_uot",
        max_iter=200,
    )

    if result['success']:
        print("\n" + "=" * 70)
        print("Step X PASSED: Optimization works with corresponding points!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Step X FAILED: Optimization did not improve significantly")
        print("=" * 70)
