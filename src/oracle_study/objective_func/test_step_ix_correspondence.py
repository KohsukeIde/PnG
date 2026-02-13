"""
Step IX: Test translation landscape with corresponding point sets.

Purpose: Verify whether the loss landscape issue is:
1. Implementation problem (if GT is minimum with correspondences)
2. 2DGS representation problem (if GT is still not minimum)

Uses COLMAP 3D point tracks to get known correspondences between images.
"""

import csv
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

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
from src.oracle_study.objective_func.test_step_f_score_functions import compute_all_scores


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix using Rodrigues formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def compute_sampson_distance(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute Sampson distance for each point pair.

    Args:
        pts1: (N, 2) array of 2D points in image 1
        pts2: (N, 2) array of 2D points in image 2
        F: (3, 3) fundamental matrix

    Returns:
        (N,) array of Sampson distances
    """
    # Homogeneous coordinates
    ones = np.ones((pts1.shape[0], 1))
    p1_hom = np.hstack([pts1, ones])  # (N, 3)
    p2_hom = np.hstack([pts2, ones])  # (N, 3)

    # Epipolar constraint: x2^T @ F @ x1
    Fp1 = F @ p1_hom.T  # (3, N)
    Ftp2 = F.T @ p2_hom.T  # (3, N)

    # Algebraic error
    x2_F_x1 = np.sum(p2_hom * Fp1.T, axis=1)  # (N,)

    # Sampson correction denominator
    denom = Fp1[0, :]**2 + Fp1[1, :]**2 + Ftp2[0, :]**2 + Ftp2[1, :]**2

    # Sampson distance
    sampson = x2_F_x1**2 / (denom + 1e-10)

    return sampson


def build_gaussians_from_points(points: np.ndarray) -> TwoDGaussians:
    """Create minimal Gaussian containers from 2D points."""
    num_points = points.shape[0]
    covs = np.tile(np.eye(2, dtype=np.float32)[None, :, :], (num_points, 1, 1))
    rgb = np.zeros((num_points, 3), dtype=np.float32)
    alpha = np.ones((num_points,), dtype=np.float32)
    rotations = np.zeros((num_points,), dtype=np.float32)
    scales = np.ones((num_points, 2), dtype=np.float32)
    return TwoDGaussians(
        means=points.astype(np.float32),
        covs=covs,
        rgb=rgb,
        alpha=alpha,
        rotations=rotations,
        scales=scales,
    )


def build_fundamental_matrix(R_wc: np.ndarray, t_wc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Build fundamental matrix from world-to-camera relative pose.

    Args:
        R_wc: (3, 3) rotation matrix (world-to-camera)
        t_wc: (3,) translation direction (world-to-camera, unit norm)
        K: (3, 3) camera intrinsic matrix

    Returns:
        (3, 3) fundamental matrix
    """
    # Skew-symmetric matrix
    t_skew = np.array([
        [0, -t_wc[2], t_wc[1]],
        [t_wc[2], 0, -t_wc[0]],
        [-t_wc[1], t_wc[0], 0]
    ])

    # Essential matrix: E = [t]_x @ R
    E = t_skew @ R_wc

    # Fundamental matrix: F = K2^{-T} @ E @ K1^{-1}
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv

    return F


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> tuple:
    """Compute relative pose (world-to-camera) from camera 1 to camera 2.

    COLMAP stores world-to-camera (w2c) pose: P_cam = R @ P_world + t.
    """
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


def run_step_ix_correspondence_test(idx1: int = 0, idx2: int = 10):
    """Run Step IX: Test translation landscape with corresponding points."""
    print("=" * 70)
    print("Step IX: Translation Loss Landscape with Corresponding Points")
    print("=" * 70)

    colmap_dir = os.path.join(project_root, "data/DTU/scan63/sparse/0")

    # Load images with 2D point observations
    print("\nLoading COLMAP images with 2D points...")
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)

    # Get corresponding points between images
    image_name1 = f"{idx1:04d}.png"
    image_name2 = f"{idx2:04d}.png"
    print(f"Getting correspondences between {image_name1} and {image_name2}...")

    pts1, pts2 = get_corresponding_points(images_with_pts, image_name1, image_name2)
    print(f"Found {len(pts1)} corresponding points")

    max_points = int(os.getenv("STEP_IX_MAX_POINTS", "300"))
    seed = int(os.getenv("STEP_IX_SEED", "0"))
    if len(pts1) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pts1), size=max_points, replace=False)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
        print(f"Subsampled to {len(pts1)} correspondences (seed={seed})")

    if len(pts1) < 10:
        print("Too few correspondences!")
        return

    # Load camera intrinsics
    cameras = load_cameras_from_colmap(colmap_dir)
    # Find camera_id for our images
    cam1_data = None
    cam2_data = None
    for img_data in images_with_pts.values():
        if img_data["name"] == image_name1:
            cam1_data = img_data
        if img_data["name"] == image_name2:
            cam2_data = img_data

    K = cameras[cam1_data["camera_id"]].get_camera_matrix()

    # Get GT relative pose (world-to-camera)
    R_wc_gt, t_wc_gt = compute_relative_pose_wc(cam1_data, cam2_data)

    print(f"\nGT pose:")
    print(f"  R_wc_gt diagonal: {np.diag(R_wc_gt)}")
    print(f"  t_wc_gt: {t_wc_gt}")

    # Verify GT Sampson cost
    F_gt = build_fundamental_matrix(R_wc_gt, t_wc_gt, K)
    sampson_gt = compute_sampson_distance(pts1, pts2, F_gt)
    print(f"\nGT Sampson cost:")
    print(f"  Mean: {np.mean(sampson_gt):.6f}")
    print(f"  Median: {np.median(sampson_gt):.6f}")
    print(f"  Max: {np.max(sampson_gt):.6f}")

    # Sweep translation direction
    max_angle = float(os.getenv("STEP_IX_MAX_ANGLE", "60"))
    angle_step = float(os.getenv("STEP_IX_ANGLE_STEP", "10"))
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    angles = np.arange(-max_angle, max_angle + 1e-6, angle_step)

    results_dir = os.path.join(project_root, "results", "step_ix_correspondence")
    os.makedirs(results_dir, exist_ok=True)

    scores_by_axis = {k: [] for k in axes}
    records = []

    print("\nComputing Sampson cost landscape...")
    for axis_name, axis_vec in axes.items():
        for angle in angles:
            R_delta = rodrigues_rotation(axis_vec, np.radians(angle))
            t_wc = R_delta @ t_wc_gt
            t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)

            F = build_fundamental_matrix(R_wc_gt, t_wc, K)
            sampson = compute_sampson_distance(pts1, pts2, F)
            avg_sampson = np.mean(sampson)

            scores_by_axis[axis_name].append(avg_sampson)
            records.append({
                "axis": axis_name,
                "angle_deg": float(angle),
                "avg_sampson": avg_sampson,
            })

    # Save CSV
    csv_path = os.path.join(results_dir, "correspondence_landscape.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "angle_deg", "avg_sampson"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved results to {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for axis_name in axes:
        ax.plot(angles, scores_by_axis[axis_name], label=f"axis {axis_name}", marker='o')
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="GT (0 deg)")
    ax.set_xlabel("Translation perturbation (deg)")
    ax.set_ylabel("Mean Sampson Distance")
    ax.set_title(f"Step IX: Sampson Landscape with {len(pts1)} Corresponding Points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correspondence_landscape.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {results_dir}/correspondence_landscape.png")

    # Print summary of minima
    print("\n" + "=" * 70)
    print("MINIMA SUMMARY (Step IX)")
    print("=" * 70)
    for axis_name in axes:
        values = np.array(scores_by_axis[axis_name])
        min_idx = int(np.argmin(values))
        min_angle = angles[min_idx]
        gt_idx = int(np.where(np.isclose(angles, 0.0))[0][0])
        gt_value = values[gt_idx]
        print(
            f"  axis={axis_name}: min at {min_angle:.1f} deg "
            f"(value={values[min_idx]:.6f}), "
            f"GT=0deg value={gt_value:.6f}"
        )

    # Key diagnostic: Is GT the minimum?
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Is GT the minimum for each axis?")
    print("=" * 70)
    all_gt_minimum = True
    for axis_name in axes:
        values = np.array(scores_by_axis[axis_name])
        min_idx = int(np.argmin(values))
        min_angle = angles[min_idx]
        is_gt_min = np.isclose(min_angle, 0.0, atol=angle_step/2)
        status = "YES" if is_gt_min else "NO"
        print(f"  axis={axis_name}: GT is minimum? {status}")
        if not is_gt_min:
            all_gt_minimum = False

    print("\n" + "=" * 70)
    if all_gt_minimum:
        print("CONCLUSION: GT IS THE MINIMUM with corresponding points!")
        print("This suggests the 2DGS representation is the problem, not the implementation.")
    else:
        print("CONCLUSION: GT IS NOT THE MINIMUM even with corresponding points!")
        print("This suggests there may still be an implementation issue.")
    print("=" * 70)

    # -------------------- OT landscape with correspondences -------------------- #
    run_ot = os.getenv("RUN_STEP_IX_OT", "1") == "1"
    if not run_ot:
        return

    print("\n" + "=" * 70)
    print("Step IX-OT: Translation Landscape with OT (Correspondence Points)")
    print("=" * 70)

    epsilon = float(os.getenv("STEP_IX_EPSILON", "0.05"))
    rho = float(os.getenv("STEP_IX_RHO", "0.5"))
    sinkhorn_max_iter = int(os.getenv("STEP_IX_SINKHORN_ITERS", "120"))

    g1 = build_gaussians_from_points(pts1)
    g2 = build_gaussians_from_points(pts2)
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
        sigma_epipolar=1.0,
        epi_clip=None,
    )
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    scores_by_axis_ot = {
        "avg_cost": {k: [] for k in axes},
        "full_uot": {k: [] for k in axes},
    }
    records_ot = []

    with torch.no_grad():
        R_wc_t = torch.tensor(R_wc_gt, dtype=torch.float32)
        for axis_name, axis_vec in axes.items():
            for angle in angles:
                R_delta = rodrigues_rotation(axis_vec, np.radians(angle))
                t_wc = R_delta @ t_wc_gt
                t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)
                t_wc_t = torch.tensor(t_wc, dtype=torch.float32)

                F = solver._build_F_from_wc(R_wc_t, t_wc_t)
                cost = solver.compute_cost_matrix(F)
                transport, _ = solver.unbalanced_sinkhorn_algorithm(
                    cost_matrix=cost,
                    epsilon=epsilon,
                    rho=rho,
                    max_iter=sinkhorn_max_iter,
                    gate_mask=solver._last_gate_mask,
                )
                eps_actual = solver._last_sinkhorn_epsilon
                rho_actual = solver._last_sinkhorn_rho
                if eps_actual is None or rho_actual is None:
                    raise RuntimeError("Sinkhorn failed to set eps/rho for OT landscape.")

                scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)
                scores_by_axis_ot["avg_cost"][axis_name].append(scores["avg_cost"])
                scores_by_axis_ot["full_uot"][axis_name].append(scores["full_uot"])
                records_ot.append({
                    "axis": axis_name,
                    "angle_deg": float(angle),
                    "avg_cost": scores["avg_cost"],
                    "full_uot": scores["full_uot"],
                    "T_sum": scores["T_sum"],
                })

    # Save CSV
    csv_ot_path = os.path.join(results_dir, "correspondence_ot_landscape.csv")
    with open(csv_ot_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "angle_deg", "avg_cost", "full_uot", "T_sum"])
        writer.writeheader()
        writer.writerows(records_ot)
    print(f"Saved OT landscape to {csv_ot_path}")

    # Plot
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for axis_name in axes:
        axes_plot[0].plot(angles, scores_by_axis_ot["avg_cost"][axis_name], label=axis_name)
        axes_plot[1].plot(angles, scores_by_axis_ot["full_uot"][axis_name], label=axis_name)
    for ax in axes_plot:
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Translation perturbation (deg)")
        ax.grid(True, alpha=0.3)
    axes_plot[0].set_title("OT avg_cost landscape")
    axes_plot[0].set_ylabel("avg_cost")
    axes_plot[1].set_title("OT full_uot landscape")
    axes_plot[1].set_ylabel("full_uot")
    axes_plot[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correspondence_ot_landscape.png"), dpi=150)
    plt.close()
    print(f"Saved OT plot to {results_dir}/correspondence_ot_landscape.png")

    # Print summary of minima
    print("\n" + "=" * 70)
    print("MINIMA SUMMARY (Step IX-OT)")
    print("=" * 70)
    for score_name in ["avg_cost", "full_uot"]:
        print(f"Score: {score_name}")
        for axis_name in axes:
            values = np.array(scores_by_axis_ot[score_name][axis_name])
            min_idx = int(np.argmin(values))
            min_angle = angles[min_idx]
            gt_idx = int(np.where(np.isclose(angles, 0.0))[0][0])
            gt_value = values[gt_idx]
            print(
                f"  axis={axis_name}: min at {min_angle:.1f} deg "
                f"(value={values[min_idx]:.6f}), "
                f"GT=0deg value={gt_value:.6f}"
            )

    # Key diagnostic: Is GT the minimum?
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Is GT the minimum for each axis? (Step IX-OT)")
    print("=" * 70)
    for score_name in ["avg_cost", "full_uot"]:
        all_gt_minimum = True
        print(f"Score: {score_name}")
        for axis_name in axes:
            values = np.array(scores_by_axis_ot[score_name][axis_name])
            min_idx = int(np.argmin(values))
            min_angle = angles[min_idx]
            is_gt_min = np.isclose(min_angle, 0.0, atol=angle_step / 2)
            status = "YES" if is_gt_min else "NO"
            print(f"  axis={axis_name}: GT is minimum? {status}")
            if not is_gt_min:
                all_gt_minimum = False
        if all_gt_minimum:
            print("  CONCLUSION: GT IS THE MINIMUM.")
        else:
            print("  CONCLUSION: GT IS NOT THE MINIMUM.")


if __name__ == "__main__":
    run_step_ix_correspondence_test()
