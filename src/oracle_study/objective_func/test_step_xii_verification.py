"""
Step XII: Comprehensive Verification of 2DGS Representation Problem

Purpose: Strengthen the evidence that the translation problem is a representation issue,
not an implementation bug.

Verification checks:
1. Transport visualization at GT vs off-minimum
2. 2DGS visual alignment verification
3. 2DGS-COLMAP track correspondence rate
4. full_uot term decomposition at GT vs off-minimum
"""

import os
import sys
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Tuple, Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
    read_images_with_points2d,
    get_corresponding_points,
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
    """Compute relative pose (world-to-camera) from camera 1 to camera 2."""
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


def verification_1_transport_visualization(
    idx1: int = 0,
    idx2: int = 10,
    output_dir: str = None,
):
    """
    Verification 1: Visualize transport at GT vs off-minimum.

    Compare what OT is doing at:
    - GT (0 deg)
    - full_uot minimum angles (x=-50, y=-10, z=-20 from Step XI)
    """
    print("=" * 70)
    print("Verification 1: Transport Visualization (GT vs Off-minimum)")
    print("=" * 70)

    if output_dir is None:
        output_dir = os.path.join(project_root, "results", "step_xii_verification")
    os.makedirs(output_dir, exist_ok=True)

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

    R_wc, t_wc_gt = compute_relative_pose_wc(cam1, cam2)

    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5

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

    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    # Test configurations: GT and off-minimum points from Step XI
    test_configs = [
        ("GT", 0.0, np.array([1, 0, 0])),  # GT
        ("x_min", -50.0, np.array([1, 0, 0])),  # x-axis minimum
        ("y_min", -10.0, np.array([0, 1, 0])),  # y-axis minimum
        ("z_min", -20.0, np.array([0, 0, 1])),  # z-axis minimum
    ]

    results = []

    with torch.no_grad():
        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)

        for label, angle, axis in test_configs:
            print(f"\nProcessing {label} (angle={angle} deg on {['x','y','z'][np.argmax(axis)]}-axis)...")

            R_delta = rodrigues_rotation(axis, np.radians(angle))
            t_wc = R_delta @ t_wc_gt
            t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)
            t_wc_t = torch.tensor(t_wc, dtype=torch.float32)

            F = solver._build_F_from_wc(R_wc_t, t_wc_t)
            cost = solver.compute_cost_matrix(F)
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix=cost,
                epsilon=epsilon,
                rho=rho,
                max_iter=120,
                gate_mask=solver._last_gate_mask,
            )

            eps_actual = solver._last_sinkhorn_epsilon
            rho_actual = solver._last_sinkhorn_rho

            # Compute scores
            scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)

            # Analyze transport
            T = transport.numpy()
            row_sum = T.sum(axis=1)
            col_sum = T.sum(axis=0)

            # Top-1 matches
            top1_idx = T.argmax(axis=1)
            top1_vals = T.max(axis=1)

            # Concentration (how peaked is the transport)
            concentration = (top1_vals / (row_sum + 1e-10)).mean()

            result = {
                'label': label,
                'angle': angle,
                'axis': ['x', 'y', 'z'][np.argmax(axis)],
                'full_uot': scores['full_uot'],
                'T_sum': scores['T_sum'],
                'transport_cost': scores['uot_cost_term'],
                'kl_term': scores['uot_kl_term'],
                'entropy_term': scores['uot_entropic_term'],
                'concentration': concentration,
                'row_sum_mean': row_sum.mean(),
                'row_sum_std': row_sum.std(),
                'col_sum_mean': col_sum.mean(),
                'col_sum_std': col_sum.std(),
                'transport': T,
                'top1_idx': top1_idx,
                'top1_vals': top1_vals,
            }
            results.append(result)

            print(f"  full_uot: {scores['full_uot']:.4f}")
            print(f"  T.sum: {scores['T_sum']:.4f}")
            print(f"  <T,C>: {scores['uot_cost_term']:.4f}")
            print(f"  rho*KL: {scores['uot_kl_term']:.4f}")
            print(f"  eps*Ent: {scores['uot_entropic_term']:.4f}")
            print(f"  concentration: {concentration:.4f}")

    # Visualization: Transport matrix comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        im = ax.imshow(r['transport'], aspect='auto', cmap='viridis')
        ax.set_title(f"{r['label']} ({r['axis']}={r['angle']:.0f}deg)\n"
                     f"full_uot={r['full_uot']:.4f}, T.sum={r['T_sum']:.3f}")
        ax.set_xlabel('Gaussian in image 2')
        ax.set_ylabel('Gaussian in image 1')
        plt.colorbar(im, ax=ax)

    plt.suptitle('Transport Matrix Comparison: GT vs Off-minimum', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transport_comparison.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(output_dir, 'transport_comparison.png')}")

    # Visualization: Top-1 correspondences on image
    means1 = g1.means
    means2 = g2.means

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax = axes[i]

        # Draw correspondences for top 50 by weight
        top_k = 50
        sorted_idx = np.argsort(-r['top1_vals'])[:top_k]

        for j in sorted_idx:
            src = means1[j]
            dst = means2[r['top1_idx'][j]]
            weight = r['top1_vals'][j]
            alpha = min(weight * 50, 1.0)  # Scale for visibility
            ax.plot([src[0], dst[0] + 1600], [src[1], dst[1]],
                   'b-', alpha=alpha, linewidth=0.5)

        # Draw Gaussian means
        ax.scatter(means1[:, 0], means1[:, 1], c='red', s=10, alpha=0.5, label='Image 1')
        ax.scatter(means2[:, 0] + 1600, means2[:, 1], c='green', s=10, alpha=0.5, label='Image 2')

        ax.set_title(f"{r['label']} ({r['axis']}={r['angle']:.0f}deg)\n"
                    f"Top-50 correspondences, conc={r['concentration']:.3f}")
        ax.set_xlim(0, 3200)
        ax.set_ylim(1200, 0)  # Flip y-axis for image coordinates
        ax.legend()

    plt.suptitle('Top-1 Correspondences: GT vs Off-minimum', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correspondences_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'correspondences_comparison.png')}")

    return results


def verification_2_visual_alignment(
    idx1: int = 0,
    idx2: int = 10,
    output_dir: str = None,
):
    """
    Verification 2: Visual verification that 2DGS is properly aligned with images.

    - Overlay Gaussian means on original images
    - Check if means are on meaningful locations
    """
    print("\n" + "=" * 70)
    print("Verification 2: 2DGS Visual Alignment")
    print("=" * 70)

    if output_dir is None:
        output_dir = os.path.join(project_root, "results", "step_xii_verification")
    os.makedirs(output_dir, exist_ok=True)

    # Load Gaussians (at fitted resolution 384x288)
    data1_fitted = load_gaussians(idx1, rescale_to_full=False)
    data2_fitted = load_gaussians(idx2, rescale_to_full=False)
    g1_fitted = data1_fitted['original_gaussians']
    g2_fitted = data2_fitted['original_gaussians']

    # Load Gaussians (rescaled to full resolution)
    data1_full = load_gaussians(idx1, rescale_to_full=True)
    data2_full = load_gaussians(idx2, rescale_to_full=True)
    g1_full = data1_full['original_gaussians']
    g2_full = data2_full['original_gaussians']

    # Try to load original images
    img_dir_resized = os.path.join(project_root, "data/DTU/scan63/images_resized")
    img_dir_full = os.path.join(project_root, "data/DTU/scan63/images")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, (idx, g_fitted, g_full) in enumerate([
        (idx1, g1_fitted, g1_full),
        (idx2, g2_fitted, g2_full)
    ]):
        # Top row: fitted resolution
        ax = axes[0, col]
        img_path = os.path.join(img_dir_resized, f"{idx:04d}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
        ax.scatter(g_fitted.means[:, 0], g_fitted.means[:, 1],
                  c='red', s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_title(f"Image {idx:04d} (384x288) - {len(g_fitted.means)} Gaussians")
        ax.set_xlim(0, 384)
        ax.set_ylim(288, 0)

        # Bottom row: full resolution with rescaled Gaussians
        ax = axes[1, col]
        img_path = os.path.join(img_dir_full, f"{idx:04d}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)
        ax.scatter(g_full.means[:, 0], g_full.means[:, 1],
                  c='red', s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.set_title(f"Image {idx:04d} (1554x1162 rescaled) - {len(g_full.means)} Gaussians")
        ax.set_xlim(0, 1554)
        ax.set_ylim(1162, 0)

    plt.suptitle('2DGS Means Overlay on Images', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gaussians_alignment.png"), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'gaussians_alignment.png')}")

    # Check spatial distribution statistics
    print("\nSpatial distribution statistics:")
    for idx, g in [(idx1, g1_full), (idx2, g2_full)]:
        means = g.means
        print(f"\n  Image {idx:04d}:")
        print(f"    X range: [{means[:, 0].min():.1f}, {means[:, 0].max():.1f}] (image: 0-1554)")
        print(f"    Y range: [{means[:, 1].min():.1f}, {means[:, 1].max():.1f}] (image: 0-1162)")
        print(f"    X mean: {means[:, 0].mean():.1f}, std: {means[:, 0].std():.1f}")
        print(f"    Y mean: {means[:, 1].mean():.1f}, std: {means[:, 1].std():.1f}")


def verification_3_correspondence_rate(
    idx1: int = 0,
    idx2: int = 10,
    radius_threshold: float = 30.0,  # pixels
    output_dir: str = None,
):
    """
    Verification 3: Quantify correspondence rate between 2DGS and COLMAP tracks.

    For each Gaussian, find nearest COLMAP keypoint and check if same track ID
    appears in both images.
    """
    print("\n" + "=" * 70)
    print("Verification 3: 2DGS-COLMAP Track Correspondence Rate")
    print("=" * 70)

    if output_dir is None:
        output_dir = os.path.join(project_root, "results", "step_xii_verification")
    os.makedirs(output_dir, exist_ok=True)

    # Load Gaussians (rescaled to full resolution)
    data1 = load_gaussians(idx1, rescale_to_full=True)
    data2 = load_gaussians(idx2, rescale_to_full=True)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']

    # Load COLMAP images with 2D points
    colmap_dir = os.path.join(project_root, "data/DTU/scan63/sparse/0")
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)

    # Get image data
    image_name1 = f"{idx1:04d}.png"
    image_name2 = f"{idx2:04d}.png"

    img1_data = None
    img2_data = None
    for img_data in images_with_pts.values():
        if img_data["name"] == image_name1:
            img1_data = img_data
        if img_data["name"] == image_name2:
            img2_data = img_data

    if img1_data is None or img2_data is None:
        print("ERROR: Could not find image data")
        return

    # Extract keypoints with valid 3D point IDs
    kpts1 = []
    track_ids1 = []
    for pt in img1_data["points2d"]:
        if pt["point3d_id"] != -1:
            kpts1.append([pt["x"], pt["y"]])
            track_ids1.append(pt["point3d_id"])
    kpts1 = np.array(kpts1)
    track_ids1 = np.array(track_ids1)

    kpts2 = []
    track_ids2 = []
    for pt in img2_data["points2d"]:
        if pt["point3d_id"] != -1:
            kpts2.append([pt["x"], pt["y"]])
            track_ids2.append(pt["point3d_id"])
    kpts2 = np.array(kpts2)
    track_ids2 = np.array(track_ids2)

    print(f"\nCOLMAP keypoints with valid 3D point:")
    print(f"  Image {idx1}: {len(kpts1)} keypoints")
    print(f"  Image {idx2}: {len(kpts2)} keypoints")

    # Find common track IDs
    common_track_ids = set(track_ids1) & set(track_ids2)
    print(f"  Common tracks: {len(common_track_ids)}")

    # For each Gaussian, find nearest keypoint
    def find_nearest_keypoint(means, kpts, track_ids, radius):
        """Find nearest keypoint for each Gaussian mean."""
        n_gaussians = len(means)
        n_kpts = len(kpts)

        # Compute distances
        dists = np.zeros((n_gaussians, n_kpts))
        for i in range(n_gaussians):
            dists[i] = np.sqrt(((kpts - means[i]) ** 2).sum(axis=1))

        # Find nearest and check threshold
        nearest_idx = dists.argmin(axis=1)
        nearest_dist = dists.min(axis=1)

        matched_gaussian_idx = []
        matched_track_ids = []
        for i in range(n_gaussians):
            if nearest_dist[i] < radius:
                matched_gaussian_idx.append(i)
                matched_track_ids.append(track_ids[nearest_idx[i]])

        return matched_gaussian_idx, matched_track_ids, nearest_dist

    matched1_idx, matched1_tracks, dists1 = find_nearest_keypoint(
        g1.means, kpts1, track_ids1, radius_threshold
    )
    matched2_idx, matched2_tracks, dists2 = find_nearest_keypoint(
        g2.means, kpts2, track_ids2, radius_threshold
    )

    print(f"\nGaussians matched to keypoints (radius < {radius_threshold}px):")
    print(f"  Image {idx1}: {len(matched1_idx)}/{len(g1.means)} ({100*len(matched1_idx)/len(g1.means):.1f}%)")
    print(f"  Image {idx2}: {len(matched2_idx)}/{len(g2.means)} ({100*len(matched2_idx)/len(g2.means):.1f}%)")

    # Find Gaussians that are matched to the SAME track in both images
    matched1_track_set = set(matched1_tracks)
    matched2_track_set = set(matched2_tracks)
    common_matched_tracks = matched1_track_set & matched2_track_set

    print(f"\nGaussians matched to COMMON tracks (same 3D point):")
    print(f"  Common matched tracks: {len(common_matched_tracks)}")

    # Count how many Gaussian pairs share the same track
    track_to_g1 = {}
    for i, track_id in zip(matched1_idx, matched1_tracks):
        if track_id not in track_to_g1:
            track_to_g1[track_id] = []
        track_to_g1[track_id].append(i)

    track_to_g2 = {}
    for i, track_id in zip(matched2_idx, matched2_tracks):
        if track_id not in track_to_g2:
            track_to_g2[track_id] = []
        track_to_g2[track_id].append(i)

    gaussian_pairs_with_correspondence = 0
    for track_id in common_matched_tracks:
        n1 = len(track_to_g1.get(track_id, []))
        n2 = len(track_to_g2.get(track_id, []))
        gaussian_pairs_with_correspondence += n1 * n2

    total_possible_pairs = len(g1.means) * len(g2.means)
    print(f"  Gaussian pairs with track correspondence: {gaussian_pairs_with_correspondence}")
    print(f"  Total possible pairs: {total_possible_pairs}")
    print(f"  Correspondence rate: {100*gaussian_pairs_with_correspondence/total_possible_pairs:.4f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, (idx, g, kpts, matched_idx, dists) in enumerate([
        (idx1, g1, kpts1, matched1_idx, dists1),
        (idx2, g2, kpts2, matched2_idx, dists2)
    ]):
        ax = axes[col]

        # Load image
        img_path = os.path.join(project_root, "data/DTU/scan63/images", f"{idx:04d}.png")
        if os.path.exists(img_path):
            img = plt.imread(img_path)
            ax.imshow(img)

        # Draw keypoints
        ax.scatter(kpts[:, 0], kpts[:, 1], c='cyan', s=10, alpha=0.5, marker='x', label='COLMAP kpts')

        # Draw Gaussians (colored by whether matched)
        matched_set = set(matched_idx)
        unmatched = [i for i in range(len(g.means)) if i not in matched_set]

        ax.scatter(g.means[unmatched, 0], g.means[unmatched, 1],
                  c='red', s=20, alpha=0.5, label=f'Unmatched ({len(unmatched)})')
        if matched_idx:
            ax.scatter(g.means[matched_idx, 0], g.means[matched_idx, 1],
                      c='green', s=30, alpha=0.7, label=f'Matched ({len(matched_idx)})')

        ax.set_title(f"Image {idx:04d}\n"
                    f"Matched: {len(matched_idx)}/{len(g.means)} ({100*len(matched_idx)/len(g.means):.1f}%)")
        ax.set_xlim(0, 1554)
        ax.set_ylim(1162, 0)
        ax.legend()

    plt.suptitle(f'2DGS-COLMAP Track Correspondence (radius={radius_threshold}px)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correspondence_rate.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(output_dir, 'correspondence_rate.png')}")

    return {
        'n_gaussians1': len(g1.means),
        'n_gaussians2': len(g2.means),
        'n_kpts1': len(kpts1),
        'n_kpts2': len(kpts2),
        'n_common_tracks': len(common_track_ids),
        'n_matched1': len(matched1_idx),
        'n_matched2': len(matched2_idx),
        'n_common_matched_tracks': len(common_matched_tracks),
        'gaussian_pairs_with_correspondence': gaussian_pairs_with_correspondence,
        'total_possible_pairs': total_possible_pairs,
    }


def verification_4_full_uot_decomposition(
    idx1: int = 0,
    idx2: int = 10,
    output_dir: str = None,
):
    """
    Verification 4: Decompose full_uot terms at GT vs off-minimum.

    Compare <T,C>, rho*KL, eps*(T log T - T) terms to understand what makes
    off-minimum "win" over GT.
    """
    print("\n" + "=" * 70)
    print("Verification 4: full_uot Term Decomposition")
    print("=" * 70)

    if output_dir is None:
        output_dir = os.path.join(project_root, "results", "step_xii_verification")
    os.makedirs(output_dir, exist_ok=True)

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

    R_wc, t_wc_gt = compute_relative_pose_wc(cam1, cam2)

    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5

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

    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    # Sweep all three axes
    axes_config = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }

    angles = np.arange(-60, 61, 10)

    results = {axis: [] for axis in axes_config}

    with torch.no_grad():
        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)

        for axis_name, axis_vec in axes_config.items():
            print(f"\nSweeping {axis_name}-axis...")
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
                    max_iter=120,
                    gate_mask=solver._last_gate_mask,
                )

                eps_actual = solver._last_sinkhorn_epsilon
                rho_actual = solver._last_sinkhorn_rho

                scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)

                results[axis_name].append({
                    'angle': angle,
                    'full_uot': scores['full_uot'],
                    'T_sum': scores['T_sum'],
                    'transport_cost': scores['uot_cost_term'],
                    'kl_term': scores['uot_kl_term'],
                    'entropy_term': scores['uot_entropic_term'],
                })

    # Visualization: Decomposition for each axis
    fig, axes_plot = plt.subplots(3, 4, figsize=(20, 12))

    for row, axis_name in enumerate(['x', 'y', 'z']):
        data = results[axis_name]
        angles_arr = np.array([d['angle'] for d in data])

        # Find minimum
        full_uot_vals = np.array([d['full_uot'] for d in data])
        min_idx = np.argmin(full_uot_vals)
        gt_idx = np.where(angles_arr == 0)[0][0]

        # Plot full_uot
        ax = axes_plot[row, 0]
        ax.plot(angles_arr, full_uot_vals, 'b-o', linewidth=2)
        ax.axvline(0, color='green', linestyle='--', label='GT')
        ax.scatter([angles_arr[min_idx]], [full_uot_vals[min_idx]], color='red', s=150,
                  zorder=5, marker='*', label=f'Min @ {angles_arr[min_idx]}°')
        ax.set_title(f'{axis_name.upper()}: full_uot')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('full_uot')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot <T,C>
        ax = axes_plot[row, 1]
        transport_cost = np.array([d['transport_cost'] for d in data])
        ax.plot(angles_arr, transport_cost, 'r-o', linewidth=2)
        ax.axvline(0, color='green', linestyle='--')
        ax.scatter([angles_arr[min_idx]], [transport_cost[min_idx]], color='red', s=100, marker='*')
        ax.set_title(f'{axis_name.upper()}: <T,C> (transport cost)')
        ax.set_xlabel('Angle (deg)')
        ax.grid(True, alpha=0.3)

        # Plot rho*KL
        ax = axes_plot[row, 2]
        kl_term = np.array([d['kl_term'] for d in data])
        ax.plot(angles_arr, kl_term, 'g-o', linewidth=2)
        ax.axvline(0, color='green', linestyle='--')
        ax.scatter([angles_arr[min_idx]], [kl_term[min_idx]], color='red', s=100, marker='*')
        ax.set_title(f'{axis_name.upper()}: rho*KL')
        ax.set_xlabel('Angle (deg)')
        ax.grid(True, alpha=0.3)

        # Plot T.sum
        ax = axes_plot[row, 3]
        t_sum = np.array([d['T_sum'] for d in data])
        ax.plot(angles_arr, t_sum, 'm-o', linewidth=2)
        ax.axvline(0, color='green', linestyle='--')
        ax.scatter([angles_arr[min_idx]], [t_sum[min_idx]], color='red', s=100, marker='*')
        ax.set_title(f'{axis_name.upper()}: T.sum')
        ax.set_xlabel('Angle (deg)')
        ax.grid(True, alpha=0.3)

    plt.suptitle('full_uot Term Decomposition: GT vs Off-minimum', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_uot_decomposition.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(output_dir, 'full_uot_decomposition.png')}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("Comparison: GT (0deg) vs Off-minimum")
    print("=" * 80)
    print(f"{'Axis':<6} {'Point':<12} {'full_uot':>10} {'<T,C>':>10} {'rho*KL':>10} {'T.sum':>8}")
    print("-" * 80)

    for axis_name in ['x', 'y', 'z']:
        data = results[axis_name]
        angles_arr = np.array([d['angle'] for d in data])
        full_uot_vals = np.array([d['full_uot'] for d in data])

        min_idx = np.argmin(full_uot_vals)
        gt_idx = np.where(angles_arr == 0)[0][0]

        gt = data[gt_idx]
        mn = data[min_idx]

        print(f"{axis_name:<6} {'GT (0deg)':<12} {gt['full_uot']:>10.4f} {gt['transport_cost']:>10.4f} "
              f"{gt['kl_term']:>10.4f} {gt['T_sum']:>8.4f}")
        print(f"{'':<6} {f'Min ({angles_arr[min_idx]:.0f}deg)':<12} {mn['full_uot']:>10.4f} {mn['transport_cost']:>10.4f} "
              f"{mn['kl_term']:>10.4f} {mn['T_sum']:>8.4f}")
        print(f"{'':<6} {'Diff':<12} {mn['full_uot']-gt['full_uot']:>10.4f} {mn['transport_cost']-gt['transport_cost']:>10.4f} "
              f"{mn['kl_term']-gt['kl_term']:>10.4f} {mn['T_sum']-gt['T_sum']:>8.4f}")
        print()

    return results


def run_all_verifications():
    """Run all verification experiments."""
    print("=" * 70)
    print("Step XII: Comprehensive Verification Experiments")
    print("=" * 70)

    output_dir = os.path.join(project_root, "results", "step_xii_verification")
    os.makedirs(output_dir, exist_ok=True)

    # Run all verifications
    results = {}

    print("\n" + "=" * 70)
    print("Running Verification 1: Transport Visualization")
    print("=" * 70)
    results['transport'] = verification_1_transport_visualization(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("Running Verification 2: Visual Alignment")
    print("=" * 70)
    verification_2_visual_alignment(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("Running Verification 3: Correspondence Rate")
    print("=" * 70)
    results['correspondence'] = verification_3_correspondence_rate(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("Running Verification 4: full_uot Decomposition")
    print("=" * 70)
    results['decomposition'] = verification_4_full_uot_decomposition(output_dir=output_dir)

    print("\n" + "=" * 70)
    print("All verifications complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_all_verifications()
