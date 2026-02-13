#!/usr/bin/env python3
"""
Step E: Track-based correspondence audit for OT transport.

This script measures how much OT mass is assigned to "true" Gaussian pairs
that correspond to the same COLMAP 3D track.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
    read_images_with_points2d,
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
    compute_closed_form_translation,
    compute_gap_metrics,
    compute_topk_weight_sum,
    compute_transport_concentration,
)


def load_colmap_cameras(scan_name: str = "scan63"):
    colmap_dir = os.path.join(PROJECT_ROOT, f"data/DTU/{scan_name}/sparse/0")
    cameras = load_cameras_from_colmap(colmap_dir)
    images = load_images_from_colmap(colmap_dir)
    return cameras, images


def get_colmap_camera_params(cameras, images, image_name: str) -> dict:
    image_data = None
    for _, img in images.items():
        if img["name"] == image_name:
            image_data = img
            break
    if image_data is None:
        raise ValueError(f"Image {image_name} not found in COLMAP data")
    camera = cameras[image_data["camera_id"]]
    K = camera.get_camera_matrix()
    R = quaternion_to_rotation_matrix(
        image_data["qw"], image_data["qx"],
        image_data["qy"], image_data["qz"],
    )
    t = np.array([image_data["tx"], image_data["ty"], image_data["tz"]])
    return {"K": K, "R": R, "t": t, "name": image_name}


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    R1, t1 = cam1["R"], cam1["t"]
    R2, t2 = cam2["R"], cam2["t"]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-10)
    return R_rel, t_rel


def load_keypoints_and_tracks(colmap_dir: str, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)

    img_data = None
    for img in images_with_pts.values():
        if img["name"] == image_name:
            img_data = img
            break
    if img_data is None:
        raise ValueError(f"Image {image_name} not found in images.bin")

    kpts = []
    track_ids = []
    for pt in img_data["points2d"]:
        if pt["point3d_id"] != -1:
            kpts.append([pt["x"], pt["y"]])
            track_ids.append(pt["point3d_id"])
    if len(kpts) == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)

    return np.array(kpts), np.array(track_ids)


def match_gaussians_to_tracks(
    means: np.ndarray,
    kpts: np.ndarray,
    track_ids: np.ndarray,
    radius: float,
) -> Dict[int, Tuple[int, float]]:
    """Assign each track ID the closest Gaussian (within radius)."""
    if kpts.shape[0] == 0:
        return {}

    diffs = means[:, None, :] - kpts[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = np.min(dists, axis=1)

    track_map: Dict[int, Tuple[int, float]] = {}
    for gi, (kp_idx, dist) in enumerate(zip(nearest_idx, nearest_dist)):
        if dist > radius:
            continue
        track_id = int(track_ids[kp_idx])
        if track_id not in track_map or dist < track_map[track_id][1]:
            track_map[track_id] = (gi, float(dist))

    return track_map


def match_tracks_to_gaussians(
    means: np.ndarray,
    kpts: np.ndarray,
    track_ids: np.ndarray,
    radius: float,
) -> Dict[int, Tuple[int, float]]:
    """Assign each track ID the closest Gaussian (within radius)."""
    if kpts.shape[0] == 0:
        return {}
    diffs = kpts[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = np.min(dists, axis=1)
    track_map: Dict[int, Tuple[int, float]] = {}
    for kp_idx, (gi, dist) in enumerate(zip(nearest_idx, nearest_dist)):
        if dist > radius:
            continue
        track_id = int(track_ids[kp_idx])
        track_map[track_id] = (int(gi), float(dist))
    return track_map


def compute_coverage_ratio(
    means: np.ndarray,
    kpts: np.ndarray,
    radius: float,
) -> Tuple[int, float]:
    if kpts.shape[0] == 0:
        return 0, 0.0
    diffs = kpts[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    min_dist = np.min(dists, axis=1)
    covered = int(np.sum(min_dist <= radius))
    return covered, float(covered / max(1, kpts.shape[0]))


def build_sequential_gate_mask(
    costs: List[Tuple[torch.Tensor, int]],
) -> torch.Tensor:
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


def compute_descriptor_cost_matrix(
    means1: np.ndarray,
    scales1: Optional[np.ndarray],
    means2: np.ndarray,
    scales2: Optional[np.ndarray],
    k: int,
) -> torch.Tensor:
    desc1 = compute_knn_descriptor(means1, scales1, k)
    desc2 = compute_knn_descriptor(means2, scales2, k)
    diff = desc1[:, None, :] - desc2[None, :, :]
    cost = np.sqrt(np.sum(diff * diff, axis=2))
    return torch.tensor(cost, dtype=torch.float32)


def compute_rank_in_row(row: np.ndarray, value: float) -> int:
    """Rank of value in row (1 = largest)."""
    if row.size == 0:
        return row.size
    return int(np.sum(row > value) + 1)


def run_track_audit(
    idx1: int,
    idx2: int,
    radius: float = 30.0,
    epsilon: float = 0.05,
    rho: float = 0.5,
    topk_list: List[int] = None,
    gate_top_m_epi: int = 0,
    gate_top_m_cov: int = 0,
    gate_top_m_desc: int = 0,
    desc_k: int = 8,
    mapping_mode: str = "gaussian_to_track",
    coverage: bool = False,
    extra_metrics: bool = False,
    gap_metric: str = "gap_norm",
    gap_topk: int = 50,
    gap_matching: str = "global_topk",
    gap_row_topk: Optional[int] = None,
    gap_grid: Optional[Tuple[int, int]] = None,
    gap_max_per_cell: int = 2,
) -> dict:
    if topk_list is None:
        topk_list = [1, 5, 10, 20, 50]

    print("\n" + "=" * 70)
    print("Step E: OT track correspondence audit")
    print(f"  Pair: ({idx1}, {idx2}), radius={radius}px")
    print(f"  epsilon={epsilon}, rho={rho}")
    if gate_top_m_epi > 0 and gate_top_m_cov > 0:
        print(f"  gate: epi_top={gate_top_m_epi}, cov_top={gate_top_m_cov}")
    if gate_top_m_desc > 0:
        print(f"  desc gate: top={gate_top_m_desc}, k={desc_k}")
    print("=" * 70)

    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1["original_gaussians"]
    g2 = data2["original_gaussians"]
    ot_mass1 = data1.get("ot_mass", None)
    ot_mass2 = data2.get("ot_mass", None)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1["K"]

    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")
    kpts1, track_ids1 = load_keypoints_and_tracks(colmap_dir, f"{idx1:04d}.png")
    kpts2, track_ids2 = load_keypoints_and_tracks(colmap_dir, f"{idx2:04d}.png")

    means1 = g1.means.numpy() if isinstance(g1.means, torch.Tensor) else g1.means
    means2 = g2.means.numpy() if isinstance(g2.means, torch.Tensor) else g2.means
    scales1 = g1.scales.numpy() if isinstance(g1.scales, torch.Tensor) else g1.scales
    scales2 = g2.scales.numpy() if isinstance(g2.scales, torch.Tensor) else g2.scales

    if coverage:
        covered1, ratio1 = compute_coverage_ratio(means1, kpts1, radius)
        covered2, ratio2 = compute_coverage_ratio(means2, kpts2, radius)
        print(f"Coverage: img1 {covered1}/{kpts1.shape[0]} ({ratio1:.3f}), "
              f"img2 {covered2}/{kpts2.shape[0]} ({ratio2:.3f})")

    if mapping_mode == "track_to_gaussian":
        track_map1 = match_tracks_to_gaussians(means1, kpts1, track_ids1, radius)
        track_map2 = match_tracks_to_gaussians(means2, kpts2, track_ids2, radius)
    else:
        track_map1 = match_gaussians_to_tracks(means1, kpts1, track_ids1, radius)
        track_map2 = match_gaussians_to_tracks(means2, kpts2, track_ids2, radius)

    common_tracks = sorted(set(track_map1.keys()) & set(track_map2.keys()))
    print(f"Common tracks with Gaussian matches: {len(common_tracks)}")

    if len(common_tracks) == 0:
        print("No common tracks found. Skipping.")
        return {}

    true_pairs = []
    for track_id in common_tracks:
        i = track_map1[track_id][0]
        j = track_map2[track_id][0]
        true_pairs.append((i, j, track_id))

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    gate_mask = None
    gate_desc_cost = None
    if gate_top_m_desc > 0:
        gate_desc_cost = compute_descriptor_cost_matrix(means1, scales1, means2, scales2, desc_k)

    if gate_top_m_epi > 0 and gate_top_m_cov > 0:
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

    R_t = torch.tensor(R_gt, dtype=torch.float32)
    t_t = torch.tensor(t_gt, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    F = solver._build_F_from_wc(R_t, t_t)
    cost = solver.compute_cost_matrix(F)
    if gate_top_m_epi > 0 and gate_top_m_cov > 0:
        epi_cost = solver_epi.compute_cost_matrix(F)
        cov_cost = solver_cov.compute_cost_matrix(F)
        gate_steps: List[Tuple[torch.Tensor, int]] = []
        if gate_top_m_epi > 0:
            gate_steps.append((epi_cost, gate_top_m_epi))
        if gate_top_m_cov > 0:
            gate_steps.append((cov_cost, gate_top_m_cov))
        if gate_top_m_desc > 0 and gate_desc_cost is not None:
            gate_steps.append((gate_desc_cost, gate_top_m_desc))
        gate_mask = build_sequential_gate_mask(gate_steps)
        row_all_false = (gate_mask.sum(dim=1) == 0).sum().item()
        col_all_false = (gate_mask.sum(dim=0) == 0).sum().item()
        cost, gate_mask = solver._apply_gate_mask(cost, gate_mask)
        valid_ratio = float(gate_mask.float().mean().item())
        print(f"  gate valid ratio: {valid_ratio:.4f} (row_fallback={row_all_false}, col_fallback={col_all_false})")

    with torch.no_grad():
        transport, _ = solver.unbalanced_sinkhorn_algorithm(
            cost, epsilon=epsilon, rho=rho,
            gate_mask=gate_mask if gate_mask is not None else solver._last_gate_mask,
        )

    T = transport.detach().cpu().numpy()
    total_mass = float(T.sum())
    if total_mass <= 0.0:
        print("Transport mass is zero. Collapse.")
        return {}

    true_masses = []
    ranks = []
    for i, j, _ in true_pairs:
        row = T[i]
        value = float(T[i, j])
        true_masses.append(value)
        ranks.append(compute_rank_in_row(row, value))

    true_masses = np.array(true_masses)
    ranks = np.array(ranks)
    total_true_mass = float(true_masses.sum())

    print(f"True pairs: {len(true_pairs)}")
    print(f"Total OT mass: {total_mass:.4f}")
    print(f"True-pair mass: {total_true_mass:.6f} ({100*total_true_mass/total_mass:.4f}%)")
    print(f"Rank stats: mean={ranks.mean():.1f}, median={np.median(ranks):.1f}")

    recall_at_k = {}
    mass_recall_at_k = {}
    for k in topk_list:
        mask = ranks <= k
        recall_at_k[k] = float(mask.mean())
        mass_recall_at_k[k] = float(true_masses[mask].sum() / (total_true_mass + 1e-10))

    print("\nRecall@k (rank within row):")
    for k in topk_list:
        print(f"  k={k:>3}: recall={recall_at_k[k]:.3f}, mass_recall={mass_recall_at_k[k]:.3f}")

    if extra_metrics:
        conc = compute_transport_concentration(torch.tensor(T))
        topk_sum = compute_topk_weight_sum(torch.tensor(T), gap_topk)
        gap_val = 0.0
        if total_mass > 0.0:
            result = compute_closed_form_translation(
                solver,
                R_t,
                transport,
                use_hard_assignment=True,
                top_k=gap_topk,
                min_transport_mass=0.05,
                min_topk_weight=0.0,
                matching_mode=gap_matching,
                row_top_k=gap_row_topk,
                diversity_grid=gap_grid,
                max_per_cell=gap_max_per_cell,
            )
            if result[1] is not None:
                gap_metrics = compute_gap_metrics(result[1])
                gap_val = gap_metrics.get(gap_metric, gap_metrics["gap_norm"])
        print(f"\nExtra metrics: conc={conc:.4f}, topk_sum={topk_sum:.4f}, {gap_metric}={gap_val:.5f}")

    return {
        "idx1": idx1,
        "idx2": idx2,
        "n_true_pairs": len(true_pairs),
        "total_mass": total_mass,
        "true_mass": total_true_mass,
        "recall_at_k": recall_at_k,
        "mass_recall_at_k": mass_recall_at_k,
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "gate_top_m_epi": gate_top_m_epi,
        "gate_top_m_cov": gate_top_m_cov,
        "gate_top_m_desc": gate_top_m_desc,
    }


def parse_pairs(pairs_str: str) -> List[Tuple[int, int]]:
    pairs = []
    for part in pairs_str.split(";"):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Track-based OT correspondence audit")
    parser.add_argument("--pairs", type=str, default="0,10",
                        help="Pairs as 'i,j;k,l' (default: 0,10)")
    parser.add_argument("--radius", type=float, default=30.0, help="Track match radius in pixels")
    parser.add_argument("--radius-list", type=str, default="",
                        help="Comma-separated radius list (overrides --radius)")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon")
    parser.add_argument("--rho", type=float, default=0.5, help="Sinkhorn rho")
    parser.add_argument("--topk", type=str, default="1,5,10,20,50",
                        help="Comma-separated k list for recall")
    parser.add_argument("--gate-epi", type=int, default=0, help="Top-M epipolar candidates")
    parser.add_argument("--gate-cov", type=int, default=0, help="Top-M cov candidates")
    parser.add_argument("--gate-desc", type=int, default=0, help="Top-M descriptor candidates")
    parser.add_argument("--desc-k", type=int, default=8, help="k for descriptor")
    parser.add_argument("--gate-epi-list", type=str, default="",
                        help="Comma-separated epi top-M list")
    parser.add_argument("--gate-cov-list", type=str, default="",
                        help="Comma-separated cov top-M list")
    parser.add_argument("--mapping-mode", type=str, default="gaussian_to_track",
                        help="Mapping: gaussian_to_track or track_to_gaussian")
    parser.add_argument("--coverage", action="store_true", help="Report coverage ratio")
    parser.add_argument("--extra-metrics", action="store_true", help="Report conc/topk_sum/gap")
    parser.add_argument("--gap-metric", type=str, default="gap_norm", help="Gap metric name")
    parser.add_argument("--gap-topk", type=int, default=50, help="Top-K for gap eval")
    parser.add_argument("--gap-matching", type=str, default="global_topk", help="Matching mode")
    parser.add_argument("--gap-row-topk", type=int, default=0, help="Row top-k for matching")
    parser.add_argument("--gap-grid", type=str, default="", help="Grid for diversity, e.g. 4x4")
    parser.add_argument("--gap-max-per-cell", type=int, default=2, help="Max per grid cell")
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    if args.radius_list:
        radii = [float(x) for x in args.radius_list.split(",") if x.strip()]
    else:
        radii = [args.radius]

    if args.gate_epi_list and args.gate_cov_list:
        gate_epi_list = [int(x) for x in args.gate_epi_list.split(",") if x.strip()]
        gate_cov_list = [int(x) for x in args.gate_cov_list.split(",") if x.strip()]
    else:
        gate_epi_list = [args.gate_epi]
        gate_cov_list = [args.gate_cov]

    row_top_k = args.gap_row_topk if args.gap_row_topk > 0 else None
    gap_grid = None
    if args.gap_grid:
        parts = args.gap_grid.lower().split("x") if "x" in args.gap_grid else args.gap_grid.split(",")
        if len(parts) == 2:
            gap_grid = (int(parts[0]), int(parts[1]))

    for idx1, idx2 in pairs:
        for radius in radii:
            for gate_epi in gate_epi_list:
                for gate_cov in gate_cov_list:
                    run_track_audit(
                        idx1,
                        idx2,
                        radius,
                        args.epsilon,
                        args.rho,
                        topk_list,
                        gate_top_m_epi=gate_epi,
                        gate_top_m_cov=gate_cov,
                        gate_top_m_desc=args.gate_desc,
                        desc_k=args.desc_k,
                        mapping_mode=args.mapping_mode,
                        coverage=args.coverage,
                        extra_metrics=args.extra_metrics,
                        gap_metric=args.gap_metric,
                        gap_topk=args.gap_topk,
                        gap_matching=args.gap_matching,
                        gap_row_topk=row_top_k,
                        gap_grid=gap_grid,
                        gap_max_per_cell=args.gap_max_per_cell,
                    )


if __name__ == "__main__":
    main()
