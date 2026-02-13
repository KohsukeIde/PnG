#!/usr/bin/env python3
"""
Step U: Oracle correspondence vs OT correspondence for closed-form t estimation.

Purpose:
  - Disentangle whether closed-form t fails due to formula/normalization
    or due to poor correspondence quality from OT.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

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
    compute_descriptor_cost_matrix,
    compute_gap_metrics,
    compute_transport_concentration,
    compute_transport_with_gate,
    rodrigues_rotation,
    rotation_error,
    translation_error,
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


def match_tracks_to_gaussians(
    means: np.ndarray,
    kpts: np.ndarray,
    track_ids: np.ndarray,
    radius: float,
) -> Dict[int, Tuple[int, float]]:
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


def build_oracle_transport(
    means1: np.ndarray,
    means2: np.ndarray,
    kpts1: np.ndarray,
    track_ids1: np.ndarray,
    kpts2: np.ndarray,
    track_ids2: np.ndarray,
    radius: float,
    weight_mode: str = "uniform",
    dedup_mode: str = "sum",
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    track_map1 = match_tracks_to_gaussians(means1, kpts1, track_ids1, radius)
    track_map2 = match_tracks_to_gaussians(means2, kpts2, track_ids2, radius)
    common_tracks = sorted(set(track_map1.keys()) & set(track_map2.keys()))
    pairs_info: List[Tuple[int, int, float]] = []
    for tid in common_tracks:
        i, d1 = track_map1[tid]
        j, d2 = track_map2[tid]
        pairs_info.append((i, j, float(d1 + d2)))
    pairs: List[Tuple[int, int]] = [(i, j) for i, j, _ in pairs_info]
    k1 = means1.shape[0]
    k2 = means2.shape[0]
    T = torch.zeros((k1, k2), dtype=torch.float32)
    if not pairs:
        return T, pairs
    # Optionally enforce one-to-one by greedy assignment on track->Gaussian distances
    if dedup_mode == "one_to_one":
        scored = sorted(pairs_info, key=lambda x: x[2])
        used_i = set()
        used_j = set()
        pairs = []
        for i, j, _ in scored:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            pairs.append((i, j))
    if weight_mode == "uniform":
        w = 1.0
        for i, j in pairs:
            if dedup_mode == "sum":
                T[i, j] += w
            else:
                T[i, j] = w
    else:
        # inverse distance weight
        for i, j in pairs:
            d = np.linalg.norm(means1[i] - means2[j])
            w = 1.0 / (d + 1e-6)
            if dedup_mode == "sum":
                T[i, j] += w
            else:
                T[i, j] = w
    # Normalize
    T = T / (T.sum() + 1e-12)
    return T, pairs


def build_keypoint_pairs(
    kpts1: np.ndarray,
    track_ids1: np.ndarray,
    kpts2: np.ndarray,
    track_ids2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired keypoints (x1, x2) for shared COLMAP tracks."""
    if kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    map1 = {int(tid): i for i, tid in enumerate(track_ids1)}
    map2 = {int(tid): i for i, tid in enumerate(track_ids2)}
    common = sorted(set(map1.keys()) & set(map2.keys()))
    if not common:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pts1 = np.stack([kpts1[map1[tid]] for tid in common], axis=0)
    pts2 = np.stack([kpts2[map2[tid]] for tid in common], axis=0)
    return pts1, pts2


def build_soft_oracle_transport(
    means1: np.ndarray,
    means2: np.ndarray,
    kpts1: np.ndarray,
    track_ids1: np.ndarray,
    kpts2: np.ndarray,
    track_ids2: np.ndarray,
    radius: float,
    top_k: int,
    sigma: float,
) -> Tuple[torch.Tensor, int]:
    """Soft oracle: each keypoint spreads mass to top-k nearby gaussians."""
    if kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return torch.zeros((means1.shape[0], means2.shape[0]), dtype=torch.float32), 0
    map1 = {int(tid): i for i, tid in enumerate(track_ids1)}
    map2 = {int(tid): i for i, tid in enumerate(track_ids2)}
    common = sorted(set(map1.keys()) & set(map2.keys()))
    if not common:
        return torch.zeros((means1.shape[0], means2.shape[0]), dtype=torch.float32), 0

    T = torch.zeros((means1.shape[0], means2.shape[0]), dtype=torch.float32)
    used = 0
    sigma_sq = max(sigma, 1e-6) ** 2

    for tid in common:
        p1 = kpts1[map1[tid]]
        p2 = kpts2[map2[tid]]
        d1 = np.linalg.norm(means1 - p1[None, :], axis=1)
        d2 = np.linalg.norm(means2 - p2[None, :], axis=1)

        idx1 = np.argsort(d1)[: max(1, min(top_k, len(d1)))]
        idx2 = np.argsort(d2)[: max(1, min(top_k, len(d2)))]

        if radius > 0:
            idx1 = idx1[d1[idx1] <= radius]
            idx2 = idx2[d2[idx2] <= radius]
        if idx1.size == 0 or idx2.size == 0:
            continue

        w1 = np.exp(-0.5 * (d1[idx1] ** 2) / sigma_sq)
        w2 = np.exp(-0.5 * (d2[idx2] ** 2) / sigma_sq)
        w1 = w1 / (w1.sum() + 1e-12)
        w2 = w2 / (w2.sum() + 1e-12)

        T[np.ix_(idx1, idx2)] += torch.tensor(np.outer(w1, w2), dtype=torch.float32)
        used += 1

    if T.sum() > 0:
        T = T / (T.sum() + 1e-12)
    return T, used


def compute_t_from_keypoints(
    K1: np.ndarray,
    K2: np.ndarray,
    R_wc: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Closed-form t from keypoint correspondences using M = sum a a^T."""
    if pts1.shape[0] < 5:
        return None, None
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    ones = np.ones((pts1.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])
    x1n = (K1_inv @ x1.T).T
    x2n = (K2_inv @ x2.T).T
    Rx1 = (R_wc @ x1n.T).T
    a = np.cross(Rx1, x2n)
    M = a.T @ a
    if not np.isfinite(M).all():
        return None, None
    if np.linalg.norm(M) < 1e-12:
        return None, None
    eigvals, eigvecs = np.linalg.eigh(M)
    t_opt = eigvecs[:, 0]
    return t_opt, eigvals


def main() -> None:
    parser = argparse.ArgumentParser(description="Step U: oracle vs OT t estimation")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--mapping-mode", type=str, default="track_to_gaussian")
    parser.add_argument("--weight-mode", type=str, default="uniform", choices=["uniform", "inv_dist"])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--gate-epi", type=int, default=0)
    parser.add_argument("--gate-cov", type=int, default=0)
    parser.add_argument("--gate-desc", type=int, default=0)
    parser.add_argument("--desc-k", type=int, default=8)
    parser.add_argument("--ot-use-hard", action="store_true")
    parser.add_argument("--ot-top-k", type=int, default=50)
    parser.add_argument("--r-perturb", type=float, default=0.0)
    parser.add_argument("--use-keypoint", action="store_true")
    parser.add_argument("--oracle-dedup", type=str, default="sum",
                        choices=["sum", "one_to_one", "overwrite"])
    parser.add_argument("--oracle-soft", action="store_true")
    parser.add_argument("--oracle-soft-k", type=int, default=5)
    parser.add_argument("--oracle-soft-sigma", type=float, default=10.0)
    args = parser.parse_args()

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{args.idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{args.idx2:04d}.png")
    K = cam1["K"]
    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

    if args.r_perturb > 0:
        R_delta = rodrigues_rotation(np.array([0.0, 1.0, 0.0]), np.deg2rad(args.r_perturb))
        R_use = R_delta @ R_gt
    else:
        R_use = R_gt

    data1 = load_gaussians(args.idx1)
    data2 = load_gaussians(args.idx2)
    g1 = data1["original_gaussians"]
    g2 = data2["original_gaussians"]
    ot_mass1 = data1.get("ot_mass", None)
    ot_mass2 = data2.get("ot_mass", None)

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    gate_solvers = None
    if args.gate_epi > 0 or args.gate_cov > 0:
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

    gate_desc_cost = None
    if args.gate_desc > 0:
        gate_desc_cost = compute_descriptor_cost_matrix(g1, g2, args.desc_k)

    # OT-based transport
    R_t = torch.tensor(R_use, dtype=torch.float32)
    t_t = torch.tensor(t_gt, dtype=torch.float32)
    cost_matrix, transport, _ = compute_transport_with_gate(
        solver,
        R_t,
        t_t,
        args.epsilon,
        args.rho,
        gate_solvers=gate_solvers,
        gate_top_m_epi=args.gate_epi,
        gate_top_m_cov=args.gate_cov,
        gate_top_m_desc=args.gate_desc,
        gate_desc_cost=gate_desc_cost,
    )
    conc = compute_transport_concentration(transport)
    t_ot, eig_ot = compute_closed_form_translation(
        solver,
        R_t,
        transport,
        use_hard_assignment=args.ot_use_hard,
        top_k=args.ot_top_k,
        min_transport_mass=0.05,
    )
    if t_ot is None:
        ot_err = float("nan")
        ot_gap = {}
    else:
        ot_err = min(
            translation_error(t_ot, t_gt),
            translation_error(-t_ot, t_gt),
        )
        ot_gap = compute_gap_metrics(eig_ot)

    # Oracle transport from tracks
    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")
    kpts1, track_ids1 = load_keypoints_and_tracks(colmap_dir, f"{args.idx1:04d}.png")
    kpts2, track_ids2 = load_keypoints_and_tracks(colmap_dir, f"{args.idx2:04d}.png")

    means1 = g1.means.numpy() if isinstance(g1.means, torch.Tensor) else g1.means
    means2 = g2.means.numpy() if isinstance(g2.means, torch.Tensor) else g2.means
    T_oracle, pairs = build_oracle_transport(
        means1, means2,
        kpts1, track_ids1,
        kpts2, track_ids2,
        args.radius,
        weight_mode=args.weight_mode,
        dedup_mode=args.oracle_dedup,
    )
    t_oracle, eig_oracle = compute_closed_form_translation(
        solver,
        R_t,
        T_oracle,
        use_hard_assignment=False,
        top_k=args.ot_top_k,
        min_transport_mass=1e-6,
    )
    if t_oracle is None:
        oracle_err = float("nan")
        oracle_gap = {}
    else:
        oracle_err = min(
            translation_error(t_oracle, t_gt),
            translation_error(-t_oracle, t_gt),
        )
        oracle_gap = compute_gap_metrics(eig_oracle)

    key_t_err = float("nan")
    key_gap = {}
    key_n = 0
    if args.use_keypoint:
        pts1, pts2 = build_keypoint_pairs(kpts1, track_ids1, kpts2, track_ids2)
        key_n = pts1.shape[0]
        t_key, eig_key = compute_t_from_keypoints(K, K, R_use, pts1, pts2)
        if t_key is not None:
            key_t_err = min(
                translation_error(t_key, t_gt),
                translation_error(-t_key, t_gt),
            )
            key_gap = compute_gap_metrics(eig_key)

    soft_t_err = float("nan")
    soft_gap = {}
    soft_used = 0
    if args.oracle_soft:
        T_soft, soft_used = build_soft_oracle_transport(
            means1,
            means2,
            kpts1,
            track_ids1,
            kpts2,
            track_ids2,
            args.radius,
            args.oracle_soft_k,
            args.oracle_soft_sigma,
        )
        t_soft, eig_soft = compute_closed_form_translation(
            solver,
            R_t,
            T_soft,
            use_hard_assignment=False,
            top_k=args.ot_top_k,
            min_transport_mass=1e-6,
        )
        if t_soft is not None:
            soft_t_err = min(
                translation_error(t_soft, t_gt),
                translation_error(-t_soft, t_gt),
            )
            soft_gap = compute_gap_metrics(eig_soft)

    r_err = rotation_error(R_use, R_gt)

    print("\n" + "=" * 70)
    print("Step U: oracle vs OT for closed-form t")
    print(f"  Pair: ({args.idx1}, {args.idx2}), radius={args.radius}px, R_err={r_err:.2f}deg")
    print(f"  gate: epi_top={args.gate_epi}, cov_top={args.gate_cov}, desc_top={args.gate_desc}")
    print(f"  OT: epsilon={args.epsilon}, rho={args.rho}, use_hard={args.ot_use_hard}")
    print("=" * 70)
    print(f"OT transport: T_sum={transport.sum().item():.4f}, conc={conc:.4f}")
    if t_ot is None:
        print("OT t: None (collapse)")
    else:
        print(f"OT t_err={ot_err:.2f}deg, gap_trace={ot_gap.get('gap_trace', float('nan')):.5f}")
    print(f"Oracle pairs: {len(pairs)}")
    if t_oracle is None:
        print("Oracle t: None (insufficient pairs)")
    else:
        print(f"Oracle t_err={oracle_err:.2f}deg, gap_trace={oracle_gap.get('gap_trace', float('nan')):.5f}")
    if args.use_keypoint:
        if key_n == 0 or np.isnan(key_t_err):
            print("Keypoint t: None (insufficient pairs)")
        else:
            print(f"Keypoint pairs: {key_n}")
            print(f"Keypoint t_err={key_t_err:.2f}deg, gap_trace={key_gap.get('gap_trace', float('nan')):.5f}")
    if args.oracle_soft:
        if soft_used == 0 or np.isnan(soft_t_err):
            print("Soft-oracle t: None (insufficient pairs)")
        else:
            print(f"Soft-oracle tracks used: {soft_used}")
            print(f"Soft-oracle t_err={soft_t_err:.2f}deg, gap_trace={soft_gap.get('gap_trace', float('nan')):.5f}")


if __name__ == "__main__":
    main()
