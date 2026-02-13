#!/usr/bin/env python3
"""
Step Q: Filter + re-rank selection for R hypotheses.

Sample random R, compute scores at epsilon_end, filter by a metric,
then re-rank within the filtered set. Report recall@1/@L for good poses.
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
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
    compute_descriptor_cost_matrix,
    compute_geometry_score,
    compute_relative_pose_wc,
    random_rotation_matrix,
    rotation_error,
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


def metric_direction(name: str) -> bool:
    return name in {"T_sum", "concentration", "topk_sum"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Step Q: filter+rerank selection")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-angle", type=float, default=90.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--filter-metric", type=str, default="mass_aware")
    parser.add_argument("--filter-frac", type=float, default=0.2)
    parser.add_argument("--rerank-metric", type=str, default="topk_cost")
    parser.add_argument("--top-l", type=int, default=5)
    parser.add_argument("--good-threshold", type=float, default=10.0)
    parser.add_argument("--gate-epi", type=int, default=0)
    parser.add_argument("--gate-cov", type=int, default=0)
    parser.add_argument("--gate-desc", type=int, default=0)
    parser.add_argument("--desc-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{args.idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{args.idx2:04d}.png")
    K = cam1["K"]
    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

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

    samples = []
    for _ in range(args.n_samples):
        R = random_rotation_matrix(args.max_angle)
        R_err = rotation_error(R, R_gt)
        scores = compute_geometry_score(
            solver,
            torch.tensor(R, dtype=torch.float32),
            torch.tensor(t_gt, dtype=torch.float32),
            args.epsilon,
            args.rho,
            top_k=args.top_k,
            gate_solvers=gate_solvers,
            gate_top_m_epi=args.gate_epi,
            gate_top_m_cov=args.gate_cov,
            gate_top_m_desc=args.gate_desc,
            gate_desc_cost=gate_desc_cost,
        )
        scores["R_err"] = R_err
        samples.append(scores)

    def _get_metric(item: Dict[str, float], name: str) -> float:
        return float(item.get(name, float("inf")))

    filter_higher = metric_direction(args.filter_metric)
    rerank_higher = metric_direction(args.rerank_metric)

    values = np.array([_get_metric(s, args.filter_metric) for s in samples], dtype=np.float64)
    order = np.argsort(values)
    if filter_higher:
        order = order[::-1]
    keep_n = max(1, int(len(order) * args.filter_frac))
    keep_idx = order[:keep_n]

    filtered = [samples[i] for i in keep_idx]
    rerank_vals = np.array([_get_metric(s, args.rerank_metric) for s in filtered], dtype=np.float64)
    rerank_order = np.argsort(rerank_vals)
    if rerank_higher:
        rerank_order = rerank_order[::-1]

    good_mask = np.array([s["R_err"] <= args.good_threshold for s in filtered], dtype=bool)
    if good_mask.sum() == 0:
        recall1 = float("nan")
        recall_l = float("nan")
    else:
        recall1 = float(good_mask[rerank_order[:1]].sum() / good_mask.sum())
        recall_l = float(good_mask[rerank_order[:args.top_l]].sum() / good_mask.sum())

    best_idx = rerank_order[0]
    best_r_err = filtered[best_idx]["R_err"]

    print("\n" + "=" * 70)
    print("Step Q: filter + rerank selection")
    print(f"  Pair: ({args.idx1}, {args.idx2}), n_samples={args.n_samples}")
    print(f"  filter: {args.filter_metric} (frac={args.filter_frac}), "
          f"rerank: {args.rerank_metric}, top-L={args.top_l}")
    print(f"  good threshold: R_err <= {args.good_threshold:.1f}deg")
    print("=" * 70)
    print(f"Filtered count: {len(filtered)}/{len(samples)}")
    print(f"Good in filtered: {int(good_mask.sum())}")
    print(f"Best R_err after rerank: {best_r_err:.2f}deg")
    print(f"Recall@1: {recall1:.3f}, Recall@{args.top_l}: {recall_l:.3f}")


if __name__ == "__main__":
    main()
