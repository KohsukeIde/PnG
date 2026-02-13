#!/usr/bin/env python3
"""
Step R: Three-view cycle consistency for pose selection.

Generate candidates per pair, keep top-L, and select combination
by minimizing cycle error R_02 ≈ R_12 @ R_01.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

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


def build_solver(idx: int, K: np.ndarray) -> Tuple[OptimalTransportSolver, dict]:
    data = load_gaussians(idx)
    g = data["original_gaussians"]
    ot_mass = data.get("ot_mass", None)
    solver = OptimalTransportSolver(
        gaussians1=g, gaussians2=g,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass, ot_mass2=ot_mass,
    )
    return solver, data


def sample_candidates(
    idx1: int,
    idx2: int,
    n_samples: int,
    max_angle: float,
    epsilon: float,
    rho: float,
    top_k: int,
    filter_metric: str,
    filter_frac: float,
    top_l: int,
    gate_epi: int,
    gate_cov: int,
    gate_desc: int,
    desc_k: int,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1["K"]
    R_gt, t_gt = compute_relative_pose_wc(cam1, cam2)

    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
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
    if gate_epi > 0 or gate_cov > 0:
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
    if gate_desc > 0:
        gate_desc_cost = compute_descriptor_cost_matrix(g1, g2, desc_k)

    samples = []
    for _ in range(n_samples):
        R = random_rotation_matrix(max_angle)
        R_err = rotation_error(R, R_gt)
        scores = compute_geometry_score(
            solver,
            torch.tensor(R, dtype=torch.float32),
            torch.tensor(t_gt, dtype=torch.float32),
            epsilon,
            rho,
            top_k=top_k,
            gate_solvers=gate_solvers,
            gate_top_m_epi=gate_epi,
            gate_top_m_cov=gate_cov,
            gate_top_m_desc=gate_desc,
            gate_desc_cost=gate_desc_cost,
        )
        scores["R_err"] = R_err
        scores["R"] = R
        samples.append(scores)

    higher = metric_direction(filter_metric)
    default_val = -float("inf") if higher else float("inf")
    metric_vals = np.array([s.get(filter_metric, default_val) for s in samples], dtype=np.float64)
    order = np.argsort(metric_vals)
    if metric_direction(filter_metric):
        order = order[::-1]
    keep_n = max(1, int(len(order) * filter_frac))
    keep_idx = order[:keep_n]
    filtered = [samples[i] for i in keep_idx]

    rerank_vals = np.array([s.get(filter_metric, default_val) for s in filtered], dtype=np.float64)
    rerank_order = np.argsort(rerank_vals)
    if metric_direction(filter_metric):
        rerank_order = rerank_order[::-1]
    top_l = max(1, min(top_l, len(rerank_order)))
    top_candidates = [filtered[i] for i in rerank_order[:top_l]]

    return top_candidates, R_gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Step R: three-view cycle consistency")
    parser.add_argument("--pair0", type=int, default=0)
    parser.add_argument("--pair1", type=int, default=10)
    parser.add_argument("--pair2", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-angle", type=float, default=90.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--filter-metric", type=str, default="mass_aware")
    parser.add_argument("--filter-frac", type=float, default=0.2)
    parser.add_argument("--top-l", type=int, default=5)
    parser.add_argument("--gate-epi", type=int, default=0)
    parser.add_argument("--gate-cov", type=int, default=0)
    parser.add_argument("--gate-desc", type=int, default=0)
    parser.add_argument("--desc-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    c01, R01_gt = sample_candidates(
        args.pair0, args.pair1,
        args.n_samples, args.max_angle,
        args.epsilon, args.rho, args.top_k,
        args.filter_metric, args.filter_frac, args.top_l,
        args.gate_epi, args.gate_cov, args.gate_desc, args.desc_k,
    )
    c12, R12_gt = sample_candidates(
        args.pair1, args.pair2,
        args.n_samples, args.max_angle,
        args.epsilon, args.rho, args.top_k,
        args.filter_metric, args.filter_frac, args.top_l,
        args.gate_epi, args.gate_cov, args.gate_desc, args.desc_k,
    )
    c02, R02_gt = sample_candidates(
        args.pair0, args.pair2,
        args.n_samples, args.max_angle,
        args.epsilon, args.rho, args.top_k,
        args.filter_metric, args.filter_frac, args.top_l,
        args.gate_epi, args.gate_cov, args.gate_desc, args.desc_k,
    )

    best_cycle = None
    for r01 in c01:
        for r12 in c12:
            R02_pred = r12["R"] @ r01["R"]
            for r02 in c02:
                cycle_err = rotation_error(R02_pred, r02["R"])
                if best_cycle is None or cycle_err < best_cycle["cycle_err"]:
                    best_cycle = {
                        "cycle_err": cycle_err,
                        "r01": r01,
                        "r12": r12,
                        "r02": r02,
                    }

    best_r01_err = best_cycle["r01"]["R_err"]
    best_r12_err = best_cycle["r12"]["R_err"]
    best_r02_err = best_cycle["r02"]["R_err"]

    print("\n" + "=" * 70)
    print("Step R: three-view cycle consistency")
    print(f"  Pairs: ({args.pair0},{args.pair1}), ({args.pair1},{args.pair2}), ({args.pair0},{args.pair2})")
    print(f"  filter: {args.filter_metric} (frac={args.filter_frac}), top-L={args.top_l}")
    print("=" * 70)
    print(f"Cycle-minimizing triple: cycle_err={best_cycle['cycle_err']:.2f}deg")
    print(f"  R_err (01/12/02): {best_r01_err:.2f}, {best_r12_err:.2f}, {best_r02_err:.2f} deg")


if __name__ == "__main__":
    main()
