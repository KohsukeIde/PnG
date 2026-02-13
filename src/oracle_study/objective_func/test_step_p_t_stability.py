#!/usr/bin/env python3
"""
Step P: Diagnose t stability vs R accuracy (R fixed, OT-based t estimation).

For each R error level, compute OT (with optional gate) and closed-form t.
Report t_err and gap statistics to decide if t is recoverable when R is good.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

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
    compute_closed_form_translation,
    compute_descriptor_cost_matrix,
    compute_gap_metrics,
    compute_topk_cost,
    compute_topk_weight_sum,
    compute_transport_concentration,
    compute_transport_with_gate,
    rotation_error,
    rodrigues_rotation,
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


def parse_grid(grid_str: str) -> Optional[Tuple[int, int]]:
    if not grid_str:
        return None
    if "x" in grid_str:
        parts = grid_str.lower().split("x")
    else:
        parts = grid_str.split(",")
    if len(parts) != 2:
        return None
    return int(parts[0]), int(parts[1])


def parse_list(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step P: t stability vs R accuracy")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--angles", type=str, default="0,5,10,15")
    parser.add_argument("--n-axes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--matching", type=str, default="global_topk")
    parser.add_argument("--row-topk", type=int, default=0)
    parser.add_argument("--grid", type=str, default="")
    parser.add_argument("--max-per-cell", type=int, default=2)
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

    angles = parse_list(args.angles)
    row_top_k = args.row_topk if args.row_topk > 0 else None
    diversity_grid = parse_grid(args.grid)

    print("\n" + "=" * 70)
    print("Step P: t stability vs R accuracy")
    print(f"  Pair: ({args.idx1}, {args.idx2}), angles={angles}, n_axes={args.n_axes}")
    print(f"  gate: epi_top={args.gate_epi}, cov_top={args.gate_cov}, desc_top={args.gate_desc}")
    print(f"  matching={args.matching}, row_top_k={row_top_k}, grid={diversity_grid}")
    print("=" * 70)

    for angle_deg in angles:
        t_errs = []
        gap_norms = []
        gap_traces = []
        concs = []
        topk_sums = []
        topk_costs = []

        for _ in range(args.n_axes):
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            R_delta = rodrigues_rotation(axis, np.deg2rad(angle_deg))
            R = R_delta @ R_gt
            R_t = torch.tensor(R, dtype=torch.float32)
            t_t = torch.tensor(t_gt, dtype=torch.float32)
            t_t = t_t / (t_t.norm() + 1e-10)

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

            concs.append(compute_transport_concentration(transport))
            topk_sums.append(compute_topk_weight_sum(transport, args.top_k))
            topk_costs.append(compute_topk_cost(transport, cost_matrix, args.top_k))

            result = compute_closed_form_translation(
                solver,
                R_t,
                transport,
                use_hard_assignment=True,
                top_k=args.top_k,
                min_transport_mass=0.05,
                min_topk_weight=0.0,
                matching_mode=args.matching,
                row_top_k=row_top_k,
                diversity_grid=diversity_grid,
                max_per_cell=args.max_per_cell,
            )
            t_opt, eigvals = result if result[0] is not None else (None, None)
            if t_opt is None:
                t_errs.append(float("nan"))
                gap_norms.append(0.0)
                gap_traces.append(0.0)
                continue

            t_err = min(
                translation_error(t_opt, t_gt),
                translation_error(-t_opt, t_gt),
            )
            t_errs.append(t_err)
            gaps = compute_gap_metrics(eigvals)
            gap_norms.append(float(gaps["gap_norm"]))
            gap_traces.append(float(gaps["gap_trace"]))

        t_errs_np = np.array(t_errs, dtype=np.float64)
        valid = np.isfinite(t_errs_np)
        if valid.sum() == 0:
            print(f"\nAngle {angle_deg:.1f}deg: no valid t updates (all collapsed)")
            continue

        print(f"\nAngle {angle_deg:.1f}deg (R_err~{angle_deg:.1f}deg):")
        print(f"  t_err mean={np.mean(t_errs_np[valid]):.2f}, "
              f"median={np.median(t_errs_np[valid]):.2f}")
        print(f"  gap_norm mean={np.mean(gap_norms):.5f}, "
              f"gap_trace mean={np.mean(gap_traces):.5f}")
        print(f"  conc mean={np.mean(concs):.3f}, topk_sum mean={np.mean(topk_sums):.3f}, "
              f"topk_cost mean={np.mean(topk_costs):.4f}")


if __name__ == "__main__":
    main()
