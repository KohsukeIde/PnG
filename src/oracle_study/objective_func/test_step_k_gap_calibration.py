#!/usr/bin/env python3
"""
Step K: Eigen-gap calibration at GT and near-GT poses.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

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
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
    compute_closed_form_translation,
    compute_transport_with_gate,
    compute_gap_metrics,
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


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


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


def run_gap_calibration(
    idx1: int,
    idx2: int,
    angles: List[float],
    n_axes: int,
    epsilon: float,
    rho: float,
    top_k: int,
    matching_mode: str,
    row_top_k: Optional[int],
    diversity_grid: Optional[Tuple[int, int]],
    max_per_cell: int,
    gate_top_m_epi: Optional[int],
    gate_top_m_cov: Optional[int],
):
    print("\n" + "=" * 70)
    print("Step K: Eigen-gap calibration")
    print(f"  Pair: ({idx1}, {idx2}), angles={angles}, n_axes={n_axes}")
    print(f"  matching={matching_mode}, row_top_k={row_top_k}, grid={diversity_grid}, max_per_cell={max_per_cell}")
    if gate_top_m_epi and gate_top_m_cov:
        print(f"  gate: epi_top={gate_top_m_epi}, cov_top={gate_top_m_cov}")
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

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    gate_solvers = None
    if gate_top_m_epi and gate_top_m_cov:
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

    for angle_deg in angles:
        gap_norm_list = []
        gap_ratio_list = []
        gap_rel_list = []
        gap_trace_list = []
        for _ in range(n_axes):
            axis = np.random.randn(3)
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            R_delta = rodrigues_rotation(axis, np.deg2rad(angle_deg))
            R = R_delta @ R_gt
            R_t = torch.tensor(R, dtype=torch.float32)
            t_t = torch.tensor(t_gt, dtype=torch.float32)
            t_t = t_t / (t_t.norm() + 1e-10)

            _, transport, _ = compute_transport_with_gate(
                solver,
                R_t,
                t_t,
                epsilon,
                rho,
                gate_solvers=gate_solvers,
                gate_top_m_epi=gate_top_m_epi or 0,
                gate_top_m_cov=gate_top_m_cov or 0,
            )

            result = compute_closed_form_translation(
                solver,
                R_t,
                transport,
                use_hard_assignment=True,
                top_k=top_k,
                min_transport_mass=0.05,
                min_topk_weight=0.0,
                matching_mode=matching_mode,
                row_top_k=row_top_k,
                diversity_grid=diversity_grid,
                max_per_cell=max_per_cell,
            )
            if result[1] is None:
                gap_norm_list.append(0.0)
                gap_ratio_list.append(0.0)
                gap_rel_list.append(0.0)
                gap_trace_list.append(0.0)
                continue
            eigvals = result[1]
            gaps = compute_gap_metrics(eigvals)
            gap_norm_list.append(float(gaps["gap_norm"]))
            gap_ratio_list.append(float(gaps["gap_ratio"]))
            gap_rel_list.append(float(gaps["gap_rel"]))
            gap_trace_list.append(float(gaps["gap_trace"]))

        gap_norm = np.array(gap_norm_list)
        gap_ratio = np.array(gap_ratio_list)
        gap_rel = np.array(gap_rel_list)
        gap_trace = np.array(gap_trace_list)

        print(f"\nAngle {angle_deg:.1f}deg:")
        print(f"  gap_norm  mean={gap_norm.mean():.5f}, median={np.median(gap_norm):.5f}, "
              f"min={gap_norm.min():.5f}, max={gap_norm.max():.5f}")
        print(f"  gap_ratio mean={gap_ratio.mean():.5f}, median={np.median(gap_ratio):.5f}, "
              f"min={gap_ratio.min():.5f}, max={gap_ratio.max():.5f}")
        print(f"  gap_rel   mean={gap_rel.mean():.5f}, median={np.median(gap_rel):.5f}, "
              f"min={gap_rel.min():.5f}, max={gap_rel.max():.5f}")
        print(f"  gap_trace mean={gap_trace.mean():.5f}, median={np.median(gap_trace):.5f}, "
              f"min={gap_trace.min():.5f}, max={gap_trace.max():.5f}")


def main():
    parser = argparse.ArgumentParser(description="Eigen-gap calibration")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--angles", type=str, default="0,5,10")
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
    args = parser.parse_args()

    angles = [float(x) for x in args.angles.split(",") if x.strip()]
    row_top_k = args.row_topk if args.row_topk > 0 else None
    diversity_grid = parse_grid(args.grid)

    run_gap_calibration(
        idx1=args.idx1,
        idx2=args.idx2,
        angles=angles,
        n_axes=args.n_axes,
        epsilon=args.epsilon,
        rho=args.rho,
        top_k=args.top_k,
        matching_mode=args.matching,
        row_top_k=row_top_k,
        diversity_grid=diversity_grid,
        max_per_cell=args.max_per_cell,
        gate_top_m_epi=args.gate_epi,
        gate_top_m_cov=args.gate_cov,
    )


if __name__ == "__main__":
    main()
