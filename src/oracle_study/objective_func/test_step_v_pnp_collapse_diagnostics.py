#!/usr/bin/env python3
"""
Step V: PnG (3D↔2D) collapse diagnostics.

Goal:
  - Distinguish numerical underflow vs UOT rejection
  - Check whether epsilon/rho stabilize transport for PnG registration
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.reconstructor.viewpoint_extender import ViewpointExtender
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
    read_images_with_points2d,
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_collapse_type_diagnostics import (
    sinkhorn_with_diagnostics,
)
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
    compute_descriptor_cost_matrix,
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


def load_tracks(colmap_dir: str, image_name: str) -> dict:
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)
    img_data = None
    for img in images_with_pts.values():
        if img["name"] == image_name:
            img_data = img
            break
    if img_data is None:
        raise ValueError(f"Image {image_name} not found in images.bin")
    tracks = {}
    for pt in img_data["points2d"]:
        if pt["point3d_id"] != -1:
            tracks[pt["point3d_id"]] = np.array([pt["x"], pt["y"]], dtype=np.float32)
    return tracks


def triangulate_points(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    P1 = K1 @ np.hstack([R1, t1.reshape(3, 1)])
    P2 = K2 @ np.hstack([R2, t2.reshape(3, 1)])
    pts1_h = pts1.T
    pts2_h = pts2.T
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X = (X_h[:3] / (X_h[3:] + 1e-10)).T
    return X


def build_3d_gaussians(points_3d: np.ndarray, sigma: float):
    gaussians = []
    cov = np.eye(3, dtype=np.float32) * (sigma ** 2)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([sigma, sigma, sigma], dtype=np.float32)
    color = np.zeros(3, dtype=np.float32)
    alpha = np.float32(1.0)
    for p in points_3d:
        gaussians.append(
            {
                "center": p.astype(np.float32),
                "covariance": cov.copy(),
                "color": color.copy(),
                "alpha": alpha,
                "quaternion": quat.copy(),
                "scale": scale.copy(),
            }
        )
    return gaussians


def rotation_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    angle = np.deg2rad(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R.astype(np.float32)


def build_gate_mask(costs: List[Tuple[torch.Tensor, int]]) -> torch.Tensor:
    if not costs:
        raise ValueError("No costs provided for gate mask.")
    k1, k2 = costs[0][0].shape
    gate = torch.zeros((k1, k2), dtype=torch.bool, device=costs[0][0].device)
    for i in range(k1):
        idx = torch.arange(k2, device=costs[0][0].device)
        for cost, top_m in costs:
            if idx.numel() == 0 or top_m <= 0:
                break
            top_m_i = min(top_m, idx.numel())
            row = cost[i, idx]
            _, sel = torch.topk(row, top_m_i, largest=False)
            idx = idx[sel]
        if idx.numel() > 0:
            gate[i, idx] = True
    # ensure each row/col has at least one allowed entry
    row_all_false = gate.sum(dim=1) == 0
    if row_all_false.any():
        cost_ref = costs[0][0]
        idx = torch.argmin(cost_ref[row_all_false], dim=1)
        gate[row_all_false, idx] = True
    col_all_false = gate.sum(dim=0) == 0
    if col_all_false.any():
        cost_ref = costs[0][0]
        idx = torch.argmin(cost_ref[:, col_all_false], dim=0)
        gate[idx, col_all_false] = True
    return gate


def run_case(
    solver,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    eps_list: List[float],
    rho_scale: float,
    label: str,
    sinkhorn_dtype: torch.dtype,
    gate_solvers=None,
    gate_desc_cost: torch.Tensor | None = None,
    gate_top_m_epi: int = 0,
    gate_top_m_cov: int = 0,
    gate_top_m_desc: int = 0,
    auto_eps: bool = False,
    append_auto: bool = False,
    eps_scale: float = 25.0,
    eps_min: float = 0.2,
) -> None:
    # Compute cost in float32 to avoid dtype mismatch inside solver
    R_t = torch.tensor(R_rel, dtype=torch.float32)
    t_t = torch.tensor(t_rel, dtype=torch.float32)
    F = solver._build_F_from_wc(R_t, t_t)
    cost = solver.compute_cost_matrix(F)
    cost_raw = cost
    gate_ratio = None
    c_med_valid = None
    if gate_solvers is not None or gate_desc_cost is not None:
        gate_costs: List[Tuple[torch.Tensor, int]] = []
        if gate_solvers is not None and gate_top_m_epi > 0:
            cost_epi = gate_solvers[0].compute_cost_matrix(F)
            gate_costs.append((cost_epi, gate_top_m_epi))
        if gate_solvers is not None and gate_top_m_cov > 0:
            cost_cov = gate_solvers[1].compute_cost_matrix(F)
            gate_costs.append((cost_cov, gate_top_m_cov))
        if gate_desc_cost is not None and gate_top_m_desc > 0:
            gate_costs.append((gate_desc_cost, gate_top_m_desc))
        if gate_costs:
            gate = build_gate_mask(gate_costs)
            gate_ratio = float(gate.float().mean().item())
            valid_costs = cost_raw[gate]
            if valid_costs.numel() > 0:
                c_med_valid = float(valid_costs.median().item())
            cost = torch.where(gate, cost, cost.max() + 1e6)
    if sinkhorn_dtype == torch.float64:
        cost = cost.double()
        cost_raw = cost_raw.double()
    c_min = float(cost.min().item())
    c_med = float(cost.median().item())
    c_max = float(cost.max().item())
    c_mean = float(cost.mean().item())
    c_med_raw = float(cost_raw.median().item())
    if c_med_valid is None:
        c_med_valid = c_med_raw
    print(
        f"\n[{label}] cost stats: med(raw)={c_med_raw:.3f} "
        f"-> med(valid)={c_med_valid:.3f} -> med(gated)={c_med:.3f}"
    )
    print(f"  min={c_min:.3f}, mean={c_mean:.3f}, max={c_max:.3f}")
    if gate_ratio is not None:
        print(f"  gate_ratio={gate_ratio:.4f} (kept)")

    a = solver.alpha1.to(sinkhorn_dtype)
    b = solver.alpha2.to(sinkhorn_dtype)

    eps_auto = max(eps_min, c_med_valid / max(eps_scale, 1e-6))
    if auto_eps:
        eps_list = [eps_auto]
    elif append_auto:
        eps_list = list(eps_list) + [eps_auto]
    if auto_eps or append_auto:
        print(f"  eps_auto={eps_auto:.3f} (med/scale, scale={eps_scale})")

    for eps in eps_list:
        rho = rho_scale * eps
        logK = (-cost / eps).detach()
        logK_max = float(logK.max().item())
        logK_min = float(logK.min().item())

        T, diag = sinkhorn_with_diagnostics(cost, a, b, epsilon=eps, rho=rho, max_iter=200)
        T_sum = float(T.sum().item())
        T_max = float(T.max().item())
        collapse = "ok"
        if not np.isfinite(T_sum):
            collapse = "nan"
        elif logK_max < -80:
            collapse = "underflow"
        elif T_sum < 1e-3:
            collapse = "uot_reject"

        print(
            f"  eps={eps:.3f} rho={rho:.3f} "
            f"logK_max={logK_max:.1f} logK_min={logK_min:.1f} "
            f"T_sum={T_sum:.4f} T_max={T_max:.4e} -> {collapse}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Step V: PnG collapse diagnostics")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--idx-new", type=int, default=20)
    parser.add_argument("--max-tracks", type=int, default=500)
    parser.add_argument("--sigma-3d", type=float, default=0.01)
    parser.add_argument("--eps-list", type=str, default="0.05,0.2,1.0")
    parser.add_argument("--rho-scale", type=float, default=10.0)
    parser.add_argument("--bad-rot-deg", type=float, default=60.0)
    parser.add_argument("--gate-epi", type=int, default=0)
    parser.add_argument("--gate-cov", type=int, default=0)
    parser.add_argument("--gate-desc", type=int, default=0)
    parser.add_argument("--desc-k", type=int, default=8)
    parser.add_argument("--auto-eps", action="store_true")
    parser.add_argument("--append-auto", action="store_true")
    parser.add_argument("--eps-scale", type=float, default=25.0)
    parser.add_argument("--eps-min", type=float, default=0.2)
    parser.add_argument("--use-f64", action="store_true")
    args = parser.parse_args()

    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")
    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{args.idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{args.idx2:04d}.png")
    cam_new = get_colmap_camera_params(cameras, images, f"{args.idx_new:04d}.png")

    tracks1 = load_tracks(colmap_dir, f"{args.idx1:04d}.png")
    tracks2 = load_tracks(colmap_dir, f"{args.idx2:04d}.png")
    shared_ids = list(set(tracks1.keys()) & set(tracks2.keys()))
    if not shared_ids:
        raise RuntimeError("No shared tracks between base views.")

    pts1 = np.stack([tracks1[i] for i in shared_ids], axis=0)
    pts2 = np.stack([tracks2[i] for i in shared_ids], axis=0)
    X = triangulate_points(
        cam1["K"], cam1["R"], cam1["t"],
        cam2["K"], cam2["R"], cam2["t"],
        pts1, pts2,
    )
    if args.max_tracks > 0 and X.shape[0] > args.max_tracks:
        X = X[: args.max_tracks]

    existing_3d = build_3d_gaussians(X, args.sigma_3d)
    new_gauss = load_gaussians(args.idx_new)["original_gaussians"]

    extender = ViewpointExtender(
        existing_3d_gaussians=existing_3d,
        camera_params_list=[(cam1["R"], cam1["t"]), (cam2["R"], cam2["t"])],
        K_new=cam_new["K"],
        reference_camera_idx=0,
        device=torch.device("cpu"),
    )
    projected_2d = extender.project_3d_gaussians()
    extender.initialize_transport_solver(projected_2d, new_gauss)
    solver = extender.transport_solver

    gate_solvers = None
    if args.gate_epi > 0 or args.gate_cov > 0:
        solver_epi = OptimalTransportSolver(
            gaussians1=projected_2d, gaussians2=new_gauss,
            k1=cam_new["K"], k2=cam_new["K"], device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
            sigma_epipolar=400.0,
        )
        solver_cov = OptimalTransportSolver(
            gaussians1=projected_2d, gaussians2=new_gauss,
            k1=cam_new["K"], k2=cam_new["K"], device="cpu",
            epipolar_mode="sampson",
            lambda_color=0.0, lambda_cov=1.0, lambda_epipolar=0.0,
            sigma_epipolar=400.0,
        )
        gate_solvers = (solver_epi, solver_cov)

    gate_desc_cost = None
    if args.gate_desc > 0:
        gate_desc_cost = compute_descriptor_cost_matrix(projected_2d, new_gauss, args.desc_k)

    # Relative pose from ref (cam1) to new view (cam_new)
    R_rel = cam_new["R"] @ cam1["R"].T
    t_rel = cam_new["t"] - R_rel @ cam1["t"]
    t_rel = t_rel / (np.linalg.norm(t_rel) + 1e-10)

    # Bad pose: rotate around y-axis
    R_bad = rotation_from_axis_angle(np.array([0.0, 1.0, 0.0]), args.bad_rot_deg) @ R_rel
    t_bad = t_rel.copy()

    eps_list = [float(x) for x in args.eps_list.split(",") if x.strip()]
    dtype = torch.float64 if args.use_f64 else torch.float32

    print("\n" + "=" * 70)
    print("Step V: PnG collapse diagnostics (3D↔2D)")
    print(f"  Base views: ({args.idx1}, {args.idx2}), new view: {args.idx_new}")
    print(f"  Triangulated points: {X.shape[0]}")
    print(f"  eps_list={eps_list}, rho_scale={args.rho_scale}, dtype={dtype}")
    print("=" * 70)

    run_case(
        solver,
        R_rel,
        t_rel,
        eps_list,
        args.rho_scale,
        "GT pose",
        dtype,
        gate_solvers=gate_solvers,
        gate_desc_cost=gate_desc_cost,
        gate_top_m_epi=args.gate_epi,
        gate_top_m_cov=args.gate_cov,
        gate_top_m_desc=args.gate_desc,
        auto_eps=args.auto_eps,
        append_auto=args.append_auto,
        eps_scale=args.eps_scale,
        eps_min=args.eps_min,
    )
    run_case(
        solver,
        R_bad,
        t_bad,
        eps_list,
        args.rho_scale,
        "Bad pose",
        dtype,
        gate_solvers=gate_solvers,
        gate_desc_cost=gate_desc_cost,
        gate_top_m_epi=args.gate_epi,
        gate_top_m_cov=args.gate_cov,
        gate_top_m_desc=args.gate_desc,
        auto_eps=args.auto_eps,
        append_auto=args.append_auto,
        eps_scale=args.eps_scale,
        eps_min=args.eps_min,
    )


if __name__ == "__main__":
    main()
