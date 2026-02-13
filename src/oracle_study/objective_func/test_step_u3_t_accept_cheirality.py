#!/usr/bin/env python3
"""
Step 3: t update accept/reject using cheirality (triangulation consistency).

Compute t from OT (closed-form), then evaluate cheirality ratio on top-k
correspondences to decide if update should be accepted.
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
    compute_transport_with_gate,
    compute_transport_concentration,
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


def select_topk_pairs(T: torch.Tensor, top_k: int) -> List[Tuple[int, int, float]]:
    T_flat = T.reshape(-1)
    if top_k >= T_flat.numel():
        idx = torch.arange(T_flat.numel(), device=T.device)
        vals = T_flat[idx]
    else:
        vals, idx = torch.topk(T_flat, top_k)
    k2 = T.shape[1]
    pairs = []
    for v, flat in zip(vals.tolist(), idx.tolist()):
        i = flat // k2
        j = flat % k2
        pairs.append((i, j, v))
    return pairs


def triangulate_positive_ratio(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> float:
    if pts1.shape[0] == 0:
        return 0.0
    K_inv = np.linalg.inv(K)
    ones = np.ones((pts1.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])
    x1n = (K_inv @ x1.T).T
    x2n = (K_inv @ x2.T).T

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    pos = 0
    for i in range(pts1.shape[0]):
        x1i = x1n[i]
        x2i = x2n[i]
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = x1i[0] * P1[2] - P1[0]
        A[1] = x1i[1] * P1[2] - P1[1]
        A[2] = x2i[0] * P2[2] - P2[0]
        A[3] = x2i[1] * P2[2] - P2[1]
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X = X[:3] / (X[3] + 1e-12)
        Z1 = X[2]
        X2 = R @ X + t
        Z2 = X2[2]
        if Z1 > 0 and Z2 > 0:
            pos += 1
    return pos / pts1.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: t accept/reject by cheirality")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--gate-epi", type=int, default=0)
    parser.add_argument("--gate-cov", type=int, default=0)
    parser.add_argument("--gate-desc", type=int, default=0)
    parser.add_argument("--desc-k", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--r-perturb", type=float, default=0.0)
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
    t_ot, _ = compute_closed_form_translation(
        solver,
        R_t,
        transport,
        use_hard_assignment=False,
        top_k=args.top_k,
        min_transport_mass=0.05,
    )

    means1 = g1.means.numpy() if hasattr(g1.means, "numpy") else g1.means
    means2 = g2.means.numpy() if hasattr(g2.means, "numpy") else g2.means

    pairs = select_topk_pairs(transport, args.top_k)
    pts1 = np.stack([means1[i] for i, _, _ in pairs], axis=0)
    pts2 = np.stack([means2[j] for _, j, _ in pairs], axis=0)

    cheir_gt = triangulate_positive_ratio(K, R_use, t_gt, pts1, pts2)
    if t_ot is None:
        cheir_ot = float("nan")
        t_err = float("nan")
    else:
        cheir_ot = triangulate_positive_ratio(K, R_use, t_ot, pts1, pts2)
        t_err = min(translation_error(t_ot, t_gt), translation_error(-t_ot, t_gt))

    r_err = rotation_error(R_use, R_gt)

    print("\n" + "=" * 70)
    print("Step 3: t accept/reject by cheirality")
    print(f"  Pair: ({args.idx1},{args.idx2}), R_err={r_err:.2f}deg")
    print(f"  gate: epi_top={args.gate_epi}, cov_top={args.gate_cov}, desc_top={args.gate_desc}")
    print(f"  OT: epsilon={args.epsilon}, rho={args.rho}, top_k={args.top_k}")
    print("=" * 70)
    print(f"  conc={conc:.4f}")
    print(f"  cheirality_ratio(GT)={cheir_gt:.3f}")
    if np.isnan(cheir_ot):
        print("  cheirality_ratio(t_ot)=nan (t_ot None)")
    else:
        print(f"  cheirality_ratio(t_ot)={cheir_ot:.3f}")
        print(f"  t_err(t_ot)={t_err:.2f}deg")


if __name__ == "__main__":
    main()
