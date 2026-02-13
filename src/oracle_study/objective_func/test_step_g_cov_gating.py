#!/usr/bin/env python3
"""
Step G: Use cov/shape to gate candidate correspondences.

Compare baseline OT vs gated OT (epipolar top-M then cov top-M2).
"""

import argparse
import os
import sys
from typing import List, Tuple

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
    compute_transport_concentration,
    compute_topk_weight_sum,
    compute_topk_cost,
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


def build_gate_mask(
    epi_cost: torch.Tensor,
    cov_cost: torch.Tensor,
    top_m_epi: int,
    top_m_cov: int,
) -> torch.Tensor:
    k1, k2 = epi_cost.shape
    mask = torch.zeros((k1, k2), dtype=torch.bool, device=epi_cost.device)
    top_m_epi = min(top_m_epi, k2)
    for i in range(k1):
        row_epi = epi_cost[i]
        _, epi_idx = torch.topk(row_epi, top_m_epi, largest=False)
        row_cov = cov_cost[i, epi_idx]
        top_m_cov_i = min(top_m_cov, epi_idx.numel())
        _, cov_idx_local = torch.topk(row_cov, top_m_cov_i, largest=False)
        cov_idx = epi_idx[cov_idx_local]
        mask[i, cov_idx] = True
    return mask


def compute_metrics(T: torch.Tensor, C: torch.Tensor, top_k: int) -> dict:
    T_sum = float(T.sum().item())
    transport_cost = float((T * C).sum().item())
    avg_cost = transport_cost / (T_sum + 1e-10)
    concentration = compute_transport_concentration(T)
    topk_sum = compute_topk_weight_sum(T, top_k)
    topk_cost = compute_topk_cost(T, C, top_k)
    return {
        "T_sum": T_sum,
        "avg_cost": avg_cost,
        "concentration": concentration,
        "topk_sum": topk_sum,
        "topk_cost": topk_cost,
    }


def run_cov_gating(
    idx1: int,
    idx2: int,
    epsilon: float = 0.05,
    rho: float = 0.5,
    top_m_epi: int = 50,
    top_m_cov: int = 10,
    top_k: int = 50,
):
    print("\n" + "=" * 70)
    print("Step G: cov-gated OT")
    print(f"  Pair: ({idx1}, {idx2})")
    print(f"  top_m_epi={top_m_epi}, top_m_cov={top_m_cov}, top_k={top_k}")
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

    solver_base = OptimalTransportSolver(
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

    F = solver_base._build_F_from_wc(R_t, t_t)
    epi_cost = solver_base.compute_cost_matrix(F)
    cov_cost = solver_cov.compute_cost_matrix(F)

    # Baseline OT
    with torch.no_grad():
        T_base, _ = solver_base.unbalanced_sinkhorn_algorithm(
            epi_cost, epsilon=epsilon, rho=rho,
            gate_mask=solver_base._last_gate_mask,
        )

    # Gated OT
    gate = build_gate_mask(epi_cost, cov_cost, top_m_epi, top_m_cov)
    gated_cost, gate = solver_base._apply_gate_mask(epi_cost, gate)

    with torch.no_grad():
        T_gate, _ = solver_base.unbalanced_sinkhorn_algorithm(
            gated_cost, epsilon=epsilon, rho=rho,
            gate_mask=gate,
        )

    base_metrics = compute_metrics(T_base, epi_cost, top_k)
    gate_metrics = compute_metrics(T_gate, gated_cost, top_k)

    valid_ratio = float(gate.float().mean().item())

    print(f"Gate valid ratio: {valid_ratio:.4f}")
    print("\nBaseline vs Gated (GT pose):")
    print(f"  T_sum:     {base_metrics['T_sum']:.4f} -> {gate_metrics['T_sum']:.4f}")
    print(f"  avg_cost:  {base_metrics['avg_cost']:.4f} -> {gate_metrics['avg_cost']:.4f}")
    print(f"  conc:      {base_metrics['concentration']:.4f} -> {gate_metrics['concentration']:.4f}")
    print(f"  topk_sum:  {base_metrics['topk_sum']:.4f} -> {gate_metrics['topk_sum']:.4f}")
    print(f"  topk_cost: {base_metrics['topk_cost']:.4f} -> {gate_metrics['topk_cost']:.4f}")

    return {
        "idx1": idx1,
        "idx2": idx2,
        "valid_ratio": valid_ratio,
        "baseline": base_metrics,
        "gated": gate_metrics,
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
    parser = argparse.ArgumentParser(description="Cov-gated OT experiment")
    parser.add_argument("--pairs", type=str, default="0,10",
                        help="Pairs as 'i,j;k,l' (default: 0,10)")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--top-m-epi", type=int, default=50)
    parser.add_argument("--top-m-cov", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    for idx1, idx2 in pairs:
        run_cov_gating(
            idx1,
            idx2,
            epsilon=args.epsilon,
            rho=args.rho,
            top_m_epi=args.top_m_epi,
            top_m_cov=args.top_m_cov,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
