#!/usr/bin/env python3
"""Loss(F_ref) vs F_bad on real data using fitted 2D Gaussians and COLMAP poses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

# Enable unpickling of TwoDGaussians saved with module name "twodgs"
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import src.primitive.twod_gaussians_rs as twodgs

sys.modules["twodgs"] = twodgs

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
)
from utils.gs_pkl_loader import load_gaussians_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Loss(F_ref) vs perturbed F on real data.")
    parser.add_argument(
        "--colmap-dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0",
        help="Path to COLMAP directory containing cameras/images files.",
    )
    parser.add_argument(
        "--gaussians-dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/apple_200_5k_pkls",
        help="Directory with fitted_gaussians_XXXX.pkl files.",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default="0000.png",
        help="First image name (must exist in COLMAP and Gaussians).",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default="0001.png",
        help="Second image name (must exist in COLMAP and Gaussians).",
    )
    parser.add_argument(
        "--angles-deg",
        type=str,
        default="10,30,60",
        help="Comma-separated rotation angles (degrees) to create F_bad (± each).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save metrics JSON (default: results under camera_pose/real_data/results).",
    )
    parser.add_argument(
        "--epipolar-mode",
        type=str,
        choices=["sampson", "sed"],
        default="sampson",
        help="Epipolar cost type (sampson recommended).",
    )
    parser.add_argument(
        "--lambda-color",
        type=float,
        default=1.0,
        help="Weight for color term.",
    )
    parser.add_argument(
        "--lambda-epipolar",
        type=float,
        default=1.0,
        help="Weight for epipolar term.",
    )
    parser.add_argument(
        "--lambda-cov",
        type=float,
        default=0.3,
        help="Weight for covariance term.",
    )
    parser.add_argument(
        "--sigma-epipolar",
        type=float,
        default=400.0,
        help="Sigma for epipolar term.",
    )
    parser.add_argument(
        "--sigma-color",
        type=float,
        default=0.5,
        help="Sigma for color term.",
    )
    parser.add_argument(
        "--sigma-cov",
        type=float,
        default=8.0,
        help="Sigma for covariance term.",
    )
    parser.add_argument(
        "--noise-model",
        type=str,
        choices=["gaussian", "cauchy", "huber"],
        default="gaussian",
        help="Noise model for costs.",
    )
    return parser.parse_args()


def load_colmap_mapping(colmap_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    cameras = load_cameras_from_colmap(colmap_dir)
    images = load_images_from_colmap(colmap_dir)
    mapping: Dict[str, Dict[str, np.ndarray]] = {}
    for _, info in images.items():
        if info["camera_id"] not in cameras:
            continue
        cam = cameras[info["camera_id"]]
        R_wc = quaternion_to_rotation_matrix(info["qw"], info["qx"], info["qy"], info["qz"])
        t_wc = np.array([info["tx"], info["ty"], info["tz"]], dtype=np.float32)
        mapping[info["name"]] = {
            "R_wc": R_wc.astype(np.float32),
            "t_wc": t_wc,
            "K": cam.get_camera_matrix().astype(np.float32),
        }
    return mapping


def angle_to_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    rotvec = axis / (np.linalg.norm(axis) + 1e-9) * np.deg2rad(angle_deg)
    R, _ = cv2.Rodrigues(rotvec.astype(np.float64))
    return R.astype(np.float32)


def build_solver(
    g1,
    g2,
    k1: np.ndarray,
    k2: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> OptimalTransportSolver:
    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=k1,
        k2=k2,
        epsilon=0.01,
        lambda_color=args.lambda_color,
        lambda_epipolar=args.lambda_epipolar,
        lambda_cov=args.lambda_cov,
        sigma_epipolar=args.sigma_epipolar,
        sigma_color=args.sigma_color,
        sigma_cov=args.sigma_cov,
        noise_model=args.noise_model,
        epipolar_mode=args.epipolar_mode,
        device=device,
    )
    return solver


def compute_loss(solver: OptimalTransportSolver, F: torch.Tensor) -> float:
    cost = solver.compute_cost_matrix(F)
    transport = solver.unbalanced_sinkhorn_algorithm(cost)
    return float((transport * cost).sum().item())


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    colmap_dir = args.colmap_dir
    gauss_dir = args.gaussians_dir

    angles = [float(a) for a in args.angles_deg.split(",") if a.strip()]
    Path(gauss_dir).mkdir(parents=True, exist_ok=True)

    mapping = load_colmap_mapping(colmap_dir)
    if args.image1 not in mapping or args.image2 not in mapping:
        raise ValueError(f"Images {args.image1}, {args.image2} not found in COLMAP data.")

    # Load Gaussians
    _, g1, _, _ = load_gaussians_torch(
        os.path.join(gauss_dir, f"fitted_gaussians_{Path(args.image1).stem}.pkl"), device=device
    )
    _, g2, _, _ = load_gaussians_torch(
        os.path.join(gauss_dir, f"fitted_gaussians_{Path(args.image2).stem}.pkl"), device=device
    )

    k1 = mapping[args.image1]["K"]
    k2 = mapping[args.image2]["K"]

    solver = build_solver(g1, g2, k1, k2, device, args)

    R_wc = torch.tensor(mapping[args.image1]["R_wc"], dtype=torch.float32, device=device)
    t_wc = torch.tensor(mapping[args.image1]["t_wc"], dtype=torch.float32, device=device)
    R_wc2 = torch.tensor(mapping[args.image2]["R_wc"], dtype=torch.float32, device=device)
    t_wc2 = torch.tensor(mapping[args.image2]["t_wc"], dtype=torch.float32, device=device)

    # Compute F_ref (camera1 at origin => relative pose from cam1 to cam2)
    R_rel = R_wc2 @ R_wc.transpose(0, 1)
    t_rel = t_wc2 - R_rel @ t_wc
    F_ref = solver._build_F_from_wc(R_rel, t_rel)
    loss_ref = compute_loss(solver, F_ref)

    # Perturb rotations around y-axis (symmetric ± angles)
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    bad_results: List[Dict[str, float]] = []
    for ang in angles:
        for sign in (+1.0, -1.0):
            R_delta = angle_to_rotation(axis, sign * ang)
            R_bad = torch.tensor(R_delta, dtype=torch.float32, device=device) @ R_rel
            F_bad = solver._build_F_from_wc(R_bad, t_rel)
            loss_bad = compute_loss(solver, F_bad)
            bad_results.append({"angle_deg": float(sign * ang), "loss": loss_bad})

    metrics = {
        "image_pair": [args.image1, args.image2],
        "epipolar_mode": args.epipolar_mode,
        "noise_model": args.noise_model,
        "lambda_color": args.lambda_color,
        "lambda_epipolar": args.lambda_epipolar,
        "lambda_cov": args.lambda_cov,
        "sigma_epipolar": args.sigma_epipolar,
        "sigma_color": args.sigma_color,
        "sigma_cov": args.sigma_cov,
        "loss_ref": loss_ref,
        "loss_bad": bad_results,
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "results" / f"{Path(args.image1).stem}_{Path(args.image2).stem}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
