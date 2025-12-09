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
        "--gt-npz",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/cameras.npz",
        help="Path to GT cameras.npz (NeRF-style) with scale_mat/world_mat/camera_mat entries.",
    )
    parser.add_argument(
        "--use-gt-cameras",
        action="store_true",
        help="Use GT cameras.npz instead of COLMAP sparse poses/intrinsics.",
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
        "--epsilon",
        type=float,
        default=None,
        help="Override Sinkhorn epsilon (if None, use auto 0.08*median).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Override Sinkhorn rho (if None, use auto 10*epsilon).",
    )
    parser.add_argument(
        "--fixed-nn-epi",
        action="store_true",
        help="Also report fixed nearest-neighbor Sampson error (no OT).",
    )
    parser.add_argument(
        "--use-colmap-matches",
        action="store_true",
        help="If images.txt is available, use shared point3D tracks as fixed correspondences and report Sampson.",
    )
    parser.add_argument(
        "--use-sift-matches",
        action="store_true",
        help="Extract SIFT matches (ratio test) and report Sampson errors on inliers.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images",
        help="Directory containing image files (for SIFT matching).",
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


def load_gt_npz_mapping(npz_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load GT cameras from NeRF-style cameras.npz (with scale_mat/world_mat/camera_mat)."""
    data = np.load(npz_path)
    mapping: Dict[str, Dict[str, np.ndarray]] = {}
    idx = 0
    while f"camera_mat_inv_{idx}" in data:
        # Intrinsics
        K = data[f"camera_mat_inv_{idx}"][:3, :3]
        # Pose: apply scale to get back to world scale used in GT
        scale = data.get(f"scale_mat_{idx}")
        world_inv = data[f"world_mat_inv_{idx}"]  # c2w before scaling
        if scale is not None:
            c2w = scale @ world_inv
        else:
            c2w = world_inv
        w2c = np.linalg.inv(c2w)
        R_wc = w2c[:3, :3]
        t_wc = w2c[:3, 3]
        name = f"{idx:04d}.png"
        mapping[name] = {"R_wc": R_wc.astype(np.float32), "t_wc": t_wc.astype(np.float32), "K": K.astype(np.float32)}
        idx += 1
    return mapping


def load_colmap_tracks(colmap_dir: str) -> Dict[str, Dict[int, np.ndarray]]:
    """Parse images.txt if available to extract 2D points keyed by point3D_id per image."""
    images_txt = Path(colmap_dir) / "images.txt"
    if not images_txt.exists():
        return {}
    tracks: Dict[str, Dict[int, np.ndarray]] = {}
    with open(images_txt, "r") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if header.startswith("#") or not header.strip():
                continue
            parts = header.strip().split()
            if len(parts) < 10:
                # malformed
                continue
            image_name = parts[9]
            points_line = f.readline()
            pts = points_line.strip().split()
            img_tracks: Dict[int, np.ndarray] = {}
            for i in range(0, len(pts), 3):
                try:
                    x = float(pts[i])
                    y = float(pts[i + 1])
                    pid = int(pts[i + 2])
                except (ValueError, IndexError):
                    continue
                if pid >= 0:
                    img_tracks[pid] = np.array([x, y], dtype=np.float32)
            tracks[image_name] = img_tracks
    return tracks


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


def compute_loss(solver: OptimalTransportSolver, F: torch.Tensor, epsilon=None, rho=None) -> float:
    cost = solver.compute_cost_matrix(F)
    transport = solver.unbalanced_sinkhorn_algorithm(cost, epsilon=epsilon, rho=rho)
    return float((transport * cost).sum().item())


def sampson_dist(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """Compute Sampson distance for pairs (pts1[i], pts2[i]) using homogeneous coords."""
    device = pts1.device
    ones1 = torch.ones((pts1.shape[0], 1), device=device)
    ones2 = torch.ones((pts2.shape[0], 1), device=device)
    x1 = torch.cat([pts1, ones1], dim=1)  # (N,3)
    x2 = torch.cat([pts2, ones2], dim=1)  # (N,3)
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = torch.sum(x2 * (F @ x1.T).T, dim=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-9
    return numer**2 / denom


def fixed_nn_sampson(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> float:
    """One-way nearest-neighbor correspondences from pts1 to pts2, Sampson averaged."""
    with torch.no_grad():
        dists = torch.cdist(pts1, pts2)  # (K1,K2)
        nn_idx = torch.argmin(dists, dim=1)
        matched_pts2 = pts2[nn_idx]
        s = sampson_dist(F, pts1, matched_pts2)
        return float(s.mean().item()) if s.numel() > 0 else float("nan")


def sampson_np(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Sampson distance in numpy, pts: (N,2), F: (3,3)."""
    ones = np.ones((pts1.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])
    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = np.sum(x2 * (F @ x1.T).T, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-9
    return numer**2 / denom


def sift_matches_sampson(F_ref: np.ndarray, F_bads: List[np.ndarray], img1_path: str, img2_path: str) -> Dict[str, float]:
    """Compute SIFT matches + RANSAC inliers, then Sampson mean for F_ref and each F_bad."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return {}
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return {}
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return {}
    pts1 = np.float64([k1[m.queryIdx].pt for m in good])
    pts2 = np.float64([k2[m.trainIdx].pt for m in good])
    F_ransac, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if mask is None:
        return {}
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    if len(pts1_in) < 8:
        return {}
    out: Dict[str, float] = {
        "sift_n_matches": len(good),
        "sift_n_inliers": int(inliers.sum()),
        "sift_sampson_ref": float(sampson_np(F_ref, pts1_in, pts2_in).mean()),
        "sift_F_ransac_est": F_ransac.tolist() if F_ransac is not None else None,
    }
    for idx, Fb in enumerate(F_bads):
        out[f"sift_sampson_bad_{idx}"] = float(sampson_np(Fb, pts1_in, pts2_in).mean())
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    colmap_dir = args.colmap_dir
    gauss_dir = args.gaussians_dir

    angles = [float(a) for a in args.angles_deg.split(",") if a.strip()]
    Path(gauss_dir).mkdir(parents=True, exist_ok=True)

    if args.use_gt_cameras:
        mapping = load_gt_npz_mapping(args.gt_npz)
    else:
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
    loss_ref = compute_loss(solver, F_ref, epsilon=args.epsilon, rho=args.rho)
    nn_ref = None
    if args.fixed_nn_epi:
        nn_ref = fixed_nn_sampson(F_ref, solver.means1.detach(), solver.means2.detach())

    # Perturb rotations around y-axis (symmetric ± angles)
    axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    bad_results: List[Dict[str, float]] = []
    F_bad_list = []
    for ang in angles:
        for sign in (+1.0, -1.0):
            R_delta = angle_to_rotation(axis, sign * ang)
            R_bad = torch.tensor(R_delta, dtype=torch.float32, device=device) @ R_rel
            F_bad = solver._build_F_from_wc(R_bad, t_rel)
            F_bad_list.append(F_bad)
            loss_bad = compute_loss(solver, F_bad, epsilon=args.epsilon, rho=args.rho)
            entry = {"angle_deg": float(sign * ang), "loss": loss_bad}
            if args.fixed_nn_epi:
                entry["nn_sampson"] = fixed_nn_sampson(
                    F_bad, solver.means1.detach(), solver.means2.detach()
                )
            bad_results.append(entry)

    shared_ref = None
    shared_bad = []
    if args.use_colmap_matches:
        tracks = load_colmap_tracks(colmap_dir)
        if args.image1 in tracks and args.image2 in tracks:
            t1 = tracks[args.image1]
            t2 = tracks[args.image2]
            shared_ids = list(set(t1.keys()) & set(t2.keys()))
            if shared_ids:
                pts1 = torch.tensor(np.stack([t1[i] for i in shared_ids], axis=0), dtype=torch.float32, device=device)
                pts2 = torch.tensor(np.stack([t2[i] for i in shared_ids], axis=0), dtype=torch.float32, device=device)
                shared_ref = float(sampson_dist(F_ref, pts1, pts2).mean().item())
                for entry in bad_results:
                    ang = entry["angle_deg"]
                    R_delta = angle_to_rotation(axis, ang)
                    R_bad = torch.tensor(R_delta, dtype=torch.float32, device=device) @ R_rel
                    F_bad = solver._build_F_from_wc(R_bad, t_rel)
                    smean = float(sampson_dist(F_bad, pts1, pts2).mean().item())
                    shared_bad.append({"angle_deg": ang, "shared_sampson": smean})

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
        "fixed_nn_sampson_ref": nn_ref,
        "shared_tracks_sampson_ref": shared_ref,
        "shared_tracks_sampson_bad": shared_bad,
    }

    if args.use_sift_matches:
        F_ref_np = F_ref.detach().cpu().numpy()
        F_bads_np = [Fb.detach().cpu().numpy() for Fb in F_bad_list]
        sift_stats = sift_matches_sampson(
            F_ref_np,
            F_bads_np,
            os.path.join(args.images_dir, args.image1),
            os.path.join(args.images_dir, args.image2),
        )
        metrics["sift"] = sift_stats
｀
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
