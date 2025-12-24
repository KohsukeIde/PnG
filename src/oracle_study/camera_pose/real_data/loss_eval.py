#!/usr/bin/env python3
"""Loss(F_ref) vs F_bad on real data using fitted 2D Gaussians and COLMAP poses."""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

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
        "--epi-clip",
        type=float,
        default=None,
        help="If set, clip Sampson residuals above this (px) by adding large cost.",
    )
    parser.add_argument(
        "--adaptive-clip-pct",
        type=float,
        default=None,
        help="If set (e.g., 0.01), choose tau so that zero-row rate (min residual > tau) <= pct using GT F.",
    )
    parser.add_argument(
        "--epi-topk",
        type=int,
        default=None,
        help="If set, keep only top-k cheapest costs per row (after clip); others get large cost.",
    )
    parser.add_argument(
        "--mutual-topk",
        action="store_true",
        help="Apply top-k on both rows and columns (intersection) after clip/window.",
    )
    parser.add_argument(
        "--pairwise-stats",
        action="store_true",
        help="If set, record pairwise Sampson min stats (mean/median/quantiles) for ref/bad.",
    )
    parser.add_argument(
        "--spatial-window",
        type=float,
        default=None,
        help="If set (px), mask costs where |p1-p2|_2 > window before top-k/clip.",
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
        "--lambda-mass",
        type=float,
        default=0.0,
        help="Weight for mass penalty KL(row||alpha)+KL(col||beta) in scoring.",
    )
    parser.add_argument(
        "--lambda-inlier-avg",
        type=float,
        default=0.0,
        help="Weight for inlier_avg in inlier-based score: score2 = -inlier_mass + lambda_inlier_avg * inlier_avg.",
    )
    parser.add_argument(
        "--dustbin-cost",
        type=float,
        default=None,
        help="If set, add dustbin row/col with this constant cost (unmatch).",
    )
    parser.add_argument(
        "--dustbin-mass",
        type=float,
        default=1.0,
        help="Relative mass assigned to dustbin rows/cols when dustbin-cost is enabled.",
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
    """Load GT cameras from DTU cameras.npz using cv2.decomposeProjectionMatrix.

    DTU format: world_mat is a 4x4 matrix where the upper 3x4 is P = K[R|t].
    We use OpenCV's RQ decomposition to extract K, R, and camera center.
    """
    data = np.load(npz_path)
    mapping: Dict[str, Dict[str, np.ndarray]] = {}
    idx = 0
    while f"world_mat_{idx}" in data:
        P = data[f"world_mat_{idx}"]  # 4x4, last row [0 0 0 1]
        P3 = P[:3, :4].astype(np.float64)

        # OpenCV's decomposeProjectionMatrix: P = K[R|t] -> returns K, R, camera_center
        K, R, t_homog, _, _, _, _ = cv2.decomposeProjectionMatrix(P3)

        # Normalize K so K[2,2] = 1
        K = K / K[2, 2]

        # t_homog is the camera center in homogeneous world coordinates (4,)
        camera_center = t_homog[:3, 0] / t_homog[3, 0]

        # t_wc = -R @ camera_center (translation in camera coordinates)
        t_wc = -R @ camera_center

        name = f"{idx:04d}.png"
        mapping[name] = {
            "R_wc": R.astype(np.float32),
            "t_wc": t_wc.astype(np.float32),
            "K": K.astype(np.float32),
        }
        idx += 1
    return mapping


def pairwise_sampson(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """Pairwise Sampson distance for all pairs (i in pts1, j in pts2)."""
    assert pts1.dim() == 2 and pts2.dim() == 2
    ones1 = torch.ones((pts1.shape[0], 1), device=pts1.device, dtype=pts1.dtype)
    ones2 = torch.ones((pts2.shape[0], 1), device=pts2.device, dtype=pts2.dtype)
    p1_h = torch.cat([pts1, ones1], dim=1)  # (N,3)
    p2_h = torch.cat([pts2, ones2], dim=1)  # (M,3)
    Fp1 = (F @ p1_h.t()).t()  # (N,3)
    FTp2 = (F.t() @ p2_h.t()).t()  # (M,3)
    numer = p2_h @ Fp1.t()  # (M,N)
    denom = (
        Fp1[:, :2].pow(2).sum(dim=1).unsqueeze(0)
        + FTp2[:, :2].pow(2).sum(dim=1).unsqueeze(1)
    )  # (M,N)
    sampson = numer.t().pow(2) / (denom + 1e-12)  # (N,M)
    return sampson


def pairwise_stats(dists: torch.Tensor) -> Dict[str, float]:
    """Summary stats for pairwise Sampson matrix."""
    min_row = dists.min(dim=1).values
    min_col = dists.min(dim=0).values
    qrow = [float(torch.quantile(min_row, q)) for q in [0.01, 0.1, 0.5]]
    qcol = [float(torch.quantile(min_col, q)) for q in [0.01, 0.1, 0.5]]
    return {
        "min_row_mean": float(min_row.mean()),
        "min_row_median": float(min_row.median()),
        "min_row_q01_q10_q50": qrow,
        "min_col_mean": float(min_col.mean()),
        "min_col_median": float(min_col.median()),
        "min_col_q01_q10_q50": qcol,
    }


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
        epi_clip=args.epi_clip,
        device=device,
    )
    return solver


def compute_loss(
    solver: OptimalTransportSolver,
    F: torch.Tensor,
    epsilon=None,
    rho=None,
    epi_topk: int | None = None,
    spatial_window: float | None = None,
    dustbin_cost: float | None = None,
    dustbin_mass: float = 1.0,
    mutual_topk: bool = False,
) -> Dict[str, float]:
    cost = solver.compute_cost_matrix(F)
    with torch.no_grad():
        c_median = torch.median(cost).item()
        epsilon_used = max(0.08 * c_median, 1e-3) if epsilon is None else epsilon
        rho_used = (10.0 * epsilon_used) if rho is None else rho
    if spatial_window is not None:
        p1 = solver.means1  # (n1,2)
        p2 = solver.means2  # (n2,2)
        diff = p1[:, None, :] - p2[None, :, :]
        dist2 = (diff**2).sum(dim=2)
        gate_sp = dist2 <= float(spatial_window) ** 2
        cost = torch.where(gate_sp, cost, cost.max() + 1e6)
        mask = gate_sp.clone()
    else:
        mask = torch.ones_like(cost, dtype=torch.bool)
    if epi_topk is not None and epi_topk > 0:
        k_row = min(epi_topk, cost.shape[1])
        idx_row = torch.topk(cost, k=k_row, dim=1, largest=False).indices
        mask_row = torch.zeros_like(cost, dtype=torch.bool)
        mask_row.scatter_(1, idx_row, True)
        if mutual_topk:
            k_col = min(epi_topk, cost.shape[0])
            idx_col = torch.topk(cost, k=k_col, dim=0, largest=False).indices
            mask_col = torch.zeros_like(cost, dtype=torch.bool)
            mask_col.scatter_(0, idx_col, True)
            mask = mask & mask_row & mask_col
        else:
            mask = mask & mask_row
    # ensure each row/col has at least one allowed entry
    row_all_false = mask.sum(dim=1) == 0
    if row_all_false.any():
        idx = torch.argmin(cost[row_all_false], dim=1)
        mask[row_all_false, idx] = True
    col_all_false = mask.sum(dim=0) == 0
    if col_all_false.any():
        idx = torch.argmin(cost[:, col_all_false], dim=0)
        mask[idx, col_all_false] = True
    cost = torch.where(mask, cost, cost.max() + 1e6)
    transport, mass_sums = solver.unbalanced_sinkhorn_algorithm(
        cost,
        epsilon=epsilon,
        rho=rho,
        record_mass=True,
        dustbin_cost=dustbin_cost,
        dustbin_mass=dustbin_mass,
    )
    if dustbin_cost is not None:
        pad_row = torch.full((cost.shape[0], 1), dustbin_cost, device=cost.device, dtype=cost.dtype)
        pad_col = torch.full((1, cost.shape[1] + 1), dustbin_cost, device=cost.device, dtype=cost.dtype)
        cost = torch.cat([torch.cat([cost, pad_row], dim=1), pad_col], dim=0)
    loss = float((transport * cost).sum().item())
    total_mass = float(transport.sum().item())
    avg_cost = loss / (total_mass + 1e-12)
    # Identify inlier region (entries not set to sentinel by clip/top-k)
    gate = cost < cost.max()
    if dustbin_cost is not None:
        gate[-1, :] = False
        gate[:, -1] = False
    inlier_mass = float((transport * gate).sum().item())
    inlier_loss = float((transport * cost * gate).sum().item())
    # KL diagnostics against alpha/beta
    eps = 1e-12
    row_sum = mass_sums[0] if mass_sums else transport.sum(dim=1)
    col_sum = mass_sums[1] if mass_sums else transport.sum(dim=0)
    if dustbin_cost is not None:
        alpha_aug = torch.cat(
            [solver.alpha1, torch.tensor([dustbin_mass], device=solver.alpha1.device, dtype=solver.alpha1.dtype)]
        )
        beta_aug = torch.cat(
            [solver.alpha2, torch.tensor([dustbin_mass], device=solver.alpha2.device, dtype=solver.alpha2.dtype)]
        )
    else:
        alpha_aug = solver.alpha1
        beta_aug = solver.alpha2
    row_min, row_med, row_max = (
        float(row_sum.min().item()),
        float(row_sum.median().item()),
        float(row_sum.max().item()),
    )
    col_min, col_med, col_max = (
        float(col_sum.min().item()),
        float(col_sum.median().item()),
        float(col_sum.max().item()),
    )
    p_row = row_sum / (row_sum.sum() + eps)
    q_row = alpha_aug / (alpha_aug.sum() + eps)
    kl_row = float((p_row * (p_row + eps).log() - p_row * (q_row + eps).log()).sum().item())
    p_col = col_sum / (col_sum.sum() + eps)
    q_col = beta_aug / (beta_aug.sum() + eps)
    kl_col = float((p_col * (p_col + eps).log() - p_col * (q_col + eps).log()).sum().item())

    inlier_avg = inlier_loss / (inlier_mass + 1e-12)
    ent = float((transport * (transport.clamp_min(1e-12).log() - 1.0)).sum().item())
    primal_score = loss + rho_used * (kl_row + kl_col) + epsilon_used * ent

    nan_inf = {
        "nan_log_u": False,  # placeholder for compatibility; solver keeps log_u internal
        "inf_log_u": False,
        "nan_transport": bool(torch.isnan(transport).any().item()),
        "inf_transport": bool(torch.isinf(transport).any().item()),
    }

    return {
        "loss": loss,
        "avg_cost": avg_cost,
        "total_mass": total_mass,
        "kl_row": kl_row,
        "kl_col": kl_col,
        "row_sum_min_med_max": [row_min, row_med, row_max],
        "col_sum_min_med_max": [col_min, col_med, col_max],
        "nan_inf": nan_inf,
        "inlier_mass": inlier_mass,
        "inlier_avg_cost": inlier_avg,
        "entropy": ent,
        "primal_score": primal_score,
        "epsilon_used": epsilon_used,
        "rho_used": rho_used,
    }


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

    # Load Gaussians (supports legacy flat files and per-image directories with gaussians.pkl)
    def resolve_gs(path_root: str, stem: str) -> str:
        cand1 = os.path.join(path_root, f"fitted_gaussians_{stem}.pkl")
        cand2 = os.path.join(path_root, stem, "gaussians.pkl")
        if os.path.exists(cand1):
            return cand1
        if os.path.exists(cand2):
            return cand2
        raise FileNotFoundError(f"Gaussians not found for {stem}: tried {cand1} and {cand2}")

    gs1_path = resolve_gs(gauss_dir, Path(args.image1).stem)
    gs2_path = resolve_gs(gauss_dir, Path(args.image2).stem)

    def load_gs_any(path: str):
        try:
            _, g, _, _ = load_gaussians_torch(path, device=device)
            return g
        except Exception:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, twodgs.TwoDGaussians):
                return obj
            raise

    g1 = load_gs_any(gs1_path)
    g2 = load_gs_any(gs2_path)

    # Optional: rescale Gaussians if they were generated on resized images
    def get_image_size(img_path: Path) -> tuple[int, int]:
        with Image.open(img_path) as im:
            w, h = im.size
        return w, h

    def infer_gs_size(gauss: twodgs.TwoDGaussians, recon_dir: Path) -> tuple[float, float]:
        recon_img = recon_dir / "reconstructed_image.png"
        if recon_img.exists():
            w, h = get_image_size(recon_img)
            return float(w), float(h)
        # fallback: infer from means max (assumed stored as x,y)
        x_max = float(np.max(gauss.means[:, 0])) if gauss.means.size > 0 else 1.0
        y_max = float(np.max(gauss.means[:, 1])) if gauss.means.size > 0 else 1.0
        return x_max + 1.0, y_max + 1.0

    def rescale_gaussians(gauss: twodgs.TwoDGaussians, target_wh: tuple[int, int], recon_dir: Path) -> twodgs.TwoDGaussians:
        tgt_w, tgt_h = target_wh
        src_w, src_h = infer_gs_size(gauss, recon_dir)
        sx = float(tgt_w) / max(src_w, 1e-6)
        sy = float(tgt_h) / max(src_h, 1e-6)
        if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
            return gauss
        # means assumed (x,y)
        gauss.means[:, 0] *= sx  # x
        gauss.means[:, 1] *= sy  # y
        gauss.scales[:, 0] *= sx
        gauss.scales[:, 1] *= sy
        if hasattr(gauss, "covs") and gauss.covs is not None:
            cov_np = np.asarray(gauss.covs)
            S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
            gauss.covs = S @ cov_np @ S.T
        return gauss

    if args.images_dir is not None:
        img1_path = Path(args.images_dir) / args.image1
        img2_path = Path(args.images_dir) / args.image2
        tgt1 = get_image_size(img1_path)
        tgt2 = get_image_size(img2_path)
        g1 = rescale_gaussians(g1, tgt1, Path(gs1_path).parent)
        g2 = rescale_gaussians(g2, tgt2, Path(gs2_path).parent)

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
    pairwise_ref = None
    pairwise_bad_stats = []
    # Optional adaptive clip: choose tau so zero-row率 <= adaptive_clip_pct using GT
    adaptive_zero_row = None
    if args.adaptive_clip_pct is not None and args.epi_clip is None:
        with torch.no_grad():
            dists = pairwise_sampson(F_ref, solver.means1, solver.means2)  # (n1,n2)
            min_row = torch.min(dists, dim=1).values
            q = max(0.0, min(1.0, 1.0 - args.adaptive_clip_pct))
            tau_sq = torch.quantile(min_row, q).item()
            args.epi_clip = math.sqrt(max(tau_sq, 1e-12))
            adaptive_zero_row = float((min_row > tau_sq).float().mean().item())
    if args.pairwise_stats:
        dists = pairwise_sampson(F_ref, solver.means1, solver.means2)
        pairwise_ref = pairwise_stats(dists)

    loss_ref_stats = compute_loss(
        solver,
        F_ref,
        epsilon=args.epsilon,
        rho=args.rho,
        epi_topk=args.epi_topk,
        dustbin_cost=args.dustbin_cost,
        dustbin_mass=args.dustbin_mass,
        mutual_topk=args.mutual_topk,
    )
    mass_pen_ref = (loss_ref_stats.get("kl_row", 0.0) or 0.0) + (loss_ref_stats.get("kl_col", 0.0) or 0.0)
    score_mass_ref = loss_ref_stats.get("avg_cost", 0.0) + args.lambda_mass * mass_pen_ref
    score_inlier_ref = -loss_ref_stats.get("inlier_mass", 0.0) + args.lambda_inlier_avg * loss_ref_stats.get(
        "inlier_avg_cost", 0.0
    )
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
            loss_bad_stats = compute_loss(
                solver,
                F_bad,
                epsilon=args.epsilon,
                rho=args.rho,
                epi_topk=args.epi_topk,
                dustbin_cost=args.dustbin_cost,
                dustbin_mass=args.dustbin_mass,
                mutual_topk=args.mutual_topk,
            )
            mass_pen = (loss_bad_stats.get("kl_row", 0.0) or 0.0) + (loss_bad_stats.get("kl_col", 0.0) or 0.0)
            score1 = loss_bad_stats.get("avg_cost", 0.0) + args.lambda_mass * mass_pen
            score2 = -loss_bad_stats.get("inlier_mass", 0.0) + args.lambda_inlier_avg * loss_bad_stats.get(
                "inlier_avg_cost", 0.0
            )
            if args.pairwise_stats:
                dists_bad = pairwise_sampson(F_bad, solver.means1, solver.means2)
                pw_bad = pairwise_stats(dists_bad)
            else:
                pw_bad = None
            entry = {
                "angle_deg": float(sign * ang),
                **loss_bad_stats,
                "score_mass_pen": score1,
                "score_inlier": score2,
                "pairwise": pw_bad,
            }
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
        "epi_clip": args.epi_clip,
        "epi_topk": args.epi_topk,
        "loss_ref": loss_ref_stats.get("loss"),
        "loss_ref_avg_cost": loss_ref_stats.get("avg_cost"),
        "loss_ref_mass": loss_ref_stats.get("total_mass"),
        "loss_ref_inlier_mass": loss_ref_stats.get("inlier_mass"),
        "loss_ref_inlier_avg_cost": loss_ref_stats.get("inlier_avg_cost"),
        "loss_ref_primal_score": loss_ref_stats.get("primal_score"),
        "loss_ref_entropy": loss_ref_stats.get("entropy"),
        "loss_ref_row_sum_min_med_max": loss_ref_stats.get("row_sum_min_med_max"),
        "loss_ref_col_sum_min_med_max": loss_ref_stats.get("col_sum_min_med_max"),
        "loss_ref_epsilon_used": loss_ref_stats.get("epsilon_used"),
        "loss_ref_rho_used": loss_ref_stats.get("rho_used"),
        "score_mass_pen_ref": score_mass_ref,
        "score_inlier_ref": score_inlier_ref,
        "adaptive_zero_row_pct": adaptive_zero_row,
        "kl_row": loss_ref_stats.get("kl_row"),
        "kl_col": loss_ref_stats.get("kl_col"),
        "pairwise_ref": pairwise_ref,
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
