#!/usr/bin/env python3
"""
Step S: 2DGS->3DGS registration test (PnG-style).

Triangulate 3D points from COLMAP tracks using two views,
create simple 3D Gaussians, and estimate pose of a new view
using ViewpointExtender + OT.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.reconstructor.viewpoint_extender import ViewpointExtender
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
    read_images_with_points2d,
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
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


def load_tracks(colmap_dir: str, image_name: str) -> Dict[int, np.ndarray]:
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)
    img_data = None
    for img in images_with_pts.values():
        if img["name"] == image_name:
            img_data = img
            break
    if img_data is None:
        raise ValueError(f"Image {image_name} not found in images.bin")
    tracks: Dict[int, np.ndarray] = {}
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


def build_3d_gaussians(points_3d: np.ndarray, sigma: float) -> List[Dict[str, np.ndarray]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Step S: PnG registration test")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--idx-new", type=int, default=20)
    parser.add_argument("--max-tracks", type=int, default=500)
    parser.add_argument("--sigma-3d", type=float, default=0.01)
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
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
    device = torch.device(args.device)

    extender = ViewpointExtender(
        existing_3d_gaussians=existing_3d,
        camera_params_list=[(cam1["R"], cam1["t"]), (cam2["R"], cam2["t"])],
        K_new=cam_new["K"],
        reference_camera_idx=0,
        device=device,
    )
    projected_2d = extender.project_3d_gaussians()
    extender.initialize_transport_solver(projected_2d, new_gauss)
    solver = extender.transport_solver
    solver.optimize_with_SE3(
        max_iter=args.max_iters,
        tol=1e-6,
        save_diagnostics=False,
        differentiable_transport=False,
        sinkhorn_epsilon=0.05,
        sinkhorn_rho=0.5,
        score_type="avg_cost",
        epsilon_annealing=True,
        epsilon_start=0.2,
        epsilon_end=0.05,
        anneal_steps=min(50, args.max_iters),
        optimize_mode="both",
    )

    with torch.no_grad():
        se3_vec = torch.cat([solver.rot_vec, solver.trans_vec])
        T_cw = solver.lie.se3_to_SE3(se3_vec)
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]
        R_wc = R_cw.t()
        t_wc = -R_wc @ t_cw

    R_ref, t_ref = extender.camera_params_list[extender.reference_camera_idx]
    R_est = R_wc.numpy()
    t_est = t_wc.numpy()

    R_rel_gt = cam_new["R"] @ cam1["R"].T
    t_rel_gt = cam_new["t"] - R_rel_gt @ cam1["t"]
    t_rel_gt = t_rel_gt / (np.linalg.norm(t_rel_gt) + 1e-10)

    r_err = rotation_error(R_est, R_rel_gt)
    t_err = min(
        translation_error(t_est, t_rel_gt),
        translation_error(-t_est, t_rel_gt),
    )

    print("\n" + "=" * 70)
    print("Step S: PnG registration test")
    print(f"  Base views: ({args.idx1}, {args.idx2}), new view: {args.idx_new}")
    print(f"  Triangulated points: {len(existing_3d)}")
    print(f"  R_err(rel)={r_err:.2f}deg, t_err(rel)={t_err:.2f}deg")
    print("=" * 70)


if __name__ == "__main__":
    main()
