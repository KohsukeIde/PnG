#!/usr/bin/env python3
"""
Step T: Covariance repeatability across views (track-based).

For shared COLMAP tracks, match to nearest Gaussians and compare
covariance features (orientation, scale, eigenratio) across views.
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

from src.utils.colmap_utils import (
    read_images_with_points2d,
)
from src.utils.gaussian_utils import load_gaussians
from src.oracle_study.objective_func.test_structural_descriptors_analysis import (
    extract_cov_features,
)


def load_keypoints_and_tracks(colmap_dir: str, image_name: str) -> Tuple[np.ndarray, np.ndarray]:
    images_bin_path = os.path.join(colmap_dir, "images.bin")
    images_with_pts = read_images_with_points2d(images_bin_path)

    img_data = None
    for img in images_with_pts.values():
        if img["name"] == image_name:
            img_data = img
            break
    if img_data is None:
        raise ValueError(f"Image {image_name} not found in images.bin")

    kpts = []
    track_ids = []
    for pt in img_data["points2d"]:
        if pt["point3d_id"] != -1:
            kpts.append([pt["x"], pt["y"]])
            track_ids.append(pt["point3d_id"])
    if len(kpts) == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=np.int64)

    return np.array(kpts), np.array(track_ids)


def match_tracks_to_gaussians(
    means: np.ndarray,
    kpts: np.ndarray,
    track_ids: np.ndarray,
    radius: float,
) -> Dict[int, Tuple[int, float]]:
    if kpts.shape[0] == 0:
        return {}
    diffs = kpts[:, None, :] - means[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = np.min(dists, axis=1)
    track_map: Dict[int, Tuple[int, float]] = {}
    for kp_idx, (gi, dist) in enumerate(zip(nearest_idx, nearest_dist)):
        if dist > radius:
            continue
        track_id = int(track_ids[kp_idx])
        track_map[track_id] = (int(gi), float(dist))
    return track_map


def angle_diff(a: float, b: float) -> float:
    """Smallest angle difference in radians (mod pi)."""
    diff = np.abs(a - b)
    diff = np.mod(diff, np.pi)
    return min(diff, np.pi - diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step T: covariance repeatability")
    parser.add_argument("--idx1", type=int, default=0)
    parser.add_argument("--idx2", type=int, default=10)
    parser.add_argument("--radius", type=float, default=30.0)
    parser.add_argument("--min-eigenratio", type=float, default=1.2)
    args = parser.parse_args()

    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")
    kpts1, track_ids1 = load_keypoints_and_tracks(colmap_dir, f"{args.idx1:04d}.png")
    kpts2, track_ids2 = load_keypoints_and_tracks(colmap_dir, f"{args.idx2:04d}.png")

    data1 = load_gaussians(args.idx1)
    data2 = load_gaussians(args.idx2)
    g1 = data1["original_gaussians"]
    g2 = data2["original_gaussians"]

    means1 = g1.means.numpy() if isinstance(g1.means, torch.Tensor) else g1.means
    means2 = g2.means.numpy() if isinstance(g2.means, torch.Tensor) else g2.means
    covs1 = g1.covs.numpy() if isinstance(g1.covs, torch.Tensor) else g1.covs
    covs2 = g2.covs.numpy() if isinstance(g2.covs, torch.Tensor) else g2.covs

    track_map1 = match_tracks_to_gaussians(means1, kpts1, track_ids1, args.radius)
    track_map2 = match_tracks_to_gaussians(means2, kpts2, track_ids2, args.radius)

    common_tracks = sorted(set(track_map1.keys()) & set(track_map2.keys()))
    if not common_tracks:
        print("No common tracks after mapping.")
        return

    feats1 = extract_cov_features(covs1)
    feats2 = extract_cov_features(covs2)

    angle_diffs = []
    ratio_diffs = []
    scale_diffs = []
    used = 0

    for track_id in common_tracks:
        i = track_map1[track_id][0]
        j = track_map2[track_id][0]

        ratio1 = feats1["eigenratio"][i]
        ratio2 = feats2["eigenratio"][j]
        if max(ratio1, ratio2) < args.min_eigenratio:
            continue

        ang1 = feats1["orientation"][i]
        ang2 = feats2["orientation"][j]
        angle_diffs.append(np.degrees(angle_diff(ang1, ang2)))

        ratio_diffs.append(abs(np.log(ratio1 + 1e-10) - np.log(ratio2 + 1e-10)))
        scale1 = feats1["scale"][i]
        scale2 = feats2["scale"][j]
        scale_diffs.append(abs(np.log(scale1 + 1e-10) - np.log(scale2 + 1e-10)))
        used += 1

    if used == 0:
        print("No tracks passed eigenratio filter.")
        return

    angle_diffs = np.array(angle_diffs)
    ratio_diffs = np.array(ratio_diffs)
    scale_diffs = np.array(scale_diffs)

    print("\n" + "=" * 70)
    print("Step T: covariance repeatability")
    print(f"  Pair: ({args.idx1}, {args.idx2}), radius={args.radius}px")
    print(f"  used tracks: {used}/{len(common_tracks)} (eigenratio>={args.min_eigenratio})")
    print("=" * 70)
    print(f"Orientation diff (deg): mean={angle_diffs.mean():.2f}, median={np.median(angle_diffs):.2f}")
    print(f"Log eigenratio diff:   mean={ratio_diffs.mean():.3f}, median={np.median(ratio_diffs):.3f}")
    print(f"Log scale diff:        mean={scale_diffs.mean():.3f}, median={np.median(scale_diffs):.3f}")


if __name__ == "__main__":
    main()
