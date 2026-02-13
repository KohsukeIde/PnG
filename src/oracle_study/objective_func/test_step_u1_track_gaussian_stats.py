#!/usr/bin/env python3
"""
Step 1: Track→Gaussian assignment stats (collision + distance + drop ratio).

Quantifies:
  - Many-to-one collisions (tracks per Gaussian)
  - Assignment distance distribution
  - One-to-one greedy drop ratio
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.colmap_utils import read_images_with_points2d
from src.utils.gaussian_utils import load_gaussians


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


def greedy_one_to_one(pairs: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    scored = sorted(pairs, key=lambda x: x[2])
    used_i = set()
    used_j = set()
    kept: List[Tuple[int, int, float]] = []
    for i, j, d in scored:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        kept.append((i, j, d))
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: track→Gaussian stats")
    parser.add_argument("--pairs", type=str, default="0,10;0,20")
    parser.add_argument("--radius", type=float, default=30.0)
    args = parser.parse_args()

    colmap_dir = os.path.join(PROJECT_ROOT, "data/DTU/scan63/sparse/0")

    def parse_pairs(pairs_str: str) -> List[Tuple[int, int]]:
        out = []
        for part in pairs_str.split(";"):
            if not part.strip():
                continue
            a, b = part.split(",")
            out.append((int(a), int(b)))
        return out

    for idx1, idx2 in parse_pairs(args.pairs):
        kpts1, track_ids1 = load_keypoints_and_tracks(colmap_dir, f"{idx1:04d}.png")
        kpts2, track_ids2 = load_keypoints_and_tracks(colmap_dir, f"{idx2:04d}.png")
        g1 = load_gaussians(idx1)["original_gaussians"]
        g2 = load_gaussians(idx2)["original_gaussians"]
        means1 = g1.means.numpy() if hasattr(g1.means, "numpy") else g1.means
        means2 = g2.means.numpy() if hasattr(g2.means, "numpy") else g2.means

        track_map1 = match_tracks_to_gaussians(means1, kpts1, track_ids1, args.radius)
        track_map2 = match_tracks_to_gaussians(means2, kpts2, track_ids2, args.radius)

        common_tracks = sorted(set(track_map1.keys()) & set(track_map2.keys()))
        pairs = []
        dists = []
        for tid in common_tracks:
            i, d1 = track_map1[tid]
            j, d2 = track_map2[tid]
            pairs.append((i, j, float(d1 + d2)))
            dists.append(float(d1 + d2))

        # collision stats: tracks per Gaussian
        g_counts = {}
        for _, (gi, _) in track_map1.items():
            g_counts[gi] = g_counts.get(gi, 0) + 1
        for _, (gi, _) in track_map2.items():
            g_counts[gi] = g_counts.get(gi, 0) + 1
        counts = np.array(list(g_counts.values())) if g_counts else np.zeros((0,))
        frac_multi = float((counts >= 2).mean()) if counts.size > 0 else 0.0
        max_multi = int(counts.max()) if counts.size > 0 else 0

        # how many tracks are assigned to multi-hit gaussians
        multi_track = 0
        for tid in common_tracks:
            i, _ = track_map1[tid]
            j, _ = track_map2[tid]
            if g_counts.get(i, 0) >= 2 or g_counts.get(j, 0) >= 2:
                multi_track += 1
        frac_track_multi = (multi_track / max(1, len(common_tracks)))

        # one-to-one drop ratio
        kept = greedy_one_to_one(pairs)
        drop_ratio = 1.0 - (len(kept) / max(1, len(pairs)))

        dists = np.array(dists) if dists else np.zeros((0,))
        print("\n" + "=" * 70)
        print("Step 1: track→Gaussian assignment stats")
        print(f"  Pair: ({idx1},{idx2}), radius={args.radius}px")
        print("=" * 70)
        print(f"  common_tracks: {len(common_tracks)}")
        print(f"  pairs: {len(pairs)}, one_to_one kept: {len(kept)} (drop={drop_ratio*100:.1f}%)")
        print(f"  gaussian multi-hit frac: {frac_multi:.3f}, max_hits: {max_multi}")
        print(f"  tracks in multi-hit gaussians: {frac_track_multi:.3f}")
        if dists.size > 0:
            print(
                f"  dist sum (d1+d2): mean={dists.mean():.2f}, "
                f"median={np.median(dists):.2f}, p90={np.quantile(dists, 0.9):.2f}, "
                f"p95={np.quantile(dists, 0.95):.2f}"
            )


if __name__ == "__main__":
    main()
