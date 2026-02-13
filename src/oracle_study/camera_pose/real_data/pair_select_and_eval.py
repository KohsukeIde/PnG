#!/usr/bin/env python3
"""Select image pairs with SIFT overlap and run loss_eval on them.

Why:
- 過去は手動でペアを選んでいたが、視野共有が小さいペアが混ざりやすい。
- SIFTベースの類似度で「重なりがありそうなペア」を自動抽出し、その上で既存の
  loss_eval.py を一括実行する。

Outputs:
- pairs.json: 選ばれたペアとスコアの一覧
- results/*.json: loss_eval.py の出力
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]

def load_gt_centers(gt_npz: Path) -> Dict[str, np.ndarray]:
    """Load camera centers from DTU cameras.npz using projection matrix decomposition.

    Note: world_mat is a projection matrix (K[R|t]). We use cv2.decomposeProjectionMatrix
    to extract K, R, and camera center. Avoids the old world_mat_inv=c2w assumption.
    """
    if not gt_npz.exists():
        return {}
    data = np.load(gt_npz)
    centers: Dict[str, np.ndarray] = {}
    idx = 0
    while f"world_mat_{idx}" in data:
        P = data[f"world_mat_{idx}"][:3, :4].astype(np.float64)  # projection matrix
        _, R, t_homog, _, _, _, _ = cv2.decomposeProjectionMatrix(P)
        C = (t_homog[:3, 0] / t_homog[3, 0]).astype(np.float64)  # camera center in world
        centers[f"{idx:04d}.png"] = C
        idx += 1
    return centers

def list_images(
    images_dir: Path, max_images: int | None, gauss_dir: Path | None = None
) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    imgs = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    if gauss_dir is not None:
        filtered = []
        for p in imgs:
            stem = p.stem
            cand_dir = gauss_dir / stem / "gaussians.pkl"
            cand_file = gauss_dir / f"fitted_gaussians_{stem}.pkl"
            if cand_dir.exists() or cand_file.exists():
                filtered.append(p)
        imgs = filtered
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


def compute_sift(path: Path, max_features: int) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None:
        desc = np.empty((0, 128), dtype=np.float32)
    return np.array(kps, dtype=object), desc.astype(np.float32)


def pair_score(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75) -> int:
    if desc1.size == 0 or desc2.size == 0:
        return 0
    index_params = dict(algorithm=1, trees=5)  # FLANN KDTree
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if n is not None and m.distance < ratio * n.distance]
    return len(good)


def select_pairs(
    images: Sequence[Path],
    descs: Dict[Path, np.ndarray],
    top_k: int,
    min_matches: int,
    min_index_gap: int | None = None,
    min_baseline: float | None = None,
    centers: Dict[str, np.ndarray] | None = None,
) -> List[Tuple[Path, Path, int]]:
    scored: List[Tuple[Path, Path, int]] = []
    for a, b in itertools.combinations(images, 2):
        if min_index_gap is not None:
            try:
                ia = int(a.stem)
                ib = int(b.stem)
                if abs(ia - ib) < min_index_gap:
                    continue
            except ValueError:
                pass
        if min_baseline is not None and centers is not None:
            ca = centers.get(a.name)
            cb = centers.get(b.name)
            if ca is not None and cb is not None:
                if np.linalg.norm(ca - cb) < min_baseline:
                    continue
        s = pair_score(descs[a], descs[b])
        if s >= min_matches:
            scored.append((a, b, s))
    scored.sort(key=lambda x: x[2], reverse=True)
    if top_k is not None and top_k > 0:
        scored = scored[:top_k]
    return scored


def run_loss_eval(
    pair: Tuple[Path, Path],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    img1, img2 = pair
    out = output_dir / f"{img1.stem}_{img2.stem}.json"
    cmd = [
        "python",
        str(REPO_ROOT / "src" / "oracle_study" / "camera_pose" / "real_data" / "loss_eval.py"),
        "--gaussians-dir",
        str(args.gaussians_dir),
        "--image1",
        img1.name,
        "--image2",
        img2.name,
        "--angles-deg",
        args.angles_deg,
        "--epsilon",
        str(args.epsilon),
        "--rho",
        str(args.rho),
        "--sigma-epipolar",
        str(args.sigma_epipolar),
        "--lambda-epipolar",
        str(args.lambda_epipolar),
        "--lambda-color",
        str(args.lambda_color),
        "--lambda-cov",
        str(args.lambda_cov),
        "--epipolar-mode",
        args.epipolar_mode,
        "--device",
        args.device,
        "--output",
        str(out),
        "--fixed-nn-epi",
    ]
    if args.use_gt_cameras:
        cmd.append("--use-gt-cameras")
    if args.images_dir is not None:
        cmd.extend(["--images-dir", str(args.images_dir)])
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select overlap-heavy pairs and run loss_eval.")
    p.add_argument("--images-dir", type=Path, required=True, help="Directory of input images (.png/.jpg).")
    p.add_argument("--gaussians-dir", type=Path, required=True, help="Directory of fitted Gaussians (legacy pkl).")
    p.add_argument("--use-gt-cameras", action="store_true", help="Use GT cameras.npz in loss_eval.")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images to consider.")
    p.add_argument("--max-features", type=int, default=4000, help="SIFT feature cap per image.")
    p.add_argument("--min-matches", type=int, default=40, help="Minimum good matches to keep a pair.")
    p.add_argument("--top-k", type=int, default=20, help="Keep top-K pairs by match count.")
    p.add_argument("--min-index-gap", type=int, default=2, help="Discard pairs with |i-j| smaller than this (if stems are int).")
    p.add_argument("--min-baseline", type=float, default=None, help="Minimum camera-center distance (requires GT).")
    p.add_argument("--angles-deg", type=str, default="60", help="Angles to perturb F (passed to loss_eval).")
    p.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn epsilon.")
    p.add_argument("--rho", type=float, default=0.5, help="Sinkhorn rho.")
    p.add_argument("--sigma-epipolar", type=float, default=400.0, help="Epipolar sigma.")
    p.add_argument("--lambda-epipolar", type=float, default=1.0, help="Epipolar weight.")
    p.add_argument("--lambda-color", type=float, default=0.0, help="Color weight (passed through).")
    p.add_argument("--lambda-cov", type=float, default=0.0, help="Cov weight (passed through).")
    p.add_argument("--epipolar-mode", type=str, default="sampson", choices=["sampson", "sed"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=Path, default=Path("src/oracle_study/camera_pose/real_data/results"))
    p.add_argument("--pairs-json", type=Path, default=None, help="Optional precomputed pairs.json to reuse.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir if args.images_dir.is_absolute() else (REPO_ROOT / args.images_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = {}
    if args.use_gt_cameras:
        centers = load_gt_centers(REPO_ROOT / "data" / "DTU" / "scan63" / "cameras.npz")

    if args.pairs_json and args.pairs_json.exists():
        pairs_data = json.loads(args.pairs_json.read_text())
        pairs = [(Path(p[0]), Path(p[1]), int(p[2])) for p in pairs_data["pairs"]]
    else:
        images = list_images(images_dir, args.max_images, gauss_dir=args.gaussians_dir)
        descs: Dict[Path, np.ndarray] = {}
        for img in images:
            _, d = compute_sift(img, args.max_features)
            descs[img] = d
        selected = select_pairs(
            images,
            descs,
            args.top_k,
            args.min_matches,
            min_index_gap=args.min_index_gap,
            min_baseline=args.min_baseline,
            centers=centers if centers else None,
        )
        pairs = selected
        pairs_json = {
            "images_dir": str(images_dir),
            "min_matches": args.min_matches,
            "top_k": args.top_k,
            "min_index_gap": args.min_index_gap,
            "min_baseline": args.min_baseline,
            "pairs": [(str(a.name), str(b.name), score) for a, b, score in selected],
        }
        (output_dir / "pairs.json").write_text(json.dumps(pairs_json, indent=2))

    for a, b, score in pairs:
        print(f"Evaluating pair {a.name}-{b.name} (matches={score})")
        run_loss_eval((a, b), args, output_dir)


if __name__ == "__main__":
    main()


