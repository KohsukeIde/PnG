#!/usr/bin/env python3
"""
Step F: Selection metrics as binary classification.

Evaluate how well each score separates good vs bad poses.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

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
from src.oracle_study.objective_func.test_step_f_score_functions import compute_all_scores
from src.oracle_study.objective_func.test_step_xix_xx_verification import (
    compute_transport_concentration,
    compute_topk_weight_sum,
    compute_topk_cost,
)
def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return float(np.degrees(angle))


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_eigen_gap(
    solver: OptimalTransportSolver,
    R_wc: torch.Tensor,
    transport: torch.Tensor,
    K: np.ndarray,
    top_k: int = 50,
    min_mass: float = 0.001,
) -> float:
    """Compute eigenvalue gap from closed-form t estimation."""
    means1 = solver.gaussians1.means
    means2 = solver.gaussians2.means
    K_inv = np.linalg.inv(K)

    T_np = transport.detach().cpu().numpy()
    T_flat = T_np.flatten()
    top_indices = np.argsort(T_flat)[-top_k:]

    n1, n2 = T_np.shape
    weights = []
    a_vectors = []

    R_np = R_wc.detach().cpu().numpy()

    for idx in top_indices:
        i = idx // n2
        j = idx % n2
        w = T_flat[idx]
        if w < min_mass:
            continue

        p1 = means1[i]
        p2 = means2[j]

        if isinstance(p1, torch.Tensor):
            p1 = p1.numpy()
            p2 = p2.numpy()

        x1_hom = np.array([p1[0], p1[1], 1.0])
        x2_hom = np.array([p2[0], p2[1], 1.0])

        x1_norm = K_inv @ x1_hom
        x2_norm = K_inv @ x2_hom

        Rx1 = R_np @ x1_norm
        a = np.cross(x2_norm, Rx1)

        weights.append(w)
        a_vectors.append(a)

    if len(weights) < 3:
        return 0.0

    weights = np.array(weights)
    a_vectors = np.array(a_vectors)

    M = np.zeros((3, 3))
    for w, a in zip(weights, a_vectors):
        M += w * np.outer(a, a)

    try:
        eigvals = np.linalg.eigvalsh(M)
        eigvals = np.sort(eigvals)
        gap = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        return float(gap)
    except Exception:
        return 0.0


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


def random_rotation_matrix(max_angle_deg: float) -> np.ndarray:
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    angle = np.random.uniform(0, np.deg2rad(max_angle_deg))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def random_unit_vector() -> np.ndarray:
    v = np.random.randn(3)
    return v / (np.linalg.norm(v) + 1e-10)


def roc_auc(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]

    tp = 0
    fp = 0
    pos = int(labels.sum())
    neg = len(labels) - pos
    tpr = [0.0]
    fpr = [0.0]

    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / max(pos, 1))
        fpr.append(fp / max(neg, 1))

    return float(np.trapezoid(tpr, fpr))


def pr_auc(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]

    tp = 0
    fp = 0
    pos = int(labels.sum())
    precisions = [1.0]
    recalls = [0.0]

    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(pos, 1)
        precisions.append(precision)
        recalls.append(recall)

    return float(np.trapezoid(precisions, recalls))


def topk_recall(scores: np.ndarray, labels: np.ndarray, k: int, higher_is_better: bool) -> float:
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    if higher_is_better:
        order = order[::-1]
    top = order[:k]
    return float(labels[top].sum() / labels.sum())


def compute_sample_metrics(
    solver: OptimalTransportSolver,
    R: np.ndarray,
    t: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    epsilon: float,
    rho: float,
    top_k: int,
) -> Dict[str, float]:
    R_t = torch.tensor(R, dtype=torch.float32)
    t_t = torch.tensor(t, dtype=torch.float32)
    t_t = t_t / (t_t.norm() + 1e-10)

    F = solver._build_F_from_wc(R_t, t_t)
    C = solver.compute_cost_matrix(F)

    with torch.no_grad():
        T, _ = solver.unbalanced_sinkhorn_algorithm(
            C, epsilon=epsilon, rho=rho,
            gate_mask=solver._last_gate_mask,
        )

    eps_actual = solver._last_sinkhorn_epsilon
    rho_actual = solver._last_sinkhorn_rho

    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()
    scores = compute_all_scores(T, C, a, b, eps_actual, rho_actual)

    concentration = compute_transport_concentration(T)
    topk_sum = compute_topk_weight_sum(T, top_k)
    topk_cost = compute_topk_cost(T, C, top_k)
    eigen_gap = compute_eigen_gap(solver, R_t, T, solver.k1.cpu().numpy(), top_k=top_k)

    R_err = rotation_error(R, R_gt)
    t_err = min(translation_error(t, t_gt), translation_error(-t, t_gt))

    geom_uot = scores["uot_cost_term"] + scores["uot_kl_term"]

    return {
        "avg_cost": scores["avg_cost"],
        "mass_aware": scores["mass_aware"],
        "full_uot": scores["full_uot"],
        "geom_uot": geom_uot,
        "T_sum": scores["T_sum"],
        "entropy": scores["entropy"],
        "concentration": concentration,
        "topk_sum": topk_sum,
        "topk_cost": topk_cost,
        "eigen_gap": eigen_gap,
        "R_err": R_err,
        "t_err": t_err,
    }


def run_selection_metrics(
    idx1: int,
    idx2: int,
    n_samples: int,
    max_R_angle: float,
    epsilon: float,
    rho: float,
    top_k: int,
    r_thresh: float,
    t_thresh: float,
    t_mode: str,
    seed: int,
):
    print("\n" + "=" * 70)
    print("Step F: Selection metrics (classification)")
    print(f"  Pair: ({idx1}, {idx2}), n_samples={n_samples}")
    print(f"  epsilon={epsilon}, rho={rho}, max_R_angle={max_R_angle}")
    print(f"  r_thresh={r_thresh}, t_thresh={t_thresh}, t_mode={t_mode}")
    print("=" * 70)

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    solver = OptimalTransportSolver(
        gaussians1=g1, gaussians2=g2,
        k1=K, k2=K, device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0, lambda_cov=0.0, lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        ot_mass1=ot_mass1, ot_mass2=ot_mass2,
    )

    samples: List[Dict[str, float]] = []

    # Random samples
    for _ in range(n_samples):
        R = random_rotation_matrix(max_R_angle)
        if t_mode == "fixed":
            t = t_gt
        else:
            t = random_unit_vector()
        samples.append(compute_sample_metrics(
            solver, R, t, R_gt, t_gt, epsilon, rho, top_k
        ))

    # Near-GT samples (10%)
    n_near = max(1, n_samples // 10)
    for _ in range(n_near):
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        angle = np.random.uniform(0, np.deg2rad(15))
        K_mat = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        R_perturb = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
        R = R_perturb @ R_gt
        t = t_gt if t_mode == "fixed" else random_unit_vector()
        samples.append(compute_sample_metrics(
            solver, R, t, R_gt, t_gt, epsilon, rho, top_k
        ))

    # Labels
    R_errs = np.array([s["R_err"] for s in samples])
    t_errs = np.array([s["t_err"] for s in samples])
    if t_thresh > 0:
        labels = (R_errs < r_thresh) & (t_errs < t_thresh)
    else:
        labels = R_errs < r_thresh

    n_good = int(labels.sum())
    print(f"  Samples: {len(samples)}, good={n_good}")

    metric_defs = {
        "avg_cost": False,
        "mass_aware": False,
        "full_uot": False,
        "geom_uot": False,
        "T_sum": True,
        "concentration": True,
        "topk_sum": True,
        "topk_cost": False,
        "eigen_gap": True,
    }

    print("\n  Metrics (ROC-AUC / PR-AUC / recall@5 / recall@10)")
    print("  " + "-" * 72)
    for name, higher_is_better in metric_defs.items():
        values = np.array([s[name] for s in samples])
        roc = roc_auc(values, labels, higher_is_better)
        pr = pr_auc(values, labels, higher_is_better)
        r5 = topk_recall(values, labels, 5, higher_is_better)
        r10 = topk_recall(values, labels, 10, higher_is_better)
        print(f"  {name:<12} {roc:>7.3f} {pr:>7.3f} {r5:>8.3f} {r10:>9.3f}")


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
    parser = argparse.ArgumentParser(description="Selection metrics as classification")
    parser.add_argument("--pairs", type=str, default="0,10",
                        help="Pairs as 'i,j;k,l' (default: 0,10)")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-R-angle", type=float, default=90.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--r-thresh", type=float, default=10.0)
    parser.add_argument("--t-thresh", type=float, default=0.0)
    parser.add_argument("--t-mode", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    for idx1, idx2 in pairs:
        run_selection_metrics(
            idx1=idx1,
            idx2=idx2,
            n_samples=args.n_samples,
            max_R_angle=args.max_R_angle,
            epsilon=args.epsilon,
            rho=args.rho,
            top_k=args.top_k,
            r_thresh=args.r_thresh,
            t_thresh=args.t_thresh,
            t_mode=args.t_mode,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
