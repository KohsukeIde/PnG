#!/usr/bin/env python3
"""Camera pose oracle evaluation for SE(3) and S³×S² optimizers."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.oracle_study.core import ToyProblemGenerator
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver


def rotation_error_deg(r_est: np.ndarray, r_gt: np.ndarray) -> float:
    """Geodesic rotation error in degrees."""
    mat = r_gt.T @ r_est
    trace = np.clip((np.trace(mat) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angular error between translation directions in degrees."""
    norm_est = np.linalg.norm(t_est)
    norm_gt = np.linalg.norm(t_gt)
    if norm_est < 1e-9 or norm_gt < 1e-9:
        return float("nan")
    cos_val = np.clip(np.dot(t_est / norm_est, t_gt / norm_gt), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))



def translation_direction_error_deg_pm(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Angular error when considering ±t equivalence."""
    def ang(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(float(a.dot(b)), -1.0, 1.0))))
    return min(ang(t_est, t_gt), ang(-t_est, t_gt))

def normalize_fundamental(F: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(F)
    if norm < 1e-12:
        return F
    return F / norm


def sampson_residuals(F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Return Sampson residuals for matched correspondences."""
    ones = np.ones((pts1.shape[0], 1), dtype=np.float64)
    x1 = np.hstack([pts1, ones])
    x2 = np.hstack([pts2, ones])

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T
    numer = np.sum(x2 * (F @ x1.T).T, axis=1)
    denom = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-9
    return numer ** 2 / denom


def diagonal_concentration(T: torch.Tensor) -> float:
    rows, cols = T.shape
    min_dim = min(rows, cols)
    if min_dim == 0:
        return float("nan")
    idx = torch.arange(min_dim, device=T.device)
    diag_sum = T[idx, idx].sum()
    total = T.sum()
    if total.abs() < 1e-12:
        return float("nan")
    return float((diag_sum / total).item())


def collect_all_scenarios() -> Dict[str, Dict[str, np.ndarray]]:
    scenarios: Dict[str, Dict[str, np.ndarray]] = {}
    providers = [
        ToyProblemGenerator.get_epipolar_standard_scenarios,
        ToyProblemGenerator.get_epipolar_baseline_scenarios,
        ToyProblemGenerator.get_epipolar_challenging_scenarios,
        ToyProblemGenerator.get_epipolar_noise_scenarios,
        ToyProblemGenerator.get_epipolar_illumination_scenarios,
        ToyProblemGenerator.get_epipolar_occlusion_scenarios,
    ]
    for provider in providers:
        scenarios.update(provider())
    return scenarios


def apply_scenario_effects(
    generator: ToyProblemGenerator,
    g1,
    g2,
    scenario_name: str,
    scenario_params: Dict[str, np.ndarray],
    seed: int,
):
    g1_mod, g2_mod = generator.apply_epipolar_scenario_effects(g1, g2, scenario_params, seed=seed)
    if "color_change" in scenario_name:
        rng = np.random.RandomState(seed ^ 0xABCDEF)
        g2_mod.rgb = rng.uniform(0.0, 1.0, size=g2_mod.rgb.shape).astype(np.float32)
    return g1_mod, g2_mod


def run_se3(
    solver_kwargs: dict,
    se3_kwargs: dict,
    init_mode: str,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    seed: int,
) -> Tuple[List[float], OptimalTransportSolver]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = OptimalTransportSolver(**solver_kwargs)
    if init_mode == "gt":
        with torch.no_grad():
            # Create SE(3) matrix in [R|t] format (3x4)
            T_gt = np.zeros((3, 4), dtype=np.float32)
            T_gt[:3, :3] = R_gt.astype(np.float32)
            T_gt[:3, 3] = t_gt.astype(np.float32)
            se3_vec = solver.lie.SE3_to_se3(torch.tensor(T_gt, device=solver.device))
            rot_init = se3_vec[:3] + 1e-3 * torch.randn(3, device=solver.device)
            trans_init = se3_vec[3:] + 1e-3 * torch.randn(3, device=solver.device)
            solver.rot_vec = torch.nn.Parameter(rot_init)
            solver.trans_vec = torch.nn.Parameter(trans_init)
    loss_history = solver.optimize_with_SE3(save_diagnostics=False, **se3_kwargs)
    return [float(v) for v in loss_history], solver


def run_geoopt(
    solver_kwargs: dict,
    geoopt_kwargs: dict,
    init_mode: str,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    seed: int,
) -> Tuple[List[float], OptimalTransportSolver]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = OptimalTransportSolver(**solver_kwargs)
    if init_mode == "gt":
        with torch.no_grad():
            R_tensor = torch.tensor(R_gt, dtype=torch.float32, device=solver.device)
            t_hat = torch.tensor(t_gt / (np.linalg.norm(t_gt) + 1e-9), dtype=torch.float32, device=solver.device)
            solver.R_wc_param = R_tensor
            solver.t_hat_param = t_hat
    loss_history = solver.optimize_with_essential_geoopt(save_diagnostics=False, **geoopt_kwargs)
    return [float(v) for v in loss_history], solver


def evaluate_method(
    method: str,
    solver_kwargs: dict,
    se3_kwargs: dict,
    geoopt_kwargs: dict,
    metrics_base: dict,
    method_dir: Path,
    seed: int,
    init_mode: str,
    pts1: np.ndarray,
    pts2: np.ndarray,
    correspondences: np.ndarray,
    F_gt: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> dict:
    start = time.perf_counter()
    if method == "se3":
        loss_history, solver = run_se3(solver_kwargs, se3_kwargs, init_mode, R_gt, t_gt, seed)
    else:
        loss_history, solver = run_geoopt(solver_kwargs, geoopt_kwargs, init_mode, R_gt, t_gt, seed)
    duration = time.perf_counter() - start

    F_est = solver.f.detach().cpu().numpy()
    R_est = solver.R_wc.detach().cpu().numpy()
    t_est = solver.t_wc.detach().cpu().numpy()

    sampson = sampson_residuals(F_est, pts1[correspondences[:, 0]], pts2[correspondences[:, 1]])
    C_est = solver.compute_cost_matrix(solver.f)
    T_est = solver.unbalanced_sinkhorn_algorithm(C_est)
    t_tensor = torch.tensor(t_est, dtype=solver.f.dtype, device=solver.device)
    R_tensor = torch.tensor(R_est, dtype=solver.f.dtype, device=solver.device)
    E_est = solver.lie.skew_symmetric(t_tensor) @ R_tensor
    E_singulars = np.linalg.svd(E_est.detach().cpu().numpy(), compute_uv=False)

    with torch.no_grad():
        F_pos = solver._build_F_from_wc(solver.R_wc, solver.t_wc)
        F_neg = solver._build_F_from_wc(solver.R_wc, -solver.t_wc)
        C_pos = solver.compute_cost_matrix(F_pos)
        C_neg = solver.compute_cost_matrix(F_neg)
        T_pos = solver.unbalanced_sinkhorn_algorithm(C_pos.detach())
        T_neg = solver.unbalanced_sinkhorn_algorithm(C_neg.detach())
        loss_pos = float((T_pos * C_pos).sum().item())
        loss_neg = float((T_neg * C_neg).sum().item())

    metrics = {
        **metrics_base,
        "method": method,
        "init_mode": init_mode,
        "duration_sec": duration,
        "iterations": len(loss_history),
        "final_loss": float(loss_history[-1]) if loss_history else float("nan"),
        "rotation_error_deg": rotation_error_deg(R_est, R_gt),
        "translation_direction_error_deg": translation_direction_error_deg(t_est, t_gt),
        "translation_direction_error_pm_deg": translation_direction_error_deg_pm(t_est, t_gt),
        "mean_sampson_error": float(np.mean(sampson)),
        "median_sampson_error": float(np.median(sampson)),
        "max_sampson_error": float(np.max(sampson)),
        "diag_concentration": diagonal_concentration(T_est),
        "frobenius_f_error": float(np.linalg.norm(normalize_fundamental(F_est) - normalize_fundamental(F_gt))),
        "E_singular_values": E_singulars.tolist(),
        "loss_history": loss_history,
        "loss_t_positive": loss_pos,
        "loss_t_negative": loss_neg,
        "loss_t_delta": loss_pos - loss_neg,
    }

    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    np.save(method_dir / "F_est.npy", F_est)
    np.save(method_dir / "R_est.npy", R_est)
    np.save(method_dir / "t_est.npy", t_est)
    np.save(method_dir / "sampson_residuals.npy", sampson)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera pose oracle evaluation")
    parser.add_argument("--scenario", default="baseline_yaw", help="Scenario name or 'all'")
    parser.add_argument("--device", default="cpu", help="Torch device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epipolar-mode", choices=["sed", "sampson", "hybrid"], default="hybrid")
    parser.add_argument("--lambda-color", type=float, default=1.0)
    parser.add_argument("--lambda-epipolar", type=float, default=1.0)
    parser.add_argument("--lambda-cov", type=float, default=0.3)
    # Recommended defaults from matching oracle analysis (K ≈ 800)
    parser.add_argument("--sigma-epipolar", type=float, default=400.0)
    parser.add_argument("--sigma-color", type=float, default=0.5)
    parser.add_argument("--sigma-cov", type=float, default=8.0)
    parser.add_argument("--noise-model", choices=["gaussian", "cauchy", "huber"], default="gaussian")
    parser.add_argument("--lambda-cheirality", type=float, default=0.0, help="Weight for cheirality regularisation")
    parser.add_argument("--cheirality-topk", type=int, default=3, help="Top-k entries per row used for cheirality loss (0 to use all)")
    parser.add_argument("--init-mode", choices=["random", "gt"], default="random", help="Initialisation strategy (random or near-GT)")
    parser.add_argument("--se3-max-iter", type=int, default=500)
    parser.add_argument("--se3-rot-lr", type=float, default=5e-3)
    parser.add_argument("--se3-trans-lr", type=float, default=5e-4)
    parser.add_argument("--geoopt-max-iter", type=int, default=1500)
    parser.add_argument("--geoopt-lr", type=float, default=3e-3)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--results-root", default=None, help="Output directory root")
    parser.add_argument("--skip-se3", action="store_true", help="Skip SE(3) baseline")
    parser.add_argument("--skip-geoopt", action="store_true", help="Skip S³×S² optimizer")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    generator = ToyProblemGenerator(seed=args.seed)
    scenarios = collect_all_scenarios()
    if args.scenario.lower() == "all":
        scenario_items = list(scenarios.items())
    else:
        if args.scenario not in scenarios:
            raise ValueError(f"Unknown scenario '{args.scenario}'. Available keys: {sorted(scenarios.keys())}")
        scenario_items = [(args.scenario, scenarios[args.scenario])]

    results_root = Path(args.results_root) if args.results_root else Path(__file__).resolve().parents[1] / "results"

    for scenario_name, params in scenario_items:
        if args.verbose:
            print(f"\n=== Scenario: {scenario_name} ===")
        R_wc_gt = params["R_wc"].astype(np.float64)
        t_wc_gt = params["t_wc"].astype(np.float64)

        K = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        g1, g2, correspondences, F_gt = generator.generate_epipolar_correspondences(
            n_gaussians=20,
            K=K,
            R_wc=R_wc_gt,
            t_wc=t_wc_gt,
        )
        g1, g2 = apply_scenario_effects(generator, g1, g2, scenario_name, params, seed=args.seed)

        pts1 = g1.means.astype(np.float64)
        pts2 = g2.means.astype(np.float64)
        scenario_dir = results_root / scenario_name

        solver_kwargs = {
            "gaussians1": g1,
            "gaussians2": g2,
            "k1": K,
            "k2": K,
            "epsilon": 0.01,
            "lambda_color": args.lambda_color,
            "lambda_epipolar": args.lambda_epipolar,
            "lambda_cov": args.lambda_cov,
            "sigma_epipolar": args.sigma_epipolar,
            "sigma_color": args.sigma_color,
            "sigma_cov": args.sigma_cov,
            "noise_model": args.noise_model,
            "epipolar_mode": args.epipolar_mode,
            "lambda_cheirality": args.lambda_cheirality,
            "cheirality_topk": None if args.cheirality_topk <= 0 else args.cheirality_topk,
            "device": device,
        }

        se3_kwargs = {
            "max_iter": args.se3_max_iter,
            "tol": args.tol,
            "rot_lr": args.se3_rot_lr,
            "trans_lr": args.se3_trans_lr,
            "seed": args.seed,
            "differentiable_transport": False,
        }
        geoopt_kwargs = {
            "max_iter": args.geoopt_max_iter,
            "tol": args.tol,
            "lr": args.geoopt_lr,
            "grad_clip": None,
            "seed": args.seed,
            "differentiable_transport": False,
        }

        metrics_base = {
            "scenario": scenario_name,
            "epipolar_mode": args.epipolar_mode,
            "lambda_color": args.lambda_color,
            "lambda_epipolar": args.lambda_epipolar,
            "lambda_cov": args.lambda_cov,
            "sigma_epipolar": args.sigma_epipolar,
            "sigma_color": args.sigma_color,
            "sigma_cov": args.sigma_cov,
            "noise_model": args.noise_model,
            "init_mode": args.init_mode,
        }

        if not args.skip_se3:
            se3_dir = scenario_dir / "se3"
            evaluate_method(
                "se3",
                solver_kwargs,
                se3_kwargs,
                geoopt_kwargs,
                metrics_base,
                se3_dir,
                args.seed,
                args.init_mode,
                pts1,
                pts2,
                correspondences,
                F_gt,
                R_wc_gt,
                t_wc_gt,
            )
            if args.verbose:
                print("  SE(3) optimisation completed.")

        if not args.skip_geoopt:
            geo_dir = scenario_dir / "s3x_s2"
            evaluate_method(
                "s3x_s2",
                solver_kwargs,
                se3_kwargs,
                geoopt_kwargs,
                metrics_base,
                geo_dir,
                args.seed,
                args.init_mode,
                pts1,
                pts2,
                correspondences,
                F_gt,
                R_wc_gt,
                t_wc_gt,
            )
            if args.verbose:
                print("  S³×S² optimisation completed.")

        if args.verbose:
            print(f"Results saved under {scenario_dir}")


if __name__ == "__main__":
    main()
