#!/usr/bin/env python3
"""
Reproduction script to verify the fix for optimization failure.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from src.oracle_study.core import ToyProblemGenerator
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.oracle_study.camera_pose.experiments.pose_oracle_basic import (
    collect_all_scenarios, apply_scenario_effects, rotation_error_deg
)

def run_optimization(differentiable: bool, scenario_name="baseline_yaw", seed=42):
    print(f"\nRunning optimization with differentiable_transport={differentiable}")
    
    # Setup
    device = torch.device("cpu")
    generator = ToyProblemGenerator(seed=seed)
    scenarios = collect_all_scenarios()
    params = scenarios[scenario_name]
    
    R_wc_gt = params["R_wc"].astype(np.float64)
    t_wc_gt = params["t_wc"].astype(np.float64)
    K = np.array([[800.0, 0.0, 400.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    
    g1, g2, correspondences, F_gt = generator.generate_epipolar_correspondences(
        n_gaussians=20, K=K, R_wc=R_wc_gt, t_wc=t_wc_gt,
    )
    g1, g2 = apply_scenario_effects(generator, g1, g2, scenario_name, params, seed=seed)
    
    solver_kwargs = {
        "gaussians1": g1,
        "gaussians2": g2,
        "k1": K,
        "k2": K,
        "epsilon": 0.01,
        "lambda_color": 1.0,
        "lambda_epipolar": 1.0,
        "lambda_cov": 0.3,
        "sigma_epipolar": 400.0,
        "sigma_color": 0.5,
        "sigma_cov": 8.0,
        "epipolar_mode": "sampson",
        "device": device,
    }
    
    se3_kwargs = {
        "max_iter": 200, 
        "tol": 1e-6,
        "rot_lr": 5e-3,
        "trans_lr": 5e-4,
        "seed": seed,
        "differentiable_transport": differentiable,
    }
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = OptimalTransportSolver(**solver_kwargs)
    
    # Initialize with random perturbation
    with torch.no_grad():
        solver._init_se3_like_cam1(rot_noise=0.1, trans_noise=0.1, seed=seed)
        
    loss_history = solver.optimize_with_SE3(save_diagnostics=False, **se3_kwargs)
    
    R_est = solver.R_wc.detach().cpu().numpy()
    rot_err = rotation_error_deg(R_est, R_wc_gt)
    final_loss = loss_history[-1]
    
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Rotation Error: {rot_err:.6f} deg")
    
    return loss_history, rot_err

def main():
    # Run ONLY non-differentiable (EM mode) to test the fix in cost function
    loss_nodiff, err_nodiff = run_optimization(differentiable=False)
    
    print("\n=== Result ===")
    print(f"Non-Diff (EM): Loss={loss_nodiff[-1]:.6f}, RotErr={err_nodiff:.6f}")

if __name__ == "__main__":
    main()
