#!/usr/bin/env python3
"""
Inspect Cost Matrix and Transport Plan at initialization.
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
    collect_all_scenarios, apply_scenario_effects
)

def inspect_initialization(scenario_name="baseline_yaw", seed=42):
    print(f"\nInspecting initialization for scenario: {scenario_name}")
    
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
        "device": device,
    }
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = OptimalTransportSolver(**solver_kwargs)
    
    # 1. GT Initialization
    print("\n--- GT Initialization ---")
    with torch.no_grad():
        F_gt_tensor = solver._build_F_from_wc(
            torch.tensor(R_wc_gt, dtype=torch.float32),
            torch.tensor(t_wc_gt, dtype=torch.float32)
        )
        C_gt = solver.compute_cost_matrix(F_gt_tensor)
        T_gt = solver.unbalanced_sinkhorn_algorithm(C_gt)
        
        print(f"Cost Matrix Stats (GT): Min={C_gt.min():.4f}, Max={C_gt.max():.4f}, Mean={C_gt.mean():.4f}")
        print(f"Transport Stats (GT): Max={T_gt.max():.4f}, Sum={T_gt.sum():.4f}")
        
        # Check if T matches correspondences
        # Correspondences are indices [i, j]
        matched_prob = 0.0
        for i, j in correspondences:
            matched_prob += T_gt[i, j].item()
        print(f"Sum of probability on GT correspondences: {matched_prob:.4f} / {len(correspondences)}")

    # 2. Random Initialization
    print("\n--- Random Initialization ---")
    with torch.no_grad():
        solver._init_se3_like_cam1(rot_noise=0.1, trans_noise=0.1, seed=seed)
        se3_vec = torch.cat([solver.rot_vec, solver.trans_vec])
        T_cw = solver.lie.se3_to_SE3(se3_vec)
        R_wc = T_cw[:3, :3].t()
        t_wc = -R_wc @ T_cw[:3, 3]
        F_rand = solver._build_F_from_wc(R_wc, t_wc)
        
        C_rand = solver.compute_cost_matrix(F_rand)
        T_rand = solver.unbalanced_sinkhorn_algorithm(C_rand)
        
        print(f"Cost Matrix Stats (Rand): Min={C_rand.min():.4f}, Max={C_rand.max():.4f}, Mean={C_rand.mean():.4f}")
        print(f"Transport Stats (Rand): Max={T_rand.max():.4f}, Sum={T_rand.sum():.4f}")
        
        matched_prob_rand = 0.0
        for i, j in correspondences:
            matched_prob_rand += T_rand[i, j].item()
        print(f"Sum of probability on GT correspondences: {matched_prob_rand:.4f} / {len(correspondences)}")
        
        # Check entropy/sparsity
        entropy = -(T_rand * (T_rand + 1e-12).log()).sum()
        print(f"Transport Entropy: {entropy:.4f}")

if __name__ == "__main__":
    inspect_initialization()
