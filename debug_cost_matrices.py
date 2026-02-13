#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/kohsukeide/dev/perspective-n-gaussian')

import numpy as np
import torch
from src.oracle_study.core.toy_problem_generator import ToyProblemGenerator
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

def R_yaw(rad: float) -> np.ndarray:
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

def get_weight_configurations():
    """Get standard weight configurations for cost function analysis."""
    return {
        'color_only': {'lambda_color': 1.0, 'lambda_epipolar': 0.0},
        'balanced': {'lambda_color': 0.5, 'lambda_epipolar': 1.0},
        'optimal': {'lambda_color': 0.8, 'lambda_epipolar': 0.2},  # 4:1 ratio
        'epi_only': {'lambda_color': 0.0, 'lambda_epipolar': 1.0},
    }

def debug_cost_matrices():
    print("🔍 Debugging Cost Matrices Generation")
    print("=" * 50)
    
    generator = ToyProblemGenerator(seed=42)
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)
    
    # Use one representative scenario for weight sensitivity analysis
    R_wc = R_yaw(0.1)
    t_wc = np.array([0.25, 0.02, 0.05], dtype=np.float32)
    g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(n_gaussians=15, K=K, R_wc=R_wc, t_wc=t_wc)
    
    weight_configs = get_weight_configurations()
    cost_matrices = {}
    
    print("\n--- Generating Cost Matrices ---")
    for name, weights in weight_configs.items():
        print(f"Testing {name}: λ_color={weights['lambda_color']}, λ_epipolar={weights['lambda_epipolar']}")
        
        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2, k1=K, k2=K, epsilon=0.01,
            lambda_color=weights['lambda_color'], lambda_epipolar=weights['lambda_epipolar'], device='cpu')
        
        with torch.no_grad():
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            C_np = C.cpu().numpy()
        
        cost_matrices[name] = C_np
        print(f"  Shape: {C_np.shape}")
        print(f"  Min: {C_np.min():.6f}, Max: {C_np.max():.6f}, Mean: {C_np.mean():.6f}")
        print(f"  Diagonal mean: {np.diagonal(C_np).mean():.6f}")
        print()
    
    print("--- Analyzing Comparisons ---")
    
    # Check which comparisons will be made
    comparisons = []
    
    # Compare optimal vs balanced
    if 'optimal' in cost_matrices and 'balanced' in cost_matrices:
        diff = cost_matrices['optimal'] - cost_matrices['balanced']
        comparisons.append(('optimal_vs_balanced', 'Optimal - Balanced', diff))
        print(f"✓ Optimal vs Balanced: diff min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # Compare epi_only vs color_only
    if 'epi_only' in cost_matrices and 'color_only' in cost_matrices:
        diff = cost_matrices['epi_only'] - cost_matrices['color_only']
        comparisons.append(('epi_vs_color', 'Epi Only - Color Only', diff))
        print(f"✓ Epi Only vs Color Only: diff min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    # Compare balanced vs color_only
    if 'balanced' in cost_matrices and 'color_only' in cost_matrices:
        diff = cost_matrices['balanced'] - cost_matrices['color_only']
        comparisons.append(('balanced_vs_color', 'Balanced - Color Only', diff))
        print(f"✓ Balanced vs Color Only: diff min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    print(f"\nTotal comparisons: {len(comparisons)}")
    
    # Check for potential issues
    print("\n--- Diagnostic Analysis ---")
    print("Cost matrix ranges:")
    for name, matrix in cost_matrices.items():
        print(f"  {name:12}: [{matrix.min():.3f}, {matrix.max():.3f}] (range: {matrix.max()-matrix.min():.3f})")
    
    print("\nDifference matrix ranges:")
    for comp_name, title, diff_matrix in comparisons:
        print(f"  {title:20}: [{diff_matrix.min():.3f}, {diff_matrix.max():.3f}] (range: {diff_matrix.max()-diff_matrix.min():.3f})")
    
    # Check if matrices are identical
    print("\nMatrix similarity check:")
    matrix_names = list(cost_matrices.keys())
    for i, name1 in enumerate(matrix_names):
        for name2 in matrix_names[i+1:]:
            diff = np.abs(cost_matrices[name1] - cost_matrices[name2]).max()
            if diff < 1e-10:
                print(f"  ⚠️  {name1} and {name2} are nearly identical (max diff: {diff:.2e})")
            else:
                print(f"  ✓ {name1} vs {name2}: max diff = {diff:.6f}")

if __name__ == "__main__":
    debug_cost_matrices()