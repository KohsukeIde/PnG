#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/kohsukeide/dev/perspective-n-gaussian')

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.oracle_study.core.toy_problem_generator import ToyProblemGenerator
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver

def R_yaw(rad: float) -> np.ndarray:
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

def validate_weight_configurations():
    print("🔍 Validating Weight Configurations for Cost Matrices")
    print("=" * 60)
    
    generator = ToyProblemGenerator(seed=42)
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)
    
    # Generate test data
    R_wc = R_yaw(0.1)
    t_wc = np.array([0.25, 0.02, 0.05], dtype=np.float32)
    g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(n_gaussians=15, K=K, R_wc=R_wc, t_wc=t_wc)
    
    print(f"Generated {len(g1.means)} Gaussians")
    print(f"Ground truth correspondences: {len(corr)} pairs")
    print(f"F matrix shape: {F_gt.shape}")
    print(f"F matrix determinant: {np.linalg.det(F_gt):.6f}")
    print()
    
    # Test individual components to verify they work correctly
    weight_configs = {
        'color_only': {'lambda_color': 1.0, 'lambda_epipolar': 0.0},
        'epi_only': {'lambda_color': 0.0, 'lambda_epipolar': 1.0},
        'balanced': {'lambda_color': 0.5, 'lambda_epipolar': 1.0},
        'optimal': {'lambda_color': 0.8, 'lambda_epipolar': 0.2},
    }
    
    print("--- Testing Each Weight Configuration ---")
    cost_matrices = {}
    transport_matrices = {}
    
    for name, weights in weight_configs.items():
        print(f"\n🧪 Testing {name}:")
        print(f"   λ_color={weights['lambda_color']}, λ_epipolar={weights['lambda_epipolar']}")
        
        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2, k1=K, k2=K, epsilon=0.01,
            lambda_color=weights['lambda_color'], 
            lambda_epipolar=weights['lambda_epipolar'], 
            device='cpu')
        
        with torch.no_grad():
            # Get cost matrix
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            C_np = C.cpu().numpy()
            cost_matrices[name] = C_np
            
            # Get transport matrix for validation
            T = solver.unbalanced_sinkhorn_algorithm(C)
            T_np = T.cpu().numpy()
            transport_matrices[name] = T_np
            
            # Calculate diagonal concentration
            diagonal_concentration = np.trace(T_np) / np.sum(T_np)
            
        print(f"   Cost matrix: min={C_np.min():.4f}, max={C_np.max():.4f}, mean={C_np.mean():.4f}")
        print(f"   Diagonal cost mean: {np.diagonal(C_np).mean():.4f}")
        print(f"   Transport diagonal concentration: {diagonal_concentration:.4f}")
        
        # Analyze cost structure
        diagonal_costs = np.diagonal(C_np)
        off_diagonal_costs = C_np[~np.eye(C_np.shape[0], dtype=bool)]
        print(f"   Diagonal vs off-diagonal cost ratio: {diagonal_costs.mean() / off_diagonal_costs.mean():.4f}")
    
    print("\n--- Validating Expected Behavior ---")
    
    # Check color_only behavior
    color_costs = cost_matrices['color_only']
    epi_costs = cost_matrices['epi_only']
    
    print(f"\n🎨 Color-only validation:")
    print(f"   Should have low diagonal costs (same colors): {np.diagonal(color_costs).mean():.6f}")
    print(f"   Off-diagonal costs should be higher: {color_costs[~np.eye(color_costs.shape[0], dtype=bool)].mean():.6f}")
    
    print(f"\n📐 Epi-only validation:")
    print(f"   Diagonal costs (geometric consistency): {np.diagonal(epi_costs).mean():.6f}")
    print(f"   Off-diagonal costs: {epi_costs[~np.eye(epi_costs.shape[0], dtype=bool)].mean():.6f}")
    
    # Compare matrices
    print(f"\n🔍 Matrix comparison validation:")
    for i, name1 in enumerate(weight_configs.keys()):
        for name2 in list(weight_configs.keys())[i+1:]:
            diff = np.abs(cost_matrices[name1] - cost_matrices[name2]).max()
            mean_diff = np.abs(cost_matrices[name1] - cost_matrices[name2]).mean()
            print(f"   {name1} vs {name2}: max_diff={diff:.4f}, mean_diff={mean_diff:.4f}")
    
    # Test the three comparisons that will be plotted
    print(f"\n📊 Plotting comparisons:")
    comparisons = [
        ('optimal', 'balanced', 'Optimal - Balanced'),
        ('epi_only', 'color_only', 'Epi Only - Color Only'), 
        ('balanced', 'color_only', 'Balanced - Color Only')
    ]
    
    for name1, name2, title in comparisons:
        if name1 in cost_matrices and name2 in cost_matrices:
            diff = cost_matrices[name1] - cost_matrices[name2]
            print(f"   {title}: range=[{diff.min():.3f}, {diff.max():.3f}], mean={diff.mean():.3f}, std={diff.std():.3f}")
        else:
            print(f"   ⚠️ {title}: Missing matrices")
    
    print("\n✅ Validation completed!")
    return cost_matrices, transport_matrices

if __name__ == "__main__":
    cost_matrices, transport_matrices = validate_weight_configurations()