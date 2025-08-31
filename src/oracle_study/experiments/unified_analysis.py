#!/usr/bin/env python3
"""
Unified Analysis for Oracle Study

This module combines transport matrix analysis and cost function analysis
into a single comprehensive experiment with proper output separation.
"""

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.oracle_study.core import ToyProblemGenerator, TransformationParams, TransportMatrixVisualizer
from typing import Tuple, Dict, Optional, List
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Create experiment-specific figure directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.dirname(SCRIPT_DIR)  # src/oracle_study
TRANSPORT_FIGURES_DIR = os.path.join(ORACLE_DIR, "results", "transport_matrix_analysis", "figures")
COST_FIGURES_DIR = os.path.join(ORACLE_DIR, "results", "cost_function_analysis", "figures")
os.makedirs(TRANSPORT_FIGURES_DIR, exist_ok=True)
os.makedirs(COST_FIGURES_DIR, exist_ok=True)


def get_standard_scenarios():
    """Get standard epipolar scenarios used across all analyses."""
    def R_yaw(rad: float) -> np.ndarray:
        return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

    return {
        'epi_translation': { 'R_wc': R_yaw(0.0),  't_wc': np.array([0.25, 0.02, 0.0], dtype=np.float32) },
        'epi_yaw_rotation': { 'R_wc': R_yaw(0.12), 't_wc': np.array([0.18, 0.01, 0.02], dtype=np.float32) },
        'epi_forward_scale_like': { 'R_wc': R_yaw(0.02), 't_wc': np.array([0.05, 0.0, 0.15], dtype=np.float32) },
        'epi_combined': { 'R_wc': R_yaw(0.10), 't_wc': np.array([0.25, 0.02, 0.05], dtype=np.float32) },
        'epi_color_change': { 'R_wc': R_yaw(0.08), 't_wc': np.array([0.20, 0.01, 0.03], dtype=np.float32) },
    }


def get_weight_configurations():
    """Get standard weight configurations for cost function analysis."""
    return {
        'color_only': {'lambda_color': 1.0, 'lambda_epipolar': 0.0},
        'balanced': {'lambda_color': 0.5, 'lambda_epipolar': 1.0},
        'optimal': {'lambda_color': 0.8, 'lambda_epipolar': 0.2},  # 4:1 ratio
        'epi_only': {'lambda_color': 0.0, 'lambda_epipolar': 1.0},
    }


def get_scenario_specific_weights(scenario_name: str) -> Tuple[float, float]:
    """Get appropriate weights for specific scenarios."""
    if 'color_change' in scenario_name:
        # Use epipolar-only for color change scenarios
        return 0.0, 1.0
    else:
        # Use optimal 4:1 ratio for normal scenarios
        return 0.8, 0.2


def analyze_transport_matrices():
    """Analyze transport matrices under epipolar-consistent scenarios."""
    print("=== Transport Matrix Analysis ===\n")

    visualizer = TransportMatrixVisualizer(figures_dir=TRANSPORT_FIGURES_DIR)
    generator = ToyProblemGenerator(seed=42)

    # Intrinsics
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)
    scenarios = get_standard_scenarios()
    transport_matrices = {}

    for name, params in scenarios.items():
        print(f"\n--- Testing {name} ---")
        g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(
            n_gaussians=15, K=K, R_wc=params['R_wc'], t_wc=params['t_wc']
        )

        # Apply scenario-specific modifications
        if 'color_change' in name:
            rng = np.random.RandomState(123)
            g2.rgb = rng.uniform(0, 1, size=g2.rgb.shape).astype(np.float32)

        # Get appropriate weights
        lambda_color, lambda_epipolar = get_scenario_specific_weights(name)

        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2, k1=K, k2=K,
            epsilon=0.01, lambda_color=lambda_color, lambda_epipolar=lambda_epipolar, device='cpu')

        with torch.no_grad():
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            T = solver.unbalanced_sinkhorn_algorithm(C)
            T_np = T.cpu().numpy()

        transport_matrices[name] = T_np
        
        # Stats and visualizations in transport_matrix_analysis directory
        visualizer.analyze_transport_statistics(T_np, name)
        visualizer.visualize_transport_matrix(T_np, title=f'Transport Matrix - {name}', save_path=f'transport_matrix_{name}.png')
        visualizer.visualize_correspondences_on_images(g1, g2, T_np, save_path=f'correspondences_{name}.png')

    # Compare all transport matrices
    visualizer.compare_transport_matrices(transport_matrices, save_path='transport_matrices_comparison_epipolar.png')

    print(f"\n🎉 Transport matrix analysis completed!")
    print(f"Results saved to: {TRANSPORT_FIGURES_DIR}")
    
    return transport_matrices


def analyze_cost_functions():
    """Analyze cost function weight sensitivity and cost matrix properties.
    
    This function focuses on:
    - Cost matrix visualization for different weight configurations
    - Weight sensitivity analysis (not transport matrix visualization)
    - Statistical analysis of cost function behavior
    
    Note: Transport matrix visualizations are handled in analyze_transport_matrices()
    """
    print("\n=== Cost Function Analysis ===\n")
    print("Analyzing cost matrices and weight sensitivity...")
    print("(Transport matrix visualizations are handled separately)\n")

    # Create visualizer for cost function analysis
    cost_visualizer = TransportMatrixVisualizer(figures_dir=COST_FIGURES_DIR)
    generator = ToyProblemGenerator(seed=42)
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)

    def R_yaw(rad: float) -> np.ndarray:
        return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

    # Use one representative scenario for weight sensitivity analysis
    R_wc = R_yaw(0.1)
    t_wc = np.array([0.25, 0.02, 0.05], dtype=np.float32)
    g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(n_gaussians=15, K=K, R_wc=R_wc, t_wc=t_wc)

    weight_configs = get_weight_configurations()
    stats_results = {}
    cost_matrices = {}

    for name, weights in weight_configs.items():
        print(f"\n--- Testing {name} weight configuration ---")
        print(f"    λ_color={weights['lambda_color']}, λ_epipolar={weights['lambda_epipolar']}")
        
        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2, k1=K, k2=K, epsilon=0.01,
            lambda_color=weights['lambda_color'], lambda_epipolar=weights['lambda_epipolar'], device='cpu')
        
        with torch.no_grad():
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            T = solver.unbalanced_sinkhorn_algorithm(C)
            T_np = T.cpu().numpy()
            C_np = C.cpu().numpy()

        # Store results for cost function analysis (internal calculation only)
        diagonal_sum = np.trace(T_np)
        total_sum = np.sum(T_np)
        diagonal_concentration = diagonal_sum / total_sum
        entropy = -np.sum(T_np * np.log(T_np + 1e-12))
        sparsity = 1.0 - (np.count_nonzero(T_np > 1e-6) / T_np.size)
        
        stats_results[name] = {
            'diagonal_concentration': diagonal_concentration,
            'entropy': entropy,
            'sparsity': sparsity
        }
        cost_matrices[name] = C_np
        
        print(f"    Diagonal concentration: {diagonal_concentration:.4f}")
        
        # Note: Transport matrix visualizations are handled in transport_matrix_analysis section
        # Cost function analysis focuses on cost matrices and weight sensitivity only

    # Generate cost function specific visualizations
    analyze_cost_matrices_visualization(cost_matrices, cost_visualizer)
    create_weight_sensitivity_summary(stats_results, cost_visualizer)

    print(f"\n🎉 Cost function analysis completed!")
    print(f"Results saved to: {COST_FIGURES_DIR}")
    
    return stats_results


def analyze_cost_matrices_visualization(cost_matrices: Dict[str, np.ndarray], visualizer):
    """Create cost matrix visualizations for different weight configurations."""
    print("\n--- Generating Cost Matrix Visualizations ---")
    
    # 1. Side-by-side cost matrix comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    configs = list(cost_matrices.keys())
    for i, (name, cost_matrix) in enumerate(cost_matrices.items()):
        if i < 4:  # Only show first 4 configurations
            ax = axes[i]
            im = ax.imshow(cost_matrix, cmap='hot', interpolation='nearest')
            ax.set_title(f'Cost Matrix - {name}')
            ax.set_xlabel('Target Gaussians')
            ax.set_ylabel('Source Gaussians')
            plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(len(cost_matrices), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(COST_FIGURES_DIR, 'cost_matrices_weight_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved cost matrix comparison to {save_path}")
    plt.close()

    # 2. Cost matrix differences analysis
    if len(cost_matrices) >= 2:
        # Count valid comparisons
        comparisons = []
        
        # Compare optimal vs balanced
        if 'optimal' in cost_matrices and 'balanced' in cost_matrices:
            diff = cost_matrices['optimal'] - cost_matrices['balanced']
            comparisons.append(('optimal_vs_balanced', 'Optimal - Balanced', diff))
        
        # Compare epi_only vs color_only
        if 'epi_only' in cost_matrices and 'color_only' in cost_matrices:
            diff = cost_matrices['epi_only'] - cost_matrices['color_only']
            comparisons.append(('epi_vs_color', 'Epi Only - Color Only', diff))
        
        # Compare balanced vs color_only
        if 'balanced' in cost_matrices and 'color_only' in cost_matrices:
            diff = cost_matrices['balanced'] - cost_matrices['color_only']
            comparisons.append(('balanced_vs_color', 'Balanced - Color Only', diff))
        
        if comparisons:
            n_comparisons = min(len(comparisons), 3)  # Max 3 subplots
            fig, axes = plt.subplots(1, n_comparisons, figsize=(6*n_comparisons, 5))
            
            # Ensure axes is always a list
            if n_comparisons == 1:
                axes = [axes]
            
            # Calculate global color scale for consistent visualization
            all_diffs = [diff for _, _, diff in comparisons[:n_comparisons]]
            global_vmin = min(diff.min() for diff in all_diffs)
            global_vmax = max(diff.max() for diff in all_diffs)
            
            print(f"Using global color scale: [{global_vmin:.3f}, {global_vmax:.3f}]")
            
            for i, (comp_name, title, diff_matrix) in enumerate(comparisons[:n_comparisons]):
                im = axes[i].imshow(diff_matrix, cmap='RdBu_r', interpolation='nearest',
                                  vmin=global_vmin, vmax=global_vmax)
                axes[i].set_title(f'{title}\n(Range: [{diff_matrix.min():.3f}, {diff_matrix.max():.3f}])')
                axes[i].set_xlabel('Target Gaussians')
                axes[i].set_ylabel('Source Gaussians')
                plt.colorbar(im, ax=axes[i])
                
                # Add statistics as text
                stats_text = f'Mean: {diff_matrix.mean():.3f}\nStd: {diff_matrix.std():.3f}'
                axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            save_path = os.path.join(COST_FIGURES_DIR, 'cost_matrices_differences.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved cost matrix differences to {save_path} ({n_comparisons} comparisons)")
            plt.close()
        else:
            print("No valid cost matrix comparisons available")



def create_weight_sensitivity_summary(stats_results: Dict, visualizer):
    """Create comprehensive weight sensitivity analysis."""
    print("\n--- Generating Weight Sensitivity Summary ---")
    
    configs = list(stats_results.keys())
    
    # Extract key metrics
    diagonal_concs = [stats_results[config]['diagonal_concentration'] for config in configs]
    entropies = [stats_results[config]['entropy'] for config in configs]
    sparsities = [stats_results[config]['sparsity'] for config in configs]
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Diagonal concentration (most important metric)
    bars1 = ax1.bar(range(len(configs)), diagonal_concs, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Weight Configuration')
    ax1.set_ylabel('Diagonal Concentration')
    ax1.set_title('Transport Quality: Diagonal Concentration')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, diagonal_concs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Entropy
    bars2 = ax2.bar(range(len(configs)), entropies, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Weight Configuration')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Transport Uncertainty: Entropy (Lower = Better)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax2.grid(True, alpha=0.3)
    
    # Sparsity
    bars3 = ax3.bar(range(len(configs)), sparsities, alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Weight Configuration')
    ax3.set_ylabel('Sparsity')
    ax3.set_title('Transport Sparsity (Higher = More Concentrated)')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax3.grid(True, alpha=0.3)
    
    # Combined analysis
    scatter = ax4.scatter(diagonal_concs, entropies, s=100, alpha=0.7, c=range(len(configs)), cmap='viridis')
    ax4.set_xlabel('Diagonal Concentration (Higher = Better)')
    ax4.set_ylabel('Entropy (Lower = Better)')
    ax4.set_title('Transport Quality vs Uncertainty')
    ax4.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, config in enumerate(configs):
        ax4.annotate(config.replace('_', '\n'), (diagonal_concs[i], entropies[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(COST_FIGURES_DIR, 'weight_sensitivity_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved weight sensitivity analysis to {save_path}")
    plt.close()


def main():
    """Run unified analysis combining transport matrix and cost function analysis."""
    print("🔍 Unified Oracle Study Analysis")
    print("=" * 60)
    print("This script combines transport matrix analysis and cost function analysis")
    print("with proper output separation into respective directories.")
    print("=" * 60)

    # 1. Transport Matrix Analysis
    transport_results = analyze_transport_matrices()

    # 2. Cost Function Analysis  
    cost_results = analyze_cost_functions()

    # 3. Summary
    print("\n" + "=" * 60)
    print("📊 Unified Analysis Summary:")
    print(f"- Transport matrix analysis (visualization): {TRANSPORT_FIGURES_DIR}")
    print(f"- Cost function analysis (matrices & weights): {COST_FIGURES_DIR}")
    print(f"- Analyzed {len(transport_results)} transport scenarios")
    print(f"- Evaluated {len(cost_results)} weight configurations")
    
    # Find best configuration
    best_config = max(cost_results.keys(), key=lambda k: cost_results[k]['diagonal_concentration'])
    best_score = cost_results[best_config]['diagonal_concentration']
    print(f"- Best weight configuration: {best_config} (diagonal concentration: {best_score:.3f})")

    print(f"\n✅ Unified analysis completed successfully!")
    print("Check the respective directories for all visualizations.")


if __name__ == "__main__":
    main()