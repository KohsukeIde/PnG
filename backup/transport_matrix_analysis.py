#!/usr/bin/env python3
"""
Transport Matrix Analysis for Oracle Study

This module analyzes transport matrices under different transformation scenarios
to understand the behavior of the optimal transport solver.
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

# Create experiment-specific figure directory
EXPERIMENT_NAME = "transport_matrix_analysis"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.dirname(SCRIPT_DIR)  # src/oracle_study
FIGURES_DIR = os.path.join(ORACLE_DIR, "results", EXPERIMENT_NAME, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def analyze_transport_matrices():
    """Analyze transport matrices under epipolar-consistent scenarios (GT F + Sampson)."""
    print("=== Analyzing Transport Matrices (Epipolar-Consistent) ===\n")

    visualizer = TransportMatrixVisualizer(figures_dir=FIGURES_DIR)
    generator = ToyProblemGenerator(seed=42)

    # Intrinsics
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)

    def R_yaw(rad: float) -> np.ndarray:
        return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

    epi_scenarios = {
        'epi_translation': { 'R_wc': R_yaw(0.0),  't_wc': np.array([0.25, 0.02, 0.0], dtype=np.float32) },
        'epi_yaw_rotation': { 'R_wc': R_yaw(0.12), 't_wc': np.array([0.18, 0.01, 0.02], dtype=np.float32) },
        'epi_forward_scale_like': { 'R_wc': R_yaw(0.02), 't_wc': np.array([0.05, 0.0, 0.15], dtype=np.float32) },
        'epi_combined': { 'R_wc': R_yaw(0.10), 't_wc': np.array([0.25, 0.02, 0.05], dtype=np.float32) },
        'epi_color_change': { 'R_wc': R_yaw(0.08), 't_wc': np.array([0.20, 0.01, 0.03], dtype=np.float32) },
    }

    transport_matrices = {}

    for name, params in epi_scenarios.items():
        print(f"\n--- Testing {name} ---")
        g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(
            n_gaussians=15, K=K, R_wc=params['R_wc'], t_wc=params['t_wc']
        )

        # If color-change scenario: randomize g2 colors (geometry/F unchanged)
        # and use epipolar-only weights to test robustness
        if 'color_change' in name:
            rng = np.random.RandomState(123)
            g2.rgb = rng.uniform(0, 1, size=g2.rgb.shape).astype(np.float32)
            # Use epipolar-only weights for color change scenario
            lambda_color, lambda_epipolar = 0.0, 1.0
        else:
            # Use balanced weights for normal scenarios
            lambda_color, lambda_epipolar = 0.5, 1.0

        solver = OptimalTransportSolver(
            gaussians1=g1, gaussians2=g2, k1=K, k2=K,
            epsilon=0.01, lambda_color=lambda_color, lambda_epipolar=lambda_epipolar, device='cpu')

        with torch.no_grad():
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            T = solver.unbalanced_sinkhorn_algorithm(C)
            T_np = T.cpu().numpy()

        transport_matrices[name] = T_np
        # Stats and visualizations
        visualizer.analyze_transport_statistics(T_np, name)
        visualizer.visualize_transport_matrix(T_np, title=f'Transport Matrix - {name}', save_path=f'transport_matrix_{name}.png')
        visualizer.visualize_correspondences_on_images(g1, g2, T_np, save_path=f'correspondences_{name}.png')

    # Compare all transport matrices in one figure
    visualizer.compare_transport_matrices(transport_matrices, save_path='transport_matrices_comparison_epipolar.png')

    print("\n🎉 Transport matrix analysis (epipolar) completed!")
    print(f"Check the {FIGURES_DIR} directory for generated visualizations.")


def analyze_transport_ablations():
    """Ablations under one epipolar-consistent scenario (GT F + Sampson)."""
    print("\n=== Analyzing Transport Matrix Ablations (Epipolar) ===\n")

    visualizer = TransportMatrixVisualizer(figures_dir=FIGURES_DIR)
    generator = ToyProblemGenerator(seed=42)
    K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)

    def R_yaw(rad: float) -> np.ndarray:
        return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)

    R_wc = R_yaw(0.1); t_wc = np.array([0.25, 0.02, 0.05], dtype=np.float32)
    g1, g2, corr, F_gt = generator.generate_epipolar_correspondences(n_gaussians=15, K=K, R_wc=R_wc, t_wc=t_wc)

    ablation_configs = {
        'color_only': {'lambda_color': 1.0, 'lambda_epipolar': 0.0},
        'balanced': {'lambda_color': 0.5, 'lambda_epipolar': 1.0},
        'epi_only': {'lambda_color': 0.0, 'lambda_epipolar': 1.0},
    }

    for name, w in ablation_configs.items():
        print(f"\n--- Testing {name} configuration ---")
        solver = OptimalTransportSolver(gaussians1=g1, gaussians2=g2, k1=K, k2=K, epsilon=0.01,
                                        lambda_color=w['lambda_color'], lambda_epipolar=w['lambda_epipolar'], device='cpu')
        with torch.no_grad():
            C = solver.compute_cost_matrix_fundamental(torch.from_numpy(F_gt))
            T = solver.unbalanced_sinkhorn_algorithm(C)
            T_np = T.cpu().numpy()
        visualizer.analyze_transport_statistics(T_np, f"Ablation - {name}")
        visualizer.visualize_transport_matrix(T_np, title=f'Transport Matrix - {name}', save_path=f'transport_ablation_{name}.png')


def create_ablation_summary_plot(stats_results: Dict):
    """Create a summary plot showing how different cost components affect transport quality."""
    configs = list(stats_results.keys())

    # Extract key metrics
    diagonal_concs = [stats_results[config]
                      ['diagonal_concentration'] for config in configs]
    entropies = [stats_results[config]['entropy'] for config in configs]
    sparsities = [stats_results[config]['sparsity'] for config in configs]

    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Diagonal concentration (most important metric)
    bars1 = ax1.bar(range(len(configs)), diagonal_concs,
                    alpha=0.7, color='skyblue')
    ax1.set_xlabel('Cost Configuration')
    ax1.set_ylabel('Diagonal Concentration')
    ax1.set_title('Transport Quality: Diagonal Concentration')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, diagonal_concs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # Entropy (lower is more concentrated)
    bars2 = ax2.bar(range(len(configs)), entropies,
                    alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Cost Configuration')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Transport Uncertainty: Entropy (Lower = Better)')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax2.grid(True, alpha=0.3)

    # Sparsity (higher means more concentrated)
    bars3 = ax3.bar(range(len(configs)), sparsities,
                    alpha=0.7, color='lightgreen')
    ax3.set_xlabel('Cost Configuration')
    ax3.set_ylabel('Sparsity')
    ax3.set_title('Transport Sparsity (Higher = More Concentrated)')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels([c.replace('_', '\n') for c in configs], rotation=0)
    ax3.grid(True, alpha=0.3)

    # Combined scatter plot: diagonal concentration vs entropy
    scatter = ax4.scatter(diagonal_concs, entropies, s=100,
                          alpha=0.7, c=range(len(configs)), cmap='viridis')
    ax4.set_xlabel('Diagonal Concentration (Higher = Better)')
    ax4.set_ylabel('Entropy (Lower = Better)')
    ax4.set_title('Transport Quality vs Uncertainty')
    ax4.grid(True, alpha=0.3)

    # Add labels for each point
    for i, config in enumerate(configs):
        ax4.annotate(config.replace('_', '\n'), (diagonal_concs[i], entropies[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'transport_ablation_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved ablation summary to {save_path}")
    plt.close()


def main():
    """Run transport matrix analysis."""
    print("🔍 Transport Matrix Analysis for Oracle Study")
    print("=" * 60)

    # Epipolar-consistent scenario analysis only
    analyze_transport_matrices()

    # Ablations (epipolar)
    analyze_transport_ablations()

    print("\n" + "=" * 60)
    print("📊 Transport Matrix Analysis Summary:")
    print("- Epipolar-consistent scenarios only (GT F + Sampson)")

    print(f"\n✅ Complete transport matrix analysis completed!")
    print(f"Check {FIGURES_DIR} directory for all visualizations.")


if __name__ == "__main__":
    main()
