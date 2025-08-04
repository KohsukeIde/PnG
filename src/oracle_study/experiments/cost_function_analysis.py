#!/usr/bin/env python3
"""
Cost Function Analysis for Oracle Study

This module analyzes the actual cost function components used in the current
OptimalTransportSolver to understand what drives the transport matrix behavior.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Create experiment-specific figure directory
EXPERIMENT_NAME = "cost_function_analysis"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.dirname(SCRIPT_DIR)  # src/oracle_study
FIGURES_DIR = os.path.join(ORACLE_DIR, "results", EXPERIMENT_NAME, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from src.oracle_study.core import ToyProblemGenerator, TransformationParams, TransportMatrixVisualizer
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver


def analyze_cost_components():
    """Analyze individual components of the current cost function."""
    print("=== Analyzing Cost Function Components ===\n")
    
    generator = ToyProblemGenerator(seed=42)
    visualizer = TransportMatrixVisualizer(figures_dir=FIGURES_DIR)
    
    # Generate test case
    gaussians1 = generator.generate_synthetic_gaussians(n_gaussians=15, color_mode='gradient')
    
    # Test different transformations
    scenarios = {
        'identical': TransformationParams(),
        'translation': TransformationParams(translation=np.array([0.3, 0.2])),
        'rotation': TransformationParams(rotation=np.pi/4),
        'scale': TransformationParams(scale=1.5),
        'color_change': TransformationParams()  # We'll modify colors manually
    }
    
    # Camera intrinsics
    K = np.array([
        [800, 0, 400],
        [0, 800, 400],
        [0, 0, 1]
    ], dtype=np.float32)
    
    results = {}
    
    for scenario_name, transform_params in scenarios.items():
        print(f"\n--- Analyzing {scenario_name} scenario ---")
        
        if scenario_name == 'color_change':
            # Create color-only change
            gaussians2 = generator.generate_synthetic_gaussians(n_gaussians=15, color_mode='random')
            correspondences = np.column_stack([np.arange(15), np.arange(15)])
        else:
            gaussians2, correspondences = generator.generate_known_correspondences(
                gaussians1, transform_params
            )
        
        # Create solver
        solver = OptimalTransportSolver(
            gaussians1=gaussians1,
            gaussians2=gaussians2,
            k1=K,
            k2=K,
            epsilon=0.01,
            lambda_mean=1.0,      # Not used in current implementation
            lambda_cov=1.0,       # Not used in current implementation
            lambda_color=0.5,
            lambda_epipolar=1.0,
            device='cpu'
        )
        
        # Analyze cost components
        with torch.no_grad():
            F_dummy = torch.eye(3, dtype=torch.float32)
            
            # Get the full cost matrix
            full_cost = solver.compute_cost_matrix_fundamental(F_dummy)
            
            # Analyze individual components by temporarily modifying weights
            # Component 1: Epipolar only
            solver_epi_only = OptimalTransportSolver(
                gaussians1=gaussians1,
                gaussians2=gaussians2,
                k1=K, k2=K,
                epsilon=0.01,
                lambda_color=0.0,      # Turn off color
                lambda_epipolar=1.0,   # Keep epipolar
                device='cpu'
            )
            epi_cost = solver_epi_only.compute_cost_matrix_fundamental(F_dummy)
            
            # Component 2: Color only
            solver_color_only = OptimalTransportSolver(
                gaussians1=gaussians1,
                gaussians2=gaussians2,
                k1=K, k2=K,
                epsilon=0.01,
                lambda_color=1.0,      # Keep color
                lambda_epipolar=0.0,   # Turn off epipolar
                device='cpu'
            )
            color_cost = solver_color_only.compute_cost_matrix_fundamental(F_dummy)
            
            # Compute transport matrices
            transport_full = solver.unbalanced_sinkhorn_algorithm(full_cost)
            transport_epi = solver_epi_only.unbalanced_sinkhorn_algorithm(epi_cost)
            transport_color = solver_color_only.unbalanced_sinkhorn_algorithm(color_cost)
            
            # Analyze statistics
            stats_full = visualizer.analyze_transport_statistics(
                transport_full.cpu().numpy(), f"{scenario_name} - Full Cost"
            )
            stats_epi = visualizer.analyze_transport_statistics(
                transport_epi.cpu().numpy(), f"{scenario_name} - Epipolar Only"
            )
            stats_color = visualizer.analyze_transport_statistics(
                transport_color.cpu().numpy(), f"{scenario_name} - Color Only"
            )
            
            results[scenario_name] = {
                'full': stats_full,
                'epipolar': stats_epi,
                'color': stats_color
            }
            
            # Visualize cost matrices
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Full cost
            im1 = axes[0].imshow(full_cost.cpu().numpy(), cmap='hot', aspect='equal')
            axes[0].set_title(f'{scenario_name.title()} - Full Cost')
            axes[0].set_xlabel('Target Gaussians')
            axes[0].set_ylabel('Source Gaussians')
            plt.colorbar(im1, ax=axes[0])
            
            # Epipolar cost
            im2 = axes[1].imshow(epi_cost.cpu().numpy(), cmap='hot', aspect='equal')
            axes[1].set_title(f'{scenario_name.title()} - Epipolar Only')
            axes[1].set_xlabel('Target Gaussians')
            axes[1].set_ylabel('Source Gaussians')
            plt.colorbar(im2, ax=axes[1])
            
            # Color cost
            im3 = axes[2].imshow(color_cost.cpu().numpy(), cmap='hot', aspect='equal')
            axes[2].set_title(f'{scenario_name.title()} - Color Only')
            axes[2].set_xlabel('Target Gaussians')
            axes[2].set_ylabel('Source Gaussians')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            save_path = os.path.join(FIGURES_DIR, f'cost_components_{scenario_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved cost component analysis to {save_path}")
            plt.close()
            
            # Visualize transport matrices
            transport_matrices = {
                f'{scenario_name} - Full': transport_full.cpu().numpy(),
                f'{scenario_name} - Epipolar': transport_epi.cpu().numpy(),
                f'{scenario_name} - Color': transport_color.cpu().numpy()
            }
            
            visualizer.compare_transport_matrices(
                transport_matrices,
                save_path=f'transport_comparison_{scenario_name}.png'
            )
    
    # Create summary plot
    create_component_summary_plot(results)
    
    return results


def create_component_summary_plot(results: Dict):
    """Create a summary plot showing how each component affects transport quality."""
    scenarios = list(results.keys())
    components = ['full', 'epipolar', 'color']
    
    # Extract diagonal concentrations
    diagonal_concs = {comp: [] for comp in components}
    
    for scenario in scenarios:
        for comp in components:
            diagonal_concs[comp].append(results[scenario][comp]['diagonal_concentration'])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    x = np.arange(len(scenarios))
    width = 0.25
    
    ax1.bar(x - width, diagonal_concs['full'], width, label='Full Cost', alpha=0.8)
    ax1.bar(x, diagonal_concs['epipolar'], width, label='Epipolar Only', alpha=0.8)
    ax1.bar(x + width, diagonal_concs['color'], width, label='Color Only', alpha=0.8)
    
    ax1.set_xlabel('Transformation Scenario')
    ax1.set_ylabel('Diagonal Concentration')
    ax1.set_title('Transport Quality by Cost Component')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Line plot
    for comp in components:
        ax2.plot(scenarios, diagonal_concs[comp], 'o-', label=comp.title(), linewidth=2, markersize=8)
    
    ax2.set_xlabel('Transformation Scenario')
    ax2.set_ylabel('Diagonal Concentration')
    ax2.set_title('Transport Quality Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'cost_component_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved component summary to {save_path}")
    plt.close()


def analyze_weight_sensitivity():
    """Analyze how different weight combinations affect transport quality."""
    print("\n=== Analyzing Weight Sensitivity ===\n")
    
    generator = ToyProblemGenerator(seed=42)
    
    # Generate a challenging test case (combined transformation)
    gaussians1 = generator.generate_synthetic_gaussians(n_gaussians=20, color_mode='gradient')
    transform_params = TransformationParams(
        translation=np.array([0.2, 0.1]),
        rotation=np.pi/6,
        scale=1.2
    )
    gaussians2, correspondences = generator.generate_known_correspondences(
        gaussians1, transform_params
    )
    
    # Camera intrinsics
    K = np.array([
        [800, 0, 400],
        [0, 800, 400],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Test different weight combinations
    weight_combinations = [
        {'lambda_epipolar': 1.0, 'lambda_color': 0.0, 'name': 'Epipolar Only'},
        {'lambda_epipolar': 0.0, 'lambda_color': 1.0, 'name': 'Color Only'},
        {'lambda_epipolar': 1.0, 'lambda_color': 0.5, 'name': 'Epi:Color = 2:1'},
        {'lambda_epipolar': 0.5, 'lambda_color': 1.0, 'name': 'Epi:Color = 1:2'},
        {'lambda_epipolar': 1.0, 'lambda_color': 1.0, 'name': 'Equal Weights'},
        {'lambda_epipolar': 2.0, 'lambda_color': 0.5, 'name': 'Epi:Color = 4:1'},
        {'lambda_epipolar': 0.5, 'lambda_color': 2.0, 'name': 'Epi:Color = 1:4'},
    ]
    
    results = []
    transport_matrices = {}
    
    for weights in weight_combinations:
        print(f"Testing {weights['name']}...")
        
        solver = OptimalTransportSolver(
            gaussians1=gaussians1,
            gaussians2=gaussians2,
            k1=K, k2=K,
            epsilon=0.01,
            lambda_color=weights['lambda_color'],
            lambda_epipolar=weights['lambda_epipolar'],
            device='cpu'
        )
        
        with torch.no_grad():
            F_dummy = torch.eye(3, dtype=torch.float32)
            cost_matrix = solver.compute_cost_matrix_fundamental(F_dummy)
            transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
            
            # Calculate diagonal concentration
            transport_np = transport_matrix.cpu().numpy()
            diagonal_conc = np.trace(transport_np) / transport_np.sum()
            
            results.append({
                'name': weights['name'],
                'lambda_epipolar': weights['lambda_epipolar'],
                'lambda_color': weights['lambda_color'],
                'diagonal_concentration': diagonal_conc
            })
            
            transport_matrices[weights['name']] = transport_np
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot of diagonal concentrations
    names = [r['name'] for r in results]
    diagonal_concs = [r['diagonal_concentration'] for r in results]
    
    bars = ax1.bar(range(len(names)), diagonal_concs, alpha=0.7)
    ax1.set_xlabel('Weight Configuration')
    ax1.set_ylabel('Diagonal Concentration')
    ax1.set_title('Transport Quality vs Weight Configuration')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diagonal_concs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Scatter plot: epipolar weight vs color weight, colored by diagonal concentration
    epi_weights = [r['lambda_epipolar'] for r in results]
    color_weights = [r['lambda_color'] for r in results]
    
    scatter = ax2.scatter(epi_weights, color_weights, c=diagonal_concs, 
                         s=100, cmap='viridis', alpha=0.7)
    ax2.set_xlabel('Epipolar Weight (λ_epipolar)')
    ax2.set_ylabel('Color Weight (λ_color)')
    ax2.set_title('Weight Space vs Transport Quality')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Diagonal Concentration')
    
    # Add labels for each point
    for i, result in enumerate(results):
        ax2.annotate(f"{i+1}", (epi_weights[i], color_weights[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'weight_sensitivity_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved weight sensitivity analysis to {save_path}")
    plt.close()
    
    # Print results table
    print("\nWeight Sensitivity Results:")
    print("=" * 80)
    print(f"{'Configuration':<20} {'λ_epi':<8} {'λ_color':<8} {'Diagonal Conc.':<15}")
    print("-" * 80)
    for result in results:
        print(f"{result['name']:<20} {result['lambda_epipolar']:<8.1f} "
              f"{result['lambda_color']:<8.1f} {result['diagonal_concentration']:<15.4f}")
    
    return results


def main():
    """Run all cost function analysis tests."""
    print("🔍 Cost Function Analysis for Oracle Study")
    print("=" * 60)
    
    # Analyze individual cost components
    component_results = analyze_cost_components()
    
    # Analyze weight sensitivity
    weight_results = analyze_weight_sensitivity()
    
    print("\n" + "=" * 60)
    print("📊 Cost Function Analysis Summary:")
    print("- Current cost function uses ONLY epipolar + color terms")
    print("- lambda_mean and lambda_cov are NOT used in compute_cost_matrix_fundamental")
    print("- Weight balance significantly affects transport quality")
    print("- Different transformations respond differently to each component")
    
    print(f"\n✅ Cost function analysis completed!")
    print(f"Check {FIGURES_DIR} directory for visualizations.")


if __name__ == "__main__":
    main()