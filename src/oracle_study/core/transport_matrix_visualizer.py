#!/usr/bin/env python3
"""
Transport Matrix Visualizer for Oracle Study

This module provides tools to visualize and analyze transport matrices
to understand the behavior of the optimal transport solver.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Create experiment-specific figure directory
EXPERIMENT_NAME = "transport_matrix_analysis"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up to src/oracle_study
FIGURES_DIR = os.path.join(ORACLE_DIR, "results", EXPERIMENT_NAME, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
# import seaborn as sns  # Using matplotlib only
from typing import Tuple, Dict, Optional, List
import cv2

# Utility class - no need for these imports
# from src.oracle_study.toy_problem_generator import ToyProblemGenerator, TransformationParams
# from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver


class TransportMatrixVisualizer:
    """Visualizes and analyzes transport matrices from optimal transport solver."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 5), figures_dir: Optional[str] = None):
        """Initialize the visualizer."""
        self.figsize = figsize
        self.figures_dir = figures_dir or FIGURES_DIR
        
    def visualize_transport_matrix(
        self,
        transport_matrix: np.ndarray,
        title: str = "Transport Matrix",
        save_path: Optional[str] = None,
        show_colorbar: bool = True
    ) -> None:
        """
        Visualize a transport matrix as a heatmap.
        
        Args:
            transport_matrix: The transport matrix to visualize
            title: Title for the plot
            save_path: Optional path to save the figure
            show_colorbar: Whether to show the colorbar
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap using matplotlib
        im = plt.imshow(transport_matrix, cmap='hot', aspect='equal', interpolation='nearest')
        if show_colorbar:
            plt.colorbar(im)
        
        plt.title(title)
        plt.xlabel('Target Gaussians (Image 2)')
        plt.ylabel('Source Gaussians (Image 1)')
        
        if save_path:
            # Ensure save_path is in the figures directory
            if not os.path.isabs(save_path):
                save_path = os.path.join(self.figures_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved transport matrix visualization to {save_path}")
        
        plt.close()
    
    def compare_transport_matrices(
        self,
        matrices: Dict[str, np.ndarray],
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare multiple transport matrices side by side.
        
        Args:
            matrices: Dictionary mapping names to transport matrices
            save_path: Optional path to save the figure
        """
        n_matrices = len(matrices)
        fig, axes = plt.subplots(1, n_matrices, figsize=(5*n_matrices, 5))
        
        if n_matrices == 1:
            axes = [axes]
        
        for i, (name, matrix) in enumerate(matrices.items()):
            im = axes[i].imshow(matrix, cmap='hot', aspect='equal', interpolation='nearest')
            plt.colorbar(im, ax=axes[i])
            axes[i].set_title(name)
            axes[i].set_xlabel('Target Gaussians')
            axes[i].set_ylabel('Source Gaussians')
        
        plt.tight_layout()
        
        if save_path:
            # Ensure save_path is in the figures directory
            if not os.path.isabs(save_path):
                save_path = os.path.join(self.figures_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved transport matrix comparison to {save_path}")
        
        plt.close()
    
    def visualize_correspondences_on_images(
        self,
        gaussians1,
        gaussians2,
        transport_matrix: np.ndarray,
        threshold: float = 0.01,
        save_path: Optional[str] = None,
        max_correspondences: int = 50,
        selection: str = "row_argmax"  # "row_argmax" or "threshold_topk"
    ) -> None:
        """
        Visualize correspondences on synthetic images.
        
        Args:
            gaussians1: First set of Gaussians
            gaussians2: Second set of Gaussians  
            transport_matrix: Transport matrix
            threshold: Minimum transport value to show correspondence
            save_path: Optional path to save the figure
            max_correspondences: Maximum number of correspondences to show
        """
        # Create synthetic images by plotting Gaussians
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot first set of Gaussians
        pos1 = gaussians1.means
        colors1 = gaussians1.rgb
        scales1 = gaussians1.scales
        
        for i in range(len(pos1)):
            circle = plt.Circle(
                pos1[i], 
                radius=np.mean(scales1[i]) * 3,  # Scale for visibility
                color=colors1[i], 
                alpha=0.6
            )
            ax1.add_patch(circle)
            ax1.text(pos1[i, 0], pos1[i, 1], str(i), 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Dynamic limits for robustness across normalized/pixel coordinates
        x1_min, y1_min = np.min(pos1, axis=0)
        x1_max, y1_max = np.max(pos1, axis=0)
        pad_x1 = 0.1 * max(1e-6, x1_max - x1_min)
        pad_y1 = 0.1 * max(1e-6, y1_max - y1_min)
        ax1.set_xlim(x1_min - pad_x1, x1_max + pad_x1)
        ax1.set_ylim(y1_min - pad_y1, y1_max + pad_y1)
        ax1.set_aspect('equal')
        ax1.set_title('Source Gaussians (Image 1)')
        ax1.grid(True, alpha=0.3)
        
        # Plot second set of Gaussians
        pos2 = gaussians2.means
        colors2 = gaussians2.rgb
        scales2 = gaussians2.scales
        
        for i in range(len(pos2)):
            circle = plt.Circle(
                pos2[i], 
                radius=np.mean(scales2[i]) * 3,  # Scale for visibility
                color=colors2[i], 
                alpha=0.6
            )
            ax2.add_patch(circle)
            ax2.text(pos2[i, 0], pos2[i, 1], str(i), 
                    ha='center', va='center', fontsize=8, fontweight='bold')
        
        x2_min, y2_min = np.min(pos2, axis=0)
        x2_max, y2_max = np.max(pos2, axis=0)
        pad_x2 = 0.1 * max(1e-6, x2_max - x2_min)
        pad_y2 = 0.1 * max(1e-6, y2_max - y2_min)
        ax2.set_xlim(x2_min - pad_x2, x2_max + pad_x2)
        ax2.set_ylim(y2_min - pad_y2, y2_max + pad_y2)
        ax2.set_aspect('equal')
        ax2.set_title('Target Gaussians (Image 2)')
        ax2.grid(True, alpha=0.3)
        
        # Draw correspondence lines using ConnectionPatch between subplots
        correspondences = []
        if selection == "row_argmax":
            # take best j per source i
            best_j = np.argmax(transport_matrix, axis=1)
            vals = transport_matrix[np.arange(transport_matrix.shape[0]), best_j]
            order = np.argsort(vals)[::-1]
            for idx in order[:max_correspondences]:
                i = int(idx)
                j = int(best_j[idx])
                v = float(vals[idx])
                if v >= threshold:
                    correspondences.append((i, j, v))
        else:
            sel = np.where(transport_matrix > threshold)
            vals = transport_matrix[sel]
            order = np.argsort(vals)[::-1][:max_correspondences]
            for k in order:
                i = int(sel[0][k])
                j = int(sel[1][k])
                v = float(vals[k])
                correspondences.append((i, j, v))

        for rank, (i, j, v) in enumerate(correspondences):
            p1 = (pos1[i, 0], pos1[i, 1])
            p2 = (pos2[j, 0], pos2[j, 1])
            lw = 1.0 + 4.0 * (v / (np.max(transport_matrix) + 1e-12))
            alpha = min(1.0, 0.2 + 0.8 * (v / (np.max(transport_matrix) + 1e-12)))
            con = ConnectionPatch(xyA=p2, coordsA=ax2.transData,
                                  xyB=p1, coordsB=ax1.transData,
                                  axesA=ax2, axesB=ax1,
                                  color='cyan', linewidth=lw, alpha=alpha)
            fig.add_artist(con)
            if rank < 10:
                print(f"Correspondence {i} -> {j}: transport = {v:.4f}")
        
        plt.tight_layout()
        
        if save_path:
            # Ensure save_path is in the figures directory
            if not os.path.isabs(save_path):
                save_path = os.path.join(self.figures_dir, save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved correspondence visualization to {save_path}")
        
        plt.close()
    
    def analyze_transport_statistics(
        self,
        transport_matrix: np.ndarray,
        name: str = "Transport Matrix"
    ) -> Dict[str, float]:
        """
        Analyze statistical properties of a transport matrix.
        
        Args:
            transport_matrix: The transport matrix to analyze
            name: Name for the analysis
            
        Returns:
            Dictionary with statistical metrics
        """
        stats = {}
        
        # Basic statistics
        stats['min'] = float(transport_matrix.min())
        stats['max'] = float(transport_matrix.max())
        stats['mean'] = float(transport_matrix.mean())
        stats['sum'] = float(transport_matrix.sum())
        stats['std'] = float(transport_matrix.std())
        
        # Sparsity analysis
        nonzero_count = np.count_nonzero(transport_matrix > 1e-6)
        total_elements = transport_matrix.size
        stats['sparsity'] = 1.0 - (nonzero_count / total_elements)
        stats['nonzero_ratio'] = nonzero_count / total_elements
        
        # Concentration analysis (how concentrated the mass is)
        flat_matrix = transport_matrix.flatten()
        sorted_values = np.sort(flat_matrix)[::-1]
        
        # Top 1%, 5%, 10% concentration
        for percent in [1, 5, 10]:
            n_elements = int(total_elements * percent / 100)
            top_sum = np.sum(sorted_values[:n_elements])
            stats[f'top_{percent}percent_concentration'] = top_sum / stats['sum']
        
        # Diagonal concentration (for square matrices)
        if transport_matrix.shape[0] == transport_matrix.shape[1]:
            diagonal_sum = np.trace(transport_matrix)
            stats['diagonal_concentration'] = diagonal_sum / stats['sum']
        
        # Row and column statistics
        row_sums = np.sum(transport_matrix, axis=1)
        col_sums = np.sum(transport_matrix, axis=0)
        
        stats['row_sum_mean'] = float(row_sums.mean())
        stats['row_sum_std'] = float(row_sums.std())
        stats['col_sum_mean'] = float(col_sums.mean())
        stats['col_sum_std'] = float(col_sums.std())
        
        # Entropy (measure of uncertainty)
        # Normalize to get probabilities
        if stats['sum'] > 0:
            prob_matrix = transport_matrix / stats['sum']
            # Avoid log(0) by adding small epsilon
            prob_matrix_safe = prob_matrix + 1e-12
            entropy = -np.sum(prob_matrix * np.log(prob_matrix_safe))
            stats['entropy'] = float(entropy)
        else:
            stats['entropy'] = 0.0
        
        print(f"\n=== {name} Statistics ===")
        print(f"Shape: {transport_matrix.shape}")
        print(f"Sum: {stats['sum']:.6f}")
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Max: {stats['max']:.6f}")
        print(f"Sparsity: {stats['sparsity']:.4f} ({stats['nonzero_ratio']:.4f} nonzero)")
        print(f"Top 1% concentration: {stats['top_1percent_concentration']:.4f}")
        print(f"Top 5% concentration: {stats['top_5percent_concentration']:.4f}")
        if 'diagonal_concentration' in stats:
            print(f"Diagonal concentration: {stats['diagonal_concentration']:.4f}")
        print(f"Entropy: {stats['entropy']:.4f}")
        
        return stats


if __name__ == "__main__":
    print("TransportMatrixVisualizer is a utility class.")
    print("For transport matrix analysis, run: python oracle_study/experiments/transport_matrix_analysis.py")