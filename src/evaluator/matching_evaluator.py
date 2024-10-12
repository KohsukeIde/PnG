# src/evaluator/matching_evaluator.py

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from src.primitive.twod_gaussians import TwoDGaussians
from scipy.optimize import linear_sum_assignment

class MatchingEvaluator:
    def __init__(self, gaussians1: TwoDGaussians, gaussians2: TwoDGaussians, transport_matrix: np.ndarray):
        self.gaussians1 = gaussians1
        self.gaussians2 = gaussians2
        self.transport_matrix = transport_matrix
        self.matches = self.extract_matches()

    def extract_matches(self) -> list:
        """Extract matches based on the transport_matrix using Hungarian algorithm."""
        num_rows, num_cols = self.transport_matrix.shape
        size = max(num_rows, num_cols)
        cost_matrix = np.full((size, size), fill_value=np.max(self.transport_matrix) * 2)

        # Set the negative transport_matrix to maximize the total transport
        cost_matrix[:num_rows, :num_cols] = -self.transport_matrix

        print("Cost Matrix for Matching:")
        print(cost_matrix)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print("Row Indices:", row_ind)
        print("Column Indices:", col_ind)

        matches = []
        threshold = 1e-6  # Transport probability threshold
        for i, j in zip(row_ind, col_ind):
            if i < num_rows and j < num_cols and self.transport_matrix[i, j] > threshold:
                matches.append((int(i), int(j)))
        print("Filtered Matches:", matches)
        return matches

    def evaluate_matches(self) -> Dict[str, float]:
        """
        Method to evaluate the quality of matches.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if not self.matches:
            return {
                "average_distance": float('inf'),
                "average_color_difference": float('inf'),
                "matching_rate": 0.0
            }
        
        # Calculate average Euclidean distance
        distances = [np.linalg.norm(self.gaussians1.means[i] - self.gaussians2.means[j]) for i, j in self.matches]
        avg_distance = np.mean(distances)
        
        # Calculate average color difference
        color_diffs = [np.linalg.norm(self.gaussians1.rgb[i] - self.gaussians2.rgb[j]) for i, j in self.matches]
        avg_color_diff = np.mean(color_diffs)
        
        # Calculate matching rate (average of the number of Gaussians in both distributions)
        matching_rate = len(self.matches) / ((self.gaussians1.k + self.gaussians2.k) / 2)
        
        return {
            "average_distance": avg_distance,
            "average_color_difference": avg_color_diff,
            "matching_rate": matching_rate
        }

    # def visualize_matches(self, output_path: str) -> None:
    #     """
    #     Method to visualize matches and save to the specified path.

    #     Args:
    #         output_path (str): Path to save the visualization image.
    #     """
    #     fig, ax = plt.subplots(figsize=(12, 6))
        
    #     # Set offset for the right image
    #     offset_x = np.max(self.gaussians1.means[:, 0]) + 5  # X-coordinate offset with some margin
        
    #     # Gaussians1 visualization (left side)
    #     ax.scatter(self.gaussians1.means[:, 0], self.gaussians1.means[:, 1], 
    #             c=self.gaussians1.rgb, marker='o', label='Image1 Gaussians')
        
    #     # Gaussians2 visualization (right side)
    #     shifted_means2 = self.gaussians2.means + np.array([offset_x, 0])
    #     ax.scatter(shifted_means2[:, 0], shifted_means2[:, 1], 
    #             c=self.gaussians2.rgb, marker='x', label='Image2 Gaussians')
        
    #     # Visualize matching
    #     for i, j in self.matches:
    #         point1 = self.gaussians1.means[i]
    #         point2 = self.gaussians2.means[j] + np.array([offset_x, 0])
    #         ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--', linewidth=0.5)
        
    #     ax.legend()
    #     ax.set_title('Gaussian Matching Visualization Between Two Images')
    #     ax.set_xlabel('X-axis')
    #     ax.set_ylabel('Y-axis')
    #     plt.tight_layout()
    #     plt.savefig(output_path)
    #     plt.close()
    
    def visualize_matches(self, output_path: str, top_k: int = 100) -> None:
        """
        Visualize matches and save the figure to the specified path.

        Args:
            output_path (str): Path to save the visualization image.
            top_k (int, optional): Number of top matches to visualize. If None, visualize all matches.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set offset between the two images
        offset_x = np.max(self.gaussians1.means[:, 0]) + 5  # Offset for the right image
        
        # Visualize Gaussians1 (left side)
        ax.scatter(self.gaussians1.means[:, 0], self.gaussians1.means[:, 1], 
                c=self.gaussians1.rgb, marker='o', label='Image1 Gaussians')
        
        # Visualize Gaussians2 (shifted to the right)
        shifted_means2 = self.gaussians2.means + np.array([offset_x, 0])
        ax.scatter(shifted_means2[:, 0], shifted_means2[:, 1], 
                c=self.gaussians2.rgb, marker='x', label='Image2 Gaussians')
        
        # Get matches and associated transport values
        matches_with_transport = []
        for i, j in self.matches:
            transport_value = self.transport_matrix[i, j]
            matches_with_transport.append((i, j, transport_value))
        
        # Sort matches by transport value in descending order
        matches_with_transport.sort(key=lambda x: x[2], reverse=True)
        
        # Select top K matches if top_k is specified
        if top_k is not None:
            matches_with_transport = matches_with_transport[:top_k]
        
        # Visualize matches
        for i, j, transport_value in matches_with_transport:
            point1 = self.gaussians1.means[i]
            point2 = self.gaussians2.means[j] + np.array([offset_x, 0])
            ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'k--', linewidth=0.5)
        
        ax.legend()
        ax.set_title(f'Gaussian Matching Visualization (Top {top_k} Matches)' if top_k else 'Gaussian Matching Visualization')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
