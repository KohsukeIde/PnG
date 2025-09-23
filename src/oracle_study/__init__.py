"""
Oracle Study Module

This module provides tools and experiments for validating matching and camera-pose
estimation with synthetic data and ground-truth correspondences.

Key Components:
- core: Shared utilities (ToyProblemGenerator, TransportMatrixVisualizer)
- matching: Oracle-study experiments for correspondence estimation
- camera_pose: Oracle-study experiments for pose estimation (upcoming)
"""

from .core import ToyProblemGenerator, TransformationParams, NoiseParams, TransportMatrixVisualizer

__all__ = [
    'ToyProblemGenerator',
    'TransformationParams',
    'NoiseParams', 
    'TransportMatrixVisualizer'
]
