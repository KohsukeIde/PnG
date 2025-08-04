"""
Oracle Study Module

This module provides tools and experiments for validating the optimal transport solver
using synthetic data with known ground truth correspondences.

Key Components:
- core: Core utilities (ToyProblemGenerator, TransportMatrixVisualizer)
- experiments: Analysis scripts for validation
- results: Generated experimental results and visualizations
"""

from .core import ToyProblemGenerator, TransformationParams, NoiseParams, TransportMatrixVisualizer

__all__ = [
    'ToyProblemGenerator',
    'TransformationParams',
    'NoiseParams', 
    'TransportMatrixVisualizer'
]