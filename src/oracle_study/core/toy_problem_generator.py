#!/usr/bin/env python3
"""
Toy Problem Generator for Oracle Study

This module generates synthetic 2D Gaussian problems with known ground truth
correspondences for validating the optimal transport solver.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from src.primitive.twod_gaussians_rs import TwoDGaussians


@dataclass
class TransformationParams:
    """Parameters for geometric transformations."""
    translation: Optional[np.ndarray] = None  # [dx, dy]
    rotation: Optional[float] = None  # radians
    scale: Optional[float] = None  # scale factor
    
    def __post_init__(self):
        if self.translation is not None:
            self.translation = np.array(self.translation, dtype=np.float32)


@dataclass
class NoiseParams:
    """Parameters for noise injection."""
    position_noise: float = 0.0  # std dev for position noise
    scale_noise: float = 0.0     # std dev for scale noise
    color_noise: float = 0.0     # std dev for color noise


class ToyProblemGenerator:
    """Generates synthetic 2D Gaussian problems with known correspondences."""
    
    def __init__(self, seed: int = 42, device: str = 'cpu'):
        """Initialize the generator."""
        self.seed = seed
        self.device = device
        self.rng = np.random.RandomState(seed)
        
    def generate_synthetic_gaussians(
        self,
        n_gaussians: int = 20,
        bounds: Tuple[float, float] = (-1.0, 1.0),
        scale_range: Tuple[float, float] = (0.05, 0.15),
        color_mode: str = 'gradient'
    ) -> TwoDGaussians:
        """
        Generate synthetic 2D Gaussians.
        
        Args:
            n_gaussians: Number of Gaussians to generate
            bounds: (min, max) bounds for positions
            scale_range: (min, max) range for scales
            color_mode: 'gradient', 'random', or 'uniform'
            
        Returns:
            TwoDGaussians object
        """
        # Generate random positions
        positions = self.rng.uniform(
            bounds[0], bounds[1], 
            size=(n_gaussians, 2)
        ).astype(np.float32)
        
        # Generate random scales (isotropic for simplicity)
        scales = self.rng.uniform(
            scale_range[0], scale_range[1],
            size=(n_gaussians, 2)
        ).astype(np.float32)
        
        # Generate rotations (set to zero for simplicity)
        rotations = np.zeros(n_gaussians, dtype=np.float32)
        
        # Generate colors based on mode
        if color_mode == 'gradient':
            # Create a smooth gradient across Gaussians
            colors = np.zeros((n_gaussians, 3), dtype=np.float32)
            for i in range(n_gaussians):
                t = i / max(1, n_gaussians - 1)
                colors[i] = [t, 1.0 - t, 0.5]
        elif color_mode == 'random':
            colors = self.rng.uniform(0, 1, size=(n_gaussians, 3)).astype(np.float32)
        else:  # uniform
            colors = np.ones((n_gaussians, 3), dtype=np.float32) * 0.5
        
        # Generate alpha values
        alpha = np.ones(n_gaussians, dtype=np.float32) * 0.8
        
        # Generate covariance matrices from scales and rotations
        covs = np.zeros((n_gaussians, 2, 2), dtype=np.float32)
        for i in range(n_gaussians):
            # Create diagonal covariance matrix from scales
            cov = np.diag(scales[i] ** 2)
            # Apply rotation if needed (for now, no rotation)
            covs[i] = cov
        
        return TwoDGaussians(
            means=positions,
            covs=covs,
            rgb=colors,
            alpha=alpha,
            rotations=rotations,
            scales=scales
        )
    
    def apply_transformation(
        self,
        gaussians: TwoDGaussians,
        params: TransformationParams
    ) -> TwoDGaussians:
        """Apply geometric transformation to Gaussians."""
        # Copy the original Gaussians
        new_means = gaussians.means.copy()
        new_scales = gaussians.scales.copy()
        new_rotations = gaussians.rotations.copy()
        new_rgb = gaussians.rgb.copy()
        new_alpha = gaussians.alpha.copy()
        new_covs = gaussians.covs.copy()
        
        # Apply translation
        if params.translation is not None:
            new_means += params.translation
        
        # Apply rotation
        if params.rotation is not None:
            cos_r = np.cos(params.rotation)
            sin_r = np.sin(params.rotation)
            rotation_matrix = np.array([
                [cos_r, -sin_r],
                [sin_r, cos_r]
            ], dtype=np.float32)
            new_means = new_means @ rotation_matrix.T
        
        # Apply scale
        if params.scale is not None:
            new_means *= params.scale
            new_scales *= params.scale
            # Update covariance matrices
            new_covs *= (params.scale ** 2)
        
        return TwoDGaussians(
            means=new_means,
            covs=new_covs,
            rgb=new_rgb,
            alpha=new_alpha,
            rotations=new_rotations,
            scales=new_scales
        )
    
    def generate_known_correspondences(
        self,
        gaussians1: TwoDGaussians,
        transform_params: TransformationParams,
        noise_params: Optional[NoiseParams] = None
    ) -> Tuple[TwoDGaussians, np.ndarray]:
        """
        Generate a second set of Gaussians with known correspondences.
        
        Args:
            gaussians1: Source Gaussians
            transform_params: Transformation to apply
            noise_params: Optional noise parameters
            
        Returns:
            (transformed_gaussians, correspondences)
            correspondences is Nx2 array of [source_idx, target_idx]
        """
        # Apply transformation
        gaussians2 = self.apply_transformation(gaussians1, transform_params)
        
        # Add noise if specified
        if noise_params is not None:
            if noise_params.position_noise > 0:
                position_noise = self.rng.normal(
                    0, noise_params.position_noise,
                    gaussians2.means.shape
                ).astype(np.float32)
                gaussians2.means += position_noise
            
            if noise_params.scale_noise > 0:
                scale_noise = self.rng.normal(
                    0, noise_params.scale_noise,
                    gaussians2.scales.shape
                ).astype(np.float32)
                gaussians2.scales += scale_noise
                gaussians2.scales = np.maximum(gaussians2.scales, 0.01)  # Prevent negative scales
                
                # Update covariance matrices
                for i in range(len(gaussians2.scales)):
                    gaussians2.covs[i] = np.diag(gaussians2.scales[i] ** 2)
            
            if noise_params.color_noise > 0:
                color_noise = self.rng.normal(
                    0, noise_params.color_noise,
                    gaussians2.rgb.shape
                ).astype(np.float32)
                gaussians2.rgb += color_noise
                gaussians2.rgb = np.clip(gaussians2.rgb, 0, 1)  # Keep colors in valid range
        
        # Create correspondence matrix (identity for now)
        n_gaussians = len(gaussians1.means)
        correspondences = np.column_stack([
            np.arange(n_gaussians),
            np.arange(n_gaussians)
        ])
        
        return gaussians2, correspondences
    
    def create_test_cases(self) -> Dict[str, Tuple[TwoDGaussians, TwoDGaussians, np.ndarray]]:
        """Create standard test cases for validation."""
        base_gaussians = self.generate_synthetic_gaussians(n_gaussians=15, color_mode='gradient')
        
        test_cases = {}
        
        # Identity case
        gaussians2, correspondences = self.generate_known_correspondences(
            base_gaussians, TransformationParams()
        )
        test_cases['identical'] = (base_gaussians, gaussians2, correspondences)
        
        # Translation case
        gaussians2, correspondences = self.generate_known_correspondences(
            base_gaussians, TransformationParams(translation=np.array([0.3, 0.2]))
        )
        test_cases['translation'] = (base_gaussians, gaussians2, correspondences)
        
        # Rotation case
        gaussians2, correspondences = self.generate_known_correspondences(
            base_gaussians, TransformationParams(rotation=np.pi/4)
        )
        test_cases['rotation'] = (base_gaussians, gaussians2, correspondences)
        
        # Scale case
        gaussians2, correspondences = self.generate_known_correspondences(
            base_gaussians, TransformationParams(scale=1.5)
        )
        test_cases['scale'] = (base_gaussians, gaussians2, correspondences)
        
        # Combined case
        gaussians2, correspondences = self.generate_known_correspondences(
            base_gaussians, TransformationParams(
                translation=np.array([0.2, 0.1]),
                rotation=np.pi/6,
                scale=1.2
            )
        )
        test_cases['combined'] = (base_gaussians, gaussians2, correspondences)
        
        return test_cases