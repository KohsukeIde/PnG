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



    def generate_epipolar_correspondences(
        self,
        n_gaussians: int,
        K: np.ndarray,
        R_wc: np.ndarray,
        t_wc: np.ndarray,
        depth_range: tuple = (2.0, 6.0),
        norm_xy_range: tuple = (-0.5, 0.5),
        scale3d_range: tuple = (0.02, 0.08),
        color_mode: str = 'gradient'
    ) -> Tuple[TwoDGaussians, TwoDGaussians, np.ndarray, np.ndarray]:
        """
        Generate epipolar-consistent 3D Gaussian splats, then project to 2D Gaussians for two views.

        - Camera-1 is identity pose (world == cam1)
        - Camera-2 has pose (R_wc, t_wc) (world->cam2)
        - 3D covariance is built from random unit quaternions and axis scales
        - 2D covariance is obtained via local Jacobian projection (see initial_3d_non_linear)
        Returns (gaussians1_2d, gaussians2_2d, correspondences, F_gt)
        """
        from src.reconstructor.initial_3d_non_linear import project_covariance_3d_to_2d

        # 1) Sample 3D GS means in cam1/world frame
        z_vals = self.rng.uniform(depth_range[0], depth_range[1], size=(n_gaussians,)).astype(np.float32)
        x_norm = self.rng.uniform(norm_xy_range[0], norm_xy_range[1], size=(n_gaussians,)).astype(np.float32)
        y_norm = self.rng.uniform(norm_xy_range[0], norm_xy_range[1], size=(n_gaussians,)).astype(np.float32)
        X_world = np.stack([x_norm * z_vals, y_norm * z_vals, z_vals], axis=1).astype(np.float32)

        # 2) Sample 3D Gaussian orientations (unit quaternion) and scales
        def random_unit_quaternion(num: int) -> np.ndarray:
            u1 = self.rng.rand(num)
            u2 = self.rng.rand(num) * 2 * np.pi
            u3 = self.rng.rand(num) * 2 * np.pi
            qw = np.sqrt(1 - u1) * np.sin(u2)
            qx = np.sqrt(1 - u1) * np.cos(u2)
            qy = np.sqrt(u1) * np.sin(u3)
            qz = np.sqrt(u1) * np.cos(u3)
            return np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)

        q_world = random_unit_quaternion(n_gaussians)
        s3 = self.rng.uniform(scale3d_range[0], scale3d_range[1], size=(n_gaussians, 3)).astype(np.float32)

        # 3) Helper to build Sigma3 from quaternion and scales (R diag(s)^2 R^T)
        def quat_to_R(q: np.ndarray) -> np.ndarray:
            qw, qx, qy, qz = q
            n = np.linalg.norm(q)
            if n < 1e-8:
                return np.eye(3, dtype=np.float32)
            qw, qx, qy, qz = q / n
            return np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)]
            ], dtype=np.float32)

        Sigma3_list = []
        for i in range(n_gaussians):
            R3 = quat_to_R(q_world[i])
            S = np.diag(s3[i]**2)
            Sigma3_list.append(R3 @ S @ R3.T)
        Sigma3 = np.stack(Sigma3_list, axis=0)

        # 4) Project means to pixels for both views
        ones = np.ones(n_gaussians, dtype=np.float32)
        x1 = X_world[:, 0] / X_world[:, 2]
        y1 = X_world[:, 1] / X_world[:, 2]
        p1_h = np.stack([x1, y1, ones], axis=1)
        p1_pix = (K @ p1_h.T).T[:, :2].astype(np.float32)

        X_cam2 = (R_wc @ X_world.T + t_wc.reshape(3, 1)).T
        x2 = X_cam2[:, 0] / X_cam2[:, 2]
        y2 = X_cam2[:, 1] / X_cam2[:, 2]
        p2_h = np.stack([x2, y2, ones], axis=1)
        p2_pix = (K @ p2_h.T).T[:, :2].astype(np.float32)

        # 5) Project 3D covariances to 2D covariances for both views
        R_cam1 = np.eye(3, dtype=np.float32)
        t_cam1 = np.zeros(3, dtype=np.float32)
        covs2d_1 = np.zeros((n_gaussians, 2, 2), dtype=np.float32)
        covs2d_2 = np.zeros((n_gaussians, 2, 2), dtype=np.float32)
        for i in range(n_gaussians):
            covs2d_1[i] = project_covariance_3d_to_2d(Sigma3[i], X_world[i], K, R_cam1, t_cam1).astype(np.float32)
            covs2d_2[i] = project_covariance_3d_to_2d(Sigma3[i], X_world[i], K, R_wc, t_wc).astype(np.float32)

        # small regularization to ensure PD
        for i in range(n_gaussians):
            covs2d_1[i] += np.eye(2, dtype=np.float32) * 1e-6
            covs2d_2[i] += np.eye(2, dtype=np.float32) * 1e-6

        # 6) Extract 2D scales and rotations from 2D covariances
        def cov2d_to_params(C: np.ndarray) -> Tuple[np.ndarray, float]:
            w, v = np.linalg.eigh(C)
            w = np.clip(w, 1e-9, None)
            idx = np.argsort(w)[::-1]
            w = w[idx]
            v = v[:, idx]
            scales = np.sqrt(w).astype(np.float32)
            angle = float(np.arctan2(v[1, 0], v[0, 0]))
            return scales, angle

        scales1 = np.zeros((n_gaussians, 2), dtype=np.float32)
        scales2 = np.zeros((n_gaussians, 2), dtype=np.float32)
        rot1 = np.zeros(n_gaussians, dtype=np.float32)
        rot2 = np.zeros(n_gaussians, dtype=np.float32)
        for i in range(n_gaussians):
            s1, a1 = cov2d_to_params(covs2d_1[i])
            s2, a2 = cov2d_to_params(covs2d_2[i])
            scales1[i] = s1
            scales2[i] = s2
            rot1[i] = a1
            rot2[i] = a2

        # 7) Colors and alpha
        if color_mode == 'gradient':
            colors = np.zeros((n_gaussians, 3), dtype=np.float32)
            for i in range(n_gaussians):
                t = i / max(1, n_gaussians - 1)
                colors[i] = [t, 1.0 - t, 0.5]
        elif color_mode == 'random':
            colors = self.rng.uniform(0, 1, size=(n_gaussians, 3)).astype(np.float32)
        else:
            colors = np.ones((n_gaussians, 3), dtype=np.float32) * 0.5
        alpha = np.ones(n_gaussians, dtype=np.float32) * 0.8

        g1 = TwoDGaussians(
            means=p1_pix,
            covs=covs2d_1,
            rgb=colors.copy(),
            alpha=alpha.copy(),
            rotations=rot1,
            scales=scales1
        )
        g2 = TwoDGaussians(
            means=p2_pix,
            covs=covs2d_2,
            rgb=colors.copy(),
            alpha=alpha.copy(),
            rotations=rot2,
            scales=scales2
        )

        correspondences = np.column_stack([np.arange(n_gaussians), np.arange(n_gaussians)])

        # GT Fundamental matrix
        def skew(t: np.ndarray) -> np.ndarray:
            return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float32)
        K_inv = np.linalg.inv(K).astype(np.float32)
        F_gt = K_inv.T @ (skew(t_wc.astype(np.float32)) @ R_wc.astype(np.float32)) @ K_inv
        return g1, g2, correspondences, F_gt.astype(np.float32)
    
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
        
        # Noisy case
        noise_params = NoiseParams(position_noise=0.05, color_noise=0.1)
        noisy_gaussians = self.add_controlled_noise(base_gaussians, noise_params)
        # For noisy case, correspondences are identity (same indices)
        correspondences = np.array([[i, i] for i in range(len(base_gaussians.means))], dtype=np.int32)
        test_cases['noisy'] = (base_gaussians, noisy_gaussians, correspondences)
        
        return test_cases
    
    def add_controlled_noise(
        self,
        gaussians: TwoDGaussians,
        noise_params: NoiseParams
    ) -> TwoDGaussians:
        """
        Add controlled noise to Gaussians.
        
        Args:
            gaussians: Source Gaussians
            noise_params: Noise parameters
            
        Returns:
            Noisy Gaussians with same structure but perturbed values
        """
        # Copy original data
        means = gaussians.means.copy()
        covs = gaussians.covs.copy()
        rgb = gaussians.rgb.copy()
        alpha = gaussians.alpha.copy()
        rotations = gaussians.rotations.copy()
        scales = gaussians.scales.copy()
        
        # Add position noise
        if noise_params.position_noise > 0:
            position_noise = np.random.normal(0, noise_params.position_noise, means.shape)
            means += position_noise.astype(np.float32)
        
        # Add scale noise
        if noise_params.scale_noise > 0:
            scale_noise = np.random.normal(1.0, noise_params.scale_noise, scales.shape)
            scales *= scale_noise.astype(np.float32)
            # Ensure scales remain positive
            scales = np.maximum(scales, 0.01)
            
            # Recompute covariances with new scales
            for i in range(len(scales)):
                theta = rotations[i]
                s = scales[i]
                cos_r = np.cos(theta)
                sin_r = np.sin(theta)
                r = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
                s_matrix = np.diag(s**2)
                covs[i] = r @ s_matrix @ r.T
        
        # Add color noise
        if noise_params.color_noise > 0:
            color_noise = np.random.normal(0, noise_params.color_noise, rgb.shape)
            rgb += color_noise.astype(np.float32)
            # Clamp colors to valid range [0, 1]
            rgb = np.clip(rgb, 0.0, 1.0)
        
        return TwoDGaussians(
            means=means,
            covs=covs,
            rgb=rgb,
            alpha=alpha,
            rotations=rotations,
            scales=scales
        )
    
    def generate_color_only_change(
        self,
        gaussians: TwoDGaussians,
        new_color_mode: str = 'random'
    ) -> Tuple[TwoDGaussians, np.ndarray]:
        """
        Generate Gaussians with same positions/scales but different colors.
        
        Args:
            gaussians: Source Gaussians
            new_color_mode: Color mode for new Gaussians ('random', 'gradient', 'uniform')
            
        Returns:
            (color_changed_gaussians, correspondences)
            correspondences is identity mapping since positions are identical
        """
        n_gaussians = len(gaussians.means)
        
        # Generate new colors based on mode
        if new_color_mode == 'gradient':
            # Create a smooth gradient across Gaussians
            new_colors = np.zeros((n_gaussians, 3), dtype=np.float32)
            for i in range(n_gaussians):
                t = i / max(1, n_gaussians - 1)
                new_colors[i] = [t, 1.0 - t, 0.5]
        elif new_color_mode == 'random':
            new_colors = self.rng.uniform(0, 1, size=(n_gaussians, 3)).astype(np.float32)
        else:  # uniform
            new_colors = np.ones((n_gaussians, 3), dtype=np.float32) * 0.5
        
        # Create new Gaussians with same geometry but different colors
        new_gaussians = TwoDGaussians(
            means=gaussians.means.copy(),      # Same positions
            covs=gaussians.covs.copy(),        # Same covariances
            rgb=new_colors,                    # Different colors
            alpha=gaussians.alpha.copy(),      # Same alpha
            rotations=gaussians.rotations.copy(),  # Same rotations
            scales=gaussians.scales.copy()     # Same scales
        )
        
        # Create identity correspondences (same positions = same indices)
        correspondences = np.column_stack([np.arange(n_gaussians), np.arange(n_gaussians)])

        return new_gaussians, correspondences

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_gaussians(gaussians: TwoDGaussians) -> TwoDGaussians:
        """Create a deep copy of a TwoDGaussians structure."""
        return TwoDGaussians(
            means=gaussians.means.copy(),
            covs=gaussians.covs.copy(),
            rgb=gaussians.rgb.copy(),
            alpha=gaussians.alpha.copy(),
            rotations=gaussians.rotations.copy(),
            scales=gaussians.scales.copy(),
        )

    @staticmethod
    def rotation_matrix(axis: str, angle: float) -> np.ndarray:
        """Return a float32 rotation matrix for the requested axis."""
        axis = axis.lower()
        c, s = np.cos(angle), np.sin(angle)

        if axis in {"x", "pitch"}:
            return np.array(
                [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
                dtype=np.float32,
            )
        if axis in {"y", "yaw"}:
            return np.array(
                [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                dtype=np.float32,
            )
        if axis in {"z", "roll"}:
            return np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        raise ValueError(f"Unknown axis: {axis}")

    @classmethod
    def get_epipolar_standard_scenarios(cls) -> Dict[str, Dict[str, np.ndarray]]:
        """Return the core epipolar scenarios used across analyses."""

        def R_yaw(angle: float) -> np.ndarray:
            return cls.rotation_matrix("yaw", angle)

        return {
            "epi_translation": {
                "R_wc": R_yaw(0.0),
                "t_wc": np.array([0.25, 0.02, 0.0], dtype=np.float32),
            },
            "epi_yaw_rotation": {
                "R_wc": R_yaw(0.12),
                "t_wc": np.array([0.18, 0.01, 0.02], dtype=np.float32),
            },
            "epi_forward_scale_like": {
                "R_wc": R_yaw(0.02),
                "t_wc": np.array([0.05, 0.0, 0.15], dtype=np.float32),
            },
            "epi_combined": {
                "R_wc": R_yaw(0.10),
                "t_wc": np.array([0.25, 0.02, 0.05], dtype=np.float32),
            },
            "epi_color_change": {
                "R_wc": R_yaw(0.08),
                "t_wc": np.array([0.20, 0.01, 0.03], dtype=np.float32),
            },
        }

    @classmethod
    def get_epipolar_baseline_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return baseline epipolar scenarios (translation/rotation variants)."""

        return {
            "baseline_translation": {
                "R_wc": cls.rotation_matrix("yaw", 0.0),
                "t_wc": np.array([0.25, 0.02, 0.0], dtype=np.float32),
                "description": "Pure translation baseline",
            },
            "baseline_yaw": {
                "R_wc": cls.rotation_matrix("yaw", 0.12),
                "t_wc": np.array([0.18, 0.01, 0.02], dtype=np.float32),
                "description": "Yaw rotation with small translation",
            },
            "baseline_forward": {
                "R_wc": cls.rotation_matrix("yaw", 0.02),
                "t_wc": np.array([0.05, 0.0, 0.15], dtype=np.float32),
                "description": "Forward motion with minimal yaw",
            },
            "baseline_combined": {
                "R_wc": cls.rotation_matrix("yaw", 0.10),
                "t_wc": np.array([0.25, 0.02, 0.05], dtype=np.float32),
                "description": "Combined yaw and translation",
            },
        }

    @classmethod
    def get_epipolar_challenging_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return challenging scenarios for stress testing the solver."""

        scenarios: Dict[str, Dict[str, Any]] = {}

        for i, scale in enumerate([0.5, 1.5, 2.0, 3.0]):
            scenarios[f"scale_test_{i}"] = {
                "R_wc": cls.rotation_matrix("yaw", 0.1),
                "t_wc": np.array([0.2, 0.02, 0.05], dtype=np.float32) * scale,
                "description": f"Scale test with factor {scale}",
            }

        for i, angle in enumerate([0.3, 0.5, 0.7, 1.0]):
            scenarios[f"large_yaw_{i}"] = {
                "R_wc": cls.rotation_matrix("yaw", angle),
                "t_wc": np.array([0.3, 0.05, 0.05], dtype=np.float32),
                "description": f"Large yaw rotation {np.degrees(angle):.1f} degrees",
            }

        angles = [0.15, 0.2, 0.25]
        for i, (pitch, yaw, roll) in enumerate(
            zip(angles, angles[::-1], angles[1:] + [angles[0]])
        ):
            R = (
                cls.rotation_matrix("pitch", pitch)
                @ cls.rotation_matrix("yaw", yaw)
                @ cls.rotation_matrix("roll", roll)
            )
            scenarios[f"multi_axis_{i}"] = {
                "R_wc": R,
                "t_wc": np.array([0.3, 0.1, 0.08], dtype=np.float32),
                "description": (
                    f"Multi-axis rotation P{np.degrees(pitch):.0f}°"
                    f"Y{np.degrees(yaw):.0f}°R{np.degrees(roll):.0f}°"
                ),
            }

        scenarios.update(
            {
                "near_identity": {
                    "R_wc": cls.rotation_matrix("yaw", 0.01),
                    "t_wc": np.array([0.01, 0.002, 0.001], dtype=np.float32),
                    "description": "Near-identity transformation",
                },
                "large_baseline": {
                    "R_wc": cls.rotation_matrix("yaw", 0.6),
                    "t_wc": np.array([1.5, 0.3, 0.2], dtype=np.float32),
                    "description": "Large baseline stereo",
                },
                "pure_forward": {
                    "R_wc": cls.rotation_matrix("yaw", 0.0),
                    "t_wc": np.array([0.0, 0.0, 0.5], dtype=np.float32),
                    "description": "Pure forward motion (challenging for epipolar)",
                },
            }
        )

        return scenarios

    @classmethod
    def get_epipolar_noise_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return scenarios for testing robustness to synthetic noise."""

        base_R = cls.rotation_matrix("yaw", 0.15)
        base_t = np.array([0.3, 0.05, 0.08], dtype=np.float32)

        scenarios: Dict[str, Dict[str, Any]] = {
            "noise_baseline": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Baseline for noise tests",
                "color_noise": 0.0,
                "gaussian_noise": 0.0,
            }
        }

        for i, noise_level in enumerate([0.1, 0.2, 0.3, 0.5]):
            scenarios[f"color_noise_{i}"] = {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": f"Color noise level {noise_level}",
                "color_noise": noise_level,
                "gaussian_noise": 0.0,
            }

        for i, pos_noise in enumerate([0.5, 1.0, 2.0, 3.0]):
            scenarios[f"position_noise_{i}"] = {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": f"Position noise {pos_noise} pixels",
                "color_noise": 0.0,
                "gaussian_noise": pos_noise,
            }

        for i, (c_noise, p_noise) in enumerate([(0.1, 0.5), (0.2, 1.0), (0.3, 1.5)]):
            scenarios[f"combined_noise_{i}"] = {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": f"Combined noise C{c_noise} P{p_noise}",
                "color_noise": c_noise,
                "gaussian_noise": p_noise,
            }

        return scenarios

    @classmethod
    def get_epipolar_illumination_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return illumination-change scenarios."""

        base_R = cls.rotation_matrix("yaw", 0.12)
        base_t = np.array([0.25, 0.03, 0.05], dtype=np.float32)

        return {
            "illumination_baseline": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Baseline illumination",
                "illumination_change": "none",
            },
            "global_brightness": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Global brightness change",
                "illumination_change": "brightness",
                "brightness_factor": 0.7,
            },
            "contrast_change": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Contrast change",
                "illumination_change": "contrast",
                "contrast_factor": 1.5,
            },
            "color_shift": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Color channel shift",
                "illumination_change": "color_shift",
                "color_shift": np.array([0.1, -0.05, 0.08], dtype=np.float32),
            },
            "random_illumination": {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": "Random illumination changes",
                "illumination_change": "random",
            },
        }

    @classmethod
    def get_epipolar_occlusion_scenarios(cls) -> Dict[str, Dict[str, Any]]:
        """Return scenarios with partial occlusions."""

        base_R = cls.rotation_matrix("yaw", 0.18)
        base_t = np.array([0.35, 0.04, 0.06], dtype=np.float32)

        scenarios: Dict[str, Dict[str, Any]] = {}

        for i, occlusion_rate in enumerate([0.1, 0.2, 0.3, 0.5]):
            scenarios[f"occlusion_{i}"] = {
                "R_wc": base_R,
                "t_wc": base_t,
                "description": f"Random occlusion {occlusion_rate*100:.0f}%",
                "occlusion_rate": occlusion_rate,
                "occlusion_type": "random",
            }

        scenarios.update(
            {
                "left_occlusion": {
                    "R_wc": base_R,
                    "t_wc": base_t,
                    "description": "Left side occlusion",
                    "occlusion_type": "left_half",
                },
                "center_occlusion": {
                    "R_wc": base_R,
                    "t_wc": base_t,
                    "description": "Center region occlusion",
                    "occlusion_type": "center_circle",
                },
                "corner_occlusion": {
                    "R_wc": base_R,
                    "t_wc": base_t,
                    "description": "Corner occlusion",
                    "occlusion_type": "corners",
                },
            }
        )

        return scenarios

    @classmethod
    def get_epipolar_comprehensive_suite(
        cls,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Return (all scenarios, representative subset) for epipolar studies."""

        categories = {
            "baseline": cls.get_epipolar_baseline_scenarios(),
            "challenge": cls.get_epipolar_challenging_scenarios(),
            "noise": cls.get_epipolar_noise_scenarios(),
            "illumination": cls.get_epipolar_illumination_scenarios(),
            "occlusion": cls.get_epipolar_occlusion_scenarios(),
        }

        all_scenarios: Dict[str, Dict[str, Any]] = {}
        for prefix, group in categories.items():
            for name, params in group.items():
                all_scenarios[f"{prefix}_{name}"] = params

        baseline = categories["baseline"]
        challenge = categories["challenge"]
        noise = categories["noise"]
        illumination = categories["illumination"]

        optimization_subset = {
            "baseline_translation": baseline["baseline_translation"],
            "baseline_yaw": baseline["baseline_yaw"],
            "baseline_combined": baseline["baseline_combined"],
            "challenge_scale_test_1": challenge["scale_test_1"],
            "challenge_large_yaw_1": challenge["large_yaw_1"],
            "challenge_multi_axis_0": challenge["multi_axis_0"],
            "challenge_large_baseline": challenge["large_baseline"],
            "noise_color_noise_1": noise["color_noise_1"],
            "noise_position_noise_1": noise["position_noise_1"],
            "noise_combined_noise_0": noise["combined_noise_0"],
            "illumination_global_brightness": illumination["global_brightness"],
            "illumination_random_illumination": illumination["random_illumination"],
        }

        return all_scenarios, optimization_subset

    @classmethod
    def get_optuna_scenarios(cls) -> List[Dict[str, Any]]:
        """Return the scenario list used for Optuna hyperparameter search."""

        def R_yaw(angle: float) -> np.ndarray:
            return cls.rotation_matrix("yaw", angle)

        def R_pitch(angle: float) -> np.ndarray:
            return cls.rotation_matrix("pitch", angle)

        return [
            {"name": "pure_translation_x", "R_wc": R_yaw(0.0), "t_wc": np.array([0.3, 0.0, 0.0], dtype=np.float32), "color_noise": False},
            {"name": "pure_translation_y", "R_wc": R_yaw(0.0), "t_wc": np.array([0.0, 0.3, 0.0], dtype=np.float32), "color_noise": False},
            {"name": "pure_translation_z", "R_wc": R_yaw(0.0), "t_wc": np.array([0.0, 0.0, 0.2], dtype=np.float32), "color_noise": False},
            {"name": "pure_yaw_small", "R_wc": R_yaw(0.05), "t_wc": np.array([0.0, 0.0, 0.0], dtype=np.float32), "color_noise": False},
            {"name": "pure_yaw_large", "R_wc": R_yaw(0.15), "t_wc": np.array([0.0, 0.0, 0.0], dtype=np.float32), "color_noise": False},
            {"name": "pure_pitch", "R_wc": R_pitch(0.08), "t_wc": np.array([0.0, 0.0, 0.0], dtype=np.float32), "color_noise": False},
            {"name": "small_motion", "R_wc": R_yaw(0.03), "t_wc": np.array([0.1, 0.02, 0.01], dtype=np.float32), "color_noise": False},
            {"name": "medium_motion", "R_wc": R_yaw(0.08), "t_wc": np.array([0.2, 0.05, 0.03], dtype=np.float32), "color_noise": False},
            {"name": "large_motion", "R_wc": R_yaw(0.12), "t_wc": np.array([0.3, 0.08, 0.05], dtype=np.float32), "color_noise": False},
            {"name": "medium_motion_color_noise", "R_wc": R_yaw(0.08), "t_wc": np.array([0.2, 0.05, 0.03], dtype=np.float32), "color_noise": True},
            {"name": "large_motion_color_noise", "R_wc": R_yaw(0.12), "t_wc": np.array([0.3, 0.08, 0.05], dtype=np.float32), "color_noise": True},
            {"name": "forward_motion", "R_wc": R_yaw(0.02), "t_wc": np.array([0.02, 0.0, 0.15], dtype=np.float32), "color_noise": False},
            {"name": "backward_motion", "R_wc": R_yaw(0.02), "t_wc": np.array([0.02, 0.0, -0.10], dtype=np.float32), "color_noise": False},
        ]

    def apply_epipolar_scenario_effects(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        scenario: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple[TwoDGaussians, TwoDGaussians]:
        """Apply scenario-specific perturbations such as noise or occlusion."""

        rng = np.random.RandomState(seed if seed is not None else self.seed)
        g1 = self._clone_gaussians(gaussians1)
        g2 = self._clone_gaussians(gaussians2)

        color_noise = float(scenario.get("color_noise", 0.0) or 0.0)
        if color_noise > 0.0:
            delta = rng.normal(0.0, color_noise, g2.rgb.shape).astype(np.float32)
            g2.rgb = np.clip(g2.rgb + delta, 0.0, 1.0)

        gaussian_noise = float(scenario.get("gaussian_noise", 0.0) or 0.0)
        if gaussian_noise > 0.0:
            delta = rng.normal(0.0, gaussian_noise, g2.means.shape).astype(np.float32)
            g2.means += delta

        illumination = scenario.get("illumination_change")
        if illumination == "brightness":
            factor = float(scenario.get("brightness_factor", 1.0))
            g2.rgb = np.clip(g2.rgb * factor, 0.0, 1.0)
        elif illumination == "contrast":
            factor = float(scenario.get("contrast_factor", 1.0))
            g2.rgb = np.clip(0.5 + factor * (g2.rgb - 0.5), 0.0, 1.0)
        elif illumination == "color_shift":
            shift = np.asarray(scenario.get("color_shift", [0.0, 0.0, 0.0]), dtype=np.float32)
            g2.rgb = np.clip(g2.rgb + shift, 0.0, 1.0)
        elif illumination == "random":
            brightness = rng.uniform(0.6, 1.25)
            contrast = rng.uniform(0.8, 1.25)
            shift = rng.normal(0.0, 0.08, size=(1, 3)).astype(np.float32)
            g2.rgb = np.clip(0.5 + contrast * (g2.rgb - 0.5), 0.0, 1.0)
            g2.rgb = np.clip(g2.rgb * brightness + shift, 0.0, 1.0)

        occlusion_type = scenario.get("occlusion_type")
        if occlusion_type is not None:
            mask = np.zeros(g2.means.shape[0], dtype=bool)
            if occlusion_type == "random":
                rate = float(scenario.get("occlusion_rate", 0.0) or 0.0)
                if rate > 0.0:
                    count = max(1, int(round(rate * g2.means.shape[0])))
                    count = min(count, g2.means.shape[0])
                    idx = rng.choice(g2.means.shape[0], size=count, replace=False)
                    mask[idx] = True
            elif occlusion_type == "left_half":
                median_x = np.median(g2.means[:, 0])
                mask = g2.means[:, 0] <= median_x
            elif occlusion_type == "center_circle":
                center = g2.means.mean(axis=0)
                extents = np.ptp(g2.means, axis=0)
                radius = scenario.get("occlusion_radius")
                if radius is None:
                    radius = 0.35 * float(max(extents[0], extents[1]))
                distance = np.linalg.norm(g2.means - center, axis=1)
                mask = distance <= radius
            elif occlusion_type == "corners":
                x = g2.means[:, 0]
                y = g2.means[:, 1]
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                margin_x = 0.15 * max(1e-6, x_max - x_min)
                margin_y = 0.15 * max(1e-6, y_max - y_min)
                top_left = (x <= x_min + margin_x) & (y >= y_max - margin_y)
                top_right = (x >= x_max - margin_x) & (y >= y_max - margin_y)
                bottom_left = (x <= x_min + margin_x) & (y <= y_min + margin_y)
                bottom_right = (x >= x_max - margin_x) & (y <= y_min + margin_y)
                mask = top_left | top_right | bottom_left | bottom_right

            if mask.any():
                g2.alpha[mask] = np.clip(g2.alpha[mask] * 0.1, 0.0, 1.0)
                g2.rgb[mask] = np.clip(g2.rgb[mask] * 0.3, 0.0, 1.0)

        return g1, g2
