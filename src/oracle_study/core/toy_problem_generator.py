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
        color_mode: str = 'gradient'
    ) -> Tuple[TwoDGaussians, TwoDGaussians, np.ndarray, np.ndarray]:
        """
        Generate a pair of Gaussian sets consistent with epipolar geometry using
        a simple two-view pinhole camera model. Camera-1 is at identity pose.

        Returns (gaussians1, gaussians2, correspondences, F_gt)
        where F_gt = K^{-T} [t]_x R K^{-1} with R=R_wc, t=t_wc.
        Inputs:
          - K: 3x3 intrinsics (same for both views)
          - R_wc, t_wc: pose of camera-2 (world->camera2)
        """
        # 1) Sample 3D points in camera-1/world frame (camera-1 is identity)
        z_vals = self.rng.uniform(depth_range[0], depth_range[1], size=(n_gaussians,)).astype(np.float32)
        x_norm = self.rng.uniform(norm_xy_range[0], norm_xy_range[1], size=(n_gaussians,)).astype(np.float32)
        y_norm = self.rng.uniform(norm_xy_range[0], norm_xy_range[1], size=(n_gaussians,)).astype(np.float32)

        # Camera-1 coordinates (also world coordinates)
        X_cam1 = np.stack([x_norm * z_vals, y_norm * z_vals, z_vals], axis=1).astype(np.float32)

        # 2) Project to image-1 pixels
        x1 = X_cam1[:, 0] / X_cam1[:, 2]
        y1 = X_cam1[:, 1] / X_cam1[:, 2]
        ones = np.ones_like(x1)
        p1_h = np.stack([x1, y1, ones], axis=1).astype(np.float32)  # normalized
        p1_pix = (K @ p1_h.T).T[:, :2].astype(np.float32)

        # 3) Transform to camera-2 frame and project
        X_cam2 = (R_wc @ X_cam1.T + t_wc.reshape(3, 1)).T
        x2 = X_cam2[:, 0] / X_cam2[:, 2]
        y2 = X_cam2[:, 1] / X_cam2[:, 2]
        p2_h = np.stack([x2, y2, np.ones_like(x2)], axis=1).astype(np.float32)
        p2_pix = (K @ p2_h.T).T[:, :2].astype(np.float32)

        # 4) Build Gaussians (simple isotropic covariances)
        scales = self.rng.uniform(1.5, 3.5, size=(n_gaussians, 2)).astype(np.float32)
        rotations = np.zeros(n_gaussians, dtype=np.float32)
        covs = np.zeros((n_gaussians, 2, 2), dtype=np.float32)
        for i in range(n_gaussians):
            covs[i] = np.diag(scales[i] ** 2)

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

        gaussians1 = TwoDGaussians(
            means=p1_pix,
            covs=covs.copy(),
            rgb=colors.copy(),
            alpha=alpha.copy(),
            rotations=rotations.copy(),
            scales=scales.copy()
        )

        gaussians2 = TwoDGaussians(
            means=p2_pix,
            covs=covs.copy(),
            rgb=colors.copy(),
            alpha=alpha.copy(),
            rotations=rotations.copy(),
            scales=scales.copy()
        )

        # 5) Identity correspondences
        correspondences = np.column_stack([np.arange(n_gaussians), np.arange(n_gaussians)])

        # 6) Ground-truth Fundamental matrix
        #   E = [t]_x R,  F = K^{-T} E K^{-1}
        def skew(t: np.ndarray) -> np.ndarray:
            return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]], dtype=np.float32)

        K_inv = np.linalg.inv(K).astype(np.float32)
        K_inv_T = K_inv.T
        E = skew(t_wc.astype(np.float32)) @ R_wc.astype(np.float32)
        F_gt = K_inv_T @ E @ K_inv

        return gaussians1, gaussians2, correspondences, F_gt.astype(np.float32)

    def generate_epipolar_correspondences_3dgs(
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