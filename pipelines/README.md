# Perspective-n-Gaussians

## Overview

Perspective-n-Gaussians (PnG) proposes a novel approach to overcoming the limitations of Gaussian Splatting (GS) in reconstructing complex 3D scenes, particularly where traditional Structure-from-Motion (SfM) techniques fail. This work addresses critical challenges by reframing image and geometry matching as optimal transport problems, enabling robust camera and geometry reconstruction even in challenging environments.

---

## Background: Challenges in Gaussian Splatting (GS)

### 1. Expressiveness Limitations of SfM addressed by GS
- **Texture-less surfaces:** Represented via Gaussian covariance matrices
- **Non-Lambertian reflections:** Handled by spherical harmonics functions
- **Semi-transparent materials:** Modeled with Gaussian weight parameters
- **Small deformations:** Captured through Gaussian rigid-body transformations

### 2. Optimization Limitations in GS
- Gradient-descent optimization confined within local minima
- Reliance on pixel intensity leads to convergence issues without good initialization
- Current GS methods require initializations from SfM solutions

**GS theoretically enables high-quality solutions unattainable by SfM, but lacks methods to reliably reach these solutions.**

---

## Problem from a 3D Computer Vision Perspective
Optimized GS generates detailed point clouds suitable for 3D recognition tasks. However, it remains underutilized because:

- View-dependent color changes allow solutions with incorrect geometry but correct renderings (local minima).
- Difficulty in assessing geometric accuracy from observations prevents GS-based dataset creation akin to LiDAR/SfM datasets.

---

## Main Hypothesis and Approach
We propose to **reformulate image-to-image and geometry-to-image matching as an optimal transport problem** using Gaussian mixture representations of 2D images, aiming to reconstruct accurate cameras and geometry for scenes problematic for SfM.

**Key insights:**
- Incorporate viewing-direction dependence and deformation into transport costs.
- Avoid explicit 1-to-1 feature matching, improving robustness.
- Preserve all observation-indistinguishable local minima.

**Redefinition of SfM’s 8-point and PnP algorithms using photometric error to ensure GS global optimality.**

---

## Core Hypotheses

### Hypothesis 1
**Gaussian mixtures represent 2D images using significantly fewer points than pixels.**

Advantages:
- Captures entire image structure, not limited to feature points.
- Gaussian centers flexibly represent the actual content.
- Covariance matrices encode local image structures, orientation, and scales.

### Hypothesis 2
**Optimal transport between Gaussian mixtures of two overlapping-view images yields effective correspondences.**

Implications:
- Transforms feature matching into a robust distribution-matching problem.
- Significantly reduces local minima challenges.

### Hypothesis 3
**Transport matrices from Hypothesis 2 enable the formulation of Perspective-n Gaussian (PnG).**

Implementation steps:
1. Use the 8-point algorithm with Gaussian-based 2D-2D correspondences to initialize geometry.
2. Leverage PnG to register new camera poses.
3. Iteratively triangulate new geometry.

### Hypothesis 4
**A Bundle Adjustment (BA) formulation initialized by PnG is possible (may be optional depending on Hypothesis 5).**

Goals:
- Joint optimization of camera parameters and 3D Gaussian points.
- Enhance stability and accuracy of reconstruction.

### Hypothesis 5
**Gaussian Splatting initialized by PnG and BA effectively reconstructs scenes where SfM fails.**

Expected outcomes:
- Improved quality in 3D reconstruction metrics (e.g., PSNR, SSIM, LPIPS).
- Enhanced optimization convergence speed.

---

## Expected Contributions
- Redefine classical SfM methods (8-point algorithm, PnP) within a photometric error optimization framework, ensuring global optimality.
- Enable GS reconstruction for previously challenging scenes, paving the way for practical datasets in 3D recognition tasks.

