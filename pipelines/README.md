# Perspective-n-Gaussians

## Overview

Perspective-n-Gaussians (PnG) proposes a novel approach to overcoming the limitations of Gaussian Splatting (GS) in reconstructing complex 3D scenes, particularly where traditional Structure-from-Motion (SfM) techniques fail (such as textureless or a scene with repeating texture. meaning that utilizing the traditional "feature points" is not possible). This work addresses critical challenges by reframing image and geometry matching as optimal transport problems, enabling robust camera and geometry reconstruction even in challenging environments while preserving all plausible solutions.

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
- Current GS methods require initializations from SfM solutions which provide only a single, potentially suboptimal solution

**GS theoretically enables high-quality solutions unattainable by SfM, but lacks methods to reliably reach these solutions due to poor initialization that discards alternative interpretations of the scene.**

---

## Problem from a 3D Computer Vision Perspective
Optimized GS generates detailed point clouds suitable for 3D recognition tasks. However, it remains underutilized because:

- View-dependent color changes allow solutions with incorrect geometry but correct renderings (local minima).
- Difficulty in assessing geometric accuracy from observations prevents GS-based dataset creation akin to LiDAR/SfM datasets.
- Traditional matching techniques make hard decisions early, discarding valid alternative interpretations of ambiguous scene regions.

---

## Main Hypothesis and Approach
We propose to **reformulate image-to-image and geometry-to-image matching as an optimal transport problem** using Gaussian mixture representations of 2D images, aiming to reconstruct accurate cameras and geometry for scenes problematic for SfM.

**Key insights:**
- **Preserve all plausible local solutions** by representing correspondences as transport probabilities rather than hard assignments
- Incorporate viewing-direction dependence and deformation into transport costs
- Avoid explicit 1-to-1 feature matching, improving robustness
- **Maintain ambiguity** where multiple interpretations are equally valid until further evidence can disambiguate

**The critical advantage of this optimal transport formulation is the preservation of all observation-compatible local minima, allowing the subsequent GS optimization to explore multiple geometrically consistent solutions simultaneously.**

---

## Core Hypotheses

### Hypothesis 1
**Gaussian mixtures represent 2D images using significantly fewer points than pixels.**

Advantages:
- Captures entire image structure, not limited to feature points
- Gaussian centers flexibly represent the actual content
- Covariance matrices encode local image structures, orientation, and scales

### Hypothesis 2
**Optimal transport between Gaussian mixtures of two overlapping-view images yields effective correspondences that preserve all plausible interpretations.**

Implications:
- Transforms feature matching into a robust distribution-matching problem
- Represents correspondences as probabilities rather than binary decisions
- **Crucially preserves multiple potential solutions** where there is ambiguity
- Significantly reduces premature commitment to incorrect local minima

### Hypothesis 3
**Transport matrices from Hypothesis 2 enable the formulation of Perspective-n Gaussian (PnG) that encodes multiple geometric interpretations.**

Implementation steps:
1. Use the 8-point algorithm with Gaussian-based 2D-2D correspondences to initialize geometry
2. Leverage PnG to register new camera poses while preserving alternative interpretations
3. Iteratively triangulate new geometry **without hard thresholding of transport values**
4. Allow multiple possible 3D locations for ambiguous features, to be resolved later in optimization

### Hypothesis 4
**A Bundle Adjustment (BA) formulation initialized by PnG can simultaneously optimize across multiple local solutions.**

Goals:
- Joint optimization of camera parameters and 3D Gaussian points
- Enhance stability and accuracy of reconstruction
- Consider multiple plausible geometric interpretations simultaneously

### Hypothesis 5
**Gaussian Splatting initialized by PnG and BA effectively reconstructs scenes where SfM fails by exploring the full solution space rather than a single local minimum.**

Expected outcomes:
- Improved quality in 3D reconstruction metrics (e.g., PSNR, SSIM, LPIPS)
- Enhanced optimization convergence speed
- **Ability to find globally optimal solutions by preserving local minima during initialization**
- Robustness to ambiguous scene elements like repetitive patterns and textureless regions

---

## Expected Contributions
- Redefine classical SfM methods within an optimal transport framework that preserves all plausible local solutions
- Eliminate premature thresholding and hard decisions in the reconstruction pipeline
- Enable GS reconstruction for previously challenging scenes by providing a richer initialization space
- Demonstrate that preserving local minima during initialization leads to better global solutions after optimization
- Pave the way for practical datasets in 3D recognition tasks that were previously impossible to reconstruct reliably