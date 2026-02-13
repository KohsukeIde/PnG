"""Utility functions for loading and manipulating 2D Gaussians.

COORDINATE CONVENTION:
======================
The EM fitting (single_image_gaussian_mixture_em.py) stores Gaussian means as:
    means[:, 0] = Y (row index)
    means[:, 1] = X (column index)

This is the standard image/matrix convention where (row, col) = (y, x).

However, the OptimalTransportSolver and COLMAP expect:
    means[:, 0] = X (horizontal coordinate)
    means[:, 1] = Y (vertical coordinate)

This module provides functions that handle this conversion automatically.
All functions in this module output Gaussians in (X, Y) format.
"""

import os
import pickle
from typing import Optional, Tuple

import numpy as np

# Get project root (assumes this file is in src/utils/)
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_UTILS_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# Lazy import to avoid circular dependencies
_TwoDGaussians = None


def _get_twod_gaussians_class():
    """Lazy import of TwoDGaussians class."""
    global _TwoDGaussians
    if _TwoDGaussians is None:
        from src.primitive.twod_gaussians_rs import TwoDGaussians
        _TwoDGaussians = TwoDGaussians
    return _TwoDGaussians


def _decompose_covariances(covs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose 2x2 covariances into rotations and scales.

    Args:
        covs: Array of shape (K, 2, 2) in (x, y) coordinates.

    Returns:
        rotations: Array of shape (K,) with rotation angles in radians.
        scales: Array of shape (K, 2) with (sx, sy).
    """
    if covs.ndim != 3 or covs.shape[1:] != (2, 2):
        raise ValueError("covs must have shape (K, 2, 2)")

    eigvals, eigvecs = np.linalg.eigh(covs)
    scales = np.sqrt(np.maximum(eigvals, 1e-12))
    rotations = np.arctan2(eigvecs[..., 1, 0], eigvecs[..., 0, 0])
    return rotations.astype(covs.dtype, copy=False), scales.astype(covs.dtype, copy=False)


def convert_yx_to_xy(gaussians) -> "TwoDGaussians":
    """Convert Gaussians from (Y, X) to (X, Y) coordinate format.

    Args:
        gaussians: TwoDGaussians object with means in (Y, X) format

    Returns:
        New TwoDGaussians object with means in (X, Y) format
    """
    TwoDGaussians = _get_twod_gaussians_class()

    # Swap means: (Y, X) -> (X, Y)
    converted_means = np.column_stack([
        gaussians.means[:, 1],  # X (was column 1)
        gaussians.means[:, 0],  # Y (was column 0)
    ])

    # Swap covariance matrix elements
    # Original: [[Cyy, Cyx], [Cxy, Cxx]]
    # Target:   [[Cxx, Cxy], [Cyx, Cyy]]
    converted_covs = np.zeros_like(gaussians.covs)
    for i in range(len(gaussians.covs)):
        converted_covs[i, 0, 0] = gaussians.covs[i, 1, 1]  # Cxx
        converted_covs[i, 0, 1] = gaussians.covs[i, 1, 0]  # Cxy
        converted_covs[i, 1, 0] = gaussians.covs[i, 0, 1]  # Cyx
        converted_covs[i, 1, 1] = gaussians.covs[i, 0, 0]  # Cyy

    # Recompute rotations/scales from converted covariances to stay consistent.
    converted_rotations, converted_scales = _decompose_covariances(converted_covs)

    return TwoDGaussians(
        means=converted_means,
        covs=converted_covs,
        scales=converted_scales,
        rotations=converted_rotations,
        rgb=gaussians.rgb.copy(),
        alpha=gaussians.alpha.copy(),
    )


def rescale_gaussians(
    gaussians,
    scale_x: float,
    scale_y: float,
) -> "TwoDGaussians":
    """Rescale Gaussian coordinates and covariances.

    Args:
        gaussians: TwoDGaussians object (assumes X, Y format)
        scale_x: Scale factor for X coordinate
        scale_y: Scale factor for Y coordinate

    Returns:
        New TwoDGaussians object with rescaled coordinates
    """
    TwoDGaussians = _get_twod_gaussians_class()

    rescaled_means = gaussians.means.copy()
    rescaled_means[:, 0] *= scale_x
    rescaled_means[:, 1] *= scale_y

    scale_mat = np.array([[scale_x, 0], [0, scale_y]])
    rescaled_covs = np.zeros_like(gaussians.covs)
    for i in range(len(gaussians.covs)):
        rescaled_covs[i] = scale_mat @ gaussians.covs[i] @ scale_mat.T

    # Recompute rotations/scales from rescaled covariances to stay consistent.
    rescaled_rotations, rescaled_scales = _decompose_covariances(rescaled_covs)

    return TwoDGaussians(
        means=rescaled_means,
        covs=rescaled_covs,
        scales=rescaled_scales,
        rotations=rescaled_rotations,
        rgb=gaussians.rgb.copy(),
        alpha=gaussians.alpha.copy(),
    )


def load_gaussians(
    image_idx: int,
    base_dir: Optional[str] = None,
    rescale_to_full: bool = True,
    source_size: Tuple[int, int] = (384, 288),  # (width, height)
    target_size: Tuple[int, int] = (1554, 1162),  # (width, height)
) -> dict:
    """Load fitted Gaussians for a given image index.

    This function handles the coordinate conversion from (Y, X) to (X, Y)
    automatically, so the returned Gaussians are always in (X, Y) format.

    Args:
        image_idx: Image index (e.g., 0 for "0000.png")
        base_dir: Base directory for Gaussian pkl files.
                  Default: data/fitted_gs/scan63_images_resized_em_200
        rescale_to_full: If True, rescale coordinates from source_size to target_size
        source_size: (width, height) of the fitted images
        target_size: (width, height) of the full resolution images

    Returns:
        dict containing:
            - 'original_gaussians': TwoDGaussians in (X, Y) format
            - 'ot_mass': Optional OT mass array (if present in pkl)
            - 'S_k': Optional S_k array (if present in pkl)

    Raises:
        FileNotFoundError: If the pkl file doesn't exist
    """
    if base_dir is None:
        base_dir = os.path.join(PROJECT_ROOT, "data/fitted_gs/scan63_images_resized_em_200")

    pkl_path = os.path.join(base_dir, f"{image_idx:04d}", "gaussians.pkl")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Gaussian pkl not found: {pkl_path}")

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Get original Gaussians (in Y, X format from EM fitting)
    g_yx = data['original_gaussians']

    # Convert to X, Y format
    g_xy = convert_yx_to_xy(g_yx)

    # Rescale if requested
    if rescale_to_full:
        scale_x = target_size[0] / source_size[0]
        scale_y = target_size[1] / source_size[1]
        g_xy = rescale_gaussians(g_xy, scale_x, scale_y)

    # Build result dict
    result = {
        'original_gaussians': g_xy,
    }

    # Include optional fields if present
    if 'ot_mass' in data:
        result['ot_mass'] = data['ot_mass']
    if 'S_k' in data:
        result['S_k'] = data['S_k']

    return result


def load_gaussians_pair(
    idx1: int,
    idx2: int,
    base_dir: Optional[str] = None,
    rescale_to_full: bool = True,
) -> Tuple[dict, dict]:
    """Load Gaussians for a pair of images.

    Convenience function for loading two images at once.

    Args:
        idx1: First image index
        idx2: Second image index
        base_dir: Base directory for Gaussian pkl files
        rescale_to_full: If True, rescale to full resolution

    Returns:
        Tuple of (data1, data2) dicts
    """
    data1 = load_gaussians(idx1, base_dir=base_dir, rescale_to_full=rescale_to_full)
    data2 = load_gaussians(idx2, base_dir=base_dir, rescale_to_full=rescale_to_full)
    return data1, data2
