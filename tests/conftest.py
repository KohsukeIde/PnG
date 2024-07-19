from typing import Final

import numpy as np
import pytest

from src.primitive.twod_gaussians import TwoDGaussians


@pytest.fixture(scope="function")
def twod_gaussians() -> TwoDGaussians:
    """Generate 2D gaussians with restricted covariances."""
    # Gaussian parameter's settings
    k: Final[int] = 512
    mean_max: Final[float] = 100.0
    variance_max: Final[float] = 4.0
    color_max: Final[float] = 255.0
    alpha_max: Final[float] = 10.0
    # Generate gaussian parameters tensor
    mean: np.ndarray = np.random.rand(k, 2) * mean_max
    color: np.ndarray = np.random.rand(k, 3) * color_max
    alpha: np.ndarray = np.random.rand(k) * alpha_max
    # Generate covariance with restriction
    cov_s: np.ndarray = np.zeros((k, 2, 2))
    cov_s[:, 0, 0] = np.random.rand(k) * variance_max + 1.0
    cov_s[:, 1, 1] = np.random.rand(k) * variance_max + 1.0
    theta: np.ndarray = np.random.rand(k) * np.pi
    cov_r: np.ndarray = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    ).transpose(2, 0, 1)
    cov_rs = np.matmul(cov_r, cov_s)
    cov = np.matmul(cov_rs, cov_rs.transpose(0, 2, 1))
    # Generate 2D Gaussians
    obj = TwoDGaussians(mean, cov, color, alpha)
    return obj
