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
    # Generate 2D Gaussians
    obj = TwoDGaussians(
        np.random.rand(k, 2) * mean_max,
        np.random.rand(k, 2, 2) * variance_max,
        np.random.rand(k, 3) * color_max,
        np.random.rand(k) * alpha_max,
    )
    # Apply covariances restrictions
    obj.covs[:, 0, 0] = np.abs(obj.covs[:, 0, 0])
    obj.covs[:, 1, 1] = np.abs(obj.covs[:, 1, 1])
    obj.covs[:, 0, 1] = -obj.covs[:, 1, 0]
    return obj
