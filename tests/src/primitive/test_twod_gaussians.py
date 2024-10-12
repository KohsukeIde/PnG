import numpy as np
import pytest

from src.primitive.twod_gaussians_rs import TwoDGaussians


def test_twod_gaussians_creation():
    """Test the creation of TwoDGaussians instance."""
    obj = TwoDGaussians(
        np.random.rand(16, 2),
        np.random.rand(16, 2, 2),
        np.random.rand(16, 3),
        np.random.rand(16),
        np.random.rand(16),
        np.random.rand(16, 2),
    )
    assert obj.k == 16


def test_twod_gaussians_invalid_creation():
    """Test that creating TwoDGaussians with invalid data raises ValueError."""
    with pytest.raises(ValueError):
        TwoDGaussians(
            np.random.rand(16, 2),
            np.random.rand(15, 2, 2),
            np.random.rand(14, 3),
            np.random.rand(13),
            np.random.rand(12),
            np.random.rand(11, 2),
        )


def test_twod_gaussians_properties():
    """Test if the properties of TwoDGaussians are set correctly."""
    means = np.random.rand(16, 2)
    covs = np.random.rand(16, 2, 2)
    rgb = np.random.rand(16, 3)
    alpha = np.random.rand(16)
    rotations = np.random.rand(16)
    scales = np.random.rand(16, 2)

    obj = TwoDGaussians(means, covs, rgb, alpha, rotations, scales)

    assert np.all(obj.means == means)
    assert np.all(obj.covs == covs)
    assert np.all(obj.rgb == rgb)
    assert np.all(obj.alpha == alpha)
    assert np.all(obj.rotations == rotations)
    assert np.all(obj.scales == scales)


def test_twod_gaussians_k_property():
    """Test the k property of TwoDGaussians."""
    for k in [1, 5, 10, 20]:
        obj = TwoDGaussians(
            np.random.rand(k, 2),
            np.random.rand(k, 2, 2),
            np.random.rand(k, 3),
            np.random.rand(k),
            np.random.rand(k),
            np.random.rand(k, 2),
        )
        assert obj.k == k


def test_twod_gaussians_invalid_means():
    """Test that creating TwoDGaussians with invalid means raises ValueError."""
    with pytest.raises(ValueError, match="Means should be 2D"):
        TwoDGaussians(
            np.random.rand(16, 3),  # 3D instead of 2D
            np.random.rand(16, 2, 2),
            np.random.rand(16, 3),
            np.random.rand(16),
            np.random.rand(16),
            np.random.rand(16, 2),
        )


# def test_twod_gaussians_invalid_covs():
#     """Test that creating TwoDGaussians with invalid covariances raises ValueError."""
#     with pytest.raises(ValueError, match="Covariances should be 2x2 matrices"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 3, 3),  # 3x3 instead of 2x2
#             np.random.rand(16, 3),
#             np.random.rand(16),
#             np.random.rand(16),
#             np.random.rand(16, 2)
#         )

# def test_twod_gaussians_invalid_rgb():
#     """Test that creating TwoDGaussians with invalid RGB values raises ValueError."""
#     with pytest.raises(ValueError, match="RGB values should have 3 channels"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 4),  # 4 channels instead of 3
#             np.random.rand(16),
#             np.random.rand(16),
#             np.random.rand(16, 2)
#         )

# def test_twod_gaussians_invalid_alpha():
#     """Test that creating TwoDGaussians with invalid alpha values raises ValueError."""
#     with pytest.raises(ValueError, match="Alpha should be a 1D array"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 3),
#             np.random.rand(16, 1),  # 2D array instead of 1D
#             np.random.rand(16),
#             np.random.rand(16, 2)
#         )

# def test_twod_gaussians_invalid_rotations():
#     """Test that creating TwoDGaussians with invalid rotations raises ValueError."""
#     with pytest.raises(ValueError, match="Rotations should be a 1D array"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 3),
#             np.random.rand(16),
#             np.random.rand(16, 1),  # 2D array instead of 1D
#             np.random.rand(16, 2)
#         )

# def test_twod_gaussians_invalid_scales():
#     """Test that creating TwoDGaussians with invalid scales raises ValueError."""
#     with pytest.raises(ValueError, match="Scales should have shape \[k, 2\]"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 3),
#             np.random.rand(16),
#             np.random.rand(16),
#             np.random.rand(16, 3)  # [k, 3] instead of [k, 2]
#         )

# def test_twod_gaussians_different_k():
#     """Test that creating TwoDGaussians with different number of Gaussians raises ValueError."""
#     with pytest.raises(ValueError, match="All arrays must have the same number of Gaussians"):
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(15, 2, 2),
#             np.random.rand(14, 3),
#             np.random.rand(13),
#             np.random.rand(12),
#             np.random.rand(11, 2)
#         )
