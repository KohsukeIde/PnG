import numpy as np

from src.primitive.twod_gaussians import TwoDGaussians


def test_twod_gaussians_creation():
    """Test the creation of TwoDGaussians instance."""
    obj = TwoDGaussians(
        np.random.rand(16, 2),
        np.random.rand(16, 2, 2),
        np.random.rand(16, 3),
        np.random.rand(16),
    )
    assert obj.k == 16


def test_twod_gaussians_invalid_creation():
    """Test that creating TwoDGaussians with invalid data raises ValueError."""
    try:
        TwoDGaussians(
            np.random.rand(16, 2),
            np.random.rand(15, 2, 2),
            np.random.rand(14, 3),
            np.random.rand(13),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("ValueError was not raised")


def test_twod_gaussians_properties():
    """Test if the properties of TwoDGaussians are set correctly."""
    means = np.random.rand(16, 2)
    covs = np.random.rand(16, 2, 2)
    rgb = np.random.rand(16, 3)
    alpha = np.random.rand(16)

    obj = TwoDGaussians(means, covs, rgb, alpha)

    assert np.all(obj.means == means)
    assert np.all(obj.covs == covs)
    assert np.all(obj.rgb == rgb)
    assert np.all(obj.alpha == alpha)


def test_twod_gaussians_k_property():
    """Test the k property of TwoDGaussians."""
    for k in [1, 5, 10, 20]:
        obj = TwoDGaussians(
            np.random.rand(k, 2),
            np.random.rand(k, 2, 2),
            np.random.rand(k, 3),
            np.random.rand(k),
        )
        assert obj.k == k


## followings was used to achieve 100% coverage
# def test_twod_gaussians_invalid_means():
#     """Test that creating TwoDGaussians with invalid means raises ValueError."""
#     try:
#         TwoDGaussians(
#             np.random.rand(16, 3),  # 3D instead of 2D
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 3),
#             np.random.rand(16)
#         )
#         assert False, "ValueError was not raised"
#     except ValueError as e:
#         assert str(e) == "Means should be 2D"

# def test_twod_gaussians_invalid_covs():
#     """Test that creating TwoDGaussians with invalid covariances raises ValueError."""
#     try:
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 3, 3),  # 3x3 instead of 2x2
#             np.random.rand(16, 3),
#             np.random.rand(16)
#         )
#         assert False, "ValueError was not raised"
#     except ValueError as e:
#         assert str(e) == "Covariances should be 2x2 matrices"

# def test_twod_gaussians_invalid_rgb():
#     """Test that creating TwoDGaussians with invalid RGB values raises ValueError."""
#     try:
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 4),  # 4 channels instead of 3
#             np.random.rand(16)
#         )
#         assert False, "ValueError was not raised"
#     except ValueError as e:
#         assert str(e) == "RGB values should have 3 channels"

# def test_twod_gaussians_invalid_alpha():
#     """Test that creating TwoDGaussians with invalid alpha values raises ValueError."""
#     try:
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(16, 2, 2),
#             np.random.rand(16, 3),
#             np.random.rand(16, 1)  # 2D array instead of 1D
#         )
#         assert False, "ValueError was not raised"
#     except ValueError as e:
#         assert str(e) == "Alpha should be a 1D array"

# def test_twod_gaussians_different_k():
#     """Test that creating TwoDGaussians with different number of Gaussians raises ValueError."""
#     try:
#         TwoDGaussians(
#             np.random.rand(16, 2),
#             np.random.rand(15, 2, 2),
#             np.random.rand(14, 3),
#             np.random.rand(13)
#         )
#         assert False, "ValueError was not raised"
#     except ValueError as e:
#         assert str(e) == "All arrays must have the same number of Gaussians"
