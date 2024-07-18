import os

import numpy as np

from src.primitive.twod_gaussians import TwoDGaussians
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer


def test_vanilla_2d_rasterizer_creatioin():
    """Test the creation of Vanilla2DRasterizer."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    assert rasterizer.height == 64
    assert rasterizer.width == 128


def test_vanilla_2d_rasterizer_rasterize():
    """Test rasterize method return image."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    k = 512
    obj = TwoDGaussians(
        np.random.rand(k, 2) * 100,
        np.random.rand(k, 2, 2) * 4,
        np.random.rand(k, 3) * 255,
        np.random.rand(k) * 10,
    )
    obj.covs[:, 0, 0] = np.abs(obj.covs[:, 0, 0])
    obj.covs[:, 1, 1] = np.abs(obj.covs[:, 1, 1])
    obj.covs[:, 0, 1] = -obj.covs[:, 1, 0]
    image = rasterizer.rasterize(obj)
    assert image.shape == (64, 128, 3)
    assert image.dtype == np.uint8


def test_vanilla_2d_rasterizer_rasterize_save_image():
    """Test rasterize method save image."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    k = 512
    obj = TwoDGaussians(
        np.random.rand(k, 2) * 100,
        np.random.rand(k, 2, 2) * 4,
        np.random.rand(k, 3) * 255,
        np.random.rand(k) * 10,
    )
    obj.covs[:, 0, 0] = np.abs(obj.covs[:, 0, 0])
    obj.covs[:, 1, 1] = np.abs(obj.covs[:, 1, 1])
    obj.covs[:, 0, 1] = -obj.covs[:, 1, 0]
    rasterizer.rasterize(obj, True)
    assert os.path.isfile("outputs/tmp.png")
