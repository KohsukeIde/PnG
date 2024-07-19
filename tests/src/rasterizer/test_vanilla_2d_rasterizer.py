import os

import numpy as np

from src.primitive.twod_gaussians import TwoDGaussians
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer


def test_vanilla_2d_rasterizer_creation():
    """Test the creation of Vanilla2DRasterizer."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    assert rasterizer.height == 64
    assert rasterizer.width == 128


def test_vanilla_2d_rasterizer_rasterize(twod_gaussians: TwoDGaussians):
    """Test rasterize method return image."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    image = rasterizer.rasterize(twod_gaussians)
    assert image.shape == (64, 128, 3)
    assert image.dtype == np.uint8


def test_vanilla_2d_rasterizer_rasterize_save_image(twod_gaussians: TwoDGaussians):
    """Test rasterize method save image."""
    rasterizer = Vanilla2DRasterizer(64, 128)
    rasterizer.rasterize(twod_gaussians, True)
    assert os.path.isfile("outputs/tmp.png")
