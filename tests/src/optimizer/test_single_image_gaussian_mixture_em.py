import os
import numpy as np
from src.optimizer.single_image_gaussian_mixture_em import SingleImageGaussianMixtureEM

def test_single_image_gaussian_mixture_em_initialization():
    """Test the initialization of SingleImageGaussianMixtureEM with a valid image."""
    
    image_path = os.path.join('data', 'tsukuba', 'scene1.row3.col1.ppm')
    assert os.path.exists(image_path), f"Test image not found: {image_path}"
    gmm = SingleImageGaussianMixtureEM(image_path)
    
    # Check if the image is correctly loaded
    assert isinstance(gmm.image, np.ndarray), "Image must be a numpy array"
    assert gmm.image.ndim == 3, "Image must have 3 dimensions"
    assert gmm.image.shape[2] == 3, "Image must have 3 color channels"
    assert gmm.image.dtype == float, "Image data type must be float"
    assert np.all((gmm.image >= 0) & (gmm.image <= 1)), "Image values must be between 0 and 1"

def test_single_image_gaussian_mixture_em_invalid_file():
    """Test the initialization with an invalid image file."""
    try:
        SingleImageGaussianMixtureEM('non_existent_image.ppm')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        assert True

# def test_single_image_gaussian_mixture_em_invalid_image():
#     """Test the initialization with an invalid image format (e.g., non-RGB image)."""
#     image_path = os.path.join('data', 'tsukuba', 'grayscale_image.png')
    
#     # Create a dummy grayscale image if it doesn't exist
#     if not os.path.exists(image_path):
#         from PIL import Image
#         dummy_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
#         Image.fromarray(dummy_image).save(image_path)
    
#     try:
#         SingleImageGaussianMixtureEM(image_path)
#         assert False, "Should have raised ValueError"
#     except ValueError:
#         assert True
