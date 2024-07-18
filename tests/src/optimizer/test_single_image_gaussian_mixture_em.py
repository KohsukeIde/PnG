import os

import numpy as np

from src.optimizer.single_image_gaussian_mixture_em import SingleImageGaussianMixtureEM
from src.primitive.twod_gaussians import TwoDGaussians


def test_single_image_gaussian_mixture_em_initialization():
    """Test the initialization of SingleImageGaussianMixtureEM with a valid image."""
    image_path = os.path.join("data", "tsukuba", "scene1.row3.col1.ppm")
    assert os.path.exists(image_path), f"Test image not found: {image_path}"
    gmm = SingleImageGaussianMixtureEM(image_path)

    # Check if the image is correctly loaded
    assert isinstance(gmm.image, np.ndarray), "Image must be a numpy array"
    assert gmm.image.ndim == 3, "Image must have 3 dimensions"
    assert gmm.image.shape[2] == 3, "Image must have 3 color channels"
    assert gmm.image.dtype == float, "Image data type must be float"
    assert np.all(
        (gmm.image >= 0) & (gmm.image <= 1)
    ), "Image values must be between 0 and 1"


def test_single_image_gaussian_mixture_em_invalid_file():
    """Test the initialization with an invalid image file."""
    try:
        SingleImageGaussianMixtureEM("non_existent_image.ppm")
        raise AssertionError("FileNotFoundError was not raised")
    except FileNotFoundError:
        pass


def test_initialize_gaussians():
    """Test the initialization of Gaussians."""
    image_path = os.path.join("data", "tsukuba", "scene1.row3.col1.ppm")
    assert os.path.exists(image_path), f"Test image not found: {image_path}"

    gmm = SingleImageGaussianMixtureEM(image_path)

    n_gauss = 16
    gaussians = gmm.initialize_gaussians(n_gauss)

    # Check if the method returns a TwoDGaussians object
    assert isinstance(gaussians, TwoDGaussians), "Should return a TwoDGaussians object"

    # Check if the number of Gaussians is correct
    assert (
        gaussians.k == n_gauss
    ), f"Expected {n_gauss} Gaussians, but got {gaussians.k}"

    # Check the shapes of the Gaussian parameters
    assert gaussians.means.shape == (n_gauss, 2), "Incorrect shape for means"
    assert gaussians.covs.shape == (n_gauss, 2, 2), "Incorrect shape for covariances"
    assert gaussians.rgb.shape == (n_gauss, 3), "Incorrect shape for RGB values"
    assert gaussians.alpha.shape == (n_gauss,), "Incorrect shape for alpha values"

    # Check if the RGB values are within the correct range
    assert np.all(
        (gaussians.rgb >= 0) & (gaussians.rgb <= 1)
    ), "RGB values should be between 0 and 1"

    # Check if the alpha values are correctly set
    assert np.all(gaussians.alpha == 0.4), "Alpha values should be 0.4"


def test_e_step():
    """Test the E-step (responsibility calculation) of the EM algorithm."""
    # Create a dummy image and initialize SingleImageGaussianMixtureEM
    dummy_image = np.random.rand(100, 100, 3)
    gmm = SingleImageGaussianMixtureEM.__new__(SingleImageGaussianMixtureEM)
    gmm.image = dummy_image

    # Create dummy Gaussians
    n_gauss = 5
    means = np.random.rand(n_gauss, 2) * [100, 100]
    covs = np.array([np.eye(2) * 10 for _ in range(n_gauss)])
    rgb = np.random.rand(n_gauss, 3)
    alpha = np.random.rand(n_gauss)
    alpha /= np.sum(alpha)  # Normalize alpha
    gaussians = TwoDGaussians(means, covs, rgb, alpha)

    # Run E-step
    responsibilities = gmm.e_step(gaussians)
    # check responsibility dtype -> fails on "make lint"
    assert isinstance(
        responsibilities, np.ndarray
    ), "responsibilities should be a numpy array"
    assert (
        responsibilities.dtype == np.float64
    ), f"Expected dtype float64, but got {responsibilities.dtype}"
    # Check shape
    assert responsibilities.shape == (
        100,
        100,
        n_gauss,
    ), "Incorrect shape of responsibilities"

    # Check if responsibilities sum to 1 for each pixel
    np.testing.assert_allclose(
        np.sum(responsibilities, axis=2),
        1,
        atol=1e-5,
        err_msg="Responsibilities don't sum to 1",
    )

    # Check that all values are between 0 and 1
    assert np.all(
        (responsibilities >= 0) & (responsibilities <= 1)
    ), "Responsibilities should be between 0 and 1"


# TODO: not sure if this test is necessary -> グレイスケールとかでもできるべき？
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
