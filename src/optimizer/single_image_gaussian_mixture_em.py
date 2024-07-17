import numpy as np
from PIL import Image

class SingleImageGaussianMixtureEM:
    """
    conduct Gaussian Mixture Model optimization on a single image using the EM algorithm.
    """

    def __init__(self, image_path: str) -> None:
        """
        Initialize the SingleImageGaussianMixtureEM with an image file.

        Args:
            image_path (str): Path to the input image file.

        Raises:
            FileNotFoundError: If the specified image file does not exist.
            ValueError: If the image cannot be opened or processed.
        """
        try:
            with Image.open(image_path) as img:
                self.image = np.array(img).astype(float) / 255.0  
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")


        # if self.image.ndim != 3 or self.image.shape[2] != 3:
        #     raise ValueError("Input image must be a 3-channel color image")