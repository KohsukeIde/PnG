from dataclasses import dataclass

import numpy as np


@dataclass
class TwoDGaussians:
    """Represents a collection of 2D Gaussians with associated properties.


    Attributes:
        means (np.ndarray): Array of shape [k, 2] representing the means of k Gaussians.
        covs (np.ndarray): Array of shape [k, 2, 2] representing the covariance matrices of k Gaussians.
        rgb (np.ndarray): Array of shape [k, 3] representing the RGB colors of k Gaussians.
        alpha (np.ndarray): Array of shape [k] representing the alpha values of k Gaussians.

    Properties:
        k (int): The number of Gaussians in the collection.
    """

    means: np.ndarray  # [k, 2, float]
    covs: np.ndarray  # [k,2, 2, float]
    rgb: np.ndarray  # [k, 3, float]
    alpha: np.ndarray  # [k, float]

    def __post_init__(self) -> None:
        """Validate the shape and dimensions of the data arrays.

        Raises:
            ValueError: If the arrays do not have the same number of Gaussians.
            ValueError: If the means are not 2D.
            ValueError: If the covariances are not 2x2 matrices.
            ValueError: If the RGB values do not have 3 channels.
            ValueError: If the alpha is not a 1D array.
        """
        # Check if all arrays have the same number of Gaussians
        if not (
            self.means.shape[0]
            == self.covs.shape[0]
            == self.rgb.shape[0]
            == self.alpha.shape[0]
        ):
            raise ValueError("All arrays must have the same number of Gaussians")

        # Check if means are 2D
        if self.means.shape[1] != 2:
            raise ValueError("Means should be 2D")

        # Check if covariances are 2x2 matrices
        if self.covs.shape[1:] != (2, 2):
            raise ValueError("Covariances should be 2x2 matrices")

        # Check if RGB values have 3 channels
        if self.rgb.shape[1] != 3:
            raise ValueError("RGB values should have 3 channels")

        # Check if alpha is a 1D array
        if self.alpha.ndim != 1:
            raise ValueError("Alpha should be a 1D array")

    @property
    def k(self) -> int:
        """Return the number of means, which is the number of Gaussians."""
        return self.means.shape[0]
