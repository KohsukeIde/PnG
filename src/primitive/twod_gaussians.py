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
    # rotation_angles: np.ndarray  # [k, float]
    # scale_x: np.ndarray  # [k, float]
    # scale_y: np.ndarray  # [k, float]

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
        
        self._covs = self.covs

    @property
    def k(self) -> int:
        """Return the number of means, which is the number of Gaussians."""
        return self.means.shape[0]
    


# @dataclass
# class TwoDGaussians:
#     """Represents a collection of 2D Gaussians with associated properties.
#     Attributes:
#         means (np.ndarray): Array of shape [k, 2] representing the means of k Gaussians.
#         covs (np.ndarray): Array of shape [k, 2, 2] representing the covariance matrices of k Gaussians.
#         rgb (np.ndarray): Array of shape [k, 3] representing the RGB colors of k Gaussians.
#         alpha (np.ndarray): Array of shape [k] representing the alpha values of k Gaussians.

#     Properties:
#         k (int): The number of Gaussians in the collection.
#     """

#     means: np.ndarray  # [k, 2, float]
#     scales: np.ndarray  # [k, 2, float]
#     rotation_angles: np.ndarray  # [k, float]
#     rgb: np.ndarray  # [k, 3, float]
#     alpha: np.ndarray  # [k, float]

#     def __post_init__(self) -> None:
#         """Validate the shape and dimensions of the data arrays."""
        
#         # Check if all arrays have the same number of Gaussians
#         if not (
#             self.means.shape[0]
#             == self.scales.shape[0]
#             == self.rotation_angles.shape[0]
#             == self.rgb.shape[0]
#             == self.alpha.shape[0]
#         ):
#             raise ValueError("All arrays must have the same number of Gaussians")

#         # Check if means are 2D
#         if self.means.shape[1] != 2:
#             raise ValueError("Means should be 2D")

#         # Check if scales are 2D
#         if self.scales.shape[1] != 2:
#             raise ValueError("Scales should be 2D")

#         # Check if rotation_angles is 1D
#         if self.rotation_angles.ndim != 1:
#             raise ValueError("Rotation angles should be a 1D array")

#         # Check if RGB values have 3 channels
#         if self.rgb.shape[1] != 3:
#             raise ValueError("RGB values should have 3 channels")

#         # Check if alpha is a 1D array
#         if self.alpha.ndim != 1:
#             raise ValueError("Alpha should be a 1D array")

#     @property
#     def k(self) -> int:
#         """Return the number of means, which is the number of Gaussians."""
#         return self.means.shape[0]

#     @property
#     def covs(self):
#         return self.compute_covs_from_params()

#     def compute_covs_from_params(self):
#         covs = np.zeros((self.k, 2, 2))
#         for i in range(self.k):
#             R = self.rotation_matrix(self.rotation_angles[i])
#             S = np.diag(self.scales[i])
#             covs[i] = R @ S @ S.T @ R.T
#         return covs

#     @staticmethod
#     def rotation_matrix(angle):
#         c, s = np.cos(angle), np.sin(angle)
#         return np.array([[c, -s], [s, c]])

#     @staticmethod
#     def cov_to_params(cov):
#         eigenvalues, eigenvectors = np.linalg.eigh(cov)
#         scales = np.sqrt(eigenvalues)
#         rotation_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
#         return scales, rotation_angle

#     @classmethod
#     def from_covs(cls, means, covs, rgb, alpha):
#         k = means.shape[0]
#         scales = np.zeros((k, 2))
#         rotation_angles = np.zeros(k)
#         for i in range(k):
#             scales[i], rotation_angles[i] = cls.cov_to_params(covs[i])
#         return cls(means, scales, rotation_angles, rgb, alpha)