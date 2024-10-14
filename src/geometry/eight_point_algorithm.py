import cv2
import numpy as np

def estimate_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Estimate the fundamental matrix using OpenCV's findFundamentalMat function.

    Args:
        points1 (np.ndarray): Array of shape (N, 2) containing points from the first image.
        points2 (np.ndarray): Array of shape (N, 2) containing corresponding points from the second image.

    Returns:
        np.ndarray: Estimated fundamental matrix of shape (3, 3).
    """
    points1 = points1.reshape(-1, 1, 2)
    points2 = points2.reshape(-1, 1, 2)

    # Use 8-point algorithm (cv2.FM_8POINT)
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_8POINT)

    # Normalize F so that ||F||_Fro = 1
    F /= np.linalg.norm(F)

    return F
