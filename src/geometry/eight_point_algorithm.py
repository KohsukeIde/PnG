import numpy as np

def estimate_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Estimate the fundamental matrix using the 8-point algorithm.

    Args:
        points1 (np.ndarray): Array of shape (N, 2) containing points from the first image.
        points2 (np.ndarray): Array of shape (N, 2) containing corresponding points from the second image.

    Returns:
        np.ndarray: Estimated fundamental matrix of shape (3, 3).
    """
    assert points1.shape == points2.shape, "Input point arrays must have the same shape."
    assert points1.shape[0] >= 8, "At least 8 point correspondences are required."

    # Step 1: Normalize coordinates
    points1_norm, T1 = normalize_points(points1)
    points2_norm, T2 = normalize_points(points2)

    # Step 2: Build the design matrix A
    N = points1_norm.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = points1_norm[i]
        x2, y2 = points2_norm[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    # Step 3: Solve for F using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Step 4: Enforce rank 2 constraint on F
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0  # Set the smallest singular value to zero
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f

    # Step 5: Denormalize the fundamental matrix
    F_denorm = T2.T @ F_rank2 @ T1

    # Normalize F so that F[2,2] = 1
    if F_denorm[-1, -1] != 0:
        F_denorm /= F_denorm[-1, -1]

    return F_denorm

def normalize_points(points: np.ndarray) -> tuple:
    """
    Normalize points for numerical stability.

    Args:
        points (np.ndarray): Array of shape (N, 2) containing 2D points.

    Returns:
        tuple: Tuple containing:
            - normalized_points (np.ndarray): Normalized points of shape (N, 2).
            - T (np.ndarray): Normalization transformation matrix of shape (3, 3).
    """
    N = points.shape[0]
    centroid = np.mean(points, axis=0)
    shifted_points = points - centroid

    avg_dist = np.mean(np.sqrt(np.sum(shifted_points ** 2, axis=1)))
    scale = np.sqrt(2) / avg_dist

    T = np.array([
        [scale, 0,     -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,     0,     1]
    ])

    points_h = np.hstack((points, np.ones((N, 1))))
    normalized_points_h = (T @ points_h.T).T
    normalized_points = normalized_points_h[:, :2] / normalized_points_h[:, 2, np.newaxis]

    return normalized_points, T
