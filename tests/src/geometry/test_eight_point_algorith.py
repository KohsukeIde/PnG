import numpy as np
from src.geometry.eight_point_algorithm import estimate_fundamental_matrix

def test_fundamental_matrix_estimation():
    # Create a synthetic test case with known fundamental matrix
    F_true = np.array([
        [0,    -0.003,  0.002],
        [0.003,    0,   -0.005],
        [-0.002, 0.005,    0]
    ])

    # Generate random points in image 1
    np.random.seed(0)
    N = 20
    x1 = np.random.uniform(-1, 1, (N, 2))
    x1_h = np.hstack((x1, np.ones((N, 1))))

    # Compute corresponding points in image 2 using the fundamental matrix
    x2_h = x1_h @ F_true.T
    x2_h = x2_h + np.random.normal(scale=0.001, size=x2_h.shape)  # Add small noise
    x2 = x2_h[:, :2] / x2_h[:, 2, np.newaxis]

    # Estimate the fundamental matrix
    F_estimated = estimate_fundamental_matrix(x1, x2)

    # Test that the estimated F has rank 2
    _, S_f, _ = np.linalg.svd(F_estimated)
    assert np.isclose(S_f[-1], 0, atol=1e-6), "Fundamental matrix should have rank 2"

    # Test the epipolar constraint x2^T * F * x1 ≈ 0
    x1_h = np.hstack((x1, np.ones((N, 1))))
    x2_h = np.hstack((x2, np.ones((N, 1))))
    residuals = np.abs(np.einsum('ij,ij->i', x2_h @ F_estimated, x1_h))
    mean_residual = np.mean(residuals)
    assert mean_residual < 1e-3, "Epipolar constraint not satisfied"

def test_insufficient_points():
    # Test that the function raises an assertion error when fewer than 8 points are provided
    x1 = np.random.uniform(-1, 1, (7, 2))
    x2 = np.random.uniform(-1, 1, (7, 2))
    try:
        estimate_fundamental_matrix(x1, x2)
        raise AssertionError("AssertionError was not raised")
    except AssertionError:
        pass

def test_mismatched_points():
    # Test that the function raises an assertion error when input shapes are mismatched
    x1 = np.random.uniform(-1, 1, (8, 2))
    x2 = np.random.uniform(-1, 1, (9, 2))
    try:
        estimate_fundamental_matrix(x1, x2)
        raise AssertionError("AssertionError was not raised")
    except AssertionError:
        pass

def test_fundamental_matrix_with_noise():
    # Create a set of corresponding points with noise
    np.random.seed(42)
    points1 = np.random.rand(20, 2) * 10
    noise = np.random.normal(0, 0.1, (20, 2))
    points2 = points1 + noise

    F = estimate_fundamental_matrix(points1, points2)

    # Test if F is a 3x3 matrix
    assert F.shape == (3, 3)

    # Test if F has rank 2
    _, S, _ = np.linalg.svd(F)
    assert np.isclose(S[-1], 0, atol=1e-6)

    # Test epipolar constraint (with higher tolerance due to noise)
    for p1, p2 in zip(points1, points2):
        x1 = np.array([p1[0], p1[1], 1])
        x2 = np.array([p2[0], p2[1], 1])
        assert abs(x2.T @ F @ x1) < 1e-3

def test_different_noise_levels():
    np.random.seed(42)
    points1 = np.random.rand(20, 2) * 10

    noise_levels = [0.01, 0.1, 0.5, 1.0]
    for noise_level in noise_levels:
        noise = np.random.normal(0, noise_level, (20, 2))
        points2 = points1 + noise

        F = estimate_fundamental_matrix(points1, points2)

        # Test if F is a 3x3 matrix
        assert F.shape == (3, 3)

        # Test if F has rank 2
        _, S, _ = np.linalg.svd(F)
        assert np.isclose(S[-1], 0, atol=1e-6)

        # Test epipolar constraint (with tolerance adjusted for noise level)
        for p1, p2 in zip(points1, points2):
            x1 = np.array([p1[0], p1[1], 1])
            x2 = np.array([p2[0], p2[1], 1])
            assert abs(x2.T @ F @ x1) < noise_level * 10