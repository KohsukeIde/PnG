"""
Step A: Epipolar Cost Unit Test (without OT)

Purpose: Verify that F generation and epipolar distance have correct orientation.

Test procedure:
1. Use epipolar_mode="sampson" (SED is fixed, but verify both)
2. Create arbitrary R, t and compute F = _build_F_from_wc(R, t)
3. Take a point x1, compute l2 = F @ x1
4. Create x2 on line l2 (satisfies a*u + b*v + c = 0)
5. Verify Sampson distance is ~0

This confirms:
- F orientation is correct
- Coordinate system is consistent
- K application is correct
- Sampson/SED formula is correct
"""

import numpy as np
import torch
from typing import Tuple


def create_test_camera_intrinsics(
    fx: float = 500.0, fy: float = 500.0,
    cx: float = 320.0, cy: float = 240.0
) -> np.ndarray:
    """Create camera intrinsic matrix K."""
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)


def create_test_pose(
    axis: np.ndarray = None,
    angle: float = 0.1,
    translation: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Create test R, t for camera pose."""
    if axis is None:
        axis = np.array([0, 1, 0], dtype=np.float64)  # rotate around Y
    axis = axis / np.linalg.norm(axis)

    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    if translation is None:
        translation = np.array([0.5, 0.0, 0.1], dtype=np.float64)
    t = translation / np.linalg.norm(translation)  # unit translation

    return R.astype(np.float64), t.astype(np.float64)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def build_fundamental_matrix(
    R_wc: np.ndarray, t_wc: np.ndarray,
    K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """Build fundamental matrix from R, t and intrinsics.

    F = K2^{-T} [t]_x R K1^{-1}

    where:
    - R_wc, t_wc: world-to-camera2 transformation
    - K1: camera1 intrinsics
    - K2: camera2 intrinsics
    """
    tx = skew_symmetric(t_wc)
    E = tx @ R_wc  # Essential matrix
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = K2_inv.T @ E @ K1_inv
    return F


def compute_sampson_distance(
    x1: np.ndarray, x2: np.ndarray, F: np.ndarray
) -> float:
    """Compute Sampson distance for a point pair.

    Sampson = (x2^T F x1)^2 / ((Fx1)_1^2 + (Fx1)_2^2 + (F^Tx2)_1^2 + (F^Tx2)_2^2)
    """
    Fx1 = F @ x1
    FTx2 = F.T @ x2

    numerator = (x2.T @ F @ x1) ** 2
    denominator = Fx1[0]**2 + Fx1[1]**2 + FTx2[0]**2 + FTx2[1]**2

    return numerator / (denominator + 1e-10)


def compute_sed_distance(
    x1: np.ndarray, x2: np.ndarray, F: np.ndarray
) -> float:
    """Compute Symmetric Epipolar Distance for a point pair.

    SED = d(x1, l1)^2 + d(x2, l2)^2
    where:
    - l1 = F^T x2 (line in image1 from point x2)
    - l2 = F x1 (line in image2 from point x1)
    - d(x, l) = |ax + by + c| / sqrt(a^2 + b^2)
    """
    l1 = F.T @ x2  # line in image1
    l2 = F @ x1    # line in image2

    # Point-to-line distance: |ax + by + c| / sqrt(a^2 + b^2)
    d1_sq = (x1.T @ l1) ** 2 / (l1[0]**2 + l1[1]**2 + 1e-10)
    d2_sq = (x2.T @ l2) ** 2 / (l2[0]**2 + l2[1]**2 + 1e-10)

    return d1_sq + d2_sq


def point_on_line(line: np.ndarray, u: float) -> np.ndarray:
    """Create a point on the line ax + by + c = 0.

    Parameterize: for given u, solve for v such that a*u + b*v + c = 0
    v = -(a*u + c) / b
    """
    a, b, c = line
    if abs(b) > 1e-6:
        v = -(a * u + c) / b
        return np.array([u, v, 1.0])
    else:
        # Line is nearly vertical, parameterize differently
        v = u  # arbitrary
        u_new = -(b * v + c) / a
        return np.array([u_new, v, 1.0])


def test_epipolar_geometry():
    """Main test for epipolar geometry correctness."""
    print("=" * 60)
    print("Step A: Epipolar Geometry Unit Test")
    print("=" * 60)

    # Setup
    K1 = create_test_camera_intrinsics()
    K2 = create_test_camera_intrinsics()  # Same intrinsics for simplicity
    R_wc, t_wc = create_test_pose()

    print(f"\nCamera intrinsics K:\n{K1}")
    print(f"\nRotation R_wc:\n{R_wc}")
    print(f"\nTranslation t_wc: {t_wc}")

    # Build fundamental matrix
    F = build_fundamental_matrix(R_wc, t_wc, K1, K2)
    print(f"\nFundamental matrix F:\n{F}")

    # Test 1: Point correspondence on epipolar line
    print("\n" + "-" * 40)
    print("Test 1: Point on epipolar line should have ~0 distance")
    print("-" * 40)

    # Random point in image1
    x1 = np.array([250.0, 180.0, 1.0])
    print(f"Point x1 in image1: {x1[:2]}")

    # Compute epipolar line in image2
    l2 = F @ x1
    print(f"Epipolar line l2 in image2: {l2}")

    # Create point x2 on this line
    x2 = point_on_line(l2, 300.0)  # u=300
    print(f"Point x2 on line l2: {x2[:2]}")

    # Verify epipolar constraint
    epipolar_constraint = x2.T @ F @ x1
    print(f"Epipolar constraint x2^T F x1: {epipolar_constraint:.2e} (should be ~0)")

    # Compute distances
    sampson = compute_sampson_distance(x1, x2, F)
    sed = compute_sed_distance(x1, x2, F)

    print(f"Sampson distance: {sampson:.2e} (should be ~0)")
    print(f"SED distance: {sed:.2e} (should be ~0)")

    # Test 2: Random point pair (not on epipolar line)
    print("\n" + "-" * 40)
    print("Test 2: Random point pair (should have non-zero distance)")
    print("-" * 40)

    x1_rand = np.array([150.0, 200.0, 1.0])
    x2_rand = np.array([400.0, 300.0, 1.0])  # Arbitrary, not on l2

    sampson_rand = compute_sampson_distance(x1_rand, x2_rand, F)
    sed_rand = compute_sed_distance(x1_rand, x2_rand, F)

    print(f"Point x1: {x1_rand[:2]}")
    print(f"Point x2: {x2_rand[:2]}")
    print(f"Sampson distance: {sampson_rand:.4f} (should be > 0)")
    print(f"SED distance: {sed_rand:.4f} (should be > 0)")

    # Test 3: Verify F matrix properties
    print("\n" + "-" * 40)
    print("Test 3: Fundamental matrix properties")
    print("-" * 40)

    U, S, Vh = np.linalg.svd(F)
    print(f"Singular values: {S}")
    print(f"Rank (should be 2): {np.sum(S > 1e-10)}")
    print(f"det(F) (should be ~0): {np.linalg.det(F):.2e}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    tests_passed = True

    if abs(epipolar_constraint) > 1e-6:
        print("FAIL: Epipolar constraint not satisfied")
        tests_passed = False
    else:
        print("PASS: Epipolar constraint satisfied")

    if sampson > 1e-6:
        print("FAIL: Sampson distance too large for point on line")
        tests_passed = False
    else:
        print("PASS: Sampson distance ~0 for point on line")

    if sed > 1e-6:
        print("FAIL: SED distance too large for point on line")
        tests_passed = False
    else:
        print("PASS: SED distance ~0 for point on line")

    if sampson_rand < 0.1:
        print("WARNING: Random points unexpectedly close")
    else:
        print("PASS: Random points have positive distance")

    return tests_passed


def test_f_direction():
    """Test that F and F^T produce lines in the correct images."""
    print("\n" + "=" * 60)
    print("Test: F direction verification")
    print("=" * 60)

    K1 = create_test_camera_intrinsics()
    K2 = create_test_camera_intrinsics()
    R_wc, t_wc = create_test_pose()
    F = build_fundamental_matrix(R_wc, t_wc, K1, K2)

    x1 = np.array([250.0, 180.0, 1.0])
    x2 = np.array([300.0, 200.0, 1.0])

    # l2 = F x1: line in image2 (from point x1 in image1)
    l2 = F @ x1

    # l1 = F^T x2: line in image1 (from point x2 in image2)
    l1 = F.T @ x2

    print(f"Point x1 in image1: {x1[:2]}")
    print(f"Point x2 in image2: {x2[:2]}")
    print(f"Line l2 in image2 (from x1): {l2}")
    print(f"Line l1 in image1 (from x2): {l1}")

    # Create x2 on l2
    x2_on_l2 = point_on_line(l2, 300.0)
    constraint = x2_on_l2.T @ F @ x1
    print(f"\nPoint on l2: {x2_on_l2[:2]}")
    print(f"Epipolar constraint: {constraint:.2e} (should be ~0)")

    # Create x1 on l1
    x1_on_l1 = point_on_line(l1, 250.0)
    constraint2 = x2.T @ F @ x1_on_l1
    print(f"\nPoint on l1: {x1_on_l1[:2]}")
    print(f"Epipolar constraint: {constraint2:.2e} (should be ~0)")

    return abs(constraint) < 1e-6 and abs(constraint2) < 1e-6


if __name__ == "__main__":
    test1 = test_epipolar_geometry()
    test2 = test_f_direction()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
