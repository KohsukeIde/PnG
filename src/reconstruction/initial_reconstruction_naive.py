# src/reconstruction/initial_reconstruction_naive.py

import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.camera.camera_model import CameraModel
from src.primitive.twod_gaussians_rs import TwoDGaussians


def perform_initial_reconstruction(
    gaussians1: TwoDGaussians,
    gaussians2: TwoDGaussians,
    camera1: CameraModel,
    camera2: CameraModel,
    transport_matrix: np.ndarray,
    threshold: float = 1e-6,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """Filter matches using epipolar constraints and reconstruct 3D point cloud using triangulation.

    Args:
        gaussians1: TwoDGaussians (Gaussian distributions for image 1, in image coordinates)
        gaussians2: TwoDGaussians (Gaussian distributions for image 2, in image coordinates)
        camera1: CameraModel (Camera model for image 1)
        camera2: CameraModel (Camera model for image 2)
        transport_matrix: np.ndarray, transport matrix
        threshold: float, threshold for the transport matrix

    Returns:
        points_3d: np.ndarray (N, 3) Reconstructed 3D point cloud
        inlier_matches: List[Tuple[int, int]] Index pairs of matches satisfying the epipolar constraint
        pts1_inliers: np.ndarray (N, 2) Inlier feature points from image 1
        pts2_inliers: np.ndarray (N, 2) Inlier feature points from image 2
    """
    # Step 1: Extract matches from the transport matrix
    matches = extract_matches(transport_matrix, threshold)

    if len(matches) == 0:
        print("No matches found after applying the threshold.")
        return np.array([]), [], np.array([]), np.array([])

    # Step 2: Use the centers of Gaussian distributions as feature points
    pts1 = gaussians1.means  # Feature points from image 1 (N, 2)
    pts2 = gaussians2.means  # Feature points from image 2 (N, 2)

    print("Feature Points Image 1 (first 5):\n", pts1[:5])
    print("Feature Points Image 2 (first 5):\n", pts2[:5])

    # Step 3: Get matching pairs
    idx1 = [i for i, _ in matches]
    idx2 = [j for _, j in matches]
    pts1_matched = pts1[idx1]
    pts2_matched = pts2[idx2]

    # Step 4: Compute the fundamental matrix from camera parameters
    F = compute_fundamental_matrix(camera1, camera2)
    print(f"{F=}")

    # Step 5: Apply epipolar constraint to extract inliers
    inlier_mask = apply_epipolar_constraint(pts1_matched, pts2_matched, F)

    inlier_matches = [match for match, inlier in zip(matches, inlier_mask) if inlier]
    pts1_inliers = pts1_matched[inlier_mask]
    pts2_inliers = pts2_matched[inlier_mask]

    print("Inlier Points Image 1 (first 5):\n", pts1_inliers[:5])
    print("Inlier Points Image 2 (first 5):\n", pts2_inliers[:5])

    if len(inlier_matches) == 0:
        print("No inlier matches found after applying the epipolar constraint.")
        return np.array([]), [], np.array([]), np.array([])

    # Step 6: Reconstruct 3D points using triangulation
    points_3d = triangulate_points(pts1_inliers, pts2_inliers, camera1, camera2)

    return points_3d, inlier_matches, pts1_inliers, pts2_inliers

def extract_matches(transport_matrix: np.ndarray, threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """Extract matches from the transport matrix.

    Args:
        transport_matrix: np.ndarray, transport matrix
        threshold: float, threshold for transport amount

    Returns:
        matches: List[Tuple[int, int]], index pairs of matches
    """
    cost_for_hungarian = -transport_matrix
    row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)
    matches = []
    for i, j in zip(row_ind, col_ind):
        if transport_matrix[i, j] > threshold:
            matches.append((i, j))
    return matches

def compute_fundamental_matrix(camera1: CameraModel, camera2: CameraModel) -> np.ndarray:
    """Compute the fundamental matrix from camera intrinsic and extrinsic parameters.

    Args:
        camera1: CameraModel, model for camera 1
        camera2: CameraModel, model for camera 2

    Returns:
        F: np.ndarray, fundamental matrix (3x3)
    """
    # Get camera parameters for camera 1 and camera 2
    R1_wc, t1_wc = camera1.R_wc, camera1.t_wc
    R2_wc, t2_wc = camera2.R_wc, camera2.t_wc

    # Compute relative rotation and translation
    R_rel = R2_wc @ R1_wc.T
    t_rel = t2_wc - R_rel @ t1_wc

    # Compute essential matrix
    t_x = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0]
    ])
    E = t_x @ R_rel

    # Compute fundamental matrix
    K1_inv = np.linalg.inv(camera1.K)
    K2_inv = np.linalg.inv(camera2.K)
    F = K2_inv.T @ E @ K1_inv

    # Normalize
    F /= np.linalg.norm(F)

    return F

def apply_epipolar_constraint(pts1: np.ndarray, pts2: np.ndarray, F: np.ndarray, threshold: float = 1e-1) -> np.ndarray:
    """Apply epipolar constraint to extract inliers.

    Args:
        pts1: np.ndarray, feature points from image 1 (N, 2)
        pts2: np.ndarray, feature points from image 2 (N, 2)
        F: np.ndarray, fundamental matrix (3x3)
        threshold: float, threshold for epipolar constraint

    Returns:
        inlier_mask: np.ndarray, boolean array indicating inliers (N,)
    """
    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N, 3)
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N, 3)

    # Compute epipolar constraint errors
    errors = np.abs(np.sum(pts2_hom * (F @ pts1_hom.T).T, axis=1))

    # Keep points with errors below the threshold
    inlier_mask = errors < threshold
    return inlier_mask

def triangulate_points(pts1: np.ndarray, pts2: np.ndarray, camera1: CameraModel, camera2: CameraModel) -> np.ndarray:
    """Reconstruct 3D points using triangulation from corresponding feature points.

    Args:
        pts1: np.ndarray, feature points from image 1 (N, 2)
        pts2: np.ndarray, feature points from image 2 (N, 2)
        camera1: CameraModel, model for camera 1
        camera2: CameraModel, model for camera 2

    Returns:
        points_3d: np.ndarray, reconstructed 3D point cloud (N, 3)
    """
    # Compute camera matrices
    P1 = camera1.P  # (3, 4)
    P2 = camera2.P  # (3, 4)

    # Triangulation
    pts1_hom = pts1.T  # (2, N)
    pts2_hom = pts2.T  # (2, N)
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_hom, pts2_hom)
    points_3d = (points_4d_hom[:3, :] / points_4d_hom[3, :]).T  # (N, 3)

    return points_3d

def visualize_reconstruction(points_3d, img1, img2, pts1_inliers, pts2_inliers):
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    if pts1_inliers.size == 0 or pts2_inliers.size == 0:
        print("No inlier points to visualize.")
        return

    plt.figure(figsize=(15, 5))

    # inlier points on image 1
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.scatter(pts1_inliers[:, 0], pts1_inliers[:, 1], c='r', marker='o')
    plt.title('Image 1 with Inlier Points')

    # inlier points on image 2
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2_inliers[:, 0], pts2_inliers[:, 1], c='r', marker='o')
    plt.title('Image 2 with Inlier Points')

    plt.savefig(os.path.join(output_dir, 'inlier_points.png'))
    plt.close()

    # Visualize 3D points
    if points_3d.size == 0:
        print("No 3D points to visualize.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

    plt.title('Reconstructed 3D Points')
    plt.savefig(os.path.join(output_dir, '3d_points.png'))
    plt.close()

def save_points_to_ply(points_3d, filename='reconstructed_points.ply', colors=None):
    """Save 3D point cloud to PLY file.

    Args:
        points_3d: np.ndarray of shape (N, 3)
        filename: str, name of the output PLY file
        colors: np.ndarray of shape (N, 3), optional RGB values for points (0-1 range)
    """
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)

    num_points = points_3d.shape[0]
    header = f'''ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
'''
    if colors is not None:
        header += '''property uchar red
property uchar green
property uchar blue
'''

    header += 'end_header\n'

    with open(file_path, 'w') as f:
        f.write(header)
        if colors is not None:
            for point, color in zip(points_3d, colors):
                # Convert color values to 0-255 range if they are in the 0-1 range
                if color.max() <= 1.0:
                    color = (color * 255).astype(int)
                f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        else:
            for point in points_3d:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
