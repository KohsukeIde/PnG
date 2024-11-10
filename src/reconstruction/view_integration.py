# src/reconstruction/view_integration.py
## NOT IN USE
import numpy as np
from typing import List, Tuple

from src.camera.camera_model import CameraModel
from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstruction.initial_reconstruction_naive import triangulate_points


def integrate_new_view(
    existing_3d_points: np.ndarray,
    new_gaussians_2d: TwoDGaussians,
    new_camera: CameraModel,
    existing_cameras: List[CameraModel],
    threshold: float = 1e-6,
    reproj_threshold: float = 2.0  # Reprojection error threshold in pixels
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Integrate a new view into existing 3D point cloud.

    Args:
        existing_3d_points: ndarray[M, 3] (existing 3D point cloud)
        new_gaussians_2d: TwoDGaussians (2D Gaussian mixture representation of new image)
        new_camera: CameraModel (camera model for new image)
        existing_cameras: List[CameraModel] (list of existing camera models)
        threshold: float, threshold for optimal transport matrix
        reproj_threshold: float, reprojection error threshold in pixels

    Returns:
        updated_3d_points: ndarray[N, 3] (updated 3D point cloud)
        new_matches: List[Tuple[int, int]] (list of new matching indices)
    """
    # Step 1: Project existing 3D points onto the new camera
    projected_points, valid_indices = project_points_to_camera(existing_3d_points, new_camera)
    if projected_points.size == 0:
        print("No valid projected points.")
        return existing_3d_points, []

    # Step 2: Match projected points with new 2D Gaussian distributions using optimal transport
    # Get the mean coordinates of Gaussians
    gaussians_means = new_gaussians_2d.means  # shape: (N_gaussians, 2)

    # Calculate cost matrix (Euclidean distance)
    cost_matrix = compute_cost_matrix(projected_points, gaussians_means)

    # Solve optimal transport
    solver = OptimalTransportSolver(projected_points, gaussians_means)
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)

    # Extract matches
    matches = extract_matches_from_transport(transport_matrix, threshold)

    if len(matches) == 0:
        print("No matches found after applying the threshold.")
        return existing_3d_points, []

    # Step 3: Filter matches based on reprojection error
    inlier_matches = filter_matches_by_reprojection_error(
        existing_3d_points, valid_indices, gaussians_means, matches, new_camera, reproj_threshold
    )

    if len(inlier_matches) == 0:
        print("No inlier matches found after reprojection error filtering.")
        return existing_3d_points, []

    # Step 4: Reconstruct new points through triangulation
    new_points_3d = reconstruct_new_points(
        new_gaussians_2d, inlier_matches, existing_cameras, new_camera
    )

    # Step 5: Combine existing 3D points with newly reconstructed points
    updated_3d_points = np.vstack([existing_3d_points, new_points_3d])

    return updated_3d_points, inlier_matches


def project_points_to_camera(points_3d: np.ndarray, camera: CameraModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points onto camera and get image coordinates.

    Args:
        points_3d: ndarray[M, 3] (3D points)
        camera: CameraModel

    Returns:
        projected_points: ndarray[K, 2] (projected 2D points)
        valid_indices: ndarray[K,] (original indices of valid projections)
    """
    # Homogeneous coordinates
    points_3d_hom = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])  # (M, 4)

    # Project points
    projected_points_hom = (camera.P @ points_3d_hom.T).T  # (M, 3)

    # Normalize to get image coordinates
    projected_points = projected_points_hom[:, :2] / projected_points_hom[:, 2][:, np.newaxis]

    # Get indices of valid projections (points within image bounds)
    img_width, img_height = camera.width, camera.height
    valid_mask = (
        (projected_points[:, 0] >= 0) & (projected_points[:, 0] < img_width) &
        (projected_points[:, 1] >= 0) & (projected_points[:, 1] < img_height) &
        (projected_points_hom[:, 2] > 0)  # Only points in front of camera
    )
    valid_indices = np.where(valid_mask)[0]
    projected_points = projected_points[valid_mask]

    return projected_points, valid_indices


def compute_cost_matrix(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculate cost matrix between two point sets (squared Euclidean distance).

    Args:
        points_a: ndarray[N, 2]
        points_b: ndarray[M, 2]

    Returns:
        cost_matrix: ndarray[N, M]
    """
    diff = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]  # (N, M, 2)
    cost_matrix = np.sum(diff ** 2, axis=2)  # (N, M)
    return cost_matrix


def extract_matches_from_transport(transport_matrix: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """
    Extract matches from optimal transport matrix.

    Args:
        transport_matrix: ndarray[N, M]
        threshold: float

    Returns:
        matches: List[Tuple[int, int]]
    """
    matches = []
    N, M = transport_matrix.shape
    for i in range(N):
        for j in range(M):
            if transport_matrix[i, j] > threshold:
                matches.append((i, j))
    return matches


def filter_matches_by_reprojection_error(
    points_3d: np.ndarray,
    valid_indices: np.ndarray,
    gaussians_means: np.ndarray,
    matches: List[Tuple[int, int]],
    camera: CameraModel,
    reproj_threshold: float
) -> List[Tuple[int, int]]:
    """
    Filter matches based on reprojection error.

    Args:
        points_3d: ndarray[M, 3]
        valid_indices: ndarray[K,]
        gaussians_means: ndarray[N, 2]
        matches: List[Tuple[int, int]]
        camera: CameraModel
        reproj_threshold: float

    Returns:
        inlier_matches: List[Tuple[int, int]]
    """
    inlier_matches = []
    for idx_proj, idx_gauss in matches:
        idx_3d = valid_indices[idx_proj]
        point_3d = points_3d[idx_3d]

        # Reproject 3D point
        point_3d_hom = np.hstack([point_3d, 1.0])  # (4,)
        projected_point_hom = camera.P @ point_3d_hom  # (3,)
        projected_point = projected_point_hom[:2] / projected_point_hom[2]

        # Compute reprojection error
        gaussian_point = gaussians_means[idx_gauss]
        error = np.linalg.norm(projected_point - gaussian_point)

        if error < reproj_threshold:
            inlier_matches.append((idx_3d, idx_gauss))

    return inlier_matches


def reconstruct_new_points(
    new_gaussians_2d: TwoDGaussians,
    inlier_matches: List[Tuple[int, int]],
    existing_cameras: List[CameraModel],
    new_camera: CameraModel
) -> np.ndarray:
    """
    Reconstruct new feature points observed from new viewpoint but not in existing 3D point cloud using triangulation.

    Args:
        new_gaussians_2d: TwoDGaussians
        inlier_matches: List[Tuple[int, int]]
        existing_cameras: List[CameraModel]
        new_camera: CameraModel

    Returns:
        new_points_3d: ndarray[L, 3]
    """
    # Feature point indices from new image
    matched_indices = [idx_gauss for _, idx_gauss in inlier_matches]

    # All feature point indices
    all_indices = set(range(len(new_gaussians_2d.means)))

    # Unmatched feature point indices
    unmatched_indices = list(all_indices - set(matched_indices))

    new_points_3d = []

    # For each unmatched feature point, find correspondences with other views (simplified here)
    # In practice, we need to find correspondences with other images for unmatched features
    # Currently returning empty array since reconstruction isn't possible from a single camera position

    return np.array(new_points_3d)

