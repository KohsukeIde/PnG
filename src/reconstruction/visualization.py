# visualization.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple
from src.camera.camera_model import CameraModel
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstruction.view_integration import project_points_to_camera

def visualize_reconstruction(points_3d, img1, img2, pts1_inliers, pts2_inliers):
    """Visualize reconstruction results.

    Args:
        points_3d: np.ndarray[N, 3], reconstructed 3D points
        img1: np.ndarray, first image
        img2: np.ndarray, second image
        pts1_inliers: np.ndarray[N, 2], inlier corresponding points in image 1
        pts2_inliers: np.ndarray[N, 2], inlier corresponding points in image 2
    """
    # Create output directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Check if inlier points exist
    if pts1_inliers.size == 0 or pts2_inliers.size == 0:
        print("No inlier points to visualize.")
        return

    plt.figure(figsize=(15, 5))

    # Plot inlier points on image 1
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.scatter(pts1_inliers[:, 0], pts1_inliers[:, 1], c='r', marker='o')
    plt.title('Image 1 with Inlier Points')

    # Plot inlier points on image 2
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(pts2_inliers[:, 0], pts2_inliers[:, 1], c='r', marker='o')
    plt.title('Image 2 with Inlier Points')

    plt.savefig(os.path.join(output_dir, 'inlier_points.png'))
    plt.close()

    # Visualize 3D point cloud
    if points_3d.size == 0:
        print("No 3D points to visualize.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Reconstructed 3D Points')
    plt.savefig(os.path.join(output_dir, '3d_points.png'))
    plt.close()

def save_points_to_ply(points_3d, filename='reconstructed_points.ply', colors=None):
    """Save 3D point cloud to PLY file.

    Args:
        points_3d: np.ndarray[N, 3], 3D point cloud
        filename: str, output PLY filename
        colors: np.ndarray[N, 3], RGB values for each point (range 0-255)
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
                # Convert color data to 0-255 range
                color = np.clip(color, 0, 255).astype(int)
                f.write(f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
        else:
            for point in points_3d:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def visualize_new_view_integration(
    new_image_path: str,
    existing_3d_points: np.ndarray,
    updated_3d_points: np.ndarray,
    inlier_matches: List[Tuple[int, int]],
    new_gaussians_2d: TwoDGaussians,
    new_camera: CameraModel
):
    """
    Visualize new view integration results.

    Args:
        new_image_path: str
        existing_3d_points: ndarray[M, 3]
        updated_3d_points: ndarray[N, 3]
        inlier_matches: List[Tuple[int, int]]
        new_gaussians_2d: TwoDGaussians
        new_camera: CameraModel
    """
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Load new image
    img_new = cv2.imread(new_image_path)

    # Project existing 3D points onto new camera
    projected_points, _ = project_points_to_camera(existing_3d_points, new_camera)

    # Plot projected points and new Gaussian centers
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB))
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c='b', marker='o', label='Projected 3D Points')
    plt.scatter(new_gaussians_2d.means[:, 0], new_gaussians_2d.means[:, 1], c='r', marker='x', label='New Gaussians')

    # Visualize matches
    for idx_3d, idx_gauss in inlier_matches:
        point_3d = existing_3d_points[idx_3d]
        point_3d_hom = np.hstack([point_3d, 1.0])
        projected_point_hom = new_camera.P @ point_3d_hom
        projected_point = projected_point_hom[:2] / projected_point_hom[2]

        gaussian_point = new_gaussians_2d.means[idx_gauss]

        plt.plot(
            [projected_point[0], gaussian_point[0]],
            [projected_point[1], gaussian_point[1]],
            'g-'
        )

    plt.legend()
    plt.title('New Image with Projected 3D Points and Gaussians')
    plt.savefig(os.path.join(output_dir, 'new_view_integration.png'))
    plt.close()

    # Visualize updated 3D point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot existing 3D points
    ax.scatter(existing_3d_points[:, 0], existing_3d_points[:, 1], existing_3d_points[:, 2], 
              c='b', marker='o', label='Existing 3D Points')

    # Plot newly added 3D points
    new_points_3d = updated_3d_points[len(existing_3d_points):]
    if new_points_3d.size > 0:
        ax.scatter(new_points_3d[:, 0], new_points_3d[:, 1], new_points_3d[:, 2], 
                  c='r', marker='^', label='New 3D Points')

    ax.legend()
    plt.title('Updated 3D Point Cloud')
    plt.savefig(os.path.join(output_dir, 'updated_3d_points.png'))
    plt.close()
