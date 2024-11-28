# visualize_tools.py

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def visualize_point_matches(img1, img2, pts1, pts2, output_dir='results'):
    """Visualize point matches between two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.
        pts1 (np.ndarray): Points from the first image (N x 2).
        pts2 (np.ndarray): Points from the second image (N x 2).
        output_dir (str): Directory to save the visualization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Stack images horizontally
    combined_img = np.hstack((img1, img2))

    # Debug information
    print(f"\nPoint matching visualization:")
    print(f"Image shapes - img1: {img1.shape}, img2: {img2.shape}")
    print(f"Points to draw - pts1: {len(pts1)}, pts2: {len(pts2)}")

    # Define colors and parameters
    color_pt1 = (0, 0, 255)    # Red
    color_pt2 = (255, 0, 0)    # Blue
    color_line = (0, 255, 0)   # Green
    point_size = 3
    line_thickness = 1

    valid_points = 0
    for pt1, pt2 in zip(pts1, pts2):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        if (0 <= x1 < img1.shape[1] and 0 <= y1 < img1.shape[0] and
            0 <= x2 < img2.shape[1] and 0 <= y2 < img2.shape[0]):
            
            pt1_int = (x1, y1)
            pt2_int = (x2 + img1.shape[1], y2)
            
            cv2.circle(combined_img, pt1_int, point_size, color_pt1, -1)
            cv2.circle(combined_img, pt2_int, point_size, color_pt2, -1)
            cv2.line(combined_img, pt1_int, pt2_int, color_line, line_thickness)
            
            valid_points += 1

    print(f"Valid points drawn: {valid_points} / {len(pts1)}")
    matches_path = os.path.join(output_dir, 'point_matches.png')
    cv2.imwrite(matches_path, combined_img)
    print(f"Point matches saved to '{matches_path}'")

def visualize_epipolar_lines(img1, img2, pts1, pts2, F, output_dir='results'):
    """Visualize epipolar lines and corresponding points on both images.

    Args:
        img1 (np.ndarray): First image (color).
        img2 (np.ndarray): Second image (color).
        pts1 (np.ndarray): Points from first image (N x 2).
        pts2 (np.ndarray): Points from second image (N x 2).
        F (np.ndarray): Fundamental matrix (3x3).
        output_dir (str): Directory to save the visualization.
    """
    def draw_lines(img, lines, pts, color=(0, 255, 0)):
        """Draw epipolar lines and points on the image.

        Args:
            img (np.ndarray): Image to draw on.
            lines (np.ndarray): Epipolar lines (N x 3).
            pts (np.ndarray): Points (N x 2).
            color (tuple): Color for lines and points.
        
        Returns:
            np.ndarray: Image with epipolar lines and points drawn.
        """
        r, c, _ = img.shape
        for r_line, pt in zip(lines, pts):
            a, b, c_line = r_line
            if b != 0:
                x0 = 0
                y0 = int(-c_line / b)
                x1 = img.shape[1]
                y1 = int(-(c_line + a * x1) / b)
            else:
                x0 = int(-c_line / a) if a != 0 else 0
                y0 = 0
                x1 = x0
                y1 = img.shape[0]
            img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
            img = cv2.circle(img, tuple(pt.astype(int)), 5, color, -1)
        return img

    os.makedirs(output_dir, exist_ok=True)

    # Compute epipolar lines for pts1 and pts2
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Draw lines on the images
    img1_with_lines = draw_lines(img1.copy(), lines1, pts1)
    img2_with_lines = draw_lines(img2.copy(), lines2, pts2)

    # Save the images with epipolar lines
    img1_path = os.path.join(output_dir, 'img1_with_epilines.png')
    img2_path = os.path.join(output_dir, 'img2_with_epilines.png')
    cv2.imwrite(img1_path, img1_with_lines)
    cv2.imwrite(img2_path, img2_with_lines)
    print(f"Epipolar lines drawn and saved to '{img1_path}' and '{img2_path}'")

def plot_epipolar_cost_change(epipolar_cost_before, epipolar_cost_after, output_dir='results'):
    """Plot the change in epipolar cost before and after optimization.

    Args:
        epipolar_cost_before (float): Epipolar cost before optimization.
        epipolar_cost_after (float): Epipolar cost after optimization.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(['Before Optimization', 'After Optimization'], [epipolar_cost_before, epipolar_cost_after], color=['red', 'green'])
    plt.ylabel('Average Epipolar Cost')
    plt.title('Epipolar Cost Before and After Optimization')
    plt.tight_layout()
    plt_path = os.path.join(output_dir, 'epipolar_cost_comparison.png')
    plt.savefig(plt_path)
    plt.close()
    print(f"Epipolar cost comparison plot saved to '{plt_path}'")

def save_warped_image(img1, img2, H_np, output_dir='results'):
    """Save warped image using homography.

    Args:
        img1 (np.ndarray): First image to be warped.
        img2 (np.ndarray): Second (target) image.
        H_np (np.ndarray): Homography matrix.
        output_dir (str): Path to output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Warp img1 using homography
    img1_warped = cv2.warpPerspective(img1, H_np, (img2.shape[1], img2.shape[0]))
    
    # Save warped image
    warped_path = os.path.join(output_dir, 'warped_image.png')
    cv2.imwrite(warped_path, img1_warped)
    print(f"Warped image saved to '{warped_path}'")
