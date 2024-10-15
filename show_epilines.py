import pickle
import sys
from src.evaluator.matching_evaluator import MatchingEvaluator
from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
import time
from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np
import matplotlib.pyplot as plt


sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

def load_gaussians(file_path: str) -> tuple:
    """
    Load Gaussian data, view matrix, and camera intrinsic matrix from a pickle file.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        tuple: (gaussians, viewmat, K)
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        gaussians = data["gaussians"]
        viewmat = data["viewmat"]
        K = data["K"]
    return gaussians, viewmat, K

def load_images(img1_path: str, img2_path: str) -> tuple:
    """
    Load two images from the specified paths.

    Args:
        img1_path (str): Path to image 1
        img2_path (str): Path to image 2

    Returns:
        tuple: (img1, img2)
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise FileNotFoundError(f"Image at path '{img1_path}' not found.")
    if img2 is None:
        raise FileNotFoundError(f"Image at path '{img2_path}' not found.")
    
    return img1, img2

def extract_feature_points(gaussians, K, viewmat):
    """
    Extract Gaussian centers as feature points in pixel coordinates.

    Args:
        gaussians: TwoDGaussians object
        K (np.ndarray): Camera intrinsic matrix (3x3)
        viewmat (np.ndarray): Camera view matrix (4x4)

    Returns:
        np.ndarray: Array of feature points (N, 2)
    """
    try:
        centers = gaussians.means  # (N, 2)
    except AttributeError:
        raise AttributeError("Gaussians object does not have 'means' attribute.")

    num_points = centers.shape[0]

    # Add Z coordinate (Z=0) and convert to homogeneous coordinates
    centers_homog = np.hstack([centers, np.zeros((num_points, 1)), np.ones((num_points, 1))])  # (N, 4)

    # Apply viewmat to convert to camera coordinates
    camera_coords_homog = (viewmat @ centers_homog.T).T  # (N, 4)
    camera_coords = camera_coords_homog[:, :3]  # (N, 3)

    # Apply camera intrinsic matrix K to convert to pixel coordinates
    pts_pixel_homog = (K @ camera_coords.T).T  # (N, 3)
    pts_pixel = pts_pixel_homog[:, :2] / pts_pixel_homog[:, 2, np.newaxis]  # (N, 2)

    return pts_pixel

def draw_epilines(img, lines, pts, colors, point_radius=5):
    """
    Draw epipolar lines and corresponding points on the image.

    Args:
        img (np.ndarray): Target image for drawing
        lines (np.ndarray): Epipolar lines (N, 3)
        pts (np.ndarray): Corresponding feature points (N, 2)
        colors (list): Colors for each feature point/epipolar line (N, 3)
        point_radius (int): Radius of feature points

    Returns:
        np.ndarray: Image with drawn epipolar lines and points
    """
    img_copy = img.copy()
    r, c, _ = img.shape
    for r_line, pt, color in zip(lines, pts, colors):
        a, b, c_line = r_line
        if b != 0:
            y0 = int(-c_line / b)
            y1 = int(-(c_line + a * c) / b)
            x0, x1 = 0, c
        else:
            x0 = int(-c_line / a)
            x1 = x0
            y0, y1 = 0, r

        # Check if coordinates are within image bounds and clip if necessary
        x0 = max(0, min(c, x0))
        x1 = max(0, min(c, x1))
        y0 = max(0, min(r, y0))
        y1 = max(0, min(r, y1))

        img_copy = cv2.line(img_copy, (int(x0), int(y0)), (int(x1), int(y1)), color.tolist(), 1)

        # Draw feature points
        pt_int = pt.astype(int)
        if 0 <= pt_int[0] < c and 0 <= pt_int[1] < r:
            img_copy = cv2.circle(img_copy, (pt_int[0], pt_int[1]), point_radius, color.tolist(), -1)
        else:
            print(f"Warning: Feature point {pt} is out of image bounds and will not be drawn.")

    return img_copy

def visualize_epilines_on_images(img1, img2, pts1_inliers, pts2_inliers, F, output_path='epilines_visualization.png'):
    """
    Draw epipolar lines and corresponding points on images and visualize them.

    Args:
        img1 (np.ndarray): Image 1
        img2 (np.ndarray): Image 2
        pts1_inliers (np.ndarray): Inlier feature points in image 1 (M, 2)
        pts2_inliers (np.ndarray): Inlier feature points in image 2 (M, 2)
        F (np.ndarray): Fundamental matrix
        output_path (str): Path to save the visualization result
    """
    if len(pts1_inliers) == 0:
        print("No inliers found. Cannot visualize epilines.")
        return

    # Calculate epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2_inliers.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    lines2 = cv2.computeCorrespondEpilines(pts1_inliers.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    # Generate unique colors for each feature point
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, (len(pts1_inliers), 3))

    img1_with_lines = draw_epilines(img1, lines1, pts1_inliers, colors)
    img2_with_lines = draw_epilines(img2, lines2, pts2_inliers, colors)

    return img1_with_lines, img2_with_lines

def compute_epipolar_errors(pts1, pts2, F):
    """
    Compute epipolar constraint errors.

    Args:
        pts1 (np.ndarray): Feature points in image 1 (N, 2)
        pts2 (np.ndarray): Feature points in image 2 (N, 2)
        F (np.ndarray): Fundamental matrix

    Returns:
        np.ndarray: Array of epipolar constraint errors (N,)
    """
    pts1_hom = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N, 3)
    pts2_hom = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N, 3)
    errors = np.abs(np.sum(pts2_hom * (F @ pts1_hom.T).T, axis=1))
    return errors

def sift_feature_matching(img1, img2):
    """
    Detect and match SIFT features between two images.

    Args:
        img1 (np.ndarray): Image 1
        img2 (np.ndarray): Image 2

    Returns:
        tuple: (pts1, pts2)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Handle case when not enough descriptors are found
    if descriptors1 is None or descriptors2 is None:
        print("Not enough descriptors found for SIFT matching.")
        return None, None

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Use only top matches
    num_matches = 100
    matches = matches[:num_matches]

    # Get matched keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    return pts1, pts2

def main():
    print("Loading Gaussians...")
    gaussians1, viewmat1, K1 = load_gaussians('/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/fitted_gaussians_22_1k.pkl')
    gaussians2, viewmat2, K2 = load_gaussians('/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/fitted_gaussians_23_1k.pkl')
    print("Gaussians loaded.")
    print(f"Number of Gaussians in gaussians1: {gaussians1.k}")
    print(f"Number of Gaussians in gaussians2: {gaussians2.k}")

    viewmat1 = viewmat1.astype(np.float32)
    viewmat2 = viewmat2.astype(np.float32)
    K1 = K1.astype(np.float32)
    K2 = K2.astype(np.float32)

    print("Initializing OptimalTransportSolver...")
    from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
    solver = OptimalTransportSolver(gaussians1, gaussians2)

    print("Computing cost matrix...")
    start_time = time.time()
    cost_matrix = solver.compute_cost_matrix()
    end_time = time.time()
    print(f"Cost matrix computed in {end_time - start_time:.2f} seconds.")

    print("Computing transport matrix using Sinkhorn algorithm...")
    start_time = time.time()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    end_time = time.time()
    print(f"Transport matrix computed in {end_time - start_time:.2f} seconds.")

    print("Initializing MatchingEvaluator...")
    from src.evaluator.matching_evaluator import MatchingEvaluator
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    print("Evaluating matches...")
    metrics = evaluator.evaluate_matches()
    print("Matching Metrics:", metrics)

    print("Visualizing matches...")
    evaluator.visualize_matches('/Users/kohsukeide/dev/perspective-n-gaussian/outputs/matching_visualization_epilines.png')
    print("Matching visualization saved to '/Users/kohsukeide/dev/perspective-n-gaussian/outputs/matching_visualization_epilines.png'.")

    print("\n=== Epipolar Lines Visualization ===")
    print("Loading images for epipolar lines visualization...")
    img1_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0022.png' 
    img2_path = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/images/0023.png'  
    img1, img2 = load_images(img1_path, img2_path)
    print("Images loaded.")

    ### using transport matrix ###
    print("\n--- Using Custom Features ---")
    print("Extracting feature points from Gaussians...")
    pts1 = extract_feature_points(gaussians1, K1, viewmat1)
    pts2 = extract_feature_points(gaussians2, K2, viewmat2)
    print(f"Extracted {pts1.shape[0]} feature points from image1.")
    print(f"Extracted {pts2.shape[0]} feature points from image2.")

    img1_height, img1_width = img1.shape[:2]
    img2_height, img2_width = img2.shape[:2]
    print(f"Image1 dimensions: width={img1_width}, height={img1_height}")
    print(f"Image2 dimensions: width={img2_width}, height={img2_height}")

    # using Hungarian algo for matching
    print("Matching features using transport matrix...")
    cost_for_hungarian = -transport_matrix
    row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)
    matches = []
    threshold = 1e-6
    for i, j in zip(row_ind, col_ind):
        if transport_matrix[i, j] > threshold:
            matches.append((i, j))
    print(f"Number of unique matched points: {len(matches)}")

    matches = np.array(matches)

    if len(matches) > 0:
        transport_values = transport_matrix[matches[:, 0], matches[:, 1]]
        sorted_indices = np.argsort(transport_values)[::-1]
        sorted_matches = matches[sorted_indices]
    else:
        sorted_matches = matches

    max_lines = 50
    limited_matches = sorted_matches[:max_lines] if len(sorted_matches) >= max_lines else sorted_matches
    print(f"Number of matched points to visualize: {len(limited_matches)}")

    # extract corersponding points
    pts1_matched = pts1[limited_matches[:, 0]]
    pts2_matched = pts2[limited_matches[:, 1]]

    print("Estimating Fundamental Matrix using OpenCV...")
    F_custom, mask_custom = cv2.findFundamentalMat(pts1_matched, pts2_matched, cv2.FM_RANSAC, ransacReprojThreshold=3.0)
    if F_custom is None or F_custom.shape != (3, 3):
        print("Fundamental matrix could not be estimated.")
        return
    print("Fundamental Matrix estimated.")

    num_inliers_custom = np.sum(mask_custom)
    num_outliers_custom = len(mask_custom) - num_inliers_custom
    print(f"Number of inliers after RANSAC: {num_inliers_custom}")
    print(f"Number of outliers after RANSAC: {num_outliers_custom}")

    if num_inliers_custom < 2:
        print("Not enough inliers to visualize epilines.")
    else:
        pts1_inliers_custom = pts1_matched[mask_custom.ravel() == 1]
        pts2_inliers_custom = pts2_matched[mask_custom.ravel() == 1]

        # compute epipolar errors
        epi_errors_custom = compute_epipolar_errors(pts1_inliers_custom, pts2_inliers_custom, F_custom)
        print(f"Epipolar Constraint Errors (Custom): mean={epi_errors_custom.mean():.4f}, std={epi_errors_custom.std():.4f}")

        print("Visualizing epipolar lines for custom features...")
        img1_epilines_custom, img2_epilines_custom = visualize_epilines_on_images(
            img1,
            img2,
            pts1_inliers_custom,
            pts2_inliers_custom,
            F_custom,
        )

    ###using SIFT ###
    print("\n--- Using SIFT Features ---")
    print("Detecting and matching features using SIFT...")
    pts1_sift, pts2_sift = sift_feature_matching(img1, img2)

    if pts1_sift is None or pts2_sift is None or len(pts1_sift) < 8:
        print("Not enough SIFT matches to compute Fundamental Matrix.")
        return

    print(f"Number of SIFT matched points: {len(pts1_sift)}")

    print("Estimating Fundamental Matrix using OpenCV...")
    F_sift, mask_sift = cv2.findFundamentalMat(pts1_sift, pts2_sift, cv2.FM_RANSAC, ransacReprojThreshold=3.0)
    if F_sift is None or F_sift.shape != (3, 3):
        print("Fundamental matrix could not be estimated with SIFT matches.")
        return
    print("Fundamental Matrix estimated.")

    num_inliers_sift = np.sum(mask_sift)
    num_outliers_sift = len(mask_sift) - num_inliers_sift
    print(f"Number of inliers after RANSAC: {num_inliers_sift}")
    print(f"Number of outliers after RANSAC: {num_outliers_sift}")

    if num_inliers_sift < 2:
        print("Not enough inliers to visualize epilines.")
        return

    pts1_inliers_sift = pts1_sift[mask_sift.ravel() == 1]
    pts2_inliers_sift = pts2_sift[mask_sift.ravel() == 1]

    epi_errors_sift = compute_epipolar_errors(pts1_inliers_sift, pts2_inliers_sift, F_sift)
    print(f"Epipolar Constraint Errors (SIFT): mean={epi_errors_sift.mean():.4f}, std={epi_errors_sift.std():.4f}")

    print("Visualizing epipolar lines for SIFT features...")
    img1_epilines_sift, img2_epilines_sift = visualize_epilines_on_images(
        img1,
        img2,
        pts1_inliers_sift,
        pts2_inliers_sift,
        F_sift,
    )

    print("\n--- Displaying Comparison ---")
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1_epilines_custom, cv2.COLOR_BGR2RGB))
    plt.title('Custom Features - Image 1 with Epilines')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2_epilines_custom, cv2.COLOR_BGR2RGB))
    plt.title('Custom Features - Image 2 with Epilines')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img1_epilines_sift, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Features - Image 1 with Epilines')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img2_epilines_sift, cv2.COLOR_BGR2RGB))
    plt.title('SIFT Features - Image 2 with Epilines')

    output_path = '/Users/kohsukeide/dev/perspective-n-gaussian/outputs/epilines_comparison.png'
    plt.savefig(output_path)
    plt.show()
    print(f"Comparison visualization saved to '{output_path}'.")

if __name__ == "__main__":
    main()