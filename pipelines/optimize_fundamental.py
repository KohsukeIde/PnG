import os
import sys
import pickle
import torch
import numpy as np
import cv2

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.camera.camera_model import CameraModel
from src.utils.colmap_utils import load_cameras_from_colmap, load_images_from_colmap
from utils.gs_pkl_loader import load_gaussians_torch

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians_rs']

############################################
# Visualization or evaluation utility
############################################
def visualize_point_matches(img1, img2, pts1, pts2, output_dir='results', prefix='matches'):
    """
    Simple function to visualize corresponding points between two images.
    """
    # Create a canvas to draw matches
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Place images side by side
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    
    # Draw lines
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        color = (0, 255, 255)  # Cyan
        p1 = (int(x1), int(y1))
        p2 = (int(x2) + w1, int(y2))
        cv2.circle(canvas, p1, 5, color, -1)
        cv2.circle(canvas, p2, 5, color, -1)
        cv2.line(canvas, p1, p2, color, 1)
    
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"{prefix}.jpg")
    cv2.imwrite(outpath, canvas)
    print(f"Saved point matches visualization to {outpath}")


def visualize_epipolar_lines(img1, img2, pts1, pts2, F, output_dir='results', prefix='epilines'):
    """
    Visualize epipolar lines for corresponding points given Fundamental matrix F.
    """
    # Convert points to int
    pts1 = pts1.astype(np.int32)
    pts2 = pts2.astype(np.int32)
    
    # Convert to homogeneous
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
    
    # Lines in image1 from pts2
    lines1 = (F @ pts2_h.T).T  # shape (N,3)
    # Lines in image2 from pts1
    lines2 = (F.T @ pts1_h.T).T  # shape (N,3)
    
    # Draw lines in img1
    img1_draw = img1.copy()
    for (a, b, c), (x, y) in zip(lines1, pts1):
        # epipolar line ax + by + c = 0
        # We find 2 extreme points for drawing
        h, w = img1_draw.shape[:2]
        # x=0 => y= -c/b, x=w => y= -(c + a*w)/b
        if abs(b) < 1e-6:
            continue
        y0 = int(-c / b)
        y1 = int(-(c + a*w) / b)
        color = (0,255,0)
        cv2.line(img1_draw, (0, y0), (w, y1), color, 1)
        cv2.circle(img1_draw, (x,y), 5, (0,0,255), -1)
    
    # Draw lines in img2
    img2_draw = img2.copy()
    for (a, b, c), (x, y) in zip(lines2, pts2):
        h, w = img2_draw.shape[:2]
        if abs(b) < 1e-6:
            continue
        y0 = int(-c / b)
        y1 = int(-(c + a*w) / b)
        color = (0,255,0)
        cv2.line(img2_draw, (0, y0), (w, y1), color, 1)
        cv2.circle(img2_draw, (x,y), 5, (0,0,255), -1)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_img1.jpg"), img1_draw)
    cv2.imwrite(os.path.join(output_dir, f"{prefix}_img2.jpg"), img2_draw)
    print(f"Saved epipolar lines to {output_dir}")


def compute_epipolar_residual(F, pts1, pts2):
    """
    Compute average epipolar distance residual: |x2^T F x1|.
    """
    # homogeneous
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N,3)
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N,3)
    
    Fx1 = F @ pts1_h.T  # shape (3,N)
    residuals = np.abs(np.sum(pts2_h * Fx1.T, axis=1))
    return np.mean(residuals)


############################################
# 1) Retrieve top correspondences based on transport
############################################
def get_top_correspondences_fundamental(solver, num_points=100):
    """
    Get top 1:1 correspondences based on transport matrix T using the *Fundamental-based* cost.
    We will call solver.compute_cost_matrix_fundamental(solver.f) and sinkhorn or unbalanced sinkhorn.
    """
    with torch.no_grad():
        cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
        transport_matrix = solver.unbalanced_sinkhorn_algorithm(cost_matrix)
    
    T_np = transport_matrix.cpu().numpy()
    points1 = solver.means1.cpu().numpy()
    points2 = solver.means2.cpu().numpy()
    
    # Sort in descending order
    flat_indices = np.argsort(-T_np.ravel())
    used_rows = set()
    used_cols = set()
    matches = []
    
    # pick top correspondences greedily
    for idx in flat_indices:
        row = idx // T_np.shape[1]
        col = idx % T_np.shape[1]
        if len(matches) >= num_points:
            break
        if row not in used_rows and col not in used_cols:
            matches.append((row,col))
            used_rows.add(row)
            used_cols.add(col)
    
    if len(matches) == 0:
        print(f"No matches found with top {num_points}.")
        return np.zeros((0,2)), np.zeros((0,2))
    
    matched_rows, matched_cols = zip(*matches)
    matched_rows = np.array(matched_rows)
    matched_cols = np.array(matched_cols)
    
    pts1 = points1[matched_rows]
    pts2 = points2[matched_cols]
    
    # Debug info
    print(f"\nTransport matrix shape: {T_np.shape}")
    print(f"Top {num_points} matches found: {len(matches)}")
    print(f"Transport max: {T_np.max()}, min: {T_np.min()}")
    return pts1, pts2

############################################
# 2) Main pipeline using Fundamental
############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data paths
    data_dir = '/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63'
    data_dir_gmm = '/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs'
    gaussians1_path = os.path.join(data_dir_gmm, 'fitted_gaussians_22_1k.pkl')
    gaussians2_path = os.path.join(data_dir_gmm, 'fitted_gaussians_23_1k.pkl')
    
    colmap_dir = os.path.join(data_dir, 'sparse/0')
    image1_name = '0022.png'
    image2_name = '0023.png'

    # 1) Load Gaussians
    _, gaussians1, _, _ = load_gaussians_torch(gaussians1_path, device)
    _, gaussians2, _, _ = load_gaussians_torch(gaussians2_path, device)

    # 2) Load camera + COLMAP info
    cameras = load_cameras_from_colmap(colmap_dir)
    images_data = load_images_from_colmap(colmap_dir)
    image_name_to_id = {data['name']: image_id for image_id, data in images_data.items()}

    image1_id = image_name_to_id.get(image1_name)
    image2_id = image_name_to_id.get(image2_name)
    if image1_id is None or image2_id is None:
        print("Could not find images in COLMAP data.")
        sys.exit(1)

    camera1_id = images_data[image1_id]['camera_id']
    camera2_id = images_data[image2_id]['camera_id']
    camera1 = CameraModel(cameras[camera1_id], image1_id, images_data)
    camera2 = CameraModel(cameras[camera2_id], image2_id, images_data)

    K1 = camera1.K
    K2 = camera2.K

    # 3) Initialize solver & Fundamental optimization
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=K1,
        k2=K2,
        epsilon=0.1,
        lambda_mean=3.0,
        lambda_cov=1.0,
        lambda_color=1.0,
        device=device
    )

    # Optimize with Fundamental
    print("\n--- Optimizing Fundamental Matrix ---")
    solver.optimize_with_fundamental(max_iter=500, tol=1e-6)
    
    # Extract final F
    F_optimized = solver.f.detach().cpu().numpy()
    print("\nOptimized Fundamental matrix:\n", F_optimized)

    # 4) Evaluate transport & get top correspondences
    pts1, pts2 = get_top_correspondences_fundamental(solver, num_points=100)

    # If images are available, visualize
    image1_path = os.path.join(data_dir, 'images', image1_name)
    image2_path = os.path.join(data_dir, 'images', image2_name)
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is not None and img2 is not None and len(pts1) > 0:
        visualize_point_matches(img1, img2, pts1, pts2, output_dir='results', prefix='matches_fundamental')
        
        # Visualize epipolar lines
        visualize_epipolar_lines(img1, img2, pts1, pts2, F_optimized, output_dir='results', prefix='epilines_fundamental')
        
        # Compute epipolar residual
        ep_residual = compute_epipolar_residual(F_optimized, pts1, pts2)
        print(f"Average epipolar residual among top correspondences: {ep_residual:.6f}")
    else:
        print("Could not visualize epipolar lines because images not loaded or no correspondences found.")

    # 5) Save results to pickle
    results = {
        'fundamental_matrix': F_optimized,
    }
    os.makedirs('results', exist_ok=True)
    with open('results/fundamental_optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Saved fundamental optimization results to 'results/fundamental_optimization_results.pkl'")

    print("\nDone.")

if __name__ == '__main__':
    main()
