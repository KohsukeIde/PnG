import os
import torch
import numpy as np
from typing import List, Tuple, Optional

def export_gaussians_for_init(
    output_path: str,
    points_3d: np.ndarray,
    quaternions: np.ndarray,
    scales: np.ndarray,
    alphas_3d: np.ndarray,
    colors_3d: np.ndarray,
    covariances_3d: Optional[np.ndarray] = None
) -> None:
    """Gaussian Splatting initialization data in Torch format.
    
    Args:
        output_path: Output file path (.pt)
        points_3d: 3D point positions [N, 3]
        quaternions: Quaternions [N, 4] 
        scales: Scales [N, 3]
        alphas_3d: Opacity values [N]
        colors_3d: RGB colors [N, 3]
        covariances_3d: 3D covariance matrices [N, 3, 3] (optional)
    """
    # Convert NumPy arrays to Tensors
    means = torch.tensor(points_3d, dtype=torch.float32)
    quats = torch.tensor(quaternions, dtype=torch.float32)
    scales = torch.tensor(scales, dtype=torch.float32)
    opacities = torch.tensor(alphas_3d, dtype=torch.float32)
    
    # Create data structure
    gaussian_data = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
    }
    
    # Add color information (as SH coefficients)
    sh0 = torch.tensor(colors_3d, dtype=torch.float32)
    gaussian_data["sh0"] = sh0
    
    # Create shN with one coefficient (for default SH coefficient setup)
    gaussian_data["shN"] = torch.zeros((len(points_3d), 0, 3), dtype=torch.float32)
    
    # Optionally include covariance matrices
    if covariances_3d is not None:
        gaussian_data["covariances"] = torch.tensor(covariances_3d, dtype=torch.float32)
    
    # Save as .pt file
    torch.save(gaussian_data, output_path)
    print(f"Saved Gaussian initialization data to {output_path}")
    
    # Save simple metadata
    meta_path = output_path.replace('.pt', '_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"Total Gaussians: {len(points_3d)}\n")
        f.write(f"Format: Torch tensors (means, quats, scales, opacities, sh0, shN)\n")
        f.write(f"means shape: {means.shape}\n")
        f.write(f"quats shape: {quats.shape}\n")
        f.write(f"scales shape: {scales.shape}\n")
        f.write(f"opacities shape: {opacities.shape}\n")
        if covariances_3d is not None:
            f.write(f"covariances shape: {tuple(torch.tensor(covariances_3d).shape)}\n")
    
    print(f"Saved metadata to {meta_path}")

def export_gaussians_to_colmap_dir(
    colmap_dir: str,
    points_3d: np.ndarray,
    covariances_3d: np.ndarray,
    colors_3d: np.ndarray,
    alphas_3d: np.ndarray,
    quaternions: np.ndarray,
    scales: np.ndarray,
    save_ellipsoids_as_ply=None,
    save_gaussians_as_ply=None
) -> None:
    """Export Gaussian information to COLMAP directory.
    
    Args:
        colmap_dir: Path to COLMAP directory
        points_3d: 3D point positions [N, 3]
        covariances_3d: 3D covariance matrices [N, 3, 3]
        colors_3d: RGB colors [N, 3]
        alphas_3d: Opacity values [N]
        quaternions: Quaternions [N, 4]
        scales: Scales [N, 3]
        save_ellipsoids_as_ply: PLY save function (skipped if None)
        save_gaussians_as_ply: PLY save function (skipped if None)
    """
    # Save ellipsoids PLY with full Gaussian information
    if save_ellipsoids_as_ply is not None:
        ply_ellipsoids_path = os.path.join(colmap_dir, "gaussians_ellipsoids.ply")
        save_ellipsoids_as_ply(
            points_3d=points_3d,
            covariances_3d=covariances_3d,
            colors_3d=colors_3d,
            alphas_3d=alphas_3d,
            filename=ply_ellipsoids_path,
            use_alpha=True
        )
        print(f"Saved ellipsoids with full Gaussian information to {ply_ellipsoids_path}")
    
    # Save Gaussian Splatting PLY format
    if save_gaussians_as_ply is not None:
        ply_gs_path = os.path.join(colmap_dir, "gaussians_splat.ply")
        save_gaussians_as_ply(
            points_3d=points_3d,
            quaternions=quaternions,
            scales=scales,
            colors_3d=colors_3d,
            alphas_3d=alphas_3d,
            filename=ply_gs_path
        )
        print(f"Saved Gaussian Splatting PLY format to {ply_gs_path}")
    
    # Save Torch format Gaussian information
    torch_path = os.path.join(colmap_dir, "gaussians.pt")
    export_gaussians_for_init(
        output_path=torch_path,
        points_3d=points_3d,
        quaternions=quaternions,
        scales=scales,
        alphas_3d=alphas_3d,
        colors_3d=colors_3d,
        covariances_3d=covariances_3d
    )

def export_colmap_format(
    output_dir: str,
    points_3d: np.ndarray,
    camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
    match_points_2d: List[List[Tuple[int, np.ndarray]]],
    intrinsics_list: List[np.ndarray],
    image_names: Optional[List[str]] = None
) -> None:
    """Export reconstruction data in COLMAP format.
    
    Args:
        output_dir: Directory to save the COLMAP files
        points_3d: 3D point coordinates [N, 3]
        camera_params_list: List of (R, t) for each camera
        match_points_2d: For each camera, list of (point_idx, [x, y]) observations
        intrinsics_list: List of 3x3 intrinsic matrices
        image_names: Optional list of image names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export cameras (cameras.txt)
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        # Header
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        # Simple pinhole camera model for all cameras
        for i, K in enumerate(intrinsics_list):
            width = int(K[0, 2] * 2)  # Approximate from principal point
            height = int(K[1, 2] * 2)
            f.write(f"{i+1} SIMPLE_PINHOLE {width} {height} {K[0, 0]} {K[0, 2]} {K[1, 2]}\n")
    
    # Export images (images.txt)
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        # Header
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, (R, t) in enumerate(camera_params_list):
            # Convert R to quaternion
            rot_mat = np.eye(4)
            rot_mat[:3, :3] = R
            rot_mat[:3, 3] = t
            
            # Extract quaternion (important: ensure the order is qw, qx, qy, qz)
            trace = np.trace(rot_mat[:3, :3])
            if trace > 0:
                s = 0.5 / np.sqrt(trace + 1.0)
                qw = 0.25 / s
                qx = (rot_mat[2, 1] - rot_mat[1, 2]) * s
                qy = (rot_mat[0, 2] - rot_mat[2, 0]) * s
                qz = (rot_mat[1, 0] - rot_mat[0, 1]) * s
            else:
                if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
                    qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                    qx = 0.25 * s
                    qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                    qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                elif rot_mat[1, 1] > rot_mat[2, 2]:
                    s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
                    qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                    qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                    qy = 0.25 * s
                    qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                else:
                    s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
                    qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
                    qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                    qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                    qz = 0.25 * s
            
            # Normalize quaternion
            quaternion = np.array([qw, qx, qy, qz])
            quat_norm = np.linalg.norm(quaternion)
            if not np.isclose(quat_norm, 1.0, rtol=1e-4):
                quaternion /= quat_norm
            qw, qx, qy, qz = quaternion
            
            # Use provided image name if available, otherwise use a default name
            if image_names and i < len(image_names):
                image_name = image_names[i]
            else:
                image_name = f"image_{i+1:06d}.jpg"
            
            # Write image info - first line
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i+1} {image_name}\n")
            
            # Write point observations - second line
            points_str = ""
            if i < len(match_points_2d):
                for point_idx, point_2d in match_points_2d[i]:
                    points_str += f"{point_2d[0]:.6f} {point_2d[1]:.6f} {point_idx+1} "
            
            # Always write the second line, even if empty
            f.write(f"{points_str}\n")
    
    # Export points3D (points3d.txt)
    with open(os.path.join(output_dir, 'points3d.txt'), 'w') as f:
        # Header
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        for i, point in enumerate(points_3d):
            # Default color (white)
            r, g, b = 255, 255, 255
            
            # Find all track elements
            track = []
            for cam_idx, observations in enumerate(match_points_2d):
                for j, (point_idx, _) in enumerate(observations):
                    if point_idx == i:
                        track.append((cam_idx + 1, j))
            
            track_str = " ".join([f"{cam_id} {point2d_idx}" for cam_id, point2d_idx in track])
            
            # Write 3D point
            f.write(f"{i+1} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {r} {g} {b} 1.0 {track_str}\n")
    
    # Also export as PLY
    export_points_as_ply(os.path.join(output_dir, 'points3d.ply'), points_3d)
    
    print(f"COLMAP format data exported to {output_dir}")

def export_points_as_ply(filepath: str, points_3d: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Export 3D points as PLY file.
    
    Args:
        filepath: Path to save the PLY file
        points_3d: 3D point coordinates [N, 3]
        colors: Optional RGB colors [N, 3]
    """
    with open(filepath, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Points (with default white color if not provided)
        if colors is None:
            colors = np.full((len(points_3d), 3), 255, dtype=np.uint8)
        elif colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = np.clip(colors, 0, 255).astype(np.uint8)
        
        for i, point in enumerate(points_3d):
            color = colors[i] if i < len(colors) else [255, 255, 255]
            f.write(f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"PLY file exported to {filepath}")