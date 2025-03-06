import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        print('initialize camera pose visualizer')

    def extrinsic2pyramid(self, extrinsic, color_map='red', focal_len_scaled=5, aspect_ratio=0.3, plotly_viz=False, legend_group='g', name='n', show_legend=True):
        vertex_std = np.array([[0, 0, 0, 1],
                               [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                               [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                            [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

        if plotly_viz:
            color = sample_colorscale('rainbow', color_map)[0]
            self.plotly_data = self.draw_interactive(meshes=meshes, color=color, legend_group=legend_group, name=name, show_legend=show_legend)

    def draw_interactive(self, meshes, color, legend_group, name, show_legend):
        x = [polygon[0] for vertice in meshes for polygon in vertice]
        y = [polygon[1] for vertice in meshes for polygon in vertice]
        z = [polygon[2] for vertice in meshes for polygon in vertice]

        # Extract camera position (first vertex is camera center)
        camera_pos = meshes[0][0]  # This is the camera center

        data = go.Mesh3d(
            x=x, 
            y=y, 
            z=z, 
            opacity=0.7,
            color=color, 
            showlegend=show_legend, 
            legendgroup=legend_group, 
            name=name,
            hoverinfo='text',
            hovertext=f"{name}<br>Position: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})",
            hoverlabel=dict(bgcolor='white', font_size=12)
        )

        return data

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc='right', bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', label='Frame Number')

    def show(self):
        plt.title('Extrinsic Parameters')
        plt.show()

def read_images_txt(images_txt_path):
    """
    Read camera poses from COLMAP's images.txt file
    """
    cameras = []
    with open(images_txt_path, 'r') as f:
        is_camera_line = True
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if is_camera_line:
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                elements = line.split()
                image_id = int(elements[0])
                qw, qx, qy, qz = map(float, elements[1:5])
                tx, ty, tz = map(float, elements[5:8])
                camera_id = int(elements[8])
                image_name = elements[9]
                
                # Construct rotation matrix from quaternion
                q = np.array([qw, qx, qy, qz])
                q = q / np.linalg.norm(q)  # Normalize
                
                # Quaternion to rotation matrix (world-to-camera rotation)
                R = np.zeros((3, 3))
                R[0, 0] = 1 - 2 * (q[2]**2 + q[3]**2)
                R[0, 1] = 2 * (q[1] * q[2] - q[3] * q[0])
                R[0, 2] = 2 * (q[1] * q[3] + q[2] * q[0])
                R[1, 0] = 2 * (q[1] * q[2] + q[3] * q[0])
                R[1, 1] = 1 - 2 * (q[1]**2 + q[3]**2)
                R[1, 2] = 2 * (q[2] * q[3] - q[1] * q[0])
                R[2, 0] = 2 * (q[1] * q[3] - q[2] * q[0])
                R[2, 1] = 2 * (q[2] * q[3] + q[1] * q[0])
                R[2, 2] = 1 - 2 * (q[1]**2 + q[2]**2)
                
                # 並進ベクトル (world-to-camera translation)
                t = np.array([tx, ty, tz]).reshape(3, 1)
                
                # カメラ中心座標（ワールド座標系）を計算
                # C = -R.T @ t が正しいカメラ中心
                C = -R.T @ t
                
                # Construct camera-to-world transformation
                # COLMAPのカメラ座標系は右手系：Z軸が前方向き
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R.T  # camera-to-worldの回転
                extrinsic[:3, 3] = C.flatten()  # カメラの中心位置
                
                cameras.append({
                    'image_id': image_id,
                    'camera_id': camera_id,
                    'name': image_name,
                    'extrinsic': extrinsic,
                    'quaternion': q,
                    'position': C.flatten(),  # カメラ中心
                    'R': R,  # 回転行列
                    't': t.flatten()  # 並進ベクトル
                })
            
            is_camera_line = not is_camera_line
    
    return cameras

def read_points3D_txt(points3D_txt_path):
    """
    Read 3D points from COLMAP's points3D.txt file
    
    Returns:
        points: Nx3 array of 3D point positions
        colors: Nx3 array of RGB colors (0-255)
    """
    points = []
    colors = []
    
    with open(points3D_txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            elements = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            point_id = int(elements[0])
            x, y, z = map(float, elements[1:4])
            r, g, b = map(int, elements[4:7])
            
            points.append([x, y, z])
            colors.append([r, g, b])
    
    return np.array(points), np.array(colors)

def read_points3D_bin(points3D_bin_path):
    """
    Read 3D points from COLMAP's points3D.bin file
    
    Returns:
        points: Nx3 array of 3D point positions
        colors: Nx3 array of RGB colors (0-255)
    """
    import struct
    
    points = []
    colors = []
    
    with open(points3D_bin_path, 'rb') as f:
        # Read number of points
        num_points = struct.unpack('Q', f.read(8))[0]
        
        # Define data structure
        point_data_struct = struct.Struct('<Q 3d 3B d Q')
        
        for i in range(num_points):
            data = point_data_struct.unpack(f.read(point_data_struct.size))
            point_id = data[0]
            x, y, z = data[1:4]
            r, g, b = data[4:7]
            
            points.append([x, y, z])
            colors.append([r, g, b])
            
            # Skip track information
            track_len = data[8]
            f.seek(track_len * 2 * 4, os.SEEK_CUR)  # 2 uint32 per track element
    
    return np.array(points), np.array(colors)

def read_points3D(points3D_path):
    """
    Read 3D points from COLMAP's points3D file (either .txt or .bin)
    """
    if points3D_path.endswith('.txt'):
        return read_points3D_txt(points3D_path)
    elif points3D_path.endswith('.bin'):
        return read_points3D_bin(points3D_path)
    else:
        raise ValueError("Points3D file must have .txt or .bin extension")

def visualize_cameras_and_points(cameras, points=None, colors=None, scene_bounds=None, use_plotly=True, max_points=50000, focal_len=5, aspect_ratio=0.3):
    """
    Visualize cameras and optionally points using CameraPoseVisualizer
    
    Args:
        cameras: List of camera dictionaries from read_images_txt
        points: Nx3 array of 3D point positions
        colors: Nx3 array of RGB colors (0-255)
        scene_bounds: Tuple of (min_bound, max_bound) for x, y, z
        use_plotly: Whether to use plotly for interactive visualization
        max_points: Maximum number of points to display
        focal_len: Focal length for camera visualization
        aspect_ratio: Aspect ratio for camera visualization
    """
    # Calculate camera positions
    camera_positions = np.array([cam['position'] for cam in cameras])
    
    # Calculate scene bounds if not provided
    if scene_bounds is None:
        # Calculate frustum vertices for all cameras
        all_vertices = []
        for cam in cameras:
            # Create standard frustum vertices
            vertex_std = np.array([
                [0, 0, 0, 1],  # Camera center
                [focal_len * aspect_ratio, -focal_len * aspect_ratio, focal_len, 1],  # Top-right
                [focal_len * aspect_ratio, focal_len * aspect_ratio, focal_len, 1],   # Top-left
                [-focal_len * aspect_ratio, focal_len * aspect_ratio, focal_len, 1],  # Bottom-left
                [-focal_len * aspect_ratio, -focal_len * aspect_ratio, focal_len, 1]  # Bottom-right
            ])
            
            # Transform vertices to world space
            vertex_transformed = vertex_std @ cam['extrinsic'].T
            
            # Add to list of all vertices (excluding homogeneous coordinate)
            all_vertices.extend([v[:-1] for v in vertex_transformed])
        
        # Convert to numpy array
        all_vertices = np.array(all_vertices)
        
        # Combine camera positions and frustum vertices
        all_points = np.vstack([camera_positions, all_vertices])
        
        # Calculate center and extent
        center = all_points.mean(axis=0)
        distances = np.linalg.norm(all_points - center, axis=1)
        max_dist = np.max(distances)
        
        # Make the bounds a bit larger to ensure all frustums are visible
        scene_bounds = (
            [center[0] - max_dist*1.2, center[0] + max_dist*1.2],
            [center[1] - max_dist*1.2, center[1] + max_dist*1.2],
            [center[2] - max_dist*1.2, center[2] + max_dist*1.2]
        )
    
    # Initialize visualizer
    visualizer = CameraPoseVisualizer(
        scene_bounds[0], scene_bounds[1], scene_bounds[2]
    )
    
    # For plotly visualization
    if use_plotly:
        final_layout = go.Figure()
        final_layout.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=scene_bounds[0]),
                yaxis=dict(nticks=4, range=scene_bounds[1]),
                zaxis=dict(nticks=4, range=scene_bounds[2]),
                aspectmode='data'
            ),
            title='Camera and Point Cloud Visualization',
            legend=dict(x=0.8, y=0.5, font=dict(color='black', size=12))
        )
    
    # Add each camera
    for i, cam in enumerate(cameras):
        color_val = i / max(1, len(cameras) - 1)  # Avoid division by zero
        
        # Create a more informative name for the camera
        camera_name = f"Camera {i+1}: {cam['name']}"
        
        visualizer.extrinsic2pyramid(
            cam['extrinsic'], 
            color_map=color_val,
            focal_len_scaled=focal_len,
            aspect_ratio=aspect_ratio,
            plotly_viz=use_plotly,
            legend_group="Cameras",  # Group all cameras together
            name=camera_name,  # Use the camera name
            show_legend=True  # Show all cameras in legend
        )
        
        if use_plotly:
            final_layout.add_trace(visualizer.plotly_data)
    
    # Add point cloud if available
    if points is not None and len(points) > 0:
        # Subsample points if there are too many
        if len(points) > max_points:
            print(f"Subsampling point cloud from {len(points)} to {max_points} points")
            indices = np.random.choice(len(points), max_points, replace=False)
            points_subset = points[indices]
            colors_subset = colors[indices] if colors is not None else None
        else:
            points_subset = points
            colors_subset = colors
        
        if use_plotly:
            # Prepare colors for plotting
            point_colors = colors_subset
            if colors_subset is None:
                point_colors = np.ones((len(points_subset), 3)) * np.array([255, 0, 0])  # Red color
            
            # Convert RGB (0-255) to hex color strings for Plotly
            point_colors_hex = [f'rgb({r},{g},{b})' for r, g, b in point_colors]
            
            # Add scatter3d trace for points
            final_layout.add_trace(
                go.Scatter3d(
                    x=points_subset[:, 0],
                    y=points_subset[:, 1],
                    z=points_subset[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1.5,
                        color=point_colors_hex,
                        opacity=0.7
                    ),
                    name='3D Points',
                    showlegend=True
                )
            )
        else:
            # Matplotlib version
            point_colors = colors_subset
            if colors_subset is None:
                point_colors = np.ones((len(points_subset), 3)) * np.array([1.0, 0.0, 0.0])  # Red color (already in 0-1 range)
            else:
                # Normalize to 0-1 range for matplotlib
                point_colors = point_colors.astype(float) / 255.0
            
            # Add scatter plot
            visualizer.ax.scatter(
                points_subset[:, 0], points_subset[:, 1], points_subset[:, 2],
                c=point_colors, s=0.5, alpha=0.5
            )
    
    # Show visualization
    if use_plotly:
        # Create camera selection buttons
        camera_buttons = []
        for i, cam in enumerate(cameras):
            camera_buttons.append(
                dict(
                    method="update",
                    args=[
                        {"visible": [j == i for j in range(len(cameras))] + [True] * (1 if points is not None else 0)},
                        {"title": f"Camera {i+1}: {cam['name']}"}
                    ],
                    label=f"Camera {i+1}"
                )
            )
        
        # Add "Show All" button
        camera_buttons.append(
            dict(
                method="update",
                args=[
                    {"visible": [True] * (len(cameras) + (1 if points is not None else 0))},
                    {"title": "All Cameras"}
                ],
                label="Show All"
            )
        )
        
        # Add buttons to layout
        final_layout.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=camera_buttons,
                    showactive=True,
                    x=0.1,
                    y=1.15,
                    xanchor="left",
                    yanchor="top"
                )
            ]
        )
        final_layout.show()
    else:
        visualizer.colorbar(len(cameras))
        visualizer.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize COLMAP camera poses and 3D points')
    parser.add_argument('--images', type=str, required=True, help='Path to COLMAP images.txt file')
    parser.add_argument('--points', type=str, default=None, help='Path to COLMAP points3D.txt or points3D.bin file')
    parser.add_argument('--plotly', action='store_true', default=True, help='Use Plotly for interactive visualization')
    parser.add_argument('--max_points', type=int, default=50000, help='Maximum number of points to display')
    parser.add_argument('--focal_len', type=float, default=2, help='Focal length for camera visualization')
    parser.add_argument('--aspect_ratio', type=float, default=0.1, help='Aspect ratio for camera visualization')
    
    args = parser.parse_args()
    
    # Read camera poses
    cameras = read_images_txt(args.images)
    print(f"Loaded {len(cameras)} cameras from {args.images}")
    
    points = None
    colors = None
    
    # Read points if path is provided
    if args.points:
        try:
            points, colors = read_points3D(args.points)
            print(f"Loaded {len(points)} points from {args.points}")
        except Exception as e:
            print(f"Error loading points: {e}")
            import traceback
            traceback.print_exc()
    
    # Visualize cameras and points
    visualize_cameras_and_points(
        cameras, 
        points, 
        colors, 
        use_plotly=args.plotly,
        max_points=args.max_points,
        focal_len=args.focal_len,
        aspect_ratio=args.aspect_ratio
    )
    
if __name__ == "__main__":
    main()

#python extrinsics_visualizer.py --images /Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/final/colmap/images.txt --points /Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/final/colmap/points3d.txt

#python extrinsics_visualizer.py --images /Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0/images_correct.txt --points /Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0/points3D.txt