import os
import matplotlib.pyplot as plt
import numpy as np
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

        data = go.Mesh3d(x=x, y=y, z=z, opacity=1, color=color, showlegend=show_legend, legendgroup=legend_group, name=name)

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
                
                # Quaternion to rotation matrix
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
                
                # Construct extrinsic matrix (camera to world)
                t = np.array([tx, ty, tz]).reshape(3, 1)
                extrinsic = np.zeros((4, 4))
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = t.flatten()
                extrinsic[3, 3] = 1.0
                
                cameras.append({
                    'image_id': image_id,
                    'camera_id': camera_id,
                    'name': image_name,
                    'extrinsic': extrinsic,
                    'quaternion': q,
                    'position': t.flatten()
                })
            
            is_camera_line = not is_camera_line
    
    return cameras

def visualize_cameras(cameras, scene_bounds=None, use_plotly=False):
    """
    Visualize cameras using CameraPoseVisualizer
    
    Args:
        cameras: List of camera dictionaries from read_images_txt
        scene_bounds: Tuple of (min_bound, max_bound) for x, y, z
        use_plotly: Whether to use plotly for interactive visualization
    """
    # Calculate scene bounds if not provided
    if scene_bounds is None:
        positions = np.array([cam['position'] for cam in cameras])
        center = positions.mean(axis=0)
        max_dist = np.max(np.linalg.norm(positions - center, axis=1))
        scene_bounds = (
            [-max_dist*10, max_dist*10],
            [-max_dist*10, max_dist*10],
            [-max_dist*10, max_dist*10]
        )
    
    # Initialize visualizer
    visualizer = CameraPoseVisualizer(
        scene_bounds[0], scene_bounds[1], scene_bounds[2]
    )
    
    # Add cameras to visualizer
    if use_plotly:
        final_layout = go.Figure()
        final_layout.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, range=scene_bounds[0]),
                yaxis=dict(nticks=4, range=scene_bounds[1]),
                zaxis=dict(nticks=4, range=scene_bounds[2]),
            ),
            legend=dict(x=0.7, y=0.5, font=dict(color='black', size=5))
        )
    
    # Add each camera
    for i, cam in enumerate(cameras):
        color_val = i / len(cameras)
        visualizer.extrinsic2pyramid(
            cam['extrinsic'], 
            color_map=color_val,
            focal_len_scaled=3,
            aspect_ratio=0.3,
            plotly_viz=use_plotly,
            legend_group=f"Camera {i+1}",
            name=cam['name'],
            show_legend=(i == 0)  # Only show legend for first camera
        )
        
        if use_plotly:
            final_layout.add_trace(visualizer.plotly_data)
    
    # Add point cloud if available (commented out as it requires points3D data)
    # if points3D is not None:
    #     # Add points as scatter
    #     ax.scatter(
    #         points3D[:, 0], points3D[:, 1], points3D[:, 2],
    #         c='gray', alpha=0.3, s=1
    #     )
    
    # Show visualization
    if use_plotly:
        final_layout.show()
    else:
        visualizer.colorbar(len(cameras))
        visualizer.show()

def main():
    # Path to COLMAP images.txt file
    images_txt_path = "/Users/kohsukeide/dev/perspective-n-gaussian/data/DTU/scan63/sparse/0/images_correct.txt"
    
    # images_txt_path = "/Users/kohsukeide/dev/perspective-n-gaussian/pipelines/results/final/colmap/images.txt"
    
    # Read camera poses
    cameras = read_images_txt(images_txt_path)
    print(f"Loaded {len(cameras)} cameras from {images_txt_path}")
    
    # Optional: Read points3D.txt to visualize point cloud
    # points3D = read_points3D_txt("path/to/your/colmap/points3D.txt")
    
    # Visualize cameras
    visualize_cameras(cameras, use_plotly=True)  # Set use_plotly=False for matplotlib
    
if __name__ == "__main__":
    main()