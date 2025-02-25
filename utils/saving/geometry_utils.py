# utils/saving/geometry_utils.py
import numpy as np
import torch


def sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta=12, n_phi=12):
    """
    Approximate the surface of a 3D ellipsoid (Gaussian) by sampling a sphere
    and applying sqrt(Sigma_3).

    Args:
        Sigma_3 (np.ndarray): 3x3 positive semi-definite covariance matrix
        center (np.ndarray): shape (3,), 3D center
        n_theta (int): number subdivisions for azimuth
        n_phi (int): number subdivisions for polar angle

    Returns:
        vertices (list of [x,y,z]): approximated ellipsoid vertices in 3D
        faces (list of [v1,v2,v3]): triangular faces (indices into vertices)
    """
    if isinstance(Sigma_3, torch.Tensor):
        Sigma_3 = Sigma_3.detach().cpu().numpy()
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    
    eigvals, eigvecs = np.linalg.eigh(Sigma_3)
    eigvals = np.clip(eigvals, 1e-12, None)
    scales = np.sqrt(eigvals)
    sqrtSigma = eigvecs @ np.diag(scales) @ eigvecs.T

    vertices = []
    faces = []
    for i in range(n_theta+1):
        theta = 2.0 * np.pi * i / n_theta
        for j in range(n_phi+1):
            phi = np.pi * j / n_phi
            x_sph = np.sin(phi) * np.cos(theta)
            y_sph = np.sin(phi) * np.sin(theta)
            z_sph = np.cos(phi)
            unit_vec = np.array([x_sph, y_sph, z_sph])
            xyz_ellip = sqrtSigma @ unit_vec
            xyz_ellip += center
            vertices.append(xyz_ellip)

    def idx(ii, jj):
        return ii*(n_phi+1) + jj

    for i in range(n_theta):
        for j in range(n_phi):
            v1 = idx(i,   j)
            v2 = idx(i+1, j)
            v3 = idx(i,   j+1)
            v4 = idx(i+1, j+1)
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])

    return vertices, faces


def create_camera_frustum_mesh(
    K,
    R_world2cam,
    t_world2cam,
    color=(1.0, 0.0, 0.0),
    alpha=1.0,
    near_z=0.1,
    far_z=0.5,
    scale_fov=1.0
):
    """
    Create a simple triangular mesh representing the camera frustum (pyramid).
    - K: Intrinsic (3x3), typically [fx, 0, cx; 0, fy, cy; 0,0,1]
    - R_world2cam, t_world2cam: The extrinsic that maps world->camera. 
      If you have camera1.R, camera1.t as world->cam, 
      then the camera center in world coords is C = -R^T * t.
    - color, alpha: color in [0..1], alpha in [0..1]
    - near_z, far_z: position of near-plane and far-plane in camera coords (z>0)
    - scale_fov: to scale the pyramid size if you want bigger/smaller frustum

    Returns:
        frustum_vertices: list of np.array([x,y,z,r,g,b,a]) shape=(N,)
        frustum_faces   : list of [v1,v2,v3] index triplets
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R_world2cam, torch.Tensor):
        R_world2cam = R_world2cam.detach().cpu().numpy()
    if isinstance(t_world2cam, torch.Tensor):
        t_world2cam = t_world2cam.detach().cpu().numpy()

    # 1) Invert (R,t) to get camera pose as cam->world
    R_cam2world = R_world2cam.T
    t_cam2world = -R_world2cam.T @ t_world2cam

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 2) Define corners in camera coords
    corners_cam = []
    for zval in [near_z, far_z]:
        # 4 corners in pixel coords (u,v)
        uvs = [
            (0, 0),
            (2*cx, 0),
            (2*cx, 2*cy),
            (0, 2*cy),
        ]
        for (u,v) in uvs:
            x = (u - cx)/fx * zval
            y = (v - cy)/fy * zval
            corners_cam.append(np.array([x, y, zval], dtype=np.float32))

    corners_cam = np.array(corners_cam)
    # optionally scale the FOV:
    corners_cam[:, :2] *= scale_fov

    # 3) Transform corners_cam to world coords
    corners_world = []
    for cc in corners_cam:
        cw = R_cam2world @ cc + t_cam2world
        corners_world.append(cw)

    corners_world = np.array(corners_world)  # shape (8,3)

    # Also define the camera center itself
    camera_center = t_cam2world  # shape (3,)

    def make_vert_xyzrgba(xyz):
        return np.array([xyz[0], xyz[1], xyz[2], color[0], color[1], color[2], alpha], dtype=np.float32)

    frustum_vertices = []
    frustum_vertices.append(make_vert_xyzrgba(camera_center))  # index 0
    for i in range(8):
        frustum_vertices.append(make_vert_xyzrgba(corners_world[i]))  # index i+1

    # 5) Build faces
    frustum_faces = []
    # center -> near-plane
    for i in range(4):
        i0 = 0
        i1 = 1 + i
        i2 = 1 + ((i+1) % 4)
        frustum_faces.append([i0, i1, i2])

    # center -> far-plane
    for i in range(4):
        i0 = 0
        i1 = 5 + i
        i2 = 5 + ((i+1) % 4)
        frustum_faces.append([i0, i1, i2])

    # near-plane ring => 2 triangles
    frustum_faces.append([1,2,3])
    frustum_faces.append([1,3,4])

    # far-plane ring => 2 triangles
    frustum_faces.append([5,6,7])
    frustum_faces.append([5,7,8])

    return frustum_vertices, frustum_faces


def save_ellipsoids_as_ply(points_3d, covariances_3d, colors_3d, alphas_3d, filename, 
                          n_theta=12, n_phi=12, camera_params=None, use_alpha=True):
    """
    Merge all ellipsoids' geometry and save as a single PLY.

    Args:
        points_3d: center points of ellipsoids, shape (N, 3)
        covariances_3d: covariance matrices, shape (N, 3, 3)
        colors_3d: RGB colors, shape (N, 3) in range [0..1]
        alphas_3d: alpha values, shape (N) in range [0..1]
        filename: output ply path
        n_theta, n_phi: ellipsoid sampling resolution
        camera_params: list of (R, t) for camera poses to visualize as frustums
        use_alpha: whether to include alpha channel in PLY
    """
    all_vertices = []
    all_faces = []

    # Process ellipsoids
    for i in range(points_3d.shape[0]):
        center = points_3d[i]
        Sigma_3 = covariances_3d[i] if covariances_3d is not None else np.eye(3)
        c3 = colors_3d[i] if colors_3d is not None else np.array([0.8, 0.2, 0.3])
        a3 = alphas_3d[i] if alphas_3d is not None else 0.9

        raw_vertices, faces = sample_ellipsoid_vertices_and_faces(Sigma_3, center, n_theta, n_phi)
        extended_vertices = []
        for vert in raw_vertices:
            combo = np.concatenate([vert, c3, [a3]])
            extended_vertices.append(combo)

        all_vertices.append(extended_vertices)
        all_faces.append(faces)

    # Add camera frustums if provided
    if camera_params is not None:
        K_dummy = np.array([
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0]
        ])
        
        for i, (R, t) in enumerate(camera_params):
            # Different color for each camera
            camera_color = [(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)][i % 3]
            
            camera_verts, camera_faces = create_camera_frustum_mesh(
                K=K_dummy,
                R_world2cam=R,
                t_world2cam=t,
                color=camera_color,
                alpha=1.0,
                near_z=0.1,
                far_z=0.5,
                scale_fov=100.0
            )
            
            all_vertices.append(camera_verts)
            all_faces.append(camera_faces)

    # Merge geometry for PLY output
    merged_vertices = []
    merged_faces = []
    v_offset = 0

    for vertices, faces in zip(all_vertices, all_faces):
        for v in vertices:
            merged_vertices.append(v)
        for f in faces:
            merged_faces.append([f[0] + v_offset, f[1] + v_offset, f[2] + v_offset])
        v_offset += len(vertices)

    # Write PLY file
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(merged_vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        if use_alpha:
            f.write('property uchar alpha\n')
        f.write(f'element face {len(merged_faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')

        for mv in merged_vertices:
            x, y, z = mv[0], mv[1], mv[2]
            r_f, g_f, b_f = mv[3], mv[4], mv[5]
            r = int(np.clip(r_f * 255, 0, 255))
            g = int(np.clip(g_f * 255, 0, 255))
            b = int(np.clip(b_f * 255, 0, 255))

            if use_alpha:
                a_f = mv[6]
                a = int(np.clip(a_f * 255, 0, 255))
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b} {a}\n")
            else:
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")

        for face in merged_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved {points_3d.shape[0]} ellipsoids to {filename}")

def save_point_cloud_as_ply(points, filename):
    """
    Save 3D points as a PLY file.

    Args:
        points (np.ndarray): shape (N,3).
        filename (str): Output PLY file path.
    """
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved {points.shape[0]} points to {filename}")