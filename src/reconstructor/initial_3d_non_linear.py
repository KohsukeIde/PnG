import numpy as np
import cv2
from scipy.optimize import least_squares
from src.primitive.twod_gaussians_rs import TwoDGaussians


def quaternion_to_rotation(q):
    """
    Convert a quaternion q = [qw, qx, qy, qz] into a 3x3 rotation matrix.
    Ensures R is orthonormal. 
    """
    qw, qx, qy, qz = q
    norm_q = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if norm_q < 1e-12:
        # fallback to identity if zero quaternion
        return np.eye(3)
    # Normalize
    qw, qx, qy, qz = qw/norm_q, qx/norm_q, qy/norm_q, qz/norm_q

    # Standard quaternion -> rotation formula
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ])
    return R


def build_covariance_3d(q, s):
    """
    Build a 3D covariance from rotation quaternion q and 3 scales s=[s1, s2, s3].
    Sigma_3 = R * diag(s1^2, s2^2, s3^2) * R^T
    """
    R = quaternion_to_rotation(q)
    S_diag = np.diag(s**2)
    Sigma_3 = R @ S_diag @ R.T
    return Sigma_3


def project_covariance_3d_to_2d(Sigma_3, point_3d, K, R_cam, t_cam):
    """
    Given a 3D covariance Sigma_3, project it to 2D:
    Sigma_2D = J * R_cam * Sigma_3 * R_cam^T * J^T

    J is the local Jacobian of the pinhole projection at the 3D point's camera coords.
    R_cam, t_cam define the transform from world to camera coordinates.
    """
    # 1) Transform 3D point to camera coords
    X_c = R_cam @ point_3d + t_cam
    X, Y, Z = X_c

    fx, fy = K[0,0], K[1,1]
    # We assume principal point is at (cx, cy), but for Jacobian we only need partial derivatives w.r.t. X, Y, Z

    # 2) Approximate Jacobian J for the projection:
    #     u = fx*(X/Z), v = fy*(Y/Z)
    # =>  J = [[fx/Z,      0,        -fx*X/(Z^2)],
    #          [    0,   fy/Z,       -fy*Y/(Z^2)]]
    J = np.array([
        [fx / Z,         0.0,       -fx * X / (Z**2)],
        [0.0,       fy / Z,         -fy * Y / (Z**2)]
    ])

    # 3) Construct "world->camera" rotation if needed
    #    In many treatments, Sigma_3 is already in world coords; 
    #    you might want Sigma_3 in camera coords: Sigma_cam = R_cam * Sigma_3 * R_cam^T
    #    But if you define Sigma_3 as in "camera coords" from the start, skip it. 
    #    We'll do the 'classical' approach:
    Sigma_cam = R_cam @ Sigma_3 @ R_cam.T

    # 4) final 2D covariance:
    #    Sigma_2D = J * Sigma_cam * J^T
    Sigma_2D_model = J @ Sigma_cam @ J.T
    return Sigma_2D_model


def single_view_cov_residual(params, point_3d, Sigma_2D_obs, K, R_cam, t_cam):
    """
    Residual function for a single camera view.

    params = [qw, qx, qy, qz, s1, s2, s3]
    Builds Sigma_3 from rotation+scale, 
    projects to 2D => Sigma_2D_model,
    returns difference from Sigma_2D_obs (some partial or full comparison).
    """
    qw, qx, qy, qz, s1, s2, s3 = params
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    s = np.array([s1, s2, s3], dtype=np.float64)

    Sigma_3 = build_covariance_3d(q, s)
    Sigma_2D_model = project_covariance_3d_to_2d(Sigma_3, point_3d, K, R_cam, t_cam)

    # Flatten difference for a least-squares residual:
    # We'll do top-left, top-right, bottom-left, bottom-right for the 2x2
    # or just the upper triangular. Here let's do the full 4 elements:
    diff = Sigma_2D_model - Sigma_2D_obs
    return diff.flatten()


def combined_two_view_cov_residual(params, point_3d, Sigma_2D_1_obs, Sigma_2D_2_obs,
                                   K1, R1, t1, K2, R2, t2):
    """
    Combined residual stacking for 2 camera views. 
    Summarizes how well the single param set [qw, qx, qy, qz, s1, s2, s3] 
    fits the 2D covariances from both cameras.
    """
    r1 = single_view_cov_residual(params, point_3d, Sigma_2D_1_obs, K1, R1, t1)
    r2 = single_view_cov_residual(params, point_3d, Sigma_2D_2_obs, K2, R2, t2)
    return np.concatenate([r1, r2])  # stack them for least_squares


class Initial3DReconstructor:
    def __init__(self, gaussians1, gaussians2, K1, K2, H):
        """
        Initialize the 3D reconstructor.

        Parameters:
        - gaussians1: TwoDGaussians instance (Gaussian distributions in image 1)
        - gaussians2: TwoDGaussians instance (Gaussian distributions in image 2)
        - K1, K2: Intrinsic camera matrices (3x3)
        - H: Homography matrix (3x3)
        """
        if not isinstance(K1, np.ndarray) or K1.shape != (3, 3):
            raise ValueError("K1 must be a 3x3 numpy array.")
        if not isinstance(K2, np.ndarray) or K2.shape != (3, 3):
            raise ValueError("K2 must be a 3x3 numpy array.")
        if not isinstance(H, np.ndarray) or H.shape != (3, 3):
            raise ValueError("H must be a 3x3 numpy array.")
        if not isinstance(gaussians1, TwoDGaussians):
            raise TypeError("gaussians1 must be an instance of TwoDGaussians.")
        if not isinstance(gaussians2, TwoDGaussians):
            raise TypeError("gaussians2 must be an instance of TwoDGaussians.")

        self.gaussians1 = gaussians1
        self.gaussians2 = gaussians2
        self.K1 = K1
        self.K2 = K2
        self.H = H

        self.P1 = None  # Projection matrix for camera 1
        self.P2 = None  # Projection matrix for camera 2
        self.points_3d = None  # 3D points from triangulation
        self.covariances_3d = None  # Covariance matrices for 3D Gaussians

        # Fix camera 1 at world coordinate origin
        self.R1 = np.eye(3)
        self.t1 = np.zeros(3)
        self.R2 = None
        self.t2 = None

    def compute_camera_matrices_from_homography(self):
        """
        Compute camera projection matrices P1 and P2 from the homography matrix.
        """
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(
            self.H, self.K1 @ self.K1.T
        )

        if retval == 0:
            raise ValueError("Homography decomposition failed.")

        selected = False
        for i in range(retval):
            R = rotations[i]
            t = translations[i].flatten()
            if t[2] > 0:
                self.R2 = R
                self.t2 = t
                selected = True
                break

        if not selected:
            self.R2 = rotations[0]
            self.t2 = translations[0].flatten()

        self.P1 = self.K1 @ np.hstack((self.R1, self.t1.reshape(3, 1)))
        self.P2 = self.K2 @ np.hstack((self.R2, self.t2.reshape(3, 1)))

    def triangulate_gaussian_centers(self, transport_matrix, threshold=1e-3):
        """
        Extract correspondences from the transport_matrix and triangulate each pair.
        """
        if self.P1 is None or self.P2 is None:
            raise ValueError("Camera matrices must be computed before triangulation.")

        centers1 = self.gaussians1.means
        centers2 = self.gaussians2.means
        k1 = centers1.shape[0]
        k2 = centers2.shape[0]

        correspondences = []
        for i in range(k1):
            row = transport_matrix[i, :]
            matched_indices = np.where(row > threshold)[0]
            for j in matched_indices:
                x1 = centers1[i]
                x2 = centers2[j]

                x1_h = np.array([x1[0], x1[1], 1.0])
                x2_h = np.array([x2[0], x2[1], 1.0])

                A = np.zeros((4, 4))
                A[0] = x1_h[0]*self.P1[2] - self.P1[0]
                A[1] = x1_h[1]*self.P1[2] - self.P1[1]
                A[2] = x2_h[0]*self.P2[2] - self.P2[0]
                A[3] = x2_h[1]*self.P2[2] - self.P2[1]

                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]
                correspondences.append(X[:3])

        if len(correspondences) == 0:
            print("No correspondences found above the threshold.")
            self.points_3d = np.zeros((0,3))
        else:
            self.points_3d = np.array(correspondences)

    def _compute_jacobian(self, point_3d, K, R, t):
        """
        Compute Jacobian for 2D projection (for reference).
        In the rotation-scale approach, we build J manually in project_covariance_3d_to_2d.
        """
        X_c = R @ point_3d + t
        X, Y, Z = X_c
        fx, fy = K[0,0], K[1,1]

        dU_dXc = fx / Z
        dU_dYc = 0
        dU_dZc = -fx * X / (Z**2)
        dV_dXc = 0
        dV_dYc = fy / Z
        dV_dZc = -fy * Y / (Z**2)

        J_image_wrt_Xc = np.array([
            [dU_dXc, dU_dYc, dU_dZc],
            [dV_dXc, dV_dYc, dV_dZc]
        ])

        J = J_image_wrt_Xc @ R
        return J

    def compute_3d_gaussian_covariances(self):
        """
        Non-linear approach: 
        For each 3D point, we find a rotation quaternion + scale = (qw,qx,qy,qz, s1,s2,s3)
        that best reproduces the observed 2D covariances from camera 1 and camera 2.
        """
        if self.P1 is None or self.P2 is None:
            raise ValueError("Camera matrices must be computed before computing covariances.")
        if self.points_3d is None:
            raise ValueError("3D points must be computed before computing covariances.")

        k = self.points_3d.shape[0]
        self.covariances_3d = np.zeros((k, 3, 3))

        # Decompose P1 => R1, t1
        R1 = np.eye(3)
        t1 = np.zeros(3)

        # Decompose P2 => R2, t2
        M = self.P2[:, :3]
        U, S_, Vt = np.linalg.svd(M)
        R2 = U @ Vt
        t2 = np.linalg.inv(self.K2) @ self.P2[:, 3]

        for i in range(k):
            point_3d = self.points_3d[i]

            # Observed 2D covariances
            Sigma_2D_1_obs = self.gaussians1.covs[i]
            Sigma_2D_2_obs = self.gaussians2.covs[i]

            # We'll define a local cost function that sums camera1 + camera2 residuals
            def two_view_resid(params):
                return combined_two_view_cov_residual(
                    params,
                    point_3d,
                    Sigma_2D_1_obs,
                    Sigma_2D_2_obs,
                    self.K1, R1, t1,
                    self.K2, R2, t2
                )

            # Initial guess => identity rotation [1,0,0,0], scale ~ [5,5,5]
            init_params = np.array([1.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0], dtype=np.float64)

            result = least_squares(
                two_view_resid,
                x0=init_params,
                method='lm',
                max_nfev=1000
            )
            final_params = result.x
            q = final_params[:4]
            s = final_params[4:]
            
            if result.status == 1:
                print("最適化が収束しました。")
            elif result.status == 2:
                print("関数評価回数の上限に達しました。")
            else:
                print("最適化が失敗しました。")

            Sigma_3 = build_covariance_3d(q, s)
            self.covariances_3d[i] = Sigma_3
