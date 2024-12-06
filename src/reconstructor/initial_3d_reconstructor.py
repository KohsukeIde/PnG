import numpy as np
import cv2
from src.primitive.twod_gaussians_rs import TwoDGaussians

class Initial3DReconstructor:
    def __init__(self, gaussians1, gaussians2, K1, K2, H):
        """
        Initialize the 3D reconstructor.

        Parameters:
        - gaussians1: TwoDGaussians instance (Gaussian distributions in image 1)
        - gaussians2: TwoDGaussians instance (Gaussian distributions in image 2)
        - K1: np.ndarray (3x3) - Intrinsic camera matrix for camera 1
        - K2: np.ndarray (3x3) - Intrinsic camera matrix for camera 2
        - H: np.ndarray (3x3) - Homography matrix
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
        retval, rotations, translations, normals = cv2.decomposeHomographyMat(self.H, self.K1 @ self.K1.T)

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
        Extract M:M correspondences using transport matrix and triangulate each corresponding pair.
        
        Parameters:
        - transport_matrix: np.ndarray (k1, k2) - Optimal transport plan matrix
        - threshold: float - Transport amount threshold. Pairs exceeding this are considered matches

        Note: This implementation uses a simple threshold-based correspondence extraction.
        For top M matches, use argsort to get the top M pairs.
        """
        if self.P1 is None or self.P2 is None:
            raise ValueError("Camera matrices must be computed before triangulation.")

        centers1 = self.gaussians1.means
        centers2 = self.gaussians2.means
        k1 = centers1.shape[0]
        k2 = centers2.shape[0]

        # 対応点ペアを抽出
        correspondences = []
        for i in range(k1):
            row = transport_matrix[i, :]
            # threshold以上の輸送量があるjを対応先とする
            matched_indices = np.where(row > threshold)[0]
            for j in matched_indices:
                x1 = centers1[i]
                x2 = centers2[j]

                # 三角測量
                x1_h = np.array([x1[0], x1[1], 1.0])
                x2_h = np.array([x2[0], x2[1], 1.0])

                A = np.zeros((4,4))
                A[0] = x1_h[0]*self.P1[2] - self.P1[0]
                A[1] = x1_h[1]*self.P1[2] - self.P1[1]
                A[2] = x2_h[0]*self.P2[2] - self.P2[0]
                A[3] = x2_h[1]*self.P2[2] - self.P2[1]

                _,_,Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]
                correspondences.append(X[:3])

        if len(correspondences) == 0:
            # 対応点が全く得られない場合はエラーにするか、警告を出す
            print("No correspondences found above the threshold.")
            self.points_3d = np.zeros((0,3))
        else:
            self.points_3d = np.array(correspondences)

    def _compute_jacobian(self, point_3d, K, R, t):
        """
        Compute Jacobian based on pinhole camera model.

        Parameters:
        - point_3d: [X, Y, Z] in world coordinates
        - K: Intrinsic camera matrix
        - R, t: Extrinsic parameters (world to camera transformation)

        Returns:
        - Jacobian matrix (2x3)
        """
        # カメラ座標系へ変換
        X_c = R @ point_3d + t
        X, Y, Z = X_c

        fx = K[0, 0]
        fy = K[1, 1]
        # cx, cyはヤコビアンには影響無し(u,vへの微分なので)

        # カメラ座標系での微分
        # u = fx(X/Z) + cx, v = fy(Y/Z) + cy
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

        # ワールド座標系への変換
        # x_c = R x_w + t
        # dx_c/dx_w = R
        # よってJ = J_image_wrt_Xc * R
        J = J_image_wrt_Xc @ R

        return J

    def compute_3d_gaussian_covariances(self):
        """
        Compute 3D Gaussian covariances from 2D projections and camera parameters.
        Must be called after computing camera matrices and 3D points.
        """
        if self.P1 is None or self.P2 is None:
            raise ValueError("Camera matrices must be computed before computing covariances.")
        if self.points_3d is None:
            raise ValueError("3D points must be computed before computing covariances.")

        k = self.points_3d.shape[0]
        self.covariances_3d = np.zeros((k, 3, 3))

        # Decompose P1 and P2 to get R1, t1 and R2, t2
        # Since K1, K2 are known, we can get R1 = I, t1 = 0 (if we assume P1 is canonical)
        # For P2, we must do a proper decomposition. Or if we assume from H and solve like in code:
        # Let's assume we already have R1=I, t1=0 by definition of P1.
        R1 = np.eye(3)
        t1 = np.zeros(3)
        # For P2:
        # Extract R2, t2 from P2. Something like:
        M = self.P2[:, :3]
        # Ensure M is orthonormal by SVD if needed
        U, S, Vt = np.linalg.svd(M)
        R2 = U @ Vt
        # t2 = ...
        # Actually t2 = np.linalg.inv(self.K2) @ self.P2[:,3], but since we only need rotation for covariance,
        # translation does not affect the differential at that small scale. However, for a correct solution:
        t2 = np.linalg.inv(self.K2) @ self.P2[:,3]

        for i in range(k):
            point_3d = self.points_3d[i]

            # 2D covariance from each camera
            Sigma_2D_1 = self.gaussians1.covs[i]
            Sigma_2D_2 = self.gaussians2.covs[i]

            # Compute J1, J2
            J1 = self._compute_jacobian(point_3d, self.K1, R1, t1)
            J2 = self._compute_jacobian(point_3d, self.K2, R2, t2)

            # Flatten Sigma_3 into s:
            # s = [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]^T
            # For each camera i, we have:
            # Σ_2D_i = J_i R_i Σ_3 R_i^T J_i^T
            # Extract upper-triangular elements from Σ_2D_i (a_i, b_i, c_i)
            a1 = Sigma_2D_1[0, 0]
            b1 = Sigma_2D_1[0, 1] # = Sigma_2D_1[1, 0]
            c1 = Sigma_2D_1[1, 1]

            a2 = Sigma_2D_2[0, 0]
            b2 = Sigma_2D_2[0, 1]
            c2 = Sigma_2D_2[1, 1]

            # Now we must express [a1,b1,c1,a2,b2,c2]^T = M * s
            # Construct M and v for each i. Actually we must solve for each Gaussian i:
            # M is 6x6, v is 6x1

            # Derivation of M is involved. For each Σ_2D_i = J_i R_i Σ_3 R_i^T J_i^T:
            # Let’s define a function to construct M_i, v_i from J_i, R_i:
            M_i, v_i = self._construct_linear_system(J1, R1, a1, b1, c1, J2, R2, a2, b2, c2)

            # Solve linear system M_i s = v_i
            # s = np.linalg.solve(M_i, v_i)
            
            try:
                s, residuals, rank, singular = np.linalg.lstsq(M_i, v_i, rcond=None)
            except np.linalg.LinAlgError as e:
                raise ValueError(f"Linear system solving failed for point {i}: {e}")


            # Reconstruct Σ_3 from s
            # s = [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
            Sigma_3 = np.array([
                [s[0], s[3], s[4]],
                [s[3], s[1], s[5]],
                [s[4], s[5], s[2]]
            ])

            # Ensure symmetry
            Sigma_3 = (Sigma_3 + Sigma_3.T) / 2
            self.covariances_3d[i] = Sigma_3

    def _construct_linear_system(self, J1, R1, a1, b1, c1, J2, R2, a2, b2, c2):
        """
        Construct linear system for 3D covariance matrix from 2D covariances from two cameras.
        
        Parameters:
            J1 (np.ndarray): Jacobian matrix for camera 1 (2x3)
            R1 (np.ndarray): Rotation matrix for camera 1 (3x3)
            a1 (float): Sigma_2D[0,0] for camera 1
            b1 (float): Sigma_2D[0,1] for camera 1
            c1 (float): Sigma_2D[1,1] for camera 1
            J2 (np.ndarray): Jacobian matrix for camera 2 (2x3)
            R2 (np.ndarray): Rotation matrix for camera 2 (3x3)
            a2 (float): Sigma_2D[0,0] for camera 2
            b2 (float): Sigma_2D[0,1] for camera 2
            c2 (float): Sigma_2D[1,1] for camera 2
        
        Returns:
            M (np.ndarray): Coefficient matrix for linear system (6x6)
            v (np.ndarray): Constant vector for linear system (6,)
        """
        M = np.zeros((6,6))
        
        # カメラ1の係数
        M[0,0] = J1[0,0]**2
        M[0,1] = J1[0,1]**2
        M[0,2] = J1[0,2]**2
        M[0,3] = 2 * J1[0,0] * J1[0,1]
        M[0,4] = 2 * J1[0,0] * J1[0,2]
        M[0,5] = 2 * J1[0,1] * J1[0,2]
        
        M[1,0] = 2 * J1[0,0] * J1[1,0]
        M[1,1] = 2 * J1[0,1] * J1[1,1]
        M[1,2] = 2 * J1[0,2] * J1[1,2]
        M[1,3] = J1[0,0] * J1[1,1] + J1[0,1] * J1[1,0]
        M[1,4] = J1[0,0] * J1[1,2] + J1[0,2] * J1[1,0]
        M[1,5] = J1[0,1] * J1[1,2] + J1[0,2] * J1[1,1]
        
        M[2,0] = J1[1,0]**2
        M[2,1] = J1[1,1]**2
        M[2,2] = J1[1,2]**2
        M[2,3] = 2 * J1[1,0] * J1[1,1]
        M[2,4] = 2 * J1[1,0] * J1[1,2]
        M[2,5] = 2 * J1[1,1] * J1[1,2]
        
        # カメラ2の係数
        M[3,0] = J2[0,0]**2
        M[3,1] = J2[0,1]**2
        M[3,2] = J2[0,2]**2
        M[3,3] = 2 * J2[0,0] * J2[0,1]
        M[3,4] = 2 * J2[0,0] * J2[0,2]
        M[3,5] = 2 * J2[0,1] * J2[0,2]
        
        M[4,0] = 2 * J2[0,0] * J2[1,0]
        M[4,1] = 2 * J2[0,1] * J2[1,1]
        M[4,2] = 2 * J2[0,2] * J2[1,2]
        M[4,3] = J2[0,0] * J2[1,1] + J2[0,1] * J2[1,0]
        M[4,4] = J2[0,0] * J2[1,2] + J2[0,2] * J2[1,0]
        M[4,5] = J2[0,1] * J2[1,2] + J2[0,2] * J2[1,1]
        
        M[5,0] = J2[1,0]**2
        M[5,1] = J2[1,1]**2
        M[5,2] = J2[1,2]**2
        M[5,3] = 2 * J2[1,0] * J2[1,1]
        M[5,4] = 2 * J2[1,0] * J2[1,2]
        M[5,5] = 2 * J2[1,1] * J2[1,2]
        
        v = np.array([a1, b1, c1, a2, b2, c2])
        
        return M, v

