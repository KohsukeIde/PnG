from typing import List, Optional, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from src.primitive.twod_gaussians_rs import TwoDGaussians

import sys

def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion [qw, qx, qy, qz] into a 3x3 rotation matrix.

    Args:
        q (np.ndarray): A 4-element array representing the quaternion (qw, qx, qy, qz).

    Returns:
        np.ndarray: A 3x3 orthonormal rotation matrix.
    """
    qw, qx, qy, qz = q
    norm_q = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    # If the quaternion is too small, return the identity matrix → to avoid rounding error/assume that there is no rotation.
    if norm_q < 1e-12:
        print("Quaternion norm is too small ({}), returning identity matrix.".format(norm_q))
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q
    r_mat = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )
    return r_mat


# def build_covariance_3d(q: np.ndarray, s: np.ndarray) -> np.ndarray:
#     """Build a 3D covariance from quaternion q and scales s=[s1, s2, s3].

#     Sigma_3 = R * diag(s^2) * R^T.

#     Args:
#         q (np.ndarray): Quaternion [qw, qx, qy, qz].
#         s (np.ndarray): Scales [s1, s2, s3].

#     Returns:
#         np.ndarray: The 3D covariance matrix (3x3).
#     """
#     r_mat = quaternion_to_rotation(q)
#     s_diag = np.diag(s**2)
#     sigma_3 = r_mat @ s_diag @ r_mat.T
#     return np.array(sigma_3, dtype=float)

def build_covariance_3d(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Build a 3D covariance from quaternion q and scales s=[s1, s2, s3].

    We now explicitly write Sigma as:
        Sigma = R * S * S^T * R^T
    where:
        - R is the 3x3 rotation matrix converted from a quaternion [qw, qx, qy, qz].
        - S = diag(s) is a 3x3 diagonal matrix with the scale factors s1, s2, s3.
          (Hence, S * S^T = diag(s^2).)

    Args:
        q (np.ndarray): Quaternion [qw, qx, qy, qz].
        s (np.ndarray): Scales [s1, s2, s3], each s_i >= 0.

    Returns:
        np.ndarray: The 3D covariance matrix (3x3).
    """
    r_mat = quaternion_to_rotation(q)

    s_diag = np.diag(s)

    # Sigma = R * S * S^T * R^T
    sigma_3 = r_mat @ s_diag @ s_diag.T @ r_mat.T

    return sigma_3


def project_covariance_3d_to_2d(
    sigma_3: np.ndarray,
    point_3d: np.ndarray,
    k: np.ndarray,
    r_cam: np.ndarray,
    t_cam: np.ndarray,
) -> np.ndarray:
    """Project a 3D covariance sigma_3 to 2D using the local Jacobian approximation.
    Given a 3D covariance Sigma_3, project it to 2D: Sigma_2D = J * R_cam * Sigma_3 * R_cam^T * J^T.

    J is the local Jacobian of the pinhole projection at the 3D point's camera coords.
    R_cam, t_cam define the transform from world to camera coordinates.

    Args:
        sigma_3 (np.ndarray): 3D covariance in world coords (3x3).
        point_3d (np.ndarray): 3D point in world coordinates (3,).
        k (np.ndarray): Intrinsic camera matrix (3x3).
        r_cam (np.ndarray): Camera rotation (world->camera) (3x3).
        t_cam (np.ndarray): Camera translation (3,).

    Returns:
        np.ndarray: Resulting 2D covariance (2x2).
    """
    # Project the 3D point to camera coordinates.
    x_c = r_cam @ point_3d + t_cam
    # Extract the x, y, z coordinates from the camera coordinates.
    x_val, y_val, z_val = x_c

    fx, fy = k[0, 0], k[1, 1]
    # Compute the Jacobian for the pinhole projection.
    j_mat = np.array(
        [
            [fx / z_val, 0.0, -fx * x_val / (z_val**2)],
            [0.0, fy / z_val, -fy * y_val / (z_val**2)],
        ],
        dtype=np.float64,
    )
    # Project the 3D covariance to camera coordinates.
    sigma_cam = r_cam @ sigma_3 @ r_cam.T
    # Propagate 3D covariance to 2D using the local Jacobian approximation.
    sigma_2d_model = j_mat @ sigma_cam @ j_mat.T
    return np.array(sigma_2d_model, dtype=float)


def single_view_cov_residual(
    params: np.ndarray,
    point_3d: np.ndarray,
    sigma_2d_obs: np.ndarray,
    k: np.ndarray,
    r_cam: np.ndarray,
    t_cam: np.ndarray,
) -> np.ndarray:
    """Residual function for a single camera view.

    Args:
        params (np.ndarray): [qw, qx, qy, qz, s1, s2, s3].
        point_3d (np.ndarray): 3D point in world coordinates.
        sigma_2d_obs (np.ndarray): Observed 2D covariance (2x2).
        k (np.ndarray): Intrinsic camera matrix (3x3).
        r_cam (np.ndarray): Camera rotation (3x3).
        t_cam (np.ndarray): Camera translation (3,).

    Returns:
        np.ndarray: Flattened difference (4,) between the modeled 2D covariance and observed 2D covariance.
    """
    if hasattr(sigma_2d_obs, "detach"):
        sigma_2d_obs = sigma_2d_obs.detach().cpu().numpy()

    qw, qx, qy, qz, s1, s2, s3 = params
    q_arr = np.array([qw, qx, qy, qz], dtype=np.float64)
    s_arr = np.array([s1, s2, s3], dtype=np.float64)

    sigma_3 = build_covariance_3d(q_arr, s_arr)
    sigma_2d_model = project_covariance_3d_to_2d(sigma_3, point_3d, k, r_cam, t_cam)
    diff = sigma_2d_model - sigma_2d_obs
    return np.array(diff.flatten(), dtype=float)


class Initial3DReconstructor:
    """Initial 3D reconstruction for Gaussian distributions from two images."""

    def __init__(
        self,
        gaussians1: TwoDGaussians,
        gaussians2: TwoDGaussians,
        k1: np.ndarray,
        k2: np.ndarray,
        h: np.ndarray,
    ) -> None:
        """Initialize the 3D reconstructor.

        Args:
            gaussians1 (TwoDGaussians): Gaussian distributions in image 1.
            gaussians2 (TwoDGaussians): Gaussian distributions in image 2.
            k1 (np.ndarray): 3x3 intrinsic camera matrix for camera 1.
            k2 (np.ndarray): 3x3 intrinsic camera matrix for camera 2.
            h (np.ndarray): 3x3 homography matrix from image 1 to image 2.
        """
        if not isinstance(k1, np.ndarray) or k1.shape != (3, 3):
            raise ValueError("k1 must be a 3x3 numpy array.")
        if not isinstance(k2, np.ndarray) or k2.shape != (3, 3):
            raise ValueError("k2 must be a 3x3 numpy array.")
        if not isinstance(h, np.ndarray) or h.shape != (3, 3):
            raise ValueError("h must be a 3x3 numpy array.")
        if not isinstance(gaussians1, TwoDGaussians):
            raise TypeError("gaussians1 must be an instance of TwoDGaussians.")
        if not isinstance(gaussians2, TwoDGaussians):
            raise TypeError("gaussians2 must be an instance of TwoDGaussians.")

        self.gaussians1 = gaussians1
        self.gaussians2 = gaussians2
        self.k1 = k1
        self.k2 = k2
        self.h = h

        self.p1: Optional[np.ndarray] = None
        self.p2: Optional[np.ndarray] = None
        self.points_3d: Optional[np.ndarray] = None
        self.covariances_3d: Optional[np.ndarray] = None

        self.r1: np.ndarray = np.eye(3)
        self.t1: np.ndarray = np.zeros(3)
        self.r2: Optional[np.ndarray] = None
        self.t2: Optional[np.ndarray] = None

        self.match_pairs: Optional[List[Tuple[int, int]]] = None

    def compute_camera_matrices_from_homography(self) -> None:
        """Compute camera projection matrices p1 and p2 from the homography matrix.

        Note:
            If you plan to use a Fundamental matrix approach (rather than a plane-induced homography),
            you can skip this method and set p1, p2 explicitly with your own extrinsics.
        """
        decomp = cv2.decomposeHomographyMat(self.h, self.k1)

        if decomp is None:
            raise ValueError("Homography decomposition returned None.")

        retval, rotations, translations, normals = decomp
        if retval == 0:
            raise ValueError("Homography decomposition failed.")

        selected = False
        for i in range(retval):
            r_candidate = rotations[i]
            if not isinstance(r_candidate, np.ndarray):
                r_candidate = np.array(r_candidate, dtype=float)

            t_candidate = translations[i]
            if not isinstance(t_candidate, np.ndarray):
                t_candidate = np.array(t_candidate, dtype=float)
            t_candidate = t_candidate.flatten()

            # Check if the translation is in front of the camera (z-coordinate is positive in camera coords).
            if t_candidate[2] > 0:
                self.r2 = r_candidate
                self.t2 = t_candidate
                selected = True
                break
        
        # If no valid translation is found, use the first solution.
        if not selected:
            r_candidate = rotations[0]
            if not isinstance(r_candidate, np.ndarray):
                r_candidate = np.array(r_candidate, dtype=float)

            t_candidate = translations[0]
            if not isinstance(t_candidate, np.ndarray):
                t_candidate = np.array(t_candidate, dtype=float)
            t_candidate = t_candidate.flatten()

            self.r2 = r_candidate
            self.t2 = t_candidate

        if self.r2 is None or self.t2 is None:
            raise ValueError("Could not find valid decomposition with t[2] > 0.")

        self.p1 = self.k1 @ np.hstack((self.r1, self.t1.reshape(3, 1)))
        self.p2 = self.k2 @ np.hstack((self.r2, self.t2.reshape(3, 1)))
        
    def recover_extrinsics_from_fundamental(f, k1, k2, pts1=None, pts2=None):
        """
        Given a fundamental matrix F and camera intrinsics k1, k2, 
        recover extrinsics (R,t) using an essential matrix approach.
        Optionally uses actual correspondences (pts1, pts2) to pick the correct solution.
        
        Args:
            f (np.ndarray): shape (3,3) fundamental matrix
            k1 (np.ndarray): shape (3,3) camera1 intrinsics
            k2 (np.ndarray): shape (3,3) camera2 intrinsics
            pts1 (Optional[np.ndarray]): shape (N,2) correspondences in image1 (pixel coords)
            pts2 (Optional[np.ndarray]): shape (N,2) correspondences in image2 (pixel coords)
        Returns:
            (R, t): a tuple of rotation(3x3) and translation(3,) 
                    up to scale (||t|| is unknown if using decomposeEssentialMat).
        """
        e = k2.T @ f @ k1  # E = K2^T * F * K1

        # Enforce rank2 if needed
        u, s, vt = np.linalg.svd(e)
        s[-1] = 0.0
        e = u @ np.diag(s) @ vt

        # If we have actual matched points, we can use recoverPose for a single solution
        if pts1 is not None and pts2 is not None:
            # We assume pts1, pts2 are pixel coords -> convert to normalized for recoverPose
            # (OpenCV can do that inside if we pass the intrinsics, but let's be consistent)
            # shape(N,1,2) needed
            pts1_undist = cv2.undistortPoints(
                pts1.reshape(-1,1,2), cameraMatrix=k1, distCoeffs=None
            )
            pts2_undist = cv2.undistortPoints(
                pts2.reshape(-1,1,2), cameraMatrix=k2, distCoeffs=None
            )
            # recoverPose => R, t
            ret_val, R_est, t_est, mask = cv2.recoverPose(
                e,
                pts1_undist,
                pts2_undist,
            )
            return R_est, t_est.flatten()
        else:
            # Without correspondences, we can only do decomposeEssentialMat => multiple solutions
            R1, R2, t = cv2.decomposeEssentialMat(e)
            # Usually, we pick the combination => (R1, t) or (R1, -t) or (R2, t) or (R2, -t)
            # Then we would do a cheirality check with some 2D-2D matches if we had them.
            # For now, let's just return the first possibility
            return R1, t.flatten()

    def set_camera_matrices_explicitly(
            self, r1: np.ndarray, t1: np.ndarray, r2: np.ndarray, t2: np.ndarray
        ) -> None:
            """Set the camera matrices p1 and p2 explicitly from external parameters.

            Args:
                r1 (np.ndarray): Rotation (3x3) for camera1 (world->camera1).
                t1 (np.ndarray): Translation (3,) for camera1.
                r2 (np.ndarray): Rotation (3x3) for camera2 (world->camera2).
                t2 (np.ndarray): Translation (3,) for camera2.
            """
            if r1.shape != (3, 3) or r2.shape != (3, 3):
                raise ValueError("r1 and r2 must be 3x3 rotation matrices.")
            if t1.shape != (3,) or t2.shape != (3,):
                raise ValueError("t1 and t2 must be shape (3,) translation vectors.")

            self.r1 = r1
            self.t1 = t1
            self.r2 = r2
            self.t2 = t2

            self.p1 = self.k1 @ np.hstack((self.r1, self.t1.reshape(3, 1)))
            self.p2 = self.k2 @ np.hstack((self.r2, self.t2.reshape(3, 1)))

    def triangulate_gaussian_centers(
        self, transport_matrix: np.ndarray, threshold: float = 1e-3, top_k: int = 100
    ) -> None:
        """Triangulate 3D points by extracting correspondences from the transport matrix."""
        if self.p1 is None or self.p2 is None:
            raise ValueError("Camera matrices must be computed before triangulation.")

        centers1 = self.gaussians1.means # (k1_num, 2)
        centers2 = self.gaussians2.means # (k2_num, 2)

        if hasattr(centers1, "detach"):
            centers1 = centers1.detach().cpu().numpy()
        if hasattr(centers2, "detach"):
            centers2 = centers2.detach().cpu().numpy()

        k1_num = centers1.shape[0] 
        k2_num = centers2.shape[0]

        print(f"transport matrix shape {transport_matrix.shape}")
        # Flatten the transport matrix.
        t_flat = transport_matrix.ravel()
        # Get all indices.
        all_indices = np.arange(t_flat.size)
        # Filter indices with values >= threshold.
        mask = t_flat >= threshold
        valid_indices = all_indices[mask]
        # If no valid indices are found, return empty 3D points and match pairs.
        if len(valid_indices) == 0:
            print(f"No transport values >= {threshold}")
            self.points_3d = np.zeros((0, 3), dtype=np.float64)
            self.match_pairs = []
            return

        # Get valid transport values.
        valid_tvals = t_flat[mask]
        # Sort indices in descending (largest to smallest) order.
        sort_desc = np.argsort(-valid_tvals)
        # Limit the number of top indices to top_k.
        top_k_limited = min(top_k, len(valid_tvals))

        # best_indices are the indices of the top k transport values.
        best_indices = valid_indices[sort_desc[:top_k_limited]]
        # Unravel the indices to get the corresponding 2D Gaussian indices.
        i_coords, j_coords = np.unravel_index(best_indices, (k1_num, k2_num))
        
        
        correspondences = []

        for idx in range(top_k_limited):
            # (x,y) in image 1 and image 2.
            i_val = i_coords[idx]
            j_val = j_coords[idx]
            
            # Convert to homogeneous coordinates (x,y,1) in image 1 and image 2.
            x1_h = np.array([centers1[i_val, 0], centers1[i_val, 1], 1.0], dtype=float)
            x2_h = np.array([centers2[j_val, 0], centers2[j_val, 1], 1.0], dtype=float)

            print(f"p1 {self.p1}")
            print(f"p2 {self.p2}")

            print(f"p1 shape {self.p1.shape}")
            print(f"p2 shape {self.p2.shape}")
            print(f"p1[2] {self.p1[2]}")
            print(f"p2[2] {self.p2[2]}")
        
            # Direct linear triangulation.
            a_mat = np.zeros((4, 4), dtype=float)
            a_mat[0, :] = x1_h[0] * self.p1[2, :] - self.p1[0, :]
            a_mat[1, :] = x1_h[1] * self.p1[2, :] - self.p1[1, :]
            a_mat[2, :] = x2_h[0] * self.p2[2, :] - self.p2[0, :]
            a_mat[3, :] = x2_h[1] * self.p2[2, :] - self.p2[1, :]
            
            print(f"a_mat shape {a_mat.shape}")
            # sys.exit()


            _, _, vt = np.linalg.svd(a_mat)
            # Get the last singular vector.
            x_val = vt[-1]
            # Homogeneous coordinates (X,Y,Z,W) -> (X/W, Y/W, Z/W).
            x_val /= x_val[3]
            # Extract the 3D point (X/W, Y/W, Z/W).
            point_3d = x_val[:3]
            # Append the correspondence.    
            correspondences.append((point_3d, i_val, j_val))

        if len(correspondences) == 0:
            print("No valid correspondences after filtering & top_k.")
            self.points_3d = np.zeros((0, 3), dtype=float)
            self.match_pairs = []
        else:
            self.points_3d = np.array([c[0] for c in correspondences], dtype=np.float64)
            self.match_pairs = [(c[1], c[2]) for c in correspondences]

        print(
            f"Selected {len(correspondences)} 3D points (threshold={threshold}, top_k={top_k})."
        )

    def compute_3d_gaussian_covariances(
        self, lambda_volume: float = 1.0, target_volume: float = 1.0, n_jobs: int = -1
    ) -> None:
        """Compute 3D Gaussian covariances via non-linear optimization with volume prior."""
        # Ensure that camera matrices and 3D points are available before proceeding
        if self.p1 is None or self.p2 is None:
            raise ValueError(
                "Camera matrices must be computed before computing covariances."
            )
        if self.points_3d is None:
            raise ValueError("3D points must be computed before computing covariances.")
        if self.match_pairs is None:
            raise ValueError(
                "match_pairs not found. Did you call triangulate_gaussian_centers first?"
            )
        if len(self.points_3d) == 0:
            raise ValueError("No 3D points available for computing covariances.")

        # Initialize an array to store the 3D covariance matrices for each point
        num_3d = self.points_3d.shape[0]
        self.covariances_3d = np.zeros((num_3d, 3, 3), dtype=np.float64)

        # Set up local camera parameters for the first camera (identity rotation and zero translation)
        r1_local = np.eye(3, dtype=float)
        t1_local = np.zeros(3, dtype=float)

        # Ensure the second camera matrix is not None
        assert self.p2 is not None, "p2 must not be None."

        # Decompose the second camera matrix to extract rotation and translation
        m_mat = self.p2[:, :3]
        u_mat, s_vals, vt_mat = np.linalg.svd(m_mat)
        r2_local = u_mat @ vt_mat
        t2_local = np.linalg.inv(self.k2) @ self.p2[:, 3]

        def solve_cov_for_gaussian(idx: int) -> np.ndarray:
            # Add assertions to satisfy type checker
            assert self.points_3d is not None, "points_3d should not be None"
            assert self.match_pairs is not None, "match_pairs should not be None"

            # Get the 3D point and corresponding 2D Gaussian indices
            point_3d_ = self.points_3d[idx]
            i_img1_, j_img2_ = self.match_pairs[idx]

            # Retrieve observed 2D covariance matrices for the Gaussian
            sigma_2d_1_obs = self.gaussians1.covs[i_img1_]
            sigma_2d_2_obs = self.gaussians2.covs[j_img2_]

            # Convert PyTorch tensors to NumPy arrays if necessary
            if hasattr(sigma_2d_1_obs, "detach"):
                sigma_2d_1_obs = sigma_2d_1_obs.detach().cpu().numpy()
            if hasattr(sigma_2d_2_obs, "detach"):
                sigma_2d_2_obs = sigma_2d_2_obs.detach().cpu().numpy()

            # Compute the determinant of the 2D covariances and estimate a scale guess
            det_2d_1 = np.linalg.det(sigma_2d_1_obs)
            det_2d_2 = np.linalg.det(sigma_2d_2_obs)
            avg_det = np.sqrt(np.abs(det_2d_1 * det_2d_2))
            scale_guess = (
                avg_det ** (1.0 / 3.0) if avg_det > 0 else target_volume ** (1.0 / 3.0)
            )

            def two_view_resid(local_params: np.ndarray) -> np.ndarray:
                # Extract quaternion and scale parameters
                qw, qx, qy, qz, ss1, ss2, ss3 = local_params
                qq = np.array([qw, qx, qy, qz], dtype=float)
                ss = np.array([ss1, ss2, ss3], dtype=float)

                # Build the 3D covariance matrix from the parameters
                sigma_3_ = build_covariance_3d(qq, ss)

                # Compute residuals for each camera view
                r1_val = single_view_cov_residual(
                    local_params, point_3d_, sigma_2d_1_obs, self.k1, r1_local, t1_local
                )
                r2_val = single_view_cov_residual(
                    local_params, point_3d_, sigma_2d_2_obs, self.k2, r2_local, t2_local
                )

                # Compute volume residual using the log determinant of the 3D covariance
                try:
                    log_det = np.log(np.linalg.det(sigma_3_))
                    volume_residual = (
                        lambda_volume * (log_det - np.log(target_volume)) ** 2
                    )
                except np.linalg.LinAlgError:
                    print("Singular matrix")
                    volume_residual = 1e-6

                # Return concatenated residuals for optimization
                return np.concatenate([r1_val, r2_val, [volume_residual]])

            # Initialize parameters for optimization
            init_params = np.array(
                [1.0, 0.0, 0.0, 0.0, scale_guess, scale_guess, scale_guess],
                dtype=float,
            )

            # Perform non-linear least squares optimization
            result = least_squares(
                two_view_resid, x0=init_params, method="lm", max_nfev=20000
            )

            # Check optimization result and return the final 3D covariance
            if result.success:
                qq_final, ss_final = result.x[:4], result.x[4:]
                sigma_3_final = build_covariance_3d(qq_final, ss_final)
                return sigma_3_final
            else:
                print(
                    f"Failed to optimize covariance for Gaussian {idx}, using fallback scale."
                )
                fallback_scale = target_volume
                return np.diag([fallback_scale, fallback_scale, fallback_scale])

        # Use joblib to parallelize the optimization across multiple Gaussians
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(solve_cov_for_gaussian)(idx) for idx in range(num_3d)
        )

        # Store the optimized 3D covariances
        for idx_, cov3_ in enumerate(results):
            self.covariances_3d[idx_] = cov3_

        print(
            f"Finished LM optimization for {num_3d} Gaussians with joblib parallelism."
        )

    def compute_3d_gaussian_colors(self, color_mode: str = "average") -> None:
        """Compute a single RGB color for each 3D Gaussian by combining matched 2D Gaussians' colors."""
        if self.points_3d is None or len(self.points_3d) == 0:
            raise ValueError("3D points must be computed before computing 3D colors.")
        if not self.match_pairs:
            raise ValueError(
                "match_pairs not found. Did you call triangulate_gaussian_centers first?"
            )

        num_3d = self.points_3d.shape[0]
        self.color_3d: np.ndarray = np.zeros((num_3d, 3), dtype=np.float32)

        for idx in range(num_3d):
            i_img1, j_img2 = self.match_pairs[idx]
            color1 = self.gaussians1.rgb[i_img1]
            color2 = self.gaussians2.rgb[j_img2]

            if hasattr(color1, "detach"):
                color1 = color1.detach().cpu().numpy()
            if hasattr(color2, "detach"):
                color2 = color2.detach().cpu().numpy()

            if color_mode == "average":
                color_val = 0.5 * (color1 + color2)
            else:
                color_val = 0.5 * (color1 + color2)

            self.color_3d[idx] = color_val.astype(np.float32)

        print(f"Computed color_3d for {num_3d} 3D Gaussians using mode='{color_mode}'.")

    def compute_3d_gaussian_alphas(self, alpha_mode: str = "average") -> None:
        """Compute a single alpha value for each 3D Gaussian after triangulation."""
        if self.points_3d is None or len(self.points_3d) == 0:
            raise ValueError("3D points must be computed before computing 3D alpha.")
        if not self.match_pairs:
            raise ValueError(
                "match_pairs not found. Did you call triangulate_gaussian_centers first?"
            )

        num_3d = self.points_3d.shape[0]
        self.alpha_3d: np.ndarray = np.zeros(num_3d, dtype=np.float32)

        for idx in range(num_3d):
            i_img1, j_img2 = self.match_pairs[idx]
            alpha1 = self.gaussians1.alpha[i_img1]
            alpha2 = self.gaussians2.alpha[j_img2]

            if hasattr(alpha1, "detach"):
                alpha1 = alpha1.detach().cpu().numpy()
            if hasattr(alpha2, "detach"):
                alpha2 = alpha2.detach().cpu().numpy()

            if alpha_mode == "average":
                alpha_3d_val = 0.5 * (alpha1 + alpha2)
            elif alpha_mode == "min":
                alpha_3d_val = min(alpha1, alpha2)
            elif alpha_mode == "max":
                alpha_3d_val = max(alpha1, alpha2)
            else:
                alpha_3d_val = 0.5 * (alpha1 + alpha2)

            self.alpha_3d[idx] = float(alpha_3d_val)

        print(f"Computed alpha_3d for {num_3d} 3D Gaussians using mode='{alpha_mode}'.")
