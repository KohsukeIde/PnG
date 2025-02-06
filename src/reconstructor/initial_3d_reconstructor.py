from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.primitive.twod_gaussians_rs import TwoDGaussians


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

        self.gaussians1: TwoDGaussians = gaussians1
        self.gaussians2: TwoDGaussians = gaussians2
        self.k1: np.ndarray = k1
        self.k2: np.ndarray = k2
        self.h: np.ndarray = h

        self.p1: Optional[np.ndarray] = None  # Projection matrix for camera 1
        self.p2: Optional[np.ndarray] = None  # Projection matrix for camera 2

        # 3D points from triangulation (N x 3) or None if not yet computed
        self.points_3d: Optional[np.ndarray] = None

        # Covariance matrices for 3D Gaussians (N x 3 x 3) or None if not yet computed
        self.covariances_3d: Optional[np.ndarray] = None

        # Fix camera 1 at world coordinate origin
        self.r1: np.ndarray = np.eye(3, dtype=float)
        self.t1: np.ndarray = np.zeros(3, dtype=float)

        # r2, t2 might remain None until homography is decomposed
        self.r2: Optional[np.ndarray] = None
        self.t2: Optional[np.ndarray] = None

        # Matched pairs from transport matrix (list of (i_img1, j_img2)) or None
        self.match_pairs: Optional[List[Tuple[int, int]]] = None

    def compute_camera_matrices_from_homography(self) -> None:
        """Compute camera projection matrices p1 and p2 from the homography matrix."""

        decomp_result = cv2.decomposeHomographyMat(self.h, self.k1 @ self.k1.T)
        if decomp_result is None:
            raise ValueError("Homography decomposition returned None.")


        retval, rotations, translations, normals = decomp_result
        if retval == 0:
            raise ValueError(
                "Homography decomposition failed or found no valid solution."
            )

        selected = False
        for i in range(retval):
            r_candidate = rotations[i]
            t_candidate = translations[i]
            # Convert to numpy if needed:
            if not isinstance(r_candidate, np.ndarray):
                r_candidate = np.array(r_candidate, dtype=float)
            if not isinstance(t_candidate, np.ndarray):
                t_candidate = np.array(t_candidate, dtype=float)

            t_flat: np.ndarray = t_candidate.flatten()
            # Choose a decomposition where t[2] > 0
            if t_flat[2] > 0:
                self.r2 = r_candidate
                self.t2 = t_flat
                selected = True
                break

        # If no valid decomposition had t[2] > 0, pick the first one anyway
        if not selected:
            r_candidate = rotations[0]
            t_candidate = translations[0]
            if not isinstance(r_candidate, np.ndarray):
                r_candidate = np.array(r_candidate, dtype=float)
            if not isinstance(t_candidate, np.ndarray):
                t_candidate = np.array(t_candidate, dtype=float)
            self.r2 = r_candidate
            self.t2 = t_candidate.flatten()

        # Guard against self.r2 or self.t2 still being None
        assert self.r2 is not None, "r2 is unexpectedly None."
        assert self.t2 is not None, "t2 is unexpectedly None."

        # Build p1, p2
        self.p1 = self.k1 @ np.hstack((self.r1, self.t1.reshape(3, 1)))
        self.p2 = self.k2 @ np.hstack((self.r2, self.t2.reshape(3, 1)))

    def triangulate_gaussian_centers(
        self,
        transport_matrix: np.ndarray,
        threshold: float = 1e-3,
        top_k: int = 100,
    ) -> None:
        """Extract correspondences from transport_matrix and triangulate 3D points.

        Args:
            transport_matrix (np.ndarray): Transport plan of shape (k1, k2).
            threshold (float): Minimum transport value to consider.
            top_k (int): Maximum number of pairs to keep (default=100).

        Raises:
            ValueError: If camera matrices are not computed before triangulation.
        """
        if self.p1 is None or self.p2 is None:
            raise ValueError("Camera matrices must be computed before triangulation.")

        centers1 = self.gaussians1.means
        centers2 = self.gaussians2.means
        k1_size = centers1.shape[0]
        k2_size = centers2.shape[0]

        print(f"transport matrix shape {transport_matrix.shape}")

        t_flat = transport_matrix.ravel()
        all_indices = np.arange(t_flat.size)

        mask = t_flat >= threshold
        valid_indices = all_indices[mask]
        if len(valid_indices) == 0:
            print(f"No transport values >= {threshold}")
            self.points_3d = np.zeros((0, 3))
            self.match_pairs = []
            return

        valid_tvals = t_flat[mask]
        sort_desc = np.argsort(-valid_tvals)
        top_k = min(top_k, len(valid_tvals))
        best_indices = valid_indices[sort_desc[:top_k]]

        i_coords, j_coords = np.unravel_index(best_indices, (k1_size, k2_size))
        correspondences: List[Tuple[np.ndarray, int, int]] = []

        assert self.p1 is not None, "p1 must be computed at this point."
        assert self.p2 is not None, "p2 must be computed at this point."

        for idx in range(top_k):
            i = i_coords[idx]
            j = j_coords[idx]
            x1 = centers1[i]
            x2 = centers2[j]

            x1_h = np.array([x1[0], x1[1], 1.0], dtype=float)
            x2_h = np.array([x2[0], x2[1], 1.0], dtype=float)

            a_mat = np.zeros((4, 4), dtype=float)
            a_mat[0] = x1_h[0] * self.p1[2] - self.p1[0]
            a_mat[1] = x1_h[1] * self.p1[2] - self.p1[1]
            a_mat[2] = x2_h[0] * self.p2[2] - self.p2[0]
            a_mat[3] = x2_h[1] * self.p2[2] - self.p2[1]

            _, _, vt = np.linalg.svd(a_mat)
            x_val = vt[-1]
            x_val /= x_val[3]
            point_3d = x_val[:3]
            correspondences.append((point_3d, i, j))

        if len(correspondences) == 0:
            print("No valid correspondences after filtering & topK.")
            self.points_3d = np.zeros((0, 3))
            self.match_pairs = []
        else:
            self.points_3d = np.array([c[0] for c in correspondences])
            self.match_pairs = [(c[1], c[2]) for c in correspondences]

        print(
            f"Selected {len(correspondences)} 3D points (threshold={threshold}, top_k={top_k})."
        )

    def _compute_jacobian(
        self, point_3d: np.ndarray, k: np.ndarray, r: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Compute the Jacobian for the pinhole camera model.

        Args:
            point_3d (np.ndarray): [X, Y, Z] in world coordinates.
            k (np.ndarray): Intrinsic camera matrix.
            r (np.ndarray): Rotation matrix (world->camera).
            t (np.ndarray): Translation vector (world->camera).

        Returns:
            np.ndarray: Jacobian matrix (2,3).
        """
        x_c = r @ point_3d + t
        x_val, y_val, z_val = x_c

        fx = k[0, 0]
        fy = k[1, 1]

        du_dxc = fx / z_val
        du_dyc = 0.0
        du_dzc = -fx * x_val / (z_val**2)

        dv_dxc = 0.0
        dv_dyc = fy / z_val
        dv_dzc = -fy * y_val / (z_val**2)

        j_image_wrt_xc = np.array(
            [[du_dxc, du_dyc, du_dzc], [dv_dxc, dv_dyc, dv_dzc]], dtype=float
        )

        j_mat = j_image_wrt_xc @ r
        return np.array(j_mat, dtype=float)

    def compute_3d_gaussian_covariances(self) -> None:
        """Compute 3D Gaussian covariances from 2D projections and camera parameters.

        Raises:
            ValueError: If camera matrices or 3D points are not yet computed.
        """
        if self.p1 is None or self.p2 is None:
            raise ValueError(
                "Camera matrices must be computed before computing covariances."
            )
        if self.points_3d is None or len(self.points_3d) == 0:
            raise ValueError("3D points must be computed before computing covariances.")
        if not self.match_pairs:
            raise ValueError(
                "match_pairs not found. triangulate_gaussian_centers must store them."
            )

        num_3d = self.points_3d.shape[0]
        self.covariances_3d = np.zeros((num_3d, 3, 3), dtype=float)

        # r1_local = I, t1_local = 0
        r1_local = np.eye(3, dtype=float)
        t1_local = np.zeros(3, dtype=float)

        # Decompose self.p2 into rotation & translation
        # Guard to ensure self.p2 is not None
        assert self.p2 is not None, "p2 must not be None here."
        m_mat = self.p2[:, :3]
        u_mat, s_vals, vt_mat = np.linalg.svd(m_mat)
        r2_local = u_mat @ vt_mat
        t2_local = np.linalg.inv(self.k2) @ self.p2[:, 3]

        for idx in range(num_3d):
            point_3d = self.points_3d[idx]
            i_img1, j_img2 = self.match_pairs[idx]

            # covs are 2D covariance from each camera
            sigma_2d_1 = self.gaussians1.covs[i_img1]
            sigma_2d_2 = self.gaussians2.covs[j_img2]

            j1 = self._compute_jacobian(point_3d, self.k1, r1_local, t1_local)
            j2 = self._compute_jacobian(point_3d, self.k2, r2_local, t2_local)

            a1 = sigma_2d_1[0, 0]
            b1 = sigma_2d_1[0, 1]
            c1 = sigma_2d_1[1, 1]

            a2 = sigma_2d_2[0, 0]
            b2 = sigma_2d_2[0, 1]
            c2 = sigma_2d_2[1, 1]

            m_i, v_i = self._construct_linear_system(
                j1, r1_local, a1, b1, c1, j2, r2_local, a2, b2, c2
            )

            try:
                s_solution, residuals, rank, singular_vals = np.linalg.lstsq(
                    m_i, v_i, rcond=None
                )
            except np.linalg.LinAlgError as e:
                raise ValueError(
                    f"Linear system solving failed for 3D point {idx}: {e}"
                ) from e

            sigma_3 = np.array(
                [
                    [s_solution[0], s_solution[3], s_solution[4]],
                    [s_solution[3], s_solution[1], s_solution[5]],
                    [s_solution[4], s_solution[5], s_solution[2]],
                ],
                dtype=float,
            )

            # Ensure symmetry
            sigma_3 = 0.5 * (sigma_3 + sigma_3.T)

            # Clamp eigenvalues
            e_vals, e_vecs = np.linalg.eigh(sigma_3)
            e_vals_clamped = np.clip(e_vals, 1e-6, None)
            sigma_3_clamped = (e_vecs * e_vals_clamped) @ e_vecs.T

            self.covariances_3d[idx] = sigma_3_clamped

    def _construct_linear_system(
        self,
        j1: np.ndarray,
        r1_local: np.ndarray,
        a1: float,
        b1: float,
        c1: float,
        j2: np.ndarray,
        r2_local: np.ndarray,
        a2: float,
        b2: float,
        c2: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct a linear system for the 3D covariance from two cameras' 2D covariances.

        Args:
            j1 (np.ndarray): Jacobian (2,3) for camera 1.
            r1_local (np.ndarray): 3x3 rotation for camera 1.
            a1 (float): sigma_2d_1[0,0].
            b1 (float): sigma_2d_1[0,1].
            c1 (float): sigma_2d_1[1,1].
            j2 (np.ndarray): Jacobian (2,3) for camera 2.
            r2_local (np.ndarray): 3x3 rotation for camera 2.
            a2 (float): sigma_2d_2[0,0].
            b2 (float): sigma_2d_2[0,1].
            c2 (float): sigma_2d_2[1,1].

        Returns:
            Tuple[np.ndarray, np.ndarray]: (M (6x6), v (6,)) for the system M*s = v.
        """
        m_arr = np.zeros((6, 6), dtype=float)

        # Camera 1 coefficients
        m_arr[0, 0] = j1[0, 0] ** 2
        m_arr[0, 1] = j1[0, 1] ** 2
        m_arr[0, 2] = j1[0, 2] ** 2
        m_arr[0, 3] = 2 * j1[0, 0] * j1[0, 1]
        m_arr[0, 4] = 2 * j1[0, 0] * j1[0, 2]
        m_arr[0, 5] = 2 * j1[0, 1] * j1[0, 2]

        m_arr[1, 0] = 2 * j1[0, 0] * j1[1, 0]
        m_arr[1, 1] = 2 * j1[0, 1] * j1[1, 1]
        m_arr[1, 2] = 2 * j1[0, 2] * j1[1, 2]
        m_arr[1, 3] = j1[0, 0] * j1[1, 1] + j1[0, 1] * j1[1, 0]
        m_arr[1, 4] = j1[0, 0] * j1[1, 2] + j1[0, 2] * j1[1, 0]
        m_arr[1, 5] = j1[0, 1] * j1[1, 2] + j1[0, 2] * j1[1, 1]

        m_arr[2, 0] = j1[1, 0] ** 2
        m_arr[2, 1] = j1[1, 1] ** 2
        m_arr[2, 2] = j1[1, 2] ** 2
        m_arr[2, 3] = 2 * j1[1, 0] * j1[1, 1]
        m_arr[2, 4] = 2 * j1[1, 0] * j1[1, 2]
        m_arr[2, 5] = 2 * j1[1, 1] * j1[1, 2]

        # Camera 2 coefficients
        m_arr[3, 0] = j2[0, 0] ** 2
        m_arr[3, 1] = j2[0, 1] ** 2
        m_arr[3, 2] = j2[0, 2] ** 2
        m_arr[3, 3] = 2 * j2[0, 0] * j2[0, 1]
        m_arr[3, 4] = 2 * j2[0, 0] * j2[0, 2]
        m_arr[3, 5] = 2 * j2[0, 1] * j2[0, 2]

        m_arr[4, 0] = 2 * j2[0, 0] * j2[1, 0]
        m_arr[4, 1] = 2 * j2[0, 1] * j2[1, 1]
        m_arr[4, 2] = 2 * j2[0, 2] * j2[1, 2]
        m_arr[4, 3] = j2[0, 0] * j2[1, 1] + j2[0, 1] * j2[1, 0]
        m_arr[4, 4] = j2[0, 0] * j2[1, 2] + j2[0, 2] * j2[1, 0]
        m_arr[4, 5] = j2[0, 1] * j2[1, 2] + j2[0, 2] * j2[1, 1]

        m_arr[5, 0] = j2[1, 0] ** 2
        m_arr[5, 1] = j2[1, 1] ** 2
        m_arr[5, 2] = j2[1, 2] ** 2
        m_arr[5, 3] = 2 * j2[1, 0] * j2[1, 1]
        m_arr[5, 4] = 2 * j2[1, 0] * j2[1, 2]
        m_arr[5, 5] = 2 * j2[1, 1] * j2[1, 2]

        v_arr = np.array([a1, b1, c1, a2, b2, c2], dtype=float)
        return m_arr, v_arr

    def compute_3d_gaussian_colors(self, color_mode: str = "average") -> None:
        """Compute a single RGB color for each 3D Gaussian by combining matched 2D Gaussians' colors.

        Args:
            color_mode (str): Strategy for combining colors. Defaults to "average".

        Raises:
            ValueError: If 3D points or match_pairs have not been computed.
        """
        if self.points_3d is None or self.points_3d.shape[0] == 0:
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

            if color_mode == "average":
                color_val = 0.5 * (color1 + color2)
            else:
                # Fallback or other strategies
                color_val = 0.5 * (color1 + color2)

            self.color_3d[idx] = color_val.astype(np.float32)

        print(f"Computed color_3d for {num_3d} 3D Gaussians using mode='{color_mode}'.")

    def compute_3d_gaussian_alphas(self, alpha_mode: str = "average") -> None:
        """Compute a single alpha value for each 3D Gaussian after triangulation.

        Args:
            alpha_mode (str): Strategy for combining alpha values ("average", "min", "max").

        Raises:
            ValueError: If 3D points or match_pairs have not been computed.
        """
        if self.points_3d is None or self.points_3d.shape[0] == 0:
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
