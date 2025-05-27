
    def _quat_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (...,4) unit quaternion [w,x,y,z]  (camera→world)
        return: (...,3,3) rotation matrix
        """
        w, x, y, z = q.unbind(-1)
        ww, xx, yy, zz = w*w, x*x, y*y, z*z
        wx, wy, wz = w*x, w*y, w*z
        xy, xz, yz = x*y, x*z, y*z
        R = torch.stack([
            torch.stack([ww+xx-yy-zz, 2*(xy-wz)   , 2*(xz+wy)], -1),
            torch.stack([2*(xy+wz)   , ww-xx+yy-zz, 2*(yz-wx)], -1),
            torch.stack([2*(xz-wy)   , 2*(yz+wx)  , ww-xx-yy+zz], -1)
        ], -2)
        return R

    
    def rodrigues(self, rvec: torch.Tensor) -> torch.Tensor:
        """SO(3) exponential map with small-angle safeguard and autograd support.
        
        rvec: (3,) -> rotation vector
        Returns: (3,3) rotation matrix using Rodrigues formula
        """
        # Calculate norm (rotation angle)
        theta = torch.linalg.norm(rvec)
        eps = 1e-6  # 小角近似の閾値
        
        # skew-symmetric matrix
        K = self.skew(rvec)
        
        # 小角度近似（θ ≈ 0の場合）
        R_small = torch.eye(3, dtype=torch.float32, device=rvec.device) + K
        
        # 通常の計算
        r_axis = rvec / torch.max(theta, torch.tensor(eps, device=rvec.device))
        K_unit = self.skew(r_axis)
        R_normal = (
            torch.eye(3, dtype=torch.float32, device=rvec.device)  # Identity matrix
            + torch.sin(theta) * K_unit                           # sin(θ)K term
            + (1.0 - torch.cos(theta)) * (K_unit @ K_unit)        # (1-cos(θ))K² term
        )
        
        # 条件に応じて値を選択（勾配は両経路に流れる）
        is_small = theta < eps
        R = torch.where(is_small, R_small, R_normal)
        
        return R


    def skew(self, v: torch.Tensor) -> torch.Tensor:
        """
        (...,3)  →  (...,3,3)   skew-symmetric matrix
        """
        K = torch.zeros((*v.shape[:-1], 3, 3), dtype=v.dtype, device=v.device)
        K[..., 0, 1] = -v[..., 2];  K[..., 0, 2] =  v[..., 1]
        K[..., 1, 0] =  v[..., 2];  K[..., 1, 2] = -v[..., 0]
        K[..., 2, 0] = -v[..., 1];  K[..., 2, 1] =  v[..., 0]
        return K

    def decompose_screw(self,
                       xi: torch.Tensor,
                       eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        xi  : (...,6)   = [omega, v]  in se(3)

        Returns
        -------
        S_hat : (...,6)   unit screw axis  (||omega_hat|| == 1  **or**  omega_hat==0)
        theta : (...,)    magnitude (rad *or* metres)

        - identical to the JAX implementation used in the iNeRF notebook
        """
        omega, v = xi[..., :3], xi[..., 3:]

        omega_norm = torch.linalg.norm(omega, dim=-1, keepdim=True)        # (...,1)
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)

        has_rot   = omega_norm > eps
        has_trans = v_norm > eps

        # theta:  ||omega||  if rotation exists,  otherwise  ||v||
        theta = torch.where(has_rot, omega_norm, v_norm).clamp_min(eps)

        # unit screw
        S_hat = torch.where(has_rot, xi / theta,                # general case
                         torch.cat([ torch.zeros_like(omega), v / theta ], dim=-1))  # pure translation

        return S_hat, theta.squeeze(-1)                         # theta shape (...,)

    def se3_exp_T(self, xi: torch.Tensor,
                  small_th: float = 1e-4,
                  eps: float = 1e-8) -> torch.Tensor:
        """
        Vectorised, NaN-safe Rodrigues / Modern-Robotics formula
        * works on an arbitrary leading batch shape

        xi : (...,6)  screw coordinates
        -->
        T  : (...,4,4) homogeneous matrix (camera-to-world in iNeRF)
        """
        omega, v  = xi[..., :3], xi[..., 3:]
        theta2    = (omega * omega).sum(dim=-1, keepdim=True)                    # (...,1)
        theta     = torch.sqrt(theta2 + eps)

        # masks
        small = (theta2 < small_th ** 2).type_as(theta)                      # 0/1
        large = 1.0 - small

        # --------  scalar coefficients A(theta) B(theta) C(theta) -------------
        sin_over_x   = torch.sin(theta) / (theta + eps)
        one_minus_cx = (1.0 - torch.cos(theta)) / (theta2 + eps)

        A = small * (1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0) + large * sin_over_x
        B = small * (0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0) + large * one_minus_cx
        C = (1.0 - A) / (theta2 + eps)

        # --------  rotation R -------------------------------------
        omega_hat = large * (omega / (theta + eps)) + small * omega               # use unit axis when theta!=0
        K     = self.skew(omega_hat)

        I3 = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(
                xi.shape[:-1] + (3, 3))

        R = I3 + A[..., None, None] * K + B[..., None, None] * (K @ K)

        # --------  translation t ----------------------------------
        V = I3 + B[..., None, None] * K + C[..., None, None] * (K @ K)
        t = (V @ v[..., None]).squeeze(-1)

        # --------  assemble T -------------------------------------
        T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(
                xi.shape[:-1] + (4, 4)).clone()
        T[..., :3, :3] = R
        T[..., :3,  3] = t
        return T
