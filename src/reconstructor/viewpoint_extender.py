# src/reconstructor/viewpoint_extender.py
import sys
import numpy as np
import torch
import os
from typing import List, Optional, Tuple, Dict, Any

from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.reconstructor.initial_3d_non_linear import (
    build_covariance_3d, 
    project_covariance_3d_to_2d,
    Initial3DReconstructor
)

class ViewpointExtender:
    """
    Extends an existing 3D Gaussian scene by adding a new viewpoint.
    
    Takes an existing 3D Gaussian distribution, projects it onto a reference viewpoint,
    matches it to a new image's 2D Gaussians, and solves for the new viewpoint's camera
    parameters using optimal transport.
    """

    def __init__(
        self,
        existing_3d_gaussians: List[Dict[str, np.ndarray]],
        camera_params_list: List[Tuple[np.ndarray, np.ndarray]],
        K_new: np.ndarray,
        reference_camera_idx: int,
        threshold_reprojection: float = 1e-3,
        device: Optional[torch.device] = None,
        # 追加するパラメータ
        source_gaussians_data: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize the ViewpointExtender.

        Args:
            existing_3d_gaussians: Array of 3D Gaussian parameters (center, covariance, etc.)
            camera_params_list: List of (R, t) for each registered camera
            K_new: Intrinsic matrix for the new viewpoint
            reference_camera_idx: Index of the reference camera for projection
            threshold_reprojection: Threshold for filtering outliers
            device: Device to run computations on (CPU/GPU)
            source_gaussians_data: 初期画像間の湧出ガウス情報 (image1, image2の各ガウスデータを含む辞書)
        """
        self.existing_3d_gaussians = existing_3d_gaussians
        self.camera_params_list = camera_params_list
        self.K_new = K_new
        self.reference_camera_idx = reference_camera_idx
        self.threshold_reprojection = threshold_reprojection
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Transport solver will be initialized later when needed
        self.transport_solver = None
        
        # Camera parameters for the new viewpoint (to be optimized)
        self.rvec = None
        self.tvec = None
        
        # 湧出ガウス情報を保存
        self.source_gaussians_data = source_gaussians_data
        
        # Validate camera parameters
        if len(self.camera_params_list) <= reference_camera_idx:
            raise ValueError(f"Reference camera index {reference_camera_idx} is out of bounds")
        
        # Convert K_new to torch tensor if it's not already
        if not isinstance(self.K_new, torch.Tensor):
            self.K_new = torch.tensor(self.K_new, dtype=torch.float32, device=self.device)

    def project_3d_gaussians(self) -> TwoDGaussians:
        """
        Project existing 3D Gaussians onto 2D using the reference camera.
        
        Returns:
            TwoDGaussians: Projected 2D Gaussians
        """
        # Get reference camera parameters
        R_ref, t_ref = self.camera_params_list[self.reference_camera_idx]
        
        # Convert to torch tensors if they're not already
        if not isinstance(R_ref, torch.Tensor):
            R_ref = torch.tensor(R_ref, dtype=torch.float32, device=self.device)
        if not isinstance(t_ref, torch.Tensor):
            t_ref = torch.tensor(t_ref, dtype=torch.float32, device=self.device)
            
        # Lists to collect 2D Gaussian parameters
        means_2d = []
        covs_2d = []
        rotations_2d = []
        scales_2d = []
        rgb_values = []
        alpha_values = []
        
        # For each 3D Gaussian
        for idx, gauss in enumerate(self.existing_3d_gaussians):
            # Extract 3D Gaussian parameters
            center_3d = gauss["center"]
            quat = gauss["quat"]
            scale_3d = gauss["scale3d"]
            color = gauss["color"]
            alpha = gauss["alpha"]
            
            # Convert to torch tensors if needed
            if not isinstance(center_3d, torch.Tensor):
                center_3d = torch.tensor(center_3d, dtype=torch.float32, device=self.device)
            
            # Build 3D covariance matrix from quaternion and scale
            sigma_3d = build_covariance_3d(quat, scale_3d)
            
            # Project center to camera coordinates
            x_cam = R_ref @ center_3d + t_ref
            
            # Check if point is in front of camera
            if x_cam[2] <= 1e-6:
                continue  # Skip points behind the camera
                
            # Project to image coordinates
            px_hom = self.K_new @ x_cam
            center_2d = px_hom[:2] / px_hom[2]
            
            # Project 3D covariance to 2D
            sigma_2d = project_covariance_3d_to_2d(
                sigma_3d, center_3d, self.K_new, R_ref, t_ref
            )
            
            # Get rotation angle and scales from 2D covariance
            rot_angle, scale_xy = self._convert_cov2d_to_params(sigma_2d)
            
            # Append to lists
            means_2d.append(center_2d.cpu().numpy() if isinstance(center_2d, torch.Tensor) else center_2d)
            covs_2d.append(sigma_2d.cpu().numpy() if isinstance(sigma_2d, torch.Tensor) else sigma_2d)
            rotations_2d.append(rot_angle)
            scales_2d.append(scale_xy)
            rgb_values.append(color)
            alpha_values.append(alpha)
        
        # Convert lists to arrays
        means_2d_arr = np.array(means_2d, dtype=np.float32)
        covs_2d_arr = np.array(covs_2d, dtype=np.float32)
        rotations_2d_arr = np.array(rotations_2d, dtype=np.float32)
        scales_2d_arr = np.array(scales_2d, dtype=np.float32)
        rgb_arr = np.array(rgb_values, dtype=np.float32)
        alpha_arr = np.array(alpha_values, dtype=np.float32)
        
        # Create TwoDGaussians object
        projected_gaussians = TwoDGaussians(
            means=means_2d_arr,
            covs=covs_2d_arr,
            rgb=rgb_arr,
            alpha=alpha_arr,
            rotations=rotations_2d_arr,
            scales=scales_2d_arr
        )
        
        return projected_gaussians

    def _convert_cov2d_to_params(self, cov_2d: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert a 2D covariance matrix to rotation angle and scales.
        
        Args:
            cov_2d: 2D covariance matrix (2x2)
            
        Returns:
            Tuple of (rotation_angle, [scale_x, scale_y])
        """
        # Compute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        
        # Sort by descending eigenvalue
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Ensure positive eigenvalues (should already be, but just in case)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Compute scales as sqrt of eigenvalues
        scales = np.sqrt(eigvals)
        
        # Compute rotation angle from principal eigenvector
        principal_axis = eigvecs[:, 0]
        angle = np.arctan2(principal_axis[1], principal_axis[0])
        
        return float(angle), scales

    def initialize_transport_solver(
        self,
        projected_2d_gaussians: TwoDGaussians,
        new_2d_gaussians: TwoDGaussians,
        epsilon: float = 0.01,
        lambda_epipolar: float = 1.0,
        lambda_color: float = 0.5,
    ) -> None:
        """
        Initialize the transport solver for 2D Gaussian matching.
        
        Args:
            projected_2d_gaussians: Projected 2D Gaussians from existing 3D model
            new_2d_gaussians: 2D Gaussians from the new viewpoint
            epsilon: Entropy regularization parameter
            lambda_epipolar: Weight for epipolar constraint
            lambda_color: Weight for color difference
        """
        self.transport_solver = OptimalTransportSolver(
            gaussians1=projected_2d_gaussians,
            gaussians2=new_2d_gaussians,
            k1=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            k2=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            epsilon=epsilon,
            lambda_mean=0.0,
            lambda_cov=0.0,
            lambda_color=lambda_color,
            lambda_epipolar=lambda_epipolar,
            device=self.device
        )
        
        # rvecとtvecを明示的に初期化
        self.transport_solver.rvec = torch.nn.Parameter(
            torch.zeros(3, dtype=torch.float32, device=self.device)
        )
        self.transport_solver.tvec = torch.nn.Parameter(
            torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32, device=self.device)
        )
        
        # ViewpointExtenderのrvecとtvecを設定
        self.rvec = self.transport_solver.rvec
        self.tvec = self.transport_solver.tvec


    def integrate_new_view(
        self, new_image_2d_gaussians: TwoDGaussians, max_iterations: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate a new viewpoint into the 3D reconstruction.
        
        Args:
            new_image_2d_gaussians: 2D Gaussians from the new viewpoint
            max_iterations: Maximum optimization iterations
            
        Returns:
            Tuple of (R_new, t_new) for the new camera viewpoint
        """
        # 1. Project existing 3D Gaussians to 2D
        projected_2d = self.project_3d_gaussians()
        
        # 2. Initialize transport solver
        self.initialize_transport_solver(
            projected_2d_gaussians=projected_2d,
            new_2d_gaussians=new_image_2d_gaussians
        )
        
        # 3. Optimize camera pose (R,t)
        self.transport_solver.optimize_with_RT(max_iter=max_iterations, tol=1e-6)
        
        # 4. Extract optimized R, t
        with torch.no_grad():
            R_est = self.transport_solver.rodrigues(self.rvec).detach().cpu().numpy()
            t_est = self.tvec.detach().cpu().numpy()
            
        # 5. Add to camera_params_list
        self.camera_params_list.append((R_est, t_est))
        
        # 6. Return the new camera parameters
        return R_est, t_est

    
    def add_new_gaussians(
        self,
        projected_2d: TwoDGaussians,
        new_image_2d_gaussians: TwoDGaussians,
        transport_matrix: np.ndarray,
        threshold: float = 1e-3,
        target_volume: float = 1.0,
        auto_threshold: bool = True
    ) -> None:
        """
        トランスポート行列を用いて新しい3Dガウスを追加する。
        不均衡最適輸送で湧出量が大きい箇所に新規ガウスを追加。
        既存のガウスは更新・削除しない。
        
        Args:
            projected_2d: 投影された2Dガウス分布
            new_image_2d_gaussians: 新しい視点の2Dガウス分布
            transport_matrix: 最適輸送行列
            threshold: 輸送閾値（auto_threshold=Falseの場合に使用）
            target_volume: 目標体積
            auto_threshold: 閾値を自動的に決定するかどうか
        """
        if len(self.camera_params_list) < 2:
            raise ValueError("At least two cameras are needed for triangulation")
        
        # 最新のカメラパラメータを取得
        R_new, t_new = self.camera_params_list[-1]
        
        # 参照カメラのパラメータを取得
        R_ref, t_ref = self.camera_params_list[self.reference_camera_idx]
        
        # 輸送行列から湧出量の大きい箇所を特定
        # 各列（新視点のガウス）の合計が小さい = 湧出量が大きい
        col_sums = transport_matrix.sum(axis=0)
        
        # 自動的に閾値を決定する場合
        if auto_threshold:
            # 列ごとの輸送量の基本統計
            mean_transport = np.mean(col_sums)
            std_transport = np.std(col_sums)
            min_transport = np.min(col_sums)
            
            # 輸送量の分布情報を表示
            print(f"Transport column sums statistics:")
            print(f"  Mean: {mean_transport:.4f}, Std: {std_transport:.4f}, Min: {min_transport:.4f}")
            print(f"  Histogram: {np.histogram(col_sums, bins=5)[0]}")
            
            # 閾値を自動計算：平均から一定のσ下回る値か、最小値を基準に
            if std_transport > 1e-4:  # 標準偏差が意味を持つ場合
                # 平均 - 1σ を閾値とする（調整可能）
                threshold = max(mean_transport - 1.0 * std_transport, min_transport)
            else:
                # 分散が小さい場合は、最小値から少し上を閾値に
                threshold = min_transport * 1.2
            
            print(f"Auto-determined threshold: {threshold:.4f}")
        
        # 湧出量の大きいガウスのインデックス（輸送量が閾値以下）
        target_indices = np.where(col_sums < threshold)[0]
        
        if len(target_indices) == 0:
            print("No significant source/sink detected. No new Gaussians added.")
            return
                
        print(f"Adding {len(target_indices)} new Gaussians from transport sinks...")
        
        # ダミーのホモグラフィ（使わないが必要）
        h_dummy = np.eye(3)
        
        # Initial3DReconstructorを初期化
        reconstructor = Initial3DReconstructor(
            gaussians1=projected_2d,
            gaussians2=new_image_2d_gaussians,
            k1=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            k2=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            h=h_dummy
        )
        
        # カメラパラメータを設定
        reconstructor.set_camera_matrices_explicitly(
            r1=R_ref,
            t1=t_ref,
            r2=R_new,
            t2=t_new
        )
        
        # 湧出量の大きいガウス間のみを考慮した輸送行列を作成
        # シャープな対応関係にするため1.0とする
        focused_transport = np.zeros_like(transport_matrix)
        
        # 各湧出ガウスに対し、最も近い投影ガウスを対応付ける
        for idx in target_indices:
            new_point = new_image_2d_gaussians.means[idx]
            
            if isinstance(new_point, torch.Tensor):
                new_point = new_point.detach().cpu().numpy()    
            distances = np.linalg.norm(projected_2d.means - new_point, axis=1)
            closest_proj_idx = np.argmin(distances)
            focused_transport[closest_proj_idx, idx] = 1.0
        
        # 三角測量
        reconstructor.triangulate_gaussian_centers(
            focused_transport, threshold=0.0, top_k=len(target_indices)
        )
        
        if len(reconstructor.points_3d) > 0:
            # 3D共分散を計算
            print(f"Computing covariances for {len(reconstructor.points_3d)} new points...")
            reconstructor.compute_3d_gaussian_covariances(
                lambda_volume=1.0, target_volume=target_volume
            )
            
            # 色と不透明度を計算
            reconstructor.compute_3d_gaussian_colors(color_mode="average")
            reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")
            
            # 新しい3Dガウスを既存のリストに追加
            for i in range(len(reconstructor.points_3d)):
                point_3d = reconstructor.points_3d[i]
                cov_3d = reconstructor.covariances_3d[i]
                color = reconstructor.color_3d[i]
                alpha = reconstructor.alpha_3d[i]
                
                # 共分散から四元数とスケールを推定
                eigvals, eigvecs = np.linalg.eigh(cov_3d)
                eigvals = np.maximum(eigvals, 1e-10)
                scales = np.sqrt(eigvals)
                
                # 単位四元数を使用
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
                # 新しい3Dガウスを作成
                new_gauss = {
                    "center": point_3d,
                    "quat": quat,
                    "scale3d": scales,
                    "color": color,
                    "alpha": alpha
                }
                
                # 既存の3Dガウスリストに追加
                self.existing_3d_gaussians.append(new_gauss)
            
            print(f"Added {len(reconstructor.points_3d)} new 3D Gaussians.")
        else:
            print("No new 3D Gaussians were triangulated.")

    # 以下は湧出ガウスを活用する新しいメソッド
    def add_new_gaussians_from_sources(
        self,
        new_image_2d_gaussians: TwoDGaussians,
        source_camera_idx: int = 0,
        target_volume: float = 1.0,
        distance_threshold: float = 30.0,
        color_threshold: float = 0.3
    ) -> int:
        """
        初期画像の湧出ガウスと新しい視点の2Dガウスを対応付けて、新しい3Dガウスを追加する
        
        Args:
            new_image_2d_gaussians: 新しい視点の2Dガウス分布
            source_camera_idx: 湧出ガウスが属する初期カメラのインデックス (0 または 1)
            target_volume: 新規ガウスの目標体積
            distance_threshold: 対応付けの距離閾値（ピクセル単位）
            color_threshold: 対応付けの色差閾値 (RGB差のL2ノルム)
            
        Returns:
            int: 追加された3Dガウスの数
        """
        # 湧出ガウス情報がない場合は処理終了
        if self.source_gaussians_data is None:
            print("No source gaussians data available.")
            return 0
            
        # カメラインデックスに応じた湧出ガウスデータを取得
        source_key = f"source_gaussians{source_camera_idx+1}_data"
        if source_key not in self.source_gaussians_data:
            print(f"No source gaussians data for camera {source_camera_idx}.")
            return 0
            
        source_data = self.source_gaussians_data[source_key]
        if source_data is None or len(source_data.get('indices', [])) == 0:
            print(f"Empty source gaussians data for camera {source_camera_idx}.")
            return 0
            
        # 湧出ガウスの特徴量を取得
        source_means = source_data['means']
        source_covs = source_data['covs']
        source_rgb = source_data['rgb']
        source_alpha = source_data['alpha']
        
        # 湧出ガウスのカメラパラメータを取得
        R_source, t_source = self.camera_params_list[source_camera_idx]
        
        # 新しい視点のカメラパラメータを取得
        R_new, t_new = self.camera_params_list[-1]
        
        # 対応の保存先
        matches = []  # (source_idx, new_idx) のリスト
        
        print(f"Finding matches between {len(source_means)} source gaussians and {len(new_image_2d_gaussians.means)} new gaussians...")
        
        # 各湧出ガウスについて、新しい視点の2Dガウスとの対応を探す
        for i, source_mean in enumerate(source_means):
            source_color = source_rgb[i]
            
            # ユークリッド距離と色差に基づいて最も近い新規ガウスを探す
            min_dist = float('inf')
            best_match_idx = -1
            
            for j, new_mean in enumerate(new_image_2d_gaussians.means):
                # 空間距離
                dist = np.linalg.norm(source_mean - new_mean)
                
                # 距離が閾値未満の場合のみ色差をチェック
                if dist < distance_threshold:
                    # 色差
                    color_diff = np.linalg.norm(source_color - new_image_2d_gaussians.rgb[j])
                    
                    # 色差が閾値未満かつ現時点での最小距離であれば更新
                    if color_diff < color_threshold and dist < min_dist:
                        min_dist = dist
                        best_match_idx = j
            
            # 対応が見つかった場合、マッチリストに追加
            if best_match_idx != -1:
                matches.append((i, best_match_idx))
        
        print(f"Found {len(matches)} matches between source gaussians and new gaussians.")
        
        if len(matches) == 0:
            return 0
            
        # 対応を使って三角測量するための簡易輸送行列を作成
        source_size = len(source_means)
        new_size = len(new_image_2d_gaussians.means)
        focused_transport = np.zeros((source_size, new_size))
        
        for src_idx, new_idx in matches:
            focused_transport[src_idx, new_idx] = 1.0
            
        # 湧出ガウスからTwoDGaussiansオブジェクトを作成
        source_rotations = source_data.get('rotations', np.zeros(len(source_means)))
        source_scales = source_data.get('scales', np.ones((len(source_means), 2)))
        
        source_gaussians = TwoDGaussians(
            means=source_means,
            covs=source_covs,
            rgb=source_rgb,
            alpha=source_alpha,
            rotations=source_rotations,
            scales=source_scales
        )
        
        # Initial3DReconstructorを初期化
        h_dummy = np.eye(3)
        
        # 湧出ガウスのカメラの内部パラメータを取得
        # ここでは簡単のため、同じK_newを使っているが、
        # 実際には湧出ガウスのカメラに対応するK値を使用するべき
        K_source = self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new
        
        # Initial3DReconstructorインスタンスを作成
        reconstructor = Initial3DReconstructor(
            gaussians1=source_gaussians,
            gaussians2=new_image_2d_gaussians,
            k1=K_source,
            k2=self.K_new.cpu().numpy() if isinstance(self.K_new, torch.Tensor) else self.K_new,
            h=h_dummy
        )
        
        # カメラパラメータを設定
        reconstructor.set_camera_matrices_explicitly(
            r1=R_source,
            t1=t_source,
            r2=R_new,
            t2=t_new
        )
        
        # 三角測量
        reconstructor.triangulate_gaussian_centers(
            focused_transport, threshold=0.0, top_k=len(matches)
        )
        
        if len(reconstructor.points_3d) == 0:
            print("No 3D points were triangulated from source gaussians.")
            return 0
            
        # 3D共分散を計算
        print(f"Computing covariances for {len(reconstructor.points_3d)} source-derived points...")
        reconstructor.compute_3d_gaussian_covariances(
            lambda_volume=1.0, target_volume=target_volume
        )
        
        # 色と不透明度を計算
        reconstructor.compute_3d_gaussian_colors(color_mode="average")
        reconstructor.compute_3d_gaussian_alphas(alpha_mode="average")
        
        # 新しい3Dガウスを既存のリストに追加
        for i in range(len(reconstructor.points_3d)):
            point_3d = reconstructor.points_3d[i]
            cov_3d = reconstructor.covariances_3d[i]
            color = reconstructor.color_3d[i]
            alpha = reconstructor.alpha_3d[i]
            
            # 共分散から四元数とスケールを推定
            eigvals, eigvecs = np.linalg.eigh(cov_3d)
            eigvals = np.maximum(eigvals, 1e-10)
            scales = np.sqrt(eigvals)
            
            # 単位四元数を使用
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # 新しい3Dガウスを作成
            new_gauss = {
                "center": point_3d,
                "quat": quat,
                "scale3d": scales,
                "color": color,
                "alpha": alpha,
                # 湧出ガウス由来であることを示すフラグ（オプション）
                "from_source": True
            }
            
            # 既存の3Dガウスリストに追加
            self.existing_3d_gaussians.append(new_gauss)
        
        print(f"Added {len(reconstructor.points_3d)} new 3D Gaussians from source gaussians.")
        return len(reconstructor.points_3d)


    def integrate_new_view_and_gaussians(
        self,
        new_image_2d_gaussians: TwoDGaussians,
        max_iterations: int = 1000,
        transport_threshold: float = 1e-3,
        target_volume: float = 1.0,
        auto_threshold: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        新しい視点を統合し、3Dガウスを追加する。
        既存の3Dガウスは更新せず、消去も行わない。
        
        Args:
            new_image_2d_gaussians: 新しい視点の2Dガウス分布
            max_iterations: 最適化の最大イテレーション回数
            transport_threshold: 最適輸送の閾値
            target_volume: 新規3Dガウスの目標体積
            auto_threshold: 閾値を自動的に決定するかどうか
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 新しいカメラパラメータ(R, t)
        """
        # 1. 新しい視点のカメラパラメータを推定
        print("Estimating camera parameters for new viewpoint...")
        R_new, t_new = self.integrate_new_view(
            new_image_2d_gaussians=new_image_2d_gaussians,
            max_iterations=max_iterations
        )
        
        # 2. 最適輸送行列を計算
        print("Computing optimal transport matrix...")
        projected_2d = self.project_3d_gaussians()
        
        # F行列を使って再計算
        cost_matrix = self.transport_solver.compute_cost_matrix_fundamental(
            self.transport_solver.f
        )
        
        transport = self.transport_solver.unbalanced_sinkhorn_algorithm(
            cost_matrix, rho=0.5, max_iter=10000, tol=1e-6
        )
        
        transport_matrix = transport.detach().cpu().numpy()
        
        # 3. 輸送行列から新規3Dガウスを追加（湧出量の大きい箇所）
        print("Adding new 3D Gaussians from current sinks...")
        self.add_new_gaussians(
            projected_2d=projected_2d,
            new_image_2d_gaussians=new_image_2d_gaussians,
            transport_matrix=transport_matrix,
            threshold=transport_threshold,
            target_volume=target_volume,
            auto_threshold=auto_threshold
        )
        
        # 4. 湧出ガウス情報がある場合は、それを使って追加の3Dガウスを追加
        if self.source_gaussians_data is not None:
            print("\nProcessing source gaussians from initial images...")
            
            # 初期画像1の湧出ガウスから追加
            added_from_img1 = self.add_new_gaussians_from_sources(
                new_image_2d_gaussians=new_image_2d_gaussians,
                source_camera_idx=0,
                target_volume=target_volume
            )
            
            # 初期画像2の湧出ガウスから追加（カメラが2つ以上ある場合）
            added_from_img2 = 0
            if len(self.camera_params_list) >= 2:
                added_from_img2 = self.add_new_gaussians_from_sources(
                    new_image_2d_gaussians=new_image_2d_gaussians,
                    source_camera_idx=1,
                    target_volume=target_volume
                )
                
            print(f"Added {added_from_img1 + added_from_img2} Gaussians from source gaussians: " 
                  f"{added_from_img1} from image 1, {added_from_img2} from image 2")
        
        return R_new, t_new