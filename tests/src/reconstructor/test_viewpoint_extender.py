import pytest
import numpy as np
import torch
from src.primitive.twod_gaussians_rs import TwoDGaussians
from src.reconstructor.viewpoint_extender import ViewpointExtender
from src.reconstructor.initial_3d_non_linear import build_covariance_3d, project_covariance_3d_to_2d


@pytest.fixture
def dummy_3d_gaussians():
    """ダミーの3Dガウス分布を作成するフィクスチャ"""
    num_gaussians = 10
    gaussians = []
    
    for i in range(num_gaussians):
        gauss = {
            "center": np.array([i*0.5, i*0.2, 10.0 + i*0.1], dtype=np.float32),
            "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # 単位四元数
            "scale3d": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "color": np.array([0.8, 0.2, 0.3], dtype=np.float32),
            "alpha": 0.9
        }
        gaussians.append(gauss)
    
    return gaussians


@pytest.fixture
def camera_params():
    """カメラパラメータを作成するフィクスチャ"""
    # カメラ1: 単位行列の回転、原点
    R1 = np.eye(3, dtype=np.float32)
    t1 = np.zeros(3, dtype=np.float32)
    
    # カメラ2: 少し回転と移動
    angle = np.deg2rad(10)
    c, s = np.cos(angle), np.sin(angle)
    R2 = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)
    t2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    return [(R1, t1), (R2, t2)]


@pytest.fixture
def intrinsics():
    """カメラ内部パラメータを作成するフィクスチャ"""
    return np.array([
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def viewpoint_extender(dummy_3d_gaussians, camera_params, intrinsics):
    """ViewpointExtenderインスタンスを作成するフィクスチャ"""
    return ViewpointExtender(
        existing_3d_gaussians=dummy_3d_gaussians,
        camera_params_list=camera_params,
        K_new=intrinsics,
        reference_camera_idx=0,
        threshold_reprojection=1e-3
    )


@pytest.fixture
def dummy_2d_gaussians():
    """ダミーの2Dガウス分布を作成するフィクスチャ"""
    num_gaussians = 5
    means = np.random.rand(num_gaussians, 2).astype(np.float32)
    covs = np.array([np.eye(2) for _ in range(num_gaussians)], dtype=np.float32)
    rgb = np.random.rand(num_gaussians, 3).astype(np.float32)
    alpha = np.ones(num_gaussians, dtype=np.float32)
    rotations = np.zeros(num_gaussians, dtype=np.float32)
    scales = np.ones((num_gaussians, 2), dtype=np.float32)
    
    return TwoDGaussians(
        means=means, 
        covs=covs, 
        rgb=rgb, 
        alpha=alpha,
        rotations=rotations,
        scales=scales
    )


def test_viewpoint_extender_init(viewpoint_extender, dummy_3d_gaussians, camera_params, intrinsics):
    """ViewpointExtenderの初期化をテスト"""
    assert viewpoint_extender.existing_3d_gaussians == dummy_3d_gaussians
    assert viewpoint_extender.camera_params_list == camera_params
    assert np.array_equal(viewpoint_extender.K_new.cpu().numpy(), intrinsics)
    assert viewpoint_extender.reference_camera_idx == 0
    assert viewpoint_extender.threshold_reprojection == 1e-3
    assert viewpoint_extender.transport_solver is None


def test_project_3d_gaussians(viewpoint_extender, dummy_3d_gaussians):
    """3Dガウスの2D投影をテスト"""
    # 3Dガウスを投影
    projected_2d = viewpoint_extender.project_3d_gaussians()
    
    # ガウスの数が正しいか確認
    assert projected_2d.k == len(dummy_3d_gaussians)
    
    # meansの形状が正しいか確認
    assert projected_2d.means.shape == (len(dummy_3d_gaussians), 2)
    
    # covsの形状が正しいか確認
    assert projected_2d.covs.shape == (len(dummy_3d_gaussians), 2, 2)
    
    # 既知の3D点が期待される2D点に投影されるか検証
    # 簡易的に、最初のガウスをチェック
    x_3d = dummy_3d_gaussians[0]["center"]
    k = viewpoint_extender.K_new.cpu().numpy()
    expected_x = k[0, 0] * x_3d[0] / x_3d[2] + k[0, 2]
    expected_y = k[1, 1] * x_3d[1] / x_3d[2] + k[1, 2]
    
    assert abs(projected_2d.means[0, 0] - expected_x) < 1e-4
    assert abs(projected_2d.means[0, 1] - expected_y) < 1e-4


def test_select_new_viewpoint_empty(viewpoint_extender):
    """空の候補リストでselect_new_viewpointがValueErrorを発生させるかテスト"""
    with pytest.raises(ValueError):
        viewpoint_extender.select_new_viewpoint([])


def test_match_2d_gaussians_no_solver(viewpoint_extender, dummy_2d_gaussians):
    """ソルバーが初期化されていない場合にmatch_2d_gaussiansがValueErrorを発生させるかテスト"""
    with pytest.raises(ValueError):
        viewpoint_extender.match_2d_gaussians(dummy_2d_gaussians, dummy_2d_gaussians)


def test_initialize_transport_solver(viewpoint_extender, dummy_2d_gaussians):
    """トランスポートソルバーの初期化をテスト"""
    # トランスポートソルバーを初期化
    viewpoint_extender.initialize_transport_solver(
        projected_2d_gaussians=dummy_2d_gaussians,
        new_2d_gaussians=dummy_2d_gaussians
    )
    
    # ソルバーが初期化されているか確認
    assert viewpoint_extender.transport_solver is not None
    
    # rvecとtvecが設定されているか確認
    assert viewpoint_extender.rvec is not None
    assert viewpoint_extender.tvec is not None


def test_convert_cov2d_to_params(viewpoint_extender):
    """2D共分散からパラメータへの変換をテスト"""
    # 単位共分散行列
    cov_2d = np.eye(2, dtype=np.float32)
    
    # パラメータに変換
    rotation, scales = viewpoint_extender._convert_cov2d_to_params(cov_2d)
    
    # 回転が0（または±π）に近いか確認
    assert abs(rotation) < 1e-6 or abs(abs(rotation) - np.pi) < 1e-6
    
    # スケールが1に近いか確認
    assert np.allclose(scales, np.ones(2), atol=1e-6)
    
    # 非等方な共分散
    cov_2d = np.array([
        [2.0, 0.0],
        [0.0, 0.5]
    ], dtype=np.float32)
    
    # パラメータに変換
    rotation, scales = viewpoint_extender._convert_cov2d_to_params(cov_2d)
    
    # 回転が0（または±π）に近いか確認
    assert abs(rotation) < 1e-6 or abs(abs(rotation) - np.pi) < 1e-6
    
    # スケールが正しいか確認（sqrt(eigenvalues)）
    expected_scales = np.array([np.sqrt(2.0), np.sqrt(0.5)])
    assert np.allclose(sorted(scales, reverse=True), sorted(expected_scales, reverse=True), atol=1e-6)