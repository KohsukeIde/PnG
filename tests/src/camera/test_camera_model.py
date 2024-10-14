import torch
from src.camera.camera_model import CameraModel

def test_camera_model_initialization():
    intrinsics = torch.tensor([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=torch.float32)
    extrinsics = torch.eye(4)
    camera_model = CameraModel(intrinsics, extrinsics)
    
    assert isinstance(camera_model, CameraModel)
    assert camera_model.intrinsics.shape == (3, 3)
    assert camera_model.extrinsics.shape == (4, 4)

def test_camera_model_initialization_invalid_input():
    try:
        CameraModel(torch.rand(2, 2), torch.eye(4))
        raise AssertionError("ValueError was not raised")
    except ValueError:
        pass

    try:
        CameraModel(torch.rand(3, 3), torch.eye(3))
        raise AssertionError("ValueError was not raised")
    except ValueError:
        pass

def test_project_3d_to_2d():
    intrinsics = torch.tensor([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=torch.float32)
    extrinsics = torch.eye(4)
    camera_model = CameraModel(intrinsics, extrinsics)
    
    points_3d = torch.tensor([
        [1, 0, 5],
        [0, 1, 5],
        [-1, -1, 5]
    ], dtype=torch.float32)
    points_2d = camera_model.project_3d_to_2d(points_3d)
    assert points_2d.shape == (3, 2)

def test_back_project_2d_to_3d():
    intrinsics = torch.tensor([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=torch.float32)
    extrinsics = torch.eye(4)
    camera_model = CameraModel(intrinsics, extrinsics)
    
    points_2d = torch.tensor([
        [600, 500],
        [500, 600],
        [400, 400]
    ], dtype=torch.float32)
    depth = torch.tensor([5, 5, 5], dtype=torch.float32)
    points_3d = camera_model.back_project_2d_to_3d(points_2d, depth)
    assert points_3d.shape == (3, 3)

def test_projection_backprojection_consistency():
    intrinsics = torch.tensor([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=torch.float32)
    extrinsics = torch.eye(4)
    camera_model = CameraModel(intrinsics, extrinsics)
    
    points_3d = torch.tensor([
        [1, 0, 5],
        [0, 1, 5],
        [-1, -1, 5]
    ], dtype=torch.float32)
    points_2d = camera_model.project_3d_to_2d(points_3d)
    depth = points_3d[:, 2]
    points_3d_recovered = camera_model.back_project_2d_to_3d(points_2d, depth)
    assert torch.allclose(points_3d, points_3d_recovered, atol=1e-5)

def test_update_extrinsics():
    intrinsics = torch.tensor([
        [1000, 0, 500],
        [0, 1000, 500],
        [0, 0, 1]
    ], dtype=torch.float32)
    extrinsics = torch.eye(4)
    camera_model = CameraModel(intrinsics, extrinsics)
    
    rotation = torch.tensor([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    translation = torch.tensor([1, 2, 3], dtype=torch.float32)
    camera_model.update_extrinsics(rotation, translation)
    assert torch.allclose(camera_model.extrinsics[:3, :3], rotation)
    assert torch.allclose(camera_model.extrinsics[:3, 3], translation)

def test_gradients():
    intrinsics = torch.rand(3, 3, requires_grad=True)
    extrinsics = torch.eye(4, requires_grad=True)
    camera_model = CameraModel(intrinsics, extrinsics)
    points_3d = torch.rand(10, 3, requires_grad=True)
    points_2d = camera_model.project_3d_to_2d(points_3d)
    loss = points_2d.sum()
    loss.backward()
    assert points_3d.grad is not None
    assert intrinsics.grad is not None
    assert extrinsics.grad is not None


# if torch.cuda.is_available():
#     def test_gpu_compatibility():
#         intrinsics = torch.rand(3, 3).cuda()
#         extrinsics = torch.eye(4).cuda()
#         camera_model = CameraModel(intrinsics, extrinsics)
#         points_3d = torch.rand(10, 3).cuda()
#         points_2d = camera_model.project_3d_to_2d(points_3d)
#         assert points_2d.is_cuda

def test_mps_compatibility():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        intrinsics = torch.rand(3, 3).to(device)
        extrinsics = torch.eye(4).to(device)
        camera_model = CameraModel(intrinsics, extrinsics)
        points_3d = torch.rand(10, 3).to(device)
        points_2d = camera_model.project_3d_to_2d(points_3d)
        assert points_2d.device == device
    else:
        print("MPS not available, skipping MPS compatibility test")