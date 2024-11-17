import torch

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.primitive.twod_gaussians_rs import TwoDGaussians


def generate_covariances_from_rotations_and_scales(rotations, scales):
    """Generate covariance matrices from rotations and scales.

    Args:
        rotations (torch.Tensor): Rotation angles in radians, shape (k,)
        scales (torch.Tensor): Scales, shape (k, 2)

    Returns:
        covs (torch.Tensor): Covariance matrices, shape (k, 2, 2)
    """
    k = rotations.shape[0]
    device = rotations.device
    covs = torch.zeros((k, 2, 2), device=device)
    for i in range(k):
        theta = rotations[i]
        s = scales[i]
        cos_r = torch.cos(theta)
        sin_r = torch.sin(theta)
        r = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]], device=device)
        s = torch.diag(s**2)
        covs[i] = r @ s @ r.T
    return covs


def test_optimal_transport_solver():
    """Test basic functionality of OptimalTransportSolver."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sample data
    means1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    means2 = torch.tensor([[0.1, 0.1], [1.1, 1.1]], device=device)
    scales = torch.ones((2, 2), device=device)
    rotations = torch.zeros(2, device=device)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    alpha = torch.ones(2, device=device)

    covs1 = generate_covariances_from_rotations_and_scales(rotations, scales)
    covs2 = generate_covariances_from_rotations_and_scales(rotations, scales)

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        epsilon=0.1,
        lambda_mean=0.3,
        lambda_color=1.0,
        lambda_cov=1.0,
        device=device
    )

    assert solver.H is None
    assert solver.epsilon == 0.1
    assert solver.lambda_mean == 0.3
    assert solver.lambda_color == 1.0
    assert solver.lambda_cov == 1.0
    assert solver.device == device


def test_cost_matrix_computation():
    """Test cost matrix computation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create larger test case to avoid dimension issues
    means1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    means2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]], device=device)
    scales = torch.ones((2, 2), device=device)
    rotations = torch.zeros(2, device=device)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    alpha = torch.ones(2, device=device)

    covs1 = generate_covariances_from_rotations_and_scales(rotations, scales)
    covs2 = generate_covariances_from_rotations_and_scales(rotations, scales)

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        epsilon=0.1,
        device=device
    )

    H = torch.eye(3, device=device)
    cost_matrix = solver.compute_cost_matrix(F)

    assert cost_matrix.shape == (2, 2)
    assert cost_matrix.device == device
    assert torch.is_tensor(cost_matrix)


def test_homography_optimization():
    """Test homography optimization process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    means1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device)
    means2 = means1 + 0.1  # Add small translation
    scales = torch.ones((2, 2), device=device)
    rotations = torch.zeros(2, device=device)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    alpha = torch.ones(2, device=device)

    covs1 = generate_covariances_from_rotations_and_scales(rotations, scales)
    covs2 = generate_covariances_from_rotations_and_scales(rotations, scales)

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha
    )

    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        epsilon=0.1,
        device=device
    )

    solver.optimize_with_homography(max_iter=10)

    assert solver.H is not None
    assert torch.is_tensor(solver.H)
    assert solver.H.shape == (3, 3)
    assert solver.H.device == device
    assert solver.H.requires_grad == False  # After optimization, F should be detached