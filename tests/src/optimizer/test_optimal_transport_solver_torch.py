import torch
import numpy as np

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


# def test_optimal_transport_solver_homography():
#     """Test OptimalTransportSolver functionality including initialization, cost computation, and optimization."""
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Create test data with smaller values and simpler configuration
#     means1 = torch.tensor([[0.0, 0.0], [0.5, 0.5]], device=device)
#     means2 = means1 + 0.1  # Small translation

#     # Use smaller scales to prevent numerical issues
#     scales = 0.1 * torch.ones((2, 2), device=device)
#     rotations = torch.zeros(2, device=device)

#     # Simpler RGB values
#     rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
#     alpha = torch.ones(2, device=device)

#     covs1 = generate_covariances_from_rotations_and_scales(rotations, scales)
#     covs2 = generate_covariances_from_rotations_and_scales(rotations, scales)

#     gaussians1 = TwoDGaussians(
#         means=means1,
#         covs=covs1,
#         scales=scales,
#         rotations=rotations,
#         rgb=rgb,
#         alpha=alpha,
#     )
#     gaussians2 = TwoDGaussians(
#         means=means2,
#         covs=covs2,
#         scales=scales,
#         rotations=rotations,
#         rgb=rgb,
#         alpha=alpha,
#     )

#     # Initialize solver with adjusted parameters
#     solver = OptimalTransportSolver(
#         gaussians1=gaussians1,
#         gaussians2=gaussians2,
#         epsilon=1.0,  # Increased epsilon for better numerical stability
#         lambda_mean=0.3,
#         lambda_color=0.3,
#         lambda_cov=0.3,
#         device=device,
#     )

#     # Test initialization
#     assert solver.h is None
#     assert solver.epsilon == 1.0
#     assert solver.device == device

#     # Test cost matrix computation
#     h = torch.eye(3, device=device)
#     cost_matrix = solver.compute_cost_matrix(h)
#     assert cost_matrix.shape == (2, 2)
#     assert cost_matrix.device == device
#     assert not torch.isnan(cost_matrix).any()  # Check for NaN values

#     # Test optimization with more iterations but smaller learning rate
#     solver.optimize_with_homography(max_iter=20, tol=1e-4)
#     assert solver.h is not None
#     assert solver.h.shape == (3, 3)
#     assert not solver.h.requires_grad
#     assert not torch.isnan(solver.h).any()  # Check for NaN values
    
def test_optimal_transport_solver_fundamental():
    """Test OptimalTransportSolver functionality with fundamental matrix optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test data with smaller values and simpler configuration
    means1 = torch.tensor([[0.0, 0.0], [0.5, 0.5]], device=device)
    means2 = means1 + 0.1  # Small translation

    # Use smaller scales to prevent numerical issues
    scales = 0.1 * torch.ones((2, 2), device=device)
    rotations = torch.zeros(2, device=device)

    # Simpler RGB values
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    alpha = torch.ones(2, device=device)

    covs1 = generate_covariances_from_rotations_and_scales(rotations, scales)
    covs2 = generate_covariances_from_rotations_and_scales(rotations, scales)

    # Create simple camera intrinsics (focal=1.0, principal point at center)
    k1 = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    k2 = k1.copy()

    gaussians1 = TwoDGaussians(
        means=means1,
        covs=covs1,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha,
    )
    gaussians2 = TwoDGaussians(
        means=means2,
        covs=covs2,
        scales=scales,
        rotations=rotations,
        rgb=rgb,
        alpha=alpha,
    )

    # Initialize solver with adjusted parameters
    solver = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=k1,
        k2=k2,
        epsilon=1.0,  # Increased epsilon for better numerical stability
        lambda_mean=0.3,
        lambda_color=0.3,
        lambda_cov=0.3,
        lambda_epipolar=0.3,
        device=device,
    )

    # Test initialization
    assert solver.f is None
    assert solver.epsilon == 1.0
    assert solver.device == device
    assert torch.allclose(solver.k1, torch.tensor(k1, dtype=torch.float32, device=device))
    assert torch.allclose(solver.k2, torch.tensor(k2, device=device))

    # Test fundamental matrix optimization
    solver.optimize_with_fundamental(max_iter=20, tol=1e-4)
    assert solver.f is not None
    assert solver.f.shape == (3, 3)
    assert not solver.f.requires_grad
    assert not torch.isnan(solver.f).any()  # Check for NaN values

    # Verify rank-2 constraint
    u, s, vh = torch.linalg.svd(solver.f)
    assert torch.abs(s[-1]) < 1e-6  # Last singular value should be close to zero

    # Test R,t optimization
    solver_rt = OptimalTransportSolver(
        gaussians1=gaussians1,
        gaussians2=gaussians2,
        k1=k1,
        k2=k2,
        epsilon=1.0,
        lambda_mean=0.3,
        lambda_color=0.3,
        lambda_cov=0.3,
        lambda_epipolar=0.3,
        device=device,
    )

    solver_rt.optimize_with_RT(max_iter=20, tol=1e-4)
    
    # Test results
    assert solver_rt.f is not None
    assert solver_rt.f.shape == (3, 3)
    assert not solver_rt.f.requires_grad
    assert not torch.isnan(solver_rt.f).any()
    
    # Check that rvec and tvec exist and have correct shapes
    assert hasattr(solver_rt, 'rvec')
    assert hasattr(solver_rt, 'tvec')
    assert solver_rt.rvec.shape == (3,)
    assert solver_rt.tvec.shape == (3,)

    # Test cost matrix computation with fundamental matrix
    cost_matrix = solver.compute_cost_matrix_fundamental(solver.f)
    assert cost_matrix.shape == (2, 2)
    assert cost_matrix.device == device
    assert not torch.isnan(cost_matrix).any()

    # Test Sampson error-based cost matrix
    cost_matrix_sampson = solver.compute_cost_matrix_fundamental_sampson(solver.f)
    assert cost_matrix_sampson.shape == (2, 2)
    assert cost_matrix_sampson.device == device
    assert not torch.isnan(cost_matrix_sampson).any()
