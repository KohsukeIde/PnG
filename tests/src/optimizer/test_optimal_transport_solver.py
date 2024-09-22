import numpy as np

from src.optimizer.optimal_transport_solver import OptimalTransportSolver
from src.primitive.twod_gaussians import TwoDGaussians


def test_optimal_transport_solver_initialization():
    """Test the initialization of OptimalTransportSolver."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=np.random.rand(7, 2, 2),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2, epsilon=0.1)
    assert isinstance(solver, OptimalTransportSolver)


def test_compute_cost_matrix():
    """Test the compute_cost_matrix method of OptimalTransportSolver."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=np.random.rand(7, 2, 2),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    cost_matrix = solver.compute_cost_matrix()
    assert cost_matrix.shape == (5, 7)


def test_sinkhorn_algorithm():
    """Test the sinkhorn_algorithm method of OptimalTransportSolver."""
    gaussians1 = TwoDGaussians(
        means=np.random.rand(5, 2),
        covs=np.random.rand(5, 2, 2),
        rgb=np.random.rand(5, 3),
        alpha=np.random.rand(5),
    )
    gaussians2 = TwoDGaussians(
        means=np.random.rand(7, 2),
        covs=np.random.rand(7, 2, 2),
        rgb=np.random.rand(7, 3),
        alpha=np.random.rand(7),
    )

    solver = OptimalTransportSolver(gaussians1, gaussians2)
    transport_matrix = solver.sinkhorn_algorithm()
    assert transport_matrix.shape == (5, 7)
