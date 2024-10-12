import pickle
import sys
from src.evaluator.matching_evaluator import MatchingEvaluator
from src.optimizer.optimal_transport_solver_rs import OptimalTransportSolver
import time

sys.modules['twodgs'] = sys.modules['src.primitive.twod_gaussians']

def load_gaussians(file_path: str) -> tuple:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        gaussians = data["gaussians"]
        viewmat = data["viewmat"]
        K = data["K"]
    return gaussians, viewmat, K

def main():
    print("Loading Gaussians...")
    gaussians1, _, _ = load_gaussians('/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/fitted_gaussians_22_1k.pkl')
    gaussians2, _, _ = load_gaussians('/Users/kohsukeide/dev/perspective-n-gaussian/data/fitted_gs/fitted_gaussians_23_1k.pkl')
    print("Gaussians loaded.")
    print(f"Number of Gaussians in gaussians1: {gaussians1.k}")
    print(f"Number of Gaussians in gaussians2: {gaussians2.k}")

    print("Initializing OptimalTransportSolver...")
    solver = OptimalTransportSolver(gaussians1, gaussians2)

    print("Computing cost matrix...")
    start_time = time.time()
    cost_matrix = solver.compute_cost_matrix()
    end_time = time.time()
    print(f"Cost matrix computed in {end_time - start_time:.2f} seconds.")

    print("Computing transport matrix using Sinkhorn algorithm...")
    start_time = time.time()
    transport_matrix = solver.sinkhorn_algorithm(cost_matrix)
    end_time = time.time()
    print(f"Transport matrix computed in {end_time - start_time:.2f} seconds.")

    print("Initializing MatchingEvaluator...")
    evaluator = MatchingEvaluator(gaussians1, gaussians2, transport_matrix)

    print("Evaluating matches...")
    metrics = evaluator.evaluate_matches()
    print("Matching Metrics:", metrics)

    print("Visualizing matches...")
    evaluator.visualize_matches('/Users/kohsukeide/dev/perspective-n-gaussian/outputs/matching_visualization.png')
    print("Matching visualization saved to 'outputs/matching_visualization.png'.")

if __name__ == "__main__":
    main()