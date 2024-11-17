import pickle
import torch

def load_gaussians(pickle_path: str) -> tuple:
    """
    Load Gaussian data, view matrix, and camera intrinsic matrix from a pickle file.

    Args:
        pickle_path (str): Path to the pickle file

    Returns:
        tuple: (original_gaussians, projected_gaussians, viewmat, K)
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        original_gaussians = data["original_gaussians"]
        projected_gaussians = data["projected_gaussians"]
        viewmat = data["viewmat"]
        K = data["K"]
    return original_gaussians, projected_gaussians, viewmat, K

def load_gaussians_torch(pickle_path: str, device: torch.device) -> tuple:
    """Load Gaussian mixtures from pickle file and convert to PyTorch tensors."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        original_gaussians = data["original_gaussians"]
        projected_gaussians = data["projected_gaussians"]
        viewmat = data["viewmat"]
        K = data["K"]

    # Convert numpy arrays to torch tensors
    projected_gaussians.means = torch.tensor(projected_gaussians.means, dtype=torch.float32, device=device)
    projected_gaussians.scales = torch.tensor(projected_gaussians.scales, dtype=torch.float32, device=device)
    projected_gaussians.rotations = torch.tensor(projected_gaussians.rotations, dtype=torch.float32, device=device)
    projected_gaussians.rgb = torch.tensor(projected_gaussians.rgb, dtype=torch.float32, device=device)
    projected_gaussians.alpha = torch.tensor(projected_gaussians.alpha, dtype=torch.float32, device=device)
    if hasattr(projected_gaussians, 'covs'):
        projected_gaussians.covs = torch.tensor(projected_gaussians.covs, dtype=torch.float32, device=device)

    return original_gaussians, projected_gaussians, viewmat, K