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
        
    print("=== Projected Gaussians Statistics ===")
    print(f"means: min={projected_gaussians.means.min().item()}, max={projected_gaussians.means.max().item()}, "
          f"mean={projected_gaussians.means.mean().item()}, std={projected_gaussians.means.std().item()}")
    print(f"scales: min={projected_gaussians.scales.min().item()}, max={projected_gaussians.scales.max().item()}, "
          f"mean={projected_gaussians.scales.mean().item()}, std={projected_gaussians.scales.std().item()}")
    print(f"rotations: min={projected_gaussians.rotations.min().item()}, max={projected_gaussians.rotations.max().item()}, "
          f"mean={projected_gaussians.rotations.mean().item()}, std={projected_gaussians.rotations.std().item()}")
    print(f"rgb: min={projected_gaussians.rgb.min().item()}, max={projected_gaussians.rgb.max().item()}, "
          f"mean={projected_gaussians.rgb.mean().item()}, std={projected_gaussians.rgb.std().item()}")
    print(f"alpha: min={projected_gaussians.alpha.min().item()}, max={projected_gaussians.alpha.max().item()}, "
          f"mean={projected_gaussians.alpha.mean().item()}, std={projected_gaussians.alpha.std().item()}")
    if hasattr(projected_gaussians, 'covs'):
        print(f"covs: min={projected_gaussians.covs.min().item()}, max={projected_gaussians.covs.max().item()}, "
              f"mean={projected_gaussians.covs.mean().item()}, std={projected_gaussians.covs.std().item()}")
    print("======================================")

    return original_gaussians, projected_gaussians, viewmat, K