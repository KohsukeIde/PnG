
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


def load_image_as_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    """
    Loads an image from disk, converts to float32 in [0..1], shape [H, W, 3].
    """
    img = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(img)  # shape [3, H, W] in [0..1]
    tensor = tensor.permute(1, 2, 0)  # shape [H, W, 3]
    return tensor.to(device)


def generate_synthetic_image(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Generates a synthetic 2D image: top-left = red, bottom-right = blue, rest = white.
    shape: [H, W, 3], dtype float32 in [0..1].
    """
    img = torch.ones((height, width, 3), device=device)
    half_h, half_w = height // 2, width // 2
    img[:half_h, :half_w, :] = torch.tensor([1.0, 0.0, 0.0], device=device)  # red
    img[half_h:, half_w:, :] = torch.tensor([0.0, 0.0, 1.0], device=device)  # blue
    return img


def build_covariance_matrices(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Given per-Gaussian scales=[N,2], rotations=[N],
    return covariance matrices covs=[N, 2, 2].
    """
    # rotations in radians
    cos_r = torch.cos(rotations)
    sin_r = torch.sin(rotations)

    # R = [[ cos_r, -sin_r],
    #      [ sin_r,  cos_r]]
    R = torch.stack([
        torch.stack([cos_r, -sin_r], dim=1),
        torch.stack([sin_r,  cos_r], dim=1),
    ], dim=1)  # shape [N, 2, 2]

    # S = diag(scales_x, scales_y)
    # scales shape [N, 2]
    S = torch.zeros((scales.shape[0], 2, 2), device=scales.device)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]

    # covariance = R @ S^2 @ R^T
    # (We treat scale as standard deviations, so we square them.)
    S_sq = S @ S  # or we can do S.pow(2) if scales are diagonal
    covs = R.bmm(S_sq).bmm(R.transpose(1, 2))  # shape [N, 2, 2]
    return covs


def render_2d_gaussians(
    means: torch.Tensor,      # [N, 2]
    covs: torch.Tensor,       # [N, 2, 2]
    rgbs: torch.Tensor,       # [N, 3]  in [-∞..+∞], we typically apply sigmoid
    alphas: torch.Tensor,     # [N]     in [-∞..+∞], also typically sigmoid
    H: int,
    W: int
) -> torch.Tensor:
    """
    Renders N 2D Gaussians onto an [H,W,3] image using alpha compositing.
    Means, covs are in pixel coordinates:
      - means[i] = (x_i, y_i)
      - covs[i] is 2x2 covariance matrix
    All shapes are torch tensors on the same device.

    Returns: final image, shape [H, W, 3], in [0..1].
    """

    device = means.device

    # Build a meshgrid of pixel coordinates, shape [H, W, 2].
    #   px_grid[y, x] = [x, y]
    # We'll keep them as floats.
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='xy')  # shape [H, W]

    # We'll reshape to [H*W, 2] so we can batch-process
    px_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W, 2]

    # Prepare accumulators for color & alpha
    accum_img   = torch.zeros(H, W, 3, device=device)  # [H, W, 3]
    accum_alpha = torch.zeros(H, W, device=device)      # [H, W]

    # For numerical stability, we might want to invert covariance once per Gaussian
    # cov_inv: [N, 2, 2], log_det = scalar if we want actual PDF normalization
    # but we might skip the normalizing constant if we just want a fuzzy shape.
    # We'll do the full PDF approach (with the 1/(2*pi sqrt(detSigma)) factor).

    # Precompute inv_cov and normalizing constant
    inv_covs = torch.inverse(covs)  # [N,2,2]
    # Determinant of each 2x2
    dets = covs[:, 0, 0]*covs[:, 1, 1] - covs[:, 0, 1]*covs[:, 1, 0]  # shape [N]
    # normalizing constant = 1 / (2*pi * sqrt(det))
    two_pi = 2.0 * math.pi
    norm_consts = 1.0 / (two_pi * dets.clamp_min(1e-12).sqrt())

    # We'll do a simple loop over N Gaussians.
    # If N is large, this can be slow, but it's simpler to illustrate.
    N = means.shape[0]
    for i in range(N):
        mean_i = means[i]  # shape [2]
        inv_cov_i = inv_covs[i]  # shape [2, 2]
        norm_i = norm_consts[i]
        color_i = torch.sigmoid(rgbs[i])   # shape [3] in [0..1]
        alpha_i = torch.sigmoid(alphas[i]) # scalar in [0..1]

        # We compute a 2D PDF value for each pixel: pdf_i(px)
        # pdf(px) = norm_i * exp(-0.5 * (px - mean_i)^T inv_cov_i (px - mean_i))
        # We'll do that in a vectorized manner for all px in [H*W,2].

        px_minus_mean = px_coords - mean_i[None, :]  # shape [H*W, 2]
        # matmul for each row: (px_minus_mean) * inv_cov_i * (px_minus_mean^T)
        # => can do (px_minus_mean @ inv_cov_i) => shape [H*W, 2]
        # then elementwise multiply by px_minus_mean and sum
        temp = px_minus_mean @ inv_cov_i           # [H*W, 2]
        quad_form = torch.sum(temp * px_minus_mean, dim=1)  # shape [H*W]

        pdf_vals = norm_i * torch.exp(-0.5 * quad_form)  # shape [H*W]

        # We'll treat pdf_vals as a "coverage" or "intensity" for that gaussian.
        # Then final alpha = alpha_i * coverage. (like "opacity * brightness")
        blend_alpha = alpha_i * pdf_vals  # shape [H*W]

        # Now we do standard "over" alpha compositing in a differentiable way:
        # out_color = accum_color + (1 - accum_alpha)*blend_alpha*color_i
        # out_alpha = accum_alpha + (1 - accum_alpha)*blend_alpha
        blend_alpha_2d = blend_alpha.reshape(H, W)
        one_minus_accum_alpha = (1.0 - accum_alpha)

        accum_img += one_minus_accum_alpha.unsqueeze(-1) * blend_alpha_2d.unsqueeze(-1) * color_i
        accum_alpha += one_minus_accum_alpha * blend_alpha_2d

    # accum_img is now the result of alpha compositing all gaussians
    # in shape [H, W, 3]. Values *should* be in [0..1], but can exceed that if alpha_i is large or many overlaps.
    # We can clamp if needed:
    out_img = accum_img.clamp(0.0, 1.0)

    return out_img


class GaussianFitter(nn.Module):
    """
    Simple class that:
      1) Holds the parameters of the Gaussians: means, scales, rotations, rgbs, alphas
      2) Renders them to an image
      3) Compares to a target image via MSE.
    """

    def __init__(self, num_points: int, height: int, width: int, device: torch.device):
        super().__init__()
        self.num_points = num_points
        self.H = height
        self.W = width
        self.device = device

        # We'll store these as nn.Parameters so they are trainable by PyTorch
        # Initialize randomly or heuristically
        # means in [0..W], [0..H] => scale them from uniform[-1..1] to [0..W, 0..H]
        means_init = torch.rand(num_points, 2, device=device)
        means_init[:, 0] *= float(self.W)  # x
        means_init[:, 1] *= float(self.H)  # y

        self.means = nn.Parameter(means_init)

        # scales in [1..10], for instance:
        scales_init = 2.0 * torch.rand(num_points, 2, device=device) + 1.0
        self.scales = nn.Parameter(scales_init)

        # rotations in [0..2pi]
        rotations_init = 2.0 * math.pi * torch.rand(num_points, device=device)
        self.rotations = nn.Parameter(rotations_init)

        # color, alpha in [-4..+4] as raw, then we apply sigmoid in rendering
        # so final color is in [0..1], alpha in [0..1]
        self.rgbs = nn.Parameter(torch.zeros(num_points, 3, device=device).normal_(0.0, 2.0))
        self.alphas = nn.Parameter(torch.zeros(num_points, device=device).normal_(0.0, 2.0))

    def forward(self) -> torch.Tensor:
        """
        Render the current Gaussians to an image [H, W, 3].
        """
        # build covariance
        covs = build_covariance_matrices(self.scales, self.rotations)
        img = render_2d_gaussians(
            means=self.means,
            covs=covs,
            rgbs=self.rgbs,
            alphas=self.alphas,
            H=self.H,
            W=self.W,
        )
        return img


def main():
    parser = argparse.ArgumentParser("Fitting a 2D image with parametric Gaussians (pure PyTorch).")
    parser.add_argument("--image_path", type=str, default=None, help="Path to target image. If omitted, a synthetic image is used.")
    parser.add_argument("--height", type=int, default=128, help="Height if synthetic.")
    parser.add_argument("--width", type=int, default=128, help="Width if synthetic.")
    parser.add_argument("--num_points", type=int, default=100, help="Number of Gaussians.")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--out", type=str, default="gaussians_fit.png", help="Output image filename.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or generate target image
    if args.image_path:
        gt_image = load_image_as_tensor(Path(args.image_path), device=device)
        H, W = gt_image.shape[0], gt_image.shape[1]
    else:
        H, W = args.height, args.width
        gt_image = generate_synthetic_image(H, W, device=device)

    # Create model
    model = GaussianFitter(args.num_points, H, W, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Training loop
    losses = []
    t_start = time.time()
    for step in range(args.iterations):
        optimizer.zero_grad()
        rendered = model()      # shape [H, W, 3]
        loss = loss_fn(rendered, gt_image)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step+1) % 100 == 0 or step == 0:
            print(f"Iteration {step+1}/{args.iterations}, Loss: {loss.item():.6f}")

    t_end = time.time()
    print(f"Training completed in {t_end - t_start:.2f} seconds.")

    # Save final rendered image
    final_render = model().detach().cpu().numpy()  # shape [H, W, 3], in [0..1]
    final_render = (final_render * 255).clip(0, 255).astype(np.uint8)
    out_img_pil = Image.fromarray(final_render)
    out_img_pil.save(args.out)
    print(f"Saved final fitting result: {args.out}")

    # Optional: Plot the loss curve
    plt.figure(figsize=(8,4))
    plt.plot(losses, label="MSE Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()
    print("Saved loss curve: loss_curve.png")


if __name__ == "__main__":
    main()
