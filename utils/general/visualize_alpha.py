#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def visualize_alpha_distribution(image_path: str, out_path: str = "alpha_visual.png"):
    """
    1) Load image as RGBA
    2) Convert to Tensor ([4,H,W]) using torchvision.transforms and extract alpha channel ([H,W])
    3) Display and save as grayscale using Matplotlib
    """
    path_obj = Path(image_path)
    if not path_obj.exists():
        print(f"Error: File not found: {image_path}")
        return

    # 1) Load RGBA with PIL
    pil_img = Image.open(path_obj).convert("RGBA")

    # 2) Convert to PyTorch tensor → shape [4,H,W]
    t = transforms.ToTensor()(pil_img)
    # t[3,:,:] is alpha component → shape [H,W]
    alpha = t[3,:,:]

    # 3) Convert alpha to numpy & display as grayscale
    alpha_np = alpha.cpu().numpy()  # shape [H,W], range [0..1]

    plt.figure(figsize=(6,6))
    plt.imshow(alpha_np, cmap='gray', vmin=0.0, vmax=1.0)
    plt.colorbar(label="Alpha")
    plt.title("Alpha Channel Visualization")
    plt.savefig(out_path)
    plt.close()

    print(f"Alpha map visualization saved to {out_path}")
    print(f"Alpha shape: {alpha_np.shape}, min={alpha_np.min():.3f}, max={alpha_np.max():.3f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_alpha.py <path_to_rgba_image> [output_png]")
        sys.exit(1)

    image_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    else:
        out_path = "alpha_visual.png"

    visualize_alpha_distribution(image_path, out_path)

if __name__ == "__main__":
    main()
