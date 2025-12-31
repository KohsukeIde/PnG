#!/usr/bin/env python3
"""
Batch processing script for running Gaussian Mixture Model EM on multiple images.
Processes all images in a directory and saves only essential outputs.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pipelines.image_to_gmm_em import run_gaussian_mixture_on_image
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)


def process_images_batch(
    input_dir: str,
    output_base_dir: str,
    k_list: str = "200",
    max_iterations: int = 500,
    tol: float = 1e-4,
    min_iterations: int = 5,
    init_mode: str = "grid",
    mse_tol: float = None,
):
    """
    Process all images in input_dir and save results to output_base_dir.
    
    Args:
        input_dir: Directory containing input images
        output_base_dir: Base directory for output (e.g., data/fitted_gs)
        k_list: Comma-separated list of K values
        max_iterations: Maximum EM iterations
        tol: Relative NLL improvement threshold
        min_iterations: Minimum iterations before early stopping
        init_mode: Initialization mode (grid|random)
        mse_tol: Optional MSE tolerance for early stopping
    """
    # Convert to absolute paths
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(project_root, input_dir)
    if not os.path.isabs(output_base_dir):
        output_base_dir = os.path.join(project_root, output_base_dir)
    
    # Parse k_list
    k_values = [int(k.strip()) for k in k_list.split(",") if k.strip()]
    
    # Get base name from input directory
    # For NeRF synthetic: data/nerf_synthetic/chair/train -> "chair_train"
    # For DTU: data/DTU/scan63/images_resized -> "scan63_images_resized"
    input_dir_path = Path(input_dir)
    parts = input_dir_path.parts
    
    # Try to extract scene name and split name
    # Check if this looks like NeRF synthetic structure (has "nerf_synthetic" in path)
    if "nerf_synthetic" in parts:
        # Find index of "nerf_synthetic"
        nerf_idx = parts.index("nerf_synthetic")
        if len(parts) > nerf_idx + 2:
            # Has scene name and split name: .../nerf_synthetic/scene/split
            scene_name = parts[nerf_idx + 1]
            split_name = parts[nerf_idx + 2]
            base_name = f"{scene_name}_{split_name}"
        elif len(parts) > nerf_idx + 1:
            # Only scene name: .../nerf_synthetic/scene
            base_name = parts[nerf_idx + 1]
        else:
            base_name = input_dir_path.name
    else:
        # For other structures, use parent name if directory name is generic
        base_name = input_dir_path.name
        if base_name in ["images_resized", "images", "train", "val", "test"]:
            parent_name = input_dir_path.parent.name
            if parent_name:
                base_name = f"{parent_name}_{base_name}"
    
    # Find all image files
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Base name: {base_name}")
    
    # Process each K value
    for k in k_values:
        # Create main output directory: output_base_dir/{base_name}_em_{k}/
        main_output_dir = os.path.join(output_base_dir, f"{base_name}_em_{k}")
        os.makedirs(main_output_dir, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"Processing with K={k}")
        print(f"Main output directory: {main_output_dir}")
        print(f"{'='*80}")
        
        # Process each image
        for image_path in image_files:
            image_name = Path(image_path).stem  # filename without extension
            
            print(f"\nProcessing: {image_name}")
            
            # Create output directory: main_output_dir/{image_name}/
            output_dir = os.path.join(main_output_dir, image_name)
            
            # Run EM algorithm (save_intermediate=False to skip intermediate visualizations)
            try:
                run_gaussian_mixture_on_image(
                    image_path=image_path,
                    n_gaussians=k,
                    max_iterations=max_iterations,
                    tol=tol,
                    min_iterations=min_iterations,
                    init_mode=init_mode,
                    mse_tol=mse_tol,
                    mask_path=None,
                    output_dir=output_dir,
                    save_intermediate=False,  # Skip intermediate visualizations
                )
                
                # Clean up: keep only essential files
                # Essential files:
                # - reconstructed_image.png (final rendering)
                # - curve_nll.png
                # - curve_mse.png
                # - curve_rates_max.png
                # - curve_alpha_alive.png
                # - gaussians.pkl
                # - metrics.json
                # - poisson_nll.txt
                
                essential_files = [
                    "reconstructed_image.png",
                    "curve_nll.png",
                    "curve_mse.png",
                    "curve_rates_max.png",
                    "curve_alpha_alive.png",
                    "gaussians.pkl",
                    "metrics.json",
                    "poisson_nll.txt",
                ]
                
                # Remove non-essential files
                all_files = os.listdir(output_dir)
                for file in all_files:
                    file_path = os.path.join(output_dir, file)
                    if file not in essential_files:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                
                print(f"✓ Completed: {image_name}")
                print(f"  Output saved to: {output_dir}")
                
            except Exception as e:
                print(f"✗ Error processing {image_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print(f"Batch processing completed!")
    print(f"Results saved to: {output_base_dir}")
    for k in k_values:
        main_output_dir = os.path.join(output_base_dir, f"{base_name}_em_{k}")
        print(f"  K={k}: {main_output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process images with Gaussian Mixture Model EM"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="data/fitted_gs",
        help="Base directory for output (default: data/fitted_gs)",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default="200",
        help="Comma-separated list of K values (e.g., '200' or '150,200,250')",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Maximum EM iterations",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Relative NLL improvement threshold for early stopping",
    )
    parser.add_argument(
        "--min_iterations",
        type=int,
        default=5,
        help="Minimum iterations before checking early stopping",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default="grid",
        help="Initialization mode for Gaussians (grid|random)",
    )
    parser.add_argument(
        "--mse_tol",
        type=float,
        default=None,
        help="Optional relative MSE improvement threshold for early stopping",
    )
    
    args = parser.parse_args()
    
    process_images_batch(
        input_dir=args.input_dir,
        output_base_dir=args.output_base_dir,
        k_list=args.k_list,
        max_iterations=args.max_iterations,
        tol=args.tol,
        min_iterations=args.min_iterations,
        init_mode=args.init_mode,
        mse_tol=args.mse_tol,
    )




