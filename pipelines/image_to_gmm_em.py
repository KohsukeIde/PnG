import os
import sys
import shutil 
import numpy as np
from typing import Optional
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.optimizer.single_image_gaussian_mixture_em import SingleImageGaussianMixtureEM
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

def visualize_gaussians(image, gaussians, iteration, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    for mean, cov in zip(gaussians.means, gaussians.covs):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * np.sqrt(eigenvalues)
        ellipse = Ellipse(xy=mean[::-1], width=width, height=height, 
                          angle=angle, fill=False, edgecolor='r', alpha=0.5)
        ax.add_artist(ellipse)

    plt.title(f'Gaussian Distribution (Iteration {iteration})')
    output_path = os.path.join(output_dir, f'gaussian_distribution_iter_{iteration}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def visualize_responsibilities(responsibilities, iteration, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(np.sum(responsibilities, axis=2), cmap='hot')
    plt.colorbar()
    plt.title(f'Responsibilities Sum (Iteration {iteration})')
    output_path = os.path.join(output_dir, f'responsibilities_sum_iter_{iteration}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def visualize_gaussian_parameters(gaussians, iteration, output_dir, image_height, image_width):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    
    # Means
    axs[0, 0].scatter(gaussians.means[:, 1], gaussians.means[:, 0])
    axs[0, 0].set_title("Gaussian Means")
    axs[0, 0].set_xlim(0, image_width)
    axs[0, 0].set_ylim(image_height, 0)
    
    # Covariances
    cov_det = np.array([np.linalg.det(cov) for cov in gaussians.covs])
    axs[0, 1].hist(cov_det, bins=50)
    axs[0, 1].set_title("Covariance Determinants")
    
    # Weights (alpha)
    axs[1, 0].bar(range(len(gaussians.alpha)), gaussians.alpha)
    axs[1, 0].set_title("Gaussian Weights (alpha)")
    
    # RGB values
    axs[1, 1].scatter(range(gaussians.k), gaussians.rgb[:, 0], c='r', alpha=0.5, label='R')
    axs[1, 1].scatter(range(gaussians.k), gaussians.rgb[:, 1], c='g', alpha=0.5, label='G')
    axs[1, 1].scatter(range(gaussians.k), gaussians.rgb[:, 2], c='b', alpha=0.5, label='B')
    axs[1, 1].set_title("RGB Values")
    axs[1, 1].legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'gaussian_parameters_iter_{iteration}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def visualize_log_likelihood(log_likelihood, iteration, output_dir):
    plt.figure(figsize=(10, 8))
    plt.imshow(log_likelihood, cmap='viridis')
    plt.colorbar()
    plt.title(f'Log Likelihood (Iteration {iteration})')
    output_path = os.path.join(output_dir, f'log_likelihood_iter_{iteration}.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")

def run_gaussian_mixture_on_image(
    image_path: str,
    n_gaussians: int = 300,
    max_iterations: int = 50,
    tol: float = 1e-4,
    min_iterations: int = 5,
    init_mode: str = "grid",
    mse_tol: float = None,
    mask_path: Optional[str] = None,
):
    # Convert relative path to absolute path
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    
    gmm = SingleImageGaussianMixtureEM(image_path, mask_path=mask_path)
    
    base_output_dir = "gaussian_mixture_results"
    output_dir = os.path.join(
        base_output_dir, f"gaussians_{n_gaussians}_auto_{max_iterations}"
    )
    
    # Check if the output directory exists, and if so, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing directory: {output_dir}")
        
    # Recreate the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    print("Image shape:", gmm.image.shape)
    print("Image min-max:", gmm.image.min(), gmm.image.max())
    
    gaussians = gmm.initialize_gaussians(n_gaussians, mode=init_mode)
    
    print("Initial gaussians statistics---------------------------------")
    print(f"Means min-max: {gaussians.means.min()}, {gaussians.means.max()}")
    print(f"Covs min-max: {gaussians.covs.min()}, {gaussians.covs.max()}")
    print(f"RGB min-max: {gaussians.rgb.min()}, {gaussians.rgb.max()}")
    print(f"Alpha min-max: {gaussians.alpha.min()}, {gaussians.alpha.max()}")
    print("------------------------------------------------------------")

    visualize_gaussians(gmm.image, gaussians, 0, output_dir)
    visualize_gaussian_parameters(gaussians, 0, output_dir, gmm.image.shape[0], gmm.image.shape[1])
    
    prev_nll = np.inf
    nll_log = []
    metrics_log = []
    rates_prev = None

    for i in range(max_iterations):
        responsibilities = gmm.e_step(gaussians)
        gaussians = gmm.m_step(responsibilities, gaussians)
        nll = gmm.poisson_nll(gaussians)
        nll_log.append(nll)
        print(f"\nIteration {i+1} completed")
        print(f"Poisson NLL: {nll:.6f}")
        # rates stats for debugging scale
        height, width = gmm.image.shape[:2]
        rasterizer = Vanilla2DRasterizer(height, width)
        rates = rasterizer.render_rates(gaussians)
        rates_min, rates_max, rates_mean = rates.min(), rates.max(), rates.mean()
        mse = np.mean((gmm.image - np.clip(rates, 0.0, 1.0)) ** 2)
        print(f"Rates stats min={rates_min:.3e}, max={rates_max:.3e}, mean={rates_mean:.3e}, MSE={mse:.6f}")
        alpha = gaussians.alpha
        alive = (alpha > 1e-4).sum()
        scales = gaussians.scales
        scale_stats = {
            "sx_median": float(np.median(scales[:, 0])),
            "sy_median": float(np.median(scales[:, 1])),
            "sx_p95": float(np.percentile(scales[:, 0], 95)),
            "sy_p95": float(np.percentile(scales[:, 1], 95)),
            "sx_max": float(scales[:, 0].max()),
            "sy_max": float(scales[:, 1].max()),
        }
        metrics_log.append(
            {
                "iter": i + 1,
                "nll": float(nll),
                "mse": float(mse),
                "rates_min": float(rates_min),
                "rates_max": float(rates_max),
                "rates_mean": float(rates_mean),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "alpha_alive_gt1e4": int(alive),
                **scale_stats,
            }
        )

        visualize_gaussians(gmm.image, gaussians, i+1, output_dir)
        # visualize_responsibilities(responsibilities, i+1, output_dir)
        visualize_gaussian_parameters(gaussians, i+1, output_dir, gmm.image.shape[0], gmm.image.shape[1])
        print(f"Iteration {i+1} - Gaussians RGB min-max: {gaussians.rgb.min()}, {gaussians.rgb.max()}")
        print(f"Iteration {i+1} - Gaussians alpha min-max: {gaussians.alpha.min()}, {gaussians.alpha.max()}")
        
        if (i + 1) % 1 == 0:  # Save every n iterations
            height, width = gmm.image.shape[:2]
            rasterizer = Vanilla2DRasterizer(height, width)
            reconstructed_image = rasterizer.rasterize(gaussians)
            reconstructed_image = reconstructed_image.astype(np.uint8)
            
            mse = np.mean((gmm.image - reconstructed_image/255.0)**2)
            print(f"\nMean Squared Error at Iteration {i+1}: {mse}")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(reconstructed_image)
            ax.set_title(f"Reconstructed Image (Iteration {i+1}, MSE: {mse:.5f})")
            ax.axis('off')
            
            reconstructed_path = os.path.join(output_dir, f"reconstructed_image_iter_{i+1}.png")
            plt.savefig(reconstructed_path)
            plt.close()
            print(f"Reconstructed image saved to {reconstructed_path}")

        # Early stopping based on relative NLL (and optional MSE) improvement
        rel_improve = abs(prev_nll - nll) / max(abs(prev_nll), 1.0)
        prev_nll = nll
        mse_improve_ok = True
        if mse_tol is not None and i > 0:
            prev_mse = np.mean((gmm.image - np.clip(rates_prev, 0.0, 1.0)) ** 2)
            rel_mse_improve = abs(prev_mse - mse) / max(abs(prev_mse), 1.0)
            mse_improve_ok = rel_mse_improve < mse_tol
        rates_prev = rates  # cache for next iter
        if (i + 1) >= min_iterations and rel_improve < tol and mse_improve_ok:
            print(f"Early stop at iter {i+1}: rel_improve={rel_improve:.3e} < tol={tol}"
                  f"{'' if mse_tol is None else f', mse_improve<{mse_tol}'}")
            break

    print("\nFinal Gaussian statistics-------------------------------------")
    print(f"Means min-max: {gaussians.means.min()}, {gaussians.means.max()}")
    print(f"Covs min-max: {gaussians.covs.min()}, {gaussians.covs.max()}")
    print(f"RGB min-max: {gaussians.rgb.min()}, {gaussians.rgb.max()}")
    print(f"Alpha min-max: {gaussians.alpha.min()}, {gaussians.alpha.max()}")
    print("---------------------------------------------------------------")

    height, width = gmm.image.shape[:2]
    rasterizer = Vanilla2DRasterizer(height, width)
    reconstructed_image = rasterizer.rasterize(gaussians)
    reconstructed_image = reconstructed_image.astype(np.uint8)

    mse = np.mean((gmm.image - reconstructed_image/255.0)**2)
    print(f"\nMean Squared Error: {mse}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(gmm.image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(reconstructed_image)
    ax2.set_title(f"Reconstructed Image ({n_gaussians} Gaussians)")
    ax2.axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(comparison_path)
    plt.close()

    reconstructed_path = os.path.join(output_dir, "reconstructed_image.png")
    Image.fromarray(reconstructed_image).save(reconstructed_path)
    print(f"Reconstructed image saved to {reconstructed_path}")

    # Save NLL log and metrics
    np.savetxt(os.path.join(output_dir, "poisson_nll.txt"), np.array(nll_log))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        import json

        json.dump(metrics_log, f, indent=2)

    # Plot basic curves (NLL, MSE, rates_max)
    try:
        iters = [m["iter"] for m in metrics_log]
        mse_vals = [m["mse"] for m in metrics_log]
        rates_max_vals = [m["rates_max"] for m in metrics_log]
        plt.figure()
        plt.plot(iters, nll_log, label="NLL")
        plt.xlabel("iter")
        plt.ylabel("NLL")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "curve_nll.png"))
        plt.close()

        plt.figure()
        plt.plot(iters, mse_vals, label="MSE", color="orange")
        plt.xlabel("iter")
        plt.ylabel("MSE")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "curve_mse.png"))
        plt.close()

        plt.figure()
        plt.plot(iters, rates_max_vals, label="rates_max", color="green")
        plt.xlabel("iter")
        plt.ylabel("rates_max")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "curve_rates_max.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting skipped due to error: {e}")

    # Save gaussians as pickle for downstream use
    try:
        import pickle

        with open(os.path.join(output_dir, "gaussians.pkl"), "wb") as f:
            pickle.dump(gaussians, f)
    except Exception as e:
        print(f"Pickle save skipped due to error: {e}")

    return gaussians, nll_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gaussian Mixture Model EM algorithm on an image")
    parser.add_argument(
        "--image_path",
        type=str,
        default=os.path.join("data", "tsukuba", "scene1.row3.col1.ppm"),
        help="Path to the input image",
    )
    parser.add_argument(
        "--n_gaussians",
        type=int,
        default=300,
        help="Number of Gaussians (used when --k_list is not provided)",
    )
    parser.add_argument(
        "--k_list",
        type=str,
        default=None,
        help="Comma-separated list of K values to sweep (e.g., '150,200,250,300,500')",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum EM iterations (upper bound)",
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
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Optional path to a mask image (same H,W). Pixels outside mask are treated as missing.",
    )
    args = parser.parse_args()

    if args.k_list:
        k_values = [int(k.strip()) for k in args.k_list.split(",") if k.strip()]
    else:
        k_values = [args.n_gaussians]

    for k in k_values:
        print(f"\n=== Running EM for K={k} ===")
        run_gaussian_mixture_on_image(
            args.image_path,
            n_gaussians=k,
            max_iterations=args.max_iterations,
            tol=args.tol,
            min_iterations=args.min_iterations,
            init_mode=args.init_mode,
            mask_path=args.mask_path,
        )