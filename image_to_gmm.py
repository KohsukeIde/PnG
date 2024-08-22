import os
import sys
import shutil 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

from src.optimizer.single_image_gaussian_mixture_em import SingleImageGaussianMixtureEM
from src.rasterizer.vanilla_2d_rasterizer import Vanilla2DRasterizer

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

def run_gaussian_mixture_on_image(image_path, n_gaussians=300, n_iterations=15):
    gmm = SingleImageGaussianMixtureEM(image_path)
    
    base_output_dir = "gaussian_mixture_results"
    output_dir = os.path.join(base_output_dir, f"gaussians_{n_gaussians}_iterations_{n_iterations}")
    
    # Check if the output directory exists, and if so, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing directory: {output_dir}")
        
    # Recreate the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    print("Image shape:", gmm.image.shape)
    print("Image min-max:", gmm.image.min(), gmm.image.max())
    
    gaussians = gmm.initialize_gaussians(n_gaussians)
    
    print("Initial gaussians statistics---------------------------------")
    print(f"Means min-max: {gaussians.means.min()}, {gaussians.means.max()}")
    print(f"Covs min-max: {gaussians.covs.min()}, {gaussians.covs.max()}")
    print(f"RGB min-max: {gaussians.rgb.min()}, {gaussians.rgb.max()}")
    print(f"Alpha min-max: {gaussians.alpha.min()}, {gaussians.alpha.max()}")
    print("------------------------------------------------------------")

    visualize_gaussians(gmm.image, gaussians, 0, output_dir)
    visualize_gaussian_parameters(gaussians, 0, output_dir, gmm.image.shape[0], gmm.image.shape[1])

    for i in range(n_iterations):
        responsibilities = gmm.e_step(gaussians)
        gaussians = gmm.m_step(responsibilities, gaussians)
        
        print(f"\nIteration {i+1} completed")
        visualize_gaussians(gmm.image, gaussians, i+1, output_dir)
        visualize_responsibilities(responsibilities, i+1, output_dir)
        visualize_gaussian_parameters(gaussians, i+1, output_dir, gmm.image.shape[0], gmm.image.shape[1])
        

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

    return gaussians

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gaussian Mixture Model EM algorithm on an image")
    parser.add_argument("--image_path", type=str, default=os.path.join("data", "tsukuba", "scene1.row3.col1.ppm"),
                        help="Path to the input image")
    parser.add_argument("--n_gaussians", type=int, default=3000, help="Number of Gaussians")
    parser.add_argument("--n_iterations", type=int, default=5, help="Number of EM iterations")
    args = parser.parse_args()

    final_gaussians = run_gaussian_mixture_on_image(args.image_path, args.n_gaussians, args.n_iterations)