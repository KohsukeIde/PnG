#!/usr/bin/env python3
"""
Test script for the ToyProblemGenerator to verify it works correctly.
"""

import sys
import os
# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Create experiment-specific figure directory
EXPERIMENT_NAME = "toy_generator_test"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_BASE = os.path.join(ORACLE_DIR, "results", EXPERIMENT_NAME)
FIGURES_DIR = os.path.join(RESULTS_BASE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

import numpy as np
import torch
import matplotlib.pyplot as plt
from src.oracle_study.core import (
    ToyProblemGenerator, 
    TransformationParams, 
    NoiseParams
)

def test_basic_generation():
    """Test basic Gaussian generation."""
    print("Testing basic Gaussian generation...")
    
    generator = ToyProblemGenerator(seed=42)
    gaussians = generator.generate_synthetic_gaussians(
        n_gaussians=20,
        color_mode='gradient'
    )
    
    print(f"Generated {len(gaussians.means)} Gaussians")
    print(f"Means shape: {gaussians.means.shape}")
    print(f"Scales shape: {gaussians.scales.shape}")
    print(f"Colors shape: {gaussians.rgb.shape}")
    print(f"Mean position: {gaussians.means.mean(axis=0)}")
    print(f"Position range: {gaussians.means.min()} to {gaussians.means.max()}")
    
    assert len(gaussians.means) == 20
    assert gaussians.means.shape == (20, 2)
    assert gaussians.scales.shape == (20, 2)
    assert gaussians.rgb.shape == (20, 3)
    print("✓ Basic generation test passed\n")

def test_transformations():
    """Test various transformations."""
    print("Testing transformations...")
    
    generator = ToyProblemGenerator(seed=42)
    base_gaussians = generator.generate_synthetic_gaussians(n_gaussians=10)
    
    # Test translation
    trans_params = TransformationParams(translation=np.array([0.2, 0.1]))
    gaussians2, correspondences = generator.generate_known_correspondences(
        base_gaussians, trans_params
    )
    
    # Check that positions are translated correctly
    expected_translation = np.array([0.2, 0.1])
    actual_translation = gaussians2.means.mean(axis=0) - base_gaussians.means.mean(axis=0)
    translation_error = np.linalg.norm(actual_translation - expected_translation)
    
    print(f"Translation error: {translation_error:.6f}")
    print(f"Correspondences shape: {correspondences.shape}")
    
    assert translation_error < 1e-5, f"Translation error too large: {translation_error}"
    assert correspondences.shape == (10, 2)
    print("✓ Translation test passed")
    
    # Test rotation
    rot_params = TransformationParams(rotation=np.pi/4)  # 45 degrees
    gaussians2_rot, corr_rot = generator.generate_known_correspondences(
        base_gaussians, rot_params
    )
    
    # Check that rotation was applied (positions should be different)
    position_diff = np.linalg.norm(gaussians2_rot.means - base_gaussians.means)
    print(f"Position difference after rotation: {position_diff:.6f}")
    
    assert position_diff > 0.1, "Rotation should change positions significantly"
    print("✓ Rotation test passed")
    
    # Test scale
    scale_params = TransformationParams(scale=2.0)
    gaussians2_scale, corr_scale = generator.generate_known_correspondences(
        base_gaussians, scale_params
    )
    
    # Check that scales are approximately doubled
    scale_ratio = gaussians2_scale.scales.mean() / base_gaussians.scales.mean()
    print(f"Scale ratio: {scale_ratio:.6f}")
    
    assert abs(scale_ratio - 2.0) < 0.1, f"Scale ratio should be ~2.0, got {scale_ratio}"
    print("✓ Scale test passed\n")

def test_noise_injection():
    """Test noise injection."""
    print("Testing noise injection...")
    
    generator = ToyProblemGenerator(seed=42)
    base_gaussians = generator.generate_synthetic_gaussians(n_gaussians=10)
    
    # Test with noise
    noise_params = NoiseParams(
        position_noise=0.05,
        color_noise=0.1
    )
    
    noisy_gaussians = generator.add_controlled_noise(base_gaussians, noise_params)
    
    # Check that noise was added
    position_diff = np.linalg.norm(noisy_gaussians.means - base_gaussians.means)
    color_diff = np.linalg.norm(noisy_gaussians.rgb - base_gaussians.rgb)
    
    print(f"Position difference after noise: {position_diff:.6f}")
    print(f"Color difference after noise: {color_diff:.6f}")
    
    assert position_diff > 0.01, "Position noise should be noticeable"
    assert color_diff > 0.01, "Color noise should be noticeable"
    print("✓ Noise injection test passed\n")

def test_standard_test_cases():
    """Test the standard test cases."""
    print("Testing standard test cases...")
    
    generator = ToyProblemGenerator(seed=42)
    test_cases = generator.create_test_cases()
    
    expected_cases = ['identical', 'translation', 'rotation', 'scale', 'combined', 'noisy']
    
    print(f"Generated test cases: {list(test_cases.keys())}")
    
    for case_name in expected_cases:
        assert case_name in test_cases, f"Missing test case: {case_name}"
        
        gaussians1, gaussians2, correspondences = test_cases[case_name]
        
        print(f"  {case_name}: {len(gaussians1.means)} -> {len(gaussians2.means)} Gaussians, "
              f"{len(correspondences)} correspondences")
        
        # Basic sanity checks
        assert len(gaussians1.means) > 0
        assert len(gaussians2.means) > 0
        assert len(correspondences) > 0
        assert correspondences.shape[1] == 2  # Should be pairs
    
    print("✓ Standard test cases passed\n")

def visualize_test_case(case_name='translation'):
    """Visualize a test case to verify it looks reasonable."""
    print(f"Visualizing test case: {case_name}")
    
    generator = ToyProblemGenerator(seed=42)
    test_cases = generator.create_test_cases()
    
    if case_name not in test_cases:
        print(f"Test case {case_name} not found")
        return
    
    gaussians1, gaussians2, correspondences = test_cases[case_name]
    
    # Get numpy arrays for plotting
    pos1 = gaussians1.means
    pos2 = gaussians2.means
    colors1 = gaussians1.rgb
    colors2 = gaussians2.rgb
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot first set
    ax1.scatter(pos1[:, 0], pos1[:, 1], c=colors1, s=100, alpha=0.7)
    ax1.set_title(f'Gaussians 1 ({case_name})')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot second set
    ax2.scatter(pos2[:, 0], pos2[:, 1], c=colors2, s=100, alpha=0.7)
    ax2.set_title(f'Gaussians 2 ({case_name})')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, f'toy_generator_test_{case_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()

def main():
    """Run all tests."""
    print("=== Testing ToyProblemGenerator ===\n")
    
    try:
        test_basic_generation()
        test_transformations()
        test_noise_injection()
        test_standard_test_cases()
        
        # Create a visualization
        visualize_test_case('translation')
        visualize_test_case('rotation')
        
        print("🎉 All tests passed! ToyProblemGenerator is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
