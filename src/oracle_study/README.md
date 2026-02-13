# Oracle Study for Perspective-N-Gaussian

A comprehensive validation framework for the optimal transport solver using synthetic data with known ground truth correspondences.

## 🎯 Purpose

The oracle study addresses critical questions about the optimal transport solver:

1. **Is the solver working correctly?** - Validate transport behavior with known correspondences
2. **Which cost components are effective?** - Analyze epipolar vs color vs geometric terms
3. **How do transformations affect performance?** - Test robustness across different scenarios
4. **What are the optimal weight configurations?** - Find best parameter settings

## 🔍 Key Findings

### **Critical Discovery: Epipolar Constraint is Broken**

Our analysis revealed fundamental issues with the current cost function:

| Cost Component | Diagonal Concentration | Status |
|----------------|----------------------|---------|
| **Color Only** | 92.44% | ✅ **Excellent** |
| **Epipolar Only** | 0.00% | ❌ **Completely Broken** |
| **Balanced (Color + Epipolar)** | 0.01% | ❌ **Broken** |

**Conclusion**: The epipolar constraint is non-functional and any epipolar weight > 0 destroys performance.

### **Performance by Transformation Type**

| Scenario | Diagonal Concentration | Description |
|----------|----------------------|-------------|
| Identical | 92.44% | Perfect baseline |
| Translation | 92.44% | Excellent (color matching works) |
| Rotation | 92.44% | Excellent (color matching works) |
| Scale | 92.44% | Excellent (color matching works) |
| **Color Change** | **7.82%** | **Poor (no color similarity)** |

This confirms that the solver relies almost entirely on color similarity for correspondence.

## 📁 Project Structure

```
src/oracle_study/
├── core/                           # Core utilities and classes
│   ├── __init__.py                 # Module exports
│   ├── toy_problem_generator.py    # Synthetic data generation
│   └── transport_matrix_visualizer.py  # Visualization tools
├── experiments/                    # Analysis and validation scripts
│   ├── __init__.py
│   ├── transport_matrix_analysis.py    # Main transport validation
│   ├── cost_function_analysis.py       # Cost component analysis
│   ├── toy_generator_validation.py     # Generator validation
│   └── integration_validation.py       # Integration tests
├── results/                        # Generated experimental results (gitignored)
│   ├── transport_matrix_analysis/  # Transport analysis outputs
│   ├── cost_function_analysis/     # Cost analysis outputs
│   └── toy_generator_test/         # Generator test outputs
├── run_experiment.py              # Convenient experiment runner
├── README.md                      # This documentation
└── CRITICAL_FINDINGS.md           # Detailed analysis results
```

## 🚀 Quick Start

### Prerequisites
- Python environment with required dependencies
- Run from project root directory

### Option 1: Direct Execution
```bash
# Run unified transport + cost analysis (recommended first)
python src/oracle_study/matching/experiments/unified_analysis.py --epipolar-mode hybrid

# Validate toy generator
python src/oracle_study/matching/experiments/toy_generator_validation.py

# Run integration tests
python src/oracle_study/matching/experiments/integration_validation.py
```

### Option 2: Using Experiment Runner
```bash
# More convenient interface
python src/oracle_study/run_experiment.py unified_analysis
python src/oracle_study/run_experiment.py toy_generator_validation
python src/oracle_study/run_experiment.py integration_validation
```

### Option 3: Import as Module
```python
from src.oracle_study import ToyProblemGenerator, TransportMatrixVisualizer
from src.oracle_study.core import TransformationParams

# Create synthetic test data
generator = ToyProblemGenerator(seed=42)
gaussians1 = generator.generate_synthetic_gaussians(n_gaussians=20)

# Apply transformations with known correspondences
gaussians2, correspondences = generator.generate_known_correspondences(
    gaussians1, TransformationParams(translation=[0.3, 0.2])
)

# Visualize results
visualizer = TransportMatrixVisualizer()
# ... use in your analysis
```

## 📊 Experiments Overview

### 1. Transport Matrix Analysis
**File**: `transport_matrix_analysis.py`

**Purpose**: Comprehensive validation of transport matrices under different scenarios and cost configurations.

**What it does**:
- **Scenario Analysis**: Tests 6 different transformation scenarios
  - `identical`: Baseline (no transformation)
  - `translation`: Pure translation
  - `rotation`: Pure rotation  
  - `scale`: Pure scaling
  - `color_change`: Same positions, different colors
  - `combined`: Complex multi-transformation
- **Ablation Studies**: Tests 5 different cost weight configurations
  - `color_only`: λ_color=1.0, λ_epipolar=0.0
  - `epipolar_only`: λ_color=0.0, λ_epipolar=1.0
  - `balanced`: λ_color=0.5, λ_epipolar=1.0
  - `color_heavy`: λ_color=2.0, λ_epipolar=0.5
  - `epipolar_heavy`: λ_color=0.5, λ_epipolar=2.0

**Outputs**:
- Transport matrix heatmaps for each scenario
- Correspondence visualizations (scenarios only)
- Side-by-side comparisons
- Quantitative analysis summary
- Ablation study results

**Key Metrics**:
- **Diagonal Concentration**: % of transport mass on diagonal (higher = better correspondences)
- **Entropy**: Measure of uncertainty (lower = more concentrated)
- **Sparsity**: Fraction of near-zero elements (higher = more concentrated)

### 2. Cost Function Analysis
**File**: `cost_function_analysis.py`

**Purpose**: Deep dive into individual cost function components.

**What it does**:
- Analyzes cost matrices (not transport matrices) for each component
- Tests component isolation (epipolar-only, color-only, combined)
- Weight sensitivity analysis across different ratios
- Visualizes cost landscapes

**Outputs**:
- Cost matrix heatmaps showing raw cost values
- Component comparison plots
- Weight sensitivity analysis
- Performance vs weight configuration plots

**Key Insight**: Shows that epipolar costs are computed but don't lead to meaningful transport behavior.

### 3. Toy Generator Validation
**File**: `toy_generator_validation.py`

**Purpose**: Validate the synthetic data generation pipeline.

**What it does**:
- Tests Gaussian generation with different color modes
- Validates geometric transformations
- Tests noise injection capabilities
- Visualizes generated test cases

**Outputs**:
- Sample Gaussian visualizations
- Transformation validation plots
- Noise effect demonstrations

### 4. Integration Validation
**File**: `integration_validation.py`

**Purpose**: End-to-end compatibility testing.

**What it does**:
- Tests toy generator + optimal transport solver integration
- Validates data format compatibility
- Basic functionality checks

**Outputs**:
- Console output with pass/fail results
- Basic transport matrix validation

## 🔧 Core Components

### ToyProblemGenerator
**Purpose**: Generate synthetic 2D Gaussian problems with known ground truth.

**Key Methods**:
```python
# Generate base Gaussians
generate_synthetic_gaussians(n_gaussians, color_mode='gradient')

# Create transformed version with known correspondences  
generate_known_correspondences(gaussians1, transform_params)

# Apply geometric transformations
apply_transformation(gaussians, transform_params)
```

**Color Modes**:
- `gradient`: Smooth color gradient across Gaussians
- `random`: Random colors for each Gaussian
- `uniform`: All Gaussians same color

**Transformation Types**:
- Translation: `TransformationParams(translation=[dx, dy])`
- Rotation: `TransformationParams(rotation=angle_radians)`
- Scale: `TransformationParams(scale=factor)`
- Combined: Multiple transformations together

### TransportMatrixVisualizer
**Purpose**: Visualize and analyze transport matrices.

**Key Methods**:
```python
# Individual matrix visualization
visualize_transport_matrix(matrix, title, save_path)

# Side-by-side comparison
compare_transport_matrices(matrices_dict, save_path)

# Statistical analysis
analyze_transport_statistics(matrix, name)

# Correspondence visualization
visualize_correspondences_on_images(gaussians1, gaussians2, transport_matrix)
```

**Analysis Metrics**:
- Basic stats (min, max, mean, sum, std)
- Sparsity analysis
- Concentration analysis (top 1%, 5%, 10%)
- Diagonal concentration (key metric)
- Entropy calculation
- Row/column statistics

## 📈 Understanding Results

### Diagonal Concentration
**Most Important Metric**: Measures how much transport mass is on the diagonal.

- **Perfect Score**: 1.0 (100%) - All mass on diagonal
- **Random Score**: ~0.067 (1/15 for 15x15 matrix)
- **Good Score**: >0.8 (80%+)
- **Poor Score**: <0.2 (20%)

**Interpretation**:
- High diagonal concentration = Correct correspondences found
- Low diagonal concentration = Random/incorrect correspondences

### Transport Matrix Visualization
**Heatmap Colors**:
- **Bright (Hot)**: High transport probability
- **Dark (Cold)**: Low transport probability
- **Diagonal Pattern**: Good correspondences
- **Scattered Pattern**: Poor correspondences

### Correspondence Visualization
**Shows**:
- Source Gaussians (left panel)
- Target Gaussians (right panel)  
- Numbered Gaussians for correspondence tracking
- Top correspondences printed to console

## 🔬 Methodology

### Oracle Study Approach
1. **Generate Known Truth**: Create synthetic problems where we know the correct answer
2. **Test Solver**: Run optimal transport solver on synthetic data
3. **Compare Results**: Measure how well solver finds known correspondences
4. **Analyze Failures**: Understand what's working and what's broken

### Validation Strategy
1. **Baseline Testing**: Start with identical Gaussians (should be perfect)
2. **Incremental Complexity**: Add transformations one by one
3. **Component Isolation**: Test each cost term separately
4. **Ablation Studies**: Systematic weight variation
5. **Edge Cases**: Test challenging scenarios (color changes, etc.)

## 🚨 Current Issues & Next Steps

### Critical Issues Identified
1. **Epipolar constraint completely non-functional**
   - 0% diagonal concentration with epipolar-only cost
   - Any epipolar weight > 0 destroys performance
   - Fundamental matrix computation or application is broken

2. **Over-reliance on color similarity**
   - 92.44% performance with color matching
   - 7.82% performance when colors change
   - No geometric understanding

### Immediate Actions Needed
1. **Fix epipolar constraint implementation**
   - Debug fundamental matrix computation
   - Verify epipolar cost calculation
   - Test with proper camera geometry

2. **Add geometric cost terms**
   - Position-based costs
   - Scale/covariance matching
   - Orientation consistency

3. **Implement robust cost function**
   - Combine working geometric and appearance terms
   - Proper weight balancing
   - Fallback mechanisms

### Long-term Improvements
1. **Enhanced validation framework**
   - More transformation types
   - Noise robustness testing
   - Real data validation

2. **Performance optimization**
   - Faster transport computation
   - Better convergence criteria
   - Memory efficiency

## 📚 References & Background

### Optimal Transport Theory
- Transport matrices represent probability of correspondence between Gaussians
- Sinkhorn algorithm approximates optimal transport solution
- Cost function design critical for good correspondences

### Epipolar Geometry
- Fundamental matrix encodes geometric constraints between views
- Epipolar lines constrain possible correspondences
- Should provide strong geometric prior for matching

### Validation Methodology
- Oracle studies use synthetic data with known ground truth
- Ablation studies isolate individual component contributions
- Quantitative metrics enable objective performance assessment

---

## 💡 Tips for Users

### Running Your First Analysis
1. Start with `transport_matrix_analysis.py` - gives comprehensive overview
2. Check diagonal concentration values - key performance indicator
3. Look at correspondence visualizations - visual validation of results
4. Review ablation summary plot - shows component effectiveness

### Interpreting Results
- **High diagonal concentration (>80%)**: Good correspondences
- **Low diagonal concentration (<20%)**: Poor correspondences  
- **Scattered transport matrix**: Random matching
- **Diagonal transport matrix**: Correct matching

### Debugging Issues
- Check console output for statistics
- Examine individual transport matrix heatmaps
- Compare different cost configurations
- Validate input data with toy generator tests

### Contributing
- Add new transformation scenarios to test robustness
- Implement additional cost function components
- Extend visualization capabilities
- Add more quantitative analysis metrics

---

**For detailed analysis results and specific findings, see `CRITICAL_FINDINGS.md`.**
