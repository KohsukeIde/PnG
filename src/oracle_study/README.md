# Oracle Study for Perspective-N-Gaussian

This module provides tools and experiments for validating the optimal transport solver using synthetic data with known ground truth correspondences.

## Purpose

The oracle study helps us understand:
1. Whether the optimal transport solver is working correctly
2. What components of the cost function are actually contributing to good correspondences
3. How different transformations affect transport quality

## Key Findings

See `CRITICAL_FINDINGS.md` for detailed analysis results.

**Critical Discovery**: 
- **Color matching works perfectly** (92.44% diagonal concentration)
- **Epipolar constraint is completely broken** (0% diagonal concentration)
- **Any epipolar weight > 0 destroys performance**

## Structure

```
src/oracle_study/
├── core/                       # Core utilities
│   ├── toy_problem_generator.py    # Synthetic data generation
│   └── transport_matrix_visualizer.py  # Visualization tools
├── experiments/                # Analysis scripts
│   ├── transport_matrix_analysis.py    # Transport matrix validation
│   ├── cost_function_analysis.py       # Cost function component analysis
│   ├── toy_generator_validation.py     # Generator validation
│   └── integration_validation.py       # Integration tests
├── results/                    # Generated experimental results
│   ├── transport_matrix_analysis/
│   ├── cost_function_analysis/
│   └── toy_generator_test/
├── run_experiment.py          # Experiment runner script
├── README.md                  # This file
└── CRITICAL_FINDINGS.md       # Key discoveries
```

## Usage

### Option 1: Direct execution
```bash
python src/oracle_study/experiments/transport_matrix_analysis.py
python src/oracle_study/experiments/cost_function_analysis.py
```

### Option 2: Using the experiment runner
```bash
python src/oracle_study/run_experiment.py transport_matrix_analysis
python src/oracle_study/run_experiment.py cost_function_analysis
```

### Option 3: Import as module
```python
from src.oracle_study import ToyProblemGenerator, TransportMatrixVisualizer
from src.oracle_study.core import TransformationParams

# Use the tools in your own analysis
generator = ToyProblemGenerator(seed=42)
visualizer = TransportMatrixVisualizer()
```

## Experiments

### 1. Transport Matrix Analysis
- **Purpose**: Analyze transport matrices under different transformations and cost configurations
- **Includes**: Scenario analysis + Ablation studies
- **Results**: Complete ground truth validation with correspondence visualizations

### 2. Cost Function Analysis  
- **Purpose**: Analyze individual cost function components
- **Focus**: Shows cost matrices (not transport matrices)
- **Results**: Component-wise analysis of epipolar vs color terms

### 3. Toy Generator Validation
- **Purpose**: Validate synthetic data generation
- **Tests**: Gaussian generation, transformations, noise injection

### 4. Integration Validation
- **Purpose**: Test compatibility between components
- **Tests**: End-to-end validation of toy problems with solver

## Results

All experimental results are saved to `src/oracle_study/results/` with organized subdirectories for each experiment type.

## Next Steps

1. **Fix the epipolar constraint** - Currently completely non-functional
2. **Implement proper fundamental matrix estimation**
3. **Add geometric cost terms** that actually work
4. **Optimize weight balancing** using the validation framework