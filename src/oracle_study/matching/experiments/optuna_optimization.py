#!/usr/bin/env python3
"""
Optuna-based Hyperparameter Optimization for Optimal Transport

This module implements systematic hyperparameter search using Optuna to find
optimal configurations for the NLL-based optimal transport cost function.
"""

import optuna
import torch
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.oracle_study.core import ToyProblemGenerator

class OptimalTransportOptimizer:
    """Optuna-based optimizer for optimal transport hyperparameters."""
    
    def __init__(self, 
                 n_trials: int = 100,
                 seed: int = 42,
                 epipolar_mode: str = 'hybrid',
                 study_name: str = None):
        """Initialize the optimizer.
        
        Args:
            n_trials: Number of optimization trials
            seed: Random seed for reproducibility
            epipolar_mode: Epipolar cost computation mode
            study_name: Name for the optimization study
        """
        self.n_trials = n_trials
        self.seed = seed
        self.epipolar_mode = epipolar_mode
        self.study_name = study_name or f"ot_optimization_{epipolar_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup output directory
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "results", "optuna", self.study_name
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize problem generator
        self.generator = ToyProblemGenerator(seed=seed)
        self.K = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]], dtype=np.float32)
        
        # Generate diverse scenarios for robust evaluation
        self.scenarios = ToyProblemGenerator.get_optuna_scenarios()
        
    
    def _evaluate_configuration(self, params: Dict[str, float]) -> float:
        """Evaluate a parameter configuration across all scenarios.
        
        Args:
            params: Dictionary of parameters to evaluate
            
        Returns:
            Average diagonal concentration across all scenarios
        """
        diagonal_concentrations = []
        
        for scenario in self.scenarios:
            try:
                # Generate Gaussians for this scenario
                g1, g2, _, F_gt = self.generator.generate_epipolar_correspondences(
                    n_gaussians=15, 
                    K=self.K, 
                    R_wc=scenario['R_wc'], 
                    t_wc=scenario['t_wc']
                )
                
                # Apply color noise if specified
                if scenario['color_noise']:
                    rng = np.random.RandomState(self.seed)
                    g2.rgb = rng.uniform(0, 1, size=g2.rgb.shape).astype(np.float32)
                
                # Create solver with suggested parameters
                solver = OptimalTransportSolver(
                    gaussians1=g1, gaussians2=g2, k1=self.K, k2=self.K,
                    epsilon=params['epsilon'],
                    lambda_color=params['lambda_color'],
                    lambda_epipolar=params['lambda_epipolar'], 
                    lambda_cov=params['lambda_cov'],
                    sigma_epipolar=params['sigma_epipolar'],
                    sigma_color=params['sigma_color'],
                    sigma_cov=params['sigma_cov'],
                    noise_model=params['noise_model'],
                    epipolar_mode=self.epipolar_mode,
                    device='cpu'
                )
                
                # Compute transport matrix
                with torch.no_grad():
                    C = solver.compute_cost_matrix(torch.from_numpy(F_gt))
                    T = solver.unbalanced_sinkhorn_algorithm(C)
                    T_np = T.cpu().numpy()
                
                # Calculate diagonal concentration
                diagonal_sum = np.trace(T_np)
                total_sum = np.sum(T_np)
                diagonal_concentration = diagonal_sum / total_sum if total_sum > 0 else 0.0
                
                diagonal_concentrations.append(diagonal_concentration)
                
            except Exception as e:
                print(f"Error in scenario {scenario['name']}: {e}")
                diagonal_concentrations.append(0.0)  # Penalty for failed cases
        
        # Return average diagonal concentration
        avg_concentration = np.mean(diagonal_concentrations)
        return avg_concentration
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value (negative because Optuna minimizes)
        """
        # Sample hyperparameters
        params = {
            # Lambda parameters (relative importance)
            'lambda_color': trial.suggest_float('lambda_color', 0.0, 2.0),
            'lambda_epipolar': trial.suggest_float('lambda_epipolar', 0.1, 2.0),
            'lambda_cov': trial.suggest_float('lambda_cov', 0.0, 1.0),
            
            # Sigma parameters (physical scales)
            'sigma_epipolar': trial.suggest_float('sigma_epipolar', 0.1, 10.0, log=True),
            'sigma_color': trial.suggest_float('sigma_color', 0.01, 1.0, log=True),
            'sigma_cov': trial.suggest_float('sigma_cov', 0.1, 50.0, log=True),
            
            # Noise model
            'noise_model': trial.suggest_categorical('noise_model', ['gaussian', 'cauchy', 'huber']),
            
            # Sinkhorn parameters
            'epsilon': trial.suggest_float('epsilon', 0.001, 0.1, log=True),
        }
        
        # Evaluate configuration
        avg_concentration = self._evaluate_configuration(params)
        
        # Log results
        trial.set_user_attr('avg_diagonal_concentration', avg_concentration)
        trial.set_user_attr('scenario_count', len(self.scenarios))
        
        # Return negative value (Optuna minimizes)
        return -avg_concentration
    
    def optimize(self) -> optuna.Study:
        """Run the optimization process.
        
        Returns:
            Completed Optuna study
        """
        print(f"🚀 Starting hyperparameter optimization for {self.epipolar_mode} mode")
        print(f"Trials: {self.n_trials}, Scenarios: {len(self.scenarios)}")
        print(f"Output directory: {self.output_dir}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study: optuna.Study):
        """Save optimization results and generate visualizations."""
        
        # Save best parameters
        best_params = {}
        for key, value in study.best_params.items():
            if hasattr(value, 'item'):  # numpy types
                best_params[key] = value.item()
            else:
                best_params[key] = float(value) if isinstance(value, (int, float)) else value
        
        best_params['best_value'] = float(-study.best_value)  # Convert back to positive
        best_params['n_trials'] = len(study.trials)
        best_params['epipolar_mode'] = self.epipolar_mode
        
        with open(os.path.join(self.output_dir, 'best_params.json'), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save all trials data
        trials_data = []
        for trial in study.trials:
            # Convert numpy types to Python native types for JSON serialization
            params = {}
            for key, value in trial.params.items():
                if hasattr(value, 'item'):  # numpy types
                    params[key] = value.item()
                else:
                    params[key] = float(value) if isinstance(value, (int, float)) else value
            
            trial_data = {
                'number': trial.number,
                'value': float(-trial.value) if trial.value is not None else None,
                'params': params,
                'state': trial.state.name,
                'avg_diagonal_concentration': float(trial.user_attrs.get('avg_diagonal_concentration', 0.0))
            }
            trials_data.append(trial_data)
        
        with open(os.path.join(self.output_dir, 'trials_data.json'), 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Generate visualizations
        self._create_visualizations(study)
        
        print(f"\n🎉 Optimization completed!")
        print(f"Best diagonal concentration: {-study.best_value:.4f}")
        print(f"Best parameters saved to: {self.output_dir}/best_params.json")
    
    def _create_visualizations(self, study: optuna.Study):
        """Create optimization visualizations."""
        
        # 1. Optimization history
        fig, ax = plt.subplots(figsize=(10, 6))
        values = [-t.value for t in study.trials if t.value is not None]
        ax.plot(values, 'b-', alpha=0.7, label='Trial Value')
        ax.plot(np.maximum.accumulate(values), 'r-', linewidth=2, label='Best Value')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Diagonal Concentration')
        ax.set_title(f'Optimization History - {self.epipolar_mode.upper()} Mode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'optimization_history.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            if importance:
                fig, ax = plt.subplots(figsize=(10, 6))
                params = list(importance.keys())
                values = list(importance.values())
                ax.barh(params, values)
                ax.set_xlabel('Importance')
                ax.set_title(f'Parameter Importance - {self.epipolar_mode.upper()} Mode')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'parameter_importance.png'), dpi=150, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Could not generate parameter importance plot: {e}")
        
        # 3. Parameter correlations
        try:
            df = study.trials_dataframe()
            if len(df) > 10:  # Only if we have enough data
                param_cols = [col for col in df.columns if col.startswith('params_')]
                if len(param_cols) > 1:
                    corr_matrix = df[param_cols + ['value']].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                    ax.set_title(f'Parameter Correlations - {self.epipolar_mode.upper()} Mode')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'parameter_correlations.png'), dpi=150, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            print(f"Could not generate correlation plot: {e}")


def run_multi_mode_optimization(n_trials: int = 100):
    """Run optimization for all epipolar modes."""
    
    modes = ['sed', 'sampson', 'hybrid']
    results = {}
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"OPTIMIZING MODE: {mode.upper()}")
        print('='*60)
        
        optimizer = OptimalTransportOptimizer(
            n_trials=n_trials,
            epipolar_mode=mode,
            seed=42
        )
        
        study = optimizer.optimize()
        results[mode] = {
            'best_params': study.best_params,
            'best_value': -study.best_value,
            'output_dir': optimizer.output_dir
        }
    
    # Create comparison summary
    summary_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "results", "optuna", "summary"
    )
    os.makedirs(summary_dir, exist_ok=True)
    
    with open(os.path.join(summary_dir, 'optimization_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print('='*60)
    
    for mode, result in results.items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  Best diagonal concentration: {result['best_value']:.4f}")
        print(f"  Best parameters: {result['best_params']}")
        print(f"  Results: {result['output_dir']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optuna-based hyperparameter optimization")
    parser.add_argument('--mode', choices=['sed', 'sampson', 'hybrid', 'all'], 
                       default='all', help='Epipolar mode to optimize')
    parser.add_argument('--trials', type=int, default=100, 
                       help='Number of optimization trials')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_multi_mode_optimization(args.trials)
    else:
        optimizer = OptimalTransportOptimizer(
            n_trials=args.trials,
            epipolar_mode=args.mode,
            seed=42
        )
        optimizer.optimize()
