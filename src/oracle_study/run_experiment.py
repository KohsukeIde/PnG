#!/usr/bin/env python3
"""
Experiment Runner for Oracle Study

Simple script to run oracle study experiments.
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    parser = argparse.ArgumentParser(description='Run oracle study experiments')
    parser.add_argument('experiment', choices=[
        'transport_matrix_analysis',
        'cost_function_analysis', 
        'toy_generator_validation',
        'integration_validation'
    ], help='Experiment to run')
    
    args = parser.parse_args()
    
    # Import and run the specified experiment
    if args.experiment == 'transport_matrix_analysis':
        from src.oracle_study.experiments.transport_matrix_analysis import main
        main()
    elif args.experiment == 'cost_function_analysis':
        from src.oracle_study.experiments.cost_function_analysis import main
        main()
    elif args.experiment == 'toy_generator_validation':
        from src.oracle_study.experiments.toy_generator_validation import main
        main()
    elif args.experiment == 'integration_validation':
        from src.oracle_study.experiments.integration_validation import main
        main()

if __name__ == "__main__":
    main()