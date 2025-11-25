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
        'unified_analysis',  # Combined transport + cost analysis
        'toy_generator_validation',
        'integration_validation',
        'pose_oracle_basic'
    ], help='Experiment to run')
    
    args = parser.parse_args()
    
    # Import and run the specified experiment
    if args.experiment == 'unified_analysis':
        from src.oracle_study.matching.experiments.unified_analysis import main
        main()
    elif args.experiment == 'toy_generator_validation':
        from src.oracle_study.matching.experiments.toy_generator_validation import main
        main()
    elif args.experiment == 'integration_validation':
        from src.oracle_study.matching.experiments.integration_validation import main
        main()
    elif args.experiment == 'pose_oracle_basic':
        from src.oracle_study.camera_pose.experiments.pose_oracle_basic import main
        main()

if __name__ == "__main__":
    main()
