#!/usr/bin/env python3
"""
Oracle Study Test Runner
Runs all oracle study tests and experiments to validate the optimal transport formulation.
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd, description, env=None):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    """Run all oracle study tests."""
    print("Oracle Study Test Suite")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    original_cwd = Path.cwd()
    
    try:
        import os
        os.chdir(project_root)
        # Ensure src is importable
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        tests = [
            # Core functionality tests (run directly to avoid pytest dependency)
            (["python", "src/oracle_study/matching/experiments/toy_generator_validation.py"], 
             "Toy Problem Generator Validation"),
            # Integration tests
            (["python", "src/oracle_study/matching/experiments/integration_validation.py"], 
             "Oracle Integration Tests"),
            
            # Unified analysis experiment (combines cost function + transport matrix)
            (["python", "src/oracle_study/matching/experiments/unified_analysis.py"], 
             "Unified Analysis (Cost Function + Transport Matrix)"),
        ]
        
        results = []
        for cmd, description in tests:
            success = run_command(cmd, description, env=env)
            results.append((description, success))
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for description, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status}: {description}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed. Check output above.")
            return 1
            
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    sys.exit(main())
