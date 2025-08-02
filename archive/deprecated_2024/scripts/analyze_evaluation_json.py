#!/usr/bin/env python3
"""
Analyze the evaluation results JSON to understand the suspicious patterns.
"""

import json
import sys
from pathlib import Path
import numpy as np

def analyze_results(json_path):
    """Analyze evaluation results from JSON file."""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("="*80)
    print("EVALUATION RESULTS ANALYSIS")
    print("="*80)
    
    # Extract key metrics - handle both formats
    results = data.get('results', data)  # If no 'results' key, assume top level is results
    
    print("\n1. OVERALL RESULTS:")
    print("-"*40)
    for method, method_data in results.items():
        if isinstance(method_data, dict):
            # Check for aggregate_metrics key
            if 'aggregate_metrics' in method_data:
                metrics = method_data['aggregate_metrics']
            else:
                metrics = method_data
                
            mean_imp = metrics.get('mean_improvement', 0)
            std_imp = metrics.get('std_improvement', 0)
            f1 = metrics.get('mean_f1_score', 0)
            shd = metrics.get('mean_shd', 0)
            
            print(f"\n{method}:")
            print(f"  Mean improvement: {mean_imp:.3f} ± {std_imp:.3f}")
            print(f"  F1 Score: {f1:.3f}")
            print(f"  SHD: {shd:.3f}")
    
    # Analyze individual SCM results
    print("\n\n2. PER-SCM ANALYSIS:")
    print("-"*40)
    
    # Collect all SCM names
    all_scms = set()
    for method, method_data in results.items():
        if isinstance(method_data, dict) and 'scm_results' in method_data:
            all_scms.update(method_data['scm_results'].keys())
    
    # Group by SCM
    scm_data = {}
    for scm_name in all_scms:
        scm_data[scm_name] = {}
        for method, method_data in results.items():
            if isinstance(method_data, dict) and 'scm_results' in method_data:
                if scm_name in method_data['scm_results']:
                    scm_data[scm_name][method] = method_data['scm_results'][scm_name]
    
    # Analyze each SCM
    for scm_name, methods in scm_data.items():
        print(f"\n{scm_name}:")
        
        # Check initial values
        initial_values = set()
        for method, result in methods.items():
            if 'initial_value' in result:
                initial_values.add(result['initial_value'])
        
        if len(initial_values) == 1:
            print(f"  Initial value: {list(initial_values)[0]:.3f}")
        else:
            print(f"  ⚠️  Different initial values: {initial_values}")
        
        # Compare improvements
        for method, result in methods.items():
            improvement = result.get('improvement', 0)
            final_value = result.get('final_value', 0)
            best_value = result.get('best_value', 0)
            
            print(f"\n  {method}:")
            print(f"    Improvement: {improvement:.3f}")
            print(f"    Final value: {final_value:.3f}")
            print(f"    Best value: {best_value:.3f}")
            
            # Check trajectory
            if 'value_trajectory' in result:
                traj = result['value_trajectory']
                print(f"    Trajectory length: {len(traj)}")
                if len(traj) > 2:
                    print(f"    First 3 values: {traj[:3]}")
                    print(f"    Last 3 values: {traj[-3:]}")
                    
                    # Check if values are changing
                    unique_values = len(set(traj))
                    if unique_values < len(traj) / 2:
                        print(f"    ⚠️  Low diversity: {unique_values} unique values in {len(traj)} steps")
        
        # Check if all methods have same F1/SHD
        f1_scores = set()
        shd_scores = set()
        for method, result in methods.items():
            if 'structure_metrics' in result:
                sm = result['structure_metrics']
                f1_scores.add(sm.get('f1_score', -1))
                shd_scores.add(sm.get('shd', -1))
        
        if len(f1_scores) == 1 and -1 not in f1_scores:
            print(f"\n  ⚠️  All methods have identical F1: {list(f1_scores)[0]:.3f}")
        if len(shd_scores) == 1 and -1 not in shd_scores:
            print(f"  ⚠️  All methods have identical SHD: {list(shd_scores)[0]:.3f}")
    
    # 3. Check for specific issues
    print("\n\n3. ISSUE DETECTION:")
    print("-"*40)
    
    # Check random baseline
    random_results = [r for k, r in results.items() if 'random' in k.lower()]
    if random_results:
        random_mean = random_results[0].get('mean_improvement', 0)
        print(f"\nRandom baseline mean improvement: {random_mean:.3f}")
        if random_mean < -2:
            print("  ⚠️  This seems too good for random! Check:")
            print("     - Is optimization_direction set correctly?")
            print("     - Are improvements calculated as initial - final?")
            print("     - Is the target variable scaled unusually?")
    
    # Check if surrogates help
    print("\n\nSurrogate impact:")
    for base_method in ['grpo', 'bc']:
        without = None
        with_surr = None
        
        for method, metrics in results.items():
            if base_method in method.lower():
                if 'surrogate' in method.lower():
                    with_surr = metrics.get('mean_improvement', 0)
                else:
                    without = metrics.get('mean_improvement', 0)
        
        if without is not None and with_surr is not None:
            diff = without - with_surr  # Lower is better
            print(f"\n{base_method.upper()}:")
            print(f"  Without surrogate: {without:.3f}")
            print(f"  With surrogate: {with_surr:.3f}")
            print(f"  Difference: {diff:+.3f} {'✓ (helps)' if diff > 0 else '✗ (hurts)'}")


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        # Try default path
        json_path = Path("evaluation_results/evaluation_results.json")
        if not json_path.exists():
            json_path = Path("evaluation_results/validation/evaluation_results.json")
    else:
        json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: Could not find results file at {json_path}")
        print("Usage: python analyze_evaluation_json.py [path/to/evaluation_results.json]")
        sys.exit(1)
    
    print(f"Analyzing: {json_path}")
    analyze_results(json_path)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("\n1. Run diagnostic script to verify calculations:")
    print("   poetry run python scripts/diagnose_acbo_results.py")
    print("\n2. Check evaluation config optimization_direction")
    print("\n3. Consider training for more episodes (1000+)")
    print("\n4. Verify surrogate training includes graph supervision")


if __name__ == "__main__":
    main()