#!/usr/bin/env python3
"""Analyze results from 12-pair evaluation."""

import json
import sys
from pathlib import Path
import pandas as pd

def analyze_results(json_path):
    """Analyze and summarize evaluation results."""
    
    with open(json_path) as f:
        results = json.load(f)
    
    # Create summary table
    summary = []
    
    for method, data in results.items():
        if 'aggregate_metrics' not in data:
            continue
            
        metrics = data['aggregate_metrics']
        
        # Parse method name
        if '+' in method:
            policy, surrogate = method.split('+')
        else:
            policy = method
            surrogate = 'none'
        
        summary.append({
            'Policy': policy,
            'Surrogate': surrogate,
            'Mean Improvement': f"{metrics['mean_improvement']:.3f}",
            'Std Improvement': f"{metrics['std_improvement']:.3f}",
            'Mean Trajectory Value': f"{metrics['mean_trajectory_value']:.3f}",
            'Mean F1': f"{metrics['mean_f1_score']:.3f}",
            'Mean SHD': f"{metrics['mean_shd']:.3f}",
            'Method': method
        })
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame(summary)
    
    # Sort by policy then surrogate
    df = df.sort_values(['Policy', 'Surrogate'])
    
    print("\n" + "="*80)
    print("12-PAIR EVALUATION RESULTS")
    print("="*80)
    
    # Group by policy
    for policy in ['random', 'oracle', 'bc', 'grpo']:
        policy_results = df[df['Policy'] == policy]
        if policy_results.empty:
            continue
            
        print(f"\n{policy.upper()} Policy:")
        print("-" * 60)
        
        for _, row in policy_results.iterrows():
            surrogate = row['Surrogate']
            print(f"\n  With {surrogate} surrogate:")
            print(f"    Performance: {row['Mean Trajectory Value']} (trajectory mean)")
            print(f"    Improvement: {row['Mean Improvement']} ± {row['Std Improvement']}")
            print(f"    Structure:   F1={row['Mean F1']}, SHD={row['Mean SHD']}")
    
    # Find best combinations
    print("\n" + "="*60)
    print("TOP PERFORMERS (by mean trajectory value):")
    print("="*60)
    
    df_sorted = df.sort_values('Mean Trajectory Value', ascending=True)  # Lower is better
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
        print(f"{i+1}. {row['Method']}: {row['Mean Trajectory Value']}")
    
    # Compare surrogate effectiveness
    print("\n" + "="*60)
    print("SURROGATE COMPARISON (averaged across policies):")
    print("="*60)
    
    for surrogate in ['bc_static', 'bc_active', 'untrained']:
        surrogate_results = df[df['Surrogate'] == surrogate]
        if surrogate_results.empty:
            continue
            
        # Convert string values back to float for averaging
        mean_traj = surrogate_results['Mean Trajectory Value'].apply(float).mean()
        mean_f1 = surrogate_results['Mean F1'].apply(float).mean()
        
        print(f"\n{surrogate}:")
        print(f"  Avg trajectory value: {mean_traj:.3f}")
        print(f"  Avg F1 score: {mean_f1:.3f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Find most recent results
        results_dir = Path("evaluation_results")
        json_files = list(results_dir.glob("*/evaluation_results.json"))
        if json_files:
            json_path = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Using most recent results: {json_path}")
        else:
            print("No results found!")
            sys.exit(1)
    
    analyze_results(json_path)