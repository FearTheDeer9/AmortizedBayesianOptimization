#!/usr/bin/env python3
"""
Verify data consistency between summary statistics and trajectory plots.

This script loads experiment results and compares:
1. Summary statistics (final values aggregated across runs)
2. Trajectory data (time series data from individual runs)

It ensures that the final values in trajectories match the summary statistics.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_experiment_results(results_path):
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def extract_final_values_from_trajectories(method_results):
    """Extract final values from trajectory data."""
    final_values = {
        'f1_scores': [],
        'shd_values': [],
        'target_values': []
    }
    
    for result in method_results:
        if 'detailed_results' in result:
            detailed = result['detailed_results']
            
            # Extract final values from trajectories
            if 'f1_scores' in detailed and detailed['f1_scores']:
                final_values['f1_scores'].append(detailed['f1_scores'][-1])
            elif 'f1_score_final' in result:
                final_values['f1_scores'].append(result['f1_score_final'])
                
            if 'shd_values' in detailed and detailed['shd_values']:
                final_values['shd_values'].append(detailed['shd_values'][-1])
            elif 'shd_final' in result:
                final_values['shd_values'].append(result['shd_final'])
                
            if 'target_progress' in detailed and detailed['target_progress']:
                final_values['target_values'].append(detailed['target_progress'][-1])
            elif 'final_target_value' in result:
                final_values['target_values'].append(result['final_target_value'])
    
    return final_values


def compare_summary_and_trajectory_data(results):
    """Compare summary statistics with trajectory final values."""
    print("\n=== Data Consistency Verification ===\n")
    
    method_results = results.get('method_results', {})
    
    for method_name, method_data in method_results.items():
        print(f"\n{method_name}:")
        print("-" * 40)
        
        # Extract final values from trajectories
        trajectory_finals = extract_final_values_from_trajectories(method_data)
        
        # Get summary statistics
        summary_values = {
            'f1_scores': [r.get('f1_score_final', 0.0) for r in method_data if 'f1_score_final' in r],
            'shd_values': [r.get('shd_final', 0.0) for r in method_data if 'shd_final' in r],
            'target_values': [r.get('final_target_value', 0.0) for r in method_data if 'final_target_value' in r]
        }
        
        # Compare values
        for metric in ['f1_scores', 'shd_values', 'target_values']:
            traj_vals = trajectory_finals[metric]
            summ_vals = summary_values[metric]
            
            if traj_vals and summ_vals:
                traj_mean = np.mean(traj_vals)
                summ_mean = np.mean(summ_vals)
                diff = abs(traj_mean - summ_mean)
                
                print(f"\n  {metric}:")
                print(f"    Trajectory final (mean): {traj_mean:.4f}")
                print(f"    Summary value (mean):    {summ_mean:.4f}")
                print(f"    Difference:              {diff:.4f}")
                
                if diff > 0.001:
                    print(f"    ⚠️  WARNING: Values don't match!")
            elif traj_vals:
                print(f"\n  {metric}: Only trajectory data available (mean: {np.mean(traj_vals):.4f})")
            elif summ_vals:
                print(f"\n  {metric}: Only summary data available (mean: {np.mean(summ_vals):.4f})")
            else:
                print(f"\n  {metric}: No data available")
    
    # Create visualization
    create_consistency_plot(method_results)


def create_consistency_plot(method_results):
    """Create plot comparing trajectory finals with summary statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Consistency: Trajectory Finals vs Summary Statistics', fontsize=16)
    
    metrics = [
        ('f1_scores', 'F1 Score', axes[0, 0]),
        ('shd_values', 'SHD', axes[0, 1]),
        ('target_values', 'Target Value', axes[1, 0])
    ]
    
    for metric_key, metric_label, ax in metrics:
        methods = []
        trajectory_means = []
        trajectory_stds = []
        summary_means = []
        summary_stds = []
        
        for method_name, method_data in method_results.items():
            # Extract final values from trajectories
            trajectory_finals = extract_final_values_from_trajectories(method_data)
            traj_vals = trajectory_finals[metric_key]
            
            # Get summary statistics
            if metric_key == 'f1_scores':
                summ_vals = [r.get('f1_score_final', np.nan) for r in method_data]
            elif metric_key == 'shd_values':
                summ_vals = [r.get('shd_final', np.nan) for r in method_data]
            elif metric_key == 'target_values':
                summ_vals = [r.get('final_target_value', np.nan) for r in method_data]
            
            # Remove NaN values
            summ_vals = [v for v in summ_vals if not np.isnan(v)]
            
            if traj_vals or summ_vals:
                methods.append(method_name)
                
                if traj_vals:
                    trajectory_means.append(np.mean(traj_vals))
                    trajectory_stds.append(np.std(traj_vals))
                else:
                    trajectory_means.append(0)
                    trajectory_stds.append(0)
                
                if summ_vals:
                    summary_means.append(np.mean(summ_vals))
                    summary_stds.append(np.std(summ_vals))
                else:
                    summary_means.append(0)
                    summary_stds.append(0)
        
        if methods:
            x = np.arange(len(methods))
            width = 0.35
            
            # Plot bars
            bars1 = ax.bar(x - width/2, trajectory_means, width, yerr=trajectory_stds,
                           label='Trajectory Finals', capsize=5, alpha=0.7)
            bars2 = ax.bar(x + width/2, summary_means, width, yerr=summary_stds,
                           label='Summary Stats', capsize=5, alpha=0.7)
            
            ax.set_xlabel('Method')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(' + ', '\n') for m in methods], rotation=0, ha='center')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_consistency_check.png', dpi=300, bbox_inches='tight')
    print("\nConsistency plot saved as 'data_consistency_check.png'")
    plt.show()


def main():
    """Main function to verify data consistency."""
    # Find the most recent results file
    results_files = list(Path('.').glob('experiment_results_*.json'))
    
    if not results_files:
        print("No experiment results files found. Please run an experiment first.")
        return
    
    # Use the most recent file
    results_file = sorted(results_files)[-1]
    print(f"Loading results from: {results_file}")
    
    # Load and analyze results
    results = load_experiment_results(results_file)
    compare_summary_and_trajectory_data(results)


if __name__ == "__main__":
    main()