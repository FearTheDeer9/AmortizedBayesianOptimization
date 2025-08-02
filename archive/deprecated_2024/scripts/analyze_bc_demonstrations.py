#!/usr/bin/env python3
"""
Analyze BC expert demonstrations to understand training signal quality.

This script loads saved BC demonstrations and analyzes:
- Distribution of selected variables
- Value distributions
- Oracle decision quality
- Diversity of demonstrations
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bc_checkpoint(checkpoint_path: Path):
    """Load BC checkpoint and extract demonstrations."""
    checkpoint_file = checkpoint_path / 'checkpoint.pkl'
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract expert buffer (stored during training)
    expert_buffer = checkpoint.get('expert_buffer', [])
    config = checkpoint.get('config', {})
    
    logger.info(f"Loaded {len(expert_buffer)} demonstrations")
    logger.info(f"Expert strategy: {config.get('expert_strategy', 'unknown')}")
    
    return expert_buffer, config


def analyze_variable_distribution(demonstrations):
    """Analyze which variables are selected for intervention."""
    var_counts = defaultdict(int)
    var_counts_by_target = defaultdict(lambda: defaultdict(int))
    
    for demo in demonstrations:
        variables = demo['variables']
        target_idx = demo['target_idx']
        expert_var_idx = demo['expert_var_idx']
        
        target_var = variables[target_idx]
        selected_var = variables[expert_var_idx]
        
        var_counts[selected_var] += 1
        var_counts_by_target[target_var][selected_var] += 1
    
    return var_counts, var_counts_by_target


def analyze_value_distribution(demonstrations):
    """Analyze intervention value distributions."""
    values = [demo['expert_value'] for demo in demonstrations]
    values_by_var = defaultdict(list)
    
    for demo in demonstrations:
        variables = demo['variables']
        expert_var_idx = demo['expert_var_idx']
        selected_var = variables[expert_var_idx]
        values_by_var[selected_var].append(demo['expert_value'])
    
    return np.array(values), values_by_var


def analyze_tensor_states(demonstrations):
    """Analyze the tensor states when decisions were made."""
    # Track how many observations/interventions existed at decision time
    trajectory_lengths = []
    intervention_counts = []
    
    for demo in demonstrations:
        tensor = demo['tensor']
        # Count non-zero timesteps (rough estimate)
        # In 3-channel format: [T, n_vars, 3]
        # Channel 2 indicates interventions
        intervention_mask = tensor[:, :, 2]
        n_interventions = np.sum(intervention_mask.any(axis=1))
        trajectory_length = np.sum(tensor[:, :, 0].any(axis=1))
        
        trajectory_lengths.append(trajectory_length)
        intervention_counts.append(n_interventions)
    
    return trajectory_lengths, intervention_counts


def plot_analysis_results(var_counts, var_counts_by_target, values, 
                         trajectory_lengths, output_dir):
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Variable selection distribution
    plt.figure(figsize=(10, 6))
    vars_sorted = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)
    variables, counts = zip(*vars_sorted) if vars_sorted else ([], [])
    plt.bar(variables, counts)
    plt.xlabel('Variable')
    plt.ylabel('Selection Count')
    plt.title('Expert Variable Selection Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'variable_distribution.png')
    plt.close()
    
    # 2. Value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Intervention Value')
    plt.ylabel('Count')
    plt.title('Expert Intervention Value Distribution')
    plt.axvline(np.mean(values), color='red', linestyle='--', 
                label=f'Mean: {np.mean(values):.3f}')
    plt.axvline(np.median(values), color='green', linestyle='--', 
                label=f'Median: {np.median(values):.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'value_distribution.png')
    plt.close()
    
    # 3. Trajectory lengths at decision time
    plt.figure(figsize=(10, 6))
    plt.scatter(trajectory_lengths, values, alpha=0.5)
    plt.xlabel('Trajectory Length at Decision')
    plt.ylabel('Intervention Value')
    plt.title('Intervention Values vs Trajectory Length')
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_analysis.png')
    plt.close()
    
    # 4. Per-target analysis
    n_targets = len(var_counts_by_target)
    if n_targets > 0 and n_targets <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (target, var_dict) in enumerate(var_counts_by_target.items()):
            if idx >= 6:
                break
            ax = axes[idx]
            
            vars_sorted = sorted(var_dict.items(), key=lambda x: x[1], reverse=True)
            if vars_sorted:
                variables, counts = zip(*vars_sorted)
                ax.bar(variables, counts)
                ax.set_title(f'Target: {target}')
                ax.set_xlabel('Selected Variable')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(var_counts_by_target), 6):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_target_distribution.png')
        plt.close()


def analyze_demonstration_quality(demonstrations):
    """Analyze quality metrics of demonstrations."""
    logger.info("\nDemonstration Quality Analysis:")
    
    # Check diversity
    unique_selections = set()
    for demo in demonstrations:
        variables = demo['variables']
        target_idx = demo['target_idx']
        expert_var_idx = demo['expert_var_idx']
        target = variables[target_idx]
        selected = variables[expert_var_idx]
        unique_selections.add((target, selected))
    
    diversity_ratio = len(unique_selections) / len(demonstrations) if demonstrations else 0
    logger.info(f"Unique (target, intervention) pairs: {len(unique_selections)}")
    logger.info(f"Diversity ratio: {diversity_ratio:.3f}")
    
    # Check if selections make sense (not intervening on isolated variables)
    # This would require SCM structure info, but we can check basic patterns
    same_var_interventions = sum(
        1 for d in demonstrations 
        if d['variables'][d['target_idx']] == d['variables'][d['expert_var_idx']]
    )
    logger.info(f"Interventions on target variable itself: {same_var_interventions}")
    
    # Value statistics
    values = [d['expert_value'] for d in demonstrations]
    if values:
        logger.info(f"Value range: [{min(values):.3f}, {max(values):.3f}]")
        logger.info(f"Value std: {np.std(values):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Analyze BC expert demonstrations')
    parser.add_argument('checkpoint', type=str, help='Path to BC checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='bc_analysis', 
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Load demonstrations
    checkpoint_path = Path(args.checkpoint)
    demonstrations, config = load_bc_checkpoint(checkpoint_path)
    
    if not demonstrations:
        logger.error("No demonstrations found in checkpoint!")
        return
    
    # Analyze demonstrations
    var_counts, var_counts_by_target = analyze_variable_distribution(demonstrations)
    values, values_by_var = analyze_value_distribution(demonstrations)
    trajectory_lengths, intervention_counts = analyze_tensor_states(demonstrations)
    
    # Print summary statistics
    logger.info(f"\nTotal demonstrations: {len(demonstrations)}")
    logger.info(f"Unique variables intervened on: {len(var_counts)}")
    logger.info(f"Most selected variable: {max(var_counts.items(), key=lambda x: x[1]) if var_counts else 'None'}")
    
    # Create plots
    plot_analysis_results(var_counts, var_counts_by_target, values, 
                         trajectory_lengths, args.output_dir)
    
    # Analyze quality
    analyze_demonstration_quality(demonstrations)
    
    logger.info(f"\nAnalysis complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()