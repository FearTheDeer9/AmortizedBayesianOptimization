#!/usr/bin/env python3
"""
Plotting utilities for evaluation results.
Generates trajectory plots for F1, SHD, and normalized target values.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_evaluation_trajectories(
    results_path: Path,
    output_dir: Optional[Path] = None,
    show_plots: bool = True
) -> None:
    """
    Plot evaluation trajectories from saved results.
    
    Args:
        results_path: Path to JSON results file from full_evaluation.py
        output_dir: Directory to save plots (default: same as results)
        show_plots: Whether to display plots
    """
    
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    if output_dir is None:
        output_dir = results_path.parent
    
    # Check if it's progression data or single evaluation
    if 'episodes' in data:
        # Single evaluation with episodes
        plot_single_evaluation(data, output_dir, show_plots)
    else:
        # Assume it's raw trajectory data
        plot_trajectory_comparison(data, output_dir, show_plots)


def plot_single_evaluation(
    data: Dict[str, Any],
    output_dir: Path,
    show_plots: bool = True
) -> None:
    """Plot trajectories from a single evaluation run."""
    
    episodes = data['episodes']
    metadata = data.get('metadata', {})
    baselines = data.get('baselines', {})
    
    # Extract trajectories for policy
    all_f1 = []
    all_targets = []
    all_parent_rates = []
    
    for episode in episodes:
        all_f1.append(episode['f1_scores'])
        all_targets.append(episode['target_values'])
        
        # Calculate cumulative parent selection rate
        parent_selections = episode['parent_selections']
        cumulative_rate = np.cumsum(parent_selections) / np.arange(1, len(parent_selections) + 1)
        all_parent_rates.append(cumulative_rate)
    
    # Extract baseline trajectories if available
    baseline_data = {}
    for baseline_name, baseline_info in baselines.items():
        baseline_episodes = baseline_info.get('episodes', [])
        baseline_f1 = []
        baseline_targets = []
        baseline_parent_rates = []
        
        for episode in baseline_episodes:
            baseline_f1.append(episode['f1_scores'])
            baseline_targets.append(episode['target_values'])
            
            parent_selections = episode['parent_selections']
            cumulative_rate = np.cumsum(parent_selections) / np.arange(1, len(parent_selections) + 1)
            baseline_parent_rates.append(cumulative_rate)
        
        baseline_data[baseline_name] = {
            'f1': baseline_f1,
            'targets': baseline_targets,
            'parent_rates': baseline_parent_rates
        }
    
    # Find global min and max across all methods for normalization
    global_min = float('inf')
    global_max = float('-inf')
    
    # Check policy targets
    for targets in all_targets:
        if targets:
            global_min = min(global_min, min(targets))
            global_max = max(global_max, max(targets))
    
    # Check baseline targets
    for baseline_name, bdata in baseline_data.items():
        for targets in bdata.get('targets', []):
            if targets:
                global_min = min(global_min, min(targets))
                global_max = max(global_max, max(targets))
    
    # Normalize all targets using the global scale
    all_targets_normalized = []
    for targets in all_targets:
        if targets:
            if global_max > global_min:
                normalized = [(v - global_min) / (global_max - global_min) for v in targets]
            else:
                normalized = [0.5] * len(targets)
            all_targets_normalized.append(normalized)
    
    # Create figure with subplots (increased to 2x3 for actual values)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Evaluation Results - Iteration {metadata.get('policy_iteration', 'N/A')}", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: F1 Score Trajectory
    ax1 = axes[0, 0]
    plot_trajectory_with_confidence(all_f1, ax1, "F1 Score", "Intervention", "F1 Score", 
                                   color='blue', label='Policy')
    
    # Add baseline F1 trajectories
    colors = {'random': 'gray', 'oracle': 'gold'}
    for baseline_name, bdata in baseline_data.items():
        if bdata['f1']:
            plot_trajectory_with_confidence(bdata['f1'], ax1, None, None, None,
                                           color=colors.get(baseline_name, 'black'),
                                           label=baseline_name.capitalize())
    
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax1.legend()
    
    # Plot 2: Normalized Target Value
    ax2 = axes[0, 1]
    
    # Normalize baseline targets using the same global scale
    for baseline_name, bdata in baseline_data.items():
        if bdata['targets']:
            baseline_normalized = []
            for targets in bdata['targets']:
                if targets:
                    if global_max > global_min:
                        normalized = [(v - global_min) / (global_max - global_min) for v in targets]
                    else:
                        normalized = [0.5] * len(targets)
                    baseline_normalized.append(normalized)
            bdata['targets_normalized'] = baseline_normalized
    
    plot_trajectory_with_confidence(all_targets_normalized, ax2, 
                                   "Normalized Target Value", "Intervention", 
                                   "Normalized Target (0=best)", color='green', label='Policy')
    
    # Add baseline normalized target trajectories
    for baseline_name, bdata in baseline_data.items():
        if 'targets_normalized' in bdata and bdata['targets_normalized']:
            plot_trajectory_with_confidence(bdata['targets_normalized'], ax2, None, None, None,
                                           color=colors.get(baseline_name, 'black'),
                                           label=baseline_name.capitalize())
    
    ax2.legend()
    
    # Plot 3: Actual Target Values (not normalized)
    ax3 = axes[0, 2]
    plot_trajectory_with_confidence(all_targets, ax3,
                                   "Actual Target Values", "Intervention",
                                   "Target Value", color='green', label='Policy')
    
    # Add baseline actual target trajectories
    for baseline_name, bdata in baseline_data.items():
        if bdata['targets']:
            plot_trajectory_with_confidence(bdata['targets'], ax3, None, None, None,
                                           color=colors.get(baseline_name, 'black'),
                                           label=baseline_name.capitalize())
    
    ax3.legend()
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.1)
    
    # Plot 4: Parent Selection Rate
    ax4 = axes[1, 0]
    plot_trajectory_with_confidence(all_parent_rates, ax4,
                                   "Cumulative Parent Selection Rate", "Intervention",
                                   "Parent Selection Rate", color='orange', label='Policy')
    
    # Add baseline parent selection rates
    for baseline_name, bdata in baseline_data.items():
        if bdata['parent_rates']:
            plot_trajectory_with_confidence(bdata['parent_rates'], ax3, None, None, None,
                                           color=colors.get(baseline_name, 'black'),
                                           label=baseline_name.capitalize())
    
    # Calculate expected random rate (depends on structure)
    num_vars = len(episodes[0].get('scm_info', {}).get('true_parents', [])) + 1
    if num_vars > 0:
        random_rate = len(episodes[0].get('scm_info', {}).get('true_parents', [])) / num_vars
        ax3.axhline(y=random_rate, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Expected random ({random_rate:.2f})')
        ax3.legend()
    
    # Plot 5: Summary Statistics
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Calculate summary stats
    final_f1s = [ep['f1_scores'][-1] if ep['f1_scores'] else 0 for ep in episodes]
    final_targets = [ep['target_values'][-1] if ep['target_values'] else 0 for ep in episodes]
    parent_rates = [np.mean(ep['parent_selections']) for ep in episodes]
    
    summary_text = f"""
    Summary Statistics (n={len(episodes)} episodes)
    
    F1 Score:
      Mean: {np.mean(final_f1s):.3f} ± {np.std(final_f1s):.3f}
      Best: {np.max(final_f1s):.3f}
      Worst: {np.min(final_f1s):.3f}
    
    Parent Selection Rate:
      Mean: {np.mean(parent_rates):.2%}
      Std: {np.std(parent_rates):.2%}
    
    Target Value (final):
      Mean: {np.mean(final_targets):.3f}
      Best: {np.min(final_targets):.3f}
    """
    
    ax5.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace')
    
    # Plot 6: Method Comparison Bar Chart
    ax6 = axes[1, 2]
    if baseline_data:
        # Gather best targets for comparison
        methods = ['Policy']
        best_targets = [min([ep['target_values'][-1] for ep in episodes if ep['target_values']])]
        
        for baseline_name, bdata in baseline_data.items():
            if bdata.get('episodes'):
                methods.append(baseline_name.capitalize())
                baseline_best = min([ep['target_values'][-1] for ep in bdata['episodes'] if ep.get('target_values')])
                best_targets.append(baseline_best)
        
        x_pos = np.arange(len(methods))
        bars = ax6.bar(x_pos, best_targets, color=['green', 'gray', 'gold'][:len(methods)])
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(methods)
        ax6.set_ylabel('Best Target Value')
        ax6.set_title('Method Comparison (Best Target)')
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.1)
        
        # Add value labels on bars
        for bar, val in zip(bars, best_targets):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    else:
        ax6.axis('off')
        ax6.text(0.5, 0.5, 'No baseline data', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "evaluation_trajectories.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_trajectory_with_confidence(
    trajectories: List[List[float]],
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: str = 'blue',
    label: Optional[str] = None
) -> None:
    """
    Plot mean trajectory with confidence intervals.
    
    Args:
        trajectories: List of trajectories (one per episode)
        ax: Matplotlib axes to plot on
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
    """
    
    if not trajectories:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        if title:
            ax.set_title(title)
        return
    
    # Convert to numpy array, padding shorter trajectories
    max_len = max(len(t) for t in trajectories)
    padded = []
    
    for traj in trajectories:
        if len(traj) < max_len:
            # Pad with last value
            padded_traj = traj + [traj[-1]] * (max_len - len(traj))
        else:
            padded_traj = traj
        padded.append(padded_traj)
    
    trajectories_array = np.array(padded)
    
    # Calculate statistics
    mean_traj = np.mean(trajectories_array, axis=0)
    std_traj = np.std(trajectories_array, axis=0)
    sem_traj = std_traj / np.sqrt(len(trajectories))
    
    x = np.arange(len(mean_traj))
    
    # Plot mean with confidence interval
    ax.plot(x, mean_traj, color=color, linewidth=2, label=label if label else 'Mean')
    ax.fill_between(x, mean_traj - sem_traj, mean_traj + sem_traj,
                    alpha=0.3, color=color)
    
    # Plot individual trajectories
    for traj in trajectories_array:
        ax.plot(x, traj, color=color, alpha=0.1, linewidth=0.5)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_training_progression(
    csv_path: Path,
    output_dir: Optional[Path] = None,
    show_plots: bool = True
) -> None:
    """
    Plot training progression from CSV file.
    
    Args:
        csv_path: Path to progression CSV from full_evaluation.py
        output_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    
    if output_dir is None:
        output_dir = csv_path.parent
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Progression Analysis", fontsize=14, fontweight='bold')
    
    # Plot 1: F1 Score progression
    ax1 = axes[0, 0]
    ax1.errorbar(df['iteration'], df['final_f1_mean'], 
                yerr=df['final_f1_std'], marker='o', capsize=5)
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Structure Learning (F1) vs Training')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['iteration'], df['final_f1_mean'], 1)
    p = np.poly1d(z)
    ax1.plot(df['iteration'], p(df['iteration']), 'r--', alpha=0.5, 
            label=f'Trend: {z[0]:.2e}x + {z[1]:.2f}')
    ax1.legend()
    
    # Plot 2: Parent Selection Rate
    ax2 = axes[0, 1]
    ax2.errorbar(df['iteration'], df['parent_selection_rate_mean'],
                yerr=df['parent_selection_rate_std'], marker='o', capsize=5,
                color='orange')
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Parent Selection Rate')
    ax2.set_title('Parent Selection Rate vs Training')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best Target Value
    ax3 = axes[1, 0]
    ax3.errorbar(df['iteration'], df['best_target_mean'],
                yerr=df['best_target_std'], marker='o', capsize=5,
                color='green')
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Best Target Value')
    ax3.set_title('Target Optimization vs Training')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance degradation analysis
    ax4 = axes[1, 1]
    
    # Calculate relative performance
    initial_f1 = df['final_f1_mean'].iloc[0]
    relative_f1 = df['final_f1_mean'] / initial_f1 if initial_f1 > 0 else df['final_f1_mean']
    
    ax4.plot(df['iteration'], relative_f1, marker='o', color='red', linewidth=2)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.fill_between(df['iteration'], 1.0, relative_f1, 
                     where=(relative_f1 < 1.0), alpha=0.3, color='red',
                     label='Performance loss')
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Relative F1 (vs initial)')
    ax4.set_title('Performance Degradation')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "training_progression.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def plot_trajectory_comparison(
    results: Dict[str, List[Dict]],
    output_dir: Path,
    show_plots: bool = True
) -> None:
    """
    Plot trajectory comparison between different methods.
    Used for multi-method evaluation results.
    """
    
    # Calculate average trajectories
    averaged = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
        
        # Stack trajectories
        all_f1 = []
        all_targets = []
        
        for episode in method_results:
            if 'f1_scores' in episode:
                all_f1.append(episode['f1_scores'])
            if 'normalized_targets' in episode:
                all_targets.append(episode['normalized_targets'])
        
        if all_f1 or all_targets:
            averaged[method_name] = {
                'f1': all_f1,
                'targets': all_targets
            }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Method Comparison", fontsize=14, fontweight='bold')
    
    colors = {'Random': 'gray', 'Oracle': 'gold', 
              'Trained Policy': 'blue', 'Trained Policy + Surrogate': 'green'}
    
    # Plot F1 comparison
    ax1 = axes[0]
    for method_name, data in averaged.items():
        if 'f1' in data and data['f1']:
            color = colors.get(method_name, 'black')
            plot_trajectory_with_confidence(data['f1'], ax1, 
                                          "F1 Score Comparison", "Intervention",
                                          "F1 Score", color=color)
    
    # Plot target comparison
    ax2 = axes[1]
    for method_name, data in averaged.items():
        if 'targets' in data and data['targets']:
            color = colors.get(method_name, 'black')
            plot_trajectory_with_confidence(data['targets'], ax2,
                                          "Target Value Comparison", "Intervention",
                                          "Normalized Target", color=color)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "method_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def main():
    """Main entry point for plotting utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument('results_path', type=Path,
                       help='Path to results file (JSON or CSV)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Directory to save plots')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots')
    
    args = parser.parse_args()
    
    if args.results_path.suffix == '.json':
        plot_evaluation_trajectories(args.results_path, args.output_dir, 
                                    not args.no_show)
    elif args.results_path.suffix == '.csv':
        plot_training_progression(args.results_path, args.output_dir,
                                not args.no_show)
    else:
        print(f"Unsupported file type: {args.results_path.suffix}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())