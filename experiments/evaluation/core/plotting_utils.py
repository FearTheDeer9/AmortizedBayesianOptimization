"""
Plotting utilities for multi-SCM evaluation results.

Provides functions for visualizing:
- Average performance trajectories with confidence bands
- Per-SCM performance heatmaps
- Method comparison bar charts
- Summary statistics and reports
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_evaluation_results(results: Dict[str, List[Dict]], 
                           output_dir: Path,
                           timestamp: str,
                           scms: Optional[List[Any]] = None):
    """
    Generate comprehensive plots for evaluation results.
    
    Args:
        results: Dictionary mapping method names to lists of trajectory dicts
        output_dir: Directory to save plots
        timestamp: Timestamp string for file naming
        scms: Optional list of SCM objects for additional context
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute average trajectories
    averaged = compute_average_trajectories(results)
    
    # Create multi-panel figure for main results
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Normalized target value trajectories
    ax1 = plt.subplot(2, 3, 1)
    plot_target_trajectories(ax1, averaged, 'Normalized Target Value')
    
    # Plot 2: F1 score trajectories
    ax2 = plt.subplot(2, 3, 2)
    plot_metric_trajectories(ax2, averaged, 'f1', 'F1 Score')
    
    # Plot 3: SHD trajectories
    ax3 = plt.subplot(2, 3, 3)
    plot_metric_trajectories(ax3, averaged, 'shd', 'Structural Hamming Distance')
    
    # Plot 4: Final performance comparison
    ax4 = plt.subplot(2, 3, 4)
    plot_final_performance_bars(ax4, averaged)
    
    # Plot 5: Per-SCM heatmap (if we have enough SCMs)
    if results and len(next(iter(results.values()))) >= 5:
        ax5 = plt.subplot(2, 3, 5)
        plot_per_scm_heatmap(ax5, results, 'normalized_targets')
    
    # Plot 6: Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    n_scms = len(next(iter(results.values()))) if results else 0
    plot_summary_table(ax6, averaged, n_scms)
    
    plt.suptitle(f'Multi-SCM Evaluation Results ({timestamp})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save main figure
    fig_path = output_dir / f"evaluation_results_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved main results plot to {fig_path}")
    
    # Create detailed trajectory plots
    create_detailed_trajectory_plots(results, averaged, output_dir, timestamp)
    
    # Create method comparison plots
    create_method_comparison_plots(results, output_dir, timestamp)
    
    plt.close('all')
    logger.info(f"All plots saved to {output_dir}")


def compute_average_trajectories(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Compute mean and std trajectories across SCMs."""
    averaged = {}
    
    for method_name, method_results in results.items():
        if not method_results:
            continue
            
        # Stack trajectories from all SCMs
        all_normalized = []
        all_f1 = []
        all_shd = []
        
        for scm_result in method_results:
            if 'normalized_targets' in scm_result:
                all_normalized.append(scm_result['normalized_targets'])
            if 'f1_scores' in scm_result:
                all_f1.append(scm_result['f1_scores'])
            if 'shd_values' in scm_result:
                all_shd.append(scm_result['shd_values'])
        
        # Convert to numpy arrays and compute statistics
        if all_normalized:
            all_normalized = np.array(all_normalized)
            normalized_mean = np.nanmean(all_normalized, axis=0)
            normalized_std = np.nanstd(all_normalized, axis=0)
        else:
            normalized_mean = np.array([])
            normalized_std = np.array([])
            
        if all_f1:
            all_f1 = np.array(all_f1)
            f1_mean = np.nanmean(all_f1, axis=0)
            f1_std = np.nanstd(all_f1, axis=0)
        else:
            f1_mean = np.array([])
            f1_std = np.array([])
            
        if all_shd:
            all_shd = np.array(all_shd)
            shd_mean = np.nanmean(all_shd, axis=0)
            shd_std = np.nanstd(all_shd, axis=0)
        else:
            shd_mean = np.array([])
            shd_std = np.array([])
        
        averaged[method_name] = {
            'normalized_mean': normalized_mean,
            'normalized_std': normalized_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'shd_mean': shd_mean,
            'shd_std': shd_std
        }
    
    return averaged


def plot_target_trajectories(ax, averaged: Dict[str, Dict], title: str):
    """Plot normalized target value trajectories with confidence bands."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(averaged)))
    
    for (method_name, metrics), color in zip(averaged.items(), colors):
        if len(metrics['normalized_mean']) == 0:
            continue
            
        x = np.arange(len(metrics['normalized_mean']))
        mean = metrics['normalized_mean']
        std = metrics['normalized_std']
        
        # Plot mean line
        ax.plot(x, mean, label=method_name, color=color, linewidth=2)
        
        # Add confidence band (±1 std)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    
    ax.set_xlabel('Intervention Step')
    ax.set_ylabel('Normalized Target Value')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Invert y-axis if minimizing (lower is better)
    ax.invert_yaxis()


def plot_metric_trajectories(ax, averaged: Dict[str, Dict], metric: str, title: str):
    """Plot metric trajectories (F1 or SHD) with confidence bands."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(averaged)))
    
    for (method_name, metrics), color in zip(averaged.items(), colors):
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        
        if mean_key not in metrics or len(metrics[mean_key]) == 0:
            continue
            
        x = np.arange(len(metrics[mean_key]))
        mean = metrics[mean_key]
        std = metrics[std_key]
        
        # Plot mean line
        ax.plot(x, mean, label=method_name, color=color, linewidth=2)
        
        # Add confidence band
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    
    ax.set_xlabel('Intervention Step')
    ax.set_ylabel(title)
    ax.set_title(f'{title} Trajectory')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # For SHD, lower is better
    if metric == 'shd':
        ax.invert_yaxis()


def plot_final_performance_bars(ax, averaged: Dict[str, Dict]):
    """Plot bar chart comparing final performance across methods."""
    method_names = []
    final_targets = []
    final_f1s = []
    final_shds = []
    
    for method_name, metrics in averaged.items():
        method_names.append(method_name)
        
        # Get final values (last intervention)
        if len(metrics['normalized_mean']) > 0:
            final_targets.append(metrics['normalized_mean'][-1])
        else:
            final_targets.append(np.nan)
            
        if len(metrics['f1_mean']) > 0:
            final_f1s.append(metrics['f1_mean'][-1])
        else:
            final_f1s.append(np.nan)
            
        if len(metrics['shd_mean']) > 0:
            final_shds.append(metrics['shd_mean'][-1])
        else:
            final_shds.append(np.nan)
    
    # Create grouped bar chart
    x = np.arange(len(method_names))
    width = 0.25
    
    # Normalize metrics to [0, 1] for comparison
    norm_targets = np.array(final_targets)
    norm_f1s = np.array(final_f1s)
    norm_shds = 1 - (np.array(final_shds) / np.nanmax(final_shds)) if np.nanmax(final_shds) > 0 else np.zeros_like(final_shds)
    
    ax.bar(x - width, 1 - norm_targets, width, label='Target (lower is better)', color='steelblue')
    ax.bar(x, norm_f1s, width, label='F1 Score', color='forestgreen')
    ax.bar(x + width, norm_shds, width, label='1 - Normalized SHD', color='coral')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Performance (higher is better)')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def plot_per_scm_heatmap(ax, results: Dict[str, List[Dict]], metric: str):
    """Create heatmap showing per-SCM performance."""
    # Create matrix of final values
    method_names = list(results.keys())
    n_scms = len(results[method_names[0]])
    
    matrix = np.zeros((len(method_names), n_scms))
    
    for i, method in enumerate(method_names):
        for j, scm_result in enumerate(results[method]):
            if metric in scm_result and len(scm_result[metric]) > 0:
                # Use final value
                matrix[i, j] = scm_result[metric][-1]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    sns.heatmap(matrix, ax=ax, annot=True, fmt='.2f', 
                xticklabels=[f'SCM {i+1}' for i in range(n_scms)],
                yticklabels=method_names,
                cmap='RdYlGn_r' if metric == 'normalized_targets' else 'RdYlGn',
                cbar_kws={'label': 'Final Normalized Target'})
    
    ax.set_title('Per-SCM Final Performance')
    ax.set_xlabel('SCM')
    ax.set_ylabel('Method')


def plot_summary_table(ax, averaged: Dict[str, Dict], n_scms: int):
    """Create summary statistics table."""
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Method', 'Final Target↓', 'Best Target↓', 'Max F1↑', 'Min SHD↓']
    
    for method_name, metrics in averaged.items():
        row = [method_name]
        
        # Final and best normalized target
        if len(metrics['normalized_mean']) > 0:
            row.append(f"{metrics['normalized_mean'][-1]:.3f}")
            row.append(f"{np.min(metrics['normalized_mean']):.3f}")
        else:
            row.extend(['N/A', 'N/A'])
        
        # Max F1
        if len(metrics['f1_mean']) > 0:
            row.append(f"{np.nanmax(metrics['f1_mean']):.3f}")
        else:
            row.append('N/A')
        
        # Min SHD
        if len(metrics['shd_mean']) > 0:
            row.append(f"{np.nanmin(metrics['shd_mean']):.1f}")
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title(f'Summary Statistics (n={n_scms} SCMs)', fontweight='bold', pad=20)


def create_detailed_trajectory_plots(results: Dict[str, List[Dict]], 
                                    averaged: Dict[str, Dict],
                                    output_dir: Path, 
                                    timestamp: str):
    """Create detailed individual trajectory plots."""
    
    # Plot 1: Detailed target value trajectories with individual SCM traces
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (method_name, method_results), color in zip(results.items(), colors):
        if not method_results:
            continue
            
        # Plot individual SCM trajectories (light)
        for scm_result in method_results:
            if 'normalized_targets' in scm_result:
                x = np.arange(len(scm_result['normalized_targets']))
                axes[0, 0].plot(x, scm_result['normalized_targets'], 
                              alpha=0.1, color=color, linewidth=0.5)
        
        # Plot average (bold)
        if method_name in averaged and len(averaged[method_name]['normalized_mean']) > 0:
            x = np.arange(len(averaged[method_name]['normalized_mean']))
            axes[0, 0].plot(x, averaged[method_name]['normalized_mean'],
                          label=method_name, color=color, linewidth=2)
    
    axes[0, 0].set_title('Target Value Trajectories (All SCMs)')
    axes[0, 0].set_xlabel('Intervention Step')
    axes[0, 0].set_ylabel('Normalized Target Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].invert_yaxis()
    
    # Plot 2: Learning curves (improvement over baseline)
    for (method_name, method_results), color in zip(results.items(), colors):
        if not method_results:
            continue
            
        improvements = []
        for scm_result in method_results:
            if 'normalized_targets' in scm_result and len(scm_result['normalized_targets']) > 1:
                initial = scm_result['normalized_targets'][0]
                trajectory = np.array(scm_result['normalized_targets'])
                improvement = (initial - trajectory) / (initial + 1e-10)
                improvements.append(improvement)
        
        if improvements:
            mean_improvement = np.mean(improvements, axis=0)
            x = np.arange(len(mean_improvement))
            axes[0, 1].plot(x, mean_improvement * 100, 
                          label=method_name, color=color, linewidth=2)
    
    axes[0, 1].set_title('Relative Improvement from Initial Value')
    axes[0, 1].set_xlabel('Intervention Step')
    axes[0, 1].set_ylabel('Improvement (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Structure learning progression
    for (method_name, method_results), color in zip(results.items(), colors):
        if 'Surrogate' in method_name:  # Only plot methods with surrogate
            f1_trajectories = []
            for scm_result in method_results:
                if 'f1_scores' in scm_result:
                    f1_trajectories.append(scm_result['f1_scores'])
            
            if f1_trajectories:
                mean_f1 = np.nanmean(f1_trajectories, axis=0)
                x = np.arange(len(mean_f1))
                axes[1, 0].plot(x, mean_f1, label=method_name, 
                              color=color, linewidth=2, marker='o', markersize=4)
    
    axes[1, 0].set_title('Structure Learning Progress (Methods with Surrogate)')
    axes[1, 0].set_xlabel('Intervention Step')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot 4: Convergence analysis
    for (method_name, method_results), color in zip(results.items(), colors):
        convergence_steps = []
        for scm_result in method_results:
            if 'normalized_targets' in scm_result:
                trajectory = np.array(scm_result['normalized_targets'])
                if len(trajectory) > 5:
                    # Find where improvement plateaus (change < 1% for 3 steps)
                    for i in range(3, len(trajectory)-2):
                        window = trajectory[i:i+3]
                        if np.std(window) < 0.01:
                            convergence_steps.append(i)
                            break
                    else:
                        convergence_steps.append(len(trajectory))
        
        if convergence_steps:
            axes[1, 1].boxplot(convergence_steps, positions=[len(results) - list(results.keys()).index(method_name)],
                             widths=0.6, patch_artist=True,
                             boxprops=dict(facecolor=color, alpha=0.7))
    
    axes[1, 1].set_title('Convergence Analysis')
    axes[1, 1].set_xlabel('Method')
    axes[1, 1].set_ylabel('Steps to Convergence')
    axes[1, 1].set_xticks(range(1, len(results) + 1))
    axes[1, 1].set_xticklabels(list(results.keys())[::-1], rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Detailed Trajectory Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / f"detailed_trajectories_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved detailed trajectories to {fig_path}")
    plt.close()


def create_method_comparison_plots(results: Dict[str, List[Dict]], 
                                  output_dir: Path,
                                  timestamp: str):
    """Create pairwise method comparison plots."""
    
    # Extract final performance for each method on each SCM
    method_names = list(results.keys())
    n_methods = len(method_names)
    
    if n_methods < 2:
        return
    
    # Create comparison matrix figure
    fig, axes = plt.subplots(n_methods-1, n_methods-1, figsize=(12, 12))
    
    if n_methods == 2:
        axes = np.array([[axes]])
    elif n_methods == 3:
        axes = axes.reshape(1, -1)
    
    for i in range(n_methods-1):
        for j in range(i+1, n_methods):
            ax_idx_i = i
            ax_idx_j = j - 1
            
            if n_methods > 2:
                ax = axes[ax_idx_i, ax_idx_j] if ax_idx_i < axes.shape[0] and ax_idx_j < axes.shape[1] else None
            else:
                ax = axes[0, 0]
                
            if ax is None:
                continue
            
            method1 = method_names[i]
            method2 = method_names[j]
            
            # Get final values for both methods
            values1 = []
            values2 = []
            
            for scm_idx in range(len(results[method1])):
                if 'normalized_targets' in results[method1][scm_idx] and \
                   'normalized_targets' in results[method2][scm_idx] and \
                   len(results[method1][scm_idx]['normalized_targets']) > 0 and \
                   len(results[method2][scm_idx]['normalized_targets']) > 0:
                    val1 = results[method1][scm_idx]['normalized_targets'][-1]
                    val2 = results[method2][scm_idx]['normalized_targets'][-1]
                    values1.append(val1)
                    values2.append(val2)
            
            if values1 and values2:
                ax.scatter(values1, values2, alpha=0.6, s=50)
                
                # Add diagonal line
                lims = [min(min(values1), min(values2)), 
                       max(max(values1), max(values2))]
                ax.plot(lims, lims, 'k--', alpha=0.3)
                
                # Add labels
                ax.set_xlabel(f'{method1}')
                ax.set_ylabel(f'{method2}')
                ax.set_title(f'{method1} vs {method2}', fontsize=9)
                ax.grid(True, alpha=0.3)
                
                # Color points by which method is better
                better = np.array(values1) < np.array(values2)  # Lower is better
                colors = ['green' if b else 'red' for b in better]
                ax.scatter(values1, values2, c=colors, alpha=0.6, s=50)
                
                # Add win rate
                win_rate = np.mean(better) * 100
                ax.text(0.05, 0.95, f'{method1} wins: {win_rate:.0f}%',
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top')
    
    # Hide unused subplots
    for i in range(n_methods-1):
        for j in range(n_methods-1):
            if j <= i and n_methods > 2:
                axes[i, j].axis('off')
    
    plt.suptitle('Pairwise Method Comparisons (Final Performance)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    fig_path = output_dir / f"method_comparisons_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved method comparisons to {fig_path}")
    plt.close()


def create_report(results: Dict[str, List[Dict]], 
                 output_dir: Path,
                 timestamp: str,
                 config: Dict[str, Any]):
    """Generate a text report summarizing the evaluation."""
    report_path = output_dir / f"evaluation_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MULTI-SCM EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Number of SCMs and methods
        n_scms = len(next(iter(results.values()))) if results else 0
        f.write(f"Number of SCMs evaluated: {n_scms}\n")
        f.write(f"Number of methods compared: {len(results)}\n")
        f.write(f"Methods: {', '.join(results.keys())}\n\n")
        
        # Compute averages
        averaged = compute_average_trajectories(results)
        
        # Performance summary
        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        for method_name, metrics in averaged.items():
            f.write(f"\n{method_name}:\n")
            
            if len(metrics['normalized_mean']) > 0:
                f.write(f"  Target Value:\n")
                f.write(f"    Initial: {metrics['normalized_mean'][0]:.4f}\n")
                f.write(f"    Final: {metrics['normalized_mean'][-1]:.4f}\n")
                f.write(f"    Best: {np.min(metrics['normalized_mean']):.4f}\n")
                f.write(f"    Improvement: {(metrics['normalized_mean'][0] - metrics['normalized_mean'][-1]):.4f}\n")
            
            if len(metrics['f1_mean']) > 0:
                f.write(f"  Structure Learning:\n")
                f.write(f"    Max F1: {np.nanmax(metrics['f1_mean']):.4f}\n")
                f.write(f"    Final F1: {metrics['f1_mean'][-1]:.4f}\n")
            
            if len(metrics['shd_mean']) > 0:
                f.write(f"    Min SHD: {np.nanmin(metrics['shd_mean']):.2f}\n")
                f.write(f"    Final SHD: {metrics['shd_mean'][-1]:.2f}\n")
        
        # Statistical comparisons
        f.write("\n\nSTATISTICAL COMPARISONS\n")
        f.write("-" * 40 + "\n")
        
        # Pairwise comparisons
        method_list = list(results.keys())
        for i in range(len(method_list)):
            for j in range(i+1, len(method_list)):
                method1, method2 = method_list[i], method_list[j]
                
                # Compare final values
                finals1 = [r['normalized_targets'][-1] for r in results[method1] 
                          if 'normalized_targets' in r and len(r['normalized_targets']) > 0]
                finals2 = [r['normalized_targets'][-1] for r in results[method2]
                          if 'normalized_targets' in r and len(r['normalized_targets']) > 0]
                
                if finals1 and finals2:
                    wins = sum(f1 < f2 for f1, f2 in zip(finals1, finals2))
                    total = len(finals1)
                    f.write(f"\n{method1} vs {method2}:\n")
                    f.write(f"  {method1} wins: {wins}/{total} ({100*wins/total:.1f}%)\n")
                    f.write(f"  Mean difference: {np.mean(finals1) - np.mean(finals2):.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
    
    logger.info(f"Saved evaluation report to {report_path}")