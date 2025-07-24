"""
Visualization utilities for BC method performance.

Creates plots for SHD, F1, and target value trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

def plot_performance_trajectories(
    results_dict: Dict[str, Dict],
    figsize: Tuple[int, int] = (15, 10),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot SHD, F1, and target value trajectories for multiple methods.
    
    Args:
        results_dict: Dictionary mapping method names to results with performance_trajectory
        figsize: Figure size
        title: Overall title for the figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title or 'BC Method Performance Trajectories', fontsize=16)
    
    # Color palette
    colors = sns.color_palette("husl", len(results_dict))
    
    # 1. Target Value Trajectory
    ax = axes[0, 0]
    for i, (method_name, result) in enumerate(results_dict.items()):
        if 'performance_trajectory' in result:
            trajectory = result['performance_trajectory']
            arrays = trajectory.get_arrays()
            
            steps = arrays.get('steps', [])
            target_values = arrays.get('target_values', [])
            
            if len(steps) > 0:
                ax.plot(steps, target_values, label=method_name, 
                       color=colors[i], marker='o', markersize=4, alpha=0.7)
    
    ax.set_xlabel('Intervention Step')
    ax.set_ylabel('Target Value')
    ax.set_title('Target Value over Interventions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cumulative Best Target Value
    ax = axes[0, 1]
    for i, (method_name, result) in enumerate(results_dict.items()):
        if 'performance_trajectory' in result:
            trajectory = result['performance_trajectory']
            arrays = trajectory.get_arrays()
            
            steps = arrays.get('steps', [])
            cumulative_best = arrays.get('cumulative_best', [])
            
            if len(steps) > 0:
                ax.plot(steps, cumulative_best, label=method_name,
                       color=colors[i], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Intervention Step')
    ax.set_ylabel('Best Target Value Found')
    ax.set_title('Cumulative Best Target Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. SHD Trajectory (if available)
    ax = axes[1, 0]
    has_shd = False
    for i, (method_name, result) in enumerate(results_dict.items()):
        if 'performance_trajectory' in result:
            trajectory = result['performance_trajectory']
            arrays = trajectory.get_arrays()
            
            steps = arrays.get('steps', [])
            shd = arrays.get('shd', [])
            
            if len(steps) > 0 and len(shd) > 0 and not np.all(np.isnan(shd)):
                has_shd = True
                # Filter out NaN values
                valid_mask = ~np.isnan(shd)
                valid_steps = np.array(steps)[valid_mask]
                valid_shd = np.array(shd)[valid_mask]
                
                if len(valid_steps) > 0:
                    ax.plot(valid_steps, valid_shd, label=method_name,
                           color=colors[i], marker='s', markersize=4, alpha=0.7)
    
    if has_shd:
        ax.set_xlabel('Intervention Step')
        ax.set_ylabel('Structural Hamming Distance')
        ax.set_title('SHD over Interventions (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'SHD not available\n(Graph estimates not tracked)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('SHD over Interventions')
    
    # 4. F1 Score Trajectory (if available)
    ax = axes[1, 1]
    has_f1 = False
    for i, (method_name, result) in enumerate(results_dict.items()):
        if 'performance_trajectory' in result:
            trajectory = result['performance_trajectory']
            arrays = trajectory.get_arrays()
            
            steps = arrays.get('steps', [])
            f1_score = arrays.get('f1_score', [])
            
            if len(steps) > 0 and len(f1_score) > 0 and not np.all(np.isnan(f1_score)):
                has_f1 = True
                # Filter out NaN values
                valid_mask = ~np.isnan(f1_score)
                valid_steps = np.array(steps)[valid_mask]
                valid_f1 = np.array(f1_score)[valid_mask]
                
                if len(valid_steps) > 0:
                    ax.plot(valid_steps, valid_f1, label=method_name,
                           color=colors[i], marker='^', markersize=4, alpha=0.7)
    
    if has_f1:
        ax.set_xlabel('Intervention Step')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score over Interventions (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    else:
        ax.text(0.5, 0.5, 'F1 Score not available\n(Graph estimates not tracked)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('F1 Score over Interventions')
    
    plt.tight_layout()
    return fig


def plot_performance_comparison(
    results_dict: Dict[str, Dict],
    metrics: List[str] = ['final_target_value', 'improvement', 'final_shd', 'final_f1'],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Create bar plots comparing final metrics across methods.
    
    Args:
        results_dict: Dictionary mapping method names to results
        metrics: List of metrics to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract performance metrics
    data = []
    for method_name, result in results_dict.items():
        if 'performance_metrics' in result:
            perf = result['performance_metrics']
            row = {'Method': method_name}
            for metric in metrics:
                if metric in perf:
                    row[metric] = perf[metric]
                elif metric == 'final_target_value' and 'final_best' in result:
                    row[metric] = result['final_best']
                elif metric == 'improvement' and 'improvement' in result:
                    row[metric] = result['improvement']
                else:
                    row[metric] = np.nan
            data.append(row)
    
    if not data:
        # Fallback to basic metrics
        for method_name, result in results_dict.items():
            row = {'Method': method_name}
            row['final_target_value'] = result.get('final_best', result.get('final_target_value', 0))
            row['improvement'] = result.get('improvement', result.get('target_improvement', 0))
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Create subplots
    n_metrics = len([m for m in metrics if m in df.columns])
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    # Color methods
    method_colors = []
    for method in df['Method']:
        if 'bc' in method.lower():
            method_colors.append('green')
        elif 'baseline' in method.lower():
            method_colors.append('red')
        else:
            method_colors.append('blue')
    
    # Plot each metric
    plot_idx = 0
    for metric in metrics:
        if metric not in df.columns:
            continue
            
        ax = axes[plot_idx]
        values = df[metric].values
        methods = df['Method'].values
        
        # Skip if all NaN
        if np.all(np.isnan(values)):
            continue
        
        bars = ax.bar(methods, values, color=method_colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Formatting
        metric_titles = {
            'final_target_value': 'Final Target Value',
            'improvement': 'Improvement',
            'final_shd': 'Final SHD',
            'final_f1': 'Final F1 Score'
        }
        ax.set_title(metric_titles.get(metric, metric))
        ax.set_ylabel('Value')
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.suptitle('Final Performance Metrics Comparison', fontsize=14)
    plt.tight_layout()
    return fig


def create_performance_summary_table(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a summary table of performance metrics.
    
    Args:
        results_dict: Dictionary mapping method names to results
        
    Returns:
        DataFrame with performance summary
    """
    data = []
    for method_name, result in results_dict.items():
        row = {'Method': method_name}
        
        # Extract basic metrics
        row['Initial Value'] = result.get('initial_best', np.nan)
        row['Final Value'] = result.get('final_best', result.get('final_target_value', np.nan))
        row['Improvement'] = result.get('improvement', result.get('target_improvement', np.nan))
        
        # Extract performance metrics if available
        if 'performance_metrics' in result:
            perf = result['performance_metrics']
            row['Final SHD'] = perf.get('final_shd', np.nan)
            row['Final F1'] = perf.get('final_f1', np.nan)
            row['Total Steps'] = perf.get('total_steps', np.nan)
        
        # Add method type
        if 'bc' in method_name.lower():
            row['Type'] = 'BC Method'
        elif 'baseline' in method_name.lower():
            row['Type'] = 'Baseline'
        else:
            row['Type'] = 'Other'
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Format numeric columns
    numeric_cols = ['Initial Value', 'Final Value', 'Improvement', 'Final SHD', 'Final F1']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A')
    
    return df