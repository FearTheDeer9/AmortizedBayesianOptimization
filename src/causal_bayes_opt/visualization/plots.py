"""
Plotting Functions for ACBO Experiment Analysis

Clean matplotlib-based plotting functions for visualizing trajectory metrics
and comparing different methods.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as onp

logger = logging.getLogger(__name__)

# Set style for clean plots
plt.style.use('default')  # Clean default style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def plot_convergence(
    trajectory_metrics: Dict[str, List[float]],
    title: str = "Convergence to True Parent Set",
    save_path: Optional[str] = None,
    show_f1: bool = True,
    show_uncertainty: bool = True
) -> plt.Figure:
    """
    Plot convergence metrics including true parent likelihood.
    
    Args:
        trajectory_metrics: Output from compute_trajectory_metrics
        title: Plot title
        save_path: Optional path to save the plot
        show_f1: Whether to show F1 score subplot
        show_uncertainty: Whether to show uncertainty subplot
        
    Returns:
        Matplotlib figure object
    """
    if not trajectory_metrics or not trajectory_metrics.get('steps'):
        logger.warning("Empty trajectory metrics provided")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Determine subplot layout
    n_subplots = 1 + int(show_f1) + int(show_uncertainty)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), sharex=True)
    
    if n_subplots == 1:
        axes = [axes]
    
    steps = trajectory_metrics['steps']
    
    # Plot 1: True Parent Likelihood
    ax1 = axes[0]
    likelihood = trajectory_metrics.get('true_parent_likelihood', [])
    if likelihood:
        ax1.plot(steps, likelihood, 'b-', linewidth=2, label='P(True Parents | Data)')
        ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Threshold')
        ax1.set_ylabel('Likelihood')
        ax1.set_title('Convergence to True Parent Set')
        ax1.legend()
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
    
    subplot_idx = 1
    
    # Plot 2: F1 Score (optional)
    if show_f1 and subplot_idx < len(axes):
        ax2 = axes[subplot_idx]
        f1_scores = trajectory_metrics.get('f1_scores', [])
        if f1_scores:
            ax2.plot(steps, f1_scores, 'g-', linewidth=2, label='F1 Score')
            ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='70% Threshold')
            ax2.set_ylabel('F1 Score')
            ax2.set_title('Structure Recovery F1 Score')
            ax2.legend()
            ax2.set_ylim(0, 1.05)
            ax2.grid(True, alpha=0.3)
        subplot_idx += 1
    
    # Plot 3: Uncertainty (optional)
    if show_uncertainty and subplot_idx < len(axes):
        ax3 = axes[subplot_idx]
        uncertainty = trajectory_metrics.get('uncertainty_bits', [])
        if uncertainty:
            ax3.plot(steps, uncertainty, 'orange', linewidth=2, label='Uncertainty (bits)')
            ax3.set_ylabel('Uncertainty (bits)')
            ax3.set_title('Model Uncertainty')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        subplot_idx += 1
    
    # Set common x-axis label
    axes[-1].set_xlabel('Intervention Steps')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot to {save_path}")
    
    return fig


def plot_target_optimization(
    trajectory_metrics: Dict[str, List[float]],
    title: str = "Target Variable Optimization",
    save_path: Optional[str] = None,
    show_confidence: bool = False
) -> plt.Figure:
    """
    Plot target value progression over interventions.
    
    Args:
        trajectory_metrics: Output from compute_trajectory_metrics
        title: Plot title
        save_path: Optional path to save the plot
        show_confidence: Whether to show confidence intervals (if available)
        
    Returns:
        Matplotlib figure object
    """
    if not trajectory_metrics or not trajectory_metrics.get('steps'):
        logger.warning("Empty trajectory metrics provided")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    steps = trajectory_metrics['steps']
    target_values = trajectory_metrics.get('target_values', [])
    rewards = trajectory_metrics.get('rewards', [])
    
    # Plot 1: Target Value Progression
    if target_values:
        ax1.plot(steps, target_values, 'b-', linewidth=2, label='Best Target Value')
        
        # Show improvement
        if len(target_values) > 1:
            improvement = target_values[-1] - target_values[0]
            ax1.text(0.02, 0.95, f'Total Improvement: {improvement:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax1.set_ylabel('Target Value')
        ax1.set_title('Target Variable Optimization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reward Progression
    if rewards and len(rewards) > 0:
        # Align rewards with steps if needed
        reward_steps = steps[:len(rewards)] if len(rewards) <= len(steps) else steps
        plot_rewards = rewards[:len(reward_steps)]
        
        ax2.plot(reward_steps, plot_rewards, 'r-', linewidth=2, label='Reward Signal')
        
        # Show moving average if enough data
        if len(plot_rewards) >= 5:
            window_size = min(5, len(plot_rewards) // 2)
            moving_avg = onp.convolve(plot_rewards, onp.ones(window_size)/window_size, mode='valid')
            moving_steps = reward_steps[window_size-1:]
            ax2.plot(moving_steps, moving_avg, 'darkred', linewidth=2, alpha=0.7, 
                    label=f'Moving Average ({window_size} steps)')
        
        ax2.set_ylabel('Reward')
        ax2.set_title('Reward Signal')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Intervention Steps')
    
    plt.suptitle(title, fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved target optimization plot to {save_path}")
    
    return fig


def plot_method_comparison(
    results_by_method: Dict[str, Dict[str, List[float]]],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    metrics: List[str] = None
) -> plt.Figure:
    """
    Compare multiple methods on the same plot with confidence intervals.
    
    Args:
        results_by_method: Dict mapping method names to learning curves from extract_learning_curves
        title: Plot title
        save_path: Optional path to save the plot
        metrics: List of metrics to plot ['likelihood', 'f1', 'target']
        
    Returns:
        Matplotlib figure object
    """
    if not results_by_method:
        logger.warning("No results provided for comparison")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    if metrics is None:
        metrics = ['likelihood', 'f1']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    # Color palette for methods
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(results_by_method.keys())}
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]
        
        for method_name, learning_curve in results_by_method.items():
            if not learning_curve or f'{metric}_mean' not in learning_curve:
                continue
            
            steps = learning_curve.get('steps', [])
            mean_values = learning_curve.get(f'{metric}_mean', [])
            std_values = learning_curve.get(f'{metric}_std', [])
            n_runs = learning_curve.get('n_runs', 1)
            
            if not steps or not mean_values:
                continue
            
            color = method_colors[method_name]
            
            # Plot mean line
            ax.plot(steps, mean_values, color=color, linewidth=2, label=f'{method_name} (n={n_runs})')
            
            # Plot confidence interval (if we have std and multiple runs)
            if std_values and n_runs > 1:
                mean_array = onp.array(mean_values)
                std_array = onp.array(std_values)
                # Standard error for confidence interval
                se = std_array / onp.sqrt(n_runs)
                ax.fill_between(steps, mean_array - se, mean_array + se, 
                               color=color, alpha=0.2)
        
        # Customize subplot
        if metric == 'likelihood':
            ax.set_ylabel('P(True Parents | Data)')
            ax.set_title('True Parent Likelihood Over Time')
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% Target')
        elif metric == 'f1':
            ax.set_ylabel('F1 Score')
            ax.set_title('Structure Recovery F1 Score')
            ax.set_ylim(0, 1.05)
            ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='70% Target')
        elif metric == 'target':
            ax.set_ylabel('Target Value')
            ax.set_title('Target Variable Optimization')
        
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Intervention Steps')
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved method comparison plot to {save_path}")
    
    return fig


def plot_intervention_efficiency(
    results_by_method: Dict[str, List[Dict[str, Any]]],
    title: str = "Intervention Efficiency",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot intervention efficiency: steps/interventions needed to reach thresholds.
    
    Args:
        results_by_method: Dict mapping method names to lists of efficiency results
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    if not results_by_method:
        logger.warning("No results provided for efficiency analysis")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = list(results_by_method.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot 1: Steps to threshold
    steps_data = []
    labels = []
    for method in methods:
        efficiency_results = results_by_method[method]
        steps_to_threshold = [r.get('steps_to_threshold') for r in efficiency_results if r.get('steps_to_threshold') is not None]
        if steps_to_threshold:
            steps_data.append(steps_to_threshold)
            labels.append(f'{method} (n={len(steps_to_threshold)})')
    
    if steps_data:
        bp1 = ax1.boxplot(steps_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors[:len(steps_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Steps to 80% Likelihood')
        ax1.set_title('Steps to Convergence')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency ratio (improvement per intervention)
    efficiency_data = []
    labels2 = []
    for method in methods:
        efficiency_results = results_by_method[method]
        efficiency_ratios = [r.get('efficiency_ratio', 0) for r in efficiency_results if r.get('efficiency_ratio', 0) > 0]
        if efficiency_ratios:
            efficiency_data.append(efficiency_ratios)
            labels2.append(f'{method} (n={len(efficiency_ratios)})')
    
    if efficiency_data:
        bp2 = ax2.boxplot(efficiency_data, labels=labels2, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors[:len(efficiency_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Improvement per Intervention')
        ax2.set_title('Learning Efficiency')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved efficiency plot to {save_path}")
    
    return fig


def create_experiment_dashboard(
    experiment_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard with multiple subplots.
    
    Args:
        experiment_results: Dictionary with experiment results and analysis
        save_path: Optional path to save the dashboard
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 2x3 grid of subplots
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Extract data
    trajectory_metrics = experiment_results.get('trajectory_metrics', {})
    convergence_analysis = experiment_results.get('convergence_analysis', {})
    method_comparison = experiment_results.get('method_comparison', {})
    
    # Plot 1: Likelihood convergence
    ax1 = fig.add_subplot(gs[0, 0])
    if trajectory_metrics.get('true_parent_likelihood'):
        steps = trajectory_metrics['steps']
        likelihood = trajectory_metrics['true_parent_likelihood']
        ax1.plot(steps, likelihood, 'b-', linewidth=2)
        ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
        ax1.set_ylabel('P(True Parents)')
        ax1.set_title('Likelihood Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
    
    # Plot 2: F1 score
    ax2 = fig.add_subplot(gs[0, 1])
    if trajectory_metrics.get('f1_scores'):
        ax2.plot(steps, trajectory_metrics['f1_scores'], 'g-', linewidth=2)
        ax2.axhline(y=0.7, color='r', linestyle='--', alpha=0.7)
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Structure Recovery')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
    
    # Plot 3: Target optimization
    ax3 = fig.add_subplot(gs[1, 0])
    if trajectory_metrics.get('target_values'):
        ax3.plot(steps, trajectory_metrics['target_values'], 'purple', linewidth=2)
        ax3.set_ylabel('Target Value')
        ax3.set_title('Target Optimization')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty reduction
    ax4 = fig.add_subplot(gs[1, 1])
    if trajectory_metrics.get('uncertainty_bits'):
        ax4.plot(steps, trajectory_metrics['uncertainty_bits'], 'orange', linewidth=2)
        ax4.set_ylabel('Uncertainty (bits)')
        ax4.set_title('Model Uncertainty')
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Convergence summary (text)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    summary_text = "Experiment Summary\\n"
    summary_text += "=" * 50 + "\\n"
    
    if convergence_analysis:
        converged = convergence_analysis.get('converged', False)
        final_likelihood = convergence_analysis.get('final_likelihood', 0)
        convergence_step = convergence_analysis.get('convergence_step')
        
        summary_text += f"Converged: {'✓' if converged else '✗'}\\n"
        summary_text += f"Final Likelihood: {final_likelihood:.3f}\\n"
        if convergence_step:
            summary_text += f"Convergence Step: {convergence_step}\\n"
    
    if trajectory_metrics:
        total_steps = len(trajectory_metrics.get('steps', []))
        summary_text += f"Total Steps: {total_steps}\\n"
        
        if trajectory_metrics.get('target_values'):
            improvement = trajectory_metrics['target_values'][-1] - trajectory_metrics['target_values'][0]
            summary_text += f"Target Improvement: {improvement:.3f}\\n"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('ACBO Experiment Dashboard', fontsize=18, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved dashboard to {save_path}")
    
    return fig


def plot_calibration_curves(
    marginals_over_time: List[Dict[str, float]],
    true_parents: List[str],
    title: str = "Probability Calibration Analysis",
    save_path: Optional[str] = None,
    n_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curves showing probability reliability over time.
    
    Args:
        marginals_over_time: List of marginal probability dicts at each step
        true_parents: List of true parent variables
        title: Plot title
        save_path: Optional path to save the plot
        n_bins: Number of bins for calibration analysis
        
    Returns:
        Matplotlib figure object
    """
    if not marginals_over_time:
        logger.warning("No marginal probabilities provided")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Import metrics functions
    from ..analysis.trajectory_metrics import (
        compute_expected_calibration_error, compute_brier_score
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Compute metrics over time
    ece_over_time = []
    brier_over_time = []
    steps = []
    
    for i, marginals in enumerate(marginals_over_time):
        if marginals:
            ece = compute_expected_calibration_error(marginals, true_parents, n_bins)
            brier = compute_brier_score(marginals, true_parents)
            ece_over_time.append(ece)
            brier_over_time.append(brier)
            steps.append(i + 1)
    
    # Plot 1: ECE over time
    ax1 = axes[0, 0]
    if ece_over_time:
        ax1.plot(steps, ece_over_time, 'b-', linewidth=2, label='ECE')
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.7, label='10% Target')
        ax1.set_xlabel('Intervention Steps')
        ax1.set_ylabel('Expected Calibration Error')
        ax1.set_title('Calibration Error Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Brier score over time
    ax2 = axes[0, 1]
    if brier_over_time:
        ax2.plot(steps, brier_over_time, 'g-', linewidth=2, label='Brier Score')
        ax2.set_xlabel('Intervention Steps')
        ax2.set_ylabel('Brier Score')
        ax2.set_title('Brier Score Over Time (Lower is Better)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final calibration plot (reliability diagram)
    ax3 = axes[1, 0]
    if marginals_over_time:
        final_marginals = marginals_over_time[-1]
        true_parents_set = set(true_parents)
        
        # Create bins for calibration plot
        bin_boundaries = onp.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        observed_freq = []
        expected_freq = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            in_bin_probs = []
            in_bin_actual = []
            
            for var, prob in final_marginals.items():
                if bin_lower <= prob < bin_upper or (bin_upper == 1.0 and prob == 1.0):
                    in_bin_probs.append(prob)
                    in_bin_actual.append(1.0 if var in true_parents_set else 0.0)
            
            if in_bin_probs:
                observed_freq.append(onp.mean(in_bin_actual))
                expected_freq.append(onp.mean(in_bin_probs))
                bin_counts.append(len(in_bin_probs))
            else:
                observed_freq.append(None)
                expected_freq.append(None)
                bin_counts.append(0)
        
        # Plot calibration curve
        valid_bins = [(o, e, c) for o, e, c in zip(observed_freq, expected_freq, bin_counts) if o is not None]
        if valid_bins:
            obs_freq = [b[0] for b in valid_bins]
            exp_freq = [b[1] for b in valid_bins]
            counts = [b[2] for b in valid_bins]
            
            # Main calibration line
            ax3.plot(exp_freq, obs_freq, 'bo-', linewidth=2, markersize=8, label='Calibration')
            
            # Perfect calibration line
            ax3.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
            
            # Add sample counts as text
            for x, y, n in zip(exp_freq, obs_freq, counts):
                ax3.text(x, y + 0.02, f'n={n}', ha='center', va='bottom', fontsize=8)
            
            ax3.set_xlabel('Mean Predicted Probability')
            ax3.set_ylabel('Observed Frequency')
            ax3.set_title('Calibration Plot (Final Step)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-0.05, 1.05)
            ax3.set_ylim(-0.05, 1.05)
    
    # Plot 4: Histogram of predicted probabilities
    ax4 = axes[1, 1]
    if marginals_over_time:
        final_marginals = marginals_over_time[-1]
        all_probs = list(final_marginals.values())
        
        # Separate by true/false parents
        true_parent_probs = [final_marginals[var] for var in final_marginals if var in true_parents_set]
        false_parent_probs = [final_marginals[var] for var in final_marginals if var not in true_parents_set]
        
        bins = onp.linspace(0, 1, 20)
        ax4.hist(false_parent_probs, bins=bins, alpha=0.5, label='Non-parents', color='red', edgecolor='black')
        ax4.hist(true_parent_probs, bins=bins, alpha=0.5, label='True Parents', color='green', edgecolor='black')
        
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of Predicted Probabilities')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved calibration plot to {save_path}")
    
    return fig


def plot_precision_recall_curves(
    marginals_over_time: List[Dict[str, float]],
    true_parents: List[str],
    title: str = "Precision-Recall Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision-recall curves at different time points.
    
    Args:
        marginals_over_time: List of marginal probability dicts at each step
        true_parents: List of true parent variables
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    if not marginals_over_time:
        logger.warning("No marginal probabilities provided")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        ax.set_title(title)
        return fig
    
    # Import metrics functions
    from ..analysis.trajectory_metrics import compute_precision_recall_curve, find_optimal_f1_threshold
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Select time points to show (early, middle, late)
    n_steps = len(marginals_over_time)
    time_points = []
    labels = []
    colors = []
    
    if n_steps > 0:
        time_points.append(0)
        labels.append(f'Step 1')
        colors.append('red')
    
    if n_steps > 5:
        mid_point = n_steps // 2
        time_points.append(mid_point)
        labels.append(f'Step {mid_point + 1}')
        colors.append('orange')
    
    if n_steps > 1:
        time_points.append(n_steps - 1)
        labels.append(f'Step {n_steps}')
        colors.append('green')
    
    # Plot 1: Precision-Recall curves
    for idx, (time_idx, label, color) in enumerate(zip(time_points, labels, colors)):
        marginals = marginals_over_time[time_idx]
        pr_data = compute_precision_recall_curve(marginals, true_parents)
        
        precision = pr_data['precision']
        recall = pr_data['recall']
        
        ax1.plot(recall, precision, color=color, linewidth=2, label=label)
        
        # Mark optimal F1 point
        optimal_threshold, max_f1 = find_optimal_f1_threshold(marginals, true_parents)
        # Find the index closest to optimal threshold
        thresholds = pr_data['thresholds']
        optimal_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - optimal_threshold))
        
        ax1.plot(recall[optimal_idx], precision[optimal_idx], 'o', color=color, markersize=8,
                label=f'Optimal F1={max_f1:.2f}')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curves Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: F1 scores vs threshold for final step
    if marginals_over_time:
        final_marginals = marginals_over_time[-1]
        pr_data = compute_precision_recall_curve(final_marginals, true_parents)
        
        thresholds = pr_data['thresholds']
        f1_scores = pr_data['f1_scores']
        
        ax2.plot(thresholds, f1_scores, 'b-', linewidth=2)
        
        # Mark optimal threshold
        optimal_threshold, max_f1 = find_optimal_f1_threshold(final_marginals, true_parents)
        ax2.axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7,
                   label=f'Optimal Threshold={optimal_threshold:.2f}')
        ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5,
                   label='Default Threshold=0.5')
        
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('F1 Score')
        ax2.set_title(f'F1 Score vs Threshold (Final Step)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.05)
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall plot to {save_path}")
    
    return fig


def save_all_plots(
    experiment_results: Dict[str, Any],
    output_dir: str = "experiment_plots",
    prefix: str = "experiment"
) -> List[str]:
    """
    Save all plots for an experiment to a directory.
    
    Args:
        experiment_results: Dictionary with experiment results
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    trajectory_metrics = experiment_results.get('trajectory_metrics', {})
    
    if trajectory_metrics:
        # Save convergence plot
        convergence_path = output_path / f"{prefix}_convergence.png"
        plot_convergence(trajectory_metrics, save_path=str(convergence_path))
        saved_files.append(str(convergence_path))
        
        # Save target optimization plot
        target_path = output_path / f"{prefix}_target_optimization.png"
        plot_target_optimization(trajectory_metrics, save_path=str(target_path))
        saved_files.append(str(target_path))
    
    # Extract marginals over time if available
    marginals_over_time = []
    true_parents = experiment_results.get('true_parents', [])
    
    # Try to extract from detailed results
    if 'detailed_results' in experiment_results:
        detailed = experiment_results['detailed_results']
        learning_history = detailed.get('learning_history', [])
        
        for step_data in learning_history:
            marginals = step_data.get('marginals', {})
            if marginals:
                marginals_over_time.append(marginals)
    
    # Save calibration plots if we have marginals
    if marginals_over_time and true_parents:
        calibration_path = output_path / f"{prefix}_calibration.png"
        plot_calibration_curves(marginals_over_time, true_parents, save_path=str(calibration_path))
        saved_files.append(str(calibration_path))
        
        # Save precision-recall plots
        pr_path = output_path / f"{prefix}_precision_recall.png"
        plot_precision_recall_curves(marginals_over_time, true_parents, save_path=str(pr_path))
        saved_files.append(str(pr_path))
    
    # Save dashboard
    dashboard_path = output_path / f"{prefix}_dashboard.png"
    create_experiment_dashboard(experiment_results, save_path=str(dashboard_path))
    saved_files.append(str(dashboard_path))
    
    logger.info(f"Saved {len(saved_files)} plots to {output_dir}")
    return saved_files


# Utility function to close all figures
def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')


# Export public functions
__all__ = [
    'plot_convergence',
    'plot_target_optimization',
    'plot_method_comparison',
    'plot_intervention_efficiency',
    'plot_calibration_curves',
    'plot_precision_recall_curves',
    'create_experiment_dashboard',
    'save_all_plots',
    'close_all_figures'
]