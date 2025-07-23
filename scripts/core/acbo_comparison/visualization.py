"""
Visualization Manager for ACBO Comparison Framework

This module provides comprehensive visualization capabilities for ACBO experiments,
including method comparisons, learning curves, and statistical analysis plots.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class VisualizationManager:
    """Manager for creating comprehensive ACBO experiment visualizations."""
    
    def __init__(self, output_dir: Optional[Path] = None, figsize: tuple = (15, 10)):
        """
        Initialize visualization manager.
        
        Args:
            output_dir: Directory for saving plots
            figsize: Default figure size (increased for better readability)
        """
        self.output_dir = output_dir or Path("experiment_plots")
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize
        
        # Configure matplotlib for better readability
        plt.rcParams.update({
            'font.size': 14,           # Increased from 12
            'axes.titlesize': 16,      # Increased from 14
            'axes.labelsize': 14,      # Increased from 12
            'xtick.labelsize': 12,     # Increased from 10
            'ytick.labelsize': 12,     # Increased from 10
            'legend.fontsize': 12,     # Increased from 11
            'figure.titlesize': 16,
            'lines.linewidth': 2.0,    # Thicker lines for better visibility
            'lines.markersize': 6      # Larger markers
        })
    
    def create_all_plots(self, method_results: Dict[str, List[Dict[str, Any]]], 
                        analysis_results: Dict[str, Any], 
                        experiment_name: str) -> Dict[str, str]:
        """
        Create all visualization plots for the experiment.
        
        Args:
            method_results: Results for all methods
            analysis_results: Statistical analysis results
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plots = {}
        
        try:
            # Method comparison plot
            plots['method_comparison'] = self.create_method_comparison_plot(
                method_results, f"{experiment_name}_method_comparison"
            )
            
            # Learning curves
            plots['learning_curves'] = self.create_learning_curves_plot(
                method_results, f"{experiment_name}_learning_curves"
            )
            
            # Statistical comparison
            plots['statistical_comparison'] = self.create_statistical_comparison_plot(
                analysis_results, f"{experiment_name}_statistical_comparison"
            )
            
            # Performance distribution
            plots['performance_distribution'] = self.create_performance_distribution_plot(
                method_results, f"{experiment_name}_performance_distribution"
            )
            
            # Intervention efficiency
            plots['intervention_efficiency'] = self.create_intervention_efficiency_plot(
                method_results, f"{experiment_name}_intervention_efficiency"
            )
            
            # Standalone trajectory plots
            plots['f1_trajectory'] = self.create_f1_trajectory_plot(
                method_results, f"{experiment_name}_f1_trajectory"
            )
            
            plots['shd_trajectory'] = self.create_shd_trajectory_plot(
                method_results, f"{experiment_name}_shd_trajectory"
            )
            
            plots['target_trajectory'] = self.create_target_trajectory_plot(
                method_results, f"{experiment_name}_target_trajectory"
            )
            
            logger.info(f"Created {len(plots)} visualization plots")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
        
        return plots
    
    def create_method_comparison_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                    filename: str) -> str:
        """Create comprehensive method comparison plot using baseline comparison style."""
        
        # Import the new baseline comparison function
        from causal_bayes_opt.visualization.plots import plot_baseline_comparison
        from causal_bayes_opt.analysis.trajectory_metrics import extract_learning_curves
        from causal_bayes_opt.data_structures.scm import get_target, get_parents
        from scripts.core.acbo_comparison.structure_metrics_helper import (
            compute_f1_from_marginals, compute_shd_from_marginals
        )
        
        # Convert method results to learning curve format
        trajectory_data = {}
        
        for method_name, results in method_results.items():
            # Extract trajectories from all successful runs
            method_trajectories = []
            
            for result in results:
                if result.get('success', True):
                    # Try to use pre-computed trajectory data from detailed_results first
                    detailed_results = result.get('detailed_results', {})
                    
                    if all(key in detailed_results for key in ['target_progress', 'f1_scores', 'shd_values']):
                        # Use pre-computed trajectory data
                        trajectory = {
                            'steps': detailed_results.get('steps', list(range(len(detailed_results['target_progress'])))),
                            'target_values': detailed_results['target_progress'],
                            'f1_scores': detailed_results['f1_scores'],
                            'shd_values': detailed_results['shd_values']
                        }
                    elif 'learning_history' in result:
                        # Compute from learning history
                        history = result['learning_history']
                        
                        # Get SCM info for structure metrics (if available)
                        scm = result.get('scm')  # May not be available
                        true_parents = []
                        target = None
                        
                        if scm:
                            try:
                                target = get_target(scm)
                                true_parents = list(get_parents(scm, target)) if target else []
                            except:
                                pass
                        
                        # Extract metrics at each step
                        trajectory = {
                            'steps': list(range(len(history))),
                            'target_values': [],
                            'f1_scores': [],
                            'shd_values': []
                        }
                        
                        for step in history:
                            # Target value
                            trajectory['target_values'].append(
                                step.get('outcome_value', step.get('target_value', 0.0))
                            )
                            
                            # F1 score - compute from marginals if available
                            marginals = step.get('marginals', step.get('marginal_probs', {}))
                            if marginals and true_parents and target:
                                f1_score, _, _ = compute_f1_from_marginals(marginals, true_parents, target)
                                trajectory['f1_scores'].append(f1_score)
                                
                                # SHD - compute from marginals
                                shd = compute_shd_from_marginals(marginals, true_parents, target)
                                trajectory['shd_values'].append(shd)
                            else:
                                # No marginals or SCM info - use defaults
                                trajectory['f1_scores'].append(0.0)
                                trajectory['shd_values'].append(len(true_parents) if true_parents else 0)
                    
                    method_trajectories.append(trajectory)
            
            # Store trajectories for this method
            if method_trajectories:
                trajectory_data[method_name] = method_trajectories
        
        # Use extract_learning_curves to get mean/std
        learning_curves = extract_learning_curves(trajectory_data)
        
        # Create the baseline comparison plot
        plot_path = self.output_dir / f"{filename}_baseline_style.png"
        plot_baseline_comparison(
            learning_curves,
            title="ACBO Methods Comparison",
            save_path=str(plot_path)
        )
        
        # Also create the original box plot for additional metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ACBO Methods Performance Distribution', fontsize=16, fontweight='bold')
        
        metrics = [
            ('target_reduction', 'Target Reduction (Higher is Better)'),
            ('structure_accuracy', 'Structure Accuracy (Higher is Better)'),
            ('sample_efficiency', 'Sample Efficiency (Higher is Better)'),
            ('convergence_steps', 'Convergence Steps')
        ]
        
        for idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Extract data for this metric
            method_names = []
            metric_values = []
            
            for method_name, results in method_results.items():
                values = [r.get(metric_key, 0.0) for r in results if r.get('success', True)]
                if values:
                    method_names.append(method_name)
                    metric_values.append(values)
            
            if metric_values:
                # Create box plot
                bp = ax.boxplot(metric_values, labels=method_names, patch_artist=True)
                
                # Color boxes
                colors = sns.color_palette("husl", len(metric_values))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Add mean markers
                means = [onp.mean(values) for values in metric_values]
                ax.scatter(range(1, len(means) + 1), means, 
                          marker='D', s=50, color='red', zorder=3, label='Mean')
            
            ax.set_title(metric_label)
            ax.set_ylabel(metric_label)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                # Only add legend if there are labeled lines
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend()
        
        plt.tight_layout()
        
        # Save box plot
        box_plot_path = self.output_dir / f"{filename}_box.png"
        plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_learning_curves_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                  filename: str) -> str:
        """Create learning curves plot showing convergence over intervention steps."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Learning Curves', fontsize=16, fontweight='bold')
        
        colors = sns.color_palette("husl", len(method_results))
        
        for (method_name, results), color in zip(method_results.items(), colors):
            # Extract trajectory data
            all_trajectories = []
            
            for result in results:
                if result.get('success', True) and 'target_values_trajectory' in result:
                    trajectory = result['target_values_trajectory']
                    if trajectory:
                        all_trajectories.append(trajectory)
            
            if all_trajectories:
                # Compute mean and std across trajectories
                max_length = max(len(traj) for traj in all_trajectories)
                
                # Pad trajectories to same length
                padded_trajectories = []
                for traj in all_trajectories:
                    if len(traj) < max_length:
                        # Pad with last value
                        padded = list(traj) + [traj[-1]] * (max_length - len(traj))
                    else:
                        padded = traj[:max_length]
                    padded_trajectories.append(padded)
                
                mean_traj = onp.mean(padded_trajectories, axis=0)
                std_traj = onp.std(padded_trajectories, axis=0)
                steps = onp.arange(len(mean_traj))
                
                # Plot mean with confidence interval
                ax1.plot(steps, mean_traj, label=method_name, color=color, linewidth=2)
                ax1.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj, 
                               alpha=0.2, color=color)
        
        ax1.set_xlabel('Intervention Step')
        ax1.set_ylabel('Target Value (Lower is Better)')
        ax1.set_title('Target Value Progression')
        # Only add legend if there are labeled lines
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Second subplot: Final improvements
        method_names = []
        final_improvements = []
        
        for method_name, results in method_results.items():
            improvements = [r.get('target_reduction', r.get('target_improvement', 0.0)) for r in results if r.get('success', True)]
            if improvements:
                method_names.append(method_name)
                final_improvements.append(improvements)
        
        if final_improvements:
            positions = onp.arange(len(method_names))
            means = [onp.mean(vals) for vals in final_improvements]
            stds = [onp.std(vals) for vals in final_improvements]
            
            bars = ax2.bar(positions, means, yerr=stds, capsize=5, 
                          color=colors[:len(means)], alpha=0.7)
            
            ax2.set_xlabel('Method')
            ax2.set_ylabel('Target Reduction (Positive = Better)')
            ax2.set_title('Final Performance Comparison')
            ax2.set_xticks(positions)
            ax2.set_xticklabels(method_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, means):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(stds) * 0.05,
                        f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_statistical_comparison_plot(self, analysis_results: Dict[str, Any], 
                                         filename: str) -> str:
        """Create statistical comparison visualization."""
        
        pairwise_comparisons = analysis_results.get('pairwise_comparisons', {})
        
        if not pairwise_comparisons:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No statistical comparisons available', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Statistical Comparison')
            
            plot_path = self.output_dir / f"{filename}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        # Create subplot layout
        n_comparisons = len(pairwise_comparisons)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Analysis Results', fontsize=16, fontweight='bold')
        
        # Extract comparison data
        comparison_names = []
        p_values = []
        effect_sizes = []
        mean_differences = []
        
        for comp_name, comp_result in pairwise_comparisons.items():
            comparison_names.append(comp_name.replace('_vs_', ' vs '))
            p_values.append(comp_result.p_value)
            effect_sizes.append(abs(comp_result.effect_size))
            mean_differences.append(comp_result.mean_difference)
        
        # P-values plot
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(p_values)), p_values, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
        ax1.set_xlabel('Comparison')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance')
        ax1.set_xticks(range(len(comparison_names)))
        ax1.set_xticklabels(comparison_names, rotation=45, ha='right')
        # Only add legend if there are labeled lines
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Color significant comparisons
        for i, (bar, p_val) in enumerate(zip(bars1, p_values)):
            if p_val < 0.05:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        # Effect sizes plot
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7, color='orange')
        ax2.set_xlabel('Comparison')
        ax2.set_ylabel('Effect Size (|Cohen\'s d|)')
        ax2.set_title('Effect Sizes')
        ax2.set_xticks(range(len(comparison_names)))
        ax2.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add effect size interpretation lines
        ax2.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Small (0.2)')
        ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium (0.5)')
        ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Large (0.8)')
        # Only add legend if there are labeled lines
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend()
        
        # Mean differences plot
        ax3 = axes[1, 0]
        colors = ['green' if diff > 0 else 'red' for diff in mean_differences]
        bars3 = ax3.bar(range(len(mean_differences)), mean_differences, 
                       color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Comparison')
        ax3.set_ylabel('Mean Difference')
        ax3.set_title('Mean Differences')
        ax3.set_xticks(range(len(comparison_names)))
        ax3.set_xticklabels(comparison_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "Statistical Summary:\n\n"
        significant_count = sum(1 for p in p_values if p < 0.05)
        summary_text += f"Significant comparisons: {significant_count}/{len(p_values)}\n"
        summary_text += f"Mean p-value: {onp.mean(p_values):.4f}\n"
        summary_text += f"Mean effect size: {onp.mean(effect_sizes):.4f}\n\n"
        
        # Add ANOVA results if available
        anova_results = analysis_results.get('anova_results', {})
        if 'error' not in anova_results and anova_results:
            summary_text += f"ANOVA F-statistic: {anova_results.get('f_statistic', 0):.4f}\n"
            summary_text += f"ANOVA p-value: {anova_results.get('p_value', 1):.4f}\n"
            summary_text += f"Overall significant: {anova_results.get('significant', False)}\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_performance_distribution_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                           filename: str) -> str:
        """Create performance distribution plots."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Distributions', fontsize=16, fontweight='bold')
        
        # Target improvement distributions
        for method_name, results in method_results.items():
            improvements = [r.get('target_reduction', r.get('target_improvement', 0.0)) for r in results if r.get('success', True)]
            if improvements:
                ax1.hist(improvements, alpha=0.6, label=method_name, bins=10, density=True)
        
        ax1.set_xlabel('Target Reduction (Positive = Better Performance)')
        ax1.set_ylabel('Density')
        ax1.set_title('Target Reduction Distribution')
        # Only add legend if there are labeled lines
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Structure accuracy distributions
        for method_name, results in method_results.items():
            accuracies = [r.get('structure_accuracy', 0.0) for r in results if r.get('success', True)]
            if accuracies:
                ax2.hist(accuracies, alpha=0.6, label=method_name, bins=10, density=True)
        
        ax2.set_xlabel('Structure Accuracy')
        ax2.set_ylabel('Density')
        ax2.set_title('Structure Accuracy Distribution')
        # Only add legend if there are labeled lines
        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_intervention_efficiency_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                          filename: str) -> str:
        """Create intervention efficiency comparison plot."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        method_names = []
        efficiency_data = []
        
        for method_name, results in method_results.items():
            efficiencies = []
            for result in results:
                if result.get('success', True):
                    # Calculate efficiency as improvement per intervention
                    improvement = result.get('target_reduction', result.get('target_improvement', 0.0))
                    interventions = result.get('intervention_count', 1)
                    efficiency = improvement / max(1, interventions)
                    efficiencies.append(efficiency)
            
            if efficiencies:
                method_names.append(method_name)
                efficiency_data.append(efficiencies)
        
        if efficiency_data:
            # Create violin plot
            parts = ax.violinplot(efficiency_data, positions=range(len(method_names)), 
                                showmeans=True, showmedians=True)
            
            # Color the violin plots
            colors = sns.color_palette("husl", len(efficiency_data))
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Intervention Efficiency\n(Reduction per Intervention)')
            ax.set_title('Intervention Efficiency Comparison')
            ax.set_xticks(range(len(method_names)))
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for i, data in enumerate(efficiency_data):
                mean_val = onp.mean(data)
                ax.text(i, max(data) + 0.01, f'{mean_val:.4f}', 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_f1_trajectory_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                 filename: str) -> str:
        """Create standalone F1 score trajectory plot."""
        
        fig, ax = plt.subplots(figsize=(15, 10))  # Larger figure for better readability
        
        # Extract trajectory data (reuse logic from method comparison)
        colors = sns.color_palette("husl", len(method_results))
        
        for (method_name, results), color in zip(method_results.items(), colors):
            all_f1_trajectories = []
            
            for result in results:
                if result.get('success', True):
                    detailed_results = result.get('detailed_results', {})
                    
                    if 'f1_scores' in detailed_results:
                        f1_scores = detailed_results['f1_scores']
                        steps = detailed_results.get('steps', list(range(len(f1_scores))))
                        all_f1_trajectories.append((steps, f1_scores))
            
            if all_f1_trajectories:
                # Plot individual trajectories with transparency
                for steps, f1_scores in all_f1_trajectories:
                    ax.plot(steps, f1_scores, color=color, alpha=0.3, linewidth=1)
                
                # Plot mean trajectory
                max_steps = max(len(scores) for _, scores in all_f1_trajectories)
                mean_f1 = []
                for step in range(max_steps):
                    step_f1_values = [scores[step] for _, scores in all_f1_trajectories 
                                     if step < len(scores)]
                    if step_f1_values:
                        mean_f1.append(onp.mean(step_f1_values))
                    else:
                        mean_f1.append(0.0)
                
                ax.plot(range(len(mean_f1)), mean_f1, color=color, linewidth=3, 
                       label=f'{method_name} (mean)', marker='o', markersize=4)
        
        ax.set_xlabel('Intervention Step', fontsize=14)
        ax.set_ylabel('F1 Score (Structure Learning)', fontsize=14)
        ax.set_title('F1 Score Trajectory Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_shd_trajectory_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                  filename: str) -> str:
        """Create standalone SHD trajectory plot."""
        
        fig, ax = plt.subplots(figsize=(15, 10))  # Larger figure for better readability
        
        # Extract trajectory data
        colors = sns.color_palette("husl", len(method_results))
        
        for (method_name, results), color in zip(method_results.items(), colors):
            all_shd_trajectories = []
            
            for result in results:
                if result.get('success', True):
                    detailed_results = result.get('detailed_results', {})
                    
                    if 'shd_values' in detailed_results:
                        shd_values = detailed_results['shd_values']
                        steps = detailed_results.get('steps', list(range(len(shd_values))))
                        all_shd_trajectories.append((steps, shd_values))
            
            if all_shd_trajectories:
                # Plot individual trajectories with transparency
                for steps, shd_values in all_shd_trajectories:
                    ax.plot(steps, shd_values, color=color, alpha=0.3, linewidth=1)
                
                # Plot mean trajectory
                max_steps = max(len(values) for _, values in all_shd_trajectories)
                mean_shd = []
                for step in range(max_steps):
                    step_shd_values = [values[step] for _, values in all_shd_trajectories 
                                      if step < len(values)]
                    if step_shd_values:
                        mean_shd.append(onp.mean(step_shd_values))
                    else:
                        mean_shd.append(0.0)
                
                ax.plot(range(len(mean_shd)), mean_shd, color=color, linewidth=3, 
                       label=f'{method_name} (mean)', marker='s', markersize=4)
        
        ax.set_xlabel('Intervention Step', fontsize=14)
        ax.set_ylabel('Structural Hamming Distance (Lower is Better)', fontsize=14)
        ax.set_title('SHD Trajectory Comparison', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def create_target_trajectory_plot(self, method_results: Dict[str, List[Dict[str, Any]]], 
                                     filename: str) -> str:
        """Create standalone target value trajectory plot."""
        
        fig, ax = plt.subplots(figsize=(15, 10))  # Larger figure for better readability
        
        # Extract trajectory data
        colors = sns.color_palette("husl", len(method_results))
        
        for (method_name, results), color in zip(method_results.items(), colors):
            all_target_trajectories = []
            
            for result in results:
                if result.get('success', True):
                    detailed_results = result.get('detailed_results', {})
                    
                    # Try multiple sources for target values
                    target_values = (detailed_results.get('target_progress') or 
                                   detailed_results.get('target_values') or 
                                   [step.get('outcome_value', 0.0) for step in result.get('learning_history', [])])
                    
                    if target_values:
                        steps = detailed_results.get('steps', list(range(len(target_values))))
                        all_target_trajectories.append((steps, target_values))
            
            if all_target_trajectories:
                # Plot individual trajectories with transparency
                for steps, target_values in all_target_trajectories:
                    ax.plot(steps, target_values, color=color, alpha=0.3, linewidth=1)
                
                # Plot mean trajectory
                max_steps = max(len(values) for _, values in all_target_trajectories)
                mean_target = []
                for step in range(max_steps):
                    step_target_values = [values[step] for _, values in all_target_trajectories 
                                         if step < len(values)]
                    if step_target_values:
                        mean_target.append(onp.mean(step_target_values))
                    else:
                        mean_target.append(0.0)
                
                ax.plot(range(len(mean_target)), mean_target, color=color, linewidth=3, 
                       label=f'{method_name} (mean)', marker='^', markersize=4)
        
        ax.set_xlabel('Intervention Step', fontsize=14)
        ax.set_ylabel('Target Variable Value (Lower is Better)', fontsize=14)
        ax.set_title('Target Value Trajectory Comparison - Minimization', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"{filename}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)