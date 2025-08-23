"""
Visualization utilities for evaluation results.

This module provides standard plotting functions for visualizing
evaluation results across different methods and experiments.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


class PlottingUtils:
    """Standard plotting functions for evaluation."""
    
    @staticmethod
    def setup_style():
        """Set up plotting style for consistent visualizations."""
        # Use a style that works with the installed matplotlib version
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('ggplot')
        
        # Set color palette
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        
    @staticmethod
    def plot_convergence(results: Dict[str, List[List[float]]], 
                        save_path: Optional[Path] = None,
                        title: str = "Convergence Comparison",
                        ylabel: str = "Target Value",
                        show_confidence: bool = True) -> plt.Figure:
        """
        Plot convergence curves for multiple methods.
        
        Args:
            results: Dictionary mapping method names to lists of trajectories
            save_path: Optional path to save the figure
            title: Plot title
            ylabel: Y-axis label
            show_confidence: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure
        """
        PlottingUtils.setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = sns.color_palette("husl", n_colors=len(results))
        
        for (method_name, trajectories), color in zip(results.items(), colors):
            if not trajectories:
                continue
                
            # Pad trajectories to same length
            max_len = max(len(t) for t in trajectories)
            padded = []
            for t in trajectories:
                if len(t) < max_len:
                    # Pad with last value
                    padded.append(t + [t[-1]] * (max_len - len(t)))
                else:
                    padded.append(t)
            
            # Compute mean and std
            padded_array = np.array(padded)
            mean_trajectory = np.mean(padded_array, axis=0)
            std_trajectory = np.std(padded_array, axis=0)
            
            x = np.arange(len(mean_trajectory))
            
            # Plot mean
            ax.plot(x, mean_trajectory, label=method_name, linewidth=2, color=color)
            
            # Plot confidence interval
            if show_confidence and len(trajectories) > 1:
                ax.fill_between(x, 
                               mean_trajectory - std_trajectory,
                               mean_trajectory + std_trajectory,
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Intervention Number')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved convergence plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_method_comparison(metrics: Dict[str, Dict[str, float]],
                              metric_names: Optional[List[str]] = None,
                              save_path: Optional[Path] = None,
                              title: str = "Method Comparison") -> plt.Figure:
        """
        Create bar plot comparing methods across metrics.
        
        Args:
            metrics: Dictionary mapping method names to metric dictionaries
            metric_names: Optional list of specific metrics to plot
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        PlottingUtils.setup_style()
        
        if not metrics:
            logger.warning("No metrics to plot")
            return None
        
        methods = list(metrics.keys())
        
        # Get metric names if not specified
        if metric_names is None:
            # Get all mean metrics (exclude std, min, max)
            all_metrics = set()
            for method_metrics in metrics.values():
                all_metrics.update(k for k in method_metrics.keys() if k.startswith('mean_'))
            metric_names = sorted(list(all_metrics))
        
        if not metric_names:
            logger.warning("No metrics found to plot")
            return None
        
        n_metrics = len(metric_names)
        n_methods = len(methods)
        
        # Create subplots
        if n_metrics <= 3:
            fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        else:
            n_rows = (n_metrics + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_metrics > 3 else axes
        
        colors = sns.color_palette("husl", n_colors=n_methods)
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get values and error bars
            values = []
            errors = []
            for method in methods:
                mean_val = metrics[method].get(metric_name, 0)
                values.append(mean_val)
                
                # Get std if available
                std_key = metric_name.replace('mean_', 'std_')
                std_val = metrics[method].get(std_key, 0)
                errors.append(std_val)
            
            # Create bar plot
            x_pos = np.arange(len(methods))
            bars = ax.bar(x_pos, values, yerr=errors if any(errors) else None,
                          capsize=5, color=colors)
            
            # Formatting
            display_name = metric_name.replace('mean_', '').replace('_', ' ').title()
            ax.set_title(display_name)
            ax.set_xlabel('Method')
            ax.set_ylabel('Value')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if 'accuracy' in metric_name:
                    label = f'{val:.1%}'
                else:
                    label = f'{val:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=10)
        
        # Remove unused subplots
        for i in range(n_metrics, len(axes)):
            fig.delaxes(axes[i])
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved comparison plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_scaling_analysis(scaling_results: Dict[int, Dict[str, float]],
                            metric: str = 'mean_best_value',
                            save_path: Optional[Path] = None,
                            title: str = "Scaling Analysis") -> plt.Figure:
        """
        Plot performance vs SCM size.
        
        Args:
            scaling_results: Dictionary mapping SCM sizes to method performance
            metric: Which metric to plot
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        PlottingUtils.setup_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if not scaling_results:
            logger.warning("No scaling results to plot")
            return fig
        
        sizes = sorted(scaling_results.keys())
        
        # Get all methods
        all_methods = set()
        for size_results in scaling_results.values():
            all_methods.update(size_results.keys())
        methods = sorted(list(all_methods))
        
        colors = sns.color_palette("husl", n_colors=len(methods))
        
        for method, color in zip(methods, colors):
            values = []
            for size in sizes:
                if method in scaling_results[size]:
                    values.append(scaling_results[size][method].get(metric, np.nan))
                else:
                    values.append(np.nan)
            
            # Plot only non-nan values
            valid_sizes = [s for s, v in zip(sizes, values) if not np.isnan(v)]
            valid_values = [v for v in values if not np.isnan(v)]
            
            if valid_sizes:
                ax.plot(valid_sizes, valid_values, marker='o', label=method,
                       linewidth=2, markersize=8, color=color)
        
        ax.set_xlabel('Number of Variables')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis to show all sizes
        ax.set_xticks(sizes)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved scaling plot to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_ablation_study(ablation_results: Dict[str, Dict[str, float]],
                           baseline_name: str = 'full_model',
                           save_path: Optional[Path] = None,
                           title: str = "Ablation Study") -> plt.Figure:
        """
        Plot ablation study results.
        
        Args:
            ablation_results: Dictionary mapping configuration names to metrics
            baseline_name: Name of the baseline configuration
            save_path: Optional path to save the figure
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        PlottingUtils.setup_style()
        
        # Convert to relative performance
        if baseline_name not in ablation_results:
            logger.warning(f"Baseline {baseline_name} not found in results")
            baseline_metrics = next(iter(ablation_results.values()))
        else:
            baseline_metrics = ablation_results[baseline_name]
        
        # Get all metrics
        metric_names = list(baseline_metrics.keys())
        configurations = list(ablation_results.keys())
        
        # Create DataFrame for heatmap
        data = []
        for config in configurations:
            row = []
            for metric in metric_names:
                if metric in ablation_results[config]:
                    # Compute relative performance
                    baseline_val = baseline_metrics.get(metric, 1.0)
                    config_val = ablation_results[config][metric]
                    if baseline_val != 0:
                        relative = (config_val / baseline_val - 1) * 100  # Percentage change
                    else:
                        relative = 0
                    row.append(relative)
                else:
                    row.append(np.nan)
            data.append(row)
        
        df = pd.DataFrame(data, index=configurations, columns=metric_names)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Relative Performance (%)'}, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Configurations')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ablation plot to {save_path}")
        
        return fig
    
    @staticmethod
    def create_summary_figure(all_results: Dict[str, Any],
                             save_path: Optional[Path] = None,
                             title: str = "Evaluation Summary") -> plt.Figure:
        """
        Create a comprehensive summary figure with multiple subplots.
        
        Args:
            all_results: Dictionary containing all evaluation results
            save_path: Optional path to save the figure
            title: Overall figure title
            
        Returns:
            Matplotlib figure
        """
        PlottingUtils.setup_style()
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Convergence plot (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'all_trajectories' in all_results:
            trajectories = all_results['all_trajectories']
            colors = sns.color_palette("husl", n_colors=len(trajectories))
            
            for (method_name, method_trajectories), color in zip(trajectories.items(), colors):
                if method_trajectories:
                    # Compute mean trajectory
                    max_len = max(len(t) for t in method_trajectories)
                    padded = [t + [t[-1]]*(max_len-len(t)) for t in method_trajectories]
                    mean_traj = np.mean(padded, axis=0)
                    
                    ax1.plot(mean_traj, label=method_name, linewidth=2, color=color)
            
            ax1.set_xlabel('Intervention Number')
            ax1.set_ylabel('Target Value')
            ax1.set_title('Convergence Comparison')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
        
        # 2. Final performance comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'comparison_metrics' in all_results:
            metrics = all_results['comparison_metrics']
            methods = list(metrics.keys())
            final_values = [metrics[m].get('mean_final_value', 0) for m in methods]
            
            colors = sns.color_palette("husl", n_colors=len(methods))
            bars = ax2.bar(range(len(methods)), final_values, color=colors)
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            ax2.set_ylabel('Final Target Value')
            ax2.set_title('Final Performance')
            
            # Add value labels
            for bar, val in zip(bars, final_values):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom')
        
        # 3. Parent accuracy (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'comparison_metrics' in all_results:
            metrics = all_results['comparison_metrics']
            methods = list(metrics.keys())
            parent_acc = [metrics[m].get('mean_parent_accuracy', 0) for m in methods]
            
            colors = sns.color_palette("husl", n_colors=len(methods))
            bars = ax3.bar(range(len(methods)), parent_acc, color=colors)
            ax3.set_xticks(range(len(methods)))
            ax3.set_xticklabels(methods, rotation=45, ha='right')
            ax3.set_ylabel('Parent Accuracy')
            ax3.set_title('Parent Selection Accuracy')
            ax3.set_ylim([0, 1])
            
            # Add percentage labels
            for bar, val in zip(bars, parent_acc):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1%}', ha='center', va='bottom')
        
        # 4. Sample efficiency (bottom middle)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'comparison_metrics' in all_results:
            metrics = all_results['comparison_metrics']
            methods = list(metrics.keys())
            efficiency = [metrics[m].get('mean_interventions_to_threshold', 
                                        float('inf')) for m in methods]
            
            # Replace inf with max value for plotting
            max_val = max(e for e in efficiency if e != float('inf'))
            efficiency = [e if e != float('inf') else max_val * 1.2 for e in efficiency]
            
            colors = sns.color_palette("husl", n_colors=len(methods))
            bars = ax4.bar(range(len(methods)), efficiency, color=colors)
            ax4.set_xticks(range(len(methods)))
            ax4.set_xticklabels(methods, rotation=45, ha='right')
            ax4.set_ylabel('Interventions to Threshold')
            ax4.set_title('Sample Efficiency')
            
            # Add value labels
            for bar, val in zip(bars, efficiency):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{int(val)}', ha='center', va='bottom')
        
        # 5. Summary statistics table (bottom right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('tight')
        ax5.axis('off')
        
        if 'comparison_metrics' in all_results:
            # Create summary table
            metrics = all_results['comparison_metrics']
            table_data = []
            for method in metrics.keys():
                row = [
                    method,
                    f"{metrics[method].get('mean_best_value', 0):.3f}",
                    f"{metrics[method].get('mean_convergence_rate', 0):.3f}",
                    f"{metrics[method].get('mean_parent_accuracy', 0):.1%}"
                ]
                table_data.append(row)
            
            table = ax5.table(cellText=table_data,
                            colLabels=['Method', 'Best Value', 'Conv. Rate', 'Parent Acc.'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax5.set_title('Summary Statistics', pad=20)
        
        # Overall title
        fig.suptitle(title, fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved summary figure to {save_path}")
        
        return fig
    
    @staticmethod
    def create_latex_table(metrics: Dict[str, Dict[str, float]],
                          metric_names: Optional[List[str]] = None,
                          caption: str = "Method Comparison",
                          label: str = "tab:comparison") -> str:
        """
        Create LaTeX table of results.
        
        Args:
            metrics: Dictionary mapping method names to metric dictionaries
            metric_names: Optional list of specific metrics to include
            caption: Table caption
            label: Table label for referencing
            
        Returns:
            LaTeX table string
        """
        if not metrics:
            return ""
        
        # Get metric names if not specified
        if metric_names is None:
            all_metrics = set()
            for method_metrics in metrics.values():
                all_metrics.update(k for k in method_metrics.keys() if k.startswith('mean_'))
            metric_names = sorted(list(all_metrics))
        
        # Create DataFrame
        data = []
        for method, method_metrics in metrics.items():
            row = [method]
            for metric in metric_names:
                value = method_metrics.get(metric, np.nan)
                if 'accuracy' in metric:
                    row.append(f"{value:.1%}")
                else:
                    row.append(f"{value:.3f}")
            data.append(row)
        
        # Create column names
        col_names = ['Method'] + [m.replace('mean_', '').replace('_', ' ').title() 
                                  for m in metric_names]
        
        df = pd.DataFrame(data, columns=col_names)
        
        # Convert to LaTeX
        latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c'*len(metric_names))
        
        # Add caption and label
        latex_str = latex_str.replace('\\begin{tabular}',
                                     f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}')
        
        return latex_str