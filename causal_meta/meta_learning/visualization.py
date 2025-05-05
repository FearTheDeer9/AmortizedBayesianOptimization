"""
Visualization utilities for neural causal discovery and optimization.

This module provides visualization functions specifically for neural causal
discovery, intervention prediction, and optimization progress. It complements
the base visualization utilities in causal_meta.graph.visualization with
specialized visualizations for neural models and benchmarks.
"""
from typing import Dict, List, Set, Any, Optional, Union, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import to_rgba
import networkx as nx

from causal_meta.graph import CausalGraph, DirectedGraph
from causal_meta.graph.visualization import (
    plot_graph, plot_graph_adjacency_matrix, plot_intervention
)
from causal_meta.meta_learning.graph_inference_utils import threshold_edge_probabilities


def plot_graph_inference_results(
    true_graph: CausalGraph,
    pred_graph_or_probs: Union[CausalGraph, np.ndarray, "torch.Tensor"],
    threshold: Optional[float] = None,
    metrics: bool = True,
    confusion_matrix: bool = True,
    ax: Optional[Union[plt.Axes, List[plt.Axes]]] = None,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 100,
    title: Optional[str] = None,
    **kwargs
) -> Union[plt.Axes, List[plt.Axes]]:
    """
    Visualize causal graph inference results by comparing ground truth and predicted graphs.
    
    Args:
        true_graph: Ground truth causal graph
        pred_graph_or_probs: Either a predicted CausalGraph or edge probability matrix
        threshold: Threshold to apply to edge probabilities (if pred_graph_or_probs is a matrix)
        metrics: Whether to include structure learning metrics like precision, recall, F1
        confusion_matrix: Whether to include a confusion matrix visualization
        ax: Optional matplotlib Axes object(s) to plot on
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        title: Optional title for the plot
        **kwargs: Additional keyword arguments passed to plot_graph
        
    Returns:
        matplotlib.pyplot.Axes or list of Axes: The axes object(s) containing the plot(s)
    """
    import torch
    
    # Convert tensor to numpy if needed
    if isinstance(pred_graph_or_probs, torch.Tensor):
        pred_graph_or_probs = pred_graph_or_probs.detach().cpu().numpy()
    
    # Convert edge probabilities to a graph if needed
    if isinstance(pred_graph_or_probs, np.ndarray):
        if threshold is None:
            threshold = 0.5
        pred_graph = threshold_edge_probabilities(
            pred_graph_or_probs, 
            true_graph.get_nodes(), 
            threshold
        )
    else:
        pred_graph = pred_graph_or_probs
    
    # Determine how many subplots we need
    n_plots = 2  # True and predicted graphs
    if confusion_matrix:
        n_plots += 1
    
    # Create subplots if ax is not provided
    if ax is None:
        fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)
        if n_plots == 1:
            axes = [axes]
    else:
        if not isinstance(ax, list) and not isinstance(ax, np.ndarray):
            ax = [ax]
        axes = ax
    
    # Plot true graph
    plot_graph(
        true_graph,
        ax=axes[0],
        title="Ground Truth Graph",
        **kwargs
    )
    
    # Plot predicted graph
    plot_graph(
        pred_graph,
        ax=axes[1],
        title="Predicted Graph",
        **kwargs
    )
    
    # Calculate and display metrics
    if metrics:
        # Get adjacency matrices
        true_adj = true_graph.get_adjacency_matrix()
        pred_adj = pred_graph.get_adjacency_matrix()
        
        # Calculate metrics
        true_pos = np.sum((true_adj == 1) & (pred_adj == 1))
        false_pos = np.sum((true_adj == 0) & (pred_adj == 1))
        false_neg = np.sum((true_adj == 1) & (pred_adj == 0))
        true_neg = np.sum((true_adj == 0) & (pred_adj == 0))
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        
        # Structural Hamming Distance (SHD)
        shd = false_pos + false_neg
        
        # Add metrics as text to the plot
        metric_text = (
            f"Precision: {precision:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"F1 Score: {f1:.3f}\n"
            f"Accuracy: {accuracy:.3f}\n"
            f"SHD: {shd}"
        )
        
        if confusion_matrix:
            # Plot metrics next to the confusion matrix
            axes[2].text(0.5, 0.5, metric_text, 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[2].transAxes,
                        bbox=dict(facecolor='white', alpha=0.8))
            axes[2].set_axis_off()
            
            # Add confusion matrix
            cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['No Edge', 'Edge'],
                yticklabels=['No Edge', 'Edge'],
                ax=axes[2]
            )
            axes[2].set_title('Confusion Matrix')
            axes[2].set_xlabel('Predicted')
            axes[2].set_ylabel('True')
        else:
            # Add text to the second plot if no confusion matrix
            axes[1].text(
                0.02, 0.02, metric_text,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=axes[1].transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
    
    # Set overall title if provided
    if title and n_plots > 1:
        plt.suptitle(title)
    elif title:
        axes[0].set_title(title)
    
    plt.tight_layout()
    
    return axes


def plot_intervention_outcomes(
    observational_data: Union[pd.DataFrame, np.ndarray, "torch.Tensor"],
    intervention_data: Union[pd.DataFrame, np.ndarray, "torch.Tensor"],
    predicted_outcomes: Union[pd.DataFrame, np.ndarray, "torch.Tensor"],
    node_names: Optional[List[str]] = None,
    intervention_nodes: Optional[List[Any]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 100,
    title: Optional[str] = None,
    show_errors: bool = True,
    show_distributions: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Visualize intervention outcomes by comparing actual and predicted post-intervention data.
    
    Args:
        observational_data: Pre-intervention observational data
        intervention_data: Post-intervention data (ground truth)
        predicted_outcomes: Model-predicted post-intervention data
        node_names: Names of the nodes/variables in the data
        intervention_nodes: Nodes on which interventions were performed
        ax: Optional matplotlib Axes object to plot on
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        title: Optional title for the plot
        show_errors: Whether to display prediction errors
        show_distributions: Whether to show full distributions rather than just means
        **kwargs: Additional keyword arguments
        
    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    import torch
    
    # Convert tensors to numpy if needed
    if isinstance(observational_data, torch.Tensor):
        observational_data = observational_data.detach().cpu().numpy()
    if isinstance(intervention_data, torch.Tensor):
        intervention_data = intervention_data.detach().cpu().numpy()
    if isinstance(predicted_outcomes, torch.Tensor):
        predicted_outcomes = predicted_outcomes.detach().cpu().numpy()
    
    # Convert to pandas DataFrames if they aren't already
    if not isinstance(observational_data, pd.DataFrame):
        if node_names is None:
            node_names = [f"X{i}" for i in range(observational_data.shape[1])]
        observational_data = pd.DataFrame(observational_data, columns=node_names)
    
    if not isinstance(intervention_data, pd.DataFrame):
        intervention_data = pd.DataFrame(intervention_data, columns=observational_data.columns)
    
    if not isinstance(predicted_outcomes, pd.DataFrame):
        predicted_outcomes = pd.DataFrame(predicted_outcomes, columns=observational_data.columns)
    
    # Default intervention nodes if not provided
    if intervention_nodes is None:
        intervention_nodes = []
    
    # Create subplots if ax is not provided
    if ax is None:
        num_nodes = len(observational_data.columns)
        if show_distributions:
            # For distributions, we need more rows
            n_rows = min(4, (num_nodes + 1) // 2)
            n_cols = min(3, (num_nodes + n_rows - 1) // n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
            axes = axes.flatten()
        else:
            # For bar plots, fewer subplots needed
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        if show_distributions:
            # If specific axes are provided for distributions
            axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
        else:
            axes = None
    
    # Create a list to store the mean values for bar plots if not showing distributions
    mean_values = {
        'Observational': observational_data.mean(),
        'Intervention (True)': intervention_data.mean(),
        'Intervention (Predicted)': predicted_outcomes.mean()
    }
    
    if show_distributions:
        # Plot distributions for each node
        for i, node in enumerate(observational_data.columns):
            if i < len(axes):
                # Only plot if we have enough axes
                is_intervention_node = node in intervention_nodes
                
                # Create a DataFrame for seaborn
                plot_data = pd.DataFrame({
                    'Value': pd.concat([
                        observational_data[node],
                        intervention_data[node],
                        predicted_outcomes[node]
                    ]),
                    'Type': ['Observational'] * len(observational_data) + 
                           ['Intervention (True)'] * len(intervention_data) + 
                           ['Intervention (Predicted)'] * len(predicted_outcomes)
                })
                
                # Plot the distributions
                sns.kdeplot(
                    data=plot_data, x='Value', hue='Type',
                    ax=axes[i], fill=True, alpha=0.3, common_norm=False
                )
                
                # Add markers for means
                for j, (label, color) in enumerate(zip(
                    ['Observational', 'Intervention (True)', 'Intervention (Predicted)'],
                    ['blue', 'orange', 'green']
                )):
                    axes[i].axvline(
                        mean_values[label][node],
                        color=color, linestyle='--', alpha=0.7,
                        label=f"{label} Mean" if i == 0 else None
                    )
                
                # Highlight intervention nodes
                if is_intervention_node:
                    axes[i].set_title(f"{node} (Intervention)", fontweight='bold')
                else:
                    axes[i].set_title(node)
                
                # Only show legend on the first plot
                if i == 0:
                    axes[i].legend()
                else:
                    axes[i].get_legend().remove() if axes[i].get_legend() else None
    else:
        # Bar plot for means
        df_means = pd.DataFrame(mean_values)
        df_means.plot(kind='bar', ax=ax, **kwargs)
        ax.set_ylabel('Mean Value')
        ax.set_title(title or "Intervention Outcomes Comparison")
        
        # Highlight intervention nodes
        if intervention_nodes:
            for i, node in enumerate(df_means.index):
                if node in intervention_nodes:
                    ax.get_xticklabels()[i].set_fontweight('bold')
    
    # Show prediction errors if requested
    if show_errors:
        mse = ((intervention_data - predicted_outcomes) ** 2).mean()
        mae = (intervention_data - predicted_outcomes).abs().mean()
        
        error_text = (
            f"Mean Squared Error: {mse.mean():.4f}\n"
            f"Mean Absolute Error: {mae.mean():.4f}"
        )
        
        if show_distributions:
            # Add text to an empty subplot or the last subplot
            text_ax_idx = min(len(observational_data.columns), len(axes) - 1)
            if text_ax_idx >= 0:
                axes[text_ax_idx].text(
                    0.5, 0.5, error_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[text_ax_idx].transAxes,
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                axes[text_ax_idx].set_axis_off()
        else:
            # Add text to the bar plot
            ax.text(
                0.02, 0.98, error_text,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
    
    # Set overall title if provided and using subplots
    if title and show_distributions:
        plt.suptitle(title)
    elif title and not show_distributions:
        ax.set_title(title)
    
    plt.tight_layout()
    
    return axes if show_distributions else ax


def plot_optimization_progress(
    iteration_data: List[Dict],
    target_variable: str,
    objective: str = "maximize",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    show_interventions: bool = True,
    show_baseline: bool = True,
    title: Optional[str] = None,
    comparison_methods: Optional[Dict[str, List[Dict]]] = None,
    **kwargs
) -> plt.Axes:
    """
    Visualize optimization progress over iterations.
    
    Args:
        iteration_data: List of dictionaries containing iteration data with keys:
                       'iteration', 'value', 'intervention', etc.
        target_variable: Variable being optimized
        objective: 'maximize' or 'minimize'
        ax: Optional matplotlib Axes object to plot on
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        show_interventions: Whether to mark intervention points
        show_baseline: Whether to show the baseline (initial) value
        title: Optional title for the plot
        comparison_methods: Optional dictionary mapping method names to their iteration data
        **kwargs: Additional keyword arguments passed to plotting functions
        
    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Extract values for plotting
    iterations = [data.get('iteration', i) for i, data in enumerate(iteration_data)]
    values = [data.get('value', 0) for data in iteration_data]
    
    # Determine best values at each iteration (cumulative best)
    if objective.lower() == 'maximize':
        cumulative_best = np.maximum.accumulate(values)
        opt_label = 'Maximum'
        best_idx = np.argmax(values)
    else:
        cumulative_best = np.minimum.accumulate(values)
        opt_label = 'Minimum'
        best_idx = np.argmin(values)
    
    # Plot the values
    ax.plot(iterations, values, 'o-', label=f'Observed values', **kwargs)
    ax.plot(iterations, cumulative_best, 'r--', label=f'Best value found', **kwargs)
    
    # Highlight best point
    ax.scatter([iterations[best_idx]], [values[best_idx]], s=100, c='red', 
              marker='*', zorder=10, label=f'Best value')
    
    # Plot comparison methods if provided
    if comparison_methods:
        colors = plt.cm.tab10.colors
        for i, (method_name, method_data) in enumerate(comparison_methods.items()):
            method_iterations = [data.get('iteration', i) for i, data in enumerate(method_data)]
            method_values = [data.get('value', 0) for data in method_data]
            
            if objective.lower() == 'maximize':
                method_best = np.maximum.accumulate(method_values)
            else:
                method_best = np.minimum.accumulate(method_values)
                
            color = colors[i % len(colors)]
            ax.plot(method_iterations, method_best, '--', color=color, label=f'{method_name} (best)', **kwargs)
    
    # Mark intervention points if requested
    if show_interventions:
        for i, data in enumerate(iteration_data):
            intervention = data.get('intervention', None)
            if intervention and isinstance(intervention, dict):
                # Format intervention as string
                intervention_str = ", ".join([f"{k}={v:.2f}" for k, v in intervention.items()])
                
                # Add annotation with arrow
                if i > 0 and i < len(iteration_data) - 1:  # Skip first and last for clarity
                    ax.annotate(
                        intervention_str,
                        xy=(iterations[i], values[i]),
                        xytext=(iterations[i], values[i] + (max(values) - min(values)) * 0.1),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
    
    # Show initial value as baseline if requested
    if show_baseline and len(values) > 0:
        ax.axhline(y=values[0], color='gray', linestyle=':', label='Initial value')
    
    # Set labels and title
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'{target_variable} Value')
    ax.set_title(title or f'Optimization Progress for {target_variable} ({objective})')
    
    # Add legend
    ax.legend()
    
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax


def plot_performance_comparison(
    benchmark_results: Dict[str, Dict],
    metrics: List[str],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
    plot_type: str = 'bar',
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Visualize performance comparison between different methods on benchmark results.
    
    Args:
        benchmark_results: Dictionary mapping method names to their performance metrics
        metrics: List of metric names to include in the comparison
        ax: Optional matplotlib Axes object to plot on
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        plot_type: Type of plot ('bar', 'radar', or 'box')
        title: Optional title for the plot
        **kwargs: Additional keyword arguments passed to plotting functions
        
    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    # Convert benchmark results to DataFrame
    data = {}
    for method, results in benchmark_results.items():
        data[method] = {metric: results.get(metric, np.nan) for metric in metrics}
    
    df = pd.DataFrame(data).T
    
    # Create axis if not provided
    if ax is None:
        if plot_type.lower() == 'radar':
            # For radar charts, we need a polar projection
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, polar=True)
        else:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if plot_type.lower() == 'bar':
        # Bar chart
        df.plot(kind='bar', ax=ax, **kwargs)
        ax.set_ylabel('Value')
        ax.set_title(title or 'Performance Comparison')
        ax.legend(title='Metric')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    elif plot_type.lower() == 'radar':
        # Radar chart
        # Number of metrics
        num_metrics = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Plot each method
        for method in df.index:
            values = df.loc[method].values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=method, **kwargs)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels for each axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Set y-limits based on data
        ax.set_ylim(bottom=min(0, df.values.min() * 1.1),
                   top=max(0, df.values.max() * 1.1))
        
        # Add title and legend
        ax.set_title(title or 'Performance Comparison (Radar)')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
    elif plot_type.lower() == 'box':
        # Box plot
        df_melted = df.reset_index().melt(id_vars='index', var_name='Metric', value_name='Value')
        sns.boxplot(data=df_melted, x='Metric', y='Value', hue='index', ax=ax, **kwargs)
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.set_title(title or 'Performance Distribution Comparison')
        ax.legend(title='Method')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    else:
        raise ValueError(f"Unsupported plot_type: {plot_type}. Use 'bar', 'radar', or 'box'.")
    
    plt.tight_layout()
    
    return ax


def plot_uncertainty(
    x_values: Union[List, np.ndarray, "torch.Tensor"],
    predictions: Union[np.ndarray, "torch.Tensor"],
    uncertainty: Union[np.ndarray, "torch.Tensor"],
    true_values: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    confidence_level: float = 0.95,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100,
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Visualize predictions with uncertainty.
    
    Args:
        x_values: X-axis values (e.g., intervention values or iterations)
        predictions: Model predictions
        uncertainty: Uncertainty estimates (standard deviations)
        true_values: Optional ground truth values for comparison
        confidence_level: Confidence level for intervals (default: 0.95)
        ax: Optional matplotlib Axes object to plot on
        figsize: Figure size as (width, height) in inches
        dpi: Dots per inch for the figure
        title: Optional title for the plot
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        **kwargs: Additional keyword arguments passed to plotting functions
        
    Returns:
        matplotlib.pyplot.Axes: The axes object containing the plot
    """
    import torch
    import scipy.stats as stats
    
    # Convert tensors to numpy if needed
    if isinstance(x_values, torch.Tensor):
        x_values = x_values.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.detach().cpu().numpy()
    if isinstance(true_values, torch.Tensor) and true_values is not None:
        true_values = true_values.detach().cpu().numpy()
    
    # Ensure arrays are 1D
    x_values = np.asarray(x_values).flatten()
    predictions = np.asarray(predictions).flatten()
    uncertainty = np.asarray(uncertainty).flatten()
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Calculate confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower_bound = predictions - z_score * uncertainty
    upper_bound = predictions + z_score * uncertainty
    
    # Plot predictions and confidence intervals
    ax.plot(x_values, predictions, 'b-', label='Predictions', **kwargs)
    ax.fill_between(x_values, lower_bound, upper_bound, color='b', alpha=0.2,
                   label=f'{confidence_level*100:.0f}% Confidence Interval')
    
    # Plot true values if provided
    if true_values is not None:
        true_values = np.asarray(true_values).flatten()
        ax.plot(x_values, true_values, 'r--', label='True Values', **kwargs)
    
    # Set labels and title
    ax.set_xlabel(x_label or 'X')
    ax.set_ylabel(y_label or 'Y')
    ax.set_title(title or 'Predictions with Uncertainty')
    
    # Add legend
    ax.legend()
    
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax 