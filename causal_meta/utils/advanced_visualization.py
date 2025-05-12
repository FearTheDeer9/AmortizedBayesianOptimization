"""
Advanced visualization utilities for causal structure learning and analysis.

This module provides visualization functions for analyzing edge probabilities,
comparing intervention strategies, and investigating model biases in causal
structure learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

def ensure_numpy(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert a tensor to numpy array if needed.
    
    Args:
        arr: Input array or tensor
        
    Returns:
        Numpy array
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

def plot_edge_probabilities(
    true_adj: Union[np.ndarray, torch.Tensor],
    edge_probs: Union[np.ndarray, torch.Tensor],
    thresholded_adj: Optional[Union[np.ndarray, torch.Tensor]] = None,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of true adjacency matrix, edge probabilities, and thresholded adjacency matrix.
    
    Args:
        true_adj: True adjacency matrix
        edge_probs: Predicted edge probabilities
        thresholded_adj: Thresholded adjacency matrix (optional)
        threshold: Threshold to apply if thresholded_adj is None
        figsize: Figure size
        cmap: Colormap for heatmaps
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    true_adj = ensure_numpy(true_adj)
    edge_probs = ensure_numpy(edge_probs)
    
    # Apply threshold if thresholded_adj not provided
    if thresholded_adj is None:
        thresholded_adj = (edge_probs > threshold).astype(float)
    else:
        thresholded_adj = ensure_numpy(thresholded_adj)
    
    # Calculate confusion matrix metrics
    tp = np.logical_and(thresholded_adj == 1, true_adj == 1).sum()
    fp = np.logical_and(thresholded_adj == 1, true_adj == 0).sum()
    tn = np.logical_and(thresholded_adj == 0, true_adj == 0).sum()
    fn = np.logical_and(thresholded_adj == 0, true_adj == 1).sum()
    
    # Calculate derived metrics
    total_edges = true_adj.sum()
    total_possible_edges = true_adj.size - np.trace(true_adj).sum()  # Exclude diagonal
    accuracy = (tp + tn) / total_possible_edges if total_possible_edges > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot true adjacency matrix
    im0 = axes[0].imshow(true_adj, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title('True Graph')
    axes[0].set_xlabel('Target Node')
    axes[0].set_ylabel('Source Node')
    
    # Add grid
    n_nodes = true_adj.shape[0]
    for i in range(n_nodes + 1):
        axes[0].axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        axes[0].axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
    
    # Plot edge probabilities
    im1 = axes[1].imshow(edge_probs, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title('Edge Probabilities')
    axes[1].set_xlabel('Target Node')
    axes[1].set_ylabel('Source Node')
    plt.colorbar(im1, ax=axes[1], label='Probability')
    
    # Add grid
    for i in range(n_nodes + 1):
        axes[1].axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        axes[1].axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
    
    # Plot thresholded adjacency matrix
    im2 = axes[2].imshow(thresholded_adj, cmap=cmap, vmin=0, vmax=1)
    axes[2].set_title(f'Thresholded Graph (t={threshold})')
    axes[2].set_xlabel('Target Node')
    axes[2].set_ylabel('Source Node')
    
    # Add grid
    for i in range(n_nodes + 1):
        axes[2].axhline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
        axes[2].axvline(i - 0.5, color='black', linestyle='-', linewidth=0.5)
    
    # Add metrics as text
    metrics_text = (
        f"True edges: {int(total_edges)}\n"
        f"Predicted edges: {int((thresholded_adj > 0).sum())}\n"
        f"TP: {int(tp)}, FP: {int(fp)}\n"
        f"TN: {int(tn)}, FN: {int(fn)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1: {f1:.4f}"
    )
    
    fig.text(0.5, 0.01, metrics_text, ha='center', va='center', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def plot_edge_probability_histogram(
    edge_probs: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    n_bins: int = 20,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of edge probabilities to analyze distribution bias.
    
    Args:
        edge_probs: Predicted edge probabilities
        threshold: Decision threshold
        n_bins: Number of histogram bins
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    edge_probs = ensure_numpy(edge_probs)
    
    # Flatten and remove diagonal (self-loops)
    n_nodes = edge_probs.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    flat_probs = edge_probs[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(flat_probs, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', 
              label=f'Threshold ({threshold})')
    
    # Calculate metrics about the distribution
    below_threshold = (flat_probs < threshold).mean() * 100
    above_threshold = (flat_probs >= threshold).mean() * 100
    near_threshold = ((flat_probs > threshold - 0.1) & 
                     (flat_probs < threshold + 0.1)).mean() * 100
    
    # Add metrics as text
    stats_text = (
        f"Mean probability: {flat_probs.mean():.4f}\n"
        f"Median probability: {np.median(flat_probs):.4f}\n"
        f"Below threshold: {below_threshold:.1f}%\n"
        f"Above threshold: {above_threshold:.1f}%\n"
        f"Near threshold (±0.1): {near_threshold:.1f}%"
    )
    
    # Add a text box with the statistics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Edge Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def plot_edge_probability_distribution(
    true_adj: Union[np.ndarray, torch.Tensor],
    edge_probs: Union[np.ndarray, torch.Tensor],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distributions of edge probabilities for true edges vs non-edges.
    
    Args:
        true_adj: True adjacency matrix
        edge_probs: Predicted edge probabilities
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    true_adj = ensure_numpy(true_adj)
    edge_probs = ensure_numpy(edge_probs)
    
    # Remove diagonal (self-loops)
    n_nodes = edge_probs.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    
    # Get probabilities for true edges and non-edges
    true_edge_probs = edge_probs[np.logical_and(true_adj > 0, mask)]
    non_edge_probs = edge_probs[np.logical_and(true_adj == 0, mask)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot distributions
    sns.kdeplot(true_edge_probs, ax=ax, color='green', 
               label=f'True Edges (n={len(true_edge_probs)})', fill=True, alpha=0.3)
    sns.kdeplot(non_edge_probs, ax=ax, color='red', 
               label=f'Non-Edges (n={len(non_edge_probs)})', fill=True, alpha=0.3)
    
    # Calculate mean probabilities
    true_edge_mean = true_edge_probs.mean() if len(true_edge_probs) > 0 else 0
    non_edge_mean = non_edge_probs.mean() if len(non_edge_probs) > 0 else 0
    
    # Add vertical lines for means
    ax.axvline(x=true_edge_mean, color='green', linestyle='--', 
              label=f'True Edges Mean ({true_edge_mean:.3f})')
    ax.axvline(x=non_edge_mean, color='red', linestyle='--',
              label=f'Non-Edges Mean ({non_edge_mean:.3f})')
    
    # Add threshold line at 0.5
    ax.axvline(x=0.5, color='black', linestyle='-', label='Threshold (0.5)')
    
    # Calculate overlap percentage
    if len(true_edge_probs) > 0 and len(non_edge_probs) > 0:
        min_true = min(true_edge_probs)
        max_true = max(true_edge_probs)
        min_non = min(non_edge_probs)
        max_non = max(non_edge_probs)
        
        overlap_start = max(min_true, min_non)
        overlap_end = min(max_true, max_non)
        
        if overlap_end > overlap_start:
            # There is overlap
            overlap_text = f"Distributions overlap from {overlap_start:.3f} to {overlap_end:.3f}"
        else:
            overlap_text = "Distributions do not overlap"
    else:
        overlap_text = "Insufficient data to calculate overlap"
    
    # Add a text box with the overlap information
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, overlap_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    # Set labels and title
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Edge Probabilities for True Edges vs Non-Edges')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def plot_threshold_sensitivity(
    true_adj: Union[np.ndarray, torch.Tensor],
    edge_probs: Union[np.ndarray, torch.Tensor],
    thresholds: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze sensitivity of graph recovery metrics to threshold choice.
    
    Args:
        true_adj: True adjacency matrix
        edge_probs: Predicted edge probabilities
        thresholds: Array of thresholds to evaluate (default: np.linspace(0.1, 0.9, 9))
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy if needed
    true_adj = ensure_numpy(true_adj)
    edge_probs = ensure_numpy(edge_probs)
    
    # Default thresholds if not provided
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    
    # Remove diagonal (self-loops)
    n_nodes = edge_probs.shape[0]
    mask = ~np.eye(n_nodes, dtype=bool)
    
    # Flatten matrices with mask applied
    y_true = true_adj[mask].flatten()
    y_score = edge_probs[mask].flatten()
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # Calculate metrics for each threshold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    true_edge_rates = []
    
    for t in thresholds:
        pred_adj = (edge_probs > t).astype(float)
        
        tp = np.logical_and(pred_adj == 1, true_adj == 1).sum()
        fp = np.logical_and(pred_adj == 1, true_adj == 0).sum()
        tn = np.logical_and(pred_adj == 0, true_adj == 0).sum()
        fn = np.logical_and(pred_adj == 0, true_adj == 1).sum()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        # Calculate true edge rate (percentage of edges predicted)
        true_edge_rate = (pred_adj > 0).sum() / (n_nodes ** 2 - n_nodes)
        
        accuracies.append(accuracy)
        precisions.append(precision_val)
        recalls.append(recall_val)
        f1_scores.append(f1)
        true_edge_rates.append(true_edge_rate)
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot precision-recall curve
    axes[0, 0].plot(recall, precision, marker='.', label=f'AP={ap:.3f}')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot metric curves vs threshold
    axes[0, 1].plot(thresholds, accuracies, 'o-', label='Accuracy')
    axes[0, 1].plot(thresholds, precisions, 's-', label='Precision')
    axes[0, 1].plot(thresholds, recalls, '^-', label='Recall')
    axes[0, 1].plot(thresholds, f1_scores, 'D-', label='F1')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_title('Metrics vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot edge prediction rate vs threshold
    axes[1, 0].plot(thresholds, true_edge_rates, 'o-', color='purple')
    axes[1, 0].axhline(y=true_adj.sum() / (n_nodes**2 - n_nodes), color='red', 
                      linestyle='--', label='True Edge Rate')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Predicted Edge Rate')
    axes[1, 0].set_title('Edge Prediction Rate vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Find optimal threshold based on F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot F1 vs threshold highlighting optimal point
    axes[1, 1].plot(thresholds, f1_scores, 'D-', color='green')
    axes[1, 1].scatter([optimal_threshold], [f1_scores[optimal_idx]], 
                      color='red', s=100, zorder=5,
                      label=f'Optimal: t={optimal_threshold:.2f}, F1={f1_scores[optimal_idx]:.4f}')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig

def compare_intervention_strategies(
    iterations: List[int],
    random_metrics: Dict[str, List[float]],
    strategic_metrics: Dict[str, List[float]],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'shd'],
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare metrics between random and strategic intervention strategies.
    
    Args:
        iterations: List of iteration numbers
        random_metrics: Dictionary of metrics for random intervention strategy
        strategic_metrics: Dictionary of metrics for strategic intervention strategy
        metrics_to_plot: List of metrics to include in the plots
        figsize: Figure size
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure
    """
    # Create figure with subplots
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(
        (n_metrics + 1) // 2, 2, 
        figsize=figsize, 
        sharex=True
    )
    axes = axes.flatten()  # Flatten for easier indexing
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes) and metric in random_metrics and metric in strategic_metrics:
            ax = axes[i]
            
            # Get metric values
            random_values = random_metrics[metric]
            strategic_values = strategic_metrics[metric]
            
            # Plot
            ax.plot(iterations, random_values, 'o-', color='blue', label='Random')
            ax.plot(iterations, strategic_values, 's-', color='red', label='Strategic')
            
            # Calculate differences and statistical significance
            if len(random_values) == len(strategic_values):
                differences = np.array(random_values) - np.array(strategic_values)
                mean_diff = differences.mean()
                
                # Add difference annotation
                if metric.lower() in ['accuracy', 'precision', 'recall', 'f1']:
                    diff_text = f"Mean Diff: {mean_diff:.4f}"
                else:
                    diff_text = f"Mean Diff: {mean_diff:.2f}"
                
                ax.text(0.05, 0.05, diff_text, transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Set title and labels
            ax.set_title(f'{metric.upper()}')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric.title())
            ax.grid(True, alpha=0.3)
            
            # Only add legend to first plot
            if i == 0:
                ax.legend()
    
    # Hide any unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    plt.suptitle('Comparison of Random vs. Strategic Intervention Strategies', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        plt.close()
    
    return fig 