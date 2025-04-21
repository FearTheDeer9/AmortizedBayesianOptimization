"""
Visualization utilities for GNN models and training results.

This module provides functions for visualizing graph structures,
training metrics, and GNN model outputs.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx

from causal_meta.inference.models.wrapper import EncoderDecoderWrapper


def visualize_graph(
    graph: Union[Data, nx.Graph],
    ax: Optional[plt.Axes] = None,
    title: str = "Graph Structure",
    node_size: int = 300,
    node_color: str = '#1f78b4',
    edge_color: str = '#999999',
    with_labels: bool = True,
    font_size: int = 10,
    layout: str = 'spring'
) -> plt.Axes:
    """
    Visualize a graph structure.

    Args:
        graph: PyTorch Geometric Data object or NetworkX graph
        ax: Matplotlib axes to plot on (creates a new one if None)
        title: Plot title
        node_size: Size of nodes in the plot
        node_color: Color of nodes
        edge_color: Color of edges
        with_labels: Whether to show node labels
        font_size: Font size for node labels
        layout: NetworkX layout algorithm ('spring', 'circular', 'kamada_kawai', 'planar')

    Returns:
        Matplotlib axes with the plotted graph
    """
    # Convert PyTorch Geometric Data to NetworkX graph if needed
    if isinstance(graph, Data):
        G = to_networkx(graph, to_undirected=False)
    else:
        G = graph

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Select layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'planar':
        try:
            pos = nx.planar_layout(G)
        except:
            # Fallback to spring layout if graph is not planar
            pos = nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw graph
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=with_labels,
        node_size=node_size,
        node_color=node_color,
        edge_color=edge_color,
        font_size=font_size
    )

    ax.set_title(title)
    ax.set_axis_off()

    return ax


def visualize_reconstruction(
    original_graph: Data,
    reconstructed_graph: Data,
    figsize: Tuple[int, int] = (12, 6),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Visualize original and reconstructed graphs side by side.

    Args:
        original_graph: Original PyTorch Geometric Data object
        reconstructed_graph: Reconstructed PyTorch Geometric Data object
        figsize: Figure size
        title: Figure title

    Returns:
        Matplotlib figure with the comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Draw original graph
    visualize_graph(
        original_graph,
        ax=ax1,
        title="Original Graph",
        node_size=200
    )

    # Draw reconstructed graph
    visualize_graph(
        reconstructed_graph,
        ax=ax2,
        title="Reconstructed Graph",
        node_size=200,
        node_color='#ff7f0e'
    )

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig


def visualize_latent_space(
    model: EncoderDecoderWrapper,
    dataset: List[Data],
    classes: Optional[List[int]] = None,
    method: str = 'pca',
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Latent Space Visualization"
) -> plt.Figure:
    """
    Visualize the latent space of the encoder.

    Args:
        model: Trained EncoderDecoderWrapper model
        dataset: List of graph data objects
        classes: Optional list of class labels for coloring points
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        figsize: Figure size
        title: Figure title

    Returns:
        Matplotlib figure with the visualization
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Encode all graphs
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for graph in dataset:
            latent = model.encoder(
                graph.to(model.encoder.convs[0].weight.device))
            latent_vectors.append(latent.cpu().numpy())

    # Stack latent vectors
    X = np.vstack(latent_vectors)

    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        X_reduced = reducer.fit_transform(X)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(random_state=42)
            X_reduced = reducer.fit_transform(X)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X)
    else:
        # Default to PCA
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points
    if classes is not None:
        # Color points by class
        scatter = ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            c=classes,
            cmap='viridis',
            alpha=0.8,
            s=50
        )
        plt.colorbar(scatter, label="Class")
    else:
        # No class information
        ax.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            alpha=0.8,
            s=50
        )

    ax.set_title(f"{title} ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation metrics.

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot (defaults to all available)
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure with the plots
    """
    # Determine which metrics to plot
    if metrics is None:
        # Default to all available metric keys that are not lists of dictionaries
        metrics = [k for k in history.keys() if not (
            isinstance(history[k], list) and
            history[k] and
            isinstance(history[k][0], dict)
        )]

    # Extract training and validation metrics from history
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Create figure
    num_plots = len(metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=figsize, sharex=True)

    # Ensure axes is always a list even for a single metric
    if num_plots == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric in history:
            ax.plot(epochs, history[metric], 'b-', label=f"Training {metric}")

            # Plot validation metric if available
            val_metric = f"val_{metric}" if not metric.startswith(
                "val_") else metric
            if val_metric in history and len(history[val_metric]) > 0:
                ax.plot(epochs, history[val_metric],
                        'r-', label=f"Validation {metric}")

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        # Only set xlabel on the bottom subplot
        if i == num_plots - 1:
            ax.set_xlabel("Epochs")

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_graph_reconstruction_examples(
    model: EncoderDecoderWrapper,
    dataset: List[Data],
    num_examples: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot examples of graph reconstructions.

    Args:
        model: Trained EncoderDecoderWrapper model
        dataset: List of graph data objects
        num_examples: Number of examples to show
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure with the plots
    """
    # Select random examples
    indices = np.random.choice(len(dataset), min(
        num_examples, len(dataset)), replace=False)
    examples = [dataset[i] for i in indices]

    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=figsize)

    # Ensure axes is 2D even for a single example
    if num_examples == 1:
        axes = axes.reshape(1, 2)

    # Generate reconstructions
    model.eval()

    with torch.no_grad():
        for i, graph in enumerate(examples):
            if i >= num_examples:
                break

            # Get device of model
            device = next(model.parameters()).device

            # Move graph to device
            graph = graph.to(device)

            # Reconstruct graph
            reconstructed_graph, _ = model(graph)

            # Draw original graph
            visualize_graph(
                graph,
                ax=axes[i, 0],
                title=f"Example {i+1}: Original",
                node_size=200
            )

            # Draw reconstructed graph
            visualize_graph(
                reconstructed_graph,
                ax=axes[i, 1],
                title=f"Example {i+1}: Reconstructed",
                node_size=200,
                node_color='#ff7f0e'
            )

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    model: EncoderDecoderWrapper,
    dataset: List[Data],
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix for edge prediction.

    Args:
        model: Trained EncoderDecoderWrapper model
        dataset: List of graph data objects
        threshold: Threshold for edge prediction
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure with the confusion matrix
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Collect true and predicted edges
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for graph in dataset:
            # Get device of model
            device = next(model.parameters()).device

            # Move graph to device
            graph = graph.to(device)

            # Get reconstruction
            reconstructed_graph, _ = model(graph)

            # Extract edges
            # Convert to adjacency matrices for comparison
            n = graph.num_nodes

            original_adj = torch.zeros(
                (n, n), dtype=torch.float, device=device)
            original_adj[graph.edge_index[0], graph.edge_index[1]] = 1.0

            # Get reconstructed adjacency matrix
            reconstructed_adj = torch.zeros(
                (n, n), dtype=torch.float, device=device)
            reconstructed_adj[reconstructed_graph.edge_index[0],
                              reconstructed_graph.edge_index[1]] = 1.0

            # Convert to binary predictions
            y_true.append(original_adj.cpu().numpy().flatten())
            y_pred.append(
                (reconstructed_adj.cpu().numpy().flatten() >= threshold).astype(int))

    # Flatten arrays
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Edge", "Edge"]
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    plt.title("Edge Prediction Confusion Matrix")
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
