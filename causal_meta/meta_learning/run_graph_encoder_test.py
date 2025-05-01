#!/usr/bin/env python
"""
Demo script to test the GraphEncoder on synthetic data and visualize the results.
This demonstrates the end-to-end pipeline from data generation to graph inference.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import precision_recall_curve, average_precision_score

from causal_meta.graph.generators.factory import GraphFactory
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.meta_learning.graph_inference_utils import threshold_graph, compute_shd, GraphMetrics
from causal_meta.meta_learning.data_generation import SyntheticDataGenerator, GraphDataset, GraphDataLoader
from causal_meta.meta_learning.graph_encoder_training import GraphEncoderTrainer
from causal_meta.graph.causal_graph import CausalGraph


def create_sample_scm(num_nodes=5, edge_probability=0.3):
    """Create a sample SCM using the full StructuralCausalModel implementation."""
    # Create a CausalGraph for our SCM
    graph = CausalGraph()
    
    # Use string variable names
    node_names = [f"X{i}" for i in range(num_nodes)]
    
    # Add nodes
    for node in node_names:
        graph.add_node(node)
    
    # Add some edges to create a simple DAG
    if num_nodes >= 3:
        graph.add_edge(node_names[0], node_names[1])
        graph.add_edge(node_names[0], node_names[2])
    if num_nodes >= 4:
        graph.add_edge(node_names[1], node_names[3])
    if num_nodes >= 5:
        graph.add_edge(node_names[2], node_names[4])
    
    # Initialize the SCM with our graph
    scm = StructuralCausalModel(causal_graph=graph, variable_names=node_names)
    
    # Define structural equations for each node
    for node in node_names:
        parents = list(graph.get_parents(node))
        
        if not parents:  # Root node
            # Define a simple gaussian equation for source nodes
            scm.define_linear_gaussian_equation(
                variable=node,
                coefficients={},  # No parents
                intercept=0.0,
                noise_std=1.0
            )
        else:
            # Define a linear equation with coefficients for each parent
            coefficients = {parent: 1.0 for parent in parents}  # Set all coefficients to 1.0
            scm.define_linear_gaussian_equation(
                variable=node,
                coefficients=coefficients,
                intercept=0.0,
                noise_std=0.5
            )
    
    return scm


def plot_graphs(true_adj, pred_adj, title='Graph Comparison'):
    """Plot true and predicted graphs side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert adjacency matrices to networkx graphs
    true_g = nx.DiGraph()
    pred_g = nx.DiGraph()
    
    n_nodes = true_adj.shape[0]
    
    # Add nodes
    for i in range(n_nodes):
        true_g.add_node(i)
        pred_g.add_node(i)
    
    # Add edges
    for i in range(n_nodes):
        for j in range(n_nodes):
            if true_adj[i, j] > 0:
                true_g.add_edge(i, j)
            if pred_adj[i, j] > 0:
                pred_g.add_edge(i, j)
    
    # Create positions for nodes - same layout for both graphs
    pos = nx.spring_layout(true_g, seed=42)
    
    # Draw true graph
    nx.draw_networkx(true_g, pos, ax=ax1, with_labels=True, 
                     node_color='lightblue', node_size=500, 
                     arrows=True, arrowsize=15)
    ax1.set_title('True Graph')
    ax1.axis('off')
    
    # Draw predicted graph
    nx.draw_networkx(pred_g, pos, ax=ax2, with_labels=True, 
                     node_color='lightgreen', node_size=500, 
                     arrows=True, arrowsize=15)
    ax2.set_title('Predicted Graph')
    ax2.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_precision_recall_curve(true_adj, edge_probs):
    """Plot precision-recall curve for edge prediction."""
    # Flatten adjacency matrices for scoring
    true_edges = true_adj.flatten()
    pred_probs = edge_probs.flatten()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_edges, pred_probs)
    avg_precision = average_precision_score(true_edges, pred_probs)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'Average Precision: {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Edge Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt.gcf()


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    num_nodes = 5  # Reduced from 6 to 5 to match the function
    edge_probability = 0.3
    n_samples = 1000  # Increased from 500 to 1000
    seq_length = 15
    batch_size = 32
    
    # Create a sample SCM
    print("Creating structural causal model...")
    scm = create_sample_scm(num_nodes=num_nodes, edge_probability=edge_probability)
    
    # Get ground truth adjacency matrix
    true_adj = torch.tensor(scm.get_adjacency_matrix(), dtype=torch.float32)
    
    # Create dataset
    print("Generating dataset...")
    dataset = GraphDataset(
        scm=scm,
        n_samples=n_samples,
        seq_length=seq_length,
        add_noise=True,
        noise_type='gaussian',
        noise_scale=0.1
    )
    
    # Create data loader
    data_loader = GraphDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Create GraphEncoder model
    print("Initializing graph encoder...")
    encoder = GraphEncoder(
        hidden_dim=64,
        attention_heads=2,
        num_layers=2,
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Create trainer
    trainer = GraphEncoderTrainer(
        model=encoder,
        lr=0.0005,  # Lower learning rate
        sparsity_weight=0.1,
        acyclicity_weight=1.0
    )
    
    # Train the model
    print("Training the model...")
    history = trainer.train(
        train_loader=data_loader,
        num_epochs=100,  # Increased from 30 to 100
        validate=False,
        verbose=True
    )
    
    # Generate a new batch for testing
    print("Generating test data...")
    test_generator = SyntheticDataGenerator(scm)
    test_batch = test_generator.generate_batch(
        batch_size=1,
        seq_length=seq_length
    )
    
    # Predict graph
    print("Predicting graph structure...")
    encoder.eval()
    with torch.no_grad():
        edge_probs = encoder(test_batch)
    
    # Try different thresholds
    print("\nResults with different thresholds:")
    thresholds = [0.3, 0.5, 0.7]
    metrics = GraphMetrics(true_adj)
    
    best_f1 = 0
    best_threshold = 0.5
    best_results = None
    
    for threshold in thresholds:
        results = metrics.compute_all_metrics(edge_probs, threshold=threshold)
        print(f"Threshold {threshold}:")
        print(f"  SHD: {results['shd']}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1']:.4f}")
        
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            best_threshold = threshold
            best_results = results
    
    # Use the best threshold for final results
    print(f"\nBest results with threshold {best_threshold}:")
    print(f"SHD: {best_results['shd']}")
    print(f"Precision: {best_results['precision']:.4f}")
    print(f"Recall: {best_results['recall']:.4f}")
    print(f"F1 Score: {best_results['f1']:.4f}")
    
    # Threshold to get binary adjacency matrix with best threshold
    pred_adj = threshold_graph(edge_probs, threshold=best_threshold)
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png')
    
    # Plot graphs
    graph_fig = plot_graphs(true_adj.numpy(), pred_adj.numpy(), 
                            title=f'Graph Comparison (SHD: {best_results["shd"]}, F1: {best_results["f1"]:.2f})')
    graph_fig.savefig('graph_comparison.png')
    
    # Plot precision-recall curve
    pr_fig = plot_precision_recall_curve(true_adj.numpy(), edge_probs.numpy())
    pr_fig.savefig('precision_recall_curve.png')
    
    print("\nDone! Visualizations saved as PNG files.")


if __name__ == "__main__":
    main() 