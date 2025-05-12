"""
Script to test the enhanced SimpleGraphLearner with anti-bias features.

This script tests the modified SimpleGraphLearner that includes:
1. Class-balanced loss with positive weighting
2. Bias initialization adjustment
3. Consistency regularization to push predictions toward 0 or 1
4. Density regularization

We compare the original model settings (with bias) against the enhanced model
to verify that our changes address the "no edge" prediction bias.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.structure_learning.training import train_simple_graph_learner
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.graph.generators.random_graphs import RandomGraphGenerator
from causal_meta.utils.advanced_visualization import (
    plot_edge_probabilities,
    plot_edge_probability_histogram,
    plot_edge_probability_distribution,
    plot_threshold_sensitivity
)

def generate_synthetic_data(num_nodes=5, num_samples=1000, seed=42):
    """
    Generate synthetic causal graph and data.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_samples: Number of data samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, true_adj_matrix, causal_graph)
    """
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random DAG with reasonable edge density
    causal_graph = RandomGraphGenerator.random_dag(
        num_nodes=num_nodes,
        edge_probability=0.3,  # 30% of possible edges are present
        seed=seed
    )
    
    # Convert to adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in causal_graph.get_children(i):
            adj_matrix[i, j] = 1
    
    # Create linear SCM
    scm = LinearSCMGenerator.generate_linear_scm(
        adj_matrix=adj_matrix,
        noise_scale=0.1,
        seed=seed
    )
    
    # Generate observational data
    data = scm.sample_data(sample_size=num_samples, as_array=False)
    
    # Convert data to numpy array
    data_numpy = np.column_stack([data[f'x{i}'] for i in range(num_nodes)])
    
    return data_numpy, adj_matrix, causal_graph

def compare_models(data, true_adj, output_dir, seed=42):
    """
    Compare biased and unbiased models on the same data.
    
    Args:
        data: Input data array
        true_adj: True adjacency matrix
        output_dir: Directory to save results
        seed: Random seed for reproducibility
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Convert data to torch tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    true_adj_tensor = torch.tensor(true_adj, dtype=torch.float32)
    
    num_nodes = data.shape[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Train the original model with biased settings
    biased_model, biased_history = train_simple_graph_learner(
        train_data=data_tensor,
        true_adj=true_adj_tensor,
        input_dim=num_nodes,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        sparsity_weight=0.1,  # Original sparsity weight
        acyclicity_weight=1.0,
        # No additional anti-bias parameters
        batch_size=32,
        lr=0.001,
        epochs=100,
        early_stopping_patience=10,
        normalize=True,
        verbose=True
    )
    
    # Train the enhanced model with anti-bias features
    unbiased_model, unbiased_history = train_simple_graph_learner(
        train_data=data_tensor,
        true_adj=true_adj_tensor,
        input_dim=num_nodes,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        sparsity_weight=0.05,  # Reduced sparsity weight
        acyclicity_weight=1.0,
        pos_weight=5.0,  # Weight positive examples 5x more
        consistency_weight=0.1,  # Encourage confident predictions
        edge_prob_bias=0.1,  # Initial bias toward predicting some edges
        expected_density=0.3,  # Expected edge density around 30%
        density_weight=0.1,  # Weight for density regularization
        batch_size=32,
        lr=0.001,
        epochs=100,
        early_stopping_patience=10,
        normalize=True,
        verbose=True
    )
    
    # Evaluate both models
    biased_model.eval()
    unbiased_model.eval()
    
    with torch.no_grad():
        # Get edge probabilities
        biased_probs = biased_model(data_tensor)
        unbiased_probs = unbiased_model(data_tensor)
        
        # Plot edge probabilities for biased model
        fig_biased = plot_edge_probabilities(
            true_adj=true_adj_tensor,
            edge_probs=biased_probs,
            save_path=os.path.join(output_dir, "biased_edge_probs.png")
        )
        
        # Plot edge probabilities for unbiased model
        fig_unbiased = plot_edge_probabilities(
            true_adj=true_adj_tensor,
            edge_probs=unbiased_probs,
            save_path=os.path.join(output_dir, "unbiased_edge_probs.png")
        )
        
        # Plot edge probability histograms
        fig_hist_biased = plot_edge_probability_histogram(
            edge_probs=biased_probs,
            save_path=os.path.join(output_dir, "biased_edge_histogram.png")
        )
        
        fig_hist_unbiased = plot_edge_probability_histogram(
            edge_probs=unbiased_probs,
            save_path=os.path.join(output_dir, "unbiased_edge_histogram.png")
        )
        
        # Plot edge probability distributions
        fig_dist_biased = plot_edge_probability_distribution(
            true_adj=true_adj_tensor,
            edge_probs=biased_probs,
            save_path=os.path.join(output_dir, "biased_edge_distribution.png")
        )
        
        fig_dist_unbiased = plot_edge_probability_distribution(
            true_adj=true_adj_tensor,
            edge_probs=unbiased_probs,
            save_path=os.path.join(output_dir, "unbiased_edge_distribution.png")
        )
        
        # Plot threshold sensitivity
        fig_thresh_biased = plot_threshold_sensitivity(
            true_adj=true_adj_tensor,
            edge_probs=biased_probs,
            save_path=os.path.join(output_dir, "biased_threshold_sensitivity.png")
        )
        
        fig_thresh_unbiased = plot_threshold_sensitivity(
            true_adj=true_adj_tensor,
            edge_probs=unbiased_probs,
            save_path=os.path.join(output_dir, "unbiased_threshold_sensitivity.png")
        )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(biased_history['train_loss'], label='Biased Model')
    plt.plot(unbiased_history['train_loss'], label='Unbiased Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if 'val_loss' in biased_history and biased_history['val_loss'] and 'val_loss' in unbiased_history and unbiased_history['val_loss']:
        plt.plot(biased_history['val_loss'], label='Biased Model')
        plt.plot(unbiased_history['val_loss'], label='Unbiased Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"))
    plt.close()
    
    # Compare metrics at threshold 0.5
    biased_adj = (biased_probs > 0.5).float()
    unbiased_adj = (unbiased_probs > 0.5).float()
    
    # Calculate metrics for biased model
    tp_biased = torch.logical_and(biased_adj == 1, true_adj_tensor == 1).sum().item()
    fp_biased = torch.logical_and(biased_adj == 1, true_adj_tensor == 0).sum().item()
    tn_biased = torch.logical_and(biased_adj == 0, true_adj_tensor == 0).sum().item()
    fn_biased = torch.logical_and(biased_adj == 0, true_adj_tensor == 1).sum().item()
    
    # Calculate metrics for unbiased model
    tp_unbiased = torch.logical_and(unbiased_adj == 1, true_adj_tensor == 1).sum().item()
    fp_unbiased = torch.logical_and(unbiased_adj == 1, true_adj_tensor == 0).sum().item()
    tn_unbiased = torch.logical_and(unbiased_adj == 0, true_adj_tensor == 0).sum().item()
    fn_unbiased = torch.logical_and(unbiased_adj == 0, true_adj_tensor == 1).sum().item()
    
    # Calculate derived metrics
    total_edges = true_adj.sum()
    total_possible_edges = true_adj.size - np.trace(true_adj).sum()
    
    accuracy_biased = (tp_biased + tn_biased) / total_possible_edges
    precision_biased = tp_biased / (tp_biased + fp_biased) if (tp_biased + fp_biased) > 0 else 0
    recall_biased = tp_biased / (tp_biased + fn_biased) if (tp_biased + fn_biased) > 0 else 0
    f1_biased = 2 * precision_biased * recall_biased / (precision_biased + recall_biased) if (precision_biased + recall_biased) > 0 else 0
    
    accuracy_unbiased = (tp_unbiased + tn_unbiased) / total_possible_edges
    precision_unbiased = tp_unbiased / (tp_unbiased + fp_unbiased) if (tp_unbiased + fp_unbiased) > 0 else 0
    recall_unbiased = tp_unbiased / (tp_unbiased + fn_unbiased) if (tp_unbiased + fn_unbiased) > 0 else 0
    f1_unbiased = 2 * precision_unbiased * recall_unbiased / (precision_unbiased + recall_unbiased) if (precision_unbiased + recall_unbiased) > 0 else 0
    
    # Save metrics to a text file
    with open(os.path.join(output_dir, "metrics_comparison.txt"), "w") as f:
        f.write("=== Metrics Comparison ===\n\n")
        f.write(f"True Graph Density: {total_edges / total_possible_edges:.4f} ({int(total_edges)} edges)\n\n")
        
        f.write("Biased Model Metrics:\n")
        f.write(f"  Accuracy:  {accuracy_biased:.4f}\n")
        f.write(f"  Precision: {precision_biased:.4f}\n")
        f.write(f"  Recall:    {recall_biased:.4f}\n")
        f.write(f"  F1 Score:  {f1_biased:.4f}\n")
        f.write(f"  TP: {tp_biased}, FP: {fp_biased}, TN: {tn_biased}, FN: {fn_biased}\n")
        f.write(f"  Predicted Edge Density: {(biased_adj.sum() / total_possible_edges).item():.4f}\n\n")
        
        f.write("Unbiased Model Metrics:\n")
        f.write(f"  Accuracy:  {accuracy_unbiased:.4f}\n")
        f.write(f"  Precision: {precision_unbiased:.4f}\n")
        f.write(f"  Recall:    {recall_unbiased:.4f}\n")
        f.write(f"  F1 Score:  {f1_unbiased:.4f}\n")
        f.write(f"  TP: {tp_unbiased}, FP: {fp_unbiased}, TN: {tn_unbiased}, FN: {fn_unbiased}\n")
        f.write(f"  Predicted Edge Density: {(unbiased_adj.sum() / total_possible_edges).item():.4f}\n\n")
        
        f.write("Improvement from Biased to Unbiased:\n")
        f.write(f"  Accuracy:  {accuracy_unbiased - accuracy_biased:.4f}\n")
        f.write(f"  Precision: {precision_unbiased - precision_biased:.4f}\n")
        f.write(f"  Recall:    {recall_unbiased - recall_biased:.4f}\n")
        f.write(f"  F1 Score:  {f1_unbiased - f1_biased:.4f}\n")
    
    print(f"Results saved to {output_dir}")

def main():
    # Set parameters
    num_nodes = 5
    num_samples = 1000
    seed = 42
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/balanced_model_comparison_{timestamp}"
    
    # Generate synthetic data
    data, true_adj, _ = generate_synthetic_data(num_nodes, num_samples, seed)
    
    # Compare models
    compare_models(data, true_adj, output_dir, seed)

if __name__ == "__main__":
    main() 