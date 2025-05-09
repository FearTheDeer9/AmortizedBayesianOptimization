"""
Example demonstrating the use of non-GNN models for causal structure inference.

This script shows how to use MLP and Transformer-based models for causal structure
inference, as alternatives to GNN-based models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from causal_meta.inference.models.mlp_encoder import MLPGraphEncoder
from causal_meta.inference.models.transformer_encoder import TransformerGraphEncoder
from causal_meta.inference.adapters import (
    MLPGraphEncoderAdapter,
    TransformerGraphEncoderAdapter
)
from causal_meta.graph.causal_graph import CausalGraph


def generate_synthetic_data(n_samples=100, n_variables=5, seq_length=20):
    """Generate synthetic time series data for testing."""
    # Create a random adjacency matrix (ground truth)
    adj_matrix = np.zeros((n_variables, n_variables))
    for i in range(n_variables):
        for j in range(n_variables):
            if i != j and np.random.rand() < 0.3:  # 30% chance of edge
                adj_matrix[i, j] = 1
    
    # Generate time series data based on the adjacency matrix
    # This is a simple AR model where each variable depends on its parents
    data = np.zeros((n_samples, seq_length, n_variables))
    
    # Initialize with random values
    data[:, 0, :] = np.random.randn(n_samples, n_variables)
    
    # Generate remaining time steps
    for t in range(1, seq_length):
        for i in range(n_variables):
            # Autoregressive component (depends on own past)
            data[:, t, i] = 0.1 * data[:, t-1, i]
            
            # Add parent influences
            for j in range(n_variables):
                if adj_matrix[j, i] == 1:  # j is a parent of i
                    data[:, t, i] += 0.3 * data[:, t-1, j]
            
            # Add noise
            data[:, t, i] += 0.1 * np.random.randn(n_samples)
    
    return data, adj_matrix


def visualize_graphs(true_adj, mlp_adj, transformer_adj):
    """Visualize the true and inferred adjacency matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True adjacency matrix
    axes[0].imshow(true_adj, cmap='Blues')
    axes[0].set_title('True Graph')
    axes[0].set_xticks(range(len(true_adj)))
    axes[0].set_yticks(range(len(true_adj)))
    
    # MLP inferred adjacency matrix
    axes[1].imshow(mlp_adj, cmap='Blues')
    axes[1].set_title('MLP Inferred Graph')
    axes[1].set_xticks(range(len(mlp_adj)))
    axes[1].set_yticks(range(len(mlp_adj)))
    
    # Transformer inferred adjacency matrix
    axes[2].imshow(transformer_adj, cmap='Blues')
    axes[2].set_title('Transformer Inferred Graph')
    axes[2].set_xticks(range(len(transformer_adj)))
    axes[2].set_yticks(range(len(transformer_adj)))
    
    plt.tight_layout()
    plt.savefig('non_gnn_model_comparison.png')
    plt.show()


def main():
    # Parameters
    n_variables = 5
    n_samples = 100
    seq_length = 20
    hidden_dim = 64
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data, true_adj = generate_synthetic_data(
        n_samples=n_samples,
        n_variables=n_variables,
        seq_length=seq_length
    )
    
    # Convert to tensor
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    # Initialize models
    print("Initializing models...")
    
    # MLP-based model
    mlp_model = MLPGraphEncoder(
        input_dim=n_variables,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        seq_length=seq_length
    )
    
    # Transformer-based model
    transformer_model = TransformerGraphEncoder(
        input_dim=n_variables,
        hidden_dim=hidden_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    )
    
    # Create adapters
    mlp_adapter = MLPGraphEncoderAdapter(mlp_model)
    transformer_adapter = TransformerGraphEncoderAdapter(transformer_model)
    
    # Prepare data for inference
    inference_data = {
        "observations": data
    }
    
    # Infer causal structures
    print("Inferring causal structures...")
    mlp_graph = mlp_adapter.infer_structure(inference_data)
    transformer_graph = transformer_adapter.infer_structure(inference_data)
    
    # Get adjacency matrices
    mlp_adj = mlp_graph.get_adjacency_matrix()
    transformer_adj = transformer_graph.get_adjacency_matrix()
    
    # Get uncertainty estimates
    print("Getting uncertainty estimates...")
    mlp_uncertainty = mlp_adapter.estimate_uncertainty()
    transformer_uncertainty = transformer_adapter.estimate_uncertainty()
    
    print(f"MLP edge probabilities shape: {mlp_uncertainty['edge_probabilities'].shape}")
    print(f"Transformer edge probabilities shape: {transformer_uncertainty['edge_probabilities'].shape}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_graphs(true_adj, mlp_adj, transformer_adj)
    
    print("Done! Visualizations saved to 'non_gnn_model_comparison.png'")


if __name__ == "__main__":
    main() 