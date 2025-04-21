"""
Graph Neural Network (GNN) encoders for converting graph structures to latent representations.

This module contains abstract base classes and implementations for GNN-based encoders
that transform graph structures into fixed-size latent representations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Optional

import torch_geometric
from torch_geometric.data import Data, Batch


class BaseGNNEncoder(nn.Module, ABC):
    """
    Abstract base class for GNN-based encoders.

    This class defines the common interface and functionality that all GNN encoder 
    implementations must provide, ensuring consistent usage patterns across different
    architectures.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, **kwargs):
        """
        Initialize the base GNN encoder.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of output latent representation
            **kwargs: Additional architecture-specific parameters
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Encode a graph structure into a latent representation.

        Args:
            graph_data: PyTorch Geometric graph data object containing
                        node features, edge indices, and potentially edge features

        Returns:
            Latent representation of the graph as a tensor
        """
        pass

    @abstractmethod
    def forward(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            graph_data: PyTorch Geometric graph data object

        Returns:
            Latent representation of the graph
        """
        pass

    def preprocess_graph(self, graph_data: Union[Data, Batch]) -> Union[Data, Batch]:
        """
        Preprocess the graph data before encoding.

        This method can be overridden by subclasses to implement specific
        preprocessing steps such as feature normalization or structure augmentation.

        Args:
            graph_data: PyTorch Geometric graph data object

        Returns:
            Preprocessed graph data
        """
        return graph_data

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the encoder.

        Returns:
            Dictionary containing the encoder configuration
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "type": self.__class__.__name__
        }


# Utility functions for evaluation metrics

def reconstruction_accuracy(original_graph: Data, reconstructed_graph: Data) -> float:
    """
    Calculate the edge reconstruction accuracy between original and reconstructed graphs.

    Args:
        original_graph: Original PyTorch Geometric graph data
        reconstructed_graph: Reconstructed graph data

    Returns:
        Accuracy score between 0 and 1
    """
    # Extract adjacency matrices
    original_edges = original_graph.edge_index
    reconstructed_edges = reconstructed_graph.edge_index

    # Convert to adjacency matrices for comparison
    n = max(original_graph.num_nodes, reconstructed_graph.num_nodes)

    original_adj = torch.zeros((n, n), dtype=torch.bool)
    original_adj[original_edges[0], original_edges[1]] = True

    reconstructed_adj = torch.zeros((n, n), dtype=torch.bool)
    reconstructed_adj[reconstructed_edges[0], reconstructed_edges[1]] = True

    # Calculate accuracy
    matches = (original_adj == reconstructed_adj).sum().item()
    total_entries = n * n

    return matches / total_entries


def edge_f1_score(original_graph: Data, reconstructed_graph: Data) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score for edge reconstruction.

    Args:
        original_graph: Original PyTorch Geometric graph data
        reconstructed_graph: Reconstructed graph data

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    # Extract adjacency matrices
    original_edges = original_graph.edge_index
    reconstructed_edges = reconstructed_graph.edge_index

    # Convert to adjacency matrices for comparison
    n = max(original_graph.num_nodes, reconstructed_graph.num_nodes)

    original_adj = torch.zeros((n, n), dtype=torch.bool)
    original_adj[original_edges[0], original_edges[1]] = True

    reconstructed_adj = torch.zeros((n, n), dtype=torch.bool)
    reconstructed_adj[reconstructed_edges[0], reconstructed_edges[1]] = True

    # Calculate true positives, false positives, false negatives
    true_positives = (original_adj & reconstructed_adj).sum().item()
    false_positives = (~original_adj & reconstructed_adj).sum().item()
    false_negatives = (original_adj & ~reconstructed_adj).sum().item()

    # Calculate metrics
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0

    return precision, recall, f1
