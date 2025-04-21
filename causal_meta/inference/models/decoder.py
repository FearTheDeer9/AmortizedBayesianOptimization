"""
Graph Neural Network (GNN) decoders for converting latent representations back to graph structures.

This module contains abstract base classes and implementations for GNN-based decoders
that transform latent representations into reconstructed graph structures.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Tuple, Optional, List

import torch_geometric
from torch_geometric.data import Data, Batch


class BaseGNNDecoder(nn.Module, ABC):
    """
    Abstract base class for GNN-based decoders.

    This class defines the common interface and functionality that all GNN decoder 
    implementations must provide, ensuring consistent usage patterns across different
    architectures.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, num_nodes: int, **kwargs):
        """
        Initialize the base GNN decoder.

        Args:
            latent_dim: Dimension of input latent representation
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node features
            num_nodes: Number of nodes in the output graph
            **kwargs: Additional architecture-specific parameters
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes

    @abstractmethod
    def decode(self, latent_representation: torch.Tensor) -> Data:
        """
        Decode a latent representation back into a graph structure.

        Args:
            latent_representation: Latent representation tensor from encoder

        Returns:
            Reconstructed PyTorch Geometric graph data object
        """
        pass

    @abstractmethod
    def forward(self, latent_representation: torch.Tensor) -> Data:
        """
        Forward pass for the decoder.

        Args:
            latent_representation: Latent representation tensor from encoder

        Returns:
            Reconstructed graph data
        """
        pass

    @abstractmethod
    def predict_edges(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges between nodes based on their embeddings.

        Args:
            node_embeddings: Node embeddings tensor

        Returns:
            Tuple of (edge_index, edge_attr) where edge_index is the predicted edge indices
            and edge_attr is the predicted edge attributes/probabilities
        """
        pass

    @abstractmethod
    def predict_node_features(self, latent_representation: torch.Tensor) -> torch.Tensor:
        """
        Predict node features from the latent representation.

        Args:
            latent_representation: Latent representation tensor from encoder

        Returns:
            Predicted node features
        """
        pass

    def postprocess_graph(self, graph_data: Data) -> Data:
        """
        Postprocess the reconstructed graph.

        This method can be overridden by subclasses to implement specific
        postprocessing steps such as thresholding edges or normalizing features.

        Args:
            graph_data: Reconstructed PyTorch Geometric graph data

        Returns:
            Postprocessed graph data
        """
        return graph_data

    def threshold_edges(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                        threshold: float = 0.5) -> torch.Tensor:
        """
        Apply thresholding to edge predictions.

        Args:
            edge_index: Edge indices tensor [2, num_edges]
            edge_attr: Edge probabilities or attributes
            threshold: Threshold value for edge existence

        Returns:
            Thresholded edge indices
        """
        # For binary edge prediction
        if edge_attr.dim() == 1:
            mask = edge_attr >= threshold
            return edge_index[:, mask]

        # For multi-class edge prediction
        else:
            # Get max values along class dimension
            values, _ = torch.max(edge_attr, dim=1)
            mask = values >= threshold
            return edge_index[:, mask]

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the decoder.

        Returns:
            Dictionary containing the decoder configuration
        """
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_nodes": self.num_nodes,
            "type": self.__class__.__name__
        }


# Utility functions for graph conversion

def adjacency_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert adjacency matrix to edge index format.

    Args:
        adj_matrix: Adjacency matrix [num_nodes, num_nodes]

    Returns:
        Edge index tensor [2, num_edges]
    """
    edge_indices = adj_matrix.nonzero(as_tuple=True)
    return torch.stack(edge_indices)


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Convert edge index to adjacency matrix.

    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph

    Returns:
        Adjacency matrix [num_nodes, num_nodes]
    """
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    if edge_index.size(1) > 0:  # Check if there are any edges
        adj_matrix[edge_index[0], edge_index[1]] = True
    return adj_matrix


def continuous_to_discrete_graph(edge_index: torch.Tensor, edge_attr: torch.Tensor,
                                 node_features: torch.Tensor, threshold: float = 0.5) -> Data:
    """
    Convert continuous edge predictions to discrete graph.

    Args:
        edge_index: Edge index tensor [2, num_potential_edges]
        edge_attr: Edge probabilities [num_potential_edges]
        node_features: Node feature tensor [num_nodes, feature_dim]
        threshold: Probability threshold for edge existence

    Returns:
        PyTorch Geometric Data object with discrete edges
    """
    # Threshold edges
    mask = edge_attr >= threshold
    discrete_edge_index = edge_index[:, mask]

    # Create graph data object
    graph = Data(x=node_features, edge_index=discrete_edge_index)

    return graph
