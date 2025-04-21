"""
Base classes for graph neural network encoders and decoders.

This module defines abstract base classes for GNN encoders and decoders,
as well as a wrapper class that combines them into a single model.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.graph_utils import batch_graphs, GraphBatcher


class GNNEncoder(nn.Module, ABC):
    """Abstract base class for GNN-based graph encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a graph into a latent representation.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes (optional)

        Returns:
            Latent representation [num_nodes, output_dim] or [batch_size, output_dim]
        """
        pass

    def encode_graph(self, graph: Data) -> torch.Tensor:
        """
        Encode a single graph object.

        Args:
            graph: PyTorch Geometric Data object

        Returns:
            Latent representation
        """
        return self.forward(graph.x, graph.edge_index)

    def encode_graphs(self, graphs: List[Data]) -> torch.Tensor:
        """
        Encode multiple graph objects.

        Args:
            graphs: List of PyTorch Geometric Data objects

        Returns:
            Batch of latent representations
        """
        if not graphs:
            return torch.empty(0, self.output_dim)

        # Use the GraphBatcher for proper handling of variable-sized graphs
        batcher = GraphBatcher()
        batch_data, batch_info = batcher.batch_graphs(graphs)

        # Forward pass with batch information
        return self.forward(batch_data.x, batch_data.edge_index, batch_data.batch)


class GNNDecoder(nn.Module, ABC):
    """Abstract base class for GNN-based graph decoders."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initialize the decoder.

        Args:
            input_dim: Input latent dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, z: torch.Tensor, edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode a latent representation back to graph structure.

        Args:
            z: Latent representation [num_nodes, input_dim] or [batch_size, input_dim]
            edge_index: Optional edge indices for reconstruction guidance [2, num_edges]
            batch: Batch assignment for nodes (optional)

        Returns:
            Tuple of:
                - Node features [num_nodes, output_dim]
                - Edge probabilities [num_nodes, num_nodes] or edge indices [2, num_predicted_edges]
        """
        pass

    def decode_to_graph(self, z: torch.Tensor, num_nodes: Optional[int] = None) -> Data:
        """
        Decode a latent representation to a graph object.

        Args:
            z: Latent representation
            num_nodes: Number of nodes to generate (if not inferred from z)

        Returns:
            PyTorch Geometric Data object
        """
        features, edge_data = self.forward(z)

        # Create graph object from decoded data
        if isinstance(edge_data, torch.Tensor) and edge_data.dim() == 2 and edge_data.size(0) == 2:
            # Edge data is already an edge index tensor
            edge_index = edge_data
        else:
            # Edge data is a probability matrix, convert to edge index
            edge_index = (edge_data > 0.5).nonzero().t()

        return Data(x=features, edge_index=edge_index)

    def decode_to_graphs(self, z: torch.Tensor, batch_size: int) -> List[Data]:
        """
        Decode batched latent representations to graph objects.

        Args:
            z: Batched latent representation
            batch_size: Number of graphs in the batch

        Returns:
            List of PyTorch Geometric Data objects
        """
        # Create fake batch assignment tensor if not provided in forward
        batch = torch.arange(batch_size, device=z.device).repeat_interleave(
            z.size(0) // batch_size)

        features, edge_data = self.forward(z, batch=batch)

        # Split decoded data by batch
        graph_list = []
        for i in range(batch_size):
            # Get indices for this graph
            indices = (batch == i).nonzero().squeeze()

            # Extract features for this graph
            graph_features = features[indices]

            # Extract edges for this graph
            if edge_data.dim() == 2 and edge_data.size(0) == 2:
                # Edge data is edge indices
                # Filter edges that connect nodes in this graph
                mask = torch.isin(edge_data[0], indices) & torch.isin(
                    edge_data[1], indices)
                graph_edge_index = edge_data[:, mask]

                # Remap indices to start from 0
                node_map = {idx.item(): i for i, idx in enumerate(indices)}
                graph_edge_index = torch.tensor([
                    [node_map[src.item()] for src in graph_edge_index[0]],
                    [node_map[dst.item()] for dst in graph_edge_index[1]]
                ], device=edge_data.device)
            else:
                # Edge data is a probability/adjacency matrix
                # Extract submatrix for this graph
                graph_edge_probs = edge_data[indices][:, indices]
                # Convert to edge index
                graph_edge_index = (graph_edge_probs > 0.5).nonzero().t()

            graph_list.append(
                Data(x=graph_features, edge_index=graph_edge_index))

        return graph_list


class EncoderDecoderWrapper(nn.Module):
    """
    Wrapper class that combines a GNN encoder and decoder into a single model.

    This class handles the combined encoding-decoding process and includes 
    methods for batch processing and loss computation.
    """

    def __init__(self, encoder: GNNEncoder, decoder: GNNDecoder):
        """
        Initialize the wrapper with encoder and decoder modules.

        Args:
            encoder: GNN encoder module
            decoder: GNN decoder module
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.graph_batcher = GraphBatcher(pad_to_max=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a graph through the encoder and decoder.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes (optional)

        Returns:
            Tuple of:
                - Reconstructed node features [num_nodes, output_dim]
                - Edge predictions (probabilities or indices)
                - Latent representation [num_nodes, latent_dim] or [batch_size, latent_dim]
        """
        # Encode the graph
        z = self.encoder(x, edge_index, batch)

        # Decode the latent representation
        reconstructed_x, edge_pred = self.decoder(z, edge_index, batch)

        return reconstructed_x, edge_pred, z

    def process_batch(self, graphs: List[Data]) -> Dict[str, Any]:
        """
        Process a batch of graphs through the encoder and decoder.

        Args:
            graphs: List of PyTorch Geometric Data objects

        Returns:
            Dictionary containing:
                - 'original_graphs': Original graph list
                - 'reconstructed_graphs': Reconstructed graph list
                - 'latent': Latent representations
                - 'batch_info': Batching metadata
        """
        if not graphs:
            raise ValueError("Cannot process an empty batch of graphs")

        # Batch the graphs with our custom batcher
        batch_data, batch_info = self.graph_batcher.batch_graphs(graphs)

        # Process the batch
        recon_x, edge_pred, z = self.forward(
            batch_data.x, batch_data.edge_index, batch_data.batch)

        # If edge_pred is a matrix of probabilities, convert to edge indices for PyG
        if edge_pred.dim() > 2:  # Dense adjacency tensor
            edge_indices = []
            batch_offset = 0

            for i, count in enumerate(batch_info['original_node_counts']):
                # Extract the probability submatrix for this graph
                if self.graph_batcher.pad_to_max:
                    max_nodes = batch_info['max_nodes']
                    prob_submatrix = edge_pred[i*max_nodes:(
                        i+1)*max_nodes, i*max_nodes:(i+1)*max_nodes]
                    # Only use the valid part (non-padding)
                    prob_submatrix = prob_submatrix[:count, :count]
                else:
                    # If not padding, use node count directly
                    prob_submatrix = edge_pred[batch_offset:batch_offset+count,
                                               batch_offset:batch_offset+count]
                    batch_offset += count

                # Threshold to get adjacency
                adj = (prob_submatrix > 0.5).nonzero().t()
                edge_indices.append(adj)
        else:
            # Edge predictions are already in edge index format
            # We need to split them by graph
            edge_indices = []
            for i, count in enumerate(batch_info['original_node_counts']):
                if self.graph_batcher.pad_to_max:
                    max_nodes = batch_info['max_nodes']
                    # Filter edges within this graph's nodes
                    node_range = torch.arange(
                        i*max_nodes, (i+1)*max_nodes, device=edge_pred.device)
                    mask = torch.isin(edge_pred[0], node_range) & torch.isin(
                        edge_pred[1], node_range)
                    graph_edges = edge_pred[:, mask]

                    # Adjust indices to start from 0 for this graph
                    graph_edges = graph_edges - i*max_nodes

                    # Keep only edges between actual nodes (not padding)
                    valid_nodes = torch.arange(count, device=edge_pred.device)
                    mask = torch.isin(graph_edges[0], valid_nodes) & torch.isin(
                        graph_edges[1], valid_nodes)
                    graph_edges = graph_edges[:, mask]
                else:
                    # Handle non-padded case
                    # TODO: Implement correct edge filtering for non-padded batches
                    pass

                edge_indices.append(graph_edges)

        # Reconstruct graphs
        reconstructed_graphs = []

        # Handle the latent representation format
        if z.dim() == 2 and batch_data.batch is not None:
            # z is per-node, gather by batch
            for i in range(batch_info['num_graphs']):
                # Get indices for this graph
                if self.graph_batcher.pad_to_max:
                    max_nodes = batch_info['max_nodes']
                    indices = torch.arange(i*max_nodes, i*max_nodes + batch_info['original_node_counts'][i],
                                           device=z.device)
                else:
                    indices = (batch_data.batch == i).nonzero().squeeze()

                # Extract features and edges for this graph
                graph_features = recon_x[indices]
                graph_edges = edge_indices[i]

                reconstructed_graphs.append(
                    Data(x=graph_features, edge_index=graph_edges))
        else:
            # z is already per-graph, create a graph for each latent vector
            for i in range(batch_info['num_graphs']):
                if self.graph_batcher.pad_to_max:
                    max_nodes = batch_info['max_nodes']
                    node_count = batch_info['original_node_counts'][i]
                    graph_features = recon_x[i *
                                             max_nodes:i*max_nodes + node_count]
                else:
                    # Handle non-padded case by using batch assignment
                    indices = (batch_data.batch == i).nonzero().squeeze()
                    graph_features = recon_x[indices]

                graph_edges = edge_indices[i]
                reconstructed_graphs.append(
                    Data(x=graph_features, edge_index=graph_edges))

        return {
            'original_graphs': graphs,
            'reconstructed_graphs': reconstructed_graphs,
            'latent': z,
            'batch_info': batch_info
        }

    def compute_reconstruction_loss(self, graphs: List[Data],
                                    node_loss_weight: float = 1.0,
                                    edge_loss_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss for a batch of graphs.

        Args:
            graphs: List of PyTorch Geometric Data objects
            node_loss_weight: Weight for node feature reconstruction loss
            edge_loss_weight: Weight for edge reconstruction loss

        Returns:
            Dictionary of loss components and total loss
        """
        batch_results = self.process_batch(graphs)
        original_graphs = batch_results['original_graphs']
        reconstructed_graphs = batch_results['reconstructed_graphs']
        batch_info = batch_results['batch_info']

        # Initialize loss components
        node_loss = torch.tensor(0.0, device=self.encoder.device)
        edge_loss = torch.tensor(0.0, device=self.encoder.device)

        # Compute loss for each graph
        for i, (orig, recon) in enumerate(zip(original_graphs, reconstructed_graphs)):
            # Node feature reconstruction loss
            node_loss += torch.nn.functional.mse_loss(recon.x, orig.x)

            # Edge reconstruction loss
            orig_adj = torch.zeros(
                (orig.num_nodes, orig.num_nodes), device=self.encoder.device)
            if orig.edge_index.shape[1] > 0:  # Check if there are any edges
                orig_adj[orig.edge_index[0], orig.edge_index[1]] = 1.0

            recon_adj = torch.zeros(
                (recon.num_nodes, recon.num_nodes), device=self.encoder.device)
            if recon.edge_index.shape[1] > 0:  # Check if there are any edges
                recon_adj[recon.edge_index[0], recon.edge_index[1]] = 1.0

            edge_loss += torch.nn.functional.binary_cross_entropy_with_logits(
                recon_adj.view(-1), orig_adj.view(-1))

        # Average over batch size
        batch_size = len(graphs)
        node_loss /= batch_size
        edge_loss /= batch_size

        # Total weighted loss
        total_loss = node_loss_weight * node_loss + edge_loss_weight * edge_loss

        return {
            'total_loss': total_loss,
            'node_loss': node_loss,
            'edge_loss': edge_loss
        }

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
