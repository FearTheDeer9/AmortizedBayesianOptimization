"""
Encoder-Decoder Wrapper for GNN-based models.

This module provides a wrapper class that combines GNN encoders and decoders
for end-to-end training, along with loss functions and regularization options.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, Optional, List, Type

import torch_geometric
from torch_geometric.data import Data, Batch

from causal_meta.inference.models.encoder import BaseGNNEncoder, reconstruction_accuracy, edge_f1_score
from causal_meta.inference.models.decoder import BaseGNNDecoder
from causal_meta.inference.models.gcn_encoder import GCNEncoder
from causal_meta.inference.models.gcn_decoder import GCNDecoder
from causal_meta.inference.models.gat_encoder import GATEncoder
from causal_meta.inference.models.gat_decoder import GATDecoder
from causal_meta.inference.models.gin_encoder import GINEncoder
from causal_meta.inference.models.gin_decoder import GINDecoder
from causal_meta.inference.models.graph_utils import GraphBatcher


class EncoderDecoderWrapper(nn.Module):
    """
    Wrapper class that combines GNN encoder and decoder for end-to-end training.

    This class provides a unified interface for training and evaluating GNN-based 
    encoder-decoder models, with support for various loss functions and regularization options.
    """

    def __init__(
        self,
        encoder: BaseGNNEncoder,
        decoder: BaseGNNDecoder,
        loss_type: str = "bce",
        edge_weight: float = 1.0,
        feature_weight: float = 0.5,
        kl_weight: float = 0.0,
        l2_weight: float = 0.0,
        pad_graphs: bool = True,
        validate_edges: bool = True
    ):
        """
        Initialize the encoder-decoder wrapper.

        Args:
            encoder: GNN encoder model
            decoder: GNN decoder model
            loss_type: Type of edge prediction loss ('bce', 'mse', 'weighted_bce')
            edge_weight: Weight for edge prediction loss
            feature_weight: Weight for node feature reconstruction loss
            kl_weight: Weight for KL divergence regularization (if applicable)
            l2_weight: Weight for L2 regularization on latent space
            pad_graphs: Whether to pad graphs to the same number of nodes in a batch
            validate_edges: Whether to validate edge indices to ensure they're within graph boundaries
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_type = loss_type
        self.edge_weight = edge_weight
        self.feature_weight = feature_weight
        self.kl_weight = kl_weight
        self.l2_weight = l2_weight
        self.graph_batcher = GraphBatcher(pad_to_max=pad_graphs)
        self.validate_edges = validate_edges

    def forward(self, graph_data: Union[Data, Batch]) -> Tuple[Data, torch.Tensor]:
        """
        Forward pass through the encoder-decoder pipeline.

        Args:
            graph_data: Input graph data

        Returns:
            Tuple of (reconstructed_graph, latent_representation)
        """
        # Save the current training state
        training = self.training

        # Put models in eval mode during testing to avoid batch norm issues
        if not training:
            self.encoder.eval()
            self.decoder.eval()

        # Convert to list for batch processing if using a single graph
        if isinstance(graph_data, Data):
            graph_list = [graph_data]
            is_single_graph = True
        elif isinstance(graph_data, Batch):
            graph_list = graph_data.to_data_list()
            is_single_graph = False
        else:
            graph_list = graph_data  # Assume it's already a list
            is_single_graph = len(graph_list) == 1

        # Use the graph batcher for proper handling
        batch_data, batch_info = self.graph_batcher.batch_graphs(graph_list)

        # Encode the batch
        latent_representation = self.encoder(batch_data)

        # Decode the latent representation
        reconstructed_batch = self.decoder(latent_representation, batch_data)

        # Validate edge indices if needed
        if self.validate_edges and hasattr(reconstructed_batch, 'edge_index') and hasattr(batch_data, 'ptr'):
            # Apply edge validation to ensure all edges reference nodes within their graph
            validated_edge_indices = self.graph_batcher.validate_edge_indices(
                reconstructed_batch.edge_index,
                batch_data.ptr,
                batch_info['num_nodes_per_graph']
            )
            reconstructed_batch.edge_index = validated_edge_indices

        # Unbatch the reconstructed graphs if needed
        if is_single_graph:
            # If original input was a single graph, return a single graph
            result = reconstructed_batch if isinstance(
                reconstructed_batch, Data) else reconstructed_batch[0]
        else:
            # If original input was a batch, return a batch
            result = reconstructed_batch if isinstance(
                reconstructed_batch, Batch) else Batch.from_data_list(reconstructed_batch)

        # Restore the previous training state
        if not training:
            self.encoder.train(training)
            self.decoder.train(training)

        return result, latent_representation

    def process_batch(self, graph_list: List[Data]) -> Dict[str, Any]:
        """
        Process a batch of graphs through the model.

        Args:
            graph_list: List of graph data objects

        Returns:
            Dictionary containing results and metadata
        """
        # Use the graph batcher
        batch_data, batch_info = self.graph_batcher.batch_graphs(graph_list)

        # Encode and decode
        latent = self.encoder(batch_data)
        reconstructed_batch = self.decoder(latent, batch_data)

        # Validate edge indices if needed
        if self.validate_edges and hasattr(reconstructed_batch, 'edge_index') and hasattr(batch_data, 'ptr'):
            # Apply edge validation to ensure all edges reference nodes within their graph
            validated_edge_indices = self.graph_batcher.validate_edge_indices(
                reconstructed_batch.edge_index,
                batch_data.ptr,
                batch_info['num_nodes_per_graph']
            )
            reconstructed_batch.edge_index = validated_edge_indices

        # Ensure the result is a list of graphs
        if not isinstance(reconstructed_batch, list):
            if isinstance(reconstructed_batch, Batch):
                reconstructed_graphs = reconstructed_batch.to_data_list()
            else:
                reconstructed_graphs = [reconstructed_batch]
        else:
            reconstructed_graphs = reconstructed_batch

        # If we padded the graphs, remove padding
        if self.graph_batcher.pad_to_max:
            reconstructed_graphs = self.graph_batcher.unbatch_graphs(
                Batch.from_data_list(reconstructed_graphs), batch_info)

        # Manual validation for edge indices within each graph to prevent errors
        for i, graph in enumerate(reconstructed_graphs):
            if hasattr(graph, 'edge_index') and graph.edge_index.size(1) > 0:
                # Ensure edge indices don't exceed the number of nodes
                num_nodes = graph.x.size(0)
                valid_edges = (graph.edge_index[0] < num_nodes) & (
                    graph.edge_index[1] < num_nodes)
                graph.edge_index = graph.edge_index[:, valid_edges]

        return {
            'original_graphs': graph_list,
            'reconstructed_graphs': reconstructed_graphs,
            'latent': latent,
            'batch_info': batch_info
        }

    def compute_loss(
        self,
        original_graph: Union[Data, Batch, List[Data]],
        reconstructed_graph: Optional[Union[Data, Batch, List[Data]]] = None,
        latent_representation: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for the encoder-decoder model.

        Args:
            original_graph: Original input graph(s)
            reconstructed_graph: Reconstructed graph(s) from the decoder (optional, 
                                will be computed if not provided)
            latent_representation: Latent representation from the encoder (optional,
                                  will be computed if not provided)

        Returns:
            Dictionary of loss components and total loss
        """
        # If reconstructed_graph and latent_representation are not provided, compute them
        if reconstructed_graph is None or latent_representation is None:
            reconstructed_graph, latent_representation = self.forward(
                original_graph)

        # Convert to lists for easier processing
        if isinstance(original_graph, Data):
            original_graphs = [original_graph]
        elif isinstance(original_graph, Batch):
            original_graphs = original_graph.to_data_list()
        else:
            original_graphs = original_graph

        if isinstance(reconstructed_graph, Data):
            reconstructed_graphs = [reconstructed_graph]
        elif isinstance(reconstructed_graph, Batch):
            reconstructed_graphs = reconstructed_graph.to_data_list()
        else:
            reconstructed_graphs = reconstructed_graph

        # Initialize losses
        device = next(self.parameters()).device
        losses = {
            'edge_loss': torch.tensor(0.0, device=device),
            'feature_loss': torch.tensor(0.0, device=device),
            'kl_loss': torch.tensor(0.0, device=device),
            'l2_loss': torch.tensor(0.0, device=device)
        }

        # Compute loss for each graph pair
        for orig, recon in zip(original_graphs, reconstructed_graphs):
            # Edge prediction loss
            edge_loss = self._compute_edge_loss(orig, recon)
            losses['edge_loss'] += edge_loss

            # Feature reconstruction loss
            if self.feature_weight > 0:
                feature_loss = self._compute_feature_loss(orig, recon)
                losses['feature_loss'] += feature_loss

        # Average losses over batch size
        batch_size = len(original_graphs)
        losses['edge_loss'] /= batch_size
        losses['feature_loss'] /= batch_size

        # KL divergence regularization (applies to the whole batch)
        if self.kl_weight > 0:
            losses['kl_loss'] = self._compute_kl_divergence(
                latent_representation)

        # L2 regularization (applies to the whole batch)
        if self.l2_weight > 0:
            losses['l2_loss'] = self._compute_l2_regularization(
                latent_representation)

        # Compute total loss
        total_loss = (
            self.edge_weight * losses['edge_loss'] +
            self.feature_weight * losses['feature_loss'] +
            self.kl_weight * losses['kl_loss'] +
            self.l2_weight * losses['l2_loss']
        )
        losses['total_loss'] = total_loss

        return losses

    def _compute_edge_loss(
        self,
        original_graph: Data,
        reconstructed_graph: Data
    ) -> torch.Tensor:
        """
        Compute the edge prediction loss.

        Args:
            original_graph: Original input graph
            reconstructed_graph: Reconstructed graph from the decoder

        Returns:
            Edge prediction loss
        """
        # Extract adjacency matrices
        original_edges = original_graph.edge_index
        reconstructed_edges = reconstructed_graph.edge_index
        reconstructed_edge_attr = getattr(
            reconstructed_graph, 'edge_attr', None)

        # Get number of nodes
        if hasattr(original_graph, 'num_nodes'):
            n = original_graph.num_nodes
        else:
            n = max(original_edges[0].max().item(),
                    original_edges[1].max().item()) + 1

        # Ensure reconstructed graph has enough nodes
        if hasattr(reconstructed_graph, 'num_nodes'):
            n = max(n, reconstructed_graph.num_nodes)
        else:
            n = max(n, reconstructed_edges[0].max().item() + 1,
                    reconstructed_edges[1].max().item() + 1)

        # Convert to adjacency matrices for comparison
        device = original_edges.device
        original_adj = torch.zeros((n, n), dtype=torch.float, device=device)
        if original_edges.shape[1] > 0:  # Check if there are any edges
            original_adj[original_edges[0], original_edges[1]] = 1.0

        # If reconstructed_edge_attr is None or empty, just use binary edges
        reconstructed_adj = torch.zeros(
            (n, n), dtype=torch.float, device=device)
        if reconstructed_edges.shape[1] > 0:  # Check if there are any edges
            if reconstructed_edge_attr is None or reconstructed_edge_attr.numel() == 0:
                # Binary edges
                reconstructed_adj[reconstructed_edges[0],
                                  reconstructed_edges[1]] = 1.0
            else:
                # Use edge attributes as probabilities/weights
                reconstructed_adj[reconstructed_edges[0],
                                  reconstructed_edges[1]] = reconstructed_edge_attr

        # Flatten matrices for loss computation
        original_adj_flat = original_adj.view(-1)
        reconstructed_adj_flat = reconstructed_adj.view(-1)

        # Compute appropriate loss
        if self.loss_type == 'bce':
            # Ensure values are in [0, 1] range for BCE
            reconstructed_adj_flat = torch.clamp(reconstructed_adj_flat, 0, 1)
            loss = F.binary_cross_entropy(
                reconstructed_adj_flat, original_adj_flat)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(reconstructed_adj_flat, original_adj_flat)
        elif self.loss_type == 'weighted_bce':
            # Weighted BCE to handle class imbalance (sparse graphs)
            edge_count = original_adj_flat.sum()
            total_count = original_adj_flat.numel()
            pos_weight = (total_count - edge_count) / \
                edge_count if edge_count > 0 else torch.tensor(
                    1.0, device=device)

            # Ensure values are in proper range for BCE logits
            if torch.any(reconstructed_adj_flat < 0) or torch.any(reconstructed_adj_flat > 1):
                # Using logits version
                loss = F.binary_cross_entropy_with_logits(
                    reconstructed_adj_flat,
                    original_adj_flat,
                    pos_weight=pos_weight
                )
            else:
                # Using regular BCE with weighted averaging
                loss = F.binary_cross_entropy(
                    reconstructed_adj_flat,
                    original_adj_flat,
                    weight=torch.where(
                        original_adj_flat > 0, pos_weight, torch.tensor(1.0, device=device))
                )
        else:
            # Default to BCE
            reconstructed_adj_flat = torch.clamp(reconstructed_adj_flat, 0, 1)
            loss = F.binary_cross_entropy(
                reconstructed_adj_flat, original_adj_flat)

        return loss

    def _compute_feature_loss(
        self,
        original_graph: Data,
        reconstructed_graph: Data
    ) -> torch.Tensor:
        """
        Compute the node feature reconstruction loss.

        Args:
            original_graph: Original input graph
            reconstructed_graph: Reconstructed graph from the decoder

        Returns:
            Feature reconstruction loss
        """
        # Extract node features
        original_features = original_graph.x
        reconstructed_features = reconstructed_graph.x

        # Ensure dimensions match (might need padding/truncation)
        if original_features.shape != reconstructed_features.shape:
            min_nodes = min(
                original_features.shape[0], reconstructed_features.shape[0])
            min_features = min(
                original_features.shape[1], reconstructed_features.shape[1])
            original_features = original_features[:min_nodes, :min_features]
            reconstructed_features = reconstructed_features[:min_nodes, :min_features]

        # Compute MSE loss for feature reconstruction
        feature_loss = F.mse_loss(reconstructed_features, original_features)

        return feature_loss

    def _compute_kl_divergence(self, latent_representation: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence regularization for the latent space.

        Args:
            latent_representation: Latent representation from the encoder

        Returns:
            KL divergence loss
        """
        # Assuming a standard normal prior
        # If latent_representation is actually a mu, sigma pair (for VAE-style models)
        if isinstance(latent_representation, tuple) and len(latent_representation) == 2:
            mu, log_var = latent_representation
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # Normalize by dimension
            kl_loss = kl_loss / mu.size(0) if len(mu.shape) > 1 else kl_loss
        else:
            # For non-VAE models, regularize towards standard normal
            mu = latent_representation
            kl_loss = 0.5 * torch.sum(mu.pow(2))
            # Normalize by dimension
            kl_loss = kl_loss / mu.size(0) if len(mu.shape) > 1 else kl_loss

        return kl_loss

    def _compute_l2_regularization(self, latent_representation: torch.Tensor) -> torch.Tensor:
        """
        Compute L2 regularization for the latent space.

        Args:
            latent_representation: Latent representation from the encoder

        Returns:
            L2 regularization loss
        """
        # Handle VAE-style latent representations
        if isinstance(latent_representation, tuple):
            mu, _ = latent_representation
            l2_loss = torch.norm(mu, p=2)
        else:
            l2_loss = torch.norm(latent_representation, p=2)

        return l2_loss

    def evaluate(self, graph_data: Union[Data, Batch, List[Data]]) -> Dict[str, Any]:
        """
        Evaluate the model on graph data.

        Args:
            graph_data: Graph data to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        # Ensure we're in evaluation mode
        self.eval()

        # Process to handle different input types
        if isinstance(graph_data, Data):
            graph_list = [graph_data]
        elif isinstance(graph_data, Batch):
            graph_list = graph_data.to_data_list()
        else:
            graph_list = graph_data  # Assume it's already a list

        # Get reconstructed graphs and compute loss
        results = self.process_batch(graph_list)
        reconstructed_graphs = results['reconstructed_graphs']
        losses = self.compute_loss(graph_list, reconstructed_graphs)

        # Initialize metrics
        metrics = {
            'loss': losses['total_loss'].item(),
            'edge_loss': losses['edge_loss'].item(),
            'feature_loss': losses['feature_loss'].item(),
        }

        # Initialize sums for averaging
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        acc_sum = 0.0

        # Handle any edge index validation issues before computing metrics
        # This ensures edge_f1_score won't fail
        for i, graph in enumerate(reconstructed_graphs):
            # Make sure edge indices are valid
            if hasattr(graph, 'edge_index') and graph.edge_index.size(1) > 0:
                num_nodes = graph.num_nodes if hasattr(
                    graph, 'num_nodes') else graph.x.size(0)
                valid_edges = (graph.edge_index[0] < num_nodes) & (
                    graph.edge_index[1] < num_nodes)
                graph.edge_index = graph.edge_index[:, valid_edges]

        for orig, recon in zip(graph_list, reconstructed_graphs):
            try:
                # Compute edge F1 score
                precision, recall, f1 = edge_f1_score(orig, recon)
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1

                # Compute reconstruction accuracy
                acc_sum += reconstruction_accuracy(orig, recon)
            except Exception as e:
                # Log the error but continue processing other graphs
                print(f"Error computing metrics: {str(e)}")
                continue

        # Average metrics
        batch_size = len(graph_list)
        if batch_size > 0:  # Avoid division by zero
            metrics['edge_precision'] = precision_sum / batch_size
            metrics['edge_recall'] = recall_sum / batch_size
            metrics['edge_f1'] = f1_sum / batch_size
            metrics['reconstruction_accuracy'] = acc_sum / batch_size

        return metrics

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the wrapper.

        Returns:
            Dictionary containing the wrapper configuration
        """
        return {
            "encoder_type": self.encoder.__class__.__name__,
            "decoder_type": self.decoder.__class__.__name__,
            "encoder_config": self.encoder.get_config() if hasattr(self.encoder, 'get_config') else {},
            "decoder_config": self.decoder.get_config() if hasattr(self.decoder, 'get_config') else {},
            "loss_type": self.loss_type,
            "edge_weight": self.edge_weight,
            "feature_weight": self.feature_weight,
            "kl_weight": self.kl_weight,
            "l2_weight": self.l2_weight
        }


def create_model(
    architecture: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    output_dim: int,
    num_nodes: int,
    **kwargs
) -> EncoderDecoderWrapper:
    """
    Factory function to create encoder-decoder models with different architectures.

    Args:
        architecture: GNN architecture to use ('gcn', 'gat', 'gin')
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        latent_dim: Dimension of latent space
        output_dim: Dimension of output node features
        num_nodes: Number of nodes in the graph
        **kwargs: Additional parameters for encoder, decoder, and wrapper

    Returns:
        Initialized EncoderDecoderWrapper
    """
    # Extract parameters for different components
    encoder_kwargs = {k: v for k,
                      v in kwargs.items() if not k.startswith('decoder_')}
    decoder_kwargs = {k[8:]: v for k,
                      v in kwargs.items() if k.startswith('decoder_')}
    wrapper_kwargs = {k: v for k, v in kwargs.items()
                      if not (k.startswith('encoder_') or k.startswith('decoder_'))}

    # Filter wrapper_kwargs to only include parameters accepted by EncoderDecoderWrapper
    allowed_wrapper_params = ['loss_type', 'edge_weight',
                              'feature_weight', 'kl_weight', 'l2_weight', 'pad_graphs']
    wrapper_kwargs = {k: v for k, v in wrapper_kwargs.items()
                      if k in allowed_wrapper_params}

    # Ensure output_dim and input_dim match for autoencoding
    decoder_kwargs['output_dim'] = output_dim

    # Create encoder and decoder based on architecture
    if architecture.lower() == 'gcn':
        encoder = GCNEncoder(input_dim, hidden_dim,
                             latent_dim, **encoder_kwargs)
        decoder = GCNDecoder(latent_dim, hidden_dim,
                             output_dim, num_nodes, **decoder_kwargs)
    elif architecture.lower() == 'gat':
        encoder = GATEncoder(input_dim, hidden_dim,
                             latent_dim, **encoder_kwargs)
        decoder = GATDecoder(latent_dim, hidden_dim,
                             output_dim, num_nodes, **decoder_kwargs)
    elif architecture.lower() == 'gin':
        encoder = GINEncoder(input_dim, hidden_dim,
                             latent_dim, **encoder_kwargs)
        decoder = GINDecoder(latent_dim, hidden_dim,
                             output_dim, num_nodes, **decoder_kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. "
                         f"Choose from 'gcn', 'gat', or 'gin'.")

    # Create wrapper
    wrapper = EncoderDecoderWrapper(encoder, decoder, **wrapper_kwargs)

    return wrapper
