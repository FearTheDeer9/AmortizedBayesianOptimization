"""
Graph Isomorphism Network (GIN) decoder implementation.

This module contains the implementation of a GIN-based decoder for converting
latent representations back to graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, Optional, List

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

from causal_meta.inference.models.decoder import BaseGNNDecoder
from causal_meta.inference.models.graph_utils import edge_list_to_adj_matrix


class GINDecoder(BaseGNNDecoder):
    """
    Graph Isomorphism Network (GIN) decoder implementation.

    This decoder uses multiple GIN layers to transform latent representations
    back into node features, followed by edge prediction to reconstruct the
    graph structure.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_nodes: int,
        edge_prediction_method: str = "mlp",
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        threshold: float = 0.5,
        batch_norm: bool = True,
        epsilon: float = 0.0,
        train_epsilon: bool = False,
        **kwargs
    ):
        """
        Initialize GIN-based decoder.

        Args:
            latent_dim: Dimension of input latent representation
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node features
            num_nodes: Number of nodes in the output graph
            edge_prediction_method: Method to use for edge prediction ("mlp", "bilinear", "attention")
            num_layers: Number of GIN layers to use
            dropout: Dropout probability between layers
            activation: Activation function to use ("relu", "leaky_relu", "elu", "tanh")
            threshold: Threshold for edge prediction
            batch_norm: Whether to use batch normalization
            epsilon: Value of epsilon in GIN update equation (0 = sum, positive values = aggregation)
            train_epsilon: Whether to make epsilon a learnable parameter
            **kwargs: Additional parameters
        """
        super().__init__(latent_dim, hidden_dim, output_dim, num_nodes, **kwargs)

        # Store configuration
        self.num_nodes = num_nodes
        self.edge_prediction_method = edge_prediction_method
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_name = activation
        self.threshold = threshold
        self.use_batch_norm = batch_norm
        self.epsilon = epsilon
        self.train_epsilon = train_epsilon

        # Create activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.relu

        # MLP to transform latent representation to initial node features
        # If we need to support batches, initial features should be [batch_size, num_nodes, hidden_dim]
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Linear(hidden_dim, num_nodes * hidden_dim)
        )

        # Create GIN layers for refining node representations
        self.gin_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Helper method to create MLP for GIN
        def create_gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                self._get_activation(),
                nn.Linear(hidden_dim, out_dim)
            )

        # GIN layers for refining node features
        for i in range(num_layers - 1):
            self.gin_layers.append(GINConv(create_gin_mlp(
                hidden_dim, hidden_dim), train_eps=train_epsilon, eps=epsilon))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Final GIN layer to output node features
        self.gin_layers.append(GINConv(create_gin_mlp(
            hidden_dim, output_dim), train_eps=train_epsilon, eps=epsilon))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(output_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Edge prediction network
        if edge_prediction_method == "mlp":
            # MLP for edge prediction (takes concatenated node features)
            self.edge_predictor = nn.Sequential(
                nn.Linear(2 * output_dim, hidden_dim),
                self._get_activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif edge_prediction_method == "bilinear":
            # Bilinear layer for edge prediction
            self.edge_predictor = nn.Bilinear(output_dim, output_dim, 1)
            self.edge_activation = nn.Sigmoid()
        elif edge_prediction_method == "attention":
            # Attention-based edge prediction
            self.edge_query = nn.Linear(output_dim, hidden_dim)
            self.edge_key = nn.Linear(output_dim, hidden_dim)
            self.edge_predictor = nn.Linear(hidden_dim, 1)
            self.edge_activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unsupported edge prediction method: {edge_prediction_method}")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_activation(self):
        """
        Get the activation function as a module.

        Returns:
            Activation function module
        """
        if self.activation_name == "relu":
            return nn.ReLU()
        elif self.activation_name == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif self.activation_name == "elu":
            return nn.ELU()
        elif self.activation_name == "tanh":
            return nn.Tanh()
        else:
            return nn.ReLU()

    def forward(self, z: torch.Tensor, graph_data=None) -> Union[Data, Batch]:
        """
        Forward pass for the decoder.

        Args:
            z: Latent representation [batch_size, latent_dim] or [latent_dim]
            graph_data: Optional original graph data to match the structure

        Returns:
            Reconstructed graph structure
        """
        return self.decode(z, graph_data)

    def decode(self, z: torch.Tensor, graph_data=None) -> Union[Data, Batch]:
        """
        Decode a latent representation into a graph structure.

        Args:
            z: Latent representation [batch_size, latent_dim] or [latent_dim]
            graph_data: Optional original graph data to match node counts

        Returns:
            Reconstructed graph as a PyTorch Geometric Data object
        """
        # Handle both batched and non-batched inputs
        is_batched = z.dim() > 1 and z.size(0) > 1

        if is_batched:
            # Process batch of graphs
            batch_size = z.size(0)
            graphs = []

            # Get the original graphs to determine node counts (if provided)
            orig_graphs = None
            if graph_data is not None and hasattr(graph_data, 'to_data_list'):
                orig_graphs = graph_data.to_data_list()

            # Process each graph in the batch separately
            for i in range(batch_size):
                # Get latent for this graph
                single_latent = z[i]

                # Generate node features for this specific graph
                node_features = self.predict_node_features(single_latent)

                # Predict edges between nodes
                edge_index, edge_attr = self.predict_edges(node_features)

                # Create graph
                graph = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )

                # Apply postprocessing
                graph = self.postprocess_graph(graph)
                graphs.append(graph)

            # Return batch of graphs
            return Batch.from_data_list(graphs)
        else:
            # Generate node features
            node_features = self.predict_node_features(z)

            # Predict edges
            edge_index, edge_attr = self.predict_edges(node_features)

            # Create graph
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # Apply postprocessing
            return self.postprocess_graph(graph)

    def predict_node_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict initial node features from latent representation.

        Args:
            z: Latent representation

        Returns:
            Initial node features
        """
        # Apply MLP to transform latent representation to initial node features
        batch_size = z.shape[0] if z.dim() > 1 else 1
        z_flat = z.view(batch_size, -1)

        # Apply transformation
        node_feats = self.latent_to_features(z_flat)

        # Reshape to [batch_size, num_nodes, hidden_dim]
        node_feats = node_feats.view(
            batch_size, self.num_nodes, self.hidden_dim)

        # If single graph (non-batched), reshape to [num_nodes, hidden_dim]
        if batch_size == 1 and z.dim() == 1:
            node_feats = node_feats.squeeze(0)

        return node_feats

    def refine_node_features(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Refine node features using multiple GIN layers.

        Args:
            node_feats: Initial node features
            edge_index: Edge indices for message passing

        Returns:
            Refined node features
        """
        # If batch dimension exists, flatten for processing
        has_batch_dim = node_feats.dim() > 2
        batch_size = node_feats.shape[0] if has_batch_dim else 1

        if has_batch_dim:
            # Reshape from [batch_size, num_nodes, hidden_dim] to [batch_size * num_nodes, hidden_dim]
            x = node_feats.view(-1, node_feats.shape[-1])

            # Adjust edge indices for batched processing
            # We need to create batched edge indices where each graph's nodes are offset
            batched_edge_indices = []
            for i in range(batch_size):
                offset = i * self.num_nodes
                batch_edges = edge_index.clone()
                batch_edges = batch_edges + offset
                batched_edge_indices.append(batch_edges)
            edge_index = torch.cat(batched_edge_indices, dim=1)

            # Create batch assignment tensor
            batch = torch.repeat_interleave(
                torch.arange(batch_size, device=node_feats.device),
                self.num_nodes
            )
        else:
            # Single graph case
            x = node_feats
            batch = None

        # Apply GIN layers for message passing
        for i, gin_layer in enumerate(self.gin_layers):
            # Apply GIN convolution
            x = gin_layer(x, edge_index)

            # Apply batch normalization if enabled
            if self.use_batch_norm and self.batch_norms is not None:
                x = self.batch_norms[i](x)

            # Apply activation (except for last layer)
            if i < len(self.gin_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)

        # If we had batch dimension, reshape back
        if has_batch_dim:
            x = x.view(batch_size, self.num_nodes, -1)

        return x

    def predict_edges(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges from node features.

        Args:
            node_feats: Node features

        Returns:
            Edge indices and edge attributes
        """
        # Handle both batched and non-batched inputs
        has_batch_dim = node_feats.dim() > 2
        batch_size = node_feats.shape[0] if has_batch_dim else 1
        num_nodes = node_feats.shape[-2] if has_batch_dim else node_feats.shape[0]

        if has_batch_dim:
            # Process each graph in the batch
            edge_index_list = []
            edge_attr_list = []
            for i in range(batch_size):
                graph_feats = node_feats[i]
                graph_edge_index, graph_edge_attr = self._predict_edges_single_graph(
                    graph_feats)
                edge_index_list.append(graph_edge_index)
                edge_attr_list.append(graph_edge_attr)
            edge_index = torch.cat(edge_index_list, dim=1)
            edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            # Process single graph
            edge_index, edge_attr = self._predict_edges_single_graph(
                node_feats)

        return edge_index, edge_attr

    def _predict_edges_single_graph(self, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges for a single graph.

        Args:
            node_feats: Node features [num_nodes, feature_dim]

        Returns:
            Edge indices and edge attributes
        """
        num_nodes = node_feats.shape[0]
        device = node_feats.device

        # Create adjacency matrix to store edge probabilities
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)

        if self.edge_prediction_method == "mlp":
            # MLP-based edge prediction
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Skip self-loops
                        # Concatenate node features
                        edge_feats = torch.cat(
                            [node_feats[i], node_feats[j]], dim=0)
                        # Predict edge probability
                        prob = self.edge_predictor(edge_feats)
                        adj_matrix[i, j] = prob

        elif self.edge_prediction_method == "bilinear":
            # Bilinear edge prediction
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Skip self-loops
                        # Use bilinear layer for prediction
                        prob = self.edge_predictor(
                            node_feats[i].unsqueeze(0),
                            node_feats[j].unsqueeze(0)
                        )
                        adj_matrix[i, j] = self.edge_activation(prob)

        elif self.edge_prediction_method == "attention":
            # Attention-based edge prediction
            # Compute queries and keys for all nodes
            queries = self.edge_query(node_feats)  # [num_nodes, hidden_dim]
            keys = self.edge_key(node_feats)      # [num_nodes, hidden_dim]

            # Compute attention scores
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # Skip self-loops
                        # Compute attention score
                        query = queries[i].unsqueeze(0)  # [1, hidden_dim]
                        key = keys[j].unsqueeze(0)       # [1, hidden_dim]

                        # Dot product attention
                        attention = query * key  # Element-wise product

                        # Process through predictor
                        prob = self.edge_predictor(attention)
                        adj_matrix[i, j] = self.edge_activation(prob)

        # Convert adjacency matrix to edge index
        edge_index = torch.nonzero(adj_matrix > 0).t()
        edge_attr = adj_matrix[edge_index[0], edge_index[1]]

        return edge_index, edge_attr

    def postprocess_graph(self, graph_data: Data) -> Data:
        """
        Process a graph data object to apply thresholding and other post-processing.

        Args:
            graph_data: PyTorch Geometric Data object with x, edge_index, and edge_attr

        Returns:
            Processed graph data
        """
        # Extract components
        node_features = graph_data.x
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr

        # Threshold the edge predictions
        if edge_index.size(1) > 0:
            # Apply threshold
            mask = edge_attr >= self.threshold
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

        # Create new graph with thresholded edges
        processed_graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # Copy any other attributes
        for key, value in graph_data:
            if key not in ['x', 'edge_index', 'edge_attr']:
                processed_graph[key] = value

        return processed_graph

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the decoder.

        Returns:
            Dictionary containing the decoder configuration
        """
        config = super().get_config()
        config.update({
            "num_nodes": self.num_nodes,
            "edge_prediction_method": self.edge_prediction_method,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "threshold": self.threshold,
            "batch_norm": self.use_batch_norm,
            "epsilon": self.epsilon,
            "train_epsilon": self.train_epsilon
        })
        return config
