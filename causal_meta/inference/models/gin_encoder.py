"""
Graph Isomorphism Network (GIN) encoder implementation.

This module contains the implementation of a GIN-based encoder for converting
graph structures to latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, Optional, List

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool

from causal_meta.inference.models.encoder import BaseGNNEncoder
from causal_meta.inference.models.graph_utils import preprocess_node_features


class GINEncoder(BaseGNNEncoder):
    """
    Graph Isomorphism Network (GIN) encoder implementation.

    This encoder uses multiple GIN layers to process node features,
    followed by pooling operations to generate a graph-level embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        pooling: str = "mean",
        batch_norm: bool = True,
        epsilon: float = 0.0,
        train_epsilon: bool = False,
        **kwargs
    ):
        """
        Initialize GIN-based encoder.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            latent_dim: Dimension of output latent representation
            num_layers: Number of GIN layers to use
            dropout: Dropout probability between layers
            activation: Activation function to use ("relu", "leaky_relu", "elu", "tanh")
            pooling: Pooling method for graph-level embeddings ("mean", "max", "sum")
            batch_norm: Whether to use batch normalization
            epsilon: Value of epsilon in GIN update equation (0 = sum, positive values = aggregation)
            train_epsilon: Whether to make epsilon a learnable parameter
            **kwargs: Additional parameters
        """
        super().__init__(input_dim, hidden_dim, latent_dim, **kwargs)

        # Store configuration
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_name = activation
        self.pooling_method = pooling
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

        # Create GIN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Helper method to create MLP for GIN
        def create_gin_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                self._get_activation(),
                nn.Linear(hidden_dim, out_dim)
            )

        # First layer: input_dim -> hidden_dim
        self.convs.append(GINConv(create_gin_mlp(
            input_dim, hidden_dim), train_eps=train_epsilon, eps=epsilon))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Middle layers: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            self.convs.append(GINConv(create_gin_mlp(
                hidden_dim, hidden_dim), train_eps=train_epsilon, eps=epsilon))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Last layer: hidden_dim -> latent_dim
        if num_layers > 1:
            self.convs.append(GINConv(create_gin_mlp(
                hidden_dim, latent_dim), train_eps=train_epsilon, eps=epsilon))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(latent_dim))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            graph_data: PyTorch Geometric graph data object

        Returns:
            Latent representation of the graph
        """
        return self.encode(graph_data)

    def encode(self, graph_data: Union[Data, Batch]) -> torch.Tensor:
        """
        Encode a graph structure into a latent representation.

        Args:
            graph_data: PyTorch Geometric graph data object

        Returns:
            Latent representation of the graph
        """
        # Get node features and edge indices
        x, edge_index = graph_data.x, graph_data.edge_index

        # Check if we're dealing with a batch or a single graph
        is_batch = hasattr(
            graph_data, 'batch') and graph_data.batch is not None
        batch = graph_data.batch if is_batch else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device)

        # Preprocess graph if needed
        x = preprocess_node_features(x, self.input_dim)

        # Apply GIN layers
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x = conv(x, edge_index)

            # Apply batch normalization if enabled
            if self.use_batch_norm and self.batch_norms is not None:
                # In evaluation mode, behavior is consistent because running stats are used
                # In training mode, we need to ensure batch stats are calculated consistently
                if self.training and not is_batch:
                    # For individual graphs during training, use running stats
                    # to mimic behavior as if it was part of a batch
                    # Store original settings
                    bn = self.batch_norms[i]
                    momentum = bn.momentum
                    track_running_stats = bn.track_running_stats

                    # Temporarily modify settings to use running stats even in training
                    bn.momentum = 0.0  # Don't update running stats
                    bn.track_running_stats = False  # Don't use running stats

                    # Apply batch norm
                    x = bn(x)

                    # Restore original settings
                    bn.momentum = momentum
                    bn.track_running_stats = track_running_stats
                else:
                    # For batches or during eval, use the batch norm as usual
                    x = self.batch_norms[i](x)

            # Apply activation (except for last layer if using an output normalization)
            if i < len(self.convs) - 1 or not getattr(self, 'normalize_output', False):
                x = self.activation(x)
                x = self.dropout(x)

        # Apply global pooling based on the specified method
        if self.pooling_method == "mean":
            pooled = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            pooled = global_max_pool(x, batch)
        elif self.pooling_method == "sum":
            pooled = global_add_pool(x, batch)
        else:
            # Default to mean pooling
            pooled = global_mean_pool(x, batch)

        # Return pooled features - keeping batch dimension for consistency
        # This change ensures we always return shape [batch_size, latent_dim], even for single graphs
        return pooled

    def preprocess_graph(self, graph_data: Union[Data, Batch]) -> Union[Data, Batch]:
        """
        Preprocess the graph data before encoding.

        Args:
            graph_data: PyTorch Geometric graph data object

        Returns:
            Preprocessed graph data
        """
        # Ensure node features have the correct dimension
        x = preprocess_node_features(graph_data.x, self.input_dim)

        # Create a new graph with the preprocessed features
        new_graph = Data(
            x=x,
            edge_index=graph_data.edge_index,
        )

        # Copy any other attributes
        for key, value in graph_data:
            if key not in ['x', 'edge_index']:
                new_graph[key] = value

        return new_graph

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the encoder.

        Returns:
            Dictionary containing the encoder configuration
        """
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "pooling": self.pooling_method,
            "batch_norm": self.use_batch_norm,
            "epsilon": self.epsilon,
            "train_epsilon": self.train_epsilon
        })
        return config
