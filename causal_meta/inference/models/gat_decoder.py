"""
Graph Attention Network (GAT) decoder implementation.

This module contains the implementation of a GAT-based decoder for converting
latent representations back to graph structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Union, Tuple, Optional, List

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

from causal_meta.inference.models.decoder import BaseGNNDecoder
from causal_meta.inference.models.graph_utils import compute_edge_weights


class GATDecoder(BaseGNNDecoder):
    """
    Graph Attention Network (GAT) decoder implementation.

    This decoder transforms latent representations back into graph structures
    by first generating node features and then predicting edges between nodes using attention.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_nodes: int,
        edge_prediction_method: str = "inner_product",
        num_layers: int = 3,
        dropout: float = 0.1,
        activation: str = "relu",
        threshold: float = 0.5,
        batch_norm: bool = True,
        heads: int = 4,
        concat_heads: bool = True,
        **kwargs
    ):
        """
        Initialize the GAT-based decoder.

        Args:
            latent_dim: Dimension of input latent representation
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output node features
            num_nodes: Number of nodes in the output graph
            edge_prediction_method: Method for predicting edges ("inner_product", "mlp", "bilinear", "attention")
            num_layers: Number of GAT layers to use
            dropout: Dropout probability between layers
            activation: Activation function to use ("relu", "leaky_relu", "elu", "tanh")
            threshold: Threshold value for edge prediction
            batch_norm: Whether to use batch normalization
            heads: Number of attention heads
            concat_heads: Whether to concatenate or average attention heads
            **kwargs: Additional parameters
        """
        super().__init__(latent_dim, hidden_dim, output_dim, num_nodes, **kwargs)

        # Store configuration
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_name = activation
        self.edge_prediction_method = edge_prediction_method
        self.threshold_value = threshold
        self.use_batch_norm = batch_norm
        self.heads = heads
        self.concat_heads = concat_heads

        # Create activation function
        if activation == "relu":
            self.activation = F.relu
            activation_module = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
            activation_module = nn.LeakyReLU(0.2)
        elif activation == "elu":
            self.activation = F.elu
            activation_module = nn.ELU()
        elif activation == "tanh":
            self.activation = torch.tanh
            activation_module = nn.Tanh()
        else:
            self.activation = F.relu
            activation_module = nn.ReLU()

        # Create MLP for latent to initial node features
        self.latent_to_nodes = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * num_nodes),
            nn.BatchNorm1d(
                hidden_dim * num_nodes) if batch_norm else nn.Identity(),
            activation_module,
            nn.Dropout(dropout),
        )

        # Create GAT layers for refining node representations
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Calculate dimensions for the GAT layers
        if concat_heads:
            # If concatenating heads, ensure dimensions are divisible by heads
            # For hidden layers, make sure each head gets at least 1 feature
            if hidden_dim < heads:
                gat_hidden_dim = 1  # Each head gets at least 1 feature
                hidden_dim = heads  # Adjust hidden_dim to match
            else:
                gat_hidden_dim = hidden_dim // heads

            # For output layer, adjust similarly
            if output_dim < heads:
                gat_output_dim = 1
                self.output_dim = heads  # Update output_dim
            else:
                gat_output_dim = output_dim // heads

            # Actual output dimensions after concatenation
            actual_hidden_dim = gat_hidden_dim * heads
            actual_output_dim = gat_output_dim * heads
        else:
            # If averaging heads, dimensions stay the same
            gat_hidden_dim = hidden_dim
            gat_output_dim = output_dim
            actual_hidden_dim = hidden_dim
            actual_output_dim = output_dim

        # First layer: hidden_dim -> hidden_dim
        self.convs.append(GATConv(hidden_dim, gat_hidden_dim,
                          heads=heads, concat=concat_heads, dropout=dropout))
        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(actual_hidden_dim))

        # Middle layers: hidden_dim -> hidden_dim
        for i in range(num_layers - 2):
            self.convs.append(GATConv(actual_hidden_dim, gat_hidden_dim,
                              heads=heads, concat=concat_heads, dropout=dropout))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(actual_hidden_dim))

        # Last layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.convs.append(GATConv(actual_hidden_dim, gat_output_dim,
                              heads=heads, concat=concat_heads, dropout=dropout))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(actual_output_dim))

        # Output projection if needed
        if concat_heads and actual_output_dim != output_dim:
            self.output_projection = nn.Linear(actual_output_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Edge prediction network based on method
        if edge_prediction_method == "mlp":
            self.edge_predictor = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if batch_norm else nn.Identity(),
                activation_module,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        elif edge_prediction_method == "bilinear":
            self.edge_predictor = nn.Bilinear(output_dim, output_dim, 1)
        elif edge_prediction_method == "attention":
            # Attention-based edge prediction - uses a scored attention mechanism
            self.edge_attention = nn.Sequential(
                nn.Linear(output_dim * 2, hidden_dim),
                activation_module,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

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
        elif isinstance(module, GATConv):
            if hasattr(module, 'lin'):
                nn.init.xavier_uniform_(module.lin.weight)
                if module.lin.bias is not None:
                    nn.init.zeros_(module.lin.bias)
            if hasattr(module, 'att'):
                nn.init.xavier_uniform_(module.att)

    def _get_activation(self):
        """
        Get the activation function as a function reference (not a module).

        Returns:
            Activation function
        """
        if self.activation_name == "relu":
            return F.relu
        elif self.activation_name == "leaky_relu":
            return F.leaky_relu
        elif self.activation_name == "elu":
            return F.elu
        elif self.activation_name == "tanh":
            return torch.tanh
        else:
            return F.relu

    def forward(self, z: torch.Tensor, batch=None):
        """
        Forward pass of the decoder.

        Args:
            z: Latent representation [batch_size, latent_dim] or [latent_dim]
            batch: Optional batch vector for batched processing

        Returns:
            Reconstructed graph as a PyTorch Geometric Data object or Batch
        """
        # Determine if we have batched input (z has shape [batch_size, latent_dim])
        is_batched = z.dim() > 1 and z.size(0) > 1

        if is_batched:
            # Handle batched input
            batch_size = z.size(0)
            graphs = []

            # Process each graph in the batch separately
            for i in range(batch_size):
                # Get latent for this graph
                single_latent = z[i]

                # Generate node features for this specific graph
                node_features = self.predict_node_features(single_latent)

                # Create a fully connected edge index for this graph
                edge_index = self._create_fully_connected_edge_index(
                    node_features.size(0), z.device)

                # Refine node features using GAT layers
                refined_features = self.refine_node_features(node_features)

                # Predict edges between nodes
                edge_index, edge_attr = self.predict_edges(refined_features)

                # Create graph
                graph = Data(
                    x=refined_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr
                )

                # Apply postprocessing
                graph = self.postprocess_graph(graph)
                graphs.append(graph)

            # Return batch of graphs
            return Batch.from_data_list(graphs)
        else:
            # Handle single graph
            # Generate node features
            node_features = self.predict_node_features(z)

            # Create a fully connected edge index
            edge_index = self._create_fully_connected_edge_index(
                node_features.size(0), z.device)

            # Refine node features using GAT layers
            refined_features = self.refine_node_features(node_features)

            # Predict edges
            edge_index, edge_attr = self.predict_edges(refined_features)

            # Create graph
            graph = Data(
                x=refined_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )

            # Apply postprocessing
            return self.postprocess_graph(graph)

    def decode(self, latent_representation: torch.Tensor, graph_data=None) -> Union[Data, Batch]:
        """
        Decode a latent representation back into a graph structure.

        Args:
            latent_representation: Latent representation tensor from encoder
            graph_data: Optional original graph data to match node counts

        Returns:
            Reconstructed PyTorch Geometric graph data object
        """
        # Simply pass to the forward method which now correctly handles both batched and non-batched inputs
        return self.forward(latent_representation)

    def predict_node_features(self, latent_representation: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
        """
        Predict node features from the latent representation.

        Args:
            latent_representation: Latent representation tensor from encoder [latent_dim] or [batch_size, latent_dim]
            num_nodes: Optional number of nodes to generate (defaults to self.num_nodes)

        Returns:
            Predicted node features tensor [num_nodes, output_dim] or [batch_size, num_nodes, output_dim]
        """
        # Use specified number of nodes or default
        num_nodes = num_nodes if num_nodes is not None else self.num_nodes
        device = latent_representation.device

        # Check if input has batch dimension
        has_batch_dim = latent_representation.dim() > 1

        if has_batch_dim:
            # Handle batched input
            batch_size = latent_representation.size(0)

            # Reshape latent representation to [batch_size, latent_dim]
            z = latent_representation.view(batch_size, -1)

            # Apply MLP to transform latent to initial node features
            # [batch_size, num_nodes * hidden_dim]
            node_features = self.latent_to_nodes(z)

            # Reshape to [batch_size, num_nodes, hidden_dim]
            node_features = node_features.view(
                batch_size, num_nodes, self.hidden_dim)

            return node_features
        else:
            # Handle single graph input
            # Apply MLP to transform latent to initial node features
            node_features = self.latent_to_nodes(
                latent_representation)  # [num_nodes * hidden_dim]

            # Reshape to [num_nodes, hidden_dim]
            node_features = node_features.view(num_nodes, self.hidden_dim)

            return node_features

    def refine_node_features(self, node_features: torch.Tensor, batch=None) -> torch.Tensor:
        """
        Refine node features using GAT layers.

        Args:
            node_features: Node features tensor of shape [num_nodes, feature_dim]
                           or [batch_size, num_nodes, feature_dim]
            batch: Optional batch tensor (unused in the updated implementation)

        Returns:
            Refined node features
        """
        device = node_features.device

        # Check if the input has a batch dimension
        has_batch_dim = node_features.dim() > 2

        if has_batch_dim:
            # Handle batched data
            batch_size = node_features.shape[0]
            num_nodes_per_graph = node_features.shape[1]

            # Reshape from [batch_size, num_nodes, hidden_dim] to [batch_size * num_nodes, hidden_dim]
            x = node_features.reshape(-1, node_features.shape[-1])

            # Create edge indices for the batched graphs
            edge_indices = []
            total_nodes = 0

            # Process each graph in the batch
            for i in range(batch_size):
                # Create fully connected edge index for this graph
                graph_edge_index = self._create_fully_connected_edge_index(
                    num_nodes_per_graph, device)

                # Add offset to edge indices
                graph_edge_index = graph_edge_index + (i * num_nodes_per_graph)
                edge_indices.append(graph_edge_index)

            # Combine edge indices from all graphs
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            # Single graph case
            x = node_features
            num_nodes = node_features.size(0)
            edge_index = self._create_fully_connected_edge_index(
                num_nodes, device)

        # Apply GAT layers
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x = conv(x, edge_index)

            # Apply activation (except for last layer)
            if i < len(self.convs) - 1:
                x = self.activation(x)

                # Apply batch normalization if enabled
                if self.use_batch_norm and self.batch_norms is not None:
                    x = self.batch_norms[i](x)

                # Apply dropout
                if hasattr(self, 'dropout'):
                    x = self.dropout(x)

        # If output dimension doesn't match the expected dimension, project it
        current_dim = x.size(-1)
        if current_dim != self.output_dim and hasattr(self, 'output_projection'):
            x = self.output_projection(x)

        # If batched, reshape back to [batch_size, num_nodes, feature_dim]
        if has_batch_dim:
            x = x.view(batch_size, num_nodes_per_graph, -1)

        return x

    def predict_edges(self, node_embeddings: torch.Tensor, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges from node embeddings.

        Args:
            node_embeddings: Node embeddings tensor [num_nodes, feature_dim] 
                            or [batch_size, num_nodes, feature_dim]
            batch: Optional batch vector (unused in the updated implementation)

        Returns:
            Tuple of (edge_index, edge_attr)
        """
        device = node_embeddings.device

        # Check if input has a batch dimension
        has_batch_dim = node_embeddings.dim() > 2

        if has_batch_dim:
            # Process batched input
            batch_size = node_embeddings.shape[0]
            edge_indices = []
            edge_attrs = []

            # Process each graph in the batch
            for i in range(batch_size):
                # Extract node embeddings for this graph
                graph_embeddings = node_embeddings[i]

                # Predict edges for this graph
                e_idx, e_attr = self._predict_edges_single_graph(
                    graph_embeddings)

                # Add offset to edge indices to account for batching in the combined output
                if i > 0:
                    e_idx = e_idx + (i * graph_embeddings.size(0))

                # Store results
                edge_indices.append(e_idx)
                edge_attrs.append(e_attr)

            # Combine results from all graphs
            if edge_indices:
                edge_index = torch.cat(edge_indices, dim=1)
                edge_attr = torch.cat(edge_attrs, dim=0)
                return edge_index, edge_attr
            else:
                # Return empty tensors if no edges were predicted
                return (torch.zeros((2, 0), dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.float, device=device))
        else:
            # Non-batched prediction (single graph)
            return self._predict_edges_single_graph(node_embeddings)

    def _predict_edges_single_graph(self, node_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict edges for a single graph.

        Args:
            node_embeddings: Node embeddings tensor [num_nodes, feature_dim]

        Returns:
            Tuple of (edge_index, edge_attr)
        """
        num_nodes = node_embeddings.size(0)
        feature_dim = node_embeddings.size(1)
        device = node_embeddings.device

        # Create all possible edges (fully connected)
        source_nodes = torch.arange(num_nodes, device=device)
        source_nodes = source_nodes.repeat_interleave(num_nodes)
        target_nodes = torch.arange(num_nodes, device=device)
        target_nodes = target_nodes.repeat(num_nodes)

        # Remove self-loops
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]

        # Create edge index
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # Predict edge probabilities based on method
        if self.edge_prediction_method == "inner_product":
            # Simple inner product
            src_embeddings = node_embeddings[source_nodes]
            dst_embeddings = node_embeddings[target_nodes]
            edge_attr = torch.sum(src_embeddings * dst_embeddings, dim=1)
            edge_attr = torch.sigmoid(edge_attr)
            return edge_index, edge_attr

        elif self.edge_prediction_method == "mlp":
            # Get source and target embeddings
            src_embeddings = node_embeddings[source_nodes]
            dst_embeddings = node_embeddings[target_nodes]

            # Concatenate embeddings
            edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
            concat_dim = edge_features.size(1)

            # Check if our edge predictor is compatible with the feature dimension
            expected_dim = self.output_dim * 2

            if hasattr(self, 'edge_predictor') and isinstance(self.edge_predictor, nn.Sequential):
                first_layer = [
                    m for m in self.edge_predictor.modules() if isinstance(m, nn.Linear)][0]
                if first_layer.in_features != concat_dim:
                    # Create a temporary MLP for this dimension
                    temp_predictor = nn.Sequential(
                        nn.Linear(concat_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, 1),
                        nn.Sigmoid()
                    ).to(device)
                    edge_attr = temp_predictor(edge_features).squeeze(-1)
                else:
                    # Use the regular edge predictor
                    edge_attr = self.edge_predictor(edge_features).squeeze(-1)
            else:
                # Create a temporary predictor if none exists
                temp_predictor = nn.Sequential(
                    nn.Linear(concat_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                ).to(device)
                edge_attr = temp_predictor(edge_features).squeeze(-1)

            return edge_index, edge_attr

        elif self.edge_prediction_method == "bilinear":
            # Bilinear prediction
            # Extract node embeddings for source and target nodes
            # [num_edges, feature_dim]
            src_embeddings = node_embeddings[source_nodes]
            # [num_edges, feature_dim]
            dst_embeddings = node_embeddings[target_nodes]

            # Check if we have a bilinear layer and if its dimensions match
            has_compatible_bilinear = (
                hasattr(self, 'edge_predictor') and
                isinstance(self.edge_predictor, nn.Bilinear) and
                self.edge_predictor.in1_features == feature_dim and
                self.edge_predictor.in2_features == feature_dim
            )

            if has_compatible_bilinear:
                # Use existing bilinear layer
                edge_attr = self.edge_predictor(
                    src_embeddings, dst_embeddings).squeeze(-1)
            else:
                # Create a temporary bilinear layer with correct dimensions
                tmp_bilinear = nn.Bilinear(
                    feature_dim, feature_dim, 1,
                    bias=hasattr(self, 'edge_predictor') and
                    isinstance(self.edge_predictor, nn.Bilinear) and
                    self.edge_predictor.bias is not None
                ).to(device)
                edge_attr = tmp_bilinear(
                    src_embeddings, dst_embeddings).squeeze(-1)

            # Apply sigmoid activation
            edge_attr = torch.sigmoid(edge_attr)
            return edge_index, edge_attr

        elif self.edge_prediction_method == "attention":
            # Get source and target embeddings
            src_embeddings = node_embeddings[source_nodes]
            dst_embeddings = node_embeddings[target_nodes]

            # Concatenate embeddings
            edge_features = torch.cat([src_embeddings, dst_embeddings], dim=1)
            concat_dim = edge_features.size(1)

            # Check if we have a compatible attention layer
            has_compatible_attention = False
            if hasattr(self, 'edge_attention') and isinstance(self.edge_attention, nn.Sequential):
                first_layer = [
                    m for m in self.edge_attention.modules() if isinstance(m, nn.Linear)][0]
                has_compatible_attention = first_layer.in_features == concat_dim

            if has_compatible_attention:
                # Use the existing attention mechanism
                edge_attr = self.edge_attention(edge_features).squeeze(-1)
            else:
                # Create a temporary attention mechanism
                temp_attention = nn.Sequential(
                    nn.Linear(concat_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid()
                ).to(device)
                edge_attr = temp_attention(edge_features).squeeze(-1)

            return edge_index, edge_attr

        else:
            # Default to inner product
            src_embeddings = node_embeddings[source_nodes]
            dst_embeddings = node_embeddings[target_nodes]
            edge_attr = torch.sum(src_embeddings * dst_embeddings, dim=1)
            edge_attr = torch.sigmoid(edge_attr)
            return edge_index, edge_attr

    def postprocess_graph(self, graph_data: Data) -> Data:
        """
        Postprocess the reconstructed graph.

        This method applies thresholding to the edge predictions and removes low-probability edges.
        It also ensures the correct node feature dimension.

        Args:
            graph_data: Reconstructed PyTorch Geometric graph data

        Returns:
            Postprocessed graph data
        """
        # Extract components
        edge_index = graph_data.edge_index
        edge_attr = graph_data.edge_attr
        node_features = graph_data.x

        # Ensure node_features have the correct output dimension
        node_feature_dim = node_features.size(-1)
        if node_feature_dim != self.output_dim:
            # Create a projection layer if needed
            projection = nn.Linear(
                node_feature_dim, self.output_dim, device=node_features.device)
            # Project the node features
            node_features = projection(node_features)

        # Threshold the edges
        mask = edge_attr >= self.threshold_value
        thresholded_edge_index = edge_index[:, mask]
        thresholded_edge_attr = edge_attr[mask]

        # Create new graph with thresholded edges and correct feature dimension
        processed_graph = Data(
            x=node_features,
            edge_index=thresholded_edge_index,
            edge_attr=thresholded_edge_attr
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
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "edge_prediction_method": self.edge_prediction_method,
            "threshold": self.threshold_value,
            "batch_norm": self.use_batch_norm,
            "heads": self.heads,
            "concat_heads": self.concat_heads
        })
        return config

    def _create_fully_connected_edge_index(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """
        Create a fully connected edge index for message passing.

        Args:
            num_nodes: Number of nodes in the graph
            device: Device to create the tensor on

        Returns:
            Edge index tensor [2, num_edges] for a fully connected graph
        """
        # Create all possible edges (fully connected)
        source_nodes = torch.arange(num_nodes, device=device)
        source_nodes = source_nodes.repeat_interleave(num_nodes)
        target_nodes = torch.arange(num_nodes, device=device)
        target_nodes = target_nodes.repeat(num_nodes)

        # Remove self-loops for message passing
        mask = source_nodes != target_nodes
        source_nodes = source_nodes[mask]
        target_nodes = target_nodes[mask]

        # Create edge index
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        return edge_index
