"""
Transformer-based causal structure inference model.

This module implements a Transformer-based model for inferring causal structure
from time series data. It leverages self-attention mechanisms to capture temporal
dependencies and infer causal relationships.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from causal_meta.inference.models.base_encoder import TransformerBaseEncoder
from causal_meta.graph.causal_graph import CausalGraph


class TransformerGraphEncoder(TransformerBaseEncoder):
    """
    Transformer-based model for causal structure learning.
    
    This model processes time series data using transformer blocks with self-attention
    and outputs a matrix of edge probabilities representing the inferred causal graph
    structure. The attention mechanism helps capture dependencies between variables.
    
    Args:
        input_dim: Number of variables in the time series
        hidden_dim: Dimension of hidden layers
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout probability for regularization
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity constraint
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0
    ):
        """Initialize the TransformerGraphEncoder."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight
        )
        
        # Linear embedding layer
        self.embedding = nn.Linear(1, hidden_dim)
        
        # Position encoding
        self.register_buffer(
            "position_encoding",
            self._generate_positional_encoding(max_seq_len=100, hidden_dim=hidden_dim)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            self._build_transformer_layer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Edge predictor (node pair features -> edge probability)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict edge probabilities.
        
        Args:
            x: Time series data tensor of shape [batch_size, seq_length, n_variables]
            
        Returns:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
        """
        # Get dimensions
        batch_size, seq_length, n_variables = x.shape
        device = x.device
        
        # Process each variable's time series with the transformer
        node_embeddings = []
        for i in range(n_variables):
            # Shape: [batch_size, seq_length, 1]
            var_i_values = x[:, :, i].unsqueeze(-1)
            
            # Embed the time series
            # Shape: [batch_size, seq_length, hidden_dim]
            embeddings = self.embedding(var_i_values)
            
            # Add positional encoding (truncate if sequence is longer than pre-computed positions)
            pos_enc = self.position_encoding[:seq_length, :]
            embeddings = embeddings + pos_enc.unsqueeze(0)
            
            # Apply transformer layers
            transformer_out = embeddings
            for layer in self.transformer_layers:
                transformer_out = layer(transformer_out)
            
            # Global pooling over sequence dimension to get node representation
            # Shape: [batch_size, hidden_dim]
            node_i_features = transformer_out.mean(dim=1)
            
            # Average over batch dimension
            # Shape: [hidden_dim]
            avg_emb = node_i_features.mean(dim=0)
            
            node_embeddings.append(avg_emb)
        
        # Stack node embeddings
        # Shape: [n_variables, hidden_dim]
        node_embeddings = torch.stack(node_embeddings)
        
        # Create pairwise features for all possible edges
        edge_probs = torch.zeros((n_variables, n_variables), device=device)
        
        # For each potential edge
        for i in range(n_variables):
            for j in range(n_variables):
                if i == j:  # No self-loops
                    continue
                
                # Create pair features (concatenate source and target node embeddings)
                # Shape: [hidden_dim * 2]
                pair_features = torch.cat([
                    node_embeddings[i],  # Source node
                    node_embeddings[j]   # Target node
                ])
                
                # Add batch dimension
                # Shape: [1, hidden_dim * 2]
                pair_features = pair_features.unsqueeze(0)
                
                # Apply edge predictor
                # Shape: [1, 1]
                edge_prob = self.edge_predictor(pair_features)
                
                # Set edge probability
                edge_probs[i, j] = edge_prob.item()
        
        # Store for uncertainty estimation
        self._last_edge_probs = edge_probs.detach().clone()
        
        return edge_probs
    
    def predict_graph_from_data(
        self,
        data: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, CausalGraph]:
        """
        Process time series data and predict a causal graph.
        
        Args:
            data: Time series data tensor of shape [batch_size, seq_length, n_variables]
            threshold: Threshold value for binarizing edge probabilities
            
        Returns:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
            graph: CausalGraph instance with nodes and edges
        """
        # Forward pass to get edge probabilities
        edge_probs = self(data)
        
        # Convert to CausalGraph
        graph = self.to_causal_graph(edge_probs, threshold=threshold)
        
        return edge_probs, graph
        
    def to_causal_graph(self, edge_probs: torch.Tensor, threshold: float = 0.5) -> CausalGraph:
        """
        Convert edge probabilities to a CausalGraph object.
        
        Args:
            edge_probs: Tensor of edge probabilities
            threshold: Threshold for binarizing edges
            
        Returns:
            CausalGraph object with nodes and edges
        """
        # Get dimensions
        n_variables = edge_probs.shape[0]
        
        # Create a new graph
        graph = CausalGraph()
        
        # Add nodes
        for i in range(n_variables):
            graph.add_node(str(i))
        
        # Create binary adjacency matrix based on threshold
        binary_adj = (edge_probs > threshold).cpu().numpy()
        
        # Add edges based on thresholded edge probabilities
        for i in range(n_variables):
            for j in range(n_variables):
                if i != j and binary_adj[i, j]:
                    graph.add_edge(str(i), str(j))
        
        return graph 