"""
MLP-based causal structure inference model.

This module implements a Multi-Layer Perceptron (MLP) based model for inferring
causal structure from time series data. Unlike Graph Neural Network approaches,
this model uses standard fully-connected layers to process time series data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

from causal_meta.inference.models.base_encoder import MLPBaseEncoder
from causal_meta.graph.causal_graph import CausalGraph


class MLPGraphEncoder(MLPBaseEncoder):
    """
    MLP-based model for causal structure learning.
    
    This model processes time series data using MLPs and outputs a matrix of edge
    probabilities representing the inferred causal graph structure. It flattens
    time series data and uses fully connected layers to predict pairwise edge probabilities.
    
    Args:
        input_dim: Number of variables in the time series
        hidden_dim: Dimension of hidden layers
        num_layers: Number of MLP layers
        dropout: Dropout probability for regularization
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity constraint
        seq_length: Default sequence length
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0,
        seq_length: int = 10  # Default sequence length
    ):
        """Initialize the MLPGraphEncoder."""
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight
        )
        self.seq_length = seq_length
        
        # Node feature encoder (time series -> hidden representation)
        self.node_encoder = self._build_mlp(
            input_dim=seq_length,  # Sequence length is the input dimension
            hidden_dim=hidden_dim, 
            output_dim=hidden_dim,
            num_layers=2
        )
        
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
        
        # Update seq_length if different from initialization
        if seq_length != self.seq_length:
            self.seq_length = seq_length
            # Recreate node encoder with correct input dimension
            self.node_encoder = self._build_mlp(
                input_dim=seq_length,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                num_layers=2
            ).to(device)
            
        # Transpose to get shape [batch_size, n_variables, seq_length]
        x = x.transpose(1, 2)
        
        # Process each variable's time series to get node embeddings
        node_embeddings = []
        for i in range(n_variables):
            # Get time series for variable i
            # Shape: [batch_size, seq_length]
            var_series = x[:, i, :]
            
            # Process through node encoder
            # Shape: [batch_size, hidden_dim]
            node_emb = self.node_encoder(var_series)
            
            # Average across batch dimension
            # Shape: [hidden_dim]
            avg_emb = node_emb.mean(dim=0)
            
            node_embeddings.append(avg_emb)
        
        # Stack embeddings
        # Shape: [n_variables, hidden_dim]
        node_embeddings = torch.stack(node_embeddings)
        
        # Create edge probability matrix
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