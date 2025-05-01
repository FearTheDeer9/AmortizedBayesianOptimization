"""
DynamicsDecoder for predicting intervention outcomes in causal graphs.

This module implements a neural network decoder that predicts the outcomes of
interventions on a causal system, given the graph structure and observational data.
The decoder uses a graph attention architecture to model the functional relationships
between variables and predict counterfactual outcomes under interventions.

Sequential Analysis:
- Problem: Predict intervention outcomes based on inferred graph structure
- Components: GNN with attention, intervention conditioning, uncertainty quantification
- Approach: Message passing with attention-based focus on relevant nodes
- Challenges: Differentiability, handling different intervention types, computational efficiency
- Plan: Build modular architecture with clear interfaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.utils import to_dense_adj
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class DynamicsDecoder(nn.Module):
    """
    Neural network module for predicting intervention outcomes in causal systems.
    
    This decoder uses a graph attention architecture to model functional relationships
    between variables in a causal system and predict counterfactual outcomes under
    different intervention scenarios.
    
    Args:
        input_dim: Dimensionality of node features
        hidden_dim: Size of hidden layers
        num_layers: Number of message passing layers
        dropout: Dropout probability for regularization
        uncertainty: Whether to provide uncertainty estimates
        num_ensembles: Number of ensemble models for uncertainty estimation
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        uncertainty: bool = False,
        num_ensembles: int = 5
    ):
        """Initialize the DynamicsDecoder."""
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.uncertainty = uncertainty
        self.num_ensembles = num_ensembles if uncertainty else 1
        
        # Initial node feature transformation
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Message passing layers with Graph Attention
        self.conv_layers = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 4,  # Divided by 4 because we use 4 heads
                heads=4,
                dropout=dropout,
                concat=True
            ) for _ in range(num_layers)
        ])
        
        # Skip connections for each layer
        self.skip_connections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Intervention conditioning mechanism
        self.intervention_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2: intervention value and mask
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers - one for each ensemble member if uncertainty is enabled
        if self.uncertainty:
            self.ensemble_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(num_ensembles)
            ])
        else:
            # Single output layer if no uncertainty
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        adj_matrices: torch.Tensor,
        interventions: Optional[Dict[str, torch.Tensor]] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the dynamics decoder.
        
        Args:
            x: Node features [num_nodes * batch_size, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes * batch_size]
            adj_matrices: Adjacency matrices from GraphEncoder [batch_size, num_nodes, num_nodes]
            interventions: Dictionary with 'targets' and 'values' for interventions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predicted outcomes, optionally with uncertainty estimates
        """
        batch_size = adj_matrices.size(0)
        num_nodes = adj_matrices.size(1)
        device = x.device
        
        # Initial node feature transformation
        h = self.node_encoder(x)
        
        # Apply intervention conditioning if provided
        if interventions is not None:
            h = self._process_interventions(h, interventions, batch, num_nodes)
        
        # Message passing through the graph with skip connections
        for i in range(self.num_layers):
            # Store original features for skip connection
            identity = h
            
            # Process through GAT layer
            h = self.conv_layers[i](h, edge_index)
            
            # Apply skip connection
            h = h + self.skip_connections[i](identity)
            
            # Apply layer normalization
            h = self.layer_norms[i](h)
            
            # Apply non-linearity and dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Generate predictions
        if self.uncertainty and return_uncertainty:
            # Generate predictions from each ensemble member
            ensemble_outputs = []
            for layer in self.ensemble_layers:
                output = layer(h)
                ensemble_outputs.append(output)
            
            # Compute mean prediction and uncertainty
            stacked_outputs = torch.stack(ensemble_outputs, dim=0)
            mean_prediction = torch.mean(stacked_outputs, dim=0)
            uncertainty = torch.std(stacked_outputs, dim=0)
            
            return mean_prediction, uncertainty
        
        elif self.uncertainty:
            # Just return mean prediction without uncertainty
            ensemble_outputs = []
            for layer in self.ensemble_layers:
                output = layer(h)
                ensemble_outputs.append(output)
            
            mean_prediction = torch.mean(torch.stack(ensemble_outputs, dim=0), dim=0)
            return mean_prediction
        
        else:
            # Single deterministic output
            outputs = self.output_layer(h)
            return outputs
    
    def predict_intervention_outcome(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        adj_matrices: torch.Tensor,
        intervention_targets: torch.Tensor,
        intervention_values: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convenience method to predict outcomes for specific interventions.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment for nodes
            adj_matrices: Adjacency matrices from GraphEncoder
            intervention_targets: Indices of nodes to intervene on
            intervention_values: Values to set for interventions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predicted outcomes, optionally with uncertainty estimates
        """
        interventions = {
            'targets': intervention_targets,
            'values': intervention_values
        }
        
        return self.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            adj_matrices=adj_matrices,
            interventions=interventions,
            return_uncertainty=return_uncertainty
        )
    
    def _process_interventions(
        self,
        x: torch.Tensor,
        interventions: Dict[str, torch.Tensor],
        batch: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Process intervention information and incorporate into node features.
        
        Args:
            x: Node features
            interventions: Dictionary with 'targets' and 'values'
            batch: Batch assignment for nodes
            num_nodes: Number of nodes per graph
            
        Returns:
            Updated node features incorporating intervention information
        """
        # Extract intervention information
        targets = interventions['targets']  # Shape: [num_interventions]
        values = interventions['values']    # Shape: [num_interventions]
        
        # Create intervention features tensor initialized with zeros
        intervention_features = torch.zeros_like(x[:, :1])
        
        # Create intervention mask tensor initialized with zeros
        intervention_mask = torch.zeros_like(x[:, :1])
        
        # Calculate global indices for intervention targets in the batch
        total_nodes = batch.size(0)
        
        # For each intervention target
        for i, target in enumerate(targets):
            # Find all instances of this target in the batch
            target_mask = (batch * num_nodes + target) == torch.arange(total_nodes, device=x.device).view(-1, 1)
            target_indices = torch.where(target_mask.any(dim=1))[0]
            
            # Set intervention values and masks for these indices
            if target_indices.numel() > 0:
                intervention_features[target_indices] = values[i]
                intervention_mask[target_indices] = 1.0
        
        # Concatenate intervention information
        intervention_info = torch.cat([intervention_features, intervention_mask], dim=1)
        
        # Encode intervention information
        encoded_intervention = self.intervention_encoder(intervention_info)
        
        # Combine with original features (additive approach)
        # Apply the intervention only where mask is 1
        intervention_applied = x + intervention_mask * encoded_intervention
        
        return intervention_applied
    
    def _compute_uncertainty(
        self,
        x: torch.Tensor,
        ensemble_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute uncertainty estimates from ensemble predictions.
        
        Args:
            x: Input features
            ensemble_outputs: List of outputs from ensemble models
            
        Returns:
            Uncertainty estimates for each prediction
        """
        # Stack ensemble outputs
        stacked_outputs = torch.stack(ensemble_outputs, dim=0)
        
        # Compute standard deviation across ensemble dimension
        uncertainty = torch.std(stacked_outputs, dim=0)
        
        return uncertainty 