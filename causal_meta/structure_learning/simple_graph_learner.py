"""
Simple MLP-based neural network for causal graph structure learning.

This module implements a simple Multi-Layer Perceptron (MLP) based model for inferring
causal graph structures from observational and interventional data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, Any, List

import networkx as nx

from causal_meta.graph import CausalGraph


class SimpleGraphLearner(nn.Module):
    """
    Simple MLP-based model for causal structure learning.
    
    This model uses Multi-Layer Perceptrons (MLPs) to process observational and
    interventional data and predict causal graph structure in the form of an
    adjacency matrix. It explicitly handles intervention indicators to improve
    structure learning under interventions.
    
    Args:
        input_dim: Number of variables in the causal system
        hidden_dim: Dimension of hidden layers
        num_layers: Number of hidden layers in the MLPs
        dropout: Dropout probability for regularization
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity constraint
        pos_weight: Weight for positive examples in BCE loss (>1 values help with "no edge" bias)
        consistency_weight: Weight for consistency regularization to push probs toward 0 or 1
        edge_prob_bias: Bias for edge probability (higher values encourage more edges)
        expected_density: Expected edge density (if provided, adds regularization toward this density)
        density_weight: Weight for density regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0,
        pos_weight: float = 5.0,
        consistency_weight: float = 0.1,
        edge_prob_bias: float = 0.0,
        expected_density: Optional[float] = None,
        density_weight: float = 0.1
    ):
        """Initialize the SimpleGraphLearner."""
        super().__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.pos_weight = pos_weight
        self.consistency_weight = consistency_weight
        self.edge_prob_bias = edge_prob_bias
        self.expected_density = expected_density
        self.density_weight = density_weight
        
        # Node feature encoder: takes raw node values and extracts features
        node_encoder_layers = []
        node_encoder_layers.append(nn.Linear(1, hidden_dim))
        node_encoder_layers.append(nn.ReLU())
        node_encoder_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            node_encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            node_encoder_layers.append(nn.ReLU())
            node_encoder_layers.append(nn.Dropout(dropout))
            
        self.node_encoder = nn.Sequential(*node_encoder_layers)
        
        # Intervention handler: processes intervention indicators
        intervention_handler_layers = []
        intervention_handler_layers.append(nn.Linear(1, hidden_dim // 2))
        intervention_handler_layers.append(nn.ReLU())
        intervention_handler_layers.append(nn.Dropout(dropout))
        intervention_handler_layers.append(nn.Linear(hidden_dim // 2, hidden_dim // 2))
        intervention_handler_layers.append(nn.ReLU())
        
        self.intervention_handler = nn.Sequential(*intervention_handler_layers)
        
        # Combined feature dimension (node features + intervention features)
        combined_dim = hidden_dim + hidden_dim // 2
        
        # Edge predictor: predicts edge probability for each node pair
        edge_predictor_layers = []
        edge_predictor_layers.append(nn.Linear(combined_dim * 2, hidden_dim))
        edge_predictor_layers.append(nn.ReLU())
        edge_predictor_layers.append(nn.Dropout(dropout))
        
        # Final layer (sigmoid applied separately to ensure proper initialization of bias)
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        # Create sequential model for all layers except final
        self.edge_predictor = nn.Sequential(*edge_predictor_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with bias adjustment for edge probabilities."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize the bias of the final layer to encourage more edge predictions
        # A positive bias makes the initial predictions favor edges more
        if self.edge_prob_bias != 0.0:
            # Convert the desired probability bias to logit space
            logit_bias = np.log(self.edge_prob_bias / (1 - self.edge_prob_bias)) if self.edge_prob_bias > 0 else 0
            # Set the bias of the final layer
            nn.init.constant_(self.final_layer.bias, logit_bias)
    
    def forward(
        self,
        data: torch.Tensor,
        intervention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to predict edge probabilities.
        
        Args:
            data: Input data tensor of shape [batch_size, n_variables]
            intervention_mask: Binary mask for interventions of shape [batch_size, n_variables]
                1 indicates intervention, 0 indicates observational data
            return_logits: If True, returns raw logits instead of probabilities
                
        Returns:
            edge_probs: Tensor of edge probabilities (or logits) of shape [n_variables, n_variables]
        """
        # Get dimensions
        batch_size, n_variables = data.shape
        
        # Process each variable's data to get node features
        node_features = []
        for i in range(n_variables):
            # Extract variable data and reshape for the encoder
            var_data = data[:, i].view(-1, 1)
            # Encode variable data
            node_feat = self.node_encoder(var_data)
            # Average over batch dimension
            node_feat = torch.mean(node_feat, dim=0, keepdim=True)
            node_features.append(node_feat)
        
        # Stack node features
        node_features = torch.cat(node_features, dim=0)  # [n_variables, hidden_dim]
        
        # Process intervention mask if provided
        if intervention_mask is None:
            # If no intervention mask provided, create zero tensor
            intervention_features = torch.zeros(
                (n_variables, self.hidden_dim // 2), device=data.device
            )
        else:
            # Process intervention data for each variable
            intervention_features = []
            for i in range(n_variables):
                # Extract intervention indicator and reshape
                int_indicator = intervention_mask[:, i].view(-1, 1)
                # Process intervention indicator
                int_feat = self.intervention_handler(int_indicator)
                # Average over batch dimension
                int_feat = torch.mean(int_feat, dim=0, keepdim=True)
                intervention_features.append(int_feat)
            
            # Stack intervention features
            intervention_features = torch.cat(intervention_features, dim=0)  # [n_variables, hidden_dim//2]
        
        # Combine node features with intervention features
        combined_features = torch.cat(
            [node_features, intervention_features], dim=1
        )  # [n_variables, hidden_dim + hidden_dim//2]
        
        # Predict edge probabilities for each node pair
        edge_logits = torch.zeros((n_variables, n_variables), device=data.device)
        
        for i in range(n_variables):
            for j in range(n_variables):
                if i == j:
                    # No self-loops
                    continue
                
                # Concatenate features of node i and node j
                node_pair_features = torch.cat(
                    [combined_features[i], combined_features[j]], dim=0
                )
                
                # Pass through edge predictor
                edge_features = self.edge_predictor(node_pair_features.unsqueeze(0))
                
                # Pass through final layer to get logits
                edge_logit = self.final_layer(edge_features)
                
                # Store raw logit
                edge_logits[i, j] = edge_logit.squeeze()
        
        # Return logits or probabilities
        if return_logits:
            return edge_logits
        else:
            # Apply sigmoid to get probabilities
            edge_probs = torch.sigmoid(edge_logits)
            return edge_probs
    
    def calculate_acyclicity_regularization(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate acyclicity regularization using the matrix exponential.
        
        Args:
            edge_probs: Tensor of edge probabilities
            
        Returns:
            Acyclicity regularization loss term
        """
        # Compute h(A) = tr(e^(A \circ A)) - d, where A is the adjacency matrix,
        # \circ is Hadamard product, d is number of nodes
        # This should be close to 0 for an acyclic graph
        
        n = edge_probs.shape[0]
        
        # Compute A \circ A
        adj_squared = edge_probs * edge_probs
        
        # Compute matrix exponential
        matrix_exp = torch.matrix_exp(adj_squared)
        
        # Compute trace
        trace = torch.trace(matrix_exp)
        
        # Acyclicity penalty
        acyclicity = trace - n
        
        return acyclicity
    
    def calculate_sparsity_regularization(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate sparsity regularization using L1 norm.
        
        Args:
            edge_probs: Tensor of edge probabilities
            
        Returns:
            Sparsity regularization loss term
        """
        # L1 regularization to promote sparsity
        return torch.sum(torch.abs(edge_probs))
    
    def calculate_consistency_regularization(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate consistency regularization to push probabilities toward 0 or 1.
        
        Args:
            edge_probs: Tensor of edge probabilities
            
        Returns:
            Consistency regularization loss term
        """
        # This creates a U-shaped penalty with minimum at 0 and 1, maximum at 0.5
        # It encourages the model to make more confident predictions
        return -torch.sum((edge_probs - 0.5).pow(2))
    
    def calculate_density_regularization(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate density regularization to encourage a specific graph density.
        
        Args:
            edge_probs: Tensor of edge probabilities
            
        Returns:
            Density regularization loss term
        """
        if self.expected_density is None:
            return torch.tensor(0.0, device=edge_probs.device)
        
        # Calculate current density (excluding diagonal)
        n = edge_probs.shape[0]
        mask = 1.0 - torch.eye(n, device=edge_probs.device)
        current_density = (edge_probs * mask).sum() / (n * (n - 1))
        
        # Penalize deviation from expected density
        return torch.abs(current_density - self.expected_density)
    
    def calculate_loss(
        self,
        edge_probs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss for the model.
        
        Args:
            edge_probs: Predicted edge probabilities or logits
            target: Target adjacency matrix (optional)
            mask: Mask for valid edges (optional)
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        loss_components = {}
        
        # Get edge probabilities for regularization terms (if edge_probs are logits)
        edge_probabilities = edge_probs
        if torch.max(edge_probs) > 1.0 or torch.min(edge_probs) < 0.0:
            # These are logits, convert to probabilities for regularization
            edge_probabilities = torch.sigmoid(edge_probs)
        
        # Supervised loss if target is provided
        if target is not None:
            # Create positive weight tensor for BCE loss if needed
            pos_weight = None
            if self.pos_weight != 1.0:
                if mask is not None:
                    target_masked = target[mask]
                    n_pos = target_masked.sum()
                    n_neg = target_masked.numel() - n_pos
                else:
                    n_pos = target.sum()
                    n_neg = target.numel() - n_pos
                
                # Calculate weight based on class imbalance, ensuring we don't divide by zero
                if n_pos > 0:
                    # The weight will be self.pos_weight * (n_neg / n_pos)
                    pos_weight = torch.tensor(self.pos_weight * (n_neg / n_pos), device=target.device)
                else:
                    # If no positive examples, set a default high weight
                    pos_weight = torch.tensor(self.pos_weight * 10.0, device=target.device)
            
            if mask is not None:
                # Apply mask to only consider valid edges
                edge_probs_masked = edge_probs[mask]
                target_masked = target[mask]
                
                # If edge_probs are probabilities, convert to logits
                if torch.max(edge_probs) <= 1.0 and torch.min(edge_probs) >= 0.0:
                    # Convert probabilities to logits
                    edge_probs_masked = torch.log(edge_probs_masked / (1 - edge_probs_masked + 1e-7) + 1e-7)
                
                supervised_loss = F.binary_cross_entropy_with_logits(
                    edge_probs_masked, target_masked, pos_weight=pos_weight
                )
            else:
                # For the full matrix case
                # If edge_probs are probabilities, convert to logits
                if torch.max(edge_probs) <= 1.0 and torch.min(edge_probs) >= 0.0:
                    # Convert probabilities to logits
                    edge_probs = torch.log(edge_probs / (1 - edge_probs + 1e-7) + 1e-7)
                
                supervised_loss = F.binary_cross_entropy_with_logits(
                    edge_probs, target, pos_weight=pos_weight
                )
            
            loss_components['supervised'] = supervised_loss
        else:
            supervised_loss = 0.0
        
        # Regularization losses
        acyclicity_reg = self.calculate_acyclicity_regularization(edge_probabilities)
        sparsity_reg = self.calculate_sparsity_regularization(edge_probabilities)
        consistency_reg = self.calculate_consistency_regularization(edge_probabilities)
        density_reg = self.calculate_density_regularization(edge_probabilities)
        
        loss_components['acyclicity'] = acyclicity_reg * self.acyclicity_weight
        loss_components['sparsity'] = sparsity_reg * self.sparsity_weight
        loss_components['consistency'] = consistency_reg * self.consistency_weight
        
        if self.expected_density is not None:
            loss_components['density'] = density_reg * self.density_weight
        
        # Total loss
        if target is not None:
            total_loss = supervised_loss + loss_components['acyclicity'] + loss_components['sparsity']
            if self.consistency_weight > 0:
                total_loss += loss_components['consistency']
            if self.expected_density is not None:
                total_loss += loss_components['density']
        else:
            # If no supervised loss, regularization becomes primary loss
            total_loss = loss_components['acyclicity'] + loss_components['sparsity']
            if self.consistency_weight > 0:
                total_loss += loss_components['consistency']
            if self.expected_density is not None:
                total_loss += loss_components['density']
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components
    
    def threshold_edge_probabilities(
        self,
        edge_probs: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Convert edge probabilities to binary adjacency matrix using a threshold.
        
        Args:
            edge_probs: Tensor of edge probabilities or logits
            threshold: Threshold value for edge inclusion
            
        Returns:
            Binary adjacency matrix
        """
        # Check if we need to convert logits to probabilities
        if torch.max(edge_probs) > 1.0 or torch.min(edge_probs) < 0.0:
            # These are logits, convert to probabilities
            edge_probs = torch.sigmoid(edge_probs)
            
        adj_matrix = (edge_probs > threshold).float()
        
        # Ensure no self-loops
        adj_matrix.fill_diagonal_(0)
        
        return adj_matrix
    
    def to_causal_graph(
        self,
        edge_probs: torch.Tensor,
        threshold: float = 0.5
    ) -> CausalGraph:
        """
        Convert edge probabilities to a CausalGraph object.
        
        Args:
            edge_probs: Tensor of edge probabilities
            threshold: Threshold value for edge inclusion
            
        Returns:
            CausalGraph instance
        """
        # Threshold the edge probabilities
        adj_matrix = self.threshold_edge_probabilities(edge_probs, threshold)
        
        # Convert to networkx DiGraph
        n_variables = adj_matrix.shape[0]
        nx_graph = nx.DiGraph()
        
        # Add nodes with proper names
        for i in range(n_variables):
            nx_graph.add_node(f'x{i}')
        
        # Add edges
        for i in range(n_variables):
            for j in range(n_variables):
                if adj_matrix[i, j] > 0:
                    nx_graph.add_edge(f'x{i}', f'x{j}')
        
        # Create CausalGraph from networkx DiGraph
        causal_graph = CausalGraph()
        
        # Add nodes to CausalGraph
        for node in nx_graph.nodes:
            causal_graph.add_node(node)
        
        # Add edges to CausalGraph
        for source, target in nx_graph.edges:
            causal_graph.add_edge(source, target)
        
        return causal_graph 