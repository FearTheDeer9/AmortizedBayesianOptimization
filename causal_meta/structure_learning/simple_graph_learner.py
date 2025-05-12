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
        sparsity_weight: float = 0.001,
        acyclicity_weight: float = 1.0,
        pos_weight: float = 5.0,
        consistency_weight: float = 0.1,
        edge_prob_bias: float = 0.0,
        expected_density: Optional[float] = None,
        density_weight: float = 0.1,
        intervention_weight=10.0
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
        self.edge_prob_bias = torch.tensor(edge_prob_bias, dtype=torch.float32)  
        self.expected_density = expected_density
        self.density_weight = density_weight
        self.intervention_weight = intervention_weight
        
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

        intervention_predictor_layers = []
        intervention_predictor_layers.append(nn.Linear(input_dim * 2, hidden_dim))
        intervention_predictor_layers.append(nn.ReLU())
        intervention_predictor_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            intervention_predictor_layers.append(nn.Linear(hidden_dim, hidden_dim))
            intervention_predictor_layers.append(nn.ReLU())
            intervention_predictor_layers.append(nn.Dropout(dropout))
        
        intervention_predictor_layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.intervention_predictor = nn.Sequential(*intervention_predictor_layers)
        
    def _init_weights(self):
        """Initialize weights with better defaults for edge detection."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize final layer with random small biases to break symmetry
        nn.init.uniform_(self.final_layer.bias, -0.1, 0.1)
    
    def reset_edge_weights(self, epoch=None, print_every=20):
        """Reset edge prediction weights to break out of local minima. Only print every N epochs if epoch is provided."""
        old_weights = self.final_layer.weight.data.clone()
        old_bias = self.final_layer.bias.data.clone()
        nn.init.xavier_normal_(self.final_layer.weight)
        nn.init.uniform_(self.final_layer.bias, -0.2, 0.2)
        self.final_layer.weight.data = 0.8 * old_weights + 0.2 * self.final_layer.weight.data
        self.final_layer.bias.data = 0.8 * old_bias + 0.2 * self.final_layer.bias.data
        if epoch is not None and epoch % print_every == 0:
            print("Edge prediction weights partially reset to break symmetry")
    
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
            # Anti-clumping: stronger scaling and bias
            edge_logits_scaled = edge_logits * 5.0  # Increased from 3.0
            noise = torch.randn_like(edge_logits) * 0.1
            edge_probs = torch.sigmoid(edge_logits_scaled + noise + self.edge_prob_bias + 0.1)  # Slightly increased bias
            n_variables = edge_probs.shape[0]
            diag_mask = torch.eye(n_variables, device=edge_probs.device)
            edge_probs = edge_probs * (1 - diag_mask)
            return edge_probs
        
    # Add this new method for debugging
    def predict_with_forced_edges(self, 
                                 data: torch.Tensor,
                                 intervention_mask: Optional[torch.Tensor] = None,
                                 true_adj: Optional[torch.Tensor] = None,
                                 threshold: float = 0.3) -> torch.Tensor:
        """Make predictions with forced edges for debugging purposes."""
        with torch.no_grad():  # Important - don't track gradients here
            # Get normal predictions
            edge_probs = self.forward(data, intervention_mask)
            
            # Create a new tensor for modified probs (avoid in-place changes)
            modified_probs = edge_probs.clone()
            
            # Force specific edges if true_adj is provided
            if true_adj is not None:
                # Force true edges to have high probability
                for i in range(true_adj.shape[0]):
                    for j in range(true_adj.shape[1]):
                        if true_adj[i, j] > 0:
                            modified_probs[i, j] = 0.99
            
            # Apply thresholding
            adj_matrix = (modified_probs > threshold).float()
            adj_matrix.fill_diagonal_(0)  # No self-loops
            
            return adj_matrix
        
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
        mask: Optional[torch.Tensor] = None,
        pre_data=None,
        intervention_mask=None,
        post_data=None,
        epoch=0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss for the model with priority on intervention-based learning.
        
        Args:
            edge_probs: Predicted edge probabilities or logits
            target: Target adjacency matrix (optional, gradually phase out)
            mask: Mask for valid edges (optional)
            pre_data: Pre-intervention data (for intervention learning)
            intervention_mask: Intervention mask (for intervention learning)
            post_data: Post-intervention data (for intervention learning)
            epoch: Current training epoch
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        loss_components = {}
        
        # Get edge probabilities for regularization terms
        edge_probabilities = edge_probs
        if torch.max(edge_probs) > 1.0 or torch.min(edge_probs) < 0.0:
            edge_probabilities = torch.sigmoid(edge_probs)
        
        # Add direct structure learning when intervention data is available
        if pre_data is not None and intervention_mask is not None and post_data is not None:
            direct_edges = self.derive_structure_from_interventions(pre_data, intervention_mask, post_data)
            structure_loss = F.binary_cross_entropy(
                edge_probabilities, direct_edges, reduction='mean')
            loss_components['direct_structure'] = structure_loss * 15.0
        else:
            structure_loss = torch.tensor(0.0, device=edge_probs.device)
            loss_components['direct_structure'] = structure_loss
        
        # Intervention prediction loss (Primary learning signal)
        has_intervention_data = (pre_data is not None and intervention_mask is not None and post_data is not None)
        
        if has_intervention_data:
            intervention_loss = self.calculate_intervention_loss(
                pre_data, intervention_mask, post_data, edge_probabilities)
            loss_components['intervention'] = intervention_loss * self.intervention_weight
        else:
            intervention_loss = torch.tensor(0.0, device=edge_probs.device)
            loss_components['intervention'] = intervention_loss
        
        # Supervised loss if target is provided (gradually phase this out)
        supervised_loss = torch.tensor(0.0, device=edge_probs.device)
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
                
                # Calculate weight based on class imbalance
                if n_pos > 0:
                    pos_weight = torch.tensor(self.pos_weight * (n_neg / n_pos), device=target.device)
                else:
                    pos_weight = torch.tensor(self.pos_weight * 10.0, device=target.device)
            
            if mask is not None:
                # Apply mask to only consider valid edges
                edge_probs_masked = edge_probs[mask]
                target_masked = target[mask]
                
                # If edge_probs are probabilities, convert to logits
                if torch.max(edge_probs) <= 1.0 and torch.min(edge_probs) >= 0.0:
                    edge_probs_masked = torch.log(edge_probs_masked / (1 - edge_probs_masked + 1e-7) + 1e-7)
                
                supervised_loss = F.binary_cross_entropy_with_logits(
                    edge_probs_masked, target_masked, pos_weight=pos_weight
                )
            else:
                # Full matrix case
                if torch.max(edge_probs) <= 1.0 and torch.min(edge_probs) >= 0.0:
                    edge_probs_logits = torch.log(edge_probs / (1 - edge_probs + 1e-7) + 1e-7)
                else:
                    edge_probs_logits = edge_probs
                    
                supervised_loss = F.binary_cross_entropy_with_logits(
                    edge_probs_logits, target, pos_weight=pos_weight
                )
            
            loss_components['supervised'] = supervised_loss
        
        # Keep existing regularization losses
        acyclicity_reg = self.calculate_acyclicity_regularization(edge_probabilities)
        sparsity_reg = self.calculate_sparsity_regularization(edge_probabilities)
        # Progressive sparsity penalty: increases with epoch
        progressive_sparsity = sparsity_reg * (self.sparsity_weight * (1.5 + 0.01 * epoch))
        loss_components['sparsity'] = progressive_sparsity
        consistency_reg = self.calculate_consistency_regularization(edge_probabilities)
        density_reg = self.calculate_density_regularization(edge_probabilities)
        n = edge_probs.shape[0]
        off_diag_mask = 1.0 - torch.eye(n, device=edge_probs.device)
        mean_edge_prob = (edge_probs * off_diag_mask).sum() / (n * (n-1))
        # Stronger anti-sparsity regularization
        anti_sparsity_reg = -torch.log(mean_edge_prob + 1e-6) * 1.0
        loss_components['anti_sparsity'] = anti_sparsity_reg
        
        loss_components['acyclicity'] = acyclicity_reg * self.acyclicity_weight
        loss_components['consistency'] = consistency_reg * self.consistency_weight
        
        if self.expected_density is not None:
            loss_components['density'] = density_reg * self.density_weight
        
        # Total loss calculation prioritizing intervention learning
        if has_intervention_data:
            # When we have intervention data, use it as the primary learning signal
            total_loss = loss_components['intervention']
            
            # Add a gradually decreasing weight to supervised loss if provided
            # This helps transition from supervised to intervention-based learning
            if target is not None:
                # Start with some weight on supervised loss, but decrease over time
                supervision_weight = 0.1  # Much lower than before
                total_loss += supervised_loss * supervision_weight
        else:
            # If no intervention data yet, fall back to supervised loss if available
            if target is not None:
                total_loss = supervised_loss
            else:
                # If neither is available, just use regularization losses
                total_loss = torch.tensor(0.0, device=edge_probs.device)
        
        # Add regularization terms
        total_loss += loss_components['acyclicity'] + loss_components['sparsity']
        if self.consistency_weight > 0:
            total_loss += loss_components['consistency']
        if self.expected_density is not None:
            total_loss += loss_components['density']
        
        total_loss += anti_sparsity_reg  # Add the anti-sparsity term
        total_loss += loss_components['direct_structure']
        
        edge_count = edge_probabilities.sum()
        expected_edges = n * self.expected_density if self.expected_density else n * 0.3
        if edge_count > expected_edges:
            fp_penalty = (edge_count - expected_edges) * 0.2  # Increased penalty
            loss_components['fp_penalty'] = fp_penalty
        else:
            loss_components['fp_penalty'] = torch.tensor(0.0, device=edge_probs.device)
        
        # Penalty for false positives based on intervention data
        if has_intervention_data:
            fp_matrix = (edge_probabilities > 0.5) & (direct_edges < 0.5)
            fp_intervention_penalty = fp_matrix.float().sum() * 0.2
            loss_components['fp_intervention_penalty'] = fp_intervention_penalty
        else:
            loss_components['fp_intervention_penalty'] = torch.tensor(0.0, device=edge_probs.device)
        
        total_loss += loss_components['fp_penalty'] + loss_components['fp_intervention_penalty']
        
        loss_components['total'] = total_loss
        
        return total_loss, loss_components
        
    def threshold_edge_probabilities(self, edge_probs, threshold=0.5, epoch=None, print_every=20):
        """Simplified thresholding with minimal output. Only print every N epochs if epoch is provided."""
        if torch.max(edge_probs) > 1.0 or torch.min(edge_probs) < 0.0:
            edge_probs = torch.sigmoid(edge_probs)
        if epoch is not None and epoch % print_every == 0:
            print(f"Edge probability range: [{edge_probs.min().item():.4f}, {edge_probs.max().item():.4f}], mean: {edge_probs.mean().item():.4f}")
            print(f"Edges detected at threshold {threshold}: {adj_matrix.sum().item()}")
        adj_matrix = (edge_probs > threshold).float()
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
    
    def predict_intervention_outcomes(self, data, intervention_mask, edge_probs):
        """
        Predict outcomes after interventions based on current graph structure
        
        Args:
            data: Input data tensor [batch_size, n_variables]
            intervention_mask: Binary mask for interventions [batch_size, n_variables]
            edge_probs: Current graph edge probabilities [n_variables, n_variables]
            
        Returns:
            Predicted post-intervention values [batch_size, n_variables]
        """
        # Guard against None inputs
        if data is None or intervention_mask is None:
            return None
            
        # Combine data and intervention info as input
        batch_size = data.shape[0]
        intervention_info = intervention_mask * data  # Values at intervention points
        combined_input = torch.cat([data, intervention_info], dim=1)
        
        # Predict the post-intervention values
        predictions = self.intervention_predictor(combined_input)
        
        # Where interventions occurred, use the intervention values
        # For non-intervention values, use the predictions
        intervention_values = intervention_mask * data
        non_intervention_mask = 1.0 - intervention_mask
        
        return intervention_values + predictions * non_intervention_mask
    
    def calculate_intervention_loss(self, pre_data, intervention_mask, post_data, edge_probs):
        """
        Calculate loss based on intervention prediction accuracy
        
        Args:
            pre_data: Pre-intervention data [batch_size, n_variables]
            intervention_mask: Intervention mask [batch_size, n_variables]
            post_data: Post-intervention data [batch_size, n_variables]
            edge_probs: Current graph edge probabilities [n_variables, n_variables]
            
        Returns:
            Intervention prediction loss
        """
        # Guard against None inputs
        if pre_data is None or intervention_mask is None or post_data is None:
            return torch.tensor(0.0, device=edge_probs.device)
            
        # Predict post-intervention outcomes
        predicted_outcomes = self.predict_intervention_outcomes(
            pre_data, intervention_mask, edge_probs)
        
        # Calculate prediction error (MSE) on non-intervened variables
        non_intervention_mask = 1.0 - intervention_mask
        squared_errors = ((predicted_outcomes - post_data) * non_intervention_mask) ** 2
        
        # Sum over variables, mean over batch
        loss = torch.sum(squared_errors, dim=1).mean()
        
        return loss

    def derive_structure_from_interventions(self, pre_data, intervention_mask, post_data):
        """Enhanced direct structure learning from interventions: uses Cohen's d and higher threshold/sigmoid scaling."""
        n_vars = pre_data.shape[1]
        edge_evidence = torch.zeros((n_vars, n_vars), device=pre_data.device)
        effect_threshold = 0.15  # More selective
        for i in range(n_vars):
            intervention_samples = (intervention_mask[:, i] > 0).nonzero(as_tuple=True)[0]
            if len(intervention_samples) == 0:
                continue
            pre_values = pre_data[intervention_samples]
            post_values = post_data[intervention_samples]
            # Cohen's d: mean difference divided by pooled std
            mean_diff = (post_values - pre_values).mean(dim=0)
            std_pre = pre_values.std(dim=0) + 1e-6
            std_post = post_values.std(dim=0) + 1e-6
            pooled_std = torch.sqrt((std_pre ** 2 + std_post ** 2) / 2)
            effect_size = torch.abs(mean_diff / pooled_std)
            effect_size[i] = 0.0
            for j in range(n_vars):
                if j != i and effect_size[j] > effect_threshold:
                    edge_evidence[i, j] += effect_size[j]
        if edge_evidence.max() > 0:
            edge_evidence = edge_evidence / edge_evidence.max()
            return torch.sigmoid(edge_evidence * 12.0)  # Sharper separation
        else:
            return 0.1 * torch.ones((n_vars, n_vars), device=pre_data.device)