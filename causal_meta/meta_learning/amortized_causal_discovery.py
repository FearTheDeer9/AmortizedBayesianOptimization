"""
AmortizedCausalDiscovery class for joint structure and dynamics inference.

This module implements the AmortizedCausalDiscovery class, which combines a GraphEncoder
for causal structure inference and a DynamicsDecoder for dynamics modeling. This unified
approach allows for end-to-end training and inference of both causal structure and
intervention effects.

Sequential Analysis:
- Problem: Combine structure and dynamics inference in a unified framework
- Components: GraphEncoder, DynamicsDecoder, joint training, data handling
- Approach: Integrate components with balanced losses and shared interfaces
- Challenges: Gradient flow, balancing objectives, computational efficiency
- Plan: Build modular architecture with comprehensive API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.meta_learning.dynamics_decoder import DynamicsDecoder
from causal_meta.graph.causal_graph import CausalGraph

# Import TaskEmbedding and MAML - add these imports
from causal_meta.meta_learning.meta_learning import TaskEmbedding, MAMLForCausalDiscovery


class AmortizedCausalDiscovery(nn.Module):
    """
    Unified framework for amortized causal discovery, combining structure inference and dynamics modeling.
    
    This class integrates a GraphEncoder for causal structure inference and a DynamicsDecoder
    for predicting intervention outcomes. It provides a joint training procedure and
    unified inference methods for both structure and dynamics.
    
    Args:
        hidden_dim: Dimension of hidden representations
        input_dim: Dimension of node features
        attention_heads: Number of attention heads for GraphEncoder
        num_layers: Number of network layers
        dropout: Dropout probability for regularization
        sparsity_weight: Weight for sparsity regularization in GraphEncoder
        acyclicity_weight: Weight for acyclicity constraint in GraphEncoder
        dynamics_weight: Weight for dynamics loss in joint training
        structure_weight: Weight for structure loss in joint training
        uncertainty: Whether to provide uncertainty estimates
        num_ensembles: Number of ensemble models for uncertainty estimation
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        input_dim: int = 3,
        attention_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0,
        dynamics_weight: float = 1.0,
        structure_weight: float = 1.0,
        uncertainty: bool = False,
        num_ensembles: int = 5
    ):
        """Initialize the AmortizedCausalDiscovery model."""
        super().__init__()
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.dynamics_weight = dynamics_weight
        self.structure_weight = structure_weight
        self.uncertainty = uncertainty
        self.num_ensembles = num_ensembles
        
        # Initialize GraphEncoder for structure inference
        self.graph_encoder = GraphEncoder(
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            num_layers=num_layers,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight
        )
        
        # Initialize DynamicsDecoder for intervention outcome prediction
        self.dynamics_decoder = DynamicsDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            uncertainty=uncertainty,
            num_ensembles=num_ensembles
        )
        
        # Node feature encoder to transform raw features for dynamics decoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        interventions: Optional[Dict[str, torch.Tensor]] = None,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the AmortizedCausalDiscovery model.
        
        Args:
            x: Time series data [batch_size, seq_length, n_variables]
            node_features: Node features [batch_size * n_variables, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes [batch_size * n_variables]
            interventions: Optional dictionary with 'targets' and 'values' for interventions
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing adjacency matrix and predictions
        """
        # Infer graph structure using GraphEncoder
        adjacency_matrix = self.graph_encoder(x)
        
        # Get dimensions
        batch_size = x.size(0)
        n_variables = x.size(2)
        
        # Expand adjacency matrix for batch processing
        # [n_variables, n_variables] -> [batch_size, n_variables, n_variables]
        adj_matrices = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process node features through dynamics decoder
        if return_uncertainty and self.uncertainty:
            predictions, uncertainty = self.dynamics_decoder(
                x=node_features,
                edge_index=edge_index,
                batch=batch,
                adj_matrices=adj_matrices,
                interventions=interventions,
                return_uncertainty=True
            )
            
            return {
                'adjacency_matrix': adjacency_matrix,
                'predictions': predictions,
                'uncertainty': uncertainty
            }
        else:
            predictions = self.dynamics_decoder(
                x=node_features,
                edge_index=edge_index,
                batch=batch,
                adj_matrices=adj_matrices,
                interventions=interventions,
                return_uncertainty=False
            )
            
            return {
                'adjacency_matrix': adjacency_matrix,
                'predictions': predictions
            }
            
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        true_adjacency: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the loss for training.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Target values for dynamics prediction
            true_adjacency: Optional ground truth adjacency matrix for supervised training
            
        Returns:
            Combined loss value
        """
        # Extract outputs
        adjacency_matrix = outputs['adjacency_matrix']
        predictions = outputs['predictions']
        
        # Compute dynamics loss
        dynamics_loss = F.mse_loss(predictions, targets)
        
        # Compute structure loss
        structure_loss = 0.0
        
        # If ground truth adjacency is provided, compute supervised loss
        if true_adjacency is not None:
            structure_loss += F.binary_cross_entropy(adjacency_matrix, true_adjacency)
        
        # Add regularization losses
        structure_loss += self.graph_encoder.get_sparsity_loss(adjacency_matrix)
        structure_loss += self.graph_encoder.get_acyclicity_loss(adjacency_matrix)
        
        # Combine losses
        total_loss = (
            self.dynamics_weight * dynamics_loss +
            self.structure_weight * structure_loss
        )
        
        return total_loss
            
    def train_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        true_adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            data_loader: DataLoader providing training data
            optimizer: Optimizer for parameter updates
            device: Device to run training on
            true_adjacency: Optional ground truth adjacency matrix
            
        Returns:
            Dictionary of metrics for the epoch
        """
        self.train()
        total_loss = 0.0
        dynamics_loss = 0.0
        structure_loss = 0.0
        num_batches = 0
        
        for batch_data in data_loader:
            # Extract batch data
            x, node_features, edge_index, batch_indicator, targets = batch_data
            
            # Move to device
            x = x.to(device)
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            batch_indicator = batch_indicator.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(
                x=x,
                node_features=node_features,
                edge_index=edge_index,
                batch=batch_indicator
            )
            
            # Compute loss
            loss = self.compute_loss(
                outputs=outputs,
                targets=targets,
                true_adjacency=true_adjacency
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'dynamics_loss': dynamics_loss / num_batches if dynamics_loss > 0 else 0.0,
            'structure_loss': structure_loss / num_batches if structure_loss > 0 else 0.0
        }
    
    def infer_causal_graph(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Infer the causal graph structure from data.
        
        Args:
            x: Time series data [batch_size, seq_length, n_variables]
            threshold: Threshold for edge pruning
            
        Returns:
            Adjacency matrix representing the inferred graph
        """
        self.eval()
        with torch.no_grad():
            # Infer graph structure
            adjacency_matrix = self.graph_encoder(x)
            
            # Apply threshold for discrete graph
            discrete_adjacency = (adjacency_matrix > threshold).float()
            
            return discrete_adjacency
    
    def predict_intervention_outcomes(
        self,
        x: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        intervention_targets: Optional[torch.Tensor] = None,
        intervention_values: Optional[torch.Tensor] = None,
        pre_computed_graph: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict outcomes under interventions.
        
        Args:
            x: Time series data [batch_size, seq_length, n_variables]
            node_features: Optional node features
            edge_index: Optional graph connectivity
            batch: Optional batch assignment
            intervention_targets: Indices of nodes to intervene on
            intervention_values: Values to set for interventions
            pre_computed_graph: Optional pre-computed adjacency matrix
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predicted outcomes, optionally with uncertainty estimates
        """
        self.eval()
        with torch.no_grad():
            # Get dimensions
            batch_size = x.size(0)
            n_variables = x.size(2)
            
            # Infer graph if not provided
            if pre_computed_graph is None:
                adjacency_matrix = self.graph_encoder(x)
            else:
                adjacency_matrix = pre_computed_graph
            
            # Expand adjacency matrix for batch processing
            adj_matrices = adjacency_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Create intervention dict if targets and values are provided
            interventions = None
            if intervention_targets is not None and intervention_values is not None:
                interventions = {
                    'targets': intervention_targets,
                    'values': intervention_values
                }
            
            # Generate node features if not provided
            if node_features is None:
                # Simple approach: use the last time step as features
                node_features = x[:, -1, :].reshape(batch_size * n_variables, 1)
                # Expand to input_dim if needed
                if node_features.size(1) < self.input_dim:
                    node_features = node_features.expand(-1, self.input_dim)
            
            # Generate edge index if not provided
            if edge_index is None:
                # Create edge index based on adjacency matrix
                edge_index = self._adjacency_to_edge_index(adjacency_matrix, batch_size)
            
            # Generate batch indicator if not provided
            if batch is None:
                batch = torch.repeat_interleave(torch.arange(batch_size), n_variables)
            
            # Predict outcomes
            if return_uncertainty and self.uncertainty:
                predictions, uncertainty = self.dynamics_decoder(
                    x=node_features,
                    edge_index=edge_index,
                    batch=batch,
                    adj_matrices=adj_matrices,
                    interventions=interventions,
                    return_uncertainty=True
                )
                return predictions, uncertainty
            else:
                predictions = self.dynamics_decoder(
                    x=node_features,
                    edge_index=edge_index,
                    batch=batch,
                    adj_matrices=adj_matrices,
                    interventions=interventions,
                    return_uncertainty=False
                )
                return predictions
    
    def to_causal_graph(
        self,
        adjacency_matrix: torch.Tensor,
        threshold: float = 0.5,
        node_names: Optional[List[Any]] = None
    ) -> CausalGraph:
        """
        Convert adjacency matrix to a CausalGraph object.
        
        Args:
            adjacency_matrix: Adjacency matrix [n_variables, n_variables]
            threshold: Threshold for edge pruning
            node_names: Optional names for the nodes
            
        Returns:
            CausalGraph object
        """
        # Apply threshold
        discrete_adjacency = (adjacency_matrix > threshold).float()
        
        # Create a CausalGraph
        n_variables = adjacency_matrix.size(0)
        
        # Use default node names if not provided
        if node_names is None:
            node_names = [f"X{i}" for i in range(n_variables)]
        
        # Create CausalGraph
        causal_graph = CausalGraph()
        
        # Add nodes
        for node in node_names:
            causal_graph.add_node(node)
        
        # Add edges
        for i in range(n_variables):
            for j in range(n_variables):
                if discrete_adjacency[i, j] > 0:
                    causal_graph.add_edge(node_names[i], node_names[j])
        
        return causal_graph
    
    def _adjacency_to_edge_index(
        self,
        adjacency_matrix: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Convert adjacency matrix to edge index format for PyTorch Geometric.
        
        Args:
            adjacency_matrix: Adjacency matrix [n_variables, n_variables]
            batch_size: Number of graphs in the batch
            
        Returns:
            Edge index tensor [2, num_edges]
        """
        n_variables = adjacency_matrix.size(0)
        device = adjacency_matrix.device
        
        # Apply threshold to get binary adjacency
        binary_adjacency = (adjacency_matrix > 0.5).float()
        
        # Find edges
        edges = torch.nonzero(binary_adjacency, as_tuple=True)
        sources, targets = edges
        
        # Create edge list for each graph in the batch
        edge_indices = []
        
        for b in range(batch_size):
            # Offset source and target by batch index
            batch_offset = b * n_variables
            batch_sources = sources + batch_offset
            batch_targets = targets + batch_offset
            
            # Stack to create edge index for this batch
            batch_edge_index = torch.stack([batch_sources, batch_targets], dim=0)
            
            edge_indices.append(batch_edge_index)
        
        # Concatenate all edge indices
        if edge_indices:
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            # If no edges, create empty edge index
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        return edge_index
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'input_dim': self.input_dim,
                'attention_heads': self.attention_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sparsity_weight': self.sparsity_weight,
                'acyclicity_weight': self.acyclicity_weight,
                'dynamics_weight': self.dynamics_weight,
                'structure_weight': self.structure_weight,
                'uncertainty': self.uncertainty,
                'num_ensembles': self.num_ensembles
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device = torch.device('cpu')):
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded AmortizedCausalDiscovery model
        """
        checkpoint = torch.load(path, map_location=device)
        
        # Create model with saved config
        model = cls(**checkpoint['config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model.to(device)
        
        return model
        
    def enable_meta_learning(
        self,
        task_embedding: Optional[TaskEmbedding] = None,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        first_order: bool = False,
        num_inner_steps: int = 5
    ) -> MAMLForCausalDiscovery:
        """
        Enable meta-learning capabilities for the model.
        
        This method wraps the current model with MAML to enable few-shot adaptation
        to new causal structures with minimal data.
        
        Args:
            task_embedding: Optional TaskEmbedding instance (one will be created if not provided)
            inner_lr: Learning rate for inner loop adaptation
            outer_lr: Learning rate for outer loop optimization
            first_order: Whether to use first-order approximation
            num_inner_steps: Number of inner loop adaptation steps
            
        Returns:
            MAMLForCausalDiscovery object that wraps this model
        """
        # Create a new TaskEmbedding if not provided
        if task_embedding is None:
            task_embedding = TaskEmbedding(
                input_dim=max(1, self.input_dim),
                embedding_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                architecture="gat",
                num_layers=self.num_layers,
                dropout=self.dropout,
                device=next(self.parameters()).device
            )
        
        # Create a MAMLForCausalDiscovery wrapper
        maml_model = MAMLForCausalDiscovery(
            model=self,
            task_embedding=task_embedding,
            inner_lr=inner_lr,
            outer_lr=outer_lr,
            first_order=first_order,
            num_inner_steps=num_inner_steps,
            device=next(self.parameters()).device
        )
        
        return maml_model
        
    def meta_adapt(
        self,
        causal_graph: CausalGraph,
        adaptation_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int = 5,
        inner_lr: float = 0.01,
        task_embedding: Optional[TaskEmbedding] = None
    ) -> 'AmortizedCausalDiscovery':
        """
        Adapt the model to a new causal graph structure using few-shot adaptation.
        
        This method creates a temporary MAML wrapper and adapts the model using
        the provided data, then returns a new instance with the adapted parameters.
        
        Args:
            causal_graph: CausalGraph structure to adapt to
            adaptation_data: Tuple of (inputs, targets) for adaptation
            num_steps: Number of adaptation steps
            inner_lr: Learning rate for adaptation
            task_embedding: Optional TaskEmbedding (one will be created if not provided)
            
        Returns:
            New AmortizedCausalDiscovery instance with adapted parameters
        """
        # Create MAML wrapper
        maml_model = self.enable_meta_learning(
            task_embedding=task_embedding,
            inner_lr=inner_lr,
            num_inner_steps=num_steps
        )
        
        # Adapt the model to the causal graph
        adapted_model = maml_model.adapt(
            graph=causal_graph,
            support_data=adaptation_data,
            num_steps=num_steps
        )
        
        # Return the adapted model
        return adapted_model 