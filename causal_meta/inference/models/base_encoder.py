"""
Base classes for causal structure inference models.

This module provides abstract base classes for causal structure inference models,
establishing common interfaces and shared functionality that specific model
implementations can build upon.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple

from causal_meta.graph.causal_graph import CausalGraph


class BaseStructureInferenceModel(nn.Module, ABC):
    """
    Abstract base class for causal structure inference models.
    
    This class defines common functionality and interfaces for models that infer
    causal structure from data. Implementations should be able to process time series
    data and output a matrix of edge probabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0
    ):
        """
        Initialize the base model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        
        # Store last edge probabilities for uncertainty estimation
        self._last_edge_probs = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict edge probabilities.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tensor of edge probabilities
        """
        pass
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def calculate_acyclicity_regularization(self, edge_probs: torch.Tensor) -> torch.Tensor:
        """
        Calculate acyclicity regularization term.
        
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
        Calculate sparsity regularization term.
        
        Args:
            edge_probs: Tensor of edge probabilities
            
        Returns:
            Sparsity regularization loss term
        """
        # L1 regularization to promote sparsity
        return torch.sum(torch.abs(edge_probs))
    
    def calculate_loss(
        self,
        edge_probs: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate loss for the model.
        
        Args:
            edge_probs: Predicted edge probabilities
            target: Target adjacency matrix (optional)
            mask: Mask for valid edges (optional)
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        loss_components = {}
        
        # Supervised loss if target is provided
        if target is not None:
            if mask is not None:
                # Apply mask to only consider valid edges
                edge_probs_masked = edge_probs[mask]
                target_masked = target[mask]
                supervised_loss = torch.nn.functional.binary_cross_entropy(
                    edge_probs_masked, target_masked
                )
            else:
                supervised_loss = torch.nn.functional.binary_cross_entropy(
                    edge_probs, target
                )
            loss_components['supervised'] = supervised_loss
        else:
            supervised_loss = 0.0
        
        # Regularization losses
        acyclicity_reg = self.calculate_acyclicity_regularization(edge_probs)
        sparsity_reg = self.calculate_sparsity_regularization(edge_probs)
        
        loss_components['acyclicity'] = acyclicity_reg * self.acyclicity_weight
        loss_components['sparsity'] = sparsity_reg * self.sparsity_weight
        
        # Total loss
        total_loss = supervised_loss + loss_components['acyclicity'] + loss_components['sparsity']
        loss_components['total'] = total_loss
        
        return total_loss, loss_components


class MLPBaseEncoder(BaseStructureInferenceModel):
    """
    Base class for MLP-based structure inference models.
    
    This class provides common functionality for MLP-based models, including
    network construction, initialization, and utility methods.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 1.0
    ):
        """
        Initialize the MLP base model.
        
        Args:
            input_dim: Number of input variables
            hidden_dim: Dimension of hidden layers
            num_layers: Number of MLP layers
            dropout: Dropout probability
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight
        )
        self.num_layers = num_layers
    
    def _build_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ) -> nn.Sequential:
        """
        Build a multi-layer perceptron with the specified dimensions.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of layers
            
        Returns:
            Sequential MLP module
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)


class TransformerBaseEncoder(BaseStructureInferenceModel):
    """
    Base class for Transformer-based structure inference models.
    
    This class provides common functionality for Transformer-based models, including
    network construction, initialization, and utility methods.
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
        """
        Initialize the Transformer base model.
        
        Args:
            input_dim: Number of input variables
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sparsity_weight=sparsity_weight,
            acyclicity_weight=acyclicity_weight
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def _generate_positional_encoding(
        self,
        max_seq_len: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """
        Generate positional encoding for transformer.
        
        Args:
            max_seq_len: Maximum sequence length
            hidden_dim: Hidden dimension
            
        Returns:
            Positional encoding tensor
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-np.log(10000.0) / hidden_dim))
        pos_encoding = torch.zeros(max_seq_len, hidden_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    
    def _build_transformer_layer(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float
    ) -> nn.TransformerEncoderLayer:
        """
        Build a transformer encoder layer.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            
        Returns:
            TransformerEncoderLayer
        """
        return nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        ) 