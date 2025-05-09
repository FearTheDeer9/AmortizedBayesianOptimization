"""
Simplified Causal Discovery Models

This module contains simplified implementations of causal discovery models
that are compatible with the EnhancedMAMLForCausalDiscovery interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Union, Any, List


class AttentionBlock(nn.Module):
    """
    Self-attention block for node interactions.
    
    This helps the model better capture dependencies between nodes 
    in larger graphs by allowing each node to attend to all others.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention: each node attends to all other nodes
        # Input shape: [batch_size, num_nodes, hidden_dim]
        residual = x
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.dropout(attn_out)
        # Add residual connection and layer normalization
        out = self.layer_norm(residual + attn_out)
        return out


class SimplifiedCausalDiscovery(nn.Module):
    """
    Simplified causal discovery model that's compatible with the EnhancedMAMLForCausalDiscovery interface.
    
    This model handles tensor shapes appropriately and provides methods for intervention encoding.
    It serves as a drop-in replacement for the full AmortizedCausalDiscovery model but with
    a simpler architecture focused on adjacency matrix prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_layers: int = 2,
        dropout: float = 0.1,
        sparsity_weight: float = 0.1,
        acyclicity_weight: float = 0.5,
        use_attention: bool = True,
        num_heads: int = 4
    ):
        """
        Initialize the simplified causal discovery model.
        
        Args:
            input_dim: Number of input dimensions (nodes in the graph)
            hidden_dim: Hidden dimension size
            num_layers: Number of hidden layers
            dropout: Dropout rate
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
            use_attention: Whether to use attention mechanism
            num_heads: Number of attention heads
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        self.use_attention = use_attention
        self.num_heads = num_heads
        
        # Flag for external usage to know we can accept intervention channels
        self.accepts_intervention_channel = True
        
        # Determine input feature dimension (accounting for interventions)
        self.feature_input_dim = input_dim * 2  # Double for intervention markers
        
        # Feature extraction network for time series
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Build encoder layers
        encoder_layers = []
        for _ in range(num_layers - 1):
            if use_attention:
                encoder_layers.append(AttentionBlock(hidden_dim, num_heads, dropout))
            encoder_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Edge prediction head
        self.edge_predictor = nn.Linear(hidden_dim, input_dim * input_dim)
        
        # Separate prediction head for outgoing and incoming edges
        self.outgoing_edge_predictor = nn.Linear(hidden_dim, input_dim)
        self.incoming_edge_predictor = nn.Linear(hidden_dim, input_dim)
        
        # Node embeddings for better handling of large graphs
        self.node_embeddings = nn.Parameter(torch.randn(1, input_dim, hidden_dim))
        
        # Initialize weights
        self._init_weights()
        
        # For tracking metrics
        self.edge_uncertainties = None
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _compute_acyclicity_loss(self, adj_matrix):
        """
        Compute acyclicity constraint loss.
        
        This encourages the model to learn DAG structures by penalizing cycles.
        Based on the h(A) = tr(e^(A ◦ A)) - d formula from the NOTEARS paper.
        
        Args:
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            acyclicity_loss: Loss term that is minimized when the graph is acyclic
        """
        batch_size, n, _ = adj_matrix.shape
        loss = 0.0
        
        for b in range(batch_size):
            # Get adjacency matrix for this batch
            A = adj_matrix[b]
            
            # Element-wise square (Hadamard product with itself)
            A_squared = A * A
            
            # Compute matrix exponential and trace
            M = torch.matrix_exp(A_squared)
            trace = torch.trace(M)
            
            # h(A) = tr(e^(A ◦ A)) - d
            h_A = trace - n
            
            # Add to batch loss
            loss += h_A
        
        # Average over batch
        return loss / batch_size
    
    def _encode_interventions(self, x, interventions):
        """
        Encode intervention information directly into the input features.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_nodes] or [batch_size, num_nodes]
            interventions: Intervention mask [batch_size, num_nodes]
            
        Returns:
            Enhanced inputs with intervention markers
        """
        batch_size = x.shape[0]
        
        # Handle different input shapes
        if x.dim() == 3:
            # Time series data [batch_size, seq_len, num_nodes]
            seq_len = x.shape[1]
            
            # Expand interventions to match sequence length
            # [batch_size, num_nodes] -> [batch_size, seq_len, num_nodes]
            interventions_expanded = interventions.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate along feature dimension
            enhanced_x = torch.cat([x, interventions_expanded.float()], dim=2)
            
        else:
            # Already aggregated data [batch_size, num_nodes]
            enhanced_x = torch.cat([x, interventions.float()], dim=1)
            
        return enhanced_x
    
    def forward(self, x, encode_interventions=None, return_uncertainty=False):
        """
        Forward pass with support for intervention encoding.
        
        Args:
            x: Input tensor with shape [batch_size, seq_len, num_nodes]
            encode_interventions: Optional tensor with intervention mask [batch_size, num_nodes]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dict with adjacency matrix and optionally uncertainty
        """
        batch_size = x.shape[0]
        
        # Process time series by averaging over the sequence dimension if needed
        if x.dim() == 3:
            # For time series data [batch_size, seq_len, num_nodes]
            if x.shape[-1] == self.input_dim * 2 and encode_interventions is None:
                # Input already contains intervention encoding
                x_processed = x.mean(dim=1)  # [batch_size, num_nodes*2]
            else:
                # Standard input
                x_mean = x.mean(dim=1)  # [batch_size, num_nodes]
                
                # Default zeros for interventions if not provided
                if encode_interventions is None:
                    interventions = torch.zeros(batch_size, self.input_dim, device=x.device)
                else:
                    interventions = encode_interventions
                
                # Create enhanced features with intervention markers
                x_processed = self._encode_interventions(x_mean, interventions)
        else:
            # Already in the right shape
            if x.shape[-1] == self.input_dim * 2 and encode_interventions is None:
                # Input already contains intervention encoding
                x_processed = x
            else:
                # Add intervention encoding
                if encode_interventions is None:
                    interventions = torch.zeros(batch_size, self.input_dim, device=x.device)
                else:
                    interventions = encode_interventions
                
                x_processed = self._encode_interventions(x, interventions)
        
        # Extract features
        h = self.feature_extractor(x_processed)
        
        # Expand node embeddings to batch size and add to features
        node_emb_expanded = self.node_embeddings.expand(batch_size, -1, -1)
        
        # Reshape h to match node embeddings if needed
        if h.dim() == 2:
            h = h.view(batch_size, self.input_dim, -1)
        
        # Add node embeddings if shapes match
        if h.shape[1:] == node_emb_expanded.shape[1:]:
            h = h + node_emb_expanded
        
        # Flatten for encoder if needed
        h_flat = h.view(batch_size, -1) if h.dim() > 2 else h
        
        # Process through encoder
        h_encoded = self.encoder(h_flat)
        
        # Predict adjacency matrix
        edge_logits = self.edge_predictor(h_encoded)
        edge_logits = edge_logits.view(batch_size, self.input_dim, self.input_dim)
        
        # Apply sigmoid to get probabilities
        adj_probs = torch.sigmoid(edge_logits)
        
        # Zero out the diagonal (no self-loops)
        diag_mask = torch.eye(self.input_dim, device=adj_probs.device).unsqueeze(0)
        adj_probs = adj_probs * (1 - diag_mask)
        
        # Calculate uncertainty (4*p*(1-p) peaks at p=0.5)
        uncertainty = 4 * adj_probs * (1 - adj_probs)
        
        # Store for later use
        self.edge_uncertainties = [uncertainty]
        
        if return_uncertainty:
            return {
                'adjacency': adj_probs,
                'uncertainty': uncertainty
            }
        
        return adj_probs
    
    def forward_with_interventions(self, x, interventions):
        """Alternative interface for intervention encoding."""
        return self.forward(x, encode_interventions=interventions)
    
    def loss(self, pred_adj, true_adj, additional_info=None):
        """
        Compute loss with regularization.
        
        Args:
            pred_adj: Predicted adjacency matrix
            true_adj: True adjacency matrix
            additional_info: Optional dictionary with additional information
            
        Returns:
            Loss value
        """
        # Binary cross entropy loss
        bce_loss = F.binary_cross_entropy(pred_adj, true_adj)
        
        # Sparsity regularization
        sparsity_loss = torch.mean(pred_adj)
        
        # Acyclicity regularization
        acyclicity_loss = self._compute_acyclicity_loss(pred_adj)
        
        # Full loss
        loss = bce_loss + self.sparsity_weight * sparsity_loss + self.acyclicity_weight * acyclicity_loss
        
        return loss


def create_simplified_model_with_maml(
    from_model=None, 
    num_nodes=None, 
    hidden_dim=64, 
    num_layers=2,
    device=None,
    use_attention=True
):
    """
    Create a simplified causal discovery model, optionally transferring weights from an existing model.
    
    Args:
        from_model: Optional existing model to transfer weights from
        num_nodes: Number of nodes in the graph
        hidden_dim: Hidden dimension size
        num_layers: Number of hidden layers
        device: PyTorch device
        use_attention: Whether to use attention mechanisms for larger graphs
        
    Returns:
        Initialized simplified model
    """
    if num_nodes is None and from_model is not None:
        # Try to infer num_nodes from the existing model
        if hasattr(from_model, 'input_dim'):
            num_nodes = from_model.input_dim
        else:
            raise ValueError("Cannot infer num_nodes from the provided model. Please specify num_nodes explicitly.")
    
    if num_nodes is None:
        raise ValueError("num_nodes must be provided if from_model is None")
    
    # Create the simplified model
    model = SimplifiedCausalDiscovery(
        input_dim=num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_attention=use_attention
    )
    
    # Transfer weights if possible and if from_model is provided
    if from_model is not None:
        try:
            # Try to copy weights for common layers
            for name, param in from_model.named_parameters():
                if name in dict(model.named_parameters()):
                    model.state_dict()[name].copy_(param.data)
        except Exception as e:
            print(f"Warning: Failed to transfer weights from the existing model: {e}")
    
    # Move to device
    if device is not None:
        model = model.to(device)
    
    return model 