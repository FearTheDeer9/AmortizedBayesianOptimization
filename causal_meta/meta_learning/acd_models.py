import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

from causal_meta.graph.causal_graph import CausalGraph


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for processing time series data
    and learning relationships between variables.
    """
    
    def __init__(self, hidden_dim, num_heads):
        """
        Initialize an attention block with multiple attention heads.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_heads: Number of attention heads
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Final layer normalization
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Apply self-attention over the input sequence.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
            
        Returns:
            output: Tensor of shape [batch_size, seq_length, hidden_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Layer normalization
        residual = x
        x = self.norm1(x)
        
        # Multi-head attention
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection with residual connection
        output = self.output_proj(context) + residual
        
        # Feed-forward network with residual connection
        residual = output
        output = self.norm2(output)
        output = self.ff_network(output) + residual
        
        return output


class GraphEncoder(nn.Module):
    """
    Neural network for causal structure learning using attention mechanisms.
    
    This encoder processes time series data and outputs a matrix of edge
    probabilities representing the inferred causal graph structure.
    """
    
    def __init__(self, hidden_dim=64, attention_heads=2, num_layers=2, 
                 sparsity_weight=0.1, acyclicity_weight=1.0):
        """
        Initialize the GraphEncoder.
        
        Args:
            hidden_dim: Dimension of hidden representations
            attention_heads: Number of attention heads in each layer
            num_layers: Number of attention layers
            sparsity_weight: Weight for sparsity regularization
            acyclicity_weight: Weight for acyclicity constraint
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.num_layers = num_layers
        self.sparsity_weight = sparsity_weight
        self.acyclicity_weight = acyclicity_weight
        
        # Initial node feature encoder (time series -> hidden representation)
        self.node_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Stack of attention layers
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, attention_heads) 
            for _ in range(num_layers)
        ])
        
        # Edge prediction network (node pair features -> edge probability)
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
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
        
        # Transpose to get shape [batch_size, n_variables, seq_length]
        x = x.transpose(1, 2)
        
        # Process each variable's time series
        node_features = []
        for i in range(n_variables):
            # Shape: [batch_size, seq_length, 1]
            var_i_values = x[:, i, :].unsqueeze(-1)
            
            # Encode time series to get initial node features
            # Shape: [batch_size, seq_length, hidden_dim]
            node_i_features = self.node_encoder(var_i_values)
            
            node_features.append(node_i_features)
        
        # Combine all node features
        # Shape: [batch_size, n_variables, seq_length, hidden_dim]
        node_features = torch.stack(node_features, dim=1)
        
        # Process each variable independently through attention layers
        processed_features = []
        for i in range(n_variables):
            # Shape: [batch_size, seq_length, hidden_dim]
            var_features = node_features[:, i]
            
            # Apply attention layers
            for layer in self.attention_layers:
                var_features = layer(var_features)
            
            # Average across sequence dimension to get node embedding
            # Shape: [batch_size, hidden_dim]
            node_embedding = var_features.mean(dim=1)
            
            processed_features.append(node_embedding)
            
        # Stack node embeddings
        # Shape: [batch_size, n_variables, hidden_dim]
        node_embeddings = torch.stack(processed_features, dim=1)
        
        # Create pairwise features for all possible edges
        edge_probs = torch.zeros((n_variables, n_variables), device=device)
        
        # For each potential edge
        for i in range(n_variables):
            for j in range(n_variables):
                if i == j:  # No self-loops
                    continue
                
                # Create pair features - source and target node embeddings
                # Shape: [batch_size, hidden_dim * 2]
                pair_features = torch.cat([node_embeddings[:, i], node_embeddings[:, j]], dim=1)
                
                # Apply edge predictor
                # Shape: [batch_size, 1]
                edge_prob = self.edge_predictor(pair_features)
                
                # Take mean over batch - shape: scalar
                mean_prob = edge_prob.mean()
                
                # Set value in adjacency matrix
                edge_probs[i, j] = mean_prob
        
        return edge_probs
    
    def get_sparsity_loss(self, edge_probs):
        """
        Calculate the sparsity regularization loss.
        
        Args:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
            
        Returns:
            loss: Scalar tensor with sparsity loss
        """
        # L1 regularization to encourage sparsity
        loss = torch.sum(torch.abs(edge_probs))
        
        return self.sparsity_weight * loss
    
    def get_acyclicity_loss(self, edge_probs):
        """
        Calculate the acyclicity constraint loss using matrix exponential.
        
        Args:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
            
        Returns:
            loss: Scalar tensor with acyclicity loss
        """
        n = edge_probs.shape[0]
        
        # Calculate h(W) = tr(e^(W ◦ W)) - n, where ◦ is element-wise product
        # This is a differentiable constraint that is minimized when the graph is acyclic
        W_squared = edge_probs * edge_probs
        
        # Compute matrix exponential
        try:
            exp_W = torch.matrix_exp(W_squared)
            acyclicity_loss = torch.trace(exp_W) - n
        except:
            # Fallback if matrix_exp encounters numerical issues
            # Approximate using power series: e^W ≈ I + W + W^2/2 + W^3/6 + ...
            identity = torch.eye(n, device=edge_probs.device)
            W_power = identity
            exp_W_approx = identity.clone()
            
            # Use first few terms of the series
            for k in range(1, 10):
                W_power = W_power @ W_squared / k
                exp_W_approx = exp_W_approx + W_power
            
            acyclicity_loss = torch.trace(exp_W_approx) - n
        
        return self.acyclicity_weight * acyclicity_loss
    
    def threshold_edge_probabilities(self, edge_probs, threshold=0.5):
        """
        Convert edge probabilities to binary adjacency matrix using a threshold.
        
        Args:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
            threshold: Threshold value for edge inclusion
            
        Returns:
            adj_matrix: Binary adjacency matrix
        """
        adj_matrix = (edge_probs > threshold).float()
        
        # Ensure no self-loops
        adj_matrix.fill_diagonal_(0)
        
        return adj_matrix
    
    def to_causal_graph(self, edge_probs, threshold=0.5):
        """
        Convert edge probabilities to a CausalGraph object.
        
        Args:
            edge_probs: Tensor of shape [n_variables, n_variables] with edge probabilities
            threshold: Threshold value for edge inclusion
            
        Returns:
            causal_graph: CausalGraph instance
        """
        # Threshold the edge probabilities
        adj_matrix = self.threshold_edge_probabilities(edge_probs, threshold)
        
        # Convert to networkx DiGraph
        n_variables = adj_matrix.shape[0]
        nx_graph = nx.DiGraph()
        
        # Add nodes
        for i in range(n_variables):
            nx_graph.add_node(i)
        
        # Add edges
        for i in range(n_variables):
            for j in range(n_variables):
                if adj_matrix[i, j] > 0:
                    nx_graph.add_edge(i, j)
        
        # Create CausalGraph from networkx DiGraph
        causal_graph = CausalGraph()
        
        # Add nodes to CausalGraph
        for node in nx_graph.nodes:
            causal_graph.add_node(node)
        
        # Add edges to CausalGraph
        for source, target in nx_graph.edges:
            causal_graph.add_edge(source, target)
        
        return causal_graph 