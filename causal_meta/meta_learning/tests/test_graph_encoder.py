import torch
import pytest
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader, TensorDataset

from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.graph.causal_graph import CausalGraph


class TestGraphEncoder:
    @pytest.fixture
    def sample_data(self):
        # Create a small dataset for testing
        batch_size = 8
        n_variables = 5
        seq_length = 10
        
        # Generate random time series data
        X = torch.randn(batch_size, seq_length, n_variables)
        
        # Create a sample adjacency matrix for a DAG
        adj_matrix = np.zeros((n_variables, n_variables))
        # Add edges 0->1, 0->2, 1->3, 2->3, 2->4
        adj_matrix[0, 1] = 1
        adj_matrix[0, 2] = 1
        adj_matrix[1, 3] = 1
        adj_matrix[2, 3] = 1
        adj_matrix[2, 4] = 1
        
        # Create a CausalGraph
        causal_graph = CausalGraph()
        
        # Add nodes
        for i in range(n_variables):
            causal_graph.add_node(i)
        
        # Add edges
        for i in range(n_variables):
            for j in range(n_variables):
                if adj_matrix[i, j] == 1:
                    causal_graph.add_edge(i, j)
        
        return {'X': X, 'causal_graph': causal_graph, 'n_variables': n_variables}
    
    def test_graph_encoder_init(self):
        """Test initialization of GraphEncoder."""
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Check basic attributes
        assert hasattr(encoder, 'hidden_dim')
        assert hasattr(encoder, 'attention_heads')
        assert hasattr(encoder, 'num_layers')
        assert hasattr(encoder, 'sparsity_weight')
        assert hasattr(encoder, 'acyclicity_weight')
        
        # Check values
        assert encoder.hidden_dim == 64
        assert encoder.attention_heads == 2
        assert encoder.num_layers == 2  # Default value
        assert encoder.sparsity_weight == 0.1  # Default value
        assert encoder.acyclicity_weight == 1.0  # Default value
    
    def test_forward_output_shape(self, sample_data):
        """Test that forward pass produces correctly shaped output."""
        X = sample_data['X']
        n_variables = sample_data['n_variables']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Forward pass
        edge_probs = encoder(X)
        
        # Check output shape
        assert edge_probs.shape == (n_variables, n_variables)
        
        # Check output is a probability matrix
        assert torch.all(edge_probs >= 0)
        assert torch.all(edge_probs <= 1)
        
        # Check diagonal is zero (no self-loops)
        assert torch.all(torch.diag(edge_probs) == 0)
    
    def test_batched_processing(self, sample_data):
        """Test that model can process batches of data."""
        X = sample_data['X']
        batch_size = X.shape[0]
        n_variables = sample_data['n_variables']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Process full batch
        edge_probs_full = encoder(X)
        
        # Process each sample individually and average
        edge_probs_individual = torch.zeros((n_variables, n_variables))
        for i in range(batch_size):
            edge_probs_individual += encoder(X[i:i+1])
        edge_probs_individual /= batch_size
        
        # Check results are similar (not exactly equal due to implementation details)
        assert torch.allclose(edge_probs_full, edge_probs_individual, atol=0.1)
    
    def test_sparsity_regularization(self, sample_data):
        """Test that sparsity regularization produces a loss term."""
        X = sample_data['X']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2, sparsity_weight=0.1)
        
        # Forward pass
        edge_probs = encoder(X)
        
        # Calculate sparsity loss
        sparsity_loss = encoder.get_sparsity_loss(edge_probs)
        
        # Loss should be positive scalar
        assert sparsity_loss.item() > 0
        assert sparsity_loss.dim() == 0  # Scalar
        
        # Test that higher weight produces larger loss
        encoder_high_weight = GraphEncoder(hidden_dim=64, attention_heads=2, sparsity_weight=1.0)
        high_weight_loss = encoder_high_weight.get_sparsity_loss(edge_probs)
        
        assert high_weight_loss > sparsity_loss
    
    def test_acyclicity_constraint(self, sample_data):
        """Test that acyclicity constraint produces a loss term."""
        X = sample_data['X']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2, acyclicity_weight=1.0)
        
        # Forward pass
        edge_probs = encoder(X)
        
        # Calculate acyclicity loss
        acyclicity_loss = encoder.get_acyclicity_loss(edge_probs)
        
        # Loss should be scalar
        assert acyclicity_loss.dim() == 0  # Scalar
        
        # Test that cyclic graph has higher loss
        # Create a cyclic graph
        cycle_probs = torch.zeros_like(edge_probs)
        n = cycle_probs.shape[0]
        for i in range(n):
            cycle_probs[i, (i + 1) % n] = 1.0
        
        cycle_loss = encoder.get_acyclicity_loss(cycle_probs)
        
        # Cyclic graph should have higher loss
        assert cycle_loss > acyclicity_loss
    
    def test_threshold_mechanism(self, sample_data):
        """Test thresholding edge probabilities to binary adjacency matrix."""
        X = sample_data['X']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Forward pass
        edge_probs = encoder(X)
        
        # Replace with controlled values for testing
        edge_probs = torch.tensor([
            [0.0, 0.7, 0.3, 0.1, 0.8],
            [0.2, 0.0, 0.6, 0.9, 0.4],
            [0.5, 0.1, 0.0, 0.3, 0.2],
            [0.2, 0.3, 0.4, 0.0, 0.6],
            [0.1, 0.2, 0.3, 0.8, 0.0]
        ])
        
        # Threshold with different values
        adj_high = encoder.threshold_edge_probabilities(edge_probs, threshold=0.7)
        adj_medium = encoder.threshold_edge_probabilities(edge_probs, threshold=0.5)
        adj_low = encoder.threshold_edge_probabilities(edge_probs, threshold=0.3)
        
        # Check binary values
        assert torch.all((adj_high == 0) | (adj_high == 1))
        assert torch.all((adj_medium == 0) | (adj_medium == 1))
        assert torch.all((adj_low == 0) | (adj_low == 1))
        
        # Check expected number of edges
        assert adj_high.sum() == 3  # Only the highest values
        assert adj_medium.sum() == 6  # Medium threshold
        assert adj_low.sum() == 11  # Low threshold
    
    def test_to_causal_graph(self, sample_data):
        """Test conversion from edge probabilities to CausalGraph."""
        X = sample_data['X']
        n_variables = sample_data['n_variables']
        
        encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        
        # Forward pass
        edge_probs = encoder(X)
        
        # Convert to CausalGraph
        causal_graph = encoder.to_causal_graph(edge_probs, threshold=0.5)
        
        # Check result is a CausalGraph
        assert isinstance(causal_graph, CausalGraph)
        
        # Check number of nodes
        assert len(causal_graph.get_nodes()) == n_variables
        
        # Convert to adjacency matrix and check structure
        adj_matrix = causal_graph.get_adjacency_matrix()
        binary_adj = encoder.threshold_edge_probabilities(edge_probs, threshold=0.5).numpy()
        
        assert np.array_equal(adj_matrix, binary_adj) 