"""
Tests for the SimpleGraphLearner model for causal graph structure learning.
"""

import pytest
import torch
import numpy as np
import pandas as pd

from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner
from causal_meta.structure_learning.data_utils import create_intervention_mask, convert_to_tensor
from causal_meta.structure_learning.graph_generators import RandomDAGGenerator


class TestSimpleGraphLearner:
    """Test suite for the SimpleGraphLearner model."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_vars = 5
        
        # Generate random data
        data = np.random.normal(0, 1, (n_samples, n_vars))
        df = pd.DataFrame(data, columns=[f'x{i}' for i in range(n_vars)])
        
        # Generate intervention mask
        intervention_mask = np.zeros((n_samples, n_vars))
        intervention_idx = 1  # Intervene on the second node (x1)
        intervention_mask[:, intervention_idx] = 1
        
        # Ground truth adjacency matrix (0->1->2, 0->3, 3->4)
        adj_matrix = np.zeros((n_vars, n_vars))
        adj_matrix[0, 1] = 1
        adj_matrix[1, 2] = 1
        adj_matrix[0, 3] = 1
        adj_matrix[3, 4] = 1
        
        return {
            'df': df,
            'intervention_mask': intervention_mask,
            'adj_matrix': adj_matrix,
            'n_vars': n_vars
        }

    def test_initialization(self, sample_data):
        """Test SimpleGraphLearner initialization."""
        n_vars = sample_data['n_vars']
        
        # Initialize model with default parameters
        model = SimpleGraphLearner(input_dim=n_vars)
        
        # Check default parameters
        assert model.input_dim == n_vars
        assert model.hidden_dim == 64
        assert model.sparsity_weight == 0.1
        assert model.acyclicity_weight == 1.0
        
        # Initialize with custom parameters
        model = SimpleGraphLearner(
            input_dim=n_vars,
            hidden_dim=32,
            sparsity_weight=0.2,
            acyclicity_weight=0.5
        )
        
        # Check custom parameters
        assert model.input_dim == n_vars
        assert model.hidden_dim == 32
        assert model.sparsity_weight == 0.2
        assert model.acyclicity_weight == 0.5

    def test_forward_pass(self, sample_data):
        """Test forward pass of SimpleGraphLearner."""
        df = sample_data['df']
        intervention_mask = sample_data['intervention_mask']
        n_vars = sample_data['n_vars']
        
        # Convert data to tensors
        data_tensor = convert_to_tensor(df)
        mask_tensor = torch.tensor(intervention_mask, dtype=torch.float32)
        
        # Initialize model
        model = SimpleGraphLearner(input_dim=n_vars)
        
        # Call forward method
        edge_probs = model(data_tensor, mask_tensor)
        
        # Check output shape and values
        assert edge_probs.shape == (n_vars, n_vars)
        assert torch.all(edge_probs >= 0) and torch.all(edge_probs <= 1)
        assert torch.all(torch.diag(edge_probs) == 0)  # No self-loops
    
    def test_loss_calculation(self, sample_data):
        """Test loss calculation with SimpleGraphLearner."""
        df = sample_data['df']
        intervention_mask = sample_data['intervention_mask']
        adj_matrix = sample_data['adj_matrix']
        n_vars = sample_data['n_vars']
        
        # Convert data to tensors
        data_tensor = convert_to_tensor(df)
        mask_tensor = torch.tensor(intervention_mask, dtype=torch.float32)
        target_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Initialize model
        model = SimpleGraphLearner(input_dim=n_vars)
        
        # Call forward method
        edge_probs = model(data_tensor, mask_tensor)
        
        # Calculate loss
        loss, loss_components = model.calculate_loss(edge_probs, target_tensor)
        
        # Check loss components
        assert 'supervised' in loss_components
        assert 'acyclicity' in loss_components
        assert 'sparsity' in loss_components
        assert 'total' in loss_components
        
        # Check that total loss is the sum of components
        assert torch.isclose(
            loss_components['total'],
            loss_components['supervised'] + loss_components['acyclicity'] + loss_components['sparsity']
        )
        
        # Check that loss works without ground truth
        loss_unsupervised, loss_comp_unsupervised = model.calculate_loss(edge_probs)
        assert 'supervised' not in loss_comp_unsupervised
        assert 'acyclicity' in loss_comp_unsupervised
        assert 'sparsity' in loss_comp_unsupervised
    
    def test_threshold_edge_probabilities(self, sample_data):
        """Test edge probability thresholding."""
        n_vars = sample_data['n_vars']
        
        # Create random edge probabilities
        edge_probs = torch.rand(n_vars, n_vars)
        # Set diagonal to 0.9 (should be set to 0 after thresholding)
        torch.diagonal(edge_probs)[:] = 0.9
        
        # Initialize model
        model = SimpleGraphLearner(input_dim=n_vars)
        
        # Apply thresholding
        adj_matrix = model.threshold_edge_probabilities(edge_probs, threshold=0.5)
        
        # Check output
        assert adj_matrix.shape == (n_vars, n_vars)
        assert torch.all((adj_matrix == 0) | (adj_matrix == 1))
        assert torch.all(torch.diag(adj_matrix) == 0)  # No self-loops
    
    def test_to_causal_graph(self, sample_data):
        """Test conversion to CausalGraph."""
        n_vars = sample_data['n_vars']
        
        # Create random edge probabilities
        edge_probs = torch.rand(n_vars, n_vars)
        # Set diagonal to 0 (no self-loops)
        torch.diagonal(edge_probs)[:] = 0
        
        # Initialize model
        model = SimpleGraphLearner(input_dim=n_vars)
        
        # Convert to CausalGraph
        graph = model.to_causal_graph(edge_probs, threshold=0.5)
        
        # Check graph properties
        assert len(graph.get_nodes()) == n_vars
        # Check that node names are x0, x1, etc.
        for i in range(n_vars):
            assert f'x{i}' in graph.get_nodes() 