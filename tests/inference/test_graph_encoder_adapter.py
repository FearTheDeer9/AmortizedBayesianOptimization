import unittest
import numpy as np
import torch
import pytest
from typing import Dict, Any

from causal_meta.inference.interfaces import CausalStructureInferenceModel, Data, UncertaintyEstimate
from causal_meta.meta_learning.acd_models import GraphEncoder
from causal_meta.inference.adapters import GraphEncoderAdapter
from causal_meta.graph.causal_graph import CausalGraph


class TestGraphEncoderAdapter(unittest.TestCase):
    """Test suite for the GraphEncoderAdapter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small dataset for testing
        batch_size = 8
        self.n_variables = 5
        seq_length = 10
        
        # Generate random time series data
        self.X = torch.randn(batch_size, seq_length, self.n_variables)
        
        # Create adapter with default GraphEncoder
        self.encoder = GraphEncoder(hidden_dim=64, attention_heads=2)
        self.adapter = GraphEncoderAdapter(self.encoder)
        
        # Create test data in the format expected by the interface
        self.test_data = {
            "observations": self.X.numpy()
        }
    
    def test_adapter_interface_compliance(self):
        """Test that the adapter correctly implements the interface."""
        self.assertIsInstance(self.adapter, CausalStructureInferenceModel)
        
    def test_infer_structure(self):
        """Test that infer_structure method works correctly."""
        # Infer structure using the adapter
        graph = self.adapter.infer_structure(self.test_data)
        
        # Check that the result is a CausalGraph
        self.assertIsInstance(graph, CausalGraph)
        
        # Check that it has the right number of nodes
        self.assertEqual(len(graph.get_nodes()), self.n_variables)
        
        # Get adjacency matrix and check it's valid
        adj_matrix = graph.get_adjacency_matrix()
        self.assertEqual(adj_matrix.shape, (self.n_variables, self.n_variables))
        
        # Values should be binary
        for row in adj_matrix:
            for val in row:
                self.assertIn(val, [0, 1])
        
        # Diagonal should be zero (no self-loops)
        for i in range(self.n_variables):
            self.assertEqual(adj_matrix[i, i], 0)
    
    def test_update_model(self):
        """Test that update_model method works correctly."""
        # Should be able to call update_model without errors
        self.adapter.update_model(self.test_data)
        
        # Add interventional data and update
        interventional_data = {
            "observations": self.X.numpy(),
            "interventions": {
                0: np.random.randn(8, 10, 1)  # Intervene on node 0
            }
        }
        
        # Should handle interventional data
        self.adapter.update_model(interventional_data)
    
    def test_estimate_uncertainty(self):
        """Test that estimate_uncertainty method works correctly."""
        # First, infer a structure to have something to estimate uncertainty for
        self.adapter.infer_structure(self.test_data)
        
        # Estimate uncertainty
        uncertainty = self.adapter.estimate_uncertainty()
        
        # Check result is a dictionary
        self.assertIsInstance(uncertainty, dict)
        
        # Should have edge probabilities
        self.assertIn('edge_probabilities', uncertainty)
        
        # Edge probabilities should have correct shape
        edge_probs = uncertainty['edge_probabilities']
        self.assertEqual(edge_probs.shape, (self.n_variables, self.n_variables))
        
        # Values should be probabilities
        self.assertTrue(np.all(edge_probs >= 0))
        self.assertTrue(np.all(edge_probs <= 1))
    
    def test_data_format_handling(self):
        """Test that the adapter handles different data formats correctly."""
        # Test with NumPy arrays
        numpy_data = {
            "observations": self.X.numpy()
        }
        graph_numpy = self.adapter.infer_structure(numpy_data)
        self.assertIsInstance(graph_numpy, CausalGraph)
        
        # Test with PyTorch tensors
        torch_data = {
            "observations": self.X
        }
        graph_torch = self.adapter.infer_structure(torch_data)
        self.assertIsInstance(graph_torch, CausalGraph)
    
    def test_input_validation(self):
        """Test that the adapter validates input correctly."""
        # Missing 'observations' key
        with self.assertRaises(ValueError):
            self.adapter.infer_structure({})
        
        # Invalid data type
        with self.assertRaises(TypeError):
            self.adapter.infer_structure({"observations": "not_a_valid_type"})
        
        # Invalid shape
        with self.assertRaises(ValueError):
            # Missing sequence dimension
            invalid_data = {"observations": np.random.randn(8, 5)}
            self.adapter.infer_structure(invalid_data)


if __name__ == '__main__':
    unittest.main() 