import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

from causal_meta.inference.interfaces import (
    CausalStructureInferenceModel,
    Graph,
    Data,
    UncertaintyEstimate
)
from causal_meta.inference.adapters import GraphEncoderAdapter


class MockGraphEncoder(torch.nn.Module):
    """Mock graph encoder for testing the adapter."""
    
    def __init__(self):
        super().__init__()
        # Mock layer for demonstration
        self.fc = torch.nn.Linear(10, 10)
        
    def forward(self, x):
        """Return mock edge probabilities."""
        batch_size = x.shape[0]
        num_nodes = x.shape[2] if len(x.shape) > 2 else x.shape[1]
        
        # Create random edge probabilities (upper triangular to ensure DAG)
        edge_probs = torch.zeros((batch_size, num_nodes, num_nodes))
        for b in range(batch_size):
            # Create upper triangular matrix of probabilities
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    edge_probs[b, i, j] = torch.rand(1)
                    
        return edge_probs
    
    def train(self, data):
        """Mock training method."""
        pass
    

class TestGraphEncoderAdapter(unittest.TestCase):
    """Test suite for the GraphEncoderAdapter."""
    
    def setUp(self):
        """Set up the test case with a mock graph encoder."""
        self.mock_encoder = MockGraphEncoder()
        self.adapter = GraphEncoderAdapter(self.mock_encoder, threshold=0.5)
        
    def test_infer_structure_numpy(self):
        """Test inference with numpy data."""
        # Create test data
        num_samples = 50
        num_variables = 5
        mock_data = {
            'observations': np.random.randn(num_samples, num_variables)
        }
        
        # Test inference
        graph = self.adapter.infer_structure(mock_data)
        
        # Verify output
        self.assertIsInstance(graph, np.ndarray)
        self.assertEqual(graph.shape, (1, num_variables, num_variables))
        
        # Verify it's a valid DAG (upper triangular)
        for b in range(graph.shape[0]):
            for i in range(num_variables):
                for j in range(i+1):
                    self.assertEqual(graph[b, i, j], 0)
        
    def test_infer_structure_torch(self):
        """Test inference with PyTorch data."""
        # Create test data
        num_samples = 50
        num_variables = 5
        mock_data = {
            'observations': torch.randn(num_samples, num_variables)
        }
        
        # Test inference
        graph = self.adapter.infer_structure(mock_data)
        
        # Verify output
        self.assertIsInstance(graph, np.ndarray)
        self.assertEqual(graph.shape, (1, num_variables, num_variables))
        
    def test_update_model(self):
        """Test model updating."""
        # Create test data
        num_samples = 50
        num_variables = 5
        mock_data = {
            'observations': np.random.randn(num_samples, num_variables)
        }
        
        # Mock the training method
        self.mock_encoder.train = MagicMock()
        
        # Call update
        self.adapter.update_model(mock_data)
        
        # Verify train was called
        self.mock_encoder.train.assert_called_once()
        
    def test_estimate_uncertainty(self):
        """Test uncertainty estimation."""
        # Create test data
        num_samples = 50
        num_variables = 5
        mock_data = {
            'observations': np.random.randn(num_samples, num_variables)
        }
        
        # Infer structure first to populate edge probabilities
        self.adapter.infer_structure(mock_data)
        
        # Get uncertainty estimates
        uncertainty = self.adapter.estimate_uncertainty()
        
        # Verify keys and shapes
        self.assertIn('edge_probabilities', uncertainty)
        self.assertIn('entropy', uncertainty)
        self.assertIn('confidence_intervals', uncertainty)
        
        self.assertEqual(uncertainty['edge_probabilities'].shape, (1, num_variables, num_variables))
        self.assertEqual(uncertainty['entropy'].shape, (1, num_variables, num_variables))
        
        self.assertIn('lower', uncertainty['confidence_intervals'])
        self.assertIn('upper', uncertainty['confidence_intervals'])
        self.assertEqual(uncertainty['confidence_intervals']['lower'].shape, 
                         (1, num_variables, num_variables))
        
    def test_error_if_estimating_uncertainty_before_inference(self):
        """Test that error is raised if estimating uncertainty before inference."""
        # Create a fresh adapter
        adapter = GraphEncoderAdapter(self.mock_encoder)
        
        # Try to estimate uncertainty before inference
        with self.assertRaises(RuntimeError):
            adapter.estimate_uncertainty()
            
    def test_error_on_invalid_data(self):
        """Test that error is raised for invalid data."""
        # Data without observations
        invalid_data = {
            'wrong_key': np.random.randn(10, 5)
        }
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.adapter.infer_structure(invalid_data)


if __name__ == '__main__':
    unittest.main() 