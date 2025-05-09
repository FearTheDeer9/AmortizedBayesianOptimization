import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

from causal_meta.inference.interfaces import (
    InterventionOutcomeModel,
    Graph,
    Data,
    UncertaintyEstimate
)
from causal_meta.inference.adapters import DynamicsDecoderAdapter


class MockDynamicsDecoder(torch.nn.Module):
    """Mock dynamics decoder for testing the adapter."""
    
    def __init__(self):
        super().__init__()
        # Mock layer for demonstration
        self.fc = torch.nn.Linear(10, 10)
        
    def forward(
        self,
        x,
        edge_index,
        batch,
        adj_matrices,
        interventions=None,
        return_uncertainty=False
    ):
        """Return mock predictions."""
        batch_size = adj_matrices.size(0)
        num_nodes = adj_matrices.size(1)
        
        # Create random predictions
        predictions = torch.rand((batch_size * num_nodes, 1))
        
        if return_uncertainty:
            # Create random uncertainty
            uncertainty = torch.rand((batch_size * num_nodes, 1)) * 0.1
            return predictions, uncertainty
        else:
            return predictions
    
    def predict_intervention_outcome(
        self,
        x,
        edge_index,
        batch,
        adj_matrices,
        intervention_targets,
        intervention_values,
        return_uncertainty=False
    ):
        """Convenience method that calls forward with intervention data."""
        interventions = {
            'targets': intervention_targets,
            'values': intervention_values
        }
        
        return self.forward(
            x=x,
            edge_index=edge_index,
            batch=batch,
            adj_matrices=adj_matrices,
            interventions=interventions,
            return_uncertainty=return_uncertainty
        )
    

class TestDynamicsDecoderAdapter(unittest.TestCase):
    """Test suite for the DynamicsDecoderAdapter."""
    
    def setUp(self):
        """Set up the test case with a mock dynamics decoder."""
        self.mock_decoder = MockDynamicsDecoder()
        self.adapter = DynamicsDecoderAdapter(self.mock_decoder)
        
    def test_predict_intervention_outcome_numpy(self):
        """Test prediction with numpy data."""
        # Create test data
        num_nodes = 5
        num_samples = 50
        mock_graph = np.zeros((1, num_nodes, num_nodes))  # Mock adjacency matrix
        mock_intervention = {'target_node': 0, 'value': 2.0}
        mock_data = {"observations": np.random.randn(num_samples, num_nodes)}
        
        # Test prediction
        predictions = self.adapter.predict_intervention_outcome(mock_graph, mock_intervention, mock_data)
        
        # Verify output
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], num_nodes)
        
    def test_predict_intervention_outcome_torch(self):
        """Test prediction with PyTorch data."""
        # Create test data
        num_nodes = 5
        num_samples = 50
        mock_graph = torch.zeros((1, num_nodes, num_nodes))  # Mock adjacency matrix
        mock_intervention = {'target_node': 0, 'value': 2.0}
        mock_data = {"observations": torch.randn(num_samples, num_nodes)}
        
        # Test prediction
        predictions = self.adapter.predict_intervention_outcome(mock_graph, mock_intervention, mock_data)
        
        # Verify output
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], num_nodes)
        
    def test_update_model(self):
        """Test model updating."""
        # Create test data
        num_nodes = 5
        num_samples = 50
        mock_data = {"observations": np.random.randn(num_samples, num_nodes)}
        
        # Mock the training method
        self.mock_decoder.train = MagicMock()
        
        # Call update
        self.adapter.update_model(mock_data)
        
        # Verify train was called
        self.mock_decoder.train.assert_called_once()
        
    def test_estimate_uncertainty(self):
        """Test uncertainty estimation."""
        # Create test data
        num_nodes = 5
        num_samples = 50
        mock_graph = np.zeros((1, num_nodes, num_nodes))  # Mock adjacency matrix
        mock_intervention = {'target_node': 0, 'value': 2.0}
        mock_data = {"observations": np.random.randn(num_samples, num_nodes)}
        
        # Test prediction with uncertainty
        self.adapter._last_predictions = np.random.randn(num_nodes, 1)
        uncertainty = self.adapter.estimate_uncertainty()
        
        # Verify keys
        self.assertIn('prediction_std', uncertainty)
        
    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty returned."""
        # Create test data
        num_nodes = 5
        num_samples = 50
        mock_graph = np.zeros((1, num_nodes, num_nodes))  # Mock adjacency matrix
        mock_intervention = {'target_node': 0, 'value': 2.0}
        mock_data = {"observations": np.random.randn(num_samples, num_nodes)}
        
        # Configure adapter to return uncertainty
        adapter = DynamicsDecoderAdapter(self.mock_decoder, return_uncertainty=True)
        
        # Test prediction
        predictions, uncertainty = adapter.predict_intervention_outcome(
            mock_graph, mock_intervention, mock_data)
        
        # Verify output
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('prediction_std', uncertainty)
        
    def test_error_if_estimating_uncertainty_before_prediction(self):
        """Test that error is raised if estimating uncertainty before prediction."""
        # Create a fresh adapter
        adapter = DynamicsDecoderAdapter(self.mock_decoder)
        
        # Try to estimate uncertainty before prediction
        with self.assertRaises(RuntimeError):
            adapter.estimate_uncertainty()
            
    def test_error_on_invalid_data(self):
        """Test that error is raised for invalid data."""
        # Data without observations
        invalid_data = {
            'wrong_key': np.random.randn(10, 5)
        }
        
        # Create mock graph and intervention
        num_nodes = 5
        mock_graph = np.zeros((1, num_nodes, num_nodes))
        mock_intervention = {'target_node': 0, 'value': 2.0}
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.adapter.predict_intervention_outcome(mock_graph, mock_intervention, invalid_data)


if __name__ == '__main__':
    unittest.main() 