import unittest
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
import torch

# Import the real interface
from causal_meta.inference.interfaces import (
    CausalStructureInferenceModel,
    Graph,
    Data,
    UncertaintyEstimate
)


class TestDetailedCausalStructureInferenceModel(unittest.TestCase):
    """Detailed test suite for the CausalStructureInferenceModel interface."""
    
    def test_interface_with_numpy_data(self):
        """Test interface with numpy array data."""
        
        class NumPyModel(CausalStructureInferenceModel):
            def infer_structure(self, data: Data) -> Graph:
                # Check that data contains observations key
                if 'observations' not in data:
                    raise ValueError("Data must contain 'observations' key")
                
                # Create a simple graph based on data dimensions
                obs = data['observations']
                n_vars = obs.shape[1]
                return np.triu(np.ones((n_vars, n_vars)) - np.eye(n_vars), k=1)
            
            def update_model(self, data: Data) -> None:
                # Validate data
                if 'observations' not in data:
                    raise ValueError("Data must contain 'observations' key")
                return None
            
            def estimate_uncertainty(self) -> UncertaintyEstimate:
                # Return mock uncertainty metrics
                return {
                    'edge_probabilities': np.array([[0.0, 0.9, 0.1], 
                                                   [0.0, 0.0, 0.8], 
                                                   [0.0, 0.0, 0.0]]),
                    'confidence_intervals': {
                        'lower': np.array([[0.0, 0.8, 0.0], 
                                          [0.0, 0.0, 0.7], 
                                          [0.0, 0.0, 0.0]]),
                        'upper': np.array([[0.0, 1.0, 0.2], 
                                          [0.0, 0.0, 0.9], 
                                          [0.0, 0.0, 0.0]])
                    }
                }
        
        # Instantiate model
        model = NumPyModel()
        
        # Create test data
        num_samples = 100
        num_variables = 3
        mock_data = {
            'observations': np.random.randn(num_samples, num_variables),
            'interventions': {
                'node_1': np.random.randn(num_samples, 1)
            }
        }
        
        # Test inference
        graph = model.infer_structure(mock_data)
        self.assertIsInstance(graph, np.ndarray)
        self.assertEqual(graph.shape, (num_variables, num_variables))
        
        # Verify it's a valid DAG (upper triangular in this case)
        self.assertTrue(np.allclose(graph, np.triu(graph)))
        
        # Test update
        model.update_model(mock_data)
        
        # Test uncertainty estimation
        uncertainty = model.estimate_uncertainty()
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('edge_probabilities', uncertainty)
        self.assertEqual(uncertainty['edge_probabilities'].shape, (num_variables, num_variables))
        
        # Test with missing data
        with self.assertRaises(ValueError):
            model.infer_structure({'wrong_key': np.random.randn(10, 3)})
    
    def test_interface_with_torch_data(self):
        """Test interface with PyTorch tensor data."""
        
        class TorchModel(CausalStructureInferenceModel):
            def infer_structure(self, data: Data) -> Graph:
                # Verify we have observations
                if 'observations' not in data:
                    raise ValueError("Data must contain 'observations' key")
                
                # Convert torch tensor to numpy if needed
                obs = data['observations']
                if isinstance(obs, torch.Tensor):
                    obs = obs.numpy()
                
                n_vars = obs.shape[1]
                # Return a simple chain graph
                adj_matrix = np.zeros((n_vars, n_vars))
                for i in range(n_vars-1):
                    adj_matrix[i, i+1] = 1
                return adj_matrix
            
            def update_model(self, data: Data) -> None:
                # Just a validation check
                if 'observations' not in data:
                    raise ValueError("Data must contain 'observations' key")
                return None
            
            def estimate_uncertainty(self) -> UncertaintyEstimate:
                # Simple uncertainty estimate
                return {
                    'edge_probabilities': np.array([[0.0, 0.95, 0.0], 
                                                   [0.0, 0.0, 0.92], 
                                                   [0.0, 0.0, 0.0]])
                }
        
        # Instantiate model
        model = TorchModel()
        
        # Create test data with PyTorch tensors
        num_samples = 100
        num_variables = 3
        mock_data = {
            'observations': torch.randn(num_samples, num_variables),
            'interventions': {
                'node_1': torch.randn(num_samples, 1)
            }
        }
        
        # Test inference
        graph = model.infer_structure(mock_data)
        self.assertIsInstance(graph, np.ndarray)
        self.assertEqual(graph.shape, (num_variables, num_variables))
        
        # Verify it's a chain graph
        expected_chain_graph = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.assertTrue(np.array_equal(graph, expected_chain_graph))
        
        # Test uncertainty
        uncertainty = model.estimate_uncertainty()
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('edge_probabilities', uncertainty)


if __name__ == '__main__':
    unittest.main() 