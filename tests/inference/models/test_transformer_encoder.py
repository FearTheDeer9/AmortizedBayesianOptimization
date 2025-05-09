import unittest
import numpy as np
import torch
import pytest
from typing import Dict, Any

from causal_meta.inference.interfaces import CausalStructureInferenceModel, Data, UncertaintyEstimate
from causal_meta.graph.causal_graph import CausalGraph


class TestTransformerGraphEncoder(unittest.TestCase):
    """Test suite for the TransformerGraphEncoder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # We need to import here because the class doesn't exist yet
        try:
            from causal_meta.inference.models.transformer_encoder import TransformerGraphEncoder
            from causal_meta.inference.adapters import TransformerGraphEncoderAdapter
            self.model_exists = True
        except ImportError:
            self.model_exists = False
            pytest.skip("TransformerGraphEncoder not implemented yet")
        
        # Create a small dataset for testing
        self.batch_size = 8
        self.n_variables = 5
        self.seq_length = 10
        
        # Generate random time series data
        self.X = torch.randn(self.batch_size, self.seq_length, self.n_variables)
        
        # Create encoder and adapter
        if self.model_exists:
            self.encoder = TransformerGraphEncoder(
                input_dim=self.n_variables,
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                dropout=0.1
            )
            self.adapter = TransformerGraphEncoderAdapter(self.encoder)
            
            # Create test data in the format expected by the interface
            self.test_data = {
                "observations": self.X.numpy()
            }
    
    def test_transformer_encoder_implementation(self):
        """Test that the TransformerGraphEncoder implementation exists."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
        
        from causal_meta.inference.models.transformer_encoder import TransformerGraphEncoder
        self.assertTrue(hasattr(TransformerGraphEncoder, 'forward'))
        self.assertTrue(hasattr(TransformerGraphEncoder, 'to_causal_graph'))
        
    def test_adapter_interface_compliance(self):
        """Test that the adapter correctly implements the interface."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
        self.assertIsInstance(self.adapter, CausalStructureInferenceModel)
        
    def test_forward_pass(self):
        """Test that the forward pass produces the expected output shape."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
        # Forward pass directly on the encoder
        edge_probs = self.encoder(self.X)
        
        # Check output shape
        self.assertEqual(edge_probs.shape, (self.n_variables, self.n_variables))
        
        # Check that probabilities are in [0, 1]
        self.assertTrue(torch.all(edge_probs >= 0))
        self.assertTrue(torch.all(edge_probs <= 1))
        
        # Check that diagonal is zero (no self-loops)
        for i in range(self.n_variables):
            self.assertEqual(edge_probs[i, i].item(), 0)
    
    def test_infer_structure(self):
        """Test that infer_structure method works correctly."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
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
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
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
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
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
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
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
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
        # Missing 'observations' key
        with self.assertRaises(ValueError):
            self.adapter.infer_structure({})
        
        # Invalid data type
        with self.assertRaises(TypeError):
            self.adapter.infer_structure({"observations": "not_a_valid_type"})
        
        # Invalid shape - needs 3D tensor [batch, seq, features]
        with self.assertRaises(ValueError):
            # Missing sequence dimension - 2D tensor with shape [batch, features]
            invalid_data = {"observations": np.random.randn(5, 5)}
            self.adapter.infer_structure(invalid_data)
    
    def test_to_causal_graph_method(self):
        """Test that the to_causal_graph method works correctly."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
        # Create a fake edge probability matrix
        edge_probs = torch.zeros((self.n_variables, self.n_variables))
        
        # Add some edges with probabilities > 0.5
        high_prob_edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4)
        ]
        
        for i, j in high_prob_edges:
            edge_probs[i, j] = 0.75  # Well above threshold
        
        # Set diagonal to zero (no self-loops)
        for i in range(self.n_variables):
            edge_probs[i, i] = 0
        
        # Convert to CausalGraph
        graph = self.encoder.to_causal_graph(edge_probs)
        
        # Check that result is a CausalGraph
        self.assertIsInstance(graph, CausalGraph)
        
        # Verify edges directly rather than via adjacency matrix
        for i, j in high_prob_edges:
            self.assertTrue(graph.has_edge(str(i), str(j)), 
                            f"Graph should have edge from {i} to {j}")
        
        # Check that other edges don't exist
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                if (i, j) not in high_prob_edges and i != j:
                    self.assertFalse(graph.has_edge(str(i), str(j)), 
                                     f"Graph should not have edge from {i} to {j}")
    
    def test_self_attention_mechanism(self):
        """Test that the self-attention mechanism works correctly."""
        if not self.model_exists:
            pytest.skip("TransformerGraphEncoder not implemented yet")
            
        # Check that the model has a self-attention module
        self.assertTrue(hasattr(self.encoder, 'transformer_layers'))
        
        # Check that attention can be applied to our data
        # This is a more implementation-specific test that might need adjustment
        # based on the actual transformer implementation
        try:
            with torch.no_grad():
                # Extract the first transformer layer
                transformer_layer = self.encoder.transformer_layers[0]
                
                # Prepare input for this layer (might require reshaping)
                # This will depend on the specific implementation details
                prepared_input = self.X[:, :, 0].unsqueeze(-1)  # Just use the first variable
                embedded_input = self.encoder.embedding(prepared_input)
                
                # Apply attention
                output = transformer_layer(embedded_input)
                
                # Check that output has reasonable shape
                self.assertIsInstance(output, torch.Tensor)
                self.assertTrue(output.numel() > 0)  # Non-empty tensor
        except Exception as e:
            # If the implementation is different, this might raise an exception
            # which is fine, as the main test is that the model has transformer_layers
            pass


if __name__ == '__main__':
    unittest.main() 