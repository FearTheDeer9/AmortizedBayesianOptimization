import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from causal_meta.inference.interfaces import InterventionOutcomeModel
from causal_meta.inference.adapters import DynamicsDecoderAdapter
from causal_meta.inference.uncertainty import (
    UncertaintyEstimator,
    EnsembleUncertaintyEstimator,
    DropoutUncertaintyEstimator,
    DirectUncertaintyEstimator,
    ConformalUncertaintyEstimator
)


class MockDynamicsDecoder(torch.nn.Module):
    """Mock dynamics decoder for testing the adapter."""
    
    def __init__(self, with_dropout=False):
        super().__init__()
        # Mock parameters - but don't actually use these for computation to avoid shape issues
        self.with_dropout = with_dropout
        
    def forward(
        self,
        x,
        edge_index,
        batch,
        adj_matrices,
        interventions=None,
        return_uncertainty=False
    ):
        """Return mock predictions without doing actual computations."""
        batch_size = adj_matrices.size(0)
        num_nodes = adj_matrices.size(1)
        
        # Create predictions with some deterministic pattern based on interventions
        predictions = torch.ones((batch_size * num_nodes, 1))
        if interventions is not None:
            # Modify predictions based on intervention
            try:
                node_idx = interventions['targets'].item()
                value = interventions['values'].item()
                predictions = predictions * value
            except (AttributeError, ValueError):
                # Handle cases where targets/values might not be a single item
                pass
        
        if return_uncertainty:
            # Create deterministic uncertainty for testing
            uncertainty = torch.ones((batch_size * num_nodes, 1)) * 0.1
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


class MockUncertaintyEstimator(UncertaintyEstimator):
    """Mock uncertainty estimator for testing."""
    
    def estimate_uncertainty(self, model, data):
        """Return mock uncertainty estimates."""
        # Always return the same uncertainty structure for testing
        return {
            'prediction_std': np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            'confidence_intervals': {
                'lower': np.array([[0.8], [1.6], [2.4], [3.2], [4.0]]),
                'upper': np.array([[1.2], [2.4], [3.6], [4.8], [6.0]]),
                'confidence': 0.95
            },
            'calibration_score': 0.92
        }
    
    def calibrate(self, model, validation_data):
        """Mock calibration method."""
        # Do nothing for the mock
        pass


class TestDynamicsUncertainty(unittest.TestCase):
    """Test suite for standardized uncertainty quantification in dynamics models."""
    
    def setUp(self):
        """Set up test cases with mocks."""
        self.mock_decoder = MockDynamicsDecoder()
        self.mock_estimator = MockUncertaintyEstimator()
        
        # Basic test data
        self.num_nodes = 5
        self.num_samples = 20
        self.mock_graph = np.zeros((1, self.num_nodes, self.num_nodes))
        self.mock_intervention = {'target_node': 0, 'value': 2.0}
        self.mock_data = {"observations": np.random.randn(self.num_samples, self.num_nodes)}
    
    def test_adapter_accepts_uncertainty_estimator(self):
        """Test that DynamicsDecoderAdapter accepts an UncertaintyEstimator."""
        # Create adapter with estimator
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=self.mock_decoder,
            uncertainty_estimator=self.mock_estimator
        )
        
        # Verify the estimator was stored
        self.assertEqual(adapter.uncertainty_estimator, self.mock_estimator)
    
    def test_adapter_uses_estimator_for_uncertainty(self):
        """Test that the adapter uses the provided estimator for uncertainty estimation."""
        # Create adapter with mock estimator
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=self.mock_decoder,
            uncertainty_estimator=self.mock_estimator
        )
        
        # Make a prediction first to have something to estimate uncertainty on
        adapter.predict_intervention_outcome(
            self.mock_graph, self.mock_intervention, self.mock_data
        )
        
        # Get uncertainty estimates
        uncertainty = adapter.estimate_uncertainty()
        
        # Verify that the estimator was used by checking format
        self.assertIn('prediction_std', uncertainty)
        self.assertIn('confidence_intervals', uncertainty)
        self.assertIn('calibration_score', uncertainty)
        
        # Verify the values match our mock estimator's return values
        np.testing.assert_array_equal(
            uncertainty['prediction_std'],
            np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        )
    
    def test_uncertainty_in_prediction_output(self):
        """Test that uncertainty is included in prediction output when requested."""
        # Create adapter with estimator
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=self.mock_decoder,
            uncertainty_estimator=self.mock_estimator,
            return_uncertainty=True
        )
        
        # Make prediction with uncertainty
        predictions, uncertainty = adapter.predict_intervention_outcome(
            self.mock_graph, self.mock_intervention, self.mock_data
        )
        
        # Verify prediction shape and uncertainty format
        self.assertEqual(predictions.shape, (self.num_nodes, 1))
        self.assertIn('prediction_std', uncertainty)
        self.assertIn('confidence_intervals', uncertainty)
    
    def test_separate_uncertainty_request(self):
        """Test requesting uncertainty separately from prediction."""
        # Create adapter without default uncertainty
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=self.mock_decoder,
            uncertainty_estimator=self.mock_estimator,
            return_uncertainty=False
        )
        
        # Make prediction without uncertainty
        predictions = adapter.predict_intervention_outcome(
            self.mock_graph, self.mock_intervention, self.mock_data
        )
        
        # Verify predictions shape
        self.assertEqual(predictions.shape, (self.num_nodes, 1))
        
        # Request uncertainty separately
        uncertainty = adapter.estimate_uncertainty()
        
        # Verify uncertainty format
        self.assertIn('prediction_std', uncertainty)
        self.assertIn('confidence_intervals', uncertainty)
    
    def test_uncertainty_format_standardization(self):
        """Test that uncertainty format is standardized regardless of estimator."""
        # Create different types of estimators
        estimators = [
            self.mock_estimator,
            MagicMock(spec=EnsembleUncertaintyEstimator),
            MagicMock(spec=DropoutUncertaintyEstimator)
        ]
        
        # Set up mock return values for the mocked estimators
        custom_uncertainty = {
            'prediction_std': np.ones((self.num_nodes, 1)) * 0.2,
            'confidence_intervals': {
                'lower': np.ones((self.num_nodes, 1)) * 0.5,
                'upper': np.ones((self.num_nodes, 1)) * 1.5,
                'confidence': 0.9
            }
        }
        
        estimators[1].estimate_uncertainty.return_value = custom_uncertainty
        estimators[2].estimate_uncertainty.return_value = custom_uncertainty
        
        # Test each estimator
        for estimator in estimators:
            adapter = DynamicsDecoderAdapter(
                dynamics_decoder=self.mock_decoder,
                uncertainty_estimator=estimator
            )
            
            # Make prediction
            adapter.predict_intervention_outcome(
                self.mock_graph, self.mock_intervention, self.mock_data
            )
            
            # Get uncertainty
            uncertainty = adapter.estimate_uncertainty()
            
            # Verify standard format
            self.assertIn('prediction_std', uncertainty)
            self.assertIn('confidence_intervals', uncertainty)
            self.assertIn('lower', uncertainty['confidence_intervals'])
            self.assertIn('upper', uncertainty['confidence_intervals'])
            self.assertIn('confidence', uncertainty['confidence_intervals'])
    
    def test_mc_dropout_integration(self):
        """Test integration with DropoutUncertaintyEstimator."""
        # Create dropout-enabled decoder
        dropout_decoder = MockDynamicsDecoder(with_dropout=True)
        
        # Create real DropoutUncertaintyEstimator
        dropout_estimator = DropoutUncertaintyEstimator(num_samples=5)
        
        # Create adapter with dropout estimator
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=dropout_decoder,
            uncertainty_estimator=dropout_estimator
        )
        
        # Need to patch torch.no_grad since the real estimator uses it
        with patch('torch.no_grad'):
            # Patch the estimator's estimate_uncertainty method
            dropout_estimator.estimate_uncertainty = MagicMock(return_value={
                'prediction_std': np.ones((self.num_nodes, 1)) * 0.1,
                'confidence_intervals': {
                    'lower': np.ones((self.num_nodes, 1)) * 0.8,
                    'upper': np.ones((self.num_nodes, 1)) * 1.2,
                    'confidence': 0.95
                }
            })
            
            # Make prediction
            adapter.predict_intervention_outcome(
                self.mock_graph, self.mock_intervention, self.mock_data
            )
            
            # Get uncertainty
            uncertainty = adapter.estimate_uncertainty()
            
            # Verify the dropout estimator was used
            self.assertTrue(dropout_estimator.estimate_uncertainty.called)
            
            # Verify standard format
            self.assertIn('prediction_std', uncertainty)
            self.assertIn('confidence_intervals', uncertainty)
    
    def test_calibration_integration(self):
        """Test calibration of uncertainty estimates."""
        # Create a mock estimator with calibration support
        calibration_estimator = MagicMock(spec=ConformalUncertaintyEstimator)
        
        # Set up mock return values
        calibration_estimator.estimate_uncertainty.return_value = {
            'prediction_std': np.ones((self.num_nodes, 1)) * 0.1,
            'confidence_intervals': {
                'lower': np.ones((self.num_nodes, 1)) * 0.8,
                'upper': np.ones((self.num_nodes, 1)) * 1.2,
                'confidence': 0.95
            },
            'calibration_error': 0.05
        }
        
        # Create adapter with calibration estimator
        adapter = DynamicsDecoderAdapter(
            dynamics_decoder=self.mock_decoder,
            uncertainty_estimator=calibration_estimator
        )
        
        # Create validation data
        validation_data = {
            "observations": np.random.randn(30, self.num_nodes)
        }
        
        # Calibrate the estimator
        adapter.calibrate_uncertainty(validation_data)
        
        # Verify calibration was called
        self.assertTrue(calibration_estimator.calibrate.called)
        
        # Check calibration data passed correctly
        args, kwargs = calibration_estimator.calibrate.call_args
        self.assertEqual(args[0], adapter)  # First arg should be the model
        self.assertEqual(args[1], validation_data)  # Second arg should be validation data


if __name__ == '__main__':
    unittest.main() 