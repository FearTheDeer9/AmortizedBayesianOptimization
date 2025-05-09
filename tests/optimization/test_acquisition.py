import unittest
import numpy as np

from causal_meta.optimization.acquisition import ExpectedImprovement, UpperConfidenceBound
from causal_meta.optimization.interfaces import AcquisitionStrategy


# Mock classes for testing
class MockGraph:
    """Mock graph for testing."""
    def __init__(self, adjacency_matrix=None):
        self.adjacency_matrix = adjacency_matrix or np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    def get_adjacency_matrix(self):
        return self.adjacency_matrix
    
    def get_nodes(self):
        return ["X0", "X1", "X2"]


class MockInterventionOutcomeModel:
    """Mock model for testing."""
    def predict_intervention_outcome(self, graph, intervention, data, return_uncertainty=False):
        # Extract target_node and value from intervention
        target_node = intervention.get('target_node', 0)
        value = intervention.get('value', 0.0)
        
        # Create mock prediction based on intervention value
        # Higher values are assumed to be better for testing
        prediction = np.array([value * 2])
        
        # Return with uncertainty if requested
        if return_uncertainty:
            uncertainty = {'prediction_std': np.array([0.1 + abs(value) * 0.05])}
            return prediction, uncertainty
        
        return prediction
        
    def estimate_uncertainty(self):
        # Return mock uncertainty
        return {'prediction_std': np.array([0.1])}
    
    def update_model(self, data):
        # Do nothing for mock
        pass


class TestExpectedImprovement(unittest.TestCase):
    """Test cases for the ExpectedImprovement acquisition strategy."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.model = MockInterventionOutcomeModel()
        self.graph = MockGraph()
        self.data = {'observations': np.random.rand(10, 3)}
        self.budget = 1.0
        
        # Create acquisition strategy
        self.acquisition = ExpectedImprovement(
            exploration_weight=0.1,
            maximize=True
        )
        
        # Set a best value for testing
        self.acquisition.set_best_value(1.0)
    
    def test_inheritance(self):
        """Test that ExpectedImprovement inherits from AcquisitionStrategy."""
        self.assertIsInstance(self.acquisition, AcquisitionStrategy)
    
    def test_compute_acquisition(self):
        """Test computing acquisition values."""
        acq_values = self.acquisition.compute_acquisition(
            self.model, self.graph, self.data
        )
        
        # Check that acquisition values were computed
        self.assertIsInstance(acq_values, dict)
        self.assertGreater(len(acq_values), 0)
        
        # Check that higher intervention values get higher acquisition values
        # This is specific to our mock model which returns value*2 as prediction
        key_positive = "node_0_value_2.0"
        key_negative = "node_0_value_-2.0"
        
        if key_positive in acq_values and key_negative in acq_values:
            self.assertGreater(acq_values[key_positive], acq_values[key_negative])
    
    def test_select_intervention(self):
        """Test selecting the best intervention."""
        intervention = self.acquisition.select_intervention(
            self.model, self.graph, self.data, self.budget
        )
        
        # Check intervention structure
        self.assertIsInstance(intervention, dict)
        self.assertIn('target_node', intervention)
        self.assertIn('value', intervention)
        
        # Our mock model returns value*2 as prediction, so the highest value should be selected
        # The candidate values are generated from -2.0 to 2.0
        self.assertGreaterEqual(intervention['value'], 1.0)
    
    def test_select_batch(self):
        """Test selecting a batch of interventions."""
        batch_size = 3
        interventions = self.acquisition.select_batch(
            self.model, self.graph, self.data, self.budget, batch_size
        )
        
        # Check batch size and structure
        self.assertIsInstance(interventions, list)
        self.assertEqual(len(interventions), batch_size)
        
        # Check that interventions are sorted by acquisition value
        # Our mock model should prefer higher values
        if len(interventions) >= 2:
            self.assertGreaterEqual(
                interventions[0]['value'],
                interventions[1]['value']
            )
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid budget
        with self.assertRaises(ValueError):
            self.acquisition.select_intervention(
                self.model, self.graph, self.data, -1.0
            )
        
        # Test invalid batch size
        with self.assertRaises(ValueError):
            self.acquisition.select_batch(
                self.model, self.graph, self.data, self.budget, 0
            )
        
        # Test invalid data
        with self.assertRaises(ValueError):
            self.acquisition.compute_acquisition(
                self.model, self.graph, {}
            )
    
    def test_maximize_vs_minimize(self):
        """Test that maximize=False properly changes behavior."""
        # Create acquisition strategy with maximize=False
        min_acquisition = ExpectedImprovement(
            exploration_weight=0.1,
            maximize=False
        )
        
        # Set a best value for testing
        min_acquisition.set_best_value(10.0)
        
        # Compute acquisition values for both strategies
        max_acq_values = self.acquisition.compute_acquisition(
            self.model, self.graph, self.data
        )
        min_acq_values = min_acquisition.compute_acquisition(
            self.model, self.graph, self.data
        )
        
        # For maximize=True, higher values are better
        # For maximize=False, lower values are better
        
        # Find a key present in both
        common_keys = set(max_acq_values.keys()) & set(min_acq_values.keys())
        if common_keys:
            # Check some keys where the behavior should be inverted
            for key in list(common_keys)[:3]:  # Check up to 3 common keys
                parts = key.split('_')
                value = float(parts[3])
                
                # For our mock model with return value*2:
                # - If value > 5.0, max prefers it, min doesn't
                # - If value < 5.0, min prefers it, max doesn't
                if value > 5.0:
                    self.assertGreater(max_acq_values[key], min_acq_values[key])
                elif value < 5.0:
                    self.assertLess(max_acq_values[key], min_acq_values[key])


class TestUpperConfidenceBound(unittest.TestCase):
    """Test cases for the UpperConfidenceBound acquisition strategy."""
    
    def setUp(self):
        """Set up test environment before each test case."""
        self.model = MockInterventionOutcomeModel()
        self.graph = MockGraph()
        self.data = {'observations': np.random.rand(10, 3)}
        self.budget = 1.0
        
        # Create acquisition strategy
        self.acquisition = UpperConfidenceBound(
            beta=2.0,
            maximize=True
        )
    
    def test_inheritance(self):
        """Test that UpperConfidenceBound inherits from AcquisitionStrategy."""
        self.assertIsInstance(self.acquisition, AcquisitionStrategy)
    
    # Add more specific tests for UCB once it's fully implemented


if __name__ == '__main__':
    unittest.main() 