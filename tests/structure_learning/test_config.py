"""Tests for the ExperimentConfig class."""

import unittest
import pytest
from causal_meta.structure_learning.config import ExperimentConfig


class TestExperimentConfig(unittest.TestCase):
    """Test cases for the ExperimentConfig class."""

    def test_default_initialization(self):
        """Test the default initialization of ExperimentConfig."""
        config = ExperimentConfig()
        self.assertEqual(config.num_nodes, 8)
        self.assertEqual(config.edge_probability, 0.3)
        self.assertEqual(config.num_obs_samples, 200)
        self.assertEqual(config.num_int_samples, 20)
        self.assertEqual(config.max_iterations, 50)
        self.assertEqual(config.hidden_dim, 64)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.noise_scale, 0.1)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.log_interval, 5)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.threshold, 0.5)

    def test_custom_initialization(self):
        """Test custom initialization of ExperimentConfig."""
        config = ExperimentConfig(
            num_nodes=10,
            edge_probability=0.4,
            num_obs_samples=300,
            num_int_samples=30,
            max_iterations=60,
            hidden_dim=128,
            learning_rate=0.002,
            noise_scale=0.2,
            random_seed=43,
            log_interval=10,
            device="cpu",
            threshold=0.6,
        )
        self.assertEqual(config.num_nodes, 10)
        self.assertEqual(config.edge_probability, 0.4)
        self.assertEqual(config.num_obs_samples, 300)
        self.assertEqual(config.num_int_samples, 30)
        self.assertEqual(config.max_iterations, 60)
        self.assertEqual(config.hidden_dim, 128)
        self.assertEqual(config.learning_rate, 0.002)
        self.assertEqual(config.noise_scale, 0.2)
        self.assertEqual(config.random_seed, 43)
        self.assertEqual(config.log_interval, 10)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.threshold, 0.6)

    def test_validation(self):
        """Test validation of ExperimentConfig parameters."""
        # Test with invalid num_nodes
        with pytest.raises(AssertionError):
            ExperimentConfig(num_nodes=0)
        
        # Test with invalid edge_probability
        with pytest.raises(AssertionError):
            ExperimentConfig(edge_probability=-0.1)
        with pytest.raises(AssertionError):
            ExperimentConfig(edge_probability=1.1)
        
        # Test with invalid num_obs_samples
        with pytest.raises(AssertionError):
            ExperimentConfig(num_obs_samples=0)
        
        # Test with invalid num_int_samples
        with pytest.raises(AssertionError):
            ExperimentConfig(num_int_samples=0)
        
        # Test with invalid max_iterations
        with pytest.raises(AssertionError):
            ExperimentConfig(max_iterations=0)
        
        # Test with invalid hidden_dim
        with pytest.raises(AssertionError):
            ExperimentConfig(hidden_dim=0)
        
        # Test with invalid learning_rate
        with pytest.raises(AssertionError):
            ExperimentConfig(learning_rate=0)
        
        # Test with invalid noise_scale
        with pytest.raises(AssertionError):
            ExperimentConfig(noise_scale=0)
        
        # Test with invalid log_interval
        with pytest.raises(AssertionError):
            ExperimentConfig(log_interval=0)
        
        # Test with invalid threshold
        with pytest.raises(AssertionError):
            ExperimentConfig(threshold=-0.1)
        with pytest.raises(AssertionError):
            ExperimentConfig(threshold=1.1)
        
        # Test with invalid device
        with pytest.raises(AssertionError):
            ExperimentConfig(device="invalid")


if __name__ == "__main__":
    unittest.main() 