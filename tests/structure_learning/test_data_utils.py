"""Tests for data generation and processing utilities."""

import unittest
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any

from causal_meta.structure_learning.graph_generators import RandomDAGGenerator
from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.environments.scm import StructuralCausalModel

# Import the module to be tested (not implemented yet, will fail initially)
# from causal_meta.structure_learning.data_utils import (
#     generate_observational_data,
#     generate_interventional_data,
#     generate_random_intervention_data,
#     create_intervention_mask,
#     convert_to_tensor
# )


class TestDataUtilsBase(unittest.TestCase):
    """Base class for data utilities tests with common setup."""

    def setUp(self):
        """Create a simple SCM for testing."""
        # Fix random seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        
        # Create a simple DAG adjacency matrix
        self.adj_matrix = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.node_names = [f"x{i}" for i in range(3)]
        self.noise_scale = 0.1
        self.n_samples = 100
        
        # Generate an SCM
        self.scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=self.adj_matrix,
            noise_scale=self.noise_scale,
            seed=self.seed
        )


class TestObservationalDataGeneration(TestDataUtilsBase):
    """Test cases for observational data generation functions."""

    def test_generate_observational_data_shape(self):
        """Test generated observational data has the correct shape."""
        from causal_meta.structure_learning.data_utils import generate_observational_data
        
        # Generate observational data as DataFrame
        data_df = generate_observational_data(
            scm=self.scm,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Check shape and type
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertEqual(data_df.shape, (self.n_samples, 3))
        
        # Check columns match node names
        self.assertListEqual(list(data_df.columns), self.node_names)
        
        # Generate observational data as tensor
        data_tensor = generate_observational_data(
            scm=self.scm,
            n_samples=self.n_samples,
            as_tensor=True
        )
        
        # Check tensor shape and type
        self.assertIsInstance(data_tensor, torch.Tensor)
        self.assertEqual(data_tensor.shape, (self.n_samples, 3))

    def test_generate_observational_data_content(self):
        """Test generated observational data has expected statistical properties."""
        from causal_meta.structure_learning.data_utils import generate_observational_data
        
        # Generate observational data
        data_df = generate_observational_data(
            scm=self.scm,
            n_samples=1000,  # Larger sample for better statistics
            as_tensor=False
        )
        
        # Check correlations (parents should correlate with children)
        corr = data_df.corr().abs()
        
        # Check if correlation exists (not checking exact value as it depends on random weights)
        self.assertGreater(corr.loc["x0", "x1"], 0.0)
        self.assertGreater(corr.loc["x0", "x2"], 0.0)
        
        # Check data range (should be centered around 0 with noise_scale variation)
        for col in data_df.columns:
            mean = data_df[col].mean()
            std = data_df[col].std()
            self.assertLess(abs(mean), 0.5)  # Mean should be close to 0
            self.assertGreater(std, 0.0)     # Should have some variation


class TestInterventionalDataGeneration(TestDataUtilsBase):
    """Test cases for interventional data generation functions."""

    def test_generate_interventional_data_shape(self):
        """Test generated interventional data has the correct shape."""
        from causal_meta.structure_learning.data_utils import generate_interventional_data
        
        # Generate interventional data as DataFrame
        data_df = generate_interventional_data(
            scm=self.scm,
            node="x0",
            value=1.0,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Check shape and type
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertEqual(data_df.shape, (self.n_samples, 3))
        
        # Check columns match node names
        self.assertListEqual(list(data_df.columns), self.node_names)
        
        # Generate interventional data as tensor
        data_tensor = generate_interventional_data(
            scm=self.scm,
            node="x0",
            value=1.0,
            n_samples=self.n_samples,
            as_tensor=True
        )
        
        # Check tensor shape and type
        self.assertIsInstance(data_tensor, torch.Tensor)
        self.assertEqual(data_tensor.shape, (self.n_samples, 3))

    def test_generate_interventional_data_content(self):
        """Test interventional data has correct intervention effects."""
        from causal_meta.structure_learning.data_utils import generate_interventional_data
        
        # Generate interventional data with intervention on x0
        data_df = generate_interventional_data(
            scm=self.scm,
            node="x0",
            value=2.0,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Check intervened node has the fixed value
        self.assertTrue(np.allclose(data_df["x0"], 2.0))
        
        # Generate interventional data with intervention on x1
        data_df2 = generate_interventional_data(
            scm=self.scm,
            node="x1",
            value=-1.0,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Check intervened node has the fixed value
        self.assertTrue(np.allclose(data_df2["x1"], -1.0))
        
        # x0 should not be affected by intervention on x1
        self.assertFalse(np.allclose(data_df2["x0"], 0.0))
        
        # Check if x2 is affected by intervention on x1 (it should be different)
        # We can't check exact values due to randomness, but we can check it's not all zeros
        self.assertFalse(np.allclose(data_df2["x2"], 0.0))


class TestRandomInterventions(TestDataUtilsBase):
    """Test cases for random intervention generation."""

    def test_generate_random_intervention_data(self):
        """Test random intervention data generation."""
        from causal_meta.structure_learning.data_utils import generate_random_intervention_data
        
        # Generate random intervention data
        data_df, intervened_node, intervention_value = generate_random_intervention_data(
            scm=self.scm,
            n_samples=self.n_samples,
            as_tensor=False
        )
        
        # Check shape and type
        self.assertIsInstance(data_df, pd.DataFrame)
        self.assertEqual(data_df.shape, (self.n_samples, 3))
        
        # Check that the intervened node has the intervention value
        self.assertTrue(np.allclose(data_df[intervened_node], intervention_value))
        
        # Test with specified intervention values
        intervention_values = {
            "x0": [1.0, 2.0],
            "x1": [-1.0, 0.0, 1.0],
            "x2": [0.5]
        }
        
        data_df, intervened_node, intervention_value = generate_random_intervention_data(
            scm=self.scm,
            n_samples=self.n_samples,
            intervention_values=intervention_values,
            as_tensor=False
        )
        
        # Check the intervened node and value are from the specified options
        self.assertIn(intervened_node, intervention_values.keys())
        self.assertIn(intervention_value, intervention_values[intervened_node])
        
        # Check that the intervened node has the intervention value
        self.assertTrue(np.allclose(data_df[intervened_node], intervention_value))


class TestInterventionMasks(TestDataUtilsBase):
    """Test cases for intervention mask generation."""

    def test_create_intervention_mask(self):
        """Test intervention mask creation."""
        from causal_meta.structure_learning.data_utils import create_intervention_mask
        
        # Create a DataFrame with random values
        df = pd.DataFrame({
            "x0": np.random.randn(self.n_samples),
            "x1": np.random.randn(self.n_samples),
            "x2": np.random.randn(self.n_samples)
        })
        
        # Create an intervention mask for a single node
        mask = create_intervention_mask(
            data=df,
            intervened_nodes=["x0"]
        )
        
        # Check mask shape and type
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (self.n_samples, 3))
        
        # Check mask values (only x0 should be 1, others 0)
        self.assertTrue(np.all(mask[:, 0] == 1))
        self.assertTrue(np.all(mask[:, 1] == 0))
        self.assertTrue(np.all(mask[:, 2] == 0))
        
        # Test with multiple intervened nodes
        mask2 = create_intervention_mask(
            data=df,
            intervened_nodes=["x0", "x2"]
        )
        
        # Check mask values (x0 and x2 should be 1, x1 should be 0)
        self.assertTrue(np.all(mask2[:, 0] == 1))
        self.assertTrue(np.all(mask2[:, 1] == 0))
        self.assertTrue(np.all(mask2[:, 2] == 1))


class TestDataConversion(TestDataUtilsBase):
    """Test cases for data conversion utilities."""

    def test_convert_to_tensor(self):
        """Test conversion of data to PyTorch tensors."""
        from causal_meta.structure_learning.data_utils import convert_to_tensor
        
        # Create a DataFrame with random values
        df = pd.DataFrame({
            "x0": np.random.randn(self.n_samples),
            "x1": np.random.randn(self.n_samples),
            "x2": np.random.randn(self.n_samples)
        })
        
        # Convert to tensor
        tensor = convert_to_tensor(df)
        
        # Check tensor shape and type
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (self.n_samples, 3))
        
        # Check values are preserved
        self.assertTrue(np.allclose(tensor.numpy(), df.values))
        
        # Test with intervention mask
        mask = np.zeros((self.n_samples, 3))
        mask[:, 0] = 1  # Intervene on x0
        
        tensor, mask_tensor = convert_to_tensor(df, intervention_mask=mask)
        
        # Check tensor and mask shapes
        self.assertEqual(tensor.shape, (self.n_samples, 3))
        self.assertEqual(mask_tensor.shape, (self.n_samples, 3))
        
        # Check mask values
        self.assertTrue(torch.all(mask_tensor[:, 0] == 1))
        self.assertTrue(torch.all(mask_tensor[:, 1] == 0))
        self.assertTrue(torch.all(mask_tensor[:, 2] == 0))


if __name__ == "__main__":
    unittest.main() 