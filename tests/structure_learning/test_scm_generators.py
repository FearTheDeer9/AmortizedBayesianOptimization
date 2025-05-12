"""Tests for the LinearSCMGenerator class."""

import unittest
import numpy as np
import pandas as pd

from causal_meta.structure_learning.scm_generators import LinearSCMGenerator
from causal_meta.structure_learning.graph_generators import RandomDAGGenerator
from causal_meta.environments.scm import StructuralCausalModel


class TestLinearSCMGenerator(unittest.TestCase):
    """Test cases for the LinearSCMGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG adjacency matrix
        self.adj_matrix = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.node_names = [f"x{i}" for i in range(3)]
        self.noise_scale = 0.1
        self.seed = 42
        self.n_samples = 100

    def test_generate_linear_scm(self):
        """Test generating a linear SCM from an adjacency matrix."""
        # Generate an SCM
        scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=self.adj_matrix,
            noise_scale=self.noise_scale,
            seed=self.seed
        )
        
        # Check type
        self.assertIsInstance(scm, StructuralCausalModel)
        
        # Check structure matches adjacency matrix
        graph = scm._causal_graph
        self.assertEqual(len(graph.get_nodes()), 3)
        
        # Check edges
        self.assertTrue(graph.has_edge(self.node_names[0], self.node_names[1]))
        self.assertTrue(graph.has_edge(self.node_names[0], self.node_names[2]))
        self.assertTrue(graph.has_edge(self.node_names[1], self.node_names[2]))
        self.assertEqual(len(graph.get_edges()), 3)
        
        # Generate data
        data = scm.sample_data(self.n_samples, as_array=False)
        
        # Check data shape
        self.assertEqual(data.shape, (self.n_samples, 3))
        
        # Check data types
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check columns match node names
        self.assertListEqual(list(data.columns), self.node_names)
        
        # Check correlations (parents should correlate with children)
        corr = data.corr().abs()
        # x0 -> x1, so they should correlate
        self.assertGreater(corr.loc["x0", "x1"], 0.1)
        # x0 -> x2, so they should correlate
        self.assertGreater(corr.loc["x0", "x2"], 0.1)
        # x1 -> x2, so they should correlate
        self.assertGreater(corr.loc["x1", "x2"], 0.1)

    def test_generate_linear_weights(self):
        """Test generating random weights for a linear SCM."""
        # Generate weights with default range
        weights = LinearSCMGenerator.generate_linear_weights(
            adj_matrix=self.adj_matrix,
            random_state=np.random.RandomState(self.seed)
        )
        
        # Check shape
        self.assertEqual(weights.shape, self.adj_matrix.shape)
        
        # Check range
        nonzero_weights = weights[weights != 0]
        self.assertTrue(np.all(nonzero_weights >= -2.0))
        self.assertTrue(np.all(nonzero_weights <= 2.0))
        
        # Check only edges in adjacency matrix have weights
        for i in range(3):
            for j in range(3):
                if self.adj_matrix[i, j] > 0:
                    self.assertNotEqual(weights[i, j], 0)
                else:
                    self.assertEqual(weights[i, j], 0)

    def test_add_noise_distributions(self):
        """Test adding noise distributions to an SCM."""
        # Create an SCM
        scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=self.adj_matrix,
            noise_scale=self.noise_scale,
            seed=self.seed
        )
        
        # Add noise distributions
        scm = LinearSCMGenerator.add_noise_distributions(
            scm=scm,
            noise_scale=0.2,  # Different noise scale
            seed=self.seed
        )
        
        # Check noise distributions exist for all variables
        for var in scm.get_variable_names():
            self.assertIn(var, scm._exogenous_functions)

    def test_non_dag_input(self):
        """Test that non-DAG adjacency matrices are rejected."""
        # Create a cyclic graph (not a DAG)
        cyclic_adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        
        # Should raise a ValueError
        with self.assertRaises(ValueError):
            LinearSCMGenerator.generate_linear_scm(cyclic_adj)

    def test_random_dag_integration(self):
        """Test integration with RandomDAGGenerator."""
        # Generate a random DAG
        num_nodes = 5
        edge_probability = 0.3
        adj_matrix = RandomDAGGenerator.generate_random_dag(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            seed=self.seed
        )
        node_names = [f"x{i}" for i in range(num_nodes)]
        
        # Generate an SCM from the DAG
        scm = LinearSCMGenerator.generate_linear_scm(
            adj_matrix=adj_matrix,
            noise_scale=self.noise_scale,
            seed=self.seed
        )
        
        # Check SCM structure
        self.assertEqual(len(scm.get_variable_names()), num_nodes)
        self.assertListEqual(list(scm.get_variable_names()), node_names)
        
        # Generate data
        data = scm.sample_data(self.n_samples)
        
        # Check data shape
        self.assertEqual(data.shape, (self.n_samples, num_nodes))
        
        # Interventions should work
        scm.do_intervention(node_names[0], 1.0)
        int_data = scm.sample_data(self.n_samples)
        
        # Check intervention effect
        self.assertTrue(np.allclose(int_data[node_names[0]], 1.0))


if __name__ == "__main__":
    unittest.main() 