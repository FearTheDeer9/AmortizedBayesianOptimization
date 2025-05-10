"""Tests for the RandomDAGGenerator class."""

import unittest
import numpy as np
import networkx as nx
from typing import List, Tuple

from causal_meta.structure_learning.graph_generators import RandomDAGGenerator
from causal_meta.graph import CausalGraph


class TestRandomDAGGenerator(unittest.TestCase):
    """Test cases for the RandomDAGGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_nodes = 5
        self.edge_probability = 0.3
        self.seed = 42

    def test_generate_random_dag_adjacency_matrix(self):
        """Test generating a random DAG as an adjacency matrix."""
        # Generate a random DAG
        adj_matrix = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            as_adjacency_matrix=True,
            seed=self.seed,
        )

        # Check shape
        self.assertEqual(adj_matrix.shape, (self.num_nodes, self.num_nodes))
        
        # Check type
        self.assertIsInstance(adj_matrix, np.ndarray)
        
        # Check acyclicity
        self.assertTrue(RandomDAGGenerator.validate_acyclicity(adj_matrix))
        
        # Check diagonal (no self-loops)
        self.assertTrue(np.all(np.diag(adj_matrix) == 0))
        
        # Check that it's upper triangular (to ensure acyclicity)
        # This assumes node ordering is preserved from the graph generation
        self.assertTrue(np.allclose(adj_matrix, np.triu(adj_matrix)))

    def test_generate_random_dag_causal_graph(self):
        """Test generating a random DAG as a CausalGraph."""
        # Generate a random DAG
        graph = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            as_adjacency_matrix=False,
            seed=self.seed,
        )

        # Check type
        self.assertIsInstance(graph, CausalGraph)
        
        # Check number of nodes
        self.assertEqual(len(graph.get_nodes()), self.num_nodes)
        
        # Check acyclicity
        self.assertTrue(graph.is_acyclic())
        
        # Check adjacency matrix
        adj_matrix = np.array(graph.get_adjacency_matrix())
        self.assertEqual(adj_matrix.shape, (self.num_nodes, self.num_nodes))
        
        # Check that it matches the direct adjacency matrix generation
        adj_matrix_direct = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            as_adjacency_matrix=True,
            seed=self.seed,
        )
        self.assertTrue(np.array_equal(adj_matrix, adj_matrix_direct))

    def test_validate_acyclicity(self):
        """Test acyclicity validation."""
        # Create a valid DAG
        dag_matrix = np.zeros((3, 3))
        dag_matrix[0, 1] = 1
        dag_matrix[1, 2] = 1
        self.assertTrue(RandomDAGGenerator.validate_acyclicity(dag_matrix))
        
        # Create a graph with a cycle
        cycle_matrix = np.zeros((3, 3))
        cycle_matrix[0, 1] = 1
        cycle_matrix[1, 2] = 1
        cycle_matrix[2, 0] = 1  # This creates a cycle
        self.assertFalse(RandomDAGGenerator.validate_acyclicity(cycle_matrix))

    def test_random_seed_reproducibility(self):
        """Test that the random seed produces reproducible results."""
        # Generate two DAGs with the same seed
        adj_matrix1 = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            seed=self.seed,
        )
        
        adj_matrix2 = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            seed=self.seed,
        )
        
        # They should be identical
        self.assertTrue(np.array_equal(adj_matrix1, adj_matrix2))
        
        # Generate a DAG with a different seed
        adj_matrix3 = RandomDAGGenerator.generate_random_dag(
            num_nodes=self.num_nodes,
            edge_probability=self.edge_probability,
            seed=self.seed + 1,
        )
        
        # It should be different
        self.assertFalse(np.array_equal(adj_matrix1, adj_matrix3))

    def test_edge_probability_effect(self):
        """Test that edge probability affects the number of edges."""
        # Generate DAGs with different edge probabilities
        # Very sparse
        adj_matrix_sparse = RandomDAGGenerator.generate_random_dag(
            num_nodes=10,
            edge_probability=0.1,
            seed=self.seed,
        )
        
        # Very dense
        adj_matrix_dense = RandomDAGGenerator.generate_random_dag(
            num_nodes=10,
            edge_probability=0.8,
            seed=self.seed,
        )
        
        # Count edges
        edge_count_sparse = np.sum(adj_matrix_sparse)
        edge_count_dense = np.sum(adj_matrix_dense)
        
        # Dense should have more edges
        self.assertLess(edge_count_sparse, edge_count_dense)


if __name__ == "__main__":
    unittest.main() 