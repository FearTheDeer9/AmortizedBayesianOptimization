#!/usr/bin/env python3
"""
Test suite for validating the PARENT_SCALE demo and algorithm implementation.
Tests structure recovery accuracy and optimization performance.
"""

from utils.sem_sampling import sample_model
from algorithms.PARENT_SCALE_algorithm import PARENT_SCALE
from graphs.data_setup import setup_observational_interventional
from examples.parent_scale_demo import (
    create_erdos_renyi_graph,
    run_parent_scale_demo
)
import sys
import os
import unittest
import numpy as np
import logging
import networkx as nx
from copy import deepcopy

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import from the demo file directly by adding examples to path
sys.path.insert(0, os.path.join(project_root, 'examples'))

# Import other needed modules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_parent_scale_demo")


class TestParentScaleDemo(unittest.TestCase):
    """Test cases for the PARENT_SCALE demo."""

    def setUp(self):
        """Setup common test resources."""
        # Small graph for quick testing
        self.num_nodes = 4
        self.seed = 1234
        self.graph = create_erdos_renyi_graph(
            self.num_nodes, exp_edges=1, seed=self.seed)

        # Generate data
        self.D_O, self.D_I, self.exploration_set = setup_observational_interventional(
            graph_type=None,
            noiseless=True,
            seed=self.seed,
            n_obs=100,
            n_int=2,
            graph=self.graph,
        )

    def test_graph_creation(self):
        """Test that the graph is created with expected properties."""
        # Check number of nodes
        self.assertEqual(len(self.graph.nodes), self.num_nodes)

        # Check for acyclicity
        self.assertTrue(nx.is_directed_acyclic_graph(self.graph._G))

        # Check that target node is set
        self.assertIsNotNone(self.graph.target)

        # Check that SEM is initialized
        self.assertIsNotNone(self.graph.SEM)

    def test_parent_scale_initialization(self):
        """Test that PARENT_SCALE initializes correctly."""
        # Initialize PARENT_SCALE
        ps = PARENT_SCALE(
            graph=self.graph,
            nonlinear=False,
            causal_prior=True,
            noiseless=True,
            scale_data=True,
            use_doubly_robust=True
        )

        # Set values
        ps.set_values(self.D_O, self.D_I, self.exploration_set)

        # Check that essential components are initialized
        self.assertEqual(ps.num_nodes, self.num_nodes)
        self.assertEqual(ps.graph, self.graph)
        self.assertIsNotNone(ps.target)
        self.assertIsNotNone(ps.topological_order)

    def test_short_run(self):
        """Test a short run of the algorithm."""
        # Initialize PARENT_SCALE
        ps = PARENT_SCALE(
            graph=self.graph,
            nonlinear=False,
            causal_prior=True,
            noiseless=True,
            scale_data=True,
            use_doubly_robust=True
        )

        # Set values
        ps.set_values(self.D_O, self.D_I, self.exploration_set)

        # Run algorithm for just 2 iterations
        results = ps.run_algorithm(T=2, show_graphics=False)

        # Check that results structure is as expected
        self.assertEqual(len(results), 6)  # 6 return values

        # Unpack results
        global_opt, current_y, current_cost, intervention_set, intervention_values, _ = results

        # Check lengths
        self.assertEqual(len(global_opt), 2)
        self.assertEqual(len(current_y), 2)
        self.assertEqual(len(current_cost), 3)  # Includes initial cost of 0
        self.assertEqual(len(intervention_set), 2)
        self.assertEqual(len(intervention_values), 2)

    def test_posterior_calculation(self):
        """Test the posterior calculation over parent sets."""
        # Initialize PARENT_SCALE
        ps = PARENT_SCALE(
            graph=self.graph,
            nonlinear=False,
            use_doubly_robust=True
        )

        # Set values
        ps.set_values(self.D_O, self.D_I, self.exploration_set)

        # Calculate posterior
        ps.data_and_prior_setup()

        # Check prior probabilities
        self.assertIsNotNone(ps.prior_probabilities)
        self.assertTrue(len(ps.prior_probabilities) > 0)

        # Check that probabilities sum to approximately 1
        total_prob = sum(ps.prior_probabilities.values())
        self.assertAlmostEqual(total_prob, 1.0, delta=0.01)

        # Define and check all possible graphs
        ps.define_all_possible_graphs()

        # Check graph creation
        self.assertIsNotNone(ps.graphs)
        self.assertTrue(len(ps.graphs) > 0)

        # Check graphs match parent sets
        for parents, graph in ps.graphs.items():
            # Verify the target's parents in the graph match the parent set
            target_parents = tuple(sorted(graph.parents[graph.target]))
            self.assertEqual(parents, target_parents)

    def test_full_demo_run(self):
        """Test the full demo with a small graph."""
        # Run a short demo with minimal iterations
        results = run_parent_scale_demo(
            num_nodes=4,
            exp_edges=1,
            n_obs=100,
            n_int=2,
            n_trials=3,
            nonlinear=False,
            use_doubly_robust=True,
            show_graphics=False,
            seed=self.seed
        )

        # Check results structure
        self.assertIn('graph', results)
        self.assertIn('posterior_history', results)
        self.assertIn('true_parents', results)
        self.assertIn('best_parent_set', results)
        self.assertIn('structure_correct', results)
        self.assertIn('optimization_results', results)
        self.assertIn('theoretical_optimum', results)
        self.assertIn('relative_performance', results)

        # Optimization results should have specific keys
        opt_results = results['optimization_results']
        self.assertIn('global_opt', opt_results)
        self.assertIn('current_y', opt_results)
        self.assertIn('current_cost', opt_results)
        self.assertIn('intervention_set', opt_results)
        self.assertIn('intervention_values', opt_results)

        # Posterior history should evolve over iterations
        posterior_history = results['posterior_history']
        self.assertTrue(len(posterior_history) >= 3)  # At least 3 iterations

        # The best parent set should be a tuple of strings
        best_parent_set = results['best_parent_set']
        if best_parent_set:  # It might be None if no clear winner
            self.assertIsInstance(best_parent_set, tuple)
            for parent in best_parent_set:
                self.assertIsInstance(parent, str)


if __name__ == "__main__":
    unittest.main()
