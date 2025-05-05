"""
Unit tests for the benchmark suite.

This module tests the functionality of the benchmark classes and utilities.
"""

import os
import torch
import numpy as np
import pandas as pd
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.meta_learning.benchmark import Benchmark, CausalDiscoveryBenchmark, CBOBenchmark
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner


class MockGraphModel:
    """Mock model for testing graph structure learning."""
    
    def __init__(self, graph_size=5, success_rate=0.8):
        self.graph_size = graph_size
        self.success_rate = success_rate
    
    def learn_graph(self, obs_data, int_data=None):
        """Return a mock graph."""
        # Create a random graph with some structure
        adj_matrix = np.zeros((self.graph_size, self.graph_size))
        
        # Add some edges
        for i in range(self.graph_size):
            for j in range(i+1, self.graph_size):
                if np.random.random() < 0.3:
                    adj_matrix[i, j] = 1
        
        # Create a new CausalGraph
        graph = CausalGraph()
        
        # Add nodes
        for i in range(self.graph_size):
            node_name = f"X{i}"
            graph.add_node(node_name)
        
        # Add edges based on adjacency matrix
        for i in range(self.graph_size):
            for j in range(self.graph_size):
                if adj_matrix[i, j] > 0:
                    graph.add_edge(f"X{i}", f"X{j}")
        
        return graph


class MockCBOModel:
    """Mock model for testing causal Bayesian optimization."""
    
    def __init__(self, success_rate=0.8):
        self.success_rate = success_rate
    
    def optimize(
        self,
        scm,
        graph,
        obs_data,
        target_node,
        potential_targets,
        intervention_ranges,
        objective_fn,
        num_iterations=10,
        maximize=True
    ):
        """Return mock optimization results."""
        # Create a mock intervention
        intervention = {}
        for target in np.random.choice(potential_targets, size=min(2, len(potential_targets)), replace=False):
            min_val, max_val = intervention_ranges[target]
            intervention[target] = np.random.uniform(min_val, max_val)
        
        # Evaluate it
        value = objective_fn(intervention)
        
        return {
            "best_intervention": intervention,
            "best_value": value,
            "num_evaluations": num_iterations,
            "intervention_sequence": [intervention],
            "value_sequence": [value]
        }


class TestBenchmark(unittest.TestCase):
    """Test the abstract Benchmark base class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a concrete subclass for testing
        class ConcreteBenchmark(Benchmark):
            def setup(self):
                pass
                
            def run(self):
                return {"result": "success"}
        
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = ConcreteBenchmark(
            name="test_benchmark",
            output_dir=self.temp_dir,
            seed=42
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test benchmark initialization."""
        self.assertEqual(self.benchmark.name, "test_benchmark")
        self.assertEqual(self.benchmark.output_dir, self.temp_dir)
        self.assertEqual(self.benchmark.seed, 42)
        self.assertTrue(os.path.exists(self.benchmark.benchmark_dir))
    
    def test_add_model(self):
        """Test adding a model to the benchmark."""
        model = MockGraphModel()
        self.benchmark.add_model("test_model", model)
        self.assertIn("test_model", self.benchmark.models)
        self.assertEqual(self.benchmark.models["test_model"], model)
    
    def test_add_baseline(self):
        """Test adding a baseline to the benchmark."""
        baseline = MockGraphModel()
        self.benchmark.add_baseline("test_baseline", baseline)
        self.assertIn("test_baseline", self.benchmark.baselines)
        self.assertEqual(self.benchmark.baselines["test_baseline"], baseline)
    
    def test_save_load_results(self):
        """Test saving and loading results."""
        # Create test results
        test_results = {
            "metric1": 0.5,
            "metric2": 0.8,
            "array": np.array([1, 2, 3])
        }
        
        # Save results
        path = self.benchmark.save_results(test_results)
        self.assertTrue(os.path.exists(path))
        
        # Load results
        loaded_results = self.benchmark.load_results()
        self.assertEqual(loaded_results["metric1"], 0.5)
        self.assertEqual(loaded_results["metric2"], 0.8)
        self.assertEqual(loaded_results["array"], [1, 2, 3])


class TestCausalDiscoveryBenchmark(unittest.TestCase):
    """Test the CausalDiscoveryBenchmark class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = CausalDiscoveryBenchmark(
            name="test_cd_benchmark",
            output_dir=self.temp_dir,
            seed=42,
            num_nodes=3,  # Small graph for testing
            num_graphs=2,  # Few graphs for testing
            num_samples=100,  # Few samples for testing
            graph_type="random"  # Use "random" instead of default
        )
        
        # Add a mock model
        self.model = MockGraphModel(graph_size=3)
        self.benchmark.add_model("test_model", self.model)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_setup(self):
        """Test benchmark setup."""
        self.benchmark.setup()
        
        # Check if graphs were generated
        self.assertEqual(len(self.benchmark.graphs), 2)
        
        # Check if SCMs were created
        self.assertEqual(len(self.benchmark.scms), 2)
        
        # Check if datasets were created
        self.assertEqual(len(self.benchmark.datasets), 2)
        
        # Check content of datasets
        for dataset_name, dataset in self.benchmark.datasets.items():
            self.assertIn("graph", dataset)
            self.assertIn("scm", dataset)
            self.assertIn("observational_data", dataset)
            
            # Check observational data
            self.assertIsInstance(dataset["observational_data"], pd.DataFrame)
            self.assertEqual(dataset["observational_data"].shape[0], 100)  # num_samples
            self.assertEqual(dataset["observational_data"].shape[1], 3)  # num_nodes
    
    def test_run(self):
        """Test running the benchmark."""
        self.benchmark.setup()
        results = self.benchmark.run()
        
        # Check results structure
        self.assertIn("benchmark_config", results)
        self.assertIn("models", results)
        self.assertIn("baselines", results)
        self.assertIn("aggregated", results)
        
        # Check model results
        self.assertIn("test_model", results["models"])
        
        # Check aggregated results
        self.assertIn("models", results["aggregated"])
        self.assertIn("test_model", results["aggregated"]["models"])
        
        # Check metrics
        metrics = results["aggregated"]["models"]["test_model"]
        self.assertIn("shd", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("runtime", metrics)


class TestCBOBenchmark(unittest.TestCase):
    """Test the CBOBenchmark class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark = CBOBenchmark(
            name="test_cbo_benchmark",
            output_dir=self.temp_dir,
            seed=42,
            num_nodes=3,  # Small graph for testing
            num_graphs=2,  # Few graphs for testing
            num_samples=100,  # Few samples for testing
            intervention_budget=3,  # Few interventions for testing
            graph_type="random"  # Use "random" instead of default
        )
        
        # Add a mock model
        self.model = MockCBOModel()
        self.benchmark.add_model("test_model", self.model)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_setup(self):
        """Test benchmark setup."""
        self.benchmark.setup()
        
        # Check if optimization problems were defined
        self.assertEqual(len(self.benchmark.optimization_problems), 2)
        
        # Check content of optimization problems
        for problem_name, problem in self.benchmark.optimization_problems.items():
            self.assertIn("graph", problem)
            self.assertIn("scm", problem)
            self.assertIn("observational_data", problem)
            self.assertIn("target_node", problem)
            self.assertIn("potential_targets", problem)
            self.assertIn("intervention_ranges", problem)
            self.assertIn("objective_fn", problem)
            
            # Check observational data
            self.assertIsInstance(problem["observational_data"], pd.DataFrame)
            self.assertEqual(problem["observational_data"].shape[0], 100)  # num_samples
            self.assertEqual(problem["observational_data"].shape[1], 3)  # num_nodes
            
            # Check target node format
            self.assertTrue(problem["target_node"].startswith("X"))
            
            # Check potential targets
            self.assertIsInstance(problem["potential_targets"], list)
            
            # Check intervention ranges
            self.assertIsInstance(problem["intervention_ranges"], dict)
            
            # Check objective function
            self.assertTrue(callable(problem["objective_fn"]))
    
    def test_run(self):
        """Test running the benchmark."""
        self.benchmark.setup()
        results = self.benchmark.run()
        
        # Check results structure
        self.assertIn("benchmark_config", results)
        self.assertIn("models", results)
        self.assertIn("baselines", results)
        self.assertIn("aggregated", results)
        
        # Check model results
        self.assertIn("test_model", results["models"])
        
        # Check aggregated results
        self.assertIn("models", results["aggregated"])
        self.assertIn("test_model", results["aggregated"]["models"])
        
        # Check metrics
        metrics = results["aggregated"]["models"]["test_model"]
        self.assertIn("best_value", metrics)
        self.assertIn("improvement_over_random", metrics)
        self.assertIn("runtime", metrics)
        self.assertIn("num_evaluations", metrics)


class TestBenchmarkRunner(unittest.TestCase):
    """Test the BenchmarkRunner class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = BenchmarkRunner(
            name="test_runner",
            output_dir=self.temp_dir,
            seed=42
        )
        
        # Create a small benchmark
        self.benchmark = CausalDiscoveryBenchmark(
            name="test_cd_benchmark",
            output_dir=self.temp_dir,
            seed=42,
            num_nodes=3,
            num_graphs=1,
            num_samples=100,
            graph_type="random"  # Use "random" instead of default
        )
        
        # Add a mock model
        self.model = MockGraphModel(graph_size=3)
        
        # Add benchmark and model to runner
        self.runner.add_benchmark(self.benchmark)
        self.runner.add_model("test_model", self.model)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_benchmark(self):
        """Test adding a benchmark to the runner."""
        self.assertIn("test_cd_benchmark", self.runner.benchmarks)
        self.assertEqual(self.runner.benchmarks["test_cd_benchmark"], self.benchmark)
    
    def test_add_model(self):
        """Test adding a model to the runner."""
        self.assertIn("test_model", self.runner.models)
        self.assertEqual(self.runner.models["test_model"], self.model)
    
    def test_run_all(self):
        """Test running all benchmarks."""
        results = self.runner.run_all()
        
        # Check results structure
        self.assertIn("run_info", results)
        self.assertIn("benchmarks", results)
        self.assertIn("test_cd_benchmark", results["benchmarks"])
    
    def test_create_standard_suite(self):
        """Test creating a standard benchmark suite."""
        suite = BenchmarkRunner.create_standard_suite(
            name="test_suite",
            output_dir=self.temp_dir,
            seed=42,
            graph_sizes=[3],
            num_graphs=1,
            num_samples=50
        )
        
        # Check if benchmarks were created
        self.assertGreater(len(suite.benchmarks), 0)
        
        # Check if there are both CD and CBO benchmarks
        has_cd = False
        has_cbo = False
        
        for name in suite.benchmarks:
            if "causal_discovery" in name:
                has_cd = True
            if "cbo" in name:
                has_cbo = True
        
        self.assertTrue(has_cd)
        self.assertTrue(has_cbo)


if __name__ == "__main__":
    unittest.main() 