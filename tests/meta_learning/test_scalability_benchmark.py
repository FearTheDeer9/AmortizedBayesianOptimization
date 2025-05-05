import pytest
import numpy as np
import torch
import os
import shutil
import tempfile
from typing import Dict, Any

from causal_meta.meta_learning.benchmark import ScalabilityBenchmark
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner

# Define a simple dummy model for testing
class DummyDiscoveryModel:
    """Mock model for testing graph structure learning."""
    
    def __init__(self, complexity: str = "linear"):
        self.complexity = complexity
        
    def learn_graph(self, data):
        """Return a mock graph."""
        # Simulate different computational complexity
        n_features = data.shape[1]
        
        # Sleep to simulate computation time that scales with complexity
        import time
        
        if self.complexity == "linear":
            time.sleep(0.001 * n_features)
        elif self.complexity == "quadratic":
            time.sleep(0.0001 * n_features**2)
        elif self.complexity == "exponential":
            time.sleep(0.0001 * (1.5**n_features))
        
        # Return a random adjacency matrix as the result
        adj_matrix = np.random.rand(n_features, n_features) > 0.8
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

# Define a simple dummy optimization model
class DummyCBOModel:
    def __init__(self, complexity: str = "linear"):
        self.complexity = complexity
    
    def optimize(self, graph, scm, obs_data, target_node, potential_targets, 
                 intervention_ranges, objective_fn):
        # Simulate different computational complexity
        n_nodes = graph.num_nodes
        
        # Sleep to simulate computation time that scales with complexity
        import time
        
        if self.complexity == "linear":
            time.sleep(0.001 * n_nodes)
        elif self.complexity == "quadratic":
            time.sleep(0.0001 * n_nodes**2)
        elif self.complexity == "exponential":
            time.sleep(0.0001 * (1.5**n_nodes))
        
        # Return a random intervention as the result
        interventions = {}
        for node in potential_targets:
            min_val, max_val = intervention_ranges[node]
            interventions[node] = np.random.uniform(min_val, max_val)
        
        return {
            "best_intervention": interventions,
            "best_value": np.random.rand(),
            "num_evaluations": 10
        }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for benchmark results."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_scalability_benchmark_initialization():
    """Test that the ScalabilityBenchmark can be initialized correctly."""
    benchmark = ScalabilityBenchmark(
        name="test_benchmark",
        min_nodes=5,
        max_nodes=15,
        step_size=5
    )
    
    assert benchmark.name == "test_benchmark"
    assert benchmark.min_nodes == 5
    assert benchmark.max_nodes == 15
    assert benchmark.step_size == 5
    assert benchmark.node_sizes == [5, 10, 15]


def test_scalability_benchmark_setup(temp_dir):
    """Test that setup creates the correct test problems."""
    benchmark = ScalabilityBenchmark(
        name="test_benchmark",
        output_dir=temp_dir,
        min_nodes=5,
        max_nodes=10,
        step_size=5,
        num_graphs_per_size=2
    )
    
    benchmark.setup()
    
    # Verify that test problems were created
    assert 5 in benchmark.test_problems
    assert 10 in benchmark.test_problems
    assert len(benchmark.test_problems[5]) == 2
    assert len(benchmark.test_problems[10]) == 2


def test_benchmark_with_models(temp_dir):
    """Test running the benchmark with different models."""
    # Create models with different computational complexity
    models = {
        "linear_model": DummyDiscoveryModel("linear"),
        "quadratic_model": DummyDiscoveryModel("quadratic"),
    }
    
    # Create benchmark runner
    runner = BenchmarkRunner(output_dir=temp_dir)
    
    # Register models
    for name, model in models.items():
        runner.register_model(name, model)
    
    # Create a small scalability suite for testing
    benchmark_ids = runner.create_scalability_suite(
        min_nodes=5,
        max_nodes=15,
        step_size=5,
        num_graphs_per_size=1,
        num_samples=100
    )
    
    assert len(benchmark_ids) == 1
    
    # Run the benchmark
    results = runner.run_scalability_analysis(
        benchmark_ids[0],
        plot_metrics=False,  # Don't display plots during tests
        save_plots=True
    )
    
    # Verify results
    assert "models" in results
    assert "linear_model" in results["models"]
    assert "quadratic_model" in results["models"]
    
    # Check that analysis was performed
    benchmark = runner.benchmarks[benchmark_ids[0]]
    assert hasattr(benchmark, "scaling_analysis")
    
    # Check that report was generated
    report_path = os.path.join(benchmark.benchmark_dir, "scalability_report.json")
    assert os.path.exists(report_path)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 