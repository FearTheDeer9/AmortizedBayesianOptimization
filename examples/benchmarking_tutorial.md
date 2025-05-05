# Benchmarking Framework Tutorial
# ===============================
#
# This file serves as both a tutorial document and executable script.
# You can run it as a Python script or convert it to a Jupyter notebook.
#
# To convert to a notebook:
# ```
# jupyter nbconvert --to notebook --execute benchmarking_tutorial.md
# ```

"""
# Comprehensive Guide to the Benchmarking Framework

This tutorial demonstrates how to use the benchmarking framework to evaluate causal discovery algorithms and causal Bayesian optimization methods.

## Key Features of the Benchmarking Framework

1. **Standard benchmarks** for causal discovery and causal Bayesian optimization
2. **Scalability testing** to evaluate performance across different graph sizes
3. **Memory and runtime profiling** to measure computational requirements
4. **Comprehensive metrics** for performance evaluation
5. **Visualization tools** for analyzing and comparing results
6. **Multi-method comparison** with statistical significance testing
7. **Integration with neural approaches** for evaluating amortized methods

## Setup and Import Required Packages
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import benchmark components
from causal_meta.meta_learning.benchmark import (
    Benchmark, 
    CausalDiscoveryBenchmark, 
    CBOBenchmark, 
    ScalabilityBenchmark
)
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner

# Import graph utilities
from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.graph.generators.factory import GraphFactory

# Import SCM utilities
from causal_meta.environments.scm import StructuralCausalModel

"""
## Part 1: Basic Benchmarking Concepts

The benchmarking framework is built around three main classes:

1. **Benchmark** (abstract base class): Defines the common interface for all benchmarks
2. **CausalDiscoveryBenchmark**: For evaluating graph structure learning algorithms
3. **CBOBenchmark**: For evaluating causal Bayesian optimization methods
4. **ScalabilityBenchmark**: For assessing how methods scale with graph size
5. **BenchmarkRunner**: For orchestrating multiple benchmarks and aggregating results

Each benchmark follows this general workflow:
1. Initialize with configuration parameters
2. Add models and baselines to evaluate
3. Set up the test environment (generate graphs, SCMs, datasets)
4. Run the benchmark
5. Analyze and visualize results
"""

"""
## Part 2: Creating a Simple Causal Discovery Benchmark

Let's start by creating a simple causal discovery benchmark to evaluate a graph structure learning algorithm.
"""

# Define a simple causal discovery method for demonstration
class SimpleCausalDiscoveryModel:
    """
    Simple causal discovery model for demonstration purposes.
    
    This model uses correlation thresholding to infer a causal graph.
    It's not meant to be accurate, just to demonstrate the benchmark framework.
    """
    
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        
    def learn_graph(self, obs_data, int_data=None):
        """Learn a causal graph from observational data."""
        # Calculate correlation matrix
        corr_matrix = obs_data.corr().abs().values
        
        # Create adjacency matrix based on threshold
        adj_matrix = np.zeros_like(corr_matrix)
        n = corr_matrix.shape[0]
        
        for i in range(n):
            for j in range(n):
                if i != j and corr_matrix[i, j] > self.threshold:
                    # Assume direction based on column index (arbitrary but deterministic)
                    if i < j:
                        adj_matrix[i, j] = 1
        
        # Create a new CausalGraph with named nodes
        graph = CausalGraph()
        
        # Add nodes
        for i, col in enumerate(obs_data.columns):
            graph.add_node(col)
        
        # Add edges based on adjacency matrix
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:
                    graph.add_edge(obs_data.columns[i], obs_data.columns[j])
        
        return graph

# Create a benchmark for small graphs (quick execution)
def run_simple_cd_benchmark():
    print("\n=== Simple Causal Discovery Benchmark ===")
    
    # Create the benchmark
    benchmark = CausalDiscoveryBenchmark(
        name="simple_cd_benchmark",
        seed=42,
        num_nodes=5,        # 5-node graphs
        num_graphs=3,       # 3 test graphs (more would be better for real evaluation)
        num_samples=500,    # 500 samples per dataset
        graph_type="random",
        edge_prob=0.3
    )
    
    # Add models to evaluate
    benchmark.add_model("correlation_0.3", SimpleCausalDiscoveryModel(threshold=0.3))
    benchmark.add_model("correlation_0.5", SimpleCausalDiscoveryModel(threshold=0.5))
    
    # Setup and run the benchmark
    print("Setting up benchmark...")
    benchmark.setup()
    print("Running benchmark...")
    results = benchmark.run()
    
    # Plot results
    print("Plotting results...")
    benchmark.plot_results(
        metrics=["shd", "precision", "recall"],
        title="Simple Causal Discovery Benchmark"
    )
    
    print(f"Results summary:")
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "summary" in model_results:
            summary = model_results["summary"]
            print(f"  {model_name}:")
            print(f"    SHD: {summary.get('shd', 'N/A'):.2f}")
            print(f"    Precision: {summary.get('precision', 'N/A'):.2f}")
            print(f"    Recall: {summary.get('recall', 'N/A'):.2f}")
    
    return benchmark, results

"""
## Part 3: Creating a Causal Bayesian Optimization Benchmark

Now let's create a benchmark for evaluating causal Bayesian optimization methods.
"""

class SimpleInterventionModel:
    """
    Simple intervention optimization model for demonstration purposes.
    
    This model randomly selects interventions within the given ranges.
    It's not meant to be effective, just to demonstrate the benchmark framework.
    """
    
    def optimize(self, scm, graph, obs_data, target_node, potential_targets, 
                intervention_ranges, objective_fn, num_iterations=10, maximize=True):
        """Run a simple optimization procedure."""
        # Track best intervention and value
        best_intervention = {}
        best_value = float('-inf') if maximize else float('inf')
        
        # Track all interventions and values
        intervention_sequence = []
        value_sequence = []
        
        # Try random interventions
        for _ in range(num_iterations):
            # Select a random subset of targets
            num_targets = min(3, len(potential_targets))
            targets = np.random.choice(potential_targets, size=num_targets, replace=False)
            
            # Create intervention
            intervention = {}
            for target in targets:
                min_val, max_val = intervention_ranges[target]
                intervention[target] = np.random.uniform(min_val, max_val)
            
            # Evaluate
            value = objective_fn(intervention)
            
            # Update best found
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                best_intervention = intervention.copy()
            
            # Track history
            intervention_sequence.append(intervention)
            value_sequence.append(value)
        
        return {
            "best_intervention": best_intervention,
            "best_value": best_value,
            "num_evaluations": num_iterations,
            "intervention_sequence": intervention_sequence,
            "value_sequence": value_sequence
        }

def run_simple_cbo_benchmark():
    print("\n=== Simple Causal Bayesian Optimization Benchmark ===")
    
    # Create the benchmark
    benchmark = CBOBenchmark(
        name="simple_cbo_benchmark",
        seed=42,
        num_nodes=5,        # 5-node graphs
        num_graphs=2,       # 2 test graphs (more would be better for real evaluation)
        num_samples=500,    # 500 samples per dataset
        graph_type="random",
        edge_prob=0.3,
        intervention_budget=10
    )
    
    # Add models to evaluate
    benchmark.add_model("random_search_10", SimpleInterventionModel())
    benchmark.add_model("random_search_20", 
                       lambda *args, **kwargs: SimpleInterventionModel().optimize(
                           *args, **kwargs, num_iterations=20
                       ))
    
    # Setup and run the benchmark
    print("Setting up benchmark...")
    benchmark.setup()
    print("Running benchmark...")
    results = benchmark.run()
    
    # Plot results
    print("Plotting results...")
    benchmark.plot_results(
        metrics=["best_value", "improvement_ratio", "runtime"],
        title="Simple CBO Benchmark Results"
    )
    
    print(f"Results summary:")
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "summary" in model_results:
            summary = model_results["summary"]
            print(f"  {model_name}:")
            print(f"    Best Value (avg): {summary.get('best_value', 'N/A'):.4f}")
            print(f"    Improvement Ratio: {summary.get('improvement_ratio', 'N/A'):.2f}")
            print(f"    Runtime (s): {summary.get('runtime', 'N/A'):.2f}")
    
    return benchmark, results

"""
## Part 4: Scalability Benchmarking

The `ScalabilityBenchmark` class allows us to evaluate how methods scale with increasing graph size.
"""

def run_scalability_benchmark():
    print("\n=== Scalability Benchmark ===")
    
    # Create the benchmark
    benchmark = ScalabilityBenchmark(
        name="scalability_benchmark",
        seed=42,
        min_nodes=5,              # Start with 5-node graphs
        max_nodes=15,             # Go up to 15-node graphs
        step_size=5,              # Test 5, 10, 15 node graphs
        num_graphs_per_size=2,    # 2 graphs per size (more would be better for real evaluation)
        num_samples=500,          # 500 samples per dataset
        graph_type="random",
        measure_mode="discovery"  # Only measure causal discovery (not CBO)
    )
    
    # Add models to evaluate
    benchmark.add_model("correlation_0.3", SimpleCausalDiscoveryModel(threshold=0.3))
    benchmark.add_model("correlation_0.5", SimpleCausalDiscoveryModel(threshold=0.5))
    
    # Setup and run the benchmark
    print("Setting up benchmark...")
    benchmark.setup()
    print("Running benchmark...")
    results = benchmark.run()
    
    # Plot scaling curves
    print("Plotting scaling curves...")
    benchmark.plot_scaling_curves(
        metric="runtime",
        log_scale=True,
        save_path="scaling_runtime.png"
    )
    
    # Generate scaling report
    print("Generating scaling report...")
    scaling_report = benchmark.generate_scaling_report()
    
    print(f"Scaling analysis summary:")
    for model_name, scaling_data in scaling_report.items():
        if "complexity_class" in scaling_data:
            complexity = scaling_data["complexity_class"]
            print(f"  {model_name} scaling complexity: {complexity}")
    
    return benchmark, results

"""
## Part 5: Using BenchmarkRunner for Multi-Benchmark Evaluation

The `BenchmarkRunner` allows you to run multiple benchmarks and aggregate results.
"""

def run_benchmark_suite():
    print("\n=== Benchmark Suite with BenchmarkRunner ===")
    
    # Create a benchmark runner
    runner = BenchmarkRunner(
        name="tutorial_benchmark_run",
        seed=42
    )
    
    # Add models to evaluate
    runner.add_model("correlation_0.3", SimpleCausalDiscoveryModel(threshold=0.3))
    runner.add_model("correlation_0.5", SimpleCausalDiscoveryModel(threshold=0.5))
    runner.add_model("random_search", SimpleInterventionModel())
    
    # Create a standard benchmark suite (returns benchmark IDs)
    print("Creating benchmark suite...")
    benchmark_ids = runner.create_standard_suite(
        graph_sizes=[5, 10],  # Test with 5 and 10 node graphs
        num_graphs=2,         # 2 graphs per size
        num_samples=500       # 500 samples per dataset
    )
    
    # Run all benchmarks
    print("Running all benchmarks...")
    suite_results = runner.run_all()
    
    # Generate summary report
    print("Generating summary report...")
    summary_path = runner.generate_summary_report()
    
    print(f"Benchmark suite results:")
    for benchmark_id, benchmark_results in suite_results.items():
        print(f"  {benchmark_id} results:")
        for model_name in benchmark_results.get("models", []):
            if "summary" in benchmark_results.get(model_name, {}):
                summary = benchmark_results[model_name]["summary"]
                if "shd" in summary:
                    print(f"    {model_name} SHD: {summary['shd']:.2f}")
                if "best_value" in summary:
                    print(f"    {model_name} Best Value: {summary['best_value']:.4f}")
    
    return runner, suite_results

"""
## Part 6: Integrating Neural Methods for Amortized Evaluation

The benchmarking framework supports neural network-based methods, including amortized causal discovery and amortized causal Bayesian optimization.
"""

def create_mock_amortized_model():
    """Create a mock amortized neural model for testing."""
    
    class MockGraphEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Mock encoder with a single linear layer
            self.linear = torch.nn.Linear(10, 5)
            
        def forward(self, x):
            # Return random edge probabilities
            batch_size = x.shape[0]
            num_nodes = 5  # Assuming 5 nodes for this example
            edge_probs = torch.sigmoid(torch.randn(batch_size, num_nodes, num_nodes))
            return edge_probs
    
    class MockDynamicsDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Mock decoder
            
        def forward(self, data, graph, interventions=None):
            # Return predictions with the same shape as input data
            return data + 0.1 * torch.randn_like(data)
    
    # Create a simple callable object for testing
    class MockAmortizedModel:
        def __init__(self):
            self.graph_encoder = MockGraphEncoder()
            self.dynamics_decoder = MockDynamicsDecoder()
            
        def infer_graph(self, x):
            # Ensure x is a tensor
            if not isinstance(x, torch.Tensor):
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                elif isinstance(x, pd.DataFrame):
                    x = torch.tensor(x.values, dtype=torch.float32)
            
            # Forward pass through the graph encoder
            return self.graph_encoder(x)
            
        def predict_intervention_outcomes(self, data, interventions):
            # Ensure data is a tensor
            if not isinstance(data, torch.Tensor):
                if isinstance(data, np.ndarray):
                    data = torch.tensor(data, dtype=torch.float32)
                elif isinstance(x, pd.DataFrame):
                    data = torch.tensor(data.values, dtype=torch.float32)
            
            # Infer graph from data
            graph = self.infer_graph(data)
            
            # Apply interventions using the dynamics decoder
            return self.dynamics_decoder(data, graph, interventions)
    
    # Return the mock model
    return MockAmortizedModel()

def run_neural_model_benchmark():
    print("\n=== Neural Model Benchmark ===")
    
    # Create a causal discovery benchmark for neural methods
    benchmark = CausalDiscoveryBenchmark(
        name="neural_cd_benchmark",
        seed=42,
        num_nodes=5,     # 5-node graphs
        num_graphs=2,    # 2 test graphs
        num_samples=500, # 500 samples per dataset
        graph_type="random"
    )
    
    # Add traditional model
    benchmark.add_model("correlation_0.3", SimpleCausalDiscoveryModel(threshold=0.3))
    
    # Add neural model
    neural_model = create_mock_amortized_model()
    benchmark.add_model("neural_model", neural_model)
    
    # Setup and run the benchmark
    print("Setting up benchmark...")
    benchmark.setup()
    print("Running benchmark...")
    results = benchmark.run()
    
    # Plot results
    print("Plotting results...")
    benchmark.plot_results(
        metrics=["shd", "precision", "recall", "runtime"],
        title="Neural vs. Traditional Method Comparison"
    )
    
    print(f"Results summary:")
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and "summary" in model_results:
            summary = model_results["summary"]
            print(f"  {model_name}:")
            print(f"    SHD: {summary.get('shd', 'N/A'):.2f}")
            print(f"    Precision: {summary.get('precision', 'N/A'):.2f}")
            print(f"    Recall: {summary.get('recall', 'N/A'):.2f}")
    
    return benchmark, results

"""
## Part 7: Best Practices for Using the Benchmarking Framework

### When to Use Each Benchmark Type

1. **CausalDiscoveryBenchmark**: For evaluating methods that infer graph structure from data
2. **CBOBenchmark**: For evaluating methods that optimize interventions in causal systems
3. **ScalabilityBenchmark**: For assessing how methods scale with problem size
4. **BenchmarkRunner**: For comparing multiple methods across multiple benchmarks

### Customizing Benchmarks

1. **Dataset Generation**: Control graph types, size, and data sample count
2. **Evaluation Metrics**: Choose appropriate metrics for your specific needs
3. **Visualization**: Customize plots for specific presentation needs
4. **Reporting**: Generate comprehensive reports for sharing results

### Integrating Your Own Methods

The benchmarking framework supports various method interfaces:

1. **Standard interface**:
   - For discovery: `model.learn_graph(obs_data, int_data)`
   - For CBO: `model.optimize(scm, graph, obs_data, target_node, ...)`

2. **Fit-predict interface**:
   - For discovery: `model.fit(obs_data)` followed by `model.predict_graph()`

3. **Callable interface**:
   - For discovery: `model(obs_data)` returns a graph or adjacency matrix
   - For CBO: `model(scm, graph, obs_data, ...)` returns intervention

4. **Neural interfaces**:
   - `model.infer_graph(data)` and `model.predict_intervention_outcomes(data, interventions)`

### Tips for Reliable Benchmarking

1. **Use multiple random seeds** for statistical significance
2. **Increase the number of test problems** for more reliable results
3. **Match evaluation to your target use case** (graph size, data characteristics)
4. **Include diverse baselines** for comprehensive comparison
5. **Analyze scaling behavior** for practical deployment considerations
"""

def main():
    """Run the tutorial examples."""
    print("Starting Benchmarking Framework Tutorial...")
    
    # Simple Causal Discovery Benchmark
    cd_benchmark, cd_results = run_simple_cd_benchmark()
    
    # Simple CBO Benchmark
    cbo_benchmark, cbo_results = run_simple_cbo_benchmark()
    
    # Scalability Benchmark
    scalability_benchmark, scalability_results = run_scalability_benchmark()
    
    # Benchmark Suite
    runner, suite_results = run_benchmark_suite()
    
    # Neural Model Benchmark
    neural_benchmark, neural_results = run_neural_model_benchmark()
    
    print("\nTutorial complete! The benchmarking framework allows you to:")
    print("1. Evaluate causal discovery algorithms")
    print("2. Evaluate causal Bayesian optimization methods")
    print("3. Assess algorithm scaling behavior")
    print("4. Compare multiple methods across multiple benchmarks")
    print("5. Integrate neural network-based approaches")
    print("6. Generate comprehensive reports and visualizations")

if __name__ == "__main__":
    main() 