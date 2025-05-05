#!/usr/bin/env python
"""
Example script for running benchmarks.

This script demonstrates how to:
1. Create and configure benchmarks
2. Add models and baselines for evaluation
3. Run benchmarks and analyze results
4. Create visualizations

Usage:
    python run_benchmarks.py [--quick] [--output_dir OUTPUT_DIR] [--seed SEED]

Options:
    --quick          Run with reduced settings for quick demonstration
    --output_dir     Directory to save results (default: benchmark_results)
    --seed           Random seed for reproducibility
"""

import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from causal_meta.meta_learning.benchmark import Benchmark, CausalDiscoveryBenchmark, CBOBenchmark
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner
from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
from causal_meta.meta_learning.amortized_cbo import AmortizedCBO
from causal_meta.graph.causal_graph import CausalGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
                    # This is obviously not a good causal discovery method!
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
    
    # Create a simple callable object instead of AmortizedCausalDiscovery
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
                elif isinstance(data, pd.DataFrame):
                    data = torch.tensor(data.values, dtype=torch.float32)
            
            # Infer graph from data
            graph = self.infer_graph(data)
            
            # Apply interventions using the dynamics decoder
            return self.dynamics_decoder(data, graph, interventions)
    
    # Return the mock model
    return MockAmortizedModel()


def create_mock_amortized_cbo():
    """Create a mock amortized CBO model for testing."""
    
    class MockAmortizedCBO:
        def __init__(self):
            # Store state
            self.X = None
            self.target_node = None
            self.feature_names = None
            self.intervention_targets = None
            self.intervention_ranges = None
            self.objective_fn = None
            self.maximize = True
            self.best_intervention = None
            self.best_value = None
            self.observations = []
            
        def configure_optimization(self, X, target_node, feature_names, 
                                  intervention_targets, intervention_ranges, 
                                  objective_fn, maximize=True):
            """Configure the optimization problem."""
            self.X = X
            self.target_node = target_node
            self.feature_names = feature_names
            self.intervention_targets = intervention_targets
            self.intervention_ranges = intervention_ranges
            self.objective_fn = objective_fn
            self.maximize = maximize
            self.best_value = float('-inf') if maximize else float('inf')
            self.best_intervention = None
            self.observations = []
            
        def suggest_intervention(self):
            """Suggest the next intervention to try."""
            # Simple random exploration strategy
            intervention = {}
            for target in self.intervention_targets:
                if len(self.observations) < len(self.intervention_targets):
                    # In early iterations, try one intervention at a time
                    if len(self.observations) == self.intervention_targets.index(target):
                        min_val, max_val = self.intervention_ranges[target]
                        intervention[target] = np.random.uniform(min_val, max_val)
                else:
                    # Later, use random combinations
                    if np.random.random() < 0.3:
                        min_val, max_val = self.intervention_ranges[target]
                        intervention[target] = np.random.uniform(min_val, max_val)
            
            return intervention
            
        def update(self, intervention, value):
            """Update the model with a new observation."""
            self.observations.append((intervention, value))
            
            # Update best if better
            if self.maximize and value > self.best_value:
                self.best_value = value
                self.best_intervention = intervention
            elif not self.maximize and value < self.best_value:
                self.best_value = value
                self.best_intervention = intervention
                
        def get_best_intervention(self):
            """Get the best intervention found so far."""
            if self.best_intervention is None:
                # If no interventions yet, return an empty dict
                return {}
            return self.best_intervention
            
    # Return the mock CBO model
    return MockAmortizedCBO()


def main():
    """Run benchmarks and analyze results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run benchmarks for causal discovery and intervention optimization')
    parser.add_argument('--quick', action='store_true', help='Run with reduced settings for quick demonstration')
    parser.add_argument('--output_dir', type=str, default='benchmark_results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"benchmark_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure benchmark settings based on quick flag
    if args.quick:
        logger.info("Running with reduced settings for quick demonstration")
        graph_sizes = [3]
        num_graphs = 2
        num_samples = 100
    else:
        graph_sizes = [5, 10]
        num_graphs = 5
        num_samples = 500
    
    # Create benchmark suite
    benchmark_suite = BenchmarkRunner.create_standard_suite(
        name=f"benchmark_suite_{timestamp}",
        output_dir=output_dir,
        seed=args.seed,
        device=device,
        graph_sizes=graph_sizes,
        num_graphs=num_graphs,
        num_samples=num_samples
    )
        
    # Create custom benchmark for demonstration
    custom_benchmark = CausalDiscoveryBenchmark(
        name="custom_small_graphs",
        output_dir=output_dir,
            seed=args.seed,
            device=device,
        num_nodes=4,
        num_graphs=3,
        num_samples=200,
        graph_type="random",
        edge_prob=0.4
        )
    benchmark_suite.add_benchmark(custom_benchmark)
        
    # Add models for evaluation
    # Simple correlation-based model
    benchmark_suite.add_model("correlation_threshold", SimpleCausalDiscoveryModel(threshold=0.3))
    
    # Mock amortized model
    benchmark_suite.add_model("amortized_neural", create_mock_amortized_model())
    
    # Add baseline for CBO
    benchmark_suite.add_baseline("random_intervention", SimpleInterventionModel())
    
    # Add mock amortized CBO
    benchmark_suite.add_model("amortized_cbo", create_mock_amortized_cbo())
    
    # Run all benchmarks
    logger.info("Running all benchmarks")
    results = benchmark_suite.run_all()
    
    # Get best models for each benchmark and metric
    best_models = benchmark_suite.get_best_models()
    
    # Print summary
    logger.info("Benchmark Results Summary:")
    for benchmark_name, metrics in best_models.items():
        logger.info(f"\nBenchmark: {benchmark_name}")
        for metric, model in metrics.items():
            logger.info(f"  Best model for {metric}: {model}")
    
    # Create plots for a specific benchmark if it exists
    if "causal_discovery_erdos_renyi_5" in benchmark_suite.benchmarks:
        cd_benchmark = benchmark_suite.benchmarks["causal_discovery_erdos_renyi_5"]
        fig = cd_benchmark.plot_results(
            metrics=["shd", "f1", "runtime"],
            title="Causal Discovery Performance (Erdos-Renyi, n=5)",
            save_path=os.path.join(output_dir, "causal_discovery_er_5_results.png")
        )
    
    if "cbo_erdos_renyi_5" in benchmark_suite.benchmarks:
        cbo_benchmark = benchmark_suite.benchmarks["cbo_erdos_renyi_5"]
        fig = cbo_benchmark.plot_results(
            metrics=["best_value", "improvement_over_random", "runtime"],
            title="CBO Performance (Erdos-Renyi, n=5)",
            save_path=os.path.join(output_dir, "cbo_er_5_results.png")
        )
    
    logger.info(f"All benchmark results saved to: {output_dir}")


if __name__ == "__main__":
    main() 