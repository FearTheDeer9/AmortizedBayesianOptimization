"""Meta-learning components for causal discovery and optimization."""

from causal_meta.meta_learning.dynamics_decoder import DynamicsDecoder
from causal_meta.meta_learning.meta_learning import TaskEmbedding, MAMLForCausalDiscovery
from causal_meta.meta_learning.amortized_cbo import AmortizedCBO
from causal_meta.meta_learning.amortized_causal_discovery import AmortizedCausalDiscovery
from causal_meta.meta_learning.benchmark import Benchmark, CausalDiscoveryBenchmark, CBOBenchmark
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner
from causal_meta.meta_learning.visualization import (
    plot_graph_inference_results,
    plot_intervention_outcomes,
    plot_optimization_progress,
    plot_performance_comparison,
    plot_uncertainty
)

__all__ = [
    "DynamicsDecoder", 
    "TaskEmbedding", 
    "MAMLForCausalDiscovery", 
    "AmortizedCBO",
    "AmortizedCausalDiscovery",
    "Benchmark",
    "CausalDiscoveryBenchmark",
    "CBOBenchmark",
    "BenchmarkRunner",
    # Visualization components
    "plot_graph_inference_results",
    "plot_intervention_outcomes",
    "plot_optimization_progress", 
    "plot_performance_comparison",
    "plot_uncertainty"
]



