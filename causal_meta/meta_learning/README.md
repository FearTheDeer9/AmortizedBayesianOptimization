# Causal Meta-Learning and Benchmarking Framework

The `causal_meta.meta_learning` module provides components for meta-learning in causal settings and a comprehensive benchmarking framework for evaluating causal discovery and causal Bayesian optimization methods.

## Benchmarking Framework

### Overview

The benchmarking framework offers a standardized way to evaluate and compare causal learning methods, with particular focus on:

1. **Causal Discovery**: How well methods recover graph structure from observational and/or interventional data
2. **Causal Bayesian Optimization**: How effectively methods can optimize interventions in causal systems
3. **Scalability**: How computational requirements scale with increasing graph sizes

### Key Components

- **Base Benchmark**: Abstract class defining the benchmark interface
- **CausalDiscoveryBenchmark**: For evaluating graph structure learning methods
- **CBOBenchmark**: For evaluating causal Bayesian optimization methods
- **ScalabilityBenchmark**: For evaluating scaling behavior of methods as graph size increases
- **BenchmarkRunner**: For managing sets of benchmarks and aggregating results

### Quick Start

```python
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner
from my_models import MyCausalDiscoveryModel, MyCBOModel

# Create your models
discovery_model = MyCausalDiscoveryModel()
cbo_model = MyCBOModel()

# Create a benchmark runner
runner = BenchmarkRunner(
    name="my_benchmark_run",
    output_dir="benchmark_results",
    seed=42
)

# Register your models
runner.add_model("discovery_model", discovery_model)
runner.add_model("cbo_model", cbo_model)

# Create a standard benchmark suite
benchmark_ids = runner.create_standard_suite(
    graph_sizes=[5, 10, 20],
    num_graphs=5,
    num_samples=1000
)

# Run all benchmarks
results = runner.run_all()

# Generate a summary report
report_path = runner.generate_summary_report()
```

## Detailed Usage

### Evaluating Causal Discovery Methods

To benchmark causal discovery methods:

```python
from causal_meta.meta_learning.benchmark import CausalDiscoveryBenchmark
from your_package import YourCausalDiscoveryMethod

# Create a benchmark
benchmark = CausalDiscoveryBenchmark(
    name="cd_benchmark",
    output_dir="benchmark_results",
    seed=42,
    num_nodes=10,  # Graph size
    num_graphs=20,  # Number of different test graphs
    num_samples=1000,  # Samples per dataset
    graph_type="random",  # Erdos-Renyi random graphs
    edge_prob=0.3  # Graph sparsity parameter
)

# Add your method and a baseline
benchmark.add_model("your_method", YourCausalDiscoveryMethod())
benchmark.add_baseline("random_baseline", RandomGraphGenerator())

# Set up and run
benchmark.setup()
results = benchmark.run()

# Visualize results
benchmark.plot_results(
    metrics=["f1", "precision", "recall", "shd", "runtime"],
    title="Causal Discovery Performance",
    save_path="benchmark_results/cd_performance.png"
)
```

### Evaluating Causal Bayesian Optimization Methods

To benchmark CBO methods:

```python
from causal_meta.meta_learning.benchmark import CBOBenchmark
from your_package import YourCBOMethod

# Create a benchmark
benchmark = CBOBenchmark(
    name="cbo_benchmark",
    output_dir="benchmark_results",
    seed=42,
    num_nodes=8,  # Graph size
    num_graphs=10,  # Number of test problems
    num_samples=1000,  # Samples per dataset
    graph_type="random",
    intervention_budget=10  # Max interventions allowed
)

# Add your method and a baseline
benchmark.add_model("your_cbo_method", YourCBOMethod())
benchmark.add_baseline("random_intervention", RandomInterventionBaseline())

# Set up and run
benchmark.setup()
results = benchmark.run()

# Visualize results
benchmark.plot_results(
    metrics=["best_value", "regret", "runtime", "num_interventions"],
    title="CBO Performance",
    save_path="benchmark_results/cbo_performance.png"
)
```

### Evaluating Method Scalability

To analyze how methods scale with increasing graph size:

```python
from causal_meta.meta_learning.benchmark import ScalabilityBenchmark
from your_package import YourMethod

# Create a scalability benchmark
benchmark = ScalabilityBenchmark(
    name="scalability_benchmark",
    output_dir="benchmark_results",
    seed=42,
    min_nodes=5,  # Smallest graph size
    max_nodes=50,  # Largest graph size
    step_size=5,  # Increments (5, 10, 15, ..., 50)
    num_graphs_per_size=3,  # Test problems per size
    measure_mode="both"  # Test both discovery and CBO
)

# Add methods with different complexity
benchmark.add_model("linear_method", LinearComplexityMethod())
benchmark.add_model("quadratic_method", QuadraticComplexityMethod())

# Set up and run
benchmark.setup()
results = benchmark.run()

# Analyze scaling behavior
scaling_analysis = benchmark.analyze_scaling()

# Visualize scaling
benchmark.plot_scaling_curves(
    metric="runtime",
    log_scale=True,
    save_path="benchmark_results/runtime_scaling.png"
)

# Generate comprehensive report
report = benchmark.generate_scaling_report(
    output_path="benchmark_results/scaling_report.json"
)
```

### Running Multiple Benchmarks

For a comprehensive evaluation across different settings:

```python
from causal_meta.meta_learning.benchmark_runner import BenchmarkRunner

# Create a runner
runner = BenchmarkRunner(
    name="comprehensive_evaluation",
    output_dir="benchmark_results",
    seed=42
)

# Register methods
runner.add_model("method_1", Method1())
runner.add_model("method_2", Method2())
runner.add_baseline("baseline", Baseline())

# Create a standard benchmark suite with different graph sizes
standard_benchmarks = runner.create_standard_suite(
    graph_sizes=[5, 10, 20, 30],
    num_graphs=10,
    num_samples=1000
)

# Create a scalability benchmark suite
scalability_benchmarks = runner.create_scalability_suite(
    min_nodes=5,
    max_nodes=50,
    step_size=5
)

# Run all benchmarks
results = runner.run_all()

# Generate summary report
report_path = runner.generate_summary_report()

# Generate comparison plots
plot_paths = runner.generate_comparison_plots()

# Find the best performing models
best_models = runner.get_best_models()
```

## Supported Metrics

### Causal Discovery Metrics
- Structural Hamming Distance (SHD)
- Precision
- Recall
- F1 score
- Runtime
- Memory usage

### CBO Metrics
- Best intervention value
- Regret
- Improvement over random
- Runtime
- Memory usage
- Number of interventions needed

### Scalability Metrics
- Runtime scaling class (linear, quadratic, cubic, exponential)
- Memory scaling class
- Accuracy vs problem size
- Maximum feasible problem size

## Integrating Your Methods

The benchmarking framework supports multiple interfaces:

### Standard Interface
```python
class StandardCausalDiscoveryMethod:
    def learn_graph(self, observational_data, interventional_data=None):
        """Learn a graph from data.
        
        Returns:
            CausalGraph or adjacency matrix
        """
        # Implementation
```

### Fit-Predict Interface
```python
class FitPredictCausalDiscoveryMethod:
    def fit(self, observational_data, interventional_data=None):
        """Fit the model to data."""
        # Implementation
        
    def predict_graph(self):
        """Return the learned graph.
        
        Returns:
            CausalGraph or adjacency matrix
        """
        # Implementation
```

### Callable Interface
```python
class CallableCausalDiscoveryMethod:
    def __call__(self, observational_data, interventional_data=None):
        """Learn a graph from data.
        
        Returns:
            CausalGraph or adjacency matrix
        """
        # Implementation
```

## Best Practices

1. **Always use a random seed** for reproducibility
2. **Test with multiple graph types** to ensure robustness
3. **Include standard baselines** for meaningful comparisons
4. **Use appropriate graph sizes** (small for initial tests, large for scalability)
5. **Measure both accuracy and runtime** for a complete evaluation
6. **Use the BenchmarkRunner** for complex evaluation setups
7. **Save benchmark results** for future reference and comparison
8. **Generate reports and visualizations** to communicate findings

## Additional Resources

- See `examples/benchmarking_tutorial.ipynb` for a comprehensive tutorial
- For neural network integration, see `examples/neural_benchmarks.ipynb`
- For custom benchmark creation, see `examples/custom_benchmark.ipynb` 