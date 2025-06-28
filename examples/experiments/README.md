# ACBO Experiment Examples

This directory contains examples demonstrating how to use the ACBO experiment infrastructure for fair method comparison and analysis.

## Files

### `quick_start.py`
Minimal example showing basic usage:
- Create test SCMs
- Run fair comparison between static and learning surrogates
- View results

**Run with:**
```bash
poetry run python examples/experiments/quick_start.py
```

### `basic_usage.py` 
Comprehensive example demonstrating:
- All infrastructure modules
- Metric computation
- Visualization generation
- Result loading and analysis

**Run with:**
```bash
poetry run python examples/experiments/basic_usage.py
```

## Key Infrastructure Modules

### Experiment Runner
```python
from src.causal_bayes_opt.experiments.runner import run_experiments, create_erdos_renyi_scms

# Create test problems
scms = create_erdos_renyi_scms(sizes=[5, 8], edge_probs=[0.3])

# Run fair comparison
results = run_experiments(
    scms=scms,
    methods=['static_surrogate', 'learning_surrogate'],
    n_runs=3,
    save_plots=True
)
```

### Analysis
```python
from src.causal_bayes_opt.analysis.trajectory_metrics import compute_trajectory_metrics

# Extract metrics from trajectory
metrics = compute_trajectory_metrics(trajectory, true_parents, target)
```

### Visualization
```python
from src.causal_bayes_opt.visualization.plots import plot_convergence, plot_method_comparison

# Plot convergence
plot_convergence(metrics, save_path="convergence.png")

# Compare methods
plot_method_comparison(learning_curves, save_path="comparison.png")
```

### Storage
```python
from src.causal_bayes_opt.storage.results import save_experiment_result, load_experiment_results

# Save results
path = save_experiment_result(result, "results")

# Load results
results = load_experiment_results("results")
```

## Best Practices

1. **Fair Comparison**: Always use the same SCM for all methods being compared
2. **Multiple Runs**: Use n_runs ≥ 3 for statistical validity
3. **Sufficient Steps**: Use n_interventions ≥ 15 to see learning effects
4. **Save Results**: Always save results for later analysis

## Documentation

For complete documentation, see:
- **Infrastructure Guide**: `docs/experiments/infrastructure_guide.md`
- **Implementation Details**: `docs/training/IMPLEMENTATION_PLAN.md`