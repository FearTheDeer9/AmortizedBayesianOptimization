# ACBO Experiment Infrastructure Guide

**Version**: 1.0  
**Last Updated**: 2025-06-27  
**Status**: Production Ready ✅

## Overview

This guide provides comprehensive documentation for the ACBO (Amortized Causal Bayesian Optimization) experiment infrastructure. The infrastructure enables fair comparison between learning and static surrogate methods while tracking detailed convergence metrics.

## Key Design Principles

### Functional Programming
- **Pure functions**: All utility functions are side-effect free
- **Immutable data**: No modification of core data structures
- **Separation of concerns**: Analysis, visualization, and storage are independent
- **Lazy computation**: Metrics computed on-demand when needed

### Fair Experimental Comparison
- **Same SCM**: Both methods use identical SCMs for valid comparison
- **Standardized outputs**: Consistent result format across all experiments
- **Statistical validity**: Multiple runs with proper random seeding
- **Trajectory tracking**: Complete intervention and posterior history

## Module Architecture

```
src/causal_bayes_opt/
├── analysis/trajectory_metrics.py     # Pure utility functions for metric computation
├── experiments/runner.py              # Fair experiment orchestration  
├── storage/results.py                 # Simple JSON-based result storage
└── visualization/plots.py             # Professional matplotlib visualizations
```

## Module Documentation

### 1. Analysis Module (`analysis/trajectory_metrics.py`)

#### Core Functions

**`compute_true_parent_likelihood(marginal_probs, true_parents) -> float`**
```python
# Calculate P(true parent set | data) from marginal probabilities
likelihood = compute_true_parent_likelihood(
    marginal_probs={'X1': 0.9, 'X2': 0.1, 'X3': 0.8},
    true_parents=['X1', 'X3']
)
# Result: 0.9 * (1-0.1) * 0.8 = 0.648
```

**`compute_trajectory_metrics(trajectory, true_parents, target) -> Dict`**
```python
# Extract all metrics from trajectory data
metrics = compute_trajectory_metrics(trajectory, ['X1', 'X3'], 'X4')
# Returns: {
#   'steps': [1, 2, 3, ...],
#   'true_parent_likelihood': [0.1, 0.3, 0.7, ...],
#   'f1_scores': [0.0, 0.4, 0.8, ...],
#   'target_values': [2.1, 2.5, 2.8, ...],
#   'uncertainties': [10.2, 8.1, 5.3, ...]
# }
```

**Key Features:**
- Zero modifications to core data structures
- Pure functional approach
- Handles missing data gracefully
- Efficient computation

### 2. Experiment Runner (`experiments/runner.py`)

#### Primary Interface

**`run_experiments(scms, methods, n_runs, ...) -> Dict`**
```python
from src.causal_bayes_opt.experiments.runner import run_experiments, create_erdos_renyi_scms

# Create test SCMs
scms = create_erdos_renyi_scms(sizes=[5, 8], edge_probs=[0.3], n_scms_per_config=3)

# Run fair comparison experiments
results = run_experiments(
    scms=scms,
    methods=['static_surrogate', 'learning_surrogate'],
    n_runs=3,                    # Statistical significance
    n_interventions=20,          # Enough steps to see learning
    output_dir="experiment_results",
    save_plots=True
)
```

**Key Features:**
- **Fair comparison**: Same SCM used for both methods
- **Fixed SCM bug**: No longer creates different SCMs per method
- **Statistical validity**: Multiple runs with proper seeding
- **Comprehensive results**: Full trajectory data and metadata

#### Fixed Issue: SCM Generation
The previous version created different SCMs for each method, making comparison invalid:
```python
# OLD (BROKEN): Each method got different SCMs
static_result = run_on_scm_A()      # F1=1.0 (easy problem)
learning_result = run_on_scm_B()    # F1=0.0 (hard problem)

# NEW (FIXED): Same SCM for fair comparison  
scm = create_erdos_renyi_scm()
static_result = run_on_scm(scm)     # F1=0.7 (same problem)
learning_result = run_on_scm(scm)   # F1=0.8 (same problem, learns better)
```

### 3. Storage Module (`storage/results.py`)

#### Core Functions

**`save_experiment_result(result, output_dir) -> str`**
```python
# Save experiment result with timestamp
path = save_experiment_result(
    result={'method': 'learning_surrogate', 'f1_score': 0.85, ...},
    output_dir="results",
    prefix="experiment"
)
# Saves to: results/experiment_20250627_142130.json
```

**`load_experiment_results(results_dir, pattern) -> List[Dict]`**
```python
# Load all matching results
results = load_experiment_results("results", "experiment_*.json")
# Returns list of all experiment dictionaries
```

**Key Features:**
- Simple timestamped JSON files
- No complex schemas or databases
- Automatic metadata addition
- Pattern-based loading and filtering

### 4. Visualization Module (`visualization/plots.py`)

#### Plotting Functions

**`plot_convergence(trajectory_metrics, save_path=None)`**
```python
# Plot convergence to true parent set
plot_convergence(
    trajectory_metrics=metrics,
    title="Learning vs Static Convergence",
    save_path="convergence_plot.png",
    show_f1=True,
    show_uncertainty=True
)
```

**`plot_method_comparison(learning_curves, save_path=None)`**
```python
# Compare multiple methods with confidence intervals
plot_method_comparison(
    learning_curves={
        'static_surrogate': {...},
        'learning_surrogate': {...}
    },
    save_path="method_comparison.png"
)
```

**Key Features:**
- Professional matplotlib styling
- Automatic confidence intervals
- Multiple plot types (convergence, optimization, comparison)
- Publication-ready output

## Usage Patterns

### Pattern 1: Quick Experiment Comparison

```python
from src.causal_bayes_opt.experiments.runner import run_experiments, create_erdos_renyi_scms

# 1. Create test problems
scms = create_erdos_renyi_scms(sizes=[5, 8], edge_probs=[0.3])

# 2. Run comparison
results = run_experiments(
    scms=scms,
    methods=['static_surrogate', 'learning_surrogate'],
    n_runs=5,
    n_interventions=15,
    save_plots=True
)

# 3. Check results
print(f"Experiments completed: {results['summary']}")
```

### Pattern 2: Custom Analysis

```python
from src.causal_bayes_opt.analysis.trajectory_metrics import compute_trajectory_metrics
from src.causal_bayes_opt.visualization.plots import plot_convergence

# 1. Load existing trajectory data
trajectory = load_trajectory_from_file("trajectory.pkl")

# 2. Compute metrics
metrics = compute_trajectory_metrics(trajectory, true_parents=['X1', 'X3'], target='X4')

# 3. Visualize
plot_convergence(metrics, save_path="custom_analysis.png")
```

### Pattern 3: Result Analysis & Comparison

```python
from src.causal_bayes_opt.storage.results import load_experiment_results, filter_results

# 1. Load all results
all_results = load_experiment_results("results")

# 2. Filter by criteria
good_results = filter_results(all_results, {'final_f1_score': {'min': 0.5}})

# 3. Group by method
by_method = {}
for result in good_results:
    method = result['method']
    if method not in by_method:
        by_method[method] = []
    by_method[method].append(result['final_f1_score'])

# 4. Compare performance
for method, f1_scores in by_method.items():
    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"{method}: {avg_f1:.3f} ± {np.std(f1_scores):.3f}")
```

## Integration with Training Pipeline

### Phase 2.5: Experimental Validation

The infrastructure integrates with the main training pipeline at Phase 2.5:

```python
# After Phase 2: Surrogate Training Complete
# Before Phase 3: Acquisition Training

# Validate surrogate performance using infrastructure
validation_results = run_experiments(
    scms=benchmark_scms,
    methods=['trained_surrogate', 'static_baseline'],
    n_runs=10,
    output_dir="phase2_validation"
)

# Proceed to Phase 3 only if validation shows improvement
if validation_results['summary']['learning_advantage'] > 0.1:
    proceed_to_acquisition_training()
```

### Expert Demonstration Collection

```python
# Collect demonstrations for acquisition training
from src.causal_bayes_opt.experiments.runner import create_erdos_renyi_scms

# Generate diverse SCMs for demonstration
demo_scms = create_erdos_renyi_scms(
    sizes=[5, 8, 10, 12],
    edge_probs=[0.2, 0.3, 0.4, 0.5],
    n_scms_per_config=20
)

# Collect expert demonstrations
for scm in demo_scms:
    demo = collect_expert_demonstration(scm)
    save_demonstration(demo)
```

## Performance Metrics

### Key Metrics Tracked

1. **Convergence Speed**: Steps to reach target likelihood threshold
2. **Final Performance**: Ultimate F1 score and target optimization
3. **Sample Efficiency**: Performance per intervention
4. **Statistical Significance**: Confidence intervals across runs
5. **Runtime**: Computational cost comparison

### Metric Definitions

```python
# True parent likelihood: P(true_parents | data)
likelihood = ∏P(is_parent) for true parents × ∏P(not_parent) for non-parents

# F1 Score: Harmonic mean of precision and recall
f1 = 2 * (precision * recall) / (precision + recall)

# Target improvement: Change in target variable value
improvement = final_target_value - initial_target_value

# Sample efficiency: Performance gain per intervention
efficiency = (final_f1 - initial_f1) / n_interventions
```

## Best Practices

### Experimental Design
1. **Multiple runs**: Always use n_runs ≥ 3 for statistical validity
2. **Fair comparison**: Same SCM for all methods being compared
3. **Sufficient steps**: Use n_interventions ≥ 15 to see learning effects
4. **Diverse problems**: Test on multiple graph sizes and edge densities

### Code Quality
1. **Pure functions**: Keep analysis functions side-effect free
2. **Error handling**: Graceful handling of missing/corrupted data
3. **Logging**: Comprehensive logging for debugging
4. **Documentation**: Clear docstrings for all public functions

### Performance
1. **Lazy computation**: Compute metrics only when needed
2. **Efficient storage**: JSON for results, pickle for trajectories
3. **Parallel execution**: Use multiprocessing for large experiments
4. **Memory management**: Clean up large trajectory objects

## Common Issues & Solutions

### Issue 1: Different SCMs Per Method
**Problem**: Methods get different SCMs, making comparison invalid
**Solution**: Fixed in runner.py - now uses same SCM for all methods

### Issue 2: Missing True Parents
**Problem**: Cannot compute true parent likelihood without ground truth
**Solution**: Extract from SCM adjacency matrix or experimental metadata

### Issue 3: Learning Not Showing Advantage
**Possible causes:**
- Too few intervention steps (increase n_interventions)
- Too simple graphs (try larger/denser graphs)
- Learning rate issues (check acquisition training)
- Random policy masking benefits (implement proper acquisition policy)

### Issue 4: Memory Issues with Large Experiments
**Solution**:
```python
# Process results in batches
results = run_experiments(
    scms=scms[:10],  # Smaller batches
    save_plots=False,  # Skip plots for large runs
    output_dir="batch_1"
)
```

## API Reference

### Quick Reference
- **Analysis**: `compute_true_parent_likelihood()`, `compute_trajectory_metrics()`
- **Experiments**: `run_experiments()`, `create_erdos_renyi_scms()`
- **Storage**: `save_experiment_result()`, `load_experiment_results()`
- **Visualization**: `plot_convergence()`, `plot_method_comparison()`

### Error Codes
- `SCM_GENERATION_ERROR`: Failed to create valid SCM
- `TRAJECTORY_EXTRACTION_ERROR`: Cannot extract metrics from trajectory
- `PLOT_GENERATION_ERROR`: Visualization failed
- `STORAGE_ERROR`: Cannot save/load results

## Future Extensions

### Planned Features
1. **Real-time monitoring**: Live experiment progress tracking
2. **Hyperparameter optimization**: Automated learning rate tuning
3. **Distributed experiments**: Multi-machine experiment execution
4. **Advanced baselines**: Integration with other causal discovery methods

### Extension Points
1. **Custom metrics**: Add new trajectory analysis functions
2. **Plot types**: Extend visualization module with new plots
3. **Storage backends**: Support for databases or cloud storage
4. **Experiment types**: Beyond Erdos-Renyi graphs

This infrastructure provides a solid foundation for rigorous ACBO experimentation while maintaining clean, functional code architecture.