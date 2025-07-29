# ACBO Evaluation Guide

This guide explains how to use the new evaluation framework for the Amortized Causal Bayesian Optimization (ACBO) system.

## Overview

The evaluation framework provides a clean, principled way to:
- Train GRPO and BC models with proper configurations
- Evaluate trained models against baselines
- Calculate comprehensive metrics (F1, SHD, target trajectories)
- Generate publication-ready visualizations

## Quick Start

### Running the Full Demonstration

The easiest way to see the system in action:

```bash
# Run complete demo (train + evaluate)
python scripts/run_full_acbo_demo.py --quick

# Full demo with standard parameters
python scripts/run_full_acbo_demo.py
```

This will:
1. Train a GRPO policy with early stopping
2. Train BC models from expert demonstrations
3. Evaluate all methods on test SCMs
4. Generate comprehensive results and plots

### Using Pre-trained Models

If you already have trained models:

```bash
python scripts/run_full_acbo_demo.py \
    --skip-training \
    --grpo-checkpoint path/to/grpo/checkpoint \
    --bc-surrogate-checkpoint path/to/bc/surrogate \
    --bc-acquisition-checkpoint path/to/bc/acquisition
```

## Component Scripts

### 1. Training GRPO (`scripts/core/train_grpo.py`)

Train a GRPO policy with collapse prevention fixes:

```bash
python scripts/core/train_grpo.py \
    --output-dir ./results/grpo \
    --optimization MINIMIZE \
    --episodes 200 \
    --learning-rate 1e-3
```

Key features:
- Early stopping to prevent overfitting
- Bootstrap surrogate for better initialization
- Global standardization for stability
- Increased entropy coefficient for exploration

### 2. Training BC Models (`scripts/core/train_bc.py`)

Train behavioral cloning models:

```bash
python scripts/core/train_bc.py \
    --output-dir ./results/bc \
    --model-type both \
    --max-epochs 50
```

Options for `--model-type`:
- `surrogate`: Train only the surrogate model
- `acquisition`: Train only the acquisition model
- `both`: Train both models (default)

### 3. Evaluating Models (`scripts/core/evaluate_methods.py`)

Evaluate trained models against baselines:

```bash
python scripts/core/evaluate_methods.py \
    --output-dir ./results/evaluation \
    --grpo-checkpoint path/to/grpo/checkpoint \
    --bc-surrogate-checkpoint path/to/bc/surrogate \
    --bc-acquisition-checkpoint path/to/bc/acquisition \
    --n-test-scms 10 \
    --n-runs 3
```

## Understanding the Metrics

### Target Value
- The objective we're trying to optimize (minimize or maximize)
- Lower is better for MINIMIZE, higher is better for MAXIMIZE
- Shown as trajectories over intervention steps

### F1 Score
- Measures structure learning accuracy (0-1, higher is better)
- Combines precision and recall of parent set prediction
- Shows how well the method learns causal relationships

### SHD (Structural Hamming Distance)
- Number of edge differences from true structure (lower is better)
- Counts missing edges + extra edges
- Complementary to F1 score

### Sample Efficiency
- How quickly methods achieve good performance
- Measured by area under the learning curve
- Important for real-world applications with limited interventions

## Interpreting Results

### Output Structure

After running evaluation, you'll find:

```
results/
├── grpo_training/         # GRPO training outputs
│   ├── checkpoints/       # Model checkpoints
│   └── training_summary.json
├── bc_training/           # BC training outputs
│   ├── checkpoints/
│   └── training_summary.json
├── evaluation/            # Evaluation results
│   ├── plots/            # Visualization plots
│   ├── comparison_results.json
│   └── evaluation_summary.txt
└── demonstration_report.txt  # Final summary
```

### Key Plots

1. **method_comparison.png**: Bar chart comparing final performance
2. **learning_curves.png**: Three panels showing target, F1, and SHD trajectories
3. **performance_distribution.png**: Box plots showing variability across runs

### Reading the Summary Report

The evaluation summary includes:
- Performance ranking of all methods
- Statistical significance tests
- Key findings and insights
- Improvement percentages vs baselines

## Troubleshooting

### Common Issues

1. **"No checkpoint found after training"**
   - Check the output directory for checkpoint files
   - Ensure training completed successfully
   - Look for error messages in training output

2. **"BC training fails with import error"**
   - Ensure expert demonstrations are available
   - Check `expert_demonstrations/raw/raw_demonstrations/` exists
   - Verify all dependencies are installed

3. **"Evaluation shows no improvement over baselines"**
   - Try training for more episodes
   - Check if optimization direction matches your objective
   - Ensure reward weights are appropriate

### Debug Mode

For detailed debugging:

```bash
# Add logging
export PYTHONPATH=/path/to/project:$PYTHONPATH
python -u scripts/run_full_acbo_demo.py --quick 2>&1 | tee debug.log
```

## Advanced Usage

### Custom SCM Generation

Modify the SCM generation in evaluation:

```python
# In evaluate_methods.py
factory = VariableSCMFactory(
    noise_scale=0.5,      # Adjust noise level
    coefficient_range=(-2.0, 2.0),  # Adjust edge strengths
    seed=custom_seed
)
```

### Custom Reward Weights

Adjust the balance between discovery and optimization:

```python
# In train_grpo.py config
"reward_weights": {
    "discovery": 0.1,      # Structure learning
    "optimization": 0.8,   # Target optimization
    "efficiency": 0.1      # Sample efficiency
}
```

### Parallel Evaluation

Enable parallel execution for faster evaluation:

```python
# In evaluation config
"parallel": True,  # Run methods in parallel
"n_workers": 4     # Number of parallel workers
```

## Best Practices

1. **Start with Quick Mode**: Use `--quick` flag for initial testing
2. **Monitor Training**: Check training logs for convergence
3. **Validate Checkpoints**: Use checkpoint utilities to verify saved models
4. **Compare Multiple Runs**: Results can vary; run multiple seeds
5. **Visualize Everything**: Plots reveal patterns not visible in numbers

## Next Steps

After successful evaluation:

1. **Analyze Results**: Study which methods work best for your SCM types
2. **Tune Hyperparameters**: Adjust learning rates, batch sizes, etc.
3. **Scale Up**: Increase SCM complexity and intervention budgets
4. **Custom Methods**: Implement new acquisition functions or surrogate models

## References

- GRPO: Group Robustness Policy Optimization
- BC: Behavioral Cloning from expert demonstrations
- SCM: Structural Causal Model
- F1/SHD: Standard causal discovery metrics

For more details, see the source code and docstrings in the implementation files.