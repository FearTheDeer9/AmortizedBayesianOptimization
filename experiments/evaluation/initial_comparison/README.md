# Initial Comparison Experiment

This experiment framework compares different causal discovery methods across varying SCM sizes to evaluate their performance on target optimization and graph structure learning.

## Overview

The experiment compares three types of methods:
1. **Random Baseline**: Random intervention selection
2. **Oracle Baseline**: Perfect knowledge of causal structure
3. **Trained/Untrained Policy**: Learned or initialized policy

Each method is evaluated on:
- Multiple SCM sizes (5, 10, 20, 50, 100 variables)
- Different graph structures (Erdos-Renyi, chain, fork)
- Graph discovery metrics (SHD, F1, precision, recall)
- Target optimization performance

## Directory Structure

```
initial_comparison/
├── configs/
│   └── experiment_config.yaml    # Main configuration file
├── src/
│   ├── graph_metrics.py         # Graph comparison metrics
│   └── experiment_runner.py     # Core experiment logic
├── scripts/
│   ├── run_experiment.py        # Main entry point
│   └── analyze_results.py       # Post-processing analysis
├── results/                     # Output directory (created at runtime)
└── README.md                    # This file
```

## Installation

Ensure you have the main project dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Run

Run with default configuration (untrained models):

```bash
cd experiments/evaluation/initial_comparison
python scripts/run_experiment.py
```

### With Trained Models

Run with trained policy and surrogate:

```bash
python scripts/run_experiment.py \
  --policy-checkpoint ../../joint-training/checkpoints/joint_ep2/policy.pkl \
  --surrogate-checkpoint ../../joint-training/checkpoints/joint_ep2/surrogate.pkl
```

### Custom Configuration

Use a custom configuration file:

```bash
python scripts/run_experiment.py \
  --config path/to/custom_config.yaml \
  --output-dir path/to/output
```

### Debug Mode

Enable detailed logging:

```bash
python scripts/run_experiment.py --debug
```

## Configuration

The experiment is configured via `configs/experiment_config.yaml`:

```yaml
experiment:
  seed: 42  # Random seed for reproducibility

scm_generation:
  sizes: [5, 10, 20, 50, 100]  # SCM sizes to test
  n_scms_per_size: 10           # Number of SCMs per size
  structure_types: ["erdos_renyi", "chain", "fork"]
  edge_density: 0.3

data_generation:
  n_observational_samples: 200  # Initial observations
  n_interventions: 50           # Number of interventions

metrics:
  track: ["shd", "f1", "precision", "recall", "final_target", "best_target"]
```

## Output

Results are saved to `results/run_[timestamp]/` containing:

### Files Generated

1. **Raw Data**:
   - `raw_results.csv`: All experiment results
   - `config.yaml`: Configuration used

2. **Aggregated Results**:
   - `aggregated/by_method.csv`: Results grouped by method
   - `aggregated/by_size.csv`: Results grouped by SCM size
   - `aggregated/by_method_and_size.csv`: Full breakdown

3. **Visualizations**:
   - `plots/target_vs_size.png`: Target value comparison
   - `plots/f1_vs_size.png`: F1 score comparison
   - `plots/shd_vs_size.png`: SHD comparison

4. **Summary**:
   - `summary_stats.json`: Statistical summary
   - `report.txt`: Human-readable report

## Post-Processing Analysis

Run additional analysis on existing results:

```bash
python scripts/analyze_results.py results/run_20240820_143022 --all
```

Options:
- `--statistical`: Perform pairwise statistical tests
- `--advanced-plots`: Generate additional visualizations
- `--latex`: Create LaTeX tables for papers
- `--all`: Run all analyses

## Metrics Explained

### Graph Metrics
- **SHD (Structural Hamming Distance)**: Number of edge operations needed to transform predicted graph to true graph (lower is better)
- **F1 Score**: Harmonic mean of precision and recall for edge prediction (0-1, higher is better)
- **Precision**: Fraction of predicted edges that are correct
- **Recall**: Fraction of true edges that were predicted

### Performance Metrics
- **Final Target**: Target variable value after all interventions
- **Best Target**: Best target value achieved during experiment
- **Convergence Rate**: Rate of improvement per intervention

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from the correct directory and project root is in path
2. **Missing checkpoints**: The experiment works with untrained models if checkpoints aren't available
3. **Memory issues**: Reduce `n_scms_per_size` or test fewer sizes

### Debug Tips

- Use `--debug` flag for detailed logging
- Check `results/*/report.txt` for high-level summary
- Examine `raw_results.csv` for detailed trajectory data

## Example Results Interpretation

After running the experiment, check the report:

```
RESULTS BY METHOD
-----------------
Random:
  Final target: -3.251 ± 1.023
  Best target: -4.892 ± 1.456
  F1 score: 0.000 ± 0.000

Oracle:
  Final target: -8.123 ± 2.341
  Best target: -8.567 ± 2.198
  F1 score: 1.000 ± 0.000

Trained Policy:
  Final target: -5.234 ± 1.567
  Best target: -6.789 ± 1.823
  F1 score: 0.423 ± 0.234
```

This shows the oracle performs best (as expected), trained policy is intermediate, and random performs worst.

## Citation

If you use this experiment framework, please cite:

```bibtex
@software{causal_bayes_opt_eval,
  title = {Causal Bayesian Optimization Evaluation Framework},
  author = {Your Name},
  year = {2024}
}
```

## Contact

For questions or issues, please open an issue on the main repository.