# BC Training Debugging & Metrics Enhancement

## Overview

This directory contains comprehensive tools for debugging and analyzing Behavioral Cloning (BC) training for both policy (intervention selection) and surrogate (structure learning) models. The tools provide detailed metrics tracking, visualization, and analysis capabilities to identify where models succeed and where learning is subpar.

## Components

### 1. **metrics_tracker.py**
Core module for tracking detailed training metrics including:
- Classification metrics (F1, precision, recall, accuracy)
- Value prediction metrics (MSE, MAE, R²)
- Node embeddings and representations
- Confusion matrices for variable selection
- Per-variable performance statistics

### 2. **enhanced_bc_trainer.py**
Extended BC trainer that wraps the base `PolicyBCTrainer` to add:
- Comprehensive metric computation during training
- Embedding extraction at configurable epochs
- Per-batch and per-epoch metric tracking
- Detailed validation metrics

### 3. **analyze_training.py**
Post-training analysis script that:
- Loads saved checkpoints and metrics
- Generates comprehensive performance reports
- Identifies problematic variables/patterns
- Compares embeddings across epochs

### 4. **visualize_metrics.py**
Visualization tools for:
- Metric trajectories over time (F1, accuracy, loss)
- Embedding evolution using t-SNE/PCA
- Confusion matrix heatmaps
- Per-variable success rate charts
- Value prediction error distributions

### 5. **test_metrics.py**
Testing script to validate all functionality:
- Runs small-scale training with metrics
- Verifies metric computation
- Tests visualization generation
- Ensures checkpoint compatibility

## Usage Examples

### Training with Enhanced Metrics

```bash
# Train BC policy with detailed metrics
python debugging-bc-training/enhanced_bc_trainer.py \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --max_demos 10 \
    --epochs 100 \
    --save_embeddings_every 10 \
    --output_dir debugging-bc-training/results/
```

### Analyzing Training Results

```bash
# Analyze a completed training run
python debugging-bc-training/analyze_training.py \
    --checkpoint checkpoints/bc_final \
    --metrics_file debugging-bc-training/results/metrics_history.pkl \
    --output_dir debugging-bc-training/results/analysis/
```

### Visualizing Metrics

```bash
# Generate visualizations from training metrics
python debugging-bc-training/visualize_metrics.py \
    --metrics_file debugging-bc-training/results/metrics_history.pkl \
    --output_dir debugging-bc-training/results/plots/
```

### Testing the Tools

```bash
# Run tests to verify everything works
python debugging-bc-training/test_metrics.py
```

## Metrics Tracked

### Classification Metrics
- **Variable Selection Accuracy**: Exact match rate for intervention variable
- **Top-k Accuracy**: Target variable in top-k predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Correct predictions / Total predictions
- **Recall**: Correct predictions / Total ground truth

### Value Prediction Metrics
- **MSE**: Mean squared error of intervention values
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination
- **Value Range Error**: Error relative to variable's value range

### Embedding Analysis
- **Embedding Diversity**: Variance in node representations
- **Cluster Quality**: How well embeddings separate by variable type
- **Temporal Stability**: How embeddings evolve during training

### Per-Variable Metrics
- **Selection Rate**: How often each variable is correctly selected
- **Confusion Patterns**: Which variables are confused with each other
- **Value Prediction Quality**: MSE per variable

## Implementation Progress

- ✅ Directory structure created
- ✅ README.md documentation
- ✅ metrics_tracker.py module
- ✅ enhanced_bc_trainer.py wrapper
- ✅ analyze_training.py script
- ✅ visualize_metrics.py tools
- ✅ test_metrics.py validation

All components have been successfully implemented and tested!

## Key Insights to Extract

1. **Training Dynamics**
   - When does learning plateau?
   - Are there sudden drops in performance?
   - Is overfitting occurring?

2. **Variable-Specific Issues**
   - Which variables are hardest to predict?
   - Are certain variable positions problematic?
   - Do embeddings collapse for some variables?

3. **Representation Quality**
   - Are node embeddings well-separated?
   - Do embeddings capture structural information?
   - How do embeddings evolve during training?

4. **Prediction Patterns**
   - Are errors systematic or random?
   - Do certain SCM structures cause issues?
   - Is there bias toward specific variables?

## Output Files

The tools generate the following outputs in `results/`:

```
results/
├── metrics_history.pkl          # Complete metrics history
├── embeddings/                  # Saved embeddings by epoch
│   ├── epoch_10.npz
│   ├── epoch_20.npz
│   └── ...
├── plots/                       # Generated visualizations
│   ├── f1_trajectory.png
│   ├── confusion_matrix.png
│   ├── embedding_evolution.png
│   └── ...
└── reports/                     # Analysis reports
    ├── training_summary.json
    └── problem_areas.txt
```

## Known Limitations

- Embedding storage can be memory-intensive for large models
- t-SNE visualization is slow for high-dimensional embeddings
- Metrics computation adds ~10-15% training overhead

## Future Enhancements

- [ ] Real-time metric dashboard
- [ ] Comparative analysis across multiple runs
- [ ] Automatic hyperparameter suggestions based on metrics
- [ ] Integration with wandb/tensorboard
- [ ] Support for distributed training metrics