# End-to-End ACBO Training and Evaluation Instructions

## Overview
This document provides step-by-step instructions for training and evaluating ACBO methods (GRPO and BC) from scratch.

## Prerequisites
- Make sure you're in the project directory
- Ensure Poetry is installed and dependencies are resolved: `poetry install`

## Step 1: Train GRPO Model

Train a GRPO (Gradient-based Reward Policy Optimization) model with continuous surrogate:

```bash
# Basic training (1000 episodes)
poetry run python scripts/train_acbo_methods.py --method grpo --episodes 1000

# Extended training with custom parameters
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 2000 \
    --learning_rate 3e-4 \
    --use_surrogate
```

The trained model will be saved to `checkpoints/clean_grpo_final/`

## Step 2: Train BC Model

Train a BC (Behavioral Cloning) model using oracle demonstrations:

```bash
# Basic training (1000 episodes)
poetry run python scripts/train_acbo_methods.py --method bc --episodes 1000

# Extended training with custom parameters
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 2000 \
    --demonstration_episodes 200 \
    --expert_strategy oracle
```

The trained model will be saved to `checkpoints/clean_bc_final/`

## Step 3: Evaluate and Compare Models

Compare all methods (Random, Oracle, GRPO, BC) on test SCMs:

```bash
# Basic evaluation
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/clean_grpo_final \
    --bc checkpoints/clean_bc_final \
    --plot

# Extended evaluation with more test cases
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/clean_grpo_final \
    --bc checkpoints/clean_bc_final \
    --n_scms 20 \
    --n_interventions 30 \
    --n_samples 20 \
    --plot
```

Results will be saved to:
- `evaluation_results/evaluation_results.json` - Detailed results
- `evaluation_results/improvement_comparison.png` - Bar chart of improvements
- `evaluation_results/f1_comparison.png` - Structure learning performance
- `evaluation_results/improvement_heatmap.png` - Per-SCM performance matrix

## Step 4: Analyze Results

### View Summary Results
```bash
poetry run python -c "
import json
with open('evaluation_results/evaluation_results.json') as f:
    results = json.load(f)
for method, data in results.items():
    print(f'{method}: {data['aggregate_metrics']['mean_improvement']:.3f} improvement')
"
```

### Debug Training Issues

If GRPO shows no improvement:
```bash
poetry run python scripts/analyze_intervention_patterns.py checkpoints/clean_grpo_final
```

If BC has low accuracy:
```bash
poetry run python scripts/analyze_bc_demonstrations.py
```

## Key Fixes Applied

1. **F1 Score Calculation**: Fixed extraction of marginal probabilities from dict posteriors
2. **Variable Name Mapping**: Created surrogate wrapper to handle X0,X1,X2 vs X,Y,Z naming
3. **Oracle Strategy**: Fixed oracle to use greedy optimization instead of random selection
4. **Surrogate Integration**: Added ContinuousParentSetPredictionModel support to evaluation

## Expected Results

- **Random Baseline**: ~4-6 improvement (varies due to randomness)
- **Oracle**: Should outperform Random by ~20-50% when properly optimizing
- **GRPO**: After 1000+ episodes, should approach Oracle performance
- **BC**: With fixed oracle, should achieve 60-80% of Oracle performance

## Troubleshooting

### "No module named 'src'" Error
Make sure to run all commands with `poetry run` prefix.

### Low F1 Scores
This is expected for Random/Oracle without real structure learning. GRPO/BC with surrogates should show better F1 scores after training.

### BC Low Accuracy
If BC accuracy remains low after fixes:
1. Check oracle is actually optimizing: `poetry run python scripts/validate_oracle_strategy.py`
2. Increase demonstration episodes: `--demonstration_episodes 500`
3. Check expert demonstrations quality: `poetry run python scripts/analyze_bc_demonstrations.py`

## Next Steps

1. **Hyperparameter Tuning**: Try different learning rates, hidden dimensions, etc.
2. **Longer Training**: Train for 5000+ episodes for better convergence
3. **Different SCM Types**: Test on specific graph structures (fork, chain, collider)
4. **Integrate PARENT_SCALE**: Use external PARENT_SCALE demonstrations for BC training