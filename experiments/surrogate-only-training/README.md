# Surrogate-Only Training Experiment

This experiment tests the surrogate model's ability to learn causal structure with a random (untrained) policy.

## Purpose

Isolate and validate the surrogate learning component by:
- Using a random policy for variable selection (no policy training)
- Training only the surrogate model with BCE loss against true parent labels
- Tracking F1 and SHD metrics throughout episodes

## Configuration

- **Episodes**: 5
- **Interventions per episode**: 75
- **Policy**: Random (frozen weights, no training)
- **Surrogate**: Learning with BCE loss
- **Learning rate**: 3e-4
- **Architecture**: ContinuousParentSetPredictionModel with NodeFeatureEncoder

## Usage

### Run experiment
```bash
python scripts/run_surrogate_experiment.py --config surrogate_test --scm chain
```

### Options
- `--scm`: SCM type (chain, fork, collider)
- `--episodes`: Override number of episodes
- `--interventions`: Override interventions per episode
- `--output`: Save results to file

### Analyze results
```bash
python scripts/analyze_metrics.py results/latest.json
```

## Metrics Tracked

- **F1 Score**: Precision/recall balance for parent prediction
- **SHD (Structural Hamming Distance)**: Graph edit distance
- **BCE Loss**: Training loss trajectory
- **Parent Prediction Accuracy**: Per-variable accuracy

## Expected Behavior

With sufficient interventions, the surrogate should:
1. Show decreasing BCE loss
2. Improve F1 score over time
3. Reduce SHD toward 0
4. Learn true parent relationships despite random interventions