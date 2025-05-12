# Edge Prediction Bias in Graph Structure Learning: Investigation and Resolution

## Problem Overview

During our work on Task 7 (Visualization and Results Analysis), we identified a significant issue with our SimpleGraphLearner model: it predominantly predicted "no edge" regardless of the true graph structure. This bias led to:

1. High accuracy but near-zero precision/recall in sparse graphs (accuracy dominated by true negatives)
2. Poor graph recovery performance despite seemingly good overall metrics
3. Unreliable intervention strategy comparisons due to incorrect graph structure learning

## Investigation Process

### Diagnostic Tools Developed

We created several diagnostic tools to analyze and understand this bias:

1. **Edge probability distribution analysis**: Visualized the distribution of predicted edge probabilities to reveal they were clustered near zero
2. **Loss component analysis**: Decomposed the loss function to understand the contribution of each regularization term
3. **Threshold sensitivity analysis**: Evaluated performance metrics across different threshold values
4. **Comparison of original vs. enhanced models**: Directly compared behavior of different model configurations

### Key Scripts Developed for Analysis

1. `examples/edge_bias_analysis.py`: Complete diagnostic analysis of edge prediction bias, comparing original and enhanced models
2. `examples/enhanced_model_tuning.py`: Systematic hyperparameter search to find optimal settings
3. `examples/train_optimal_graphlearner.py`: Training script using the discovered optimal parameters

## Root Causes Identified

We discovered several contributing factors to the no-edge prediction bias:

1. **Excessive sparsity regularization**: The default sparsity_weight=0.1 was too strong, pushing all edge probabilities toward 0
2. **Class imbalance in sparse graphs**: With few positive examples (edges) vs. many negative examples (non-edges), the model learned to predict "no edge" as the default
3. **Lack of expected density guidance**: Without knowledge of typical graph density, the model defaulted to maximum sparsity
4. **No edge probability initialization bias**: Edge probabilities initialized around 0.5 quickly converged to near 0 during training
5. **Insufficient training data**: Small sample sizes made the bias more pronounced

## Solution Implemented

We developed an enhanced version of the SimpleGraphLearner with the following modifications:

### Parameter Modifications

| Parameter | Original Value | Enhanced Value | Impact |
|-----------|----------------|----------------|--------|
| sparsity_weight | 0.1 | 0.07 | Reduced sparsity regularization |
| pos_weight | 1.0 | 5.0 | Stronger positive class weighting |
| edge_prob_bias | None | 0.3 | Bias toward predicting some edges |
| consistency_weight | None | 0.1 | Push probabilities toward 0 or 1 |
| expected_density | None | 0.4 | Guide toward typical graph density |

### Parameter Tuning Process

We conducted a systematic parameter search testing combinations of:
- Sparsity weight: [0.01, 0.03, 0.05, 0.07]
- Positive class weight: [5.0, 7.0, 10.0]
- Edge probability bias: [0.1, 0.2, 0.3]
- Consistency weight: [0.1, 0.2]
- Expected density: [0.3, 0.4]

This yielded an optimal parameter set that achieved the lowest average SHD across multiple random seeds.

## Results and Performance Impact

The enhanced model with optimized parameters significantly improved graph structure learning:

1. **Edge probability distribution**: Shifted from heavily skewed toward 0 to a more balanced distribution
2. **F1 score**: Increased from near 0 to 0.6-0.9 on test graphs
3. **Perfect recovery**: Achieved SHD=0 on several small graph instances
4. **Balance of metrics**: Improved precision and recall while maintaining high accuracy

## How to Use the Enhanced Model

### Example Code

```python
from causal_meta.structure_learning.simple_graph_learner import SimpleGraphLearner

# Create enhanced model with optimized parameters
model = SimpleGraphLearner(
    input_dim=num_nodes,
    hidden_dim=64,
    num_layers=2,
    sparsity_weight=0.07,     # Reduced sparsity regularization
    acyclicity_weight=1.0,    # Default acyclicity weight
    pos_weight=5.0,           # Stronger positive class weighting
    consistency_weight=0.1,   # Push probabilities toward 0 or 1
    edge_prob_bias=0.3,       # Bias toward predicting edges
    expected_density=0.4      # Expected graph density
)
```

### Running the Training Script

To train a model with the optimal parameters, run:

```bash
python examples/train_optimal_graphlearner.py --num_nodes 4 --num_samples 2000 --epochs 200 --seed 42
```

The results will be saved in the `results/optimal_model_[timestamp]` directory and include visualizations, metrics, and model checkpoints.

## Lessons Learned

1. **Class imbalance matters**: In sparse graphs, special attention must be paid to class imbalance
2. **Regularization balance is crucial**: Sparsity regularization must be balanced against data fit
3. **Metric selection is important**: Accuracy alone can be misleading; F1 score is more informative for sparse graphs
4. **Visual diagnostics reveal hidden issues**: Visualizing the edge probability distribution was key to identifying the bias
5. **Parameter tuning is essential**: Default parameters may not work well across all graph types and densities

## Next Steps

1. **Automatic parameter adjustment**: Develop methods to automatically set parameters based on graph properties
2. **Revisit intervention strategies**: Reevaluate strategic vs. random intervention with the properly balanced model
3. **Larger graph performance**: Test the enhanced model on larger graphs to evaluate scaling behavior
4. **Integrating with other components**: Update other components like ProgressiveInterventionLoop to use the enhanced model 