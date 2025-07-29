# GRPO Training Collapse Analysis Summary

## Executive Summary
We investigated posterior collapse in GRPO training where all variable embeddings become nearly identical. Through systematic analysis, we identified when collapse occurs, potential causes, and the cascade of effects.

## Key Findings

### 1. Temporal Dynamics
- **Collapse occurs between episodes 100-150**
- Mean reward drops sharply: 0.6940 → 0.6207 (10.6% decrease)
- Policy loss drops to near zero after episode 100, indicating convergence
- Optimization improvement becomes negative after collapse

### 2. Location of Collapse
Through layer-by-layer analysis, we found:
- Collapse happens at the **temporal aggregation step**
- Similarity jumps from 0.8132 (after attention) to 0.9642 (after aggregation)
- The temporal attention mechanism fails to differentiate between variables

### 3. Root Causes Identified

#### a) Bootstrap Surrogate Bug (Fixed)
- Found bug in `_compute_distances_to_target` treating directed graphs as undirected
- This reduced parent probability diversity in the enriched history
- Fix: Removed line that adds reverse edges

#### b) Low Variance in Parent Probabilities Channel
- Parent probabilities channel has variance of only 0.0116
- Bootstrap surrogate outputs nearly identical probabilities for all variables
- This provides insufficient signal for the encoder to differentiate variables

#### c) Training Dynamics
- Policy loss approaching zero suggests premature convergence
- Model achieves perfect structure accuracy but fails at optimization
- Lack of exploration after episode 100

### 4. Architecture Issues
- Current encoder architecture (without role-based projection) struggles to maintain variable identity
- Temporal aggregation uses learned attention weights that converge to uniform distribution
- No explicit mechanism to preserve variable-specific information through aggregation

## Timeline of Collapse

1. **Episodes 1-50**: Healthy training, reward increases
2. **Episodes 50-100**: Peak performance (reward = 0.6940)
3. **Episode 100**: Policy loss drops to ~0, indicating convergence
4. **Episodes 100-150**: Collapse occurs, reward drops significantly
5. **Episodes 150-250**: Model remains in collapsed state

## Recommended Solutions

### Immediate Fixes
1. ✅ Fix bootstrap surrogate bug (already implemented)
2. Add diversity regularization to parent probability predictions
3. Implement exploration bonuses to prevent premature convergence

### Architectural Improvements
1. Use role-based projection to maintain variable identity
2. Add skip connections around temporal aggregation
3. Implement variable-specific output heads
4. Add embedding diversity loss term

### Training Improvements
1. Increase exploration through:
   - Entropy regularization
   - Epsilon-greedy exploration
   - Temperature scheduling
2. Monitor embedding diversity during training
3. Early stopping based on diversity metrics

## Technical Details

### Metrics at Collapse Point (Episode 150)
- Mean similarity: 0.9642 (collapse threshold = 0.9)
- Effective rank: 1-2 (should be 5 for 5 variables)
- Policy loss: 0.0000
- Value loss: 0.0000
- Structure accuracy: 1.0000 (perfect, but misleading)

### Data Flow Analysis
```
Experience Buffer (diverse) 
    → Bootstrap Surrogate (low variance output)
    → Enriched History (low variance parent_probs channel)
    → Temporal Encoder (attempts differentiation)
    → Temporal Aggregation (COLLAPSE POINT)
    → Policy Heads (receives collapsed embeddings)
```

## Next Steps

1. **Implement role-based projection** (currently on refactor branch)
2. **Add diversity loss**: L_diversity = -log(std(embeddings))
3. **Fix exploration**: Add entropy bonus to policy loss
4. **Monitor collapse**: Add real-time diversity tracking during training
5. **Validate fixes**: Run ablation studies with each improvement

## Conclusion

The collapse is caused by a combination of:
1. Insufficient input diversity (bootstrap surrogate issue)
2. Architectural limitations (temporal aggregation)
3. Training dynamics (premature convergence)

The problem is solvable through targeted improvements to each component. The role-based projection branch represents a promising architectural fix that should be combined with training improvements for best results.