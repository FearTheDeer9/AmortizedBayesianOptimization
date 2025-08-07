# Attention Architecture Test

## What We're Testing

We're comparing two policy architectures for GRPO training:

### 1. Simple Policy (Current Default)
- Initial embedding projection 
- Temporal processing with residual connections
- Mean pooling across time
- MLP layers for final output
- Separate heads for variable selection and value prediction

### 2. Alternating Attention Policy
- Based on CAASL architecture
- Alternates between:
  - Attention over time (for each variable)
  - Attention over variables (for each timestep)
- 4 layers with 4-head multi-head attention
- Layer normalization and residual connections
- Should better capture complex temporal and cross-variable dependencies

## Key Metrics We're Tracking

1. **Target Value Improvement**: Does the policy learn to minimize target values?
2. **Parent Selection Rate**: Does the policy learn to intervene on causal parents?
3. **Logit Spread**: How confident is the policy in its choices?
4. **Parent Logit Advantage**: Do parents have higher logits than non-parents?
5. **Discrimination Over Time**: Do these metrics improve with training?

## Hypotheses

1. **Attention > Simple for Structure Learning**: The attention mechanism should better identify causal relationships
2. **Better Credit Assignment**: Cross-variable attention might help with value learning
3. **More Stable Training**: Layer norm and residual connections should stabilize gradients

## Configurations Being Tested

1. Simple Policy (LR=3e-4) - Baseline
2. Attention Policy (LR=3e-4) - Direct comparison
3. Attention Policy (LR=3e-3) - Higher learning rate
4. Attention Policy (LR=3e-2) - Much higher learning rate

This will help us determine if the architecture is the bottleneck in GRPO training.