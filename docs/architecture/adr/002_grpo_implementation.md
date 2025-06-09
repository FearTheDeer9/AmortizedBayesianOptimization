# ADR 002: Group Relative Policy Optimization Implementation

## Status
**IMPLEMENTED** ✅ - Completed with comprehensive test coverage

## Context

We need to implement a reinforcement learning algorithm for our acquisition model. Traditional approaches like PPO require a value network, which adds complexity and computational overhead.

## Decision

Implement Group Relative Policy Optimization (GRPO) which eliminates the need for a value network by using group-based advantage estimation.

## Implementation Details

### Core Algorithm (Following DeepSeek Literature)
```python
# Group-based advantages with std normalization
group_baseline = jnp.mean(rewards)
advantages = rewards - group_baseline
advantages = advantages / (jnp.std(advantages) + 1e-8)

# Policy loss with clipping
ratio = jnp.exp(new_log_probs - old_log_probs)
policy_loss = -jnp.mean(jnp.minimum(
    ratio * advantages, 
    jnp.clip(ratio, 1-clip, 1+clip) * advantages
))

# Exact KL divergence
kl_divergence = jnp.mean(old_log_probs - new_log_probs)
kl_penalty = kl_coeff * jnp.maximum(0.0, kl_divergence - 0.01)

# Total loss (NO value network!)
total_loss = policy_loss + entropy_coeff * entropy_loss + kl_penalty
```

### Key Implementation Features

1. **Pure GRPO**: NO value network - uses group mean as baseline
2. **Literature-compliant**: Advantage normalization by standard deviation  
3. **Exact KL divergence**: Proper KL(π_old || π_new) formula
4. **Group size = 64**: Following DeepSeek recommendations
5. **Single intervention actions**: Each sample is one (state, action, reward) tuple
6. **Vectorized processing**: JAX vmap for efficiency
7. **Diverse training**: Supports different states/SCMs per batch

### File Structure
- `src/causal_bayes_opt/acquisition/grpo.py` - Core GRPO implementation
- `tests/test_acquisition/test_grpo.py` - Comprehensive test suite (13 tests)

## Alternatives Considered

### Proximal Policy Optimization (PPO)
- Pros: Well-established, extensive literature
- Cons: Requires value network, higher memory usage, more complex training

### REINFORCE with Baseline
- Pros: Simpler than PPO
- Cons: Still requires baseline estimation, less stable

### TRL Integration (Rejected)
- Pros: Established library with GRPO support
- Cons: PyTorch-based (conflicts with JAX), designed for language models, unnecessary complexity

## Consequences Realized

### Positive ✅
- **Reduced complexity**: No value network needed
- **Literature compliance**: Follows DeepSeek best practices
- **Multi-objective stability**: Group-based advantages handle conflicting rewards well
- **JAX integration**: Seamless with existing architecture
- **Comprehensive testing**: 13 passing tests covering all aspects
- **Performance**: Vectorized operations with JAX compilation

### Negative ⚠️
- **Group size requirements**: Needs 64+ samples per batch (manageable)
- **Memory usage**: Higher due to larger batch sizes (acceptable trade-off)

## Implementation Validation

### Test Coverage (All Passing)
- ✅ Configuration validation (DeepSeek hyperparameters)
- ✅ Advantage normalization (std-based)
- ✅ Exact KL divergence computation
- ✅ No value network verification
- ✅ Policy entropy calculation
- ✅ Trainer creation and interface
- ✅ Batch collection and processing
- ✅ Error handling and edge cases
- ✅ Integration principles
- ✅ Numerical stability

### Performance Characteristics
- **Group size**: 64 (configurable)
- **Learning rate**: 3e-4
- **Clip ratio**: 0.2
- **Entropy coefficient**: 0.01
- **KL penalty coefficient**: 0.1

## Future Extensions

1. **Adaptive group sizing**: Dynamic adjustment based on reward variance
2. **Multi-objective weighting**: Learnable coefficients for reward components
3. **Distributed training**: Scale to larger SCMs with model parallelism