# Enhanced Context Architecture Issues

## Problem Statement

The current policy network architecture in `src/causal_bayes_opt/acquisition/policy.py` suffers from a fundamental design flaw: **post-transformer feature concatenation**. The transformer learns good representations from intervention history, but these learned representations are then ignored in favor of hand-crafted feature engineering.

## Current Architecture Flow

```python
# Stage 1: Clean transformer input
history = [MAX_HISTORY_SIZE, n_vars, 3]  # values, intervention_flags, target_flags
state_emb = transformer(history)  # [n_vars, hidden_dim] - learned representation

# Stage 2: Feature engineering soup (the problem)
context_features = jnp.stack([
    marginal_probs, uncertainty_features, uncertainty_bits, 
    best_value_feat, step_feat,
    mechanism_confidence_features, predicted_effects, high_impact_indicators,
    mechanism_type_features  # 9 hand-crafted features
], axis=1)  # [n_vars, 9]

# Stage 3: Concatenation (throwing away learned structure)
combined_features = jnp.concatenate([state_emb, context_features], axis=1)
# [n_vars, hidden_dim + 9]
```

## Root Cause Analysis

### 1. **Architectural Contradiction**
- **Assumption**: Transformer can learn useful representations from intervention history
- **Reality**: We immediately concatenate 9 hand-crafted features, suggesting transformer output is insufficient
- **Implication**: Either the transformer is learning nothing useful, or we're wasting its learned structure

### 2. **Separation of Concerns Violation** 
- **Transformer role**: Sequence modeling over intervention history
- **MLP role**: Feature engineering from state components
- **Problem**: These should be integrated, not concatenated

### 3. **Information Loss**
The transformer learns temporal dependencies and variable interactions, but concatenation treats these learned patterns as just another set of features to combine with hand-crafted ones.

## Evidence from Code

**Location**: `src/causal_bayes_opt/acquisition/policy.py:434-442`

```python
# The transformer has learned rich representations...
state_emb = encoder(history, is_training)  # [n_vars, hidden_dim]

# ...but we immediately dilute them with feature engineering
context_features = jnp.stack([...], axis=1)  # [n_vars, 9] 
combined_features = jnp.concatenate([state_emb, context_features], axis=1)
```

**Comments in code reveal the problem**:
```python
# Enhanced for mechanism-aware intervention selection (Architecture Enhancement Pivot - Part C):
# - Leverages uncertainty information from ParentSetPosterior
# - Uses mechanism confidence and predicted effects
# - Prioritizes high-impact variables with uncertain mechanisms
```

These comments show **incremental feature addition** without architectural planning.

## Proposed Solution: Enriched Transformer Input

### Core Principle
**Move context INTO the transformer input, not after it**. This allows the transformer to learn temporal integration of all relevant information.

### New Architecture

```python
# Stage 1: Rich transformer input (PROPOSED)
enriched_history = [MAX_HISTORY_SIZE, n_vars, enriched_channels] where:

# Core intervention data
[:, :, 0] = standardized variable values
[:, :, 1] = intervention indicators
[:, :, 2] = target indicators

# Uncertainty context (per timestep)
[:, :, 3] = marginal parent probabilities (uncertainty over time)
[:, :, 4] = uncertainty bits (how uncertain we were at each timestep)

# Mechanism context (per timestep) 
[:, :, 5] = mechanism confidence (confidence in mechanism at each timestep)
[:, :, 6] = predicted effect magnitude (expected effect size)
[:, :, 7] = mechanism type encoding (what type we predicted)

# Optimization context (per timestep)
[:, :, 8] = best value so far (optimization progress)
[:, :, 9] = steps since improvement (stagnation indicator)
```

### Benefits

1. **Temporal Integration**: Transformer learns how uncertainty/confidence evolves over time
2. **Natural Attention**: Model learns which historical contexts matter for current decisions
3. **Reduced Feature Engineering**: No manual feature combination - learned through attention
4. **Better Generalization**: Learned patterns transfer across different SCM structures

## Implementation Plan

### Phase 1: Data Pipeline Enhancement
```python
# File: src/causal_bayes_opt/acquisition/state_enrichment.py
class EnrichedHistoryBuilder:
    def build_enriched_history(self, state: AcquisitionState) -> jnp.ndarray:
        """Convert AcquisitionState to enriched transformer input."""
        # Create enriched channels for each timestep in history
        # Track uncertainty/mechanism/optimization context evolution
```

### Phase 2: Transformer Architecture Update
```python
# File: src/causal_bayes_opt/acquisition/enriched_policy.py
class EnrichedAttentionEncoder(hk.Module):
    def __call__(self, enriched_history: jnp.ndarray):
        # Input: [MAX_HISTORY_SIZE, n_vars, enriched_channels]
        # Process ALL context through transformer attention
        # No post-processing feature concatenation
```

### Phase 3: Policy Head Simplification
```python
class SimplifiedPolicyHeads(hk.Module):
    def variable_selection_head(self, state_emb: jnp.ndarray):
        # Input: [n_vars, hidden_dim] from enriched transformer
        # No additional feature engineering needed
        return hk.Linear(1)(state_emb).squeeze(-1)
```

## Expected Outcomes

### Performance Improvements
- **Better temporal reasoning**: Model learns how context evolves over intervention sequence
- **Reduced overfitting**: Less hand-crafted feature engineering means better generalization
- **Cleaner gradients**: Direct path from input to output without feature concatenation bottleneck

### Code Quality Improvements  
- **Simpler policy heads**: No complex feature engineering in policy network
- **Better separation**: State enrichment separate from policy learning
- **Easier testing**: Can test enriched input generation independently

### Architectural Benefits
- **Functional design**: Clear data pipeline with pure functions
- **JAX compatibility**: No complex state extraction in compiled functions
- **Composability**: Enriched history builder can be reused across different models

## Migration Strategy

### Step 1: Create Parallel Implementation
- Implement `EnrichedAttentionEncoder` alongside existing `AlternatingAttentionEncoder`
- Add feature flag for A/B testing

### Step 2: Comparative Validation
- Train both architectures on same data
- Compare performance on causal discovery tasks
- Validate that enriched approach matches or exceeds current performance

### Step 3: Gradual Migration
- Default to enriched architecture for new experiments
- Deprecate feature concatenation approach
- Remove legacy code after validation period

## Related Issues

This architectural issue is related to:
- **Discrete parent set modeling** (see `continuous_parent_set_modeling.md`)
- **JAX compilation blockers** in current implementation
- **Feature engineering vs representation learning** philosophical differences

## References

- **Current implementation**: `src/causal_bayes_opt/acquisition/policy.py:434-442`
- **Transformer encoder**: `src/causal_bayes_opt/acquisition/policy.py:68-252`
- **Related discussion**: Architecture analysis conversation (2024-12-26)