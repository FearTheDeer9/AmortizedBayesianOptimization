# Continuous Parent Set Modeling

## Problem Statement

The current surrogate model (`ParentSetPredictionModel`) uses **discrete enumeration with pre-computed lookup tables** for parent set prediction. This approach is fundamentally misaligned with modern differentiable programming and creates multiple architectural problems that block JAX compilation and limit scalability.

## Current Discrete Approach

### Architecture Overview
**Location**: `src/causal_bayes_opt/avici_integration/parent_set/jax_model.py`

```python
# Current approach: Pre-enumerate all possible parent sets
from .jax_enumeration import (
    precompute_parent_set_tables,        # Enumerate all combinations
    filter_parent_sets_for_target_jax,   # Filter by target variable  
    encode_parent_sets_vectorized,       # Convert to integer indices
    select_top_k_parent_sets_jax,        # Select best k sets
    create_parent_set_lookup             # Create lookup table
)
```

### The Enumeration Problem
```python
# This is the wrong abstraction:
parent_sets = [tuple(sorted(combo)) 
               for combo in itertools.combinations(range(d), r)]
# Creates exponential number of discrete objects: C(d,r) parent sets

# Then requires lookup tables for JAX compilation:
parent_set_indices = precompute_parent_set_tables(max_vars, max_parent_size)
# Pre-computed tables become massive for larger graphs
```

## Root Cause Analysis

### 1. **Wrong Mathematical Abstraction**
- **Current**: Parent sets as discrete combinatorial objects
- **Problem**: Exponential enumeration, combinatorial explosion
- **Better**: Parent relationships as continuous probability distributions

### 2. **JAX Compilation Blockers**
```python
# This blocks JAX compilation:
for parent_set in enumerate_all_parent_sets():  # Python iteration
    if parent_set in target_compatible_sets:    # String/tuple operations
        scores.append(score_parent_set(parent_set))  # Dynamic list building
```

### 3. **Scalability Issues**
- **C(10,3) = 120** parent sets (manageable)
- **C(20,3) = 1,140** parent sets (getting large)  
- **C(50,3) = 19,600** parent sets (memory intensive)
- **Exponential growth** makes approach impractical for realistic problems

### 4. **Lookup Table Hack**
The need for pre-computed lookup tables indicates **wrong abstraction**:
```python
# This suggests we're fighting the framework:
parent_set_lookup = create_parent_set_lookup(...)  # Pre-compute everything
encoded_sets = encode_parent_sets_vectorized(...)  # Convert to tensors
```

## Evidence from Codebase

### JAX Model Comments
**Location**: `src/causal_bayes_opt/avici_integration/parent_set/jax_model.py:1-13`

```python
"""
Key improvements:
1. No Python loops in forward pass
2. No string operations or .index() calls  
3. Fixed-size tensor operations with padding
4. Full @jax.jit compatibility
5. Maintains numerical equivalence with original model
"""
```

**Analysis**: The "improvements" are all **workarounds** for the discrete enumeration approach. The fact that we need to eliminate "string operations" and "Python loops" suggests the wrong modeling choice.

### Enumeration Module Evidence
**Location**: `src/causal_bayes_opt/avici_integration/parent_set/jax_enumeration.py`

The existence of a whole module dedicated to "JAX-compatible enumeration" is a red flag. Enumeration shouldn't need special JAX compatibility - it should be naturally differentiable.

## Proposed Solution: Continuous Parent Set Learning

### Core Principle
**Replace discrete enumeration with continuous probability learning**. Model parent relationships directly as learnable probability distributions.

### Mathematical Framework

#### Current (Discrete)
```python
# Enumerate all possible parent sets for variable i
parent_sets_i = {∅, {1}, {2}, {1,2}, {1,3}, {2,3}, {1,2,3}, ...}
# Score each set discretely
scores = [score(parent_set) for parent_set in parent_sets_i]
# Select argmax
best_parent_set = parent_sets_i[argmax(scores)]
```

#### Proposed (Continuous)
```python
# Learn parent probabilities directly
parent_logits = ParentAttention(target_node=i, all_nodes=range(d))  # [d]
parent_probs = softmax(parent_logits)  # [d] - probability each var is parent

# Continuous parent set representation
weighted_parent_effects = sum(parent_probs[j] * effect_function(j→i) 
                             for j in range(d))
```

### Architecture Design

#### 1. **Parent Probability Learning**
```python
class ContinuousParentSetModel(hk.Module):
    def __call__(self, node_features, target_node_idx):
        # Learn which variables are likely parents
        parent_logits = self.parent_attention(
            query=node_features[target_node_idx],    # Target node embedding
            key_value=node_features                  # All node embeddings  
        )  # Output: [n_vars] - logits for each potential parent
        
        # Convert to probabilities
        parent_probs = jax.nn.softmax(parent_logits)
        return parent_probs
```

#### 2. **Differentiable Parent Set Sampling**
```python
class DifferentiableParentSampling(hk.Module):
    def sample_parent_set(self, parent_probs, temperature=1.0):
        # Gumbel-Softmax for differentiable discrete sampling
        gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(...)))
        relaxed_sample = jax.nn.softmax((parent_probs + gumbel_noise) / temperature)
        
        # During training: use soft sample (differentiable)
        # During inference: use hard sample (discrete)
        return relaxed_sample
```

#### 3. **Continuous Effect Modeling**
```python
class ContinuousEffectModel(hk.Module):
    def __call__(self, parent_probs, parent_features):
        # Weight parent effects by their probabilities
        weighted_effects = jnp.sum(
            parent_probs[:, None] * self.effect_network(parent_features),
            axis=0
        )
        return weighted_effects
```

## Implementation Plan

### Phase 1: Core Continuous Model
```python
# File: src/causal_bayes_opt/avici_integration/continuous/model.py
class ContinuousParentSetPredictionModel(hk.Module):
    """Continuous alternative to discrete parent set enumeration."""
    
    def __call__(self, data, target_variable):
        # Input: [N, d, 3] intervention data
        # Output: [d] parent probabilities for target variable
        
        # Learn node representations
        node_embeddings = self.node_encoder(data)  # [d, hidden_dim]
        
        # Learn parent probabilities 
        parent_probs = self.parent_predictor(
            target_emb=node_embeddings[target_variable],
            all_embs=node_embeddings
        )  # [d]
        
        return parent_probs
```

### Phase 2: Differentiable Structure Learning
```python
# File: src/causal_bayes_opt/avici_integration/continuous/structure.py
class DifferentiableStructureLearning(hk.Module):
    """Learn full causal structure using continuous parent probabilities."""
    
    def __call__(self, data):
        # Learn parent probabilities for ALL variables simultaneously
        all_parent_probs = jax.vmap(
            self.single_variable_parents, 
            in_axes=(None, 0)
        )(data, jnp.arange(self.n_vars))  # [n_vars, n_vars]
        
        # Enforce acyclicity constraint (differentiable)
        acyclic_probs = self.enforce_acyclicity(all_parent_probs)
        
        return acyclic_probs
```

### Phase 3: Integration with ACBO
```python
# File: src/causal_bayes_opt/avici_integration/continuous/integration.py
def create_continuous_surrogate_model(config):
    """Factory for continuous surrogate model."""
    
    def model_fn(data, target_variable):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )
        return model(data, target_variable)
    
    return hk.transform(model_fn)
```

## Benefits of Continuous Approach

### 1. **Natural JAX Compatibility**
- No enumeration, no lookup tables
- Pure tensor operations throughout
- Natural @jax.jit compilation
- Vectorizable across all variables

### 2. **Scalability**
- **O(d²)** parameters instead of **O(2^d)** enumeration
- Linear scaling with number of variables
- Memory efficient for large graphs

### 3. **Differentiable Learning**
- End-to-end gradient flow
- No discrete argmax bottlenecks
- Can backpropagate through structure learning

### 4. **Uncertainty Quantification**
```python
# Natural uncertainty representation
parent_probs = softmax(parent_logits)  # [d]
entropy = -sum(p * log(p) for p in parent_probs)  # Uncertainty measure
```

### 5. **Compositional Design**
- Separate parent prediction from effect modeling
- Reusable components across different causal discovery methods
- Clean functional interfaces

## Comparison: Discrete vs Continuous

| Aspect | Discrete Enumeration | Continuous Learning |
|--------|---------------------|-------------------|
| **Scalability** | O(2^d) exponential | O(d²) quadratic |
| **JAX Compatibility** | Requires workarounds | Natural tensor ops |
| **Memory Usage** | Massive lookup tables | Linear parameters |
| **Differentiability** | Discrete argmax | Fully differentiable |
| **Uncertainty** | Hard to quantify | Natural probabilities |
| **Code Complexity** | Enumeration + encoding | Simple attention |

## Migration Strategy

### Step 1: Parallel Implementation
- Implement continuous model alongside discrete version
- Ensure same input/output interface for drop-in replacement

### Step 2: Empirical Validation
- Compare performance on same causal discovery benchmarks
- Validate that continuous approach matches discrete accuracy
- Measure computational improvements

### Step 3: Integration Testing
- Test with existing ACBO pipeline
- Ensure uncertainty estimates are compatible with policy network
- Validate end-to-end training pipeline

### Step 4: Deprecation Path
- Feature flag for continuous vs discrete models
- Gradual migration of experiments to continuous approach
- Remove discrete enumeration code after validation

## Expected Challenges

### 1. **Acyclicity Constraints**
Ensuring learned parent probabilities represent valid DAGs requires careful constraint handling.

**Solution**: Use differentiable acyclicity penalties:
```python
def acyclicity_penalty(parent_probs):
    # Differentiable penalty for cycles
    adj_matrix = parent_probs  # [d, d]
    trace_penalty = jnp.trace(jnp.linalg.matrix_power(adj_matrix, d))
    return trace_penalty
```

### 2. **Discrete-Continuous Gap**
Some downstream components may expect discrete parent sets.

**Solution**: Provide both continuous and discrete interfaces:
```python
def get_top_k_parents(parent_probs, k=3):
    """Convert continuous probs to discrete top-k parent sets."""
    top_indices = jnp.argsort(parent_probs)[-k:]
    return frozenset(top_indices)
```

## Related Improvements

This architectural change enables:
- **Enhanced context integration** (see `enhanced_context_architecture_issues.md`)
- **Full JAX compilation** throughout the pipeline
- **Scalable causal discovery** for larger graphs
- **Better uncertainty quantification** for active learning

## References

- **Current discrete implementation**: `src/causal_bayes_opt/avici_integration/parent_set/`
- **JAX enumeration workarounds**: `src/causal_bayes_opt/avici_integration/parent_set/jax_enumeration.py`
- **Related architecture discussion**: Enhanced context injection analysis (2024-12-26)
- **Theoretical foundation**: Continuous relaxations of combinatorial optimization