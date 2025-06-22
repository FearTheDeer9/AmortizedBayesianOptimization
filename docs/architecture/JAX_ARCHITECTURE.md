# JAX-Native Architecture for Causal Bayesian Optimization

## Overview

This document describes the JAX-native architecture implemented in Phase 1.5 of the causal Bayesian optimization system. This represents a fundamental shift from a Python-first to a JAX-first design, enabling significant performance improvements through optimal compilation and tensor operations.

## Architecture Philosophy

### Core Principles

1. **JAX-First Design**: Data structures and operations designed primarily for JAX compilation
2. **Static Tensor Shapes**: All operations use fixed-size tensors for optimal compilation
3. **Immutable Functional Programming**: Following functional programming principles with immutable data
4. **Two-Tier Compilation Strategy**: JAX compilation where it matters, convenience APIs for usability
5. **Zero Circular Dependencies**: Clean separation between performance tiers

### Design Pattern: Two-Tier Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier 2: Convenience APIs                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ JAXAcquisition  │  │ JAXSampleBuffer │  │ High-level  │ │
│  │     State       │  │                 │  │  Functions  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│           │                     │                   │       │
│           └─────────────────────┼───────────────────┘       │
│                                 │                           │
└─────────────────────────────────┼───────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────┐
│              Tier 1: JAX-Compiled Tensor Operations        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │@jax.jit         │  │@jax.jit         │  │@jax.jit     │ │
│  │tensor_ops_jax() │  │confidence_jax() │  │features_jax()│ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Tier 1 (JAX-Compiled)**: Pure tensor operations, optimal performance
**Tier 2 (Convenience)**: Dataclass interfaces, developer productivity

## Core Components

### 1. Static Configuration (`JAXConfig`)

**Location**: `src/causal_bayes_opt/jax_native/config.py`

Immutable configuration establishing all static parameters for JAX compilation:

```python
@dataclass(frozen=True)
class JAXConfig:
    n_vars: int                    # Number of variables
    target_idx: int               # Target variable index
    max_samples: int              # Buffer capacity
    max_history: int              # History length for attention
    variable_names: Tuple[str, ...] # For interpretation only
    mechanism_types: Tuple[int, ...] # Mechanism type indices
    feature_dim: int = 3          # Mechanism feature dimension
```

**Key Features**:
- Comprehensive validation at initialization
- JAX-compatible mask creation methods
- Integer indexing throughout (no string lookups)
- Variable names stored for interpretation only

### 2. Pure Tensor Sample Buffer (`JAXSampleBuffer`)

**Location**: `src/causal_bayes_opt/jax_native/sample_buffer.py`

Circular buffer using pure JAX tensors with immutable operations:

```python
@dataclass(frozen=True)
class JAXSampleBuffer:
    values: jnp.ndarray           # [max_samples, n_vars]
    interventions: jnp.ndarray    # [max_samples, n_vars]
    targets: jnp.ndarray          # [max_samples]
    valid_mask: jnp.ndarray       # [max_samples]
    write_idx: int                # Circular buffer position
    n_samples: int                # Number of valid samples
    config: JAXConfig             # Static configuration
```

**JAX-Compiled Operations**:
```python
@jax.jit
def add_sample_to_tensors_jax(
    values_array, interventions_array, targets_array, valid_mask,
    write_idx, n_samples, max_samples,
    new_values, new_interventions, new_target
) -> Tuple[jnp.ndarray, ...]:
    # Pure tensor circular buffer update
```

### 3. JAX Acquisition State (`JAXAcquisitionState`)

**Location**: `src/causal_bayes_opt/jax_native/state.py`

Immutable state using pure JAX tensors:

```python
@dataclass(frozen=True)
class JAXAcquisitionState:
    sample_buffer: JAXSampleBuffer
    mechanism_features: jnp.ndarray      # [n_vars, feature_dim]
    marginal_probs: jnp.ndarray          # [n_vars]
    confidence_scores: jnp.ndarray       # [n_vars]
    best_value: float
    current_step: int
    uncertainty_bits: float
    config: JAXConfig
```

### 4. Pure JAX Operations (`operations.py`)

**Location**: `src/causal_bayes_opt/jax_native/operations.py`

All computational operations as JAX-compiled functions:

```python
@jax.jit
def compute_mechanism_confidence_from_tensors_jax(
    mechanism_features: jnp.ndarray,  # [n_vars, feature_dim]
    target_mask: jnp.ndarray          # [n_vars] boolean
) -> jnp.ndarray:
    # Pure tensor confidence computation
    
@jax.jit
def update_mechanism_features_jax(
    current_features: jnp.ndarray,
    new_observations: jnp.ndarray,
    learning_rate: float = 0.1
) -> jnp.ndarray:
    # JAX-compiled feature updates
```

## Performance Characteristics

### JAX Compilation Verification

All core operations pass JAX compilation validation:

```python
from causal_bayes_opt.jax_native.operations import validate_jax_compilation
assert validate_jax_compilation() == True
```

### Compilation Strategy

**✅ JAX-Compiled (Tier 1)**:
- `add_sample_to_tensors_jax`: Core buffer operations
- `compute_mechanism_confidence_from_tensors_jax`: Confidence computation
- `update_mechanism_features_jax`: Feature updates
- `compute_acquisition_scores_jax`: Scoring functions

**📋 Non-Compiled (Tier 2)**:
- `add_sample_jax`: Convenience wrapper for buffer updates
- `compute_optimization_progress_jax`: Dictionary returns (by design)
- `get_latest_samples_jax`: Dynamic shapes (by design)

### Performance Improvements

**Eliminated Performance Bottlenecks**:
- ❌ Dictionary operations in hot paths
- ❌ Python loops in tensor computations
- ❌ Format conversion overhead
- ❌ Circular dependencies causing fallbacks

**Achieved Optimizations**:
- ✅ Static tensor shapes throughout
- ✅ Pure JAX operations in critical paths
- ✅ Zero compilation blockers
- ✅ GPU-ready tensor operations

## Data Flow Architecture

### Input Processing

```python
# Legacy (Dictionary-based)
state = {
    'marginal_probs': {'X': 0.7, 'Y': 0.3},
    'mechanism_features': {'X': {...}, 'Y': {...}}
}

# JAX-Native (Tensor-based)  
state = JAXAcquisitionState(
    marginal_probs=jnp.array([0.7, 0.3]),        # [n_vars]
    mechanism_features=jnp.array([              # [n_vars, 3]
        [1.0, 0.2, 0.8],  # Variable 0
        [2.0, 0.1, 0.9]   # Variable 1
    ])
)
```

### Computation Pipeline

```python
# 1. Extract tensors (Tier 2)
features = state.mechanism_features
target_mask = state.config.create_target_mask()

# 2. JAX-compiled computation (Tier 1)
confidence = compute_mechanism_confidence_from_tensors_jax(features, target_mask)

# 3. Return results (tensors stay in JAX format)
return confidence  # jnp.ndarray[n_vars]
```

### Memory Layout

**Fixed Tensor Shapes**:
- `mechanism_features`: `[n_vars, feature_dim]`
- `marginal_probs`: `[n_vars]`
- `sample_buffer.values`: `[max_samples, n_vars]`
- `sample_buffer.interventions`: `[max_samples, n_vars]`

**Benefits**:
- Optimal GPU memory allocation
- Predictable compilation behavior
- No dynamic shape overhead

## Integration Points

### Policy Networks

JAX-native state integrates seamlessly with vectorized policy networks:

```python
# Extract policy input tensor
policy_features = compute_policy_features_jax(jax_state)  # [n_vars, total_features]

# Feed to vectorized attention network
logits = policy_network(policy_features)  # JAX-compiled
```

### Training Loops

Direct tensor operations eliminate conversion overhead:

```python
# JAX-compiled training step
def training_step(jax_state, action):
    # Pure tensor operations throughout
    new_state = add_sample_to_state_jax(jax_state, values, interventions, target)
    features = compute_policy_features_jax(new_state)
    return new_state, features
```

## Migration Strategy

### Compatibility Layers

The architecture provides smooth migration through bridge functions:

```python
# Legacy code continues working
legacy_state = AcquisitionState(...)

# Convert to JAX-native when needed
jax_state = convert_legacy_to_jax(legacy_state, config)

# Use JAX-optimized operations
confidence = compute_mechanism_confidence_jax(jax_state)
```

### Performance Tiers

1. **JAX-Native (Recommended)**: Full performance optimization
2. **Hybrid**: Legacy data, JAX operations via bridges
3. **Legacy (Deprecated)**: Original implementation with fallbacks

## Testing & Validation

### Comprehensive Test Suite

- **Unit Tests**: Each component independently validated
- **Integration Tests**: End-to-end JAX pipeline verification
- **Compilation Tests**: JAX JIT compilation validation
- **Performance Tests**: Benchmarking against legacy implementation

### Property-Based Testing

Using Hypothesis for robust validation:

```python
@given(st.integers(1, 10), st.integers(0, 9))
def test_jax_config_properties(n_vars, target_idx):
    assume(target_idx < n_vars)
    config = create_jax_config(...)
    assert config.n_vars == n_vars
    assert config.target_idx == target_idx
```

## Future Enhancements

### Planned Optimizations

1. **XLA Optimizations**: Target-specific GPU compilation
2. **Batched Operations**: Multi-environment parallel processing
3. **Memory Optimizations**: Gradient checkpointing for large models
4. **Distributed Training**: Multi-GPU state parallelization

### Scalability Considerations

- **Variable Scaling**: Support for problems with 100+ variables
- **Buffer Scaling**: Efficient circular buffers for large sample histories
- **Feature Scaling**: High-dimensional mechanism feature spaces

## Conclusion

The JAX-native architecture provides a solid foundation for high-performance causal Bayesian optimization while maintaining developer productivity through well-designed APIs. The two-tier design ensures optimal performance where it matters most while preserving usability and maintainability.

**Key Benefits**:
- **Performance**: JAX compilation eliminates computational bottlenecks
- **Scalability**: Static tensor shapes enable efficient GPU utilization
- **Maintainability**: Clean separation between performance and convenience
- **Future-Proof**: Foundation for advanced optimizations and scaling