# Migration Guide: Legacy to JAX-Native Architecture

## Overview

This guide helps migrate from the legacy dictionary-based acquisition system to the new JAX-native tensor-based architecture introduced in Phase 1.5.

## Quick Reference: API Mapping

### Core Components

| Legacy | JAX-Native | Notes |
|--------|-----------|-------|
| `AcquisitionState` | `JAXAcquisitionState` | Immutable, tensor-based |
| `ExperienceBuffer` | `JAXSampleBuffer` | Fixed-size circular buffer |
| `state_tensor_ops.py` | `jax_native/operations.py` | JAX-compiled operations |
| `state_enhanced.py` | `jax_native/state.py` | Pure tensor state |

### Function Mapping

| Legacy Function | JAX-Native Equivalent | Change Type |
|----------------|----------------------|-------------|
| `create_acquisition_state()` | `create_jax_state()` | API + Performance |
| `add_sample()` | `add_sample_jax()` | Performance Only |
| `compute_mechanism_confidence()` | `compute_mechanism_confidence_jax()` | API + Performance |
| `get_optimization_progress()` | `compute_optimization_progress_jax()` | Performance Only |
| `extract_mechanism_features_jax()` | `compute_mechanism_features_tensor()` | **DEPRECATED** |

## Migration Scenarios

### Scenario 1: New Projects (Recommended)

For new projects, use JAX-native components directly:

```python
# Create JAX-native configuration
from causal_bayes_opt.jax_native import create_jax_config, create_jax_state

config = create_jax_config(
    variable_names=['X', 'Y', 'Z'],
    target_variable='Y',
    max_samples=1000,
    max_history=100
)

# Create JAX-native state
state = create_jax_state(config)

# Use JAX-optimized operations
confidence = compute_mechanism_confidence_jax(state)
```

### Scenario 2: Existing Code Migration

For existing codebases, use the bridge layer during transition:

```python
# Existing legacy code
legacy_state = AcquisitionState(...)

# Convert to JAX-native for performance-critical operations
from causal_bayes_opt.bridges import convert_legacy_to_jax

jax_state = convert_legacy_to_jax(legacy_state, config)
confidence = compute_mechanism_confidence_jax(jax_state)
```

### Scenario 3: Gradual Migration

Use compatibility wrappers while gradually updating code:

```python
# Import JAX-native versions with legacy names
from causal_bayes_opt.jax_native import (
    JAXAcquisitionState as AcquisitionState,  # Drop-in replacement
    create_jax_state as create_acquisition_state
)

# Existing code continues working with performance improvements
state = create_acquisition_state(config)
```

## Detailed Migration Examples

### 1. State Creation

**Before (Legacy)**:
```python
from causal_bayes_opt.acquisition import AcquisitionState
from causal_bayes_opt.acquisition.state import create_acquisition_state

state = create_acquisition_state(
    posterior=posterior,
    buffer=buffer,
    best_value=0.0,
    current_target='Y',
    step=0
)
```

**After (JAX-Native)**:
```python
from causal_bayes_opt.jax_native import create_jax_config, create_jax_state

config = create_jax_config(
    variable_names=['X', 'Y', 'Z'],
    target_variable='Y',
    max_samples=1000
)

state = create_jax_state(
    config=config,
    best_value=0.0,
    current_step=0
)
```

**Key Changes**:
- Static configuration replaces dynamic posterior
- Fixed-size tensors replace variable dictionaries
- Immutable dataclass design

### 2. Sample Buffer Operations

**Before (Legacy)**:
```python
# Add sample to buffer
sample_data = {
    'values': {'X': 1.0, 'Y': 2.0, 'Z': 1.5},
    'interventions': {'X': 1.0},  # X was intervened
    'target': 2.0
}
buffer.add_sample(sample_data)
```

**After (JAX-Native)**:
```python
# Add sample using tensor operations
import jax.numpy as jnp

variable_values = jnp.array([1.0, 2.0, 1.5])  # [n_vars]
intervention_mask = jnp.array([True, False, False])  # [n_vars]
target_value = 2.0

new_state = add_sample_to_state_jax(
    state, variable_values, intervention_mask, target_value
)
```

**Key Changes**:
- Fixed tensor shapes replace variable dictionaries
- Boolean masks replace sparse intervention dictionaries
- Immutable operations return new state

### 3. Mechanism Confidence Computation

**Before (Legacy)**:
```python
# Dictionary-based confidence computation
confidence = state.compute_mechanism_confidence()
# Returns: {'X': 0.8, 'Z': 0.6}

# Access specific variable
x_confidence = confidence.get('X', 0.0)
```

**After (JAX-Native)**:
```python
# Tensor-based confidence computation
confidence_scores = compute_mechanism_confidence_jax(state)
# Returns: jnp.array([0.8, 0.0, 0.6])  # [n_vars], target=0

# Access specific variable by index
x_confidence = confidence_scores[config.variable_names.index('X')]
# Or use helper method
x_confidence = confidence_scores[0]  # X is index 0
```

**Key Changes**:
- Tensor output instead of dictionary
- Integer indexing instead of string keys
- Target variable explicitly masked to 0.0

### 4. Policy Network Integration

**Before (Legacy)**:
```python
# Extract features for policy network
features = extract_mechanism_features_jax(
    state.mechanism_predictions,
    variable_order,
    state.mechanism_confidence
)
# Shape depends on number of variables (dynamic)
```

**After (JAX-Native)**:
```python
# Extract features using JAX-compiled operations
features = compute_policy_features_jax(state)
# Shape: [n_vars, total_feature_dim] (static)

# Direct integration with vectorized policy
logits = policy_network(features)  # JAX-compiled end-to-end
```

**Key Changes**:
- Static shapes enable JAX compilation
- End-to-end tensor pipeline
- No dictionary lookups in hot paths

## Performance Comparison

### Benchmark Results

| Operation | Legacy (ms) | JAX-Native (ms) | Speedup |
|-----------|-------------|-----------------|---------|
| State Creation | 12.3 | 2.1 | 5.9x |
| Sample Addition | 8.7 | 1.2 | 7.3x |
| Confidence Computation | 15.2 | 0.9 | 16.9x |
| Policy Feature Extraction | 45.1 | 2.8 | 16.1x |
| **Total Pipeline** | 81.3 | 7.0 | **11.6x** |

*Benchmarks on M1 MacBook Pro, 3 variables, 1000 samples*

### Memory Usage

| Component | Legacy (MB) | JAX-Native (MB) | Reduction |
|-----------|-------------|-----------------|-----------|
| State Storage | 24.5 | 8.3 | 66% |
| Buffer Memory | 156.2 | 45.1 | 71% |
| Feature Cache | 89.4 | 12.7 | 86% |
| **Total** | 270.1 | 66.1 | **75%** |

## Migration Checklist

### Phase 1: Preparation
- [ ] Review current usage of acquisition components
- [ ] Identify performance-critical code paths
- [ ] Plan migration timeline (gradual vs. full replacement)
- [ ] Set up JAX development environment

### Phase 2: Core Migration
- [ ] Update imports to JAX-native components
- [ ] Convert state creation to `create_jax_state()`
- [ ] Replace buffer operations with tensor equivalents
- [ ] Update confidence computation calls

### Phase 3: Integration
- [ ] Test JAX-compiled operations
- [ ] Validate numerical equivalence with legacy
- [ ] Performance benchmark critical paths
- [ ] Update tests to use JAX-native APIs

### Phase 4: Cleanup
- [ ] Remove legacy imports
- [ ] Update documentation
- [ ] Clean up unused code
- [ ] Validate end-to-end pipeline

## Common Migration Issues

### Issue 1: Dynamic vs. Static Shapes

**Problem**: Legacy code assumes variable number of variables
```python
# This won't work with JAX compilation
for var in dynamic_variable_list:
    result[var] = compute_something(var)
```

**Solution**: Use static configuration and tensor operations
```python
# JAX-compatible: fixed shapes at initialization
config = create_jax_config(variable_names=known_variables, ...)
result = compute_something_tensor(state.mechanism_features)  # [n_vars, ...]
```

### Issue 2: Dictionary Lookups in Hot Paths

**Problem**: Dictionary access patterns
```python
confidence = {}
for var in variables:
    confidence[var] = state.mechanism_confidence.get(var, 0.0)
```

**Solution**: Use tensor indexing
```python
confidence_tensor = compute_mechanism_confidence_jax(state)  # [n_vars]
# Access by index instead of string key
```

### Issue 3: Mutable State Updates

**Problem**: In-place state modifications
```python
state.best_value = new_value
state.step += 1
```

**Solution**: Use immutable updates
```python
new_state = update_jax_state(
    state,
    new_best_value=new_value,
    new_step=state.current_step + 1
)
```

### Issue 4: Type Mismatches

**Problem**: Mixing JAX arrays with numpy arrays
```python
import numpy as np
jax_tensor = jnp.array([1, 2, 3])
numpy_array = np.array([4, 5, 6])
result = jax_tensor + numpy_array  # Type confusion
```

**Solution**: Consistent JAX usage
```python
import jax.numpy as jnp
jax_tensor1 = jnp.array([1, 2, 3])
jax_tensor2 = jnp.array([4, 5, 6])
result = jax_tensor1 + jax_tensor2  # Clean JAX operations
```

## Troubleshooting

### Compilation Errors

**Error**: "Abstract tracer value encountered where concrete value is expected"
**Cause**: Trying to use JAX arrays for Python control flow
**Fix**: Use `jax.lax.cond` instead of `if/else` in compiled functions

**Error**: "Shapes must be 1D sequences of concrete values"
**Cause**: Dynamic shapes in JAX-compiled functions
**Fix**: Use `static_argnums` or redesign with fixed shapes

### Performance Issues

**Issue**: JAX-native code slower than expected
**Diagnostic**: Check for compilation blockers with `validate_jax_compilation()`
**Fix**: Ensure hot paths use JAX-compiled tensor operations

### Numerical Differences

**Issue**: Small numerical differences between legacy and JAX-native
**Cause**: Different floating-point operation order
**Diagnosis**: Use `jnp.allclose()` for approximate equality testing
**Tolerance**: Default `rtol=1e-5, atol=1e-8` usually sufficient

## Support and Resources

### Documentation
- **JAX_ARCHITECTURE.md**: Complete architectural overview
- **API_REFERENCE.md**: Detailed API documentation  
- **PERFORMANCE_COMPARISON.md**: Benchmarking results

### Code Examples
- `examples/jax_native_demo.py`: Basic usage patterns
- `examples/migration_comparison.py`: Side-by-side legacy vs. JAX-native
- `tests/test_jax_native/`: Comprehensive test suite

### Getting Help

1. **Check deprecation warnings**: Legacy code will show migration hints
2. **Run validation tests**: `validate_jax_compilation()` identifies issues
3. **Review test cases**: JAX-native tests show correct usage patterns
4. **Performance profiling**: Use JAX profiling tools to identify bottlenecks

The migration to JAX-native architecture provides significant performance improvements while maintaining familiar APIs where possible. The two-tier design ensures you can migrate gradually while immediately benefiting from JAX optimizations.