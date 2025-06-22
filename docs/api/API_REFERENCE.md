# JAX-Native API Reference

## Overview

This document provides comprehensive API documentation for the JAX-native causal Bayesian optimization system introduced in Phase 1.5.

## Package Structure

```
causal_bayes_opt.jax_native/
├── config.py          # Static configuration management
├── sample_buffer.py   # Circular buffer with tensor operations
├── state.py           # Immutable acquisition state
├── operations.py      # JAX-compiled computational functions
└── __init__.py        # Public API exports
```

## Configuration API

### JAXConfig

**Location**: `causal_bayes_opt.jax_native.config`

Immutable configuration for JAX-native optimization.

```python
@dataclass(frozen=True)
class JAXConfig:
    n_vars: int                    # Number of variables
    target_idx: int               # Target variable index  
    max_samples: int              # Maximum buffer capacity
    max_history: int              # History length for attention
    variable_names: Tuple[str, ...] # Variable names (interpretation only)
    mechanism_types: Tuple[int, ...] # Mechanism type indices
    feature_dim: int = 3          # Mechanism feature dimension
```

**Methods**:

#### `get_variable_name(idx: int) -> str`
Get variable name by index.

**Parameters**:
- `idx`: Variable index (0 <= idx < n_vars)

**Returns**: Variable name string

**Example**:
```python
config = create_jax_config(['X', 'Y', 'Z'], 'Y')
assert config.get_variable_name(0) == 'X'
```

#### `get_target_name() -> str`
Get target variable name.

**Returns**: Target variable name

#### `get_non_target_indices() -> Tuple[int, ...]`
Get indices of non-target variables.

**Returns**: Tuple of variable indices excluding target

#### `create_target_mask() -> jnp.ndarray`
Create boolean mask for target variable.

**Returns**: Boolean array `[n_vars]` with `True` at target index

#### `create_non_target_mask() -> jnp.ndarray`
Create boolean mask for non-target variables.

**Returns**: Boolean array `[n_vars]` with `False` at target index

### Configuration Functions

#### `create_jax_config(variable_names, target_variable, **kwargs) -> JAXConfig`

Create JAX configuration from variable specifications.

**Parameters**:
- `variable_names: List[str]` - List of variable names
- `target_variable: str` - Name of target variable
- `max_samples: int = 1000` - Maximum buffer capacity
- `max_history: int = 100` - History length
- `mechanism_types: Optional[List[int]] = None` - Mechanism type indices
- `feature_dim: int = 3` - Feature dimension

**Returns**: Validated JAX configuration

**Example**:
```python
config = create_jax_config(
    variable_names=['X', 'Y', 'Z'],
    target_variable='Y',
    max_samples=1000,
    max_history=100
)
```

#### `validate_jax_config(config: JAXConfig) -> None`

Validate configuration for consistency and reasonable parameters.

**Parameters**:
- `config`: Configuration to validate

**Raises**: `ValueError` if configuration is invalid

## Sample Buffer API

### JAXSampleBuffer

**Location**: `causal_bayes_opt.jax_native.sample_buffer`

Immutable circular buffer using pure JAX tensors.

```python
@dataclass(frozen=True)
class JAXSampleBuffer:
    values: jnp.ndarray           # [max_samples, n_vars] - variable values
    interventions: jnp.ndarray    # [max_samples, n_vars] - intervention indicators
    targets: jnp.ndarray          # [max_samples] - target values
    valid_mask: jnp.ndarray       # [max_samples] - validity indicators
    write_idx: int                # Current write position
    n_samples: int                # Number of valid samples
    config: JAXConfig             # Static configuration
```

**Properties**:

#### `is_empty() -> bool`
Check if buffer contains no samples.

#### `is_full() -> bool`
Check if buffer is at maximum capacity.

#### `get_latest_samples(n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]`
Get the n most recent samples.

**Parameters**:
- `n`: Number of samples to retrieve

**Returns**: Tuple of (values, interventions, targets) for latest samples

### Buffer Functions

#### `create_empty_jax_buffer(config: JAXConfig) -> JAXSampleBuffer`

Create an empty JAX sample buffer.

**Parameters**:
- `config`: Static configuration

**Returns**: Empty buffer ready for sample addition

#### `add_sample_jax(buffer, variable_values, intervention_mask, target_value) -> JAXSampleBuffer`

Add a sample to the buffer using JAX operations.

**Parameters**:
- `buffer: JAXSampleBuffer` - Current buffer state
- `variable_values: jnp.ndarray` - Values for all variables `[n_vars]`
- `intervention_mask: jnp.ndarray` - Boolean intervention indicators `[n_vars]`
- `target_value: float` - Target variable value

**Returns**: New buffer with sample added

**Example**:
```python
values = jnp.array([1.0, 2.0, 1.5])
interventions = jnp.array([True, False, False])  # X intervened
new_buffer = add_sample_jax(buffer, values, interventions, 2.0)
```

### JAX-Compiled Buffer Operations

#### `add_sample_to_tensors_jax(...) -> Tuple[jnp.ndarray, ...]`

**JAX-compiled** function to add sample to tensor arrays.

**Compilation**: `@jax.jit` - Optimal performance

**Parameters**:
- `values_array: jnp.ndarray` - Current values `[max_samples, n_vars]`
- `interventions_array: jnp.ndarray` - Current interventions `[max_samples, n_vars]`
- `targets_array: jnp.ndarray` - Current targets `[max_samples]`
- `valid_mask: jnp.ndarray` - Current validity mask `[max_samples]`
- `write_idx: int` - Current write position
- `n_samples: int` - Current sample count
- `max_samples: int` - Buffer capacity
- `new_values: jnp.ndarray` - New sample values `[n_vars]`
- `new_interventions: jnp.ndarray` - New interventions `[n_vars]`
- `new_target: float` - New target value

**Returns**: Tuple of updated tensor arrays and indices

## Acquisition State API

### JAXAcquisitionState

**Location**: `causal_bayes_opt.jax_native.state`

Immutable acquisition state using pure JAX tensors.

```python
@dataclass(frozen=True)
class JAXAcquisitionState:
    sample_buffer: JAXSampleBuffer
    mechanism_features: jnp.ndarray      # [n_vars, feature_dim]
    marginal_probs: jnp.ndarray          # [n_vars] - parent probabilities
    confidence_scores: jnp.ndarray       # [n_vars] - mechanism confidence
    best_value: float                    # Current best target value
    current_step: int                    # Current optimization step
    uncertainty_bits: float              # Posterior uncertainty estimate
    config: JAXConfig                    # Static configuration
```

**Properties**:

#### `get_target_name() -> str`
Get target variable name from configuration.

#### `get_n_samples() -> int`
Get number of samples in buffer.

#### `is_buffer_empty() -> bool`
Check if sample buffer is empty.

### State Functions

#### `create_jax_state(config, **kwargs) -> JAXAcquisitionState`

Create JAX acquisition state with default or specified values.

**Parameters**:
- `config: JAXConfig` - Static configuration
- `sample_buffer: Optional[JAXSampleBuffer] = None` - Sample buffer (creates empty if None)
- `mechanism_features: Optional[jnp.ndarray] = None` - Feature tensor
- `marginal_probs: Optional[jnp.ndarray] = None` - Probability tensor
- `confidence_scores: Optional[jnp.ndarray] = None` - Confidence tensor
- `best_value: float = 0.0` - Initial best value
- `current_step: int = 0` - Initial step number
- `uncertainty_bits: float = 1.0` - Initial uncertainty

**Returns**: Initialized JAX acquisition state

**Example**:
```python
state = create_jax_state(
    config=config,
    best_value=1.5,
    current_step=10
)
```

#### `update_jax_state(state, **kwargs) -> JAXAcquisitionState`

**JAX-compiled** function to update acquisition state.

**Compilation**: `@jax.jit` - Optimal performance

**Parameters**:
- `state: JAXAcquisitionState` - Current state
- `new_sample_buffer: Optional[JAXSampleBuffer] = None` - Updated buffer
- `new_mechanism_features: Optional[jnp.ndarray] = None` - Updated features
- `new_marginal_probs: Optional[jnp.ndarray] = None` - Updated probabilities
- `new_confidence_scores: Optional[jnp.ndarray] = None` - Updated confidence
- `new_best_value: Optional[float] = None` - Updated best value
- `new_step: Optional[int] = None` - Updated step
- `new_uncertainty: Optional[float] = None` - Updated uncertainty

**Returns**: Updated state with new values

#### `add_sample_to_state_jax(state, variable_values, intervention_mask, target_value) -> JAXAcquisitionState`

**JAX-compiled** function to add sample to state.

**Parameters**:
- `state: JAXAcquisitionState` - Current state
- `variable_values: jnp.ndarray` - Variable values `[n_vars]`
- `intervention_mask: jnp.ndarray` - Intervention indicators `[n_vars]`
- `target_value: float` - Target value

**Returns**: State with updated buffer and best value

#### `get_policy_input_tensor_jax(state) -> jnp.ndarray`

**JAX-compiled** function to extract policy input tensor.

**Parameters**:
- `state: JAXAcquisitionState` - JAX acquisition state

**Returns**: Policy input tensor `[n_vars, total_features]`

## Operations API

### JAX-Compiled Operations

**Location**: `causal_bayes_opt.jax_native.operations`

All operations optimized for JAX compilation.

#### `compute_mechanism_confidence_from_tensors_jax(mechanism_features, target_mask) -> jnp.ndarray`

**JAX-compiled** computation of mechanism confidence from tensor inputs.

**Compilation**: `@jax.jit` - Optimal performance

**Parameters**:
- `mechanism_features: jnp.ndarray` - Features `[n_vars, feature_dim]`
- `target_mask: jnp.ndarray` - Target mask `[n_vars]` boolean

**Returns**: Confidence scores `[n_vars]` with target set to 0.0

**Algorithm**:
```python
effect_magnitude = jnp.abs(mechanism_features[:, 0])
uncertainty = mechanism_features[:, 1]
confidence = effect_magnitude / (1.0 + uncertainty)
confidence = jnp.where(target_mask, 0.0, confidence)
```

#### `update_mechanism_features_jax(current_features, new_observations, learning_rate=0.1) -> jnp.ndarray`

**JAX-compiled** feature updates using exponential moving average.

**Compilation**: `@jax.jit` - Optimal performance

**Parameters**:
- `current_features: jnp.ndarray` - Current features `[n_vars, feature_dim]`
- `new_observations: jnp.ndarray` - New observations `[n_vars, feature_dim]`
- `learning_rate: float = 0.1` - Learning rate for updates

**Returns**: Updated mechanism features

#### `compute_acquisition_scores_jax(policy_features, target_idx, exploration_weight=0.1) -> jnp.ndarray`

**JAX-compiled** acquisition function for variable selection.

**Compilation**: `@jax.jit` - Optimal performance

**Parameters**:
- `policy_features: jnp.ndarray` - Features `[n_vars, feature_dim]`
- `target_idx: int` - Target variable index to mask
- `exploration_weight: float = 0.1` - Weight for exploration term

**Returns**: Acquisition scores `[n_vars]` with target masked to `-inf`

### State Computation Functions

#### `compute_mechanism_confidence_jax(state) -> jnp.ndarray`

Compute mechanism confidence scores using JAX operations.

**Parameters**:
- `state: JAXAcquisitionState` - JAX acquisition state

**Returns**: Confidence scores `[n_vars]`

**Implementation**: Delegates to `compute_mechanism_confidence_from_tensors_jax`

#### `compute_optimization_progress_jax(state) -> Dict[str, float]`

Compute optimization progress metrics using JAX operations.

**Parameters**:
- `state: JAXAcquisitionState` - JAX acquisition state

**Returns**: Dictionary with progress metrics:
- `improvement_from_start: float` - Total improvement
- `recent_improvement: float` - Recent progress
- `optimization_rate: float` - Improvement per step
- `stagnation_steps: int` - Steps since last improvement

#### `compute_exploration_coverage_jax(state) -> Dict[str, float]`

Compute exploration coverage metrics using JAX operations.

**Parameters**:
- `state: JAXAcquisitionState` - JAX acquisition state

**Returns**: Dictionary with coverage metrics:
- `target_coverage_rate: float` - Fraction of variables explored
- `intervention_diversity: float` - Entropy of intervention distribution
- `unexplored_variables: float` - Fraction unexplored

#### `compute_policy_features_jax(state) -> jnp.ndarray`

Compute comprehensive policy features using JAX operations.

**Parameters**:
- `state: JAXAcquisitionState` - JAX acquisition state

**Returns**: Policy features tensor `[n_vars, total_feature_dim]`

**Features Included**:
- Mechanism features `[n_vars, feature_dim]`
- Marginal probabilities `[n_vars, 1]`
- Confidence scores `[n_vars, 1]`
- Global context (broadcasted) `[n_vars, 8]`
- Variable-specific exploration `[n_vars, 2]`

## Validation and Testing

### Compilation Validation

#### `validate_jax_compilation() -> bool`

Validate that all operations can be JAX-compiled successfully.

**Returns**: `True` if all operations compile successfully

**Usage**:
```python
from causal_bayes_opt.jax_native.operations import validate_jax_compilation
assert validate_jax_compilation() == True
```

### Test Utilities

#### `create_test_config() -> JAXConfig`

Create a test configuration for unit testing.

**Returns**: Test configuration with 3 variables

#### `create_test_buffer() -> JAXSampleBuffer`

Create a test buffer with sample data.

**Returns**: Buffer with 5 test samples

#### `create_test_state() -> JAXAcquisitionState`

Create a test state for unit testing.

**Returns**: Complete test state with sample data

## Type Annotations

### Common Type Aliases

```python
from typing import Dict, List, Tuple, Optional, Any
import jax.numpy as jnp

# Core types
NodeId = str
InterventionValue = float
Shape = Tuple[int, ...]

# JAX tensor types
Array = jnp.ndarray
PRNGKey = jnp.ndarray

# Configuration types
VariableNames = Tuple[str, ...]
MechanismTypes = Tuple[int, ...]

# Function signatures
StateTransition = Callable[[JAXAcquisitionState, Array, Array, float], JAXAcquisitionState]
FeatureExtractor = Callable[[JAXAcquisitionState], Array]
ConfidenceComputer = Callable[[Array, Array], Array]
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised for invalid configuration parameters or tensor shapes.

**Common Causes**:
- Invalid target index
- Mismatched tensor dimensions
- Out-of-range parameters

#### `TypeError`
Raised for incorrect argument types.

**Common Causes**:
- Passing numpy arrays instead of JAX arrays
- Wrong dataclass types

### JAX-Specific Errors

#### `ConcretizationTypeError`
Raised when JAX compilation encounters dynamic values.

**Common Causes**:
- Using dynamic shapes in `@jax.jit` functions
- Incorrect `static_argnums` specification

**Solutions**:
- Use static shapes throughout
- Mark dynamic arguments with `static_argnums`

## Performance Considerations

### JAX Compilation Best Practices

1. **Use Static Shapes**: All tensor operations should use fixed shapes
2. **Minimize Python Calls**: Keep Python code outside `@jax.jit` functions
3. **Batch Operations**: Vectorize computations when possible
4. **Memory Layout**: Use column-major arrays for optimal performance

### Memory Management

1. **Pre-allocation**: Use `create_empty_jax_buffer()` for efficient initialization
2. **Immutable Updates**: All operations return new objects rather than mutating
3. **Tensor Reuse**: JAX automatically optimizes memory reuse in compiled functions

### GPU Optimization

1. **Data Transfer**: Keep data in JAX arrays to avoid CPU-GPU transfers
2. **Kernel Fusion**: JAX automatically fuses operations for optimal GPU utilization
3. **Memory Bandwidth**: Static shapes enable optimal memory access patterns

## Migration from Legacy API

### Key Differences

| Legacy | JAX-Native | Change |
|--------|-----------|---------|
| `AcquisitionState` | `JAXAcquisitionState` | Immutable, tensor-based |
| `buffer.add_sample(dict)` | `add_sample_jax(buffer, tensors)` | Tensor arguments |
| `state.mechanism_confidence` | `compute_mechanism_confidence_jax(state)` | Function call |
| `{'X': 0.8, 'Y': 0.6}` | `jnp.array([0.8, 0.6])` | Tensor output |

### Migration Checklist

1. **Replace Imports**: Use `causal_bayes_opt.jax_native`
2. **Update Data Types**: Convert dictionaries to tensors
3. **Use Integer Indexing**: Replace string keys with indices
4. **Check Shapes**: Ensure all tensors have expected dimensions
5. **Validate Compilation**: Use `validate_jax_compilation()`

See `docs/migration/MIGRATION_GUIDE.md` for detailed migration instructions.