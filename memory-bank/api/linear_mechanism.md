# Linear Mechanisms API Reference

## Overview
The Linear Mechanisms module provides functions for creating and working with linear causal mechanisms in structural causal models (SCMs). It supports both variables with parents (using linear combinations of parent values) and root variables (variables with no parents), with optional Gaussian noise injection for realistic data generation.

## Core Types

### MechanismFunction
```python
MechanismFunction = Callable[[Dict[str, float], jax.Array], float]
```
A callable that represents a causal mechanism. Takes parent variable values and a noise key, returns the computed value for the variable.

### SampleList
```python
SampleList = List[pyr.PMap]
```
A list of Sample objects representing generated observational data from an SCM.

## Constants

- `DEFAULT_NOISE_SCALE: float = 1.0` - Default standard deviation for Gaussian noise
- `DEFAULT_INTERCEPT: float = 0.0` - Default intercept value for linear mechanisms

## Core Functions

### create_linear_mechanism(parents, coefficients, intercept=0.0, noise_scale=1.0)
Create a linear mechanism function for a variable with parents.

Creates a mechanism that computes: Y = intercept + sum(coeff_i * X_i) + noise, where X_i are the parent variables and coeff_i are their coefficients.

**Parameters:**
- `parents: List[str]` - List of parent variable names (can be empty for root variables)
- `coefficients: Dict[str, float]` - Mapping from parent variable names to their coefficients
- `intercept: float` - Constant term in the linear equation (default: 0.0)
- `noise_scale: float` - Standard deviation of Gaussian noise, must be >= 0 (default: 1.0)

**Returns:**
`MechanismFunction` - A mechanism function that takes (parent_values, noise_key) and returns a scalar value

**Raises:**
- `ValueError` - If inputs are inconsistent (e.g., missing coefficients for parents, coefficients for non-parents, invalid parameter values)

**Example:**
```python
import jax.random as random

# Variable Y with parents X and Z: Y = 1.0 + 2.0*X - 1.5*Z + noise
mechanism = create_linear_mechanism(
    parents=['X', 'Z'],
    coefficients={'X': 2.0, 'Z': -1.5},
    intercept=1.0,
    noise_scale=0.1
)

# Generate a value
key = random.PRNGKey(42)
value = mechanism({'X': 1.0, 'Z': 2.0}, key)
```

### create_root_mechanism(mean=0.0, noise_scale=1.0)
Create a mechanism for root variables (variables with no parents).

**Parameters:**
- `mean: float` - Mean value for the root variable (default: 0.0)
- `noise_scale: float` - Standard deviation of Gaussian noise, must be >= 0 (default: 1.0)

**Returns:**
`MechanismFunction` - A mechanism function for a root variable

**Raises:**
- `ValueError` - If parameters are invalid (non-finite values, negative noise_scale)

**Example:**
```python
import jax.random as random

# Root variable with mean 5.0 and noise
mechanism = create_root_mechanism(mean=5.0, noise_scale=1.0)

# Generate a value (empty parent_values for root variables)
key = random.PRNGKey(42)
value = mechanism({}, key)
```

### sample_from_linear_scm(scm, n_samples, seed=42)
Generate observational samples from a linear SCM.

Samples variables in topological order to respect causal dependencies. Each sample is generated independently with proper random key threading for reproducible results.

**Parameters:**
- `scm: pyr.PMap` - The structural causal model to sample from
- `n_samples: int` - Number of samples to generate (must be positive)
- `seed: int` - Random seed for reproducible sampling (default: 42)

**Returns:**
`SampleList` - List of Sample objects containing the generated data

**Raises:**
- `ValueError` - If SCM is invalid for sampling (missing mechanisms, cyclic dependencies) or n_samples is not positive

**Example:**
```python
# Assume we have an SCM with linear mechanisms
samples = sample_from_linear_scm(scm, n_samples=100, seed=42)

# Access sample data
print(f"Generated {len(samples)} samples")
first_sample = samples[0]
# Each sample contains variable values as a dictionary-like structure
```

## Key Usage Patterns

Linear mechanisms support two primary use cases:

1. **Variables with parents**: Use `create_linear_mechanism()` to define how a variable depends linearly on its parents
2. **Root variables**: Use `create_root_mechanism()` for variables with no causal parents

Both mechanism types support configurable Gaussian noise for realistic data generation. The `sample_from_linear_scm()` function coordinates sampling across an entire SCM, ensuring causal ordering is respected.