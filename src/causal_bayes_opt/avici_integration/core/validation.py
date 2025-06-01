"""Comprehensive input validation with helpful error messages."""

import jax.numpy as jnp
import pyrsistent as pyr
from typing import List

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]


def validate_conversion_inputs(
    samples: SampleList,
    variable_order: VariableOrder, 
    target_variable: str
) -> None:
    """
    Validate all inputs for conversion functions.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        target_variable: Name of target variable for conditioning
        
    Raises:
        ValueError: If any validation fails with specific error message
    """
    # Basic input validation
    if not samples:
        raise ValueError("Samples list cannot be empty")
    
    if not variable_order:
        raise ValueError("Variable order cannot be empty")
    
    if not target_variable:
        raise ValueError("Target variable cannot be empty")
    
    # Check for duplicates in variable order
    if len(set(variable_order)) != len(variable_order):
        duplicates = [var for var in set(variable_order) if variable_order.count(var) > 1]
        raise ValueError(f"Variable order contains duplicates: {duplicates}")
    
    # Validate target variable
    if target_variable not in variable_order:
        raise ValueError(
            f"Target variable '{target_variable}' not in variable_order. "
            f"Available variables: {sorted(variable_order)}"
        )
    
    # Validate sample structure and variable consistency
    variable_set = set(variable_order)
    
    for i, sample in enumerate(samples):
        # Check sample structure
        if 'values' not in sample:
            raise ValueError(f"Sample {i} missing 'values' key")
        
        if 'intervention_type' not in sample:
            raise ValueError(f"Sample {i} missing 'intervention_type' key")
        
        if 'intervention_targets' not in sample:
            raise ValueError(f"Sample {i} missing 'intervention_targets' key")
        
        # Check variable consistency
        sample_values = sample['values']
        sample_vars = set(sample_values.keys())
        
        missing = sample_vars - variable_set
        if missing:
            raise ValueError(
                f"Sample {i} contains variables not in variable_order: {sorted(missing)}"
            )
        
        extra = variable_set - sample_vars
        if extra:
            raise ValueError(
                f"Sample {i} missing variables from variable_order: {sorted(extra)}"
            )
        
        # Validate intervention targets if present
        if sample['intervention_type'] is not None:
            intervention_targets = sample['intervention_targets']
            if not isinstance(intervention_targets, (set, frozenset, pyr.PSet)):
                raise ValueError(f"Sample {i} intervention_targets must be a set-like object")
            
            invalid_targets = set(intervention_targets) - variable_set
            if invalid_targets:
                raise ValueError(
                    f"Sample {i} has intervention targets not in variable_order: {sorted(invalid_targets)}"
                )


def validate_training_batch_inputs(
    scm: pyr.PMap,
    samples: SampleList,
    target_variable: str
) -> None:
    """
    Validate inputs for training batch creation.
    
    Args:
        scm: The structural causal model
        samples: List of Sample objects
        target_variable: Name of target variable
        
    Raises:
        ValueError: If any validation fails with specific error message
    """
    # Validate SCM structure
    if 'variables' not in scm:
        raise ValueError("SCM missing 'variables' key")
    
    if 'edges' not in scm:
        raise ValueError("SCM missing 'edges' key")
    
    if not scm['variables']:
        raise ValueError("SCM variables cannot be empty")
    
    # Get variable order from SCM
    variables = scm['variables']
    if isinstance(variables, (set, frozenset, pyr.PSet)):
        variable_order = sorted(variables)
    elif isinstance(variables, (list, tuple)):
        variable_order = list(variables)
    else:
        raise ValueError(f"SCM variables must be a set or list, got {type(variables)}")
    
    # Validate edges structure
    edges = scm['edges']
    for edge in edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            raise ValueError(f"Each edge must be a (parent, child) pair, got {edge}")
        
        parent, child = edge
        if parent not in variables:
            raise ValueError(f"Edge parent '{parent}' not in SCM variables")
        
        if child not in variables:
            raise ValueError(f"Edge child '{child}' not in SCM variables")
    
    # Use standard conversion validation for samples
    validate_conversion_inputs(samples, variable_order, target_variable)


def validate_avici_data_structure(
    avici_data: jnp.ndarray,
    expected_n_samples: int,
    expected_n_variables: int
) -> None:
    """
    Validate structure of AVICI data tensor.
    
    Args:
        avici_data: AVICI data tensor
        expected_n_samples: Expected number of samples
        expected_n_variables: Expected number of variables
        
    Raises:
        ValueError: If structure validation fails
    """
    if avici_data.ndim != 3:
        raise ValueError(f"AVICI data must be 3D tensor, got {avici_data.ndim}D")
    
    actual_shape = avici_data.shape
    expected_shape = (expected_n_samples, expected_n_variables, 3)
    
    if actual_shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {actual_shape}")
    
    # Validate data types and ranges
    if not jnp.issubdtype(avici_data.dtype, jnp.floating):
        raise ValueError(f"AVICI data must be floating point, got {avici_data.dtype}")
    
    # Check for NaN or infinity values
    if not jnp.isfinite(avici_data).all():
        raise ValueError("AVICI data contains NaN or infinity values")
    
    # Validate binary channels (interventions and targets)
    intervention_channel = avici_data[:, :, 1]
    target_channel = avici_data[:, :, 2]
    
    # Check that intervention indicators are binary (0 or 1)
    unique_interventions = jnp.unique(intervention_channel)
    valid_interventions = jnp.all(jnp.isin(unique_interventions, jnp.array([0.0, 1.0])))
    if not valid_interventions:
        raise ValueError(
            f"Intervention indicators must be 0 or 1, found values: {unique_interventions}"
        )
    
    # Check that target indicators are binary (0 or 1)
    unique_targets = jnp.unique(target_channel)
    valid_targets = jnp.all(jnp.isin(unique_targets, jnp.array([0.0, 1.0])))
    if not valid_targets:
        raise ValueError(
            f"Target indicators must be 0 or 1, found values: {unique_targets}"
        )
    
    # Check that exactly one variable is marked as target in each sample
    target_sums = jnp.sum(target_channel, axis=1)
    if not jnp.allclose(target_sums, 1.0):
        invalid_samples = jnp.where(target_sums != 1.0)[0]
        raise ValueError(
            f"Each sample must have exactly one target variable. "
            f"Samples with invalid target counts: {invalid_samples[:10]}"  # Show first 10
        )


def validate_training_batch_structure(
    batch: dict,
    expected_n_samples: int,
    expected_n_variables: int,
    expected_target: str
) -> None:
    """
    Validate structure and content of training batch.
    
    Args:
        batch: Training batch dictionary
        expected_n_samples: Expected number of samples
        expected_n_variables: Expected number of variables
        expected_target: Expected target variable name
        
    Raises:
        ValueError: If validation fails with specific error message
    """
    required_keys = ['x', 'g', 'is_count_data', 'target_variable', 'variable_order']
    
    # Check required keys
    for key in required_keys:
        if key not in batch:
            raise ValueError(f"Training batch missing required key: '{key}'")
    
    # Validate x tensor
    x_data = batch['x']
    validate_avici_data_structure(x_data, expected_n_samples, expected_n_variables)
    
    # Validate g matrix (adjacency matrix)
    g_matrix = batch['g']
    expected_g_shape = (expected_n_variables, expected_n_variables)
    
    if g_matrix.shape != expected_g_shape:
        raise ValueError(f"Expected g matrix shape {expected_g_shape}, got {g_matrix.shape}")
    
    if not jnp.issubdtype(g_matrix.dtype, jnp.floating):
        raise ValueError(f"g matrix must be floating point, got {g_matrix.dtype}")
    
    # g matrix should be binary (0 or 1)
    unique_g_values = jnp.unique(g_matrix)
    valid_g_values = jnp.all(jnp.isin(unique_g_values, jnp.array([0.0, 1.0])))
    if not valid_g_values:
        raise ValueError(f"g matrix must contain only 0 or 1, found: {unique_g_values}")
    
    # Validate target variable
    if batch['target_variable'] != expected_target:
        raise ValueError(
            f"Expected target '{expected_target}', got '{batch['target_variable']}'"
        )
    
    # Validate variable order
    variable_order = batch['variable_order']
    if len(variable_order) != expected_n_variables:
        raise ValueError(
            f"Expected {expected_n_variables} variables in order, got {len(variable_order)}"
        )
    
    if expected_target not in variable_order:
        raise ValueError(
            f"Target variable '{expected_target}' not in variable_order: {variable_order}"
        )
    
    # Validate is_count_data
    is_count_data = batch['is_count_data']
    if not isinstance(is_count_data, (bool, jnp.ndarray)):
        raise ValueError(f"is_count_data must be boolean or JAX array, got {type(is_count_data)}")
