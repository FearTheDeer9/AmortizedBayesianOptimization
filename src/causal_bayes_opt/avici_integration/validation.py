"""
Validation functions for AVICI data format bridge.

This module contains all validation logic to ensure data conversion
preserves information and catches errors early with helpful messages.
"""

# Standard library imports
import logging
from typing import List

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]

logger = logging.getLogger(__name__)


def _validate_variable_order(
    samples: SampleList,
    variable_order: VariableOrder
) -> None:
    """
    Validate that variable order is consistent with samples.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        
    Raises:
        ValueError: If validation fails
    """
    if not variable_order:
        raise ValueError("Variable order cannot be empty")
    
    if not samples:
        raise ValueError("Samples list cannot be empty")
    
    # Check that all variables in samples are in the order
    variable_set = set(variable_order)
    
    for i, sample in enumerate(samples):
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


def _validate_target_variable(
    target_variable: str,
    variable_order: VariableOrder
) -> None:
    """
    Validate that target variable is in the variable order.
    
    Args:
        target_variable: Name of target variable
        variable_order: Ordered list of variable names
        
    Raises:
        ValueError: If target variable is not in variable order
    """
    if target_variable not in variable_order:
        raise ValueError(
            f"Target variable '{target_variable}' not in variable_order. "
            f"Available variables: {sorted(variable_order)}"
        )


def validate_data_conversion(
    original_samples: SampleList,
    converted_data: jnp.ndarray,
    variable_order: VariableOrder,
    target_variable: str,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that data conversion preserves all information.
    
    Args:
        original_samples: Original Sample objects
        converted_data: Converted AVICI data tensor [N, d, 3]
        variable_order: Variable order used in conversion
        target_variable: Target variable name
        tolerance: Tolerance for numerical comparisons
        
    Returns:
        True if conversion preserved all information, False otherwise
        
    Raises:
        ValueError: If inputs are inconsistent
        
    Example:
        >>> samples = [create_observational_sample({'X': 1.0, 'Y': 2.0})]
        >>> variable_order = ['X', 'Y']
        >>> data = samples_to_avici_format(samples, variable_order, 'Y', standardize=False)
        >>> is_valid = validate_data_conversion(samples, data, variable_order, 'Y')
        >>> is_valid
        True
    """
    # Validate inputs
    if len(original_samples) != converted_data.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {len(original_samples)} vs {converted_data.shape[0]}"
        )
    
    if len(variable_order) != converted_data.shape[1]:
        raise ValueError(
            f"Variable count mismatch: {len(variable_order)} vs {converted_data.shape[1]}"
        )
    
    if converted_data.shape[2] != 3:
        raise ValueError(f"Expected 3 channels, got {converted_data.shape[2]}")
    
    logger.debug(f"Validating conversion for {len(original_samples)} samples")
    
    # Extract channels
    values_channel = converted_data[:, :, 0]
    intervention_channel = converted_data[:, :, 1]
    target_channel = converted_data[:, :, 2]
    
    # Validate target indicators
    target_idx = variable_order.index(target_variable)
    expected_target_indicators = jnp.zeros((len(original_samples), len(variable_order)))
    expected_target_indicators = expected_target_indicators.at[:, target_idx].set(1.0)
    
    if not jnp.allclose(target_channel, expected_target_indicators, atol=tolerance):
        logger.error("Target indicators validation failed")
        return False
    
    # Validate intervention indicators
    for i, sample in enumerate(original_samples):
        expected_interventions = jnp.zeros(len(variable_order))
        
        if sample['intervention_type'] is not None:
            intervention_targets = sample['intervention_targets']
            for j, var_name in enumerate(variable_order):
                if var_name in intervention_targets:
                    expected_interventions = expected_interventions.at[j].set(1.0)
        
        if not jnp.allclose(intervention_channel[i], expected_interventions, atol=tolerance):
            logger.error(f"Intervention indicators validation failed for sample {i}")
            return False
    
    logger.debug("Data conversion validation passed")
    return True


def validate_training_batch(
    batch: dict,
    expected_n_samples: int,
    expected_n_variables: int,
    expected_target: str
) -> bool:
    """
    Validate that a training batch has the correct structure and content.
    
    Args:
        batch: Training batch dictionary
        expected_n_samples: Expected number of samples
        expected_n_variables: Expected number of variables
        expected_target: Expected target variable name
        
    Returns:
        True if batch is valid, False otherwise
    """
    required_keys = ['x', 'g', 'is_count_data', 'target_variable', 'variable_order']
    
    # Check required keys
    for key in required_keys:
        if key not in batch:
            logger.error(f"Training batch missing required key: {key}")
            return False
    
    # Check tensor shapes
    x_data = batch['x']
    g_matrix = batch['g']
    
    expected_x_shape = (expected_n_samples, expected_n_variables, 3)
    if x_data.shape != expected_x_shape:
        logger.error(f"Expected x shape {expected_x_shape}, got {x_data.shape}")
        return False
    
    expected_g_shape = (expected_n_variables, expected_n_variables)
    if g_matrix.shape != expected_g_shape:
        logger.error(f"Expected g shape {expected_g_shape}, got {g_matrix.shape}")
        return False
    
    # Check target variable
    if batch['target_variable'] != expected_target:
        logger.error(f"Expected target '{expected_target}', got '{batch['target_variable']}'")
        return False
    
    # Check variable order length
    if len(batch['variable_order']) != expected_n_variables:
        logger.error(f"Expected {expected_n_variables} variables in order, got {len(batch['variable_order'])}")
        return False
    
    logger.debug("Training batch validation passed")
    return True