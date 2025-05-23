"""
Core conversion functions for AVICI data format bridge.

This module contains the main user-facing functions for converting
Sample objects to AVICI's expected tensor format with target conditioning.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from ._helpers import (
    _extract_values_matrix,
    _extract_intervention_indicators, 
    _create_target_indicators,
    _standardize_values
)
from .validation import _validate_variable_order, _validate_target_variable

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]
AVICIDataBatch = Dict[str, jnp.ndarray]

# Constants
DEFAULT_STANDARDIZATION = "default"

logger = logging.getLogger(__name__)


def samples_to_avici_format(
    samples: SampleList,
    variable_order: VariableOrder,
    target_variable: str,
    standardize: bool = True,
    standardization_type: str = DEFAULT_STANDARDIZATION
) -> jnp.ndarray:
    """
    Convert Sample objects to AVICI's expected input format [N, d, 3].
    
    This function transforms our internal sample representation into the format
    expected by AVICI models, adding target variable conditioning information.
    
    Args:
        samples: List of Sample objects (observational + interventional)
        variable_order: Ordered list of variable names for consistent indexing
        target_variable: Name of target variable for conditioning
        standardize: Whether to standardize variable values
        standardization_type: Type of standardization ("default" or "count")
        
    Returns:
        JAX array of shape [N, d, 3] where:
        - [:, :, 0] = variable values (standardized if requested)
        - [:, :, 1] = intervention indicators (1 if intervened, 0 otherwise)
        - [:, :, 2] = target indicators (1 if target variable, 0 otherwise)
        
    Raises:
        ValueError: If target_variable not in variable_order
        ValueError: If samples contain variables not in variable_order
        ValueError: If inputs are invalid
        
    Example:
        >>> samples = [create_observational_sample({'X': 1.0, 'Y': 2.0})]
        >>> variable_order = ['X', 'Y']
        >>> data = samples_to_avici_format(samples, variable_order, 'Y')
        >>> data.shape
        (1, 2, 3)
        >>> data[0, 1, 2]  # Target indicator for Y
        1.0
    """
    # Validate inputs
    _validate_variable_order(samples, variable_order)
    _validate_target_variable(target_variable, variable_order)
    
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    logger.debug(
        f"Converting {n_samples} samples with {n_vars} variables, target='{target_variable}'"
    )
    
    # Extract variable values [N, d]
    values = _extract_values_matrix(samples, variable_order)
    
    # Standardize values if requested
    if standardize:
        values = _standardize_values(values, standardization_type)
    
    # Extract intervention indicators [N, d]
    intervention_indicators = _extract_intervention_indicators(samples, variable_order)
    
    # Create target indicators [N, d]
    target_indicators = _create_target_indicators(target_variable, variable_order, n_samples)
    
    # Stack into [N, d, 3] tensor
    avici_data = jnp.stack([values, intervention_indicators, target_indicators], axis=2)
    
    logger.debug(f"Created AVICI data tensor with shape: {avici_data.shape}")
    
    return avici_data


def create_training_batch(
    scm: pyr.PMap,
    samples: SampleList,
    target_variable: str,
    standardize: bool = True,
    is_count_data: bool = False
) -> AVICIDataBatch:
    """
    Create a training batch compatible with AVICI's training pipeline.
    
    Args:
        scm: The structural causal model
        samples: List of Sample objects
        target_variable: Name of target variable
        standardize: Whether to standardize the data
        is_count_data: Whether the data should be treated as count data
        
    Returns:
        Dictionary containing AVICI-compatible batch data with keys:
        - 'x': Input data tensor [N, d, 3]
        - 'g': Ground truth adjacency matrix [d, d] (if available)
        - 'is_count_data': Boolean flag for standardization type
        - 'target_variable': Name of target variable
        - 'variable_order': List of variable names in order
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> scm = create_simple_test_scm()
        >>> samples = sample_from_linear_scm(scm, n_samples=100)
        >>> batch = create_training_batch(scm, samples, 'Y')
        >>> batch['x'].shape
        (100, 3, 3)
    """
    # Get variable order from SCM (simplified - use sorted order)
    variables = scm['variables']
    variable_order = sorted(variables)
    
    # Validate target variable
    _validate_target_variable(target_variable, variable_order)
    
    # Convert samples to AVICI format
    standardization_type = "count" if is_count_data else "default"
    x_data = samples_to_avici_format(
        samples=samples,
        variable_order=variable_order,
        target_variable=target_variable,
        standardize=standardize,
        standardization_type=standardization_type
    )
    
    # Create ground truth adjacency matrix if available
    n_vars = len(variable_order)
    g_matrix = jnp.zeros((n_vars, n_vars), dtype=jnp.float32)
    
    # Fill in ground truth edges from SCM
    edges = scm['edges']
    for parent, child in edges:
        parent_idx = variable_order.index(parent)
        child_idx = variable_order.index(child)
        g_matrix = g_matrix.at[parent_idx, child_idx].set(1.0)
    
    # Create batch dictionary
    batch = {
        'x': x_data,
        'g': g_matrix,
        'is_count_data': jnp.array(is_count_data),
        'target_variable': target_variable,
        'variable_order': variable_order,
        'metadata': {
            'n_samples': len(samples),
            'n_variables': n_vars,
            'standardization_type': standardization_type
        }
    }
    
    logger.info(
        f"Created training batch: {len(samples)} samples, {n_vars} variables, "
        f"target='{target_variable}', standardized={standardize}"
    )
    
    return batch