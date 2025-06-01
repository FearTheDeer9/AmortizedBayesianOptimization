"""Pure conversion functions with no side effects."""

import jax.numpy as jnp
import pyrsistent as pyr
from typing import Dict, List, Optional, Any

from .data_extraction import (
    extract_values_matrix,
    extract_intervention_indicators,
    create_target_indicators,
    extract_ground_truth_adjacency
)
from .standardization import (
    compute_standardization_params,
    apply_standardization,
    StandardizationParams,
    StandardizationType
)

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]
AVICIDataBatch = Dict[str, Any]


def samples_to_avici_format(
    samples: SampleList,
    variable_order: VariableOrder,
    target_variable: str,
    standardization_params: Optional[StandardizationParams] = None
) -> jnp.ndarray:
    """
    Pure function: Convert samples to AVICI format.
    
    NO LOGGING OR SIDE EFFECTS.
    Assumes inputs are pre-validated.
    
    Args:
        samples: List of Sample objects (observational + interventional)
        variable_order: Ordered list of variable names for consistent indexing
        target_variable: Name of target variable for conditioning
        standardization_params: Pre-computed standardization parameters
        
    Returns:
        JAX array of shape [N, d, 3] where:
        - [:, :, 0] = variable values (standardized if params provided)
        - [:, :, 1] = intervention indicators (1 if intervened, 0 otherwise)
        - [:, :, 2] = target indicators (1 if target variable, 0 otherwise)
    """
    n_samples = len(samples)
    
    # Extract variable values [N, d]
    values = extract_values_matrix(samples, variable_order)
    
    # Apply standardization if parameters provided
    if standardization_params is not None:
        values = apply_standardization(values, standardization_params)
    
    # Extract intervention indicators [N, d]
    intervention_indicators = extract_intervention_indicators(samples, variable_order)
    
    # Create target indicators [N, d]
    target_indicators = create_target_indicators(target_variable, variable_order, n_samples)
    
    # Stack into [N, d, 3] tensor
    avici_data = jnp.stack([values, intervention_indicators, target_indicators], axis=2)
    
    return avici_data


def create_avici_batch_from_components(
    x_data: jnp.ndarray,
    ground_truth_adjacency: jnp.ndarray,
    target_variable: str,
    variable_order: VariableOrder,
    is_count_data: bool = False
) -> AVICIDataBatch:
    """
    Pure function: Assemble components into training batch.
    
    Separated from data conversion for better composability.
    
    Args:
        x_data: AVICI data tensor [N, d, 3]
        ground_truth_adjacency: Adjacency matrix [d, d]
        target_variable: Name of target variable
        variable_order: Ordered list of variable names
        is_count_data: Whether data is count-based
        
    Returns:
        Dictionary containing AVICI-compatible batch data
    """
    n_samples = x_data.shape[0]
    n_variables = x_data.shape[1]
    
    # Determine standardization type
    standardization_type = "count" if is_count_data else "default"
    
    batch = {
        'x': x_data,
        'g': ground_truth_adjacency,
        'is_count_data': jnp.array(is_count_data),
        'target_variable': target_variable,
        'variable_order': variable_order,
        'metadata': {
            'n_samples': n_samples,
            'n_variables': n_variables,
            'standardization_type': standardization_type
        }
    }
    
    return batch


# High-level API functions with validation
def samples_to_avici_format_validated(
    samples: SampleList,
    variable_order: VariableOrder,
    target_variable: str,
    standardize: bool = True,
    standardization_type: StandardizationType = "default"
) -> jnp.ndarray:
    """
    High-level function that validates inputs then converts.
    
    This is the main user-facing function.
    
    Args:
        samples: List of Sample objects (observational + interventional)
        variable_order: Ordered list of variable names for consistent indexing
        target_variable: Name of target variable for conditioning
        standardize: Whether to standardize variable values
        standardization_type: Type of standardization ("default", "count", or "none")
        
    Returns:
        JAX array of shape [N, d, 3] ready for AVICI models
        
    Raises:
        ValueError: If inputs are invalid
    """
    from .validation import validate_conversion_inputs
    
    # Validate inputs
    validate_conversion_inputs(samples, variable_order, target_variable)
    
    # Compute standardization if needed
    if standardize and standardization_type != "none":
        values = extract_values_matrix(samples, variable_order)
        std_params = compute_standardization_params(values, standardization_type)
    else:
        std_params = None
    
    # Convert (pure function)
    return samples_to_avici_format(samples, variable_order, target_variable, std_params)


def create_training_batch_validated(
    scm: pyr.PMap,
    samples: SampleList,
    target_variable: str,
    standardize: bool = True,
    is_count_data: bool = False
) -> AVICIDataBatch:
    """
    High-level function to create training batch with validation.
    
    This is the main user-facing function.
    
    Args:
        scm: The structural causal model
        samples: List of Sample objects
        target_variable: Name of target variable
        standardize: Whether to standardize the data
        is_count_data: Whether the data should be treated as count data
        
    Returns:
        Dictionary containing AVICI-compatible batch data
        
    Raises:
        ValueError: If inputs are invalid
    """
    from .validation import validate_training_batch_inputs
    
    # Validate inputs
    validate_training_batch_inputs(scm, samples, target_variable)
    
    # Get variable order from SCM
    variables = scm['variables']
    if isinstance(variables, (set, frozenset, pyr.PSet)):
        variable_order = sorted(variables)
    else:
        variable_order = list(variables)
    
    # Convert data
    std_type = "count" if is_count_data else "default"
    x_data = samples_to_avici_format_validated(
        samples, variable_order, target_variable, standardize, std_type
    )
    
    # Extract ground truth
    g_matrix = extract_ground_truth_adjacency(scm, variable_order)
    
    # Assemble batch
    return create_avici_batch_from_components(
        x_data, g_matrix, target_variable, variable_order, is_count_data
    )
