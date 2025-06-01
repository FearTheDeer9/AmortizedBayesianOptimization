"""Pure data extraction functions."""

import jax.numpy as jnp
import pyrsistent as pyr
from typing import List

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]


def extract_values_matrix(
    samples: SampleList,
    variable_order: VariableOrder
) -> jnp.ndarray:
    """
    Pure function: Extract values matrix [N, d] from samples.
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array of shape [N, d] with variable values
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize values matrix
    values_matrix = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Extract values for each sample
    for i, sample in enumerate(samples):
        sample_values = sample['values']
        
        # Extract values in variable order
        for j, var_name in enumerate(variable_order):
            values_matrix = values_matrix.at[i, j].set(float(sample_values[var_name]))
    
    return values_matrix


def extract_intervention_indicators(
    samples: SampleList,
    variable_order: VariableOrder
) -> jnp.ndarray:
    """
    Pure function: Extract intervention indicators [N, d].
    
    Args:
        samples: List of Sample objects
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array of shape [N, d] with intervention indicators
        (1 if variable was intervened upon, 0 otherwise)
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize intervention indicators matrix
    intervention_matrix = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Extract intervention indicators for each sample
    for i, sample in enumerate(samples):
        if sample['intervention_type'] is not None:
            intervention_targets = sample['intervention_targets']
            
            # Set indicators for intervened variables
            for j, var_name in enumerate(variable_order):
                if var_name in intervention_targets:
                    intervention_matrix = intervention_matrix.at[i, j].set(1.0)
    
    return intervention_matrix


def create_target_indicators(
    target_variable: str,
    variable_order: VariableOrder,
    n_samples: int
) -> jnp.ndarray:
    """
    Pure function: Create target indicators [N, d].
    
    Args:
        target_variable: Name of target variable
        variable_order: Ordered list of variable names
        n_samples: Number of samples
        
    Returns:
        JAX array of shape [N, d] with target indicators
        (1 for target variable, 0 for all others)
    """
    n_vars = len(variable_order)
    target_indicators = jnp.zeros((n_samples, n_vars), dtype=jnp.float32)
    
    # Find target variable index
    target_idx = variable_order.index(target_variable)
    
    # Set target indicators (1 for target variable, 0 for others)
    target_indicators = target_indicators.at[:, target_idx].set(1.0)
    
    return target_indicators


def extract_ground_truth_adjacency(
    scm: pyr.PMap,
    variable_order: VariableOrder
) -> jnp.ndarray:
    """
    Pure function: Extract adjacency matrix from SCM.
    
    Args:
        scm: The structural causal model
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array of shape [d, d] with adjacency matrix
        (1 if edge exists from parent to child, 0 otherwise)
    """
    n_vars = len(variable_order)
    g_matrix = jnp.zeros((n_vars, n_vars), dtype=jnp.float32)
    
    # Fill in ground truth edges from SCM
    edges = scm['edges']
    for parent, child in edges:
        parent_idx = variable_order.index(parent)
        child_idx = variable_order.index(child)
        g_matrix = g_matrix.at[parent_idx, child_idx].set(1.0)
    
    return g_matrix
