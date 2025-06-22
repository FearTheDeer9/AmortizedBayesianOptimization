"""
JAX-compatible parent set encoding with tensor operations.

This module provides JAX-compiled functions for encoding parent sets
using tensor operations instead of Python loops and string operations.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


@jax.jit
def encode_parent_set_jax(
    parent_set_indicator: jnp.ndarray,  # [n_vars] binary indicator
    variable_embeddings: jnp.ndarray    # [n_vars, dim]
) -> jnp.ndarray:
    """
    Encode a single parent set using JAX tensor operations.
    
    Replaces the Python loop-based encoding with vectorized operations.
    
    Args:
        parent_set_indicator: [n_vars] binary array indicating parent variables
        variable_embeddings: [n_vars, dim] embeddings for all variables
        
    Returns:
        parent_set_embedding: [dim] aggregated embedding for the parent set
    """
    # Weighted sum of embeddings based on parent set indicators
    # [n_vars] @ [n_vars, dim] -> [dim]
    weighted_sum = jnp.dot(parent_set_indicator, variable_embeddings)
    
    # Normalize by number of parents (avoid division by zero)
    num_parents = jnp.sum(parent_set_indicator)
    normalized_embedding = weighted_sum / jnp.maximum(num_parents, 1.0)
    
    return normalized_embedding


@jax.jit
def create_target_mask_jax(
    n_vars: int,
    target_idx: int
) -> jnp.ndarray:
    """
    Create a mask that excludes the target variable from parent sets.
    
    Args:
        n_vars: Total number of variables
        target_idx: Index of target variable
        
    Returns:
        mask: [n_vars] binary mask (1 for valid parents, 0 for target)
    """
    mask = jnp.ones(n_vars)
    mask = mask.at[target_idx].set(0.0)
    return mask


@jax.jit
def validate_parent_set_indicator_jax(
    parent_set_indicator: jnp.ndarray,  # [n_vars]
    target_mask: jnp.ndarray,           # [n_vars]
    max_parent_size: int
) -> jnp.ndarray:
    """
    Validate that a parent set indicator is valid using JAX operations.
    
    Args:
        parent_set_indicator: Binary indicator for parent set
        target_mask: Mask excluding target variable
        max_parent_size: Maximum allowed parent set size
        
    Returns:
        is_valid: Scalar boolean indicating if parent set is valid
    """
    # Check that target variable is not included
    target_excluded = jnp.sum(parent_set_indicator * (1.0 - target_mask)) == 0.0
    
    # Check that parent set size is within limits
    size_valid = jnp.sum(parent_set_indicator) <= max_parent_size
    
    return target_excluded & size_valid


@jax.jit
def batch_encode_parent_sets_jax(
    parent_set_indicators: jnp.ndarray,  # [batch_size, n_vars]
    variable_embeddings: jnp.ndarray,    # [n_vars, dim]
    valid_mask: jnp.ndarray              # [batch_size]
) -> jnp.ndarray:
    """
    Encode multiple parent sets in a single vectorized operation.
    
    Args:
        parent_set_indicators: [batch_size, n_vars] binary indicators
        variable_embeddings: [n_vars, dim] variable embeddings
        valid_mask: [batch_size] mask for valid parent sets
        
    Returns:
        encoded_parent_sets: [batch_size, dim] encoded parent set embeddings
    """
    # Vectorized weighted sum: [batch_size, n_vars] @ [n_vars, dim] -> [batch_size, dim]
    weighted_sums = jnp.matmul(parent_set_indicators, variable_embeddings)
    
    # Normalize by parent set sizes
    parent_set_sizes = jnp.sum(parent_set_indicators, axis=1, keepdims=True)  # [batch_size, 1]
    parent_set_sizes = jnp.maximum(parent_set_sizes, 1.0)  # Avoid division by zero
    
    normalized_embeddings = weighted_sums / parent_set_sizes
    
    # Apply validity mask
    valid_mask_expanded = valid_mask[:, None]  # [batch_size, 1]
    masked_embeddings = jnp.where(
        valid_mask_expanded,
        normalized_embeddings,
        jnp.zeros_like(normalized_embeddings)
    )
    
    return masked_embeddings


@jax.jit
def compute_parent_set_statistics_jax(
    parent_set_indicators: jnp.ndarray,  # [max_parent_sets, n_vars]
    valid_mask: jnp.ndarray              # [max_parent_sets]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute statistics about parent sets using JAX operations.
    
    Args:
        parent_set_indicators: Binary indicators for all parent sets
        valid_mask: Mask indicating valid parent sets
        
    Returns:
        mean_parent_set_size: Average size of valid parent sets
        max_parent_set_size: Maximum size of valid parent sets  
        num_valid_parent_sets: Number of valid parent sets
    """
    # Compute sizes for all parent sets
    all_sizes = jnp.sum(parent_set_indicators, axis=1)  # [max_parent_sets]
    
    # Filter to valid parent sets only
    valid_sizes = jnp.where(valid_mask, all_sizes, 0.0)
    
    # Compute statistics
    num_valid = jnp.sum(valid_mask)
    mean_size = jnp.sum(valid_sizes) / jnp.maximum(num_valid, 1.0)
    max_size = jnp.max(jnp.where(valid_mask, all_sizes, 0.0))
    
    return mean_size, max_size, num_valid


@jax.jit
def create_marginal_parent_probabilities_jax(
    parent_set_indicators: jnp.ndarray,  # [k, n_vars]
    parent_set_probs: jnp.ndarray,       # [k]
    target_mask: jnp.ndarray             # [n_vars]
) -> jnp.ndarray:
    """
    Compute marginal parent probabilities using JAX operations.
    
    Args:
        parent_set_indicators: Binary indicators for top-k parent sets
        parent_set_probs: Probabilities for top-k parent sets
        target_mask: Mask excluding target variable
        
    Returns:
        marginal_probs: [n_vars] marginal probability for each variable being a parent
    """
    # Weight parent set indicators by their probabilities
    # [k, n_vars] * [k, 1] -> [k, n_vars]
    weighted_indicators = parent_set_indicators * parent_set_probs[:, None]
    
    # Sum over parent sets to get marginal probabilities
    marginal_probs = jnp.sum(weighted_indicators, axis=0)  # [n_vars]
    
    # Apply target mask (target variable cannot be its own parent)
    masked_marginals = marginal_probs * target_mask
    
    return masked_marginals


@jax.jit
def compute_parent_set_compatibility_matrix_jax(
    parent_set_indicators: jnp.ndarray,  # [max_parent_sets, n_vars]
    variable_similarities: jnp.ndarray   # [n_vars, n_vars]
) -> jnp.ndarray:
    """
    Compute pairwise compatibility matrix between parent sets.
    
    Args:
        parent_set_indicators: Binary indicators for parent sets
        variable_similarities: Pairwise variable similarity matrix
        
    Returns:
        compatibility_matrix: [max_parent_sets, max_parent_sets] compatibility scores
    """
    # For each pair of parent sets, compute similarity based on shared variables
    # and variable similarities
    
    # Compute intersection sizes: how many variables are shared
    # [max_parent_sets, n_vars] @ [n_vars, max_parent_sets] -> [max_parent_sets, max_parent_sets]
    intersection_matrix = jnp.matmul(parent_set_indicators, parent_set_indicators.T)
    
    # Compute union sizes
    set_sizes = jnp.sum(parent_set_indicators, axis=1)  # [max_parent_sets]
    union_matrix = set_sizes[:, None] + set_sizes[None, :] - intersection_matrix
    
    # Jaccard similarity (intersection / union)
    jaccard_similarity = intersection_matrix / jnp.maximum(union_matrix, 1.0)
    
    # Weight by variable similarities for shared variables
    # This is more complex and may not be needed for basic functionality
    
    return jaccard_similarity


# Utility functions for debugging and validation

@jax.jit
def validate_encoding_consistency_jax(
    parent_set_indicator: jnp.ndarray,
    encoded_embedding: jnp.ndarray,
    variable_embeddings: jnp.ndarray,
    tolerance: float = 1e-6
) -> jnp.ndarray:
    """
    Validate that encoded embedding is consistent with manual computation.
    
    Returns boolean indicating if encoding is consistent.
    """
    # Recompute encoding manually
    manual_encoding = encode_parent_set_jax(parent_set_indicator, variable_embeddings)
    
    # Check if embeddings are close
    diff = jnp.abs(encoded_embedding - manual_encoding)
    max_diff = jnp.max(diff)
    
    return max_diff < tolerance


def convert_string_parent_set_to_indicators(
    parent_set: frozenset,
    variable_names: list,
    n_vars: int
) -> jnp.ndarray:
    """
    Convert a string-based parent set to binary indicators for JAX operations.
    
    This function bridges between the old string-based representation
    and the new JAX-compatible integer representation.
    """
    indicators = jnp.zeros(n_vars)
    
    for var_name in parent_set:
        if var_name in variable_names:
            var_idx = variable_names.index(var_name)
            indicators = indicators.at[var_idx].set(1.0)
    
    return indicators


def convert_indicators_to_string_parent_set(
    indicators: jnp.ndarray,
    variable_names: list,
    threshold: float = 0.5
) -> frozenset:
    """
    Convert binary indicators back to string-based parent set.
    
    This function bridges from JAX tensors back to interpretable strings.
    """
    parent_variables = []
    
    for i, indicator in enumerate(indicators):
        if indicator > threshold and i < len(variable_names):
            parent_variables.append(variable_names[i])
    
    return frozenset(parent_variables)