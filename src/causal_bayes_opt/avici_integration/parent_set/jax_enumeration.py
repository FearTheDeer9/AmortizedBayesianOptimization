"""
JAX-compatible parent set enumeration with tensor-based operations.

This module replaces the Python-based enumeration in enumeration.py with
JAX-compatible tensor operations that can be compiled with @jax.jit.

Key improvements:
1. Pre-computed parent set lookup tables as JAX arrays
2. Integer-based parent set IDs instead of frozensets
3. Fixed-size tensor operations with padding
4. No Python loops or dynamic control flow in forward pass
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, FrozenSet
from itertools import combinations
import numpy as np


def precompute_parent_set_tables(
    n_vars: int, 
    max_parent_size: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[FrozenSet[int], int]]:
    """
    Pre-compute all possible parent sets as lookup tables for JAX compilation.
    
    This function runs once during model initialization to create static
    lookup tables that can be used in the JAX-compiled forward pass.
    
    Args:
        n_vars: Number of variables in the graph
        max_parent_size: Maximum parent set size to consider
        
    Returns:
        parent_set_indicators: [max_parent_sets, n_vars] binary indicator matrix
        parent_set_sizes: [max_parent_sets] size of each parent set
        parent_set_map: Dictionary mapping frozensets to indices (for debugging)
    """
    # Generate all possible parent sets using integer indices
    all_parent_sets = []
    parent_set_map = {}
    
    # Add empty parent set first (always index 0)
    all_parent_sets.append(frozenset())
    parent_set_map[frozenset()] = 0
    
    # Generate all combinations up to max_parent_size
    potential_parents = list(range(n_vars))  # Use integers instead of strings
    
    for size in range(1, min(max_parent_size + 1, n_vars)):
        for parent_combo in combinations(potential_parents, size):
            parent_set = frozenset(parent_combo)
            all_parent_sets.append(parent_set)
            parent_set_map[parent_set] = len(all_parent_sets) - 1
    
    max_parent_sets = len(all_parent_sets)
    
    # Create indicator matrix: [max_parent_sets, n_vars]
    parent_set_indicators = np.zeros((max_parent_sets, n_vars), dtype=np.float32)
    parent_set_sizes = np.zeros(max_parent_sets, dtype=np.int32)
    
    for i, parent_set in enumerate(all_parent_sets):
        parent_set_sizes[i] = len(parent_set)
        for parent_idx in parent_set:
            if parent_idx < n_vars:  # Safety check
                parent_set_indicators[i, parent_idx] = 1.0
    
    return (
        jnp.array(parent_set_indicators), 
        jnp.array(parent_set_sizes),
        parent_set_map
    )


def compute_adaptive_k_jax(n_vars: int, max_parent_size: int) -> int:
    """
    Compute adaptive k value for JAX compilation.
    
    This is a static computation that doesn't depend on data.
    """
    # Conservative estimate: include reasonable number of top parent sets
    base_k = min(10, 2 ** min(max_parent_size, 4))
    
    # Scale with number of variables
    if n_vars <= 5:
        return min(base_k, 8)
    elif n_vars <= 10:
        return min(base_k, 12)
    else:
        return min(base_k, 16)


@jax.jit
def filter_parent_sets_for_target_jax(
    parent_set_indicators: jnp.ndarray,  # [max_parent_sets, n_vars]
    parent_set_sizes: jnp.ndarray,       # [max_parent_sets]
    target_idx: int,                     # Target variable index
    max_parent_size: int                 # Maximum parent set size
) -> jnp.ndarray:
    """
    Filter parent sets to exclude target variable using JAX operations.
    
    Args:
        parent_set_indicators: Binary indicator matrix for all parent sets
        parent_set_sizes: Size of each parent set
        target_idx: Index of target variable to exclude
        max_parent_size: Maximum allowed parent set size
        
    Returns:
        valid_mask: [max_parent_sets] boolean mask for valid parent sets
    """
    # Mask out parent sets that include the target variable
    target_mask = parent_set_indicators[:, target_idx] == 0.0
    
    # Mask out parent sets that are too large
    size_mask = parent_set_sizes <= max_parent_size
    
    # Combine masks
    valid_mask = target_mask & size_mask
    
    return valid_mask


@jax.jit
def encode_parent_sets_vectorized(
    parent_set_indicators: jnp.ndarray,  # [max_parent_sets, n_vars]
    variable_embeddings: jnp.ndarray,    # [n_vars, dim]
    valid_mask: jnp.ndarray              # [max_parent_sets]
) -> jnp.ndarray:
    """
    Vectorized parent set encoding using JAX operations.
    
    Replaces the Python loop in the original implementation with
    vectorized tensor operations that can be JAX-compiled.
    
    Args:
        parent_set_indicators: Binary indicators for parent sets
        variable_embeddings: Variable embeddings from transformer
        valid_mask: Mask indicating which parent sets are valid
        
    Returns:
        parent_set_embeddings: [max_parent_sets, dim] encoded parent sets
    """
    # Weighted sum of variable embeddings based on parent set indicators
    # [max_parent_sets, n_vars] @ [n_vars, dim] -> [max_parent_sets, dim]
    parent_set_embeddings = jnp.matmul(parent_set_indicators, variable_embeddings)
    
    # Normalize by parent set size (avoid division by zero)
    parent_set_sizes = jnp.sum(parent_set_indicators, axis=1, keepdims=True)  # [max_parent_sets, 1]
    parent_set_sizes = jnp.maximum(parent_set_sizes, 1.0)  # Avoid division by zero
    
    normalized_embeddings = parent_set_embeddings / parent_set_sizes
    
    # Mask out invalid parent sets
    valid_mask_expanded = valid_mask[:, None]  # [max_parent_sets, 1]
    masked_embeddings = jnp.where(
        valid_mask_expanded,
        normalized_embeddings,
        jnp.zeros_like(normalized_embeddings)
    )
    
    return masked_embeddings


def select_top_k_parent_sets_jax(
    parent_set_logits: jnp.ndarray,  # [max_parent_sets]
    valid_mask: jnp.ndarray,         # [max_parent_sets]
    k: int                           # Number of top parent sets to return
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Select top-k parent sets using JAX operations.
    
    Simplified version without JIT to avoid static slicing issues during development.
    """
    # Mask out invalid parent sets by setting their logits to -inf
    masked_logits = jnp.where(valid_mask, parent_set_logits, -jnp.inf)
    
    # Get ALL indices sorted by logits (descending order)
    all_sorted_indices = jnp.argsort(masked_logits)[::-1]
    
    # Clamp k to available number of parent sets
    max_available = all_sorted_indices.shape[0]
    actual_k = jnp.minimum(k, max_available)
    
    # Use jax.lax.dynamic_slice to avoid static slicing requirements
    top_k_indices = jax.lax.dynamic_slice(all_sorted_indices, (0,), (actual_k,))
    
    # Get corresponding logits  
    top_k_logits = masked_logits[top_k_indices]
    
    # For now, return variable-size results (simpler for development)
    # TODO: Add fixed-size padding for production use
    return top_k_indices, top_k_logits


def create_parent_set_lookup(
    n_vars: int,
    variable_names: List[str],
    max_parent_size: int = 3
) -> Dict[str, any]:
    """
    Create complete parent set lookup tables for a given variable configuration.
    
    This function should be called once during model initialization to create
    all the static data structures needed for JAX-compiled parent set operations.
    
    Args:
        n_vars: Number of variables
        variable_names: List of variable names (for debugging/interpretation)
        max_parent_size: Maximum parent set size
        
    Returns:
        Dictionary containing all lookup tables and metadata
    """
    indicators, sizes, mapping = precompute_parent_set_tables(n_vars, max_parent_size)
    
    # Create name-to-index mapping
    name_to_idx = {name: i for i, name in enumerate(variable_names)}
    idx_to_name = {i: name for i, name in enumerate(variable_names)}
    
    # Compute adaptive k for this configuration
    adaptive_k = compute_adaptive_k_jax(n_vars, max_parent_size)
    
    return {
        'parent_set_indicators': indicators,      # [max_parent_sets, n_vars]
        'parent_set_sizes': sizes,               # [max_parent_sets]
        'parent_set_mapping': mapping,           # For debugging
        'name_to_idx': name_to_idx,             # str -> int
        'idx_to_name': idx_to_name,             # int -> str
        'n_vars': n_vars,
        'max_parent_size': max_parent_size,
        'adaptive_k': adaptive_k,
        'max_parent_sets': len(indicators)
    }


def interpret_parent_set_results(
    top_k_indices: jnp.ndarray,
    top_k_logits: jnp.ndarray,
    lookup_tables: Dict[str, any],
    target_variable: str
) -> List[Tuple[FrozenSet[str], float]]:
    """
    Convert JAX tensor results back to interpretable parent sets with names.
    
    This function is used after the JAX-compiled forward pass to convert
    integer indices back to variable names for human interpretation.
    
    Args:
        top_k_indices: [k] tensor of parent set indices
        top_k_logits: [k] tensor of corresponding logits
        lookup_tables: Lookup tables from create_parent_set_lookup
        target_variable: Name of target variable
        
    Returns:
        List of (parent_set, logit) tuples with variable names
    """
    parent_set_mapping = lookup_tables['parent_set_mapping']
    idx_to_name = lookup_tables['idx_to_name']
    target_idx = lookup_tables['name_to_idx'][target_variable]
    
    results = []
    
    for i, (ps_idx, logit) in enumerate(zip(top_k_indices, top_k_logits)):
        ps_idx = int(ps_idx)
        logit = float(logit)
        
        # Skip padding entries
        if ps_idx == -1 or not jnp.isfinite(logit):
            continue
            
        # Find the parent set corresponding to this index
        for parent_set_int, mapped_idx in parent_set_mapping.items():
            if mapped_idx == ps_idx:
                # Convert integer parent set to variable names
                parent_set_names = frozenset(
                    idx_to_name[parent_idx] 
                    for parent_idx in parent_set_int 
                    if parent_idx != target_idx  # Exclude target
                )
                results.append((parent_set_names, logit))
                break
    
    return results


# Test utilities for validation

def validate_jax_enumeration_equivalence(
    variable_names: List[str],
    target_variable: str,
    max_parent_size: int = 3
) -> bool:
    """
    Validate that JAX enumeration produces equivalent results to Python version.
    
    Returns True if outputs are numerically equivalent, False otherwise.
    """
    # This function can be used during development to ensure numerical equivalence
    # Implementation would compare outputs between old and new systems
    pass


def benchmark_jax_enumeration_performance(
    n_vars_list: List[int],
    max_parent_size: int = 3
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark performance improvements from JAX enumeration.
    
    Returns timing comparisons between Python and JAX implementations.
    """
    # This function can be used to measure performance improvements
    # Implementation would run timing comparisons
    pass