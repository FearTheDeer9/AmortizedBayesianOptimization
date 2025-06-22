"""
Utility functions for the unified parent set model.
"""

from typing import List, Dict, Any
import jax.numpy as jnp
import warnings
from math import comb

from .config import TargetAwareConfig


def compute_adaptive_max_parents(n_variables: int, config: TargetAwareConfig) -> int:
    """
    Compute adaptive max_parents based on graph size and density constraints.
    
    This addresses the hard-coded max_parents=5 scalability issue.
    
    Args:
        n_variables: Number of variables in the graph
        config: Configuration containing density constraints
        
    Returns:
        Adaptive max_parents value
    """
    if config.max_parent_size is not None:
        # Explicit override - respect user choice but validate bounds
        return max(
            config.min_max_parents,
            min(config.max_parent_size, config.max_max_parents, n_variables - 1)
        )
    
    # Adaptive computation based on density
    max_possible = n_variables - 1  # Can't include target itself
    
    # Compute based on density constraint
    density_based = int(config.max_parent_density * max_possible)
    
    # Apply bounds
    adaptive_max = max(
        config.min_max_parents,
        min(density_based, config.max_max_parents, max_possible)
    )
    
    return adaptive_max


def validate_unified_config(config: TargetAwareConfig, n_variables: int) -> bool:
    """
    Validate that configuration is appropriate for the given graph size.
    
    Args:
        config: Configuration to validate
        n_variables: Number of variables in the graph
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if n_variables < 2:
        raise ValueError("Graph must have at least 2 variables")
        
    max_parents = compute_adaptive_max_parents(n_variables, config)
    
    # The actual constraint is max_parents < (n_variables - 1) since target can't be its own parent
    max_possible_parents = n_variables - 1
    if max_parents > max_possible_parents:
        raise ValueError(
            f"Computed max_parents ({max_parents}) must be <= max_possible_parents ({max_possible_parents}) for {n_variables} variables"
        )
        
    # Check computational feasibility
    total_parent_sets = sum(
        comb(n_variables - 1, k) for k in range(max_parents + 1)
    )
    
    # Warn if extremely large number of parent sets
    if total_parent_sets > 10000:
        warnings.warn(
            f"Large number of parent sets ({total_parent_sets}) may impact performance. "
            f"Consider reducing max_parent_density from {config.max_parent_density}"
        )
    
    return True


def count_total_parent_sets(n_variables: int, max_parents: int) -> int:
    """
    Count total number of possible parent sets.
    
    Args:
        n_variables: Number of variables in graph
        max_parents: Maximum parent set size
        
    Returns:
        Total number of parent sets
    """
    # Parent sets can have size 0 to max_parents
    # Target variable cannot be its own parent, so n_variables - 1 possible parents
    possible_parents = n_variables - 1
    actual_max = min(max_parents, possible_parents)
    
    total = sum(comb(possible_parents, k) for k in range(actual_max + 1))
    return total


def estimate_memory_usage(config: TargetAwareConfig, n_variables: int, batch_size: int = 1) -> Dict[str, float]:
    """
    Estimate memory usage for the unified model.
    
    Args:
        config: Model configuration
        n_variables: Number of variables
        batch_size: Batch size for training
        
    Returns:
        Dictionary with memory estimates in MB
    """
    max_parents = compute_adaptive_max_parents(n_variables, config)
    k = count_total_parent_sets(n_variables, max_parents)
    
    # Estimate parameter counts (based on analysis results)
    base_params = 4_565_768  # From original model analysis
    
    # Add parameters for target conditioning
    if config.enable_target_conditioning:
        target_params = config.target_embedding_dim * n_variables
        base_params += target_params
    
    # Add parameters for mechanism prediction
    if config.predict_mechanisms:
        n_types = len(config.mechanism_types)
        # Mechanism type classifier
        mech_type_params = config.dim * (config.dim // 2) + (config.dim // 2) * (k * n_types)
        # Parameter regressor  
        mech_param_params = config.dim * (config.dim // 2) + (config.dim // 2) * (k * n_types * config.mechanism_param_dim)
        base_params += mech_type_params + mech_param_params
    
    # Estimate activation memory (major component)
    hidden_dim = config.dim
    if config.enable_target_conditioning:
        hidden_dim += config.target_embedding_dim
    
    # Transformer activations: batch_size * n_variables * hidden_dim * n_layers
    transformer_memory = batch_size * n_variables * hidden_dim * config.layers * 4  # 4 bytes per float32
    
    # Parent set logits and features
    parent_set_memory = batch_size * k * 4  # Parent set logits
    
    # Total memory in MB
    param_memory_mb = (base_params * 4) / (1024 ** 2)  # Parameters
    activation_memory_mb = (transformer_memory + parent_set_memory) / (1024 ** 2)  # Activations
    total_memory_mb = param_memory_mb + activation_memory_mb
    
    return {
        "parameters_mb": param_memory_mb,
        "activations_mb": activation_memory_mb,
        "total_mb": total_memory_mb,
        "estimated_params": base_params,
        "parent_sets": k
    }


def suggest_optimal_config(n_variables: int, memory_budget_mb: float = 2048) -> TargetAwareConfig:
    """
    Suggest optimal configuration based on graph size and memory constraints.
    
    Args:
        n_variables: Number of variables in graph
        memory_budget_mb: Available memory in MB
        
    Returns:
        Suggested configuration
    """
    # Start with base configuration
    if n_variables <= 5:
        base_config = SMALL_GRAPH_CONFIG
    elif n_variables <= 15:
        base_config = MEDIUM_GRAPH_CONFIG  
    else:
        base_config = LARGE_GRAPH_CONFIG
    
    # Adjust based on memory budget
    test_config = base_config
    memory_estimate = estimate_memory_usage(test_config, n_variables)
    
    if memory_estimate["total_mb"] > memory_budget_mb:
        # Reduce configuration to fit memory budget
        if n_variables > 10:
            # For large graphs, reduce density and layer count
            adjusted_config = TargetAwareConfig(
                layers=6,                    # Reduce layers
                dim=96,                      # Reduce hidden dimension
                max_parent_density=0.15,     # Lower density
                min_max_parents=2,
                max_max_parents=6,
                enable_target_conditioning=True,
                predict_mechanisms=False     # Disable mechanism prediction for memory
            )
        else:
            # For medium graphs, just reduce complexity
            adjusted_config = TargetAwareConfig(
                layers=6,
                dim=112,
                max_parent_density=0.25,
                min_max_parents=2,
                max_max_parents=7,
                enable_target_conditioning=True,
                predict_mechanisms=False
            )
        
        return adjusted_config
    
    return base_config


def validate_model_outputs(
    outputs: Dict[str, Any],
    expected_keys: List[str],
    parent_sets: List[frozenset],
    config: TargetAwareConfig
) -> bool:
    """
    Validate model outputs have expected structure and shapes.
    
    Args:
        outputs: Model outputs to validate
        expected_keys: Expected keys in outputs
        parent_sets: Parent sets being predicted
        config: Model configuration
        
    Returns:
        True if outputs are valid
        
    Raises:
        ValueError: If outputs are invalid
    """
    # Check required keys
    for key in expected_keys:
        if key not in outputs:
            raise ValueError(f"Missing required output key: {key}")
    
    # Validate parent set logits
    if "parent_set_logits" in outputs:
        logits = outputs["parent_set_logits"]
        expected_k = len(parent_sets)
        
        if logits.shape != (expected_k,):
            raise ValueError(
                f"parent_set_logits shape {logits.shape} != expected ({expected_k},)"
            )
        
        if not jnp.isfinite(logits).all():
            raise ValueError("parent_set_logits contains non-finite values")
    
    # Validate mechanism outputs if present
    if config.predict_mechanisms and "mechanism_predictions" in outputs:
        mech_outputs = outputs["mechanism_predictions"]
        k = len(parent_sets)
        n_types = len(config.mechanism_types)
        
        if "mechanism_type_logits" in mech_outputs:
            type_logits = mech_outputs["mechanism_type_logits"]
            if type_logits.shape != (k, n_types):
                raise ValueError(
                    f"mechanism_type_logits shape {type_logits.shape} != expected ({k}, {n_types})"
                )
        
        if "mechanism_parameters" in mech_outputs:
            params = mech_outputs["mechanism_parameters"]
            expected_shape = (k, n_types, config.mechanism_param_dim)
            if params.shape != expected_shape:
                raise ValueError(
                    f"mechanism_parameters shape {params.shape} != expected {expected_shape}"
                )
    
    return True


# Import preset configurations for easy access
from .config import (
    SMALL_GRAPH_CONFIG,
    MEDIUM_GRAPH_CONFIG, 
    LARGE_GRAPH_CONFIG,
    MECHANISM_AWARE_CONFIG
)