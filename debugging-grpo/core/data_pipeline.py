"""
Clean data pipeline for GRPO training.

Focuses on:
1. Simple, efficient buffer→tensor conversion
2. Proper per-variable normalization (preserves relationships)
3. Surrogate integration hooks (for future information gain)
4. Strong gradient flow optimization
"""

import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleBuffer:
    """Minimal buffer interface focused on efficiency."""
    
    def __init__(self):
        self.samples = []
        self.interventions = []  # Track which samples are interventions
    
    def add_sample(self, sample: Dict[str, float], is_intervention: bool = False):
        """Add sample to buffer."""
        self.samples.append(sample)
        self.interventions.append(is_intervention)
    
    def get_recent_samples(self, n: int) -> List[Dict[str, float]]:
        """Get n most recent samples."""
        return self.samples[-n:] if n <= len(self.samples) else self.samples
    
    def get_recent_interventions(self, n: int) -> List[bool]:
        """Get intervention indicators for recent samples."""
        return self.interventions[-n:] if n <= len(self.interventions) else self.interventions
    
    def size(self) -> int:
        return len(self.samples)
    
    def get_variables(self) -> List[str]:
        """Get all variable names present in buffer."""
        if not self.samples:
            return []
        return sorted(self.samples[0].keys())


def create_clean_tensor(
    buffer: SimpleBuffer,
    target_variable: str,
    max_history: int = 50,
    surrogate_fn: Optional[Callable] = None
) -> Tuple[jnp.ndarray, List[str]]:
    """
    Create clean 4-channel tensor optimized for gradient flow.
    
    Design principles:
    1. Preserve variable relationships (per-variable normalization)
    2. Minimal processing (avoid information loss)
    3. Surrogate integration ready
    4. Strong gradient flow
    
    Args:
        buffer: Simple buffer with samples
        target_variable: Target variable name
        max_history: Maximum temporal history to include
        surrogate_fn: Optional surrogate for parent probability predictions
        
    Returns:
        Tuple of (tensor [T, n_vars, 4], variable_order)
        
    Channels:
        0. Values: Per-variable normalized values (preserves X→Y relationships)
        1. Target: Binary indicator [0,0,1,0] for target position
        2. Intervention: Binary indicator for intervention history  
        3. Probs: Parent probabilities (0.5 uniform or surrogate predictions)
    """
    if buffer.size() == 0:
        raise ValueError("Buffer is empty")
    
    # Get variable order and dimensions
    variables = buffer.get_variables()
    if target_variable not in variables:
        raise ValueError(f"Target {target_variable} not in buffer variables: {variables}")
    
    n_vars = len(variables)
    target_idx = variables.index(target_variable)
    
    # Get recent samples
    recent_samples = buffer.get_recent_samples(max_history)
    recent_interventions = buffer.get_recent_interventions(max_history)
    T = len(recent_samples)
    
    # Initialize tensor
    tensor = jnp.zeros((T, n_vars, 4))
    
    # Fill tensor efficiently
    for t, (sample, is_intervention) in enumerate(zip(recent_samples, recent_interventions)):
        
        # Channel 0: Values (raw, will normalize per-variable later)
        values = jnp.array([sample.get(var, 0.0) for var in variables])
        
        # Channel 1: Target indicator
        target_indicator = jnp.zeros(n_vars)
        target_indicator = target_indicator.at[target_idx].set(1.0)
        
        # Channel 2: Intervention indicator
        intervention_indicator = jnp.ones(n_vars) * (1.0 if is_intervention else 0.0)
        
        # Channel 3: Parent probabilities
        if surrogate_fn is not None:
            # Future: Get surrogate predictions
            # For now, placeholder for surrogate integration
            parent_probs = jnp.full(n_vars, 0.5)  # TODO: surrogate_fn(buffer_state)
        else:
            # No surrogate: uniform probabilities (max entropy)
            parent_probs = jnp.full(n_vars, 0.5)
        
        # Set tensor values
        tensor = tensor.at[t, :, 0].set(values)
        tensor = tensor.at[t, :, 1].set(target_indicator)
        tensor = tensor.at[t, :, 2].set(intervention_indicator)
        tensor = tensor.at[t, :, 3].set(parent_probs)
    
    # CRITICAL: Per-variable normalization (preserves relationships)
    tensor = _normalize_per_variable(tensor)
    
    return tensor, variables


def _normalize_per_variable(tensor: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize each variable independently to preserve relationships.
    
    Key insight: X and Y have 10x relationship that must be preserved.
    Global normalization destroys this, per-variable preserves it.
    """
    T, n_vars, n_channels = tensor.shape
    
    # Only normalize values channel (channel 0)
    values = tensor[:, :, 0]  # [T, n_vars]
    
    # Per-variable normalization
    normalized_values = jnp.zeros_like(values)
    
    for var_idx in range(n_vars):
        var_values = values[:, var_idx]  # [T] - this variable across time
        
        # Normalize this variable relative to its own distribution
        var_mean = jnp.mean(var_values)
        var_std = jnp.std(var_values) + 1e-8
        
        normalized_var = (var_values - var_mean) / var_std
        normalized_values = normalized_values.at[:, var_idx].set(normalized_var)
    
    # Update tensor with normalized values
    return tensor.at[:, :, 0].set(normalized_values)


def compute_information_gain(
    buffer_before: SimpleBuffer,
    buffer_after: SimpleBuffer, 
    surrogate_fn: Callable,
    target_variable: str
) -> float:
    """
    Compute information gain for future surrogate integration.
    
    This is a hook for future use when surrogate is integrated.
    Currently returns 0.0 for compatibility.
    """
    if surrogate_fn is None:
        return 0.0
    
    # Future implementation:
    # tensor_before = create_clean_tensor(buffer_before, target_variable, surrogate_fn=None)
    # tensor_after = create_clean_tensor(buffer_after, target_variable, surrogate_fn=None)
    # posterior_before = surrogate_fn(tensor_before)
    # posterior_after = surrogate_fn(tensor_after)
    # entropy_before = compute_entropy(posterior_before)
    # entropy_after = compute_entropy(posterior_after)
    # return entropy_before - entropy_after
    
    logger.debug("Information gain computation: surrogate not implemented yet")
    return 0.0


def validate_tensor_quality(tensor: jnp.ndarray, variables: List[str], target_variable: str):
    """Validate tensor has good properties for learning."""
    T, n_vars, n_channels = tensor.shape
    
    print(f"🔍 TENSOR QUALITY VALIDATION:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Variables: {variables}")
    print(f"  Target: {target_variable}")
    
    # Check channel 0 (values) for distinguishability
    values = tensor[:, :, 0]
    var_means = jnp.mean(values, axis=0)  # Mean per variable
    distinguishability = jnp.std(var_means)
    
    print(f"  Variable distinguishability: {distinguishability:.6f}")
    
    if distinguishability < 0.01:
        print(f"  ❌ CRITICAL: Variables not distinguishable!")
        return False
    
    # Check for information content
    total_variation = jnp.std(tensor)
    non_zero_ratio = jnp.sum(tensor != 0.0) / tensor.size
    
    print(f"  Total variation: {total_variation:.6f}")
    print(f"  Non-zero ratio: {non_zero_ratio:.3f}")
    
    if total_variation < 0.1:
        print(f"  ⚠️ WARNING: Low variation - may be insufficient signal")
    elif non_zero_ratio < 0.2:
        print(f"  ⚠️ WARNING: Mostly zeros - check data population")
    else:
        print(f"  ✅ Tensor quality looks good")
        return True
    
    return distinguishability > 0.01