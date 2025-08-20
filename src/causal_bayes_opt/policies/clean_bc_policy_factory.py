#!/usr/bin/env python3
"""
Shared BC policy factory to prevent Haiku parameter mismatches.

This module provides a centralized factory for creating BC (Behavioral Cloning)
policy functions. By using the same factory function in both training and
inference, we ensure Haiku creates identical module paths, preventing the
"Unable to retrieve parameter" errors.

Key Principle: 
- Functions must be defined in the SAME way across all contexts
- Use this factory everywhere BC policies are needed
"""

import logging
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

logger = logging.getLogger(__name__)


def create_clean_bc_policy(hidden_dim: int = 256, architecture: str = "permutation_invariant") -> callable:
    """
    Create BC policy function with consistent module paths.
    
    This function returns a Haiku module function that can be transformed
    and will produce consistent parameter structures across different contexts.
    
    Args:
        hidden_dim: Hidden dimension for policy network
        architecture: Architecture type - "simple", "attention", "alternating_attention", or "permutation_invariant"
                     Default is "permutation_invariant" for BC as it handles permutation symmetries better
        
    Returns:
        Policy function ready for hk.transform
    """
    # Import from clean_policy_factory to reuse existing implementations
    import sys
    from pathlib import Path
    # Add parent to path to import from clean_policy_factory
    sys.path.insert(0, str(Path(__file__).parent))
    from clean_policy_factory import (
        create_simple_policy,
        create_attention_policy, 
        create_alternating_attention_policy
    )
    from permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
    
    # Select architecture - reuse the same implementations as GRPO
    if architecture == "permutation_invariant":
        return create_permutation_invariant_alternating_policy(hidden_dim)
    elif architecture == "alternating_attention":
        return create_alternating_attention_policy(hidden_dim)
    elif architecture == "attention":
        return create_attention_policy(hidden_dim)
    elif architecture == "simple":
        return create_simple_policy(hidden_dim)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def create_bc_loss_fn(policy_fn: hk.Transformed) -> callable:
    """
    Create BC loss function for training.
    
    Args:
        policy_fn: Transformed BC policy
        
    Returns:
        Loss function for behavioral cloning
    """
    def loss_fn(params: hk.Params,
                rng: jax.Array,
                tensor: jnp.ndarray,
                target_idx: int,
                expert_var_idx: int,
                expert_value: float) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute BC loss (cross-entropy for variable, MSE for value).
        
        Args:
            params: Policy parameters
            rng: Random key (unused but kept for consistency)
            tensor: Input tensor [T, n_vars, 3]
            target_idx: Target variable index
            expert_var_idx: Expert's chosen variable index
            expert_value: Expert's intervention value
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Get policy predictions
        outputs = policy_fn.apply(params, rng, tensor, target_idx)
        var_logits = outputs['variable_logits']
        value_params = outputs['value_params']
        
        # Variable selection loss (cross-entropy)
        var_loss = -jax.nn.log_softmax(var_logits)[expert_var_idx]
        
        # Value prediction loss (negative log likelihood of Gaussian)
        mean = value_params[expert_var_idx, 0]
        log_std = value_params[expert_var_idx, 1]
        std = jnp.exp(jnp.clip(log_std, -5, 2))  # Clip for stability
        
        # NLL of expert value under predicted Gaussian
        value_loss = 0.5 * jnp.log(2 * jnp.pi) + log_std + 0.5 * ((expert_value - mean) / std) ** 2
        
        # Combined loss
        total_loss = var_loss + 0.5 * value_loss  # Weight value loss less
        
        metrics = {
            'total_loss': total_loss,
            'var_loss': var_loss,
            'value_loss': value_loss,
            'predicted_var_probs': jax.nn.softmax(var_logits),
            'predicted_mean': mean,
            'predicted_std': std
        }
        
        return total_loss, metrics
    
    return loss_fn


def verify_bc_parameter_compatibility(
    saved_params: Dict[str, Any],
    model_fn: hk.Transformed,
    dummy_input: jnp.ndarray,
    target_idx: int = 0
) -> bool:
    """
    Verify that saved BC parameters are compatible with model.
    
    Args:
        saved_params: Parameters loaded from checkpoint
        model_fn: Transformed BC model function
        dummy_input: Dummy input tensor for testing
        target_idx: Target variable index
        
    Returns:
        True if parameters are compatible
    """
    try:
        # Try to apply the model with saved parameters
        rng = random.PRNGKey(0)
        output = model_fn.apply(saved_params, rng, dummy_input, target_idx)
        
        # Check output structure
        required_keys = {'variable_logits', 'value_params'}
        if not all(k in output for k in required_keys):
            logger.error(f"Missing output keys. Got: {output.keys()}, need: {required_keys}")
            return False
            
        # Verify shapes
        n_vars = dummy_input.shape[1]
        if output['variable_logits'].shape != (n_vars,):
            logger.error(f"Wrong variable_logits shape: {output['variable_logits'].shape}")
            return False
            
        if output['value_params'].shape != (n_vars, 2):
            logger.error(f"Wrong value_params shape: {output['value_params'].shape}")
            return False
            
        logger.info("✓ BC parameters are compatible with model")
        return True
        
    except Exception as e:
        logger.error(f"BC parameter compatibility check failed: {e}")
        return False