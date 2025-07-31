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


def create_clean_bc_policy(hidden_dim: int = 256) -> callable:
    """
    Create BC policy function with consistent module paths.
    
    This function returns a Haiku module function that can be transformed
    and will produce consistent parameter structures across different contexts.
    
    Args:
        hidden_dim: Hidden dimension for policy network
        
    Returns:
        Policy function ready for hk.transform
    """
    def bc_policy_fn(tensor: jnp.ndarray, target_idx: int) -> Dict[str, jnp.ndarray]:
        """
        BC policy network for intervention selection.
        
        Args:
            tensor: Input tensor of shape [T, n_vars, C] where C can be 3 or 5
            target_idx: Index of target variable (to mask from selection)
            
        Returns:
            Dictionary with 'variable_logits' and 'value_params'
        """
        # Handle both 3 and 5 channel inputs
        T, n_vars, n_channels = tensor.shape
        
        if n_channels == 3:
            # Legacy 3-channel format - pad with zeros
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor)
            tensor = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # Flatten time and variable dimensions
        flat_input = tensor.reshape(T * n_vars, n_channels)
        
        # Simple MLP encoder
        net = hk.Sequential([
            hk.Linear(hidden_dim),
            jax.nn.relu,
            hk.Linear(hidden_dim),
            jax.nn.relu,
            hk.Linear(hidden_dim // 2)
        ])
        
        # Encode all timesteps
        encoded = net(flat_input)  # [T * n_vars, hidden_dim // 2]
        
        # Aggregate across time (mean pooling)
        encoded = encoded.reshape(T, n_vars, hidden_dim // 2)
        aggregated = jnp.mean(encoded, axis=0)  # [n_vars, hidden_dim // 2]
        
        # Variable selection head
        var_head = hk.Linear(1)
        var_logits = var_head(aggregated).squeeze(-1)  # [n_vars]
        
        # Mask target variable (set to -inf)
        mask = jnp.arange(n_vars) == target_idx
        var_logits = jnp.where(mask, -jnp.inf, var_logits)
        
        # Value prediction head (mean and log_std for each variable)
        value_head = hk.Linear(2)  # mean and log_std
        value_params = value_head(aggregated)  # [n_vars, 2]
        
        return {
            'variable_logits': var_logits,
            'value_params': value_params
        }
    
    return bc_policy_fn


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