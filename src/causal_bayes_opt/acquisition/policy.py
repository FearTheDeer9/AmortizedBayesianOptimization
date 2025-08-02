"""
Minimal policy module stub for backward compatibility.

This module wraps the unified_policy implementation to maintain
compatibility with existing imports while using the new architecture.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import haiku as hk

# Import from the actual implementation
from ..policies.unified_policy import create_unified_policy


@dataclass
class PolicyConfig:
    """Configuration for acquisition policy network."""
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout_rate: float = 0.1
    feature_dim: int = 64


# For backward compatibility - create dummy classes
class AlternatingAttentionEncoder:
    """Deprecated - use create_unified_policy instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use create_unified_policy instead")

class AcquisitionPolicyNetwork:
    """Deprecated - use create_unified_policy instead."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use create_unified_policy instead")


def create_acquisition_policy(config: PolicyConfig) -> hk.Transformed:
    """Create acquisition policy network.
    
    Args:
        config: Policy configuration
        
    Returns:
        Haiku transformed policy network
    """
    policy_fn = create_unified_policy(
        hidden_dim=config.hidden_dim,
        num_layers=config.n_layers,
        dropout_rate=config.dropout_rate
    )
    return hk.transform(policy_fn)


def sample_intervention_from_policy(
    policy_fn: hk.Transformed,
    params: Any,
    state_tensor: jnp.ndarray,
    target_idx: int,
    rng_key: jax.Array,
    temperature: float = 1.0
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Sample intervention from policy network.
    
    Args:
        policy_fn: Transformed policy network
        params: Network parameters
        state_tensor: State representation tensor
        target_idx: Index of target variable
        rng_key: JAX random key
        temperature: Sampling temperature
        
    Returns:
        Tuple of (intervention dict, log probabilities)
    """
    # Apply policy network
    outputs = policy_fn.apply(params, rng_key, state_tensor, target_idx)
    
    # Sample variable
    var_key, val_key = jax.random.split(rng_key)
    var_logits = outputs['variable_logits'] / temperature
    var_probs = jax.nn.softmax(var_logits)
    selected_var = jax.random.categorical(var_key, var_logits)
    
    # Sample value
    value_params = outputs['value_params']
    mean = value_params[selected_var, 0]
    log_std = value_params[selected_var, 1]
    std = jnp.exp(jnp.clip(log_std, -5, 2))
    value = mean + std * jax.random.normal(val_key)
    
    intervention = {
        'variable_idx': int(selected_var),
        'value': float(value)
    }
    
    log_probs = {
        'variable': float(jnp.log(var_probs[selected_var])),
        'value': float(-0.5 * jnp.log(2 * jnp.pi) - log_std - 
                      0.5 * ((value - mean) / std) ** 2)
    }
    
    return intervention, log_probs


def compute_action_log_probability(
    policy_fn: hk.Transformed,
    params: Any,
    state_tensor: jnp.ndarray,
    target_idx: int,
    action: Dict[str, Any],
    rng_key: jax.Array
) -> float:
    """Compute log probability of an action under the policy.
    
    Args:
        policy_fn: Transformed policy network
        params: Network parameters
        state_tensor: State representation tensor
        target_idx: Index of target variable
        action: Action dict with 'variable_idx' and 'value'
        rng_key: JAX random key
        
    Returns:
        Log probability of the action
    """
    outputs = policy_fn.apply(params, rng_key, state_tensor, target_idx)
    
    # Variable selection log probability
    var_logits = outputs['variable_logits']
    var_log_probs = jax.nn.log_softmax(var_logits)
    var_log_prob = var_log_probs[action['variable_idx']]
    
    # Value log probability
    value_params = outputs['value_params']
    mean = value_params[action['variable_idx'], 0]
    log_std = value_params[action['variable_idx'], 1]
    std = jnp.exp(jnp.clip(log_std, -5, 2))
    
    value_log_prob = -0.5 * jnp.log(2 * jnp.pi) - log_std - \
                     0.5 * ((action['value'] - mean) / std) ** 2
    
    return float(var_log_prob + value_log_prob)


def compute_policy_entropy(
    policy_fn: hk.Transformed,
    params: Any,
    state_tensor: jnp.ndarray,
    target_idx: int,
    rng_key: jax.Array
) -> float:
    """Compute entropy of the policy distribution.
    
    Args:
        policy_fn: Transformed policy network
        params: Network parameters
        state_tensor: State representation tensor
        target_idx: Index of target variable
        rng_key: JAX random key
        
    Returns:
        Policy entropy
    """
    outputs = policy_fn.apply(params, rng_key, state_tensor, target_idx)
    
    # Variable selection entropy
    var_logits = outputs['variable_logits']
    var_probs = jax.nn.softmax(var_logits)
    var_entropy = -jnp.sum(var_probs * jax.nn.log_softmax(var_logits))
    
    # Value entropy (continuous Gaussian)
    value_params = outputs['value_params']
    log_stds = value_params[:, 1]
    # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
    value_entropies = 0.5 * jnp.log(2 * jnp.pi * jnp.e) + log_stds
    # Expected value entropy
    value_entropy = jnp.sum(var_probs * value_entropies)
    
    return float(var_entropy + value_entropy)


def analyze_policy_output(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze policy network outputs for debugging.
    
    Args:
        outputs: Policy network outputs
        
    Returns:
        Analysis dictionary
    """
    var_logits = outputs['variable_logits']
    value_params = outputs['value_params']
    
    var_probs = jax.nn.softmax(var_logits)
    top_vars = jnp.argsort(-var_probs)[:3]
    
    return {
        'top_variables': [(int(idx), float(var_probs[idx])) for idx in top_vars],
        'entropy': float(-jnp.sum(var_probs * jnp.log(var_probs + 1e-8))),
        'value_means': float(jnp.mean(value_params[:, 0])),
        'value_stds': float(jnp.mean(jnp.exp(value_params[:, 1])))
    }


def validate_policy_output(outputs: Dict[str, Any], n_vars: int) -> bool:
    """Validate policy network outputs.
    
    Args:
        outputs: Policy network outputs
        n_vars: Expected number of variables
        
    Returns:
        True if outputs are valid
    """
    if 'variable_logits' not in outputs or 'value_params' not in outputs:
        return False
        
    var_logits = outputs['variable_logits']
    value_params = outputs['value_params']
    
    if var_logits.shape != (n_vars,):
        return False
        
    if value_params.shape != (n_vars, 2):
        return False
        
    if jnp.any(jnp.isnan(var_logits)) or jnp.any(jnp.isnan(value_params)):
        return False
        
    return True