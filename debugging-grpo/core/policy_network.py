"""
Clean policy network optimized for strong gradient flow.

Design principles:
1. Simple architecture based on working reference patterns
2. Direct input→output mapping (minimal processing layers)  
3. Optimized for gradient magnitude (not sophistication)
4. Handles 4-channel input with surrogate integration ready
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Callable


def create_clean_policy(hidden_dim: int = 256) -> Callable:
    """
    Create clean policy optimized for strong gradients.
    
    Architecture:
    - Input: [T, n_vars, 4] tensor (values, target, intervention, parent_probs)
    - Processing: Simple temporal pooling + direct MLP (like reference)
    - Output: Variable logits + value parameters
    - Focus: Maximum gradient flow, minimal complexity
    
    Based on adaptive_tanh_gaussian_policy.py pattern:
    input → encoder → pooling → emitter (variable + value heads)
    """
    
    def clean_policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        Clean policy network focused on gradient efficiency.
        
        Args:
            tensor_input: [T, n_vars, 4] tensor  
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with variable_logits [n_vars] and value_params [n_vars, 2]
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Validate input
        if n_channels != 4:
            raise ValueError(f"Expected 4 channels, got {n_channels}")
        
        # ENCODER: Simple MLP processing (like reference)
        # Flatten temporal and variable dimensions for direct processing
        flat_input = tensor_input.reshape(T * n_vars, 4)  # [T*n_vars, 4]
        
        # Encoder network - optimized for gradient flow
        encoded = hk.Sequential([
            hk.Linear(hidden_dim, name="encoder_1"),
            jax.nn.relu,
            hk.Linear(hidden_dim // 2, name="encoder_2"), 
            jax.nn.relu,
            hk.Linear(hidden_dim // 4, name="encoder_3")
        ])(flat_input)  # [T*n_vars, hidden_dim//4]
        
        # Reshape back to temporal structure
        encoded = encoded.reshape(T, n_vars, hidden_dim // 4)  # [T, n_vars, hidden_dim//4]
        
        # POOLING: Simple sum pooling (like reference - more info than mean)
        # Sum preserves more information than mean pooling
        pooled = jnp.sum(encoded, axis=0)  # [n_vars, hidden_dim//4]
        
        # EMITTER: Direct output heads (like reference)
        # Variable selection head
        var_logits = hk.Sequential([
            hk.Linear(hidden_dim // 8, name="var_head_1"),
            jax.nn.relu,
            hk.Linear(1, name="var_head_output")
        ])(pooled).squeeze(-1)  # [n_vars]
        
        # Mask target variable (cannot intervene on target)
        var_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,  # Large negative value
            var_logits
        )
        
        # Value prediction head (mean and log_std)
        value_params = hk.Sequential([
            hk.Linear(hidden_dim // 8, name="val_head_1"),
            jax.nn.relu, 
            hk.Linear(2, name="val_head_output")  # [mean, log_std]
        ])(pooled)  # [n_vars, 2]
        
        return {
            'variable_logits': var_logits,
            'value_params': value_params
        }
    
    return clean_policy_fn


def create_ultra_simple_policy(hidden_dim: int = 256) -> Callable:
    """
    Ultra-simple policy for maximum gradient flow testing.
    
    Even simpler than clean_policy - single MLP layer.
    Use this if clean_policy still has gradient issues.
    """
    
    def ultra_simple_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        T, n_vars, n_channels = tensor_input.shape
        
        # Ultra-simple: flatten everything
        flat_input = tensor_input.flatten()  # [T * n_vars * 4]
        
        # Single hidden layer
        hidden = hk.Linear(hidden_dim, name="hidden")(flat_input)
        hidden = jax.nn.relu(hidden)
        
        # Direct outputs
        var_logits = hk.Linear(n_vars, name="var_output")(hidden)
        value_flat = hk.Linear(n_vars * 2, name="val_output")(hidden)
        value_params = value_flat.reshape(n_vars, 2)
        
        # Mask target
        var_logits = jnp.where(jnp.arange(n_vars) == target_idx, -jnp.inf, var_logits)
        
        return {
            'variable_logits': var_logits,
            'value_params': value_params
        }
    
    return ultra_simple_fn


def test_policy_gradient_flow(policy_apply_fn, params, tensor_input: jnp.ndarray, target_idx: int):
    """Test policy for gradient flow quality."""
    
    # Test gradient magnitude with existing parameters
    def loss_fn(p):
        dummy_key = jax.random.PRNGKey(42)
        output = policy_apply_fn(p, dummy_key, tensor_input, target_idx)
        # Simple loss: negative log probability of first variable
        var_probs = jax.nn.softmax(output['variable_logits'])
        return -jnp.log(var_probs[0] + 1e-8)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Compute gradient statistics
    grad_norms = jax.tree.map(jnp.linalg.norm, grads)
    total_grad_norm = sum(jax.tree_leaves(grad_norms))
    
    print(f"🔍 POLICY GRADIENT FLOW TEST:")
    print(f"  Loss value: {loss:.6f}")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm > 0.01:
        print(f"  ✅ Strong gradients - good for learning")
        return True
    elif total_grad_norm > 0.001:
        print(f"  ⚠️ Moderate gradients - may need higher LR")
        return True
    else:
        print(f"  ❌ Weak gradients - architecture issue")
        return False


def create_surrogate_integration_hooks():
    """
    Placeholder functions for future surrogate integration.
    
    These will be implemented when testing with trained surrogates.
    """
    
    def compute_entropy_from_posterior(posterior: Dict[str, Any], target_var: str) -> float:
        """Compute Shannon entropy of posterior distribution."""
        # Future implementation for information gain rewards
        return 0.0
    
    def apply_surrogate_to_buffer(buffer: SimpleBuffer, surrogate_fn: Callable, target_var: str):
        """Apply surrogate to buffer state to get posterior."""
        # Future implementation
        return {'marginal_parent_probs': {}}
    
    def update_tensor_with_surrogate(tensor: jnp.ndarray, posterior: Dict, variables: List[str]) -> jnp.ndarray:
        """Update channel 3 with surrogate predictions instead of uniform 0.5."""
        # Future implementation - will make channel 3 dynamic
        return tensor
    
    return {
        'compute_entropy': compute_entropy_from_posterior,
        'apply_surrogate': apply_surrogate_to_buffer,
        'update_tensor': update_tensor_with_surrogate
    }