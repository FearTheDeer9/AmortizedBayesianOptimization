"""
Shared policy definitions for clean ACBO implementation.

This module ensures that the same function definitions are used
for both training and inference, preventing Haiku module path mismatches.

Key insight: Haiku creates different module hierarchies based on WHERE
functions are defined, not just HOW they're structured. This factory
ensures consistent module paths across training and inference.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Dict, Any, Callable


def create_clean_grpo_policy(hidden_dim: int = 256) -> Callable:
    """
    Create GRPO policy function with consistent module paths.
    
    This factory ensures the same function is used for both
    training and inference, preventing Haiku parameter mismatches.
    
    Args:
        hidden_dim: Hidden dimension for the network
        
    Returns:
        Policy function that maps tensor inputs to action distributions
    """
    def policy_fn(tensor_input: jnp.ndarray, target_idx: int = 0) -> Dict[str, jnp.ndarray]:
        """
        GRPO policy network processing 5-channel tensor input.
        
        Args:
            tensor_input: [T, n_vars, C] tensor where C can be 3 or 5
            target_idx: Index of target variable to mask
            
        Returns:
            Dictionary with:
            - variable_logits: [n_vars] logits for variable selection
            - value_params: [n_vars, 2] mean and log_std for each variable
        """
        T, n_vars, n_channels = tensor_input.shape
        
        # Handle both 3 and 5 channel inputs
        if n_channels == 3:
            # Legacy 3-channel format - pad with zeros
            padded = jnp.zeros((T, n_vars, 5))
            padded = padded.at[:, :, :3].set(tensor_input)
            tensor_input = padded
            n_channels = 5
        elif n_channels != 5:
            raise ValueError(f"Expected 3 or 5 channels, got {n_channels}")
        
        # Project to hidden dimension (now from 5 channels)
        flat_input = tensor_input.reshape(T * n_vars, 5)
        x = hk.Linear(hidden_dim, name="input_projection")(flat_input)
        x = jax.nn.relu(x)
        
        # Reshape for temporal processing
        x = x.reshape(T, n_vars, hidden_dim)
        
        # Process each timestep independently
        def process_timestep(timestep_data):
            """Process single timestep with residual connection."""
            # Layer normalization
            x_norm = hk.LayerNorm(
                axis=-1, 
                create_scale=True, 
                create_offset=True, 
                name="timestep_norm"
            )(timestep_data)
            
            # MLP with expansion
            x_hidden = hk.Linear(hidden_dim * 2, name="timestep_hidden")(x_norm)
            x_hidden = jax.nn.relu(x_hidden)
            x_out = hk.Linear(hidden_dim, name="timestep_output")(x_hidden)
            
            # Residual connection
            return x_out + timestep_data
        
        # Process all timesteps
        x = jax.vmap(process_timestep)(x)
        
        # Aggregate over time (simple mean pooling)
        x_agg = jnp.mean(x, axis=0)  # [n_vars, hidden_dim]
        
        # Final layer norm before output heads
        x_agg = hk.LayerNorm(
            axis=-1,
            create_scale=True,
            create_offset=True,
            name="output_norm"
        )(x_agg)
        
        # Variable selection head
        variable_head = hk.Linear(1, name="variable_head")(x_agg)
        variable_logits = variable_head.squeeze(-1)  # [n_vars]
        
        # Mask out target variable (cannot intervene on target)
        variable_logits = jnp.where(
            jnp.arange(n_vars) == target_idx,
            -jnp.inf,
            variable_logits
        )
        
        # Value prediction head (mean and log_std for each variable)
        value_head = hk.Linear(2, name="value_head")(x_agg)  # [n_vars, 2]
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_head
        }
    
    return policy_fn


def verify_parameter_compatibility(
    saved_params: Dict[str, Any],
    model_fn: hk.Transformed,
    dummy_input: jnp.ndarray,
    target_idx: int = 0
) -> bool:
    """
    Verify that saved parameters match expected model structure.
    
    This is crucial for catching Haiku module path mismatches early.
    
    Args:
        saved_params: Parameters loaded from checkpoint
        model_fn: Haiku transformed function
        dummy_input: Example input for initialization
        target_idx: Target variable index
        
    Returns:
        True if parameters are compatible, False otherwise
    """
    import jax.tree_util as tree
    
    try:
        # Get expected parameter structure
        rng = jax.random.PRNGKey(0)
        expected_params = model_fn.init(rng, dummy_input, target_idx)
        
        # Get parameter keys
        saved_flat, saved_tree = tree.tree_flatten(saved_params)
        expected_flat, expected_tree = tree.tree_flatten(expected_params)
        
        # Extract keys (parameter paths)
        def get_keys(params):
            """Extract all parameter paths."""
            keys = []
            
            def traverse(path, node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        traverse(path + "/" + k if path else k, v)
                else:
                    keys.append(path)
            
            traverse("", params)
            return set(keys)
        
        saved_keys = get_keys(saved_params)
        expected_keys = get_keys(expected_params)
        
        # Check compatibility
        if saved_keys != expected_keys:
            missing = expected_keys - saved_keys
            extra = saved_keys - expected_keys
            
            print("Parameter mismatch detected!")
            if missing:
                print(f"Missing parameters: {missing}")
            if extra:
                print(f"Extra parameters: {extra}")
            
            return False
        
        # Check shapes match
        for key in saved_keys:
            saved_shape = tree.tree_map(lambda x: x.shape, saved_params)
            expected_shape = tree.tree_map(lambda x: x.shape, expected_params)
            
            if saved_shape != expected_shape:
                print(f"Shape mismatch for {key}")
                return False
        
        print("Parameters are compatible!")
        return True
        
    except Exception as e:
        print(f"Error verifying parameters: {e}")
        return False


def create_parameter_migration_util() -> Dict[str, Callable]:
    """
    Create utilities for migrating parameters between different module structures.
    
    Returns:
        Dictionary of migration utilities
    """
    def extract_leaf_params(params: Dict) -> Dict[str, jnp.ndarray]:
        """Extract all leaf parameters with their paths."""
        leaves = {}
        
        def traverse(path, node):
            if isinstance(node, dict):
                for k, v in node.items():
                    new_path = f"{path}/{k}" if path else k
                    traverse(new_path, v)
            else:
                leaves[path] = node
        
        traverse("", params)
        return leaves
    
    def remap_parameters(
        old_params: Dict,
        old_to_new_mapping: Dict[str, str]
    ) -> Dict:
        """Remap parameters from old structure to new structure."""
        # Extract leaves
        old_leaves = extract_leaf_params(old_params)
        
        # Build new structure
        new_params = {}
        for old_path, new_path in old_to_new_mapping.items():
            if old_path in old_leaves:
                # Navigate and create nested structure
                parts = new_path.split('/')
                current = new_params
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                current[parts[-1]] = old_leaves[old_path]
        
        return new_params
    
    return {
        'extract_leaves': extract_leaf_params,
        'remap': remap_parameters
    }