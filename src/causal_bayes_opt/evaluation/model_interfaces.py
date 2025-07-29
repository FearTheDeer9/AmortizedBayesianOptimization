"""
Simple model interfaces for universal ACBO evaluation.

This module provides functions that convert trained models (GRPO, BC, etc.)
into simple acquisition functions that work with the universal evaluator.

Key principle: Models are just functions that map:
- (tensor, posterior, target) → intervention

No complex state management, no evaluation logic, just pure functions.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

from ..interventions.handlers import create_perfect_intervention
from ..policies.clean_policy_factory import create_clean_grpo_policy, verify_parameter_compatibility
from ..policies.clean_bc_policy_factory import create_clean_bc_policy, verify_bc_parameter_compatibility

logger = logging.getLogger(__name__)


def create_grpo_acquisition(checkpoint_path: Path, 
                           seed: int = 42) -> Callable:
    """
    Create acquisition function from trained GRPO checkpoint.
    
    Args:
        checkpoint_path: Path to GRPO checkpoint
        seed: Random seed for stochastic policy
        
    Returns:
        Function that maps (tensor, posterior, target) → intervention
    """
    # Load checkpoint
    with open(checkpoint_path / 'checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    
    policy_params = checkpoint['policy_params']
    config = checkpoint.get('config', {})
    
    # Extract architecture config
    arch_config = config.get('architecture', {})
    hidden_dim = arch_config.get('hidden_dim', 256)
    
    # Use shared policy factory - SAME as training!
    # This ensures Haiku creates identical module paths
    policy_fn = create_clean_grpo_policy(hidden_dim=hidden_dim)
    transformed_fn = hk.transform(policy_fn)
    
    # Verify parameter compatibility
    dummy_tensor = jnp.zeros((10, 5, 3))
    if not verify_parameter_compatibility(policy_params, transformed_fn, dummy_tensor):
        logger.warning("Parameter mismatch detected! Model may not load correctly.")
    rng_key = random.PRNGKey(seed)
    
    def grpo_acquisition(tensor: jnp.ndarray, 
                        posterior: Optional[Dict[str, Any]], 
                        target: str,
                        variables: List[str]) -> Dict[str, Any]:
        """
        GRPO acquisition function.
        
        Args:
            tensor: [T, n_vars, 3] tensor in 3-channel format
            posterior: Optional structure prediction (unused by GRPO)
            target: Target variable name
            variables: List of variable names in order
            
        Returns:
            Intervention dictionary with targets and values
        """
        nonlocal rng_key
        
        target_idx = variables.index(target)
        
        # Apply policy
        rng_key, policy_key = random.split(rng_key)
        outputs = transformed_fn.apply(policy_params, policy_key, tensor, target_idx)
        
        # Sample intervention
        var_logits = outputs['variable_logits']
        value_params = outputs['value_params']
        
        # For evaluation, use deterministic selection
        selected_idx = int(jnp.argmax(var_logits))
        selected_var = variables[selected_idx]
        
        # Use mean value
        intervention_value = float(value_params[selected_idx, 0])
        
        return {
            'targets': frozenset([selected_var]),
            'values': {selected_var: intervention_value}
        }
    
    return grpo_acquisition


def create_bc_acquisition(checkpoint_path: Path, 
                         seed: int = 42) -> Callable:
    """
    Create acquisition function from trained BC model.
    
    Args:
        checkpoint_path: Path to BC checkpoint
        seed: Random seed for stochastic policy
        
    Returns:
        Function that maps (tensor, posterior, target) → intervention
    """
    # Load checkpoint
    with open(checkpoint_path / 'checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
    
    policy_params = checkpoint.get('policy_params')
    config = checkpoint.get('config', {})
    
    # Extract hidden dim
    hidden_dim = config.get('hidden_dim', 256)
    
    # Use shared BC policy factory - SAME as training!
    policy_fn = create_clean_bc_policy(hidden_dim=hidden_dim)
    transformed_fn = hk.transform(policy_fn)
    
    # Verify parameter compatibility
    dummy_tensor = jnp.zeros((10, 5, 3))
    if not verify_bc_parameter_compatibility(policy_params, transformed_fn, dummy_tensor):
        logger.warning("BC parameter mismatch detected! Model may not load correctly.")
    
    rng_key = random.PRNGKey(seed)
    
    def bc_acquisition(tensor: jnp.ndarray,
                      posterior: Optional[Dict[str, Any]],
                      target: str,
                      variables: List[str]) -> Dict[str, Any]:
        """
        BC acquisition function.
        
        Args:
            tensor: [T, n_vars, 3] tensor in 3-channel format
            posterior: Optional structure prediction (can be used by BC)
            target: Target variable name
            variables: List of variable names in order
            
        Returns:
            Intervention dictionary with targets and values
        """
        nonlocal rng_key
        
        target_idx = variables.index(target)
        
        # Apply BC policy
        rng_key, policy_key = random.split(rng_key)
        outputs = transformed_fn.apply(policy_params, policy_key, tensor, target_idx)
        
        # Get predictions
        var_logits = outputs['variable_logits']
        value_params = outputs['value_params']
        
        # For evaluation, use deterministic selection (highest probability)
        selected_idx = int(jnp.argmax(var_logits))
        selected_var = variables[selected_idx]
        
        # Use mean value prediction
        intervention_value = float(value_params[selected_idx, 0])
        
        return {
            'targets': frozenset([selected_var]),
            'values': {selected_var: intervention_value}
        }
    
    return bc_acquisition


def create_random_acquisition(seed: int = 42) -> Callable:
    """
    Create random baseline acquisition function.
    
    Args:
        seed: Random seed
        
    Returns:
        Random acquisition function
    """
    rng_key = random.PRNGKey(seed)
    
    def random_acquisition(tensor: jnp.ndarray,
                          posterior: Optional[Dict[str, Any]],
                          target: str,
                          variables: List[str]) -> Dict[str, Any]:
        """Random intervention selection."""
        nonlocal rng_key
        
        # Remove target from candidates
        candidates = [v for v in variables if v != target]
        
        # Random selection
        rng_key, var_key, val_key = random.split(rng_key, 3)
        selected_idx = random.randint(var_key, (), 0, len(candidates))
        selected_var = candidates[selected_idx]
        
        # Random value in [-2, 2]
        intervention_value = float(random.uniform(val_key, (), minval=-2.0, maxval=2.0))
        
        return {
            'targets': frozenset([selected_var]),
            'values': {selected_var: intervention_value}
        }
    
    return random_acquisition


def create_oracle_acquisition(scm_edges: Dict[str, List[str]], 
                             seed: int = 42,
                             optimization_direction: str = 'MINIMIZE') -> Callable:
    """
    Create oracle acquisition that knows true causal structure and optimizes greedily.
    
    Args:
        scm_edges: True causal edges {child: [parents]}
        seed: Random seed
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        
    Returns:
        Oracle acquisition function that selects optimal interventions
    """
    rng_key = random.PRNGKey(seed)
    intervention_count = 0  # Track number of interventions for rotation
    
    def oracle_acquisition(tensor: jnp.ndarray,
                          posterior: Optional[Dict[str, Any]],
                          target: str,
                          variables: List[str]) -> Dict[str, Any]:
        """Oracle intervention selection using true structure."""
        nonlocal rng_key, intervention_count
        
        # Get true parents of target
        true_parents = scm_edges.get(target, [])
        
        if true_parents:
            # Rotate through parents to ensure diversity
            # This helps when multiple parents have similar effects
            selected_var = true_parents[intervention_count % len(true_parents)]
            intervention_count += 1
            
            # Choose intervention value based on optimization direction
            # Use stronger values for better optimization
            if optimization_direction == 'MINIMIZE':
                # Use strong negative values to minimize
                # Vary between -3.0 and -1.5 for diversity
                base_value = -2.5
                rng_key, range_key = random.split(rng_key)
                range_offset = float(random.uniform(range_key, (), minval=-0.5, maxval=1.0))
                intervention_value = base_value + range_offset
            else:
                # Use strong positive values to maximize
                base_value = 2.5
                rng_key, range_key = random.split(rng_key)
                range_offset = float(random.uniform(range_key, (), minval=-1.0, maxval=0.5))
                intervention_value = base_value + range_offset
            
        else:
            # No parents - target is a root node
            # Intervene on a random non-target variable
            candidates = [v for v in variables if v != target]
            
            if candidates:
                rng_key, var_key = random.split(rng_key)
                selected_idx = random.randint(var_key, (), 0, len(candidates))
                selected_var = candidates[selected_idx]
                intervention_value = 0.0
            else:
                # Edge case: only one variable
                selected_var = target
                intervention_value = 0.0
        
        return {
            'targets': frozenset([selected_var]),
            'values': {selected_var: intervention_value}
        }
    
    return oracle_acquisition


def create_uniform_exploration_acquisition(value_range: Tuple[float, float] = (-2.0, 2.0),
                                         seed: int = 42) -> Callable:
    """
    Create acquisition that uniformly explores all variables.
    
    Args:
        value_range: Range of intervention values
        seed: Random seed
        
    Returns:
        Uniform exploration acquisition function
    """
    rng_key = random.PRNGKey(seed)
    step_counter = 0
    
    def uniform_acquisition(tensor: jnp.ndarray,
                           posterior: Optional[Dict[str, Any]],
                           target: str,
                           variables: List[str]) -> Dict[str, Any]:
        """Uniform exploration over variables."""
        nonlocal rng_key, step_counter
        
        # Remove target from candidates
        candidates = [v for v in variables if v != target]
        
        # Cycle through variables uniformly
        selected_var = candidates[step_counter % len(candidates)]
        step_counter += 1
        
        # Random value in range
        rng_key, val_key = random.split(rng_key)
        intervention_value = float(random.uniform(
            val_key, (), minval=value_range[0], maxval=value_range[1]
        ))
        
        return {
            'targets': frozenset([selected_var]),
            'values': {selected_var: intervention_value}
        }
    
    return uniform_acquisition