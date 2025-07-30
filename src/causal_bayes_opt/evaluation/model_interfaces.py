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
    # Load checkpoint - handle both file path and directory path
    if checkpoint_path.is_file():
        checkpoint_file = checkpoint_path
    else:
        # Look for checkpoint.pkl in directory
        checkpoint_file = checkpoint_path / 'checkpoint.pkl'
        if not checkpoint_file.exists():
            # Try direct pickle file
            checkpoint_file = checkpoint_path
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Handle new checkpoint format from general_bc_trainer
    if 'params' in checkpoint:
        # New format
        policy_params = checkpoint['params']
        config = checkpoint.get('config', {})
    else:
        # Old format compatibility
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


def create_bc_surrogate(checkpoint_path: Path, 
                       allow_updates: bool = False,
                       learning_rate: float = 1e-4) -> Tuple[Callable, Optional[Callable]]:
    """
    Create BC-trained surrogate function for structure learning.
    
    Args:
        checkpoint_path: Path to BC surrogate checkpoint
        allow_updates: Whether to allow online updates during evaluation
        learning_rate: Learning rate for online updates (if enabled)
        
    Returns:
        Tuple of (predict_fn, update_fn) where update_fn is None if updates disabled
    """
    import optax
    from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
    from ..avici_integration.core import samples_to_avici_format
    from ..avici_integration.parent_set import predict_parent_posterior
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract parameters
    params = checkpoint.get('params', checkpoint.get('surrogate_params'))
    config = checkpoint.get('config', {})
    metadata = checkpoint.get('metadata', {})
    
    # Extract model config
    hidden_dim = config.get('surrogate_hidden_dim', 128)
    num_layers = config.get('surrogate_layers', 4)
    num_heads = config.get('surrogate_heads', 8)
    
    # Create the model function
    def model_fn(data: jnp.ndarray, target_variable: int, is_training: bool = False):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=32,
            dropout=0.1
        )
        return model(data, target_variable, is_training)
    
    # Transform it
    net = hk.transform(model_fn)
    
    def bc_surrogate_predict(tensor: jnp.ndarray, target: str) -> Dict[str, Any]:
        """
        Predict parent posterior for target variable.
        
        Args:
            tensor: [T, n_vars, 3] tensor in 3-channel format
            target: Target variable name
            
        Returns:
            Posterior dictionary with parent probabilities
        """
        # Convert to AVICI format
        avici_data = tensor  # Already in correct format
        
        # Get variable order from tensor
        n_vars = tensor.shape[1]
        variables = [f"var_{i}" for i in range(n_vars)]
        if target not in variables:
            # Map target to appropriate index
            target_idx = n_vars - 1  # Assume target is last
            variables[target_idx] = target
        
        # Need to find target index
        target_idx = variables.index(target) if target in variables else len(variables) - 1
        
        # Apply the model
        output = net.apply(params, None, avici_data, target_idx, False)
        
        # The output should be parent predictions - need to convert to posterior format
        # For now, create a simple posterior
        from ..avici_integration.parent_set.posterior import create_parent_set_posterior
        
        # Extract parent probabilities from model output
        # ContinuousParentSetPredictionModel returns a dict with 'parent_probabilities'
        if isinstance(output, dict) and 'parent_probabilities' in output:
            parent_probs = output['parent_probabilities']
        else:
            # Fallback
            parent_probs = output
        
        # Convert to marginal probabilities
        marginal_probs = {}
        for i, var in enumerate(variables):
            if var != target:
                marginal_probs[var] = float(parent_probs[i])
        
        # Create posterior with single "set" representing marginal probabilities
        posterior = create_parent_set_posterior(
            target_variable=target,
            parent_sets=[frozenset()],  # Empty set as placeholder
            probabilities=jnp.array([1.0]),
            metadata={'marginal_parent_probs': marginal_probs}
        )
        
        return posterior
    
    if not allow_updates:
        return bc_surrogate_predict, None
    
    # Create update function for online learning
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    def bc_surrogate_update(params, opt_state, posterior, samples, variables, target):
        """Update surrogate parameters based on new data."""
        # This would implement the update logic
        # For now, return unchanged
        return params, opt_state, (0.0, 0.0, 0.0, 0.0)
    
    return bc_surrogate_predict, bc_surrogate_update


def create_learning_surrogate(scm: Any,
                             learning_rate: float = 1e-3,
                             scoring_method: str = "bic",
                             seed: int = 42) -> Tuple[Callable, Callable]:
    """
    Create a fresh learnable surrogate that learns from scratch.
    
    This is equivalent to the "learning baseline" - no pre-training.
    
    Args:
        scm: The structural causal model (to get variable names)
        learning_rate: Learning rate for online updates
        scoring_method: Scoring method for structure learning
        seed: Random seed
        
    Returns:
        Tuple of (predict_fn, update_fn) for online learning
    """
    from ..training.active_learning import create_active_learning_surrogate
    
    # Use production active learning implementation
    return create_active_learning_surrogate(
        scm=scm,
        initial_checkpoint=None,  # Start from scratch
        learning_rate=learning_rate,
        scoring_method=scoring_method,
        seed=seed
    )


def create_hybrid_surrogate(scm: Any,
                           checkpoint_path: Path,
                           learning_rate: float = 1e-4,
                           seed: int = 42) -> Tuple[Callable, Callable]:
    """
    Create hybrid surrogate that starts with BC weights but continues learning.
    
    This combines pre-training benefits with online adaptation.
    
    Args:
        scm: The structural causal model (to get variable names)
        checkpoint_path: Path to BC surrogate checkpoint
        learning_rate: Learning rate for online updates
        seed: Random seed
        
    Returns:
        Tuple of (predict_fn, update_fn) for hybrid approach
    """
    from ..training.active_learning import create_active_learning_surrogate
    
    # Use production active learning with initial checkpoint
    return create_active_learning_surrogate(
        scm=scm,
        initial_checkpoint=checkpoint_path,
        learning_rate=learning_rate,
        scoring_method="bic",  # Default to BIC for hybrid
        seed=seed
    )


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