"""
Simple model interfaces for universal ACBO evaluation.

This module provides functions that convert trained models (GRPO, BC, etc.)
into simple acquisition functions that work with the universal evaluator.

Key principle: Models are just functions that map:
- (tensor, posterior, target) → intervention

No complex state management, no evaluation logic, just pure functions.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

from ..interventions.handlers import create_perfect_intervention
from ..policies.clean_policy_factory import create_clean_grpo_policy, verify_parameter_compatibility
from ..policies.clean_bc_policy_factory import create_clean_bc_policy, verify_bc_parameter_compatibility
from ..training.five_channel_converter import convert_three_to_five_channel
from ..utils.posterior_validator import PosteriorValidator

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
    from ..utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
    
    # Load checkpoint using unified utilities
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Verify it's a policy model
    if checkpoint['model_type'] != 'policy':
        raise ValueError(f"Expected policy model, got {checkpoint['model_type']}")
    
    # Create model from checkpoint
    transformed_fn, policy_params = create_model_from_checkpoint(checkpoint)
    
    # Extract architecture
    hidden_dim = checkpoint['architecture']['hidden_dim']
    
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
            posterior: Optional structure prediction (now integrated)
            target: Target variable name
            variables: List of variable names in order
            
        Returns:
            Intervention dictionary with targets and values
        """
        nonlocal rng_key
        
        target_idx = variables.index(target)
        
        # Convert to 5-channel format if we have posterior
        if posterior is not None:
            # Create surrogate function that returns the posterior
            surrogate_fn = lambda t, tgt: posterior
            tensor_5ch, diagnostics = convert_three_to_five_channel(
                tensor, variables, target, surrogate_fn
            )
            
            # Log diagnostics
            if diagnostics.get('had_surrogate'):
                logger.debug(f"GRPO using surrogate predictions")
                PosteriorValidator.log_posterior_summary(
                    posterior, variables, target, prefix="GRPO"
                )
            
            # Use 5-channel tensor
            input_tensor = tensor_5ch
        else:
            # No posterior - use 3-channel tensor
            # Pad with zeros to make it 5-channel for compatibility
            T, n_vars, _ = tensor.shape
            tensor_5ch = jnp.zeros((T, n_vars, 5))
            tensor_5ch = tensor_5ch.at[:, :, :3].set(tensor)
            input_tensor = tensor_5ch
        
        # Apply policy
        rng_key, policy_key = random.split(rng_key)
        outputs = transformed_fn.apply(policy_params, policy_key, input_tensor, target_idx)
        
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
    from ..utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
    
    # Load checkpoint using unified utilities
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Verify it's a policy model
    if checkpoint['model_type'] != 'policy':
        raise ValueError(f"Expected policy model, got {checkpoint['model_type']}")
    
    # Create model from checkpoint
    transformed_fn, policy_params = create_model_from_checkpoint(checkpoint)
    
    # Extract architecture
    hidden_dim = checkpoint['architecture']['hidden_dim']
    
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
            posterior: Optional structure prediction (now integrated)
            target: Target variable name
            variables: List of variable names in order
            
        Returns:
            Intervention dictionary with targets and values
        """
        nonlocal rng_key
        
        target_idx = variables.index(target)
        
        # Convert to 5-channel format if we have posterior
        if posterior is not None:
            # Create surrogate function that returns the posterior
            surrogate_fn = lambda t, tgt: posterior
            tensor_5ch, diagnostics = convert_three_to_five_channel(
                tensor, variables, target, surrogate_fn
            )
            
            # Log diagnostics
            if diagnostics.get('had_surrogate'):
                logger.debug(f"BC using surrogate predictions")
                PosteriorValidator.log_posterior_summary(
                    posterior, variables, target, prefix="BC"
                )
            
            # Use 5-channel tensor
            input_tensor = tensor_5ch
        else:
            # No posterior - use 3-channel tensor padded to 5
            T, n_vars, _ = tensor.shape
            tensor_5ch = jnp.zeros((T, n_vars, 5))
            tensor_5ch = tensor_5ch.at[:, :, :3].set(tensor)
            input_tensor = tensor_5ch
        
        # Apply BC policy
        rng_key, policy_key = random.split(rng_key)
        outputs = transformed_fn.apply(policy_params, policy_key, input_tensor, target_idx)
        
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
                       learning_rate: float = 1e-4,
                       return_components: bool = False) -> Union[Tuple[Callable, Optional[Callable]], Tuple[Callable, Optional[Callable], Any, Any, Any]]:
    """
    Create BC-trained surrogate function for structure learning.
    
    Args:
        checkpoint_path: Path to BC surrogate checkpoint
        allow_updates: Whether to allow online updates during evaluation
        learning_rate: Learning rate for online updates (if enabled)
        return_components: If True, also return (net, params, opt_state)
        
    Returns:
        Tuple of (predict_fn, update_fn) where update_fn is None if updates disabled
        If return_components=True, returns (predict_fn, update_fn, net, params, opt_state)
    """
    import optax
    from ..utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint
    from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
    from ..avici_integration.core import samples_to_avici_format
    from ..avici_integration.parent_set import predict_parent_posterior
    
    # Load checkpoint using unified utilities
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Verify it's a surrogate model
    if checkpoint['model_type'] != 'surrogate':
        raise ValueError(f"Expected surrogate model, got {checkpoint['model_type']}")
    
    # Create model from checkpoint
    net, params = create_model_from_checkpoint(checkpoint)
    # Extract metadata
    metadata = checkpoint.get('metadata', {})
    
    # Internal predict function with full signature
    def _bc_surrogate_predict_full(tensor: jnp.ndarray, target: str, variables: List[str] = None) -> Dict[str, Any]:
        """
        Predict parent posterior for target variable.
        
        Args:
            tensor: [T, n_vars, 3] tensor in 3-channel format
            target: Target variable name
            variables: Optional list of variable names in tensor order
            
        Returns:
            Posterior dictionary with parent probabilities
        """
        # Convert to AVICI format
        avici_data = tensor  # Already in correct format
        
        # Get variable order from tensor
        n_vars = tensor.shape[1]
        
        # Use provided variables or create default names
        if variables is None:
            variables = [f"X{i}" for i in range(n_vars)]
            if target not in variables:
                # Assume target is last
                variables[-1] = target
        
        # Need to find target index
        target_idx = variables.index(target) if target in variables else len(variables) - 1
        
        # Apply the model
        output = net.apply(params, None, avici_data, target_idx, False)
        
        # The output should be parent predictions - need to convert to posterior format
        from ..avici_integration.parent_set.posterior import create_parent_set_posterior
        
        # Extract parent probabilities from model output
        # ContinuousParentSetPredictionModel returns a dict with 'parent_probabilities'
        if isinstance(output, dict) and 'parent_probabilities' in output:
            parent_probs = output['parent_probabilities']
        else:
            # Fallback
            parent_probs = output
        
        # Ensure we have the right variable mapping
        # The model uses indices, but we need variable names
        if target not in variables:
            # Create proper variable mapping
            variables = [f"X{i}" for i in range(n_vars)]
            if target not in variables:
                variables[-1] = target  # Assume target is last
        
        # Convert to marginal probabilities
        marginal_probs = {}
        logger.info(f"\nBC Surrogate - Converting parent probabilities:")
        logger.info(f"  Target: {target} (index {target_idx})")
        logger.info(f"  Parent probs shape: {parent_probs.shape if hasattr(parent_probs, 'shape') else 'unknown'}")
        logger.info(f"  Variables: {variables}")
        
        # The model outputs probabilities for all variables in order
        # We need to map them correctly, skipping the target
        for i, var in enumerate(variables):
            if i < len(parent_probs):
                prob = float(parent_probs[i])
                if var == target:
                    # Skip target - it can't be its own parent
                    logger.info(f"  {var}: (target, skipped)")
                else:
                    marginal_probs[var] = prob
                    logger.info(f"  {var}: {prob:.3f}")
            else:
                if var != target:
                    marginal_probs[var] = 0.0
                    logger.info(f"  {var}: 0.0 (no prediction)")
        
        # Log summary
        non_zero_probs = [p for v, p in marginal_probs.items() if v != target and p > 0]
        logger.info(f"  Non-zero probabilities: {len(non_zero_probs)} out of {len(variables)-1}")
        if non_zero_probs:
            logger.info(f"  Range: [{min(non_zero_probs):.3f}, {max(non_zero_probs):.3f}]")
            logger.info(f"  Mean: {sum(non_zero_probs)/len(non_zero_probs):.3f}")
            # Check if all values are exactly 0.5 (dummy surrogate indicator)
            if all(abs(p - 0.5) < 1e-6 for p in non_zero_probs):
                logger.warning("  WARNING: All probabilities are exactly 0.5 - this suggests the model is not properly trained or loaded!")
        
        # Create posterior with marginal probabilities in metadata
        # This format is expected by the 5-channel converter
        posterior = create_parent_set_posterior(
            target_variable=target,
            parent_sets=[frozenset()],  # Empty set as placeholder
            probabilities=jnp.array([1.0]),
            metadata={'marginal_parent_probs': marginal_probs}
        )
        
        return posterior
    
    # Store the internal function for testing
    bc_surrogate_predict_full = _bc_surrogate_predict_full
    
    # Create wrapper that works with standard (tensor, target, variables) signature
    def bc_surrogate_predict(tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
        """
        Standard interface wrapper matching SurrogateFn signature.
        
        Args:
            tensor: [T, n_vars, 3] tensor
            target: Target variable name
            variables: List of variable names in tensor order
            
        Returns:
            Posterior with parent predictions
        """
        return _bc_surrogate_predict_full(tensor, target, variables=variables)
    
    # Add the full version as an attribute for callers that can provide variables
    bc_surrogate_predict.predict_with_variables = bc_surrogate_predict_full
    
    if not allow_updates:
        if return_components:
            return bc_surrogate_predict, None, net, params, None
        return bc_surrogate_predict, None
    
    # Create update function for online learning
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    def bc_surrogate_update(params, opt_state, posterior, samples, variables, target):
        """Update surrogate parameters based on new data using gradient descent.
        
        This implements the same approach as active learning: nudge parameters
        to maximize data likelihood.
        
        Args:
            params: Current model parameters
            opt_state: Optimizer state
            posterior: Current posterior predictions (not used)
            samples: List of samples (observational and interventional)
            variables: List of variable names
            target: Target variable name
            
        Returns:
            Tuple of (updated_params, updated_opt_state, metrics)
        """
        if len(samples) < 5:
            logger.debug("Skipping BC update: insufficient samples")
            return params, opt_state, {"loss": 0.0, "grad_norm": 0.0, "update_norm": 0.0, "skipped": True}
        
        # Import required functions
        from ..data_structures.sample import get_values
        from ..avici_integration.core import samples_to_avici_format
        
        # Convert samples to tensor format
        data = samples_to_avici_format(samples, variables, target)
        
        # Get target index
        target_idx = variables.index(target) if target in variables else len(variables) - 1
        
        def compute_parent_set_scores(parent_sets, samples, target_variable):
            """Compute data likelihood scores for parent sets."""
            # Extract target values
            target_values = jnp.array([get_values(s)[target_variable] for s in samples])
            n_samples = len(target_values)
            
            # Create value matrix for all variables
            all_values = jnp.zeros((n_samples, len(variables)))
            for i, sample in enumerate(samples):
                sample_values = get_values(sample)
                for j, var_name in enumerate(variables):
                    if var_name in sample_values:
                        all_values = all_values.at[i, j].set(sample_values[var_name])
            
            scores = []
            
            for parent_set in parent_sets:
                if len(parent_set) == 0:
                    # No parents: simple mean/variance model
                    mean_pred = jnp.mean(target_values)
                    var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                    log_likelihood = -0.5 * jnp.sum(
                        jnp.log(2 * jnp.pi * var_pred) + 
                        (target_values - mean_pred)**2 / var_pred
                    )
                else:
                    # Linear regression with parents
                    parent_indices = [variables.index(p) for p in parent_set 
                                    if p in variables]
                    
                    if parent_indices and n_samples > len(parent_indices):
                        # Extract parent values
                        parent_values = all_values[:, jnp.array(parent_indices)]
                        
                        # Add intercept
                        X = jnp.column_stack([jnp.ones(n_samples), parent_values])
                        
                        # Solve least squares with regularization
                        XTX = X.T @ X + 1e-6 * jnp.eye(X.shape[1])
                        XTy = X.T @ target_values
                        beta = jnp.linalg.solve(XTX, XTy)
                        
                        # Compute residuals
                        predictions = X @ beta
                        residuals = target_values - predictions
                        residual_var = jnp.maximum(jnp.var(residuals), 0.01)
                        
                        # Log likelihood
                        log_likelihood = -0.5 * jnp.sum(
                            jnp.log(2 * jnp.pi * residual_var) + 
                            residuals**2 / residual_var
                        )
                    else:
                        # Fallback
                        mean_pred = jnp.mean(target_values)
                        var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                        log_likelihood = -0.5 * jnp.sum(
                            jnp.log(2 * jnp.pi * var_pred) + 
                            (target_values - mean_pred)**2 / var_pred
                        )
                
                # Apply BIC penalty
                n_params = len(parent_set) + 2  # coefficients + intercept + variance
                bic_penalty = 0.5 * n_params * jnp.log(n_samples)
                scores.append(log_likelihood - bic_penalty)
            
            return jnp.array(scores)
        
        # Define loss function
        def loss_fn(params):
            # Get model predictions
            output = net.apply(params, None, data, target_idx, False)
            
            # Extract parent set predictions
            if isinstance(output, dict):
                logits = output.get('parent_set_logits', jnp.zeros(1))
                parent_sets = output.get('parent_sets', [[]])
            else:
                # Fallback
                logits = jnp.zeros(1)
                parent_sets = [[]]
            
            # Compute scores for each parent set
            scores = compute_parent_set_scores(parent_sets, samples, target)
            
            # Temperature-scaled softmax
            temperature = 2.0
            probs = jax.nn.softmax(logits / temperature)
            
            # Expected score under model distribution
            expected_score = jnp.sum(probs * scores)
            
            # Loss is negative expected score
            loss = -expected_score
            
            # Add L2 regularization
            l2_reg = 1e-4 * sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
            
            return loss + l2_reg
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        # Clip gradients
        grad_norm = optax.global_norm(grads)
        max_grad_norm = 1.0
        
        if grad_norm > max_grad_norm:
            grads = jax.tree.map(
                lambda g: g * max_grad_norm / (grad_norm + 1e-8),
                grads
            )
            clipped_grad_norm = max_grad_norm
        else:
            clipped_grad_norm = grad_norm
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Track update statistics
        update_norm = optax.global_norm(updates)
        
        metrics = {
            "loss": float(loss_val),
            "grad_norm": float(clipped_grad_norm),
            "update_norm": float(update_norm),
            "n_samples": len(samples)
        }
        
        logger.debug(f"BC surrogate update: loss={loss_val:.4f}, "
                    f"grad_norm={clipped_grad_norm:.4f}, "
                    f"update_norm={update_norm:.4f}")
        
        return new_params, new_opt_state, metrics
    
    if return_components:
        return bc_surrogate_predict, bc_surrogate_update, net, params, opt_state
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
    # NOTE: This function is deprecated - use ActiveLearningSurrogateWrapper instead
    raise NotImplementedError("create_learning_surrogate is deprecated. Use ActiveLearningSurrogateWrapper from surrogate_interface.py")
    
    # from ..training.active_learning import create_active_learning_surrogate
    # al_predict_fn, al_update_fn = create_active_learning_surrogate(
    #     scm=scm,
    #     initial_checkpoint=None,  # Start from scratch
    #     learning_rate=learning_rate,
    #     scoring_method=scoring_method,
    #     seed=seed
    # )
    # 
    # # Wrap to match the new SurrogateFn signature
    # def learning_surrogate_predict(tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
    #     # Active learning already knows the SCM variables, but we pass them for consistency
    #     return al_predict_fn(tensor, target)
    # 
    # return learning_surrogate_predict, al_update_fn


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
    # NOTE: This function is deprecated - use ActiveLearningSurrogateWrapper instead
    raise NotImplementedError("create_hybrid_surrogate is deprecated. Use ActiveLearningSurrogateWrapper from surrogate_interface.py")
    
    # from ..training.active_learning import create_active_learning_surrogate
    # al_predict_fn, al_update_fn = create_active_learning_surrogate(
    #     scm=scm,
    #     initial_checkpoint=checkpoint_path,
    #     learning_rate=learning_rate,
    #     scoring_method="bic",  # Default to BIC for hybrid
    #     seed=seed
    # )
    # 
    # # Wrap to match the new SurrogateFn signature
    # def hybrid_surrogate_predict(tensor: jnp.ndarray, target: str, variables: List[str]) -> Dict[str, Any]:
    #     # Active learning already knows the SCM variables, but we pass them for consistency
    #     return al_predict_fn(tensor, target)
    # 
    # return hybrid_surrogate_predict, al_update_fn


def create_bc_active_learning_wrapper(checkpoint_path: Path,
                                    scm: Any,
                                    learning_rate: float = 1e-4,
                                    seed: int = 42) -> Tuple[Callable, Callable, Any, Any]:
    """
    Create BC surrogate with active learning that exposes params/opt_state.
    
    This wrapper bridges the gap between BC's internal state management
    and the evaluator's expectation of external state management.
    
    Args:
        checkpoint_path: Path to BC surrogate checkpoint  
        scm: The structural causal model
        learning_rate: Learning rate for online updates
        seed: Random seed
        
    Returns:
        Tuple of (predict_fn, update_fn, initial_params, initial_opt_state)
    """
    import optax
    from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
    from ..data_structures.scm import get_variables
    
    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract parameters and config
    params = checkpoint.get('params', checkpoint.get('surrogate_params'))
    config = checkpoint.get('config', {})
    
    # Model configuration
    hidden_dim = config.get('surrogate_hidden_dim', 128)
    num_layers = config.get('surrogate_layers', 4)
    num_heads = config.get('surrogate_heads', 8)
    
    # Get variables from SCM
    variables = list(get_variables(scm))
    
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
    
    net = hk.transform(model_fn)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Create predict function
    def predict_fn(tensor: jnp.ndarray, target: str, vars: List[str]) -> Dict[str, Any]:
        """Predict using current parameters."""
        # Use provided variables or SCM variables
        vars_to_use = vars if vars else variables
        target_idx = vars_to_use.index(target) if target in vars_to_use else len(vars_to_use) - 1
        
        # Apply model
        output = net.apply(params, None, tensor, target_idx, False)
        
        # Extract parent probabilities
        if isinstance(output, dict) and 'parent_probabilities' in output:
            parent_probs = output['parent_probabilities']
        else:
            parent_probs = output
        
        # Convert to marginal probabilities
        marginal_probs = {}
        for i, var in enumerate(vars_to_use):
            if i < len(parent_probs) and var != target:
                marginal_probs[var] = float(parent_probs[i])
        
        # Create posterior
        from ..avici_integration.parent_set.posterior import create_parent_set_posterior
        posterior = create_parent_set_posterior(
            target_variable=target,
            parent_sets=[frozenset()],
            probabilities=jnp.array([1.0]),
            metadata={'marginal_parent_probs': marginal_probs}
        )
        
        return posterior
    
    # Create update function that matches evaluator expectations
    def update_fn(current_params, current_opt_state, posterior, samples, vars, target):
        """Update function matching evaluator signature."""
        # Inline the BC surrogate update logic
        if len(samples) < 5:
            logger.debug("Skipping BC update: insufficient samples")
            return current_params, current_opt_state, {"loss": 0.0, "grad_norm": 0.0, "update_norm": 0.0, "skipped": True}
        
        # Import required functions
        from ..data_structures.sample import get_values
        from ..avici_integration.core import samples_to_avici_format
        
        # Convert samples to tensor format
        data = samples_to_avici_format(samples, vars, target)
        
        # Get target index
        target_idx = vars.index(target) if target in vars else len(vars) - 1
        
        # Define loss function based on data likelihood
        def loss_fn(params):
            # Get model predictions
            output = net.apply(params, None, data, target_idx, False)
            
            # Extract parent probabilities
            if isinstance(output, dict) and 'parent_probabilities' in output:
                parent_probs = output['parent_probabilities']
            else:
                parent_probs = output
            
            # Convert to proper shape if needed
            if len(parent_probs.shape) == 1:
                parent_probs = parent_probs[None, :]  # Add batch dimension
            
            # Simple cross-entropy loss encouraging higher probabilities for variables
            # that have high correlation with target
            # This is a simplified version - ideally would use proper causal scoring
            
            # Extract target values from data
            target_values = data[:, target_idx, 0]  # Values channel
            
            # Compute correlations with other variables
            correlations = []
            for i in range(data.shape[1]):
                if i != target_idx:
                    var_values = data[:, i, 0]
                    # Simple correlation as proxy for parent likelihood
                    corr = jnp.abs(jnp.corrcoef(var_values, target_values)[0, 1])
                    correlations.append(corr)
                else:
                    correlations.append(0.0)
            
            correlations = jnp.array(correlations)
            # Normalize to get pseudo-targets
            pseudo_targets = correlations / (jnp.sum(correlations) + 1e-8)
            
            # Cross-entropy loss
            epsilon = 1e-8
            ce_loss = -jnp.sum(pseudo_targets * jnp.log(parent_probs[0] + epsilon))
            
            # Add L2 regularization
            l2_reg = 1e-4 * sum(jnp.sum(p**2) for p in jax.tree.leaves(params))
            
            return ce_loss + l2_reg
        
        # Compute gradients
        loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
        
        # Update parameters
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        # Update the closure params for predict_fn
        nonlocal params
        params = new_params
        
        metrics = {
            "loss": float(loss_val),
            "grad_norm": float(optax.global_norm(grads)),
            "update_norm": float(optax.global_norm(updates)),
            "n_samples": len(samples)
        }
        
        logger.info(f"BC surrogate update: loss={loss_val:.4f}, grad_norm={metrics['grad_norm']:.6f}, update_norm={metrics['update_norm']:.6f}, n_samples={len(samples)}")
        
        return new_params, new_opt_state, metrics
    
    return predict_fn, update_fn, params, opt_state




def create_optimal_oracle_acquisition(scm: Any,
                                    optimization_direction: str = 'MINIMIZE',
                                    intervention_range: Union[Tuple[float, float], Dict[str, Tuple[float, float]]] = (-2.0, 2.0),
                                    seed: int = 42) -> Callable:
    """
    Create optimal oracle acquisition that uses exact SCM coefficients.
    
    This oracle has perfect knowledge of the SCM structure AND coefficients.
    For perfect interventions, it selects the parent with maximum |coefficient|
    and sets it to the extreme value that optimizes the target.
    
    Args:
        scm: The full structural causal model with mechanisms
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        intervention_range: Range of allowed intervention values
        seed: Random seed (unused but kept for compatibility)
        
    Returns:
        Optimal oracle acquisition function
    """
    from ..data_structures.scm import get_mechanisms, get_parents, get_target
    from ..mechanisms.serializable_mechanisms import LinearMechanism
    
    # Extract SCM information
    mechanisms = get_mechanisms(scm)
    target_var = get_target(scm)
    
    def optimal_oracle_acquisition(tensor: jnp.ndarray,
                                 posterior: Optional[Dict[str, Any]],
                                 target: str,
                                 variables: List[str]) -> Dict[str, Any]:
        """Optimal oracle intervention using exact SCM knowledge."""
        # Get true parents of target
        true_parents = list(get_parents(scm, target))
        
        if not true_parents:
            # Target is a root node - no parents to intervene on
            # Return a dummy intervention
            return {
                'targets': frozenset([variables[0] if variables[0] != target else variables[1]]),
                'values': {variables[0] if variables[0] != target else variables[1]: 0.0}
            }
        
        # Get the target mechanism to access coefficients
        target_mechanism = mechanisms[target]
        
        # Only proceed if it's a LinearMechanism (which it should be for our SCMs)
        if isinstance(target_mechanism, LinearMechanism):
            coefficients = target_mechanism.coefficients
            
            # Find parent with largest |coefficient * extremum| effect
            best_parent = None
            best_effect = 0.0
            best_value = 0.0
            
            for parent in true_parents:
                if parent not in coefficients:
                    continue
                    
                coeff = coefficients[parent]
                
                # Get intervention range for this parent
                if isinstance(intervention_range, dict):
                    # Variable-specific ranges
                    parent_range = intervention_range.get(parent, (-2.0, 2.0))
                else:
                    # Global range
                    parent_range = intervention_range
                
                min_val, max_val = parent_range
                
                # Calculate maximum possible effect magnitude
                # Effect = coeff * intervention_value
                effect_at_min = abs(coeff * min_val)
                effect_at_max = abs(coeff * max_val)
                max_effect = max(effect_at_min, effect_at_max)
                
                # Track which parent can produce the largest effect
                if max_effect > best_effect:
                    best_effect = max_effect
                    best_parent = parent
                    
                    # Determine optimal intervention value
                    if optimization_direction == 'MINIMIZE':
                        # We want to minimize target (most negative effect)
                        if coeff > 0:
                            # Positive coeff: lower value decreases target
                            best_value = min_val
                        else:
                            # Negative coeff: higher value decreases target
                            best_value = max_val
                    else:  # MAXIMIZE
                        # We want to maximize target (most positive effect)
                        if coeff > 0:
                            # Positive coeff: higher value increases target
                            best_value = max_val
                        else:
                            # Negative coeff: lower value increases target
                            best_value = min_val
            
            if best_parent is not None:
                return {
                    'targets': frozenset([best_parent]),
                    'values': {best_parent: best_value}
                }
        
        # Fallback if we couldn't find coefficients
        best_parent = true_parents[0]
        best_value = intervention_range[0] if optimization_direction == 'MINIMIZE' else intervention_range[1]
        
        return {
            'targets': frozenset([best_parent]),
            'values': {best_parent: best_value}
        }
    
    return optimal_oracle_acquisition


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