#!/usr/bin/env python3
"""
Core learning functions for ACBO demo experiments.

Provides model creation, training, and evaluation utilities.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.data_structures import get_values, get_parents
from causal_bayes_opt.avici_integration.core import samples_to_avici_format
from causal_bayes_opt.avici_integration.parent_set import (
    ParentSetPredictionModel, predict_parent_posterior,
    get_marginal_parent_probabilities
)
from causal_bayes_opt.acquisition import AcquisitionState
from causal_bayes_opt.interventions import create_perfect_intervention

# Type aliases for cleaner signatures
PredictorFunction = Callable[..., object]
UpdateFunction = Callable[..., Tuple[object, object, Tuple[float, float, float, float]]]
ModelTuple = Tuple[PredictorFunction, object, object, object, UpdateFunction]


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for progressive learning demo."""
    n_observational_samples: int = 30  # More initial data for complex SCM
    n_intervention_steps: int = 20  # Reasonable number of steps
    learning_rate: float = 1e-3  # More realistic learning rate
    intervention_value_range: Tuple[float, float] = (-2.0, 2.0)
    random_seed: int = 42
    scoring_method: str = "bic"  # Scoring method: "bic", "aic", "mdl", or "likelihood"


def _get_true_parents_for_scm(scm: pyr.PMap, target: str) -> List[str]:
    """Get true parents for the target variable in the given SCM."""
    true_parents_set = get_parents(scm, target)
    return sorted(list(true_parents_set))


def _extract_target_coefficients_from_scm(scm: pyr.PMap, target_variable: str) -> Dict[str, float]:
    """
    Extract actual coefficients for target variable from SCM mechanism descriptors.
    
    This function deterministically extracts the true coefficients used in the SCM
    construction, enabling the oracle to use perfect knowledge of the structure.
    
    Args:
        scm: The structural causal model containing mechanism descriptors
        target_variable: Name of the target variable
        
    Returns:
        Dictionary mapping parent variable names to their coefficients
        
    Raises:
        ValueError: If coefficients cannot be extracted from the SCM
    """
    from causal_bayes_opt.data_structures.scm import get_parents
    
    # Get true parents of target variable
    parents = get_parents(scm, target_variable)
    
    # Try mechanism descriptors first (preferred method)
    if 'mechanism_descriptors' in scm and target_variable in scm['mechanism_descriptors']:
        descriptor = scm['mechanism_descriptors'][target_variable]
        if hasattr(descriptor, 'coefficients'):
            return dict(descriptor.coefficients)
    
    # Fallback: examine coefficients from SCM metadata
    # Erdos-Renyi SCMs store coefficients in metadata during creation
    if 'metadata' in scm:
        metadata = scm['metadata']
        
        # Check if coefficients are stored directly in metadata
        if 'coefficients' in metadata:
            coefficients = {}
            for parent in parents:
                edge_key = (parent, target_variable)
                if edge_key in metadata['coefficients']:
                    coefficients[parent] = metadata['coefficients'][edge_key]
            
            if coefficients:
                return coefficients
    
    # Additional fallback: try to extract from mechanisms directly
    # This is a last resort and involves examining the mechanism function
    mechanisms = scm.get('mechanisms', {})
    if target_variable in mechanisms:
        mechanism = mechanisms[target_variable]
        
        # For mechanisms created with _return_descriptor=True,
        # the descriptor might be stored alongside the function
        if hasattr(mechanism, '_descriptor'):
            descriptor = mechanism._descriptor
            if hasattr(descriptor, 'coefficients'):
                return dict(descriptor.coefficients)
    
    # If all else fails, we cannot extract coefficients
    raise ValueError(
        f"Cannot extract coefficients for target variable '{target_variable}' from SCM. "
        f"SCM may not have mechanism descriptors or coefficient metadata stored. "
        f"Available SCM keys: {list(scm.keys())}, "
        f"Metadata keys: {list(scm.get('metadata', {}).keys()) if 'metadata' in scm else 'no metadata'}"
    )


def compute_likelihood_per_parent_set_jax(parent_sets: List, new_samples: List, 
                                         target_variable: str, variable_order: List[str],
                                         scoring_method: str = "bic") -> jnp.ndarray:
    """
    JAX-compatible computation of scores for each parent set.
    
    For each parent set, compute a score that balances fit quality with model complexity.
    Supports multiple scoring methods to prevent overfitting to large parent sets.
    
    Args:
        parent_sets: List of parent sets (each is a frozenset of variable names)
        new_samples: List of samples to compute likelihood for
        target_variable: Target variable name
        variable_order: Ordered list of all variable names
        scoring_method: One of "bic", "aic", "mdl", or "likelihood" (default: "bic")
        
    Returns:
        JAX array of shape [k] with scores for each parent set (higher is better)
    """
    if not new_samples:
        return jnp.array([0.0] * len(parent_sets))
    
    # Extract target values
    target_values = jnp.array([get_values(s)[target_variable] for s in new_samples])
    n_samples = len(target_values)
    
    # Create a matrix of all variable values [n_samples, n_vars]
    all_values = jnp.zeros((n_samples, len(variable_order)))
    for i, sample in enumerate(new_samples):
        sample_values = get_values(sample)
        for j, var_name in enumerate(variable_order):
            if var_name in sample_values:
                all_values = all_values.at[i, j].set(sample_values[var_name])
    
    # Compute scores for each parent set
    scores = []
    
    for parent_set in parent_sets:
        # Number of parameters in the model
        n_params = len(parent_set) + 1  # coefficients + intercept
        
        if len(parent_set) == 0:
            # No parents: Y ~ Normal(μ, σ²)
            mean_pred = jnp.mean(target_values)
            var_pred = jnp.maximum(jnp.var(target_values), 0.01)  # Avoid zero variance
            # Log likelihood under normal distribution
            log_likelihood = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var_pred) + (target_values - mean_pred)**2 / var_pred)
        else:
            # Has parents: use linear regression likelihood
            # Get parent indices
            parent_indices = jnp.array([variable_order.index(p) for p in parent_set if p in variable_order])
            
            if len(parent_indices) > 0 and n_samples > len(parent_indices):
                # Extract parent values [n_samples, n_parents]
                parent_values = all_values[:, parent_indices]
                
                # Add intercept column [n_samples, n_parents + 1]
                X = jnp.column_stack([jnp.ones(n_samples), parent_values])
                
                # Solve least squares: β = (X^T X)^{-1} X^T y
                XTX = X.T @ X
                XTy = X.T @ target_values
                
                # Add small regularization for numerical stability
                XTX_reg = XTX + 1e-6 * jnp.eye(XTX.shape[0])
                beta = jnp.linalg.solve(XTX_reg, XTy)
                
                # Predictions and residuals
                predictions = X @ beta
                residuals = target_values - predictions
                residual_var = jnp.maximum(jnp.var(residuals), 0.01)
                
                # Log likelihood under normal residuals
                log_likelihood = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * residual_var) + residuals**2 / residual_var)
            else:
                # Fallback: not enough data for regression
                mean_pred = jnp.mean(target_values)
                var_pred = jnp.maximum(jnp.var(target_values), 0.01)
                log_likelihood = -0.5 * jnp.sum(jnp.log(2 * jnp.pi * var_pred) + (target_values - mean_pred)**2 / var_pred)
        
        # Apply scoring method to balance fit and complexity
        if scoring_method == "bic":
            # Bayesian Information Criterion: -2*LL + k*log(n)
            # We negate to make higher scores better
            score = log_likelihood - 0.5 * n_params * jnp.log(n_samples)
        elif scoring_method == "aic":
            # Akaike Information Criterion: -2*LL + 2*k
            # We negate to make higher scores better
            score = log_likelihood - n_params
        elif scoring_method == "mdl":
            # Minimum Description Length: -LL + (k/2)*log(n)
            # We negate to make higher scores better
            score = log_likelihood - 0.5 * n_params * jnp.log(n_samples)
        else:  # "likelihood" or default
            # Raw likelihood (no complexity penalty)
            score = log_likelihood
        
        scores.append(score)
    
    return jnp.array(scores)


def create_learnable_surrogate_model(
    variables: List[str], 
    key: jax.Array, 
    learning_rate: float = 1e-3,
    scoring_method: str = "bic"
) -> ModelTuple:
    """Create learnable ParentSetPredictionModel with online training capability.
    
    Creates a complete training setup including the prediction function,
    neural network, parameters, optimizer state, and update function for
    self-supervised learning using data likelihood.
    
    Args:
        variables: List of variable names in the SCM
        key: JAX random key for parameter initialization
        learning_rate: Learning rate for the Adam optimizer
        scoring_method: Scoring method for parent set evaluation
        
    Returns:
        Tuple containing:
        - Prediction function for getting posteriors
        - Haiku network object
        - Initial parameters
        - Initial optimizer state  
        - Update function for training steps
    """
    import haiku as hk
    import optax
    
    n_vars = len(variables)
    
    def model_fn(x: jnp.ndarray, variable_order: List[str], target_variable: str, is_training: bool = False):
        model = ParentSetPredictionModel(
            layers=4,
            dim=64, 
            max_parent_size=min(5, n_vars-1)
        )
        return model(x, variable_order, target_variable, is_training)
    
    net = hk.transform(model_fn)
    dummy_data = jnp.zeros((10, n_vars, 3))
    params = net.init(key, dummy_data, variables, variables[0], False)
    
    # Create optimizer for online learning
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    def predict_posterior_wrapper(data: jnp.ndarray, variable_order: List[str], 
                                 target_variable: str, current_params: object = None) -> object:
        """Use current model parameters for prediction."""
        active_params = current_params if current_params is not None else params
        return predict_parent_posterior(
            net=net,
            params=active_params, 
            x=data,
            variable_order=variable_order,
            target_variable=target_variable,
            metadata={'model_type': 'ParentSetPredictionModel', 'learning': True}
        )
    
    def update_model_with_data_likelihood(current_params: object, current_opt_state: object, 
                                        _posterior: object, new_samples: List, 
                                        variable_order: List[str], target_variable: str) -> Tuple[object, object, Tuple[float, float, float, float]]:
        """Update model parameters using data likelihood as training signal."""
        
        # Data likelihood loss function for self-supervised learning
        def loss_fn(params):
            # Get new prediction with current params
            data = convert_samples_for_prediction(new_samples, variable_order, target_variable)
            # Use the actual neural network directly for gradient computation
            rng_key = random.PRNGKey(42)  # Fixed key for deterministic gradients
            pred_output = net.apply(params, rng_key, data, variable_order, target_variable, False)
            
            # Convert model output to data likelihood
            # pred_output is a dict with 'parent_set_logits', 'parent_sets', 'k'
            logits = pred_output['parent_set_logits']  # [k] logits for top-k parent sets
            parent_sets = pred_output['parent_sets']   # List of k parent sets
            
            # Convert logits to probabilities
            probs = jax.nn.softmax(logits)  # [k] probabilities over parent sets
            
            # Compute scores for each parent set (JAX-compatible version)
            # Using configurable scoring method to prevent overfitting to large parent sets
            scores_per_set = compute_likelihood_per_parent_set_jax(
                parent_sets, new_samples, target_variable, variable_order, 
                scoring_method=scoring_method
            )  # [k] scores for each parent set
            
            # Weighted score (expectation over posterior)
            total_score = jnp.sum(probs * scores_per_set)
            
            # Return negative score as loss (minimize this)
            return -total_score
        
        # Compute gradients and update
        loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        # Compute diagnostics for monitoring learning
        param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(current_params)))
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree_util.tree_leaves(updates)))
        
        return new_params, new_opt_state, (float(loss_val), float(param_norm), float(grad_norm), float(update_norm))
    
    return predict_posterior_wrapper, net, params, opt_state, update_model_with_data_likelihood


def convert_samples_for_prediction(samples: List[pyr.PMap], variables: List[str], target: str) -> jnp.ndarray:
    """Convert samples to AVICI format for prediction."""
    return samples_to_avici_format(samples, variables, target)


def create_acquisition_state_from_buffer(buffer, surrogate_fn: callable, 
                                       variables: List[str], target: str, _step: int, 
                                       current_params: object = None) -> AcquisitionState:
    """Create acquisition state using surrogate model."""
    all_samples = buffer.get_all_samples()
    avici_data = convert_samples_for_prediction(all_samples, variables, target)
    
    # Get posterior from surrogate model with current parameters
    posterior = surrogate_fn(avici_data, variables, target, current_params)
    
    # Get best value from samples
    target_values = [get_values(s)[target] for s in all_samples]
    best_value = float(max(target_values))
    
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=target,
        step=_step,
        metadata=pyr.m()
    )


def create_random_intervention_policy(variables: List[str], target_variable: str, 
                                    intervention_range: Tuple[float, float]) -> Callable[..., pyr.PMap]:
    """Create random intervention policy for unbiased exploration."""
    # Get candidate variables for intervention (exclude target)
    candidate_vars = [v for v in variables if v != target_variable]
    min_val, max_val = intervention_range
    
    def select_random_intervention(_state: object = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """Randomly select variable and intervention value."""
        if key is None:
            key = random.PRNGKey(42)
            
        # Randomly select variable to intervene on
        var_key, val_key = random.split(key)
        var_idx = random.randint(var_key, (), 0, len(candidate_vars))
        chosen_var = candidate_vars[var_idx]
        
        # Randomly select intervention value
        intervention_value = float(random.uniform(val_key, (), minval=min_val, maxval=max_val))
        
        return create_perfect_intervention(
            targets=frozenset([chosen_var]),
            values={chosen_var: intervention_value}
        )
    
    return select_random_intervention


def create_fixed_intervention_policy(variables: List[str], target_variable: str, 
                                   fixed_variable: str, fixed_value: float) -> Callable[..., pyr.PMap]:
    """Create fixed intervention policy that always intervenes on the same variable with the same value.
    
    This policy is designed to make the structure unidentifiable by providing no variation
    in intervention targets or values, which should prevent correct causal discovery.
    
    Args:
        variables: List of all variables in the SCM
        target_variable: The target variable (excluded from interventions)  
        fixed_variable: The variable to always intervene on
        fixed_value: The value to always set the variable to
        
    Returns:
        Intervention policy function that always returns the same intervention
    """
    # Validate inputs
    candidate_vars = [v for v in variables if v != target_variable]
    assert fixed_variable in candidate_vars, f"Fixed variable {fixed_variable} must be a non-target variable"
    
    def select_fixed_intervention(_state: object = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """Always select the same variable and intervention value."""
        return create_perfect_intervention(
            targets=frozenset([fixed_variable]),
            values={fixed_variable: fixed_value}
        )
    
    return select_fixed_intervention


def create_oracle_intervention_policy(variables: List[str], target_variable: str, 
                                     scm: pyr.PMap, intervention_value_range: Tuple[float, float] = (-2.0, 2.0),
                                     intervention_strength: float = 2.0) -> Callable[..., pyr.PMap]:
    """Create oracle intervention policy that always chooses the absolute best intervention.
    
    This policy has perfect knowledge of the true causal structure AND target optimization.
    It deterministically selects the intervention that maximizes the target variable.
    
    Args:
        variables: List of all variables in the SCM
        target_variable: The target variable (excluded from interventions)
        scm: The true structural causal model
        intervention_value_range: Range for intervention values
        intervention_strength: Strength of interventions (default value)
        
    Returns:
        Intervention policy function that uses oracle knowledge for optimal selection
    """
    # Get true parents of target variable
    true_parents = set(get_parents(scm, target_variable))
    candidate_vars = [v for v in variables if v != target_variable]
    
    # Extract true coefficients from SCM deterministically
    try:
        target_coefficients = _extract_target_coefficients_from_scm(scm, target_variable)
    except ValueError as e:
        # Fallback: use positive coefficients if extraction fails
        logger.warning(f"Failed to extract coefficients from SCM: {e}")
        logger.warning("Using fallback positive coefficients for oracle policy")
        target_coefficients = {parent: 1.0 for parent in true_parents}
    
    min_val, max_val = intervention_value_range
    
    def select_oracle_intervention(_state: object = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """Select the absolute best intervention using perfect oracle knowledge."""
        
        # Calculate expected improvement for each possible intervention
        best_improvement = float('-inf')
        best_var = None
        best_value = None
        
        # Only consider true parents (oracle knows they are the only ones that matter)
        for var in candidate_vars:
            if var in true_parents and var in target_coefficients:
                coeff = target_coefficients[var]
                
                # Choose intervention value that maximizes target
                if coeff > 0:
                    # Positive coefficient: use maximum positive value
                    optimal_value = max_val
                else:
                    # Negative coefficient: use minimum (most negative) value  
                    optimal_value = min_val
                
                # Calculate expected improvement: coefficient * intervention_value
                expected_improvement = coeff * optimal_value
                
                if expected_improvement > best_improvement:
                    best_improvement = expected_improvement
                    best_var = var
                    best_value = optimal_value
        
        # Fallback if no parents found or no coefficients known
        if best_var is None:
            # Choose the first parent variable with max positive intervention
            if true_parents:
                best_var = next(iter(true_parents))
                best_value = max_val  # Assume positive coefficient
            else:
                # No parents, choose first candidate variable
                best_var = candidate_vars[0] if candidate_vars else variables[0]
                best_value = max_val
        
        # Create intervention
        intervention = create_perfect_intervention(
            targets=frozenset([best_var]),
            values={best_var: best_value}
        )
        
        # Optional debug logging (can be enabled for debugging)
        # print(f"ORACLE: Optimal intervention on {best_var} = {best_value:.2f} "
        #       f"(coeff: {target_coefficients.get(best_var, 'unknown')}, improvement: {best_improvement:.3f})")
        
        return intervention
    
    return select_oracle_intervention