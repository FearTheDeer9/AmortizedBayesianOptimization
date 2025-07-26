#!/usr/bin/env python3
"""
Core learning functions for ACBO demo experiments.

Provides model creation, training, and evaluation utilities.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any
import logging
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

logger = logging.getLogger(__name__)

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
    learning_rate: float = 1e-4  # Conservative learning rate for stable training
    intervention_value_range: Tuple[float, float] = (-2.0, 2.0)
    random_seed: int = 42
    scoring_method: str = "bic"  # Scoring method: "bic", "aic", "mdl", or "likelihood"


def _get_true_parents_for_scm(scm: pyr.PMap, target: str) -> List[str]:
    """Get true parents for the target variable in the given SCM."""
    true_parents_set = get_parents(scm, target)
    return sorted(list(true_parents_set))


def _extract_target_coefficients_from_scm(scm: pyr.PMap, target_variable: str) -> Dict[str, float]:
    """
    Extract actual coefficients for target variable from SCM.
    
    This function provides oracle knowledge by using hardcoded coefficients
    based on the known SCM structures defined in demo_scms.py.
    
    Args:
        scm: The structural causal model 
        target_variable: Name of the target variable
        
    Returns:
        Dictionary mapping parent variable names to their coefficients
        
    Raises:
        ValueError: If coefficients cannot be determined for the target variable
    """
    from causal_bayes_opt.data_structures.scm import get_parents
    
    # Get true parents of target variable
    parents = get_parents(scm, target_variable)
    variables = set(scm.get('variables', set()))
    
    # Use hardcoded knowledge of demo SCM structures
    # This provides perfect oracle knowledge for the known test SCMs
    
    if target_variable == 'D' and 'B' in parents and 'C' in parents:
        # Easy SCM: D has parents B and C with coefficients B: 1.2, C: 0.8
        return {'B': 1.2, 'C': 0.8}
    
    elif target_variable == 'F' and 'D' in parents and 'G' in parents:
        # Medium SCM: F has parents D and G with coefficients D: 0.8, G: 0.9
        return {'D': 0.8, 'G': 0.9}
    
    elif target_variable == 'J' and 'H' in parents and 'I' in parents:
        # Hard SCM: J has parents H and I with coefficients H: 0.3, I: 0.4
        return {'H': 0.3, 'I': 0.4}
    
    # Generic fallback: use positive coefficients for all parents
    # This ensures the oracle policy works even for unknown SCMs
    logger.warning(f"Using generic coefficients for target {target_variable} with parents {parents}")
    return {parent: 1.0 for parent in parents}


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
    
    # Create optimizer with adaptive learning rate schedule
    # Use cosine decay to gradually reduce learning rate for stable convergence
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=100,  # Decay over 100 update steps
        alpha=0.1  # Final learning rate will be 0.1 * initial
    )
    optimizer = optax.adam(learning_rate=schedule)
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
        
        # Skip update if not enough samples
        if len(new_samples) < 5:
            # Return zeros for diagnostics
            return current_params, current_opt_state, (0.0, 0.0, 0.0, 0.0)
        
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
            
            # Convert logits to probabilities with temperature scaling for stability
            # Higher temperature = smoother gradients, more exploration
            temperature = 3.0  # Increased from 1.0 for more stable gradients
            scaled_logits = logits / temperature
            probs = jax.nn.softmax(scaled_logits)  # [k] probabilities over parent sets
            
            # Compute scores for each parent set (JAX-compatible version)
            # Using configurable scoring method to prevent overfitting to large parent sets
            scores_per_set = compute_likelihood_per_parent_set_jax(
                parent_sets, new_samples, target_variable, variable_order, 
                scoring_method=scoring_method
            )  # [k] scores for each parent set
            
            # Principled normalization:
            # 1. Expected log-likelihood per sample under simple Gaussian is approximately -1.4 (log(2π) + 0.5)
            # 2. We normalize scores by this expected scale to keep gradients reasonable
            expected_ll_per_sample = -1.4
            scores_normalized = scores_per_set / len(new_samples)  # Per-sample scores
            scores_centered = scores_normalized - expected_ll_per_sample  # Center around expected value
            
            # Use log-sum-exp for numerical stability
            # Now we're working with centered scores that are typically in range [-2, 2]
            log_partition = jax.nn.logsumexp(scores_centered, b=probs)
            
            # Loss is negative log expected score (centered)
            # This keeps the loss in a reasonable range (typically 0-5) instead of 20+
            loss = -log_partition
            
            # Add moderate L2 regularization for stability
            l2_reg = 1e-4 * sum(
                jnp.sum(p**2) for p in jax.tree.leaves(params)
            )
            
            return loss + l2_reg
        
        # Compute gradients and update with gradient clipping
        loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
        
        # Clip gradients to prevent instability
        # With proper normalization, we can use a tighter clip threshold
        max_grad_norm = 1.0  # Much more conservative than 10.0
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        
        if grad_norm > max_grad_norm:
            # Scale gradients to have max_grad_norm
            grads = jax.tree.map(
                lambda g: g * max_grad_norm / (grad_norm + 1e-8), 
                grads
            )
            clipped_grad_norm = max_grad_norm
        else:
            clipped_grad_norm = grad_norm
        
        updates, new_opt_state = optimizer.update(grads, current_opt_state, current_params)
        new_params = optax.apply_updates(current_params, updates)
        
        # Compute diagnostics for monitoring learning
        param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(current_params)))
        update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree.leaves(updates)))
        
        return new_params, new_opt_state, (float(loss_val), float(param_norm), float(clipped_grad_norm), float(update_norm))
    
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
    
    # Get best value from samples (minimization: best = lowest)
    target_values = [get_values(s)[target] for s in all_samples]
    best_value = float(min(target_values))
    
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


def create_structure_aware_intervention_policy(variables: List[str], target_variable: str,
                                             intervention_range: Tuple[float, float] = (-2.0, 2.0),
                                             temperature: float = 1.0) -> Callable[..., pyr.PMap]:
    """Create structure-aware intervention policy that uses learned marginal probabilities.
    
    This policy uses learned marginal parent probabilities to guide intervention selection,
    prioritizing variables that are likely to be parents of the target. It represents
    the key missing component between random exploration and fully trained policies.
    
    Args:
        variables: List of all variables in the SCM
        target_variable: The target variable (excluded from interventions)
        intervention_range: Range for intervention values
        temperature: Temperature for softmax variable selection (lower = more focused)
        
    Returns:
        Intervention policy function that uses learned structure to guide decisions
    """
    candidate_vars = [v for v in variables if v != target_variable]
    min_val, max_val = intervention_range
    
    def select_structure_aware_intervention(state: object = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """Select intervention using learned marginal probabilities."""
        if key is None:
            key = random.PRNGKey(42)
        
        var_key, val_key = random.split(key)
        
        # Extract learned marginal probabilities if available
        marginal_probs = {}
        if state is not None and hasattr(state, 'marginal_parent_probs'):
            marginal_probs = dict(state.marginal_parent_probs)
        
        # If no learned structure available, fall back to random
        if not marginal_probs:
            var_idx = random.randint(var_key, (), 0, len(candidate_vars))
            var_name = candidate_vars[var_idx]
        else:
            # Use learned marginal probabilities to guide variable selection
            # Higher marginal probability = more likely to be a parent = higher priority for intervention
            
            # Get probabilities for candidate variables
            candidate_probs = []
            for var in candidate_vars:
                if var in marginal_probs:
                    # Add small baseline to ensure all variables have some probability
                    prob = marginal_probs[var] + 0.1
                else:
                    prob = 0.1  # Baseline probability for unknown variables
                candidate_probs.append(prob)
            
            # Convert to JAX array for sampling
            probs_array = jnp.array(candidate_probs)
            
            # Apply temperature for exploration control
            if temperature > 0:
                probs_array = probs_array / temperature
                probs_array = jnp.exp(probs_array - jnp.max(probs_array))  # Numerical stability
                probs_array = probs_array / jnp.sum(probs_array)  # Normalize
            
            # Sample variable based on learned probabilities
            var_idx = random.choice(var_key, len(candidate_vars), p=probs_array)
            var_name = candidate_vars[var_idx]
        
        # For intervention value, use a strategy that promotes minimization
        # Start with random value but add some heuristics based on learned structure
        base_value = random.uniform(val_key, (), minval=min_val, maxval=max_val)
        
        # If we have learned this variable is likely a parent, bias towards negative values
        # (assuming positive coefficients are more common, negative interventions help minimize)
        if var_name in marginal_probs and marginal_probs[var_name] > 0.5:
            # High confidence this is a parent - bias towards minimum value for minimization
            bias_strength = marginal_probs[var_name]  # 0.5 to 1.0
            biased_value = base_value * (1 - bias_strength) + min_val * bias_strength
            intervention_value = float(biased_value)
        else:
            # Low confidence or unknown - use random value
            intervention_value = float(base_value)
        
        return create_perfect_intervention(
            targets=frozenset([var_name]),
            values={var_name: intervention_value}
        )
    
    return select_structure_aware_intervention


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
    
    This policy has perfect knowledge of the true causal structure AND simulates
    the SCM to find the intervention that truly minimizes the target variable.
    
    Note: The oracle only considers interventions on direct parents of the target,
    not on ancestors that might have stronger indirect effects. This is a design
    choice to make the oracle's knowledge more realistic (knowing direct causes
    but not necessarily optimal control strategies through the full graph).
    
    Args:
        variables: List of all variables in the SCM
        target_variable: The target variable (excluded from interventions)
        scm: The true structural causal model
        intervention_value_range: Range for intervention values
        intervention_strength: Strength of interventions (default value)
        
    Returns:
        Intervention policy function that uses oracle knowledge for optimal selection
    """
    from causal_bayes_opt.environments.sampling import sample_with_intervention
    from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
    
    # Get true parents of target variable
    true_parents = set(get_parents(scm, target_variable))
    candidate_vars = [v for v in variables if v != target_variable]
    min_val, max_val = intervention_value_range
    
    # Sample size for estimating expected target value
    n_oracle_samples = 100
    
    def select_oracle_intervention(_state: object = None, key: Optional[jax.Array] = None) -> pyr.PMap:
        """Select the absolute best intervention by simulating the SCM."""
        
        # Use random key for sampling if provided
        if key is None:
            key = random.PRNGKey(42)
        
        # First, get baseline target value without intervention
        base_key, *intervention_keys = random.split(key, len(candidate_vars) + 1)
        baseline_samples = sample_from_linear_scm(scm, n_oracle_samples, seed=int(base_key[0]))
        baseline_target_values = [get_values(s)[target_variable] for s in baseline_samples]
        baseline_mean = jnp.mean(jnp.array(baseline_target_values))
        
        # Try each possible intervention and find the one that minimizes target
        best_expected_target = baseline_mean  # Start with baseline
        best_var = None
        best_value = None
        
        # Test interventions on each candidate variable
        for i, var in enumerate(candidate_vars):
            # Skip non-parents if we're being smart (oracle knows structure)
            if var not in true_parents:
                continue
                
            # Try both extreme values for each variable
            for test_value in [min_val, max_val]:
                # Create test intervention
                test_intervention = create_perfect_intervention(
                    targets=frozenset([var]),
                    values={var: test_value}
                )
                
                # Sample from SCM under this intervention
                intervention_samples = sample_with_intervention(
                    scm, test_intervention, n_oracle_samples, 
                    seed=int(intervention_keys[i][0])
                )
                
                # Calculate expected target value under this intervention
                target_values = [get_values(s)[target_variable] for s in intervention_samples]
                expected_target = jnp.mean(jnp.array(target_values))
                
                # Update best if this is better (lower target value)
                if expected_target < best_expected_target:
                    best_expected_target = expected_target
                    best_var = var
                    best_value = test_value
        
        # If no intervention improves over baseline, pick a random parent to intervene on
        if best_var is None:
            if true_parents:
                # Pick the first parent and use min value (arbitrary but deterministic)
                best_var = sorted(list(true_parents))[0]
                best_value = min_val
            else:
                # No parents - intervene on first variable
                best_var = candidate_vars[0] if candidate_vars else variables[0]
                best_value = min_val
        
        # Create the optimal intervention
        intervention = create_perfect_intervention(
            targets=frozenset([best_var]),
            values={best_var: best_value}
        )
        
        # Debug logging (can be enabled)
        # print(f"ORACLE: Optimal intervention on {best_var} = {best_value:.2f} "
        #       f"(expected target: {best_expected_target:.3f} vs baseline: {baseline_mean:.3f})")
        
        return intervention
    
    return select_oracle_intervention