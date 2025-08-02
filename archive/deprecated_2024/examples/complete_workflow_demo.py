#!/usr/bin/env python3
"""
ACBO Complete Workflow Demo

Validates the end-to-end ACBO pipeline with real untrained models:
1. Create SCM with known structure  
2. Generate observational data
3. Use real ParentSetPredictionModel for posteriors
4. Apply interventions and update posteriors iteratively

Organized into modular components with comprehensive validation capabilities.
Includes progressive learning, intervention strategy testing, and difficulty studies.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

# Import modular components
try:
    # Try relative imports first (when used as module)
    from .demo_scms import create_easy_scm, create_easy_scm_with_disconnected_var, create_medium_scm, create_hard_scm
    from .demo_learning import (
        DemoConfig, _get_true_parents_for_scm, create_learnable_surrogate_model,
        create_acquisition_state_from_buffer, create_random_intervention_policy,
        create_fixed_intervention_policy, create_oracle_intervention_policy,
        create_structure_aware_intervention_policy
    )
    from .demo_evaluation import (
        analyze_convergence, compute_data_likelihood_from_posterior,
        compare_intervention_tests, print_learning_progress, print_final_summary,
        validate_experiment_results
    )
except ImportError:
    # Fall back to absolute imports (when run directly)
    from demo_scms import create_easy_scm, create_easy_scm_with_disconnected_var, create_medium_scm, create_hard_scm
    from demo_learning import (
        DemoConfig, _get_true_parents_for_scm, create_learnable_surrogate_model,
        create_acquisition_state_from_buffer, create_random_intervention_policy,
        create_fixed_intervention_policy, create_oracle_intervention_policy,
        create_structure_aware_intervention_policy
    )
    from demo_evaluation import (
        analyze_convergence, compute_data_likelihood_from_posterior,
        compare_intervention_tests, print_learning_progress, print_final_summary,
        validate_experiment_results
    )

# Core framework imports
from causal_bayes_opt.data_structures import (
    get_target, get_variables, create_sample, get_values, 
    ExperienceBuffer, create_empty_buffer
)
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.environments.sampling import sample_with_intervention
from causal_bayes_opt.avici_integration.parent_set import get_marginal_parent_probabilities
from causal_bayes_opt.analysis.trajectory_metrics import compute_f1_score_from_marginals


def generate_initial_data(scm: pyr.PMap, config: DemoConfig, key: jax.Array) -> Tuple[List[pyr.PMap], ExperienceBuffer]:
    """Generate initial observational data."""
    if config.n_observational_samples > 0:
        initial_samples = sample_from_linear_scm(scm, config.n_observational_samples, seed=int(key[0]))
        
        # Create buffer and add samples
        buffer = create_empty_buffer()
        for sample in initial_samples:
            buffer.add_observation(sample)
        
        return initial_samples, buffer
    else:
        # No initial observational data
        return [], create_empty_buffer()


def run_progressive_learning_demo_with_scm(
    scm: pyr.PMap, 
    config: DemoConfig, 
    pretrained_surrogate: Optional[Tuple[Any, Any, Any, Any, Any]] = None,
    pretrained_acquisition: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Run progressive learning demo with a specific SCM.
    
    This is the core learning loop that demonstrates:
    1. Self-supervised learning using data likelihood 
    2. Progressive model parameter updates
    3. Diverse intervention selection
    4. Convergence to true causal structure
    
    Args:
        scm: The structural causal model to run experiments on
        config: Demo configuration parameters
        pretrained_surrogate: Optional tuple of (surrogate_fn, net, params, opt_state, update_fn)
                              for BC methods. If None, creates a new learnable model.
        pretrained_acquisition: Optional pretrained acquisition function for BC methods.
                               If None, creates a random intervention policy.
    """
    print(f"🧠 Progressive Learning Demo")
    print("=" * 60)
    
    # Setup
    key = random.PRNGKey(config.random_seed)
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    
    # Get true parents based on SCM structure (needed for F1 score computation)
    true_parents = _get_true_parents_for_scm(scm, target)
    
    # Create or use surrogate model and random intervention policy
    key, surrogate_key = random.split(key)
    if pretrained_surrogate is not None:
        # Use pretrained BC surrogate model
        surrogate_fn, _net, params, opt_state, update_fn = pretrained_surrogate
        print("🎯 Using pretrained BC surrogate model")
    else:
        # Create new learnable surrogate model
        surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
            variables, surrogate_key, config.learning_rate, config.scoring_method
        )
        print("🧠 Using learnable surrogate model")
    
    # Create or use intervention policy
    if pretrained_acquisition is not None:
        # Use pretrained acquisition policy (could be BC, GRPO, or Oracle)
        intervention_fn = pretrained_acquisition
        print("🎯 Using pretrained acquisition policy")
    else:
        # Create random intervention policy
        intervention_fn = create_random_intervention_policy(variables, target, config.intervention_value_range)
        print("🎲 Using random intervention policy")
    
    # Generate initial data
    key, data_key = random.split(key)
    initial_samples, buffer = generate_initial_data(scm, config, data_key)
    
    # Track learning progress
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Initial progress tracking (minimization)
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        initial_mean = jnp.mean(jnp.array(initial_values))  # Use mean instead of min
        best_so_far = min(initial_values)  # Still track best for other purposes
    else:
        initial_mean = 0.0  # No initial data
        best_so_far = float('inf')
    target_progress.append(initial_mean)  # Store mean as first entry
    
    # Get initial posterior and metrics (only if we have samples)
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
        
        # Initial data likelihood
        initial_likelihood = compute_data_likelihood_from_posterior(current_state.posterior, initial_samples, target)
        data_likelihood_progress.append(initial_likelihood)
    else:
        # No initial data - start with high uncertainty
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        data_likelihood_progress.append(0.0)
    
    # Run intervention and learning loop
    keys = random.split(key, config.n_intervention_steps)
    current_params = params
    current_opt_state = opt_state
    
    for step in range(config.n_intervention_steps):
        # Get current posterior with latest parameters (only if we have samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 1:
            current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
        else:
            current_state = None  # No samples yet for meaningful state
        
        # Execute random intervention
        intervention = intervention_fn(key=keys[step])
        _, outcome_key = random.split(keys[step])
        
        if intervention and intervention.get('values'):
            # Apply intervention
            outcome = sample_with_intervention(scm, intervention, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_intervention(intervention, outcome)
        else:
            # Observational fallback
            outcome = sample_from_linear_scm(scm, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_observation(outcome)
        
        # Update model with more samples for better learning signal
        all_samples = buffer.get_all_samples()
        new_samples = all_samples[-15:] if len(all_samples) >= 15 else all_samples
        
        # Try to update model parameters using self-supervised signal (only if we have enough samples)
        if len(all_samples) >= 5 and current_state is not None:
            try:
                current_params, current_opt_state, (loss, param_norm, grad_norm, update_norm) = update_fn(
                    current_params, current_opt_state, current_state.posterior, 
                    new_samples, variables, target
                )
            except Exception:
                # If update fails, continue with current parameters
                loss, param_norm, grad_norm, update_norm = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            # Not enough samples yet for meaningful updates
            loss, param_norm, grad_norm, update_norm = 0.0, 0.0, 0.0, 0.0
        
        # Track progress
        outcome_value = get_values(outcome)[target]
        best_so_far = min(best_so_far, outcome_value) if best_so_far != float('inf') else outcome_value
        target_progress.append(outcome_value)  # Track actual value, not cumulative best
        
        # Get updated state for metrics (only if we have enough samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step+1, current_params)
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
            
            # Compute data likelihood progress
            likelihood = compute_data_likelihood_from_posterior(updated_state.posterior, all_samples, target)
            data_likelihood_progress.append(likelihood)
        else:
            # Not enough samples for meaningful metrics yet
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
        
        # Store step info (only if we have meaningful metrics)
        if len(all_samples) >= 5:
            # Compute F1 score from marginal probabilities
            f1_score = compute_f1_score_from_marginals(
                dict(updated_state.marginal_parent_probs),
                true_parents,
                threshold=0.5
            )
            
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': updated_state.uncertainty_bits,
                'marginals': dict(updated_state.marginal_parent_probs),
                'data_likelihood': likelihood,
                'f1_score': f1_score
            })
        else:
            # Minimal logging for early steps
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0,
                'f1_score': 0.0  # No meaningful F1 score yet
            })
    
    # Final analysis
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, current_params)
    
    # Count intervention types
    intervention_counts = {}
    for step_info in learning_history:
        intervention = step_info['intervention']
        if intervention and intervention.get('values'):
            vars_intervened = list(intervention['values'].keys())
            for var in vars_intervened:
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
        else:
            intervention_counts['observational'] = intervention_counts.get('observational', 0) + 1
    
    # Get true parents based on SCM structure
    true_parents = _get_true_parents_for_scm(scm, target)
    
    return {
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'initial_mean': target_progress[0],
        'intervention_mean': jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0],
        'reduction': target_progress[0] - (jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0]),  # Positive reduction is good
        'total_samples': buffer.size(),
        'intervention_counts': intervention_counts,
        
        # Learning progress metrics
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        
        # Final state
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_state.marginal_parent_probs,
        'converged_to_truth': analyze_convergence(marginal_prob_progress, true_parents)
    }


def run_progressive_learning_demo_with_custom_policy(
    scm: pyr.PMap, 
    config: DemoConfig,
    intervention_fn: Callable[[Any, pyr.PMap, jax.Array], pyr.PMap]
) -> Dict[str, Any]:
    """
    Run progressive learning demo with a custom intervention policy.
    
    This function is identical to run_progressive_learning_demo_with_scm except
    it accepts a custom intervention function that can use the acquisition state,
    SCM, and random key to make decisions. This enables integration with trained
    policies like the enriched GRPO policy.
    
    Args:
        scm: Structural causal model
        config: Demo configuration
        intervention_fn: Custom intervention function with signature (state, scm, key) -> intervention
        
    Returns:
        Results dictionary with learning history and metrics
    """
    print(f"🎯 Progressive Learning with Custom Policy")
    print("=" * 60)
    
    # Setup
    key = random.PRNGKey(config.random_seed)
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    
    # Create learnable surrogate model
    key, surrogate_key = random.split(key)
    surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, surrogate_key, config.learning_rate, config.scoring_method
    )
    
    # Generate initial data
    key, data_key = random.split(key)
    initial_samples, buffer = generate_initial_data(scm, config, data_key)
    
    # Track learning progress
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Initial progress tracking (minimization)
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        initial_mean = jnp.mean(jnp.array(initial_values))  # Use mean instead of min
        best_so_far = min(initial_values)  # Still track best for other purposes
    else:
        initial_mean = 0.0  # No initial data
        best_so_far = float('inf')
    target_progress.append(initial_mean)  # Store mean as first entry
    
    # Get initial posterior and metrics (only if we have samples)
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
        
        # Initial data likelihood
        initial_likelihood = compute_data_likelihood_from_posterior(current_state.posterior, initial_samples, target)
        data_likelihood_progress.append(initial_likelihood)
    else:
        # No initial data - start with high uncertainty
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        data_likelihood_progress.append(0.0)
    
    # Run intervention and learning loop
    keys = random.split(key, config.n_intervention_steps)
    current_params = params
    current_opt_state = opt_state
    
    for step in range(config.n_intervention_steps):
        # Get current posterior with latest parameters (only if we have samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 1:
            current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
        else:
            current_state = None  # No samples yet for meaningful state
        
        # Execute custom intervention policy
        intervention = intervention_fn(current_state, scm, keys[step])
        _, outcome_key = random.split(keys[step])
        
        if intervention and intervention.get('values'):
            # Apply intervention
            outcome = sample_with_intervention(scm, intervention, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_intervention(intervention, outcome)
        else:
            # Observational fallback
            outcome = sample_from_linear_scm(scm, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_observation(outcome)
        
        # Update model with more samples for better learning signal
        all_samples = buffer.get_all_samples()
        new_samples = all_samples[-15:] if len(all_samples) >= 15 else all_samples
        
        # Try to update model parameters using self-supervised signal (only if we have enough samples)
        if len(all_samples) >= 5 and current_state is not None:
            try:
                current_params, current_opt_state, (loss, param_norm, grad_norm, update_norm) = update_fn(
                    current_params, current_opt_state, current_state.posterior, 
                    new_samples, variables, target
                )
            except Exception:
                # If update fails, continue with current parameters
                loss, param_norm, grad_norm, update_norm = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            # Not enough samples yet for meaningful updates
            loss, param_norm, grad_norm, update_norm = 0.0, 0.0, 0.0, 0.0
        
        # Track progress
        outcome_value = get_values(outcome)[target]
        best_so_far = min(best_so_far, outcome_value) if best_so_far != float('inf') else outcome_value
        target_progress.append(outcome_value)  # Track actual value, not cumulative best
        
        # Get updated state for metrics (only if we have enough samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step+1, current_params)
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
            
            # Compute data likelihood progress
            likelihood = compute_data_likelihood_from_posterior(updated_state.posterior, all_samples, target)
            data_likelihood_progress.append(likelihood)
        else:
            # Not enough samples for meaningful metrics yet
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
        
        # Store step info (only if we have meaningful metrics)
        if len(all_samples) >= 5:
            # Compute F1 score from marginal probabilities
            f1_score = compute_f1_score_from_marginals(
                marginal_prob_progress[-1],
                true_parents,
                threshold=0.5
            )
            
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': uncertainty_progress[-1],
                'marginals': marginal_prob_progress[-1],
                'data_likelihood': data_likelihood_progress[-1],
                'f1_score': f1_score
            })
        else:
            # Minimal logging for early steps
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0,
                'f1_score': 0.0  # No meaningful F1 score yet
            })
    
    # Final analysis (identical to regular demo)
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, current_params)
    
    # Count intervention types
    intervention_counts = {}
    for step_info in learning_history:
        intervention = step_info['intervention']
        if intervention and intervention.get('values'):
            vars_intervened = list(intervention['values'].keys())
            for var in vars_intervened:
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
        else:
            intervention_counts['observational'] = intervention_counts.get('observational', 0) + 1
    
    # Get true parents based on SCM structure
    true_parents = _get_true_parents_for_scm(scm, target)
    
    return {
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'initial_mean': target_progress[0],
        'intervention_mean': jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0],
        'reduction': target_progress[0] - (jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0]),  # Positive reduction is good
        'total_samples': buffer.size(),
        'intervention_counts': intervention_counts,
        
        # Learning progress metrics
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        
        # Final state
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_state.marginal_parent_probs,
        'converged_to_truth': analyze_convergence(marginal_prob_progress, true_parents),
        
        # Mark as custom policy method
        'method': 'custom_policy_learning'
    }


def run_progressive_learning_demo_with_oracle_interventions(scm: pyr.PMap, config: DemoConfig) -> Dict[str, Any]:
    """
    Run progressive learning demo with oracle intervention policy.
    
    This version uses the same neural network surrogate learning as the regular demo,
    but replaces random intervention selection with oracle intervention policy that
    has perfect knowledge of the true causal structure.
    
    Args:
        scm: The structural causal model (used for oracle knowledge)
        config: Configuration for the demo
        
    Returns:
        Dictionary with learning results and convergence metrics
    """
    print(f"🧠 Progressive Learning Demo (Oracle Interventions)")
    print("=" * 60)
    
    # Setup (identical to regular demo)
    key = random.PRNGKey(config.random_seed)
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    
    # Create learnable surrogate model (same as regular demo)
    key, surrogate_key = random.split(key)
    surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, surrogate_key, config.learning_rate, config.scoring_method
    )
    
    # Create ORACLE intervention policy (this is the key difference)
    intervention_fn = create_oracle_intervention_policy(
        variables, target, scm, config.intervention_value_range
    )
    
    # Generate initial data (identical to regular demo)
    key, data_key = random.split(key)
    initial_samples, buffer = generate_initial_data(scm, config, data_key)
    
    # Track learning progress (identical to regular demo)
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Initial progress tracking (minimization)
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        initial_mean = jnp.mean(jnp.array(initial_values))  # Use mean instead of min
        best_so_far = min(initial_values)  # Still track best for other purposes
    else:
        initial_mean = 0.0  # No initial data
        best_so_far = float('inf')
    target_progress.append(initial_mean)  # Store mean as first entry
    
    # Get initial posterior and metrics (identical to regular demo)
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
        
        # Initial data likelihood
        initial_likelihood = compute_data_likelihood_from_posterior(current_state.posterior, initial_samples, target)
        data_likelihood_progress.append(initial_likelihood)
    else:
        # No initial data - start with high uncertainty
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        data_likelihood_progress.append(0.0)
    
    # Run intervention and learning loop (same logic, different intervention selection)
    keys = random.split(key, config.n_intervention_steps)
    current_params = params
    current_opt_state = opt_state
    
    for step in range(config.n_intervention_steps):
        step_key = keys[step]
        
        # ORACLE intervention selection (this is the key difference from regular demo)
        intervention = intervention_fn(key=step_key)
        
        # Sample outcome from SCM with intervention (identical to regular demo)
        from causal_bayes_opt.environments.sampling import sample_with_intervention
        outcome_samples = sample_with_intervention(scm, intervention, n_samples=1, seed=int(step_key[0]))
        
        if outcome_samples:
            outcome = outcome_samples[0]
            buffer.add_intervention(intervention, outcome)
            
            # Track target value progress (identical to regular demo)
            outcome_value = get_values(outcome)[target]
            best_so_far = min(best_so_far, outcome_value)  # Minimization
            target_progress.append(outcome_value)  # Track actual value, not cumulative best
            
            # Update surrogate model with data likelihood (identical to regular demo)
            all_samples = buffer.get_all_samples()
            if len(all_samples) >= 2:  # Need some data for learning
                # Get posterior before update
                current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
                
                # Update model parameters
                current_params, current_opt_state, (loss, param_norm, grad_norm, update_norm) = update_fn(
                    current_params, current_opt_state, current_state.posterior, [outcome], variables, target
                )
                
                # Get posterior after update
                updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
                
                # Track metrics
                uncertainty_progress.append(updated_state.uncertainty_bits)
                marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
                
                # Data likelihood
                likelihood = compute_data_likelihood_from_posterior(updated_state.posterior, all_samples, target)
                data_likelihood_progress.append(likelihood)
            else:
                # Not enough data for meaningful learning yet
                loss = 0.0
                param_norm = 0.0
                grad_norm = 0.0
                update_norm = 0.0
                uncertainty_progress.append(float('inf'))
                marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
                data_likelihood_progress.append(0.0)
        else:
            # Failed to sample - record defaults
            outcome_value = 0.0
            target_progress.append(outcome_value)  # Track actual value even for failed samples
            loss = 0.0
            param_norm = 0.0
            grad_norm = 0.0
            update_norm = 0.0
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
        
        # Record learning step (identical to regular demo)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': uncertainty_progress[-1],
                'marginals': marginal_prob_progress[-1],
                'data_likelihood': data_likelihood_progress[-1]
            })
        else:
            # Minimal logging for early steps
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0,
                'f1_score': 0.0  # No meaningful F1 score yet
            })
    
    # Final analysis (identical to regular demo)
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, current_params)
    
    # Count intervention types
    intervention_counts = {}
    for step_info in learning_history:
        intervention = step_info['intervention']
        if intervention and intervention.get('values'):
            vars_intervened = list(intervention['values'].keys())
            for var in vars_intervened:
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
        else:
            intervention_counts['observational'] = intervention_counts.get('observational', 0) + 1
    
    # Get true parents based on SCM structure
    true_parents = _get_true_parents_for_scm(scm, target)
    
    return {
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'initial_mean': target_progress[0],
        'intervention_mean': jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0],
        'reduction': target_progress[0] - (jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0]),  # Positive reduction is good
        'total_samples': buffer.size(),
        'intervention_counts': intervention_counts,
        
        # Learning progress metrics
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        
        # Final state
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_state.marginal_parent_probs,
        'converged_to_truth': analyze_convergence(marginal_prob_progress, true_parents),
        
        # Mark as oracle method
        'method': 'oracle_learning_surrogate'
    }


def run_structure_guided_learning_demo_with_scm(scm: pyr.PMap, config: DemoConfig) -> Dict[str, Any]:
    """
    Run structure-guided learning demo with a specific SCM.
    
    This uses the structure-aware intervention policy that:
    1. Learns causal structure through progressive learning
    2. Uses learned marginal probabilities to guide intervention selection
    3. Combines the benefits of learning with intelligent intervention targeting
    
    This replaces the misleading "Random + Learning" method with proper
    structure-guided acquisition.
    """
    print(f"🎯 Structure-Guided Learning Demo")
    print("=" * 60)
    
    # Setup
    key = random.PRNGKey(config.random_seed)
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    
    # Create learnable surrogate model and structure-aware intervention policy
    key, surrogate_key = random.split(key)
    surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, surrogate_key, config.learning_rate, config.scoring_method
    )
    intervention_fn = create_structure_aware_intervention_policy(
        variables, target, config.intervention_value_range, temperature=1.0
    )
    
    # Generate initial data
    key, data_key = random.split(key)
    initial_samples, buffer = generate_initial_data(scm, config, data_key)
    
    # Track learning progress
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Initial progress tracking (minimization)
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        initial_mean = jnp.mean(jnp.array(initial_values))  # Use mean instead of min
        best_so_far = min(initial_values)  # Still track best for other purposes
    else:
        initial_mean = 0.0  # No initial data
        best_so_far = float('inf')
    target_progress.append(initial_mean)  # Store mean as first entry
    
    # Get initial posterior and metrics (only if we have samples)
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
        
        # Initial data likelihood
        initial_likelihood = compute_data_likelihood_from_posterior(current_state.posterior, initial_samples, target)
        data_likelihood_progress.append(initial_likelihood)
    else:
        # No initial data - start with high uncertainty
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        data_likelihood_progress.append(0.0)
    
    # Run intervention and learning loop
    keys = random.split(key, config.n_intervention_steps)
    current_params = params
    current_opt_state = opt_state
    
    for step in range(config.n_intervention_steps):
        # Get current posterior with latest parameters (only if we have samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 1:
            current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
        else:
            current_state = None  # No samples yet for meaningful state
        
        # Execute structure-aware intervention (pass state for marginal probabilities)
        intervention = intervention_fn(state=current_state, key=keys[step])
        _, outcome_key = random.split(keys[step])
        
        if intervention and intervention.get('values'):
            # Apply intervention
            outcome = sample_with_intervention(scm, intervention, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_intervention(intervention, outcome)
        else:
            # Observational fallback
            outcome = sample_from_linear_scm(scm, n_samples=1, seed=int(outcome_key[0]))[0]
            buffer.add_observation(outcome)
        
        # Update model with more samples for better learning signal
        all_samples = buffer.get_all_samples()
        new_samples = all_samples[-15:] if len(all_samples) >= 15 else all_samples
        
        # Try to update model parameters using self-supervised signal (only if we have enough samples)
        if len(all_samples) >= 5 and current_state is not None:
            try:
                current_params, current_opt_state, (loss, param_norm, grad_norm, update_norm) = update_fn(
                    current_params, current_opt_state, current_state.posterior, 
                    new_samples, variables, target
                )
            except Exception:
                # If update fails, continue with current parameters
                loss, param_norm, grad_norm, update_norm = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            # Not enough samples yet for meaningful updates
            loss, param_norm, grad_norm, update_norm = 0.0, 0.0, 0.0, 0.0
        
        # Track progress
        outcome_value = get_values(outcome)[target]
        best_so_far = min(best_so_far, outcome_value) if best_so_far != float('inf') else outcome_value
        target_progress.append(outcome_value)  # Track actual value, not cumulative best
        
        # Get updated state for metrics (only if we have enough samples)
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step+1, current_params)
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
            
            # Compute data likelihood progress
            likelihood = compute_data_likelihood_from_posterior(updated_state.posterior, all_samples, target)
            data_likelihood_progress.append(likelihood)
        else:
            # Not enough samples for meaningful metrics yet
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
        
        # Store step info (only if we have meaningful metrics)
        if len(all_samples) >= 5:
            # Compute F1 score from marginal probabilities
            f1_score = compute_f1_score_from_marginals(
                marginal_prob_progress[-1],
                true_parents,
                threshold=0.5
            )
            
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': uncertainty_progress[-1],
                'marginals': marginal_prob_progress[-1],
                'data_likelihood': data_likelihood_progress[-1],
                'f1_score': f1_score
            })
        else:
            # Minimal logging for early steps
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0,
                'f1_score': 0.0  # No meaningful F1 score yet
            })
    
    # Final analysis
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, current_params)
    
    # Count intervention types
    intervention_counts = {}
    for step_info in learning_history:
        intervention = step_info['intervention']
        if intervention and intervention.get('values'):
            vars_intervened = list(intervention['values'].keys())
            for var in vars_intervened:
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
        else:
            intervention_counts['observational'] = intervention_counts.get('observational', 0) + 1
    
    # Get true parents based on SCM structure
    true_parents = _get_true_parents_for_scm(scm, target)
    
    return {
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'initial_mean': target_progress[0],
        'intervention_mean': jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0],
        'reduction': target_progress[0] - (jnp.mean(jnp.array(target_progress[1:])) if len(target_progress) > 1 else target_progress[0]),  # Positive reduction is good
        'total_samples': buffer.size(),
        'intervention_counts': intervention_counts,
        
        # Learning progress metrics
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        
        # Final state
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_state.marginal_parent_probs,
        'converged_to_truth': analyze_convergence(marginal_prob_progress, true_parents),
        
        # Mark as structure-guided method
        'method': 'structure_guided_learning'
    }


def run_zero_obs_random_intervention_test(config: Optional[DemoConfig] = None) -> Dict[str, Any]:
    """Test random interventions with zero observational data."""
    if config is None:
        config = DemoConfig(
            n_observational_samples=0,
            n_intervention_steps=12,
            learning_rate=1e-3,
            random_seed=42
        )
    
    print("🎲 Zero Observational Data + Random Interventions Test")
    print("=" * 60)
    print("Testing structure learning with ONLY diverse interventions (no observational data)")
    
    scm = create_easy_scm()
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    true_parents = _get_true_parents_for_scm(scm, target)
    
    print(f"📊 Test Setup:")
    print(f"   SCM: {variables}")
    print(f"   Target: {target}")
    print(f"   True parents: {true_parents}")
    print(f"   Observational samples: {config.n_observational_samples}")
    print(f"   Intervention steps: {config.n_intervention_steps}")
    print(f"   Expected: Should learn structure through diverse interventions")
    
    # Run the test
    results = run_progressive_learning_demo_with_scm(scm, config)
    
    # Print results
    converged = results['converged_to_truth']['converged']
    accuracy = results['converged_to_truth']['final_accuracy']
    true_parent_probs = results['converged_to_truth']['true_parent_probs']
    
    print(f"\n📈 Random Intervention Results (Zero Obs Data):")
    print(f"   Converged to truth: {converged}")
    print(f"   Final accuracy: {accuracy:.3f}")
    print(f"   True parent probs: {true_parent_probs}")
    print(f"   Final uncertainty: {results['final_uncertainty']:.2f} bits")
    print(f"   Total samples used: {results['total_samples']} (all interventional)")
    
    return results


def run_zero_obs_fixed_intervention_test(config: Optional[DemoConfig] = None) -> Dict[str, Any]:
    """Test fixed interventions on disconnected variable with zero observational data."""
    if config is None:
        config = DemoConfig(
            n_observational_samples=0,
            n_intervention_steps=12,
            learning_rate=1e-3,
            random_seed=42
        )
    
    print("🔒 Zero Observational Data + Fixed Interventions Test")
    print("=" * 60)
    print("Testing structure learning with ONLY fixed interventions (no observational data)")
    
    # Use SCM with disconnected variable for negative control
    scm = create_easy_scm_with_disconnected_var()
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    true_parents = _get_true_parents_for_scm(scm, target)
    
    # Choose disconnected variable X for fixed intervention
    fixed_variable = 'X'
    fixed_value = 1.0
    
    print(f"📊 Test Setup:")
    print(f"   SCM: {variables} (X is disconnected)")
    print(f"   Structure: A → B → D ← C ← E, X (isolated)")
    print(f"   Target: {target}")
    print(f"   True parents: {true_parents}")
    print(f"   Observational samples: {config.n_observational_samples}")
    print(f"   Intervention steps: {config.n_intervention_steps}")
    print(f"   Fixed intervention: always do({fixed_variable} = {fixed_value})")
    print(f"   Expected: Should NOT learn structure - X provides ZERO causal information")
    
    # Create fixed intervention policy
    key = random.PRNGKey(config.random_seed)
    key, surrogate_key = random.split(key)
    surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, surrogate_key, config.learning_rate, config.scoring_method
    )
    intervention_fn = create_fixed_intervention_policy(variables, target, fixed_variable, fixed_value)
    
    # Generate initial data (should be empty)
    key, data_key = random.split(key)
    initial_samples, buffer = generate_initial_data(scm, config, data_key)
    
    # Run the same learning loop but with fixed interventions
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Initial progress tracking
    if initial_samples:
        initial_values = [get_values(s)[target] for s in initial_samples]
        best_so_far = max(initial_values)
    else:
        best_so_far = float('-inf')
    target_progress.append(best_so_far)
    
    # Initial metrics
    if initial_samples:
        current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, 0, params)
        uncertainty_progress.append(current_state.uncertainty_bits)
        marginal_prob_progress.append(dict(current_state.marginal_parent_probs))
        initial_likelihood = compute_data_likelihood_from_posterior(current_state.posterior, initial_samples, target)
        data_likelihood_progress.append(initial_likelihood)
    else:
        uncertainty_progress.append(float('inf'))
        marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
        data_likelihood_progress.append(0.0)
    
    # Run fixed intervention loop
    keys = random.split(key, config.n_intervention_steps)
    current_params = params
    current_opt_state = opt_state
    
    for step in range(config.n_intervention_steps):
        # Get current state
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 1:
            current_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step, current_params)
        else:
            current_state = None
        
        # Apply fixed intervention
        intervention = intervention_fn()
        _, outcome_key = random.split(keys[step])
        outcome = sample_with_intervention(scm, intervention, n_samples=1, seed=int(outcome_key[0]))[0]
        buffer.add_intervention(intervention, outcome)
        
        # Update model parameters
        all_samples = buffer.get_all_samples()
        new_samples = all_samples[-15:] if len(all_samples) >= 15 else all_samples
        
        if len(all_samples) >= 5 and current_state is not None:
            try:
                current_params, current_opt_state, (loss, param_norm, grad_norm, update_norm) = update_fn(
                    current_params, current_opt_state, current_state.posterior, 
                    new_samples, variables, target
                )
            except Exception:
                loss, param_norm, grad_norm, update_norm = float('nan'), float('nan'), float('nan'), float('nan')
        else:
            loss, param_norm, grad_norm, update_norm = 0.0, 0.0, 0.0, 0.0
        
        # Track progress
        outcome_value = get_values(outcome)[target]
        best_so_far = min(best_so_far, outcome_value) if best_so_far != float('inf') else outcome_value
        target_progress.append(outcome_value)  # Track actual value, not cumulative best
        
        # Update metrics
        all_samples = buffer.get_all_samples()
        if len(all_samples) >= 5:
            updated_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, step+1, current_params)
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(dict(updated_state.marginal_parent_probs))
            likelihood = compute_data_likelihood_from_posterior(updated_state.posterior, all_samples, target)
            data_likelihood_progress.append(likelihood)
            
            # Compute F1 score from marginal probabilities
            f1_score = compute_f1_score_from_marginals(
                dict(updated_state.marginal_parent_probs),
                true_parents,
                threshold=0.5
            )
            
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': updated_state.uncertainty_bits,
                'marginals': dict(updated_state.marginal_parent_probs),
                'data_likelihood': likelihood,
                'f1_score': f1_score
            })
        else:
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
            
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'loss': loss,
                'param_norm': param_norm,
                'grad_norm': grad_norm,
                'update_norm': update_norm,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0
            })
    
    # Final analysis
    final_state = create_acquisition_state_from_buffer(buffer, surrogate_fn, variables, target, config.n_intervention_steps, current_params)
    final_marginal_probs = final_state.marginal_parent_probs
    final_uncertainty = final_state.uncertainty_bits
    
    # Analyze convergence
    convergence_analysis = analyze_convergence(marginal_prob_progress, true_parents)
    
    # Count intervention types (should all be the same fixed intervention)
    intervention_counts = {}
    for step_info in learning_history:
        intervention = step_info['intervention']
        if intervention and intervention.get('values'):
            vars_intervened = list(intervention['values'].keys())
            for var in vars_intervened:
                intervention_counts[var] = intervention_counts.get(var, 0) + 1
        else:
            intervention_counts['observational'] = intervention_counts.get('observational', 0) + 1
    
    print(f"\n📈 Fixed Intervention Results (Zero Obs Data):")
    print(f"   Fixed variable used: {fixed_variable} (intervened {intervention_counts.get(fixed_variable, 0)} times)")
    print(f"   Converged to truth: {convergence_analysis['converged']} (should be False)")
    print(f"   Final accuracy: {convergence_analysis['final_accuracy']:.3f} (should be low)")
    print(f"   Final uncertainty: {final_uncertainty:.2f} bits (should be high)")
    print(f"   Total samples used: {len(buffer.get_all_samples())} (all interventional)")
    
    # DEBUG: Print detailed results
    print(f"\n🔍 DETAILED DEBUG INFO:")
    print(f"   Final marginal probs: {final_marginal_probs}")
    print(f"   True parents: {true_parents}")
    print(f"   Convergence details: {convergence_analysis}")
    
    # DEBUG: Print intervention history
    print(f"\n🔍 INTERVENTION HISTORY:")
    for i, step_info in enumerate(learning_history[-5:]):  # Last 5 steps
        step_num = step_info['step']
        intervention = step_info['intervention']
        marginals = step_info['marginals']
        print(f"   Step {step_num}: {intervention} -> marginals: {marginals}")
    
    return {
        'test_type': 'zero_obs_fixed_interventions',
        'fixed_variable': fixed_variable,
        'fixed_value': fixed_value,
        'target_variable': target,
        'true_parents': true_parents,
        'config': config,
        'observational_samples_used': 0,
        'learning_from_interventions_only': True,
        
        # Results (should show failure to identify structure)
        'converged_to_truth': convergence_analysis,
        'final_uncertainty': final_uncertainty,
        'final_marginal_probs': final_marginal_probs,
        
        # Progress tracking
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        
        # Intervention tracking
        'intervention_counts': intervention_counts,
        'total_samples': len(buffer.get_all_samples()),
        'initial_best': target_progress[0] if target_progress else float('-inf'),
        'final_best': target_progress[-1] if target_progress else float('-inf'),
        'improvement': (target_progress[-1] - target_progress[0]) if len(target_progress) > 1 else 0.0
    }


def run_difficulty_comparative_study(config: Optional[DemoConfig] = None) -> Dict[str, Any]:
    """
    Run comparative study across 3 difficulty levels to validate performance degradation.
    
    Tests the same progressive learning protocol on easy, medium, and hard SCMs
    to demonstrate that performance appropriately degrades with complexity.
    
    Args:
        config: Demo configuration (uses defaults if None)
        
    Returns:
        Dictionary with comparative results across all difficulty levels
    """
    if config is None:
        config = DemoConfig(
            n_observational_samples=15,  # Reduced for testing
            n_intervention_steps=8,   # Reduced for testing
            learning_rate=1e-3,
            random_seed=42
        )
    
    print("🔬 Running Difficulty Comparative Study")
    print("=" * 60)
    
    # Define difficulty levels with their configurations
    difficulty_levels = [
        {
            'name': 'Easy',
            'scm_fn': create_easy_scm,
            'description': '5 vars, strong signal (coeff: 1.2, 0.8), low noise (0.5)',
            'expected_difficulty': 'Should converge quickly'
        },
        {
            'name': 'Medium', 
            'scm_fn': create_medium_scm,
            'description': '8 vars, moderate signal (coeff: 0.7, 0.6), ancestral correlation',
            'expected_difficulty': 'Moderate convergence with some confusion'
        },
        {
            'name': 'Hard',
            'scm_fn': create_hard_scm,
            'description': '10 vars, weak signal (coeff: 0.3, 0.4, 0.3), high noise (2.0)',
            'expected_difficulty': 'Difficult convergence due to weak signal and confounding'
        }
    ]
    
    all_results = {}
    
    for level_info in difficulty_levels:
        level_name = level_info['name']
        scm_fn = level_info['scm_fn']
        
        print(f"\n📊 Testing {level_name} Level:")
        print(f"   {level_info['description']}")
        print(f"   {level_info['expected_difficulty']}")
        print("-" * 40)
        
        # Create SCM and run test
        scm = scm_fn()
        results = run_progressive_learning_demo_with_scm(scm, config)
        
        # Print results
        converged = results['converged_to_truth']['converged']
        accuracy = results['converged_to_truth']['final_accuracy']
        uncertainty = results['final_uncertainty']
        
        print(f"   Result: {'✅ Converged' if converged else '❌ Did not converge'}")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Uncertainty: {uncertainty:.2f} bits")
        print(f"   Target improvement: {results['improvement']:.3f}")
        
        all_results[level_name.lower()] = results
    
    # Compare results across difficulty levels
    print(f"\n🔍 Comparative Analysis:")
    print(f"{'Level':<10} {'Converged':<12} {'Accuracy':<10} {'Uncertainty':<12} {'Improvement'}")
    print("-" * 60)
    
    for level_name in ['Easy', 'Medium', 'Hard']:
        results = all_results[level_name.lower()]
        converged = '✅' if results['converged_to_truth']['converged'] else '❌'
        accuracy = results['converged_to_truth']['final_accuracy']
        uncertainty = results['final_uncertainty']
        improvement = results['improvement']
        
        print(f"{level_name:<10} {converged:<12} {accuracy:<10.3f} {uncertainty:<12.2f} {improvement:<10.3f}")
    
    return {
        'config': config,
        'difficulty_levels': difficulty_levels,
        'results': all_results,
        'summary': {
            'easy_converged': all_results['easy']['converged_to_truth']['converged'],
            'medium_converged': all_results['medium']['converged_to_truth']['converged'],
            'hard_converged': all_results['hard']['converged_to_truth']['converged'],
            'shows_degradation': (
                all_results['easy']['converged_to_truth']['final_accuracy'] >= 
                all_results['medium']['converged_to_truth']['final_accuracy'] >= 
                all_results['hard']['converged_to_truth']['final_accuracy']
            )
        }
    }


# Main functions for external use
def run_progressive_learning_demo(config: Optional[DemoConfig] = None) -> Dict[str, Any]:
    """Run progressive learning demo with Easy SCM."""
    if config is None:
        config = DemoConfig()
    
    scm = create_easy_scm()
    return run_progressive_learning_demo_with_scm(scm, config)


def compare_zero_obs_intervention_tests(random_results: Dict[str, Any], fixed_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare zero observational data tests."""
    return compare_intervention_tests(random_results, fixed_results)


# Example usage and testing
if __name__ == "__main__":
    # Test the refactored demo
    print("🧪 Testing Refactored ACBO Demo")
    print("=" * 50)
    
    # Quick test with reduced steps
    test_config = DemoConfig(
        n_observational_samples=10,
        n_intervention_steps=5,
        learning_rate=1e-3,
        random_seed=42
    )
    
    # Test basic progressive learning
    print("\n1. Testing basic progressive learning...")
    results = run_progressive_learning_demo(test_config)
    print_final_summary(results, "Basic Progressive Learning")
    
    # Test difficulty comparison
    print("\n2. Testing difficulty comparison...")
    difficulty_results = run_difficulty_comparative_study(test_config)
    print(f"Difficulty degradation observed: {difficulty_results['summary']['shows_degradation']}")
    
    print("\n✅ Refactored demo testing complete!")
    
    # Test visualization integration
    print("\n3. Testing visualization integration...")
    try:
        from demo_evaluation import extract_trajectory_metrics_from_demo
        from causal_bayes_opt.visualization.plots import plot_convergence, plot_structure_learning_dashboard
        
        # Extract trajectory metrics
        trajectory_metrics = extract_trajectory_metrics_from_demo(results)
        
        # Create simple plots (without saving to avoid cluttering)
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        # Generate convergence plot
        fig1 = plot_convergence(trajectory_metrics, title="Demo Convergence")
        plt.close(fig1)
        
        # Generate dashboard
        fig2 = plot_structure_learning_dashboard(trajectory_metrics, title="Demo Dashboard")
        plt.close(fig2)
        
        print("   ✅ Visualization integration working!")
        
    except Exception as e:
        print(f"   ⚠️  Visualization test failed: {e}")
    
    print("\n✅ All demo tests complete!")