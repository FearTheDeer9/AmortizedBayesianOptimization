"""
Proper implementation of enriched policy with progressive learning.

This integrates the trained enriched policy with actual surrogate model updates
to track structure learning progress properly.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from causal_bayes_opt.data_structures.scm import get_variables, get_target, get_parents
from causal_bayes_opt.environments.sampling import sample_with_intervention
from causal_bayes_opt.data_structures.sample import create_sample, get_values
from causal_bayes_opt.acquisition.grpo_enriched_integration import (
    create_enriched_policy_intervention_function
)
from causal_bayes_opt.avici_integration.parent_set import (
    get_marginal_parent_probabilities
)

logger = logging.getLogger(__name__)


def run_enriched_policy_with_progressive_learning(
    scm: pyr.PMap, 
    acbo_config,
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Run enriched policy with actual progressive learning.
    
    This properly integrates:
    1. Trained enriched policy for intervention selection
    2. Progressive surrogate model learning with parameter updates
    3. Proper tracking of parent marginals for structure learning metrics
    
    Args:
        scm: Structural causal model
        acbo_config: Configuration for ACBO experiment
        checkpoint_path: Path to trained policy checkpoint
        
    Returns:
        Complete results with learning history including parent marginals
    """
    # Import required modules
    from examples.demo_learning import (
        create_learnable_surrogate_model,
        create_acquisition_state_from_buffer,
        convert_samples_for_prediction
    )
    from examples.complete_workflow_demo import generate_initial_data
    
    # Get SCM structure
    variables = sorted(get_variables(scm))
    target = get_target(scm)
    true_parents = list(get_parents(scm, target)) if target else []
    
    # Initialize
    key = jax.random.PRNGKey(acbo_config.random_seed)
    key, subkey = random.split(key)
    
    # Generate initial observational data
    initial_samples, buffer = generate_initial_data(scm, acbo_config, subkey)
    
    # Create learnable surrogate model
    key, subkey = random.split(key)
    surrogate_fn, _net, params, opt_state, update_fn = create_learnable_surrogate_model(
        variables, subkey, acbo_config.learning_rate, acbo_config.scoring_method
    )
    current_params = params
    
    # Create enriched policy function
    try:
        enriched_policy_fn = create_enriched_policy_intervention_function(
            checkpoint_path=checkpoint_path,
            intervention_value_range=acbo_config.intervention_value_range
        )
        logger.info(f"Successfully loaded enriched policy from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load enriched policy: {e}")
        raise
    
    # Initialize tracking
    learning_history = []
    target_progress = []
    uncertainty_progress = []
    marginal_prob_progress = []
    data_likelihood_progress = []
    
    # Handle initial target value (minimization)
    if initial_samples:
        target_values = [get_values(s).get(target, 0.0) for s in initial_samples]
        initial_target = min(target_values)  # Best (lowest) for minimization
        best_value = initial_target
    else:
        initial_target = float('inf')  # No data yet
        best_value = float('inf')
    
    # Progressive learning loop
    for step in range(acbo_config.n_intervention_steps):
        # Create acquisition state with current model
        state = create_acquisition_state_from_buffer(
            buffer, surrogate_fn, variables, target, step, current_params
        )
        
        # Get intervention from enriched policy
        key, subkey = random.split(key)
        try:
            intervention = enriched_policy_fn(state, scm, subkey)
        except Exception as e:
            logger.warning(f"Policy failed at step {step}: {e}")
            # Fallback to random intervention
            from causal_bayes_opt.acquisition.random_policy import create_random_policy
            random_policy = create_random_policy(
                variables, 
                target,
                acbo_config.intervention_value_range
            )
            intervention = random_policy(state, scm, subkey)
        
        # Apply intervention and get outcome
        key, subkey = random.split(key)
        outcome_samples = sample_with_intervention(
            scm, intervention, n_samples=1, seed=int(subkey[0])
        )
        outcome = outcome_samples[0]  # Already a properly formatted interventional sample
        
        # Add to buffer
        buffer.add_intervention(intervention, outcome)
        
        # Update best value (minimization)
        outcome_value = get_values(outcome)[target]
        if best_value == float('inf') or outcome_value < best_value:
            best_value = outcome_value
        
        target_improvement = initial_target - best_value  # Positive when we reduce
        
        # Update surrogate model if we have enough samples
        all_samples = buffer.get_all_samples()
        loss = 0.0
        param_norm = 0.0
        grad_norm = 0.0
        update_norm = 0.0
        
        if len(all_samples) >= 5:
            # Convert samples for training
            avici_data = convert_samples_for_prediction(all_samples, variables, target)
            
            # Compute loss and gradients
            (loss, outputs), grads = jax.value_and_grad(
                lambda p: apply_fn(p, avici_data, is_training=True), 
                has_aux=True
            )(current_params)
            
            # Update parameters
            updates, opt_state = update_fn(grads, opt_state)
            current_params = jax.tree_map(
                lambda p, u: p - acbo_config.learning_rate * u, 
                current_params, updates
            )
            
            # Track metrics
            param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_leaves(current_params)))
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(grads)))
            update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree_leaves(updates)))
            
            # Get updated state with new parameters
            updated_state = create_acquisition_state_from_buffer(
                buffer, predict_fn, variables, target, step+1, current_params
            )
            
            # Extract parent marginals (the key fix!)
            parent_marginals = dict(updated_state.marginal_parent_probs)
            
            # Track progress
            uncertainty_progress.append(updated_state.uncertainty_bits)
            marginal_prob_progress.append(parent_marginals)
            
            # Compute data likelihood if available
            likelihood = 0.0
            if hasattr(updated_state.posterior, 'data_likelihood'):
                likelihood = updated_state.posterior.data_likelihood
            data_likelihood_progress.append(likelihood)
            
            # Record step with proper marginals
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'target_improvement': target_improvement,
                'loss': float(loss),
                'param_norm': float(param_norm),
                'grad_norm': float(grad_norm),
                'update_norm': float(update_norm),
                'uncertainty': updated_state.uncertainty_bits,
                'marginals': parent_marginals,  # Parent marginals, not node selection!
                'data_likelihood': likelihood,
                'policy_type': 'enriched_grpo',
                'learning_enabled': True
            })
        else:
            # Early steps without enough data for learning
            learning_history.append({
                'step': step + 1,
                'intervention': intervention,
                'outcome_value': outcome_value,
                'target_improvement': target_improvement,
                'loss': 0.0,
                'param_norm': 0.0,
                'grad_norm': 0.0,
                'update_norm': 0.0,
                'uncertainty': float('inf'),
                'marginals': {v: 0.0 for v in variables if v != target},
                'data_likelihood': 0.0,
                'policy_type': 'enriched_grpo',
                'learning_enabled': False
            })
            
            uncertainty_progress.append(float('inf'))
            marginal_prob_progress.append({v: 0.0 for v in variables if v != target})
            data_likelihood_progress.append(0.0)
        
        target_progress.append(outcome_value)
    
    # Final analysis
    final_state = create_acquisition_state_from_buffer(
        buffer, surrogate_fn, variables, target, 
        acbo_config.n_intervention_steps, current_params
    )
    final_marginal_probs = dict(final_state.marginal_parent_probs)
    
    # Compute final metrics
    from scripts.core.acbo_comparison.structure_metrics_helper import (
        compute_f1_from_marginals,
        compute_parent_probability,
        compute_shd_from_marginals
    )
    
    final_f1, final_precision, final_recall = compute_f1_from_marginals(
        final_marginal_probs, true_parents, target
    )
    final_parent_prob = compute_parent_probability(final_marginal_probs, true_parents)
    final_shd = compute_shd_from_marginals(final_marginal_probs, true_parents, target)
    
    # Build trajectory arrays
    steps = list(range(len(learning_history)))
    f1_scores = []
    shd_values = []
    true_parent_likelihood = []
    
    for step_data in learning_history:
        step_marginals = step_data.get('marginals', {})
        if step_marginals and step_data.get('learning_enabled', False):
            f1, _, _ = compute_f1_from_marginals(step_marginals, true_parents, target)
            parent_prob = compute_parent_probability(step_marginals, true_parents)
            shd = compute_shd_from_marginals(step_marginals, true_parents, target)
        else:
            f1 = 0.0
            parent_prob = 0.0
            shd = len(true_parents)
        
        f1_scores.append(f1)
        true_parent_likelihood.append(parent_prob)
        shd_values.append(shd)
    
    # Return comprehensive results
    return {
        'method': 'learned_enriched_policy',
        'learning_history': learning_history,
        'target_progress': target_progress,
        'uncertainty_progress': uncertainty_progress,
        'marginal_prob_progress': marginal_prob_progress,
        'data_likelihood_progress': data_likelihood_progress,
        'initial_best': initial_target if initial_target != float('inf') else 0.0,
        'final_best': best_value if best_value != float('inf') else 0.0,
        'improvement': initial_target - best_value if initial_target != float('inf') else 0.0,  # Positive reduction is good
        'target_improvement': initial_target - best_value if initial_target != float('inf') else 0.0,  # Positive reduction is good
        'reduction': initial_target - best_value if initial_target != float('inf') else 0.0,  # Alias for clarity
        'total_samples': acbo_config.n_observational_samples + acbo_config.n_intervention_steps,
        'final_uncertainty': final_state.uncertainty_bits,
        'final_marginal_probs': final_marginal_probs,
        'structure_accuracy': final_f1,  # Use F1 as structure accuracy
        'converged_to_truth': final_f1 > 0.8,  # High F1 indicates convergence
        'policy_checkpoint_used': checkpoint_path,
        'detailed_results': {
            'learning_history': learning_history,
            'target_progress': target_progress,
            'f1_scores': f1_scores,
            'shd_values': shd_values,
            'true_parent_likelihood': true_parent_likelihood,
            'steps': steps,
            'final_f1': final_f1,
            'final_precision': final_precision,
            'final_recall': final_recall,
            'final_parent_prob': final_parent_prob,
            'final_shd': final_shd
        }
    }