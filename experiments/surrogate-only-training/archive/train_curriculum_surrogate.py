#!/usr/bin/env python3
"""Train surrogate model with curriculum learning on progressively harder SCMs."""

import sys
import os
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.policies.permutation_invariant_alternating_policy import create_permutation_invariant_alternating_policy
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper


def create_curriculum() -> List[Dict]:
    """Create curriculum of SCMs from easy to hard."""
    curriculum = [
        # Warm-up (3-5 variables)
        {'num_vars': 3, 'structure': 'fork'},
        {'num_vars': 3, 'structure': 'chain'},
        {'num_vars': 4, 'structure': 'collider'},
        {'num_vars': 5, 'structure': 'mixed'},
        
        # Medium (6-10 variables)
        {'num_vars': 6, 'structure': 'fork'},
        {'num_vars': 8, 'structure': 'chain'},
        {'num_vars': 10, 'structure': 'collider'},
        
        # Large (12-20 variables)
        {'num_vars': 12, 'structure': 'mixed'},
        {'num_vars': 15, 'structure': 'random', 'edge_density': 0.3},
        {'num_vars': 20, 'structure': 'fork'},
        
        # Very Large (25-50 variables) - Test the limits!
        {'num_vars': 25, 'structure': 'chain'},
        {'num_vars': 30, 'structure': 'mixed'},
        {'num_vars': 40, 'structure': 'random', 'edge_density': 0.2},
        {'num_vars': 50, 'structure': 'fork'},
    ]
    return curriculum


def initialize_models(config: Dict, key: jax.random.PRNGKey, max_vars: int) -> Tuple:
    """Initialize policy and surrogate models for maximum variable count."""
    # Create dummy data for initialization
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    # Initialize policy
    policy_key, surrogate_key = random.split(key)
    policy_fn = create_permutation_invariant_alternating_policy(config['hidden_dim'])
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    policy_params = policy_net.init(policy_key, dummy_tensor, 0)
    
    # Initialize surrogate
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=config['hidden_dim'],
            num_heads=4,
            num_layers=4,
            dropout=0.1
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    surrogate_params = surrogate_net.init(surrogate_key, dummy_tensor, 0, True)
    
    # Initialize optimizer
    surrogate_optimizer = optax.adam(config['learning_rate'])
    surrogate_opt_state = surrogate_optimizer.init(surrogate_params)
    
    return policy_net, policy_params, surrogate_net, surrogate_params, surrogate_optimizer, surrogate_opt_state


def compute_surrogate_loss(params, net, buffer, target_idx, target_var, true_parents, variables, rng_key):
    """Compute BCE loss for surrogate predictions."""
    tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
    # Use RNG for surrogate network
    predictions = net.apply(params, rng_key, tensor, target_idx, True)
    
    if 'parent_probabilities' in predictions:
        pred_probs = predictions['parent_probabilities']
    else:
        raw_logits = predictions.get('attention_logits', jnp.zeros(len(variables)))
        pred_probs = jax.nn.sigmoid(raw_logits)
    
    pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
    
    # Create ground truth labels
    labels = []
    for i, var in enumerate(variables):
        if i != target_idx:
            label = 1.0 if var in true_parents else 0.0
            labels.append(label)
    
    labels = jnp.array(labels)
    mask = jnp.ones(len(variables), dtype=bool).at[target_idx].set(False)
    pred_probs_filtered = pred_probs[mask]
    
    # BCE loss
    loss = -jnp.mean(
        labels * jnp.log(pred_probs_filtered) + 
        (1 - labels) * jnp.log(1 - pred_probs_filtered)
    )
    
    return loss, pred_probs


def evaluate_predictions(pred_probs, true_parents, variables, target_idx, threshold=0.5):
    """Evaluate surrogate predictions and compute F1 score."""
    predicted_parents = set()
    for i, var in enumerate(variables):
        if i != target_idx and pred_probs[i] > threshold:
            predicted_parents.add(var)
    
    true_parent_set = set(true_parents)
    
    # Handle empty parent sets correctly
    if len(true_parent_set) == 0 and len(predicted_parents) == 0:
        return 1.0  # Perfect F1 for correctly predicting no parents
    elif len(true_parent_set) == 0 or len(predicted_parents) == 0:
        return 0.0  # One is empty but not both
    
    tp = len(predicted_parents & true_parent_set)
    fp = len(predicted_parents - true_parent_set)
    fn = len(true_parent_set - predicted_parents)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def train_on_scm(scm, policy_net, policy_params, surrogate_net, surrogate_params,
                 surrogate_optimizer, surrogate_opt_state, config, rng_key):
    """Train surrogate on a single SCM until F1=1 or max interventions."""
    
    # Get SCM info
    mapper = VariableMapper(list(get_variables(scm)))
    variables = mapper.variables  # Sorted variables
    num_vars = len(variables)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    rng_key, obs_key = random.split(rng_key)
    obs_data = sample_from_linear_scm(scm, 20, seed=int(obs_key[0]))
    for sample in obs_data:
        buffer.add_observation(sample)
    
    interventions_count = 0
    f1_history = []
    
    for intervention_idx in range(config['max_interventions_per_scm']):
        interventions_count += 1
        
        # Progress indicator
        if intervention_idx == 0:
            print(f"    Starting training (max {config['max_interventions_per_scm']} interventions)...")
        
        # Select intervention using policy
        rng_key, target_key, select_key = random.split(rng_key, 3)
        target_idx = random.choice(target_key, num_vars)
        target_var = variables[int(target_idx)]
        
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        output = policy_net.apply(policy_params, tensor, int(target_idx))
        logits = output['variable_logits']
        
        # Use random selection for diversity
        valid_mask = jnp.ones(num_vars).at[int(target_idx)].set(0)
        valid_indices = jnp.where(valid_mask)[0]
        selected_idx = random.choice(select_key, valid_indices)
        selected_var = variables[int(selected_idx)]
        
        # Perform intervention
        rng_key, int_key, post_key = random.split(rng_key, 3)
        intervention_value = random.normal(int_key) * 2.0
        
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        post_data = sample_with_intervention(scm, intervention, 1, seed=int(post_key[0]))
        if post_data:
            buffer.add_intervention(intervention, post_data[0])
        
        # Train surrogate on all variables
        all_f1_scores = []
        
        for target_var in variables:
            target_idx = mapper.get_index(target_var)
            true_parents = list(get_parents(scm, target_var))
            
            # Multiple gradient steps
            for _ in range(config['gradient_steps_per_intervention']):
                rng_key, loss_key = random.split(rng_key)
                
                def loss_fn(params):
                    loss, pred_probs = compute_surrogate_loss(
                        params, surrogate_net, buffer, target_idx, target_var,
                        true_parents, variables, loss_key
                    )
                    return loss, pred_probs
                
                (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(surrogate_params)
                updates, surrogate_opt_state = surrogate_optimizer.update(grads, surrogate_opt_state, surrogate_params)
                surrogate_params = optax.apply_updates(surrogate_params, updates)
            
            # Evaluate F1 for this variable
            f1 = evaluate_predictions(pred_probs, true_parents, variables, target_idx)
            all_f1_scores.append(f1)
        
        # Average F1 across all variables
        avg_f1 = np.mean(all_f1_scores)
        f1_history.append(avg_f1)
        
        # Check if we've achieved target F1
        if avg_f1 >= config['target_f1']:
            print(f"    ✓ Achieved F1={avg_f1:.3f} after {interventions_count} interventions")
            return surrogate_params, surrogate_opt_state, True, interventions_count, f1_history
        
        # Log progress every intervention for debugging
        print(f"    Intervention {intervention_idx+1}: F1={avg_f1:.3f}")
    
    print(f"    ✗ Max interventions reached. Final F1={f1_history[-1]:.3f}")
    return surrogate_params, surrogate_opt_state, False, interventions_count, f1_history


def main():
    parser = argparse.ArgumentParser(description='Train surrogate with curriculum learning')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension for models')
    parser.add_argument('--lr', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--max-interventions', type=int, default=100,
                       help='Max interventions per SCM')
    parser.add_argument('--target-f1', type=float, default=1.0,
                       help='Target F1 score to move to next SCM')
    parser.add_argument('--gradient-steps', type=int, default=5,
                       help='Gradient steps per intervention')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CURRICULUM LEARNING FOR SURROGATE MODEL")
    print("="*70)
    
    # Configuration
    config = {
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.lr,
        'max_interventions_per_scm': args.max_interventions,
        'target_f1': args.target_f1,
        'gradient_steps_per_intervention': args.gradient_steps,
        'seed': args.seed
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize RNG
    rng_key = random.PRNGKey(config['seed'])
    
    # Create curriculum
    curriculum = create_curriculum()
    print(f"\nCurriculum: {len(curriculum)} SCMs")
    
    # Find max variables for model initialization
    max_vars = max(scm_cfg['num_vars'] for scm_cfg in curriculum)
    print(f"Maximum variables: {max_vars}")
    
    # Initialize models
    rng_key, init_key = random.split(rng_key)
    policy_net, policy_params, surrogate_net, surrogate_params, surrogate_optimizer, surrogate_opt_state = \
        initialize_models(config, init_key, max_vars)
    
    # Save initial model
    checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    initial_checkpoint = {
        'params': surrogate_params,
        'opt_state': surrogate_opt_state,
        'config': config,
        'trained_on': []
    }
    
    with open(checkpoint_dir / 'initial_model.pkl', 'wb') as f:
        pickle.dump(initial_checkpoint, f)
    print(f"\nSaved initial model to {checkpoint_dir / 'initial_model.pkl'}")
    
    # Training loop
    print("\n" + "-"*70)
    print("Starting Curriculum Training")
    print("-"*70)
    
    training_log = {
        'config': config,
        'curriculum': curriculum,
        'results': [],
        'total_interventions': 0,
        'completed_scms': 0
    }
    
    factory = VariableSCMFactory(seed=config['seed'])
    
    for i, scm_config in enumerate(curriculum):
        print(f"\nSCM {i+1}/{len(curriculum)}: {scm_config['structure']} with {scm_config['num_vars']} variables")
        
        # Create SCM
        scm = factory.create_variable_scm(
            num_variables=scm_config['num_vars'],
            structure_type=scm_config['structure'],
            edge_density=scm_config.get('edge_density', 0.5)
        )
        
        # Train on this SCM
        rng_key, train_key = random.split(rng_key)
        start_time = datetime.now()
        
        surrogate_params, surrogate_opt_state, success, interventions, f1_history = train_on_scm(
            scm, policy_net, policy_params, surrogate_net, surrogate_params,
            surrogate_optimizer, surrogate_opt_state, config, train_key
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log results
        result = {
            'scm_index': i,
            'scm_config': scm_config,
            'success': success,
            'interventions': interventions,
            'final_f1': f1_history[-1] if f1_history else 0,
            'f1_history': f1_history,
            'time_seconds': elapsed
        }
        training_log['results'].append(result)
        training_log['total_interventions'] += interventions
        
        if success:
            training_log['completed_scms'] += 1
        
        # Early stopping if struggling
        if not success and scm_config['num_vars'] > 20:
            print(f"\nStruggling with large SCMs. Stopping curriculum.")
            break
    
    # Save final model
    final_checkpoint = {
        'params': surrogate_params,
        'opt_state': surrogate_opt_state,
        'config': config,
        'trained_on': [r['scm_config'] for r in training_log['results'] if r['success']]
    }
    
    with open(checkpoint_dir / 'trained_model.pkl', 'wb') as f:
        pickle.dump(final_checkpoint, f)
    print(f"\nSaved trained model to {checkpoint_dir / 'trained_model.pkl'}")
    
    # Save training log
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = results_dir / f'curriculum_training_{timestamp}.json'
    
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"Saved training log to {log_path}")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Completed SCMs: {training_log['completed_scms']}/{len(curriculum)}")
    print(f"Total interventions: {training_log['total_interventions']}")
    
    if training_log['results']:
        successful = [r for r in training_log['results'] if r['success']]
        if successful:
            avg_interventions = np.mean([r['interventions'] for r in successful])
            print(f"Average interventions per success: {avg_interventions:.1f}")
            
            largest_solved = max(successful, key=lambda r: r['scm_config']['num_vars'])
            print(f"Largest SCM solved: {largest_solved['scm_config']['num_vars']} variables")


if __name__ == '__main__':
    main()