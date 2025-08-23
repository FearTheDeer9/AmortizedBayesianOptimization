#!/usr/bin/env python3
"""Evaluate trained and initial surrogate models on unseen SCMs."""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint

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


def create_test_scms() -> List[Dict]:
    """Create test set of unseen SCMs."""
    test_scms = [
        # Small to medium (different from training)
        {'num_vars': 4, 'structure': 'chain'},
        {'num_vars': 5, 'structure': 'fork'},
        {'num_vars': 7, 'structure': 'collider'},
        
        # Large
        {'num_vars': 10, 'structure': 'mixed'},
        {'num_vars': 15, 'structure': 'chain'},
        {'num_vars': 20, 'structure': 'collider'},
        
        # Very large
        {'num_vars': 25, 'structure': 'mixed'},
        {'num_vars': 30, 'structure': 'random', 'edge_density': 0.25},
        {'num_vars': 35, 'structure': 'fork'},
        {'num_vars': 40, 'structure': 'chain'},
    ]
    return test_scms


def initialize_policy(config: Dict, key: jax.random.PRNGKey, max_vars: int):
    """Initialize policy model."""
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    policy_fn = create_permutation_invariant_alternating_policy(config['hidden_dim'])
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    policy_params = policy_net.init(key, dummy_tensor, 0)
    
    return policy_net, policy_params


def initialize_surrogate(config: Dict, key: jax.random.PRNGKey, max_vars: int):
    """Initialize surrogate model."""
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=config['hidden_dim'],
            num_heads=4,
            num_layers=4,
            dropout=0.1
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    surrogate_params = surrogate_net.init(key, dummy_tensor, 0, True)
    
    return surrogate_net, surrogate_params


def evaluate_predictions(surrogate_net, surrogate_params, buffer, scm, mapper, rng_key):
    """Evaluate surrogate predictions on all variables."""
    variables = mapper.variables
    all_f1_scores = []
    all_metrics = []
    
    for target_var in variables:
        target_idx = mapper.get_index(target_var)
        true_parents = list(get_parents(scm, target_var))
        
        # Get predictions
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        predictions = surrogate_net.apply(surrogate_params, rng_key, tensor, target_idx, False)
        
        if 'parent_probabilities' in predictions:
            pred_probs = predictions['parent_probabilities']
        else:
            raw_logits = predictions.get('attention_logits', jnp.zeros(len(variables)))
            pred_probs = jax.nn.sigmoid(raw_logits)
        
        # Compute F1
        predicted_parents = set()
        for i, var in enumerate(variables):
            if i != target_idx and pred_probs[i] > 0.5:
                predicted_parents.add(var)
        
        true_parent_set = set(true_parents)
        
        if len(true_parent_set) == 0 and len(predicted_parents) == 0:
            f1 = 1.0
        elif len(true_parent_set) == 0 or len(predicted_parents) == 0:
            f1 = 0.0
        else:
            tp = len(predicted_parents & true_parent_set)
            fp = len(predicted_parents - true_parent_set)
            fn = len(true_parent_set - predicted_parents)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        all_f1_scores.append(f1)
        all_metrics.append({
            'variable': target_var,
            'true_parents': list(true_parent_set),
            'predicted_parents': list(predicted_parents),
            'f1': f1
        })
    
    return np.mean(all_f1_scores), all_metrics


def evaluate_on_scm(scm, policy_net, policy_params, surrogate_net, surrogate_params,
                   num_interventions, rng_key):
    """Evaluate model on a single SCM with fixed interventions."""
    
    # Get SCM info
    mapper = VariableMapper(list(get_variables(scm)))
    variables = mapper.variables
    num_vars = len(variables)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    rng_key, obs_key = random.split(rng_key)
    obs_data = sample_from_linear_scm(scm, 20, seed=int(obs_key[0]))
    for sample in obs_data:
        buffer.add_observation(sample)
    
    f1_trajectory = []
    
    for intervention_idx in range(num_interventions):
        # Select intervention using policy
        rng_key, target_key, select_key = random.split(rng_key, 3)
        target_idx = random.choice(target_key, num_vars)
        target_var = variables[int(target_idx)]
        
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        output = policy_net.apply(policy_params, tensor, int(target_idx))
        logits = output['variable_logits']
        
        # Random selection for diversity
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
        
        # Evaluate predictions (no gradient updates!)
        rng_key, eval_key = random.split(rng_key)
        avg_f1, _ = evaluate_predictions(surrogate_net, surrogate_params, buffer, scm, mapper, eval_key)
        f1_trajectory.append(avg_f1)
    
    return f1_trajectory


def main():
    parser = argparse.ArgumentParser(description='Evaluate surrogate models')
    parser.add_argument('--interventions', type=int, default=10,
                       help='Number of interventions per SCM')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (different from training)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SURROGATE MODEL EVALUATION")
    print("="*70)
    
    # Load checkpoints
    checkpoint_dir = Path(__file__).parent.parent / 'checkpoints'
    
    print("\nLoading models...")
    initial_checkpoint = load_checkpoint(checkpoint_dir / 'initial_model.pkl')
    trained_checkpoint = load_checkpoint(checkpoint_dir / 'trained_model.pkl')
    
    config = trained_checkpoint['config']
    print(f"Model config: hidden_dim={config['hidden_dim']}")
    print(f"Trained on {len(trained_checkpoint['trained_on'])} SCMs")
    
    # Initialize RNG
    rng_key = random.PRNGKey(args.seed)
    
    # Create test SCMs
    test_scms_config = create_test_scms()
    print(f"\nTest set: {len(test_scms_config)} SCMs")
    
    # Find max variables
    max_vars = max(scm_cfg['num_vars'] for scm_cfg in test_scms_config)
    
    # Initialize models
    rng_key, policy_key, surrogate_key = random.split(rng_key, 3)
    policy_net, policy_params = initialize_policy(config, policy_key, max_vars)
    surrogate_net, _ = initialize_surrogate(config, surrogate_key, max_vars)
    
    # Evaluation results
    results = {
        'config': {
            'interventions_per_scm': args.interventions,
            'seed': args.seed
        },
        'test_scms': test_scms_config,
        'initial_model': [],
        'trained_model': []
    }
    
    factory = VariableSCMFactory(seed=args.seed)
    
    print("\n" + "-"*70)
    print("Evaluating Models")
    print("-"*70)
    
    for i, scm_config in enumerate(test_scms_config):
        print(f"\nSCM {i+1}/{len(test_scms_config)}: {scm_config['structure']} with {scm_config['num_vars']} variables")
        
        # Create SCM
        scm = factory.create_variable_scm(
            num_variables=scm_config['num_vars'],
            structure_type=scm_config['structure'],
            edge_density=scm_config.get('edge_density', 0.5)
        )
        
        # Evaluate initial model
        rng_key, eval_key = random.split(rng_key)
        initial_f1_trajectory = evaluate_on_scm(
            scm, policy_net, policy_params, surrogate_net,
            initial_checkpoint['params'], args.interventions, eval_key
        )
        
        # Evaluate trained model
        rng_key, eval_key = random.split(rng_key)
        trained_f1_trajectory = evaluate_on_scm(
            scm, policy_net, policy_params, surrogate_net,
            trained_checkpoint['params'], args.interventions, eval_key
        )
        
        # Store results
        results['initial_model'].append({
            'scm_config': scm_config,
            'f1_trajectory': initial_f1_trajectory,
            'final_f1': initial_f1_trajectory[-1],
            'max_f1': max(initial_f1_trajectory)
        })
        
        results['trained_model'].append({
            'scm_config': scm_config,
            'f1_trajectory': trained_f1_trajectory,
            'final_f1': trained_f1_trajectory[-1],
            'max_f1': max(trained_f1_trajectory)
        })
        
        print(f"  Initial model: F1={initial_f1_trajectory[-1]:.3f} (max={max(initial_f1_trajectory):.3f})")
        print(f"  Trained model: F1={trained_f1_trajectory[-1]:.3f} (max={max(trained_f1_trajectory):.3f})")
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'evaluation_results_{timestamp}.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Initial model stats
    initial_final_f1s = [r['final_f1'] for r in results['initial_model']]
    initial_max_f1s = [r['max_f1'] for r in results['initial_model']]
    
    print("\nInitial Model:")
    print(f"  Average final F1: {np.mean(initial_final_f1s):.3f} ± {np.std(initial_final_f1s):.3f}")
    print(f"  Average max F1: {np.mean(initial_max_f1s):.3f} ± {np.std(initial_max_f1s):.3f}")
    
    # Trained model stats
    trained_final_f1s = [r['final_f1'] for r in results['trained_model']]
    trained_max_f1s = [r['max_f1'] for r in results['trained_model']]
    
    print("\nTrained Model:")
    print(f"  Average final F1: {np.mean(trained_final_f1s):.3f} ± {np.std(trained_final_f1s):.3f}")
    print(f"  Average max F1: {np.mean(trained_max_f1s):.3f} ± {np.std(trained_max_f1s):.3f}")
    
    # Improvement
    improvement = np.mean(trained_final_f1s) - np.mean(initial_final_f1s)
    print(f"\nImprovement: {improvement:+.3f} F1 points")
    
    # Performance by SCM size
    print("\nPerformance by SCM size (trained model):")
    size_groups = {}
    for result in results['trained_model']:
        size = result['scm_config']['num_vars']
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(result['final_f1'])
    
    for size in sorted(size_groups.keys()):
        avg_f1 = np.mean(size_groups[size])
        print(f"  {size} variables: F1={avg_f1:.3f}")


if __name__ == '__main__':
    main()