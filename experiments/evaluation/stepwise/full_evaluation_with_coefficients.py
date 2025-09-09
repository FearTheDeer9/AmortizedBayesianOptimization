#!/usr/bin/env python3
"""
Extended evaluation script with coefficient channel support.
Based on full_evaluation.py but adds ability to use ground truth coefficients.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import everything from original full_evaluation
from full_evaluation import (
    MetricsTracker,
    load_models,
    evaluate_checkpoint_pair,
    evaluate_training_progression,
    create_baseline
)

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target, get_mechanisms
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values, get_intervention_targets, is_observational
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def buffer_to_coefficient_channel_tensor(buffer, target_var, scm, **kwargs):
    """
    Create tensor with actual coefficient values in fourth channel.
    Based on train_grpo_ground_truth_coefficients.py implementation.
    """
    # Get all samples from buffer
    all_samples = buffer.get_all_samples()
    if not all_samples:
        raise ValueError("Buffer is empty")
    
    # Get variable order and create mapper
    variable_order = sorted(buffer.get_variable_coverage())
    if target_var not in variable_order:
        raise ValueError(f"Target '{target_var}' not in buffer variables: {variable_order}")
    
    mapper = VariableMapper(variable_order, target_var)
    n_vars = len(variable_order)
    target_idx = mapper.get_index(target_var)
    
    # Get ground truth parents and coefficients
    true_parents = set(get_parents(scm, target_var))
    
    # Extract coefficients from target mechanism
    mechanisms = get_mechanisms(scm)
    target_mechanism = mechanisms.get(target_var)
    coefficients = {}
    if target_mechanism and hasattr(target_mechanism, 'coefficients'):
        for parent, coeff in target_mechanism.coefficients.items():
            coefficients[parent] = coeff
    
    # Create coefficient array for all variables
    coefficient_values = np.zeros(n_vars)
    for i, var in enumerate(variable_order):
        if var == target_var:
            # Target itself gets 0
            coefficient_values[i] = 0.0
        elif var in coefficients:
            # Parents get their actual coefficient
            coefficient_values[i] = coefficients[var]
        else:
            # Non-parents get 0
            coefficient_values[i] = 0.0
    
    coefficient_values = jnp.array(coefficient_values)
    
    # Log the coefficient values
    logger.info(f"    🔢 COEFFICIENT CHANNEL for {target_var}:")
    logger.info(f"       True parents: {true_parents}")
    logger.info(f"       Coefficients: {coefficients}")
    
    # Determine actual history size
    max_history_size = kwargs.get('max_history_size', 100)
    if max_history_size is None:
        max_history_size = 100  # Default value
    actual_size = min(len(all_samples), max_history_size)
    recent_samples = all_samples[-actual_size:]
    
    # Initialize 4-channel tensor
    tensor = jnp.zeros((max_history_size, n_vars, 4))
    
    # Fill tensor with recent samples
    for t, sample in enumerate(recent_samples):
        tensor_idx = max_history_size - actual_size + t
        
        # Channel 0: Values
        sample_values = get_values(sample)
        values = jnp.array([float(sample_values.get(var, 0.0)) for var in variable_order])
        
        # Channel 1: Target indicator
        target_mask = jnp.array([1.0 if var == target_var else 0.0 for var in variable_order])
        
        # Channel 2: Intervention indicator
        intervention_targets = get_intervention_targets(sample)
        intervention_mask = jnp.array([1.0 if var in intervention_targets else 0.0 for var in variable_order])
        
        # Channel 3: Coefficient values (constant for all samples)
        # This is the KEY - using actual coefficients instead of probabilities
        
        # Set tensor values
        tensor = tensor.at[tensor_idx, :, 0].set(values)
        tensor = tensor.at[tensor_idx, :, 1].set(target_mask)
        tensor = tensor.at[tensor_idx, :, 2].set(intervention_mask)
        tensor = tensor.at[tensor_idx, :, 3].set(coefficient_values)  # COEFFICIENTS!
    
    # Standardize values if requested (per-variable)
    if kwargs.get('standardize', True) and actual_size > 1:
        # Standardize channel 0 (values) per-variable
        start_idx = max_history_size - actual_size
        for var_idx in range(n_vars):
            var_actual_data = tensor[start_idx:, var_idx, 0]
            if len(var_actual_data) > 1:
                var_mean = jnp.mean(var_actual_data)
                var_std = jnp.std(var_actual_data) + 1e-8
                standardized_var = (tensor[:, var_idx, 0] - var_mean) / var_std
                mask = jnp.zeros(max_history_size)
                mask = mask.at[start_idx:].set(1.0)
                new_var_values = tensor[:, var_idx, 0] * (1 - mask) + standardized_var * mask
                tensor = tensor.at[:, var_idx, 0].set(new_var_values)
    
    # Count obs and int samples for diagnostics
    n_obs = sum(1 for s in all_samples if is_observational(s))
    n_int = len(all_samples) - n_obs
    
    # Diagnostics
    diagnostics = {
        'n_variables': n_vars,
        'n_samples': len(all_samples),
        'n_obs': n_obs,
        'n_int': n_int,
        'actual_history_size': actual_size,
        'target_variable': target_var,
        'target_idx': target_idx,
        'true_parents': list(true_parents),
        'coefficients': coefficients,
        'coefficient_channel': 'ACTIVE'
    }
    
    return tensor, mapper, diagnostics


def evaluate_episode_with_coefficients(
    policy_net, policy_params,
    scm, metrics_tracker: MetricsTracker,
    num_interventions: int = 30,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    max_history_size: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
    baseline: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate one episode using ground truth coefficients in 4th channel.
    Modified from evaluate_episode to use coefficient channel.
    """
    
    # Get SCM information
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    variable_ranges = scm.get('metadata', {}).get('variable_ranges', {})
    
    # Extract parent coefficients from metadata
    all_coefficients = scm.get('metadata', {}).get('coefficients', {})
    parent_coefficients = {}
    for edge_str, coeff in all_coefficients.items():
        # Parse edge string
        if isinstance(edge_str, str) and ',' in edge_str:
            edge_str = edge_str.strip('()')
            parts = [p.strip().strip("'").strip('"') for p in edge_str.split(',')]
            if len(parts) == 2:
                from_var, to_var = parts
                if to_var == target_var and from_var in true_parents:
                    parent_coefficients[from_var] = coeff
        elif isinstance(edge_str, tuple) and len(edge_str) == 2:
            from_var, to_var = edge_str
            if to_var == target_var and from_var in true_parents:
                parent_coefficients[from_var] = coeff
    
    # Calculate optimal value (placeholder)
    optimal_value = -5.0
    
    # Start episode tracking
    coeff_stats = {}
    if parent_coefficients:
        coeff_magnitudes = [abs(c) for c in parent_coefficients.values()]
        coeff_stats = {
            'min_coefficient': min(coeff_magnitudes),
            'max_coefficient': max(coeff_magnitudes),
            'mean_coefficient': float(np.mean(coeff_magnitudes)),
            'parent_coefficients': parent_coefficients
        }
    
    metrics_tracker.start_episode({
        'num_variables': len(variables),
        'target': target_var,
        'true_parents': list(true_parents),
        'structure_type': scm.get('metadata', {}).get('structure_type', 'unknown'),
        **coeff_stats
    })
    
    if verbose:
        print(f"\n📊 Episode: {len(variables)} vars, target={target_var}, parents={true_parents}")
        if parent_coefficients:
            coeff_magnitudes = [abs(c) for c in parent_coefficients.values()]
            print(f"   Coefficient channel active - using actual coefficient values")
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    
    # Add observational samples
    rng_key, sample_key = random.split(rng_key)
    samples = sample_from_linear_scm(scm, n_samples=initial_observations, seed=int(sample_key[0]))
    for sample in samples:
        buffer.add_observation(sample)
    
    # Add initial random interventions if specified
    if initial_interventions > 0:
        non_target_vars = [v for v in variables if v != target_var]
        
        for i in range(initial_interventions):
            rng_key, var_key = random.split(rng_key)
            selected_var_idx = random.choice(var_key, len(non_target_vars))
            selected_var = non_target_vars[int(selected_var_idx)]
            
            var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
            rng_key, value_key = random.split(rng_key)
            intervened_value = float(random.uniform(value_key, 
                                                   minval=var_range[0], 
                                                   maxval=var_range[1]))
            
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: intervened_value}
            )
            
            samples = sample_with_intervention(
                scm, intervention, n_samples=1,
                seed=seed + i + 5000
            )
            
            for sample in samples:
                buffer.add_intervention({selected_var: intervened_value}, sample)
    
    # Run interventions using coefficient channel
    for intervention_idx in range(num_interventions):
        # Convert buffer to 4-channel tensor with coefficients
        tensor_4ch, mapper, _ = buffer_to_coefficient_channel_tensor(
            buffer, target_var, scm,
            max_history_size=max_history_size,
            standardize=True
        )
        
        # Get current parent probability predictions (actually coefficients now)
        current_parent_probs = {}
        for i, var in enumerate(mapper.variables):
            if var != target_var:
                # Note: These are coefficients, not probabilities
                coeff = float(tensor_4ch[-1, i, 3])
                # Convert to pseudo-probability for compatibility
                # Parents with non-zero coefficients get high "probability"
                prob = 1.0 if coeff != 0 else 0.0
                current_parent_probs[var] = prob
        
        # Select intervention
        if baseline is not None:
            # Use baseline for intervention selection
            selected_var, intervened_value = baseline.select_intervention(
                buffer, target_var, variables, variable_ranges,
                parent_probs=current_parent_probs
            )
        else:
            # Use policy for intervention selection
            target_idx = mapper.get_index(target_var)
            
            # Call policy
            policy_output = policy_net.apply(policy_params, tensor_4ch, target_idx)
            
            # Decode quantile output
            quantile_scores = policy_output['quantile_scores']
            flat_scores = quantile_scores.flatten()
            
            # Sample from distribution
            rng_key, sample_key = random.split(rng_key)
            probs = jax.nn.softmax(flat_scores)
            selected_flat_idx = random.choice(sample_key, len(flat_scores), p=probs)
            
            # Map back to variable and quantile
            selected_var_idx = selected_flat_idx // 3
            selected_quantile_idx = selected_flat_idx % 3
            selected_var = mapper.variables[int(selected_var_idx)]
            
            # Map quantile to value
            quantile_values = {0: 0.25, 1: 0.50, 2: 0.75}
            percentile = quantile_values[int(selected_quantile_idx)]
            var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
            intervened_value = var_range[0] + (var_range[1] - var_range[0]) * percentile
            
            # Add exploration noise
            rng_key, noise_key = random.split(rng_key)
            noise = random.normal(noise_key) * 0.1
            intervened_value = float(jnp.clip(intervened_value + noise, var_range[0], var_range[1]))
        
        # Check if parent
        is_parent = selected_var in true_parents
        
        # Apply intervention
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: intervened_value}
        )
        
        samples = sample_with_intervention(
            scm, intervention, n_samples=1, 
            seed=seed + intervention_idx + 1000
        )
        
        # Get target value after intervention
        for sample in samples:
            target_value = float(get_values(sample)[target_var])
            buffer.add_intervention({selected_var: intervened_value}, sample)
        
        # Track metrics
        metrics_tracker.add_intervention({
            'intervention': (selected_var, intervened_value),
            'parent_probs': current_parent_probs,
            'target_value': target_value,
            'is_parent': is_parent,
            'true_parents': true_parents,
            'optimal_value': optimal_value
        })
        
        if verbose and (intervention_idx + 1) % 10 == 0:
            current_f1 = metrics_tracker.current_episode['f1_scores'][-1]
            print(f"  Intervention {intervention_idx+1}: F1={current_f1:.3f}, "
                  f"target={target_value:.2f}, parent_rate="
                  f"{np.mean(metrics_tracker.current_episode['parent_selections']):.2%}")
    
    # End episode
    metrics_tracker.end_episode()
    
    return metrics_tracker.episodes[-1]['summary']


def evaluate_checkpoint_pair_with_coefficients(
    policy_path: Path,
    num_episodes: int = 10,
    num_interventions: int = 30,
    initial_observations: int = 20,
    initial_interventions: int = 10,
    max_history_size: Optional[int] = None,
    structure_types: List[str] = ['chain'],
    num_vars_list: List[int] = [8],
    seed: int = 42,
    verbose: bool = True,
    include_baselines: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a policy using coefficient channel (no surrogate needed).
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating Policy with Coefficient Channel")
    print(f"Policy: {policy_path.name}")
    print(f"Channel 4: Ground truth coefficient values")
    print(f"{'='*70}")
    
    # Load policy
    policy_checkpoint = load_checkpoint(policy_path)
    policy_params = policy_checkpoint['params']
    policy_architecture = policy_checkpoint.get('architecture', {})
    
    # Create policy network
    from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy
    
    policy_fn = create_quantile_policy(
        hidden_dim=policy_architecture.get('hidden_dim', 256)
    )
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Initialize metrics trackers
    metrics_tracker = MetricsTracker()
    random_tracker = MetricsTracker() if include_baselines else None
    oracle_tracker = MetricsTracker() if include_baselines else None
    
    # Create SCM factory
    factory = VariableSCMFactory(
        seed=seed,
        noise_scale=0.5,
        coefficient_range=(-3.0, 3.0),
        vary_intervention_ranges=True,
        use_output_bounds=True
    )
    
    # Evaluate across different SCM configurations
    episode_count = 0
    for structure_type in structure_types:
        for num_vars in num_vars_list:
            for episode_idx in range(num_episodes):
                episode_count += 1
                
                if verbose:
                    print(f"\n📊 Episode {episode_count}: {structure_type} with {num_vars} vars")
                
                # Create SCM
                scm = factory.create_variable_scm(
                    num_variables=num_vars,
                    structure_type=structure_type
                )
                
                # Evaluate episode with coefficient channel
                summary = evaluate_episode_with_coefficients(
                    policy_net, policy_params,
                    scm, metrics_tracker,
                    num_interventions=num_interventions,
                    initial_observations=initial_observations,
                    initial_interventions=initial_interventions,
                    max_history_size=max_history_size,
                    seed=seed + episode_count * 1000,
                    verbose=False
                )
                
                if verbose:
                    print(f"  Policy with coefficients: F1={summary['final_f1']:.3f}, "
                          f"Parent Rate={summary['parent_selection_rate']:.2%}, "
                          f"Target={summary['best_target']:.2f}")
                
                # Run baselines if requested
                if include_baselines:
                    # Random baseline
                    random_baseline = create_baseline('random', seed=seed + episode_count * 2000)
                    random_summary = evaluate_episode_with_coefficients(
                        policy_net, policy_params,
                        scm, random_tracker,
                        num_interventions=num_interventions,
                        initial_observations=initial_observations,
                        initial_interventions=initial_interventions,
                        max_history_size=max_history_size,
                        seed=seed + episode_count * 1000,
                        verbose=False,
                        baseline=random_baseline
                    )
                    
                    # Oracle baseline
                    oracle_baseline = create_baseline('oracle', scm=scm)
                    oracle_summary = evaluate_episode_with_coefficients(
                        policy_net, policy_params,
                        scm, oracle_tracker,
                        num_interventions=num_interventions,
                        initial_observations=initial_observations,
                        initial_interventions=initial_interventions,
                        max_history_size=max_history_size,
                        seed=seed + episode_count * 1000,
                        verbose=False,
                        baseline=oracle_baseline
                    )
                    
                    if verbose:
                        print(f"  Random: F1={random_summary['final_f1']:.3f}, "
                              f"Parent Rate={random_summary['parent_selection_rate']:.2%}")
                        print(f"  Oracle: F1={oracle_summary['final_f1']:.3f}, "
                              f"Parent Rate={oracle_summary['parent_selection_rate']:.2%}")
    
    # Get aggregate metrics
    aggregate_metrics = metrics_tracker.get_aggregate_metrics()
    
    # Add metadata
    result = {
        'metadata': {
            'policy_path': str(policy_path),
            'policy_architecture': policy_architecture,
            'coefficient_channel': True
        },
        'evaluation_config': {
            'num_episodes': episode_count,
            'num_interventions': num_interventions,
            'structure_types': structure_types,
            'num_vars_list': num_vars_list
        },
        'aggregate_metrics': aggregate_metrics,
        'episodes': metrics_tracker.episodes
    }
    
    # Add baseline results if computed
    if include_baselines:
        result['baselines'] = {
            'random': {
                'aggregate_metrics': random_tracker.get_aggregate_metrics(),
                'episodes': random_tracker.episodes
            },
            'oracle': {
                'aggregate_metrics': oracle_tracker.get_aggregate_metrics(),
                'episodes': oracle_tracker.episodes
            }
        }
    
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full evaluation with coefficient support")
    
    # Model paths
    parser.add_argument('--policy-path', type=Path, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--surrogate-path', type=Path,
                       help='Path to surrogate checkpoint (not needed for coefficient evaluation)')
    
    # Evaluation config
    parser.add_argument('--num-episodes', type=int, default=30,
                       help='Number of episodes per configuration')
    parser.add_argument('--num-interventions', type=int, default=40,
                       help='Number of interventions per episode')
    parser.add_argument('--initial-observations', type=int, default=20,
                       help='Number of initial observational samples')
    parser.add_argument('--initial-interventions', type=int, default=10,
                       help='Number of initial random interventional samples')
    parser.add_argument('--structures', nargs='+', 
                       default=['chain'],
                       help='SCM structure types to test')
    parser.add_argument('--num-vars', nargs='+', type=int, default=[8],
                       help='Number of variables to test')
    
    # Special modes
    parser.add_argument('--use-coefficient-channel', action='store_true',
                       help='Use ground truth coefficient channel instead of surrogate')
    parser.add_argument('--oracle-surrogate', action='store_true',
                       help='Use oracle surrogate with perfect parent predictions')
    
    # Output
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation_results'),
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--baselines', action='store_true',
                       help='Include random and oracle baselines')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FULL EVALUATION WITH COEFFICIENT SUPPORT")
    print("="*70)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.use_coefficient_channel:
        # Evaluate with coefficient channel
        print(f"\n📊 Evaluating with coefficient channel: {args.policy_path}")
        
        result = evaluate_checkpoint_pair_with_coefficients(
            policy_path=args.policy_path,
            num_episodes=args.num_episodes,
            num_interventions=args.num_interventions,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            structure_types=args.structures,
            num_vars_list=args.num_vars,
            seed=args.seed,
            include_baselines=args.baselines
        )
        
        # Save results
        json_path = args.output_dir / f"coefficient_evaluation_{timestamp}.json"
        
    else:
        # Use original evaluation with surrogate
        if not args.surrogate_path:
            print("Error: --surrogate-path required when not using coefficient channel")
            return 1
        
        print(f"\n📊 Evaluating with surrogate: {args.policy_path}")
        
        result = evaluate_checkpoint_pair(
            policy_path=args.policy_path,
            surrogate_path=args.surrogate_path,
            num_episodes=args.num_episodes,
            num_interventions=args.num_interventions,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            structure_types=args.structures,
            num_vars_list=args.num_vars,
            seed=args.seed,
            include_baselines=args.baselines,
            use_oracle_surrogate=args.oracle_surrogate
        )
        
        # Save results
        mode = "oracle" if args.oracle_surrogate else "learned"
        json_path = args.output_dir / f"evaluation_{mode}_{timestamp}.json"
    
    # Convert to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(result), f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Display summary
    metrics = result['aggregate_metrics']
    print(f"\n📋 Evaluation Summary:")
    print(f"  F1 Score: {metrics.get('final_f1_mean', 0):.3f} ± "
          f"{metrics.get('final_f1_std', 0):.3f}")
    print(f"  Parent Selection: {metrics.get('parent_selection_rate_mean', 0):.2%} ± "
          f"{metrics.get('parent_selection_rate_std', 0):.2%}")
    print(f"  Best Target: {metrics.get('best_target_mean', 0):.2f} ± "
          f"{metrics.get('best_target_std', 0):.2f}")
    
    print(f"\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())