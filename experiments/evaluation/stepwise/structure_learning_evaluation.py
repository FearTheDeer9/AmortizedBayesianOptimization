#!/usr/bin/env python3
"""
Structure Learning Evaluation

Evaluates how different intervention strategies help reveal causal structure.
Starts with no initial data and compares how quickly different approaches
help the surrogate learn the true graph structure.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy

# Import baselines
from baselines import create_baseline


class StructureLearningTracker:
    """Track structure learning progress over interventions."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset tracking for new episode."""
        self.f1_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.parent_discovery = []  # Track which parents have been discovered
        self.intervention_count = 0
        self.interventions = []  # List of (variable, value) tuples
    
    def add_evaluation(self, predicted_parents, true_parents, intervention=None):
        """Add structure learning evaluation after an intervention."""
        tp = len(true_parents & predicted_parents)
        fp = len(predicted_parents - true_parents)
        fn = len(true_parents - predicted_parents)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        self.f1_scores.append(f1)
        self.precision_scores.append(precision)
        self.recall_scores.append(recall)
        self.parent_discovery.append(tp)  # Number of true parents discovered
        
        if intervention:
            self.interventions.append(intervention)
        
        self.intervention_count += 1
        
        return f1


def evaluate_structure_learning_episode(
    policy_net, policy_params,
    surrogate_net, surrogate_params,
    scm,
    num_interventions: int = 50,
    initial_observations: int = 0,  # Start with NO data
    strategy: str = 'policy',  # 'policy', 'random', 'oracle'
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate structure learning for one episode.
    
    Args:
        policy_net, policy_params: Policy model
        surrogate_net, surrogate_params: Surrogate model for structure learning
        scm: Structural causal model
        num_interventions: Number of interventions to perform
        initial_observations: Initial observational samples (default 0)
        strategy: Intervention strategy ('policy', 'random', 'oracle')
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary with structure learning metrics
    """
    
    # Get SCM information
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    variable_ranges = scm.get('metadata', {}).get('variable_ranges', {})
    non_target_vars = [v for v in variables if v != target_var]
    
    if verbose:
        print(f"\nEpisode: Target={target_var}, Parents={true_parents}")
        print(f"Strategy: {strategy}")
    
    # Initialize tracking
    tracker = StructureLearningTracker()
    
    # Initialize buffer
    buffer = ExperienceBuffer()
    
    # Add initial observations if any
    if initial_observations > 0:
        samples = sample_from_linear_scm(scm, n_samples=initial_observations, seed=seed)
        for sample in samples:
            buffer.add_observation(sample)
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Create baseline if needed
    baseline = None
    if strategy == 'random':
        baseline = create_baseline('random', seed=seed)
    elif strategy == 'oracle':
        baseline = create_baseline('oracle', scm=scm)
    
    # Create surrogate wrapper
    def surrogate_fn(tensor_3ch, target_var_name, variable_list):
        """Wrapper for surrogate predictions."""
        target_idx = list(variable_list).index(target_var_name)
        rng_key_surrogate = random.PRNGKey(42)
        predictions = surrogate_net.apply(surrogate_params, rng_key_surrogate, 
                                        tensor_3ch, target_idx, False)
        parent_probs = predictions.get('parent_probabilities', 
                                      jnp.full(len(variable_list), 0.5))
        return {'parent_probs': parent_probs}
    
    # Perform interventions
    for intervention_idx in range(num_interventions):
        # Get current buffer state
        if len(buffer.get_all_samples()) == 0:
            # No data yet - need at least one sample
            # Add one random intervention to bootstrap
            rng_key, var_key, value_key = random.split(rng_key, 3)
            selected_var = non_target_vars[int(random.choice(var_key, len(non_target_vars)))]
            var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
            intervened_value = float(random.uniform(value_key, 
                                                   minval=var_range[0], 
                                                   maxval=var_range[1]))
        else:
            # Convert buffer to tensor
            tensor_4ch, mapper, _ = buffer_to_four_channel_tensor(
                buffer, target_var,
                surrogate_fn=surrogate_fn,
                max_history_size=None,
                standardize=True
            )
            
            # Get current structure predictions
            current_parent_probs = {}
            for i, var in enumerate(mapper.variables):
                if var != target_var:
                    prob = float(tensor_4ch[-1, i, 3])
                    current_parent_probs[var] = prob
            
            # Select intervention based on strategy
            if strategy == 'policy':
                # Use policy
                target_idx = mapper.get_index(target_var)
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
                intervened_value = float(jnp.clip(intervened_value + noise, 
                                                 var_range[0], var_range[1]))
            
            elif baseline is not None:
                # Use baseline strategy
                selected_var, intervened_value = baseline.select_intervention(
                    buffer, target_var, variables, variable_ranges,
                    parent_probs=current_parent_probs
                )
            
            # Evaluate current structure learning
            predicted_parents = {var for var, prob in current_parent_probs.items() 
                               if prob > 0.5}
            f1 = tracker.add_evaluation(predicted_parents, true_parents, 
                                       (selected_var, intervened_value))
            
            if verbose and (intervention_idx + 1) % 10 == 0:
                print(f"  Intervention {intervention_idx+1}: F1={f1:.3f}, "
                      f"Predicted={predicted_parents}, True={true_parents}")
        
        # Apply intervention
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: intervened_value}
        )
        
        samples = sample_with_intervention(
            scm, intervention, n_samples=1,
            seed=seed + intervention_idx + 1000
        )
        
        for sample in samples:
            buffer.add_intervention({selected_var: intervened_value}, sample)
    
    # Final evaluation
    if len(buffer.get_all_samples()) > 0:
        tensor_4ch, mapper, _ = buffer_to_four_channel_tensor(
            buffer, target_var,
            surrogate_fn=surrogate_fn,
            max_history_size=None,
            standardize=True
        )
        
        current_parent_probs = {}
        for i, var in enumerate(mapper.variables):
            if var != target_var:
                prob = float(tensor_4ch[-1, i, 3])
                current_parent_probs[var] = prob
        
        predicted_parents = {var for var, prob in current_parent_probs.items() 
                           if prob > 0.5}
        final_f1 = tracker.add_evaluation(predicted_parents, true_parents)
        
        if verbose:
            print(f"  Final F1={final_f1:.3f}")
    
    # Calculate summary metrics
    return {
        'strategy': strategy,
        'num_interventions': num_interventions,
        'true_parents': list(true_parents),
        'f1_progression': tracker.f1_scores,
        'precision_progression': tracker.precision_scores,
        'recall_progression': tracker.recall_scores,
        'parent_discovery': tracker.parent_discovery,
        'final_f1': tracker.f1_scores[-1] if tracker.f1_scores else 0.0,
        'interventions_to_90': next((i for i, f1 in enumerate(tracker.f1_scores) 
                                    if f1 >= 0.9), num_interventions),
        'interventions': tracker.interventions
    }


def compare_structure_learning_strategies(
    policy_path: Path,
    surrogate_path: Path,
    structure_types: List[str] = ['chain', 'fork'],
    num_vars_list: List[int] = [5],
    num_episodes: int = 10,
    num_interventions: int = 50,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different intervention strategies for structure learning.
    
    Returns DataFrame with comparison results.
    """
    
    print("="*70)
    print("STRUCTURE LEARNING COMPARISON")
    print("="*70)
    
    # Load models
    print(f"\nLoading models...")
    print(f"Policy: {policy_path.name}")
    print(f"Surrogate: {surrogate_path.name}")
    
    # Load policy
    policy_checkpoint = load_checkpoint(policy_path)
    policy_params = policy_checkpoint['params']
    policy_architecture = policy_checkpoint.get('architecture', {})
    
    policy_fn = create_quantile_policy(
        hidden_dim=policy_architecture.get('hidden_dim', 256)
    )
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Load surrogate
    surrogate_checkpoint = load_checkpoint(surrogate_path)
    surrogate_params = surrogate_checkpoint['params']
    surrogate_architecture = surrogate_checkpoint.get('architecture', {})
    
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=surrogate_architecture.get('hidden_dim', 128),
            num_heads=surrogate_architecture.get('num_heads', 8),
            num_layers=surrogate_architecture.get('num_layers', 8),
            key_size=surrogate_architecture.get('key_size', 32),
            dropout=0.0,
            use_temperature_scaling=True,
            temperature_init=0.0
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    
    # Create SCM factory
    factory = VariableSCMFactory(seed=seed, noise_scale=0.5)
    
    # Collect results
    all_results = []
    
    # Test each strategy
    strategies = ['policy', 'random', 'oracle']
    
    for structure_type in structure_types:
        for num_vars in num_vars_list:
            print(f"\n\nStructure: {structure_type}, Variables: {num_vars}")
            print("-"*50)
            
            for episode_idx in range(num_episodes):
                # Create SCM
                scm = factory.create_variable_scm(
                    num_variables=num_vars,
                    structure_type=structure_type
                )
                
                if verbose and episode_idx == 0:
                    print(f"Episode {episode_idx+1}: Target={get_target(scm)}, "
                          f"Parents={get_parents(scm, get_target(scm))}")
                
                for strategy in strategies:
                    result = evaluate_structure_learning_episode(
                        policy_net, policy_params,
                        surrogate_net, surrogate_params,
                        scm,
                        num_interventions=num_interventions,
                        initial_observations=0,  # Start with no data
                        strategy=strategy,
                        seed=seed + episode_idx * 1000,
                        verbose=False
                    )
                    
                    # Add metadata
                    result['structure_type'] = structure_type
                    result['num_vars'] = num_vars
                    result['episode'] = episode_idx
                    
                    all_results.append({
                        'structure': structure_type,
                        'num_vars': num_vars,
                        'episode': episode_idx,
                        'strategy': strategy,
                        'final_f1': result['final_f1'],
                        'interventions_to_90': result['interventions_to_90'],
                        'num_parents': len(result['true_parents'])
                    })
            
            # Print summary for this configuration
            df_config = pd.DataFrame([r for r in all_results 
                                     if r['structure'] == structure_type 
                                     and r['num_vars'] == num_vars])
            
            for strategy in strategies:
                df_strat = df_config[df_config['strategy'] == strategy]
                if len(df_strat) > 0:
                    print(f"\n{strategy.upper()}:")
                    print(f"  Final F1: {df_strat['final_f1'].mean():.3f} ± "
                          f"{df_strat['final_f1'].std():.3f}")
                    print(f"  Interventions to F1>0.9: {df_strat['interventions_to_90'].mean():.1f}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    summary = df.groupby('strategy').agg({
        'final_f1': ['mean', 'std'],
        'interventions_to_90': 'mean'
    }).round(3)
    
    print(summary)
    
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Structure learning evaluation")
    
    parser.add_argument('--policy-path', type=Path, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--surrogate-path', type=Path, required=True,
                       help='Path to surrogate checkpoint')
    parser.add_argument('--structures', nargs='+', default=['chain', 'fork'],
                       help='SCM structure types')
    parser.add_argument('--num-vars', nargs='+', type=int, default=[5],
                       help='Number of variables')
    parser.add_argument('--num-episodes', type=int, default=10,
                       help='Episodes per configuration')
    parser.add_argument('--num-interventions', type=int, default=50,
                       help='Interventions per episode')
    parser.add_argument('--output-dir', type=Path, default=Path('structure_learning_results'),
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run comparison
    df = compare_structure_learning_strategies(
        policy_path=args.policy_path,
        surrogate_path=args.surrogate_path,
        structure_types=args.structures,
        num_vars_list=args.num_vars,
        num_episodes=args.num_episodes,
        num_interventions=args.num_interventions,
        seed=args.seed
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"structure_learning_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())