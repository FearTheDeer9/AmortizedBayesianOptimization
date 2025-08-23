#!/usr/bin/env python3
"""Evaluate trained surrogate models on unseen SCMs up to 100 variables."""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
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
    """Create comprehensive test set of unseen SCMs."""
    test_scms = [
        # Small (different seeds/parameters from training)
        {'num_vars': 5, 'structure': 'fork'},
        {'num_vars': 10, 'structure': 'chain'},
        
        # Medium
        {'num_vars': 20, 'structure': 'collider'},
        {'num_vars': 30, 'structure': 'mixed'},
        {'num_vars': 40, 'structure': 'chain'},  # Hard structure
        
        # Large
        {'num_vars': 50, 'structure': 'fork'},
        {'num_vars': 60, 'structure': 'random', 'edge_density': 0.2},
        {'num_vars': 70, 'structure': 'collider'},
        
        # Extra Large
        {'num_vars': 80, 'structure': 'mixed'},
        {'num_vars': 90, 'structure': 'chain'},  # Very hard
        {'num_vars': 100, 'structure': 'fork'},  # Easiest 100-var
        {'num_vars': 100, 'structure': 'random', 'edge_density': 0.1},  # Sparse 100-var
    ]
    return test_scms


def initialize_models(model_config: Dict, key: jax.random.PRNGKey, max_vars: int):
    """Initialize policy and surrogate models."""
    dummy_buffer = ExperienceBuffer()
    dummy_values = {f'X{i}': 0.0 for i in range(max_vars)}
    dummy_sample = create_sample(dummy_values, intervention_type=None)
    dummy_buffer.add_observation(dummy_sample)
    dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
    
    # Initialize policy
    policy_fn = create_permutation_invariant_alternating_policy(model_config['hidden_dim'])
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    policy_params = policy_net.init(key, dummy_tensor, 0)
    
    # Initialize surrogate
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=model_config['hidden_dim'],
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 6),
            dropout=model_config.get('dropout', 0.1)
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    
    return policy_net, policy_params, surrogate_net


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
            'f1': f1,
            'precision': precision if 'precision' in locals() else 0,
            'recall': recall if 'recall' in locals() else 0
        })
    
    return np.mean(all_f1_scores), all_metrics


def evaluate_on_scm(scm, policy_net, policy_params, surrogate_net, surrogate_params,
                   num_interventions, rng_key, verbose=False):
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
    detailed_metrics = []
    
    for intervention_idx in range(num_interventions):
        # Select intervention using policy
        rng_key, target_key, select_key = random.split(rng_key, 3)
        target_idx = random.choice(target_key, num_vars)
        target_var = variables[int(target_idx)]
        
        tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
        output = policy_net.apply(policy_params, tensor, int(target_idx))
        
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
        avg_f1, metrics = evaluate_predictions(surrogate_net, surrogate_params, buffer, scm, mapper, eval_key)
        f1_trajectory.append(avg_f1)
        detailed_metrics.append(metrics)
        
        if verbose and intervention_idx % 5 == 0:
            print(f"      Intervention {intervention_idx+1}: F1={avg_f1:.3f}")
    
    return f1_trajectory, detailed_metrics


def load_model_checkpoint(checkpoint_path: Path) -> Dict:
    """Load a model checkpoint using standardized format."""
    return load_checkpoint(checkpoint_path)


def evaluate_checkpoint_dir(checkpoint_dir: Path, test_scms: List[Dict], 
                           num_interventions: int, seed: int) -> Dict:
    """Evaluate all models in a checkpoint directory."""
    results = {}
    
    # Find all model files
    model_files = []
    
    # Check for staged checkpoints
    for stage in range(1, 6):
        stage_file = checkpoint_dir / f"stage_{stage}_complete.pkl"
        if stage_file.exists():
            model_files.append(('stage', stage, stage_file))
    
    # Check for size-specific best models
    best_dir = checkpoint_dir / "best_per_size"
    if best_dir.exists():
        for best_file in sorted(best_dir.glob("best_*var.pkl")):
            size = int(best_file.stem.split('_')[1].replace('var', ''))
            model_files.append(('best', size, best_file))
    
    # Check for final model
    final_file = checkpoint_dir / "final_model.pkl"
    if final_file.exists():
        model_files.append(('final', None, final_file))
    
    print(f"\nFound {len(model_files)} model checkpoints to evaluate")
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Load config if available
    config_file = checkpoint_dir / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        model_config = config.get('model_config', {'hidden_dim': 768})
    else:
        # Default config
        model_config = {'hidden_dim': 768, 'num_layers': 8, 'num_heads': 12}
    
    # Find max variables in test set
    max_vars = max(scm['num_vars'] for scm in test_scms)
    
    # Initialize models
    rng_key, init_key = random.split(rng_key)
    policy_net, policy_params, surrogate_net = initialize_models(model_config, init_key, max_vars)
    
    # Create SCM factory
    factory = VariableSCMFactory(seed=seed)
    
    # Evaluate each model
    for model_type, model_id, model_path in model_files:
        print(f"\nEvaluating {model_type} model" + (f" {model_id}" if model_id else ""))
        
        # Load checkpoint
        checkpoint = load_model_checkpoint(model_path)
        surrogate_params = checkpoint['params']
        
        # Extract metadata if available
        metadata = checkpoint.get('metadata', {})
        step = metadata.get('step', 'unknown')
        
        model_results = {
            'type': model_type,
            'id': model_id,
            'path': str(model_path),
            'evaluations': []
        }
        
        # Evaluate on each test SCM
        for scm_config in test_scms:
            print(f"  Testing on {scm_config['structure']} with {scm_config['num_vars']} variables...")
            
            # Create SCM
            scm = factory.create_variable_scm(
                num_variables=scm_config['num_vars'],
                structure_type=scm_config['structure'],
                edge_density=scm_config.get('edge_density', 0.5)
            )
            
            # Evaluate
            rng_key, eval_key = random.split(rng_key)
            f1_trajectory, detailed = evaluate_on_scm(
                scm, policy_net, policy_params, surrogate_net,
                surrogate_params, num_interventions, eval_key, verbose=False
            )
            
            model_results['evaluations'].append({
                'scm_config': scm_config,
                'f1_trajectory': f1_trajectory,
                'final_f1': f1_trajectory[-1],
                'max_f1': max(f1_trajectory),
                'avg_f1': np.mean(f1_trajectory)
            })
            
            print(f"    F1: {f1_trajectory[-1]:.3f} (max: {max(f1_trajectory):.3f})")
        
        # Summary for this model
        all_final_f1 = [e['final_f1'] for e in model_results['evaluations']]
        model_results['summary'] = {
            'mean_f1': np.mean(all_final_f1),
            'std_f1': np.std(all_final_f1),
            'min_f1': min(all_final_f1),
            'max_f1': max(all_final_f1)
        }
        
        results[f"{model_type}_{model_id}" if model_id else model_type] = model_results
    
    return results


def plot_results(results: Dict, output_dir: Path):
    """Create visualization plots for evaluation results."""
    output_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    model_names = []
    mean_f1s = []
    std_f1s = []
    
    for name, data in results.items():
        if 'summary' in data:
            model_names.append(name)
            mean_f1s.append(data['summary']['mean_f1'])
            std_f1s.append(data['summary']['std_f1'])
    
    # Plot 1: Overall performance comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    ax.bar(x, mean_f1s, yerr=std_f1s, capsize=5)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_comparison.png', dpi=150)
    plt.close()
    
    # Plot 2: Performance by SCM size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for name, data in results.items():
        if 'evaluations' not in data:
            continue
        
        sizes = []
        f1_scores = []
        
        for eval_data in data['evaluations']:
            sizes.append(eval_data['scm_config']['num_vars'])
            f1_scores.append(eval_data['final_f1'])
        
        # Sort by size
        sorted_pairs = sorted(zip(sizes, f1_scores))
        sizes, f1_scores = zip(*sorted_pairs)
        
        ax.plot(sizes, f1_scores, marker='o', label=name, linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of Variables')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance Scaling with SCM Size')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_performance.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate scaled surrogate models')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--interventions', type=int, default=20,
                       help='Number of interventions per SCM')
    parser.add_argument('--test-sizes', type=str, default=None,
                       help='Comma-separated list of SCM sizes to test')
    parser.add_argument('--test-structures', type=str, default='all',
                       help='Structures to test (all, fork, chain, collider, mixed, random)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (different from training)')
    parser.add_argument('--output-plots', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("SCALED SURROGATE MODEL EVALUATION")
    print("="*70)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist")
        sys.exit(1)
    
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Interventions per SCM: {args.interventions}")
    
    # Create test SCMs
    if args.test_sizes:
        sizes = [int(s) for s in args.test_sizes.split(',')]
        test_scms = []
        for size in sizes:
            if size <= 20:
                structures = ['fork', 'chain', 'collider']
            elif size <= 50:
                structures = ['fork', 'chain', 'mixed']
            else:
                structures = ['fork', 'random', 'chain']
            
            for struct in structures:
                if args.test_structures != 'all' and struct not in args.test_structures:
                    continue
                config = {'num_vars': size, 'structure': struct}
                if struct == 'random':
                    config['edge_density'] = 0.2 if size <= 50 else 0.1
                test_scms.append(config)
    else:
        test_scms = create_test_scms()
        if args.test_structures != 'all':
            test_scms = [s for s in test_scms if s['structure'] in args.test_structures]
    
    print(f"Test set: {len(test_scms)} SCMs")
    
    # Run evaluation
    results = evaluate_checkpoint_dir(checkpoint_dir, test_scms, args.interventions, args.seed)
    
    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'scaled_evaluation_{timestamp}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(v) for v in obj]
        else:
            return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_arrays(results), f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Generate plots if requested
    if args.output_plots:
        plot_dir = results_dir / f'plots_{timestamp}'
        plot_results(results, plot_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for name, data in results.items():
        if 'summary' in data:
            summary = data['summary']
            print(f"\n{name}:")
            print(f"  Mean F1: {summary['mean_f1']:.3f} ± {summary['std_f1']:.3f}")
            print(f"  Range: [{summary['min_f1']:.3f}, {summary['max_f1']:.3f}]")
            
            # Performance by size
            size_performance = {}
            for eval_data in data['evaluations']:
                size = eval_data['scm_config']['num_vars']
                if size not in size_performance:
                    size_performance[size] = []
                size_performance[size].append(eval_data['final_f1'])
            
            print("  Performance by size:")
            for size in sorted(size_performance.keys()):
                mean_f1 = np.mean(size_performance[size])
                print(f"    {size} vars: {mean_f1:.3f}")


if __name__ == '__main__':
    main()