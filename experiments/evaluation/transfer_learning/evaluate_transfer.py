#!/usr/bin/env python3
"""
Transfer Learning Evaluation Script
Evaluates surrogate models trained on one graph type on all other graph types.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime
import argparse
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.four_channel_converter import buffer_to_four_channel_tensor
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.policies.clean_policy_factory import create_quantile_policy


# Define standard model configurations
# Note: There's a legacy naming mismatch in the API:
# - "fork" in API creates a star structure
# - "true_fork" in API creates the actual fork structure
MODEL_CONFIGS = {
    'fork': {  # This model was trained on true fork structures
        'path': 'checkpoints/avici_runs/avici_style_20250903_220822/checkpoint_step_1000.pkl',
        'name': 'Fork-trained',
        'structure_type': 'true_fork'
    },
    'chain': {
        'path': 'checkpoints/avici_runs/avici_style_20250903_154909/checkpoint_step_1000.pkl',
        'name': 'Chain-trained',
        'structure_type': 'chain'
    },
    'scale_free': {
        'path': 'imperial-vm-checkpoints/checkpoints/production_12hr_10hr_trainingy/surrogate_phase_33.pkl',
        'name': 'ScaleFree-trained',
        'structure_type': 'scale_free'
    },
    'star': {  # This model was trained on star structures (called "fork" in API)
        'path': 'imperial-vm-checkpoints/checkpoints/production_12hr_fork_and_chain/surrogate_phase_31.pkl',
        'name': 'Star-trained',
        'structure_type': 'fork'  # API calls star structures "fork"
    },
    'random': {
        'path': 'imperial-vm-checkpoints/avici_style_20250905_222147/checkpoint_step_1000.pkl',
        'name': 'Random-trained',
        'structure_type': 'random'
    }
}

# Test configurations - using display names
TEST_GRAPH_TYPES = ['true_fork', 'chain', 'scale_free', 'star', 'random']

# Mapping from display names to API structure types
GRAPH_TYPE_TO_API = {
    'true_fork': 'true_fork',  # Actual fork structure
    'chain': 'chain',
    'scale_free': 'scale_free',
    'star': 'fork',  # API calls star "fork"
    'random': 'random'
}
TEST_GRAPH_SIZES = [3, 8, 15, 30, 100]


def load_surrogate_model(model_path: Path) -> Tuple[Any, Any, Dict]:
    """Load surrogate model from checkpoint."""
    checkpoint = load_checkpoint(model_path)
    params = checkpoint['params']
    architecture = checkpoint.get('architecture', {})
    
    # Create surrogate network
    def surrogate_fn(x, target_idx, is_training):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=architecture.get('hidden_dim', 128),
            num_heads=architecture.get('num_heads', 8),
            num_layers=architecture.get('num_layers', 8),
            key_size=architecture.get('key_size', 32),
            dropout=architecture.get('dropout', 0.1) if is_training else 0.0,
            use_temperature_scaling=True,
            temperature_init=0.0
        )
        return model(x, target_idx, is_training)
    
    surrogate_net = hk.transform(surrogate_fn)
    
    metadata = {
        'architecture': architecture,
        'model_path': str(model_path),
        'model_type': checkpoint.get('model_type', 'surrogate'),
        'training_steps': checkpoint.get('step', -1)
    }
    
    return surrogate_net, params, metadata


def load_dummy_policy() -> Tuple[Any, Any]:
    """Load a dummy policy to satisfy the evaluation script requirements."""
    # Use a simple policy that won't be used (num_interventions=0)
    policy_fn = create_quantile_policy(hidden_dim=256)
    policy_net = hk.without_apply_rng(hk.transform(policy_fn))
    
    # Create dummy params (won't be used)
    rng_key = random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10, 4))
    dummy_target = 0
    policy_params = policy_net.init(rng_key, dummy_input, dummy_target)
    
    return policy_net, policy_params


def evaluate_single_episode(
    surrogate_net,
    surrogate_params,
    scm,
    initial_observations: int = 80,
    initial_interventions: int = 40,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate a single episode with data-only evaluation."""
    
    # Get SCM information
    variables = list(get_variables(scm))
    target_var = get_target(scm)
    true_parents = set(get_parents(scm, target_var))
    variable_ranges = scm.get('metadata', {}).get('variable_ranges', {})
    
    # Initialize RNG
    rng_key = random.PRNGKey(seed)
    
    # Initialize buffer with observations
    buffer = ExperienceBuffer()
    
    # Add observational samples
    rng_key, sample_key = random.split(rng_key)
    samples = sample_from_linear_scm(scm, n_samples=initial_observations, seed=int(sample_key[0]))
    for sample in samples:
        buffer.add_observation(sample)
    
    # Add initial random interventions
    non_target_vars = [v for v in variables if v != target_var]
    
    for i in range(initial_interventions):
        # Randomly select a variable to intervene on
        rng_key, var_key = random.split(rng_key)
        selected_var_idx = random.choice(var_key, len(non_target_vars))
        selected_var = non_target_vars[int(selected_var_idx)]
        
        # Sample intervention value from variable's range
        var_range = variable_ranges.get(selected_var, (-2.0, 2.0))
        rng_key, value_key = random.split(rng_key)
        intervened_value = float(random.uniform(value_key, 
                                               minval=var_range[0], 
                                               maxval=var_range[1]))
        
        # Apply intervention and sample
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: intervened_value}
        )
        
        samples = sample_with_intervention(
            scm, intervention, n_samples=1,
            seed=seed + i + 5000
        )
        
        # Add to buffer
        for sample in samples:
            buffer.add_intervention({selected_var: intervened_value}, sample)
    
    # Create surrogate wrapper
    def surrogate_fn(tensor_3ch, target_var_name, variable_list):
        """Wrapper for surrogate predictions."""
        target_idx = list(variable_list).index(target_var_name)
        rng_key_surrogate = random.PRNGKey(42)
        predictions = surrogate_net.apply(surrogate_params, rng_key_surrogate, tensor_3ch, target_idx, False)
        parent_probs = predictions.get('parent_probabilities', jnp.full(len(variable_list), 0.5))
        return {'parent_probs': parent_probs}
    
    # Get final buffer state and make predictions
    tensor_4ch, mapper, _ = buffer_to_four_channel_tensor(
        buffer, target_var, 
        surrogate_fn=surrogate_fn,
        max_history_size=None,  # Use all data
        standardize=True
    )
    
    # Get parent probability predictions
    parent_probs = {}
    for i, var in enumerate(mapper.variables):
        if var != target_var:
            prob = float(tensor_4ch[-1, i, 3])
            parent_probs[var] = prob
    
    # Calculate metrics
    predicted_parents = {var for var, prob in parent_probs.items() if prob > 0.5}
    tp = len(true_parents & predicted_parents)
    fp = len(predicted_parents - true_parents)
    fn = len(true_parents - predicted_parents)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate average parent probability for true parents
    avg_true_parent_prob = np.mean([parent_probs.get(p, 0.0) for p in true_parents]) if true_parents else 0.0
    
    # Calculate average parent probability for non-parents
    non_parents = set(v for v in variables if v != target_var and v not in true_parents)
    avg_non_parent_prob = np.mean([parent_probs.get(p, 0.5) for p in non_parents]) if non_parents else 0.5
    
    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'num_variables': len(variables),
        'num_true_parents': len(true_parents),
        'predicted_parents': list(predicted_parents),
        'true_parents': list(true_parents),
        'parent_probabilities': parent_probs,
        'avg_true_parent_prob': avg_true_parent_prob,
        'avg_non_parent_prob': avg_non_parent_prob,
        'structure_type': scm.get('metadata', {}).get('structure_type', 'unknown')
    }


def evaluate_transfer(
    trained_on: str,
    test_on: str,
    num_vars: int,
    num_episodes: int = 15,
    initial_observations: int = 80,
    initial_interventions: int = 40,
    min_parent_coefficient: float = 0.5,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """Evaluate a model trained on one graph type on another graph type.
    
    Args:
        trained_on: Model name (e.g., 'fork', 'chain', 'star')
        test_on: Test graph type display name (e.g., 'true_fork', 'star')
    """
    
    if verbose:
        print(f"\nEvaluating {trained_on}-trained model on {test_on} graphs with {num_vars} variables")
    
    # Load surrogate model
    model_config = MODEL_CONFIGS[trained_on]
    model_path = Path(model_config['path'])
    
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}")
        return None
    
    surrogate_net, surrogate_params, metadata = load_surrogate_model(model_path)
    
    # Convert test_on display name to API structure type
    api_structure_type = GRAPH_TYPE_TO_API.get(test_on, test_on)
    
    # Create SCM factory
    factory = VariableSCMFactory(
        seed=seed,
        noise_scale=0.5,
        coefficient_range=(-3.0, 3.0),
        min_parent_coefficient=min_parent_coefficient,
        vary_intervention_ranges=True,
        use_output_bounds=True
    )
    
    # Run episodes
    episode_results = []
    for episode_idx in range(num_episodes):
        # Create SCM using API structure type
        scm = factory.create_variable_scm(
            num_variables=num_vars,
            structure_type=api_structure_type
        )
        
        # Evaluate episode
        result = evaluate_single_episode(
            surrogate_net,
            surrogate_params,
            scm,
            initial_observations=initial_observations,
            initial_interventions=initial_interventions,
            seed=seed + episode_idx * 1000
        )
        
        episode_results.append(result)
        
        if verbose and (episode_idx + 1) % 5 == 0:
            avg_f1 = np.mean([r['f1_score'] for r in episode_results])
            print(f"  Episodes {episode_idx + 1}/{num_episodes}: Avg F1 = {avg_f1:.3f}")
    
    # Aggregate results
    aggregate = {
        'trained_on': trained_on,
        'tested_on': test_on,
        'num_vars': num_vars,
        'num_episodes': num_episodes,
        'initial_observations': initial_observations,
        'initial_interventions': initial_interventions,
        'min_parent_coefficient': min_parent_coefficient,
        'f1_mean': np.mean([r['f1_score'] for r in episode_results]),
        'f1_std': np.std([r['f1_score'] for r in episode_results]),
        'precision_mean': np.mean([r['precision'] for r in episode_results]),
        'precision_std': np.std([r['precision'] for r in episode_results]),
        'recall_mean': np.mean([r['recall'] for r in episode_results]),
        'recall_std': np.std([r['recall'] for r in episode_results]),
        'avg_true_parent_prob_mean': np.mean([r['avg_true_parent_prob'] for r in episode_results]),
        'avg_non_parent_prob_mean': np.mean([r['avg_non_parent_prob'] for r in episode_results]),
        'episodes': episode_results,
        'metadata': metadata
    }
    
    return aggregate


def run_full_transfer_evaluation(
    output_dir: Path,
    trained_models: List[str] = None,
    test_graphs: List[str] = None,
    graph_sizes: List[int] = None,
    **kwargs
):
    """Run complete transfer learning evaluation across all combinations."""
    
    if trained_models is None:
        trained_models = list(MODEL_CONFIGS.keys())
    if test_graphs is None:
        test_graphs = TEST_GRAPH_TYPES
    if graph_sizes is None:
        graph_sizes = TEST_GRAPH_SIZES
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create summary for all results
    all_results = []
    
    # Total evaluations
    total_evals = len(trained_models) * len(test_graphs) * len(graph_sizes)
    pbar = tqdm(total=total_evals, desc="Transfer evaluations")
    
    for trained_on in trained_models:
        for test_on in test_graphs:
            for num_vars in graph_sizes:
                # Skip if model doesn't exist
                model_path = Path(MODEL_CONFIGS[trained_on]['path'])
                if not model_path.exists():
                    print(f"\nSkipping {trained_on} (model not found)")
                    pbar.update(1)
                    continue
                
                # Run evaluation (test_on is the display name, handled in evaluate_transfer)
                result = evaluate_transfer(
                    trained_on=trained_on,
                    test_on=test_on,
                    num_vars=num_vars,
                    verbose=False,
                    **kwargs
                )
                
                if result:
                    # Save individual result
                    result_dir = output_dir / trained_on / test_on
                    result_dir.mkdir(parents=True, exist_ok=True)
                    
                    result_file = result_dir / f"eval_{num_vars}vars_{timestamp}.json"
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
                    
                    # Add to summary
                    all_results.append({
                        'trained_on': trained_on,
                        'tested_on': test_on,
                        'num_vars': num_vars,
                        'f1_mean': result['f1_mean'],
                        'f1_std': result['f1_std'],
                        'precision_mean': result['precision_mean'],
                        'recall_mean': result['recall_mean']
                    })
                    
                    pbar.set_postfix({
                        'Current': f"{trained_on}->{test_on}",
                        'Vars': num_vars,
                        'F1': f"{result['f1_mean']:.3f}"
                    })
                
                pbar.update(1)
    
    pbar.close()
    
    # Save summary
    summary_file = output_dir / f"transfer_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'configurations': {
                'trained_models': trained_models,
                'test_graphs': test_graphs,
                'graph_sizes': graph_sizes
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✅ Transfer evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    # Print quick summary matrix for smallest graph size
    print("\n📊 Transfer Matrix (F1 scores for 3 variables):")
    print(f"{'Trained on':<12} | " + " | ".join([f"{g:<10}" for g in test_graphs]))
    print("-" * (13 + 13 * len(test_graphs)))
    
    for trained in trained_models:
        row = f"{trained:<12} |"
        for test in test_graphs:
            result = next((r for r in all_results 
                          if r['trained_on'] == trained and r['tested_on'] == test and r['num_vars'] == 3), None)
            if result:
                row += f" {result['f1_mean']:>10.3f} |"
            else:
                row += f" {'N/A':>10} |"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Transfer Learning Evaluation")
    
    # Model selection
    parser.add_argument('--trained-models', nargs='+', 
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Models to evaluate (default: all)')
    parser.add_argument('--test-graphs', nargs='+',
                       choices=TEST_GRAPH_TYPES,
                       help='Graph types to test on (default: all)')
    parser.add_argument('--graph-sizes', nargs='+', type=int,
                       default=[3, 8, 15, 30, 100],
                       help='Graph sizes to test')
    
    # Evaluation parameters
    parser.add_argument('--num-episodes', type=int, default=15,
                       help='Number of episodes per configuration')
    parser.add_argument('--initial-observations', type=int, default=80,
                       help='Number of initial observational samples')
    parser.add_argument('--initial-interventions', type=int, default=40,
                       help='Number of initial random interventions')
    parser.add_argument('--min-parent-coefficient', type=float, default=0.5,
                       help='Minimum parent coefficient magnitude')
    
    # Output
    parser.add_argument('--output-dir', type=Path, 
                       default=Path('thesis_results/transfer_learning'),
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Single evaluation mode
    parser.add_argument('--single', action='store_true',
                       help='Run single evaluation (requires --trained-on and --test-on)')
    parser.add_argument('--trained-on', choices=list(MODEL_CONFIGS.keys()),
                       help='Model trained on this graph type')
    parser.add_argument('--test-on', choices=TEST_GRAPH_TYPES,
                       help='Test on this graph type')
    parser.add_argument('--num-vars', type=int, default=8,
                       help='Number of variables for single evaluation')
    
    args = parser.parse_args()
    
    if args.single:
        if not args.trained_on or not args.test_on:
            print("Error: --single mode requires --trained-on and --test-on")
            return 1
        
        # Run single evaluation (test_on is already the display name)
        result = evaluate_transfer(
            trained_on=args.trained_on,
            test_on=args.test_on,
            num_vars=args.num_vars,
            num_episodes=args.num_episodes,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            min_parent_coefficient=args.min_parent_coefficient,
            seed=args.seed,
            verbose=True
        )
        
        if result:
            print(f"\n📊 Results:")
            print(f"  F1 Score: {result['f1_mean']:.3f} ± {result['f1_std']:.3f}")
            print(f"  Precision: {result['precision_mean']:.3f} ± {result['precision_std']:.3f}")
            print(f"  Recall: {result['recall_mean']:.3f} ± {result['recall_std']:.3f}")
            
            # Save result in the same nested structure as batch mode
            # Format: output_dir/trained_on/tested_on/eval_Nvars_timestamp.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create nested directory structure
            result_dir = args.output_dir / args.trained_on / args.test_on
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Use same naming convention as batch mode
            result_file = result_dir / f"eval_{args.num_vars}vars_{timestamp}.json"
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
            print(f"\n💾 Result saved to: {result_file}")
    else:
        # Run full evaluation
        run_full_transfer_evaluation(
            output_dir=args.output_dir,
            trained_models=args.trained_models,
            test_graphs=args.test_graphs,
            graph_sizes=args.graph_sizes,
            num_episodes=args.num_episodes,
            initial_observations=args.initial_observations,
            initial_interventions=args.initial_interventions,
            min_parent_coefficient=args.min_parent_coefficient,
            seed=args.seed
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())