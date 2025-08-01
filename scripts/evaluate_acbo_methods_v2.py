#!/usr/bin/env python3
"""
Refactored ACBO evaluation script with clean surrogate management.

This script evaluates different acquisition methods paired with various surrogates
using a principled registry-based approach.
"""

import argparse
import logging
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_bc_acquisition,
    create_random_acquisition,
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.evaluation.surrogate_registry import (
    SurrogateRegistry, get_registry, register_surrogate, get_surrogate
)
from src.causal_bayes_opt.evaluation.surrogate_interface import DummySurrogate
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm,
    create_chain_scm,
    create_collider_scm,
    create_dense_scm
)
from src.causal_bayes_opt.data_structures.scm import get_parents, get_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_method(
    method_name: str,
    acquisition_fn: callable,
    scms: List[tuple],
    config: Dict[str, Any],
    surrogate_name: str,
    registry: SurrogateRegistry
) -> Dict[str, Any]:
    """
    Evaluate a single method with a specific surrogate on all test SCMs.
    
    Args:
        method_name: Name of the method
        acquisition_fn: Acquisition function to evaluate
        scms: List of (name, scm) tuples
        config: Evaluation configuration
        surrogate_name: Name of surrogate in registry
        registry: SurrogateRegistry instance
        
    Returns:
        Evaluation results
    """
    logger.info(f"\nEvaluating {method_name} with surrogate '{surrogate_name}'...")
    
    # Get surrogate from registry
    surrogate = registry.get(surrogate_name)
    if surrogate is None:
        logger.error(f"Surrogate '{surrogate_name}' not found in registry")
        return {}
    
    logger.info(f"  Using {surrogate.name} (type: {surrogate.surrogate_type})")
    
    evaluator = create_universal_evaluator()
    results = {
        'method': method_name,
        'surrogate': surrogate_name,
        'surrogate_type': surrogate.surrogate_type,
        'scm_results': {},
        'aggregate_metrics': {}
    }
    
    all_improvements = []
    all_f1_scores = []
    all_final_values = []
    
    for scm_name, scm in scms:
        logger.info(f"  Testing on {scm_name}...")
        
        # Create surrogate predict function that matches evaluator expectations
        def surrogate_fn(tensor, target, variables):
            return surrogate.predict(tensor, target, variables)
        
        # Evaluate
        eval_result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            surrogate_fn=surrogate_fn if surrogate.surrogate_type != 'dummy' else None,
            seed=config['seed']
        )
        
        if eval_result.success:
            metrics = eval_result.final_metrics
            results['scm_results'][scm_name] = {
                'improvement': metrics['improvement'],
                'final_f1': metrics.get('final_f1', 0.0),
                'final_value': metrics['final_value'],
                'trajectory': eval_result.history
            }
            
            all_improvements.append(metrics['improvement'])
            all_f1_scores.append(metrics.get('final_f1', 0.0))
            all_final_values.append(metrics['final_value'])
    
    # Aggregate metrics
    if all_improvements:
        results['aggregate_metrics'] = {
            'mean_improvement': float(np.mean(all_improvements)),
            'std_improvement': float(np.std(all_improvements)),
            'mean_f1': float(np.mean(all_f1_scores)),
            'mean_final_value': float(np.mean(all_final_values))
        }
    
    logger.info(f"  {method_name} mean improvement: {results['aggregate_metrics'].get('mean_improvement', 0):.3f}")
    logger.info(f"  {method_name} mean F1 score: {results['aggregate_metrics'].get('mean_f1', 0):.3f}")
    
    return results


def create_test_scm_set(n_scms: int = 10, seed: int = 42) -> List[tuple]:
    """Create a diverse set of test SCMs."""
    test_scms = []
    
    # Add specific benchmark SCMs
    test_scms.append(('fork', create_fork_scm()))
    test_scms.append(('chain_3', create_chain_scm(3)))
    test_scms.append(('chain_5', create_chain_scm(5)))
    test_scms.append(('collider', create_collider_scm()))
    
    # Add more if requested
    if n_scms > 4:
        # Create additional dense SCMs with varying sizes
        for i in range(n_scms - 4):
            n_vars = 4 + (i % 3)  # Vary between 4-6 variables
            scm = create_dense_scm(n_vars, edge_prob=0.3, seed=seed + i)
            test_scms.append((f'dense_{n_vars}_{i}', scm))
    
    logger.info(f"Created {len(test_scms)} test SCMs")
    return test_scms[:n_scms]


def plot_comparison_results(all_results: Dict[str, Dict], output_dir: Path):
    """Create comparison plots for all evaluated methods."""
    # Implementation same as before, but cleaner
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    methods = []
    improvements = []
    f1_scores = []
    
    for key, result in all_results.items():
        if 'aggregate_metrics' in result:
            methods.append(key)
            improvements.append(result['aggregate_metrics']['mean_improvement'])
            f1_scores.append(result['aggregate_metrics']['mean_f1'])
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Improvement plot
    ax1.bar(methods, improvements)
    ax1.set_ylabel('Mean Improvement')
    ax1.set_title('Target Value Improvement by Method')
    ax1.tick_params(axis='x', rotation=45)
    
    # F1 score plot
    ax2.bar(methods, f1_scores)
    ax2.set_ylabel('Mean F1 Score')
    ax2.set_title('Structure Learning Performance')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'method_comparison.png', dpi=150)
    plt.close()
    
    logger.info(f"Comparison plots saved to {output_dir}/")


def main():
    """Main evaluation script with clean surrogate management."""
    parser = argparse.ArgumentParser(description='Evaluate ACBO methods with clean surrogate management')
    
    # Surrogate registration
    parser.add_argument('--register_surrogate', nargs=2, action='append', metavar=('NAME', 'PATH'),
                       help='Register a surrogate: NAME PATH_OR_TYPE (can be used multiple times)')
    
    # Policy registration  
    parser.add_argument('--register_policy', nargs=2, action='append', metavar=('NAME', 'PATH'),
                       help='Register a policy model: NAME PATH (can be used multiple times)')
    
    # Evaluation pairs
    parser.add_argument('--evaluate_pairs', nargs=2, action='append', metavar=('POLICY', 'SURROGATE'),
                       help='Evaluate policy-surrogate pair: POLICY_NAME SURROGATE_NAME')
    
    # Built-in baselines
    parser.add_argument('--include_baselines', action='store_true',
                       help='Include Random and Oracle baselines')
    parser.add_argument('--baseline_surrogate', type=str, default='dummy',
                       help='Surrogate to use for baselines')
    
    # Evaluation parameters
    parser.add_argument('--n_scms', type=int, default=10, help='Number of test SCMs')
    parser.add_argument('--n_obs', type=int, default=100, help='Initial observations')
    parser.add_argument('--n_interventions', type=int, default=20, help='Number of interventions')
    parser.add_argument('--n_samples', type=int, default=10, help='Samples per intervention')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='Directory for results')
    parser.add_argument('--plot', action='store_true', help='Create comparison plots')
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = get_registry()
    
    # Register surrogates
    if args.register_surrogate:
        for name, path_or_type in args.register_surrogate:
            try:
                # Handle special cases cleanly
                if path_or_type.lower() in ['dummy', 'none']:
                    registry.register(name, 'dummy')
                elif path_or_type.lower() == 'active_learning':
                    registry.register(name, 'active_learning')
                else:
                    # Assume it's a path
                    registry.register(name, Path(path_or_type))
            except Exception as e:
                logger.error(f"Failed to register surrogate '{name}': {e}")
    
    # Register built-in policies
    policy_registry = {
        'random': create_random_acquisition(seed=args.seed)
    }
    
    # Register custom policies
    if args.register_policy:
        for name, path in args.register_policy:
            try:
                checkpoint_path = Path(path)
                # Try to detect policy type from checkpoint
                if 'grpo' in path.lower():
                    policy_registry[name] = create_grpo_acquisition(checkpoint_path, seed=args.seed)
                else:
                    policy_registry[name] = create_bc_acquisition(checkpoint_path, seed=args.seed)
                logger.info(f"Registered policy '{name}' from {path}")
            except Exception as e:
                logger.error(f"Failed to register policy '{name}': {e}")
    
    # Create evaluation config
    eval_config = {
        'n_observational': args.n_obs,
        'max_interventions': args.n_interventions,
        'n_intervention_samples': args.n_samples,
        'optimization_direction': 'MINIMIZE',
        'seed': args.seed
    }
    
    # Create test SCMs
    test_scms = create_test_scm_set(args.n_scms, args.seed)
    
    # Evaluate all requested pairs
    all_results = {}
    
    # Include baselines if requested
    if args.include_baselines:
        # Ensure baseline surrogate is registered
        if not registry.get(args.baseline_surrogate):
            registry.register(args.baseline_surrogate, 'dummy')
        
        # Random baseline
        random_results = evaluate_method(
            'Random', policy_registry['random'], test_scms, eval_config,
            args.baseline_surrogate, registry
        )
        all_results['Random'] = random_results
        
        # Oracle baseline (special handling as it's created per SCM)
        oracle_results = {'method': 'Oracle', 'scm_results': {}, 'aggregate_metrics': {}}
        improvements = []
        f1_scores = []
        
        for scm_name, scm in test_scms:
            oracle_fn = create_optimal_oracle_acquisition(scm, optimization_direction='MINIMIZE', seed=args.seed)
            evaluator = create_universal_evaluator()
            
            # Get surrogate
            surrogate = registry.get(args.baseline_surrogate)
            surrogate_fn = (lambda t, tgt, v: surrogate.predict(t, tgt, v)) if surrogate else None
            
            eval_result = evaluator.evaluate(
                acquisition_fn=oracle_fn,
                scm=scm,
                config=eval_config,
                surrogate_fn=surrogate_fn,
                seed=args.seed
            )
            
            if eval_result.success:
                metrics = eval_result.final_metrics
                oracle_results['scm_results'][scm_name] = {
                    'improvement': metrics['improvement'],
                    'final_f1': metrics.get('final_f1', 0.0),
                    'trajectory': eval_result.history
                }
                improvements.append(metrics['improvement'])
                f1_scores.append(metrics.get('final_f1', 0.0))
        
        if improvements:
            oracle_results['aggregate_metrics'] = {
                'mean_improvement': float(np.mean(improvements)),
                'mean_f1': float(np.mean(f1_scores))
            }
        
        all_results['Oracle'] = oracle_results
    
    # Evaluate requested pairs
    if args.evaluate_pairs:
        for policy_name, surrogate_name in args.evaluate_pairs:
            if policy_name not in policy_registry:
                logger.error(f"Policy '{policy_name}' not found")
                continue
                
            pair_name = f"{policy_name}+{surrogate_name}"
            results = evaluate_method(
                pair_name, policy_registry[policy_name], test_scms,
                eval_config, surrogate_name, registry
            )
            all_results[pair_name] = results
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to JSON-serializable format
    def make_serializable(obj):
        """Convert numpy/jax arrays and dataclasses to serializable format."""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # Handle dataclasses/objects
            return {k: make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        else:
            return obj
    
    serializable_results = make_serializable(all_results)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create plots if requested
    if args.plot:
        plot_comparison_results(all_results, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    # List registered surrogates
    logger.info("\nRegistered Surrogates:")
    for name, info in registry.list_registered().items():
        logger.info(f"  {name}: {info}")
    
    # Results summary
    logger.info("\nResults Summary:")
    for method, results in all_results.items():
        if 'aggregate_metrics' in results:
            metrics = results['aggregate_metrics']
            logger.info(f"  {method}:")
            logger.info(f"    Mean improvement: {metrics.get('mean_improvement', 0):.3f}")
            logger.info(f"    Mean F1 score: {metrics.get('mean_f1', 0):.3f}")
    
    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()