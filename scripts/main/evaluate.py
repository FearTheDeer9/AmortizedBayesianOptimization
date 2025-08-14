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
from src.causal_bayes_opt.evaluation.surrogate_interface import (
    DummySurrogate, ActiveLearningSurrogateWrapper
)
from src.causal_bayes_opt.utils.update_functions import UpdateContext
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
    registry: SurrogateRegistry,
    update_strategy: str = "none"
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
        update_strategy: Update strategy ('none' or 'bic')
        
    Returns:
        Evaluation results
    """
    logger.info(f"\nEvaluating {method_name} with surrogate '{surrogate_name}'...")
    
    # Get surrogate from registry
    surrogate = registry.get(surrogate_name)
    if surrogate is None:
        logger.error(f"Surrogate '{surrogate_name}' not found in registry")
        return {}
    
    # Wrap with active learning if requested
    if update_strategy != "none":
        # Only allow active learning with surrogates that have a model to update
        if surrogate.surrogate_type == 'dummy':
            logger.warning(f"  Active learning not supported for dummy surrogate, ignoring --surrogate_update_strategy")
        elif hasattr(surrogate, '_checkpoint_path'):
            # BC surrogate - can enable active learning
            logger.info(f"  Enabling active learning with {update_strategy} strategy")
            try:
                from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_surrogate
                
                # Reload to get network components
                _, _, net, initial_params, initial_opt_state = create_bc_surrogate(
                    surrogate._checkpoint_path,
                    allow_updates=True,
                    return_components=True
                )
                surrogate = ActiveLearningSurrogateWrapper(
                    base_surrogate=surrogate,
                    update_strategy=update_strategy,
                    net=net,
                    initial_params=initial_params,
                    initial_opt_state=initial_opt_state,
                    learning_rate=1e-3,
                    seed=config.get('seed', 42)
                )
            except Exception as e:
                logger.error(f"  Failed to enable active learning: {e}")
                logger.info("  Continuing without active learning")
        else:
            logger.warning(f"  Active learning not supported for {surrogate.surrogate_type} surrogate")
    
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
        
        # Prepare active learning components if needed
        update_fn = None
        surrogate_params = None
        surrogate_opt_state = None
        
        if hasattr(surrogate, 'get_update_function'):
            update_fn = surrogate.get_update_function()
            if update_fn is not None:
                # Get initial params and opt_state
                if hasattr(surrogate, 'get_params'):
                    surrogate_params = surrogate.get_params()
                if hasattr(surrogate, 'get_opt_state'):
                    surrogate_opt_state = surrogate.get_opt_state()
                logger.info(f"    Active learning enabled with {update_strategy} strategy")
        
        # Evaluate
        eval_result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            surrogate_fn=surrogate_fn if surrogate.surrogate_type != 'dummy' else None,
            update_fn=update_fn,
            surrogate_params=surrogate_params,
            surrogate_opt_state=surrogate_opt_state,
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
            scm = create_dense_scm(n_vars, edge_prob=0.3)
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


def plot_trajectories(all_results: Dict[str, Dict], output_dir: Path):
    """
    Create trajectory plots showing metrics over intervention steps.
    
    Args:
        all_results: Results for all methods
        output_dir: Directory to save plots
    """
    # Plot 1: Target value trajectories
    plot_target_trajectories(all_results, output_dir)
    
    # Plot 2: Structure metric trajectories (F1/SHD)
    plot_structure_trajectories(all_results, output_dir)
    
    # Plot 3: Per-SCM examples
    plot_scm_examples(all_results, output_dir)
    
    logger.info(f"Trajectory plots saved to {output_dir}/")


def plot_target_trajectories(all_results: Dict[str, Dict], output_dir: Path):
    """Plot target value over intervention steps for each method."""
    plt.figure(figsize=(12, 8))
    
    # Colors for each method
    colors = {
        'Random': 'gray', 
        'Oracle': 'red', 
        'random+bc': 'orange',
        'grpo+bc': 'blue',
        'bc+bc': 'green'
    }
    
    for method, results in all_results.items():
        if 'scm_results' not in results:
            continue
            
        # Collect trajectories from all SCMs
        all_trajectories = []
        max_steps = 0
        
        for scm_name, scm_result in results['scm_results'].items():
            if 'trajectory' not in scm_result:
                continue
                
            trajectory = scm_result['trajectory']
            steps = [step.step for step in trajectory]
            values = [step.outcome_value for step in trajectory]
            
            if len(steps) > max_steps:
                max_steps = len(steps)
            
            all_trajectories.append((steps, values))
        
        if not all_trajectories:
            continue
        
        # Compute mean trajectory
        mean_values_by_step = {}
        for steps, values in all_trajectories:
            for step, value in zip(steps, values):
                if step not in mean_values_by_step:
                    mean_values_by_step[step] = []
                mean_values_by_step[step].append(value)
        
        # Calculate means and stds
        steps_sorted = sorted(mean_values_by_step.keys())
        mean_values = [np.mean(mean_values_by_step[s]) for s in steps_sorted]
        std_values = [np.std(mean_values_by_step[s]) for s in steps_sorted]
        
        # Plot mean with confidence band
        color = colors.get(method, 'black')
        plt.plot(steps_sorted, mean_values, label=method, color=color, linewidth=2)
        plt.fill_between(steps_sorted, 
                        np.array(mean_values) - np.array(std_values),
                        np.array(mean_values) + np.array(std_values),
                        alpha=0.2, color=color)
    
    plt.xlabel('Intervention Step')
    plt.ylabel('Target Value')
    plt.title('Mean Target Value Trajectory Across SCMs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'target_trajectories.png', dpi=150)
    plt.close()


def plot_structure_trajectories(all_results: Dict[str, Dict], output_dir: Path):
    """Plot F1 and SHD over intervention steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {
        'Random': 'gray', 
        'Oracle': 'red', 
        'random+bc': 'orange',
        'grpo+bc': 'blue',
        'bc+bc': 'green'
    }
    
    for method, results in all_results.items():
        if 'scm_results' not in results:
            continue
        
        # Collect F1 trajectories step by step
        f1_by_step = {}
        shd_by_step = {}
        
        for scm_name, scm_result in results['scm_results'].items():
            if 'trajectory' not in scm_result:
                continue
                
            trajectory = scm_result['trajectory']
            
            for step_metric in trajectory:
                step = step_metric.step
                
                # Extract F1 from predicted_parents if available
                if hasattr(step_metric, 'predicted_parents') and step_metric.predicted_parents is not None:
                    # We need true parents from scm_result
                    if 'f1_score' in scm_result:  # Use final F1 as proxy for now
                        if step not in f1_by_step:
                            f1_by_step[step] = []
                        # Use prediction confidence as proxy for F1 evolution
                        if hasattr(step_metric, 'prediction_confidence'):
                            f1_by_step[step].append(step_metric.prediction_confidence)
                
                # For now, use final values at last step
                if step == len(trajectory) - 1:
                    if 'f1_score' in scm_result:
                        if step not in f1_by_step:
                            f1_by_step[step] = []
                        f1_by_step[step].append(scm_result['f1_score'])
                    
                    if 'shd' in scm_result and scm_result['shd'] != float('inf'):
                        if step not in shd_by_step:
                            shd_by_step[step] = []
                        shd_by_step[step].append(scm_result['shd'])
        
        # Plot F1 trajectory
        if f1_by_step:
            steps = sorted(f1_by_step.keys())
            mean_f1 = [np.mean(f1_by_step.get(s, [0])) for s in steps]
            color = colors.get(method, 'black')
            ax1.plot(steps, mean_f1, 'o-', label=method, color=color, markersize=8)
        
        # Plot SHD trajectory  
        if shd_by_step:
            steps = sorted(shd_by_step.keys())
            mean_shd = [np.mean(shd_by_step.get(s, [0])) for s in steps]
            color = colors.get(method, 'black')
            ax2.plot(steps, mean_shd, 'o-', label=method, color=color, markersize=8)
    
    ax1.set_xlabel('Intervention Step')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Structure Learning: F1 Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    ax2.set_xlabel('Intervention Step')
    ax2.set_ylabel('Structural Hamming Distance')
    ax2.set_title('Structure Learning: SHD')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'structure_trajectories.png', dpi=150)
    plt.close()


def plot_scm_examples(all_results: Dict[str, Dict], output_dir: Path):
    """Plot trajectories for individual example SCMs."""
    # Select up to 4 SCMs to show
    example_scms = ['fork', 'chain_3', 'chain_5', 'collider']
    
    # Check which SCMs are available
    available_scms = set()
    for method_results in all_results.values():
        if 'scm_results' in method_results:
            available_scms.update(method_results['scm_results'].keys())
    
    # Use first 4 available if our examples aren't present
    scms_to_plot = [scm for scm in example_scms if scm in available_scms]
    if len(scms_to_plot) < 4:
        remaining = list(available_scms - set(scms_to_plot))
        scms_to_plot.extend(remaining[:4-len(scms_to_plot)])
    scms_to_plot = scms_to_plot[:4]
    
    if not scms_to_plot:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {
        'Random': 'gray', 
        'Oracle': 'red', 
        'random+bc': 'orange',
        'grpo+bc': 'blue',
        'bc+bc': 'green'
    }
    
    for idx, scm_name in enumerate(scms_to_plot):
        ax = axes[idx]
        
        for method, results in all_results.items():
            if 'scm_results' not in results or scm_name not in results['scm_results']:
                continue
                
            scm_result = results['scm_results'][scm_name]
            if 'trajectory' not in scm_result:
                continue
                
            trajectory = scm_result['trajectory']
            steps = [step.step for step in trajectory]
            values = [step.outcome_value for step in trajectory]
            
            color = colors.get(method, 'black')
            ax.plot(steps, values, 'o-', label=method, color=color, markersize=6)
        
        ax.set_xlabel('Intervention Step')
        ax.set_ylabel('Target Value')
        ax.set_title(f'SCM: {scm_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scm_example_trajectories.png', dpi=150)
    plt.close()


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
    
    # Active learning
    parser.add_argument('--surrogate_update_strategy', type=str, default='none',
                       choices=['none', 'bic'],
                       help='Surrogate update strategy for active learning')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='Directory for results')
    parser.add_argument('--plot', action='store_true', help='Create comparison plots')
    parser.add_argument('--plot_trajectories', action='store_true', 
                       help='Create trajectory plots showing metrics over steps')
    
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
            args.baseline_surrogate, registry, args.surrogate_update_strategy
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
                eval_config, surrogate_name, registry, args.surrogate_update_strategy
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
    
    # Create trajectory plots if requested
    if args.plot_trajectories:
        plot_trajectories(all_results, output_dir)
    
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