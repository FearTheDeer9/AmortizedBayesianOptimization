#!/usr/bin/env python3
"""
Comparison evaluation script for ACBO methods.

This script evaluates and compares different ACBO methods:
- GRPO (trained policy)
- BC (behavioral cloning)
- Random baseline
- Oracle baseline (knows true structure)

Usage:
    python scripts/evaluate_acbo_methods.py --grpo checkpoints/clean_grpo_final --bc checkpoints/clean_bc_final
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import pickle
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
    create_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm,
    create_chain_scm,
    create_collider_scm,
    create_sparse_scm,
    create_dense_scm
)
from src.causal_bayes_opt.data_structures.scm import get_parents, get_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_scms(n_scms: int = 10, seed: int = 42) -> List[Any]:
    """
    Create a diverse set of test SCMs.
    
    Args:
        n_scms: Number of SCMs to create
        seed: Random seed
        
    Returns:
        List of SCM objects
    """
    import random
    random.seed(seed)
    
    scms = []
    
    # Add canonical structures
    scms.append(('fork', create_fork_scm(noise_scale=1.0)))
    scms.append(('chain_3', create_chain_scm(chain_length=3)))
    scms.append(('chain_5', create_chain_scm(chain_length=5)))
    scms.append(('collider', create_collider_scm(noise_scale=1.0)))
    
    # Add random sparse graphs
    for i in range(n_scms - 4):
        n_vars = random.randint(4, 8)
        scm = create_sparse_scm(
            num_vars=n_vars,
            edge_prob=0.3,
            noise_scale=1.0
        )
        scms.append((f'sparse_{n_vars}vars', scm))
    
    return scms


def evaluate_method(
    method_name: str,
    acquisition_fn: callable,
    scms: List[tuple],
    config: Dict[str, Any],
    surrogate_fn: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Evaluate a single method on all test SCMs.
    
    Args:
        method_name: Name of the method
        acquisition_fn: Acquisition function to evaluate
        scms: List of (name, scm) tuples
        config: Evaluation configuration
        surrogate_fn: Optional surrogate function
        
    Returns:
        Evaluation results
    """
    logger.info(f"\nEvaluating {method_name}...")
    
    evaluator = create_universal_evaluator()
    results = {
        'method': method_name,
        'scm_results': {},
        'aggregate_metrics': {}
    }
    
    all_improvements = []
    all_f1_scores = []
    all_final_values = []
    
    for scm_name, scm in scms:
        logger.info(f"  Testing on {scm_name}...")
        
        # Get true parents for oracle
        target = get_target(scm)
        true_parents = list(get_parents(scm, target)) if hasattr(scm, 'edges') else []
        
        # Evaluate
        eval_result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            surrogate_fn=surrogate_fn,
            seed=config['seed']
        )
        
        # Store results
        results['scm_results'][scm_name] = {
            'initial_value': eval_result.final_metrics['initial_value'],
            'best_value': eval_result.final_metrics['best_value'],
            'improvement': eval_result.final_metrics['improvement'],
            'regret': eval_result.final_metrics.get('simple_regret', 0.0),
            'n_unique_interventions': eval_result.final_metrics.get('n_unique_interventions', 0),
            'true_parents': true_parents
        }
        
        # Add structure metrics if available
        if 'final_f1' in eval_result.final_metrics:
            results['scm_results'][scm_name]['f1_score'] = eval_result.final_metrics['final_f1']
            results['scm_results'][scm_name]['shd'] = eval_result.final_metrics.get('final_shd', float('inf'))
            all_f1_scores.append(eval_result.final_metrics['final_f1'])
        
        all_improvements.append(eval_result.final_metrics['improvement'])
        all_final_values.append(eval_result.final_metrics['best_value'])
    
    # Compute aggregate metrics
    results['aggregate_metrics'] = {
        'mean_improvement': np.mean(all_improvements),
        'std_improvement': np.std(all_improvements),
        'mean_final_value': np.mean(all_final_values),
        'mean_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0,
        'n_scms': len(scms)
    }
    
    logger.info(f"  Mean improvement: {results['aggregate_metrics']['mean_improvement']:.3f}")
    if all_f1_scores:
        logger.info(f"  Mean F1 score: {results['aggregate_metrics']['mean_f1_score']:.3f}")
    
    return results


def plot_comparison(all_results: Dict[str, Dict], output_dir: Path):
    """
    Create comparison plots.
    
    Args:
        all_results: Results for all methods
        output_dir: Directory to save plots
    """
    output_dir.mkdir(exist_ok=True)
    
    # Extract data for plotting
    methods = list(all_results.keys())
    mean_improvements = [all_results[m]['aggregate_metrics']['mean_improvement'] for m in methods]
    std_improvements = [all_results[m]['aggregate_metrics']['std_improvement'] for m in methods]
    
    # 1. Bar plot of mean improvements
    plt.figure(figsize=(10, 6))
    x = np.arange(len(methods))
    plt.bar(x, mean_improvements, yerr=std_improvements, capsize=10)
    plt.xticks(x, methods)
    plt.ylabel('Mean Improvement')
    plt.title('ACBO Method Comparison: Target Value Improvement')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_comparison.png', dpi=150)
    plt.close()
    
    # 2. F1 scores if available
    mean_f1s = []
    for m in methods:
        f1 = all_results[m]['aggregate_metrics'].get('mean_f1_score', 0.0)
        mean_f1s.append(f1)
    
    if any(f1 > 0 for f1 in mean_f1s):
        plt.figure(figsize=(10, 6))
        plt.bar(x, mean_f1s)
        plt.xticks(x, methods)
        plt.ylabel('Mean F1 Score')
        plt.title('Structure Learning Performance')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'f1_comparison.png', dpi=150)
        plt.close()
    
    # 3. Per-SCM heatmap
    scm_names = list(next(iter(all_results.values()))['scm_results'].keys())
    improvement_matrix = np.zeros((len(methods), len(scm_names)))
    
    for i, method in enumerate(methods):
        for j, scm in enumerate(scm_names):
            improvement_matrix[i, j] = all_results[method]['scm_results'][scm]['improvement']
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(improvement_matrix, aspect='auto', cmap='RdBu_r')
    plt.colorbar(im, label='Improvement')
    plt.xticks(range(len(scm_names)), scm_names, rotation=45, ha='right')
    plt.yticks(range(len(methods)), methods)
    plt.title('Improvement by Method and SCM Type')
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmap.png', dpi=150)
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}/")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate and compare ACBO methods')
    
    # Checkpoint paths
    parser.add_argument('--grpo', type=str, help='Path to GRPO checkpoint')
    parser.add_argument('--bc', type=str, help='Path to BC checkpoint')
    
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
    
    # Create evaluation config
    eval_config = {
        'n_initial_obs': args.n_obs,
        'max_interventions': args.n_interventions,
        'n_intervention_samples': args.n_samples,
        'optimization_direction': 'MINIMIZE',
        'seed': args.seed
    }
    
    # Create test SCMs
    test_scms = create_test_scms(args.n_scms, args.seed)
    logger.info(f"Created {len(test_scms)} test SCMs")
    
    # Evaluate all methods
    all_results = {}
    
    # Create a simple dummy surrogate for baselines (to test structure learning)
    # This will just return uniform probabilities
    def create_dummy_surrogate():
        def dummy_surrogate(tensor, target_var):
            n_vars = tensor.shape[1]
            # Return uniform probabilities for all non-target variables
            if n_vars == 3 and target_var in ['X', 'Y', 'Z']:
                variables = ['X', 'Y', 'Z']
            else:
                variables = [f'X{i}' for i in range(n_vars)]
            
            marginals = {}
            for var in variables:
                if var != target_var:
                    marginals[var] = 0.5  # Uniform probability
                else:
                    marginals[var] = 0.0
            
            return {
                'marginal_parent_probs': marginals,
                'entropy': 1.0,
                'model_type': 'dummy'
            }
        return dummy_surrogate
    
    dummy_surrogate_fn = create_dummy_surrogate()
    
    # 1. Random baseline (always evaluate)
    logger.info("\n" + "="*60)
    logger.info("Evaluating Random Baseline")
    random_fn = create_random_acquisition(seed=args.seed)
    all_results['Random'] = evaluate_method(
        'Random', random_fn, test_scms, eval_config, surrogate_fn=dummy_surrogate_fn
    )
    
    # 2. Oracle baseline (always evaluate)
    logger.info("\n" + "="*60)
    logger.info("Evaluating Oracle Baseline")
    # Note: Oracle needs to be created per SCM
    oracle_results = {'method': 'Oracle', 'scm_results': {}, 'aggregate_metrics': {}}
    oracle_improvements = []
    oracle_final_values = []
    
    evaluator = create_universal_evaluator()
    for scm_name, scm in test_scms:
        # Get true structure
        target = get_target(scm)
        scm_edges = {}
        # Build parent-child relationships from edges
        from src.causal_bayes_opt.data_structures.scm import get_edges
        edges = get_edges(scm)  # Returns frozenset of (parent, child) tuples
        for parent, child in edges:
            if child not in scm_edges:
                scm_edges[child] = []
            scm_edges[child].append(parent)
        
        # Create oracle for this SCM
        oracle_fn = create_oracle_acquisition(
            scm_edges, 
            seed=args.seed,
            optimization_direction=eval_config['optimization_direction']
        )
        
        # Evaluate with dummy surrogate
        eval_result = evaluator.evaluate(
            acquisition_fn=oracle_fn,
            scm=scm,
            config=eval_config,
            surrogate_fn=dummy_surrogate_fn,
            seed=eval_config['seed']
        )
        
        oracle_results['scm_results'][scm_name] = {
            'initial_value': eval_result.final_metrics['initial_value'],
            'best_value': eval_result.final_metrics['best_value'],
            'improvement': eval_result.final_metrics['improvement'],
            'regret': eval_result.final_metrics.get('simple_regret', 0.0),
            'n_unique_interventions': eval_result.final_metrics.get('n_unique_interventions', 0)
        }
        
        oracle_improvements.append(eval_result.final_metrics['improvement'])
        oracle_final_values.append(eval_result.final_metrics['best_value'])
    
    oracle_results['aggregate_metrics'] = {
        'mean_improvement': np.mean(oracle_improvements),
        'std_improvement': np.std(oracle_improvements),
        'mean_final_value': np.mean(oracle_final_values),
        'n_scms': len(test_scms)
    }
    
    all_results['Oracle'] = oracle_results
    logger.info(f"  Oracle mean improvement: {oracle_results['aggregate_metrics']['mean_improvement']:.3f}")
    
    # 3. GRPO (if checkpoint provided)
    if args.grpo:
        logger.info("\n" + "="*60)
        logger.info("Evaluating GRPO")
        grpo_path = Path(args.grpo)
        if grpo_path.exists():
            grpo_fn = create_grpo_acquisition(grpo_path, seed=args.seed)
            
            # Try to load surrogate from GRPO checkpoint
            grpo_surrogate_fn = None
            try:
                with open(grpo_path / 'checkpoint.pkl', 'rb') as f:
                    checkpoint = pickle.load(f)
                if checkpoint.get('has_surrogate', False):
                    # Import surrogate wrapper
                    from src.causal_bayes_opt.training.continuous_surrogate_integration import (
                        create_continuous_learnable_surrogate,
                        create_surrogate_fn_wrapper
                    )
                    import jax
                    # Create surrogate from checkpoint
                    key = jax.random.PRNGKey(42)
                    net, _, _, _, _ = create_continuous_learnable_surrogate(
                        n_variables=8,  # Max size
                        key=key
                    )
                    surrogate_params = checkpoint.get('surrogate_params')
                    if surrogate_params:
                        grpo_surrogate_fn = create_surrogate_fn_wrapper(net, surrogate_params)
                        logger.info("Loaded surrogate from GRPO checkpoint")
            except Exception as e:
                logger.warning(f"Could not load GRPO surrogate: {e}")
                grpo_surrogate_fn = dummy_surrogate_fn
            
            all_results['GRPO'] = evaluate_method(
                'GRPO', grpo_fn, test_scms, eval_config, 
                surrogate_fn=grpo_surrogate_fn or dummy_surrogate_fn
            )
        else:
            logger.warning(f"GRPO checkpoint not found: {grpo_path}")
    
    # 4. BC (if checkpoint provided)
    if args.bc:
        logger.info("\n" + "="*60)
        logger.info("Evaluating BC")
        bc_path = Path(args.bc)
        if bc_path.exists():
            bc_fn = create_bc_acquisition(bc_path, seed=args.seed)
            
            # Try to load surrogate from BC checkpoint
            bc_surrogate_fn = None
            try:
                with open(bc_path / 'checkpoint.pkl', 'rb') as f:
                    checkpoint = pickle.load(f)
                if checkpoint.get('has_surrogate', False):
                    # Import surrogate wrapper if not already imported
                    try:
                        from src.causal_bayes_opt.training.continuous_surrogate_integration import (
                            create_continuous_learnable_surrogate,
                            create_surrogate_fn_wrapper
                        )
                    except ImportError:
                        pass  # Already imported
                    import jax
                    # Create surrogate from checkpoint
                    key = jax.random.PRNGKey(42)
                    net, _, _, _, _ = create_continuous_learnable_surrogate(
                        n_variables=8,  # Max size
                        key=key
                    )
                    surrogate_params = checkpoint.get('surrogate_params')
                    if surrogate_params:
                        bc_surrogate_fn = create_surrogate_fn_wrapper(net, surrogate_params)
                        logger.info("Loaded surrogate from BC checkpoint")
            except Exception as e:
                logger.warning(f"Could not load BC surrogate: {e}")
                bc_surrogate_fn = dummy_surrogate_fn
                
            all_results['BC'] = evaluate_method(
                'BC', bc_fn, test_scms, eval_config,
                surrogate_fn=bc_surrogate_fn or dummy_surrogate_fn
            )
        else:
            logger.warning(f"BC checkpoint not found: {bc_path}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    # Create plots if requested
    if args.plot:
        plot_comparison(all_results, output_dir)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    
    for method, results in all_results.items():
        metrics = results['aggregate_metrics']
        logger.info(f"\n{method}:")
        logger.info(f"  Mean improvement: {metrics['mean_improvement']:.3f} ± {metrics.get('std_improvement', 0):.3f}")
        logger.info(f"  Mean final value: {metrics['mean_final_value']:.3f}")
        if 'mean_f1_score' in metrics and metrics['mean_f1_score'] > 0:
            logger.info(f"  Mean F1 score: {metrics['mean_f1_score']:.3f}")
    
    # Relative performance
    if 'Random' in all_results and len(all_results) > 1:
        logger.info("\nRelative to Random baseline:")
        random_improvement = all_results['Random']['aggregate_metrics']['mean_improvement']
        for method in all_results:
            if method != 'Random':
                relative = (all_results[method]['aggregate_metrics']['mean_improvement'] / 
                           random_improvement if random_improvement != 0 else float('inf'))
                logger.info(f"  {method}: {relative:.2f}x")


if __name__ == "__main__":
    main()