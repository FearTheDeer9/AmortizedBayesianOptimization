#!/usr/bin/env python3
"""
Validate oracle strategy implementation.

This script tests whether the oracle is actually optimal by:
- Testing on simple SCMs with known optimal strategies
- Comparing to true optimal interventions
- Checking value selection strategies
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Set, Any, Tuple, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.data_structures.scm import (
    create_scm, get_parents, get_target, get_variables
)
from src.causal_bayes_opt.mechanisms.linear import (
    create_linear_edge_function, sample_from_linear_scm
)
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.evaluation.model_interfaces import create_oracle_acquisition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_chain_2var():
    """Create X -> Y chain where optimal is to intervene on X."""
    edges = {
        'Y': [('X', create_linear_edge_function(weight=2.0))]
    }
    exogenous = {
        'X': {'mean': 0.0, 'std': 1.0},
        'Y': {'mean': 0.0, 'std': 0.1}
    }
    return create_scm(
        variables=['X', 'Y'],
        edges=edges,
        exogenous=exogenous,
        target='Y'
    )


def create_simple_collider():
    """Create X -> Z <- Y collider where both X and Y affect Z."""
    edges = {
        'Z': [
            ('X', create_linear_edge_function(weight=1.0)),
            ('Y', create_linear_edge_function(weight=-1.0))
        ]
    }
    exogenous = {
        'X': {'mean': 0.0, 'std': 1.0},
        'Y': {'mean': 0.0, 'std': 1.0},
        'Z': {'mean': 0.0, 'std': 0.1}
    }
    return create_scm(
        variables=['X', 'Y', 'Z'],
        edges=edges,
        exogenous=exogenous,
        target='Z'
    )


def compute_intervention_effect(scm, intervention_var, intervention_value, 
                               n_samples=1000, seed=42):
    """Compute expected target value under intervention."""
    intervention = create_perfect_intervention(
        targets=frozenset([intervention_var]),
        values={intervention_var: intervention_value}
    )
    
    samples = sample_with_intervention(scm, intervention, n_samples, seed=seed)
    target_var = get_target(scm)
    
    target_values = []
    for sample in samples:
        values = sample.get('values', {})
        if target_var in values:
            target_values.append(values[target_var])
    
    return np.mean(target_values) if target_values else 0.0


def find_optimal_intervention(scm, intervention_budget=1.0, n_grid=20):
    """Find truly optimal intervention by grid search."""
    target = get_target(scm)
    variables = list(get_variables(scm))
    candidates = [v for v in variables if v != target]
    
    best_var = None
    best_value = None
    best_effect = float('inf')  # We minimize
    
    # Grid search over variables and values
    for var in candidates:
        values = np.linspace(-intervention_budget, intervention_budget, n_grid)
        for val in values:
            effect = compute_intervention_effect(scm, var, float(val))
            if effect < best_effect:
                best_effect = effect
                best_var = var
                best_value = float(val)
    
    return best_var, best_value, best_effect


def test_oracle_on_scm(scm, scm_name, n_tests=20):
    """Test oracle performance on a specific SCM."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing oracle on {scm_name}")
    logger.info(f"{'='*60}")
    
    # Get true optimal intervention
    optimal_var, optimal_value, optimal_effect = find_optimal_intervention(scm)
    logger.info(f"True optimal: Intervene on {optimal_var}={optimal_value:.3f}, effect={optimal_effect:.3f}")
    
    # Get oracle's choice
    target = get_target(scm)
    from src.causal_bayes_opt.data_structures.scm import get_edges
    
    # Build parent structure for oracle
    scm_edges = {}
    edges = get_edges(scm)
    for parent, child in edges:
        if child not in scm_edges:
            scm_edges[child] = []
        scm_edges[child].append(parent)
    
    oracle_fn = create_oracle_acquisition(scm_edges, seed=42)
    
    # Test oracle multiple times
    oracle_vars = []
    oracle_values = []
    oracle_effects = []
    
    for i in range(n_tests):
        # Create dummy tensor (oracle doesn't use it but interface requires it)
        dummy_tensor = jnp.zeros((10, len(get_variables(scm)), 3))
        dummy_posterior = None
        
        # Get oracle's intervention
        intervention = oracle_fn(dummy_tensor, dummy_posterior, target, list(get_variables(scm)))
        
        oracle_var = list(intervention['targets'])[0] if intervention['targets'] else None
        oracle_value = intervention['values'].get(oracle_var, 0.0)
        
        # Compute effect
        if oracle_var:
            effect = compute_intervention_effect(scm, oracle_var, oracle_value, seed=42+i)
            oracle_vars.append(oracle_var)
            oracle_values.append(oracle_value)
            oracle_effects.append(effect)
    
    # Analyze oracle choices
    var_counts = {}
    for var in oracle_vars:
        var_counts[var] = var_counts.get(var, 0) + 1
    
    logger.info(f"\nOracle variable selection over {n_tests} runs:")
    for var, count in var_counts.items():
        logger.info(f"  {var}: {count} times ({100*count/n_tests:.1f}%)")
    
    logger.info(f"\nOracle value statistics:")
    logger.info(f"  Mean: {np.mean(oracle_values):.3f}")
    logger.info(f"  Std: {np.std(oracle_values):.3f}")
    logger.info(f"  Range: [{np.min(oracle_values):.3f}, {np.max(oracle_values):.3f}]")
    
    logger.info(f"\nOracle effect statistics:")
    logger.info(f"  Mean: {np.mean(oracle_effects):.3f}")
    logger.info(f"  Std: {np.std(oracle_effects):.3f}")
    logger.info(f"  Best: {np.min(oracle_effects):.3f}")
    
    # Compare to optimal
    suboptimality = np.mean(oracle_effects) - optimal_effect
    logger.info(f"\nSuboptimality gap: {suboptimality:.3f}")
    
    return {
        'optimal_var': optimal_var,
        'optimal_value': optimal_value,
        'optimal_effect': optimal_effect,
        'oracle_mean_effect': np.mean(oracle_effects),
        'oracle_best_effect': np.min(oracle_effects),
        'suboptimality_gap': suboptimality,
        'oracle_var_distribution': var_counts
    }


def plot_intervention_landscape(scm, output_file):
    """Plot the intervention effect landscape."""
    target = get_target(scm)
    variables = list(get_variables(scm))
    candidates = [v for v in variables if v != target]
    
    n_points = 50
    intervention_range = 2.0
    
    fig, axes = plt.subplots(1, len(candidates), figsize=(5*len(candidates), 4))
    if len(candidates) == 1:
        axes = [axes]
    
    for idx, var in enumerate(candidates):
        values = np.linspace(-intervention_range, intervention_range, n_points)
        effects = []
        
        for val in values:
            effect = compute_intervention_effect(scm, var, float(val))
            effects.append(effect)
        
        ax = axes[idx]
        ax.plot(values, effects, 'b-', linewidth=2)
        ax.set_xlabel(f'Intervention value on {var}')
        ax.set_ylabel(f'E[{target}]')
        ax.set_title(f'Effect of intervening on {var}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark optimal
        optimal_idx = np.argmin(effects)
        ax.plot(values[optimal_idx], effects[optimal_idx], 'r*', markersize=15,
                label=f'Optimal: {values[optimal_idx]:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    logger.info(f"Saved intervention landscape to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate oracle strategy')
    parser.add_argument('--n_tests', type=int, default=50,
                       help='Number of tests per SCM')
    parser.add_argument('--output_dir', type=str, default='oracle_validation',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Test SCMs
    test_scms = [
        ('simple_chain', create_simple_chain_2var()),
        ('simple_collider', create_simple_collider())
    ]
    
    # Also test on standard SCMs
    from src.causal_bayes_opt.experiments.benchmark_scms import (
        create_fork_scm, create_chain_scm, create_collider_scm
    )
    test_scms.extend([
        ('fork', create_fork_scm(noise_scale=0.5)),
        ('chain_3', create_chain_scm(chain_length=3)),
        ('collider', create_collider_scm(noise_scale=0.5))
    ])
    
    results = {}
    
    for scm_name, scm in test_scms:
        # Plot intervention landscape
        plot_file = output_dir / f'{scm_name}_landscape.png'
        plot_intervention_landscape(scm, plot_file)
        
        # Test oracle
        result = test_oracle_on_scm(scm, scm_name, n_tests=args.n_tests)
        results[scm_name] = result
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    scm_names = list(results.keys())
    gaps = [results[name]['suboptimality_gap'] for name in scm_names]
    
    plt.bar(scm_names, gaps)
    plt.xlabel('SCM')
    plt.ylabel('Suboptimality Gap')
    plt.title('Oracle Suboptimality Across Different SCMs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'oracle_suboptimality.png')
    plt.close()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ORACLE VALIDATION SUMMARY")
    logger.info("="*60)
    
    for scm_name, result in results.items():
        logger.info(f"\n{scm_name}:")
        logger.info(f"  Optimal: {result['optimal_var']}={result['optimal_value']:.3f}")
        logger.info(f"  Oracle mean effect: {result['oracle_mean_effect']:.3f}")
        logger.info(f"  Suboptimality gap: {result['suboptimality_gap']:.3f}")
    
    # Check if oracle is using structure correctly
    logger.info("\n" + "="*60)
    logger.info("ORACLE STRUCTURE USAGE")
    logger.info("="*60)
    
    for scm_name, result in results.items():
        logger.info(f"\n{scm_name}:")
        oracle_vars = result['oracle_var_distribution']
        if len(oracle_vars) == 1:
            logger.info("  Oracle consistently selects same variable (GOOD)")
        else:
            logger.info("  Oracle selects multiple variables (BAD - should be deterministic)")
            for var, count in oracle_vars.items():
                logger.info(f"    {var}: {count} times")


if __name__ == "__main__":
    main()