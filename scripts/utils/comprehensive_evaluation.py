#!/usr/bin/env python3
"""Comprehensive evaluation of surrogate models with different policies."""

import sys
import json
from pathlib import Path
sys.path.append('src')

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from causal_bayes_opt.evaluation.surrogate_registry import get_registry
from causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_optimal_oracle_acquisition
)
from causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm,
    create_chain_scm,
    create_collider_scm,
    create_dense_scm,
    create_sparse_scm
)
import numpy as np

def create_test_scms(n_scms=10):
    """Create diverse test SCMs."""
    scms = []
    
    # Standard structures
    scms.append(('fork', create_fork_scm()))
    scms.append(('chain_3', create_chain_scm(3)))
    scms.append(('chain_5', create_chain_scm(5)))
    scms.append(('collider', create_collider_scm()))
    
    # Add sparse/dense SCMs
    for i in range(n_scms - 4):
        if i % 2 == 0:
            n_vars = 4 + (i // 2)
            scm = create_sparse_scm(n_vars, edge_prob=0.3)
            scms.append((f'sparse_{n_vars}', scm))
        else:
            n_vars = 4 + (i // 2)
            scm = create_dense_scm(n_vars, edge_prob=0.5)
            scms.append((f'dense_{n_vars}', scm))
    
    return scms[:n_scms]

def evaluate_pair(policy_name, policy_fn, surrogate_name, surrogate, scms, config):
    """Evaluate a policy-surrogate pair on all SCMs."""
    evaluator = create_universal_evaluator()
    
    results = {
        'policy': policy_name,
        'surrogate': surrogate_name,
        'scm_f1_scores': {},
        'scm_improvements': {},
        'scm_final_values': {}
    }
    
    for scm_name, scm in scms:
        # Create surrogate function
        if surrogate is not None:
            def surrogate_fn(tensor, target, variables):
                return surrogate.predict(tensor, target, variables)
        else:
            surrogate_fn = None
        
        # For oracle, need to create per-SCM
        if policy_name == 'oracle':
            acquisition_fn = create_optimal_oracle_acquisition(
                scm, optimization_direction='MINIMIZE', seed=config['seed']
            )
        else:
            acquisition_fn = policy_fn
        
        # Evaluate
        eval_result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            surrogate_fn=surrogate_fn,
            seed=config['seed']
        )
        
        if eval_result.success:
            metrics = eval_result.final_metrics
            results['scm_f1_scores'][scm_name] = metrics.get('final_f1', 0.0)
            results['scm_improvements'][scm_name] = metrics.get('improvement', 0.0)
            results['scm_final_values'][scm_name] = metrics.get('final_value', 0.0)
    
    # Calculate aggregates
    f1_scores = list(results['scm_f1_scores'].values())
    improvements = list(results['scm_improvements'].values())
    
    results['mean_f1'] = float(np.mean(f1_scores)) if f1_scores else 0.0
    results['std_f1'] = float(np.std(f1_scores)) if f1_scores else 0.0
    results['mean_improvement'] = float(np.mean(improvements)) if improvements else 0.0
    results['std_improvement'] = float(np.std(improvements)) if improvements else 0.0
    
    return results

def main():
    print("="*80)
    print("COMPREHENSIVE SURROGATE EVALUATION")
    print("="*80)
    
    # Setup
    registry = get_registry()
    
    # Register surrogates
    print("\n1. Registering surrogates...")
    registry.register('dummy', 'dummy')
    registry.register('trained', Path('test_checkpoints/full_training/bc_surrogate_final'))
    
    dummy_surrogate = registry.get('dummy')
    trained_surrogate = registry.get('trained')
    print(f"   ✓ Dummy surrogate: {dummy_surrogate.name}")
    print(f"   ✓ Trained surrogate: {trained_surrogate.name}")
    
    # Create policies
    print("\n2. Creating policies...")
    random_policy = create_random_acquisition(seed=42)
    print("   ✓ Random policy created")
    print("   ✓ Oracle policy (created per SCM)")
    
    # Create test SCMs
    print("\n3. Creating test SCMs...")
    scms = create_test_scms(10)
    for name, scm in scms:
        n_vars = len(scm.variables)
        print(f"   - {name}: {n_vars} variables")
    
    # Evaluation config
    config = {
        'n_observational': 100,
        'max_interventions': 10,
        'n_intervention_samples': 10,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Evaluate all pairs
    print("\n4. Running evaluations...")
    print("   (This may take a few minutes...)")
    
    all_results = {}
    
    # Random + Dummy
    print("\n   Evaluating: Random Policy + Dummy Surrogate")
    results = evaluate_pair('random', random_policy, 'dummy', dummy_surrogate, scms, config)
    all_results['random_dummy'] = results
    
    # Random + Trained
    print("   Evaluating: Random Policy + Trained Surrogate")
    results = evaluate_pair('random', random_policy, 'trained', trained_surrogate, scms, config)
    all_results['random_trained'] = results
    
    # Oracle + Dummy
    print("   Evaluating: Oracle Policy + Dummy Surrogate")
    results = evaluate_pair('oracle', None, 'dummy', dummy_surrogate, scms, config)
    all_results['oracle_dummy'] = results
    
    # Oracle + Trained
    print("   Evaluating: Oracle Policy + Trained Surrogate")
    results = evaluate_pair('oracle', None, 'trained', trained_surrogate, scms, config)
    all_results['oracle_trained'] = results
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Per-pair summary
    for pair_name, results in all_results.items():
        policy = results['policy']
        surrogate = results['surrogate']
        print(f"\n{policy.upper()} POLICY + {surrogate.upper()} SURROGATE:")
        print(f"  Overall F1: {results['mean_f1']:.3f} ± {results['std_f1']:.3f}")
        print(f"  Overall Improvement: {results['mean_improvement']:.3f} ± {results['std_improvement']:.3f}")
        
        print(f"\n  Per-SCM F1 Scores:")
        for scm_name in sorted(results['scm_f1_scores'].keys()):
            f1 = results['scm_f1_scores'][scm_name]
            print(f"    {scm_name:12s}: {f1:.3f}")
    
    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Pair':<25} {'Mean F1':>10} {'Std F1':>10} {'Mean Imp':>10}")
    print("-"*55)
    
    for pair_name, results in all_results.items():
        policy = results['policy']
        surrogate = results['surrogate']
        label = f"{policy} + {surrogate}"
        print(f"{label:<25} {results['mean_f1']:10.3f} {results['std_f1']:10.3f} {results['mean_improvement']:10.3f}")
    
    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Compare dummy vs trained with same policy
    random_improvement = all_results['random_trained']['mean_f1'] - all_results['random_dummy']['mean_f1']
    oracle_improvement = all_results['oracle_trained']['mean_f1'] - all_results['oracle_dummy']['mean_f1']
    
    print(f"\n1. Trained vs Dummy Surrogate:")
    print(f"   - With Random Policy: F1 improvement of {random_improvement:.3f}")
    print(f"   - With Oracle Policy: F1 improvement of {oracle_improvement:.3f}")
    
    # Compare policies with same surrogate
    if all_results['random_trained']['mean_f1'] > 0:
        policy_diff = all_results['oracle_trained']['mean_f1'] - all_results['random_trained']['mean_f1']
        print(f"\n2. Oracle vs Random Policy (with trained surrogate):")
        print(f"   - F1 improvement of {policy_diff:.3f}")
    
    # Save results
    output_file = Path('evaluation_results/comprehensive_results.json')
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n3. Results saved to: {output_file}")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()