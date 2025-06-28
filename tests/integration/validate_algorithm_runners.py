#!/usr/bin/env python3
"""
Validate Algorithm Runners

Compare the original algorithm runner with the enhanced history-tracking version
to ensure they produce identical results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend(['external', 'external/parent_scale'])

import numpy as onp
from src.causal_bayes_opt.training.expert_collection.scm_generation import generate_scm
from src.causal_bayes_opt.integration.parent_scale.algorithm_runner import run_full_parent_scale_algorithm
from src.causal_bayes_opt.integration.parent_scale.algorithm_runner_with_history import run_full_parent_scale_algorithm_with_history


def compare_algorithm_outputs():
    """Compare outputs from both algorithm runners."""
    print("🔬 Comparing Algorithm Runners")
    print("=" * 60)
    
    # Generate test SCM
    import jax.random as random
    key = random.PRNGKey(42)
    scm = generate_scm(key=key, n_nodes=4, graph_type='chain')
    print(f"✓ Generated test SCM with {len(scm.get('variables', frozenset()))} nodes")
    
    # Common parameters
    params = {
        'scm': scm,
        'T': 3,  # Short trajectory for testing
        'nonlinear': True,
        'causal_prior': True,
        'individual': False,
        'use_doubly_robust': True,
        'n_observational': 100,
        'n_interventional': 2,
        'seed': 42
    }
    
    print(f"\n📊 Running comparison with parameters:")
    for key, value in params.items():
        if key != 'scm':
            print(f"  - {key}: {value}")
    
    # Run original algorithm
    print(f"\n1️⃣ Running original algorithm...")
    try:
        original_result = run_full_parent_scale_algorithm(**params)
        print(f"  ✓ Original algorithm completed")
        print(f"  - Status: {original_result['status']}")
        print(f"  - Final optimum: {original_result['final_optimum']:.6f}")
        print(f"  - Final posterior size: {len(original_result.get('final_posterior', {}))}")
    except Exception as e:
        print(f"  ❌ Original algorithm failed: {e}")
        return False
    
    # Run enhanced algorithm with history
    print(f"\n2️⃣ Running enhanced algorithm with history...")
    try:
        enhanced_result = run_full_parent_scale_algorithm_with_history(**params)
        print(f"  ✓ Enhanced algorithm completed")
        print(f"  - Status: {enhanced_result['status']}")
        print(f"  - Final optimum: {enhanced_result['final_optimum']:.6f}")
        print(f"  - Final posterior size: {len(enhanced_result.get('final_posterior', {}))}")
        print(f"  - Posterior history states: {len(enhanced_result.get('posterior_history', []))}")
    except Exception as e:
        print(f"  ❌ Enhanced algorithm failed: {e}")
        return False
    
    # Compare results
    print(f"\n🔍 Comparing Results:")
    print(f"=" * 40)
    
    # Compare statuses
    status_match = original_result['status'] == enhanced_result['status']
    print(f"Status match: {'✅' if status_match else '❌'} "
          f"({original_result['status']} vs {enhanced_result['status']})")
    
    # Compare final optimum
    optimum_diff = abs(original_result['final_optimum'] - enhanced_result['final_optimum'])
    optimum_match = optimum_diff < 1e-6
    print(f"Final optimum match: {'✅' if optimum_match else '❌'} "
          f"(diff: {optimum_diff:.8f})")
    
    # Compare final posterior
    orig_posterior = original_result.get('final_posterior', {})
    enh_posterior = enhanced_result.get('final_posterior', {})
    
    posterior_match = compare_posteriors(orig_posterior, enh_posterior)
    print(f"Final posterior match: {'✅' if posterior_match else '❌'}")
    
    # Compare discovered parents
    orig_parents = original_result.get('discovered_parent_set', frozenset())
    enh_parents = enhanced_result.get('discovered_parent_set', frozenset())
    parents_match = orig_parents == enh_parents
    print(f"Discovered parents match: {'✅' if parents_match else '❌'} "
          f"({orig_parents} vs {enh_parents})")
    
    # Verify history capture
    history = enhanced_result.get('posterior_history', [])
    history_valid = len(history) == params['T'] + 1  # Initial + T iterations
    print(f"\nHistory capture validation: {'✅' if history_valid else '❌'}")
    print(f"  - Expected states: {params['T'] + 1}")
    print(f"  - Captured states: {len(history)}")
    
    if history:
        print(f"\n📈 Posterior Evolution:")
        for i, state in enumerate(history):
            n_parent_sets = len(state.get('posterior', {}))
            print(f"  - State {i}: {n_parent_sets} parent sets")
    
    # Overall validation
    all_match = status_match and optimum_match and posterior_match and parents_match and history_valid
    
    print(f"\n{'✅ VALIDATION PASSED' if all_match else '❌ VALIDATION FAILED'}")
    print(f"=" * 60)
    
    return all_match


def compare_posteriors(posterior1, posterior2, tolerance=1e-6):
    """Compare two posterior distributions."""
    # Check same parent sets
    keys1 = set(posterior1.keys())
    keys2 = set(posterior2.keys())
    
    if keys1 != keys2:
        print(f"  - Different parent sets: {keys1 ^ keys2}")
        return False
    
    # Check probabilities match
    for parent_set in keys1:
        prob1 = posterior1[parent_set]
        prob2 = posterior2[parent_set]
        diff = abs(prob1 - prob2)
        
        if diff > tolerance:
            print(f"  - Probability mismatch for {parent_set}: {prob1} vs {prob2} (diff: {diff})")
            return False
    
    return True


if __name__ == "__main__":
    success = compare_algorithm_outputs()
    sys.exit(0 if success else 1)