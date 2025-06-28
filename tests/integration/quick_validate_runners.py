#!/usr/bin/env python3
"""
Quick validation of algorithm runners
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend(['external', 'external/parent_scale'])

import jax.random as random
from src.causal_bayes_opt.training.expert_collection.scm_generation import generate_scm
from src.causal_bayes_opt.integration.parent_scale.algorithm_runner import run_full_parent_scale_algorithm
from src.causal_bayes_opt.integration.parent_scale.algorithm_runner_with_history import run_full_parent_scale_algorithm_with_history


def quick_validation():
    """Quick validation of both runners."""
    print("🔬 Quick Algorithm Runner Validation")
    print("=" * 50)
    
    # Simple 3-node SCM
    key = random.PRNGKey(42)
    scm = generate_scm(key=key, n_nodes=3, graph_type='chain')
    
    # Minimal parameters
    params = {
        'scm': scm,
        'T': 2,  # Very short
        'nonlinear': True,
        'causal_prior': True,
        'n_observational': 50,
        'n_interventional': 2,
        'seed': 42
    }
    
    print("Running original algorithm...")
    orig = run_full_parent_scale_algorithm(**params)
    print(f"✓ Original: {orig['status']}, optimum={orig['final_optimum']:.4f}")
    
    print("\nRunning enhanced algorithm...")
    enh = run_full_parent_scale_algorithm_with_history(**params)
    print(f"✓ Enhanced: {enh['status']}, optimum={enh['final_optimum']:.4f}")
    print(f"  History states: {len(enh.get('posterior_history', []))}")
    
    # Quick comparison
    match = abs(orig['final_optimum'] - enh['final_optimum']) < 1e-4
    print(f"\n{'✅ MATCH' if match else '❌ MISMATCH'}")
    
    return match


if __name__ == "__main__":
    success = quick_validation()
    sys.exit(0 if success else 1)