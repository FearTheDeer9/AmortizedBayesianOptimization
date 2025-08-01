#!/usr/bin/env python3
"""
Quick test to check if BC surrogate is working properly with v2 evaluation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO)

from src.causal_bayes_opt.evaluation.surrogate_registry import get_registry
from src.causal_bayes_opt.evaluation.model_interfaces import create_bc_acquisition
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

# Create registry and register surrogate
registry = get_registry()
registry.register('bc_test', Path('checkpoints/test_v2/bc_surrogate_final'))

# Get the surrogate
surrogate = registry.get('bc_test')
print(f"Surrogate type: {surrogate.surrogate_type}")
print(f"Surrogate name: {surrogate.name}")

# Create a simple test
import jax.numpy as jnp
import numpy as np

# Create a test tensor
test_tensor = jnp.zeros((10, 3, 3))
variables = ['X', 'Y', 'Z']
target_var = 'Y'

# Test prediction
try:
    marginals = surrogate.predict(test_tensor, target_var, variables)
    print(f"\nMarginals for variables: {marginals}")
    print(f"Are all probabilities 0.5? {all(v == 0.5 for v in marginals.values())}")
except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()

# Now test full evaluation
print("\n" + "="*60)
print("Testing full evaluation with BC surrogate...")
print("="*60)

scm = create_fork_scm()
bc_policy = create_bc_acquisition(Path('checkpoints/test_v2/bc_final'), seed=42)
evaluator = create_universal_evaluator()

config = {
    'n_observational': 10,
    'max_interventions': 3,
    'n_intervention_samples': 10,
    'optimization_direction': 'MINIMIZE',
    'seed': 42
}

# Create surrogate function
def surrogate_fn(tensor, target, variables):
    return surrogate.predict(tensor, target, variables)

try:
    result = evaluator.evaluate(
        acquisition_fn=bc_policy,
        scm=scm,
        config=config,
        surrogate_fn=surrogate_fn,
        seed=42
    )
    
    print(f"\nEvaluation success: {result.success}")
    if result.success:
        print(f"Final improvement: {result.final_metrics['improvement']:.3f}")
        print(f"Final F1: {result.final_metrics.get('final_f1', 0):.3f}")
    else:
        print(f"Error message: {result.error_message}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()