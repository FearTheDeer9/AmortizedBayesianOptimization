#!/usr/bin/env python3
"""Test marginal probability logging to debug why all policies show same F1/SHD."""

import logging
from pathlib import Path

from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_baseline,
    create_optimal_oracle_acquisition,
    create_bc_surrogate
)

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def test_marginals():
    """Test marginal probabilities across different policies."""
    
    # Get test SCM
    scm_name = 'fork'
    scm = create_fork_scm(noise_scale=1.0)
    
    # Create BC surrogate
    surrogate_path = Path('checkpoints/acbo/bc_surrogate/1733589468')
    if not surrogate_path.exists():
        logger.error(f"BC surrogate checkpoint not found at {surrogate_path}")
        return
    
    bc_surrogate_fn, _ = create_bc_surrogate(surrogate_path, allow_updates=False)
    
    # Create policies
    policies = {
        'Random': create_random_baseline(seed=42),
        'Oracle': create_optimal_oracle_acquisition(scm, optimization_direction='MINIMIZE', seed=42)
    }
    
    # Evaluation config
    eval_config = {
        'n_observational': 100,
        'max_interventions': 5,  # Just a few steps
        'n_intervention_samples': 100,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    print("\n" + "="*80)
    print("MARGINAL PROBABILITY COMPARISON")
    print("="*80)
    print(f"\nTesting on SCM: {scm_name}")
    
    # Run evaluation for each policy
    for policy_name, acquisition_fn in policies.items():
        print(f"\n\n{'='*60}")
        print(f"POLICY: {policy_name}")
        print(f"{'='*60}")
        
        # Create evaluator with policy name
        evaluator = create_universal_evaluator()
        evaluator.name = f"{policy_name}+BC"
        
        # Run evaluation
        result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=eval_config,
            surrogate_fn=bc_surrogate_fn,
            seed=42
        )
        
        # Extract final marginals
        final_metrics = result.final_metrics
        print(f"\nFinal F1: {final_metrics.get('final_f1', 0):.3f}")
        print(f"Final SHD: {final_metrics.get('final_shd', 'N/A')}")

if __name__ == "__main__":
    test_marginals()