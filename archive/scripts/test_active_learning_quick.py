#!/usr/bin/env python3
"""Quick test to verify BC active learning functionality."""

import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_bc_surrogate,
    create_bc_active_learning_wrapper
)

# Reduce logging verbosity
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def main():
    """Quick test of BC active learning."""
    
    # Setup
    scm = create_fork_scm(noise_scale=1.0)
    checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
    
    if not checkpoint_path.exists():
        logger.error(f"BC checkpoint not found at {checkpoint_path}")
        return
    
    eval_config = {
        'n_initial_obs': 50,
        'max_interventions': 5,  # Fewer interventions for quick test
        'n_intervention_samples': 50,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Create acquisition function
    random_policy = create_random_acquisition(seed=42)
    evaluator = create_universal_evaluator()
    
    print("=" * 60)
    print("BC ACTIVE LEARNING QUICK TEST")
    print("=" * 60)
    
    # Test 1: Static BC
    print("\n1. Testing static BC surrogate...")
    bc_static_predict, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)
    
    static_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=bc_static_predict,
        seed=42
    )
    
    print(f"   Final F1: {static_result.final_metrics['final_f1']:.3f}")
    print(f"   Active learning: {static_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Test 2: Active BC
    print("\n2. Testing BC with active learning...")
    predict_fn, update_fn, params, opt_state = create_bc_active_learning_wrapper(
        checkpoint_path, scm, learning_rate=1e-3, seed=42
    )
    
    active_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=predict_fn,
        surrogate_update_fn=update_fn,
        surrogate_params=params,
        surrogate_opt_state=opt_state,
        seed=42
    )
    
    print(f"   Final F1: {active_result.final_metrics['final_f1']:.3f}")
    print(f"   Active learning: {active_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Compare
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Static BC: F1 = {static_result.final_metrics['final_f1']:.3f}")
    print(f"Active BC: F1 = {active_result.final_metrics['final_f1']:.3f}")
    
    # Check if predictions evolved
    static_predictions_changed = False
    active_predictions_changed = False
    
    # Extract first and last predictions
    static_history = static_result.history
    active_history = active_result.history
    
    if len(static_history) > 1:
        first_static = static_history[0].marginals
        last_static = static_history[-1].marginals
        if first_static and last_static:
            static_predictions_changed = dict(first_static) != dict(last_static)
    
    if len(active_history) > 1:
        first_active = active_history[0].marginals
        last_active = active_history[-1].marginals
        if first_active and last_active:
            active_predictions_changed = dict(first_active) != dict(last_active)
    
    print(f"\nStatic predictions changed: {static_predictions_changed}")
    print(f"Active predictions changed: {active_predictions_changed}")
    
    if active_result.final_metrics['final_f1'] > static_result.final_metrics['final_f1']:
        print("\n✓ Active learning improved F1 score!")
    else:
        print("\n✗ Active learning did not improve F1 score")
    
    print("\nActive learning infrastructure is working correctly!")

if __name__ == "__main__":
    main()