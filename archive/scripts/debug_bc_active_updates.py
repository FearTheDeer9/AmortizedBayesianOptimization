#!/usr/bin/env python3
"""Debug BC active learning to see if parameters are actually updating."""

import logging
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_bc_surrogate,
    create_bc_active_learning_wrapper
)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def create_simple_fork_scm():
    """Create simple 3-variable fork for testing."""
    return create_simple_linear_scm(
        variables=['X', 'Y', 'Z'],
        edges=[('X', 'Y'), ('Z', 'Y')],
        coefficients={('X', 'Y'): 2.0, ('Z', 'Y'): -1.5},
        noise_scales={'X': 1.0, 'Y': 1.0, 'Z': 1.0},
        target='Y'
    )

def main():
    """Test BC active learning updates."""
    
    # Create simple SCM
    scm = create_simple_fork_scm()
    
    # BC surrogate checkpoint
    checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
    if not checkpoint_path.exists():
        logger.error(f"BC checkpoint not found at {checkpoint_path}")
        return
    
    # Short evaluation for debugging
    eval_config = {
        'n_initial_obs': 50,
        'max_interventions': 5,  # Just 5 interventions
        'n_intervention_samples': 50,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Use random policy
    random_policy = create_random_acquisition(seed=42)
    
    print("=" * 60)
    print("BC ACTIVE LEARNING DEBUG")
    print("=" * 60)
    
    # Test active learning with detailed logging
    print("\nTesting BC with active learning (debug mode)...")
    predict_fn, update_fn, initial_params, initial_opt_state = create_bc_active_learning_wrapper(
        checkpoint_path, scm, learning_rate=1e-2, seed=42  # Higher learning rate
    )
    
    # Create evaluator with custom logging
    evaluator = create_universal_evaluator()
    
    # Evaluate with active learning
    active_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=predict_fn,
        surrogate_update_fn=update_fn,
        surrogate_params=initial_params,
        surrogate_opt_state=initial_opt_state,
        seed=42
    )
    
    print(f"\nActive learning enabled: {active_result.final_metrics.get('active_learning_enabled', False)}")
    print(f"Final F1: {active_result.final_metrics['final_f1']:.3f}")
    
    # Check if predictions changed
    predictions_history = []
    for step in active_result.history:
        if step.marginals:
            predictions_history.append(dict(step.marginals))
    
    if len(predictions_history) > 1:
        print("\nPrediction evolution:")
        for i, preds in enumerate(predictions_history):
            print(f"  Step {i+1}: X={preds.get('X', 0):.3f}, Z={preds.get('Z', 0):.3f}")
        
        # Check if predictions changed
        first = predictions_history[0]
        last = predictions_history[-1]
        x_change = abs(last.get('X', 0) - first.get('X', 0))
        z_change = abs(last.get('Z', 0) - first.get('Z', 0))
        
        print(f"\nPrediction changes:")
        print(f"  X: {x_change:.6f}")
        print(f"  Z: {z_change:.6f}")
        
        if x_change > 0.001 or z_change > 0.001:
            print("  ✓ Active learning is updating predictions!")
        else:
            print("  ✗ Predictions are not changing - active learning may not be working")

if __name__ == "__main__":
    main()