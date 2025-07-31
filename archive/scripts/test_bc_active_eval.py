#!/usr/bin/env python3
"""Test BC static vs active learning with proper evaluation infrastructure."""

import logging
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_bc_surrogate,
    create_bc_active_learning_wrapper
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    """Compare BC static vs BC with active learning."""
    
    # Setup
    scm = create_fork_scm(noise_scale=1.0)
    checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
    
    if not checkpoint_path.exists():
        logger.error(f"BC checkpoint not found at {checkpoint_path}")
        return
    
    eval_config = {
        'n_initial_obs': 50,
        'max_interventions': 10,
        'n_intervention_samples': 50,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    # Create acquisition function (same for both)
    random_policy = create_random_acquisition(seed=42)
    evaluator = create_universal_evaluator()
    
    logger.info("="*60)
    logger.info("BC STATIC VS ACTIVE LEARNING COMPARISON")
    logger.info("="*60)
    
    # Test 1: BC Static (no updates)
    logger.info("\n1. BC STATIC (no active learning):")
    logger.info("-"*40)
    
    bc_static_predict, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)
    
    static_result = evaluator.evaluate(
        acquisition_fn=random_policy,
        scm=scm,
        config=eval_config,
        surrogate_fn=bc_static_predict,
        seed=42
    )
    
    logger.info(f"\nStatic Results:")
    logger.info(f"  Final F1: {static_result.final_metrics['final_f1']:.3f}")
    logger.info(f"  Final SHD: {static_result.final_metrics['final_shd']}")
    logger.info(f"  Mean trajectory value: {static_result.final_metrics['mean_trajectory_value']:.3f}")
    logger.info(f"  Active learning enabled: {static_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Test 2: BC with active learning
    logger.info("\n\n2. BC ACTIVE (with online updates):")
    logger.info("-"*40)
    
    # Use the wrapper that exposes params/opt_state
    predict_fn, update_fn, initial_params, initial_opt_state = create_bc_active_learning_wrapper(
        checkpoint_path, scm, learning_rate=1e-3, seed=42
    )
    
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
    
    logger.info(f"\nActive Results:")
    logger.info(f"  Final F1: {active_result.final_metrics['final_f1']:.3f}")
    logger.info(f"  Final SHD: {active_result.final_metrics['final_shd']}")
    logger.info(f"  Mean trajectory value: {active_result.final_metrics['mean_trajectory_value']:.3f}")
    logger.info(f"  Active learning enabled: {active_result.final_metrics.get('active_learning_enabled', False)}")
    
    # Compare
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY:")
    logger.info("="*60)
    
    f1_improvement = active_result.final_metrics['final_f1'] - static_result.final_metrics['final_f1']
    shd_improvement = static_result.final_metrics['final_shd'] - active_result.final_metrics['final_shd']
    
    logger.info(f"\nStructure Learning Improvement:")
    logger.info(f"  F1 Score: {static_result.final_metrics['final_f1']:.3f} → {active_result.final_metrics['final_f1']:.3f} "
                f"({'↑' if f1_improvement > 0 else '↓'}{abs(f1_improvement):.3f})")
    logger.info(f"  SHD: {static_result.final_metrics['final_shd']} → {active_result.final_metrics['final_shd']} "
                f"({'↓' if shd_improvement > 0 else '↑'}{abs(shd_improvement)})")
    
    # Analyze prediction evolution
    logger.info("\nPrediction Evolution:")
    
    # Extract marginals from history
    static_marginals_history = []
    active_marginals_history = []
    
    for step in static_result.history:
        if step.marginals:
            static_marginals_history.append((step.step, dict(step.marginals)))
    
    for step in active_result.history:
        if step.marginals:
            active_marginals_history.append((step.step, dict(step.marginals)))
    
    # Show how predictions changed
    if static_marginals_history and active_marginals_history:
        logger.info("\n  Static predictions (first vs last):")
        first_static = static_marginals_history[0][1]
        last_static = static_marginals_history[-1][1]
        for var in sorted(first_static.keys()):
            if var in last_static:
                logger.info(f"    {var}: {first_static[var]:.3f} → {last_static[var]:.3f}")
        
        logger.info("\n  Active predictions (first vs last):")
        first_active = active_marginals_history[0][1]
        last_active = active_marginals_history[-1][1]
        for var in sorted(first_active.keys()):
            if var in last_active:
                change = last_active[var] - first_active[var]
                logger.info(f"    {var}: {first_active[var]:.3f} → {last_active[var]:.3f} "
                          f"(Δ={change:+.3f})")
    
    logger.info("\nConclusion:")
    if f1_improvement > 0.05:
        logger.info("  ✓ Active learning significantly improved structure discovery")
    elif f1_improvement > 0:
        logger.info("  ✓ Active learning slightly improved structure discovery")
    else:
        logger.info("  ✗ Active learning did not improve structure discovery")
    
    logger.info(f"  Static predictions remained constant: {len(set(str(m) for _, m in static_marginals_history)) == 1}")
    logger.info(f"  Active predictions evolved: {len(set(str(m) for _, m in active_marginals_history)) > 1}")

if __name__ == "__main__":
    main()