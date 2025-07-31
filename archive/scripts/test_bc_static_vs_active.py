#!/usr/bin/env python3
"""Direct comparison of BC static vs BC with active learning."""

import logging
from pathlib import Path

from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.evaluation.universal_evaluator import create_universal_evaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_random_acquisition,
    create_bc_surrogate
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_static_vs_active():
    """Compare BC static vs BC with active learning."""
    
    # Setup
    scm = create_fork_scm(noise_scale=1.0)
    random_policy = create_random_acquisition(seed=42)
    checkpoint_path = Path('checkpoints/validation/bc_surrogate_final')
    
    if not checkpoint_path.exists():
        logger.error(f"BC checkpoint not found at {checkpoint_path}")
        return
    
    eval_config = {
        'n_observational': 50,
        'max_interventions': 10,
        'n_intervention_samples': 50,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    logger.info("Testing BC Static vs BC Active Learning")
    logger.info("="*60)
    
    # Test 1: BC Static (no updates)
    logger.info("\n1. BC STATIC (no active learning):")
    logger.info("-"*40)
    
    bc_static_predict, _ = create_bc_surrogate(checkpoint_path, allow_updates=False)
    evaluator = create_universal_evaluator()
    
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
    logger.info(f"  Trajectory mean: {static_result.final_metrics['mean_trajectory_value']:.3f}")
    
    # Extract prediction history
    static_predictions = []
    for step in static_result.history:
        if step.marginals:
            static_predictions.append({
                'step': step.step,
                'marginals': dict(step.marginals),
                'confidence': step.prediction_confidence
            })
    
    # Test 2: BC with active learning
    logger.info("\n\n2. BC ACTIVE (with online updates):")
    logger.info("-"*40)
    logger.info("Note: Current infrastructure doesn't support BC active learning properly")
    logger.info("The evaluator needs params/opt_state which aren't exposed by BC model")
    
    # This is the issue - we can create the functions but can't use them properly
    bc_active_predict, bc_update = create_bc_surrogate(checkpoint_path, allow_updates=True)
    
    # The problem is that the evaluator expects to manage params/opt_state
    # but the BC model encapsulates them internally
    
    logger.info("\nAnalysis:")
    logger.info("- BC static works correctly")
    logger.info("- BC active learning update function is implemented")
    logger.info("- But integration with evaluator needs work")
    logger.info("- Need to either:")
    logger.info("  1. Expose BC params/opt_state")
    logger.info("  2. Use ActiveLearningSurrogate wrapper")
    logger.info("  3. Modify evaluator to handle encapsulated models")
    
    # Show how predictions stayed constant for static
    if len(static_predictions) > 1:
        logger.info(f"\nStatic prediction consistency:")
        first_marginals = static_predictions[0]['marginals']
        all_same = all(
            p['marginals'] == first_marginals 
            for p in static_predictions[1:]
        )
        logger.info(f"  All predictions identical: {all_same}")
        
        # Show evolution
        logger.info(f"\n  Prediction evolution:")
        for i in [0, len(static_predictions)//2, -1]:
            if i < len(static_predictions):
                p = static_predictions[i]
                logger.info(f"    Step {p['step']}: {p['marginals']}")

if __name__ == "__main__":
    test_static_vs_active()