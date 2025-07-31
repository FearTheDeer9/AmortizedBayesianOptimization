#!/usr/bin/env python3
"""
Simple test of the evaluation framework with detailed logging.

This script uses the UniversalACBOEvaluator directly to avoid signature issues.
"""

import logging
import sys
from pathlib import Path

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import UniversalACBOEvaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_bc_acquisition,
    create_bc_surrogate,
    create_random_acquisition,
    create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

logger = logging.getLogger(__name__)


def main():
    """Run simple evaluation test."""
    logger.info("="*80)
    logger.info("SIMPLE EVALUATION TEST")
    logger.info("="*80)
    
    # Create test SCM
    scm = create_fork_scm(noise_scale=1.0)
    logger.info(f"\nCreated fork SCM:")
    logger.info(f"  Variables: {scm['variables']}")
    logger.info(f"  Target: {scm['target']}")
    logger.info(f"  Edges: {scm['edges']}")
    
    # Create evaluator
    evaluator = UniversalACBOEvaluator()
    
    # Simple evaluation config
    eval_config = {
        'n_observational': 20,
        'max_interventions': 5,
        'n_intervention_samples': 20,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Test 1: Random baseline
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Random Baseline")
    logger.info("="*60)
    
    random_fn = create_random_acquisition(seed=42)
    result = evaluator.evaluate(
        acquisition_fn=random_fn,
        scm=scm,
        config=eval_config,
        seed=42
    )
    
    if result.success:
        logger.info("✓ Random evaluation succeeded!")
        logger.info(f"  Initial value: {result.final_metrics['initial_value']:.4f}")
        logger.info(f"  Best value: {result.final_metrics['best_value']:.4f}")
        logger.info(f"  Improvement: {result.final_metrics['improvement']:.4f}")
        
        # Show interventions
        logger.info("\n  Interventions made:")
        for i, step in enumerate(result.history[1:6]):  # Skip initial, show first 5
            if step.intervention:
                targets = list(step.intervention.get('targets', []))
                if targets:
                    target_var = targets[0]
                    value = step.intervention['values'][target_var]
                    logger.info(f"    Step {i+1}: {target_var} = {value:.3f} → outcome = {step.outcome_value:.3f}")
    else:
        logger.error(f"✗ Random evaluation failed: {result.error_message}")
    
    # Test 2: Oracle baseline  
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Oracle Baseline")
    logger.info("="*60)
    
    oracle_fn = create_optimal_oracle_acquisition(scm)
    result = evaluator.evaluate(
        acquisition_fn=oracle_fn,
        scm=scm,
        config=eval_config,
        seed=42
    )
    
    if result.success:
        logger.info("✓ Oracle evaluation succeeded!")
        logger.info(f"  Initial value: {result.final_metrics['initial_value']:.4f}")
        logger.info(f"  Best value: {result.final_metrics['best_value']:.4f}")
        logger.info(f"  Improvement: {result.final_metrics['improvement']:.4f}")
        
        # Show interventions
        logger.info("\n  Interventions made:")
        for i, step in enumerate(result.history[1:6]):
            if step.intervention:
                targets = list(step.intervention.get('targets', []))
                if targets:
                    target_var = targets[0]
                    value = step.intervention['values'][target_var]
                    logger.info(f"    Step {i+1}: {target_var} = {value:.3f} → outcome = {step.outcome_value:.3f}")
    else:
        logger.error(f"✗ Oracle evaluation failed: {result.error_message}")
    
    # Test 3: GRPO with BC surrogate
    logger.info("\n" + "="*60)
    logger.info("TEST 3: GRPO + BC Surrogate")
    logger.info("="*60)
    
    # Check if checkpoints exist
    grpo_path = Path("checkpoints/validation/unified_grpo_final")
    surrogate_path = Path("checkpoints/validation/bc_surrogate_final")
    
    if grpo_path.exists() and surrogate_path.exists():
        grpo_fn = create_grpo_acquisition(grpo_path)
        surrogate_fn, _ = create_bc_surrogate(surrogate_path)
        
        result = evaluator.evaluate(
            acquisition_fn=grpo_fn,
            scm=scm,
            config=eval_config,
            surrogate_fn=surrogate_fn,
            seed=42
        )
        
        if result.success:
            logger.info("✓ GRPO+Surrogate evaluation succeeded!")
            logger.info(f"  Initial value: {result.final_metrics['initial_value']:.4f}")
            logger.info(f"  Best value: {result.final_metrics['best_value']:.4f}")
            logger.info(f"  Improvement: {result.final_metrics['improvement']:.4f}")
            
            # Check if surrogate provided predictions
            has_marginals = any(step.marginals is not None for step in result.history)
            logger.info(f"  Surrogate predictions used: {'Yes' if has_marginals else 'No'}")
            
            # Show interventions with surrogate predictions
            logger.info("\n  Interventions made:")
            for i, step in enumerate(result.history[1:6]):
                if step.intervention:
                    targets = list(step.intervention.get('targets', []))
                    if targets:
                        target_var = targets[0]
                        value = step.intervention['values'][target_var]
                        
                        msg = f"    Step {i+1}: {target_var} = {value:.3f} → outcome = {step.outcome_value:.3f}"
                        
                        # Add surrogate prediction if available
                        if step.marginals and target_var in step.marginals:
                            prob = step.marginals[target_var]
                            msg += f" (surrogate prob: {prob:.3f})"
                        
                        logger.info(msg)
        else:
            logger.error(f"✗ GRPO+Surrogate evaluation failed: {result.error_message}")
    else:
        logger.warning("Checkpoints not found - skipping GRPO+Surrogate test")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT CALCULATION VERIFICATION")
    logger.info("="*80)
    logger.info("\nFormula: improvement = initial_value - best_value")
    logger.info("For MINIMIZE optimization:")
    logger.info("  - Positive improvement = target was reduced (good)")
    logger.info("  - Larger improvement = better performance")
    logger.info("\nOracle should have the highest improvement since it knows the true structure.")


if __name__ == "__main__":
    main()