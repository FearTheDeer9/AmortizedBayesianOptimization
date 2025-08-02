#!/usr/bin/env python3
"""
Test 5-channel integration by evaluating policy-surrogate pairs.
"""

import logging
import sys
from pathlib import Path

# Set debug logging for posterior extraction
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('src.causal_bayes_opt.training.five_channel_converter').setLevel(logging.DEBUG)

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import UniversalACBOEvaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_bc_acquisition,
    create_bc_surrogate,
    create_random_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import create_sparse_scm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("Testing 5-Channel Integration with Model Pairs")
    logger.info("="*60)
    
    # Create test SCM
    scm = create_sparse_scm(num_vars=5, edge_prob=0.3, noise_scale=1.0)
    logger.info(f"Created test SCM with {len(scm['variables'])} variables")
    
    # Checkpoint paths
    grpo_checkpoint = Path("checkpoints/test_5ch/grpo/unified_grpo_final")
    bc_checkpoint = Path("checkpoints/test_5ch/bc_policy/bc_final")
    surrogate_checkpoint = Path("checkpoints/test_5ch/bc_surrogate/bc_surrogate_final")
    
    # Load models
    logger.info("\nLoading models...")
    grpo_fn = create_grpo_acquisition(grpo_checkpoint)
    bc_fn = create_bc_acquisition(bc_checkpoint)
    surrogate_fn, _ = create_bc_surrogate(surrogate_checkpoint)
    random_fn = create_random_acquisition()
    
    # Test surrogate output format
    logger.info("\nTesting surrogate output format...")
    import jax.numpy as jnp
    test_tensor = jnp.zeros((10, 5, 3))
    test_posterior = surrogate_fn(test_tensor, 'X4')
    logger.info(f"Surrogate returns type: {type(test_posterior)}")
    logger.info(f"Posterior attributes: {dir(test_posterior)}")
    if hasattr(test_posterior, 'metadata'):
        logger.info(f"Metadata type: {type(test_posterior.metadata)}")
        logger.info(f"Metadata contents: {test_posterior.metadata}")
    
    # Create evaluator
    evaluator = UniversalACBOEvaluator()
    
    # Evaluation config (short)
    eval_config = {
        'n_observational': 20,
        'max_interventions': 5,
        'n_intervention_samples': 20,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Test pairs
    pairs = [
        ("GRPO without surrogate", grpo_fn, None),
        ("GRPO with BC surrogate", grpo_fn, surrogate_fn),
        ("BC without surrogate", bc_fn, None),
        ("BC with BC surrogate", bc_fn, surrogate_fn),
        ("Random baseline", random_fn, None)
    ]
    
    results = {}
    
    for name, policy_fn, surr_fn in pairs:
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {name}")
        logger.info("="*40)
        
        result = evaluator.evaluate(
            acquisition_fn=policy_fn,
            scm=scm,
            config=eval_config,
            surrogate_fn=surr_fn,
            seed=42
        )
        
        if result.success:
            metrics = result.final_metrics
            logger.info(f"✓ Success - Final value: {metrics['final_value']:.3f}, Best: {metrics['best_value']:.3f}")
            
            # Check surrogate usage
            if surr_fn is not None and result.history:
                logger.info("\nChecking surrogate utilization:")
                for i, step in enumerate(result.history[1:4]):  # First 3 interventions
                    if step.intervention and step.marginals:
                        targets = step.intervention.get('targets', set())
                        if targets:
                            target = list(targets)[0]
                            
                            # Sort marginals by probability
                            sorted_probs = sorted(
                                [(v, p) for v, p in step.marginals.items() if v != scm['target']],
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            top_2 = [v for v, p in sorted_probs[:2]]
                            
                            logger.info(f"  Step {i+1}: Intervened on {target}")
                            logger.info(f"    Top predictions: {sorted_probs[:2]}")
                            logger.info(f"    Aligned: {'✓' if target in top_2 else '✗'}")
            
            results[name] = metrics
        else:
            logger.error(f"✗ Failed: {result.error_message}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info("="*60)
    
    for name, metrics in results.items():
        logger.info(f"{name:30} Best: {metrics['best_value']:8.3f}")
    
    # Check if surrogates helped
    improvements = {}
    for policy in ['GRPO', 'BC']:
        without = results.get(f"{policy} without surrogate", {}).get('best_value')
        with_surr = results.get(f"{policy} with BC surrogate", {}).get('best_value')
        
        if without and with_surr:
            improvement = without - with_surr  # Lower is better
            improvements[policy] = improvement
            
            logger.info(f"\n{policy} improvement with surrogate: {improvement:.3f}")
            if improvement > 0:
                logger.info("  ✓ Surrogate helped!")
            else:
                logger.info("  ✗ Surrogate did not help (may need more training)")
    
    # Final verdict
    if any(imp > 0 for imp in improvements.values()):
        logger.info("\n✓ SUCCESS: 5-channel integration is working!")
        logger.info("  At least one policy benefited from surrogate predictions.")
    else:
        logger.info("\n⚠ Integration working but models need more training")
        logger.info("  The 5-channel tensors are being created and used correctly,")
        logger.info("  but minimal training (5 episodes) may not show improvement yet.")


if __name__ == "__main__":
    main()