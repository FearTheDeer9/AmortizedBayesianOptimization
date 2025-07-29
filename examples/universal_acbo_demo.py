#!/usr/bin/env python3
"""
Demonstration of Universal ACBO Evaluator.

This demo shows the clean architecture where:
1. Models are simple functions
2. Evaluation is generic
3. No method-specific code needed

The same evaluator works with GRPO, BC, random baselines, etc.
"""

import logging
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import (
    UniversalACBOEvaluator, create_universal_evaluator
)
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition,
    create_random_acquisition,
    create_oracle_acquisition,
    create_uniform_exploration_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.data_structures.scm import get_parents, get_target

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_universal_evaluation():
    """Demonstrate universal evaluator with different acquisition functions."""
    logger.info("=== Universal ACBO Evaluator Demo ===\n")
    
    # Create test SCM
    scm = create_fork_scm(noise_scale=1.0, target="Y")
    target = get_target(scm)
    
    # Extract true structure for oracle
    scm_edges = {}
    variables = ['X', 'Y', 'Z']
    for var in variables:
        scm_edges[var] = list(get_parents(scm, var))
    
    logger.info(f"Test SCM: Fork structure")
    logger.info(f"Target: {target}")
    logger.info(f"True edges: {scm_edges}\n")
    
    # Evaluation config
    config = {
        'n_observational': 100,
        'max_interventions': 10,
        'n_intervention_samples': 100,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Create universal evaluator
    evaluator = create_universal_evaluator()
    
    # Test different acquisition functions
    acquisition_functions = {
        'Random': create_random_acquisition(seed=42),
        'Uniform': create_uniform_exploration_acquisition(seed=43),
        'Oracle': create_oracle_acquisition(scm_edges, seed=44)
    }
    
    results = {}
    
    for name, acquisition_fn in acquisition_functions.items():
        logger.info(f"\nEvaluating {name} acquisition...")
        
        # Run evaluation - same evaluator for all methods!
        result = evaluator.evaluate(
            acquisition_fn=acquisition_fn,
            scm=scm,
            config=config,
            surrogate_fn=None,  # No surrogate for this demo
            seed=42
        )
        
        if result.success:
            logger.info(f"  Initial value: {result.final_metrics['initial_value']:.3f}")
            logger.info(f"  Final value: {result.final_metrics['final_value']:.3f}")
            logger.info(f"  Best value: {result.final_metrics['best_value']:.3f}")
            logger.info(f"  Improvement: {result.final_metrics['improvement']:.3f}")
            logger.info(f"  Mean reward: {result.final_metrics['mean_reward']:.3f}")
            logger.info(f"  Time: {result.total_time:.2f}s")
            
            results[name] = result
        else:
            logger.error(f"  Evaluation failed: {result.error_message}")
    
    return results


def demonstrate_grpo_evaluation():
    """Demonstrate evaluating a trained GRPO model."""
    logger.info("\n\n=== GRPO Model Evaluation ===\n")
    
    # Check if we have a trained GRPO checkpoint
    checkpoint_path = Path('checkpoints/clean_grpo/clean_grpo_final')
    
    if not checkpoint_path.exists():
        logger.warning("No GRPO checkpoint found. Train a model first with clean_acbo_demo.py")
        return None
    
    logger.info(f"Loading GRPO checkpoint from {checkpoint_path}")
    
    # Create GRPO acquisition function
    grpo_acquisition = create_grpo_acquisition(checkpoint_path, seed=42)
    
    # Test on multiple SCMs
    test_scms = [
        ("3-var fork", create_fork_scm(noise_scale=1.0, target="Y")),
        ("4-var chain", create_chain_scm(chain_length=4, noise_scale=1.0)),
        ("3-var collider", create_collider_scm(noise_scale=1.0))
    ]
    
    # Evaluation config
    config = {
        'n_observational': 100,
        'max_interventions': 10,
        'n_intervention_samples': 100,
        'optimization_direction': 'MINIMIZE'
    }
    
    # Single evaluator for all SCMs
    evaluator = create_universal_evaluator()
    
    for scm_name, scm in test_scms:
        logger.info(f"\nEvaluating GRPO on {scm_name}...")
        
        result = evaluator.evaluate(
            acquisition_fn=grpo_acquisition,
            scm=scm,
            config=config,
            seed=42
        )
        
        if result.success:
            logger.info(f"  Improvement: {result.final_metrics['improvement']:.3f}")
            logger.info(f"  Final value: {result.final_metrics['final_value']:.3f}")


def compare_architectures():
    """Compare old vs new architecture complexity."""
    logger.info("\n\n=== Architecture Comparison ===\n")
    
    logger.info("OLD Architecture (method-specific):")
    logger.info("  - GRPOEvaluator: ~400 lines")
    logger.info("  - BCEvaluator: ~350 lines")
    logger.info("  - BaselineEvaluator: ~300 lines")
    logger.info("  - Total: ~1050 lines of evaluation code")
    logger.info("  - Each method needs custom evaluation logic")
    logger.info("  - Models tightly coupled to evaluation\n")
    
    logger.info("NEW Architecture (universal):")
    logger.info("  - UniversalACBOEvaluator: ~300 lines")
    logger.info("  - Model interfaces: ~50 lines per method")
    logger.info("  - Total: ~450 lines for ALL methods")
    logger.info("  - Single evaluation logic for all")
    logger.info("  - Models are simple functions\n")
    
    logger.info("Benefits:")
    logger.info("  ✓ 57% less code")
    logger.info("  ✓ Much simpler to understand")
    logger.info("  ✓ Easy to add new methods")
    logger.info("  ✓ Models can be tested independently")
    logger.info("  ✓ No wrappers or adapters needed")


def main():
    """Run full universal evaluator demonstration."""
    start_time = time.time()
    
    # Basic demonstration
    demonstrate_universal_evaluation()
    
    # GRPO evaluation (if checkpoint exists)
    demonstrate_grpo_evaluation()
    
    # Architecture comparison
    compare_architectures()
    
    logger.info(f"\n\nTotal demo time: {time.time() - start_time:.2f}s")
    
    logger.info("\n" + "="*60)
    logger.info("Universal ACBO Evaluator Benefits:")
    logger.info("1. ONE evaluator for ALL methods")
    logger.info("2. Models are just functions: (tensor, posterior, target) → intervention")
    logger.info("3. No method-specific evaluation code")
    logger.info("4. Clean separation of concerns")
    logger.info("5. Easy to extend with new methods")
    logger.info("="*60)


if __name__ == "__main__":
    main()