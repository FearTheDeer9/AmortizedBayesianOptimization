#!/usr/bin/env python3
"""
Debug structure learning and F1 score calculations.

This script runs evaluation with detailed logging to understand:
- Why F1 scores are always 0
- Whether surrogate models are being called
- What posteriors look like
- How marginals are computed
"""

import argparse
import logging
from pathlib import Path
import sys
import json
import numpy as np
import jax.numpy as jnp

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.evaluation.universal_evaluator import UniversalACBOEvaluator
from src.causal_bayes_opt.evaluation.model_interfaces import (
    create_grpo_acquisition, create_bc_acquisition, 
    create_random_acquisition, create_optimal_oracle_acquisition
)
from src.causal_bayes_opt.experiments.benchmark_scms import (
    create_fork_scm, create_chain_scm, create_collider_scm
)
from src.causal_bayes_opt.data_structures.scm import get_parents, get_target, get_edges
from src.causal_bayes_opt.training.continuous_surrogate_integration import (
    create_continuous_learnable_surrogate
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DebugEvaluator(UniversalACBOEvaluator):
    """Extended evaluator with detailed logging for debugging."""
    
    def __init__(self, name: str = "DebugEvaluator"):
        super().__init__(name)
        self.posterior_log = []
        self.marginal_log = []
        
    def evaluate(self, acquisition_fn, scm, config, surrogate_fn=None, seed=42):
        """Override to add detailed logging."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting evaluation with surrogate_fn: {surrogate_fn is not None}")
        logger.info(f"{'='*60}")
        
        # Get true structure for debugging
        target = get_target(scm)
        true_parents = list(get_parents(scm, target))
        logger.info(f"True structure - Target: {target}, Parents: {true_parents}")
        
        # Run normal evaluation with our overridden methods
        result = super().evaluate(acquisition_fn, scm, config, surrogate_fn, seed)
        
        # Log final structure metrics
        logger.info(f"\nFinal metrics: {result.final_metrics}")
        
        return result
        
    def _compute_final_metrics(self, history, initial_value, best_value, 
                              true_parents, optimization_direction):
        """Override to add detailed F1 calculation logging."""
        logger.info(f"\n{'='*40}")
        logger.info("Computing final metrics...")
        logger.info(f"History length: {len(history)}")
        logger.info(f"True parents: {true_parents}")
        
        final_step = history[-1] if history else None
        
        if final_step:
            logger.info(f"Final step has marginals: {final_step.marginals is not None}")
            if final_step.marginals:
                logger.info(f"Marginals: {final_step.marginals}")
                
                # Compute predicted parents
                pred_parents = {
                    var for var, prob in final_step.marginals.items() 
                    if prob > 0.5
                }
                logger.info(f"Predicted parents (>0.5): {pred_parents}")
                
                # Compute F1 manually for debugging
                true_set = set(true_parents)
                pred_set = pred_parents
                
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                logger.info(f"TP={tp}, FP={fp}, FN={fn}")
                logger.info(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        else:
            logger.info("No history available!")
        
        # Call parent implementation
        metrics = super()._compute_final_metrics(
            history, initial_value, best_value, true_parents, optimization_direction
        )
        
        logger.info(f"Computed metrics: {metrics}")
        return metrics


def create_debug_surrogate():
    """Create a surrogate function that logs when called."""
    
    # Create actual surrogate
    import jax
    key = jax.random.PRNGKey(42)
    net, params, opt_state, predict_fn, update_fn = create_continuous_learnable_surrogate(
        n_variables=8,  # Max size
        key=key,
        learning_rate=1e-3
    )
    
    call_count = [0]  # Mutable counter
    
    def debug_surrogate_fn(tensor, target_var):
        call_count[0] += 1
        logger.info(f"\n{'='*40}")
        logger.info(f"Surrogate called (#{call_count[0]}) for target: {target_var}")
        logger.info(f"Tensor shape: {tensor.shape}")
        
        # Get variables from tensor
        n_vars = tensor.shape[1]
        variables = [f'X{i}' for i in range(n_vars)]
        
        # Handle special variable names
        if target_var in ['Y', 'Z', 'X']:
            # Map to indices for special cases
            var_map = {'X': 0, 'Y': 1, 'Z': 2}
            target_idx = var_map.get(target_var, 0)
        else:
            # Extract number from X0, X1, etc.
            try:
                target_idx = int(target_var[1:])
            except:
                target_idx = 0
        
        # Call actual prediction
        result = predict_fn(tensor, target_idx, variables)
        
        logger.info(f"Surrogate output: {result}")
        if 'marginal_parent_probs' in result:
            logger.info(f"Marginal probs: {result['marginal_parent_probs']}")
        
        return result
    
    return debug_surrogate_fn


def test_single_method(method_name, acquisition_fn, scm, config, use_surrogate=True):
    """Test a single method with detailed logging."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {method_name}")
    logger.info(f"{'='*60}")
    
    evaluator = DebugEvaluator(name=f"Debug_{method_name}")
    
    # Create debug surrogate if needed
    surrogate_fn = create_debug_surrogate() if use_surrogate else None
    
    # Run evaluation
    result = evaluator.evaluate(
        acquisition_fn=acquisition_fn,
        scm=scm,
        config=config,
        surrogate_fn=surrogate_fn,
        seed=config.get('seed', 42)
    )
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Debug structure learning and F1 scores')
    parser.add_argument('--grpo_checkpoint', type=str, help='Path to GRPO checkpoint')
    parser.add_argument('--bc_checkpoint', type=str, help='Path to BC checkpoint')
    parser.add_argument('--n_interventions', type=int, default=5, 
                       help='Number of interventions (keep small for debugging)')
    parser.add_argument('--output_file', type=str, default='debug_structure_learning.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create simple test SCM
    scm = create_fork_scm(noise_scale=1.0)
    
    # Evaluation config
    config = {
        'n_observational': 50,
        'max_interventions': args.n_interventions,
        'n_intervention_samples': 10,
        'optimization_direction': 'MINIMIZE',
        'seed': 42
    }
    
    results = {}
    
    # Test Random baseline
    logger.info("\n" + "="*80)
    logger.info("TESTING RANDOM BASELINE")
    logger.info("="*80)
    random_fn = create_random_acquisition(seed=42)
    random_result = test_single_method("Random", random_fn, scm, config, use_surrogate=True)
    results['random'] = {
        'final_f1': random_result.final_metrics.get('final_f1', -1),
        'has_marginals': any(step.marginals is not None for step in random_result.history)
    }
    
    # Test Oracle
    logger.info("\n" + "="*80)
    logger.info("TESTING ORACLE")
    logger.info("="*80)
    target = get_target(scm)
    scm_edges = {}
    edges = get_edges(scm)
    for parent, child in edges:
        if child not in scm_edges:
            scm_edges[child] = []
        scm_edges[child].append(parent)
    oracle_fn = create_optimal_oracle_acquisition(scm, optimization_direction='MINIMIZE', seed=42)
    oracle_result = test_single_method("Oracle", oracle_fn, scm, config, use_surrogate=True)
    results['oracle'] = {
        'final_f1': oracle_result.final_metrics.get('final_f1', -1),
        'has_marginals': any(step.marginals is not None for step in oracle_result.history)
    }
    
    # Test GRPO if checkpoint provided
    if args.grpo_checkpoint:
        logger.info("\n" + "="*80)
        logger.info("TESTING GRPO")
        logger.info("="*80)
        grpo_fn = create_grpo_acquisition(Path(args.grpo_checkpoint), seed=42)
        grpo_result = test_single_method("GRPO", grpo_fn, scm, config, use_surrogate=True)
        results['grpo'] = {
            'final_f1': grpo_result.final_metrics.get('final_f1', -1),
            'has_marginals': any(step.marginals is not None for step in grpo_result.history)
        }
    
    # Test BC if checkpoint provided
    if args.bc_checkpoint:
        logger.info("\n" + "="*80)
        logger.info("TESTING BC")
        logger.info("="*80)
        bc_fn = create_bc_acquisition(Path(args.bc_checkpoint), seed=42)
        bc_result = test_single_method("BC", bc_fn, scm, config, use_surrogate=True)
        results['bc'] = {
            'final_f1': bc_result.final_metrics.get('final_f1', -1),
            'has_marginals': any(step.marginals is not None for step in bc_result.history)
        }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    for method, result in results.items():
        logger.info(f"{method}: F1={result['final_f1']:.3f}, Has marginals={result['has_marginals']}")


if __name__ == "__main__":
    main()