#!/usr/bin/env python3
"""
Two-Phase GRPO + Active Learning Training

This script implements the two-phase approach:
1. Phase 1: Train GRPO policy with bootstrap features (already done)
2. Phase 2: Use trained GRPO policy with active learning surrogate

The key insight is that a good intervention policy (from GRPO) should
accelerate structure learning compared to random interventions.
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from omegaconf import DictConfig, OmegaConf

from src.causal_bayes_opt.training.grpo_policy_loader import (
    load_grpo_policy, create_grpo_intervention_fn
)
from examples.demo_learning import DemoConfig, create_learnable_surrogate_model
from examples.complete_workflow_demo import run_progressive_learning_demo_with_scm
from examples.demo_scms import create_easy_scm, create_medium_scm, create_hard_scm
from src.causal_bayes_opt.data_structures.scm import get_variables, get_target
from src.causal_bayes_opt.analysis.trajectory_metrics import (
    compute_f1_score_from_marginals,
    compute_shd_from_marginals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_phase2_config(
    n_observational_samples: int = 30,
    n_intervention_steps: int = 50,
    learning_rate: float = 1e-3,
    scoring_method: str = "bic"
) -> DemoConfig:
    """Create configuration for Phase 2 active learning."""
    return DemoConfig(
        n_observational_samples=n_observational_samples,
        n_intervention_steps=n_intervention_steps,
        learning_rate=learning_rate,
        intervention_value_range=(-2.0, 2.0),
        random_seed=42,
        scoring_method=scoring_method
    )


def run_phase2_active_learning(
    scm: pyr.PMap,
    grpo_checkpoint_path: str,
    config: DemoConfig,
    track_structure_learning: bool = True
) -> Dict[str, Any]:
    """
    Run Phase 2: Active learning with fixed GRPO policy.
    
    Args:
        scm: Structural causal model
        grpo_checkpoint_path: Path to trained GRPO checkpoint
        config: Configuration for active learning
        track_structure_learning: Whether to track F1/SHD metrics
        
    Returns:
        Results dictionary with learning history and metrics
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Active Learning with GRPO Policy")
    logger.info("=" * 60)
    
    # Load trained GRPO policy
    logger.info(f"Loading GRPO policy from: {grpo_checkpoint_path}")
    loaded_policy = load_grpo_policy(grpo_checkpoint_path)
    
    # Verify SCM compatibility
    scm_variables = list(get_variables(scm))
    scm_target = get_target(scm)
    
    if set(scm_variables) != set(loaded_policy.variables):
        logger.warning(
            f"SCM variables {scm_variables} differ from "
            f"policy variables {loaded_policy.variables}"
        )
    
    if scm_target != loaded_policy.target_variable:
        raise ValueError(
            f"SCM target {scm_target} != policy target "
            f"{loaded_policy.target_variable}"
        )
    
    # Create GRPO intervention function
    grpo_intervention_fn = create_grpo_intervention_fn(
        loaded_policy=loaded_policy,
        scm=scm,
        intervention_range=config.intervention_value_range
    )
    
    # Run active learning with GRPO interventions
    logger.info("Starting active learning with GRPO-guided interventions")
    result = run_progressive_learning_demo_with_scm(
        scm=scm,
        config=config,
        pretrained_surrogate=None,  # Create new learning surrogate
        pretrained_acquisition=grpo_intervention_fn  # Use GRPO policy
    )
    
    # Add phase 2 specific metadata
    result['phase'] = 'phase2_active_learning'
    result['grpo_checkpoint_used'] = str(grpo_checkpoint_path)
    result['used_grpo_policy'] = True
    result['used_learning_surrogate'] = True
    
    # Track structure learning metrics if requested
    if track_structure_learning and 'learning_history' in result:
        f1_scores = []
        shd_scores = []
        
        for step_info in result['learning_history']:
            if 'marginals' in step_info:
                marginals = step_info['marginals']
                # Compute F1 score
                f1 = compute_f1_score_from_marginals(
                    marginals, scm, threshold=0.5
                )
                f1_scores.append(f1)
                
                # Compute SHD
                shd = compute_shd_from_marginals(
                    marginals, scm, threshold=0.5
                )
                shd_scores.append(shd)
        
        result['structure_learning_metrics'] = {
            'f1_scores': f1_scores,
            'shd_scores': shd_scores,
            'final_f1': f1_scores[-1] if f1_scores else 0.0,
            'final_shd': shd_scores[-1] if shd_scores else float('inf')
        }
    
    return result


def compare_approaches(
    scm: pyr.PMap,
    grpo_checkpoint_path: str,
    config: Optional[DemoConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compare three approaches:
    1. Random + Active Learning (baseline)
    2. GRPO + Active Learning (Phase 2)
    3. GRPO + Bootstrap (current approach)
    
    Args:
        scm: Structural causal model
        grpo_checkpoint_path: Path to trained GRPO checkpoint
        config: Configuration (uses defaults if None)
        
    Returns:
        Dictionary with results for each approach
    """
    if config is None:
        config = create_phase2_config()
    
    results = {}
    
    # 1. Random + Active Learning (baseline)
    logger.info("\n" + "=" * 60)
    logger.info("Running Random + Active Learning (baseline)")
    results['random_active'] = run_progressive_learning_demo_with_scm(
        scm=scm,
        config=config,
        pretrained_surrogate=None,  # Learning surrogate
        pretrained_acquisition=None  # Random interventions
    )
    
    # 2. GRPO + Active Learning (Phase 2)
    logger.info("\n" + "=" * 60)
    logger.info("Running GRPO + Active Learning (Phase 2)")
    results['grpo_active'] = run_phase2_active_learning(
        scm=scm,
        grpo_checkpoint_path=grpo_checkpoint_path,
        config=config,
        track_structure_learning=True
    )
    
    # 3. For GRPO + Bootstrap, we would need to run the original GRPO evaluation
    # This is included for reference but not implemented here
    logger.info("\n" + "=" * 60)
    logger.info("GRPO + Bootstrap results should be obtained from GRPO evaluation")
    
    # Compare results
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    
    for approach_name, result in results.items():
        if 'final_target_value' in result:
            logger.info(f"\n{approach_name}:")
            logger.info(f"  Final target value: {result['final_target_value']:.4f}")
            logger.info(f"  Target improvement: {result['target_improvement']:.4f}")
            
            if 'structure_learning_metrics' in result:
                metrics = result['structure_learning_metrics']
                logger.info(f"  Final F1 score: {metrics['final_f1']:.4f}")
                logger.info(f"  Final SHD: {metrics['final_shd']:.1f}")
    
    return results


def main():
    """Main entry point for two-phase training."""
    parser = argparse.ArgumentParser(
        description="Two-Phase GRPO + Active Learning Training"
    )
    parser.add_argument(
        '--grpo-checkpoint',
        type=str,
        required=True,
        help='Path to trained GRPO checkpoint'
    )
    parser.add_argument(
        '--scm-type',
        type=str,
        choices=['easy', 'medium', 'hard'],
        default='easy',
        help='SCM difficulty level'
    )
    parser.add_argument(
        '--n-interventions',
        type=int,
        default=50,
        help='Number of intervention steps'
    )
    parser.add_argument(
        '--n-observational',
        type=int,
        default=30,
        help='Number of initial observational samples'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate for active learning'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple approaches'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='phase2_results',
        help='Directory for saving results'
    )
    
    args = parser.parse_args()
    
    # Create SCM
    scm_creators = {
        'easy': create_easy_scm,
        'medium': create_medium_scm,
        'hard': create_hard_scm
    }
    scm = scm_creators[args.scm_type]()
    
    # Create configuration
    config = create_phase2_config(
        n_observational_samples=args.n_observational,
        n_intervention_steps=args.n_interventions,
        learning_rate=args.learning_rate
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        # Run comparison
        results = compare_approaches(
            scm=scm,
            grpo_checkpoint_path=args.grpo_checkpoint,
            config=config
        )
        
        # Save results
        import pickle
        results_file = output_dir / f"comparison_results_{args.scm_type}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"Saved comparison results to {results_file}")
    else:
        # Run Phase 2 only
        result = run_phase2_active_learning(
            scm=scm,
            grpo_checkpoint_path=args.grpo_checkpoint,
            config=config,
            track_structure_learning=True
        )
        
        # Save result
        import pickle
        result_file = output_dir / f"phase2_result_{args.scm_type}.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved Phase 2 result to {result_file}")
        
        # Print summary
        logger.info("\nPhase 2 Summary:")
        logger.info(f"Final target value: {result.get('final_target_value', 'N/A')}")
        logger.info(f"Target improvement: {result.get('target_improvement', 'N/A')}")
        
        if 'structure_learning_metrics' in result:
            metrics = result['structure_learning_metrics']
            logger.info(f"Final F1 score: {metrics['final_f1']:.4f}")
            logger.info(f"Final SHD: {metrics['final_shd']:.1f}")


if __name__ == "__main__":
    main()