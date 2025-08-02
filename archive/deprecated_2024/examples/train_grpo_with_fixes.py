#!/usr/bin/env python3
"""Example script for training GRPO with collapse prevention fixes.

This script demonstrates how to use the fixed GRPO configuration that prevents
posterior collapse where all variable embeddings become identical.

Key fixes applied:
1. Global standardization instead of per-variable standardization
2. Increased entropy coefficient (0.1 instead of 0.01)
3. Bootstrap surrogate with structural priors
4. Adaptive reward system that shifts from discovery to optimization

Usage:
    python examples/train_grpo_with_fixes.py --n_variables 4 --max_steps 10000
"""

import argparse
import logging
from pathlib import Path

import jax
import jax.random as random

from src.causal_bayes_opt.training.grpo_fixed_config import (
    create_grpo_config_with_fixes,
    create_bootstrap_phase_config,
    create_bootstrap_config,
    validate_fixed_config
)
from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm


def main():
    parser = argparse.ArgumentParser(
        description="Train GRPO policy with collapse prevention fixes"
    )
    parser.add_argument(
        "--n_variables", 
        type=int, 
        default=4,
        help="Number of variables in the SCM"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.1,
        help="Entropy coefficient for exploration"
    )
    parser.add_argument(
        "--use_bootstrap",
        action="store_true",
        default=True,
        help="Use bootstrap surrogate"
    )
    parser.add_argument(
        "--use_adaptive_rewards",
        action="store_true", 
        default=True,
        help="Use adaptive reward system"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/grpo_fixed",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Creating GRPO configuration with collapse prevention fixes...")
    
    # Create configuration
    config = create_grpo_config_with_fixes(
        max_training_steps=args.max_steps,
        batch_size=args.batch_size,
        entropy_coefficient=args.entropy_coeff,
        use_bootstrap=args.use_bootstrap,
        use_adaptive_rewards=args.use_adaptive_rewards
    )
    
    # Validate configuration
    try:
        validate_fixed_config(config)
        logger.info("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Create bootstrap configurations if enabled
    if args.use_bootstrap:
        phase_config = create_bootstrap_phase_config()
        bootstrap_config = create_bootstrap_config()
        logger.info("Bootstrap surrogate enabled with structural priors")
    
    # Log key configuration parameters
    logger.info(f"Key configuration parameters:")
    logger.info(f"  - Entropy coefficient: {config.grpo_algorithm.entropy_coeff}")
    logger.info(f"  - Group size: {config.grpo_algorithm.group_size}")
    logger.info(f"  - Batch size: {config.experience_management.batch_size}")
    logger.info(f"  - Max training steps: {config.max_training_steps}")
    logger.info(f"  - Training mode: {config.training_mode}")
    logger.info(f"  - Checkpoint directory: {args.checkpoint_dir}")
    
    # Create example SCM for demonstration
    logger.info(f"Creating example SCM with {args.n_variables} variables...")
    
    variables = [f'X{i}' for i in range(args.n_variables)]
    edges = [(f'X{i}', f'X{i+1}') for i in range(args.n_variables - 1)]
    
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients={e: 1.5 for e in edges},
        noise_scales={v: 0.1 for v in variables},
        target=variables[-1]
    )
    
    logger.info(f"SCM created with target: {variables[-1]}")
    logger.info(f"Edges: {edges}")
    
    # Note: Actual training would happen here
    # This is just a configuration demonstration
    logger.info("\nConfiguration created successfully!")
    logger.info("To use this configuration:")
    logger.info("1. Pass it to your GRPO trainer")
    logger.info("2. Ensure EnrichedHistoryBuilder uses global standardization")
    logger.info("3. Monitor embedding similarity during training")
    logger.info("4. Check that parent_prob channel variance stays > 0.01")
    
    # Save configuration summary
    config_summary = {
        "entropy_coefficient": config.grpo_algorithm.entropy_coeff,
        "group_size": config.grpo_algorithm.group_size,
        "batch_size": config.experience_management.batch_size,
        "use_bootstrap": args.use_bootstrap,
        "use_adaptive_rewards": args.use_adaptive_rewards,
        "max_training_steps": config.max_training_steps,
        "standardization": "global",
        "fixes_applied": [
            "global_standardization",
            "increased_entropy", 
            "bootstrap_surrogate",
            "adaptive_rewards"
        ]
    }
    
    logger.info(f"\nConfiguration summary: {config_summary}")
    
    return 0


if __name__ == "__main__":
    exit(main())