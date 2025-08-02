#!/usr/bin/env python3
"""
Modular Enriched GRPO Policy Training Script

This script demonstrates the refactored, modular approach to training enriched
GRPO policies following CLAUDE.md principles. Each component has a single
responsibility and can be tested independently.

Usage:
    # Basic training
    poetry run python scripts/train_enriched_acbo_modular.py
    
    # With WandB logging
    poetry run python scripts/train_enriched_acbo_modular.py logging.wandb.enabled=true
    
    # Custom configuration
    poetry run python scripts/train_enriched_acbo_modular.py training.n_episodes=500

Features:
- Modular, single-responsibility components
- Comprehensive test coverage through dependency injection
- Immutable state management with pyrsistent
- Proper JAX key threading
- Clean separation of concerns
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_bayes_opt.training.enriched_trainer import EnrichedGRPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="enriched_training")
def main(cfg: DictConfig) -> None:
    """
    Main training function using modular components.
    
    This function demonstrates clean, functional programming principles:
    - Single responsibility (just coordinates training)
    - Immutable configuration
    - Clear separation of concerns
    - Error handling at the right level
    """
    logger.info("Starting modular enriched GRPO training")
    logger.info(f"Configuration: {cfg}")
    
    try:
        # Create trainer with dependency injection
        trainer = EnrichedGRPOTrainer(cfg)
        
        # Run training (trainer handles all complexity internally)
        results = trainer.train()
        
        # Log results
        logger.info("Training completed successfully")
        logger.info(f"Final checkpoint: {results['checkpoint_path']}")
        logger.info(f"Performance summary: {results['performance']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()