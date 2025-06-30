#!/usr/bin/env python3
"""
Hydra + WandB Integration Demo for Causal Bayesian Optimization

This script demonstrates minimal but powerful integration of Hydra configuration
management with WandB experiment tracking, working alongside existing Pydantic configs.

Usage:
    # Basic run
    python scripts/hydra_wandb_demo.py
    
    # Override config values
    python scripts/hydra_wandb_demo.py training.algorithm.learning_rate=0.001
    
    # Use different config compositions
    python scripts/hydra_wandb_demo.py training=grpo_high_performance
    
    # Disable WandB for local testing
    python scripts/hydra_wandb_demo.py logging=local_dev
    
    # Hyperparameter sweep
    python scripts/hydra_wandb_demo.py --multirun \
        training.algorithm.learning_rate=0.0001,0.001,0.01 \
        training.algorithm.batch_size=32,64,128

Features Demonstrated:
- Hydra configuration composition and overrides
- WandB experiment tracking with Hydra configs
- Seamless integration with existing Pydantic configs
- Hyperparameter sweeps with minimal setup
- Environment-specific configurations
"""

import logging
from pathlib import Path
from typing import Dict, Any
import time

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp

# Import existing project components
from causal_bayes_opt.training.config import TrainingConfig, GRPOTrainingConfig
from causal_bayes_opt.training.grpo_config import LoggingConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def demo_hydra_wandb_integration(cfg: DictConfig) -> None:
    """Demo function showing Hydra + WandB integration."""
    
    logger.info("Starting Hydra + WandB Integration Demo")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # 1. Initialize WandB if enabled
    wandb_run = None
    if cfg.logging.wandb.enabled and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.logging.wandb.tags,
            group=cfg.logging.wandb.group,
            name=f"demo_{cfg.experiment.problem.difficulty}_{int(time.time())}"
        )
        logger.info(f"WandB initialized: {wandb.run.url}")
    elif cfg.logging.wandb.enabled:
        logger.warning("WandB enabled in config but not available. Install with: pip install wandb")
    
    # 2. Convert Hydra config to existing Pydantic configs
    # This shows how to bridge Hydra configs with your existing system
    logging_config = convert_to_logging_config(cfg.logging)
    grpo_config = convert_to_grpo_config(cfg.training)
    
    logger.info("Converted Hydra configs to existing Pydantic configs")
    logger.info(f"LoggingConfig: enable_wandb={logging_config.enable_wandb}")
    logger.info(f"GRPOConfig: learning_rate={grpo_config}")
    
    # 3. Simulate a simple training loop with metrics
    logger.info("Simulating training loop...")
    
    key = jax.random.PRNGKey(cfg.seed)
    
    for step in range(min(cfg.max_steps, 50)):  # Limit demo steps
        key, subkey = jax.random.split(key)
        
        # Simulate some causal discovery metrics
        metrics = simulate_training_step(subkey, step, cfg)
        
        # Log every few steps
        if step % cfg.logging.console.frequency == 0:
            logger.info(f"Step {step}: Loss={metrics['loss']:.4f}, "
                       f"SHD={metrics['shd']:.2f}, "
                       f"Precision={metrics['precision']:.3f}")
        
        # Log to WandB if available
        if wandb_run is not None:
            wandb.log(metrics, step=step)
    
    # 4. Demonstrate config-dependent behavior
    demonstrate_config_effects(cfg)
    
    # 5. Log final summary
    final_metrics = {
        "demo_completed": True,
        "total_steps": min(cfg.max_steps, 50),
        "config_hash": hash(str(cfg)),
        "difficulty": cfg.experiment.problem.difficulty,
    }
    
    if wandb_run is not None:
        wandb.log(final_metrics)
        wandb.finish()
    
    logger.info("Demo completed successfully!")

def convert_to_logging_config(hydra_logging_cfg: DictConfig) -> LoggingConfig:
    """Convert Hydra logging config to existing Pydantic LoggingConfig."""
    return LoggingConfig(
        log_level=hydra_logging_cfg.console.level,
        log_frequency=hydra_logging_cfg.console.frequency,
        log_gradients=hydra_logging_cfg.metrics.log_gradients,
        log_weights=hydra_logging_cfg.metrics.log_weights,
        log_activations=hydra_logging_cfg.metrics.log_activations,
        enable_tensorboard=hydra_logging_cfg.tensorboard.enabled,
        enable_wandb=hydra_logging_cfg.wandb.enabled,
        project_name=hydra_logging_cfg.wandb.project,
        tags=hydra_logging_cfg.wandb.tags,
    )

def convert_to_grpo_config(hydra_training_cfg: DictConfig) -> Dict[str, Any]:
    """Convert Hydra training config to GRPO parameters."""
    return {
        "learning_rate": hydra_training_cfg.algorithm.learning_rate,
        "batch_size": hydra_training_cfg.algorithm.batch_size,
        "entropy_coefficient": hydra_training_cfg.algorithm.entropy_coefficient,
        "clip_ratio": hydra_training_cfg.algorithm.clip_ratio,
    }

def simulate_training_step(key: jax.Array, step: int, cfg: DictConfig) -> Dict[str, float]:
    """Simulate training step with realistic causal discovery metrics."""
    # Simulate convergence behavior based on config
    progress = step / cfg.max_steps
    lr_factor = cfg.training.algorithm.learning_rate * 1000  # Scale for demo
    
    # Loss decreases with progress, varies with learning rate
    base_loss = 2.0 * jnp.exp(-progress * lr_factor)
    noise = jax.random.normal(key, shape=()) * 0.1
    loss = float(base_loss + noise)
    
    # Causal discovery metrics improve with progress
    shd = float(5 * (1 - progress) + jax.random.normal(key, shape=()) * 0.5)  # Structural Hamming Distance
    precision = float(0.3 + 0.6 * progress + jax.random.normal(key, shape=()) * 0.05)
    recall = float(0.2 + 0.7 * progress + jax.random.normal(key, shape=()) * 0.05)
    
    # Intervention efficiency (higher is better)
    intervention_efficiency = float(0.5 + 0.4 * progress)
    
    return {
        "loss": max(0.01, loss),
        "shd": max(0, shd),
        "precision": jnp.clip(precision, 0, 1),
        "recall": jnp.clip(recall, 0, 1),
        "f1": 2 * precision * recall / (precision + recall + 1e-8),
        "intervention_efficiency": intervention_efficiency,
        "learning_rate": cfg.training.algorithm.learning_rate,
        "batch_size": cfg.training.algorithm.batch_size,
    }

def demonstrate_config_effects(cfg: DictConfig) -> None:
    """Show how different configurations affect behavior."""
    logger.info("\\n=== Configuration Effects Demo ===")
    
    # Show different behavior based on difficulty
    if cfg.experiment.problem.difficulty == "easy":
        logger.info("Easy problem: Using simpler algorithms and faster convergence")
    elif cfg.experiment.problem.difficulty == "hard":
        logger.info("Hard problem: Using more sophisticated algorithms and longer training")
    else:
        logger.info(f"Medium problem ({cfg.experiment.problem.difficulty}): Balanced approach")
    
    # Show curriculum effects
    if cfg.training.curriculum.enabled:
        logger.info(f"Curriculum learning enabled with {cfg.training.curriculum.num_stages} stages")
    else:
        logger.info("Standard training without curriculum")
    
    # Show optimization effects
    if cfg.training.algorithm.learning_rate > 0.001:
        logger.info("High learning rate: Faster but potentially less stable training")
    else:
        logger.info("Conservative learning rate: Slower but more stable training")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo_hydra_wandb_integration()
