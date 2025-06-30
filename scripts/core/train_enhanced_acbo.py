#!/usr/bin/env python3
"""
Enhanced ACBO Training Script with GRPO.

This script trains enhanced ACBO policies using GRPO with our improved architectures:
- Continuous parent set models for scalable structure learning
- Enriched transformer architecture with multi-channel temporal input
- Enhanced reward systems with verifiable ground truth

Usage:
    # Train with default enhanced configuration
    python scripts/train_enhanced_acbo.py
    
    # Train with specific architecture level
    python scripts/train_enhanced_acbo.py training.algorithm.enhanced_architecture_level=simplified
    
    # Train with curriculum learning
    python scripts/train_enhanced_acbo.py training.curriculum.enabled=true
    
    # Train with custom problem size
    python scripts/train_enhanced_acbo.py environment.num_variables=10 environment.edge_density=0.3

Features:
- Hydra configuration management
- WandB logging and monitoring
- Curriculum learning progression
- Enhanced architecture validation
- Checkpointing and resume capability
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# JAX and numerical libraries
import numpy as onp

# Enhanced ACBO components
from causal_bayes_opt.training.grpo_training_manager import (
    GRPOTrainingManager, create_grpo_training_manager
)
from causal_bayes_opt.training.grpo_config import (
    ComprehensiveGRPOConfig, create_production_grpo_config
)
from causal_bayes_opt.acquisition.enhanced_policy_network import (
    create_enhanced_policy_for_grpo, validate_enhanced_policy_integration
)
from causal_bayes_opt.avici_integration.enhanced_surrogate import (
    create_enhanced_surrogate_for_grpo, validate_enhanced_surrogate_integration
)
from causal_bayes_opt.acquisition.reward_rubric import CausalRewardRubric
from causal_bayes_opt.environments.intervention_env import InterventionEnvironment
from causal_bayes_opt.experiments.benchmark_graphs import create_erdos_renyi_scm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="enhanced_acbo_training_config")
def train_enhanced_acbo(cfg: DictConfig) -> None:
    """Main enhanced ACBO training function."""
    
    logger.info("Starting Enhanced ACBO Training with GRPO")
    logger.info(f"Configuration:\\n{OmegaConf.to_yaml(cfg)}")
    
    # Validate enhanced integration
    if not validate_enhanced_integration():
        logger.error("Enhanced integration validation failed")
        return
    
    # Initialize WandB if enabled
    wandb_run = None
    if cfg.training.logging.enable_wandb and WANDB_AVAILABLE:
        wandb_run = initialize_wandb_logging(cfg)
    
    # Create training environment
    try:
        training_results = run_enhanced_acbo_training(cfg, wandb_run)
        
        # Log final results
        if wandb_run:
            log_final_training_results(training_results, wandb_run)
        
        # Save training artifacts
        save_training_artifacts(training_results, cfg)
        
        logger.info("Enhanced ACBO training completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced ACBO training failed: {e}")
        if wandb_run:
            wandb.log({"training_status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


def validate_enhanced_integration() -> bool:
    """Validate that enhanced components are working correctly."""
    logger.info("Validating enhanced integration...")
    
    # Validate enhanced policy integration
    if not validate_enhanced_policy_integration():
        logger.error("Enhanced policy integration validation failed")
        return False
    
    # Validate enhanced surrogate integration
    if not validate_enhanced_surrogate_integration():
        logger.error("Enhanced surrogate integration validation failed")
        return False
    
    logger.info("Enhanced integration validation passed")
    return True


def run_enhanced_acbo_training(cfg: DictConfig, wandb_run: Optional[Any]) -> Dict[str, Any]:
    """Run the main enhanced ACBO training loop."""
    
    # Extract training parameters
    variables = [f'X{i}' for i in range(cfg.environment.num_variables)]
    target_variable = f'X{cfg.environment.get("target_variable_idx", 0)}'
    
    logger.info(f"Training enhanced ACBO with {len(variables)} variables, target: {target_variable}")
    
    # Create enhanced networks
    enhanced_policy_fn, policy_config = create_enhanced_policy_for_grpo(
        variables=variables,
        target_variable=target_variable,
        architecture_level=cfg.training.algorithm.get('enhanced_architecture_level', 'full'),
        performance_mode=cfg.training.algorithm.get('enhanced_performance_mode', 'balanced')
    )
    
    enhanced_surrogate_fn, surrogate_config = create_enhanced_surrogate_for_grpo(
        variables=variables,
        target_variable=target_variable,
        model_complexity=cfg.training.surrogate.get('model_complexity', 'full'),
        use_continuous=cfg.training.surrogate.get('use_continuous', True),
        performance_mode=cfg.training.algorithm.get('enhanced_performance_mode', 'balanced')
    )
    
    # Create training environment
    environment = create_enhanced_training_environment(cfg, variables, target_variable)
    
    # Create reward rubric
    reward_rubric = create_enhanced_reward_rubric(cfg, variables, target_variable)
    
    # Create GRPO training configuration
    grpo_config = create_enhanced_grpo_config(cfg, policy_config, surrogate_config)
    
    # Create and run training manager
    training_manager = create_grpo_training_manager(
        config=grpo_config,
        environment=environment,
        reward_rubric=reward_rubric,
        policy_network=enhanced_policy_fn,
        value_network=enhanced_policy_fn  # Use same network for value estimation
    )
    
    # Run training
    logger.info("Starting GRPO training with enhanced architectures...")
    training_session = training_manager.train()
    
    # Extract results
    training_results = {
        'session': training_session,
        'policy_config': policy_config,
        'surrogate_config': surrogate_config,
        'grpo_config': grpo_config,
        'final_performance': training_session.final_performance,
        'training_time': training_session.training_time,
        'total_steps': training_session.total_steps,
        'convergence_achieved': training_session.convergence_achieved,
        'best_checkpoint': training_session.best_checkpoint
    }
    
    return training_results


def create_enhanced_training_environment(
    cfg: DictConfig, 
    variables: List[str], 
    target_variable: str
) -> InterventionEnvironment:
    """Create enhanced training environment."""
    
    # Create SCM for training
    scm = create_erdos_renyi_scm(
        n_nodes=len(variables),
        edge_prob=cfg.environment.edge_density,
        noise_scale=cfg.environment.get('noise_scale', 0.1),
        seed=cfg.seed
    )
    
    # Create intervention environment
    # Note: This is a placeholder - in a full implementation would create proper environment
    class MockInterventionEnvironment:
        def __init__(self, scm, variables, target_variable):
            self.scm = scm
            self.variables = variables
            self.target_variable = target_variable
            
        def reset(self, key):
            # Return mock state
            return type('State', (), {
                'mechanism_features': jnp.ones(len(self.variables))
            })()
            
        def step(self, action):
            # Return mock next state and info
            next_state = type('State', (), {
                'mechanism_features': jnp.ones(len(self.variables))
            })()
            env_info = type('EnvInfo', (), {
                'episode_complete': True
            })()
            return next_state, env_info
    
    return MockInterventionEnvironment(scm, variables, target_variable)


def create_enhanced_reward_rubric(
    cfg: DictConfig,
    variables: List[str],
    target_variable: str
) -> CausalRewardRubric:
    """Create enhanced reward rubric."""
    
    # Create reward configuration
    reward_config = {
        'use_verifiable_rewards': cfg.training.rewards.get('use_verifiable_rewards', True),
        'target_improvement_weight': cfg.training.rewards.get('target_improvement_weight', 2.0),
        'true_parent_weight': cfg.training.rewards.get('true_parent_weight', 1.0),
        'exploration_weight': cfg.training.rewards.get('exploration_weight', 0.5),
    }
    
    # Note: This is a placeholder - in a full implementation would create proper reward rubric
    class MockCausalRewardRubric:
        def __init__(self, config):
            self.config = config
            
        def compute_reward(self, state, action, next_state):
            # Return mock reward
            return type('RewardResult', (), {
                'total_reward': 1.0,
                'components': {'improvement': 0.5, 'exploration': 0.3, 'structure': 0.2}
            })()
    
    return MockCausalRewardRubric(reward_config)


def create_enhanced_grpo_config(
    cfg: DictConfig,
    policy_config: Dict[str, Any],
    surrogate_config: Dict[str, Any]
) -> ComprehensiveGRPOConfig:
    """Create enhanced GRPO configuration."""
    
    # Start with production config
    grpo_config = create_production_grpo_config(
        max_training_steps=cfg.training.get('max_training_steps', 5000)
    )
    
    # Enhance with specific configuration
    # Note: This would need to be expanded to properly configure all enhanced features
    
    return grpo_config


def initialize_wandb_logging(cfg: DictConfig) -> Any:
    """Initialize WandB logging for enhanced training."""
    
    if not WANDB_AVAILABLE:
        logger.warning("WandB not available - install with: pip install wandb")
        return None
    
    # Prepare config for logging
    wandb_config = {
        "enhanced_training": True,
        "architecture_level": cfg.training.algorithm.get('enhanced_architecture_level', 'full'),
        "performance_mode": cfg.training.algorithm.get('enhanced_performance_mode', 'balanced'),
        "use_continuous_surrogate": cfg.training.surrogate.get('use_continuous', True),
        "num_variables": cfg.environment.num_variables,
        "edge_density": cfg.environment.edge_density,
        "learning_rate": cfg.training.algorithm.learning_rate,
        "batch_size": cfg.training.algorithm.batch_size,
        "max_training_steps": cfg.training.get('max_training_steps', 5000),
        "curriculum_enabled": cfg.training.curriculum.get('enabled', False),
    }
    
    # Initialize wandb run
    wandb_run = wandb.init(
        project=cfg.training.logging.wandb.get('project', 'enhanced_acbo_training'),
        name=f"enhanced_acbo_{cfg.training.algorithm.enhanced_architecture_level}_{int(time.time())}",
        config=wandb_config,
        tags=cfg.training.logging.wandb.get('tags', ['enhanced', 'acbo', 'grpo']),
        group="enhanced_architectures"
    )
    
    logger.info(f"WandB logging initialized: {wandb.run.url}")
    return wandb_run


def log_final_training_results(training_results: Dict[str, Any], wandb_run: Any) -> None:
    """Log final training results to WandB."""
    
    session = training_results['session']
    
    final_metrics = {
        "training/final_performance": session.final_performance.get('best_reward', 0.0),
        "training/total_steps": session.total_steps,
        "training/total_episodes": session.total_episodes,
        "training/training_time": session.training_time,
        "training/convergence_achieved": session.convergence_achieved,
        "training/early_stopped": session.early_stopped,
        "training/checkpoints_saved": len(session.checkpoints_saved),
        "enhanced/architecture_validation": True,
        "enhanced/policy_config": str(training_results['policy_config']),
        "enhanced/surrogate_config": str(training_results['surrogate_config']),
    }
    
    wandb_run.log(final_metrics)
    logger.info("Final training results logged to WandB")


def save_training_artifacts(training_results: Dict[str, Any], cfg: DictConfig) -> None:
    """Save training artifacts locally."""
    
    timestamp = int(time.time())
    artifacts_dir = Path(f"training_artifacts/enhanced_acbo_{timestamp}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = artifacts_dir / "training_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    # Save training results summary
    results_summary = {
        'final_performance': training_results['session'].final_performance,
        'training_time': training_results['session'].training_time,
        'total_steps': training_results['session'].total_steps,
        'convergence_achieved': training_results['session'].convergence_achieved,
        'enhanced_architecture_level': cfg.training.algorithm.get('enhanced_architecture_level', 'full'),
        'num_variables': cfg.environment.num_variables,
        'timestamp': timestamp
    }
    
    import json
    results_path = artifacts_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Training artifacts saved to {artifacts_dir}")


# Create default enhanced training config
def create_default_enhanced_config() -> DictConfig:
    """Create default enhanced training configuration."""
    
    config = OmegaConf.create({
        'seed': 42,
        'environment': {
            'num_variables': 8,
            'edge_density': 0.3,
            'noise_scale': 0.1,
            'target_variable_idx': 0
        },
        'training': {
            'max_training_steps': 2000,
            'algorithm': {
                'learning_rate': 0.0003,
                'batch_size': 32,
                'enhanced_architecture_level': 'full',
                'enhanced_performance_mode': 'balanced',
                'use_enhanced_networks': True
            },
            'surrogate': {
                'model_complexity': 'full',
                'use_continuous': True
            },
            'curriculum': {
                'enabled': False
            },
            'rewards': {
                'use_verifiable_rewards': True,
                'target_improvement_weight': 2.0,
                'true_parent_weight': 1.0,
                'exploration_weight': 0.5
            },
            'logging': {
                'enable_wandb': False,
                'wandb': {
                    'project': 'enhanced_acbo_training',
                    'tags': ['enhanced', 'acbo', 'grpo']
                }
            }
        }
    })
    
    return config


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create enhanced training config if needed
    config_path = Path(__file__).parent.parent / "config" / "enhanced_acbo_training_config.yaml"
    if not config_path.exists():
        logger.info(f"Creating default enhanced training config at {config_path}")
        default_config = create_default_enhanced_config()
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            OmegaConf.save(default_config, f)
    
    # Run training
    train_enhanced_acbo()