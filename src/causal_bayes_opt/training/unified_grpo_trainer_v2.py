"""
Unified GRPO trainer - refactored with modular components.

This is a clean refactored version that maintains the same public interface
but uses modular components for better maintainability.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
from omegaconf import DictConfig
import pyrsistent as pyr

from .grpo_trainer_core import GRPOTrainerCore, GRPOTrainerConfig
from .grpo_reward_computer import GRPORewardComputer, create_reward_computer_from_config
from .grpo_logger import GRPOLogger
from .convergence_detector import ConvergenceDetector, ConvergenceConfig
from ..acquisition.better_rewards import RunningStats
from ..data_structures.scm import get_variables, get_target, get_parents

logger = logging.getLogger(__name__)


class UnifiedGRPOTrainerV2:
    """
    Refactored unified GRPO trainer using modular components.
    
    Maintains the same public interface as the original UnifiedGRPOTrainer
    but with much cleaner internal structure.
    """
    
    def __init__(self, 
                 config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
                 **kwargs):
        """Initialize trainer with backward compatibility."""
        # Handle config conversion
        if config is not None:
            if isinstance(config, DictConfig):
                config = dict(config)
            self.config = config
        else:
            # Create config from kwargs for backward compatibility
            self.config = kwargs
        
        # Create core trainer
        self.core = GRPOTrainerCore(self.config)
        
        # Initialize convergence detection if needed
        self.use_early_stopping = self.config.get('use_early_stopping', False)
        if self.use_early_stopping:
            conv_config = self.config.get('convergence', {})
            self.convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=conv_config.get('accuracy_threshold', 0.95),
                patience=conv_config.get('patience', 5),
                min_episodes=conv_config.get('min_episodes', 5),
                max_episodes_per_scm=conv_config.get('max_episodes_per_scm', 200)
            )
            self.convergence_detector = ConvergenceDetector(self.convergence_config)
        else:
            self.convergence_detector = None
        
        # Expose core components for backward compatibility
        self.policy_params = self.core.policy_params
        self.policy_fn = self.core.policy_fn
        self.optimizer_state = self.core.optimizer_state
        self.reward_stats = self.core.reward_computer.get_stats()
        self.rng_key = self.core.rng_key  # Expose RNG key
        self.grpo_config = self.core.grpo_config  # Expose GRPO config
        
        # Training state for compatibility
        self.training_metrics = []
        self.episode_count = 0
        self.training_step = 0
        
        # Configuration attributes for compatibility
        self.max_episodes = self.config.get('max_episodes', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.optimization_direction = self.config.get('optimization_direction', 'MINIMIZE')
        self.use_surrogate = self.config.get('use_surrogate', False)
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        self.reward_weights = self.config.get('reward_weights', {'target': 0.7, 'parent': 0.1, 'info_gain': 0.2})
        
        # Additional attributes expected by JointACBOTrainer
        self.obs_per_episode = self.config.get('obs_per_episode', 100)
        self.max_interventions = self.config.get('max_interventions', 10)
        self.n_variables_range = self.config.get('n_variables_range', [3, 8])
        self.seed = self.config.get('seed', 42)
        
        logger.info("Initialized UnifiedGRPOTrainerV2 with modular components")
    
    def train(self, 
              scms: Union[List[Any], Dict[str, Any], Callable[[], Any]],
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Train GRPO policy - delegates to core trainer.
        
        Maintains the same interface as original UnifiedGRPOTrainer.
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting unified GRPO training (v2 - modular)")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info(f"  Reward type: {self.config.get('reward_type', 'composite')}")
        logger.info(f"{'='*70}\n")
        
        # Delegate to core trainer
        results = self.core.train(scms)
        
        # Update training state for compatibility
        self.policy_params = self.core.policy_params
        self.training_metrics = results.get('all_metrics', [])
        self.episode_count = len(self.training_metrics)
        
        # Add compatibility fields
        results['converged'] = False  # Would need convergence detection
        results['episodes_per_scm'] = {}  # Not tracked in core
        
        return results
    
    def _run_grpo_episode(self, episode_idx: int, scm: Any, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Run single episode - delegates to core trainer."""
        return self.core._run_episode(episode_idx, scm, scm_name, key)
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save checkpoint using simplified logic."""
        from ..utils.checkpoint_utils import save_checkpoint
        
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = "unified_grpo_final" if is_final else f"unified_grpo_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        
        architecture = {
            'hidden_dim': self.core.config.hidden_dim,
            'architecture_type': self.core.config.policy_architecture
        }
        
        training_config = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_episodes': self.max_episodes,
            'optimization_direction': self.optimization_direction
        }
        
        metadata = {
            'trainer_type': 'UnifiedGRPOTrainerV2',
            'episode': self.episode_count,
            'is_final': is_final,
            'uses_modular_components': True
        }
        
        save_checkpoint(
            path=checkpoint_path,
            params=self.policy_params,
            architecture=architecture,
            model_type='policy',
            model_subtype='grpo',
            training_config=training_config,
            metadata=metadata
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _prepare_scms(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> List[Tuple[str, Any]]:
        """Convert various SCM formats to standard list of (name, scm) tuples."""
        if isinstance(scms, list):
            if scms and isinstance(scms[0], tuple) and len(scms[0]) == 2:
                return scms
            else:
                return [(f"scm_{i}", scm) for i, scm in enumerate(scms)]
        elif isinstance(scms, dict):
            return list(scms.items())
        elif callable(scms):
            generated = []
            for i in range(10):
                scm = scms()
                generated.append((f"generated_{i}", scm))
            return generated
        else:
            raise ValueError(f"Unsupported SCM format: {type(scms)}")


# Factory function for backward compatibility
def create_unified_grpo_trainer(config: Union[DictConfig, Dict[str, Any], None] = None,
                               **kwargs) -> UnifiedGRPOTrainerV2:
    """Create unified GRPO trainer with backward compatibility."""
    return UnifiedGRPOTrainerV2(config=config, **kwargs)