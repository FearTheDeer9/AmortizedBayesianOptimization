"""
Unified GRPO reward computation module.

This module provides a clean interface for all reward computation types:
- Composite rewards (target + parent + info_gain)
- Binary rewards (+1/-1 based on mean)
- Clean rewards (improvement-based)

Consolidates the various reward systems into a single interface.
"""

import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..acquisition.composite_reward import compute_composite_reward, RewardConfig
from ..acquisition.clean_rewards import compute_clean_reward
from ..acquisition.better_rewards import compute_better_clean_reward, RunningStats
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values

logger = logging.getLogger(__name__)


@dataclass
class GRPORewardConfig:
    """Unified configuration for GRPO reward computation."""
    # Reward type
    reward_type: str = "composite"  # "composite", "clean", "binary", "better_clean"
    
    # Optimization
    optimization_direction: str = "MINIMIZE"
    
    # Weights (for composite rewards)
    target_weight: float = 0.7
    parent_weight: float = 0.1
    info_gain_weight: float = 0.2
    
    # Binary reward settings
    use_running_stats: bool = True
    stats_window_size: int = 1000
    
    # Better clean reward settings
    better_reward_type: str = "adaptive_sigmoid"
    temperature_factor: float = 2.0
    
    def to_legacy_config(self) -> Dict[str, Any]:
        """Convert to legacy config format for compatibility."""
        return {
            'reward_type': self.reward_type,
            'optimization_direction': self.optimization_direction,
            'weights': {
                'target': self.target_weight,
                'parent': self.parent_weight,
                'info_gain': self.info_gain_weight,
                'diversity': 0.0,
                'exploration': 0.0
            },
            'temperature_factor': self.temperature_factor
        }


class GRPORewardComputer:
    """Unified reward computation for GRPO training."""
    
    def __init__(self, config: GRPORewardConfig):
        self.config = config
        
        # Initialize running stats for binary rewards
        if config.use_running_stats:
            self.stats = RunningStats(window_size=config.stats_window_size)
        else:
            self.stats = None
        
        # Create legacy RewardConfig for composite rewards
        self.legacy_reward_config = RewardConfig(
            target_weight=config.target_weight,
            parent_weight=config.parent_weight,
            info_gain_weight=config.info_gain_weight,
            optimization_direction=config.optimization_direction,
            reward_type=config.reward_type,
            stats=self.stats
        )
        
        logger.info(f"Initialized GRPORewardComputer with reward_type={config.reward_type}")
    
    def compute_reward(self,
                      intervention: Dict[str, Any],
                      outcome_sample: Any,
                      buffer: ExperienceBuffer,
                      scm: Any,
                      target_variable: str,
                      variables: list,
                      surrogate_predict_fn: Optional[Callable] = None,
                      tensor_5ch: Optional[Any] = None,
                      mapper: Optional[Any] = None) -> Dict[str, Any]:
        """
        Unified reward computation interface.
        
        Args:
            intervention: Applied intervention
            outcome_sample: Result sample
            buffer: Experience buffer before intervention
            scm: Structural causal model
            target_variable: Target variable name
            variables: List of all variables
            surrogate_predict_fn: Optional surrogate function
            tensor_5ch: Optional 5-channel tensor
            mapper: Optional variable mapper
            
        Returns:
            Dictionary with reward components and total
        """
        if self.config.reward_type == "composite":
            return compute_composite_reward(
                intervention=intervention,
                outcome_sample=outcome_sample,
                buffer=buffer,
                scm=scm,
                target_variable=target_variable,
                variables=variables,
                surrogate_predict_fn=surrogate_predict_fn,
                config=self.legacy_reward_config,
                tensor_5ch=tensor_5ch,
                mapper=mapper,
                reward_type=self.config.reward_type,
                stats=self.stats
            )
        
        elif self.config.reward_type == "clean":
            return compute_clean_reward(
                buffer_before=buffer,
                intervention=intervention,
                outcome=outcome_sample,
                target_variable=target_variable,
                config=self.config.to_legacy_config()
            )
        
        elif self.config.reward_type == "better_clean" or self.config.reward_type == "binary":
            return compute_better_clean_reward(
                buffer_before=buffer,
                intervention=intervention,
                outcome=outcome_sample,
                target_variable=target_variable,
                config=self.config.to_legacy_config(),
                stats=self.stats
            )
        
        else:
            # Default to composite
            logger.warning(f"Unknown reward_type {self.config.reward_type}, using composite")
            return compute_composite_reward(
                intervention=intervention,
                outcome_sample=outcome_sample,
                buffer=buffer,
                scm=scm,
                target_variable=target_variable,
                variables=variables,
                surrogate_predict_fn=surrogate_predict_fn,
                config=self.legacy_reward_config,
                tensor_5ch=tensor_5ch,
                mapper=mapper,
                reward_type="composite",
                stats=self.stats
            )
    
    def get_stats(self) -> Optional[RunningStats]:
        """Get running statistics for external access."""
        return self.stats


def create_reward_computer_from_config(config: Dict[str, Any]) -> GRPORewardComputer:
    """Factory function to create reward computer from training config."""
    reward_config = GRPORewardConfig(
        reward_type=config.get('reward_type', 'composite'),
        optimization_direction=config.get('optimization_direction', 'MINIMIZE'),
        target_weight=config.get('reward_weights', {}).get('target', 0.7),
        parent_weight=config.get('reward_weights', {}).get('parent', 0.1),
        info_gain_weight=config.get('reward_weights', {}).get('info_gain', 0.2),
        use_running_stats=config.get('use_running_stats', True),
        stats_window_size=config.get('stats_window_size', 1000),
        better_reward_type=config.get('better_reward_type', 'adaptive_sigmoid'),
        temperature_factor=config.get('temperature_factor', 2.0)
    )
    
    return GRPORewardComputer(reward_config)