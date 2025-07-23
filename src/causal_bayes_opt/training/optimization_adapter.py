#!/usr/bin/env python3
"""
Optimization Direction Adapter for GRPO Training

Handles conversion between minimization and maximization objectives
to ensure consistent training signals regardless of optimization direction.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class OptimizationAdapter:
    """
    Adapter for handling different optimization directions in GRPO training.
    
    This ensures that the internal GRPO training always maximizes rewards,
    while the external interface can support both minimization and maximization.
    """
    direction: str  # "MINIMIZE" or "MAXIMIZE"
    target_baseline: float = 0.0
    
    def __post_init__(self):
        if self.direction not in ["MINIMIZE", "MAXIMIZE"]:
            raise ValueError(f"Invalid optimization direction: {self.direction}")
        self.is_minimizing = self.direction == "MINIMIZE"
    
    def adapt_target_value_for_reward(self, target_value: float) -> float:
        """
        Convert target value to reward signal for GRPO training.
        
        For maximization: reward = target_value
        For minimization: reward = -target_value + baseline
        
        This ensures GRPO always maximizes, but the effect is to
        minimize the target when needed.
        """
        if self.is_minimizing:
            # Negate so that lower target values produce higher rewards
            return -target_value + self.target_baseline
        else:
            # Direct mapping for maximization
            return target_value
    
    def adapt_reward_components(self, components: Dict[str, float]) -> Dict[str, float]:
        """
        Adapt reward components based on optimization direction.
        
        This modifies the optimization component to align with the
        desired direction while keeping other components unchanged.
        """
        adapted = components.copy()
        
        if 'optimization' in adapted and self.is_minimizing:
            # Negate optimization component for minimization
            adapted['optimization'] = -adapted['optimization']
            adapted['optimization_note'] = 'negated_for_minimization'
        
        return adapted
    
    def compute_intervention_reward(self,
                                   intervention_targets: set,
                                   intervention_values: Dict[str, float],
                                   target_variable: str,
                                   target_change: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for an intervention based on optimization direction.
        
        Args:
            intervention_targets: Variables being intervened on
            intervention_values: Values of interventions
            target_variable: The target variable to optimize
            target_change: Optional change in target value (if available)
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        reward_components = {}
        
        # Base reward components (direction-agnostic)
        if intervention_targets:
            # Exploration bonus for intervening
            exploration_bonus = 0.2
            reward_components['exploration'] = exploration_bonus
            
            # Penalty if intervening on target (invalid action)
            if target_variable in intervention_targets:
                target_penalty = -0.5
                reward_components['target_penalty'] = target_penalty
            else:
                # Bonus for valid intervention
                valid_bonus = 0.3
                reward_components['valid_intervention'] = valid_bonus
            
            # Intervention magnitude component
            magnitude = sum(abs(v) for v in intervention_values.values())
            magnitude_bonus = min(0.5, magnitude * 0.1)
            reward_components['magnitude'] = magnitude_bonus
        else:
            # No intervention penalty
            no_action_penalty = -0.1
            reward_components['no_action'] = no_action_penalty
        
        # Target optimization component (if target change is known)
        if target_change is not None and target_variable not in intervention_targets:
            # Scale target change to reasonable reward range
            optimization_reward = target_change * 0.1  # Scale factor
            
            # Adapt based on direction
            if self.is_minimizing:
                # Negate so that decreases in target produce positive rewards
                optimization_reward = -optimization_reward
                reward_components['optimization'] = optimization_reward
                reward_components['optimization_direction'] = 'minimize'
            else:
                reward_components['optimization'] = optimization_reward
                reward_components['optimization_direction'] = 'maximize'
        
        # Sum components
        total_reward = sum(reward_components.values())
        
        # Clip to reasonable range
        total_reward = max(-1.0, min(1.0, total_reward))
        
        return total_reward, reward_components
    
    def format_metrics_for_display(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics for display based on optimization direction.
        
        This ensures that metrics are shown in a way that makes sense
        for the optimization direction (e.g., "reduction" vs "improvement").
        """
        formatted = metrics.copy()
        
        if self.is_minimizing:
            # Rename metrics for minimization context
            if 'target_improvement' in formatted:
                formatted['target_reduction'] = -formatted['target_improvement']
                del formatted['target_improvement']
            
            if 'best_target_value' in formatted:
                formatted['best_target_value_note'] = 'lower_is_better'
        else:
            if 'best_target_value' in formatted:
                formatted['best_target_value_note'] = 'higher_is_better'
        
        formatted['optimization_direction'] = self.direction
        
        return formatted
    
    def convert_demonstration_reward(self, demonstration_reward: float) -> float:
        """
        Convert demonstration rewards from external sources.
        
        This is important when using demonstrations from algorithms like
        PARENT_SCALE that minimize, while GRPO training expects maximization.
        """
        if self.is_minimizing:
            # If we're training to minimize and demos are from minimization,
            # no conversion needed
            return demonstration_reward
        else:
            # If we're training to maximize but demos are from minimization,
            # we need to negate
            logger.info("Converting minimization demonstrations to maximization format")
            return -demonstration_reward


def create_optimization_adapter(config: Dict[str, Any]) -> OptimizationAdapter:
    """
    Create optimization adapter from configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        OptimizationAdapter instance
    """
    optimization_config = config.get('optimization', {})
    
    # Default to maximization if not specified
    direction = optimization_config.get('direction', 'MAXIMIZE')
    target_baseline = optimization_config.get('target_baseline', 0.0)
    
    return OptimizationAdapter(
        direction=direction,
        target_baseline=target_baseline
    )