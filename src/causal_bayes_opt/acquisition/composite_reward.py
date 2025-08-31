"""
Configurable composite reward system for ACBO training.

This module provides a clean, configurable reward system with three components:
1. Target reward: Raw target node value optimization
2. Information gain: Entropy reduction in surrogate posterior
3. Parent reward: Binary reward for intervening on direct parents

All components are configurable via RewardConfig.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import jax.numpy as jnp
import numpy as np

from ..data_structures.scm import get_parents
from ..data_structures.sample import get_values
from ..data_structures.buffer import ExperienceBuffer
from ..training.three_channel_converter import buffer_to_three_channel_tensor

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for composite reward system."""
    target_weight: float = 0.7
    info_gain_weight: float = 0.2
    parent_weight: float = 0.1
    optimization_direction: str = "MINIMIZE"
    reward_type: str = "continuous"
    stats: Optional[Any] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if abs(self.target_weight + self.info_gain_weight + self.parent_weight - 1.0) > 0.01:
            logger.warning(f"Reward weights don't sum to 1.0: {self.target_weight + self.info_gain_weight + self.parent_weight:.3f}")


def compute_target_reward(outcome_sample: Any, target_variable: str, 
                         optimization_direction: str = "MINIMIZE", 
                         reward_type: str = "continuous",
                         stats: Optional[Any] = None) -> float:
    """
    Compute raw target node value reward.
    
    Args:
        outcome_sample: Sample containing target value
        target_variable: Name of target variable
        optimization_direction: "MINIMIZE" or "MAXIMIZE"
        reward_type: "continuous" or "binary"
        stats: Running statistics for binary reward (required if reward_type="binary")
        
    Returns:
        Raw target reward (GRPO will handle baseline/normalization)
    """
    try:
        target_value = get_values(outcome_sample)[target_variable]
        
        if reward_type == "binary" and stats is not None:
            # Binary reward: +1 if above mean, -1 if below mean
            stats.update(target_value)
            current_mean = stats.mean
            
            if target_value > current_mean:
                reward = 1.0
            else:
                reward = -1.0
            
            # Flip sign if minimizing (above mean is bad for minimization)
            if optimization_direction == "MINIMIZE":
                reward = -reward
                
            logger.info(f"[BINARY TARGET REWARD] Value: {target_value:.3f}, Mean: {current_mean:.3f}, Binary reward: {reward:.1f}")
            
        else:
            # Continuous reward (original behavior)
            # For MINIMIZE: negative target values are good (positive reward)  
            # For MAXIMIZE: positive target values are good (positive reward)
            if optimization_direction == "MINIMIZE":
                reward = -float(target_value)
            else:
                reward = float(target_value)
        
        return reward
        
    except Exception as e:
        logger.error(f"Error computing target reward: {e}")
        return 0.0


def compute_information_gain_reward(
    buffer: ExperienceBuffer,
    intervention: Dict[str, Any],
    outcome_sample: Any,
    surrogate_predict_fn: Optional[Callable],
    target_variable: str,
    variables: list,
    tensor_5ch: Optional[Any] = None,
    mapper: Optional[Any] = None,
    info_gain_type: str = "entropy_reduction"
) -> float:
    """
    Compute information gain using entropy reduction or probability change.
    
    Args:
        buffer: Current experience buffer (before intervention)
        intervention: Intervention that was applied
        outcome_sample: Result of the intervention
        surrogate_predict_fn: Function to get surrogate predictions
        target_variable: Target variable name
        variables: List of all variables
        tensor_5ch: Optional pre-computed 5-channel tensor
        mapper: Optional pre-computed mapper
        info_gain_type: "entropy_reduction" or "probability_change"
        
    Returns:
        Information gain (positive = good)
    """
    if surrogate_predict_fn is None:
        return 0.0
    
    try:
        if info_gain_type == "probability_change":
            # User's preferred: sum of absolute probability changes
            return _compute_probability_change_info_gain(
                buffer, intervention, outcome_sample, surrogate_predict_fn, 
                target_variable, variables
            )
        else:
            # Original: entropy reduction
            return _compute_entropy_reduction_info_gain(
                buffer, intervention, outcome_sample, surrogate_predict_fn,
                target_variable, variables, tensor_5ch, mapper
            )
        
    except Exception as e:
        logger.error(f"Error computing information gain: {e}")
        return 0.0


def _compute_probability_change_info_gain(
    buffer: ExperienceBuffer,
    intervention: Dict[str, Any], 
    outcome_sample: Any,
    surrogate_predict_fn: Callable,
    target_variable: str,
    variables: list
) -> float:
    """
    Compute info gain as sum of absolute probability changes.
    
    Returns:
        sum(|prob_after[i] - prob_before[i]| for i in variables)
    """
    # Get before probabilities
    from ..training.three_channel_converter import buffer_to_three_channel_tensor
    tensor_before, _ = buffer_to_three_channel_tensor(
        buffer, target_variable, max_history_size=100, standardize=False
    )
    
    posterior_before = surrogate_predict_fn(tensor_before, target_variable, variables)
    if 'parent_probs' not in posterior_before:
        return 0.0
    
    probs_before = posterior_before['parent_probs']
    
    # Get after probabilities  
    hypo_buffer = _copy_buffer(buffer)
    hypo_buffer.add_intervention(intervention, outcome_sample)
    
    tensor_after, _ = buffer_to_three_channel_tensor(
        hypo_buffer, target_variable, max_history_size=100, standardize=False
    )
    
    posterior_after = surrogate_predict_fn(tensor_after, target_variable, variables)
    if 'parent_probs' not in posterior_after:
        return 0.0
    
    probs_after = posterior_after['parent_probs']
    
    # Compute absolute change sum
    prob_changes = []
    for i in range(min(len(probs_before), len(probs_after))):
        change = abs(float(probs_after[i]) - float(probs_before[i]))
        prob_changes.append(change)
    
    total_change = sum(prob_changes)
    
    logger.debug(f"Probability changes: {prob_changes}, Total: {total_change:.4f}")
    return total_change


def _compute_entropy_reduction_info_gain(
    buffer: ExperienceBuffer,
    intervention: Dict[str, Any],
    outcome_sample: Any, 
    surrogate_predict_fn: Callable,
    target_variable: str,
    variables: list,
    tensor_5ch: Optional[Any] = None,
    mapper: Optional[Any] = None
) -> float:
    """Original entropy reduction implementation."""
    # EFFICIENT: Use stored posteriors from Channel 3 if tensor provided
    if tensor_5ch is not None and mapper is not None:
        # Extract entropy from Channel 3 (stored posteriors)
        entropy_before = _compute_entropy_from_channel(tensor_5ch[:, :, 3], target_variable, mapper)
    else:
        # Fallback: compute from buffer (less efficient)
        from ..training.three_channel_converter import buffer_to_three_channel_tensor
        tensor_before, mapper = buffer_to_three_channel_tensor(
            buffer, target_variable, max_history_size=100, standardize=True
        )
        posterior_before = surrogate_predict_fn(tensor_before, target_variable, variables)
        entropy_before = _compute_posterior_entropy(posterior_before, target_variable)
    
    # Create hypothetical buffer WITH intervention for "after" state
    hypo_buffer = _copy_buffer(buffer)
    hypo_buffer.add_intervention(intervention, outcome_sample)
    
    # Get tensor with posteriors for "after" state
    from ..training.five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
    tensor_after, mapper_after, _ = buffer_to_five_channel_tensor_with_posteriors(
        hypo_buffer, target_variable, max_history_size=100, standardize=True
    )
    
    # Extract entropy from Channel 3 of "after" tensor
    entropy_after = _compute_entropy_from_channel(tensor_after[:, :, 3], target_variable, mapper_after)
    
    # Information gain = entropy reduction
    info_gain = entropy_before - entropy_after
    
    logger.debug(f"Info gain (entropy): {entropy_before:.4f} -> {entropy_after:.4f} = {info_gain:.4f}")
    return float(info_gain)


def compute_parent_reward(scm: Any, intervention_variable: str, target_variable: str) -> float:
    """
    Compute binary parent reward.
    
    Args:
        scm: Structural causal model
        intervention_variable: Variable that was intervened on
        target_variable: Target variable
        
    Returns:
        1.0 if intervention_variable is a parent of target_variable, 0.0 otherwise
    """
    try:
        true_parents = set(get_parents(scm, target_variable))
        return 1.0 if intervention_variable in true_parents else 0.0
    except Exception as e:
        logger.error(f"Error computing parent reward: {e}")
        return 0.0


def compute_composite_reward(
    intervention: Dict[str, Any],
    outcome_sample: Any,
    buffer: ExperienceBuffer,
    scm: Any,
    target_variable: str,
    variables: list,
    surrogate_predict_fn: Optional[Callable] = None,
    config: Optional[RewardConfig] = None,
    tensor_5ch: Optional[Any] = None,
    mapper: Optional[Any] = None,
    reward_type: str = "continuous",
    stats: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compute weighted combination of all reward components.
    
    Args:
        intervention: Applied intervention
        outcome_sample: Result sample
        buffer: Experience buffer (before intervention)
        scm: Structural causal model
        target_variable: Target variable name
        variables: List of all variables
        surrogate_predict_fn: Optional surrogate prediction function
        config: Reward configuration
        
    Returns:
        Dictionary with total reward and component breakdown
    """
    if config is None:
        config = RewardConfig()
    
    # Extract intervention variable (assume single variable intervention)
    intervention_variable = list(intervention['targets'])[0]
    
    # Compute individual components
    target_reward = compute_target_reward(
        outcome_sample, target_variable, config.optimization_direction,
        reward_type=reward_type, stats=stats
    )
    
    info_gain = compute_information_gain_reward(
        buffer, intervention, outcome_sample, surrogate_predict_fn, 
        target_variable, variables, tensor_5ch, mapper
    )
    
    parent_reward = compute_parent_reward(scm, intervention_variable, target_variable)
    
    # Weighted combination
    total_reward = (
        config.target_weight * target_reward +
        config.info_gain_weight * info_gain +
        config.parent_weight * parent_reward
    )
    
    # Detailed logging for debugging
    logger.debug(
        f"[REWARD BREAKDOWN] {intervention_variable}:\n"
        f"  Target: {target_reward:.4f} × {config.target_weight:.2f} = {config.target_weight * target_reward:.4f}\n"
        f"  Info gain: {info_gain:.4f} × {config.info_gain_weight:.2f} = {config.info_gain_weight * info_gain:.4f}\n"
        f"  Parent: {parent_reward:.4f} × {config.parent_weight:.2f} = {config.parent_weight * parent_reward:.4f}\n"
        f"  TOTAL: {total_reward:.4f}"
    )
    
    # Return detailed breakdown for debugging/analysis
    return {
        'total': float(total_reward),
        'target': float(target_reward),
        'diversity': 0.0,  # Legacy compatibility
        'exploration': 0.0,  # Legacy compatibility  
        'info_gain': float(info_gain),
        'parent': float(parent_reward),
        'weights': {
            'target': config.target_weight,
            'info_gain': config.info_gain_weight,
            'parent': config.parent_weight
        },
        'components': {
            'target_raw': float(target_reward),
            'info_gain_raw': float(info_gain),
            'parent_raw': float(parent_reward),
            'target_weighted': float(config.target_weight * target_reward),
            'info_gain_weighted': float(config.info_gain_weight * info_gain),
            'parent_weighted': float(config.parent_weight * parent_reward)
        },
        'reward_type': 'composite'
    }


def _compute_entropy_from_channel(parent_prob_channel: Any, target_variable: str, mapper: Any) -> float:
    """
    Compute entropy directly from Channel 3 (stored parent probabilities).
    
    Args:
        parent_prob_channel: Channel 3 from 5-channel tensor [T, n_vars, 1] 
        target_variable: Target variable name
        mapper: Variable mapper for index lookup
        
    Returns:
        Entropy of stored posterior (Shannon entropy)
    """
    try:
        # Get target variable index
        target_idx = mapper.variables.index(target_variable) if hasattr(mapper, 'variables') else 0
        
        # Extract parent probabilities for target variable from most recent timestep
        # Channel 3 stores parent probabilities for each variable
        recent_timestep = -1  # Most recent
        target_probs = parent_prob_channel[recent_timestep, :, :]  # [n_vars] or [n_vars, 1]
        
        # Flatten if needed
        if len(target_probs.shape) > 1:
            target_probs = target_probs.flatten()
        
        # Filter out zero/near-zero probabilities
        valid_probs = target_probs[target_probs > 1e-8]
        
        if len(valid_probs) == 0:
            return 0.0
        
        # Compute Shannon entropy: -sum(p * log(p))
        entropy = -float(jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8)))
        return entropy
        
    except Exception as e:
        logger.error(f"Error computing entropy from channel: {e}")
        return 0.0


def _compute_posterior_entropy(posterior: Dict[str, Any], target_variable: str) -> float:
    """Compute entropy of posterior distribution over parent sets (fallback method)."""
    try:
        if 'marginal_parent_probs' not in posterior:
            return 0.0
            
        probs = posterior['marginal_parent_probs'].get(target_variable, {})
        if not probs:
            return 0.0
        
        # Compute Shannon entropy: -sum(p * log(p))
        prob_values = np.array(list(probs.values()))
        prob_values = prob_values[prob_values > 1e-8]  # Filter near-zero probs
        
        if len(prob_values) == 0:
            return 0.0
            
        entropy = -np.sum(prob_values * np.log(prob_values + 1e-8))
        return float(entropy)
        
    except Exception as e:
        logger.error(f"Error computing posterior entropy: {e}")
        return 0.0


def _copy_buffer(buffer: ExperienceBuffer) -> ExperienceBuffer:
    """Create a copy of experience buffer (temporary implementation)."""
    # TODO: Implement proper copy method in ExperienceBuffer
    new_buffer = ExperienceBuffer()
    
    # Copy observations
    for obs in buffer.get_observations():
        new_buffer.add_observation(obs)
    
    # Copy interventions  
    for intervention, sample in buffer.get_interventions():
        new_buffer.add_intervention(intervention, sample)
    
    return new_buffer


# Legacy compatibility functions
def compute_binary_parent_reward(scm: Any, intervention_variable: str, target_variable: str, **kwargs) -> Dict[str, float]:
    """Legacy compatibility wrapper."""
    parent_reward = compute_parent_reward(scm, intervention_variable, target_variable)
    return {
        'total': parent_reward,
        'target': parent_reward,
        'diversity': 0.0,
        'exploration': 0.0,
        'info_gain': 0.0,
        'weights': {},
        'reward_type': 'binary_parent_legacy'
    }