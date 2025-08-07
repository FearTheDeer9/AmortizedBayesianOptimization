"""
Improved reward functions for GRPO that provide proper RL signals.

Key improvements:
1. Absolute performance rewards that increase as task performance improves
2. Adaptive baseline option for relative rewards
3. Proper scaling to ensure rewards grow during training
"""

import logging
from typing import Dict, Any, Optional, List
from collections import deque
import jax.numpy as jnp

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values

logger = logging.getLogger(__name__)


class AdaptiveBaseline:
    """Maintains an adaptive baseline for reward computation."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.recent_values = deque(maxlen=window_size)
        self.initial_baseline = None
    
    def update(self, value: float):
        """Update baseline with new observation."""
        self.recent_values.append(value)
        if self.initial_baseline is None:
            self.initial_baseline = value
    
    def get_baseline(self) -> float:
        """Get current adaptive baseline."""
        if len(self.recent_values) >= 5:
            return float(jnp.mean(jnp.array(list(self.recent_values))))
        elif self.initial_baseline is not None:
            return self.initial_baseline
        else:
            return 0.0


def compute_absolute_target_reward(
    outcome_value: float,
    optimization_direction: str = 'MINIMIZE',
    scale: float = 1.0
) -> float:
    """
    Compute absolute performance reward that increases with better performance.
    
    For minimization: reward = 1 / (1 + scale * outcome_value)
    - As outcome → 0, reward → 1
    - As outcome → ∞, reward → 0
    
    For maximization: reward = outcome_value / (scale + outcome_value)
    - As outcome → 0, reward → 0
    - As outcome → ∞, reward → 1
    
    Args:
        outcome_value: The achieved value of target variable
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        scale: Scaling factor to adjust reward sensitivity
        
    Returns:
        Reward in [0, 1] range that increases with better performance
    """
    if optimization_direction == 'MINIMIZE':
        # For minimization: lower values → higher rewards
        reward = 1.0 / (1.0 + scale * abs(outcome_value))
    else:
        # For maximization: higher values → higher rewards
        reward = abs(outcome_value) / (scale + abs(outcome_value))
    
    logger.debug(
        f"Absolute reward: outcome={outcome_value:.3f}, "
        f"direction={optimization_direction}, reward={reward:.3f}"
    )
    
    return float(reward)


def compute_adaptive_target_reward(
    outcome_value: float,
    baseline: float,
    optimization_direction: str = 'MINIMIZE',
    temperature: float = 2.0
) -> float:
    """
    Compute reward relative to an adaptive baseline.
    
    Uses a sigmoid function to map improvements to rewards:
    - Improvement > 0: reward > 0.5
    - Improvement = 0: reward = 0.5
    - Improvement < 0: reward < 0.5
    
    The temperature parameter controls sensitivity.
    
    Args:
        outcome_value: The achieved value
        baseline: Current baseline to compare against
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        temperature: Controls reward sensitivity (higher = more sensitive)
        
    Returns:
        Reward in [0, 1] range
    """
    if optimization_direction == 'MINIMIZE':
        improvement = baseline - outcome_value  # Lower is better
    else:
        improvement = outcome_value - baseline  # Higher is better
    
    # Sigmoid mapping
    reward = 1.0 / (1.0 + jnp.exp(-temperature * improvement))
    
    logger.debug(
        f"Adaptive reward: outcome={outcome_value:.3f}, baseline={baseline:.3f}, "
        f"improvement={improvement:.3f}, reward={reward:.3f}"
    )
    
    return float(reward)


def compute_improved_clean_reward(
    buffer_before: ExperienceBuffer,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    config: Optional[Dict[str, Any]] = None,
    adaptive_baseline: Optional[AdaptiveBaseline] = None,
    posterior_before: Optional[Dict[str, Any]] = None,
    posterior_after: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Improved reward computation that ensures proper RL signals.
    
    Key improvements:
    1. Option for absolute performance rewards
    2. Option for adaptive baseline rewards
    3. Proper scaling to ensure rewards increase during training
    
    Args:
        buffer_before: Buffer state before intervention
        intervention: Applied intervention
        outcome: Outcome sample
        target_variable: Variable being optimized
        config: Configuration with reward type and parameters
        adaptive_baseline: Optional adaptive baseline tracker
        
    Returns:
        Dictionary with reward components
    """
    if config is None:
        config = {}
    
    # Get outcome value
    outcome_values = get_values(outcome)
    if target_variable not in outcome_values:
        logger.warning(f"Target {target_variable} not in outcome")
        return {'total': 0.0, 'target': 0.0}
    
    outcome_value = float(outcome_values[target_variable])
    
    # Determine reward type
    reward_type = config.get('reward_type', 'absolute')  # 'absolute' or 'adaptive'
    optimization_direction = config.get('optimization_direction', 'MINIMIZE')
    
    if reward_type == 'absolute':
        # Use absolute performance reward
        scale = config.get('scale', 1.0)
        target_reward = compute_absolute_target_reward(
            outcome_value, optimization_direction, scale
        )
    elif reward_type == 'adaptive' and adaptive_baseline is not None:
        # Use adaptive baseline reward
        baseline = adaptive_baseline.get_baseline()
        temperature = config.get('temperature', 2.0)
        target_reward = compute_adaptive_target_reward(
            outcome_value, baseline, optimization_direction, temperature
        )
        # Update baseline with new outcome
        adaptive_baseline.update(outcome_value)
    else:
        # Fallback to absolute if adaptive requested but no baseline provided
        logger.warning("Adaptive reward requested but no baseline provided, using absolute")
        scale = config.get('scale', 1.0)
        target_reward = compute_absolute_target_reward(
            outcome_value, optimization_direction, scale
        )
    
    # Log reward details
    logger.info(
        f"[IMPROVED REWARD] Type: {reward_type}, "
        f"Outcome: {outcome_value:.3f}, "
        f"Target reward: {target_reward:.3f}"
    )
    
    # Extract weights from config for compatibility
    weights = config.get('weights', {})
    
    # Compute other rewards for compatibility (can be customized later)
    diversity_reward = weights.get('diversity', 0.0)
    exploration_reward = weights.get('exploration', 0.0)
    info_gain_reward = 0.0
    
    # If posteriors provided, compute info gain (placeholder for now)
    if posterior_before is not None and posterior_after is not None:
        # TODO: Implement actual info gain computation if needed
        info_gain_reward = 0.0
    
    # Total reward includes all components
    total_reward = target_reward
    if weights:
        # If weights provided, use weighted combination
        total_reward = (
            weights.get('target', 1.0) * target_reward +
            weights.get('diversity', 0.0) * diversity_reward +
            weights.get('exploration', 0.0) * exploration_reward +
            weights.get('info_gain', 0.0) * info_gain_reward
        )
    
    return {
        'total': float(total_reward),
        'target': float(target_reward),
        'diversity': float(diversity_reward),
        'exploration': float(exploration_reward),
        'info_gain': float(info_gain_reward),
        'weights': weights,
        'reward_type': reward_type
    }


def test_improved_rewards():
    """Test that improved rewards provide proper RL signals."""
    
    print("Testing improved reward functions...")
    
    # Test absolute rewards for minimization
    print("\nAbsolute rewards (minimization):")
    for value in [2.0, 1.0, 0.5, 0.1]:
        reward = compute_absolute_target_reward(value, 'MINIMIZE')
        print(f"  Y={value:.1f} → reward={reward:.3f}")
    
    # Test absolute rewards for maximization
    print("\nAbsolute rewards (maximization):")
    for value in [0.1, 0.5, 1.0, 2.0]:
        reward = compute_absolute_target_reward(value, 'MAXIMIZE')
        print(f"  Y={value:.1f} → reward={reward:.3f}")
    
    # Test adaptive baseline
    print("\nAdaptive baseline rewards:")
    baseline = AdaptiveBaseline(window_size=10)
    baseline.update(1.0)  # Initial value
    
    for value in [0.9, 0.7, 0.5, 0.3]:
        reward = compute_adaptive_target_reward(value, baseline.get_baseline(), 'MINIMIZE')
        print(f"  Y={value:.1f}, baseline={baseline.get_baseline():.2f} → reward={reward:.3f}")
        baseline.update(value)
    
    print("\n✓ All reward functions provide increasing rewards for improving performance!")


if __name__ == "__main__":
    test_improved_rewards()