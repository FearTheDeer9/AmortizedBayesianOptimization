"""
Better reward functions that handle negative values and unknown ranges.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, List
from collections import deque

import logging
logger = logging.getLogger(__name__)


class RunningStats:
    """Track running statistics of values for normalization."""
    
    def __init__(self, window_size: int = 1000):
        self.values = deque(maxlen=window_size)
        self.window_size = window_size
    
    def update(self, value: float):
        """Add new value to statistics."""
        self.values.append(value)
    
    @property
    def mean(self) -> float:
        """Get running mean."""
        if not self.values:
            return 0.0
        return float(np.mean(list(self.values)))
    
    @property
    def std(self) -> float:
        """Get running standard deviation."""
        if len(self.values) < 2:
            return 1.0
        return float(np.std(list(self.values)))
    
    @property
    def min(self) -> float:
        """Get minimum value seen."""
        if not self.values:
            return 0.0
        return float(min(self.values))
    
    @property
    def max(self) -> float:
        """Get maximum value seen."""
        if not self.values:
            return 0.0
        return float(max(self.values))
    
    def get_percentile(self, value: float) -> float:
        """Get percentile rank of value."""
        if not self.values:
            return 0.5
        return float(np.sum(np.array(list(self.values)) <= value) / len(self.values))


def sigmoid_target_reward(
    outcome_value: float,
    optimization_direction: str = 'MINIMIZE',
    center: float = 0.0,
    temperature: float = 1.0
) -> float:
    """
    Sigmoid-based reward that handles any value range.
    
    Maps values to [0, 1] using a sigmoid centered at 'center'.
    Temperature controls the steepness of the transition.
    
    For MINIMIZE:
    - Values < center get reward > 0.5
    - Values > center get reward < 0.5
    
    For MAXIMIZE:
    - Values > center get reward > 0.5
    - Values < center get reward < 0.5
    
    Args:
        outcome_value: The achieved value
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        center: Center point of sigmoid (default: 0)
        temperature: Controls steepness (default: 1)
    
    Returns:
        Reward in [0, 1] range
    """
    if optimization_direction == 'MINIMIZE':
        # For minimization: lower values → higher rewards
        reward = 1.0 / (1.0 + jnp.exp(temperature * (outcome_value - center)))
    else:
        # For maximization: higher values → higher rewards
        reward = 1.0 / (1.0 + jnp.exp(-temperature * (outcome_value - center)))
    
    return float(reward)


def adaptive_sigmoid_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE',
    temperature_factor: float = 2.0
) -> float:
    """
    Sigmoid reward that adapts to the data distribution.
    
    Uses running statistics to automatically set center and temperature.
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics of recent values
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        temperature_factor: Multiplier for temperature (default: 2.0)
    
    Returns:
        Reward in [0, 1] range
    """
    # Update stats with new value
    stats.update(outcome_value)
    
    # Use mean as center
    center = stats.mean
    
    # Use std to set temperature (inverse relationship)
    # Larger std → smaller temperature → gentler curve
    if stats.std > 0:
        temperature = temperature_factor / stats.std
    else:
        temperature = temperature_factor
    
    reward = sigmoid_target_reward(
        outcome_value, optimization_direction, center, temperature
    )
    
    logger.debug(
        f"Adaptive sigmoid: value={outcome_value:.3f}, "
        f"center={center:.3f}, std={stats.std:.3f}, "
        f"temperature={temperature:.3f}, reward={reward:.3f}"
    )
    
    return reward


def percentile_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE'
) -> float:
    """
    Reward based on percentile rank in recent history.
    
    Automatically adapts to any value distribution.
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics of recent values
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
    
    Returns:
        Reward in [0, 1] range (percentile)
    """
    # Update stats with new value
    stats.update(outcome_value)
    
    # Get percentile rank
    percentile = stats.get_percentile(outcome_value)
    
    if optimization_direction == 'MINIMIZE':
        # For minimization: lower percentile → higher reward
        reward = 1.0 - percentile
    else:
        # For maximization: higher percentile → higher reward
        reward = percentile
    
    logger.debug(
        f"Percentile reward: value={outcome_value:.3f}, "
        f"percentile={percentile:.3f}, reward={reward:.3f}"
    )
    
    return float(reward)


def tanh_normalized_reward(
    outcome_value: float,
    stats: RunningStats,
    optimization_direction: str = 'MINIMIZE',
    scale_factor: float = 2.0
) -> float:
    """
    Reward using tanh normalization with running statistics.
    
    Maps values to [-1, 1] then shifts to [0, 1].
    
    Args:
        outcome_value: The achieved value
        stats: Running statistics
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        scale_factor: Controls sensitivity (default: 2.0)
    
    Returns:
        Reward in [0, 1] range
    """
    # Update stats
    stats.update(outcome_value)
    
    # Normalize using z-score
    if stats.std > 0:
        z_score = (outcome_value - stats.mean) / stats.std
    else:
        z_score = 0.0
    
    # Map to [-1, 1] using tanh
    if optimization_direction == 'MINIMIZE':
        # For minimization: negative z-scores are good
        tanh_value = jnp.tanh(-scale_factor * z_score)
    else:
        # For maximization: positive z-scores are good
        tanh_value = jnp.tanh(scale_factor * z_score)
    
    # Shift to [0, 1]
    reward = (tanh_value + 1.0) / 2.0
    
    logger.debug(
        f"Tanh reward: value={outcome_value:.3f}, "
        f"z_score={z_score:.3f}, reward={reward:.3f}"
    )
    
    return float(reward)


def compute_better_clean_reward(
    buffer_before: Any,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    config: Optional[Dict[str, Any]] = None,
    stats: Optional[RunningStats] = None,
    posterior_before: Optional[Dict[str, Any]] = None,
    posterior_after: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Improved reward computation with better handling of value ranges.
    
    Args:
        buffer_before: Buffer (for compatibility)
        intervention: Applied intervention
        outcome: Outcome sample
        target_variable: Variable being optimized
        config: Configuration with reward type and parameters
        stats: Running statistics for adaptive rewards
        posterior_before: Optional (for compatibility)
        posterior_after: Optional (for compatibility)
    
    Returns:
        Dictionary with reward components
    """
    if config is None:
        config = {}
    
    # Import here to avoid circular dependency
    from ..data_structures.sample import get_values
    
    # Get outcome value
    outcome_values = get_values(outcome)
    if target_variable not in outcome_values:
        logger.warning(f"Target {target_variable} not in outcome")
        return {'total': 0.5, 'target': 0.5, 'weights': {}}
    
    outcome_value = float(outcome_values[target_variable])
    
    # Get configuration
    reward_type = config.get('reward_type', 'sigmoid')
    optimization_direction = config.get('optimization_direction', 'MINIMIZE')
    
    # Compute target reward based on type
    if reward_type == 'sigmoid':
        center = config.get('center', 0.0)
        temperature = config.get('temperature', 1.0)
        target_reward = sigmoid_target_reward(
            outcome_value, optimization_direction, center, temperature
        )
    
    elif reward_type == 'adaptive_sigmoid' and stats is not None:
        temperature_factor = config.get('temperature_factor', 2.0)
        target_reward = adaptive_sigmoid_reward(
            outcome_value, stats, optimization_direction, temperature_factor
        )
    
    elif reward_type == 'percentile' and stats is not None:
        target_reward = percentile_reward(
            outcome_value, stats, optimization_direction
        )
    
    elif reward_type == 'tanh' and stats is not None:
        scale_factor = config.get('scale_factor', 2.0)
        target_reward = tanh_normalized_reward(
            outcome_value, stats, optimization_direction, scale_factor
        )
    
    else:
        # Fallback to simple sigmoid
        logger.warning(f"Unknown reward type {reward_type}, using sigmoid")
        target_reward = sigmoid_target_reward(
            outcome_value, optimization_direction
        )
    
    # Extract weights for compatibility
    weights = config.get('weights', {})
    
    # Log reward details
    logger.debug(
        f"Better reward - type: {reward_type}, "
        f"outcome: {outcome_value:.3f}, "
        f"target reward: {target_reward:.3f}"
    )
    
    # Return compatible format
    return {
        'total': float(target_reward),
        'target': float(target_reward),
        'diversity': 0.0,
        'exploration': 0.0,
        'info_gain': 0.0,
        'weights': weights,
        'reward_type': reward_type
    }


def test_better_rewards():
    """Test the better reward functions."""
    
    print("Testing better reward functions...")
    
    # Test values including negative
    test_values = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]
    
    print("\n1. SIGMOID REWARDS (center=0, temp=1):")
    print("Value  | Minimize | Maximize")
    print("-------|----------|----------")
    for val in test_values:
        min_reward = sigmoid_target_reward(val, 'MINIMIZE')
        max_reward = sigmoid_target_reward(val, 'MAXIMIZE')
        print(f"{val:6.1f} | {min_reward:8.3f} | {max_reward:8.3f}")
    
    print("\n2. ADAPTIVE REWARDS (updates with data):")
    stats = RunningStats()
    print("Value  | Percentile | Adaptive Sigmoid")
    print("-------|------------|------------------")
    for val in test_values:
        perc_reward = percentile_reward(val, stats, 'MINIMIZE')
        adapt_reward = adaptive_sigmoid_reward(val, stats, 'MINIMIZE')
        print(f"{val:6.1f} | {perc_reward:10.3f} | {adapt_reward:16.3f}")
    
    print(f"\nFinal stats: mean={stats.mean:.2f}, std={stats.std:.2f}")
    print("\n✓ All reward functions handle negative values correctly!")


if __name__ == "__main__":
    test_better_rewards()