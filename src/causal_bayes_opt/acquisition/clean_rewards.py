"""
Clean reward computation for ACBO without AcquisitionState.

This module provides verifiable reward functions that work directly with
experience buffers, avoiding the complexity of AcquisitionState objects.

Key principles:
- All rewards are verifiable (no human feedback)
- Work directly with buffers and samples
- Support both optimization and exploration objectives
- Simple, clear interfaces
"""

import logging
from typing import Dict, Any, Optional, List
from collections import defaultdict

import jax.numpy as jnp

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import get_values, get_intervention_targets

logger = logging.getLogger(__name__)


def compute_clean_reward(
    buffer_before: ExperienceBuffer,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    config: Optional[Dict[str, Any]] = None,
    posterior_before: Optional[Dict[str, Any]] = None,
    posterior_after: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Compute verifiable reward directly from buffer without AcquisitionState.
    
    This function computes multiple reward components:
    1. Target optimization reward (primary objective)
    2. Intervention diversity reward (explore different variables)
    3. Value exploration reward (explore different intervention values)
    4. Information gain reward (reduce structure uncertainty) - when surrogate available
    
    Args:
        buffer_before: Buffer state before intervention
        intervention: Applied intervention {targets, values}
        outcome: Outcome sample from intervention
        target_variable: Variable being optimized
        config: Optional configuration with weights and parameters
        posterior_before: Optional posterior before intervention (for info gain)
        posterior_after: Optional posterior after intervention (for info gain)
        
    Returns:
        Dictionary with reward components and total
    """
    if config is None:
        config = {}
    
    # Default weights for different objectives
    default_weights = {
        'target': 1.0,      # Primary objective: optimize target
        'diversity': 0.2,   # Secondary: try different variables
        'exploration': 0.1, # Tertiary: try different values
        'info_gain': 0.0    # Information gain (activated when surrogate available)
    }
    
    # Use provided weights or defaults
    weights = config.get('weights', {})
    for key, default_val in default_weights.items():
        if key not in weights:
            weights[key] = default_val
    
    # 1. Target improvement reward
    target_reward = compute_target_reward(
        buffer_before, target_variable, outcome,
        optimization_direction=config.get('optimization_direction', 'MINIMIZE')
    )
    
    # 2. Diversity bonus for trying less-intervened variables
    diversity_reward = compute_diversity_reward(buffer_before, intervention)
    
    # 3. Exploration bonus for trying new intervention values
    exploration_reward = compute_exploration_reward(buffer_before, intervention)
    
    # 4. Information gain reward (if posteriors provided)
    info_gain_reward = 0.0
    if posterior_before is not None and posterior_after is not None:
        info_gain_reward = compute_information_gain_reward(
            posterior_before, posterior_after
        )
        logger.info(
            f"[REWARD] Computing info gain reward WITH surrogate: "
            f"entropy_before={posterior_before.get('entropy', 0):.3f}, "
            f"entropy_after={posterior_after.get('entropy', 0):.3f}, "
            f"info_gain_reward={info_gain_reward:.3f}"
        )
    else:
        logger.info(
            f"[REWARD] No surrogate available - info gain reward = 0.0 "
            f"(posterior_before={posterior_before is not None}, "
            f"posterior_after={posterior_after is not None})"
        )
    
    # Weighted combination
    total_reward = (
        weights['target'] * target_reward +
        weights['diversity'] * diversity_reward +
        weights['exploration'] * exploration_reward +
        weights['info_gain'] * info_gain_reward
    )
    
    # Log detailed reward breakdown
    logger.info(
        f"[REWARD] Reward breakdown:\n"
        f"  - Target reward: {target_reward:.3f} (weight={weights['target']:.2f})\n"
        f"  - Diversity reward: {diversity_reward:.3f} (weight={weights['diversity']:.2f})\n" 
        f"  - Exploration reward: {exploration_reward:.3f} (weight={weights['exploration']:.2f})\n"
        f"  - Info gain reward: {info_gain_reward:.3f} (weight={weights['info_gain']:.2f})\n"
        f"  - Total reward: {total_reward:.3f}"
    )
    
    return {
        'total': float(total_reward),
        'target': float(target_reward),
        'diversity': float(diversity_reward),
        'exploration': float(exploration_reward),
        'info_gain': float(info_gain_reward),
        'weights': weights
    }


def compute_target_reward(
    buffer: ExperienceBuffer,
    target_variable: str,
    intervention_outcome: Any,
    optimization_direction: str = 'MINIMIZE'
) -> float:
    """
    Compute reward based on target variable improvement.
    
    Uses observational data as baseline and rewards improvement
    from that baseline.
    
    Args:
        buffer: Experience buffer with historical data
        target_variable: Variable to optimize
        intervention_outcome: Outcome from intervention
        optimization_direction: 'MINIMIZE' or 'MAXIMIZE'
        
    Returns:
        Reward value (higher is better)
    """
    # Get baseline from observational samples
    obs_samples = buffer.get_observations()
    if not obs_samples:
        # No baseline available, use neutral reward
        return 0.0
    
    # Compute baseline as mean of observational values
    obs_values = []
    for sample in obs_samples[-100:]:  # Use recent samples
        values = get_values(sample)
        if target_variable in values:
            obs_values.append(float(values[target_variable]))
    
    if not obs_values:
        logger.warning(f"No observational values for target {target_variable}")
        return 0.0
    
    baseline = float(jnp.mean(jnp.array(obs_values)))
    
    # Get intervention outcome value
    outcome_values = get_values(intervention_outcome)
    if target_variable not in outcome_values:
        logger.warning(f"Target {target_variable} not in outcome")
        return 0.0
    
    outcome_value = float(outcome_values[target_variable])
    
    # Compute improvement from baseline
    if optimization_direction == 'MINIMIZE':
        improvement = baseline - outcome_value  # Lower is better
    else:
        improvement = outcome_value - baseline  # Higher is better
    
    # Normalize by standard deviation for scale-invariant reward
    if len(obs_values) > 1:
        std_dev = float(jnp.std(jnp.array(obs_values)))
        if std_dev > 0:
            normalized_improvement = improvement / std_dev
        else:
            normalized_improvement = improvement
    else:
        normalized_improvement = improvement
    
    # Convert to [0, 1] range using sigmoid
    # Centered at 0 improvement, positive improvement gives reward > 0.5
    reward = float(1.0 / (1.0 + jnp.exp(-2.0 * normalized_improvement)))
    
    logger.debug(
        f"Target reward: baseline={baseline:.3f}, outcome={outcome_value:.3f}, "
        f"improvement={improvement:.3f}, reward={reward:.3f}"
    )
    
    return reward


def compute_diversity_reward(
    buffer: ExperienceBuffer,
    intervention: Dict[str, Any]
) -> float:
    """
    Reward intervening on less-explored variables.
    
    This encourages the policy to try different variables rather
    than repeatedly intervening on the same one.
    
    Args:
        buffer: Experience buffer with intervention history
        intervention: New intervention being evaluated
        
    Returns:
        Diversity reward in [0, 1] range
    """
    # Count previous interventions per variable
    intervention_counts = defaultdict(int)
    total_interventions = 0
    
    for _, sample in buffer.get_interventions():
        targets = get_intervention_targets(sample)
        for var in targets:
            intervention_counts[var] += 1
            total_interventions += 1
    
    if total_interventions == 0:
        # First intervention, maximum diversity reward
        return 1.0
    
    # Get the variable being intervened on
    new_targets = intervention.get('targets', set())
    if not new_targets:
        return 0.0
    
    # For simplicity, assume single-variable interventions
    new_var = list(new_targets)[0] if isinstance(new_targets, (set, frozenset)) else new_targets[0]
    
    # Compute diversity score
    # Variables never intervened on get score 1.0
    # Most intervened variable gets score 0.0
    var_count = intervention_counts.get(new_var, 0)
    max_count = max(intervention_counts.values()) if intervention_counts else 1
    
    diversity_score = 1.0 - (var_count / (max_count + 1))
    
    logger.debug(
        f"Diversity reward: var={new_var}, count={var_count}, "
        f"max_count={max_count}, reward={diversity_score:.3f}"
    )
    
    return diversity_score


def compute_exploration_reward(
    buffer: ExperienceBuffer,
    intervention: Dict[str, Any]
) -> float:
    """
    Reward trying new intervention values for variables.
    
    This encourages exploring different parts of the intervention
    space for each variable.
    
    Args:
        buffer: Experience buffer with intervention history
        intervention: New intervention being evaluated
        
    Returns:
        Exploration reward in [0, 1] range
    """
    # Get intervention details
    targets = intervention.get('targets', set())
    values = intervention.get('values', {})
    
    if not targets or not values:
        return 0.0
    
    # For simplicity, handle single-variable interventions
    var = list(targets)[0] if isinstance(targets, (set, frozenset)) else targets[0]
    if var not in values:
        return 0.0
    
    new_value = float(values[var])
    
    # Collect previous intervention values for this variable
    previous_values = []
    for _, sample in buffer.get_interventions():
        sample_targets = get_intervention_targets(sample)
        if var in sample_targets:
            sample_values = get_values(sample)
            if var in sample_values:
                previous_values.append(float(sample_values[var]))
    
    if not previous_values:
        # First intervention on this variable, max exploration reward
        return 1.0
    
    # Compute minimum distance to previous values
    distances = [abs(new_value - v) for v in previous_values]
    min_distance = min(distances)
    
    # Normalize by value range
    value_range = max(previous_values) - min(previous_values)
    if value_range < 1e-8:
        # All previous values are the same
        # Reward any deviation
        return 1.0 if min_distance > 0 else 0.0
    
    # Normalize distance to [0, 1] range
    normalized_distance = min(min_distance / value_range, 1.0)
    
    logger.debug(
        f"Exploration reward: var={var}, value={new_value:.3f}, "
        f"min_distance={min_distance:.3f}, reward={normalized_distance:.3f}"
    )
    
    return normalized_distance


def compute_information_gain_reward(
    posterior_before: Dict[str, Any],
    posterior_after: Dict[str, Any]
) -> float:
    """
    Compute information gain as reduction in entropy.
    
    Information gain measures how much the intervention reduced our
    uncertainty about the causal structure.
    
    The reward uses a sigmoid normalization centered at 0.5:
    - reward = 1 / (1 + exp(-4 * info_gain))
    - When info_gain = 0: reward = 0.5 (neutral)
    - When info_gain > 0: reward > 0.5 (positive reward)
    - When info_gain < 0: reward < 0.5 (negative reward)
    
    This design choice allows for:
    1. Smooth gradients for learning
    2. Both positive rewards (information gained) and negative (information lost)
    3. A neutral point at 0.5 when no information is gained
    
    Example values:
    - info_gain = 0.1 nats → reward ≈ 0.60
    - info_gain = 0.5 nats → reward ≈ 0.88
    - info_gain = 1.0 nats → reward ≈ 0.98
    
    Args:
        posterior_before: Posterior before intervention (with 'entropy' key)
        posterior_after: Posterior after intervention (with 'entropy' key)
        
    Returns:
        Information gain reward in [0, 1] range, centered at 0.5
    """
    # Extract entropy values
    entropy_before = posterior_before.get('entropy', 0.0)
    entropy_after = posterior_after.get('entropy', 0.0)
    
    # Compute raw information gain (can be negative if entropy increases)
    info_gain = entropy_before - entropy_after
    
    # Normalize to [0, 1] range
    # Use sigmoid-like function centered at meaningful gain
    # Gain of 0.1 nats ~ 0.5 reward, gain of 0.5 nats ~ 0.88 reward
    normalized_gain = float(1.0 / (1.0 + jnp.exp(-4.0 * info_gain)))
    
    logger.debug(
        f"Information gain: entropy_before={entropy_before:.3f}, "
        f"entropy_after={entropy_after:.3f}, gain={info_gain:.3f}, "
        f"reward={normalized_gain:.3f}"
    )
    
    return normalized_gain


def compute_structure_aware_reward(
    buffer_before: ExperienceBuffer,
    buffer_after: ExperienceBuffer,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    surrogate_model: Optional[Any] = None,
    surrogate_params: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Extended reward function that includes structure learning rewards.
    
    This is used when a surrogate model is available to compute
    information gain and parent intervention rewards.
    
    Args:
        buffer_before: Buffer before intervention
        buffer_after: Buffer after intervention
        intervention: Applied intervention
        outcome: Intervention outcome
        target_variable: Target being optimized
        surrogate_model: Optional structure learning model
        surrogate_params: Model parameters
        config: Configuration
        
    Returns:
        Extended reward dictionary including structure rewards
    """
    # Start with basic rewards
    rewards = compute_clean_reward(
        buffer_before, intervention, outcome, target_variable, config
    )
    
    if surrogate_model is None:
        # No structure learning rewards available
        rewards['info_gain'] = 0.0
        rewards['parent_bonus'] = 0.0
        return rewards
    
    # TODO: Implement structure learning rewards when surrogate is integrated
    # For now, placeholder values
    rewards['info_gain'] = 0.0
    rewards['parent_bonus'] = 0.0
    
    # Update total with structure rewards
    weights = config.get('weights', {}) if config else {}
    structure_weight = weights.get('structure', 0.5)
    parent_weight = weights.get('parent', 0.3)
    
    rewards['total'] += (
        structure_weight * rewards['info_gain'] +
        parent_weight * rewards['parent_bonus']
    )
    
    return rewards