"""
GRPO-specific reward functions following RLVR best practices.

This module implements reward functions specifically designed for GRPO training
with proper credit assignment, ground truth parent information, and verifiable
reward components.
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
import logging

from ..data_structures.sample import get_values
from ..data_structures.scm import get_parents, get_target

logger = logging.getLogger(__name__)


@dataclass
class GRPORewardComponents:
    """Decomposed reward components for GRPO training."""
    # Core rewards
    target_improvement: float  # Did we improve the target variable?
    parent_intervention: float  # Did we intervene on a true parent?
    value_optimization: float  # How optimal was the intervention value?
    structure_discovery: float  # Information gain about structure
    
    # Binary verifiable signals (RLVR-style)
    improved_beyond_threshold: bool  # Binary: significant improvement?
    correct_parent: bool  # Binary: intervened on true parent?
    
    # Aggregated rewards
    total_reward: float
    variable_selection_reward: float  # Credit for choosing the right variable
    value_selection_reward: float  # Credit for choosing the right value
    
    # Metadata
    metadata: Dict[str, Any]


def compute_grpo_reward(
    scm: Any,
    intervention: Dict[str, Any],
    outcome: Any,
    target_variable: str,
    buffer_before: Any,
    config: Optional[Dict[str, Any]] = None,
    group_outcomes: Optional[List[Tuple[Dict, Any]]] = None
) -> GRPORewardComponents:
    """
    Compute GRPO reward with proper credit assignment and ground truth information.
    
    Args:
        scm: Ground truth SCM for parent information
        intervention: Applied intervention {targets, values}
        outcome: Observed outcome after intervention
        target_variable: Variable being optimized
        buffer_before: Experience buffer before intervention
        config: Reward configuration
        group_outcomes: Other outcomes in the same group (for relative rewards)
        
    Returns:
        GRPORewardComponents with detailed reward breakdown
    """
    if config is None:
        config = {}
    
    # Get configuration
    optimization_direction = config.get('optimization_direction', 'MINIMIZE')
    improvement_threshold = config.get('improvement_threshold', 0.1)
    
    # Extract intervention details
    intervention_targets = intervention.get('targets', set())
    if isinstance(intervention_targets, frozenset):
        intervention_targets = set(intervention_targets)
    
    # Get ground truth parents
    true_parents = set(get_parents(scm, target_variable))
    
    # Get outcome value
    outcome_values = get_values(outcome)
    target_value = float(outcome_values.get(target_variable, 0.0))
    
    # 1. Parent Intervention Reward (using ground truth)
    intervened_vars = intervention_targets
    intervened_parents = intervened_vars & true_parents
    
    if len(intervened_vars) > 0:
        parent_intervention_reward = len(intervened_parents) / len(intervened_vars)
        correct_parent = len(intervened_parents) > 0
    else:
        parent_intervention_reward = 0.0
        correct_parent = False
    
    # 2. Target Improvement Reward
    # Get baseline from buffer
    baseline_value = _compute_baseline_value(
        buffer_before, target_variable, group_outcomes
    )
    
    if optimization_direction == 'MINIMIZE':
        improvement = baseline_value - target_value
    else:
        improvement = target_value - baseline_value
    
    # Continuous improvement reward with better gradient
    if baseline_value != 0:
        relative_improvement = improvement / abs(baseline_value)
    else:
        relative_improvement = improvement
    
    # Use tanh for smoother gradients across the range
    target_improvement_reward = float(0.5 + 0.5 * jnp.tanh(2.0 * relative_improvement))
    improved_beyond_threshold = improvement > improvement_threshold
    
    # 3. Value Optimization Reward
    # How close is the intervention value to optimal for this variable?
    debug_value = relative_improvement > 0.5  # Debug high-reward cases
    value_optimization_reward = _compute_value_optimization_reward(
        scm, intervention, intervened_vars, optimization_direction, debug=debug_value
    )
    
    # 4. Structure Discovery Reward
    # Simplified: reward interventions on high-uncertainty variables
    structure_discovery_reward = _compute_structure_discovery_reward(
        intervened_vars, true_parents, buffer_before
    )
    
    # 5. Credit Assignment
    # Separate rewards for variable vs value selection
    if correct_parent:
        # Good variable choice
        variable_selection_reward = parent_intervention_reward + 0.5 * structure_discovery_reward
    else:
        # Bad variable choice, but might still learn structure
        variable_selection_reward = 0.2 * structure_discovery_reward
    
    if improved_beyond_threshold:
        # Good value choice
        value_selection_reward = target_improvement_reward + value_optimization_reward
    else:
        # Bad value choice
        value_selection_reward = 0.1 * value_optimization_reward
    
    # 6. Total Reward Composition
    # Weight components based on configuration
    weights = config.get('reward_weights', {
        'variable_selection': 0.5,
        'value_selection': 0.5,
        'parent_bonus': 0.3,
        'improvement_bonus': 0.2
    })
    
    total_reward = (
        weights['variable_selection'] * variable_selection_reward +
        weights['value_selection'] * value_selection_reward +
        weights['parent_bonus'] * float(correct_parent) +
        weights['improvement_bonus'] * float(improved_beyond_threshold)
    )
    
    # Create metadata
    metadata = {
        'intervened_vars': list(intervened_vars),
        'true_parents': list(true_parents),
        'intervened_parents': list(intervened_parents),
        'target_value': target_value,
        'baseline_value': baseline_value,
        'improvement': improvement,
        'relative_improvement': relative_improvement,
        'optimization_direction': optimization_direction
    }
    
    return GRPORewardComponents(
        target_improvement=target_improvement_reward,
        parent_intervention=parent_intervention_reward,
        value_optimization=value_optimization_reward,
        structure_discovery=structure_discovery_reward,
        improved_beyond_threshold=improved_beyond_threshold,
        correct_parent=correct_parent,
        total_reward=total_reward,
        variable_selection_reward=variable_selection_reward,
        value_selection_reward=value_selection_reward,
        metadata=metadata
    )


def compute_group_advantages(
    rewards: List[float],
    method: str = 'zscore'
) -> List[float]:
    """
    Compute advantages for a group of rewards following GRPO best practices.
    
    Args:
        rewards: List of rewards for interventions in the same group
        method: 'zscore' for normalized advantages, 'mean' for simple baseline
        
    Returns:
        List of advantages
    """
    rewards_array = jnp.array(rewards)
    
    if method == 'zscore':
        # Z-score normalization (GRPO standard)
        mean_reward = jnp.mean(rewards_array)
        std_reward = jnp.std(rewards_array)
        
        if std_reward > 1e-8:
            advantages = (rewards_array - mean_reward) / std_reward
        else:
            # All rewards are the same
            advantages = jnp.zeros_like(rewards_array)
    
    elif method == 'mean':
        # Simple mean baseline
        mean_reward = jnp.mean(rewards_array)
        advantages = rewards_array - mean_reward
    
    else:
        raise ValueError(f"Unknown advantage method: {method}")
    
    return advantages.tolist()


def _compute_baseline_value(
    buffer: Any,
    target_variable: str,
    group_outcomes: Optional[List[Tuple[Dict, Any]]] = None
) -> float:
    """
    Compute baseline value for reward calculation.
    
    Uses group outcomes if available (GRPO), otherwise uses buffer mean.
    """
    if group_outcomes and len(group_outcomes) > 0:
        # Use mean of group outcomes as baseline (GRPO approach)
        group_values = []
        for _, outcome in group_outcomes:
            values = get_values(outcome)
            if target_variable in values:
                group_values.append(float(values[target_variable]))
        
        if group_values:
            return float(jnp.mean(jnp.array(group_values)))
    
    # Fallback to buffer mean
    if buffer is not None:
        buffer_values = []
        # Get recent samples from buffer
        recent_samples = buffer.get_observations()[-20:]  # Last 20 observations
        
        for sample in recent_samples:
            values = get_values(sample)
            if target_variable in values:
                buffer_values.append(float(values[target_variable]))
        
        if buffer_values:
            return float(jnp.mean(jnp.array(buffer_values)))
    
    # Ultimate fallback
    return 0.0


def _compute_value_optimization_reward(
    scm: Any,
    intervention: Dict[str, Any],
    intervened_vars: Set[str],
    optimization_direction: str,
    debug: bool = False
) -> float:
    """
    Compute how optimal the intervention values were.
    
    This uses knowledge of the SCM structure to evaluate if the
    intervention values push in the right direction.
    """
    if not intervened_vars:
        return 0.0
    
    intervention_values = intervention.get('values', {})
    target_var = get_target(scm)
    
    # Simple heuristic: for parents of target, check if intervention
    # values align with optimization direction and edge weights
    value_rewards = []
    
    for var in intervened_vars:
        if var not in intervention_values:
            continue
        
        value = float(intervention_values[var])
        
        # Check if this variable is a parent of target
        true_parents = get_parents(scm, target_var)
        if var in true_parents:
            # For parents, reward based on value magnitude and direction
            # This is a simplified version - in practice would use edge weights
            if optimization_direction == 'MINIMIZE':
                # For minimization, more negative values are better
                # Stronger linear component for better gradient
                linear_component = -0.4 * value  # Stronger negative slope
                sigmoid_component = 0.5 / (1.0 + jnp.exp(3.0 * value))
                value_reward = jnp.clip(linear_component + sigmoid_component, 0.0, 3.0)
            else:
                # For maximization, more positive values are better
                linear_component = 0.4 * value  # Stronger positive slope
                sigmoid_component = 0.5 / (1.0 + jnp.exp(-3.0 * value))
                value_reward = jnp.clip(linear_component + sigmoid_component, 0.0, 3.0)
            
            if debug:
                logger.debug(f"Parent {var}: value={value:.3f}, reward={value_reward:.3f} (direction={optimization_direction})")
            
            value_rewards.append(value_reward)
        else:
            # For non-parents, small values are better (less disruption)
            value_reward = jnp.exp(-abs(value))
            value_rewards.append(value_reward)
    
    if value_rewards:
        return float(jnp.mean(jnp.array(value_rewards)))
    else:
        return 0.5  # Neutral reward


def _compute_structure_discovery_reward(
    intervened_vars: Set[str],
    true_parents: Set[str],
    buffer: Any
) -> float:
    """
    Reward interventions that help discover causal structure.
    
    Simplified version that rewards:
    1. Intervening on high-uncertainty variables
    2. Intervening on potential parents
    """
    if not intervened_vars:
        return 0.0
    
    # Simple heuristic: reward overlap with true parents
    # In practice, would use uncertainty estimates from posterior
    parent_overlap = len(intervened_vars & true_parents) / len(intervened_vars)
    
    # Bonus for single-variable interventions (more informative)
    single_var_bonus = 0.2 if len(intervened_vars) == 1 else 0.0
    
    return parent_overlap + single_var_bonus


def create_counterfactual_baseline(
    scm: Any,
    current_intervention: Dict[str, Any],
    buffer: Any,
    n_counterfactuals: int = 4
) -> float:
    """
    Create counterfactual baseline by considering alternative interventions.
    
    This helps with credit assignment by asking "what if we had intervened
    on a different variable?"
    
    Args:
        scm: Ground truth SCM
        current_intervention: The actual intervention taken
        buffer: Experience buffer
        n_counterfactuals: Number of counterfactual interventions to consider
        
    Returns:
        Counterfactual baseline value
    """
    # Get all variables except target
    all_vars = list(get_values(buffer.get_observations()[0]).keys())
    target_var = get_target(scm)
    available_vars = [v for v in all_vars if v != target_var]
    
    # Current intervention variable
    current_vars = list(current_intervention.get('targets', []))
    
    # Generate counterfactual interventions
    counterfactual_values = []
    
    for var in available_vars:
        if var not in current_vars:
            # Estimate expected outcome if we had intervened on this variable
            # This is a simplified estimation - in practice would use a model
            if var in get_parents(scm, target_var):
                # Parent variables have higher expected impact
                expected_value = 0.7
            else:
                # Non-parents have lower expected impact
                expected_value = 0.3
            
            counterfactual_values.append(expected_value)
    
    if counterfactual_values:
        # Return mean of counterfactual values
        return float(jnp.mean(jnp.array(counterfactual_values[:n_counterfactuals])))
    else:
        return 0.5  # Neutral baseline


def analyze_reward_distribution(
    reward_components: List[GRPORewardComponents]
) -> Dict[str, Any]:
    """
    Analyze distribution of reward components for debugging.
    
    Args:
        reward_components: List of reward components from training
        
    Returns:
        Dictionary with distribution statistics
    """
    if not reward_components:
        return {}
    
    # Extract component arrays
    target_improvements = [r.target_improvement for r in reward_components]
    parent_interventions = [r.parent_intervention for r in reward_components]
    value_optimizations = [r.value_optimization for r in reward_components]
    structure_discoveries = [r.structure_discovery for r in reward_components]
    
    # Binary signals
    improvement_rate = sum(r.improved_beyond_threshold for r in reward_components) / len(reward_components)
    parent_rate = sum(r.correct_parent for r in reward_components) / len(reward_components)
    
    return {
        'target_improvement': {
            'mean': float(jnp.mean(jnp.array(target_improvements))),
            'std': float(jnp.std(jnp.array(target_improvements))),
            'min': float(jnp.min(jnp.array(target_improvements))),
            'max': float(jnp.max(jnp.array(target_improvements)))
        },
        'parent_intervention': {
            'mean': float(jnp.mean(jnp.array(parent_interventions))),
            'rate': parent_rate
        },
        'value_optimization': {
            'mean': float(jnp.mean(jnp.array(value_optimizations))),
            'std': float(jnp.std(jnp.array(value_optimizations)))
        },
        'structure_discovery': {
            'mean': float(jnp.mean(jnp.array(structure_discoveries))),
            'std': float(jnp.std(jnp.array(structure_discoveries)))
        },
        'binary_signals': {
            'improvement_rate': improvement_rate,
            'parent_selection_rate': parent_rate
        },
        'n_samples': len(reward_components)
    }