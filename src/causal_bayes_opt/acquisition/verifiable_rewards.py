#!/usr/bin/env python3
"""
Simple Ground-Truth Verifiable Rewards for ACBO

Implements binary reward functions that are hard to game and based on ground truth.
Following successful GRPO patterns from DeepSeek, math reasoning, and code generation.

Key Principles:
1. Binary rewards (1 or 0) - hard to game
2. Ground truth based - uses true SCM structure 
3. Decoupled from surrogate quality - acquisition judged on intervention choice only
4. Simple and verifiable - no complex multi-component balancing

Reward Components:
- Target Improvement: Binary reward for meaningful target variable improvement
- True Parent Intervention: Binary reward for intervening on actual parents 
- Exploration Diversity: Binary reward for exploring under-sampled variables
"""

import logging
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass

import jax.numpy as jnp
import pyrsistent as pyr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimpleRewardComponents:
    """
    Simple decomposed reward components for analysis.
    
    All components are binary (0 or 1) to prevent gaming and ensure
    clear, objective evaluation of intervention choices.
    """
    target_improvement: float      # 1 if target improved meaningfully, 0 otherwise
    true_parent_intervention: float # 1 if intervened on true parent, 0 otherwise  
    exploration_diversity: float   # 1 if exploring under-sampled variables, 0 otherwise
    total_reward: float           # Weighted sum of components
    
    # Metadata for analysis
    intervention_targets: Set[str]
    target_variable: str
    outcome_value: float
    current_best: float
    true_parents: Set[str]
    
    def summary(self) -> Dict[str, Any]:
        """Create human-readable summary."""
        return {
            'total_reward': self.total_reward,
            'target_improvement': self.target_improvement,
            'true_parent_intervention': self.true_parent_intervention, 
            'exploration_diversity': self.exploration_diversity,
            'intervention_targets': list(self.intervention_targets),
            'target_variable': self.target_variable,
            'improvement_amount': self.outcome_value - self.current_best,
            'intervened_on_true_parent': len(self.intervention_targets & self.true_parents) > 0
        }


def target_improvement_reward(
    outcome_value: float,
    current_best: float,
    improvement_threshold: float = 0.1
) -> float:
    """
    Binary reward for meaningful target improvement.
    
    Args:
        outcome_value: Target variable value from intervention outcome
        current_best: Current best target value achieved
        improvement_threshold: Minimum improvement required for reward
        
    Returns:
        1.0 if improvement > threshold, 0.0 otherwise
        
    Example:
        >>> target_improvement_reward(5.5, 5.0, 0.1)
        1.0  # Improvement of 0.5 > 0.1 threshold
        >>> target_improvement_reward(5.05, 5.0, 0.1) 
        0.0  # Improvement of 0.05 < 0.1 threshold
    """
    if not jnp.isfinite(outcome_value) or not jnp.isfinite(current_best):
        logger.warning(f"Non-finite values: outcome={outcome_value}, best={current_best}")
        return 0.0
    
    improvement = outcome_value - current_best
    reward = 1.0 if improvement > improvement_threshold else 0.0
    
    logger.debug(
        f"Target improvement: value={outcome_value:.3f}, best={current_best:.3f}, "
        f"improvement={improvement:.3f}, threshold={improvement_threshold:.3f}, reward={reward}"
    )
    
    return reward


def true_parent_intervention_reward(
    intervention_targets: Set[str],
    true_parents: Set[str],
    target_variable: str
) -> float:
    """
    Binary reward for intervening on actual parents of target variable.
    
    This uses ground truth SCM structure to guide learning toward
    informative interventions without relying on model predictions.
    
    Args:
        intervention_targets: Variables that were intervened upon
        true_parents: True parent variables of target (from ground truth SCM)
        target_variable: Name of target variable
        
    Returns:
        1.0 if any intervention target is a true parent, 0.0 otherwise
        
    Example:
        >>> true_parent_intervention_reward({'X', 'Z'}, {'X', 'Y'}, 'T')
        1.0  # X is both in intervention targets and true parents
        >>> true_parent_intervention_reward({'Z'}, {'X', 'Y'}, 'T')
        0.0  # Z is not a true parent
    """
    # Can't intervene on target itself
    if target_variable in intervention_targets:
        logger.debug(f"Cannot intervene on target variable {target_variable}")
        return 0.0
    
    # Check if any intervention target is a true parent
    intersection = intervention_targets & true_parents
    reward = 1.0 if len(intersection) > 0 else 0.0
    
    logger.debug(
        f"True parent intervention: targets={intervention_targets}, "
        f"true_parents={true_parents}, intersection={intersection}, reward={reward}"
    )
    
    return reward


def exploration_diversity_reward(
    intervention_targets: Set[str],
    previous_interventions: List[Set[str]],
    diversity_threshold: int = 3
) -> float:
    """
    Binary reward for exploring under-sampled variables.
    
    Encourages diversity in intervention strategies to prevent mode collapse
    and ensure adequate exploration of the intervention space.
    
    Args:
        intervention_targets: Variables intervened upon in current step
        previous_interventions: List of intervention target sets from previous steps
        diversity_threshold: Maximum frequency before reward becomes 0
        
    Returns:
        1.0 if intervention frequency < threshold, 0.0 otherwise
        
    Example:
        >>> prev = [{'X'}, {'Y'}, {'X'}]  # X intervened 2 times, Y once
        >>> exploration_diversity_reward({'Z'}, prev, 3)
        1.0  # Z never intervened (frequency 0 < 3)
        >>> exploration_diversity_reward({'X'}, prev, 3) 
        1.0  # X intervened 2 times (frequency 2 < 3)
        >>> exploration_diversity_reward({'X'}, prev + [{'X'}], 3)
        0.0  # X would be intervened 3 times (frequency 3 >= 3)
    """
    if not intervention_targets:
        return 0.0
    
    # Count frequency of these exact intervention targets
    frequency = sum(1 for prev_targets in previous_interventions 
                   if len(intervention_targets & prev_targets) > 0)
    
    reward = 1.0 if frequency < diversity_threshold else 0.0
    
    logger.debug(
        f"Exploration diversity: targets={intervention_targets}, "
        f"frequency={frequency}, threshold={diversity_threshold}, reward={reward}"
    )
    
    return reward


def compute_simple_verifiable_reward(
    intervention_targets: Set[str],
    outcome_value: float,
    current_best: float,
    true_parents: Set[str],
    target_variable: str,
    previous_interventions: List[Set[str]],
    weights: Optional[Dict[str, float]] = None,
    improvement_threshold: float = 0.1,
    diversity_threshold: int = 3
) -> SimpleRewardComponents:
    """
    Compute simple verifiable reward using ground truth.
    
    This is the main reward function that combines binary components
    in a weighted sum. All components are binary to prevent gaming.
    
    Args:
        intervention_targets: Variables that were intervened upon
        outcome_value: Target variable value from intervention outcome
        current_best: Current best target value achieved
        true_parents: True parent variables of target (from ground truth)
        target_variable: Name of target variable being optimized
        previous_interventions: History of previous intervention targets
        weights: Optional weight override for reward components
        improvement_threshold: Minimum improvement for target reward
        diversity_threshold: Maximum frequency for diversity reward
        
    Returns:
        SimpleRewardComponents with detailed decomposition
        
    Example:
        >>> reward = compute_simple_verifiable_reward(
        ...     intervention_targets={'X'},
        ...     outcome_value=5.5,
        ...     current_best=5.0,
        ...     true_parents={'X', 'Y'},
        ...     target_variable='Z',
        ...     previous_interventions=[{'Y'}]
        ... )
        >>> reward.total_reward  # Should be positive (target improved + true parent)
        3.5
    """
    # Default weights emphasizing target improvement
    if weights is None:
        weights = {
            'target_improvement': 2.0,     # Primary objective (optimization)
            'true_parent': 1.0,           # Structure guidance from ground truth
            'exploration': 0.5            # Diversity maintenance
        }
    
    # Validate and convert inputs to sets if needed
    intervention_targets = set(intervention_targets) if not isinstance(intervention_targets, set) else intervention_targets
    true_parents = set(true_parents) if not isinstance(true_parents, set) else true_parents
    
    # Compute binary reward components
    target_reward = target_improvement_reward(
        outcome_value, current_best, improvement_threshold
    )
    
    parent_reward = true_parent_intervention_reward(
        intervention_targets, true_parents, target_variable
    )
    
    diversity_reward = exploration_diversity_reward(
        intervention_targets, previous_interventions, diversity_threshold
    )
    
    # Weighted combination
    total = (
        weights['target_improvement'] * target_reward +
        weights['true_parent'] * parent_reward +
        weights['exploration'] * diversity_reward
    )
    
    return SimpleRewardComponents(
        target_improvement=target_reward,
        true_parent_intervention=parent_reward,
        exploration_diversity=diversity_reward,
        total_reward=total,
        intervention_targets=intervention_targets,
        target_variable=target_variable,
        outcome_value=outcome_value,
        current_best=current_best,
        true_parents=true_parents
    )


def validate_reward_consistency(
    reward_history: List[SimpleRewardComponents],
    window_size: int = 50
) -> Dict[str, Any]:
    """
    Validate reward consistency and detect potential gaming.
    
    Args:
        reward_history: Recent reward components from training
        window_size: Number of recent rewards to analyze
        
    Returns:
        Validation metrics and gaming detection results
    """
    if not reward_history:
        return {'valid': True, 'warning': 'No reward history'}
    
    recent_rewards = reward_history[-window_size:]
    
    # Component statistics
    target_rewards = [r.target_improvement for r in recent_rewards]
    parent_rewards = [r.true_parent_intervention for r in recent_rewards]
    diversity_rewards = [r.exploration_diversity for r in recent_rewards]
    total_rewards = [r.total_reward for r in recent_rewards]
    
    # Check for gaming patterns
    gaming_issues = []
    
    # 1. Check if target improvement is consistently 0 (no optimization progress)
    target_rate = sum(target_rewards) / len(target_rewards)
    if target_rate < 0.1:
        gaming_issues.append(f"Very low target improvement rate: {target_rate:.2f}")
    
    # 2. Check if parent intervention rate is suspiciously high (gaming structure guidance)
    parent_rate = sum(parent_rewards) / len(parent_rewards)
    if parent_rate > 0.9:
        gaming_issues.append(f"Suspiciously high parent intervention rate: {parent_rate:.2f}")
    
    # 3. Check if diversity is consistently 0 (mode collapse)
    diversity_rate = sum(diversity_rewards) / len(diversity_rewards)
    if diversity_rate < 0.1:
        gaming_issues.append(f"Very low exploration diversity: {diversity_rate:.2f}")
    
    # 4. Check reward variance (too little variance suggests gaming)
    total_variance = jnp.var(jnp.array(total_rewards))
    if total_variance < 0.01:
        gaming_issues.append(f"Very low reward variance: {total_variance:.4f}")
    
    return {
        'valid': len(gaming_issues) == 0,
        'gaming_issues': gaming_issues,
        'component_rates': {
            'target_improvement': target_rate,
            'true_parent_intervention': parent_rate,
            'exploration_diversity': diversity_rate
        },
        'statistics': {
            'mean_total_reward': float(jnp.mean(jnp.array(total_rewards))),
            'reward_variance': float(total_variance),
            'n_samples': len(recent_rewards)
        }
    }


def compute_adaptive_thresholds(
    scm: pyr.PMap,
    difficulty_level: Optional[int] = None,
    base_improvement_threshold: float = 0.1,
    base_diversity_threshold: int = 3
) -> Dict[str, float]:
    """
    Compute adaptive thresholds based on SCM characteristics.
    
    Args:
        scm: The SCM structure to analyze
        difficulty_level: Optional curriculum difficulty level (1-5)
        base_improvement_threshold: Base threshold for improvement
        base_diversity_threshold: Base threshold for diversity
        
    Returns:
        Dictionary with adaptive threshold values
    """
    n_variables = len(scm.get('variables', set()))
    n_edges = len(scm.get('edges', frozenset()))
    
    # Scale thresholds based on graph size
    if n_variables <= 5:
        size_factor = 1.0  # Small graphs - use base thresholds
    elif n_variables <= 10:
        size_factor = 0.7  # Medium graphs - easier thresholds
    else:
        size_factor = 0.5  # Large graphs - more relaxed thresholds
    
    # Adjust for graph density
    expected_edges = n_variables - 1  # Minimum for connected graph
    density_factor = 1.0 if n_edges <= expected_edges else 0.8  # Dense graphs are harder
    
    # Adjust for curriculum difficulty
    difficulty_factor = 1.0
    if difficulty_level is not None:
        difficulty_factor = 1.0 - (difficulty_level - 1) * 0.1  # Easier thresholds for higher difficulty
    
    # Compute adaptive thresholds
    improvement_threshold = base_improvement_threshold * size_factor * density_factor * difficulty_factor
    diversity_threshold = max(1, int(base_diversity_threshold * size_factor))
    
    return {
        'improvement_threshold': float(improvement_threshold),
        'diversity_threshold': int(diversity_threshold),
        'size_factor': size_factor,
        'density_factor': density_factor,
        'difficulty_factor': difficulty_factor
    }


def create_reward_config(
    target_improvement_weight: float = 2.0,
    true_parent_weight: float = 1.0,
    exploration_weight: float = 0.5,
    improvement_threshold: float = 0.1,
    diversity_threshold: int = 3
) -> Dict[str, Any]:
    """
    Create a validated reward configuration.
    
    Args:
        target_improvement_weight: Weight for target optimization reward
        true_parent_weight: Weight for true parent intervention reward
        exploration_weight: Weight for exploration diversity reward
        improvement_threshold: Minimum improvement for target reward
        diversity_threshold: Maximum frequency for diversity reward
        
    Returns:
        Validated reward configuration dictionary
    """
    config = {
        'weights': {
            'target_improvement': target_improvement_weight,
            'true_parent': true_parent_weight,
            'exploration': exploration_weight
        },
        'improvement_threshold': improvement_threshold,
        'diversity_threshold': diversity_threshold
    }
    
    # Validate configuration
    if any(w < 0 for w in config['weights'].values()):
        raise ValueError("All weights must be non-negative")
    
    if config['improvement_threshold'] <= 0:
        raise ValueError("Improvement threshold must be positive")
    
    if config['diversity_threshold'] <= 0:
        raise ValueError("Diversity threshold must be positive")
    
    return config


def create_adaptive_reward_config(
    scm: pyr.PMap,
    difficulty_level: Optional[int] = None,
    target_improvement_weight: float = 2.0,
    true_parent_weight: float = 1.0,
    exploration_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Create a reward configuration with adaptive thresholds based on SCM characteristics.
    
    Args:
        scm: The SCM structure to analyze
        difficulty_level: Optional curriculum difficulty level (1-5)
        target_improvement_weight: Weight for target optimization reward
        true_parent_weight: Weight for true parent intervention reward
        exploration_weight: Weight for exploration diversity reward
        
    Returns:
        Validated reward configuration with adaptive thresholds
        
    Example:
        >>> scm = create_scm(variables={'X', 'Y', 'Z'}, edges={('X', 'Y')}, ...)
        >>> config = create_adaptive_reward_config(scm, difficulty_level=2)
        >>> config['improvement_threshold']  # Will be adapted to SCM size
        0.07  # Easier threshold for medium difficulty
    """
    # Compute adaptive thresholds
    adaptive_thresholds = compute_adaptive_thresholds(scm, difficulty_level)
    
    # Create configuration with adaptive thresholds
    config = create_reward_config(
        target_improvement_weight=target_improvement_weight,
        true_parent_weight=true_parent_weight,
        exploration_weight=exploration_weight,
        improvement_threshold=adaptive_thresholds['improvement_threshold'],
        diversity_threshold=adaptive_thresholds['diversity_threshold']
    )
    
    # Add adaptive threshold metadata
    config['adaptive_thresholds'] = adaptive_thresholds
    config['scm_characteristics'] = {
        'n_variables': len(scm.get('variables', set())),
        'n_edges': len(scm.get('edges', frozenset())),
        'difficulty_level': difficulty_level
    }
    
    return config


# Convenience function for integration with existing reward system
def compute_verifiable_reward_simple(
    intervention: pyr.PMap,
    outcome: pyr.PMap,
    scm: pyr.PMap,
    current_best: float,
    previous_interventions: List[pyr.PMap],
    config: Optional[Dict[str, Any]] = None
) -> SimpleRewardComponents:
    """
    Convenience function for integration with existing ACBO interfaces.
    
    Args:
        intervention: Intervention specification (pyrsistent map)
        outcome: Outcome sample (pyrsistent map)
        scm: True SCM structure (pyrsistent map)
        current_best: Current best target value
        previous_interventions: History of previous interventions
        config: Optional reward configuration
        
    Returns:
        SimpleRewardComponents with computed rewards
    """
    if config is None:
        config = create_reward_config()
    
    # Extract intervention targets
    intervention_targets = intervention.get('targets', set())
    if not isinstance(intervention_targets, set):
        intervention_targets = set(intervention_targets)
    
    # Extract outcome value for target
    target_variable = scm.get('target')
    if target_variable is None:
        raise ValueError("SCM must specify target variable")
    
    outcome_values = outcome.get('values', {})
    outcome_value = outcome_values.get(target_variable)
    if outcome_value is None:
        raise ValueError(f"Outcome must contain value for target variable {target_variable}")
    
    # Extract true parents from SCM
    edges = scm.get('edges', frozenset())
    true_parents = set(parent for parent, child in edges if child == target_variable)
    
    # Extract previous intervention targets
    prev_targets_list = []
    for prev_intervention in previous_interventions:
        prev_targets = prev_intervention.get('targets', set())
        if not isinstance(prev_targets, set):
            prev_targets = set(prev_targets)
        prev_targets_list.append(prev_targets)
    
    return compute_simple_verifiable_reward(
        intervention_targets=intervention_targets,
        outcome_value=outcome_value,
        current_best=current_best,
        true_parents=true_parents,
        target_variable=target_variable,
        previous_interventions=prev_targets_list,
        weights=config['weights'],
        improvement_threshold=config['improvement_threshold'],
        diversity_threshold=config['diversity_threshold']
    )