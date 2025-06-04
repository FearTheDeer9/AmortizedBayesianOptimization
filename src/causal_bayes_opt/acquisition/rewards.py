"""
Multi-Component Verifiable Rewards for Dual-Objective ACBO.

This module implements sophisticated reward decomposition for our dual-objective system
that balances both optimization and structure learning, unlike pure structure learning
approaches that focus on a single objective.

Key Components:
1. RewardComponents: Decomposed reward structure for analysis
2. compute_verifiable_reward: Main reward computation for dual objectives  
3. Individual reward functions for each component
4. Configurable weighting system for different problem requirements

The reward system is designed to be verifiable (no human feedback required) and
provides detailed decomposition for understanding what drives learning.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from .state import AcquisitionState
from ..data_structures.sample import get_values, get_intervention_targets
from ..data_structures.buffer import ExperienceBuffer
from ..avici_integration.parent_set.posterior import ParentSetPosterior

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardComponents:
    """
    Decomposed reward components for dual-objective learning analysis.
    
    This decomposition allows us to understand what drives policy learning
    and adjust the balance between optimization and structure discovery.
    
    Attributes:
        optimization_reward: Target variable improvement (unique to our dual-objective approach)
        structure_discovery_reward: AVICI improvement through information gain  
        parent_intervention_reward: Bonus for intervening on likely parents
        exploration_bonus: Incentive for diverse intervention strategies
        total_reward: Weighted combination of all components
        metadata: Additional context and debugging information
    """
    optimization_reward: float        # Target variable improvement (not in pure structure learning)
    structure_discovery_reward: float # AVICI improvement (similar to CAASL approach)
    parent_intervention_reward: float # Bonus for intervening on likely parents
    exploration_bonus: float          # Encourage diverse interventions
    total_reward: float              # Weighted combination
    metadata: pyr.PMap[str, Any] = pyr.m()
    
    def __post_init__(self):
        """Validate reward components."""
        # Check that individual rewards are finite
        for field_name in ['optimization_reward', 'structure_discovery_reward', 
                          'parent_intervention_reward', 'exploration_bonus', 'total_reward']:
            value = getattr(self, field_name)
            if not jnp.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
    
    def summary(self) -> Dict[str, float]:
        """Create human-readable summary of reward decomposition."""
        return {
            'total_reward': self.total_reward,
            'optimization_component': self.optimization_reward,
            'structure_component': self.structure_discovery_reward,
            'parent_guidance_component': self.parent_intervention_reward,
            'exploration_component': self.exploration_bonus,
            'optimization_fraction': self.optimization_reward / max(abs(self.total_reward), 1e-8),
            'structure_fraction': self.structure_discovery_reward / max(abs(self.total_reward), 1e-8),
        }


def compute_verifiable_reward(
    state_before: AcquisitionState,
    intervention: pyr.PMap,
    outcome: pyr.PMap,
    state_after: AcquisitionState,
    config: Optional[pyr.PMap] = None
) -> RewardComponents:
    """
    Compute verifiable reward for dual objectives.
    
    This is the main reward function that balances multiple objectives:
    1. Target variable optimization (key for our use case vs pure structure learning)
    2. Structure discovery through information gain
    3. Intervention quality bonuses
    4. Exploration incentives
    
    Args:
        state_before: Acquisition state before intervention
        state_after: Acquisition state after intervention and outcome
        intervention: Applied intervention specification
        outcome: Observed outcome sample from the intervention
        config: Optional configuration for reward weights and parameters
        
    Returns:
        RewardComponents with detailed decomposition
        
    Raises:
        ValueError: If states or intervention are inconsistent
        
    Example:
        >>> rewards = compute_verifiable_reward(
        ...     state_before, intervention, outcome, state_after, config
        ... )
        >>> print(f"Total reward: {rewards.total_reward:.3f}")
        >>> print(f"Optimization: {rewards.optimization_reward:.3f}")
        >>> print(f"Structure: {rewards.structure_discovery_reward:.3f}")
    """
    # Validate inputs
    if state_before.current_target != state_after.current_target:
        raise ValueError("State target mismatch between before and after")
    
    if state_before.step >= state_after.step:
        raise ValueError("After state step must be greater than before state step")
    
    # Set default config if not provided
    if config is None:
        config = pyr.m()
    
    # Get reward weights (can be adjusted for different problem requirements)
    default_weights = {
        'optimization': 1.0,      # Primary objective for ACBO
        'structure': 0.5,         # Secondary objective (structure learning)
        'parent': 0.3,           # Learning guidance through parent intervention
        'exploration': 0.1       # Diversity maintenance and exploration
    }
    
    weights = config.get('reward_weights', default_weights)
    target_variable = state_before.current_target
    
    # 1. Optimization reward: target variable improvement (CRITICAL for our dual-objective approach)
    opt_reward = _compute_optimization_reward(
        state_before, outcome, target_variable
    )
    
    # 2. Structure discovery reward: AVICI improvement (similar to CAASL but adapted for dual objectives)
    struct_reward = _compute_structure_discovery_reward(
        state_before.posterior, state_after.posterior
    )
    
    # 3. Parent intervention reward: guide structure learning with uncertainty information
    parent_reward = _compute_parent_intervention_reward(
        intervention, state_before.marginal_parent_probs
    )
    
    # 4. Exploration bonus: prevent mode collapse and encourage diversity
    exploration_bonus = _compute_exploration_bonus(
        intervention, state_before.buffer, config.get('exploration_weight', 0.1)
    )
    
    # Combine with learnable/configurable weights
    total = (
        weights['optimization'] * opt_reward +
        weights['structure'] * struct_reward + 
        weights['parent'] * parent_reward +
        weights['exploration'] * exploration_bonus
    )
    
    # Create metadata for analysis and debugging
    metadata = pyr.m(**{
        'weights_used': weights,
        'target_variable': target_variable,
        'intervention_type': intervention.get('type', 'unknown'),
        'intervention_targets': list(intervention.get('targets', set())),
        'uncertainty_reduction': state_before.uncertainty_bits - state_after.uncertainty_bits,
        'step_before': state_before.step,
        'step_after': state_after.step,
        'best_value_before': state_before.best_value,
        'best_value_after': state_after.best_value,
        'buffer_size_before': state_before.buffer_statistics.total_samples,
        'buffer_size_after': state_after.buffer_statistics.total_samples,
    })
    
    return RewardComponents(
        optimization_reward=opt_reward,
        structure_discovery_reward=struct_reward,
        parent_intervention_reward=parent_reward,
        exploration_bonus=exploration_bonus,
        total_reward=total,
        metadata=metadata
    )


def _compute_optimization_reward(
    state_before: AcquisitionState,
    outcome: pyr.PMap,
    target_variable: str
) -> float:
    """
    Reward based on target variable improvement.
    
    This is a key component for optimization objectives that pure structure 
    learning approaches don't include. It directly incentivizes finding
    interventions that improve the target variable.
    
    Args:
        state_before: State before intervention
        outcome: Observed outcome sample
        target_variable: Name of the target variable being optimized
        
    Returns:
        Optimization reward in [-1, 1] range (bounded via tanh)
        
    Note:
        Uses tanh normalization to ensure bounded rewards for stable training.
        Positive rewards for improvements, negative for deteriorations.
    """
    try:
        # Get target value from outcome
        outcome_values = get_values(outcome)
        if target_variable not in outcome_values:
            logger.warning(f"Target variable '{target_variable}' not in outcome")
            return 0.0
        
        target_value = float(outcome_values[target_variable])
        
        # Check for invalid values
        if not jnp.isfinite(target_value):
            logger.warning(f"Non-finite target value: {target_value}")
            return 0.0
        
        # Compute improvement over current best
        improvement = target_value - state_before.best_value
        
        # Apply tanh normalization for bounded rewards
        # Scale factor can be adjusted based on typical improvement magnitudes
        scale_factor = 1.0  # Can be made configurable
        normalized_reward = float(jnp.tanh(improvement / scale_factor))
        
        logger.debug(
            f"Optimization reward: target_value={target_value:.3f}, "
            f"best_before={state_before.best_value:.3f}, "
            f"improvement={improvement:.3f}, reward={normalized_reward:.3f}"
        )
        
        return normalized_reward
        
    except Exception as e:
        logger.error(f"Error computing optimization reward: {e}")
        return 0.0


def _compute_structure_discovery_reward(
    posterior_before: ParentSetPosterior,
    posterior_after: ParentSetPosterior  
) -> float:
    """
    Reward based on information gain using our rich posterior representation.
    
    Uses uncertainty reduction rather than simple accuracy metrics for more
    nuanced structure discovery guidance. This is similar to CAASL but adapted
    for our dual-objective setting.
    
    Args:
        posterior_before: Posterior before intervention
        posterior_after: Posterior after intervention
        
    Returns:
        Information gain reward in [0, 1] range
        
    Note:
        Normalized by maximum possible uncertainty reduction to ensure
        bounded rewards across different problem sizes.
    """
    try:
        # Validate same target variable
        if posterior_before.target_variable != posterior_after.target_variable:
            logger.warning(
                f"Posterior target mismatch: '{posterior_before.target_variable}' vs "
                f"'{posterior_after.target_variable}'"
            )
            return 0.0
        
        # Compute uncertainty reduction (information gain)
        uncertainty_reduction = posterior_before.uncertainty - posterior_after.uncertainty
        
        # Normalize by maximum possible uncertainty reduction
        max_uncertainty = posterior_before.uncertainty
        if max_uncertainty > 1e-8:  # Avoid division by zero
            normalized_reduction = uncertainty_reduction / max_uncertainty
            # Clip to [0, 1] range (negative reduction means uncertainty increased)
            reward = float(jnp.clip(normalized_reduction, 0.0, 1.0))
        else:
            # Already very certain - no reward for structure discovery
            reward = 0.0
        
        logger.debug(
            f"Structure discovery reward: uncertainty_before={posterior_before.uncertainty:.3f}, "
            f"uncertainty_after={posterior_after.uncertainty:.3f}, "
            f"reduction={uncertainty_reduction:.3f}, reward={reward:.3f}"
        )
        
        return reward
        
    except Exception as e:
        logger.error(f"Error computing structure discovery reward: {e}")
        return 0.0


def _compute_parent_intervention_reward(
    intervention: pyr.PMap,
    marginal_parent_probs: Dict[str, float]
) -> float:
    """
    Reward for intervening on variables likely to be parents.
    
    This guides the policy toward interventions that are likely to be informative
    for structure learning, using our rich uncertainty information rather than
    simple heuristics.
    
    Args:
        intervention: Applied intervention specification
        marginal_parent_probs: Marginal parent probabilities for all variables
        
    Returns:
        Parent intervention reward in [0, 1] range
        
    Note:
        Only applies to perfect interventions. Could be extended to other types.
    """
    try:
        # Only reward perfect interventions (can be extended later)
        if intervention.get('type') != 'perfect':
            return 0.0
        
        targets = intervention.get('targets', set())
        if not targets:
            return 0.0
        
        # Average parent probability of intervened variables
        target_probs = []
        for var in targets:
            prob = marginal_parent_probs.get(var, 0.0)
            if jnp.isfinite(prob) and 0.0 <= prob <= 1.0:
                target_probs.append(prob)
            else:
                logger.warning(f"Invalid marginal probability for {var}: {prob}")
        
        if not target_probs:
            return 0.0
        
        avg_parent_prob = float(jnp.mean(jnp.array(target_probs)))
        
        logger.debug(
            f"Parent intervention reward: targets={list(targets)}, "
            f"avg_parent_prob={avg_parent_prob:.3f}"
        )
        
        return avg_parent_prob
        
    except Exception as e:
        logger.error(f"Error computing parent intervention reward: {e}")
        return 0.0


def _compute_exploration_bonus(
    intervention: pyr.PMap,
    buffer: ExperienceBuffer, 
    weight: float
) -> float:
    """
    Bonus for exploring under-sampled intervention types.
    
    Encourages diversity in intervention strategies to prevent mode collapse
    and ensure adequate exploration of the intervention space.
    
    Args:
        intervention: Applied intervention specification
        buffer: Experience buffer with intervention history
        weight: Weight for exploration bonus (typically small, e.g., 0.1)
        
    Returns:
        Exploration bonus in [0, weight] range
        
    Note:
        Uses inverse frequency weighting to encourage under-explored interventions.
    """
    try:
        # Only apply to perfect interventions for now
        if intervention.get('type') != 'perfect':
            return 0.0
        
        targets = intervention.get('targets', set())
        if not targets:
            return 0.0
        
        # Count previous interventions on these specific targets
        previous_count = 0
        all_interventions = buffer.get_interventions()
        
        for prev_intervention, _ in all_interventions:
            prev_targets = prev_intervention.get('targets', set())
            if prev_targets == targets:  # Exact match of target set
                previous_count += 1
        
        total_interventions = len(all_interventions)
        
        if total_interventions == 0:
            # First intervention gets full exploration bonus
            bonus = weight
            frequency = 0.0  # For logging
        else:
            # Inverse frequency bonus
            frequency = previous_count / total_interventions
            bonus = weight * (1.0 - frequency)
            bonus = max(0.0, bonus)  # Ensure non-negative
        
        logger.debug(
            f"Exploration bonus: targets={list(targets)}, "
            f"previous_count={previous_count}, total={total_interventions}, "
            f"frequency={frequency:.3f}, bonus={bonus:.3f}"
        )
        
        return float(bonus)
        
    except Exception as e:
        logger.error(f"Error computing exploration bonus: {e}")
        return 0.0


# Additional utility functions for reward analysis and debugging

def analyze_reward_trends(
    reward_history: List[RewardComponents],
    window_size: int = 10
) -> Dict[str, Any]:
    """
    Analyze trends in reward components over time.
    
    Args:
        reward_history: List of RewardComponents from training
        window_size: Size of moving average window
        
    Returns:
        Dictionary with trend analysis
        
    Example:
        >>> trends = analyze_reward_trends(training_rewards, window_size=20)
        >>> print(f"Recent optimization trend: {trends['optimization_trend']:.3f}")
    """
    if not reward_history:
        return {}
    
    # Extract component time series
    total_rewards = [r.total_reward for r in reward_history]
    opt_rewards = [r.optimization_reward for r in reward_history]
    struct_rewards = [r.structure_discovery_reward for r in reward_history]
    parent_rewards = [r.parent_intervention_reward for r in reward_history]
    explore_rewards = [r.exploration_bonus for r in reward_history]
    
    def compute_trend(values: List[float], window: int) -> float:
        """Compute slope of recent trend."""
        if len(values) < window:
            return 0.0
        recent = values[-window:]
        x = jnp.arange(len(recent))
        y = jnp.array(recent)
        # Linear regression slope
        slope = jnp.sum((x - jnp.mean(x)) * (y - jnp.mean(y))) / jnp.sum((x - jnp.mean(x))**2)
        return float(slope)
    
    return {
        'total_reward_trend': compute_trend(total_rewards, window_size),
        'optimization_trend': compute_trend(opt_rewards, window_size),
        'structure_trend': compute_trend(struct_rewards, window_size),
        'parent_trend': compute_trend(parent_rewards, window_size),
        'exploration_trend': compute_trend(explore_rewards, window_size),
        'recent_avg_total': float(jnp.mean(jnp.array(total_rewards[-window_size:]))),
        'recent_avg_optimization': float(jnp.mean(jnp.array(opt_rewards[-window_size:]))),
        'recent_avg_structure': float(jnp.mean(jnp.array(struct_rewards[-window_size:]))),
        'reward_volatility': float(jnp.std(jnp.array(total_rewards[-window_size:]))),
        'n_samples': len(reward_history)
    }


def validate_reward_config(config: pyr.PMap) -> bool:
    """
    Validate reward configuration for common issues.
    
    Args:
        config: Configuration map to validate
        
    Returns:
        True if configuration is valid, False otherwise
        
    Example:
        >>> config = pyr.m(**
        {'reward_weights': {'optimization': 1.0, 'structure': 0.5}})
        >>> valid = validate_reward_config(config)
    """
    try:
        # Check reward weights
        weights = config.get('reward_weights', {})
        
        required_weights = ['optimization', 'structure', 'parent', 'exploration']
        for weight_name in required_weights:
            if weight_name not in weights:
                logger.warning(f"Missing reward weight: {weight_name}")
                return False
            
            weight_value = weights[weight_name]
            if not isinstance(weight_value, (int, float)):
                logger.error(f"Reward weight {weight_name} must be numeric, got {type(weight_value)}")
                return False
            
            if not jnp.isfinite(weight_value):
                logger.error(f"Reward weight {weight_name} must be finite, got {weight_value}")
                return False
            
            if weight_value < 0:
                logger.warning(f"Negative reward weight {weight_name}: {weight_value}")
        
        # Check exploration weight
        exploration_weight = config.get('exploration_weight', 0.1)
        if not (0.0 <= exploration_weight <= 1.0):
            logger.warning(f"Exploration weight {exploration_weight} outside [0,1] range")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating reward config: {e}")
        return False


def create_default_reward_config(
    optimization_weight: float = 1.0,
    structure_weight: float = 0.5,
    parent_weight: float = 0.3,
    exploration_weight: float = 0.1
) -> pyr.PMap:
    """
    Create a default reward configuration.
    
    Args:
        optimization_weight: Weight for target variable optimization
        structure_weight: Weight for structure discovery  
        parent_weight: Weight for parent intervention guidance
        exploration_weight: Weight for exploration bonus
        
    Returns:
        Validated reward configuration
        
    Example:
        >>> config = create_default_reward_config(
        ...     optimization_weight=2.0,  # Emphasize optimization
        ...     structure_weight=0.3       # De-emphasize structure discovery
        ... )
    """
    config = pyr.m(**{
        'reward_weights': {
            'optimization': optimization_weight,
            'structure': structure_weight,
            'parent': parent_weight,
            'exploration': exploration_weight
        },
        'exploration_weight': exploration_weight
    })
    
    if not validate_reward_config(config):
        raise ValueError("Invalid reward configuration")
    
    return config
