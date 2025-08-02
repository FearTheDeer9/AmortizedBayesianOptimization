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
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState
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
        state_before, outcome, target_variable, config
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
    target_variable: str,
    config: Optional[pyr.PMap] = None
) -> float:
    """
    Continuous SCM-objective reward that doesn't rely on relative improvement.
    
    This reward measures how close the intervention gets us to the theoretical
    optimum for the SCM, avoiding the problem where optimal interventions are
    avoided once found (because relative improvement becomes zero).
    
    The reward uses either:
    1. True SCM parameters (when available for validation/testing)
    2. Predicted mechanism parameters from AcquisitionState
    3. Fallback to improved relative reward (normalized by value range)
    
    Args:
        state_before: State before intervention
        outcome: Observed outcome sample
        target_variable: Name of the target variable being optimized
        
    Returns:
        Optimization reward in [0, 1] range (0 = worst possible, 1 = optimal)
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
        
        # Try SCM-objective reward first (using predicted mechanisms)
        scm_reward = _compute_scm_objective_reward(
            state_before, target_value, target_variable
        )
        
        if scm_reward is not None:
            logger.debug(
                f"SCM-objective reward: target_value={target_value:.3f}, "
                f"scm_reward={scm_reward:.3f}"
            )
            return scm_reward
        
        # Fallback to improved relative reward (better than original)
        relative_reward = _compute_improved_relative_reward(
            state_before, target_value, target_variable, config
        )
        
        logger.debug(
            f"Relative reward fallback: target_value={target_value:.3f}, "
            f"relative_reward={relative_reward:.3f}"
        )
        
        return relative_reward
        
    except Exception as e:
        logger.error(f"Error computing optimization reward: {e}")
        return 0.0


def _compute_scm_objective_reward(
    state_before: AcquisitionState,
    target_value: float,
    target_variable: str
) -> Optional[float]:
    """
    Compute reward based on distance to theoretical SCM optimum.
    
    Uses mechanism predictions from AcquisitionState to estimate the
    theoretical optimal value for the target variable, then rewards
    based on how close we got to that optimum.
    
    Args:
        state_before: State with mechanism predictions
        target_value: Observed target value from intervention
        target_variable: Name of target variable
        
    Returns:
        Reward in [0, 1] range, or None if cannot compute
    """
    try:
        # Check if we have mechanism predictions
        if not hasattr(state_before, 'mechanism_predictions') or \
           not state_before.mechanism_predictions:
            return None
        
        # Get mechanism predictions for target variable
        mechanism_preds = state_before.mechanism_predictions
        if target_variable not in mechanism_preds:
            return None
        
        target_mechanism = mechanism_preds[target_variable]
        
        # Extract predicted coefficients and intercept
        if not hasattr(target_mechanism, 'coefficients') or \
           not hasattr(target_mechanism, 'intercept'):
            return None
        
        coefficients = target_mechanism.coefficients
        intercept = float(target_mechanism.intercept)
        
        # Compute theoretical optimum with realistic intervention bounds
        intervention_bounds = getattr(state_before, 'intervention_bounds', {})
        default_bound = 3.0  # Reasonable default for normalized variables
        
        theoretical_optimum = intercept
        worst_case_value = intercept
        
        for parent, coeff in coefficients.items():
            lower_bound = intervention_bounds.get(parent, (-default_bound, default_bound))[0]
            upper_bound = intervention_bounds.get(parent, (-default_bound, default_bound))[1]
            
            coeff_val = float(coeff)
            if coeff_val > 0:
                # Positive coefficient: optimal is upper bound, worst is lower
                theoretical_optimum += coeff_val * upper_bound
                worst_case_value += coeff_val * lower_bound
            else:
                # Negative coefficient: optimal is lower bound, worst is upper
                theoretical_optimum += coeff_val * lower_bound
                worst_case_value += coeff_val * upper_bound
        
        # Compute reward as fraction of distance to optimum
        total_range = abs(theoretical_optimum - worst_case_value)
        if total_range == 0:
            return 1.0  # If no improvement possible, perfect score
        
        distance_from_worst = abs(target_value - worst_case_value)
        reward = distance_from_worst / total_range
        
        # Ensure bounded in [0, 1]
        return float(jnp.clip(reward, 0.0, 1.0))
        
    except Exception as e:
        logger.debug(f"Could not compute SCM-objective reward: {e}")
        return None


def _compute_improved_relative_reward(
    state_before: AcquisitionState,
    target_value: float,
    target_variable: str,
    config: Optional[pyr.PMap] = None
) -> float:
    """
    Improved relative reward that compares against expected/mean value baseline.
    
    This reward:
    1. Uses expected value (mean) as baseline instead of range center
    2. Normalizes by the observed value range to provide context
    3. Rewards proportional to outcome improvement (not intervention magnitude)
    4. Prevents "nudging" by using stable baseline
    
    Args:
        state_before: State before intervention
        target_value: Observed target value
        target_variable: Target variable name
        config: Optional configuration with optimization_direction
        
    Returns:
        Reward in [0, 1] range
    """
    try:
        # Extract optimization direction from config
        optimization_direction = 'MAXIMIZE'  # Default
        if config:
            optimization_direction = config.get('optimization_direction', 'MAXIMIZE')
        
        # Use expected value baseline if configured
        use_expected_baseline = config.get('use_expected_value_baseline', True) if config else True
        
        # Estimate expected value and range from experience buffer
        expected_value = state_before.best_value  # Fallback
        value_range = 2.0  # Default range
        
        if hasattr(state_before, 'buffer') and state_before.buffer:
            # Extract target values from buffer to compute statistics
            buffer_values = []
            for sample in state_before.buffer.samples[-50:]:  # Use more samples for stable estimate
                sample_values = get_values(sample)
                if target_variable in sample_values:
                    buffer_values.append(float(sample_values[target_variable]))
            
            if len(buffer_values) > 1:
                # Compute expected value (mean) as baseline
                expected_value = float(jnp.mean(jnp.array(buffer_values)))
                value_range = max(buffer_values) - min(buffer_values)
                
                # If range is too small, use standard deviation based range
                if value_range < 0.1:
                    std_dev = float(jnp.std(jnp.array(buffer_values)))
                    value_range = max(4 * std_dev, 0.5)  # At least 4 std devs or 0.5
            else:
                # Not enough data, use current best as expected value
                expected_value = state_before.best_value
        
        # Use expected value or best value as baseline based on configuration
        baseline_value = expected_value if use_expected_baseline else state_before.best_value
        
        # Avoid division by zero
        if value_range == 0:
            value_range = 1.0
        
        # Compute reward based on improvement from baseline
        if optimization_direction == 'MINIMIZE':
            # For minimization: reward = how much we decreased from baseline
            improvement = baseline_value - target_value
        else:
            # For maximization: reward = how much we increased from baseline
            improvement = target_value - baseline_value
        
        # Normalize improvement by value range
        normalized_improvement = improvement / value_range
        
        # Convert to [0, 1] range using sigmoid
        # Center at 0 improvement, positive improvement gives reward > 0.5
        reward = float(1.0 / (1.0 + jnp.exp(-4.0 * normalized_improvement)))
        
        # Log details for debugging
        logger.debug(
            f"Relative reward: target={target_value:.3f}, baseline={baseline_value:.3f}, "
            f"improvement={improvement:.3f}, normalized={normalized_improvement:.3f}, "
            f"reward={reward:.3f}, direction={optimization_direction}"
        )
        
        return reward
        
    except Exception as e:
        logger.debug(f"Error in improved relative reward: {e}")
        # Ultimate fallback: neutral reward
        return 0.5


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
    exploration_weight: float = 0.1,
    optimization_direction: str = 'MAXIMIZE'
) -> pyr.PMap:
    """
    Create a default reward configuration.
    
    Args:
        optimization_weight: Weight for target variable optimization
        structure_weight: Weight for structure discovery  
        parent_weight: Weight for parent intervention guidance
        exploration_weight: Weight for exploration bonus
        optimization_direction: Direction of optimization ('MINIMIZE' or 'MAXIMIZE')
        
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
        'exploration_weight': exploration_weight,
        'optimization_direction': optimization_direction
    })
    
    if not validate_reward_config(config):
        raise ValueError("Invalid reward configuration")
    
    return config


def compute_adaptive_thresholds(
    scm: pyr.PMap,
    difficulty_level: Optional[int] = None,
    base_improvement_threshold: float = 0.1,
    base_diversity_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Compute adaptive thresholds based on SCM characteristics.
    
    Adapted from the old binary reward system but updated for continuous rewards.
    
    Args:
        scm: The SCM structure to analyze
        difficulty_level: Optional curriculum difficulty level (1-5)
        base_improvement_threshold: Base threshold for optimization (continuous)
        base_diversity_threshold: Base threshold for exploration (continuous)
        
    Returns:
        Dictionary with adaptive threshold values
    """
    try:
        # Handle different SCM structures - be flexible about field names
        variables = scm.get('variables', scm.get('nodes', set()))
        edges = scm.get('edges', scm.get('edge_list', frozenset()))
        
        n_variables = len(variables) if variables else 3  # Default fallback
        n_edges = len(edges) if edges else 0
        
        # Scale thresholds based on graph size
        if n_variables <= 5:
            size_factor = 1.0  # Small graphs - use base thresholds
        elif n_variables <= 10:
            size_factor = 0.7  # Medium graphs - easier thresholds
        else:
            size_factor = 0.5  # Large graphs - more relaxed thresholds
        
        # Adjust for graph density
        expected_edges = max(1, n_variables - 1)  # Minimum for connected graph
        density_factor = 1.0 if n_edges <= expected_edges else 0.8  # Dense graphs are harder
        
        # Adjust for curriculum difficulty
        difficulty_factor = 1.0
        if difficulty_level is not None:
            difficulty_factor = max(0.5, 1.0 - (difficulty_level - 1) * 0.1)  # Easier thresholds for higher difficulty
        
        # Compute adaptive thresholds
        improvement_threshold = base_improvement_threshold * size_factor * density_factor * difficulty_factor
        diversity_threshold = base_diversity_threshold * size_factor * difficulty_factor
        
        return {
            'improvement_threshold': float(improvement_threshold),
            'diversity_threshold': float(diversity_threshold),
            'size_factor': size_factor,
            'density_factor': density_factor,
            'difficulty_factor': difficulty_factor,
            'n_variables': n_variables,
            'n_edges': n_edges
        }
        
    except Exception as e:
        logger.warning(f"Error computing adaptive thresholds: {e}, using defaults")
        return {
            'improvement_threshold': base_improvement_threshold,
            'diversity_threshold': base_diversity_threshold,
            'size_factor': 1.0,
            'density_factor': 1.0,
            'difficulty_factor': 1.0,
            'n_variables': 3,
            'n_edges': 0
        }


def create_adaptive_reward_config(
    scm: pyr.PMap,
    difficulty_level: Optional[int] = None,
    optimization_weight: float = 1.0,
    structure_weight: float = 0.5,
    parent_weight: float = 0.3,
    exploration_weight: float = 0.1
) -> pyr.PMap:
    """
    Create a reward configuration with adaptive thresholds based on SCM characteristics.
    
    Updated to work with our continuous reward system instead of the old binary system.
    
    Args:
        scm: The SCM structure to analyze
        difficulty_level: Optional curriculum difficulty level (1-5)
        optimization_weight: Weight for target optimization reward
        structure_weight: Weight for structure discovery reward
        parent_weight: Weight for parent intervention reward
        exploration_weight: Weight for exploration bonus
        
    Returns:
        Validated reward configuration with adaptive thresholds
        
    Example:
        >>> scm = create_scm(variables={'X', 'Y', 'Z'}, edges={('X', 'Y')})
        >>> config = create_adaptive_reward_config(scm, difficulty_level=2)
        >>> config['adaptive_thresholds']['improvement_threshold']  # Adapted threshold
    """
    # Compute adaptive thresholds
    adaptive_thresholds = compute_adaptive_thresholds(scm, difficulty_level)
    
    # Create base configuration
    config = create_default_reward_config(
        optimization_weight=optimization_weight,
        structure_weight=structure_weight,
        parent_weight=parent_weight,
        exploration_weight=exploration_weight
    )
    
    # Add adaptive threshold metadata  
    enhanced_config = config.set('adaptive_thresholds', adaptive_thresholds)
    enhanced_config = enhanced_config.set('scm_characteristics', pyr.m(**{
        'n_variables': adaptive_thresholds['n_variables'],
        'n_edges': adaptive_thresholds['n_edges'], 
        'difficulty_level': difficulty_level
    }))
    
    return enhanced_config


def validate_reward_consistency(
    reward_history: List[RewardComponents],
    window_size: int = 50
) -> Dict[str, Any]:
    """
    Validate reward consistency and detect potential gaming patterns.
    
    Updated to work with continuous RewardComponents instead of binary SimpleRewardComponents.
    
    Args:
        reward_history: Recent reward components from training
        window_size: Number of recent rewards to analyze
        
    Returns:
        Validation metrics and gaming detection results
    """
    if not reward_history:
        return {'valid': True, 'warning': 'No reward history'}
    
    recent_rewards = reward_history[-window_size:]
    
    # Component statistics for continuous rewards
    opt_rewards = [r.optimization_reward for r in recent_rewards]
    struct_rewards = [r.structure_discovery_reward for r in recent_rewards]
    parent_rewards = [r.parent_intervention_reward for r in recent_rewards]
    explore_rewards = [r.exploration_bonus for r in recent_rewards]
    total_rewards = [r.total_reward for r in recent_rewards]
    
    # Check for gaming patterns (adapted for continuous rewards)
    gaming_issues = []
    
    # 1. Check if optimization reward is consistently very low (no optimization progress)
    opt_mean = float(jnp.mean(jnp.array(opt_rewards)))
    if opt_mean < 0.1:
        gaming_issues.append(f"Very low optimization reward mean: {opt_mean:.3f}")
    
    # 2. Check if parent intervention rate is suspiciously high (gaming structure guidance)
    parent_mean = float(jnp.mean(jnp.array(parent_rewards)))
    if parent_mean > 0.9:
        gaming_issues.append(f"Suspiciously high parent intervention mean: {parent_mean:.3f}")
    
    # 3. Check if exploration is consistently very low (mode collapse)
    explore_mean = float(jnp.mean(jnp.array(explore_rewards)))
    if explore_mean < 0.01:
        gaming_issues.append(f"Very low exploration reward mean: {explore_mean:.3f}")
    
    # 4. Check reward variance (too little variance suggests gaming)
    total_variance = float(jnp.var(jnp.array(total_rewards)))
    if total_variance < 0.001:
        gaming_issues.append(f"Very low total reward variance: {total_variance:.6f}")
    
    # 5. Check for NaN or infinite values
    all_rewards = opt_rewards + struct_rewards + parent_rewards + explore_rewards + total_rewards
    if not all(jnp.isfinite(r) for r in all_rewards):
        gaming_issues.append("Found non-finite reward values")
    
    return {
        'valid': len(gaming_issues) == 0,
        'gaming_issues': gaming_issues,
        'component_means': {
            'optimization_reward': opt_mean,
            'structure_discovery_reward': float(jnp.mean(jnp.array(struct_rewards))),
            'parent_intervention_reward': parent_mean,
            'exploration_bonus': explore_mean
        },
        'statistics': {
            'mean_total_reward': float(jnp.mean(jnp.array(total_rewards))),
            'total_reward_variance': total_variance,
            'optimization_variance': float(jnp.var(jnp.array(opt_rewards))),
            'structure_variance': float(jnp.var(jnp.array(struct_rewards))),
            'n_samples': len(recent_rewards)
        }
    }


# Legacy compatibility functions for reward_rubric.py
def target_improvement_reward(
    outcome_value: float,
    current_best: float,
    improvement_threshold: float = 0.1
) -> float:
    """
    Legacy wrapper for binary target improvement reward.
    
    This provides compatibility for the reward_rubric system while using
    our continuous reward approach under the hood.
    
    Args:
        outcome_value: Target variable value from intervention outcome
        current_best: Current best target value achieved
        improvement_threshold: Minimum improvement required for reward
        
    Returns:
        1.0 if improvement > threshold, 0.0 otherwise
    """
    if not jnp.isfinite(outcome_value) or not jnp.isfinite(current_best):
        return 0.0
    
    improvement = outcome_value - current_best
    return 1.0 if improvement > improvement_threshold else 0.0


def exploration_diversity_reward(
    intervention_targets: set,
    previous_interventions: list,
    diversity_threshold: int = 3
) -> float:
    """
    Legacy wrapper for binary exploration diversity reward.
    
    This provides compatibility for the reward_rubric system.
    
    Args:
        intervention_targets: Variables intervened upon in current step
        previous_interventions: List of intervention target sets from previous steps
        diversity_threshold: Maximum frequency before reward becomes 0
        
    Returns:
        1.0 if intervention frequency < threshold, 0.0 otherwise
    """
    if not intervention_targets:
        return 0.0
    
    # Count frequency of these exact intervention targets
    frequency = sum(1 for prev_targets in previous_interventions 
                   if len(intervention_targets & prev_targets) > 0)
    
    return 1.0 if frequency < diversity_threshold else 0.0
