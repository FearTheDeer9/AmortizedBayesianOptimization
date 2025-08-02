# Uncertainty-Guided Exploration Strategies for ACBO
# Implementation following Phase 3 Component 5 specification

import jax.numpy as jnp
import pyrsistent as pyr
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..data_structures.buffer import ExperienceBuffer
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies."""
    uncertainty_weight: float = 1.0
    count_weight: float = 0.1
    variable_uncertainty_weight: float = 0.5
    temperature: float = 1.0
    
    # Adaptive exploration parameters
    initial_temperature: float = 2.0
    final_temperature: float = 0.1
    adaptation_steps: int = 1000
    stagnation_threshold: int = 100


class UncertaintyGuidedExploration:
    """
    Exploration strategy leveraging our rich uncertainty infrastructure.
    
    Uses ParentSetPosterior uncertainty plus optimization progress 
    to guide exploration vs. exploitation decisions.
    """
    
    def __init__(self, config: ExplorationConfig):
        """Initialize uncertainty-guided exploration strategy.
        
        Args:
            config: Configuration parameters for exploration
        """
        self.config = config
    
    def compute_exploration_bonus(
        self,
        state: AcquisitionState,
        candidate_intervention: pyr.PMap
    ) -> float:
        """Compute exploration bonus for candidate intervention.
        
        Args:
            state: Current acquisition state with uncertainty information
            candidate_intervention: Intervention to evaluate
            
        Returns:
            Exploration bonus value (higher = more exploration incentive)
        """
        # Epistemic uncertainty: encourage exploration when intervention provides information gain
        epistemic_bonus = self._compute_epistemic_bonus(state, candidate_intervention)
        
        # Count-based bonus: encourage under-explored variable combinations
        count_bonus = self._compute_count_bonus(candidate_intervention, state.buffer)
        
        # Variable uncertainty: prefer variables with uncertain parent status
        var_uncertainty_bonus = self._compute_variable_uncertainty_bonus(
            candidate_intervention, state.marginal_parent_probs
        )
        
        # Combine bonuses with weights
        total_bonus = (
            self.config.uncertainty_weight * epistemic_bonus +
            self.config.count_weight * count_bonus +
            self.config.variable_uncertainty_weight * var_uncertainty_bonus
        )
        
        # Apply temperature scaling
        return total_bonus / self.config.temperature
    
    def _compute_epistemic_bonus(self, state: AcquisitionState, intervention: pyr.PMap) -> float:
        """Compute bonus based on expected information gain from intervention.
        
        Args:
            state: Current acquisition state
            intervention: Candidate intervention to evaluate
            
        Returns:
            Expected epistemic bonus (information gain prediction)
        """
        # For interventions on variables with uncertain parent status,
        # we expect higher information gain about the causal structure
        if intervention.get('type') != 'perfect':
            return 0.0
            
        targets = intervention.get('targets', frozenset())
        if not targets:
            return 0.0
        
        # Compute expected information gain based on target variable uncertainties
        # Variables with uncertain parent status (near 0.5) provide more information
        # when intervened upon, as they help resolve structural uncertainty
        total_expected_gain = 0.0
        for var in targets:
            # Get marginal parent probability for this variable
            parent_prob = state.marginal_parent_probs.get(var, 0.0)
            
            # Expected information gain is higher for variables with uncertain status
            # Maximum gain when prob ~= 0.5, minimum when prob ~= 0.0 or 1.0
            uncertainty_factor = 1.0 - 2.0 * abs(parent_prob - 0.5)
            
            # Scale by overall posterior uncertainty - more uncertain posteriors
            # have more room for information gain
            expected_gain = uncertainty_factor * state.uncertainty_bits
            total_expected_gain += expected_gain
        
        # Average information gain across all target variables
        return float(total_expected_gain / len(targets))
    
    def _compute_count_bonus(self, intervention: pyr.PMap, buffer: ExperienceBuffer) -> float:
        """Bonus inversely proportional to intervention frequency.
        
        Args:
            intervention: Candidate intervention
            buffer: Experience buffer with intervention history
            
        Returns:
            Count-based exploration bonus
        """
        if intervention.get('type') != 'perfect':
            return 0.0
            
        targets = intervention.get('targets', frozenset())
        if not targets:
            return 0.0
        
        # Count previous interventions on these targets
        try:
            count = len(buffer.filter_interventions_by_targets(targets))
            total = buffer.num_interventions()
        except AttributeError:
            # Fallback if buffer doesn't have these methods
            return self.config.count_weight
        
        if total == 0:
            return 1.0  # Maximum bonus for first intervention
        
        # Inverse frequency bonus
        frequency = count / total
        return 1.0 - frequency
    
    def _compute_variable_uncertainty_bonus(
        self, 
        intervention: pyr.PMap, 
        marginal_probs: Dict[str, float]
    ) -> float:
        """Bonus for variables with uncertain parent status (prob ~0.5).
        
        Args:
            intervention: Candidate intervention
            marginal_probs: Marginal parent probabilities for each variable
            
        Returns:
            Variable uncertainty bonus
        """
        if intervention.get('type') != 'perfect':
            return 0.0
            
        targets = intervention.get('targets', frozenset())
        if not targets:
            return 0.0
        
        uncertainties = []
        for var in targets:
            prob = marginal_probs.get(var, 0.0)
            # Maximum uncertainty at prob=0.5, minimum at prob=0.0 or 1.0
            uncertainty = 1.0 - 2.0 * abs(prob - 0.5)
            uncertainties.append(uncertainty)
        
        return float(jnp.mean(jnp.array(uncertainties)))


class AdaptiveExploration:
    """
    Adaptive exploration balancing optimization and structure discovery.
    
    Adapts exploration based on both optimization progress and 
    structural uncertainty - our dual-objective advantage.
    """
    
    def __init__(self, config: ExplorationConfig):
        """Initialize adaptive exploration strategy.
        
        Args:
            config: Configuration parameters for adaptive exploration
        """
        self.config = config
    
    def get_exploration_temperature(self, step: int, state: AcquisitionState) -> float:
        """Get temperature based on step and optimization progress.
        
        Args:
            step: Current training step
            state: Current acquisition state
            
        Returns:
            Exploration temperature (higher = more exploration)
        """
        # Base temperature decay over training steps
        base_progress = min(step / self.config.adaptation_steps, 1.0)
        
        # Adjust based on optimization stagnation
        stagnation_bonus = 0.0
        if hasattr(state, 'optimization_stagnation_steps'):
            stagnation_steps = getattr(state, 'optimization_stagnation_steps', 0)
            if isinstance(stagnation_steps, (int, float)) and stagnation_steps > 0:
                stagnation_bonus = min(stagnation_steps / self.config.stagnation_threshold, 0.5)
                # Reduce progress if we're stagnating (increase exploration)
                base_progress = max(0.0, base_progress - stagnation_bonus)
        
        # Linear interpolation between initial and final temperatures
        temperature = (
            self.config.initial_temperature * (1 - base_progress) + 
            self.config.final_temperature * base_progress
        )
        
        return temperature
    
    def should_explore(self, state: AcquisitionState, step: int) -> bool:
        """Decide whether to prioritize exploration vs exploitation.
        
        Args:
            state: Current acquisition state
            step: Current training step
            
        Returns:
            True if should prioritize exploration, False for exploitation
        """
        # Get current exploration temperature
        temperature = self.get_exploration_temperature(step, state)
        
        # High uncertainty -> explore regardless of temperature
        if state.uncertainty_bits > 2.0:
            return True
        
        # Check for optimization progress stagnation
        if hasattr(state, 'recent_target_improvements'):
            recent_improvements = getattr(state, 'recent_target_improvements', [])
            if isinstance(recent_improvements, list) and len(recent_improvements) > 5:
                try:
                    max_recent_improvement = max(recent_improvements[-5:])
                    if isinstance(max_recent_improvement, (int, float)) and max_recent_improvement < 0.01:
                        return True  # Explore if optimization has stagnated
                except (TypeError, ValueError):
                    pass  # Skip if improvements aren't numeric
        
        # Temperature-dependent exploration decision
        explore_prob = temperature / self.config.initial_temperature
        
        # Use deterministic threshold for reproducibility in testing
        # In practice, could use random sampling
        return explore_prob > 0.5
    
    def get_exploration_bonus_schedule(self, step: int) -> float:
        """Get exploration bonus scaling based on training progress.
        
        Args:
            step: Current training step
            
        Returns:
            Exploration bonus multiplier
        """
        # Calculate temperature directly without state-dependent adjustments
        # This is used for schedule-based bonus scaling, not state-aware decisions
        base_progress = min(step / self.config.adaptation_steps, 1.0)
        temperature = (
            self.config.initial_temperature * (1 - base_progress) + 
            self.config.final_temperature * base_progress
        )
        # Scale bonus with temperature
        return temperature / self.config.initial_temperature


def create_exploration_strategy(
    strategy_type: str = "uncertainty_guided",
    config: Optional[ExplorationConfig] = None,
    **kwargs
) -> Any:
    """Factory function for creating exploration strategies.
    
    Args:
        strategy_type: Type of exploration strategy ("uncertainty_guided", "adaptive")
        config: Optional configuration object
        **kwargs: Additional configuration parameters
        
    Returns:
        Exploration strategy instance
        
    Raises:
        ValueError: If strategy_type is not recognized
    """
    if config is None:
        # Create config from kwargs
        config = ExplorationConfig(**kwargs)
    
    if strategy_type == "uncertainty_guided":
        return UncertaintyGuidedExploration(config)
    elif strategy_type == "adaptive":
        return AdaptiveExploration(config)
    else:
        raise ValueError(f"Unknown exploration strategy: {strategy_type}")


def compute_exploration_value(
    exploration_strategy: Any,
    state: AcquisitionState,
    intervention: pyr.PMap,
    step: int = 0
) -> float:
    """Compute exploration value for an intervention using given strategy.
    
    Args:
        exploration_strategy: Exploration strategy instance
        state: Current acquisition state
        intervention: Candidate intervention
        step: Current training step (for adaptive strategies)
        
    Returns:
        Exploration value (higher = more valuable for exploration)
    """
    if hasattr(exploration_strategy, 'compute_exploration_bonus'):
        # UncertaintyGuidedExploration
        return exploration_strategy.compute_exploration_bonus(state, intervention)
    elif hasattr(exploration_strategy, 'get_exploration_bonus_schedule'):
        # AdaptiveExploration - combine with uncertainty-guided computation
        base_strategy = UncertaintyGuidedExploration(exploration_strategy.config)
        base_bonus = base_strategy.compute_exploration_bonus(state, intervention)
        schedule_multiplier = exploration_strategy.get_exploration_bonus_schedule(step)
        return base_bonus * schedule_multiplier
    else:
        raise ValueError(f"Unknown exploration strategy type: {type(exploration_strategy)}")


def select_exploration_intervention(
    candidates: list[pyr.PMap],
    exploration_strategy: Any,
    state: AcquisitionState,
    step: int = 0,
    top_k: int = 1
) -> list[pyr.PMap]:
    """Select top-k interventions for exploration from candidates.
    
    Args:
        candidates: List of candidate interventions
        exploration_strategy: Exploration strategy to use
        state: Current acquisition state
        step: Current training step
        top_k: Number of interventions to select
        
    Returns:
        List of top-k interventions ranked by exploration value
    """
    if not candidates:
        return []
    
    # Compute exploration values for all candidates
    exploration_values = []
    for intervention in candidates:
        value = compute_exploration_value(exploration_strategy, state, intervention, step)
        exploration_values.append((value, intervention))
    
    # Sort by exploration value (descending) and return top-k
    exploration_values.sort(key=lambda x: x[0], reverse=True)
    return [intervention for _, intervention in exploration_values[:top_k]]


def balance_exploration_exploitation(
    exploration_value: float,
    exploitation_value: float,
    exploration_strategy: Any,
    state: AcquisitionState,
    step: int = 0,
    alpha: float = 0.5
) -> float:
    """Balance exploration and exploitation values.
    
    Args:
        exploration_value: Value from exploration perspective
        exploitation_value: Value from exploitation perspective
        exploration_strategy: Exploration strategy for adaptive weighting
        state: Current acquisition state
        step: Current training step
        alpha: Base weighting between exploration (α) and exploitation (1-α)
        
    Returns:
        Combined value balancing both objectives
    """
    # Adaptive weighting based on exploration strategy
    if hasattr(exploration_strategy, 'should_explore'):
        should_explore = exploration_strategy.should_explore(state, step)
        if should_explore:
            # Increase exploration weight
            effective_alpha = min(1.0, alpha + 0.3)
        else:
            # Increase exploitation weight
            effective_alpha = max(0.0, alpha - 0.3)
    else:
        effective_alpha = alpha
    
    return effective_alpha * exploration_value + (1 - effective_alpha) * exploitation_value


# Export main classes and functions
__all__ = [
    'ExplorationConfig',
    'UncertaintyGuidedExploration', 
    'AdaptiveExploration',
    'create_exploration_strategy',
    'compute_exploration_value',
    'select_exploration_intervention',
    'balance_exploration_exploitation'
]