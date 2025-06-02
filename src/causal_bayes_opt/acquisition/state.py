"""
State representation for RL-based acquisition in ACBO.

This module provides immutable state representation that combines:
- Structural uncertainty (from ParentSetPosterior) 
- Optimization progress (from ExperienceBuffer)
- Decision-making context for policy networks

The state serves as the primary input to the policy network for intervention selection.
"""

# Standard library imports
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports 
from ..data_structures.buffer import ExperienceBuffer, BufferStatistics
from ..avici_integration.parent_set import (
    ParentSetPosterior, 
    get_marginal_parent_probabilities,
    compute_posterior_entropy
)

# Type aliases
InterventionSpec = pyr.PMap  # From interventions framework
Sample = pyr.PMap  # From data_structures framework

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AcquisitionState:
    """
    Immutable state representation for RL-based acquisition.
    
    Combines structural uncertainty (from ParentSetPosterior) with 
    optimization progress (from ExperienceBuffer) for decision making.
    
    This state serves as input to policy networks and reward computation,
    providing all necessary context for intelligent intervention selection.
    
    Attributes:
        posterior: Current posterior distribution over parent sets
        buffer: Experience buffer containing all observational/interventional data
        best_value: Current best observed value of the target variable
        current_target: Name of the target variable being optimized
        step: Current step number in the acquisition process
        metadata: Additional context information
        
        # Derived properties (computed at creation for efficiency)
        uncertainty_bits: Posterior uncertainty in bits (derived)
        buffer_statistics: Comprehensive buffer statistics (derived)
        marginal_parent_probs: Marginal parent probabilities for all variables (derived)
    """
    # Core components from Phases 1 & 2
    posterior: ParentSetPosterior
    buffer: ExperienceBuffer
    best_value: float
    current_target: str
    
    # Context information
    step: int
    metadata: pyr.PMap[str, Any] = pyr.m()
    
    # Derived properties (computed once for efficiency)
    uncertainty_bits: float = field(init=False)
    buffer_statistics: BufferStatistics = field(init=False)
    marginal_parent_probs: Dict[str, float] = field(init=False)
    
    def __post_init__(self):
        """Compute derived properties for efficient access during training."""
        try:
            # Convert uncertainty from nats to bits for interpretability
            uncertainty_bits = self.posterior.uncertainty / jnp.log(2.0)
            object.__setattr__(self, 'uncertainty_bits', float(uncertainty_bits))
            
            # Get comprehensive buffer statistics
            buffer_stats = self.buffer.get_statistics()
            object.__setattr__(self, 'buffer_statistics', buffer_stats)
            
            # Compute marginal parent probabilities for all variables
            all_variables = list(self.buffer.get_variable_coverage())
            if all_variables:
                marginal_probs = get_marginal_parent_probabilities(self.posterior, all_variables)
            else:
                marginal_probs = {}
            object.__setattr__(self, 'marginal_parent_probs', marginal_probs)
            
            # Validate state consistency
            self._validate_state_consistency()
            
            logger.debug(
                f"Created AcquisitionState: step={self.step}, "
                f"uncertainty={self.uncertainty_bits:.3f} bits, "
                f"best_value={self.best_value:.3f}, "
                f"buffer_size={self.buffer_statistics.total_samples}"
            )
            
        except Exception as e:
            logger.error(f"Error creating AcquisitionState: {e}")
            raise ValueError(f"Failed to create AcquisitionState: {e}")
    
    def _validate_state_consistency(self) -> None:
        """Validate internal consistency of the acquisition state."""
        # Check target variable consistency
        if self.current_target != self.posterior.target_variable:
            raise ValueError(
                f"Target mismatch: state target '{self.current_target}' != "
                f"posterior target '{self.posterior.target_variable}'"
            )
        
        # Check that target is in buffer variables
        buffer_vars = self.buffer.get_variable_coverage()
        if self.current_target not in buffer_vars:
            raise ValueError(
                f"Target variable '{self.current_target}' not found in buffer variables: {buffer_vars}"
            )
        
        # Check step is non-negative
        if self.step < 0:
            raise ValueError(f"Step must be non-negative, got {self.step}")
        
        # Check that best_value is finite
        if not jnp.isfinite(self.best_value):
            raise ValueError(f"Best value must be finite, got {self.best_value}")
    
    def get_optimization_progress(self) -> Dict[str, float]:
        """
        Get optimization progress metrics derived from buffer history.
        
        Returns:
            Dictionary with optimization progress information
        """
        # Get all target values from buffer
        all_samples = self.buffer.get_all_samples()
        if not all_samples:
            return {
                'improvement_from_start': 0.0,
                'recent_improvement': 0.0,
                'optimization_rate': 0.0,
                'stagnation_steps': 0.0
            }
        
        target_values = []
        for sample in all_samples:
            # Access sample values using established API
            from ..data_structures.sample import get_values
            values = get_values(sample)
            if self.current_target in values:
                target_values.append(float(values[self.current_target]))
        
        if not target_values:
            return {
                'improvement_from_start': 0.0,
                'recent_improvement': 0.0,
                'optimization_rate': 0.0,
                'stagnation_steps': 0.0
            }
        
        # Compute progress metrics
        initial_value = target_values[0]
        improvement_from_start = self.best_value - initial_value
        
        # Recent improvement (last 10 samples vs last 20)
        recent_improvement = 0.0
        if len(target_values) >= 20:
            recent_10 = jnp.mean(jnp.array(target_values[-10:]))
            prev_10 = jnp.mean(jnp.array(target_values[-20:-10]))
            recent_improvement = float(recent_10 - prev_10)
        
        # Optimization rate (improvement per sample)
        optimization_rate = improvement_from_start / len(target_values) if target_values else 0.0
        
        # Stagnation (steps since last improvement)
        stagnation_steps = 0
        for i in range(len(target_values) - 1, -1, -1):
            if target_values[i] < self.best_value - 1e-6:  # Small tolerance
                break
            stagnation_steps += 1
        
        return {
            'improvement_from_start': float(improvement_from_start),
            'recent_improvement': float(recent_improvement),
            'optimization_rate': float(optimization_rate),
            'stagnation_steps': float(stagnation_steps)
        }
    
    def get_exploration_coverage(self) -> Dict[str, float]:
        """
        Get exploration coverage metrics from intervention history.
        
        Returns:
            Dictionary with exploration coverage information
        """
        all_variables = list(self.buffer.get_variable_coverage())
        intervention_targets = self.buffer.get_intervention_targets_coverage()
        
        if not all_variables:
            return {
                'target_coverage_rate': 0.0,
                'intervention_diversity': 0.0,
                'unexplored_variables': 1.0
            }
        
        # Exclude target variable from potential intervention targets
        potential_targets = [v for v in all_variables if v != self.current_target]
        
        if not potential_targets:
            return {
                'target_coverage_rate': 1.0,  # All variables explored (just target)
                'intervention_diversity': 0.0,
                'unexplored_variables': 0.0
            }
        
        # Coverage rate (fraction of variables that have been intervention targets)
        explored_count = len([v for v in potential_targets if v in intervention_targets])
        target_coverage_rate = explored_count / len(potential_targets)
        
        # Intervention diversity (entropy of intervention target distribution)
        intervention_counts = {}
        for intervention_spec, outcome in self.buffer.get_interventions():
            # Get intervention targets from outcome sample
            from ..data_structures.sample import get_intervention_targets
            targets = get_intervention_targets(outcome)
            
            # Count interventions by target variable combinations
            target_key = frozenset(targets)
            intervention_counts[target_key] = intervention_counts.get(target_key, 0) + 1
        
        # Compute intervention diversity as normalized entropy
        if intervention_counts:
            total_interventions = sum(intervention_counts.values())
            probs = [count / total_interventions for count in intervention_counts.values()]
            entropy = -sum(p * jnp.log(p + 1e-12) for p in probs)
            max_entropy = jnp.log(len(potential_targets)) if potential_targets else 0
            intervention_diversity = float(entropy / max(max_entropy, 1e-12))
        else:
            intervention_diversity = 0.0
        
        unexplored_variables = 1.0 - target_coverage_rate
        
        return {
            'target_coverage_rate': float(target_coverage_rate),
            'intervention_diversity': float(intervention_diversity),
            'unexplored_variables': float(unexplored_variables)
        }
    
    def summary(self) -> Dict[str, Any]:
        """
        Create human-readable summary of the acquisition state.
        
        Returns:
            Dictionary with comprehensive state information
        """
        optimization_progress = self.get_optimization_progress()
        exploration_coverage = self.get_exploration_coverage()
        
        return {
            # Core state information
            'step': self.step,
            'target_variable': self.current_target,
            'best_value': self.best_value,
            
            # Uncertainty information
            'uncertainty_bits': self.uncertainty_bits,
            'uncertainty_nats': self.posterior.uncertainty,
            'posterior_concentration': 1.0 - jnp.exp(-self.posterior.uncertainty),
            
            # Buffer information
            'total_samples': self.buffer_statistics.total_samples,
            'observational_samples': self.buffer_statistics.num_observations,
            'interventional_samples': self.buffer_statistics.num_interventions,
            'unique_variables': self.buffer_statistics.unique_variables,
            
            # Most likely parents
            'most_likely_parents': list(self.posterior.top_k_sets[0][0]) if self.posterior.top_k_sets else [],
            'most_likely_probability': self.posterior.top_k_sets[0][1] if self.posterior.top_k_sets else 0.0,
            
            # Marginal probabilities for key variables (top 5)
            'top_marginal_parent_probs': dict(sorted(
                self.marginal_parent_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]),
            
            # Progress metrics
            'optimization_progress': optimization_progress,
            'exploration_coverage': exploration_coverage,
            
            # Metadata
            'metadata': dict(self.metadata)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AcquisitionState(step={self.step}, target='{self.current_target}', "
            f"best_value={self.best_value:.3f}, uncertainty={self.uncertainty_bits:.2f} bits, "
            f"samples={self.buffer_statistics.total_samples})"
        )


# Factory functions for creating and updating states
def create_acquisition_state(
    scm: pyr.PMap,
    buffer: ExperienceBuffer,
    surrogate_model: Any,
    surrogate_params: Any,
    target_variable: str,
    step: int = 0,
    metadata: Optional[Dict[str, Any]] = None
) -> AcquisitionState:
    """
    Create acquisition state from current buffer and surrogate model predictions.
    
    This factory function integrates the buffer data with surrogate model predictions
    to create a complete state representation for the acquisition policy.
    
    Args:
        scm: Structural causal model (for context)
        buffer: Current experience buffer with all data
        surrogate_model: Trained surrogate model for posterior prediction
        surrogate_params: Current parameters of the surrogate model
        target_variable: Name of the optimization target variable
        step: Current step number in acquisition process
        metadata: Optional additional context information
        
    Returns:
        Complete AcquisitionState ready for policy input
        
    Raises:
        ValueError: If inputs are inconsistent or insufficient data
        
    Example:
        >>> state = create_acquisition_state(
        ...     scm, buffer, surrogate_model, params, target='Y', step=42
        ... )
        >>> print(f"Uncertainty: {state.uncertainty_bits:.2f} bits")
    """
    # Validate inputs
    if not target_variable:
        raise ValueError("Target variable cannot be empty")
    
    if step < 0:
        raise ValueError("Step must be non-negative")
    
    if buffer.size() == 0:
        raise ValueError("Buffer must contain at least one sample")
    
    # Check target variable is in buffer
    buffer_vars = buffer.get_variable_coverage()
    if target_variable not in buffer_vars:
        raise ValueError(f"Target '{target_variable}' not in buffer variables: {buffer_vars}")
    
    # Get current best value for target variable
    all_samples = buffer.get_all_samples()
    target_values = []
    
    for sample in all_samples:
        from ..data_structures.sample import get_values
        values = get_values(sample)
        if target_variable in values:
            target_values.append(float(values[target_variable]))
    
    if not target_values:
        raise ValueError(f"No samples contain target variable '{target_variable}'")
    
    best_value = float(jnp.max(jnp.array(target_values)))
    
    # Get posterior prediction from surrogate model
    # This requires integration with Phase 2 AVICI components
    try:
        # Import prediction function from Phase 2
        from ..avici_integration.parent_set import predict_parent_posterior
        from ..avici_integration.core import create_training_batch
        
        # Create prediction batch from buffer
        batch = create_training_batch(scm, all_samples, target_variable)
        
        # Predict posterior using surrogate model
        posterior = predict_parent_posterior(
            surrogate_model, surrogate_params,
            batch['x'], batch['variable_order'], target_variable
        )
        
    except ImportError as e:
        logger.error(f"Failed to import Phase 2 components: {e}")
        # Fallback: create a uniform posterior for testing
        logger.warning("Using uniform posterior fallback for testing")
        from ..avici_integration.parent_set import create_parent_set_posterior
        
        # Create a simple uniform posterior over empty set and single parents
        parent_candidates = [v for v in buffer_vars if v != target_variable]
        parent_sets = [frozenset()]  # Empty parent set
        parent_sets.extend([frozenset([var]) for var in parent_candidates[:3]])  # Top 3 single parents
        
        n_sets = len(parent_sets)
        uniform_probs = jnp.ones(n_sets) / n_sets
        
        posterior = create_parent_set_posterior(
            target_variable=target_variable,
            parent_sets=parent_sets,
            probabilities=uniform_probs,
            metadata={'fallback_uniform': True, 'step': step}
        )
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    state_metadata = pyr.pmap({
        **metadata,
        'creation_time': time.time(),
        'scm_variables': set(scm.get('variables', [])) if scm else set(),
        'buffer_creation_time': buffer.get_statistics().creation_time,
        'surrogate_model_type': type(surrogate_model).__name__ if surrogate_model else 'None'
    })
    
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=target_variable,
        step=step,
        metadata=state_metadata
    )


def update_state_with_intervention(
    current_state: AcquisitionState,
    intervention: InterventionSpec,
    outcome: Sample,
    new_posterior: ParentSetPosterior
) -> AcquisitionState:
    """
    Create new state after applying an intervention and observing outcome.
    
    This function efficiently updates the acquisition state with new information
    while maintaining immutability principles.
    
    Args:
        current_state: Current acquisition state
        intervention: Applied intervention specification
        outcome: Observed outcome sample from the intervention
        new_posterior: Updated posterior after incorporating new data
        
    Returns:
        New AcquisitionState with updated information
        
    Raises:
        ValueError: If intervention/outcome are inconsistent with current state
        
    Example:
        >>> new_state = update_state_with_intervention(
        ...     state, intervention, outcome, updated_posterior
        ... )
        >>> print(f"Step {new_state.step}: uncertainty reduced by "
        ...       f"{state.uncertainty_bits - new_state.uncertainty_bits:.2f} bits")
    """
    # Validate inputs
    if new_posterior.target_variable != current_state.current_target:
        raise ValueError(
            f"Posterior target '{new_posterior.target_variable}' doesn't match "
            f"state target '{current_state.current_target}'"
        )
    
    # Create updated buffer (this creates a copy for safety)
    new_buffer = ExperienceBuffer()
    
    # Copy existing data
    for obs in current_state.buffer.get_observations():
        new_buffer.add_observation(obs)
    
    for int_spec, int_outcome in current_state.buffer.get_interventions():
        new_buffer.add_intervention(int_spec, int_outcome)
    
    # Add new intervention-outcome pair
    new_buffer.add_intervention(intervention, outcome)
    
    # Update best value
    from ..data_structures.sample import get_values
    outcome_values = get_values(outcome)
    current_target_value = outcome_values.get(current_state.current_target)
    
    if current_target_value is not None:
        new_best_value = max(current_state.best_value, float(current_target_value))
    else:
        new_best_value = current_state.best_value
        logger.warning(f"Outcome doesn't contain target variable '{current_state.current_target}'")
    
    # Update metadata
    new_metadata = current_state.metadata.set('last_intervention_step', current_state.step)
    new_metadata = new_metadata.set('last_update_time', time.time())
    
    # Add intervention analysis to metadata
    from ..interventions import validate_intervention_spec
    try:
        intervention_analysis = {
            'intervention_type': intervention.get('type', 'unknown'),
            'intervention_targets': list(intervention.get('targets', set())),
            'outcome_target_value': current_target_value
        }
        new_metadata = new_metadata.set('last_intervention_analysis', intervention_analysis)
    except Exception as e:
        logger.warning(f"Could not analyze intervention: {e}")
    
    # Create new state
    return AcquisitionState(
        posterior=new_posterior,
        buffer=new_buffer,
        best_value=new_best_value,
        current_target=current_state.current_target,
        step=current_state.step + 1,  # Increment step
        metadata=new_metadata
    )


# Utility functions for accessing state information
def get_state_uncertainty_bits(state: AcquisitionState) -> float:
    """Get uncertainty in bits (for interpretability)."""
    return state.uncertainty_bits


def get_state_optimization_progress(state: AcquisitionState) -> Dict[str, float]:
    """Get optimization progress metrics."""
    return state.get_optimization_progress()


def get_state_marginal_probabilities(state: AcquisitionState) -> Dict[str, float]:
    """Get marginal parent probabilities for all variables."""
    return state.marginal_parent_probs
