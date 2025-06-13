"""
State representation for RL-based acquisition in ACBO.

This module provides immutable state representation that combines:
- Structural uncertainty (from ParentSetPosterior) 
- Optimization progress (from ExperienceBuffer)
- Decision-making context for policy networks

The state serves as the primary input to the policy network for intervention selection.

This module contains ONLY the core data structure and pure methods with no external 
dependencies beyond immediate neighbors. Factory functions and integration logic 
are in the services module.
"""

# Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Local imports (only immediate neighbors)
from ..data_structures.buffer import ExperienceBuffer, BufferStatistics
from ..avici_integration.parent_set import ParentSetPosterior

# Type aliases
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
                from ..avici_integration.parent_set import get_marginal_parent_probabilities
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
            # Access sample values - must use functional interface to avoid imports
            values = sample.get('values', pyr.m())
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
            # Get intervention targets from outcome sample - use direct access to avoid imports
            targets = outcome.get('intervention_targets', frozenset())
            
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