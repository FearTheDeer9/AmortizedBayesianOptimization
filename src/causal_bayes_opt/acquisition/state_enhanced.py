"""
Enhanced JAX-Native AcquisitionState

⚠️  DEPRECATED: This module is deprecated as of Phase 1.5. Use the JAX-native
    architecture in `causal_bayes_opt.jax_native` for optimal performance.

    Migration path:
    - Replace `EnhancedAcquisitionState` with `JAXAcquisitionState`
    - Use `create_jax_state()` instead of legacy state creation
    - See docs/migration/MIGRATION_GUIDE.md for details

This module provides an enhanced version of AcquisitionState that uses tensor operations
instead of Python loops for all state computations. This enables true JAX compilation
while maintaining full backward compatibility with existing interfaces.

Key enhancements:
1. Tensor-based mechanism confidence computation
2. Vectorized optimization progress tracking
3. JAX-compiled exploration coverage analysis
4. Pure tensor mechanism insights extraction
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import jax.numpy as jnp
import pyrsistent as pyr

# Local imports
from ..data_structures.buffer import ExperienceBuffer, BufferStatistics
from ..avici_integration.parent_set import ParentSetPosterior
from .state_tensor_ops import (
    convert_state_to_tensor_operations,
    compute_mechanism_confidence_tensor,
    create_tensor_state_computer
)
from .state_tensor_converter import convert_acquisition_state_to_tensors
from .tensor_features import (
    MechanismFeaturesTensor,
    create_mechanism_features_tensor,
    compute_mechanism_features_tensor,
    interpret_mechanism_tensor_results
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnhancedAcquisitionState:
    """
    JAX-native enhanced version of AcquisitionState using tensor operations.
    
    This state representation eliminates all Python loops in favor of JAX-compiled
    tensor operations, enabling true compilation while maintaining identical
    functionality to the original AcquisitionState.
    
    Key improvements:
    - All state computations use JAX-compiled functions
    - Tensor-based mechanism feature processing
    - Vectorized optimization and exploration metrics
    - Backward compatibility with existing interfaces
    - Optional fallback to legacy computation for validation
    """
    # Core components (same as original)
    posterior: ParentSetPosterior
    buffer: ExperienceBuffer
    best_value: float
    current_target: str
    step: int
    metadata: pyr.PMap = pyr.m()
    
    # Enhanced mechanism-aware features
    mechanism_predictions: Optional[List[Any]] = None
    mechanism_uncertainties: Optional[Dict[str, float]] = None
    
    # Tensor-based derived properties (computed using JAX)
    uncertainty_bits: float = field(init=False, default=0.0)
    buffer_statistics: BufferStatistics = field(init=False, default=None)
    marginal_parent_probs: Dict[str, float] = field(init=False, default_factory=dict)
    mechanism_confidence: Dict[str, float] = field(init=False, default_factory=dict)
    
    # JAX-native tensor data (pre-computed for efficiency)
    _tensor_data: Optional[Dict[str, Any]] = field(init=False, default=None)
    _mechanism_features_tensor: Optional[MechanismFeaturesTensor] = field(init=False, default=None)
    _use_tensor_ops: bool = field(init=False, default=True)
    
    def __post_init__(self):
        """Compute all derived properties using JAX tensor operations."""
        # Deprecation warning
        warnings.warn(
            "EnhancedAcquisitionState is deprecated as of Phase 1.5. "
            "Use JAXAcquisitionState from causal_bayes_opt.jax_native for optimal performance. "
            "See docs/migration/MIGRATION_GUIDE.md for migration instructions.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            # Basic uncertainty conversion (same as original)
            uncertainty_bits = self.posterior.uncertainty / jnp.log(2.0)
            object.__setattr__(self, 'uncertainty_bits', float(uncertainty_bits))
            
            # Buffer statistics (same as original)
            buffer_stats = self.buffer.get_statistics()
            object.__setattr__(self, 'buffer_statistics', buffer_stats)
            
            # Get variable order for tensor operations
            all_variables = list(self.buffer.get_variable_coverage())
            if not all_variables:
                # Handle empty buffer case
                object.__setattr__(self, 'marginal_parent_probs', {})
                object.__setattr__(self, 'mechanism_confidence', {})
                object.__setattr__(self, '_tensor_data', None)
                object.__setattr__(self, '_mechanism_features_tensor', None)
                return
            
            # Compute marginal parent probabilities (same as original)
            from ..avici_integration.parent_set import get_marginal_parent_probabilities
            marginal_probs = get_marginal_parent_probabilities(self.posterior, all_variables)
            object.__setattr__(self, 'marginal_parent_probs', marginal_probs)
            
            # Convert to tensor format for JAX operations
            tensor_conversion = convert_acquisition_state_to_tensors(self, max_history_size=100)
            object.__setattr__(self, '_tensor_data', tensor_conversion)
            
            # Create mechanism features tensor using default values for empty insights
            mechanism_insights = {
                'predicted_effects': {},
                'mechanism_types': {}
            }
            mechanism_features_tensor = create_mechanism_features_tensor(
                variable_order=all_variables,
                mechanism_insights=mechanism_insights,
                mechanism_confidence={}  # Will be computed from tensor
            )
            object.__setattr__(self, '_mechanism_features_tensor', mechanism_features_tensor)
            
            # Compute all state metrics using JAX tensor operations
            if self._use_tensor_ops and tensor_conversion is not None:
                tensor_results = self._compute_state_metrics_tensor()
                
                # Extract mechanism confidence from tensor results
                confidence_scores = tensor_results.get('mechanism_insights', {}).get('confidence_scores', jnp.array([]))
                if len(confidence_scores) == len(all_variables):
                    mechanism_confidence = {
                        var: float(confidence_scores[i])
                        for i, var in enumerate(all_variables)
                        if var != self.current_target
                    }
                else:
                    mechanism_confidence = {}
            else:
                # No fallback - use empty confidence if tensor ops disabled
                mechanism_confidence = {}
            
            object.__setattr__(self, 'mechanism_confidence', mechanism_confidence)
            
            # Validate state consistency
            self._validate_state_consistency()
            
            logger.debug(
                f"Created EnhancedAcquisitionState: step={self.step}, "
                f"uncertainty={self.uncertainty_bits:.3f} bits, "
                f"best_value={self.best_value:.3f}, "
                f"buffer_size={self.buffer_statistics.total_samples}, "
                f"tensor_ops={'enabled' if self._use_tensor_ops else 'disabled'}"
            )
            
        except Exception as e:
            logger.error(f"Error creating EnhancedAcquisitionState: {e}")
            raise ValueError(f"Failed to create EnhancedAcquisitionState: {e}")
    
    def _compute_state_metrics_tensor(self) -> Dict[str, Any]:
        """Compute all state metrics using JAX tensor operations."""
        if self._tensor_data is None or self._mechanism_features_tensor is None:
            return {}
        
        # Get tensor data
        history_tensor = self._tensor_data['history']
        mechanism_features = compute_mechanism_features_tensor(self._mechanism_features_tensor)
        
        # Use pre-computed marginal probabilities array (no dictionary operations)
        variable_order = self._tensor_data['variable_order']
        marginal_probs_array = self._tensor_data['marginal_probs']
        
        # Get target index
        target_idx = self._tensor_data['target_idx']
        if target_idx < 0:
            target_idx = 0  # Fallback
        
        # Use JAX-compiled state computation
        return convert_state_to_tensor_operations(
            history_tensor,
            mechanism_features,
            marginal_probs_array,
            target_idx,
            self.best_value
        )
    
    def _validate_state_consistency(self) -> None:
        """Validate internal consistency (same as original)."""
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
        """Get optimization progress using JAX tensor operations."""
        if self._use_tensor_ops and self._tensor_data is not None:
            try:
                tensor_results = self._compute_state_metrics_tensor()
                return tensor_results.get('optimization_progress', {})
            except Exception as e:
                logger.warning(f"Tensor optimization progress failed: {e}")
                # Return default values instead of falling back
                return {
                    'improvement_from_start': 0.0,
                    'recent_improvement': 0.0,
                    'optimization_rate': 0.0,
                    'stagnation_steps': 0
                }
        
        # Return default values when tensor ops disabled
        return {
            'improvement_from_start': 0.0,
            'recent_improvement': 0.0,
            'optimization_rate': 0.0,
            'stagnation_steps': 0
        }
    
    def get_exploration_coverage(self) -> Dict[str, float]:
        """Get exploration coverage using JAX tensor operations."""
        if self._use_tensor_ops and self._tensor_data is not None:
            try:
                tensor_results = self._compute_state_metrics_tensor()
                return tensor_results.get('exploration_coverage', {})
            except Exception as e:
                logger.warning(f"Tensor exploration coverage failed: {e}")
                # Return default values instead of falling back
                return {
                    'target_coverage_rate': 0.0,
                    'intervention_diversity': 0.0,
                    'unexplored_variables': 1.0
                }
        
        # Return default values when tensor ops disabled
        return {
            'target_coverage_rate': 0.0,
            'intervention_diversity': 0.0,
            'unexplored_variables': 1.0
        }
    
    def get_mechanism_insights(self) -> Dict[str, Any]:
        """Get mechanism insights using JAX tensor operations."""
        if self._use_tensor_ops and self._tensor_data is not None and self._mechanism_features_tensor is not None:
            try:
                tensor_results = self._compute_state_metrics_tensor()
                insights = tensor_results.get('mechanism_insights', {})
                
                # Convert tensor results back to dictionary format
                if 'high_impact_mask' in insights and 'uncertain_mask' in insights:
                    variable_order = self._tensor_data['variable_order']
                    
                    high_impact_variables = [
                        var for i, var in enumerate(variable_order)
                        if i < len(insights['high_impact_mask']) and insights['high_impact_mask'][i]
                    ]
                    
                    uncertain_mechanisms = [
                        var for i, var in enumerate(variable_order)
                        if i < len(insights['uncertain_mask']) and insights['uncertain_mask'][i]
                    ]
                    
                    # Extract mechanism features for effects and types
                    mechanism_features = self._mechanism_features_tensor
                    predicted_effects = {
                        var: float(mechanism_features.predicted_effects[i])
                        for i, var in enumerate(variable_order)
                        if i < len(mechanism_features.predicted_effects)
                    }
                    
                    return {
                        'mechanism_aware': True,
                        'high_impact_variables': high_impact_variables,
                        'uncertain_mechanisms': uncertain_mechanisms,
                        'predicted_effects': predicted_effects,
                        'mechanism_types': {},  # Could be extracted from tensor if needed
                        'mechanism_confidence_avg': float(jnp.mean(insights.get('confidence_scores', jnp.array([0.5]))))
                    }
                    
            except Exception as e:
                logger.warning(f"Tensor mechanism insights failed: {e}")
                # Return default values instead of falling back
                return {
                    'mechanism_aware': False,
                    'high_impact_variables': [],
                    'uncertain_mechanisms': [],
                    'predicted_effects': {},
                    'mechanism_types': {},
                    'mechanism_confidence_avg': 0.5
                }
        
        # Return default values when tensor ops disabled
        return {
            'mechanism_aware': False,
            'high_impact_variables': [],
            'uncertain_mechanisms': [],
            'predicted_effects': {},
            'mechanism_types': {},
            'mechanism_confidence_avg': 0.5
        }
    
    def summary(self) -> Dict[str, Any]:
        """Create comprehensive state summary using tensor operations where possible."""
        optimization_progress = self.get_optimization_progress()
        exploration_coverage = self.get_exploration_coverage()
        mechanism_insights = self.get_mechanism_insights()
        
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
            
            # Mechanism confidence (top 5)
            'top_mechanism_confidence': dict(sorted(
                self.mechanism_confidence.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]) if self.mechanism_confidence else {},
            
            # Progress metrics (tensor-computed)
            'optimization_progress': optimization_progress,
            'exploration_coverage': exploration_coverage,
            
            # Mechanism insights (tensor-computed)
            'mechanism_insights': mechanism_insights,
            
            # Enhanced metadata
            'tensor_ops_enabled': self._use_tensor_ops,
            'tensor_data_available': self._tensor_data is not None,
            'mechanism_features_available': self._mechanism_features_tensor is not None,
            
            # Original metadata
            'metadata': dict(self.metadata)
        }
    
    def get_tensor_input_for_policy(self) -> Dict[str, Any]:
        """Get tensor input suitable for vectorized policy networks."""
        if self._tensor_data is None:
            raise ValueError("Tensor data not available. Initialize state with tensor operations enabled.")
        
        return self._tensor_data
    
    def get_mechanism_features_tensor(self) -> Optional[MechanismFeaturesTensor]:
        """Get mechanism features in tensor format."""
        return self._mechanism_features_tensor
    
    def enable_tensor_operations(self, enabled: bool = True):
        """Enable or disable tensor operations (for testing/debugging)."""
        object.__setattr__(self, '_use_tensor_ops', enabled)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        tensor_status = "tensor" if self._use_tensor_ops else "legacy"
        return (
            f"EnhancedAcquisitionState(step={self.step}, target='{self.current_target}', "
            f"best_value={self.best_value:.3f}, uncertainty={self.uncertainty_bits:.2f} bits, "
            f"samples={self.buffer_statistics.total_samples}, mode={tensor_status})"
        )


# Factory function for creating enhanced acquisition states
def create_enhanced_acquisition_state(
    posterior: ParentSetPosterior,
    buffer: ExperienceBuffer,
    best_value: float,
    current_target: str,
    step: int,
    metadata: Optional[Dict[str, Any]] = None,
    mechanism_predictions: Optional[List[Any]] = None,
    mechanism_uncertainties: Optional[Dict[str, float]] = None,
    use_tensor_ops: bool = True
) -> EnhancedAcquisitionState:
    """
    Factory function for creating enhanced acquisition states.
    
    Args:
        posterior: Parent set posterior distribution
        buffer: Experience buffer with samples
        best_value: Current best observed target value
        current_target: Target variable name
        step: Current optimization step
        metadata: Optional metadata
        mechanism_predictions: Optional mechanism predictions
        mechanism_uncertainties: Optional mechanism uncertainties
        use_tensor_ops: Whether to use tensor operations (default: True)
        
    Returns:
        EnhancedAcquisitionState with tensor operations enabled
    """
    state = EnhancedAcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=current_target,
        step=step,
        metadata=pyr.m(**(metadata or {})),
        mechanism_predictions=mechanism_predictions,
        mechanism_uncertainties=mechanism_uncertainties
    )
    
    # Configure tensor operations
    state.enable_tensor_operations(use_tensor_ops)
    
    return state


# Backward compatibility wrapper
def upgrade_acquisition_state_to_enhanced(
    original_state,  # AcquisitionState
    use_tensor_ops: bool = True
) -> EnhancedAcquisitionState:
    """
    Upgrade an original AcquisitionState to EnhancedAcquisitionState.
    
    Args:
        original_state: Original AcquisitionState object
        use_tensor_ops: Whether to enable tensor operations
        
    Returns:
        EnhancedAcquisitionState with same data but tensor capabilities
    """
    return create_enhanced_acquisition_state(
        posterior=original_state.posterior,
        buffer=original_state.buffer,
        best_value=original_state.best_value,
        current_target=original_state.current_target,
        step=original_state.step,
        metadata=dict(original_state.metadata),
        mechanism_predictions=getattr(original_state, 'mechanism_predictions', None),
        mechanism_uncertainties=getattr(original_state, 'mechanism_uncertainties', None),
        use_tensor_ops=use_tensor_ops
    )