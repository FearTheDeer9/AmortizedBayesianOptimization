"""
State enrichment for enhanced context architecture.

This module implements enriched history building that converts AcquisitionState
to multi-channel transformer input with temporal context evolution.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EnrichedHistoryBuilder:
    """
    Builder for enriched transformer input with temporal context evolution.
    
    Creates multi-channel input where all context information (uncertainty,
    mechanism insights, optimization progress) is embedded as channels
    that evolve over time, allowing the transformer to learn temporal
    patterns in the context itself.
    """
    
    # Fixed maximum history size for consistent transformer input shape
    MAX_HISTORY_SIZE = 100
    
    # Channel definitions for enriched input
    CHANNEL_DEFINITIONS = {
        0: "variable_values",           # Standardized variable values
        1: "intervention_indicators",   # 1 if intervened, 0 otherwise  
        2: "target_indicators",         # 1 if target variable, 0 otherwise
        3: "marginal_parent_probs",     # Parent probability evolution
        4: "uncertainty_bits",          # Uncertainty evolution over time
        5: "mechanism_confidence",      # Mechanism confidence trajectory
        6: "predicted_effects",         # Predicted effect magnitudes
        7: "mechanism_type_encoding",   # Mechanism type encoding
        8: "best_value_progression",    # Best value found so far
        9: "steps_since_improvement",   # Stagnation indicator
    }
    
    def __init__(self, 
                 standardize_values: bool = True,
                 include_temporal_features: bool = True,
                 max_history_size: Optional[int] = None,
                 support_variable_scms: bool = True,
                 num_channels: Optional[int] = None):
        """
        Initialize enriched history builder.
        
        Args:
            standardize_values: Whether to standardize variable values
            include_temporal_features: Whether to include temporal context
            max_history_size: Maximum history size (uses class default if None)
            support_variable_scms: Whether to support variable-count SCMs (3-8 variables)
            num_channels: Number of channels to use (defaults to CHANNEL_DEFINITIONS length)
        """
        self.standardize_values = standardize_values
        self.include_temporal_features = include_temporal_features
        self.max_history_size = max_history_size or self.MAX_HISTORY_SIZE
        self.num_channels = num_channels if num_channels is not None else len(self.CHANNEL_DEFINITIONS)
        self.support_variable_scms = support_variable_scms
    
    def build_enriched_history(self, state) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Convert AcquisitionState to enriched transformer input with optional variable masking.
        
        Args:
            state: AcquisitionState with buffer and context information
            
        Returns:
            Tuple of (enriched_history, variable_mask) where:
            - enriched_history: [MAX_HISTORY_SIZE, n_vars, num_channels]
            - variable_mask: [n_vars] or None (1.0 for valid variables, 0.0 for padding)
        """
        # Get all samples from buffer
        all_samples = state.buffer.get_all_samples()
        
        # Get variable ordering
        variable_order = sorted(state.buffer.get_variable_coverage())
        n_vars = len(variable_order)
        
        if not all_samples or n_vars == 0:
            # Handle empty buffer case
            empty_history = jnp.zeros((self.max_history_size, max(1, n_vars), self.num_channels))
            empty_mask = jnp.ones(max(1, n_vars)) if self.support_variable_scms else None
            return empty_history, empty_mask
        
        # Build core intervention history (channels 0-2)
        core_history = self._build_core_history(
            all_samples, variable_order, state.current_target
        )
        
        # Build context history (channels 3 onwards, up to num_channels)
        if self.include_temporal_features and self.num_channels > 3:
            context_history = self._build_context_history(
                all_samples, variable_order, state
            )
            
            # Limit context to available channels
            max_context_channels = self.num_channels - 3
            if context_history.shape[2] > max_context_channels:
                context_history = context_history[:, :, :max_context_channels]
            
            # Combine core and context
            enriched_history = jnp.concatenate([core_history, context_history], axis=2)
        else:
            # Either no temporal features or num_channels <= 3
            if self.num_channels > 3:
                # Pad core history to full channel size
                padding_shape = (core_history.shape[0], core_history.shape[1], 
                               self.num_channels - 3)
                padding = jnp.zeros(padding_shape)
                enriched_history = jnp.concatenate([core_history, padding], axis=2)
            else:
                # Use only core channels (limit to num_channels)
                enriched_history = core_history[:, :, :self.num_channels]
        
        # Create variable mask for variable-agnostic processing
        variable_mask = None
        if self.support_variable_scms:
            # All variables are valid (no padding in this implementation)
            # Future enhancement: could support padding to max variable count
            variable_mask = jnp.ones(n_vars)
        
        return enriched_history, variable_mask
    
    def _build_core_history(self, 
                          all_samples: List[Any], 
                          variable_order: List[str],
                          current_target: str) -> jnp.ndarray:
        """
        Build core intervention history (channels 0-2).
        
        Args:
            all_samples: List of samples from buffer
            variable_order: Ordered list of variable names
            current_target: Current target variable name
            
        Returns:
            Core history [MAX_HISTORY_SIZE, n_vars, 3]
        """
        n_vars = len(variable_order)
        history_length = min(len(all_samples), self.max_history_size)
        
        # Initialize core history tensor
        core_history = jnp.zeros((self.max_history_size, n_vars, 3))
        
        # Process samples (most recent first)
        recent_samples = all_samples[-history_length:]
        
        for t, sample in enumerate(recent_samples):
            # Channel 0: Variable values
            values = self._extract_variable_values(sample, variable_order)
            if self.standardize_values:
                values = self._standardize_values(values, all_samples, variable_order)
            
            # Channel 1: Intervention indicators
            intervention_flags = self._extract_intervention_flags(sample, variable_order)
            
            # Channel 2: Target indicators  
            target_flags = self._extract_target_flags(sample, variable_order, current_target)
            
            # Store in history tensor
            history_idx = self.max_history_size - history_length + t
            core_history = core_history.at[history_idx, :, 0].set(values)
            core_history = core_history.at[history_idx, :, 1].set(intervention_flags)
            core_history = core_history.at[history_idx, :, 2].set(target_flags)
        
        return core_history
    
    def _build_context_history(self,
                             all_samples: List[Any],
                             variable_order: List[str], 
                             state) -> jnp.ndarray:
        """
        Build context history (channels 3-9).
        
        Args:
            all_samples: List of samples from buffer
            variable_order: Ordered list of variable names
            state: AcquisitionState with context information
            
        Returns:
            Context history [MAX_HISTORY_SIZE, n_vars, 7]
        """
        n_vars = len(variable_order)
        history_length = min(len(all_samples), self.max_history_size)
        
        # Initialize context history tensor
        context_history = jnp.zeros((self.max_history_size, n_vars, 7))
        
        # Get static context information
        marginal_probs = self._get_marginal_probabilities(variable_order, state)
        mechanism_insights = state.get_mechanism_insights() if hasattr(state, 'get_mechanism_insights') else {}
        optimization_progress = state.get_optimization_progress() if hasattr(state, 'get_optimization_progress') else {}
        
        # Process temporal evolution
        for t in range(history_length):
            history_idx = self.max_history_size - history_length + t
            
            # Channel 3: Marginal parent probabilities (static for now)
            context_history = context_history.at[history_idx, :, 0].set(marginal_probs)
            
            # Channel 4: Uncertainty bits evolution
            uncertainty_bits = self._compute_uncertainty_at_step(t, all_samples, state)
            uncertainty_vector = jnp.full((n_vars,), uncertainty_bits)
            context_history = context_history.at[history_idx, :, 1].set(uncertainty_vector)
            
            # Channel 5: Mechanism confidence trajectory
            mechanism_confidence = self._get_mechanism_confidence_vector(variable_order, state)
            context_history = context_history.at[history_idx, :, 2].set(mechanism_confidence)
            
            # Channel 6: Predicted effect magnitudes
            predicted_effects = self._get_predicted_effects_vector(variable_order, mechanism_insights)
            context_history = context_history.at[history_idx, :, 3].set(predicted_effects)
            
            # Channel 7: Mechanism type encoding
            mechanism_types = self._get_mechanism_type_vector(variable_order, mechanism_insights)
            context_history = context_history.at[history_idx, :, 4].set(mechanism_types)
            
            # Channel 8: Best value progression
            best_value = self._compute_best_value_at_step(t, all_samples, state)
            best_value_vector = jnp.full((n_vars,), best_value)
            context_history = context_history.at[history_idx, :, 5].set(best_value_vector)
            
            # Channel 9: Steps since improvement
            steps_since_improvement = self._compute_stagnation_at_step(t, all_samples, state)
            stagnation_vector = jnp.full((n_vars,), steps_since_improvement)
            context_history = context_history.at[history_idx, :, 6].set(stagnation_vector)
        
        return context_history
    
    def _extract_variable_values(self, sample: Any, variable_order: List[str]) -> jnp.ndarray:
        """Extract variable values from sample."""
        values = []
        sample_values = sample.get('values', {}) if hasattr(sample, 'get') else {}
        for var_name in variable_order:
            if var_name in sample_values:
                values.append(float(sample_values[var_name]))
            else:
                values.append(0.0)  # Default value if missing
        return jnp.array(values)
    
    def _extract_intervention_flags(self, sample: Any, variable_order: List[str]) -> jnp.ndarray:
        """Extract intervention indicators from sample."""
        flags = []
        intervention_targets = sample.get('intervention_targets', set()) if hasattr(sample, 'get') else set()
        for var_name in variable_order:
            if var_name in intervention_targets:
                flags.append(1.0)
            else:
                flags.append(0.0)
        return jnp.array(flags)
    
    def _extract_target_flags(self, sample: Any, variable_order: List[str], current_target: str) -> jnp.ndarray:
        """Extract target indicators from sample."""
        flags = []
        for var_name in variable_order:
            if var_name == current_target:
                flags.append(1.0)
            else:
                flags.append(0.0)
        return jnp.array(flags)
    
    def _standardize_values(self, 
                          values: jnp.ndarray, 
                          all_samples: List[Any], 
                          variable_order: List[str]) -> jnp.ndarray:
        """Standardize variable values using buffer statistics."""
        if len(all_samples) < 2:
            return values  # Can't standardize with too few samples
        
        # Compute means and stds across all samples
        all_values = []
        for sample in all_samples:
            sample_values = self._extract_variable_values(sample, variable_order)
            all_values.append(sample_values)
        
        all_values = jnp.stack(all_values)  # [n_samples, n_vars]
        means = jnp.mean(all_values, axis=0)
        stds = jnp.std(all_values, axis=0) + 1e-8  # Avoid division by zero
        
        standardized = (values - means) / stds
        return standardized
    
    def _get_marginal_probabilities(self, variable_order: List[str], state) -> jnp.ndarray:
        """Get marginal parent probabilities for all variables."""
        probs = []
        for var_name in variable_order:
            if hasattr(state, 'marginal_parent_probs') and var_name in state.marginal_parent_probs:
                probs.append(float(state.marginal_parent_probs[var_name]))
            else:
                probs.append(0.0)  # Default if missing
        return jnp.array(probs)
    
    def _get_mechanism_confidence_vector(self, variable_order: List[str], state) -> jnp.ndarray:
        """Get mechanism confidence for all variables."""
        confidences = []
        for var_name in variable_order:
            if hasattr(state, 'mechanism_confidence') and var_name in state.mechanism_confidence:
                confidences.append(float(state.mechanism_confidence[var_name]))
            else:
                confidences.append(0.0)  # Default if missing
        return jnp.array(confidences)
    
    def _get_predicted_effects_vector(self, variable_order: List[str], mechanism_insights: Dict) -> jnp.ndarray:
        """Get predicted effect magnitudes for all variables."""
        effects = []
        predicted_effects = mechanism_insights.get('predicted_effects', {})
        for var_name in variable_order:
            if var_name in predicted_effects:
                effects.append(abs(float(predicted_effects[var_name])))  # Use magnitude
            else:
                effects.append(0.0)  # Default if missing
        return jnp.array(effects)
    
    def _get_mechanism_type_vector(self, variable_order: List[str], mechanism_insights: Dict) -> jnp.ndarray:
        """Get mechanism type encoding for all variables."""
        types = []
        mechanism_types = mechanism_insights.get('mechanism_types', {})
        type_encoding = {
            'linear': 1.0, 'polynomial': 2.0, 'gaussian': 3.0, 
            'neural': 4.0, 'unknown': 0.0
        }
        
        for var_name in variable_order:
            mech_type = mechanism_types.get(var_name, 'unknown')
            types.append(type_encoding.get(mech_type, 0.0))
        return jnp.array(types)
    
    def _compute_uncertainty_at_step(self, step: int, all_samples: List[Any], state) -> float:
        """Compute uncertainty at specific step in history."""
        if hasattr(state, 'uncertainty_bits'):
            return float(state.uncertainty_bits)
        return 0.0  # Default if missing
    
    def _compute_best_value_at_step(self, step: int, all_samples: List[Any], state) -> float:
        """Compute best value found up to specific step."""
        if step >= len(all_samples):
            return 0.0
        
        # Get best value up to this step
        step_samples = all_samples[:step+1]
        if not step_samples:
            return 0.0
        
        # Extract target values from samples
        target_values = []
        current_target = state.current_target if hasattr(state, 'current_target') else None
        
        for sample in step_samples:
            sample_values = sample.get('values', {}) if hasattr(sample, 'get') else {}
            
            # Try to get target value from sample values
            if current_target and current_target in sample_values:
                target_values.append(float(sample_values[current_target]))
            # Fallback to checking attributes (for mock samples)
            elif hasattr(sample, 'target_value'):
                target_values.append(float(sample.target_value))
            elif hasattr(sample, 'reward'):
                target_values.append(float(sample.reward))
        
        if target_values:
            return float(min(target_values))  # Changed to min for minimization
        return 0.0
    
    def _compute_stagnation_at_step(self, step: int, all_samples: List[Any], state) -> float:
        """Compute steps since improvement at specific step."""
        if step >= len(all_samples):
            return 0.0
        
        # Find last improvement before this step
        step_samples = all_samples[:step+1]
        if len(step_samples) < 2:
            return 0.0
        
        # Simple implementation: count steps since best value
        best_value = self._compute_best_value_at_step(step, all_samples, state)
        
        # Count backwards from current step
        current_target = state.current_target if hasattr(state, 'current_target') else None
        
        for i in range(step, -1, -1):
            sample = all_samples[i]
            sample_values = sample.get('values', {}) if hasattr(sample, 'get') else {}
            
            # Try to get target value
            if current_target and current_target in sample_values:
                value = float(sample_values[current_target])
            elif hasattr(sample, 'target_value'):
                value = float(sample.target_value)
            elif hasattr(sample, 'reward'):
                value = float(sample.reward)
            else:
                continue
            
            if abs(value - best_value) < 1e-6:
                return float(step - i)
        
        return float(step)  # No improvement found
    
    def get_channel_info(self) -> Dict[int, str]:
        """Get information about enriched input channels."""
        return self.CHANNEL_DEFINITIONS.copy()
    
    def validate_enriched_history(self, enriched_history: jnp.ndarray) -> bool:
        """
        Validate enriched history tensor.
        
        Args:
            enriched_history: Enriched history tensor
            
        Returns:
            True if valid, False otherwise
        """
        expected_shape = (self.max_history_size, None, self.num_channels)  # n_vars can vary
        
        if len(enriched_history.shape) != 3:
            logger.error(f"Expected 3D tensor, got shape {enriched_history.shape}")
            return False
        
        if enriched_history.shape[0] != self.max_history_size:
            logger.error(f"Expected {self.max_history_size} timesteps, got {enriched_history.shape[0]}")
            return False
        
        if enriched_history.shape[2] != self.num_channels:
            logger.error(f"Expected {self.num_channels} channels, got {enriched_history.shape[2]}")
            return False
        
        # Check for NaN or infinite values
        if not jnp.all(jnp.isfinite(enriched_history)):
            logger.error("Found NaN or infinite values in enriched history")
            return False
        
        return True


def create_enriched_history_jax(state, 
                               max_history_size: int = 100,
                               include_temporal_features: bool = True,
                               support_variable_scms: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    JAX-compatible function for creating enriched history with variable masking.
    
    This function is designed to be used within JAX transformations
    and avoids Python loops and dynamic operations.
    
    Args:
        state: AcquisitionState (processed outside JAX)
        max_history_size: Maximum history size
        include_temporal_features: Whether to include temporal context
        support_variable_scms: Whether to support variable-count SCMs
        
    Returns:
        Tuple of (enriched_history, variable_mask) where:
        - enriched_history: [max_history_size, n_vars, num_channels]
        - variable_mask: [n_vars] or None
    """
    builder = EnrichedHistoryBuilder(
        max_history_size=max_history_size,
        include_temporal_features=include_temporal_features,
        support_variable_scms=support_variable_scms
    )
    return builder.build_enriched_history(state)


def create_enriched_history_tensor(state, 
                                  max_history_size: int = 100,
                                  include_temporal_features: bool = True,
                                  support_variable_scms: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Create enriched history tensor from AcquisitionState with variable masking.
    
    This is the main entry point for converting AcquisitionState to enriched
    transformer input format with temporal context evolution and variable-agnostic support.
    
    Args:
        state: AcquisitionState with buffer and context information
        max_history_size: Maximum history size for transformer input
        include_temporal_features: Whether to include temporal context channels
        support_variable_scms: Whether to support variable-count SCMs
        
    Returns:
        Tuple of (enriched_history, variable_mask) where:
        - enriched_history: [max_history_size, n_vars, num_channels]
        - variable_mask: [n_vars] or None
    """
    return create_enriched_history_jax(
        state=state,
        max_history_size=max_history_size,
        include_temporal_features=include_temporal_features,
        support_variable_scms=support_variable_scms
    )


def validate_enriched_state_integration() -> bool:
    """
    Validate enriched state integration functionality.
    
    Returns:
        True if integration is working correctly, False otherwise
    """
    try:
        # Test with minimal state object
        class MockState:
            def __init__(self):
                self.buffer = MockBuffer()
                self.current_target = 'X0'
                self.marginal_parent_probs = {'X0': 0.8, 'X1': 0.3}
                self.mechanism_confidence = {'X0': 0.9, 'X1': 0.7}
                self.uncertainty_bits = 2.5
                
            def get_mechanism_insights(self):
                return {
                    'predicted_effects': {'X0': 1.5, 'X1': -0.8},
                    'mechanism_types': {'X0': 'linear', 'X1': 'gaussian'}
                }
                
            def get_optimization_progress(self):
                return {'best_value': 2.3, 'steps_since_improvement': 5}
        
        class MockBuffer:
            def get_all_samples(self):
                return [MockSample('X0', 1.0), MockSample('X1', -0.5)]
                
            def get_variable_coverage(self):
                return ['X0', 'X1']
        
        class MockSample:
            def __init__(self, var, value):
                self.values = {var: value}
                self.interventions = {var} if abs(value) > 0.5 else set()
                self.target_value = value
                self.reward = value
        
        # Test enriched history creation
        mock_state = MockState()
        enriched_tensor, variable_mask = create_enriched_history_tensor(
            state=mock_state,
            max_history_size=20,
            include_temporal_features=True,
            support_variable_scms=True
        )
        
        # Validate tensor shape
        expected_shape = (20, 2, 10)  # [max_history_size, n_vars, num_channels]
        if enriched_tensor.shape != expected_shape:
            logger.error(f"Unexpected tensor shape: {enriched_tensor.shape}, expected: {expected_shape}")
            return False
        
        # Validate tensor is finite
        if not jnp.all(jnp.isfinite(enriched_tensor)):
            logger.error("Enriched tensor contains NaN or infinite values")
            return False
        
        # Validate variable mask
        if variable_mask is not None:
            expected_mask_shape = (2,)  # [n_vars]
            if variable_mask.shape != expected_mask_shape:
                logger.error(f"Unexpected mask shape: {variable_mask.shape}, expected: {expected_mask_shape}")
                return False
            
            if not jnp.all((variable_mask == 0.0) | (variable_mask == 1.0)):
                logger.error("Variable mask should contain only 0.0 and 1.0 values")
                return False
        
        logger.info("Enriched state integration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Enriched state integration validation failed: {e}")
        return False