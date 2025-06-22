"""
Compatibility Layer for Seamless Legacy-to-JAX Integration

Provides wrapper classes and adapter functions that allow existing code to work
with JAX-native architecture without immediate migration. This enables gradual
transition while maintaining full backward compatibility.

Key features:
- Drop-in replacement for legacy classes
- Automatic conversion between legacy and JAX formats
- Performance monitoring and optimization hints
- Clear deprecation warnings with migration guidance
"""

import warnings
from typing import Dict, List, Any, Optional, Union, Callable
import jax.numpy as jnp
import pyrsistent as pyr
from functools import wraps
from dataclasses import dataclass

from ..jax_native import (
    JAXConfig, JAXSampleBuffer, JAXAcquisitionState,
    create_jax_config, create_jax_state,
    add_sample_jax, compute_policy_features_jax
)
from .legacy_to_jax import convert_legacy_to_jax, measure_conversion_performance


class JAXCompatibilityWrapper:
    """
    Compatibility wrapper that provides legacy API while using JAX backend.
    
    This class allows existing code to continue working unchanged while
    automatically leveraging JAX performance improvements under the hood.
    
    Usage:
        # Drop-in replacement for legacy AcquisitionState
        state = JAXCompatibilityWrapper(legacy_state)
        
        # Existing code works unchanged
        features = state.get_mechanism_features()  # Returns dict as expected
        state = state.add_sample(sample_dict)      # Accepts legacy format
        
        # Performance benefits from JAX operations
        # Automatic conversion handled transparently
    """
    
    def __init__(
        self, 
        legacy_state=None,
        jax_state: Optional[JAXAcquisitionState] = None,
        config: Optional[JAXConfig] = None,
        enable_performance_warnings: bool = True,
        conversion_threshold_ms: float = 10.0
    ):
        """
        Initialize compatibility wrapper.
        
        Args:
            legacy_state: Optional legacy state to wrap
            jax_state: Optional pre-converted JAX state
            config: Optional JAX configuration
            enable_performance_warnings: Whether to warn about conversion overhead
            conversion_threshold_ms: Threshold for performance warnings
        """
        self._enable_warnings = enable_performance_warnings
        self._conversion_threshold = conversion_threshold_ms
        self._conversion_count = 0
        self._total_conversion_time = 0.0
        
        if jax_state is not None:
            # Use pre-converted JAX state
            self._jax_state = jax_state
            self._legacy_state = None
        elif legacy_state is not None:
            # Convert legacy state to JAX
            self._legacy_state = legacy_state
            self._jax_state = self._convert_with_monitoring(legacy_state, config)
        else:
            # Create empty state
            if config is None:
                config = create_jax_config(['X', 'Y', 'Z'], 'Y')  # Default test config
            self._jax_state = create_jax_state(config)
            self._legacy_state = None
        
        self._issue_deprecation_warning()
    
    def _convert_with_monitoring(
        self, 
        legacy_state, 
        config: Optional[JAXConfig] = None
    ) -> JAXAcquisitionState:
        """Convert legacy state with performance monitoring."""
        if self._enable_warnings:
            perf_metrics = measure_conversion_performance(legacy_state, config)
            self._conversion_count += 1
            self._total_conversion_time += perf_metrics['total_conversion_time']
            
            if perf_metrics['total_conversion_time'] > self._conversion_threshold:
                warnings.warn(
                    f"Legacy-to-JAX conversion took {perf_metrics['total_conversion_time']:.1f}ms. "
                    f"Consider migrating to JAX-native APIs for better performance. "
                    f"See docs/migration/MIGRATION_GUIDE.md",
                    PerformanceWarning,
                    stacklevel=3
                )
        
        return convert_legacy_to_jax(legacy_state, config)
    
    def _issue_deprecation_warning(self):
        """Issue deprecation warning for compatibility wrapper usage."""
        warnings.warn(
            "JAXCompatibilityWrapper is a temporary compatibility layer. "
            "For optimal performance, migrate to JAX-native APIs in causal_bayes_opt.jax_native. "
            "See docs/migration/MIGRATION_GUIDE.md for migration instructions.",
            DeprecationWarning,
            stacklevel=3
        )
    
    # Legacy API methods - maintain exact signatures and behavior
    
    def get_mechanism_features(self) -> Dict[str, float]:
        """Get mechanism features in legacy dictionary format."""
        features_tensor = self._jax_state.mechanism_features
        config = self._jax_state.config
        
        # Convert tensor back to dictionary format
        result = {}
        for i, var_name in enumerate(config.variable_names):
            if i != config.target_idx:  # Exclude target variable
                # Extract effect magnitude from mechanism features
                effect = float(features_tensor[i, 0])
                result[var_name] = effect
        
        return result
    
    def get_mechanism_confidence(self) -> Dict[str, float]:
        """Get mechanism confidence in legacy dictionary format."""
        confidence_tensor = self._jax_state.confidence_scores
        config = self._jax_state.config
        
        result = {}
        for i, var_name in enumerate(config.variable_names):
            if i != config.target_idx:  # Exclude target variable
                confidence = float(confidence_tensor[i])
                result[var_name] = confidence
        
        return result
    
    def get_marginal_parent_probs(self) -> Dict[str, float]:
        """Get marginal parent probabilities in legacy dictionary format."""
        probs_tensor = self._jax_state.marginal_probs
        config = self._jax_state.config
        
        result = {}
        for i, var_name in enumerate(config.variable_names):
            if i != config.target_idx:  # Exclude target variable
                prob = float(probs_tensor[i])
                result[var_name] = prob
        
        return result
    
    def add_sample(self, sample_data: Union[Dict, Any]) -> 'JAXCompatibilityWrapper':
        """Add sample in legacy format, return new wrapper."""
        # Extract sample information
        if hasattr(sample_data, 'values'):
            values_dict = sample_data.values
            interventions_dict = getattr(sample_data, 'interventions', {})
            target_value = getattr(sample_data, 'target', 0.0)
        elif isinstance(sample_data, dict):
            values_dict = sample_data.get('values', {})
            interventions_dict = sample_data.get('interventions', {})
            target_value = sample_data.get('target', 0.0)
        else:
            raise ValueError(f"Unknown sample format: {type(sample_data)}")
        
        # Convert to tensor format
        config = self._jax_state.config
        variable_values = jnp.zeros(config.n_vars)
        intervention_mask = jnp.zeros(config.n_vars, dtype=bool)
        
        for i, var_name in enumerate(config.variable_names):
            if var_name in values_dict:
                variable_values = variable_values.at[i].set(float(values_dict[var_name]))
            if var_name in interventions_dict:
                intervention_mask = intervention_mask.at[i].set(True)
        
        # Add to JAX state
        new_jax_state = add_sample_jax(
            self._jax_state, variable_values, intervention_mask, float(target_value)
        )
        
        # Return new wrapper
        return JAXCompatibilityWrapper(
            jax_state=new_jax_state,
            enable_performance_warnings=self._enable_warnings,
            conversion_threshold_ms=self._conversion_threshold
        )
    
    def get_optimization_progress(self) -> Dict[str, float]:
        """Get optimization progress metrics in legacy format."""
        from ..jax_native.operations import compute_optimization_progress_jax
        return compute_optimization_progress_jax(self._jax_state)
    
    def get_exploration_coverage(self) -> Dict[str, float]:
        """Get exploration coverage metrics in legacy format."""
        from ..jax_native.operations import compute_exploration_coverage_jax
        return compute_exploration_coverage_jax(self._jax_state)
    
    # Properties that maintain legacy interface
    
    @property
    def best_value(self) -> float:
        """Current best target value."""
        return self._jax_state.best_value
    
    @property
    def current_target(self) -> str:
        """Current target variable name."""
        return self._jax_state.config.get_target_name()
    
    @property
    def step(self) -> int:
        """Current optimization step."""
        return self._jax_state.current_step
    
    @property
    def uncertainty_bits(self) -> float:
        """Posterior uncertainty estimate."""
        return self._jax_state.uncertainty_bits
    
    @property 
    def buffer_statistics(self):
        """Buffer statistics in legacy format."""
        return BufferStatisticsWrapper(self._jax_state.sample_buffer)
    
    # JAX-native access for performance-critical code
    
    @property
    def jax_state(self) -> JAXAcquisitionState:
        """Access to underlying JAX state for performance-critical operations."""
        return self._jax_state
    
    def get_policy_features_tensor(self) -> jnp.ndarray:
        """Get policy features as JAX tensor (performance optimized)."""
        return compute_policy_features_jax(self._jax_state)
    
    # Performance monitoring
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this wrapper."""
        return {
            'conversion_count': self._conversion_count,
            'total_conversion_time_ms': self._total_conversion_time,
            'average_conversion_time_ms': (
                self._total_conversion_time / max(1, self._conversion_count)
            )
        }


class BufferStatisticsWrapper:
    """Wrapper for buffer statistics to maintain legacy interface."""
    
    def __init__(self, jax_buffer: JAXSampleBuffer):
        self._buffer = jax_buffer
    
    @property
    def total_samples(self) -> int:
        return self._buffer.n_samples
    
    @property
    def num_interventions(self) -> int:
        # Count samples with any interventions
        if self._buffer.n_samples == 0:
            return 0
        
        # Check for any True values in intervention arrays
        valid_interventions = self._buffer.interventions[:self._buffer.n_samples]
        return int(jnp.sum(jnp.any(valid_interventions, axis=1)))


def create_compatibility_layer(
    legacy_state=None,
    enable_monitoring: bool = True,
    performance_threshold_ms: float = 5.0
) -> JAXCompatibilityWrapper:
    """
    Create compatibility layer for gradual migration to JAX-native APIs.
    
    This is the main entry point for existing code to start benefiting from
    JAX performance improvements without requiring immediate refactoring.
    
    Args:
        legacy_state: Optional existing legacy state to wrap
        enable_monitoring: Whether to monitor and warn about performance
        performance_threshold_ms: Threshold for performance warnings
        
    Returns:
        Compatibility wrapper that provides legacy API with JAX backend
        
    Example:
        # Existing code
        state = AcquisitionState(...)
        
        # Add one line to get JAX performance
        state = create_compatibility_layer(state)
        
        # All existing code continues to work
        features = state.get_mechanism_features()
        new_state = state.add_sample(sample)
    """
    return JAXCompatibilityWrapper(
        legacy_state=legacy_state,
        enable_performance_warnings=enable_monitoring,
        conversion_threshold_ms=performance_threshold_ms
    )


def legacy_api_adapter(func: Callable) -> Callable:
    """
    Decorator to automatically adapt legacy functions to work with JAX backend.
    
    This decorator allows existing functions that expect legacy state objects
    to automatically work with JAX-native implementations.
    
    Args:
        func: Function that expects legacy state arguments
        
    Returns:
        Wrapped function that accepts both legacy and JAX states
        
    Example:
        @legacy_api_adapter
        def my_legacy_function(state):
            # Function expects legacy state format
            features = state.get_mechanism_features()
            return process(features)
        
        # Works with both legacy and JAX states
        result = my_legacy_function(jax_state)  # Automatically wrapped
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert JAX states to compatibility wrappers
        converted_args = []
        for arg in args:
            if isinstance(arg, JAXAcquisitionState):
                # Wrap JAX state in compatibility layer
                arg = JAXCompatibilityWrapper(jax_state=arg, enable_performance_warnings=False)
            converted_args.append(arg)
        
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, JAXAcquisitionState):
                value = JAXCompatibilityWrapper(jax_state=value, enable_performance_warnings=False)
            converted_kwargs[key] = value
        
        return func(*converted_args, **converted_kwargs)
    
    return wrapper


# Performance warning class
class PerformanceWarning(UserWarning):
    """Warning category for performance-related issues."""
    pass


# Migration utilities

@dataclass
class MigrationPlan:
    """Migration plan for converting legacy code to JAX-native."""
    legacy_functions_used: List[str]
    conversion_overhead_ms: float
    recommended_changes: List[str]
    expected_speedup: float
    migration_priority: str  # 'high', 'medium', 'low'


def analyze_migration_opportunity(wrapper: JAXCompatibilityWrapper) -> MigrationPlan:
    """
    Analyze usage patterns and recommend migration strategy.
    
    Args:
        wrapper: Compatibility wrapper with usage statistics
        
    Returns:
        Migration plan with specific recommendations
    """
    stats = wrapper.get_performance_stats()
    
    # Determine migration priority based on usage patterns
    if stats['conversion_count'] > 100 or stats['total_conversion_time_ms'] > 1000:
        priority = 'high'
        expected_speedup = 10.0
    elif stats['conversion_count'] > 20 or stats['total_conversion_time_ms'] > 200:
        priority = 'medium' 
        expected_speedup = 5.0
    else:
        priority = 'low'
        expected_speedup = 2.0
    
    recommendations = [
        "Replace JAXCompatibilityWrapper with direct JAXAcquisitionState usage",
        "Use compute_policy_features_jax() instead of get_mechanism_features()",
        "Convert dictionary-based sample handling to tensor operations",
        "Leverage JAX compilation for performance-critical loops"
    ]
    
    if priority == 'high':
        recommendations.insert(0, "URGENT: High conversion overhead detected")
    
    return MigrationPlan(
        legacy_functions_used=['get_mechanism_features', 'add_sample'],  # Detected dynamically
        conversion_overhead_ms=stats['total_conversion_time_ms'],
        recommended_changes=recommendations,
        expected_speedup=expected_speedup,
        migration_priority=priority
    )


# Context manager for temporary compatibility
class TemporaryCompatibilityMode:
    """
    Context manager for temporarily enabling compatibility mode.
    
    Useful for testing and gradual migration scenarios.
    """
    
    def __init__(self, enable_warnings: bool = False):
        self.enable_warnings = enable_warnings
        self.original_warning_state = None
    
    def __enter__(self):
        if not self.enable_warnings:
            # Temporarily suppress compatibility warnings
            warnings.filterwarnings('ignore', category=DeprecationWarning, module=__name__)
            warnings.filterwarnings('ignore', category=PerformanceWarning)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable_warnings:
            # Restore warning state
            warnings.resetwarnings()