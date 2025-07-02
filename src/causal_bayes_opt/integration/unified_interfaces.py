"""
Unified interfaces for architecture switching and compatibility.

This module provides drop-in replacement interfaces that can switch between
different architectural implementations based on feature flags while
maintaining exact API compatibility.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from typing import Dict, Any, List, Optional, Union, Callable
import time
import logging

from ..config.architecture_flags import ArchitectureConfig, ArchitectureFlagManager, ParentSetModelType, PolicyArchitectureType

logger = logging.getLogger(__name__)


class UnifiedParentSetModel:
    """
    Unified parent set model interface supporting both discrete and continuous implementations.
    
    This class provides a drop-in replacement for existing parent set models while
    allowing seamless switching between discrete enumeration and continuous approaches.
    """
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.flag_manager = ArchitectureFlagManager(config)
        
        # Initialize models based on configuration
        self._init_models()
    
    def _init_models(self):
        """Initialize the appropriate model implementations."""
        if self.flag_manager.should_use_continuous_parent_model():
            self._init_continuous_model()
        else:
            self._init_discrete_model()
        
        # Initialize secondary model for A/B testing if needed
        if self.config.enable_side_by_side_validation:
            self._init_secondary_model()
    
    def _init_continuous_model(self):
        """Initialize continuous parent set model."""
        from ..avici_integration.continuous.integration import create_continuous_surrogate_model
        
        self.model_fn = create_continuous_surrogate_model(self.config.parent_set_config)
        self.transformed = hk.transform(self.model_fn)
        self.model_type = "continuous"
        
        logger.info("Initialized continuous parent set model")
    
    def _init_discrete_model(self):
        """Initialize discrete parent set model."""
        from ..avici_integration.parent_set.jax_model import create_jax_parent_set_model
        
        # Extract necessary parameters for discrete model
        config = self.config.parent_set_config
        n_vars = config.get('n_vars', 10)  # Will be set during initialization
        
        self.model_fn = create_jax_parent_set_model(
            layers=config.get('layers', 8),
            dim=config.get('dim', 128),
            key_size=config.get('key_size', 32),
            num_heads=config.get('num_heads', 8),
            widening_factor=config.get('widening_factor', 4),
            dropout=config.get('dropout', 0.1),
            max_parent_size=config.get('max_parent_size', 3),
            n_vars=n_vars
        )
        self.transformed = hk.transform(self.model_fn)
        self.model_type = "discrete"
        
        logger.info("Initialized discrete parent set model")
    
    def _init_secondary_model(self):
        """Initialize secondary model for A/B testing."""
        # For A/B testing, initialize the opposite model type
        if self.model_type == "continuous":
            # Primary is continuous, secondary is discrete
            logger.info("Initializing discrete model for A/B testing")
            # Implementation would go here
            self.secondary_model = None
        else:
            # Primary is discrete, secondary is continuous
            logger.info("Initializing continuous model for A/B testing")
            from ..avici_integration.continuous.integration import create_continuous_surrogate_model
            
            secondary_config = self.config.parent_set_config.copy()
            # Adjust config for continuous model
            continuous_config = {
                'hidden_dim': secondary_config.get('dim', 128),
                'num_layers': secondary_config.get('layers', 4),
                'num_heads': secondary_config.get('num_heads', 8),
                'dropout': secondary_config.get('dropout', 0.1)
            }
            
            secondary_fn = create_continuous_surrogate_model(continuous_config)
            self.secondary_transformed = hk.transform(secondary_fn)
            self.secondary_model_type = "continuous"
    
    def init(self, rng_key: jnp.ndarray, data: jnp.ndarray, target_variable: Union[int, str], **kwargs):
        """Initialize model parameters."""
        self.params = self.transformed.init(rng_key, data, target_variable, **kwargs)
        
        if hasattr(self, 'secondary_transformed'):
            # Initialize secondary model for A/B testing
            self.secondary_params = self.secondary_transformed.init(rng_key, data, target_variable)
    
    def predict_parent_sets(self,
                          data: jnp.ndarray,
                          target_variable: Union[int, str],
                          variable_order: Optional[List[str]] = None,
                          rng_key: Optional[jnp.ndarray] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Predict parent sets with unified interface.
        
        Args:
            data: Intervention data [N, d, 3]
            target_variable: Target variable (index or name)
            variable_order: Variable names (for compatibility)
            rng_key: Random key (optional)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with parent set predictions
        """
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        
        # Time the primary prediction
        start_time = time.time()
        primary_result = self._predict_with_model(
            self.transformed, self.params, data, target_variable, 
            variable_order, rng_key, **kwargs
        )
        primary_time = time.time() - start_time
        
        # Log performance metrics
        self.flag_manager.log_architecture_metric(
            "prediction_time", primary_time, self.model_type
        )
        
        # Add model type to result
        primary_result['model_type'] = self.model_type
        primary_result['prediction_time'] = primary_time
        
        # Run secondary model for validation if needed
        if (hasattr(self, 'secondary_transformed') and 
            self.flag_manager.should_run_side_by_side_validation(str(target_variable))):
            
            start_time = time.time()
            secondary_result = self._predict_with_model(
                self.secondary_transformed, self.secondary_params,
                data, target_variable, variable_order, rng_key, **kwargs
            )
            secondary_time = time.time() - start_time
            
            # Log secondary metrics
            self.flag_manager.log_architecture_metric(
                "prediction_time", secondary_time, self.secondary_model_type
            )
            
            # Add validation information
            primary_result['validation'] = {
                'secondary_model_type': self.secondary_model_type,
                'secondary_result': secondary_result,
                'secondary_time': secondary_time,
                'time_ratio': secondary_time / primary_time if primary_time > 0 else 1.0
            }
            
            logger.info(f"A/B test: {self.model_type} vs {self.secondary_model_type}, "
                       f"time ratio: {secondary_time/primary_time:.2f}")
        
        return primary_result
    
    def _predict_with_model(self,
                          transformed_model,
                          params,
                          data: jnp.ndarray,
                          target_variable: Union[int, str],
                          variable_order: Optional[List[str]],
                          rng_key: jnp.ndarray,
                          **kwargs) -> Dict[str, Any]:
        """Make prediction with specific model."""
        # Convert target variable to index if needed
        if isinstance(target_variable, str):
            if variable_order is None:
                raise ValueError("variable_order required when target_variable is string")
            target_idx = variable_order.index(target_variable)
        else:
            target_idx = target_variable
        
        # Make prediction
        if self.model_type == "continuous" or hasattr(self, 'secondary_model_type'):
            # Continuous model returns probabilities directly
            parent_probs = transformed_model.apply(params, rng_key, data, target_idx)
            
            # Convert to legacy format
            from ..avici_integration.continuous.integration import convert_to_legacy_format
            result = convert_to_legacy_format(parent_probs, variable_order)
        else:
            # Discrete model returns full result dictionary
            result = transformed_model.apply(params, rng_key, data, variable_order, target_variable, **kwargs)
        
        return result


class UnifiedAcquisitionPolicy:
    """
    Unified acquisition policy interface supporting both legacy and enriched architectures.
    
    This class provides seamless switching between post-transformer feature concatenation
    and enriched transformer input approaches.
    """
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.flag_manager = ArchitectureFlagManager(config)
        
        # Initialize policy based on configuration
        self._init_policy()
    
    def _init_policy(self):
        """Initialize the appropriate policy implementation."""
        if self.flag_manager.should_use_enriched_policy():
            self._init_enriched_policy()
        else:
            self._init_legacy_policy()
        
        # Initialize secondary policy for A/B testing if needed
        if self.config.enable_side_by_side_validation:
            self._init_secondary_policy()
    
    def _init_enriched_policy(self):
        """Initialize enriched policy architecture."""
        from ..acquisition.enriched.policy_heads import create_enriched_policy_factory
        
        self.policy_fn = create_enriched_policy_factory(self.config.policy_config)
        self.transformed = hk.transform(self.policy_fn)
        self.policy_type = "enriched"
        
        logger.info("Initialized enriched acquisition policy")
    
    def _init_legacy_policy(self):
        """Initialize legacy policy architecture."""
        from ..acquisition.policy import create_acquisition_policy
        
        # Create legacy policy using existing function
        self.policy_fn = lambda *args, **kwargs: create_acquisition_policy(self.config.policy_config)(*args, **kwargs)
        self.transformed = hk.transform(self.policy_fn)
        self.policy_type = "legacy"
        
        logger.info("Initialized legacy acquisition policy")
    
    def _init_secondary_policy(self):
        """Initialize secondary policy for A/B testing."""
        # For A/B testing, initialize the opposite policy type
        if self.policy_type == "enriched":
            # Primary is enriched, secondary is legacy
            logger.info("Initializing legacy policy for A/B testing")
            # Implementation would go here
            self.secondary_policy = None
        else:
            # Primary is legacy, secondary is enriched
            logger.info("Initializing enriched policy for A/B testing")
            from ..acquisition.enriched.policy_heads import create_enriched_policy_factory
            
            secondary_fn = create_enriched_policy_factory(self.config.policy_config)
            self.secondary_transformed = hk.transform(secondary_fn)
            self.secondary_policy_type = "enriched"
    
    def init(self, rng_key: jnp.ndarray, *args, **kwargs):
        """Initialize policy parameters."""
        self.params = self.transformed.init(rng_key, *args, **kwargs)
        
        if hasattr(self, 'secondary_transformed'):
            # Initialize secondary policy for A/B testing
            self.secondary_params = self.secondary_transformed.init(rng_key, *args, **kwargs)
    
    def __call__(self,
                 state_or_history,  # Either AcquisitionState or enriched_history
                 target_variable_idx: Optional[int] = None,
                 is_training: bool = True,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Run policy inference with unified interface.
        
        Args:
            state_or_history: AcquisitionState (legacy) or enriched_history (enriched)
            target_variable_idx: Target variable index (for enriched policy)
            is_training: Training mode flag
            **kwargs: Additional arguments
            
        Returns:
            Policy outputs dictionary
        """
        # Time the primary prediction
        start_time = time.time()
        
        if self.policy_type == "enriched":
            # Enriched policy expects enriched_history and target_variable_idx
            if target_variable_idx is None:
                raise ValueError("target_variable_idx required for enriched policy")
            primary_result = self.transformed.apply(
                self.params, random.PRNGKey(42), state_or_history, target_variable_idx, is_training
            )
        else:
            # Legacy policy expects AcquisitionState
            primary_result = self.transformed.apply(
                self.params, random.PRNGKey(42), state_or_history, is_training
            )
        
        primary_time = time.time() - start_time
        
        # Log performance metrics
        self.flag_manager.log_architecture_metric(
            "policy_time", primary_time, self.policy_type
        )
        
        # Add metadata
        primary_result['policy_type'] = self.policy_type
        primary_result['inference_time'] = primary_time
        
        # Run secondary policy for validation if needed
        if (hasattr(self, 'secondary_transformed') and 
            self.flag_manager.should_run_side_by_side_validation()):
            
            start_time = time.time()
            
            try:
                if self.secondary_policy_type == "enriched":
                    secondary_result = self.secondary_transformed.apply(
                        self.secondary_params, random.PRNGKey(42), 
                        state_or_history, target_variable_idx, is_training
                    )
                else:
                    secondary_result = self.secondary_transformed.apply(
                        self.secondary_params, random.PRNGKey(42), state_or_history, is_training
                    )
                
                secondary_time = time.time() - start_time
                
                # Log secondary metrics
                self.flag_manager.log_architecture_metric(
                    "policy_time", secondary_time, self.secondary_policy_type
                )
                
                # Add validation information
                primary_result['validation'] = {
                    'secondary_policy_type': self.secondary_policy_type,
                    'secondary_result': secondary_result,
                    'secondary_time': secondary_time,
                    'time_ratio': secondary_time / primary_time if primary_time > 0 else 1.0
                }
                
            except Exception as e:
                logger.warning(f"Secondary policy validation failed: {e}")
                primary_result['validation_error'] = str(e)
        
        return primary_result


def create_unified_acbo_pipeline(config: ArchitectureConfig) -> Dict[str, Any]:
    """
    Create unified ACBO pipeline with architecture switching.
    
    Args:
        config: Architecture configuration
        
    Returns:
        Dictionary with unified pipeline components
    """
    # Validate configuration
    if not config.validate():
        raise ValueError("Invalid architecture configuration")
    
    # Create unified components
    parent_set_model = UnifiedParentSetModel(config)
    acquisition_policy = UnifiedAcquisitionPolicy(config)
    flag_manager = ArchitectureFlagManager(config)
    
    logger.info(f"Created unified ACBO pipeline:")
    logger.info(f"  Parent Set Model: {config.parent_set_model_type.value}")
    logger.info(f"  Policy Architecture: {config.policy_architecture_type.value}")
    logger.info(f"  A/B Testing: {config.enable_side_by_side_validation}")
    
    return {
        'parent_set_model': parent_set_model,
        'acquisition_policy': acquisition_policy,
        'flag_manager': flag_manager,
        'config': config
    }


class ArchitecturePerformanceMonitor:
    """Monitor and compare performance between architectures."""
    
    def __init__(self, flag_manager: ArchitectureFlagManager):
        self.flag_manager = flag_manager
        self.start_times = {}
    
    def start_timing(self, operation: str, architecture: str) -> str:
        """Start timing an operation."""
        timing_id = f"{operation}_{architecture}_{id(self)}"
        self.start_times[timing_id] = time.time()
        return timing_id
    
    def end_timing(self, timing_id: str, operation: str, architecture: str):
        """End timing and log the result."""
        if timing_id in self.start_times:
            elapsed = time.time() - self.start_times[timing_id]
            self.flag_manager.log_architecture_metric(operation, elapsed, architecture)
            del self.start_times[timing_id]
            return elapsed
        return None
    
    def compare_architectures(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Compare specific metric between architectures."""
        comparison = self.flag_manager.get_performance_comparison()
        
        results = {}
        for arch_type, metrics in comparison.items():
            if metric_name in metrics:
                results[arch_type] = metrics[metric_name]['mean']
        
        return results if results else None