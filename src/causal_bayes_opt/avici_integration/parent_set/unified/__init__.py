"""
Unified Parent Set Prediction Model Architecture

This module provides the unified parent set model that combines:
- Proven AVICI-style transformer architecture (from original model)
- Target-aware conditioning (from modular model)
- Optional mechanism prediction capabilities
- Adaptive max_parents based on graph density

Key Components:
- JAXUnifiedParentSetModel: JAX-compiled main model (RECOMMENDED)
- UnifiedParentSetModel: Original unified model (DEPRECATED - use JAX version)
- TargetAwareConfig: Configuration with adaptive parameters
- Mechanism prediction heads (optional via config)
- Modular file structure for maintainability

MIGRATION GUIDE:
- OLD: UnifiedParentSetModel (has JAX compilation issues)
- NEW: JAXUnifiedParentSetModel (full JAX compatibility, same features)
- Use JAXUnifiedParentSetModelWrapper for drop-in replacement
"""

import warnings
from .config import TargetAwareConfig, create_structure_only_config, create_mechanism_aware_config
from .target_conditioning import add_target_conditioning
from .mechanism_heads import MechanismPredictionHeads
from .utils import compute_adaptive_max_parents, validate_unified_config

# NEW: JAX-compatible model (RECOMMENDED)
from .jax_model import (
    JAXUnifiedParentSetModel,
    JAXUnifiedParentSetModelWrapper,
    create_jax_unified_parent_set_model,
    predict_with_jax_unified_model
)

# DEPRECATED: Original unified model (has JAX compilation issues)
from .model import UnifiedParentSetModel, create_unified_parent_set_model

def create_parent_set_model(*args, use_jax=True, **kwargs):
    """
    Factory function that creates the appropriate parent set model.
    
    Args:
        use_jax: If True (default), uses JAX-compatible model. If False, uses original model.
        *args, **kwargs: Arguments passed to model creation function
        
    Returns:
        JAX-compatible model by default, original model if use_jax=False
    """
    if use_jax:
        return create_jax_unified_parent_set_model(*args, **kwargs)
    else:
        warnings.warn(
            "Using original UnifiedParentSetModel which has JAX compilation issues. "
            "Consider switching to JAX-compatible version with use_jax=True (default).",
            DeprecationWarning,
            stacklevel=2
        )
        return create_unified_parent_set_model(*args, **kwargs)

# Provide backward compatibility with deprecation warning
def _create_unified_parent_set_model_with_warning(*args, **kwargs):
    warnings.warn(
        "create_unified_parent_set_model is deprecated due to JAX compilation issues. "
        "Use create_jax_unified_parent_set_model or create_parent_set_model() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_unified_parent_set_model(*args, **kwargs)

__all__ = [
    # RECOMMENDED: JAX-compatible models
    'JAXUnifiedParentSetModel',
    'JAXUnifiedParentSetModelWrapper', 
    'create_jax_unified_parent_set_model',
    'predict_with_jax_unified_model',
    'create_parent_set_model',  # Factory function (uses JAX by default)
    
    # Configuration
    'TargetAwareConfig', 
    'create_structure_only_config',
    'create_mechanism_aware_config',
    
    # Components
    'add_target_conditioning',
    'MechanismPredictionHeads',
    
    # Utilities
    'compute_adaptive_max_parents',
    'validate_unified_config',
    
    # DEPRECATED: Original models (kept for backward compatibility)
    'UnifiedParentSetModel',           # DEPRECATED - use JAXUnifiedParentSetModel
    'create_unified_parent_set_model',  # DEPRECATED - use create_jax_unified_parent_set_model
]

# Mark deprecated exports
_deprecated_exports = {
    'UnifiedParentSetModel': 'JAXUnifiedParentSetModel',
    'create_unified_parent_set_model': 'create_jax_unified_parent_set_model'
}

def __getattr__(name):
    """Provide deprecation warnings for old exports."""
    if name in _deprecated_exports:
        warnings.warn(
            f"'{name}' is deprecated due to JAX compilation issues. "
            f"Use '{_deprecated_exports[name]}' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if name == 'UnifiedParentSetModel':
            return UnifiedParentSetModel
        elif name == 'create_unified_parent_set_model':
            return _create_unified_parent_set_model_with_warning
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")