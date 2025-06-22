"""
Parent Set Prediction Module

This module provides a clean, modular API for parent set prediction models.

CURRENT ARCHITECTURE (2024):
- unified/: JAX-compatible unified model (RECOMMENDED)
- model.py: Original ParentSetPredictionModel (DEPRECATED - limited features)
- mechanism_aware.py: Mechanism-aware extensions (DEPRECATED - use unified)

MIGRATION GUIDE:
- OLD: ParentSetPredictionModel (basic model, no target conditioning)
- OLD: ModularParentSetModel (mechanism-aware but not JAX-compatible)
- NEW: JAXUnifiedParentSetModel (combines all features + JAX compatibility)

For new code, use: from .unified import JAXUnifiedParentSetModel
"""

import warnings

# RECOMMENDED: Unified JAX-compatible models
from .unified import (
    JAXUnifiedParentSetModel,
    JAXUnifiedParentSetModelWrapper,
    create_jax_unified_parent_set_model,
    create_parent_set_model,  # Factory function (uses JAX by default)
    TargetAwareConfig,
    create_structure_only_config,
    create_mechanism_aware_config,
    predict_with_jax_unified_model
)

# Core utilities (shared across all models) - UPDATED WITH JAX OPTIMIZATIONS
from .inference import (
    predict_parent_posterior,  # Now JAX-optimized internally 
    compute_loss, 
    create_train_step,
    create_jax_optimized_model,  # NEW: Easy JAX model creation
    benchmark_model_performance  # NEW: Performance validation
)
from .enumeration import enumerate_possible_parent_sets, compute_adaptive_k
from .encoding import encode_parent_set, create_parent_set_indicators
from .posterior import (
    ParentSetPosterior,
    create_parent_set_posterior,
    get_most_likely_parents,
    get_parent_set_probability,
    get_marginal_parent_probabilities,
    compute_posterior_entropy,
    compute_posterior_concentration,
    filter_parent_sets_by_probability,
    compare_posteriors,
    summarize_posterior
)

# DEPRECATED: Original models (kept for backward compatibility)
from .model import ParentSetPredictionModel
from .mechanism_aware import (
    ModularParentSetModel,
    MechanismAwareConfig,
    MechanismPrediction,
    MechanismType,
    create_modular_parent_set_model,
    predict_with_mechanisms,
    get_all_mechanism_types,
    validate_mechanism_types,
    create_enhanced_config,
    compare_model_outputs
)

# Override old create_parent_set_model with deprecation warning
def _deprecated_create_parent_set_model(*args, **kwargs):
    """DEPRECATED: Use create_parent_set_model from unified module instead."""
    warnings.warn(
        "Importing create_parent_set_model from .model is deprecated. "
        "The basic ParentSetPredictionModel lacks target conditioning and JAX compatibility. "
        "Use create_parent_set_model from .unified (default JAX model) or "
        "create_jax_unified_parent_set_model for full features.",
        DeprecationWarning,
        stacklevel=2
    )
    from .model import create_parent_set_model as old_create
    return old_create(*args, **kwargs)

__all__ = [
    # RECOMMENDED: JAX-compatible unified models
    'JAXUnifiedParentSetModel',         # Main JAX model class
    'JAXUnifiedParentSetModelWrapper',  # Drop-in replacement wrapper
    'create_jax_unified_parent_set_model',  # Factory for JAX model
    'create_parent_set_model',          # Smart factory (uses JAX by default)
    'predict_with_jax_unified_model',   # Direct prediction interface
    
    # Configuration (from unified)
    'TargetAwareConfig',
    'create_structure_only_config',
    'create_mechanism_aware_config',
    
    # Core utilities (shared) - JAX-OPTIMIZED
    'predict_parent_posterior',        # Now 10-100x faster internally
    'compute_loss',
    'create_train_step',
    'create_jax_optimized_model',      # NEW: Easy JAX model creation  
    'benchmark_model_performance',     # NEW: Performance validation
    
    # Posterior data structure and utilities
    'ParentSetPosterior',
    'create_parent_set_posterior',
    'get_most_likely_parents',
    'get_parent_set_probability',
    'get_marginal_parent_probabilities',
    'compute_posterior_entropy',
    'compute_posterior_concentration',
    'filter_parent_sets_by_probability',
    'compare_posteriors',
    'summarize_posterior',
    
    # Enumeration and encoding utilities
    'enumerate_possible_parent_sets', 
    'compute_adaptive_k',
    'encode_parent_set',
    'create_parent_set_indicators',
    
    # DEPRECATED: Legacy models (backward compatibility only)
    'ParentSetPredictionModel',         # DEPRECATED - use JAXUnifiedParentSetModel
    'ModularParentSetModel',            # DEPRECATED - use JAXUnifiedParentSetModel  
    'MechanismAwareConfig',             # DEPRECATED - use TargetAwareConfig
    'MechanismPrediction',
    'MechanismType',
    'create_modular_parent_set_model',  # DEPRECATED - use create_jax_unified_parent_set_model
    'predict_with_mechanisms',          # DEPRECATED - use predict_with_jax_unified_model
    'get_all_mechanism_types',
    'validate_mechanism_types',
    'create_enhanced_config',           # DEPRECATED - use create_mechanism_aware_config
    'compare_model_outputs',
]

# Deprecation mapping for helpful error messages
_deprecated_imports = {
    'ParentSetPredictionModel': 'JAXUnifiedParentSetModel',
    'ModularParentSetModel': 'JAXUnifiedParentSetModel', 
    'MechanismAwareConfig': 'TargetAwareConfig',
    'create_modular_parent_set_model': 'create_jax_unified_parent_set_model',
    'predict_with_mechanisms': 'predict_with_jax_unified_model',
    'create_enhanced_config': 'create_mechanism_aware_config'
}

def __getattr__(name):
    """Provide helpful deprecation messages for old imports."""
    if name in _deprecated_imports:
        warnings.warn(
            f"'{name}' is deprecated. Use '{_deprecated_imports[name]}' instead. "
            f"The unified JAX model provides all features with better performance.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Return the actual deprecated object
        if name == 'ParentSetPredictionModel':
            return ParentSetPredictionModel
        elif name == 'ModularParentSetModel':
            return ModularParentSetModel
        elif name == 'MechanismAwareConfig':
            return MechanismAwareConfig
        elif name == 'create_modular_parent_set_model':
            return create_modular_parent_set_model
        elif name == 'predict_with_mechanisms':
            return predict_with_mechanisms
        elif name == 'create_enhanced_config':
            return create_enhanced_config
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
