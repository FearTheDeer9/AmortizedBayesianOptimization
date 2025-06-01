"""
AVICI Integration Module - Clean Public API

Provides functional interface for converting SCM data to AVICI format.
"""

# Main user-facing functions
from .core.conversion import (
    samples_to_avici_format_validated as samples_to_avici_format,
    create_training_batch_validated as create_training_batch,
)

# Standardization utilities (for advanced users)
from .core.standardization import (
    compute_standardization_params,
    apply_standardization,
    reverse_standardization,
    StandardizationType,
    StandardizationParams,
)

# Validation (for advanced users)
from .core.validation import (
    validate_conversion_inputs,
    validate_training_batch_inputs,
    validate_avici_data_structure,
    validate_training_batch_structure,
)

# Analysis utilities
from .utils.analysis import (
    analyze_avici_data,
    compare_conversions,
    compute_data_quality_metrics,
)

# Parent set functionality (unchanged)
from .parent_set import (
    create_parent_set_model,
    predict_parent_sets,
    ParentSetPredictionModel,
    compute_loss,
    create_train_step,
)

# Type aliases for user convenience
from typing import List, Dict, Any
import pyrsistent as pyr
import jax.numpy as jnp

SampleList = List[pyr.PMap]
VariableOrder = List[str]
AVICIDataBatch = Dict[str, Any]

__all__ = [
    # Main API (what most users will use)
    "samples_to_avici_format",
    "create_training_batch",
    
    # Advanced standardization
    "compute_standardization_params",
    "apply_standardization", 
    "reverse_standardization",
    "StandardizationType",
    "StandardizationParams",
    
    # Validation
    "validate_conversion_inputs",
    "validate_training_batch_inputs",
    "validate_avici_data_structure",
    "validate_training_batch_structure",
    
    # Analysis
    "analyze_avici_data",
    "compare_conversions",
    "compute_data_quality_metrics",
    
    # Parent set model (unchanged API)
    "create_parent_set_model", 
    "predict_parent_sets",
    "ParentSetPredictionModel",
    "compute_loss",
    "create_train_step",
    
    # Type aliases
    "SampleList",
    "VariableOrder", 
    "AVICIDataBatch",
]
