"""
AVICI Integration Module - Clean Public API

Provides functional interface for converting SCM data to AVICI format.
"""

# Main user-facing functions
from .core.conversion import (
    samples_to_avici_format_validated as samples_to_avici_format,
    create_training_batch_validated as create_training_batch,
    get_variable_order_from_scm,
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
    validate_data_conversion,
)

# Analysis utilities
from .utils.analysis import (
    analyze_avici_data,
    compare_data_conversions,
    compute_data_quality_metrics,
    reconstruct_samples_from_avici_data,
)

# Parent set functionality
from .parent_set import (
    create_parent_set_model,
    predict_parent_posterior,
    ParentSetPredictionModel,
    ParentSetPosterior,
    create_parent_set_posterior,
    get_most_likely_parents,
    get_parent_set_probability,
    get_marginal_parent_probabilities,
    compute_posterior_entropy,
    compute_posterior_concentration,
    filter_parent_sets_by_probability,
    compare_posteriors,
    summarize_posterior,
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
    "get_variable_order_from_scm",
    
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
    "validate_data_conversion",
    
    # Analysis
    "analyze_avici_data",
    "compare_data_conversions",
    "compute_data_quality_metrics",
    "reconstruct_samples_from_avici_data",
    
    # Parent set model
    "create_parent_set_model", 
    "predict_parent_posterior",
    "ParentSetPredictionModel",
    "ParentSetPosterior",
    "create_parent_set_posterior",
    "get_most_likely_parents",
    "get_parent_set_probability", 
    "get_marginal_parent_probabilities",
    "compute_posterior_entropy",
    "compute_posterior_concentration",
    "filter_parent_sets_by_probability",
    "compare_posteriors",
    "summarize_posterior",
    "compute_loss",
    "create_train_step",
    
    # Type aliases
    "SampleList",
    "VariableOrder", 
    "AVICIDataBatch",
]
