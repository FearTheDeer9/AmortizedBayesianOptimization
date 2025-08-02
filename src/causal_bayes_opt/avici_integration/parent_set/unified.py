"""
Stub for removed unified module.

This module provides minimal compatibility shims for the removed unified models.
The actual implementation should use ContinuousParentSetPredictionModel.
"""

import warnings
from typing import List, Dict, Any, Optional
import jax.numpy as jnp
from dataclasses import dataclass

# Import the actual implementation
from ..continuous.model import ContinuousParentSetPredictionModel


# Stub classes for compatibility
@dataclass
class TargetAwareConfig:
    """Stub for removed config class."""
    layers: int = 4
    dim: int = 128
    max_parent_size: int = 5
    
    
class JAXUnifiedParentSetModel:
    """Stub for removed model class."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "JAXUnifiedParentSetModel was removed. Use ContinuousParentSetPredictionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


class JAXUnifiedParentSetModelWrapper:
    """Stub for removed wrapper class."""
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "JAXUnifiedParentSetModelWrapper was removed. Use ContinuousParentSetPredictionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


def create_jax_unified_parent_set_model(*args, **kwargs):
    """Stub for removed factory function."""
    warnings.warn(
        "create_jax_unified_parent_set_model was removed. Use ContinuousParentSetPredictionModel instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


def create_parent_set_model(*args, **kwargs):
    """Create parent set model - returns continuous model."""
    return ContinuousParentSetPredictionModel(**kwargs)


def create_structure_only_config(**kwargs):
    """Stub for removed config function."""
    return TargetAwareConfig(**kwargs)


def create_mechanism_aware_config(**kwargs):
    """Stub for removed config function."""
    return TargetAwareConfig(**kwargs)


def predict_with_jax_unified_model(*args, **kwargs):
    """Stub for removed prediction function."""
    warnings.warn(
        "predict_with_jax_unified_model was removed. Use predict methods from continuous module.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Use prediction methods from continuous module")