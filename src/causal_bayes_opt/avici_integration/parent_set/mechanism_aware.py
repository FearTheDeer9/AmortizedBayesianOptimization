"""
Stub for removed mechanism_aware module.

This module provides minimal compatibility shims for the removed mechanism-aware models.
"""

import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import jax.numpy as jnp


class MechanismType(Enum):
    """Stub for mechanism types."""
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    UNKNOWN = "unknown"


@dataclass
class MechanismPrediction:
    """Stub for mechanism prediction class."""
    mechanism_type: MechanismType
    confidence: float
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class MechanismAwareConfig:
    """Stub for mechanism config."""
    predict_mechanisms: bool = False
    mechanism_dim: int = 64


class ModularParentSetModel:
    """Stub for removed model class."""
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ModularParentSetModel was removed. Use ContinuousParentSetPredictionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


def create_modular_parent_set_model(*args, **kwargs):
    """Stub for removed factory function."""
    warnings.warn(
        "create_modular_parent_set_model was removed. Use ContinuousParentSetPredictionModel instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


def predict_with_mechanisms(*args, **kwargs):
    """Stub for removed prediction function."""
    warnings.warn(
        "predict_with_mechanisms was removed. Use standard prediction methods.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Use prediction methods from continuous module")


def get_all_mechanism_types():
    """Return all mechanism types."""
    return list(MechanismType)


def validate_mechanism_types(types: List[MechanismType]) -> bool:
    """Validate mechanism types."""
    return all(isinstance(t, MechanismType) for t in types)


def create_enhanced_config(**kwargs):
    """Stub for removed config function."""
    return MechanismAwareConfig(**kwargs)


def compare_model_outputs(*args, **kwargs):
    """Stub for removed comparison function."""
    warnings.warn(
        "compare_model_outputs was removed.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Function was removed")