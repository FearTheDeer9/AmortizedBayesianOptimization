"""
Stub for removed model module.

This module provides minimal compatibility shims for the removed parent set models.
The actual implementation should use ContinuousParentSetPredictionModel.
"""

import warnings
from typing import List, Dict, Any
import haiku as hk
import jax.numpy as jnp


class ParentSetPredictionModel(hk.Module):
    """Stub for removed model class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        warnings.warn(
            "ParentSetPredictionModel was removed. Use ContinuousParentSetPredictionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
    def __call__(self, x, variable_order, target_variable, is_training=False):
        raise NotImplementedError("Use ContinuousParentSetPredictionModel from continuous module")


def create_parent_set_model(*args, **kwargs):
    """Stub for removed factory function."""
    warnings.warn(
        "create_parent_set_model from model.py was removed. Import from continuous module instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from ..continuous.model import ContinuousParentSetPredictionModel
    return ContinuousParentSetPredictionModel(**kwargs)