"""
Continuous parent set modeling module.

This module provides a continuous alternative to discrete parent set enumeration,
using attention mechanisms and probability distributions for scalable causal discovery.
"""

from .model import ContinuousParentSetPredictionModel
from .structure import DifferentiableStructureLearning
from .sampling import DifferentiableParentSampling
from .integration import create_continuous_surrogate_model

__all__ = [
    "ContinuousParentSetPredictionModel",
    "DifferentiableStructureLearning", 
    "DifferentiableParentSampling",
    "create_continuous_surrogate_model",
]