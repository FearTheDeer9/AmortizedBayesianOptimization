#!/usr/bin/env python3
"""
Common data structures for training modules.

This module defines shared data structures to avoid circular imports.
"""

from dataclasses import dataclass
from typing import List, Any, Optional
import jax.numpy as jnp


@dataclass(frozen=True)
class TrainingExample:
    """Single training example for surrogate model."""
    observational_data: jnp.ndarray  # [N, d, 3] - N samples, d variables, 3 channels
    expert_posterior: Any  # Expert's posterior over parent sets
    target_variable: str
    variable_order: List[str]
    expert_accuracy: float
    problem_difficulty: str
    parent_sets: List[frozenset]  # List of parent sets
    expert_probs: jnp.ndarray  # Expert probabilities for parent sets


@dataclass(frozen=True)
class TrainingBatchJAX:
    """JAX-optimized training batch for BC surrogate training."""
    observational_data: jnp.ndarray  # [batch_size, N, d, 3]
    expert_probs: jnp.ndarray  # [batch_size, num_parent_sets]
    expert_accuracies: jnp.ndarray  # [batch_size]
    parent_sets: List[List[frozenset]]  # [batch_size][num_parent_sets]
    variable_orders: List[List[str]]  # [batch_size][d]
    target_variables: List[int]  # [batch_size] - indices into variable_orders


@dataclass(frozen=True)
class TrainingMetrics:
    """Training metrics for surrogate models."""
    total_loss: float
    kl_loss: float
    accuracy: float
    step: int
    learning_rate: float
    grad_norm: float
    timestamp: float = 0.0


@dataclass(frozen=True)
class ValidationResults:
    """Validation results for BC training."""
    average_loss: float
    average_accuracy: float
    kl_divergence: float
    uncertainty_calibration: float
    num_samples: int