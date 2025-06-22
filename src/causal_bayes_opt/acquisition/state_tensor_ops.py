"""
JAX-Native Tensor Operations for State Management

This module provides JAX-compiled functions that replace Python loops in state
management operations. These functions operate on pure tensors and can be
compiled for maximum performance.

Key functions:
1. Mechanism confidence computation using tensor operations
2. Optimization progress tracking with vectorized operations  
3. Exploration coverage analysis without Python loops
4. Mechanism insights extraction using tensor indexing
"""

import warnings

warnings.warn(
    "This module is deprecated as of Phase 1.5. "
    "Use causal_bayes_opt.jax_native.operations instead. "
    "See docs/migration/MIGRATION_GUIDE.md for migration instructions. "
    "This module will be removed on 2024-02-01.",
    DeprecationWarning,
    stacklevel=2
)


import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationMetricsTensor:
    """Pure tensor representation of optimization metrics."""
    target_values: jnp.ndarray        # [n_samples] target values over time
    improvement_from_start: float     # Total improvement since start
    recent_improvement: float         # Improvement in recent window
    optimization_rate: float          # Improvement per sample
    stagnation_steps: int            # Steps since last improvement


@dataclass(frozen=True) 
class ExplorationMetricsTensor:
    """Pure tensor representation of exploration metrics."""
    intervention_targets: jnp.ndarray  # [n_interventions, n_vars] binary matrix
    target_coverage_rate: float       # Fraction of variables explored
    intervention_diversity: float     # Entropy-based diversity measure
    unexplored_variables: float       # Fraction of variables not explored


@jax.jit
def compute_mechanism_confidence_tensor(
    mechanism_predictions: jnp.ndarray,    # [n_vars, n_features] mechanism features
    mechanism_types: jnp.ndarray,          # [n_vars] mechanism type indices
    target_idx: int,                       # Index of target variable to exclude
    confidence_weights: jnp.ndarray = None # [n_features] weights for combining features
) -> jnp.ndarray:
    """
    JAX-compiled function to compute mechanism confidence scores.
    
    Args:
        mechanism_predictions: Tensor with mechanism features for each variable
        mechanism_types: Integer indices for mechanism types
        target_idx: Index of target variable (excluded from confidence)
        confidence_weights: Optional weights for combining features
        
    Returns:
        Confidence scores [n_vars] (0 for target variable)
    """
    n_vars = mechanism_predictions.shape[0]
    
    # Default weights if not provided
    if confidence_weights is None:
        confidence_weights = jnp.array([0.4, 0.3, 0.3])  # [effect_magnitude, type_confidence, uncertainty]
    
    # Compute weighted confidence for each variable
    raw_confidence = jnp.sum(mechanism_predictions * confidence_weights[None, :], axis=1)
    
    # Mask out target variable
    mask = jnp.ones(n_vars)
    mask = mask.at[target_idx].set(0.0)
    
    # Apply mask and ensure [0, 1] range
    confidence_scores = jnp.clip(raw_confidence * mask, 0.0, 1.0)
    
    return confidence_scores


@jax.jit
def compute_optimization_progress_tensor(
    target_values: jnp.ndarray,         # [n_samples] target values over time
    current_best: float,                # Current best value
    recent_window: int = 10             # Window size for recent improvement
) -> OptimizationMetricsTensor:
    """
    JAX-compiled function to compute optimization progress metrics.
    
    Args:
        target_values: Array of target values over time
        current_best: Current best observed value
        recent_window: Number of recent samples for improvement calculation
        
    Returns:
        OptimizationMetricsTensor with progress metrics
    """
    n_samples = target_values.shape[0]
    
    # Handle empty case with JAX-compatible operations
    empty_case = n_samples == 0
    
    # Improvement from start (safe indexing)
    initial_value = jnp.where(empty_case, 0.0, target_values[0])
    improvement_from_start = current_best - initial_value
    
    # Recent improvement (vectorized computation with safe indexing)
    has_enough_samples = n_samples >= 2 * recent_window
    recent_values = target_values[-recent_window:] 
    prev_values = target_values[-2*recent_window:-recent_window]
    recent_improvement = jnp.where(
        has_enough_samples,
        jnp.mean(recent_values) - jnp.mean(prev_values),
        0.0
    )
    
    # Optimization rate (safe division)
    optimization_rate = jnp.where(
        n_samples > 0,
        improvement_from_start / n_samples,
        0.0
    )
    
    # Stagnation steps (vectorized using cumulative operations)
    value_improvements = target_values - (current_best - 1e-6)  # Small tolerance
    is_improving = value_improvements > 0
    
    # Count steps since last improvement using JAX ops (avoiding if statements)
    indices = jnp.arange(n_samples)
    improving_indices = jnp.where(is_improving, indices, -1)
    has_improvements = jnp.any(is_improving)
    last_improvement_idx = jnp.max(improving_indices)
    stagnation_steps = jnp.where(
        has_improvements,
        n_samples - 1 - last_improvement_idx,
        n_samples
    )
    
    # Return raw values instead of dataclass for JAX compatibility
    return (
        improvement_from_start,
        recent_improvement,
        optimization_rate,
        stagnation_steps
    )


@jax.jit
def compute_exploration_coverage_tensor(
    intervention_matrix: jnp.ndarray,    # [n_interventions, n_vars] binary matrix
    target_idx: int,                     # Index of target variable to exclude
    epsilon: float = 1e-12              # Small value for numerical stability
) -> ExplorationMetricsTensor:
    """
    JAX-compiled function to compute exploration coverage metrics.
    
    Args:
        intervention_matrix: Binary matrix where entry [i,j] = 1 if intervention i targeted variable j
        target_idx: Index of target variable (excluded from coverage)
        epsilon: Small value for log stability
        
    Returns:
        ExplorationMetricsTensor with coverage metrics
    """
    n_interventions, n_vars = intervention_matrix.shape
    
    # Handle empty case with JAX-compatible operations
    empty_case = (n_interventions == 0) | (n_vars <= 1)
    
    # Exclude target variable from analysis
    potential_targets_mask = jnp.ones(n_vars)
    potential_targets_mask = potential_targets_mask.at[target_idx].set(0.0)
    n_potential_targets = jnp.sum(potential_targets_mask)
    no_potential_targets = n_potential_targets == 0
    
    # Coverage rate: fraction of non-target variables that have been intervention targets
    variable_was_targeted = jnp.any(intervention_matrix, axis=0)  # [n_vars]
    targeted_non_targets = variable_was_targeted * potential_targets_mask
    explored_count = jnp.sum(targeted_non_targets)
    
    # Safe division for target coverage rate
    target_coverage_rate = jnp.where(
        no_potential_targets | empty_case,
        jnp.where(no_potential_targets, 1.0, 0.0),  # 1.0 if no potential targets, 0.0 if empty
        explored_count / jnp.maximum(n_potential_targets, 1.0)
    )
    
    # Intervention diversity: entropy of intervention pattern distribution
    # For each intervention, compute a hash-like signature (simplified)
    intervention_signatures = jnp.sum(
        intervention_matrix * jnp.arange(n_vars)[None, :], axis=1
    )  # [n_interventions]
    
    # Compute approximate entropy using signature variance (simpler than unique)
    signature_mean = jnp.mean(intervention_signatures)
    signature_var = jnp.var(intervention_signatures)
    max_variance = jnp.maximum(n_potential_targets, 1.0)
    
    # Normalized diversity based on signature variance
    intervention_diversity = jnp.where(
        empty_case | (n_interventions <= 1),
        0.0,
        jnp.clip(signature_var / max_variance, 0.0, 1.0)
    )
    
    unexplored_variables = 1.0 - target_coverage_rate
    
    # Return raw values instead of dataclass for JAX compatibility
    return (
        target_coverage_rate,
        intervention_diversity,
        unexplored_variables
    )


@jax.jit
def extract_mechanism_insights_tensor(
    mechanism_features: jnp.ndarray,     # [n_vars, n_features] mechanism features
    marginal_parent_probs: jnp.ndarray,  # [n_vars] marginal parent probabilities
    target_idx: int,                     # Index of target variable to exclude
    impact_threshold: float = 0.5,       # Threshold for high impact
    uncertainty_threshold: float = 0.5,  # Threshold for uncertain mechanisms
    parent_prob_threshold: float = 0.3   # Threshold for likely parents
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX-compiled function to extract mechanism insights.
    
    Args:
        mechanism_features: Features for each variable [effect_magnitude, uncertainty, scaling]
        marginal_parent_probs: Probability each variable is a parent
        target_idx: Index of target variable to exclude
        impact_threshold: Minimum effect magnitude for high impact
        uncertainty_threshold: Minimum uncertainty for uncertain mechanism
        parent_prob_threshold: Minimum parent probability to consider
        
    Returns:
        Tuple of (high_impact_mask, uncertain_mechanism_mask) both [n_vars]
    """
    n_vars = mechanism_features.shape[0]
    
    # Extract effect magnitudes and uncertainties
    effect_magnitudes = mechanism_features[:, 0]  # First column
    uncertainties = mechanism_features[:, 1]      # Second column
    
    # High impact variables: large effect magnitude, not target
    high_impact_mask = (effect_magnitudes > impact_threshold)
    
    # Uncertain mechanisms: high uncertainty + likely parent + not target
    uncertain_mask = (
        (uncertainties > uncertainty_threshold) & 
        (marginal_parent_probs > parent_prob_threshold)
    )
    
    # Exclude target variable from both masks
    target_exclusion = jnp.ones(n_vars)
    target_exclusion = target_exclusion.at[target_idx].set(0.0)
    
    high_impact_mask = high_impact_mask * target_exclusion
    uncertain_mask = uncertain_mask * target_exclusion
    
    return high_impact_mask.astype(bool), uncertain_mask.astype(bool)


def create_intervention_matrix_from_samples(
    intervention_indicators: jnp.ndarray,  # [n_samples, n_vars] intervention indicators
    max_interventions: int                  # Maximum number of interventions to track
) -> jnp.ndarray:
    """
    Create intervention matrix from sample indicators (non-JIT due to dynamic shapes).
    
    Args:
        intervention_indicators: Binary matrix from sample history
        max_interventions: Maximum number of intervention steps to track
        
    Returns:
        Intervention matrix [max_interventions, n_vars] for coverage analysis
    """
    n_samples, n_vars = intervention_indicators.shape
    
    # Count total interventions without dynamic filtering
    total_intervention_samples = jnp.sum(jnp.any(intervention_indicators, axis=1))
    
    # Create result matrix with fixed size
    result = jnp.zeros((max_interventions, n_vars))
    
    # Fill with the last max_interventions intervention samples
    # Use a simpler approach: take last max_interventions samples that have any intervention
    intervention_count = 0
    for i in range(n_samples - 1, -1, -1):  # Go backwards through samples
        if jnp.any(intervention_indicators[i]) and intervention_count < max_interventions:
            result = result.at[max_interventions - 1 - intervention_count].set(intervention_indicators[i])
            intervention_count += 1
    
    return result


# JAX-compatible alternative that uses fixed-size operations
def _create_intervention_matrix_impl(
    intervention_indicators: jnp.ndarray,  # [n_samples, n_vars] intervention indicators
    max_interventions: int                  # Maximum number of interventions to track
) -> jnp.ndarray:
    """
    Internal implementation for intervention matrix creation.
    """
    n_samples, n_vars = intervention_indicators.shape
    
    # Create padding tensor with fixed size
    padding = jnp.zeros((max_interventions, n_vars))
    
    # Select the last min(n_samples, max_interventions) samples
    take_size = min(n_samples, max_interventions)
    start_idx = max(0, n_samples - take_size)
    
    # Extract the samples we want
    selected_samples = intervention_indicators[start_idx:start_idx + take_size]
    
    # Place at the end of the result matrix
    if take_size > 0:
        result = padding.at[-take_size:].set(selected_samples)
    else:
        result = padding
    
    return result

# Create JIT-compiled version with static max_interventions
create_intervention_matrix_fixed = jax.jit(_create_intervention_matrix_impl, static_argnums=(1,))


def convert_state_to_tensor_operations(
    samples_history: jnp.ndarray,        # [n_samples, n_vars, 3] from prepare_samples_for_history_jax
    mechanism_features: jnp.ndarray,     # [n_vars, n_features] from tensor_features
    marginal_parent_probs: jnp.ndarray,  # [n_vars] marginal probabilities
    target_idx: int,                     # Index of target variable
    current_best: float                  # Current best target value
) -> Dict[str, Any]:
    """
    High-level function to convert state data to tensor operations.
    
    This function coordinates all tensor-based state computations and returns
    the results in a format compatible with existing interfaces.
    
    Args:
        samples_history: Full sample history in tensor format
        mechanism_features: Mechanism features from tensor_features module
        marginal_parent_probs: Marginal parent probabilities
        target_idx: Index of target variable  
        current_best: Current best observed target value
        
    Returns:
        Dictionary with all state metrics computed using JAX operations
    """
    # Extract target values over time (vectorized)
    target_values = samples_history[:, target_idx, 0]  # [n_samples] values for target variable
    
    # Compute optimization progress using JAX
    opt_improvement, opt_recent, opt_rate, opt_stagnation = compute_optimization_progress_tensor(target_values, current_best)
    
    # Create intervention matrix for exploration analysis
    intervention_indicators = samples_history[:, :, 1]  # [n_samples, n_vars] intervention indicators
    intervention_matrix = create_intervention_matrix_fixed(
        intervention_indicators, max_interventions=100
    )
    
    # Compute exploration coverage using JAX
    exp_coverage, exp_diversity, exp_unexplored = compute_exploration_coverage_tensor(intervention_matrix, target_idx)
    
    # Extract mechanism insights using JAX
    high_impact_mask, uncertain_mask = extract_mechanism_insights_tensor(
        mechanism_features, marginal_parent_probs, target_idx
    )
    
    # Compute mechanism confidence using JAX
    confidence_scores = compute_mechanism_confidence_tensor(
        mechanism_features, 
        jnp.zeros(len(marginal_parent_probs), dtype=jnp.int32),  # Placeholder for mechanism types
        target_idx
    )
    
    return {
        'optimization_progress': {
            'improvement_from_start': float(opt_improvement),
            'recent_improvement': float(opt_recent),
            'optimization_rate': float(opt_rate),
            'stagnation_steps': int(opt_stagnation)
        },
        'exploration_coverage': {
            'target_coverage_rate': float(exp_coverage),
            'intervention_diversity': float(exp_diversity),
            'unexplored_variables': float(exp_unexplored)
        },
        'mechanism_insights': {
            'high_impact_mask': high_impact_mask,
            'uncertain_mask': uncertain_mask,
            'confidence_scores': confidence_scores
        },
        'tensor_data': {
            'target_values': target_values,
            'intervention_matrix': intervention_matrix,
            'mechanism_features': mechanism_features
        }
    }


# Factory function for creating tensor-based state operations
def create_tensor_state_computer(
    max_history_size: int = 100,
    max_interventions: int = 100
):
    """
    Create a configured tensor-based state computation function.
    
    Args:
        max_history_size: Maximum number of samples to track
        max_interventions: Maximum number of interventions to track
        
    Returns:
        Configured state computation function
    """
    def compute_state_metrics(
        samples_history: jnp.ndarray,
        mechanism_features: jnp.ndarray,
        marginal_parent_probs: jnp.ndarray,
        target_idx: int,
        current_best: float
    ):
        return convert_state_to_tensor_operations(
            samples_history, mechanism_features, marginal_parent_probs,
            target_idx, current_best
        )
    
    return compute_state_metrics