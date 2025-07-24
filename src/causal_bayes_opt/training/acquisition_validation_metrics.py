#!/usr/bin/env python3
"""
Acquisition Policy Validation Metrics

Provides meaningful validation metrics for acquisition policy training that go beyond
simple exact-match accuracy. These metrics are designed to evaluate intervention 
quality in causal discovery contexts where multiple variables could be valid choices.

Key metrics:
1. Top-k accuracy: Is expert choice in top-k predictions?
2. Mean reciprocal rank: How highly ranked is expert choice?
3. Diversity metrics: How diverse are intervention choices?
4. Exploration coverage: What % of variables explored?
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict

import jax
import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

logger = logging.getLogger(__name__)


def top_k_accuracy(policy_logits: jnp.ndarray, expert_choices: jnp.ndarray, k: int = 3) -> float:
    """
    Compute top-k accuracy: is expert choice in top-k predicted choices?
    
    Args:
        policy_logits: [batch_size, n_variables] - policy output logits
        expert_choices: [batch_size] - expert variable indices
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as float between 0 and 1
    """
    batch_size, n_vars = policy_logits.shape
    k = min(k, n_vars)  # Can't have k > n_vars
    
    # Get top-k predictions for each sample
    top_k_indices = jnp.argsort(policy_logits, axis=1)[:, -k:]  # Last k are highest
    
    # Check if expert choice is in top-k for each sample
    expert_choices_expanded = expert_choices[:, None]  # [batch_size, 1]
    matches = jnp.any(top_k_indices == expert_choices_expanded, axis=1)  # [batch_size]
    
    return float(jnp.mean(matches))


def mean_reciprocal_rank(policy_logits: jnp.ndarray, expert_choices: jnp.ndarray) -> float:
    """
    Compute mean reciprocal rank of expert choices.
    
    Higher is better. MRR = 1.0 means expert choice always ranked 1st.
    MRR = 0.5 means expert choice ranked 2nd on average.
    
    Args:
        policy_logits: [batch_size, n_variables] - policy output logits  
        expert_choices: [batch_size] - expert variable indices
    
    Returns:
        Mean reciprocal rank as float between 0 and 1
    """
    # Get ranking of each variable (higher logits = lower rank index)
    ranks = jnp.argsort(jnp.argsort(policy_logits, axis=1), axis=1)  # [batch_size, n_vars]
    n_vars = policy_logits.shape[1]
    
    # Convert to 1-based ranking (1 = best, n_vars = worst)
    ranks = n_vars - ranks  # Flip so higher logits get rank 1
    
    # Get rank of expert choice for each sample
    batch_indices = jnp.arange(policy_logits.shape[0])
    expert_ranks = ranks[batch_indices, expert_choices]  # [batch_size]
    
    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / expert_ranks.astype(jnp.float32)
    
    return float(jnp.mean(reciprocal_ranks))


def expert_choice_percentile(policy_logits: jnp.ndarray, expert_choices: jnp.ndarray) -> float:
    """
    Compute what percentile expert choice is in policy distribution.
    
    1.0 = expert choice has highest logit
    0.5 = expert choice is median
    0.0 = expert choice has lowest logit
    
    Args:
        policy_logits: [batch_size, n_variables] - policy output logits
        expert_choices: [batch_size] - expert variable indices
    
    Returns:
        Mean percentile as float between 0 and 1
    """
    batch_size, n_vars = policy_logits.shape
    percentiles = []
    
    for i in range(batch_size):
        expert_logit = policy_logits[i, expert_choices[i]]
        # Count how many variables have lower logit
        lower_count = jnp.sum(policy_logits[i] < expert_logit)
        percentile = lower_count / (n_vars - 1)  # -1 to exclude expert choice itself
        percentiles.append(percentile)
    
    return float(jnp.mean(jnp.array(percentiles)))


def intervention_diversity_score(predicted_variables: jnp.ndarray, window_size: int = 10) -> float:
    """
    Measure diversity of recent intervention choices.
    
    Higher is better. 1.0 = all recent choices were different variables.
    
    Args:
        predicted_variables: [sequence_length] - sequence of predicted variable indices
        window_size: Number of recent predictions to consider
    
    Returns:
        Diversity score as float between 0 and 1
    """
    if len(predicted_variables) == 0:
        return 0.0
    
    # Take last window_size predictions
    recent_choices = predicted_variables[-window_size:]
    
    # Count unique choices
    unique_choices = len(jnp.unique(recent_choices))
    max_possible_unique = min(len(recent_choices), window_size)
    
    return float(unique_choices / max_possible_unique) if max_possible_unique > 0 else 0.0


def exploration_coverage(predicted_variables: jnp.ndarray, total_variables: int) -> float:
    """
    Compute percentage of variables that have been explored.
    
    Args:
        predicted_variables: [sequence_length] - sequence of predicted variable indices
        total_variables: Total number of variables in the system
    
    Returns:
        Coverage as float between 0 and 1
    """
    if len(predicted_variables) == 0 or total_variables == 0:
        return 0.0
    
    unique_variables = len(jnp.unique(predicted_variables))
    return float(unique_variables / total_variables)


def compute_diversity_bonus(expert_choice: int, intervention_history: List[int], 
                          decay_factor: float = 0.9, max_history: int = 20) -> float:
    """
    Compute diversity bonus for expert choice based on intervention history.
    
    Higher bonus for variables that haven't been chosen recently.
    Used to weight the cross-entropy loss.
    
    Args:
        expert_choice: Variable index chosen by expert
        intervention_history: List of recently chosen variable indices
        decay_factor: How quickly to discount older interventions
        max_history: Maximum history length to consider
    
    Returns:
        Diversity bonus as float >= 0
    """
    if not intervention_history:
        return 1.0  # Maximum bonus for first choice
    
    # Consider only recent history
    recent_history = intervention_history[-max_history:]
    
    # Compute weighted count of expert choice in recent history
    weighted_count = 0.0
    for i, choice in enumerate(reversed(recent_history)):
        weight = decay_factor ** i  # More recent = higher weight
        if choice == expert_choice:
            weighted_count += weight
    
    # Bonus is inversely related to how often this choice was made recently
    # Add 1 to avoid division by zero and to give some bonus even for repeated choices
    diversity_bonus = 1.0 / (1.0 + weighted_count)
    
    return diversity_bonus


def compute_comprehensive_validation_metrics(
    policy_logits: jnp.ndarray,
    expert_choices: jnp.ndarray,
    intervention_history: Optional[List[int]] = None,
    total_variables: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute comprehensive validation metrics for acquisition policy.
    
    Args:
        policy_logits: [batch_size, n_variables] - policy output logits
        expert_choices: [batch_size] - expert variable indices  
        intervention_history: Historical sequence of intervention choices
        total_variables: Total number of variables in system
    
    Returns:
        Dictionary with validation metrics
    """
    metrics = {}
    
    # Core ranking metrics
    metrics['top_1_accuracy'] = top_k_accuracy(policy_logits, expert_choices, k=1)
    metrics['top_3_accuracy'] = top_k_accuracy(policy_logits, expert_choices, k=3)
    metrics['top_5_accuracy'] = top_k_accuracy(policy_logits, expert_choices, k=5)
    metrics['mean_reciprocal_rank'] = mean_reciprocal_rank(policy_logits, expert_choices)
    metrics['expert_percentile'] = expert_choice_percentile(policy_logits, expert_choices)
    
    # Diversity and exploration metrics
    if intervention_history is not None:
        predicted_vars = jnp.argmax(policy_logits, axis=1)  # Get policy's top choices
        all_predictions = list(intervention_history) + list(predicted_vars)
        
        metrics['diversity_score'] = intervention_diversity_score(jnp.array(all_predictions))
        
        if total_variables is not None:
            metrics['exploration_coverage'] = exploration_coverage(
                jnp.array(all_predictions), total_variables
            )
    
    # Policy confidence metrics
    max_logits = jnp.max(policy_logits, axis=1)
    metrics['mean_confidence'] = float(jnp.mean(max_logits))
    metrics['confidence_std'] = float(jnp.std(max_logits))
    
    # Policy entropy (lower = more confident/certain)
    policy_probs = jax.nn.softmax(policy_logits, axis=1)
    entropies = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-8), axis=1)
    metrics['mean_entropy'] = float(jnp.mean(entropies))
    
    return metrics


def log_validation_metrics(metrics: Dict[str, float], epoch: int, level: str = "") -> None:
    """
    Log validation metrics in a readable format.
    
    Args:
        metrics: Dictionary of validation metrics
        epoch: Current epoch number
        level: Optional difficulty level string
    """
    level_str = f" (Level: {level})" if level else ""
    logger.info(f"Validation Metrics - Epoch {epoch}{level_str}:")
    
    # Group metrics for better readability
    ranking_metrics = {k: v for k, v in metrics.items() 
                      if k in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 
                              'mean_reciprocal_rank', 'expert_percentile']}
    
    exploration_metrics = {k: v for k, v in metrics.items()
                          if k in ['diversity_score', 'exploration_coverage']}
    
    confidence_metrics = {k: v for k, v in metrics.items()
                         if k in ['mean_confidence', 'confidence_std', 'mean_entropy']}
    
    # Log each group
    if ranking_metrics:
        logger.info("  Ranking Metrics:")
        for name, value in ranking_metrics.items():
            logger.info(f"    {name}: {value:.4f}")
    
    if exploration_metrics:
        logger.info("  Exploration Metrics:")
        for name, value in exploration_metrics.items():
            logger.info(f"    {name}: {value:.4f}")
    
    if confidence_metrics:
        logger.info("  Confidence Metrics:")
        for name, value in confidence_metrics.items():
            logger.info(f"    {name}: {value:.4f}")


def create_validation_summary(metrics_history: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Create summary of validation metrics over training history.
    
    Args:
        metrics_history: List of metric dictionaries from each epoch
    
    Returns:
        Summary statistics and trends
    """
    if not metrics_history:
        return {}
    
    summary = {}
    
    # Get all metric names
    all_metrics = set()
    for epoch_metrics in metrics_history:
        all_metrics.update(epoch_metrics.keys())
    
    # Compute summary statistics for each metric
    for metric_name in all_metrics:
        values = []
        for epoch_metrics in metrics_history:
            if metric_name in epoch_metrics:
                values.append(epoch_metrics[metric_name])
        
        if values:
            summary[metric_name] = {
                'final': values[-1],
                'best': max(values),
                'worst': min(values),
                'mean': onp.mean(values),
                'std': onp.std(values),
                'trend': 'improving' if values[-1] > values[0] else 'declining'
            }
    
    return summary