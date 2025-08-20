"""
Loss functions for joint ACBO training.

This module implements the principled loss functions for joint training:
- Absolute target loss (actual value improvement, not relative)
- Information gain loss (entropy reduction in posterior)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import entropy as scipy_entropy

from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.scm import get_target

logger = logging.getLogger(__name__)


def compute_absolute_target_loss(
    buffer: ExperienceBuffer,
    target_var: str,
    minimize: bool = False
) -> float:
    """
    Compute absolute improvement in target variable value.
    
    This measures the actual change in target value from initial observations
    to final interventions, not relative improvement.
    
    Args:
        buffer: Experience buffer with observations and interventions
        target_var: Name of target variable
        minimize: If True, lower values are better; if False, higher is better
        
    Returns:
        Absolute improvement in target value (positive is good)
    """
    # Get all samples
    all_samples = buffer.get_all_samples()
    
    if len(all_samples) < 2:
        return 0.0
    
    # Get initial observational values
    obs_samples = buffer.get_observations()
    if not obs_samples:
        return 0.0
        
    # Extract target values from observations
    initial_values = []
    for sample in obs_samples[:10]:  # Use first 10 observations
        values = sample.get('values', sample)
        if target_var in values:
            initial_values.append(float(values[target_var]))
    
    if not initial_values:
        logger.warning(f"No initial values found for target {target_var}")
        return 0.0
    
    initial_mean = np.mean(initial_values)
    
    # Get intervention samples
    interventions = buffer.get_interventions()
    if not interventions:
        # No interventions, no improvement
        return 0.0
    
    # Extract final target values from intervention outcomes
    final_values = []
    for intervention, outcome in interventions[-10:]:  # Use last 10 interventions
        values = outcome.get('values', outcome)
        if target_var in values:
            final_values.append(float(values[target_var]))
    
    if not final_values:
        logger.warning(f"No final values found for target {target_var}")
        return 0.0
    
    final_mean = np.mean(final_values)
    
    # Compute absolute improvement
    if minimize:
        improvement = initial_mean - final_mean  # Lower is better
    else:
        improvement = final_mean - initial_mean  # Higher is better
    
    logger.debug(f"Absolute target improvement: {initial_mean:.3f} -> {final_mean:.3f} = {improvement:.3f}")
    
    return improvement


def compute_information_gain(
    posterior_before: Dict[str, Any],
    posterior_after: Dict[str, Any],
    target_var: str
) -> float:
    """
    Compute information gain as entropy reduction in posterior.
    
    Information gain = H(P_before) - H(P_after)
    
    Args:
        posterior_before: Posterior distribution before intervention
        posterior_after: Posterior distribution after intervention
        target_var: Target variable name
        
    Returns:
        Information gain (positive means uncertainty reduced)
    """
    if not posterior_before or not posterior_after:
        return 0.0
    
    # Extract marginal parent probabilities
    probs_before = posterior_before.get('marginal_parent_probs', {})
    probs_after = posterior_after.get('marginal_parent_probs', {})
    
    if not probs_before or not probs_after:
        return 0.0
    
    # Compute entropy for each posterior
    entropy_before = compute_posterior_entropy(probs_before, target_var)
    entropy_after = compute_posterior_entropy(probs_after, target_var)
    
    # Information gain is reduction in entropy
    info_gain = entropy_before - entropy_after
    
    logger.debug(f"Information gain: H_before={entropy_before:.3f}, H_after={entropy_after:.3f}, gain={info_gain:.3f}")
    
    return info_gain


def compute_posterior_entropy(
    marginal_probs: Dict[str, float],
    target_var: str
) -> float:
    """
    Compute entropy of posterior distribution over parent sets.
    
    Args:
        marginal_probs: Marginal probabilities for each potential parent
        target_var: Target variable (to exclude from parents)
        
    Returns:
        Entropy of the posterior distribution
    """
    # Filter out target variable and get parent probabilities
    parent_probs = []
    for var, prob in marginal_probs.items():
        if var != target_var:
            parent_probs.append(prob)
    
    if not parent_probs:
        return 0.0
    
    # Convert to binary distribution for each parent
    # Each parent can be in/out of parent set independently
    total_entropy = 0.0
    
    for p in parent_probs:
        # Binary entropy for each parent: H = -p*log(p) - (1-p)*log(1-p)
        p = np.clip(p, 1e-8, 1 - 1e-8)  # Avoid log(0)
        binary_entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
        total_entropy += binary_entropy
    
    return total_entropy


def compute_joint_policy_loss(
    buffer: ExperienceBuffer,
    target_var: str,
    posteriors: List[Tuple[Optional[Dict], Optional[Dict]]],
    weights: Dict[str, float],
    minimize_target: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Compute combined policy loss for joint training.
    
    Loss = λ₁ * absolute_target_loss + λ₂ * information_gain_loss
    
    Args:
        buffer: Experience buffer
        target_var: Target variable name
        posteriors: List of (before, after) posterior pairs for each intervention
        weights: Loss component weights {'absolute_target': λ₁, 'information_gain': λ₂}
        minimize_target: Whether to minimize (True) or maximize (False) target
        
    Returns:
        Tuple of (total_loss, component_dict)
    """
    # Default weights
    if not weights:
        weights = {
            'absolute_target': 0.7,
            'information_gain': 0.3
        }
    
    # Compute absolute target improvement
    target_improvement = compute_absolute_target_loss(
        buffer, target_var, minimize_target
    )
    
    # Compute average information gain across interventions
    info_gains = []
    for before, after in posteriors:
        if before and after:
            gain = compute_information_gain(before, after, target_var)
            info_gains.append(gain)
    
    avg_info_gain = np.mean(info_gains) if info_gains else 0.0
    
    # Combine losses (negate for maximization -> minimization)
    target_loss = -target_improvement  # Negative because we want to maximize improvement
    info_loss = -avg_info_gain  # Negative because we want to maximize information gain
    
    # Weighted combination
    total_loss = (
        weights.get('absolute_target', 0.7) * target_loss +
        weights.get('information_gain', 0.3) * info_loss
    )
    
    components = {
        'total_loss': total_loss,
        'target_loss': target_loss,
        'info_loss': info_loss,
        'target_improvement': target_improvement,
        'avg_info_gain': avg_info_gain
    }
    
    return total_loss, components


def compute_trajectory_info_gain(
    buffer: ExperienceBuffer,
    target_var: str
) -> float:
    """
    Compute total information gain across a full trajectory.
    
    This measures how much the posterior uncertainty was reduced
    from start to end of the trajectory.
    
    Args:
        buffer: Experience buffer with posteriors
        target_var: Target variable
        
    Returns:
        Total information gain for the trajectory
    """
    # Get all samples with posteriors
    samples_with_posteriors = buffer.get_all_samples_with_posteriors()
    
    if len(samples_with_posteriors) < 2:
        return 0.0
    
    # Get first and last posteriors
    first_posterior = None
    last_posterior = None
    
    # Find first non-None posterior
    for _, posterior in samples_with_posteriors:
        if posterior is not None:
            first_posterior = posterior
            break
    
    # Find last non-None posterior
    for _, posterior in reversed(samples_with_posteriors):
        if posterior is not None:
            last_posterior = posterior
            break
    
    if not first_posterior or not last_posterior:
        return 0.0
    
    # Compute information gain from first to last
    return compute_information_gain(first_posterior, last_posterior, target_var)


def validate_loss_computation(
    buffer: ExperienceBuffer,
    target_var: str,
    posteriors: List[Tuple[Optional[Dict], Optional[Dict]]]
) -> Dict[str, Any]:
    """
    Validate that loss computation is working correctly.
    
    Returns diagnostic information about the losses.
    
    Args:
        buffer: Experience buffer
        target_var: Target variable
        posteriors: Posterior pairs
        
    Returns:
        Dictionary with validation results
    """
    validation = {}
    
    # Check buffer has data
    validation['has_observations'] = len(buffer.get_observations()) > 0
    validation['has_interventions'] = len(buffer.get_interventions()) > 0
    validation['n_samples'] = buffer.size()
    
    # Check target improvement
    target_improvement = compute_absolute_target_loss(buffer, target_var)
    validation['target_improvement'] = float(target_improvement)
    validation['target_improved'] = target_improvement > 0
    
    # Check information gains
    info_gains = []
    for before, after in posteriors:
        if before and after:
            gain = compute_information_gain(before, after, target_var)
            info_gains.append(gain)
    
    validation['n_posteriors'] = len(posteriors)
    validation['n_valid_gains'] = len(info_gains)
    validation['avg_info_gain'] = float(np.mean(info_gains)) if info_gains else 0.0
    validation['max_info_gain'] = float(np.max(info_gains)) if info_gains else 0.0
    validation['min_info_gain'] = float(np.min(info_gains)) if info_gains else 0.0
    
    # Check trajectory info gain
    trajectory_gain = compute_trajectory_info_gain(buffer, target_var)
    validation['trajectory_info_gain'] = float(trajectory_gain)
    
    # Overall health check
    validation['losses_valid'] = (
        validation['has_observations'] and
        validation['has_interventions'] and
        validation['n_valid_gains'] > 0
    )
    
    return validation