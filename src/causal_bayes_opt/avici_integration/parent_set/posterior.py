"""
Parent set posterior representation and utilities.

This module provides the core data structure for representing posterior distributions
over parent sets, along with utilities for manipulation and analysis.
"""

# Standard library imports
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, FrozenSet, Optional, Any

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Type aliases
ParentSet = FrozenSet[str]
ParentSetProbs = pyr.PMap[ParentSet, float]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentSetPosterior:
    """
    Immutable representation of a posterior distribution over parent sets.
    
    This is the core output format for parent set prediction models, providing
    a clean interface for working with predicted parent set probabilities.
    
    Attributes:
        target_variable: Name of the target variable
        parent_set_probs: Mapping from parent sets to their probabilities
        uncertainty: Entropy-based uncertainty measure (in nats)
        top_k_sets: Most likely parent sets with their probabilities
        metadata: Additional information about the prediction
    """
    target_variable: str
    parent_set_probs: ParentSetProbs
    uncertainty: float
    top_k_sets: List[Tuple[ParentSet, float]]
    metadata: pyr.PMap[str, Any] = pyr.m()
    
    def __post_init__(self):
        """Validate the posterior distribution."""
        _validate_posterior_consistency(self)


# Core utility functions
def create_parent_set_posterior(
    target_variable: str,
    parent_sets: List[ParentSet], 
    probabilities: jnp.ndarray,
    metadata: Optional[Dict[str, Any]] = None
) -> ParentSetPosterior:
    """
    Factory function for creating a ParentSetPosterior from model outputs.
    
    Args:
        target_variable: Name of the target variable
        parent_sets: List of parent sets from model prediction
        probabilities: Corresponding probabilities (should sum to ~1.0)
        metadata: Optional metadata about the prediction
        
    Returns:
        Validated ParentSetPosterior object
        
    Raises:
        ValueError: If inputs are inconsistent or invalid
        
    Example:
        >>> parent_sets = [frozenset(), frozenset(['X']), frozenset(['X', 'Z'])]
        >>> probs = jnp.array([0.1, 0.3, 0.6])
        >>> posterior = create_parent_set_posterior('Y', parent_sets, probs)
        >>> posterior.get_most_likely_parents()
        frozenset(['X', 'Z'])
    """
    # Validate inputs
    if not parent_sets:
        raise ValueError("Parent sets list cannot be empty")
    
    if len(parent_sets) != len(probabilities):
        raise ValueError(f"Mismatch: {len(parent_sets)} parent sets, {len(probabilities)} probabilities")
    
    if not target_variable:
        raise ValueError("Target variable cannot be empty")
    
    # Validate probabilities
    prob_sum = float(jnp.sum(probabilities))
    if not (0.95 <= prob_sum <= 1.05):  # Allow small numerical errors
        logger.warning(f"Probabilities sum to {prob_sum:.6f}, not ~1.0")
    
    # Ensure probabilities are non-negative
    if jnp.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")
    
    # Create probability mapping
    parent_set_probs = pyr.pmap({
        ps: float(prob) for ps, prob in zip(parent_sets, probabilities)
    })
    
    # Compute uncertainty (entropy in nats)
    # H = -sum(p * log(p)) where log is natural logarithm
    # Clamp to ensure non-negative due to numerical precision
    uncertainty = float(jnp.maximum(0.0, -jnp.sum(probabilities * jnp.log(probabilities + 1e-12))))
    
    # Create top-k list (sorted by probability, descending)
    sorted_indices = jnp.argsort(probabilities)[::-1]  # Descending order
    top_k_sets = [
        (parent_sets[int(idx)], float(probabilities[idx])) 
        for idx in sorted_indices
    ]
    
    # Create metadata
    if metadata is None:
        metadata = {}
    
    metadata_map = pyr.pmap({
        **metadata,
        'n_parent_sets': len(parent_sets),
        'probability_sum': prob_sum,
        'max_probability': float(jnp.max(probabilities)),
        'min_probability': float(jnp.min(probabilities))
    })
    
    return ParentSetPosterior(
        target_variable=target_variable,
        parent_set_probs=parent_set_probs,
        uncertainty=uncertainty,
        top_k_sets=top_k_sets,
        metadata=metadata_map
    )


def get_most_likely_parents(posterior: ParentSetPosterior, k: int = 1) -> List[ParentSet]:
    """
    Get the k most likely parent sets from the posterior.
    
    Args:
        posterior: The parent set posterior
        k: Number of top parent sets to return
        
    Returns:
        List of k most likely parent sets, ordered by probability (descending)
        
    Example:
        >>> top_3 = get_most_likely_parents(posterior, k=3)
        >>> print(f"Most likely: {top_3[0]}")
    """
    if k <= 0:
        raise ValueError("k must be positive")
    
    k = min(k, len(posterior.top_k_sets))
    return [ps for ps, _ in posterior.top_k_sets[:k]]


def get_parent_set_probability(posterior: ParentSetPosterior, parent_set: ParentSet) -> float:
    """
    Get the probability of a specific parent set.
    
    Args:
        posterior: The parent set posterior
        parent_set: The parent set to query
        
    Returns:
        Probability of the parent set (0.0 if not in posterior)
        
    Example:
        >>> prob = get_parent_set_probability(posterior, frozenset(['X', 'Z']))
        >>> print(f"P(parents={{X,Z}}) = {prob:.3f}")
    """
    return float(posterior.parent_set_probs.get(parent_set, 0.0))


def get_marginal_parent_probabilities(posterior: ParentSetPosterior, all_variables: List[str]) -> Dict[str, float]:
    """
    Compute marginal probabilities that each variable is a parent.
    
    Args:
        posterior: The parent set posterior
        all_variables: List of all possible parent variables
        
    Returns:
        Dictionary mapping variable names to marginal parent probabilities
        
    Example:
        >>> marginals = get_marginal_parent_probabilities(posterior, ['X', 'Y', 'Z'])
        >>> print(f"P(X is parent) = {marginals['X']:.3f}")
    """
    marginals = {var: 0.0 for var in all_variables}
    
    # Remove target variable from candidates (can't be parent of itself)
    candidates = [var for var in all_variables if var != posterior.target_variable]
    marginals = {var: 0.0 for var in candidates}
    
    # Sum probabilities over all parent sets that contain each variable
    for parent_set, prob in posterior.parent_set_probs.items():
        for var in parent_set:
            if var in marginals:
                marginals[var] += prob
    
    return marginals


def compute_posterior_entropy(posterior: ParentSetPosterior) -> float:
    """
    Compute the entropy of the posterior distribution.
    
    Args:
        posterior: The parent set posterior
        
    Returns:
        Entropy in nats (natural units)
        
    Note:
        This is the same as the uncertainty field, but provided as a
        standalone function for clarity.
    """
    return posterior.uncertainty


def compute_posterior_concentration(posterior: ParentSetPosterior) -> float:
    """
    Compute concentration measure (inverse of effective number of parent sets).
    
    Args:
        posterior: The parent set posterior
        
    Returns:
        Concentration measure in [0, 1], where 1 = all probability on one set
        
    Note:
        This is computed as 1 - exp(-entropy), which gives an intuitive
        measure of how concentrated the posterior is.
    """
    return 1.0 - jnp.exp(-posterior.uncertainty)


def filter_parent_sets_by_probability(
    posterior: ParentSetPosterior, 
    min_probability: float = 0.01
) -> ParentSetPosterior:
    """
    Create a filtered posterior containing only high-probability parent sets.
    
    Args:
        posterior: The original posterior
        min_probability: Minimum probability threshold
        
    Returns:
        New posterior with filtered parent sets (probabilities renormalized)
        
    Example:
        >>> filtered = filter_parent_sets_by_probability(posterior, min_probability=0.1)
        >>> print(f"Filtered to {len(filtered.parent_set_probs)} parent sets")
    """
    # Filter parent sets
    filtered_items = [
        (ps, prob) for ps, prob in posterior.parent_set_probs.items()
        if prob >= min_probability
    ]
    
    if not filtered_items:
        raise ValueError(f"No parent sets have probability >= {min_probability}")
    
    # Extract and renormalize
    filtered_parent_sets, filtered_probs = zip(*filtered_items)
    filtered_probs = jnp.array(filtered_probs)
    filtered_probs = filtered_probs / jnp.sum(filtered_probs)  # Renormalize
    
    # Create new metadata
    new_metadata = posterior.metadata.set('filtered_threshold', min_probability)
    new_metadata = new_metadata.set('original_n_parent_sets', len(posterior.parent_set_probs))
    
    return create_parent_set_posterior(
        target_variable=posterior.target_variable,
        parent_sets=list(filtered_parent_sets),
        probabilities=filtered_probs,
        metadata=dict(new_metadata)
    )


def compare_posteriors(
    posterior1: ParentSetPosterior, 
    posterior2: ParentSetPosterior
) -> Dict[str, float]:
    """
    Compare two posteriors over the same target variable.
    
    Args:
        posterior1: First posterior
        posterior2: Second posterior
        
    Returns:
        Dictionary with comparison metrics
        
    Raises:
        ValueError: If posteriors have different target variables
        
    Example:
        >>> metrics = compare_posteriors(predicted_posterior, true_posterior)
        >>> print(f"KL divergence: {metrics['kl_divergence']:.3f}")
    """
    if posterior1.target_variable != posterior2.target_variable:
        raise ValueError(
            f"Cannot compare posteriors for different targets: "
            f"'{posterior1.target_variable}' vs '{posterior2.target_variable}'"
        )
    
    # Get all parent sets from both posteriors
    all_parent_sets = set(posterior1.parent_set_probs.keys()) | set(posterior2.parent_set_probs.keys())
    
    # Create probability arrays
    probs1 = jnp.array([
        posterior1.parent_set_probs.get(ps, 0.0) for ps in all_parent_sets
    ])
    probs2 = jnp.array([
        posterior2.parent_set_probs.get(ps, 0.0) for ps in all_parent_sets
    ])
    
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    probs1_safe = probs1 + eps
    probs2_safe = probs2 + eps
    
    # Compute metrics
    kl_div_1_to_2 = float(jnp.sum(probs1 * jnp.log(probs1_safe / probs2_safe)))
    kl_div_2_to_1 = float(jnp.sum(probs2 * jnp.log(probs2_safe / probs1_safe)))
    
    # Symmetric KL divergence
    symmetric_kl = (kl_div_1_to_2 + kl_div_2_to_1) / 2.0
    
    # Total variation distance
    tv_distance = float(jnp.sum(jnp.abs(probs1 - probs2)) / 2.0)
    
    # Overlap (sum of min probabilities)
    overlap = float(jnp.sum(jnp.minimum(probs1, probs2)))
    
    return {
        'kl_divergence_1_to_2': kl_div_1_to_2,
        'kl_divergence_2_to_1': kl_div_2_to_1,
        'symmetric_kl_divergence': symmetric_kl,
        'total_variation_distance': tv_distance,
        'overlap': overlap,
        'n_common_parent_sets': len(set(posterior1.parent_set_probs.keys()) & 
                                    set(posterior2.parent_set_probs.keys()))
    }


def summarize_posterior(posterior: ParentSetPosterior) -> Dict[str, Any]:
    """
    Create a human-readable summary of the posterior.
    
    Args:
        posterior: The parent set posterior
        
    Returns:
        Dictionary with summary information
        
    Example:
        >>> summary = summarize_posterior(posterior)
        >>> print(f"Target: {summary['target_variable']}")
        >>> print(f"Most likely parents: {summary['most_likely_parents']}")
    """
    # Get top parent set
    most_likely_ps, most_likely_prob = posterior.top_k_sets[0]
    
    # Compute marginals if we have variable information
    marginals = None
    if 'all_variables' in posterior.metadata:
        all_vars = list(posterior.metadata['all_variables'])
        marginals = get_marginal_parent_probabilities(posterior, all_vars)
    
    return {
        'target_variable': posterior.target_variable,
        'n_parent_sets': len(posterior.parent_set_probs),
        'uncertainty_nats': posterior.uncertainty,
        'uncertainty_bits': posterior.uncertainty / jnp.log(2),  # Convert to bits
        'concentration': compute_posterior_concentration(posterior),
        'most_likely_parents': set(most_likely_ps) if most_likely_ps else set(),
        'most_likely_probability': most_likely_prob,
        'marginal_parent_probabilities': marginals,
        'probability_mass_top_3': sum(prob for _, prob in posterior.top_k_sets[:3]),
        'effective_n_parent_sets': jnp.exp(posterior.uncertainty),
        'metadata': dict(posterior.metadata)
    }


# Validation functions
def _validate_posterior_consistency(posterior: ParentSetPosterior) -> None:
    """
    Validate internal consistency of a ParentSetPosterior.
    
    Args:
        posterior: The posterior to validate
        
    Raises:
        ValueError: If posterior is inconsistent
    """
    # Check that target variable is not empty
    if not posterior.target_variable:
        raise ValueError("Target variable cannot be empty")
    
    # Check that we have parent sets
    if not posterior.parent_set_probs:
        raise ValueError("Must have at least one parent set")
    
    # Check probability consistency
    total_prob = sum(posterior.parent_set_probs.values())
    if not (0.95 <= total_prob <= 1.05):
        raise ValueError(f"Probabilities sum to {total_prob:.6f}, not ~1.0")
    
    # Check that all probabilities are non-negative
    for ps, prob in posterior.parent_set_probs.items():
        if prob < 0:
            raise ValueError(f"Negative probability {prob} for parent set {ps}")
    
    # Check top_k_sets consistency
    if len(posterior.top_k_sets) != len(posterior.parent_set_probs):
        raise ValueError("top_k_sets length doesn't match parent_set_probs")
    
    # Check that target variable is not in any parent set
    for ps in posterior.parent_set_probs.keys():
        if posterior.target_variable in ps:
            raise ValueError(
                f"Target variable '{posterior.target_variable}' cannot be in its own parent set: {ps}"
            )
    
    # Check uncertainty bounds (should be non-negative and bounded by log(n))
    if posterior.uncertainty < 0:
        raise ValueError(f"Uncertainty cannot be negative: {posterior.uncertainty}")
    
    max_entropy = jnp.log(len(posterior.parent_set_probs))
    if posterior.uncertainty > max_entropy + 0.01:  # Small tolerance for numerical errors
        raise ValueError(
            f"Uncertainty {posterior.uncertainty:.6f} exceeds maximum possible "
            f"{max_entropy:.6f} for {len(posterior.parent_set_probs)} parent sets"
        )
