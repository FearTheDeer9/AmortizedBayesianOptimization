"""
Pure utility functions for working with AcquisitionState objects.

This module contains simple, pure functions that extract information from
AcquisitionState objects without any external dependencies. These functions
follow functional programming principles and have no side effects.
"""

from typing import Dict, List, Any

from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState


def get_state_uncertainty_bits(state: AcquisitionState) -> float:
    """
    Get uncertainty in bits (for interpretability).
    
    Args:
        state: Acquisition state
        
    Returns:
        Uncertainty in bits
    """
    return state.uncertainty_bits


def get_state_uncertainty_nats(state: AcquisitionState) -> float:
    """
    Get uncertainty in nats (natural logarithm base).
    
    Args:
        state: Acquisition state
        
    Returns:
        Uncertainty in nats
    """
    return state.posterior.uncertainty


def get_state_optimization_progress(state: AcquisitionState) -> Dict[str, float]:
    """
    Get optimization progress metrics.
    
    Args:
        state: Acquisition state
        
    Returns:
        Dictionary with optimization progress metrics
    """
    return state.get_optimization_progress()


def get_state_exploration_coverage(state: AcquisitionState) -> Dict[str, float]:
    """
    Get exploration coverage metrics.
    
    Args:
        state: Acquisition state
        
    Returns:
        Dictionary with exploration coverage metrics
    """
    return state.get_exploration_coverage()


def get_state_marginal_probabilities(state: AcquisitionState) -> Dict[str, float]:
    """
    Get marginal parent probabilities for all variables.
    
    Args:
        state: Acquisition state
        
    Returns:
        Dictionary mapping variable names to marginal parent probabilities
    """
    return state.marginal_parent_probs


def get_state_best_value(state: AcquisitionState) -> float:
    """
    Get the current best value for the target variable.
    
    Args:
        state: Acquisition state
        
    Returns:
        Best observed value for the target variable
    """
    return state.best_value


def get_state_target_variable(state: AcquisitionState) -> str:
    """
    Get the target variable name.
    
    Args:
        state: Acquisition state
        
    Returns:
        Name of the target variable
    """
    return state.current_target


def get_state_step(state: AcquisitionState) -> int:
    """
    Get the current step number.
    
    Args:
        state: Acquisition state
        
    Returns:
        Current step number
    """
    return state.step


def get_state_buffer_size(state: AcquisitionState) -> int:
    """
    Get the total number of samples in the buffer.
    
    Args:
        state: Acquisition state
        
    Returns:
        Total number of samples
    """
    return state.buffer_statistics.total_samples


def get_state_observation_count(state: AcquisitionState) -> int:
    """
    Get the number of observational samples.
    
    Args:
        state: Acquisition state
        
    Returns:
        Number of observational samples
    """
    return state.buffer_statistics.num_observations


def get_state_intervention_count(state: AcquisitionState) -> int:
    """
    Get the number of interventional samples.
    
    Args:
        state: Acquisition state
        
    Returns:
        Number of interventional samples
    """
    return state.buffer_statistics.num_interventions


def get_state_unique_variables(state: AcquisitionState) -> int:
    """
    Get the number of unique variables in the buffer.
    
    Args:
        state: Acquisition state
        
    Returns:
        Number of unique variables
    """
    return state.buffer_statistics.unique_variables


def get_most_likely_parents(state: AcquisitionState) -> frozenset[str]:
    """
    Get the most likely parent set for the target variable.
    
    Args:
        state: Acquisition state
        
    Returns:
        Most likely parent set (empty frozenset if none available)
    """
    if state.posterior.top_k_sets:
        return state.posterior.top_k_sets[0][0]
    return frozenset()


def get_most_likely_parent_probability(state: AcquisitionState) -> float:
    """
    Get the probability of the most likely parent set.
    
    Args:
        state: Acquisition state
        
    Returns:
        Probability of the most likely parent set (0.0 if none available)
    """
    if state.posterior.top_k_sets:
        return state.posterior.top_k_sets[0][1]
    return 0.0


def get_top_k_parent_sets(state: AcquisitionState, k: int = 3) -> List[tuple[frozenset[str], float]]:
    """
    Get the top k most likely parent sets with their probabilities.
    
    Args:
        state: Acquisition state
        k: Number of top parent sets to return
        
    Returns:
        List of (parent_set, probability) tuples
    """
    return state.posterior.top_k_sets[:k]


def is_state_highly_uncertain(state: AcquisitionState, threshold_bits: float = 2.0) -> bool:
    """
    Check if the state has high uncertainty about the causal structure.
    
    Args:
        state: Acquisition state
        threshold_bits: Uncertainty threshold in bits
        
    Returns:
        True if uncertainty is above threshold
    """
    return state.uncertainty_bits > threshold_bits


def is_state_converged(state: AcquisitionState, uncertainty_threshold: float = 0.5) -> bool:
    """
    Check if the state has converged (low uncertainty).
    
    Args:
        state: Acquisition state
        uncertainty_threshold: Uncertainty threshold in bits
        
    Returns:
        True if uncertainty is below threshold
    """
    return state.uncertainty_bits < uncertainty_threshold


def has_state_improved(
    old_state: AcquisitionState, 
    new_state: AcquisitionState
) -> bool:
    """
    Check if the target variable value has improved between states.
    
    Args:
        old_state: Previous acquisition state
        new_state: Current acquisition state
        
    Returns:
        True if best value has improved
    """
    return new_state.best_value > old_state.best_value


def has_uncertainty_decreased(
    old_state: AcquisitionState, 
    new_state: AcquisitionState
) -> bool:
    """
    Check if uncertainty has decreased between states.
    
    Args:
        old_state: Previous acquisition state
        new_state: Current acquisition state
        
    Returns:
        True if uncertainty has decreased
    """
    return new_state.uncertainty_bits < old_state.uncertainty_bits


def compute_information_gain(
    old_state: AcquisitionState, 
    new_state: AcquisitionState
) -> float:
    """
    Compute information gain between two states (reduction in uncertainty).
    
    Args:
        old_state: Previous acquisition state
        new_state: Current acquisition state
        
    Returns:
        Information gain in bits (positive means uncertainty decreased)
    """
    return old_state.uncertainty_bits - new_state.uncertainty_bits


def compute_optimization_improvement(
    old_state: AcquisitionState, 
    new_state: AcquisitionState
) -> float:
    """
    Compute optimization improvement between two states.
    
    Args:
        old_state: Previous acquisition state
        new_state: Current acquisition state
        
    Returns:
        Improvement in target variable value
    """
    return new_state.best_value - old_state.best_value


def get_state_summary_compact(state: AcquisitionState) -> str:
    """
    Get a compact string summary of the state.
    
    Args:
        state: Acquisition state
        
    Returns:
        Compact string representation
    """
    return (
        f"Step {state.step}: {state.current_target}={state.best_value:.3f}, "
        f"uncertainty={state.uncertainty_bits:.2f}bits, "
        f"samples={state.buffer_statistics.total_samples}"
    )


def filter_states_by_uncertainty(
    states: List[AcquisitionState], 
    min_uncertainty: float = 0.0,
    max_uncertainty: float = float('inf')
) -> List[AcquisitionState]:
    """
    Filter states by uncertainty range.
    
    Args:
        states: List of acquisition states
        min_uncertainty: Minimum uncertainty in bits
        max_uncertainty: Maximum uncertainty in bits
        
    Returns:
        Filtered list of states
    """
    return [
        state for state in states 
        if min_uncertainty <= state.uncertainty_bits <= max_uncertainty
    ]


def sort_states_by_uncertainty(
    states: List[AcquisitionState], 
    reverse: bool = False
) -> List[AcquisitionState]:
    """
    Sort states by uncertainty.
    
    Args:
        states: List of acquisition states
        reverse: If True, sort from high to low uncertainty
        
    Returns:
        Sorted list of states
    """
    return sorted(states, key=lambda s: s.uncertainty_bits, reverse=reverse)


def sort_states_by_best_value(
    states: List[AcquisitionState], 
    reverse: bool = True
) -> List[AcquisitionState]:
    """
    Sort states by best target value.
    
    Args:
        states: List of acquisition states
        reverse: If True, sort from high to low value (default)
        
    Returns:
        Sorted list of states
    """
    return sorted(states, key=lambda s: s.best_value, reverse=reverse)


def find_best_state(states: List[AcquisitionState]) -> AcquisitionState:
    """
    Find the state with the highest best value.
    
    Args:
        states: List of acquisition states
        
    Returns:
        State with highest best value
        
    Raises:
        ValueError: If states list is empty
    """
    if not states:
        raise ValueError("Cannot find best state from empty list")
    
    return max(states, key=lambda s: s.best_value)


def find_most_uncertain_state(states: List[AcquisitionState]) -> AcquisitionState:
    """
    Find the state with the highest uncertainty.
    
    Args:
        states: List of acquisition states
        
    Returns:
        State with highest uncertainty
        
    Raises:
        ValueError: If states list is empty
    """
    if not states:
        raise ValueError("Cannot find most uncertain state from empty list")
    
    return max(states, key=lambda s: s.uncertainty_bits)