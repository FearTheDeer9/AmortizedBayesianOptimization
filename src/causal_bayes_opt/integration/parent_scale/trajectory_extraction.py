#!/usr/bin/env python3
"""
PARENT_SCALE Trajectory Extraction

This module contains pure functions for extracting expert demonstration
trajectories from PARENT_SCALE algorithm outputs and converting them to
our standard format for training ACBO components.
"""

from typing import Dict, Any, List, Tuple, Optional
import pyrsistent as pyr


def validate_algorithm_results(results: Any) -> bool:
    """
    Validate that PARENT_SCALE algorithm results have the expected structure.
    
    Args:
        results: Output from parent_scale.run_algorithm()
        
    Returns:
        bool: True if results are valid, False otherwise
    """
    if not results:
        return False
    
    if not hasattr(results, '__len__') or len(results) != 6:
        return False
    
    # Check that none of the critical components are None
    global_opt, current_y, current_cost, intervention_set, intervention_values, average_uncertainty = results
    
    return all(x is not None for x in [global_opt, current_y, intervention_set])


def extract_trajectory_components(results: Tuple[Any, ...]) -> Dict[str, Any]:
    """
    Extract individual components from PARENT_SCALE algorithm results.
    
    Args:
        results: Tuple of (global_opt, current_y, current_cost, intervention_set, 
                          intervention_values, average_uncertainty)
        
    Returns:
        Dictionary with extracted components
        
    Raises:
        ValueError: If results structure is invalid
    """
    if not validate_algorithm_results(results):
        raise ValueError(f"Invalid algorithm results: {type(results)} with content {results}")
    
    (
        global_opt,
        current_y,
        current_cost,
        intervention_set,
        intervention_values,
        average_uncertainty
    ) = results
    
    # Validate individual components
    if global_opt is None:
        raise ValueError("global_opt is None")
    if current_y is None:
        raise ValueError("current_y is None")
    if intervention_set is None:
        raise ValueError("intervention_set is None")
    
    return {
        'global_opt': global_opt,
        'current_y': current_y,
        'current_cost': current_cost,
        'intervention_set': intervention_set,
        'intervention_values': intervention_values,
        'average_uncertainty': average_uncertainty
    }


def compute_performance_metrics(components: Dict[str, Any], T: int) -> Dict[str, float]:
    """
    Compute performance metrics from trajectory components.
    
    Args:
        components: Extracted trajectory components
        T: Number of algorithm iterations
        
    Returns:
        Dictionary with computed performance metrics
    """
    global_opt = components['global_opt']
    intervention_set = components['intervention_set']
    
    metrics = {}
    
    # Convergence rate: fraction of steps that improved the global optimum
    if len(global_opt) > 1:
        improvements = sum(1 for i, (prev, curr) in enumerate(zip(global_opt[:-1], global_opt[1:])) if curr < prev)
        metrics['convergence_rate'] = improvements / T
    else:
        metrics['convergence_rate'] = 0.0
    
    # Exploration efficiency: diversity of interventions
    if intervention_set:
        unique_interventions = len(set(tuple(intervention) if isinstance(intervention, list) else intervention 
                                        for intervention in intervention_set))
        metrics['exploration_efficiency'] = unique_interventions / len(intervention_set)
    else:
        metrics['exploration_efficiency'] = 0.0
    
    return metrics


def create_expert_trajectory(
    components: Dict[str, Any],
    target_variable: str,
    T: int,
    n_observational: int,
    n_interventional: int,
    algorithm_config: Dict[str, Any],
    parent_scale_algorithm: Any = None
) -> Dict[str, Any]:
    """
    Create a complete expert demonstration trajectory from algorithm components.
    
    Args:
        components: Extracted trajectory components
        target_variable: Name of the optimization target variable
        T: Number of algorithm iterations
        n_observational: Number of observational samples used
        n_interventional: Number of interventional samples per exploration element
        algorithm_config: Configuration used for the algorithm
        parent_scale_algorithm: Optional PARENT_SCALE algorithm instance for additional data
        
    Returns:
        Complete expert demonstration trajectory dictionary
    """
    # Extract components
    global_opt = components['global_opt']
    current_y = components['current_y']
    current_cost = components['current_cost']
    intervention_set = components['intervention_set']
    intervention_values = components['intervention_values']
    average_uncertainty = components['average_uncertainty']
    
    # Compute performance metrics
    metrics = compute_performance_metrics(components, T)
    
    # Extract posterior information if available
    final_posterior = {}
    final_graphs = []
    if parent_scale_algorithm:
        final_posterior = getattr(parent_scale_algorithm, 'prior_probabilities', {})
        final_graphs = list(getattr(parent_scale_algorithm, 'graphs', {}).keys())
    
    # Create complete trajectory
    trajectory = {
        # Algorithm metadata
        'algorithm': 'PARENT_SCALE_CBO',
        'target_variable': target_variable,
        'iterations': T,
        'total_samples_used': n_observational + (n_interventional * len(intervention_set)) if intervention_set else n_observational,
        'status': 'completed',
        
        # Complete trajectory data for training
        'intervention_sequence': intervention_set,
        'intervention_values': intervention_values,
        'target_outcomes': current_y,
        'global_optimum_trajectory': global_opt,
        'cost_trajectory': current_cost,
        'uncertainty_trajectory': average_uncertainty,
        
        # Final algorithm state
        'final_optimum': global_opt[-1] if global_opt else None,
        'final_cost': current_cost[-1] if current_cost else None,
        'total_interventions': len(intervention_set) if intervention_set else 0,
        
        # Algorithm configuration
        'config': algorithm_config,
        
        # Posterior information (if available)
        'final_posterior': final_posterior,
        'final_graphs': final_graphs,
        
        # Performance metrics
        'convergence_rate': metrics['convergence_rate'],
        'exploration_efficiency': metrics['exploration_efficiency']
    }
    
    return trajectory


def create_failed_trajectory(
    error: str,
    target_variable: str,
    T: int,
    n_observational: int,
    algorithm_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a trajectory record for a failed algorithm run.
    
    Args:
        error: Error message describing the failure
        target_variable: Name of the optimization target variable
        T: Number of algorithm iterations attempted
        n_observational: Number of observational samples used
        algorithm_config: Configuration used for the algorithm
        
    Returns:
        Failed trajectory record for debugging
    """
    return {
        'algorithm': 'PARENT_SCALE_CBO',
        'target_variable': target_variable,
        'iterations': T,
        'total_samples_used': n_observational,
        'error': str(error),
        'status': 'failed',
        'config': algorithm_config
    }


def convert_to_acbo_format(trajectory: Dict[str, Any]) -> pyr.PMap:
    """
    Convert trajectory to immutable ACBO format for consistency.
    
    Args:
        trajectory: Expert demonstration trajectory dictionary
        
    Returns:
        Immutable trajectory representation using pyrsistent
    """
    # Convert lists to pyrsistent vectors for immutability
    immutable_trajectory = {}
    
    for key, value in trajectory.items():
        if isinstance(value, list):
            immutable_trajectory[key] = pyr.v(*value)
        elif isinstance(value, dict):
            immutable_trajectory[key] = pyr.m(**value)
        else:
            immutable_trajectory[key] = value
    
    return pyr.m(**immutable_trajectory)


def extract_state_action_pairs(trajectory: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Extract (state, action) pairs from trajectory for acquisition model training.
    
    Args:
        trajectory: Expert demonstration trajectory
        
    Returns:
        List of (state, action) tuples for imitation learning
        
    Note:
        This is a placeholder implementation. The actual state representation
        would depend on the final AcquisitionState design and how we reconstruct
        the decision context from PARENT_SCALE's internal state.
    """
    state_action_pairs = []
    
    if trajectory.get('status') != 'completed':
        return state_action_pairs
    
    intervention_set = trajectory.get('intervention_sequence', [])
    intervention_values = trajectory.get('intervention_values', [])
    target_outcomes = trajectory.get('target_outcomes', [])
    uncertainty_trajectory = trajectory.get('uncertainty_trajectory', [])
    
    # Extract state-action pairs for each intervention decision
    for i, (intervention_vars, intervention_vals) in enumerate(zip(intervention_set, intervention_values)):
        # State: Information available before making intervention decision
        state = {
            'step': i,
            'uncertainty': uncertainty_trajectory[i] if i < len(uncertainty_trajectory) else None,
            'current_best': min(target_outcomes[:i]) if i > 0 and target_outcomes else None,
            'intervention_history': intervention_set[:i],
            'target_history': target_outcomes[:i]
        }
        
        # Action: Intervention decision made by expert
        action = {
            'intervention_variables': intervention_vars,
            'intervention_values': intervention_vals
        }
        
        state_action_pairs.append((state, action))
    
    return state_action_pairs


def extract_posterior_sequence(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract sequence of posterior updates for surrogate model training.
    
    Args:
        trajectory: Expert demonstration trajectory
        
    Returns:
        List of posterior states for behavioral cloning
        
    Note:
        This is a placeholder implementation. The actual posterior extraction
        would require access to PARENT_SCALE's internal posterior updates
        during the algorithm run.
    """
    posterior_sequence = []
    
    if trajectory.get('status') != 'completed':
        return posterior_sequence
    
    # This would require modification of PARENT_SCALE algorithm to capture
    # posterior states at each step, or reconstruction from final posterior
    # and intervention history
    
    # Placeholder: use uncertainty trajectory as proxy for posterior evolution
    uncertainty_trajectory = trajectory.get('uncertainty_trajectory', [])
    
    for i, uncertainty in enumerate(uncertainty_trajectory):
        posterior_state = {
            'step': i,
            'uncertainty': uncertainty,
            'method': 'placeholder'  # Would be actual posterior distribution
        }
        posterior_sequence.append(posterior_state)
    
    return posterior_sequence