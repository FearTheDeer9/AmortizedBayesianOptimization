#!/usr/bin/env python3
"""
PARENT_SCALE Data Generation

This module contains pure functions for generating data using PARENT_SCALE's
exact original data generation process to ensure 100% identical behavior.
"""

from typing import Dict, Any, Tuple, List
import sys

# Add PARENT_SCALE to path
sys.path.insert(0, 'external/parent_scale')
sys.path.insert(0, 'external')
sys.path.insert(0, 'causal_bayes_opt_old')

# Import PARENT_SCALE components - handle gracefully if not available
try:
    from graphs.linear_collider_graph import LinearColliderGraph
    from graphs.data_setup import setup_observational_interventional
    PARENT_SCALE_AVAILABLE = True
except ImportError:
    # Create dummy classes when PARENT_SCALE not available
    class LinearColliderGraph:
        pass
    def setup_observational_interventional(*args, **kwargs):
        raise ImportError("PARENT_SCALE not available")
    PARENT_SCALE_AVAILABLE = False


def check_parent_scale_availability() -> bool:
    """
    Check if PARENT_SCALE components are available.
    
    Returns:
        bool: True if PARENT_SCALE is available, False otherwise
    """
    return PARENT_SCALE_AVAILABLE


def generate_parent_scale_data(
    n_observational: int,
    n_interventional: int,
    seed: int
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
    """
    Generate data using PARENT_SCALE's exact original data generation process.
    
    This function ensures 100% identical behavior with the original algorithm
    by using LinearColliderGraph and setup_observational_interventional exactly
    as the original implementation does.
    
    Args:
        n_observational: Number of observational samples to generate
        n_interventional: Number of interventional samples per exploration element
        seed: Random seed for reproducible data generation
        
    Returns:
        Tuple of (D_O, D_I, exploration_set) where:
        - D_O: Observational data dictionary
        - D_I: Interventional data dictionary 
        - exploration_set: List of variables available for intervention
        
    Raises:
        ImportError: If PARENT_SCALE components are not available
    """
    if not PARENT_SCALE_AVAILABLE:
        raise ImportError(
            "PARENT_SCALE components not available. Please ensure external/parent_scale is properly set up."
        )
    
    # CRITICAL: Set random seed exactly like the original algorithm
    import numpy as np
    np.random.seed(seed)
    
    # Create the original graph structure (exactly like original algorithm)
    graph = LinearColliderGraph(noiseless=False)
    
    # Generate data using PARENT_SCALE's exact original process
    D_O, D_I, exploration_set = setup_observational_interventional(
        graph_type="Toy",
        n_obs=n_observational,
        n_int=n_interventional,
        noiseless=True,
        seed=seed,
        graph=graph,
        use_iscm=False
    )
    
    return D_O, D_I, exploration_set


def validate_data_completeness(
    D_O: Dict[str, Any],
    D_I: Dict[str, Any], 
    exploration_set: List[Any],
    target_variable: str = 'Y'
) -> None:
    """
    Validate that generated data is complete and consistent.
    
    Args:
        D_O: Observational data dictionary
        D_I: Interventional data dictionary
        exploration_set: List of variables available for intervention
        target_variable: Target variable name for validation
        
    Raises:
        ValueError: If data is incomplete or inconsistent
    """
    # Check observational data
    if target_variable not in D_O:
        raise ValueError(f"Target variable '{target_variable}' not found in observational data")
    
    if len(D_O[target_variable]) == 0:
        raise ValueError("No observational samples generated")
    
    # Check interventional data completeness
    missing_exploration_elements = [es for es in exploration_set if es not in D_I]
    if missing_exploration_elements:
        raise ValueError(
            f"Missing D_I data for exploration elements: {missing_exploration_elements}. "
            "This should not happen with original data generation."
        )
    
    # Check that all exploration elements have target data
    for es in exploration_set:
        es_tuple = tuple(es)
        if es_tuple not in D_I:
            raise ValueError(f"Missing interventional data for exploration element: {es_tuple}")
        
        if target_variable not in D_I[es_tuple]:
            raise ValueError(f"Missing target variable data for exploration element: {es_tuple}")


def get_data_summary(
    D_O: Dict[str, Any],
    D_I: Dict[str, Any],
    exploration_set: List[Any],
    target_variable: str = 'Y'
) -> Dict[str, Any]:
    """
    Get a summary of the generated data for logging and validation.
    
    Args:
        D_O: Observational data dictionary
        D_I: Interventional data dictionary
        exploration_set: List of variables available for intervention
        target_variable: Target variable name
        
    Returns:
        Dictionary containing data summary information
    """
    summary = {
        'observational_samples': len(D_O.get(target_variable, [])),
        'intervention_groups': len(D_I),
        'exploration_elements': len(exploration_set),
        'target_variable': target_variable,
        'observational_variables': list(D_O.keys()),
        'exploration_set': exploration_set
    }
    
    # Calculate total interventional samples
    total_interventional = 0
    for es_tuple in D_I:
        if target_variable in D_I[es_tuple]:
            total_interventional += len(D_I[es_tuple][target_variable])
    
    summary['total_interventional_samples'] = total_interventional
    
    return summary


def create_graph_instance() -> Any:
    """
    Create a LinearColliderGraph instance for use with PARENT_SCALE.
    
    Returns:
        LinearColliderGraph instance
        
    Raises:
        ImportError: If PARENT_SCALE components are not available
    """
    if not PARENT_SCALE_AVAILABLE:
        raise ImportError(
            "PARENT_SCALE components not available. Please ensure external/parent_scale is properly set up."
        )
    
    return LinearColliderGraph(noiseless=False)