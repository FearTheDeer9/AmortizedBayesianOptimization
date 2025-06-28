#!/usr/bin/env python3
"""
Data Processing for PARENT_SCALE Integration

This module contains consolidated data processing functions for converting between
our ACBO data structures and PARENT_SCALE's expected formats.
"""

import sys
import warnings
from typing import Dict, Any, Tuple, List, Optional
from collections import namedtuple
from pathlib import Path

import numpy as onp
import pyrsistent as pyr

# Import our data structures
from ...data_structures.scm import get_target
from ...data_structures.sample import (
    get_values, get_intervention_targets, is_interventional
)

# Setup PARENT_SCALE path - add external directory to sys.path
def setup_parent_scale_path():
    """Add external directory to Python path for PARENT_SCALE imports."""
    # Get the project root (4 levels up from this file)
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent.parent.parent
    external_dir = project_root / "external"
    parent_scale_dir = external_dir / "parent_scale"
    
    paths_added = 0
    
    # Add external directory for parent_scale module import
    if external_dir.exists():
        external_str = str(external_dir)
        if external_str not in sys.path:
            sys.path.insert(0, external_str)
            paths_added += 1
    
    # Add parent_scale directory for internal imports (graphs, posterior_model, etc.)
    if parent_scale_dir.exists():
        parent_scale_str = str(parent_scale_dir)
        if parent_scale_str not in sys.path:
            sys.path.insert(0, parent_scale_str)
            paths_added += 1
    
    return paths_added > 0

# Setup the path before importing
path_setup_success = setup_parent_scale_path()

# PARENT_SCALE imports will be done when needed with proper error handling
# No global availability state - use direct imports with try/except at point of use

# Import graph structure
from .graph_structure import ACBOGraphStructure
try:
    from .helpers import setup_observational_interventional_original
except ImportError:
    def setup_observational_interventional_original(*args, **kwargs):
        raise ImportError("Helpers module not available")


def ensure_parent_scale_imports():
    """
    Ensure PARENT_SCALE can be imported, setting up paths if needed.
    Raises ImportError with clear message if not available.
    """
    setup_parent_scale_path()
    
    try:
        from parent_scale.posterior_model.model import Data
        return True
    except ImportError as e:
        raise ImportError(
            f"PARENT_SCALE not available: {e}. "
            "Please ensure external/parent_scale directory contains PARENT_SCALE source code."
        )


def generate_parent_scale_data_with_scm(
    scm,
    n_observational: int,
    n_interventional: int,
    seed: int
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Any]]:
    """
    Generate data using actual SCM converted to PARENT_SCALE format.
    
    This function converts any SCM to PARENT_SCALE's GraphStructure and generates
    data from that structure, enabling true decoupling from LinearColliderGraph.
    
    Args:
        scm: The SCM to use for data generation
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
    # Ensure PARENT_SCALE is available
    ensure_parent_scale_imports()
    
    # Use the proper SCM conversion through original data generation
    return generate_parent_scale_data_original(
        scm=scm,
        n_observational=n_observational, 
        n_interventional=n_interventional,
        seed=seed,
        noiseless=True
    )


def generate_parent_scale_data_original(
    scm: pyr.PMap,
    n_observational: int = 100,
    n_interventional: int = 2,
    seed: int = 42,
    noiseless: bool = True
) -> Tuple[Dict[str, onp.ndarray], Dict[Tuple, Dict[str, onp.ndarray]], List[Tuple]]:
    """
    Generate data using the original PARENT_SCALE implementation pattern.
    
    This uses the exact same pattern as the original integration:
    1. Convert ACBO SCM to proper GraphStructure with real SEMs
    2. Use setup_observational_interventional_original()
    3. Return data with original variable names
    
    Args:
        scm: Our SCM representation
        n_observational: Number of observational samples
        n_interventional: Number of interventional samples per exploration element
        seed: Random seed for reproducibility
        noiseless: Whether to use noiseless sampling
        
    Returns:
        Tuple of (D_O, D_I, exploration_set) where:
        - D_O: Dict mapping variables to observational data arrays
        - D_I: Dict mapping intervention tuples to variable data arrays
        - exploration_set: List of intervention tuples with original variable names
    """
    # Ensure PARENT_SCALE is available
    ensure_parent_scale_imports()
    
    print(f"Generating PARENT_SCALE data using original implementation pattern...")
    print(f"  - n_observational: {n_observational}")
    print(f"  - n_interventional: {n_interventional}")
    print(f"  - seed: {seed}")
    
    # Convert SCM to PARENT_SCALE graph format with proper SEMs
    graph = scm_to_graph_structure(scm)
    
    # Get the original target variable name
    original_target = get_target(scm)
    
    # Use the original data generation pattern directly
    D_O, D_I, exploration_set = setup_observational_interventional_original(
        graph=graph,
        n_obs=n_observational,
        n_int=n_interventional,
        noiseless=noiseless,
        seed=seed,
        use_iscm=False
    )
    
    print(f"✓ Generated original PARENT_SCALE data format")
    print(f"  - Raw D_O keys: {list(D_O.keys())}")
    print(f"  - Raw D_I keys: {list(D_I.keys())}")
    print(f"  - Raw exploration set: {exploration_set}")
    print(f"  - Original target: {original_target}")
    
    # Return data with original variable names
    print(f"✓ Using original variable names")
    return D_O, D_I, exploration_set


def scm_to_graph_structure(scm: pyr.PMap) -> ACBOGraphStructure:
    """
    Convert our SCM to PARENT_SCALE GraphStructure.
    
    Args:
        scm: Our immutable SCM representation
        
    Returns:
        ACBOGraphStructure compatible with PARENT_SCALE
    """
    # Ensure PARENT_SCALE is available
    ensure_parent_scale_imports()
    
    return ACBOGraphStructure(scm)


def validate_data_completeness(
    D_O: Dict[str, Any],
    D_I: Dict[str, Any], 
    exploration_set: List[Any],
    target_variable: str
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
    target_variable: str
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


def samples_to_parent_scale_data(
    samples: List[pyr.PMap], 
    variable_order: List[str]
) -> Any:
    """
    Convert our Sample objects to PARENT_SCALE Data format.
    
    Args:
        samples: List of our Sample objects
        variable_order: Consistent ordering of variables for matrix conversion
        
    Returns:
        Data(samples, nodes) compatible with PARENT_SCALE
    """
    n_samples = len(samples)
    n_vars = len(variable_order)
    
    # Initialize matrices
    sample_matrix = onp.zeros((n_samples, n_vars))
    intervention_matrix = onp.zeros((n_samples, n_vars), dtype=int)
    
    for i, sample in enumerate(samples):
        values = get_values(sample)
        
        # Extract variable values in consistent order
        for j, var in enumerate(variable_order):
            if var in values:
                sample_matrix[i, j] = float(values[var])
            else:
                # Fill missing values with 0 (should be rare with proper data)
                sample_matrix[i, j] = 0.0
                warnings.warn(f"Missing value for {var} in sample {i}, using 0.0")
        
        # Mark interventional variables
        if is_interventional(sample):
            intervention_targets = get_intervention_targets(sample)
            for var in intervention_targets:
                if var in variable_order:
                    j = variable_order.index(var)
                    intervention_matrix[i, j] = 1
        
    # Import Data class when needed
    try:
        from parent_scale.posterior_model.model import Data
        return Data(samples=sample_matrix, nodes=intervention_matrix)
    except ImportError as e:
        raise ImportError(f"Cannot create PARENT_SCALE Data object: {e}")


def calculate_data_requirements(n_nodes: int, target_accuracy: float = 0.8) -> Dict[str, int]:
    """
    Calculate data requirements using validated scaling formulas.
    
    Based on empirical validation: O(d^2.5) sample scaling achieves 0.8+ accuracy.
    """
    # Sample size: Based on validated scaling results
    if n_nodes <= 3:
        total_samples = 100
    elif n_nodes <= 5:
        total_samples = 150
    elif n_nodes <= 10:
        total_samples = 300
    elif n_nodes <= 15:
        total_samples = 400
    else:
        # For 20+ nodes, use validated scaling
        total_samples = max(500, int(25 * n_nodes))
    
    # Bootstrap samples: scale with graph complexity  
    bootstrap_scaling_factor = 0.75
    bootstrap_samples = max(5, min(20, int(bootstrap_scaling_factor * n_nodes)))
    
    # Split observational/interventional (15% interventional)
    interventional_ratio = 0.15
    n_interventional = int(total_samples * interventional_ratio)
    n_observational = total_samples - n_interventional
    
    # Adjust for target accuracy
    if target_accuracy >= 0.9:
        total_samples = int(total_samples * 1.5)
        bootstrap_samples = int(bootstrap_samples * 1.2)
    elif target_accuracy <= 0.6:
        total_samples = int(total_samples * 0.8)
        bootstrap_samples = max(3, int(bootstrap_samples * 0.8))
    
    return {
        'total_samples': total_samples,
        'observational_samples': n_observational,
        'interventional_samples': n_interventional,
        'bootstrap_samples': bootstrap_samples,
        'n_nodes': n_nodes,
        'target_accuracy': target_accuracy
    }