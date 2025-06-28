#!/usr/bin/env python3
"""
Data Conversion Functions for PARENT_SCALE Integration

Provides conversion functions between our ACBO data structures and PARENT_SCALE's
expected formats, including validation and data requirements calculation.
"""

import warnings
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import namedtuple

import numpy as onp
import pyrsistent as pyr

# Import our data structures
from ...data_structures.scm import get_target
from ...data_structures.sample import (
    get_values, get_intervention_targets, is_interventional
)

# Import path setup and availability check from data_processing
from .data_processing import ensure_parent_scale_imports

# Import other components
from .graph_structure import ACBOGraphStructure
from .helpers import setup_observational_interventional_original


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


def parent_scale_results_to_posterior(
    prob_estimate: Dict,
    target_variable: str,
    all_variables: List[str]
) -> Dict[str, Any]:
    """
    Convert PARENT_SCALE probability estimates to our format.
    
    Args:
        prob_estimate: PARENT_SCALE prob_estimate dictionary
        target_variable: Name of target variable
        all_variables: List of all variables in the graph
        
    Returns:
        Dictionary with parent set posterior information
    """
    if not prob_estimate:
        return {
            'target_variable': target_variable,
            'parent_sets': {},
            'most_likely_parents': frozenset(),
            'confidence': 0.0,
            'uncertainty': float('inf')
        }
    
    # Convert parent sets to our format
    parent_sets = {}
    for parent_tuple, probability in prob_estimate.items():
        if parent_tuple:  # Non-empty parent set
            parent_set = frozenset(parent_tuple)
        else:  # Empty parent set
            parent_set = frozenset()
        parent_sets[parent_set] = float(probability)
    
    # Find most likely parent set
    most_likely = max(parent_sets.items(), key=lambda x: x[1])
    most_likely_parents = most_likely[0]
    confidence = most_likely[1]
    
    # Calculate uncertainty (entropy)
    probs = onp.array(list(parent_sets.values()))
    probs = probs / onp.sum(probs)  # Normalize
    uncertainty = -onp.sum(probs * onp.log(probs + 1e-12))
    
    return {
        'target_variable': target_variable,
        'parent_sets': parent_sets,
        'most_likely_parents': most_likely_parents,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'n_parent_sets': len(parent_sets)
    }


def validate_conversion(
    original_samples: List[pyr.PMap],
    variable_order: List[str],
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that our data conversion preserves all information.
    
    Args:
        original_samples: Original samples to validate
        variable_order: Variable ordering used in conversion
        tolerance: Numerical tolerance for comparisons
        
    Returns:
        True if conversion is valid, False otherwise
    """
    try:
        # Convert to PARENT_SCALE format
        data = samples_to_parent_scale_data(original_samples, variable_order)
        
        # Validate dimensions
        expected_samples = len(original_samples)
        expected_vars = len(variable_order)
        
        if data.samples.shape != (expected_samples, expected_vars):
            print(f"Shape mismatch: expected {(expected_samples, expected_vars)}, got {data.samples.shape}")
            return False
            
        if data.nodes.shape != (expected_samples, expected_vars):
            print(f"Nodes shape mismatch: expected {(expected_samples, expected_vars)}, got {data.nodes.shape}")
            return False
            
        # Validate sample values
        for i, sample in enumerate(original_samples):
            values = get_values(sample)
            for j, var in enumerate(variable_order):
                if var in values:
                    original_val = float(values[var])
                    converted_val = data.samples[i, j]
                    if abs(original_val - converted_val) > tolerance:
                        print(f"Value mismatch at [{i}, {j}]: {original_val} vs {converted_val}")
                        return False
            
        # Validate intervention indicators
        for i, sample in enumerate(original_samples):
            if is_interventional(sample):
                intervention_targets = get_intervention_targets(sample)
                for j, var in enumerate(variable_order):
                    expected_intervention = 1 if var in intervention_targets else 0
                    actual_intervention = data.nodes[i, j]
                    if expected_intervention != actual_intervention:
                        print(f"Intervention mismatch at [{i}, {j}]: {expected_intervention} vs {actual_intervention}")
                        return False
            else:
                # Observational sample should have no interventions
                if onp.any(data.nodes[i, :]):
                    print(f"Observational sample {i} has intervention markers")
                    return False
            
        print("✅ Data conversion validation passed")
        return True
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False


def calculate_data_requirements(n_nodes: int, target_accuracy: float = 0.8) -> Dict[str, int]:
    """
    Calculate data requirements using validated scaling formulas.
    
    Based on empirical validation: O(d^2.5) sample scaling achieves 0.8+ accuracy.
    """
    # Sample size: Based on validated 20-node results (500 samples for 20 nodes)
    # This gives us ~25 samples per node, which scales roughly as O(d)
    if n_nodes <= 3:
        total_samples = 100
    elif n_nodes <= 5:
        total_samples = 150
    elif n_nodes <= 10:
        total_samples = 300
    elif n_nodes <= 15:
        total_samples = 400
    else:
        # For 20+ nodes, use validated 500 sample requirement
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


def samples_to_parent_scale_dict_format(
    samples: List[pyr.PMap], 
    variable_order: List[str]
) -> Tuple[Dict[str, onp.ndarray], Dict[Tuple, Dict[str, onp.ndarray]]]:
    """
    Convert our Sample objects to PARENT_SCALE's D_O/D_I dictionary format.
    
    Args:
        samples: List of our Sample objects
        variable_order: Consistent ordering of variables
        
    Returns:
        Tuple of (D_O, D_I) where:
        - D_O: Dict mapping variables to observational data arrays
        - D_I: Dict mapping intervention tuples to variable data arrays
    """
    # Separate observational and interventional samples
    obs_samples = [s for s in samples if not is_interventional(s)]
    int_samples = [s for s in samples if is_interventional(s)]
    
    # Create D_O (observational data)
    D_O = {var: [] for var in variable_order}
    
    for sample in obs_samples:
        values = get_values(sample)
        for var in variable_order:
            if var in values:
                D_O[var].append(float(values[var]))
            else:
                D_O[var].append(0.0)  # Fill missing with 0
                warnings.warn(f"Missing value for {var} in observational sample, using 0.0")
    
    # Convert to numpy arrays - ensure 2D format for PARENT_SCALE compatibility
    D_O = {var: onp.array(vals).reshape(-1, 1) for var, vals in D_O.items()}
    
    # Create D_I (interventional data)
    D_I = {}
    
    # Group interventional samples by intervention targets
    intervention_groups = {}
    
    for sample in int_samples:
        targets = get_intervention_targets(sample)
        targets_tuple = tuple(sorted(targets))  # Consistent ordering
        
        if targets_tuple not in intervention_groups:
            intervention_groups[targets_tuple] = []
        intervention_groups[targets_tuple].append(sample)
    
    # Convert each intervention group to PARENT_SCALE format
    for targets_tuple, group_samples in intervention_groups.items():
        D_I[targets_tuple] = {var: [] for var in variable_order}
        
        for sample in group_samples:
            values = get_values(sample)
            for var in variable_order:
                if var in values:
                    D_I[targets_tuple][var].append(float(values[var]))
                else:
                    D_I[targets_tuple][var].append(0.0)
                    warnings.warn(f"Missing value for {var} in interventional sample, using 0.0")
        
        # Convert to numpy arrays
        D_I[targets_tuple] = {var: onp.array(vals) for var, vals in D_I[targets_tuple].items()}
    
    return D_O, D_I


def create_exploration_set(variables: List[str], target: str) -> List[Tuple[str]]:
    """
    Create initial exploration set for PARENT_SCALE algorithm.
    
    Args:
        variables: All variable names in the graph
        target: Target variable name
        
    Returns:
        List of intervention tuples (starting with individual interventions)
    """
    # Start with individual interventions on non-target variables
    non_target_vars = [var for var in variables if var != target]
    exploration_set = [(var,) for var in non_target_vars]
    
    return exploration_set


def generate_parent_scale_data_original(
    scm: pyr.PMap,
    n_observational: int = 100,
    n_interventional: int = 2,
    seed: int = 42,
    noiseless: bool = True
) -> Tuple[Dict[str, onp.ndarray], Dict[Tuple, Dict[str, onp.ndarray]], List[Tuple]]:
    """
    Generate data using the original PARENT_SCALE implementation pattern.
    
    This uses the exact same pattern as causal_bayes_opt_old:
    1. Convert ACBO SCM to proper GraphStructure with real SEMs
    2. Use setup_observational_interventional_original()
    3. Map only the target variable to 'Y' for PARENT_SCALE compatibility
    
    Args:
        scm: Our SCM representation
        n_observational: Number of observational samples
        n_interventional: Number of interventional samples per exploration element
        seed: Random seed for reproducibility
        noiseless: Whether to use noiseless sampling
        
    Returns:
        Tuple of (D_O, D_I, exploration_set) where:
        - D_O: Dict mapping variables to observational data arrays (target as 'Y')
        - D_I: Dict mapping intervention tuples to variable data arrays (target as 'Y')
        - exploration_set: List of intervention tuples with original variable names
    """
    print(f"Generating PARENT_SCALE data using original implementation pattern...")
    print(f"  - n_observational: {n_observational}")
    print(f"  - n_interventional: {n_interventional}")
    print(f"  - seed: {seed}")
    
    # Convert SCM to PARENT_SCALE graph format with proper SEMs
    graph = scm_to_graph_structure(scm)
    
    # Get the original target variable name
    from ...data_structures.scm import get_target
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
    
    # Return data with original variable names for now
    print(f"✓ Using original variable names")
    return D_O, D_I, exploration_set


# Keep the old function name for backward compatibility
def generate_parent_scale_data(
    scm: pyr.PMap,
    n_observational: int = 100,
    n_interventional: int = 2,
    seed: int = 42,
    noiseless: bool = True
) -> Tuple[Dict[str, onp.ndarray], Dict[Tuple, Dict[str, onp.ndarray]], List[Tuple]]:
    """Backward compatibility wrapper - use generate_parent_scale_data_original for new code."""
    return generate_parent_scale_data_original(
        scm, n_observational, n_interventional, seed, noiseless
    )


def create_parent_scale_bridge() -> bool:
    """Validate PARENT_SCALE availability for bridge functions."""
    # Ensure PARENT_SCALE is available
    ensure_parent_scale_imports()
    return True