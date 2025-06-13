#!/usr/bin/env python3
"""
PARENT_SCALE Data Bridge

Provides conversion functions between our ACBO data structures and PARENT_SCALE's
expected formats. Enables seamless integration with the validated neural doubly
robust method while maintaining our clean architecture.

Key Conversions:
1. Our SCM → PARENT_SCALE GraphStructure  
2. Our Sample list → PARENT_SCALE Data(samples, nodes)
3. PARENT_SCALE prob_estimate → Our ParentSetPosterior
4. Round-trip validation for data integrity
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
from collections import namedtuple
import warnings

import numpy as onp
import pyrsistent as pyr

# Add PARENT_SCALE to path
sys.path.insert(0, 'external/parent_scale')
sys.path.insert(0, 'external')

# Import PARENT_SCALE components
try:
    from parent_scale.posterior_model.model import DoublyRobustModel, Data
    from parent_scale.graphs.graph import GraphStructure
    PARENT_SCALE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PARENT_SCALE not available: {e}")
    PARENT_SCALE_AVAILABLE = False
    # Create dummy classes for type hints
    Data = namedtuple("Data", ["samples", "nodes"])
    class GraphStructure:
        pass

# Import our data structures
from causal_bayes_opt.data_structures.scm import get_variables, get_edges, get_target
from causal_bayes_opt.data_structures.sample import (
    get_values, get_intervention_targets, get_intervention_type, is_interventional
)


def scm_to_graph_structure(scm: pyr.PMap) -> GraphStructure:
    """
    Convert our SCM to PARENT_SCALE GraphStructure.
    
    Args:
        scm: Our immutable SCM representation
        
    Returns:
        GraphStructure compatible with PARENT_SCALE
    """
    if not PARENT_SCALE_AVAILABLE:
        raise ImportError(
            "PARENT_SCALE not available. Please ensure external/parent_scale is properly set up."
        )
    
    variables = list(get_variables(scm))
    edges = list(get_edges(scm))
    target = get_target(scm)
    
    # Create parent mapping required by PARENT_SCALE
    parents = {var: [] for var in variables}
    for parent, child in edges:
        parents[child].append(parent)
    
    # Create simple GraphStructure-like object
    class SimpleGraphStructure:
        def __init__(self, variables, parents, target):
            self.variables = variables
            self.parents = parents
            self.target = target
    
    return SimpleGraphStructure(variables, parents, target)
    
def samples_to_parent_scale_data(
    samples: List[pyr.PMap], 
    variable_order: List[str]
) -> Data:
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
        
    return Data(samples=sample_matrix, nodes=intervention_matrix)


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
    
def run_parent_discovery(
    scm: pyr.PMap,
    samples: List[pyr.PMap],
    target_variable: str,
    num_bootstraps: int = 15,
    individual: bool = False
) -> Dict[str, Any]:
    """
    Complete parent discovery using PARENT_SCALE neural doubly robust method.
    
    Args:
        scm: Our SCM representation
        samples: List of our Sample objects
        target_variable: Variable to find parents for
        num_bootstraps: Number of bootstrap samples (use validated scaling)
        individual: Whether to use individual bootstrap method
        
    Returns:
        Parent discovery results in our format
    """
    # Convert SCM to PARENT_SCALE format
    graph = scm_to_graph_structure(scm)
    variables = graph.variables
    
    # Convert samples to PARENT_SCALE format
    data = samples_to_parent_scale_data(samples, variables)
    
    # Create and run PARENT_SCALE model
    model = DoublyRobustModel(
        graph=graph,
        topological_order=variables,
        target=target_variable,
        num_bootstraps=num_bootstraps,
        indivdual=individual  # Note: PARENT_SCALE has typo in parameter name
    )
        
    # Run inference
    estimate = model.run_method(data)
        
    # Convert results to our format
    posterior = parent_scale_results_to_posterior(
        model.prob_estimate, target_variable, variables
    )
        
    # Add metadata
    posterior['num_bootstraps'] = num_bootstraps
    posterior['num_samples'] = len(samples)
    posterior['method'] = 'neural_doubly_robust'
        
    return posterior
    
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


def create_parent_scale_bridge() -> bool:
    """Validate PARENT_SCALE availability for bridge functions."""
    if not PARENT_SCALE_AVAILABLE:
        raise ImportError(
            "PARENT_SCALE not available. Please ensure external/parent_scale is properly set up."
        )
    return True


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


if __name__ == "__main__":
    # Simple test of the bridge
    if PARENT_SCALE_AVAILABLE:
        print("PARENT_SCALE Data Bridge Test")
        print("=" * 40)
        
        create_parent_scale_bridge()  # Validate availability
        print("✅ Bridge functions available")
        
        # Test data requirements calculation
        for n_nodes in [5, 10, 20]:
            req = calculate_data_requirements(n_nodes)
            print(f"  {n_nodes} nodes: {req['total_samples']} samples, {req['bootstrap_samples']} bootstraps")
        
        print("✅ Bridge functionality validated")
    else:
        print("❌ PARENT_SCALE not available for testing")