"""
Analysis and debugging utilities for AVICI data format bridge.

This module contains functions for analyzing converted data,
debugging issues, and providing insights into the conversion process.
"""

# Standard library imports
from typing import Dict, List, Optional, Any

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

# Type aliases
SampleList = List[pyr.PMap]
VariableOrder = List[str]


def analyze_avici_data(avici_data: jnp.ndarray, variable_order: VariableOrder) -> Dict[str, Any]:
    """
    Analyze AVICI data tensor and return summary statistics.
    
    Args:
        avici_data: AVICI data tensor [N, d, 3]
        variable_order: Variable order used in conversion
        
    Returns:
        Dictionary with analysis results
    """
    n_samples, n_vars, n_channels = avici_data.shape
    
    # Extract channels
    values = avici_data[:, :, 0]
    interventions = avici_data[:, :, 1]
    targets = avici_data[:, :, 2]
    
    # Compute statistics
    analysis = {
        'shape': avici_data.shape,
        'n_samples': n_samples,
        'n_variables': n_vars,
        'variable_order': variable_order,
        'values_stats': {
            'mean': float(jnp.mean(values)),
            'std': float(jnp.std(values)),
            'min': float(jnp.min(values)),
            'max': float(jnp.max(values))
        },
        'intervention_stats': {
            'total_interventions': int(jnp.sum(interventions)),
            'samples_with_interventions': int(jnp.sum(jnp.any(interventions, axis=1))),
            'variables_intervened': [
                var for i, var in enumerate(variable_order) 
                if jnp.sum(interventions[:, i]) > 0
            ]
        },
        'target_stats': {
            'target_variable': variable_order[jnp.argmax(jnp.sum(targets, axis=0))],
            'target_indicator_sum': float(jnp.sum(targets))
        }
    }
    
    return analysis


def reconstruct_samples_from_avici_data(
    avici_data: jnp.ndarray,
    variable_order: VariableOrder,
    target_variable: str,
    standardization_params: Optional[Dict[str, jnp.ndarray]] = None
) -> SampleList:
    """
    Reconstruct Sample objects from AVICI data tensor (for validation).
    
    Args:
        avici_data: AVICI data tensor [N, d, 3]
        variable_order: Variable order used in conversion
        target_variable: Target variable name
        standardization_params: Parameters for reversing standardization
        
    Returns:
        List of reconstructed Sample objects
        
    Note:
        This function is primarily for validation purposes.
        Standardization reversal is approximate if standardization_params not provided.
    """
    n_samples = avici_data.shape[0]
    reconstructed_samples = []
    
    # Extract channels
    values_channel = avici_data[:, :, 0]
    intervention_channel = avici_data[:, :, 1]
    
    for i in range(n_samples):
        # Extract values for this sample
        sample_values = {}
        for j, var_name in enumerate(variable_order):
            sample_values[var_name] = float(values_channel[i, j])
        
        # Extract intervention information
        intervention_targets = set()
        for j, var_name in enumerate(variable_order):
            if intervention_channel[i, j] > 0.5:  # Threshold for binary indicator
                intervention_targets.add(var_name)
        
        # Create Sample object (simplified structure for validation)
        if intervention_targets:
            sample = pyr.m(
                values=pyr.m(**sample_values),
                intervention_type='perfect',  # Simplified
                intervention_targets=pyr.s(*intervention_targets),
                metadata=pyr.m()
            )
        else:
            sample = pyr.m(
                values=pyr.m(**sample_values),
                intervention_type=None,
                intervention_targets=pyr.s(),
                metadata=pyr.m()
            )
        
        reconstructed_samples.append(sample)
    
    return reconstructed_samples


def get_variable_order_from_scm(scm: pyr.PMap) -> VariableOrder:
    """
    Get a consistent variable ordering from an SCM.
    
    Args:
        scm: The structural causal model
        
    Returns:
        List of variable names in a consistent order
        
    Note:
        This is a simplified version. In a full implementation,
        would use topological_sort from scm module for proper ordering.
    """
    variables = scm['variables']
    # Simplified: just sort alphabetically
    # In real implementation: return topological_sort(scm)
    return sorted(variables)


def compare_data_conversions(
    samples: SampleList,
    avici_data1: jnp.ndarray,
    avici_data2: jnp.ndarray,
    variable_order: VariableOrder,
    labels: List[str] = None
) -> Dict[str, Any]:
    """
    Compare two different AVICI data conversions for debugging.
    
    Args:
        samples: Original sample data
        avici_data1: First AVICI conversion
        avici_data2: Second AVICI conversion  
        variable_order: Variable order used
        labels: Labels for the two conversions (for reporting)
        
    Returns:
        Dictionary with comparison results
    """
    if labels is None:
        labels = ["Conversion 1", "Conversion 2"]
    
    # Analyze both conversions
    analysis1 = analyze_avici_data(avici_data1, variable_order)
    analysis2 = analyze_avici_data(avici_data2, variable_order)
    
    # Compare shapes
    shape_match = avici_data1.shape == avici_data2.shape
    
    # Compare values (allowing for standardization differences)
    values_diff = jnp.mean(jnp.abs(avici_data1[:, :, 0] - avici_data2[:, :, 0]))
    intervention_diff = jnp.mean(jnp.abs(avici_data1[:, :, 1] - avici_data2[:, :, 1]))
    target_diff = jnp.mean(jnp.abs(avici_data1[:, :, 2] - avici_data2[:, :, 2]))
    
    # Exact matches for binary channels
    intervention_match = jnp.allclose(avici_data1[:, :, 1], avici_data2[:, :, 1])
    target_match = jnp.allclose(avici_data1[:, :, 2], avici_data2[:, :, 2])
    
    comparison = {
        'labels': labels,
        'shape_match': shape_match,
        'shapes': [avici_data1.shape, avici_data2.shape],
        'channel_differences': {
            'values': float(values_diff),
            'interventions': float(intervention_diff),
            'targets': float(target_diff)
        },
        'binary_channel_matches': {
            'interventions': intervention_match,
            'targets': target_match
        },
        'individual_analyses': {
            labels[0]: analysis1,
            labels[1]: analysis2
        }
    }
    
    return comparison


def debug_sample_conversion(
    sample: pyr.PMap,
    sample_idx: int,
    avici_data: jnp.ndarray,
    variable_order: VariableOrder,
    target_variable: str
) -> Dict[str, Any]:
    """
    Debug the conversion of a specific sample for detailed inspection.
    
    Args:
        sample: Original Sample object
        sample_idx: Index of sample in the batch
        avici_data: Converted AVICI data tensor
        variable_order: Variable order used in conversion
        target_variable: Target variable name
        
    Returns:
        Dictionary with detailed debug information
    """
    # Extract original sample information
    original_values = dict(sample['values'])
    original_intervention_type = sample['intervention_type']
    original_intervention_targets = set(sample['intervention_targets'])
    
    # Extract converted information
    converted_values = {}
    converted_interventions = {}
    converted_targets = {}
    
    for j, var_name in enumerate(variable_order):
        converted_values[var_name] = float(avici_data[sample_idx, j, 0])
        converted_interventions[var_name] = float(avici_data[sample_idx, j, 1])
        converted_targets[var_name] = float(avici_data[sample_idx, j, 2])
    
    # Check consistency
    intervention_consistency = set()
    for var_name in variable_order:
        if converted_interventions[var_name] > 0.5:
            intervention_consistency.add(var_name)
    
    target_consistency = None
    for var_name in variable_order:
        if converted_targets[var_name] > 0.5:
            target_consistency = var_name
            break
    
    debug_info = {
        'sample_index': sample_idx,
        'original': {
            'values': original_values,
            'intervention_type': original_intervention_type,
            'intervention_targets': sorted(original_intervention_targets)
        },
        'converted': {
            'values': converted_values,
            'intervention_indicators': converted_interventions,
            'target_indicators': converted_targets
        },
        'consistency_checks': {
            'intervention_targets_match': intervention_consistency == original_intervention_targets,
            'target_variable_match': target_consistency == target_variable,
            'expected_target': target_variable,
            'detected_target': target_consistency,
            'expected_interventions': sorted(original_intervention_targets),
            'detected_interventions': sorted(intervention_consistency)
        }
    }
    
    return debug_info