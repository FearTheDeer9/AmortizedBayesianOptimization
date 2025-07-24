#!/usr/bin/env python3
"""
Data Format Adapter for Expert Demonstrations

Converts between ExpertDemonstration format (actual pickled data) and
DemonstrationData format (expected by training pipeline).

Following functional programming principles with immutable conversions.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
from dataclasses import dataclass
import time

import jax.numpy as jnp
import pyrsistent as pyr

from .expert_collection.data_structures import ExpertDemonstration
from .pure_data_loader import DemonstrationData

logger = logging.getLogger(__name__)


def _convert_pmap_samples_to_avici_format(samples: List[pyr.PMap], variable_order: List[str]) -> jnp.ndarray:
    """
    Convert pyrsistent PMap samples to AVICI format [N, d, 3].
    
    Args:
        samples: List of PMap samples with variable -> value mapping
        variable_order: Ordered list of variable names
        
    Returns:
        JAX array in AVICI format [N, d, 3]
    """
    if not samples:
        # Return empty array with correct shape
        return jnp.zeros((0, len(variable_order), 3))
    
    N = len(samples)
    d = len(variable_order)
    
    # Convert samples to structured format
    data = jnp.zeros((N, d, 3))
    
    for i, sample in enumerate(samples):
        for j, var in enumerate(variable_order):
            if var in sample:
                value = float(sample[var])
                # AVICI format: [value, indicator, context]
                data = data.at[i, j, 0].set(value)
                data = data.at[i, j, 1].set(1.0)  # Indicator that value is present
                data = data.at[i, j, 2].set(0.0)  # Context (observational)
    
    return data


def _extract_variable_order_from_scm(scm: pyr.PMap) -> List[str]:
    """
    Extract ordered variable names from SCM structure.
    
    Args:
        scm: SCM structure containing variable information
        
    Returns:
        Ordered list of variable names
    """
    # Try to extract variables from SCM metadata
    if 'variables' in scm:
        variables = scm['variables']
        if isinstance(variables, (list, tuple)):
            return list(variables)
        elif isinstance(variables, (set, frozenset)):
            return sorted(list(variables))
    
    # Fallback: try to extract from edges
    if 'edges' in scm:
        edges = scm['edges']
        all_vars = set()
        if isinstance(edges, (list, tuple)):
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    all_vars.add(edge[0])
                    all_vars.add(edge[1])
        return sorted(list(all_vars))
    
    # Final fallback: empty list
    logger.warning("Could not extract variable order from SCM, using empty list")
    return []


def _create_mock_posterior_history(target_variable: str, discovered_parents: FrozenSet[str]) -> List[Dict[str, Any]]:
    """
    Create mock posterior history based on discovered parents.
    
    Args:
        target_variable: Target variable name
        discovered_parents: Set of discovered parent variables
        
    Returns:
        List of posterior history entries
    """
    # Create simple progression showing parent discovery
    history = []
    
    # Initial uniform posterior
    history.append({
        'step': 0,
        'posterior_entropy': 1.0,
        'top_parent_sets': [
            {'parent_set': frozenset(), 'probability': 1.0}
        ],
        'action': 'initial_observation'
    })
    
    # Add steps for each discovered parent
    for i, parent in enumerate(sorted(discovered_parents)):
        entropy = 1.0 - (i + 1) / (len(discovered_parents) + 1)
        
        # Create posterior with increasing confidence
        parent_set = frozenset(list(sorted(discovered_parents))[:i+1])
        prob = 0.5 + 0.4 * (i + 1) / len(discovered_parents)
        
        history.append({
            'step': i + 1,
            'posterior_entropy': entropy,
            'top_parent_sets': [
                {'parent_set': parent_set, 'probability': prob},
                {'parent_set': frozenset(), 'probability': 1.0 - prob}
            ],
            'action': f'intervene_{parent}'
        })
    
    return history


def _create_mock_intervention_sequence(discovered_parents: FrozenSet[str]) -> List[Dict[str, Any]]:
    """
    Create mock intervention sequence based on discovered parents.
    
    Args:
        discovered_parents: Set of discovered parent variables
        
    Returns:
        List of intervention entries
    """
    interventions = []
    
    # Create intervention for each discovered parent
    for i, parent in enumerate(sorted(discovered_parents)):
        interventions.append({
            'step': i + 1,
            'variable': parent,
            'value': 1.0,  # Simple intervention value
            'type': 'do_intervention',
            'samples_collected': 50  # Mock sample count
        })
    
    return interventions


def convert_expert_demonstration_to_training_data(
    expert_demo: ExpertDemonstration,
    demo_id: Optional[str] = None
) -> DemonstrationData:
    """
    Convert ExpertDemonstration to DemonstrationData format.
    
    Args:
        expert_demo: Original expert demonstration
        demo_id: Optional demonstration ID
        
    Returns:
        Converted demonstration data
    """
    # Generate demo ID if not provided
    if demo_id is None:
        demo_id = f"demo_{int(time.time() * 1000)}"
    
    # Extract variable order from SCM
    variable_order = _extract_variable_order_from_scm(expert_demo.scm)
    
    # If no variables found, create minimal order with target
    if not variable_order:
        variable_order = [expert_demo.target_variable]
        if expert_demo.discovered_parents:
            variable_order.extend(sorted(expert_demo.discovered_parents))
    
    # Ensure target variable is in variable order
    if expert_demo.target_variable not in variable_order:
        variable_order.insert(0, expert_demo.target_variable)
    
    # Convert observational samples to AVICI format
    observational_avici = _convert_pmap_samples_to_avici_format(
        expert_demo.observational_samples, 
        variable_order
    )
    
    # Convert interventional samples to AVICI format
    interventional_avici = _convert_pmap_samples_to_avici_format(
        expert_demo.interventional_samples, 
        variable_order
    )
    
    # Combine observational and interventional data
    if observational_avici.shape[0] > 0 and interventional_avici.shape[0] > 0:
        # Mark interventional samples in context dimension
        interventional_marked = interventional_avici.at[:, :, 2].set(1.0)
        avici_data = jnp.concatenate([observational_avici, interventional_marked], axis=0)
    elif observational_avici.shape[0] > 0:
        avici_data = observational_avici
    elif interventional_avici.shape[0] > 0:
        avici_data = interventional_avici.at[:, :, 2].set(1.0)
    else:
        # No data - create minimal structure
        avici_data = jnp.zeros((1, len(variable_order), 3))
    
    # Create mock posterior history
    posterior_history = _create_mock_posterior_history(
        expert_demo.target_variable,
        expert_demo.discovered_parents
    )
    
    # Create mock intervention sequence
    intervention_sequence = _create_mock_intervention_sequence(
        expert_demo.discovered_parents
    )
    
    # Create DemonstrationData
    return DemonstrationData(
        demo_id=demo_id,
        avici_data=avici_data,
        target_variable=expert_demo.target_variable,
        variable_order=variable_order,
        posterior_history=posterior_history,
        intervention_sequence=intervention_sequence,
        expert_accuracy=expert_demo.accuracy,
        confidence_score=expert_demo.confidence,
        metadata=pyr.pmap({
            'original_format': 'ExpertDemonstration',
            'conversion_timestamp': time.time(),
            'n_nodes': expert_demo.n_nodes,
            'graph_type': expert_demo.graph_type,
            'discovered_parents': list(expert_demo.discovered_parents),
            'inference_time': expert_demo.inference_time,
            'total_samples_used': expert_demo.total_samples_used,
            'collection_timestamp': expert_demo.collection_timestamp,
            'validation_passed': expert_demo.validation_passed,
            'observational_samples': len(expert_demo.observational_samples),
            'interventional_samples': len(expert_demo.interventional_samples)
        })
    )


def convert_expert_demonstrations_batch(
    expert_demos: List[ExpertDemonstration],
    demo_id_prefix: str = "converted"
) -> List[DemonstrationData]:
    """
    Convert a batch of ExpertDemonstration objects to DemonstrationData format.
    
    Args:
        expert_demos: List of expert demonstrations
        demo_id_prefix: Prefix for generated demo IDs
        
    Returns:
        List of converted demonstration data
    """
    converted_demos = []
    
    for i, expert_demo in enumerate(expert_demos):
        demo_id = f"{demo_id_prefix}_{i:04d}"
        try:
            converted_demo = convert_expert_demonstration_to_training_data(expert_demo, demo_id)
            converted_demos.append(converted_demo)
        except Exception as e:
            logger.error(f"Failed to convert demonstration {i}: {e}")
            continue
    
    logger.info(f"Successfully converted {len(converted_demos)}/{len(expert_demos)} demonstrations")
    return converted_demos


def validate_converted_data(demo_data: DemonstrationData) -> bool:
    """
    Validate that converted data meets expected format requirements.
    
    Args:
        demo_data: Converted demonstration data
        
    Returns:
        True if validation passes
    """
    try:
        # Check basic structure
        assert demo_data.demo_id is not None
        assert demo_data.avici_data is not None
        assert demo_data.target_variable is not None
        assert demo_data.variable_order is not None
        
        # Check AVICI data format
        assert len(demo_data.avici_data.shape) == 3
        N, d, channels = demo_data.avici_data.shape
        assert channels == 3
        assert d == len(demo_data.variable_order)
        
        # Check target variable is in variable order
        assert demo_data.target_variable in demo_data.variable_order
        
        # Check posterior history structure
        assert isinstance(demo_data.posterior_history, list)
        for entry in demo_data.posterior_history:
            assert 'step' in entry
            assert 'posterior_entropy' in entry
            assert 'top_parent_sets' in entry
            assert 'action' in entry
        
        # Check intervention sequence structure
        assert isinstance(demo_data.intervention_sequence, list)
        for entry in demo_data.intervention_sequence:
            assert 'step' in entry
            assert 'variable' in entry
            assert 'value' in entry
            assert 'type' in entry
        
        # Check metadata
        assert demo_data.metadata is not None
        assert 'original_format' in demo_data.metadata
        
        return True
        
    except Exception as e:
        logger.error(f"Validation failed for demo {demo_data.demo_id}: {e}")
        return False