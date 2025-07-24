#!/usr/bin/env python3
"""
Pure Data Loader for Behavioral Cloning

Immutable, functional data loading following Rich Hickey's principles:
- Pure functions with no side effects
- Fail fast with clear error messages
- Complete type safety (no Any types)
- JAX-native operations for performance
- Immutable data structures using pyrsistent

This replaces the problematic behavioral_cloning_adapter.py and data_extraction_fixed.py
"""

import pickle
import logging
from typing import List, Dict, FrozenSet, Tuple, Optional, NamedTuple, Union
from pathlib import Path
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr
from pyrsistent import PRecord, field, PVector

# Import existing infrastructure
from ..avici_integration.core.conversion import samples_to_avici_format
from ..data_structures.scm import get_variables, get_edges
from .expert_collection.data_structures import ExpertDemonstration, DemonstrationBatch

logger = logging.getLogger(__name__)


# Immutable data structures
class AVICIData(NamedTuple):
    """Immutable AVICI data representation"""
    samples: jnp.ndarray  # [N, d, 3] format
    variables: Tuple[str, ...]  # Ordered variable names
    target_variable: str
    sample_count: int


class PosteriorStep(NamedTuple):
    """Immutable posterior distribution at a single step"""
    step: int
    posterior: Dict[FrozenSet[str], float]  # parent_set -> probability
    entropy: float


class InterventionStep(NamedTuple):
    """Immutable intervention representation"""
    step: int
    variables: FrozenSet[str]
    values: Tuple[float, ...]


class DemonstrationData(PRecord):
    """Immutable demonstration data structure - JAX-native"""
    demo_id: str = field(type=str)
    avici_data = field()  # JAX array [N, d, 3] 
    target_variable: str = field(type=str)
    variable_order = field()  # List[str]
    posterior_history = field()  # List[Dict[str, Any]]
    intervention_sequence = field()  # List[Dict[str, Any]]
    expert_accuracy: float = field(type=float)
    confidence_score: float = field(type=float)
    metadata = field()  # pyr.PMap for additional data


class LoadError(Exception):
    """Specific error for data loading failures"""
    pass


class ValidationError(Exception):
    """Specific error for data validation failures"""
    pass


def load_demonstration_batch(file_path: str) -> DemonstrationBatch:
    """
    Pure function: Load demonstration batch from pickle file.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        DemonstrationBatch object
        
    Raises:
        LoadError: If file cannot be loaded or is invalid
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise LoadError(f"Demonstration file not found: {file_path}")
    
    if not path_obj.is_file():
        raise LoadError(f"Path is not a file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            batch = pickle.load(f)
    except Exception as e:
        raise LoadError(f"Failed to unpickle file {file_path}: {str(e)}")
    
    if not isinstance(batch, DemonstrationBatch):
        raise LoadError(f"Expected DemonstrationBatch, got {type(batch).__name__}")
    
    if not batch.demonstrations:
        raise LoadError(f"Batch contains no demonstrations: {file_path}")
    
    return batch


@jax.jit
def _compute_entropy(probabilities: jnp.ndarray) -> float:
    """JAX-optimized entropy computation"""
    # Add small epsilon to prevent log(0)
    safe_probs = jnp.where(probabilities > 0, probabilities, 1e-10)
    return -jnp.sum(probabilities * jnp.log(safe_probs))


def _validate_demonstration(demo: ExpertDemonstration) -> None:
    """
    Validate demonstration data structure.
    
    Args:
        demo: Demonstration to validate
        
    Raises:
        ValidationError: If demonstration is invalid
    """
    if not isinstance(demo, ExpertDemonstration):
        raise ValidationError(f"Expected ExpertDemonstration, got {type(demo).__name__}")
    
    if demo.n_nodes <= 0:
        raise ValidationError(f"Invalid node count: {demo.n_nodes}")
    
    if not demo.target_variable:
        raise ValidationError("Missing target variable")
    
    if not demo.scm:
        raise ValidationError("Missing SCM")
    
    if not (demo.observational_samples or demo.interventional_samples):
        raise ValidationError("No samples available")
    
    if not (0 <= demo.accuracy <= 1):
        raise ValidationError(f"Invalid accuracy: {demo.accuracy}")


def _extract_avici_data(demo: ExpertDemonstration) -> AVICIData:
    """
    Pure function: Extract AVICI data from demonstration.
    
    Args:
        demo: Validated demonstration
        
    Returns:
        AVICIData object
        
    Raises:
        ValidationError: If data extraction fails
    """
    try:
        variables = sorted(list(get_variables(demo.scm)))
        all_samples = demo.observational_samples + demo.interventional_samples
        
        if not all_samples:
            raise ValidationError("No samples to extract")
        
        # Convert to AVICI format
        avici_array = samples_to_avici_format(
            samples=all_samples,
            variable_order=variables,
            target_variable=demo.target_variable,
            standardization_params=None
        )
        
        return AVICIData(
            samples=avici_array,
            variables=tuple(variables),
            target_variable=demo.target_variable,
            sample_count=len(all_samples)
        )
        
    except Exception as e:
        raise ValidationError(f"Failed to extract AVICI data: {str(e)}")


def _extract_posterior_history(demo: ExpertDemonstration) -> PVector[PosteriorStep]:
    """
    Pure function: Extract posterior history from demonstration.
    
    Args:
        demo: Validated demonstration
        
    Returns:
        PVector of PosteriorStep objects
        
    Raises:
        ValidationError: If posterior extraction fails
    """
    try:
        trajectory = demo.parent_posterior.get('trajectory', {})
        posterior_history = trajectory.get('posterior_history', [])
        
        if not posterior_history:
            # Use final posterior as fallback
            final_posterior = demo.parent_posterior.get('posterior_distribution', {})
            if not final_posterior:
                raise ValidationError("No posterior information available")
            posterior_history = [{'posterior': final_posterior}]
        
        steps = []
        for i, step_data in enumerate(posterior_history):
            posterior_dict = step_data.get('posterior', {})
            
            # Convert to consistent format
            converted_posterior = {}
            probabilities = []
            
            for parents, prob in posterior_dict.items():
                if isinstance(parents, tuple):
                    parent_set = frozenset(parents)
                elif isinstance(parents, frozenset):
                    parent_set = parents
                else:
                    parent_set = frozenset()
                
                prob_float = float(prob)
                converted_posterior[parent_set] = prob_float
                probabilities.append(prob_float)
            
            # Compute entropy
            if probabilities:
                entropy = float(_compute_entropy(jnp.array(probabilities)))
            else:
                entropy = 0.0
            
            steps.append(PosteriorStep(
                step=i,
                posterior=converted_posterior,
                entropy=entropy
            ))
        
        return pyr.pvector(steps)
        
    except Exception as e:
        raise ValidationError(f"Failed to extract posterior history: {str(e)}")


def _extract_intervention_sequence(demo: ExpertDemonstration) -> PVector[InterventionStep]:
    """
    Pure function: Extract intervention sequence from demonstration.
    
    Args:
        demo: Validated demonstration
        
    Returns:
        PVector of InterventionStep objects
        
    Raises:
        ValidationError: If intervention extraction fails
    """
    try:
        trajectory = demo.parent_posterior.get('trajectory', {})
        intervention_sequence = trajectory.get('intervention_sequence', [])
        intervention_values = trajectory.get('intervention_values', [])
        
        if len(intervention_sequence) != len(intervention_values):
            raise ValidationError("Mismatch between intervention sequence and values")
        
        steps = []
        for i, (vars_tuple, values_tuple) in enumerate(zip(intervention_sequence, intervention_values)):
            # Convert variables to frozenset
            if isinstance(vars_tuple, tuple):
                vars_set = frozenset(vars_tuple)
            else:
                vars_set = frozenset()
            
            # Convert values to tuple
            if isinstance(values_tuple, (tuple, list)):
                values = tuple(float(v) for v in values_tuple)
            else:
                values = (float(values_tuple),)
            
            steps.append(InterventionStep(
                step=i,
                variables=vars_set,
                values=values
            ))
        
        return pyr.pvector(steps)
        
    except Exception as e:
        raise ValidationError(f"Failed to extract intervention sequence: {str(e)}")


def _compute_complexity_score(
    n_nodes: int,
    edge_count: int,
    n_iterations: int,
    graph_type_weight: float
) -> float:
    """Complexity score computation"""
    edge_density = edge_count / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
    
    complexity = (
        n_nodes * 0.3 +               # Node count contribution
        edge_density * 10.0 +         # Edge density contribution  
        n_iterations * 0.1 +          # Convergence difficulty
        graph_type_weight * 2.0       # Graph type complexity
    )
    
    return float(complexity)


def _compute_demonstration_complexity(demo: ExpertDemonstration) -> float:
    """
    Pure function: Compute complexity score for demonstration.
    
    Args:
        demo: Validated demonstration
        
    Returns:
        Complexity score (higher = more complex)
    """
    try:
        edges = get_edges(demo.scm)
        edge_count = len(edges)
        
        trajectory = demo.parent_posterior.get('trajectory', {})
        n_iterations = trajectory.get('iterations', 1)
        
        # Graph type complexity weights
        graph_type_weights = {
            'chain': 1.0,
            'fork': 1.2,
            'collider': 1.5,
            'erdos_renyi': 2.0,
            'scale_free': 2.5
        }
        graph_weight = graph_type_weights.get(demo.graph_type, 1.0)
        
        return _compute_complexity_score(
            demo.n_nodes,
            edge_count,
            n_iterations,
            graph_weight
        )
        
    except Exception as e:
        raise ValidationError(f"Failed to compute complexity: {str(e)}")


def process_demonstration(demo: ExpertDemonstration, demo_id: str) -> DemonstrationData:
    """
    Pure function: Process a single demonstration into immutable data structure.
    
    Args:
        demo: Expert demonstration
        demo_id: Unique identifier for this demonstration
        
    Returns:
        DemonstrationData object
        
    Raises:
        ValidationError: If processing fails
    """
    # Validate input
    _validate_demonstration(demo)
    
    # Extract all components
    avici_data = _extract_avici_data(demo)
    posterior_history = _extract_posterior_history(demo)
    intervention_sequence = _extract_intervention_sequence(demo)
    complexity_score = _compute_demonstration_complexity(demo)
    
    return DemonstrationData(
        demo_id=demo_id,
        avici_data=avici_data,
        target_variable=demo.target_variable,
        variable_order=list(get_variables(demo.scm)) if demo.scm else [],
        posterior_history=posterior_history,
        intervention_sequence=intervention_sequence,
        expert_accuracy=demo.accuracy,
        confidence_score=complexity_score,
        metadata=pyr.pmap({
            'n_nodes': demo.n_nodes,
            'graph_type': demo.graph_type,
            'original_demo': demo
        })
    )


def load_demonstrations_from_directory(demo_dir: str) -> List[DemonstrationData]:
    """
    Pure function: Load all demonstrations from directory.
    
    Args:
        demo_dir: Directory containing pickle files
        
    Returns:
        List of DemonstrationData objects
        
    Raises:
        LoadError: If directory cannot be processed
    """
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        raise LoadError(f"Directory not found: {demo_dir}")
    
    if not demo_path.is_dir():
        raise LoadError(f"Path is not a directory: {demo_dir}")
    
    pickle_files = list(demo_path.glob("*.pkl"))
    if not pickle_files:
        raise LoadError(f"No pickle files found in {demo_dir}")
    
    logger.info(f"Loading {len(pickle_files)} demonstration files")
    
    all_demonstrations = []
    failed_count = 0
    
    for pickle_file in pickle_files:
        try:
            batch = load_demonstration_batch(str(pickle_file))
            
            for i, demo in enumerate(batch.demonstrations):
                demo_id = f"{pickle_file.stem}_{i}"
                try:
                    demo_data = process_demonstration(demo, demo_id)
                    all_demonstrations.append(demo_data)
                except ValidationError as e:
                    logger.warning(f"Skipping invalid demonstration {demo_id}: {e}")
                    failed_count += 1
                    
        except LoadError as e:
            logger.warning(f"Skipping file {pickle_file}: {e}")
            failed_count += 1
    
    if not all_demonstrations:
        raise LoadError(f"No valid demonstrations loaded from {demo_dir}")
    
    logger.info(f"Successfully loaded {len(all_demonstrations)} demonstrations, "
                f"failed to load {failed_count}")
    
    return all_demonstrations


def validate_demonstration_data(demo_data: DemonstrationData) -> None:
    """
    Pure function: Validate processed demonstration data.
    
    Args:
        demo_data: DemonstrationData to validate
        
    Raises:
        ValidationError: If data is invalid
    """
    if not demo_data.demo_id:
        raise ValidationError("Missing demo ID")
    
    if demo_data.n_nodes <= 0:
        raise ValidationError(f"Invalid node count: {demo_data.n_nodes}")
    
    if not demo_data.target_variable:
        raise ValidationError("Missing target variable")
    
    if demo_data.avici_data.samples.shape[0] == 0:
        raise ValidationError("No samples in AVICI data")
    
    if len(demo_data.posterior_history) == 0:
        raise ValidationError("No posterior history")
    
    if demo_data.complexity_score < 0:
        raise ValidationError(f"Invalid complexity score: {demo_data.complexity_score}")