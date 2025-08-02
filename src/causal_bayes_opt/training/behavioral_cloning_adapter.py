#!/usr/bin/env python3
"""
Behavioral Cloning Adapter for Expert Demonstrations

Pure functional adapter that transforms expert demonstrations from PARENT_SCALE
optimization trajectories into training data formats for both surrogate and
acquisition models.

Key Features:
1. Pure functions with no side effects
2. Transforms demonstrations to AVICI [N, d, 3] format
3. Extracts (state, action) pairs for behavioral cloning
4. Handles variable graph sizes and types
5. Integrates with existing JAX infrastructure

Design Principles (Rich Hickey Approved):
- Simple over easy: Clear data transformations
- Pure functions: No side effects or mutations
- Single responsibility: Each function does ONE thing well
- Composable design: Build complexity from simple functions
"""

import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional, FrozenSet
from pathlib import Path

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

# Import existing infrastructure
from ..avici_integration.core.conversion import samples_to_avici_format
from ..avici_integration.core.standardization import compute_standardization_params
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState
from ..data_structures.scm import get_variables, get_edges
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import create_sample
from .expert_collection.data_structures import ExpertDemonstration, DemonstrationBatch
# Import TrainingExample from shared data structures
from .data_structures import TrainingExample
from ..avici_integration.parent_set.posterior import ParentSetPosterior

logger = logging.getLogger(__name__)


def load_demonstration_batch(path: str) -> DemonstrationBatch:
    """
    Pure function: Load expert demonstration batch from pickle file.
    
    Args:
        path: Path to pickle file containing DemonstrationBatch
        
    Returns:
        DemonstrationBatch object with expert demonstrations
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.UnpicklingError: If file is corrupted
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Demonstration file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            batch = pickle.load(f)
        
        if not isinstance(batch, DemonstrationBatch):
            raise ValueError(f"Expected DemonstrationBatch, got {type(batch)}")
            
        return batch
    except Exception as e:
        raise pickle.UnpicklingError(f"Failed to load demonstration batch: {e}")


def extract_avici_samples(demo: ExpertDemonstration) -> jnp.ndarray:
    """
    Pure function: Convert expert demonstration samples to AVICI [N, d, 3] format.
    
    Args:
        demo: Expert demonstration with observational and interventional samples
        
    Returns:
        JAX array of shape [N, d, 3] where:
        - [:, :, 0] = variable values (standardized)
        - [:, :, 1] = intervention indicators
        - [:, :, 2] = target indicators
    """
    # Extract variables and create consistent ordering
    variables = sorted(list(get_variables(demo.scm)))
    target_variable = demo.target_variable
    
    # Combine observational and interventional samples
    all_samples = demo.observational_samples + demo.interventional_samples
    
    if not all_samples:
        raise ValueError("No samples found in demonstration")
    
    # Convert to AVICI format using existing utility
    # The samples_to_avici_format expects samples with proper structure
    avici_data = samples_to_avici_format(
        samples=all_samples,
        variable_order=variables,
        target_variable=target_variable,
        standardization_params=None  # Let it compute standardization
    )
    
    return avici_data


def extract_posterior_history(demo: ExpertDemonstration) -> List[Dict[FrozenSet[str], float]]:
    """
    Pure function: Extract posterior history from expert demonstration.
    
    Args:
        demo: Expert demonstration with trajectory information
        
    Returns:
        List of posterior distributions at each step
    """
    trajectory = demo.parent_posterior.get('trajectory', {})
    posterior_history = trajectory.get('posterior_history', [])
    
    if not posterior_history:
        # Fallback to final posterior if no history available
        final_posterior = demo.parent_posterior.get('posterior_distribution', {})
        if final_posterior:
            return [final_posterior]
        else:
            raise ValueError("No posterior information found in demonstration")
    
    # Convert posterior history to consistent format
    posteriors = []
    for step in posterior_history:
        posterior_dict = step.get('posterior', {})
        
        # Convert tuple keys to frozensets for consistency
        converted_posterior = {}
        for parents, prob in posterior_dict.items():
            if isinstance(parents, tuple):
                parent_set = frozenset(parents)
            elif isinstance(parents, frozenset):
                parent_set = parents
            else:
                parent_set = frozenset()
            converted_posterior[parent_set] = float(prob)
        
        posteriors.append(converted_posterior)
    
    return posteriors


def create_experience_buffer_from_demo(demo: ExpertDemonstration) -> ExperienceBuffer:
    """
    Pure function: Create ExperienceBuffer from expert demonstration data.
    
    Args:
        demo: Expert demonstration with observational and interventional samples
        
    Returns:
        ExperienceBuffer containing the demonstration data
    """
    buffer = ExperienceBuffer()
    
    # Add observational samples
    for obs_sample in demo.observational_samples:
        if isinstance(obs_sample, pyr.PMap):
            # Already in the right format
            values = obs_sample.get('values', obs_sample)
        else:
            # Convert dict to values dict
            values = obs_sample
        
        # Create sample in proper format
        sample = create_sample(
            values=dict(values),
            intervention_type=None,  # Observational
            intervention_targets=None,
            metadata={'source': 'expert_demonstration'}
        )
        buffer.add_observation(sample)
    
    # Add interventional samples
    for int_sample in demo.interventional_samples:
        if isinstance(int_sample, pyr.PMap):
            values = int_sample.get('values', int_sample)
            intervention_targets = int_sample.get('intervention_targets', frozenset())
            intervention_type = int_sample.get('intervention_type', 'perfect')
        else:
            # Basic dict format - try to infer intervention info
            values = int_sample
            intervention_targets = frozenset()  # Will be populated if available
            intervention_type = 'perfect'
        
        # Create intervention sample
        sample = create_sample(
            values=dict(values),
            intervention_type=intervention_type,
            intervention_targets=intervention_targets,
            metadata={'source': 'expert_demonstration'}
        )
        
        # For interventional samples, we need to create an intervention descriptor
        # For simplicity, use empty intervention - this could be enhanced
        intervention = pyr.m(
            type=intervention_type,
            targets=intervention_targets,
            values=pyr.m()  # Values would be specified in real interventions
        )
        
        buffer.add_intervention(intervention, sample)
    
    return buffer


def extract_intervention_sequence(demo: ExpertDemonstration) -> List[Tuple[FrozenSet[str], Tuple[float, ...]]]:
    """
    Pure function: Extract intervention sequence from expert trajectory.
    
    Args:
        demo: Expert demonstration with trajectory information
        
    Returns:
        List of (intervention_variables, intervention_values) pairs
    """
    trajectory = demo.parent_posterior.get('trajectory', {})
    intervention_sequence = trajectory.get('intervention_sequence', [])
    intervention_values = trajectory.get('intervention_values', [])
    
    if len(intervention_sequence) != len(intervention_values):
        raise ValueError("Mismatch between intervention sequence and values")
    
    # Convert to consistent format
    interventions = []
    for vars_tuple, values_tuple in zip(intervention_sequence, intervention_values):
        if isinstance(vars_tuple, tuple):
            vars_set = frozenset(vars_tuple)
        else:
            vars_set = frozenset()
            
        if isinstance(values_tuple, (tuple, list)):
            values = tuple(float(v) for v in values_tuple)
        else:
            values = (float(values_tuple),)
            
        interventions.append((vars_set, values))
    
    return interventions


def create_parent_set_posterior_from_demo(
    demo: ExpertDemonstration, 
    step: int = 0
) -> ParentSetPosterior:
    """
    Pure function: Create ParentSetPosterior from expert demonstration.
    
    Args:
        demo: Expert demonstration with posterior information
        step: Step in trajectory (default: use final posterior)
        
    Returns:
        ParentSetPosterior object
    """
    target_variable = demo.target_variable
    
    # Extract posterior history
    try:
        posterior_history = extract_posterior_history(demo)
        if step < len(posterior_history):
            posterior_dict = posterior_history[step]
        else:
            posterior_dict = posterior_history[-1] if posterior_history else {}
    except ValueError:
        # Fallback to empty posterior
        posterior_dict = {}
    
    if not posterior_dict:
        # Create minimal posterior with empty parent set
        posterior_dict = {frozenset(): 1.0}
    
    # Convert to proper format
    parent_set_probs = pyr.pmap(posterior_dict)
    
    # Calculate uncertainty (entropy)
    probabilities = list(posterior_dict.values())
    uncertainty = -sum(p * jnp.log(p + 1e-10) for p in probabilities if p > 0)
    
    # Get top-k parent sets
    top_k_sets = sorted(
        posterior_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    return ParentSetPosterior(
        target_variable=target_variable,
        parent_set_probs=parent_set_probs,
        uncertainty=float(uncertainty),
        top_k_sets=top_k_sets,
        metadata=pyr.m(
            demo_id=getattr(demo, 'id', 'unknown'),
            step=step,
            source='expert_demonstration'
        )
    )


def create_acquisition_state(
    demo: ExpertDemonstration,
    step: int,
    avici_data: jnp.ndarray,
    intervention_history: List[Tuple[FrozenSet[str], Tuple[float, ...]]]
) -> AcquisitionState:
    """
    Pure function: Create AcquisitionState for a specific step in trajectory.
    
    Args:
        demo: Expert demonstration
        step: Step number in trajectory
        avici_data: AVICI-formatted sample data
        intervention_history: History of interventions up to this step
        
    Returns:
        AcquisitionState object representing the state at this step
    """
    target_variable = demo.target_variable
    
    # Create ExperienceBuffer from demonstration data
    buffer = create_experience_buffer_from_demo(demo)
    
    # Create ParentSetPosterior at this step
    posterior = create_parent_set_posterior_from_demo(demo, step)
    
    # Calculate best value from samples (simplified - use target variable mean)
    all_samples = demo.observational_samples + demo.interventional_samples
    target_values = []
    for sample in all_samples:
        if isinstance(sample, pyr.PMap):
            values = sample.get('values', sample)
        else:
            values = sample
        
        if target_variable in values:
            try:
                target_values.append(float(values[target_variable]))
            except (ValueError, TypeError):
                continue
    
    best_value = max(target_values) if target_values else 0.0
    
    # Get variables from SCM using the same approach as in extract_avici_samples
    variables = sorted(list(get_variables(demo.scm)))
    
    # Create SCM info for variable indexing
    scm_info = {
        'variables': variables,
        'n_nodes': demo.n_nodes,
        'graph_type': demo.graph_type
    }
    
    # Create metadata with demonstration info and SCM info
    metadata = pyr.m(
        n_nodes=demo.n_nodes,
        graph_type=demo.graph_type,
        demo_step=step,
        total_interventions=len(intervention_history),
        accuracy=demo.accuracy,
        confidence=demo.confidence,
        source='expert_demonstration',
        scm_info=scm_info  # Store SCM info in metadata for variable indexing
    )
    
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=best_value,
        current_target=target_variable,
        step=step,
        metadata=metadata
    )


def extract_expert_action(
    demo: ExpertDemonstration,
    step: int,
    intervention_sequence: List[Tuple[FrozenSet[str], Tuple[float, ...]]]
) -> Dict[str, Any]:
    """
    Pure function: Extract expert action at a specific step.
    
    Args:
        demo: Expert demonstration
        step: Step number in trajectory
        intervention_sequence: Full intervention sequence
        
    Returns:
        Dictionary representing the expert's action at this step
    """
    if step >= len(intervention_sequence):
        raise ValueError(f"Step {step} beyond intervention sequence length {len(intervention_sequence)}")
    
    intervention_vars, intervention_vals = intervention_sequence[step]
    
    return {
        'intervention_variables': intervention_vars,
        'intervention_values': intervention_vals,
        'step': step
    }


def create_surrogate_training_example(
    demo: ExpertDemonstration,
    step: int,
    avici_data: jnp.ndarray
) -> TrainingExample:
    """
    Pure function: Create TrainingExample for surrogate model training.
    
    Args:
        demo: Expert demonstration
        step: Step number in trajectory  
        avici_data: AVICI-formatted sample data
        
    Returns:
        TrainingExample for surrogate model behavioral cloning
    """
    variables = sorted(list(get_variables(demo.scm)))
    target_variable = demo.target_variable
    
    # Get posterior at this step
    posterior_history = extract_posterior_history(demo)
    if step < len(posterior_history):
        posterior_dict = posterior_history[step]
    else:
        posterior_dict = posterior_history[-1] if posterior_history else {}
    
    # Convert to ParentSetPosterior format
    parent_sets = []
    probabilities = []
    
    for parent_set, prob in posterior_dict.items():
        parent_sets.append(parent_set)
        probabilities.append(float(prob))
    
    if not parent_sets:
        # Create empty posterior as fallback
        parent_sets = [frozenset()]
        probabilities = [1.0]
    
    # Create ParentSetPosterior object
    parent_set_probs = pyr.pmap(dict(zip(parent_sets, probabilities)))
    uncertainty = -sum(p * jnp.log(p + 1e-10) for p in probabilities)  # Entropy
    top_k_sets = sorted(zip(parent_sets, probabilities), key=lambda x: x[1], reverse=True)[:5]
    
    expert_posterior = ParentSetPosterior(
        target_variable=target_variable,
        parent_set_probs=parent_set_probs,
        uncertainty=float(uncertainty),
        top_k_sets=top_k_sets
    )
    
    # Extract parent sets and probabilities from expert posterior
    parent_sets = list(parent_set_probs.keys())
    expert_probs = jnp.array(list(parent_set_probs.values()))
    
    return TrainingExample(
        observational_data=avici_data,
        target_variable=target_variable,
        variable_order=variables,
        expert_posterior=expert_posterior,
        expert_accuracy=demo.accuracy,
        problem_difficulty='medium',  # Will be refined by curriculum manager
        parent_sets=parent_sets,
        expert_probs=expert_probs
    )


def compute_demonstration_complexity(demo: ExpertDemonstration) -> float:
    """
    Pure function: Compute complexity score for demonstration.
    
    Args:
        demo: Expert demonstration
        
    Returns:
        Complexity score (higher = more complex)
    """
    # Base complexity from graph structure
    n_nodes = demo.n_nodes
    edges = get_edges(demo.scm)
    edge_density = len(edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
    
    # Complexity from convergence behavior
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
    
    # Combined complexity score
    complexity = (
        n_nodes * 0.3 +               # Node count contribution
        edge_density * 10.0 +         # Edge density contribution  
        n_iterations * 0.1 +          # Convergence difficulty
        graph_weight * 2.0            # Graph type complexity
    )
    
    return float(complexity)


def load_all_demonstrations(demo_dir: str) -> List[ExpertDemonstration]:
    """
    Pure function: Load all expert demonstrations from directory.
    
    Args:
        demo_dir: Directory containing demonstration pickle files
        
    Returns:
        List of all expert demonstrations
    """
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        raise FileNotFoundError(f"Demonstration directory not found: {demo_dir}")
    
    all_demonstrations = []
    pickle_files = list(demo_path.glob("*.pkl"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {demo_dir}")
    
    logger.info(f"Loading {len(pickle_files)} demonstration files")
    
    for pickle_file in pickle_files:
        try:
            batch = load_demonstration_batch(str(pickle_file))
            all_demonstrations.extend(batch.demonstrations)
        except Exception as e:
            logger.warning(f"Failed to load {pickle_file}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_demonstrations)} total demonstrations")
    return all_demonstrations