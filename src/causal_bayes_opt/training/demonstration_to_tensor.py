"""
Convert expert demonstrations to 5-channel tensors for BC training.

This module provides a direct transformation from expert demonstrations to
the 5-channel tensor format expected by policy networks, bypassing the
unnecessary buffer and PolicyTrainingData abstractions.

The key insight: Expert demonstrations contain posterior information that
should be included in BC training to teach policies how to use structural
knowledge, not just mimic actions.
"""

import jax.numpy as jnp
from typing import List, Tuple, Dict, Any
import logging

from ..data_structures.scm import get_variables
from ..data_structures.sample import get_values, get_intervention_targets

logger = logging.getLogger(__name__)


def demonstration_to_five_channel_tensor(
    demonstration: Any,
    max_trajectory_length: int = 100
) -> Tuple[List[jnp.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert expert demonstration to 5-channel training examples.
    
    This function extracts the complete trajectory from an expert demonstration
    and creates 5-channel tensor inputs paired with intervention outputs for
    BC training.
    
    Args:
        demonstration: ExpertDemonstration object containing:
            - observational_samples: Initial observations
            - interventional_samples: Samples from interventions
            - parent_posterior: Contains posterior distribution and trajectory
            - scm: The structural causal model
            - target_variable: Target variable name
        max_trajectory_length: Maximum trajectory length to process
        
    Returns:
        Tuple of:
        - input_tensors: List of 5-channel tensors [T, n_vars, 5]
        - intervention_labels: List of intervention dicts with 'targets' and 'values'
        - metadata: Dictionary with debugging/validation info
    """
    # Extract basic information
    variables = sorted(list(get_variables(demonstration.scm)))
    target_variable = demonstration.target_variable
    target_idx = variables.index(target_variable)
    n_vars = len(variables)
    
    # Extract posterior distribution for marginal probabilities
    posterior_dist = demonstration.parent_posterior.get('posterior_distribution', {})
    marginal_probs = _compute_marginal_parent_probabilities(
        posterior_dist, variables, target_variable
    )
    
    # Extract trajectory information
    trajectory = demonstration.parent_posterior.get('trajectory', {})
    intervention_sequence = trajectory.get('intervention_sequence', [])
    intervention_values = trajectory.get('intervention_values', [])
    
    if not intervention_sequence:
        logger.warning("No intervention trajectory found in demonstration")
        return [], [], {'error': 'no_trajectory'}
    
    # Build tensors step by step through the trajectory
    all_samples = list(demonstration.observational_samples)
    input_tensors = []
    intervention_labels = []
    
    # Track intervention history for recency channel
    intervention_history = {var: -1 for var in variables}  # -1 means never intervened
    
    for step, (int_var_spec, int_value) in enumerate(zip(intervention_sequence, intervention_values)):
        if step >= max_trajectory_length:
            break
            
        # Parse intervention variable
        if isinstance(int_var_spec, tuple):
            int_var = int_var_spec[0]
        else:
            int_var = int_var_spec
            
        # Parse intervention value
        if isinstance(int_value, tuple):
            int_val = float(int_value[0])
        else:
            int_val = float(int_value)
            
        # Create 5-channel tensor for current state
        tensor_5ch = _create_five_channel_tensor(
            samples=all_samples,
            variables=variables,
            target_variable=target_variable,
            marginal_probs=marginal_probs,
            intervention_history=intervention_history,
            step=step
        )
        
        # Create intervention label with variable info
        intervention_label = {
            'targets': frozenset([int_var]),
            'values': {int_var: int_val},
            'variables': variables,  # Include variable names for this demo
            'target_variable': target_variable
        }
        
        input_tensors.append(tensor_5ch)
        intervention_labels.append(intervention_label)
        
        # Update intervention history
        intervention_history[int_var] = step
        
        # Add synthetic interventional sample (simplified - in practice would execute intervention)
        # This is a limitation: we don't have the actual post-intervention samples
        # For now, we'll just track that an intervention occurred
    
    metadata = {
        'n_examples': len(input_tensors),
        'variables': variables,
        'target_variable': target_variable,
        'target_idx': target_idx,
        'marginal_probs': marginal_probs,
        'scm_type': demonstration.graph_type,
        'true_parents': demonstration.parent_posterior.get('most_likely_parents', [])
    }
    
    return input_tensors, intervention_labels, metadata


def _compute_marginal_parent_probabilities(
    posterior_distribution: Dict[frozenset, float],
    variables: List[str],
    target_variable: str
) -> jnp.ndarray:
    """
    Compute marginal parent probabilities as a vector.
    
    Args:
        posterior_distribution: Parent set -> probability mapping
        variables: Ordered list of all variables
        target_variable: The target variable
        
    Returns:
        Array of shape [n_vars] with marginal probabilities
    """
    marginal_vector = jnp.zeros(len(variables))
    
    for i, var in enumerate(variables):
        if var != target_variable:
            # Sum probabilities of all parent sets containing this variable
            prob = sum(
                p for parent_set, p in posterior_distribution.items() 
                if var in parent_set
            )
            marginal_vector = marginal_vector.at[i].set(float(prob))
    
    return marginal_vector


def _create_five_channel_tensor(
    samples: List[Any],
    variables: List[str],
    target_variable: str,
    marginal_probs: jnp.ndarray,
    intervention_history: Dict[str, int],
    step: int,
    history_length: int = 20
) -> jnp.ndarray:
    """
    Create a single 5-channel tensor from current state.
    
    Args:
        samples: All samples up to current point
        variables: Ordered list of variables
        target_variable: Target variable name
        marginal_probs: Marginal parent probabilities [n_vars]
        intervention_history: Dict mapping variables to last intervention step
        step: Current step number
        history_length: How many recent samples to include
        
    Returns:
        Tensor of shape [history_length, n_vars, 5]
    """
    n_vars = len(variables)
    tensor = jnp.zeros((history_length, n_vars, 5))
    
    # Use most recent samples
    recent_samples = samples[-history_length:] if len(samples) > history_length else samples
    actual_length = len(recent_samples)
    
    # Fill channels 0-2 from samples
    for t, sample in enumerate(recent_samples):
        t_idx = history_length - actual_length + t
        
        # Channel 0: Values
        values = get_values(sample)
        for i, var in enumerate(variables):
            if var in values:
                tensor = tensor.at[t_idx, i, 0].set(float(values[var]))
        
        # Channel 1: Target indicator
        target_idx = variables.index(target_variable)
        tensor = tensor.at[t_idx, target_idx, 1].set(1.0)
        
        # Channel 2: Intervention indicator
        intervention_targets = get_intervention_targets(sample)
        for i, var in enumerate(variables):
            if var in intervention_targets:
                tensor = tensor.at[t_idx, i, 2].set(1.0)
    
    # Channel 3: Marginal parent probabilities (constant over time)
    for t in range(history_length):
        tensor = tensor.at[t, :, 3].set(marginal_probs)
    
    # Channel 4: Intervention recency
    recency_vector = jnp.zeros(n_vars)
    for i, var in enumerate(variables):
        last_intervention = intervention_history.get(var, -1)
        if last_intervention >= 0:
            recency = step - last_intervention
            # Normalize to [0, 1] with exponential decay
            recency_vector = recency_vector.at[i].set(jnp.exp(-0.1 * recency))
    
    for t in range(history_length):
        tensor = tensor.at[t, :, 4].set(recency_vector)
    
    # Standardize values channel
    values_channel = tensor[:, :, 0]
    mean = jnp.mean(values_channel)
    std = jnp.std(values_channel) + 1e-8
    tensor = tensor.at[:, :, 0].set((values_channel - mean) / std)
    
    return tensor


def create_bc_training_dataset(
    demonstrations: List[Any],
    max_trajectory_length: int = 100
) -> Tuple[List[jnp.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Create complete BC training dataset from multiple demonstrations.
    
    Args:
        demonstrations: List of ExpertDemonstration objects
        max_trajectory_length: Maximum trajectory length per demonstration
        
    Returns:
        Tuple of:
        - all_inputs: List of 5-channel input tensors
        - all_labels: List of intervention labels
        - dataset_metadata: Summary statistics
    """
    all_inputs = []
    all_labels = []
    
    for demo_idx, demo in enumerate(demonstrations):
        try:
            inputs, labels, metadata = demonstration_to_five_channel_tensor(
                demo, max_trajectory_length
            )
            all_inputs.extend(inputs)
            all_labels.extend(labels)
            
            logger.info(f"Processed demonstration {demo_idx}: "
                       f"{metadata['n_examples']} examples from {metadata['scm_type']} SCM")
            
        except Exception as e:
            logger.warning(f"Failed to process demonstration {demo_idx}: {e}")
            continue
    
    # Get variable info from first successfully processed demo
    if all_inputs and demonstrations:
        # Find first valid demo to extract metadata
        for demo in demonstrations:
            try:
                _, _, first_metadata = demonstration_to_five_channel_tensor(demo, 1)
                break
            except:
                continue
    
    dataset_metadata = {
        'total_examples': len(all_inputs),
        'n_demonstrations': len(demonstrations),
        'tensor_shape': all_inputs[0].shape if all_inputs else None,
        'variables': first_metadata.get('variables', []),
        'target_variable': first_metadata.get('target_variable'),
        'target_idx': first_metadata.get('target_idx', 0)
    }
    
    logger.info(f"Created BC training dataset: {dataset_metadata['total_examples']} examples "
               f"from {dataset_metadata['n_demonstrations']} demonstrations")
    
    return all_inputs, all_labels, dataset_metadata