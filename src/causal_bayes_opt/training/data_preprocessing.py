#!/usr/bin/env python3
"""
Data preprocessing pipeline for ACBO demonstrations.

This module provides clean, efficient conversion of collected expert demonstrations
into formats suitable for surrogate and policy training. It replaces the complex
runtime conversion logic with a preprocessing step that handles all data transformation
once, before training begins.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import logging
import jax.numpy as jnp
import numpy as onp

from ..data_structures.scm import get_variables
from ..avici_integration.core import samples_to_avici_format

logger = logging.getLogger(__name__)


@dataclass
class SurrogateTrainingData:
    """Preprocessed data for surrogate model training."""
    state_tensor: jnp.ndarray  # [N, d, 3]
    target_idx: int
    marginal_parent_probs: Dict[str, float]  # Variable -> probability
    variables: List[str]
    scm_id: str  # For tracking


@dataclass
class PolicyTrainingData:
    """Preprocessed data for policy model training."""
    states: jnp.ndarray  # [T, d, 3]
    intervention_vars: jnp.ndarray  # [T-1]
    intervention_values: jnp.ndarray  # [T-1]
    rewards: jnp.ndarray  # [T-1]
    target_idx: int
    variables: List[str]
    scm_id: str  # For tracking


def compute_marginal_parent_probabilities(
    posterior_distribution: Dict[frozenset, float],
    variables: List[str],
    target_variable: str
) -> Dict[str, float]:
    """
    Compute marginal parent probabilities from posterior distribution.
    
    Args:
        posterior_distribution: Parent set -> probability mapping
        variables: All variables in the SCM
        target_variable: The target variable
        
    Returns:
        Dictionary mapping each variable to its marginal parent probability
    """
    marginals = {}
    
    for var in variables:
        if var != target_variable:
            # Sum probabilities of all parent sets containing this variable
            prob = sum(
                p for parent_set, p in posterior_distribution.items() 
                if var in parent_set
            )
            marginals[var] = float(prob)
    
    return marginals


def extract_surrogate_training_data(demonstration: Any) -> SurrogateTrainingData:
    """
    Extract and preprocess data for surrogate model training.
    
    Args:
        demonstration: ExpertDemonstration object
        
    Returns:
        SurrogateTrainingData with all necessary information
    """
    # Get variables from SCM
    variables = list(get_variables(demonstration.scm))
    
    # Combine all samples
    all_samples = demonstration.observational_samples + demonstration.interventional_samples
    
    # Convert to tensor format
    state_tensor = samples_to_avici_format(
        all_samples, 
        variables, 
        demonstration.target_variable
    )
    
    # Get target index
    target_idx = variables.index(demonstration.target_variable)
    
    # Compute marginal parent probabilities
    posterior_dist = demonstration.parent_posterior.get('posterior_distribution', {})
    marginal_probs = compute_marginal_parent_probabilities(
        posterior_dist,
        variables,
        demonstration.target_variable
    )
    
    # Create SCM ID for tracking
    scm_id = f"{demonstration.graph_type}_n{demonstration.n_nodes}_t{demonstration.target_variable}"
    
    return SurrogateTrainingData(
        state_tensor=jnp.array(state_tensor),
        target_idx=target_idx,
        marginal_parent_probs=marginal_probs,
        variables=variables,
        scm_id=scm_id
    )


def extract_policy_training_data(demonstration: Any) -> Optional[PolicyTrainingData]:
    """
    Extract and preprocess data for policy model training.
    
    Args:
        demonstration: ExpertDemonstration object
        
    Returns:
        PolicyTrainingData if trajectory data is available, None otherwise
    """
    # Check if we have trajectory data
    trajectory = demonstration.parent_posterior.get('trajectory', {})
    if not isinstance(trajectory, dict):
        return None
    
    # Extract intervention sequence and outcomes
    interventions = trajectory.get('intervention_sequence', [])
    values = trajectory.get('intervention_values', [])
    outcomes = trajectory.get('target_outcomes', [])
    
    if not interventions or not values or not outcomes:
        return None
    
    # Need at least 2 outcomes to compute rewards
    if len(outcomes) < 2:
        return None
    
    # Get variables
    variables = list(get_variables(demonstration.scm))
    target_idx = variables.index(demonstration.target_variable)
    
    # Convert to tensor format for states
    # For now, use the final state as representative
    # In production, we'd track buffer snapshots at each intervention
    all_samples = demonstration.observational_samples + demonstration.interventional_samples
    final_state = samples_to_avici_format(
        all_samples,
        variables,
        demonstration.target_variable
    )
    
    # Create states by duplicating final state (simplified for now)
    # In production, we'd have actual buffer snapshots
    T = len(interventions) + 1
    states = jnp.tile(final_state[-1:], (T, 1, 1))
    
    # Convert intervention variables to indices
    intervention_var_indices = []
    for int_var in interventions:
        # Handle both string and tuple formats
        if isinstance(int_var, tuple) and len(int_var) > 0:
            var_name = int_var[0]
        else:
            var_name = int_var
            
        if var_name in variables:
            intervention_var_indices.append(variables.index(var_name))
        else:
            # Skip invalid interventions
            continue
    
    if not intervention_var_indices:
        return None
    
    # Ensure we have matching lengths
    min_len = min(len(intervention_var_indices), len(values), len(outcomes) - 1)
    intervention_var_indices = intervention_var_indices[:min_len]
    values = values[:min_len]
    
    # Compute rewards as negative change (for minimization)
    rewards = []
    for i in range(min_len):
        reward = -(outcomes[i+1] - outcomes[i])
        rewards.append(float(reward))
    
    # Create SCM ID
    scm_id = f"{demonstration.graph_type}_n{demonstration.n_nodes}_t{demonstration.target_variable}"
    
    return PolicyTrainingData(
        states=states[:min_len+1],  # Include initial state
        intervention_vars=jnp.array(intervention_var_indices),
        intervention_values=jnp.array(values),
        rewards=jnp.array(rewards),
        target_idx=target_idx,
        variables=variables,
        scm_id=scm_id
    )


def preprocess_demonstration_batch(
    demonstrations: List[Any]
) -> Dict[str, List[Any]]:
    """
    Preprocess a batch of demonstrations for training.
    
    Args:
        demonstrations: List of ExpertDemonstration objects or DemonstrationBatch
        
    Returns:
        Dictionary with 'surrogate_data' and 'policy_data' lists
    """
    surrogate_data = []
    policy_data = []
    
    # Flatten if we have DemonstrationBatch objects
    flat_demos = []
    for item in demonstrations:
        if hasattr(item, 'demonstrations'):
            # This is a DemonstrationBatch
            flat_demos.extend(item.demonstrations)
        else:
            # This is an individual demonstration
            flat_demos.append(item)
    
    for demo in flat_demos:
        # Extract surrogate training data
        try:
            surrogate_data.append(extract_surrogate_training_data(demo))
        except Exception as e:
            logger.warning(f"Failed to extract surrogate data: {e}")
        
        # Extract policy training data if available
        try:
            policy_datum = extract_policy_training_data(demo)
            if policy_datum is not None:
                policy_data.append(policy_datum)
        except Exception as e:
            logger.warning(f"Failed to extract policy data: {e}")
    
    return {
        'surrogate_data': surrogate_data,
        'policy_data': policy_data
    }


def create_surrogate_training_batch(
    data_list: List[SurrogateTrainingData]
) -> Dict[str, Any]:
    """
    Create a batched format for surrogate training.
    
    Args:
        data_list: List of SurrogateTrainingData objects
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # Stack all state tensors
    # Note: This assumes all SCMs have the same number of variables
    # In production, we'd handle variable-size batching
    state_tensors = [d.state_tensor for d in data_list]
    
    # Find max dimensions
    max_samples = max(t.shape[0] for t in state_tensors)
    n_vars = state_tensors[0].shape[1]
    
    # Pad tensors to same size
    padded_tensors = []
    masks = []
    
    for tensor in state_tensors:
        n_samples = tensor.shape[0]
        if n_samples < max_samples:
            # Pad with zeros
            padding = jnp.zeros((max_samples - n_samples, n_vars, 3))
            padded = jnp.concatenate([tensor, padding], axis=0)
            mask = jnp.concatenate([
                jnp.ones(n_samples),
                jnp.zeros(max_samples - n_samples)
            ])
        else:
            padded = tensor
            mask = jnp.ones(max_samples)
        
        padded_tensors.append(padded)
        masks.append(mask)
    
    # Stack into batch
    batch_states = jnp.stack(padded_tensors)  # [B, N, d, 3]
    batch_masks = jnp.stack(masks)  # [B, N]
    
    # Collect other data
    target_indices = jnp.array([d.target_idx for d in data_list])
    
    # Convert marginal probabilities to matrix format
    # Shape: [B, d] where each row is probabilities for one example
    marginal_prob_matrix = []
    for datum in data_list:
        probs = onp.zeros(n_vars)
        for var, prob in datum.marginal_parent_probs.items():
            var_idx = datum.variables.index(var)
            probs[var_idx] = prob
        marginal_prob_matrix.append(probs)
    
    marginal_prob_matrix = jnp.array(marginal_prob_matrix)
    
    return {
        'states': batch_states,
        'masks': batch_masks,
        'target_indices': target_indices,
        'marginal_parent_probs': marginal_prob_matrix,
        'scm_ids': [d.scm_id for d in data_list],
        'variables': data_list[0].variables  # Assuming same variables
    }


def create_policy_training_batch(
    data_list: List[PolicyTrainingData]
) -> Dict[str, Any]:
    """
    Create a batched format for policy training.
    
    Args:
        data_list: List of PolicyTrainingData objects
        
    Returns:
        Dictionary with batched tensors and metadata
    """
    # For policy training, we typically train on individual trajectories
    # But we can batch them for efficiency
    
    # Find max trajectory length
    max_length = max(d.states.shape[0] for d in data_list)
    n_vars = data_list[0].states.shape[1]
    
    # Check that all have same number of variables
    if not all(d.states.shape[1] == n_vars for d in data_list):
        # Group by number of variables and return first group
        from collections import defaultdict
        by_nvars = defaultdict(list)
        for d in data_list:
            by_nvars[d.states.shape[1]].append(d)
        # Use the largest group
        largest_group = max(by_nvars.values(), key=len)
        data_list = largest_group
        n_vars = data_list[0].states.shape[1]
        max_length = max(d.states.shape[0] for d in data_list)
    
    # Pad trajectories
    padded_states = []
    padded_actions = []
    padded_values = []
    padded_rewards = []
    trajectory_masks = []
    
    for datum in data_list:
        T = datum.states.shape[0]
        T_actions = T - 1  # Actions are one less than states
        
        # Pad states
        if T < max_length:
            state_padding = jnp.zeros((max_length - T, n_vars, 3))
            padded_state = jnp.concatenate([datum.states, state_padding], axis=0)
        else:
            padded_state = datum.states
        
        # Pad actions and rewards
        if T_actions < max_length - 1:
            action_padding = jnp.zeros(max_length - 1 - T_actions)
            padded_action = jnp.concatenate([datum.intervention_vars, action_padding])
            padded_value = jnp.concatenate([datum.intervention_values, action_padding])
            padded_reward = jnp.concatenate([datum.rewards, action_padding])
        else:
            padded_action = datum.intervention_vars
            padded_value = datum.intervention_values
            padded_reward = datum.rewards
        
        # Create mask
        mask = jnp.zeros(max_length - 1)
        mask = mask.at[:T_actions].set(1.0)
        
        padded_states.append(padded_state)
        padded_actions.append(padded_action)
        padded_values.append(padded_value)
        padded_rewards.append(padded_reward)
        trajectory_masks.append(mask)
    
    return {
        'states': jnp.stack(padded_states),  # [B, T, d, 3]
        'intervention_vars': jnp.stack(padded_actions),  # [B, T-1]
        'intervention_values': jnp.stack(padded_values),  # [B, T-1]
        'rewards': jnp.stack(padded_rewards),  # [B, T-1]
        'masks': jnp.stack(trajectory_masks),  # [B, T-1]
        'target_indices': jnp.array([d.target_idx for d in data_list]),
        'scm_ids': [d.scm_id for d in data_list],
        'variables': data_list[0].variables  # Assuming same variables
    }


def load_demonstrations_from_path(path: Union[str, Path], max_files: Optional[int] = None) -> List[Any]:
    """
    Load expert demonstrations from file or directory.
    
    Args:
        path: Path to demonstrations file or directory
        max_files: Maximum number of files to load (for testing)
        
    Returns:
        List of demonstration objects
    """
    path = Path(path)
    
    if path.is_file():
        # Load single file
        with open(path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    elif path.is_dir():
        # Load all .pkl files from directory
        all_demos = []
        files = sorted(path.glob("*.pkl"))
        
        # Limit files if requested
        if max_files is not None:
            logger.info(f"*** LIMITING DEMONSTRATIONS: Found {len(files)} files, loading only first {max_files} ***")
            files = files[:max_files]
        
        for file_path in files:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, list):
                    all_demos.extend(data)
                else:
                    all_demos.append(data)
        
        logger.info(f"Total demonstrations loaded: {len(all_demos)}")
        return all_demos
    else:
        raise ValueError(f"Path does not exist: {path}")


def load_and_preprocess_demonstrations(path: Union[str, Path]) -> Dict[str, List]:
    """
    Single entry point for loading and preprocessing demonstrations.
    
    This function loads expert demonstrations from the standard format
    and preprocesses them into both surrogate and policy training data.
    
    Args:
        path: Path to expert demonstrations directory or file
        
    Returns:
        Dictionary with 'surrogate_data' and 'policy_data' lists
    """
    logger.info(f"Loading demonstrations from {path}")
    
    # Load raw demonstrations
    raw_demos = load_demonstrations_from_path(path)
    logger.info(f"Loaded {len(raw_demos)} raw demonstrations")
    
    # Preprocess all demonstrations
    all_surrogate_data = []
    all_policy_data = []
    
    for demo in raw_demos:
        preprocessed = preprocess_demonstration_batch([demo])
        
        if preprocessed['surrogate_data']:
            all_surrogate_data.extend(preprocessed['surrogate_data'])
            
        if preprocessed['policy_data']:
            all_policy_data.extend(preprocessed['policy_data'])
    
    logger.info(f"Preprocessed into {len(all_surrogate_data)} surrogate examples "
                f"and {len(all_policy_data)} policy trajectories")
    
    return {
        'surrogate_data': all_surrogate_data,
        'policy_data': all_policy_data
    }