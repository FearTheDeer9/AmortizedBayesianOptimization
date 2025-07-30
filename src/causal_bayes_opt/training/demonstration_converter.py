"""
Converter for transforming expert demonstrations between formats.

This module provides utilities to convert ExpertDemonstration objects
(used for structure learning) into policy demonstrations (used for
acquisition function training).
"""

import logging
from typing import List, Dict, Any, Optional, Set
import jax.numpy as jnp

from ..training.expert_collection.data_structures import (
    ExpertDemonstration, 
    DemonstrationBatch,
    ExpertTrajectoryDemonstration
)
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import (
    get_values, 
    get_intervention_targets,
    is_interventional,
    is_observational
)
from ..interventions.handlers import create_perfect_intervention
from ..training.three_channel_converter import buffer_to_three_channel_tensor

logger = logging.getLogger(__name__)


def convert_expert_demonstration_to_policy_data(
    demo: ExpertDemonstration,
    max_history_size: int = 100,
    standardize: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert an ExpertDemonstration to policy training demonstrations.
    
    This extracts the intervention decisions made by the expert algorithm
    and creates (state, action, outcome) tuples suitable for behavioral cloning.
    
    Args:
        demo: ExpertDemonstration containing observational and interventional samples
        max_history_size: Maximum history for three-channel tensor
        standardize: Whether to standardize the tensor
        
    Returns:
        List of policy demonstrations, each containing:
            - tensor: Three-channel state representation
            - target_idx: Index of target variable
            - expert_var_idx: Index of intervened variable
            - expert_value: Intervention value
            - variables: Variable ordering
            - outcome_value: Target value after intervention
    """
    policy_demonstrations = []
    
    # Build initial buffer from observational samples
    buffer = ExperienceBuffer()
    for sample in demo.observational_samples:
        # Only add truly observational samples
        if is_observational(sample):
            buffer.add_observation(sample)
        else:
            logger.warning("Skipping non-observational sample in observational_samples list")
    
    # Get variable information
    target_var = demo.target_variable
    
    # Group interventional samples by intervention
    intervention_groups = {}
    for sample in demo.interventional_samples:
        if is_interventional(sample):
            targets = get_intervention_targets(sample)
            if targets:
                # Use frozenset of targets as key
                key = tuple(sorted(targets))
                if key not in intervention_groups:
                    intervention_groups[key] = []
                intervention_groups[key].append(sample)
    
    # Process each unique intervention
    for intervention_targets, samples in intervention_groups.items():
        if not samples:
            continue
            
        # Get intervention values from first sample
        first_sample = samples[0]
        values_dict = get_values(first_sample)
        
        # Extract intervention target and value
        # For single-variable interventions (most common case)
        if len(intervention_targets) == 1:
            intervened_var = intervention_targets[0]
            intervention_value = float(values_dict[intervened_var])
            
            # Create state tensor from buffer BEFORE adding intervention
            tensor, var_order = buffer_to_three_channel_tensor(
                buffer, target_var, 
                max_history_size=max_history_size,
                standardize=standardize
            )
            
            # Get variable indices
            try:
                target_idx = var_order.index(target_var)
                expert_var_idx = var_order.index(intervened_var)
            except ValueError:
                logger.warning(f"Variable not found in tensor: target={target_var}, intervened={intervened_var}")
                continue
            
            # Calculate outcome (mean target value after intervention)
            target_values = []
            for sample in samples:
                sample_values = get_values(sample)
                if target_var in sample_values:
                    target_values.append(float(sample_values[target_var]))
            
            if target_values:
                outcome_value = float(jnp.mean(jnp.array(target_values)))
            else:
                outcome_value = 0.0
            
            # Create policy demonstration
            policy_demo = {
                'tensor': tensor.copy(),
                'target_idx': target_idx,
                'expert_var_idx': expert_var_idx,
                'expert_value': intervention_value,
                'variables': var_order.copy(),
                'outcome_value': outcome_value,
                'n_nodes': demo.n_nodes
            }
            
            policy_demonstrations.append(policy_demo)
            
            # Update buffer with intervention samples for next iteration
            intervention = create_perfect_intervention(
                targets=frozenset([intervened_var]),
                values={intervened_var: intervention_value}
            )
            for sample in samples:
                buffer.add_intervention(intervention, sample)
                
        else:
            # Multi-variable interventions - could extend to handle these
            logger.debug(f"Skipping multi-variable intervention: {intervention_targets}")
    
    logger.info(f"Converted {len(demo.interventional_samples)} interventional samples "
               f"into {len(policy_demonstrations)} policy demonstrations")
    
    return policy_demonstrations


def convert_demonstration_batch_to_policy_data(
    batch: DemonstrationBatch,
    max_history_size: int = 100,
    standardize: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert a DemonstrationBatch to policy training demonstrations.
    
    Args:
        batch: DemonstrationBatch containing multiple ExpertDemonstrations
        max_history_size: Maximum history for three-channel tensor
        standardize: Whether to standardize the tensor
        
    Returns:
        List of all policy demonstrations from all experts in the batch
    """
    all_demonstrations = []
    
    for demo in batch.demonstrations:
        if isinstance(demo, ExpertDemonstration):
            policy_demos = convert_expert_demonstration_to_policy_data(
                demo, max_history_size, standardize
            )
            all_demonstrations.extend(policy_demos)
        else:
            logger.warning(f"Skipping non-ExpertDemonstration of type {type(demo)}")
    
    return all_demonstrations


def is_policy_demonstration(data: Any) -> bool:
    """
    Check if data is already in policy demonstration format.
    
    Policy demonstrations should have:
    - tensor: State representation
    - expert_var_idx: Variable to intervene on
    - expert_value: Value to set
    """
    if isinstance(data, dict):
        required_keys = {'tensor', 'expert_var_idx', 'expert_value'}
        return required_keys.issubset(data.keys())
    return False


def is_structure_demonstration(data: Any) -> bool:
    """
    Check if data is a structure learning demonstration.
    """
    from ..training.expert_collection.data_structures import ExpertDemonstration
    return isinstance(data, ExpertDemonstration)