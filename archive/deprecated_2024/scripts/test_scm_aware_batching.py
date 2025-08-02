#!/usr/bin/env python3
"""
Test SCM-aware batching for acquisition training.

This script tests that the new SCM-aware batching function correctly groups
trajectory steps by demonstration ID, preventing variable mismatch errors.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass

import jax.numpy as jnp
import pyrsistent as pyr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
from causal_bayes_opt.training.bc_data_pipeline import create_acquisition_scm_aware_batches
from causal_bayes_opt.training.trajectory_processor import TrajectoryStep
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.avici_integration.parent_set.posterior import ParentSetPosterior
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.data_structures.sample import create_sample


def create_mock_acquisition_state(n_vars: int, demo_id: str, step: int) -> AcquisitionState:
    """Create a mock AcquisitionState with specified number of variables."""
    # Create variable list
    variables = [f"X{i}" for i in range(n_vars)]
    
    # Create mock posterior
    posterior = ParentSetPosterior(
        target_variable=variables[0],
        parent_set_probs=pyr.pmap({frozenset(): 1.0}),
        uncertainty=0.0,
        top_k_sets=[(frozenset(), 1.0)]
    )
    
    # Create buffer with some samples
    buffer = ExperienceBuffer()
    # Add a few observational samples
    for i in range(3):
        values = {var: float(i + j) for j, var in enumerate(variables)}
        sample = create_sample(
            values=values,
            intervention_type=None,
            intervention_targets=None
        )
        buffer.add_observation(sample)
    
    # Create metadata with SCM info
    metadata = pyr.m(
        n_nodes=n_vars,
        graph_type='test',
        demo_step=step,
        scm_info={
            'variables': variables,
            'n_nodes': n_vars,
            'graph_type': 'test'
        }
    )
    
    return AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=0.0,
        current_target=variables[0],
        step=step,
        metadata=metadata
    )


def create_mock_trajectory_steps() -> List[TrajectoryStep]:
    """Create mock trajectory steps with different SCM sizes."""
    trajectory_steps = []
    
    # Demo 1: 3 variables, 5 steps
    for step in range(5):
        state = create_mock_acquisition_state(3, "demo_0001", step)
        action = {
            'intervention_variables': frozenset([f"X{step % 3}"]),
            'intervention_values': (1.0,)
        }
        trajectory_steps.append(TrajectoryStep(
            state=state,
            action=action,
            step_number=step,
            demonstration_id="demo_0001"
        ))
    
    # Demo 2: 7 variables, 3 steps  
    for step in range(3):
        state = create_mock_acquisition_state(7, "demo_0002", step)
        action = {
            'intervention_variables': frozenset([f"X{step % 7}"]),
            'intervention_values': (2.0,)
        }
        trajectory_steps.append(TrajectoryStep(
            state=state,
            action=action,
            step_number=step,
            demonstration_id="demo_0002"
        ))
    
    # Demo 3: 3 variables, 4 steps (same size as demo 1 but different demo)
    for step in range(4):
        state = create_mock_acquisition_state(3, "demo_0003", step)
        action = {
            'intervention_variables': frozenset([f"X{(step + 1) % 3}"]),
            'intervention_values': (3.0,)
        }
        trajectory_steps.append(TrajectoryStep(
            state=state,
            action=action,
            step_number=step,
            demonstration_id="demo_0003"
        ))
    
    return trajectory_steps


def test_scm_aware_batching():
    """Test SCM-aware batching function."""
    print("\n=== Testing SCM-Aware Batching for Acquisition Training ===\n")
    
    # Create mock trajectory steps
    trajectory_steps = create_mock_trajectory_steps()
    print(f"Created {len(trajectory_steps)} trajectory steps from 3 demonstrations")
    print(f"- Demo 1: 3 variables, 5 steps")
    print(f"- Demo 2: 7 variables, 3 steps")
    print(f"- Demo 3: 3 variables, 4 steps")
    
    # Test with different batch sizes
    for batch_size in [2, 4, 8]:
        print(f"\n--- Testing with batch_size={batch_size} ---")
        
        # Create SCM-aware batches
        batch_indices = create_acquisition_scm_aware_batches(
            trajectory_steps=trajectory_steps,
            batch_size=batch_size,
            shuffle=False,  # No shuffle for reproducible testing
            random_seed=42
        )
        
        print(f"Created {len(batch_indices)} batches")
        
        # Verify each batch contains steps from the same demonstration
        for i, indices in enumerate(batch_indices):
            batch_steps = [trajectory_steps[idx] for idx in indices]
            
            # Check demonstration consistency
            demo_ids = set(step.demonstration_id for step in batch_steps)
            assert len(demo_ids) == 1, f"Batch {i} contains mixed demonstrations: {demo_ids}"
            
            # Check variable consistency
            n_vars_list = []
            for step in batch_steps:
                if hasattr(step.state.metadata, 'scm_info'):
                    scm_info = step.state.metadata['scm_info']
                    n_vars = len(scm_info['variables'])
                    n_vars_list.append(n_vars)
            
            unique_n_vars = set(n_vars_list)
            assert len(unique_n_vars) == 1, f"Batch {i} contains mixed variable counts: {unique_n_vars}"
            
            demo_id = list(demo_ids)[0]
            n_vars = list(unique_n_vars)[0] if unique_n_vars else "?"
            print(f"  Batch {i}: {len(batch_steps)} steps from {demo_id} with {n_vars} variables")
    
    print("\n✅ All tests passed! SCM-aware batching correctly groups steps by demonstration.")
    
    # Test that variable mapping works correctly within each batch
    print("\n--- Testing Variable Mapping within Batches ---")
    batch_indices = create_acquisition_scm_aware_batches(
        trajectory_steps=trajectory_steps,
        batch_size=4,
        shuffle=False,
        random_seed=42
    )
    
    for i, indices in enumerate(batch_indices):
        batch_steps = [trajectory_steps[idx] for idx in indices]
        
        # Extract variable list from first state (as done in trainer)
        first_state = batch_steps[0].state
        variables = []
        if hasattr(first_state.metadata, 'scm_info'):
            scm_info = first_state.metadata['scm_info']
            variables = list(scm_info['variables'])
        
        var_to_idx = {var: idx for idx, var in enumerate(variables)}
        
        # Verify all expert actions can be mapped
        for j, step in enumerate(batch_steps):
            action = step.action
            intervention_vars = action.get('intervention_variables', frozenset())
            
            if intervention_vars:
                var_name = next(iter(intervention_vars))
                assert var_name in var_to_idx, f"Variable {var_name} not in mapping {var_to_idx}"
                var_idx = var_to_idx[var_name]
                print(f"  Batch {i}, Step {j}: {var_name} -> index {var_idx} ✓")
    
    print("\n✅ Variable mapping works correctly within all batches!")


if __name__ == "__main__":
    test_scm_aware_batching()