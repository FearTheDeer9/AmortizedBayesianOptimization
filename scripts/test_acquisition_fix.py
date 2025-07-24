#!/usr/bin/env python3
"""
Test the fix for acquisition model astronomical losses.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    load_demonstration_batch,
    extract_expert_action,
    extract_intervention_sequence
)
from src.causal_bayes_opt.training.trajectory_processor import (
    extract_trajectory_steps,
    TrajectoryStep
)
from src.causal_bayes_opt.training.bc_acquisition_trainer import (
    BCAcquisitionTrainer,
    BCPolicyConfig
)
from src.causal_bayes_opt.acquisition.policy import PolicyConfig


def test_acquisition_conversion():
    """Test that expert actions are properly converted."""
    print("="*60)
    print("TESTING ACQUISITION ACTION CONVERSION")
    print("="*60)
    
    # Load demonstration
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    print(f"\n1. Demonstration info:")
    # Get variables from SCM - handle both dict and PMap formats
    if hasattr(demo.scm, 'get'):
        variables = sorted(list(demo.scm.get('variables', [])))
    else:
        variables = sorted(list(demo.scm))
    print(f"   Variables: {variables}")
    print(f"   Target: {demo.target_variable}")
    
    # Extract intervention sequence
    intervention_sequence = extract_intervention_sequence(demo)
    print(f"\n2. Intervention sequence (first 3):")
    for i, (vars, vals) in enumerate(intervention_sequence[:3]):
        print(f"   Step {i}: vars={vars}, vals={vals}")
    
    # Extract trajectory steps
    trajectory_steps = extract_trajectory_steps(demo, "test_demo")
    print(f"\n3. Trajectory steps extracted: {len(trajectory_steps)}")
    
    if trajectory_steps:
        # Check first step
        first_step = trajectory_steps[0]
        print(f"\n4. First trajectory step:")
        print(f"   Action: {first_step.action}")
        print(f"   State has variables: {hasattr(first_step.state, 'scm_info')}")
        
        # Check action format
        action_vars = first_step.action.get('intervention_variables', frozenset())
        if action_vars:
            var_name = next(iter(action_vars))
            print(f"   First intervention variable: {var_name} (type: {type(var_name)})")
            
            # This is the issue - var_name is a string like 'X0'
            # But the model expects integer indices
            if isinstance(var_name, str):
                print("   ⚠️  Variable is string - needs conversion to index!")
            
    return trajectory_steps


def test_cross_entropy_calculation():
    """Test cross-entropy loss calculation."""
    print("\n" + "="*60)
    print("TESTING CROSS-ENTROPY LOSS")
    print("="*60)
    
    # Test with correct prediction
    n_vars = 5
    correct_idx = 2
    
    # Perfect prediction
    logits_correct = jnp.zeros(n_vars)
    logits_correct = logits_correct.at[correct_idx].set(10.0)  # High score for correct var
    
    loss_correct = optax.softmax_cross_entropy_with_integer_labels(
        logits_correct[None, :], jnp.array([correct_idx])
    )[0]
    
    print(f"\n1. Perfect prediction:")
    print(f"   Logits: {logits_correct}")
    print(f"   Target: {correct_idx}")
    print(f"   Loss: {float(loss_correct):.6f}")
    
    # Uniform prediction
    logits_uniform = jnp.zeros(n_vars)
    loss_uniform = optax.softmax_cross_entropy_with_integer_labels(
        logits_uniform[None, :], jnp.array([correct_idx])
    )[0]
    
    print(f"\n2. Uniform prediction:")
    print(f"   Logits: {logits_uniform}")
    print(f"   Target: {correct_idx}")
    print(f"   Loss: {float(loss_uniform):.6f}")
    
    # Wrong index (out of bounds) - THIS CAUSES ASTRONOMICAL LOSS
    wrong_idx = 999  # Way out of bounds
    try:
        # This will likely produce nan or very large values
        loss_wrong = optax.softmax_cross_entropy_with_integer_labels(
            logits_uniform[None, :], jnp.array([wrong_idx])
        )[0]
        print(f"\n3. Out-of-bounds index:")
        print(f"   Logits shape: {logits_uniform.shape}")
        print(f"   Target: {wrong_idx}")
        print(f"   Loss: {float(loss_wrong)}")
        
        if jnp.isnan(loss_wrong) or loss_wrong > 1000:
            print("   ⚠️  ASTRONOMICAL LOSS DETECTED!")
    except Exception as e:
        print(f"\n3. Out-of-bounds index caused error: {e}")


def validate_acquisition_fix():
    """Validate that variable name to index conversion works."""
    print("\n" + "="*60)
    print("VALIDATING ACQUISITION FIX")
    print("="*60)
    
    # Create a simple test case
    variables = ['X0', 'X1', 'X2', 'X3', 'X4']
    var_to_idx = {var: idx for idx, var in enumerate(variables)}
    
    print(f"\n1. Variable mapping:")
    print(f"   {var_to_idx}")
    
    # Test conversion
    test_actions = [
        {'intervention_variables': frozenset(['X2']), 'intervention_values': (1.5,)},
        {'intervention_variables': frozenset(['X0']), 'intervention_values': (-0.5,)},
        {'intervention_variables': frozenset(['X4']), 'intervention_values': (2.0,)},
    ]
    
    print(f"\n2. Converting actions:")
    for i, action in enumerate(test_actions):
        vars = action.get('intervention_variables', frozenset())
        if vars:
            var_name = next(iter(vars))
            var_idx = var_to_idx.get(var_name, -1)
            print(f"   Action {i}: {var_name} -> index {var_idx}")
            
            if var_idx < 0 or var_idx >= len(variables):
                print(f"      ⚠️  Invalid index! Will cause astronomical loss!")
    
    print("\n3. Fix: Ensure variable names are properly mapped to indices")
    print("   - Create var_to_idx mapping from state.scm_info['variables']")
    print("   - Convert string variable names to integer indices")
    print("   - Handle missing variables gracefully (default to 0)")


if __name__ == "__main__":
    import optax  # Import here to use in test
    
    # Test action conversion
    trajectory_steps = test_acquisition_conversion()
    
    # Test cross-entropy calculation
    test_cross_entropy_calculation()
    
    # Validate fix approach
    validate_acquisition_fix()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nThe acquisition model has the same issue as the surrogate model:")
    print("- Expert actions use string variable names ('X0', 'X1', etc.)")
    print("- Cross-entropy loss expects integer indices (0, 1, 2, etc.)")
    print("- When indices are out of bounds, loss becomes astronomical")
    print("\nThe fix is to properly convert variable names to indices in")
    print("the acquisition trainer's training step.")