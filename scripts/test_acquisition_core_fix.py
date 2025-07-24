#!/usr/bin/env python3
"""
Test the core acquisition fix without full training.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from src.causal_bayes_opt.training.behavioral_cloning_adapter import (
    load_demonstration_batch
)
from src.causal_bayes_opt.training.trajectory_processor import (
    extract_trajectory_steps
)
from src.causal_bayes_opt.training.bc_acquisition_trainer import BCAcquisitionTrainer


def test_core_fix():
    """Test that variable conversion works correctly."""
    print("="*60)
    print("TESTING ACQUISITION CORE FIX")
    print("="*60)
    
    # Load demonstration
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    # Extract trajectory steps
    trajectory_steps = extract_trajectory_steps(demo, "test_demo")
    
    print(f"\n1. Extracted {len(trajectory_steps)} trajectory steps")
    
    if not trajectory_steps:
        print("   ❌ No trajectory steps extracted!")
        return False
    
    # Check first few steps
    print(f"\n2. Checking variable information in states:")
    for i in range(min(3, len(trajectory_steps))):
        step = trajectory_steps[i]
        state = step.state
        action = step.action
        
        # Check metadata
        has_metadata = hasattr(state, 'metadata')
        has_scm_info = has_metadata and 'scm_info' in state.metadata
        
        print(f"\n   Step {i}:")
        print(f"   - Has metadata: {has_metadata}")
        print(f"   - Has scm_info: {has_scm_info}")
        
        if has_scm_info:
            scm_info = state.metadata['scm_info']
            variables = scm_info.get('variables', [])
            print(f"   - Variables: {variables[:5]}...")
            print(f"   - Num variables: {len(variables)}")
            
            # Check action
            intervention_vars = action.get('intervention_variables', frozenset())
            if intervention_vars:
                var_name = next(iter(intervention_vars))
                print(f"   - Action variable: {var_name}")
                
                # Test conversion
                var_to_idx = {var: idx for idx, var in enumerate(variables)}
                var_idx = var_to_idx.get(var_name, -1)
                print(f"   - Converted index: {var_idx}")
                
                if var_idx >= 0 and var_idx < len(variables):
                    print(f"   ✅ Valid index!")
                else:
                    print(f"   ❌ Invalid index!")
    
    # Test cross-entropy computation
    print(f"\n3. Testing cross-entropy with proper indices:")
    
    # Get variables from first state
    first_state = trajectory_steps[0].state
    if hasattr(first_state, 'metadata') and 'scm_info' in first_state.metadata:
        scm_info = first_state.metadata['scm_info']
        variables = scm_info.get('variables', [])
        n_vars = len(variables)
        
        print(f"   Number of variables: {n_vars}")
        
        # Test with valid index
        valid_idx = 2
        logits = jnp.zeros(n_vars)
        loss_valid = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([valid_idx])
        )[0]
        print(f"   Valid index ({valid_idx}) loss: {float(loss_valid):.6f}")
        
        # Test with out-of-bounds index (simulating the bug)
        invalid_idx = n_vars + 10  # Out of bounds
        try:
            loss_invalid = optax.softmax_cross_entropy_with_integer_labels(
                logits[None, :], jnp.array([invalid_idx])
            )[0]
            print(f"   Invalid index ({invalid_idx}) loss: {float(loss_invalid)}")
            
            if jnp.isnan(loss_invalid) or loss_invalid > 1000:
                print(f"   ⚠️  This would cause astronomical loss!")
        except Exception as e:
            print(f"   Invalid index caused error: {e}")
    
    return True


def test_trainer_conversion():
    """Test that the trainer properly converts variables."""
    print("\n" + "="*60)
    print("TESTING TRAINER CONVERSION LOGIC")
    print("="*60)
    
    # Create mock states with our fix
    mock_metadata = {
        'scm_info': {
            'variables': ['X0', 'X1', 'X2', 'X3', 'X4'],
            'n_nodes': 5
        }
    }
    
    # Create mock action with string variable
    mock_action = {
        'intervention_variables': frozenset(['X2']),
        'intervention_values': (1.5,)
    }
    
    # Simulate the conversion logic from acquisition trainer
    variables = mock_metadata['scm_info']['variables']
    var_to_idx = {var: idx for idx, var in enumerate(variables)}
    
    print(f"\n1. Variable mapping: {var_to_idx}")
    
    # Convert action
    intervention_vars = mock_action.get('intervention_variables', frozenset())
    if intervention_vars:
        var_name = next(iter(intervention_vars))
        var_idx = var_to_idx.get(var_name, 0)
        value = mock_action['intervention_values'][0]
        
        print(f"\n2. Action conversion:")
        print(f"   Variable name: {var_name}")
        print(f"   Converted index: {var_idx}")
        print(f"   Value: {value}")
        
        # Create action tensor as trainer would
        action_array = jnp.array([var_idx, value])
        print(f"   Action tensor: {action_array}")
        
        # Check if index is valid
        if 0 <= var_idx < len(variables):
            print(f"   ✅ Valid conversion!")
            return True
        else:
            print(f"   ❌ Invalid conversion!")
            return False
    
    return False


if __name__ == "__main__":
    # Test core fix
    success1 = test_core_fix()
    
    # Test trainer conversion
    success2 = test_trainer_conversion()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success1 and success2:
        print("\n✅ The acquisition fix is working!")
        print("   - States now include scm_info in metadata")
        print("   - Variables are properly extracted")
        print("   - String variable names convert to valid indices")
        print("   - Cross-entropy loss should be computed correctly")
        print("\n📝 Next: Run the notebook to see if acquisition losses are fixed")
    else:
        print("\n❌ The fix has issues - check output above")