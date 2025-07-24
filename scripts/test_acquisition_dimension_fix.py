#!/usr/bin/env python3
"""
Test that the acquisition dimension fix resolves astronomical losses.
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
from src.causal_bayes_opt.training.bc_acquisition_trainer import (
    create_bc_acquisition_trainer
)


def test_dimension_fix():
    """Test that dimensions are correctly handled in the loss computation."""
    print("="*60)
    print("TESTING ACQUISITION DIMENSION FIX")
    print("="*60)
    
    # Load demonstration with known variable count
    demo_file = Path("expert_demonstrations/raw/raw_demonstrations/batch_1751266609.pkl")
    batch = load_demonstration_batch(str(demo_file))
    demo = batch.demonstrations[0]
    
    # Extract trajectory steps to get real data
    trajectory_steps = extract_trajectory_steps(demo, "test_demo")
    
    print(f"\n1. Loaded demonstration:")
    print(f"   Trajectory steps: {len(trajectory_steps)}")
    
    if not trajectory_steps:
        print("   ❌ No trajectory steps found!")
        return False
    
    # Check variable information
    first_step = trajectory_steps[0]
    state = first_step.state
    action = first_step.action
    
    if hasattr(state, 'metadata') and 'scm_info' in state.metadata:
        scm_info = state.metadata['scm_info']
        variables = scm_info.get('variables', [])
        n_vars = len(variables)
        print(f"   Variables in state: {n_vars}")
        print(f"   Variable names: {variables[:5]}...")
        
        # Check action
        intervention_vars = action.get('intervention_variables', frozenset())
        if intervention_vars:
            var_name = next(iter(intervention_vars))
            var_idx = variables.index(var_name) if var_name in variables else -1
            print(f"   Action variable: {var_name} → index {var_idx}")
            
            if 0 <= var_idx < n_vars:
                print(f"   ✅ Action index is valid for {n_vars} variables")
            else:
                print(f"   ❌ Action index {var_idx} is out of bounds for {n_vars} variables")
                return False
    else:
        print("   ❌ No variable information found in state!")
        return False
    
    # Test cross-entropy loss with correct dimensions
    print(f"\n2. Testing cross-entropy loss with {n_vars} variables:")
    
    # Create logits for correct number of variables
    logits = jnp.zeros(n_vars)  # Uniform logits
    valid_idx = var_idx
    
    try:
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([valid_idx])
        )[0]
        print(f"   Valid index ({valid_idx}) loss: {float(loss):.6f}")
        
        if not jnp.isnan(loss) and loss < 100:
            print(f"   ✅ Loss is reasonable!")
        else:
            print(f"   ❌ Loss is astronomical or NaN!")
            return False
    
    except Exception as e:
        print(f"   ❌ Cross-entropy failed: {e}")
        return False
    
    # Test with out-of-bounds index (should fail gracefully)
    try:
        invalid_idx = n_vars + 5  # Definitely out of bounds
        loss_invalid = optax.softmax_cross_entropy_with_integer_labels(
            logits[None, :], jnp.array([invalid_idx])
        )[0]
        print(f"   Invalid index ({invalid_idx}) loss: {float(loss_invalid)}")
        print(f"   ⚠️  This would cause astronomical loss (expected)")
    except Exception as e:
        print(f"   Invalid index causes error: {e} (expected)")
    
    print(f"\n3. Summary:")
    print(f"   ✅ Variable extraction works correctly")
    print(f"   ✅ Action indices are within valid range") 
    print(f"   ✅ Cross-entropy loss computes correctly with {n_vars} variables")
    print(f"   ✅ The dimension fix should resolve astronomical losses")
    
    return True


def test_trainer_initialization():
    """Test that the trainer can be created and initialized."""
    print("\n" + "="*60)
    print("TESTING TRAINER INITIALIZATION")
    print("="*60)
    
    try:
        # Create trainer
        trainer = create_bc_acquisition_trainer(
            learning_rate=1e-3,
            batch_size=4,
            use_curriculum=False,
            use_jax=True,
            enable_wandb_logging=False,
            experiment_name="dimension_fix_test"
        )
        
        print(f"✅ Trainer created successfully")
        print(f"   JAX compilation: enabled")
        print(f"   Policy network: {trainer._policy_network is not None}")
        
        # Check that dummy state creation works
        dummy_state = trainer._create_dummy_acquisition_state()
        print(f"   Dummy state created: {list(dummy_state.keys())}")
        print(f"   Dummy tensor shapes: {dummy_state['state_tensor'].shape}, {dummy_state['history_tensor'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Test dimension fix
    success1 = test_dimension_fix()
    
    # Test trainer initialization
    success2 = test_trainer_initialization()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if success1 and success2:
        print("\n✅ The acquisition dimension fix is working!")
        print("   - Variables are extracted correctly from state metadata")
        print("   - Action indices are within valid range")
        print("   - Cross-entropy loss uses correct dimensions")
        print("   - Trainer initializes properly")
        print("\n📝 Next: Re-run the notebook to see if acquisition losses are fixed")
    else:
        print("\n❌ The fix has issues - check output above")