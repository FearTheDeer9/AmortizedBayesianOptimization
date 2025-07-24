#!/usr/bin/env python3
"""
Test that the model is truly variable-agnostic.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel


def test_variable_agnostic():
    """Test model works with different numbers of variables."""
    print("Testing Variable-Agnostic Behavior")
    print("="*60)
    
    key = random.PRNGKey(42)
    
    # Create model function
    def model_fn(data, target):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=64, num_layers=2, num_heads=4, key_size=32, dropout=0.0
        )
        return model(data, target, is_training=False)
    
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Test 1: Initialize with 3 variables
    print("\n1. Initialize model with 3 variables:")
    data_3 = jnp.zeros((10, 3, 3))
    data_3 = data_3.at[:, :, 0].set(random.normal(key, (10, 3)))
    data_3 = data_3.at[:, :, 2].set(1.0)
    
    params = model.init(key, data_3, 0)
    output_3 = model.apply(params, data_3, 1)
    print(f"   Output shape: {output_3['parent_probabilities'].shape}")
    print(f"   Probabilities: {output_3['parent_probabilities']}")
    
    # Test 2: Use same params with 5 variables
    print("\n2. Use same parameters with 5 variables:")
    data_5 = jnp.zeros((10, 5, 3))
    key, subkey = random.split(key)
    data_5 = data_5.at[:, :, 0].set(random.normal(subkey, (10, 5)))
    data_5 = data_5.at[:, :, 2].set(1.0)
    
    try:
        output_5 = model.apply(params, data_5, 2)
        print(f"   Output shape: {output_5['parent_probabilities'].shape}")
        print(f"   Probabilities: {output_5['parent_probabilities']}")
        print("   ✅ SUCCESS: Model handles different variable counts!")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("   Model is NOT variable-agnostic")
        return False
    
    # Test 3: Use with 2 variables (fewer than initialization)
    print("\n3. Use same parameters with 2 variables:")
    data_2 = jnp.zeros((10, 2, 3))
    key, subkey = random.split(key)
    data_2 = data_2.at[:, :, 0].set(random.normal(subkey, (10, 2)))
    data_2 = data_2.at[:, :, 2].set(1.0)
    
    try:
        output_2 = model.apply(params, data_2, 0)
        print(f"   Output shape: {output_2['parent_probabilities'].shape}")
        print(f"   Probabilities: {output_2['parent_probabilities']}")
        print("   ✅ SUCCESS: Model handles fewer variables too!")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: Verify outputs are different (not all uniform)
    print("\n4. Checking output variation:")
    all_same_3 = jnp.allclose(output_3['parent_probabilities'][0], output_3['parent_probabilities'][1])
    all_same_5 = jnp.allclose(output_5['parent_probabilities'][0], output_5['parent_probabilities'][1])
    
    if not all_same_3 and not all_same_5:
        print("   ✅ Model produces varied outputs for different variables")
    else:
        print("   ⚠️  Model might still be producing uniform outputs")
    
    return True


def test_shared_parameters():
    """Test that parameters are shared across variables."""
    print("\n\nTesting Shared Parameter Structure")
    print("="*60)
    
    key = random.PRNGKey(42)
    
    def model_fn(data, target):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=32, num_layers=2, num_heads=2, key_size=16, dropout=0.0
        )
        return model(data, target, is_training=False)
    
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Initialize with some data
    data = jnp.zeros((5, 4, 3))
    params = model.init(key, data, 0)
    
    # Check parameter names
    print("\nParameter structure:")
    def print_params(params, prefix=""):
        for k, v in params.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}/")
                print_params(v, prefix + "  ")
            else:
                print(f"{prefix}{k}: shape={v.shape}")
    
    print_params(params)
    
    # Check for variable-specific parameters
    var_specific = []
    def find_var_specific(params, path=""):
        for k, v in params.items():
            if isinstance(v, dict):
                find_var_specific(v, path + k + "/")
            else:
                if "var" in k and any(str(i) in k for i in range(10)):
                    var_specific.append(path + k)
    
    find_var_specific(params)
    
    if var_specific:
        print(f"\n⚠️  Found variable-specific parameters: {var_specific}")
        print("Model is NOT fully variable-agnostic")
        return False
    else:
        print("\n✅ No variable-specific parameters found")
        print("Model uses shared parameters for all variables")
        return True


def main():
    """Run all tests."""
    print("="*80)
    print("VARIABLE-AGNOSTIC MODEL TEST")
    print("="*80)
    
    test1 = test_variable_agnostic()
    test2 = test_shared_parameters()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✅ ALL TESTS PASSED!")
        print("\nThe model is truly variable-agnostic and can handle any number of variables.")
    else:
        print("❌ Some tests failed.")
        print("\nThe model has variable-specific components that limit flexibility.")


if __name__ == "__main__":
    main()