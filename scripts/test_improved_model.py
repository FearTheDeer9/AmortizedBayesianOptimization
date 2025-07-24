#!/usr/bin/env python3
"""
Test the improved continuous parent set prediction model.
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
from src.causal_bayes_opt.avici_integration.continuous.improved_model import ImprovedContinuousParentSetPredictionModel


def test_model_outputs(model_class, model_name):
    """Test a model with different inputs and check for variation."""
    print(f"\nTesting {model_name}")
    print("=" * 60)
    
    # Model parameters
    hidden_dim = 64
    num_layers = 3
    num_heads = 4
    key_size = 32
    
    # Create model function
    def model_fn(data, target_idx):
        model = model_class(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=key_size,
            dropout=0.0
        )
        return model(data, target_idx, is_training=False)
    
    # Transform with Haiku
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Initialize parameters
    key = random.PRNGKey(42)
    dummy_data = jnp.zeros((10, 4, 3))
    params = model.init(key, dummy_data, 0)
    
    # Test data
    n_samples = 20
    n_vars = 4
    target_idx = 3
    
    results = []
    
    # Test 1: Random observational data
    print("\n1. Random observational data:")
    key, subkey = random.split(key)
    data1 = jnp.zeros((n_samples, n_vars, 3))
    data1 = data1.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
    data1 = data1.at[:, :, 2].set(1.0)  # All observed
    
    output1 = model.apply(params, data1, target_idx)
    probs1 = output1['parent_probabilities']
    results.append(("Random", probs1))
    print(f"   Parent probs: {probs1}")
    
    # Test 2: Strong correlation X1->X3
    print("\n2. Strong X1->X3 correlation:")
    key, subkey = random.split(key)
    data2 = jnp.zeros((n_samples, n_vars, 3))
    x1_vals = random.normal(subkey, (n_samples,))
    data2 = data2.at[:, 1, 0].set(x1_vals)
    data2 = data2.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
    data2 = data2.at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
    data2 = data2.at[:, :, 2].set(1.0)
    
    output2 = model.apply(params, data2, target_idx)
    probs2 = output2['parent_probabilities']
    results.append(("X1->X3", probs2))
    print(f"   Parent probs: {probs2}")
    
    # Test 3: Perfect correlation X2->X3
    print("\n3. Perfect X2->X3 correlation:")
    key, subkey = random.split(key)
    data3 = jnp.zeros((n_samples, n_vars, 3))
    x2_vals = random.normal(subkey, (n_samples,))
    data3 = data3.at[:, 2, 0].set(x2_vals)
    data3 = data3.at[:, 3, 0].set(x2_vals)  # Perfect correlation
    data3 = data3.at[:, [0, 1], 0].set(random.normal(subkey, (n_samples, 2)))
    data3 = data3.at[:, :, 2].set(1.0)
    
    output3 = model.apply(params, data3, target_idx)
    probs3 = output3['parent_probabilities']
    results.append(("X2->X3", probs3))
    print(f"   Parent probs: {probs3}")
    
    # Test 4: Interventional data
    print("\n4. Mixed interventional data:")
    key, subkey = random.split(key)
    data4 = jnp.zeros((n_samples, n_vars, 3))
    data4 = data4.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
    # Intervene on X1 for half the samples
    data4 = data4.at[:10, 1, 1].set(1.0)
    data4 = data4.at[:10, 1, 2].set(0.0)
    # Rest observational
    data4 = data4.at[10:, :, 2].set(1.0)
    data4 = data4.at[:10, [0, 2, 3], 2].set(1.0)
    
    output4 = model.apply(params, data4, target_idx)
    probs4 = output4['parent_probabilities']
    results.append(("Mixed", probs4))
    print(f"   Parent probs: {probs4}")
    
    # Analyze variation
    print("\nVariation Analysis:")
    all_probs = [r[1] for r in results]
    
    # Check if all probabilities are identical
    all_same = all(jnp.allclose(all_probs[0], p, atol=1e-6) for p in all_probs[1:])
    
    if all_same:
        print("   ❌ All outputs are identical!")
        print(f"   Fixed output: {all_probs[0]}")
    else:
        print("   ✅ Model produces varied outputs!")
        
        # Show which variables have highest probability for each test
        for name, probs in results:
            var_names = ['X0', 'X1', 'X2', 'X3']
            sorted_idx = jnp.argsort(probs)[::-1]
            top_var = var_names[sorted_idx[0]] if sorted_idx[0] != target_idx else var_names[sorted_idx[1]]
            top_prob = probs[sorted_idx[0]] if sorted_idx[0] != target_idx else probs[sorted_idx[1]]
            print(f"   {name}: Top parent = {top_var} (p={top_prob:.4f})")
    
    return not all_same


def main():
    """Compare original and improved models."""
    print("Comparing Parent Set Prediction Models")
    print("=" * 60)
    
    # Test original model
    original_works = test_model_outputs(
        ContinuousParentSetPredictionModel,
        "Original Model"
    )
    
    # Test improved model
    improved_works = test_model_outputs(
        ImprovedContinuousParentSetPredictionModel,
        "Improved Model"
    )
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Original model produces varied outputs: {original_works}")
    print(f"Improved model produces varied outputs: {improved_works}")
    
    if not original_works and improved_works:
        print("\n✅ The improved model successfully fixes the aggregation issue!")
    elif original_works:
        print("\n🤔 Original model already works - issue might be in the checkpoint or training")
    else:
        print("\n❌ Both models have issues - need further investigation")


if __name__ == "__main__":
    main()