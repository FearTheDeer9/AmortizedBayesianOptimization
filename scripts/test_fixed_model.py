#!/usr/bin/env python3
"""
Test the fixed continuous parent set prediction model.
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
from src.causal_bayes_opt.avici_integration.continuous.fixed_model import FixedContinuousParentSetPredictionModel


def test_model(model_class, model_name):
    """Test a model to see if it produces varied outputs."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print('='*60)
    
    # Create model
    def model_fn(data, target_idx):
        model = model_class(
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            key_size=32,
            dropout=0.0
        )
        return model(data, target_idx, is_training=False)
    
    # Transform with Haiku
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Initialize
    key = random.PRNGKey(42)
    dummy_data = jnp.zeros((10, 4, 3))
    params = model.init(key, dummy_data, 0)
    
    # Test data
    n_samples = 20
    n_vars = 4
    target_idx = 3
    
    # Test 1: Random data
    print("\n1. Random uncorrelated data:")
    key, subkey = random.split(key)
    data1 = jnp.zeros((n_samples, n_vars, 3))
    data1 = data1.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
    data1 = data1.at[:, :, 2].set(1.0)
    
    output1 = model.apply(params, data1, target_idx)
    probs1 = output1['parent_probabilities']
    print(f"   Parent probs: {probs1}")
    print(f"   Attention logits: {output1['attention_logits']}")
    
    # Test 2: Strong X1->X3 correlation
    print("\n2. Strong X1->X3 correlation:")
    data2 = jnp.zeros((n_samples, n_vars, 3))
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data2 = data2.at[:, 1, 0].set(x1_vals)
    data2 = data2.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
    data2 = data2.at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
    data2 = data2.at[:, :, 2].set(1.0)
    
    output2 = model.apply(params, data2, target_idx)
    probs2 = output2['parent_probabilities']
    print(f"   Parent probs: {probs2}")
    print(f"   Attention logits: {output2['attention_logits']}")
    
    # Test 3: Perfect X2->X3 correlation
    print("\n3. Perfect X2->X3 correlation:")
    data3 = jnp.zeros((n_samples, n_vars, 3))
    x2_vals = jnp.linspace(-2, 2, n_samples)
    data3 = data3.at[:, 2, 0].set(x2_vals)
    data3 = data3.at[:, 3, 0].set(x2_vals)  # Perfect correlation
    data3 = data3.at[:, [0, 1], 0].set(random.normal(subkey, (n_samples, 2)))
    data3 = data3.at[:, :, 2].set(1.0)
    
    output3 = model.apply(params, data3, target_idx)
    probs3 = output3['parent_probabilities']
    print(f"   Parent probs: {probs3}")
    print(f"   Attention logits: {output3['attention_logits']}")
    
    # Check variation
    all_probs = [probs1, probs2, probs3]
    all_same = all(jnp.allclose(all_probs[0], p, atol=1e-6) for p in all_probs[1:])
    
    if all_same:
        print(f"\n❌ {model_name} produces UNIFORM outputs!")
    else:
        print(f"\n✅ {model_name} produces VARIED outputs!")
        
        # Show which test identified which parent
        test_names = ["Random", "X1->X3", "X2->X3"]
        for i, (name, probs) in enumerate(zip(test_names, all_probs)):
            top_parent_idx = jnp.argmax(probs[:3])  # Exclude target itself
            print(f"   {name}: Top parent = X{top_parent_idx} (p={probs[top_parent_idx]:.4f})")
    
    return not all_same


def main():
    """Test both original and fixed models."""
    print("="*60)
    print("MODEL COMPARISON: Original vs Fixed")
    print("="*60)
    
    # Test original model
    original_works = test_model(
        ContinuousParentSetPredictionModel,
        "Original Model"
    )
    
    # Test fixed model
    fixed_works = test_model(
        FixedContinuousParentSetPredictionModel,
        "Fixed Model"
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if not original_works and fixed_works:
        print("\n✅ SUCCESS: The fixed model resolves the uniform output issue!")
        print("\nThe key fixes were:")
        print("1. Proper attention mechanism with query-key interaction")
        print("2. Variable-specific encoding without destructive averaging")
        print("3. Preserving correlation information throughout the pipeline")
    elif original_works:
        print("\n🤔 Unexpected: Original model already works?")
    else:
        print("\n❌ Both models have issues - need more investigation")


if __name__ == "__main__":
    main()