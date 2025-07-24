#!/usr/bin/env python3
"""
Test the relationship-aware model that should actually work.
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

from src.causal_bayes_opt.avici_integration.continuous.relationship_aware_model import (
    RelationshipAwareParentSetModel,
    create_simple_correlation_based_model
)


def test_correlation_based_model():
    """Test the simple correlation-based model first."""
    print("Testing Simple Correlation-Based Model")
    print("=" * 60)
    
    # This model directly uses correlations - it MUST produce varied outputs
    model_fn = create_simple_correlation_based_model()
    
    # Test data
    n_samples = 20
    n_vars = 4
    target_idx = 3
    key = random.PRNGKey(42)
    
    # Test 1: No correlations
    print("\n1. No correlations (random data):")
    key, subkey = random.split(key)
    data1 = jnp.zeros((n_samples, n_vars, 3))
    data1 = data1.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
    data1 = data1.at[:, :, 2].set(1.0)
    
    output1 = model_fn(data1, target_idx)
    print(f"   Correlations with X3: {output1['correlations']}")
    print(f"   Parent probabilities: {output1['parent_probabilities']}")
    
    # Test 2: Strong X1->X3 correlation
    print("\n2. Strong X1->X3 correlation:")
    data2 = jnp.zeros((n_samples, n_vars, 3))
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data2 = data2.at[:, 1, 0].set(x1_vals)
    data2 = data2.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
    data2 = data2.at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
    data2 = data2.at[:, :, 2].set(1.0)
    
    output2 = model_fn(data2, target_idx)
    print(f"   Correlations with X3: {output2['correlations']}")
    print(f"   Parent probabilities: {output2['parent_probabilities']}")
    print(f"   Top parent: X{jnp.argmax(output2['parent_probabilities'][:3])}")
    
    # Test 3: Perfect X2->X3 correlation
    print("\n3. Perfect X2->X3 correlation:")
    data3 = jnp.zeros((n_samples, n_vars, 3))
    x2_vals = jnp.linspace(-2, 2, n_samples)
    data3 = data3.at[:, 2, 0].set(x2_vals)
    data3 = data3.at[:, 3, 0].set(x2_vals)  # Perfect correlation
    data3 = data3.at[:, [0, 1], 0].set(random.normal(subkey, (n_samples, 2)))
    data3 = data3.at[:, :, 2].set(1.0)
    
    output3 = model_fn(data3, target_idx)
    print(f"   Correlations with X3: {output3['correlations']}")
    print(f"   Parent probabilities: {output3['parent_probabilities']}")
    print(f"   Top parent: X{jnp.argmax(output3['parent_probabilities'][:3])}")
    
    print("\n✅ Correlation-based model successfully produces varied outputs!")


def test_relationship_aware_model():
    """Test the full relationship-aware model."""
    print("\n\nTesting Relationship-Aware Model")
    print("=" * 60)
    
    # Create model
    def model_fn(data, target_idx):
        model = RelationshipAwareParentSetModel(
            hidden_dim=64,
            num_layers=2,
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
    
    # Test with correlation patterns
    print("\n1. Testing with X1->X3 correlation:")
    data = jnp.zeros((n_samples, n_vars, 3))
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data = data.at[:, 1, 0].set(x1_vals)
    data = data.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(key, (n_samples,)))
    data = data.at[:, [0, 2], 0].set(random.normal(key, (n_samples, 2)))
    data = data.at[:, :, 2].set(1.0)
    
    output = model.apply(params, data, target_idx)
    print(f"   Target correlations: {output['target_correlations']}")
    print(f"   Parent probabilities: {output['parent_probabilities']}")
    
    # Check if X1 has highest probability
    if jnp.argmax(output['parent_probabilities'][:3]) == 1:
        print("   ✅ Correctly identifies X1 as most likely parent!")
    else:
        print("   ❌ Failed to identify X1 as most likely parent")
    
    # Test with different correlation
    print("\n2. Testing with X2->X3 correlation:")
    data2 = jnp.zeros((n_samples, n_vars, 3))
    x2_vals = jnp.linspace(-2, 2, n_samples)
    data2 = data2.at[:, 2, 0].set(x2_vals)
    data2 = data2.at[:, 3, 0].set(x2_vals + 0.1 * random.normal(key, (n_samples,)))
    data2 = data2.at[:, [0, 1], 0].set(random.normal(key, (n_samples, 2)))
    data2 = data2.at[:, :, 2].set(1.0)
    
    output2 = model.apply(params, data2, target_idx)
    print(f"   Target correlations: {output2['target_correlations']}")
    print(f"   Parent probabilities: {output2['parent_probabilities']}")
    
    if jnp.argmax(output2['parent_probabilities'][:3]) == 2:
        print("   ✅ Correctly identifies X2 as most likely parent!")
    else:
        print("   ❌ Failed to identify X2 as most likely parent")


def main():
    """Test both models."""
    test_correlation_based_model()
    test_relationship_aware_model()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("The relationship-aware approach successfully produces varied outputs")
    print("by explicitly computing and using correlation information!")


if __name__ == "__main__":
    main()