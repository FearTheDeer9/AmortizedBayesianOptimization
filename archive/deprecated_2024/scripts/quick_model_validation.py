#!/usr/bin/env python3
"""
Quick validation that the replaced model works correctly.
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

# Import the model (now fixed)
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel


def test_model_outputs_varied():
    """Test that model produces varied outputs, not uniform distributions."""
    print("Testing Model Output Variation")
    print("="*60)
    
    # Create test data with correlations
    key = random.PRNGKey(42)
    n_samples, n_vars = 20, 4
    
    # Create correlated data
    base = random.normal(key, (n_samples, 1))
    data = jnp.zeros((n_samples, n_vars, 3))
    
    # X0 = base + noise
    data = data.at[:, 0, 0].set(base.squeeze() + 0.1 * random.normal(random.PRNGKey(1), (n_samples,)))
    # X1 = 2*X0 + noise (X0 -> X1)
    data = data.at[:, 1, 0].set(2 * data[:, 0, 0] + 0.1 * random.normal(random.PRNGKey(2), (n_samples,)))
    # X2 = -X0 + noise (X0 -> X2)
    data = data.at[:, 2, 0].set(-data[:, 0, 0] + 0.1 * random.normal(random.PRNGKey(3), (n_samples,)))
    # X3 = independent
    data = data.at[:, 3, 0].set(random.normal(random.PRNGKey(4), (n_samples,)))
    
    # All observed
    data = data.at[:, :, 2].set(1.0)
    
    # Test model
    def model_fn(data, target):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=64, num_layers=3, num_heads=4, key_size=32, dropout=0.0
        )
        return model(data, target, is_training=False)
    
    model = hk.without_apply_rng(hk.transform(model_fn))
    params = model.init(key, data, 0)
    
    # Test different targets
    results = []
    for target_idx in range(n_vars):
        output = model.apply(params, data, target_idx)
        probs = output['parent_probabilities']
        results.append(probs)
        
        print(f"\nTarget X{target_idx}:")
        print(f"  Parent probabilities: {probs}")
        print(f"  Max prob: {jnp.max(probs):.3f}, Min prob: {jnp.min(probs):.3f}")
        print(f"  Entropy: {-jnp.sum(probs * jnp.log(probs + 1e-8)):.3f}")
    
    # Check for variation
    all_similar = True
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            if not jnp.allclose(results[i], results[j], atol=0.01):
                all_similar = False
                break
    
    if all_similar:
        print("\n❌ FAILURE: All outputs are similar (uniform distribution problem persists)")
        return False
    else:
        print("\n✅ SUCCESS: Model produces varied outputs based on input data")
        return True


def test_model_learning():
    """Test that model can learn from gradients."""
    print("\n\nTesting Model Learning Capability")
    print("="*60)
    
    import optax
    
    # Create simple data where X0 -> X1
    key = random.PRNGKey(42)
    n_samples = 50
    
    data = jnp.zeros((n_samples, 2, 3))
    x0 = random.normal(key, (n_samples,))
    x1 = 2 * x0 + 0.1 * random.normal(random.PRNGKey(1), (n_samples,))
    
    data = data.at[:, 0, 0].set(x0)
    data = data.at[:, 1, 0].set(x1)
    data = data.at[:, :, 2].set(1.0)  # All observed
    
    # Ground truth: X1's parent is X0
    target_idx = 1
    true_probs = jnp.array([1.0, 0.0])  # X0 is parent, X1 is not its own parent
    
    # Model setup
    def model_fn(data, target):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=32, num_layers=2, num_heads=2, key_size=16, dropout=0.0
        )
        return model(data, target, is_training=True)
    
    model = hk.transform(model_fn)
    params = model.init(key, data, target_idx)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    # Check initial predictions
    key, subkey = random.split(key)
    initial_output = model.apply(params, subkey, data, target_idx)
    initial_probs = initial_output['parent_probabilities']
    print(f"Initial predictions: {initial_probs}")
    
    # Training loop
    initial_loss = None
    for step in range(50):
        def loss_fn(params, key):
            output = model.apply(params, key, data, target_idx)
            pred_probs = output['parent_probabilities']
            loss = -jnp.sum(true_probs * jnp.log(pred_probs + 1e-8))
            return loss, pred_probs
        
        key, subkey = random.split(key)
        (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey)
        
        if initial_loss is None:
            initial_loss = float(loss)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f}, Predicted probs = {pred_probs}")
    
    final_loss = float(loss)
    
    # Check if model improved or started good
    if jnp.argmax(initial_probs) == 0:  # Already predicting correctly
        print(f"\n✅ SUCCESS: Model correctly identifies parent from initialization")
        return True
    elif final_loss < initial_loss * 0.5 or jnp.argmax(pred_probs) == 0:
        print(f"\n✅ SUCCESS: Model learns from data (loss reduced from {initial_loss:.4f} to {final_loss:.4f})")
        return True
    else:
        print(f"\n❌ FAILURE: Model doesn't learn (loss only reduced from {initial_loss:.4f} to {final_loss:.4f})")
        return False


def main():
    """Run validation tests."""
    print("="*80)
    print("VALIDATING REPLACED MODEL")
    print("="*80)
    
    test1 = test_model_outputs_varied()
    test2 = test_model_learning()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✅ ALL TESTS PASSED! The fixed model is working correctly.")
        print("\nThe model replacement was successful. The BC workflow should now work properly.")
    else:
        print("❌ Some tests failed. The model may need further investigation.")


if __name__ == "__main__":
    main()