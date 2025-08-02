#!/usr/bin/env python3
"""
Test model learning capability after variable-agnostic changes.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
from pathlib import Path
import numpy as np

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel


def test_learning_on_complex_data():
    """Test learning on more complex data where model needs to actually learn."""
    print("Testing Learning on Complex Multi-Parent Data")
    print("="*60)
    
    key = random.PRNGKey(42)
    n_samples = 100
    n_vars = 5
    
    # Create data with complex relationships
    # X0 and X1 are independent
    # X2 = X0 + noise
    # X3 = X1 + noise  
    # X4 = X2 + X3 + noise (has two parents)
    
    data = jnp.zeros((n_samples, n_vars, 3))
    
    # Generate values
    key, k1, k2, k3, k4, k5 = random.split(key, 6)
    x0 = random.normal(k1, (n_samples,))
    x1 = random.normal(k2, (n_samples,))
    x2 = x0 + 0.3 * random.normal(k3, (n_samples,))
    x3 = x1 + 0.3 * random.normal(k4, (n_samples,))
    x4 = x2 + x3 + 0.3 * random.normal(k5, (n_samples,))
    
    values = jnp.stack([x0, x1, x2, x3, x4], axis=1)
    data = data.at[:, :, 0].set(values)
    data = data.at[:, :, 2].set(1.0)  # All observed
    
    # True parent relationships
    true_parents = {
        0: [],      # X0 has no parents
        1: [],      # X1 has no parents
        2: [0],     # X2's parent is X0
        3: [1],     # X3's parent is X1
        4: [2, 3]   # X4's parents are X2 and X3
    }
    
    # Test learning for each variable
    def create_model():
        def model_fn(data, target):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=64, num_layers=3, num_heads=4, key_size=32, dropout=0.0
            )
            return model(data, target, is_training=True)
        return hk.transform(model_fn)
    
    print("\nTesting learning for each target variable:")
    
    all_learned = True
    for target_idx in range(n_vars):
        # Create true parent probabilities
        true_probs = jnp.zeros(n_vars)
        for parent in true_parents[target_idx]:
            true_probs = true_probs.at[parent].set(1.0 / len(true_parents[target_idx]) if true_parents[target_idx] else 0.0)
        
        # Skip if no parents (nothing to learn)
        if not true_parents[target_idx]:
            continue
            
        print(f"\n{'='*40}")
        print(f"Target X{target_idx}, true parents: {true_parents[target_idx]}")
        
        # Initialize model
        model = create_model()
        key, subkey = random.split(key)
        params = model.init(subkey, data, target_idx)
        
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        # Check initial predictions
        key, subkey = random.split(key)
        initial_output = model.apply(params, subkey, data, target_idx)
        initial_probs = initial_output['parent_probabilities']
        
        # Training loop
        losses = []
        for step in range(100):
            def loss_fn(params, key):
                output = model.apply(params, key, data, target_idx)
                pred_probs = output['parent_probabilities']
                
                # Simple MSE loss for testing
                loss = jnp.mean((pred_probs - true_probs) ** 2)
                return loss, pred_probs
            
            key, subkey = random.split(key)
            (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            losses.append(float(loss))
            
            if step % 20 == 0:
                top_pred = jnp.argmax(pred_probs)
                print(f"  Step {step}: Loss = {loss:.4f}, Top prediction = X{top_pred}")
        
        # Check final predictions
        key, subkey = random.split(key) 
        final_output = model.apply(params, subkey, data, target_idx)
        final_probs = final_output['parent_probabilities']
        
        # Analyze results
        predicted_parents = []
        for i in range(n_vars):
            if final_probs[i] > 0.2 and i != target_idx:  # Threshold for considering as parent
                predicted_parents.append(i)
        
        correct = set(predicted_parents) == set(true_parents[target_idx])
        
        print(f"\nFinal predictions:")
        print(f"  Probabilities: {final_probs}")
        print(f"  Predicted parents: {predicted_parents}")
        print(f"  True parents: {true_parents[target_idx]}")
        print(f"  Correct: {'✅' if correct else '❌'}")
        
        # Check if model learned (loss decreased)
        if losses[-1] < losses[0] * 0.5:
            print(f"  Learning: ✅ (loss decreased from {losses[0]:.4f} to {losses[-1]:.4f})")
        else:
            print(f"  Learning: ❌ (loss only decreased from {losses[0]:.4f} to {losses[-1]:.4f})")
            all_learned = False
    
    return all_learned


def test_different_variable_counts():
    """Test that model learns equally well with different variable counts."""
    print("\n\nTesting Learning with Different Variable Counts")
    print("="*60)
    
    results = {}
    
    for n_vars in [3, 5, 7]:
        print(f"\nTesting with {n_vars} variables:")
        
        key = random.PRNGKey(42 + n_vars)
        n_samples = 50
        
        # Create simple chain: X0 -> X1 -> X2 -> ...
        data = jnp.zeros((n_samples, n_vars, 3))
        
        key, subkey = random.split(key)
        values = [random.normal(subkey, (n_samples,))]
        
        for i in range(1, n_vars):
            key, subkey = random.split(key)
            values.append(values[i-1] + 0.3 * random.normal(subkey, (n_samples,)))
        
        data = data.at[:, :, 0].set(jnp.stack(values, axis=1))
        data = data.at[:, :, 2].set(1.0)
        
        # Test learning X1's parent (should be X0)
        target_idx = 1
        true_parent = 0
        
        # Create and train model
        def model_fn(data, target):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=32, num_layers=2, num_heads=2, key_size=16, dropout=0.0
            )
            return model(data, target, is_training=True)
        
        model = hk.transform(model_fn)
        params = model.init(key, data, target_idx)
        
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        
        # Quick training
        initial_loss = None
        for step in range(50):
            def loss_fn(params, key):
                output = model.apply(params, key, data, target_idx)
                pred_probs = output['parent_probabilities']
                
                # Want high probability on true parent
                loss = -jnp.log(pred_probs[true_parent] + 1e-8)
                return loss, pred_probs
            
            key, subkey = random.split(key)
            (loss, pred_probs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, subkey)
            
            if initial_loss is None:
                initial_loss = float(loss)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        
        final_loss = float(loss)
        top_pred = jnp.argmax(pred_probs)
        
        results[n_vars] = {
            'learned': top_pred == true_parent,
            'loss_reduction': (initial_loss - final_loss) / initial_loss,
            'final_probs': pred_probs
        }
        
        print(f"  Predicted X{top_pred} (true: X{true_parent})")
        print(f"  Loss reduction: {results[n_vars]['loss_reduction']:.2%}")
        print(f"  Success: {'✅' if results[n_vars]['learned'] else '❌'}")
    
    # Check consistency across variable counts
    all_learned = all([r['learned'] for r in results.values()])
    similar_performance = np.std([r['loss_reduction'] for r in results.values()]) < 0.2
    
    print(f"\nConsistent learning across variable counts: {'✅' if all_learned and similar_performance else '❌'}")
    
    return all_learned


def main():
    """Run all learning tests."""
    print("="*80)
    print("MODEL LEARNING CAPABILITY TEST")
    print("="*80)
    
    test1 = test_learning_on_complex_data()
    test2 = test_different_variable_counts()
    
    print("\n" + "="*80)
    if test1 and test2:
        print("✅ ALL TESTS PASSED!")
        print("\nThe variable-agnostic model maintains strong learning capability.")
    else:
        print("❌ Some tests failed.")
        print("\nThe model may have reduced learning capability.")


if __name__ == "__main__":
    main()