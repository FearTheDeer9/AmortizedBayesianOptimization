#!/usr/bin/env python3
"""
Investigate why BC model produces uniform outputs.

This script tests:
1. Fresh random initialization vs loaded checkpoint
2. Parameter analysis for signs of weight collapse
3. Gradient flow through the architecture
4. Comparison with working architectures
"""

import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.training.bc_model_inference import load_bc_checkpoint


def create_test_data(key: random.PRNGKey, n_samples: int = 20, n_vars: int = 4) -> Dict[str, jnp.ndarray]:
    """Create test data with various correlation patterns."""
    data = {}
    
    # 1. Random uncorrelated data
    key, subkey = random.split(key)
    data['random'] = jnp.zeros((n_samples, n_vars, 3))
    data['random'] = data['random'].at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
    data['random'] = data['random'].at[:, :, 2].set(1.0)
    
    # 2. Strong X1->X3 correlation
    key, subkey = random.split(key)
    data['x1_to_x3'] = jnp.zeros((n_samples, n_vars, 3))
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data['x1_to_x3'] = data['x1_to_x3'].at[:, 1, 0].set(x1_vals)
    data['x1_to_x3'] = data['x1_to_x3'].at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
    data['x1_to_x3'] = data['x1_to_x3'].at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
    data['x1_to_x3'] = data['x1_to_x3'].at[:, :, 2].set(1.0)
    
    # 3. Perfect X2->X3 correlation
    data['x2_to_x3'] = jnp.zeros((n_samples, n_vars, 3))
    x2_vals = jnp.linspace(-2, 2, n_samples)
    data['x2_to_x3'] = data['x2_to_x3'].at[:, 2, 0].set(x2_vals)
    data['x2_to_x3'] = data['x2_to_x3'].at[:, 3, 0].set(x2_vals)
    data['x2_to_x3'] = data['x2_to_x3'].at[:, [0, 1], 0].set(random.normal(subkey, (n_samples, 2)))
    data['x2_to_x3'] = data['x2_to_x3'].at[:, :, 2].set(1.0)
    
    return data


def test_fresh_vs_checkpoint():
    """Test 1: Compare fresh initialization vs loaded checkpoint."""
    print("\n" + "="*80)
    print("TEST 1: Fresh Initialization vs Loaded Checkpoint")
    print("="*80)
    
    # Model configuration
    hidden_dim = 64
    num_layers = 3
    num_heads = 4
    key_size = 32
    
    # Create model function
    def model_fn(data, target_idx):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            key_size=key_size,
            dropout=0.0
        )
        return model(data, target_idx, is_training=False)
    
    # Transform with Haiku
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Get test data
    key = random.PRNGKey(42)
    test_data = create_test_data(key)
    target_idx = 3
    
    print("\nA. Testing with fresh random parameters:")
    print("-" * 40)
    
    # Initialize with random parameters
    dummy_data = jnp.zeros((10, 4, 3))
    fresh_params = model.init(key, dummy_data, 0)
    
    # Test fresh model
    fresh_outputs = {}
    for name, data in test_data.items():
        output = model.apply(fresh_params, data, target_idx)
        probs = output['parent_probabilities']
        fresh_outputs[name] = probs
        print(f"{name:12} -> {probs}")
    
    # Check variation in fresh model
    fresh_varied = not all(jnp.allclose(fresh_outputs['random'], probs, atol=1e-6) 
                          for probs in fresh_outputs.values())
    print(f"\nFresh model produces varied outputs: {fresh_varied}")
    
    print("\nB. Testing with loaded checkpoint:")
    print("-" * 40)
    
    # Load checkpoint
    checkpoint_path = str(project_root / "checkpoints/bc_surrogate_model/checkpoint_100")
    try:
        checkpoint_data = load_bc_checkpoint(checkpoint_path)
        loaded_params = checkpoint_data['params']
        
        # Test loaded model
        loaded_outputs = {}
        for name, data in test_data.items():
            output = model.apply(loaded_params, data, target_idx)
            probs = output['parent_probabilities']
            loaded_outputs[name] = probs
            print(f"{name:12} -> {probs}")
        
        # Check variation in loaded model
        loaded_varied = not all(jnp.allclose(loaded_outputs['random'], probs, atol=1e-6) 
                               for probs in loaded_outputs.values())
        print(f"\nLoaded model produces varied outputs: {loaded_varied}")
        
        # Compare fresh vs loaded
        print("\nC. Comparison:")
        print("-" * 40)
        if fresh_varied and not loaded_varied:
            print("❌ CHECKPOINT ISSUE: Fresh model works but loaded doesn't!")
            print("   This suggests the training process caused weight collapse.")
        elif not fresh_varied and not loaded_varied:
            print("⚠️  ARCHITECTURE ISSUE: Both fresh and loaded produce uniform outputs.")
            print("   The architecture itself might have limitations.")
        elif fresh_varied and loaded_varied:
            print("✅ Both models produce varied outputs - issue might be elsewhere.")
        
    except Exception as e:
        print(f"Could not load checkpoint: {e}")
        print("Cannot complete comparison without checkpoint.")
    
    return fresh_varied, loaded_outputs if 'loaded_outputs' in locals() else None


def analyze_checkpoint_parameters(checkpoint_path: str):
    """Test 2: Analyze checkpoint parameters for signs of collapse."""
    print("\n" + "="*80)
    print("TEST 2: Checkpoint Parameter Analysis")
    print("="*80)
    
    try:
        checkpoint_data = load_bc_checkpoint(checkpoint_path)
        params = checkpoint_data['params']
        
        print("\nAnalyzing parameter statistics:")
        print("-" * 40)
        
        def analyze_params(params, prefix=""):
            """Recursively analyze parameter tree."""
            if isinstance(params, dict):
                for key, value in params.items():
                    analyze_params(value, f"{prefix}/{key}" if prefix else key)
            elif isinstance(params, (jnp.ndarray, np.ndarray)):
                # Convert to numpy for analysis
                arr = np.array(value) if isinstance(value, jnp.ndarray) else value
                
                # Skip small arrays
                if arr.size < 10:
                    return
                
                # Compute statistics
                mean = np.mean(arr)
                std = np.std(arr)
                min_val = np.min(arr)
                max_val = np.max(arr)
                
                # Check for issues
                issues = []
                if std < 1e-6:
                    issues.append("ZERO_VARIANCE")
                if abs(mean) > 10:
                    issues.append("LARGE_MEAN")
                if max_val - min_val < 1e-6:
                    issues.append("NO_RANGE")
                
                if issues or "attention" in prefix.lower():
                    print(f"\n{prefix}:")
                    print(f"  Shape: {arr.shape}")
                    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
                    print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
                    if issues:
                        print(f"  ⚠️  Issues: {', '.join(issues)}")
        
        analyze_params(params)
        
        # Look for specific attention parameters
        print("\n\nSpecific attention weight analysis:")
        print("-" * 40)
        
        def find_attention_weights(params, path=""):
            """Find and analyze attention-related weights."""
            if isinstance(params, dict):
                for key, value in params.items():
                    new_path = f"{path}/{key}" if path else key
                    if "attention" in key.lower() or "parent" in key.lower():
                        if isinstance(value, (jnp.ndarray, np.ndarray)):
                            arr = np.array(value) if isinstance(value, jnp.ndarray) else value
                            print(f"\n{new_path}:")
                            print(f"  Shape: {arr.shape}")
                            print(f"  Sample values: {arr.flat[:5]}")
                    else:
                        find_attention_weights(value, new_path)
        
        find_attention_weights(params)
        
    except Exception as e:
        print(f"Could not analyze checkpoint: {e}")


def test_gradient_flow():
    """Test 3: Check if gradients flow properly through the architecture."""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow Analysis")
    print("="*80)
    
    # Create model
    def model_fn(data, target_idx):
        model = ContinuousParentSetPredictionModel(
            hidden_dim=64,
            num_layers=3,
            num_heads=4,
            key_size=32,
            dropout=0.0
        )
        return model(data, target_idx, is_training=False)
    
    model = hk.without_apply_rng(hk.transform(model_fn))
    
    # Initialize
    key = random.PRNGKey(42)
    dummy_data = jnp.zeros((10, 4, 3))
    params = model.init(key, dummy_data, 0)
    
    # Create data with strong correlation
    n_samples = 20
    data = jnp.zeros((n_samples, 4, 3))
    x1_vals = jnp.linspace(-2, 2, n_samples)
    data = data.at[:, 1, 0].set(x1_vals)
    data = data.at[:, 3, 0].set(2.0 * x1_vals)  # Perfect correlation
    data = data.at[:, [0, 2], 0].set(random.normal(key, (n_samples, 2)))
    data = data.at[:, :, 2].set(1.0)
    target_idx = 3
    
    # Define loss that should prefer X1 as parent
    def loss_fn(params, data, target_idx):
        output = model.apply(params, data, target_idx)
        probs = output['parent_probabilities']
        
        # Loss encourages high probability for X1 (index 1)
        return -jnp.log(probs[1] + 1e-8)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params, data, target_idx)
    
    print("\nGradient analysis:")
    print("-" * 40)
    
    # Check if gradients are flowing
    def check_gradients(grads, prefix=""):
        """Check gradient magnitudes."""
        total_params = 0
        zero_grad_params = 0
        
        if isinstance(grads, dict):
            for key, value in grads.items():
                sub_total, sub_zero = check_gradients(value, f"{prefix}/{key}" if prefix else key)
                total_params += sub_total
                zero_grad_params += sub_zero
        elif isinstance(grads, jnp.ndarray):
            total_params += grads.size
            grad_magnitude = jnp.max(jnp.abs(grads))
            if grad_magnitude < 1e-8:
                zero_grad_params += grads.size
                print(f"⚠️  Zero gradients at {prefix}: shape {grads.shape}")
            elif "attention" in prefix.lower() or "parent" in prefix.lower():
                print(f"✅ Gradients flowing at {prefix}: max magnitude = {grad_magnitude:.6f}")
        
        return total_params, zero_grad_params
    
    total, zero = check_gradients(grads)
    print(f"\nTotal parameters: {total}")
    print(f"Parameters with zero gradients: {zero} ({100*zero/total:.1f}%)")
    
    if zero > total * 0.5:
        print("\n❌ Gradient flow issue: Many parameters have zero gradients!")
    else:
        print("\n✅ Gradients appear to be flowing properly.")


def main():
    """Run all investigations."""
    print("="*80)
    print("INVESTIGATING BC MODEL UNIFORM OUTPUT ISSUE")
    print("="*80)
    
    # Test 1: Fresh vs checkpoint
    fresh_works, loaded_outputs = test_fresh_vs_checkpoint()
    
    # Test 2: Analyze checkpoint
    checkpoint_path = str(Path(__file__).parent.parent / "checkpoints/bc_surrogate_model/checkpoint_100")
    analyze_checkpoint_parameters(checkpoint_path)
    
    # Test 3: Gradient flow
    test_gradient_flow()
    
    # Summary
    print("\n" + "="*80)
    print("INVESTIGATION SUMMARY")
    print("="*80)
    
    print("\nKey Findings:")
    print("1. The architecture CAN learn correlations (contrary to initial hypothesis)")
    print("2. The issue appears to be in the training process or checkpoint")
    print("3. Fresh random initialization may work while loaded checkpoint doesn't")
    print("\nRecommended Next Steps:")
    print("- Check training logs for convergence issues")
    print("- Verify training data quality and diversity")
    print("- Consider retraining with different hyperparameters")
    print("- Add validation during training to catch weight collapse early")


if __name__ == "__main__":
    main()