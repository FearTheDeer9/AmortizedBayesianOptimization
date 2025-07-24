#!/usr/bin/env python3
"""
Diagnose why BC surrogate produces fixed outputs.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import pickle
import gzip
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.bc_model_inference import load_bc_checkpoint, create_bc_surrogate_inference_fn


def inspect_model_behavior():
    """Inspect BC surrogate model behavior in detail."""
    print("Diagnosing BC Surrogate Model")
    print("=" * 60)
    
    # Load checkpoint
    checkpoint_path = project_root / "checkpoints/behavioral_cloning/dev/surrogate/surrogate_bc_development_epoch_22_level_3_1753298905.pkl"
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    
    # Check checkpoint contents
    print("\nCheckpoint contents:")
    for key in checkpoint_data.keys():
        print(f"  - {key}")
    
    # Check training state
    if 'training_state' in checkpoint_data:
        state = checkpoint_data['training_state']
        print(f"\nTraining state:")
        print(f"  - Total samples seen: {getattr(state, 'total_samples_seen', 'N/A')}")
        print(f"  - Best validation loss: {getattr(state, 'best_val_loss', 'N/A')}")
        print(f"  - Current validation loss: {getattr(state, 'current_val_loss', 'N/A')}")
    
    # Create inference function
    inference_fn = create_bc_surrogate_inference_fn(str(checkpoint_path))
    
    # Test with extreme inputs
    n_samples = 10
    n_vars = 4
    variables = ["X0", "X1", "X2", "X3"]
    target = "X3"
    
    print("\n\nTesting with extreme inputs:")
    
    # Test 1: All zeros
    print("\n1. All zeros input:")
    data_zeros = jnp.zeros((n_samples, n_vars, 3))
    posterior_zeros = inference_fn(data_zeros, variables, target)
    if 'continuous_probs' in posterior_zeros.metadata:
        print(f"   Continuous probs: {posterior_zeros.metadata['continuous_probs']}")
    print(f"   Top set: {posterior_zeros.top_k_sets[0]}")
    
    # Test 2: All ones
    print("\n2. All ones input:")
    data_ones = jnp.ones((n_samples, n_vars, 3))
    posterior_ones = inference_fn(data_ones, variables, target)
    if 'continuous_probs' in posterior_ones.metadata:
        print(f"   Continuous probs: {posterior_ones.metadata['continuous_probs']}")
    print(f"   Top set: {posterior_ones.top_k_sets[0]}")
    
    # Test 3: Large values
    print("\n3. Large values input:")
    data_large = jnp.ones((n_samples, n_vars, 3)) * 100
    posterior_large = inference_fn(data_large, variables, target)
    if 'continuous_probs' in posterior_large.metadata:
        print(f"   Continuous probs: {posterior_large.metadata['continuous_probs']}")
    print(f"   Top set: {posterior_large.top_k_sets[0]}")
    
    # Test 4: Perfect correlation pattern
    print("\n4. Perfect correlation (X1->X3):")
    data_corr = jnp.zeros((n_samples, n_vars, 3))
    x_vals = jnp.linspace(-2, 2, n_samples)
    data_corr = data_corr.at[:, 1, 0].set(x_vals)
    data_corr = data_corr.at[:, 3, 0].set(x_vals)  # Perfect correlation
    data_corr = data_corr.at[:, :, 2].set(1.0)  # All observational
    posterior_corr = inference_fn(data_corr, variables, target)
    if 'continuous_probs' in posterior_corr.metadata:
        print(f"   Continuous probs: {posterior_corr.metadata['continuous_probs']}")
    print(f"   Top set: {posterior_corr.top_k_sets[0]}")
    
    # Check if continuous probabilities are all the same
    if all('continuous_probs' in p.metadata for p in [posterior_zeros, posterior_ones, posterior_large, posterior_corr]):
        probs_list = [p.metadata['continuous_probs'] for p in [posterior_zeros, posterior_ones, posterior_large, posterior_corr]]
        all_same = all(probs_list[0] == probs for probs in probs_list[1:])
        print(f"\n\nAll continuous probabilities identical? {all_same}")
        
        if all_same:
            print("⚠️  Model is producing fixed outputs regardless of input!")
            print("This suggests the model either:")
            print("  1. Has collapsed during training")
            print("  2. Is not properly using the input data")
            print("  3. Has numerical issues (e.g., exploding/vanishing gradients)")


def check_model_parameters():
    """Check if model parameters look reasonable."""
    print("\n\nChecking Model Parameters")
    print("=" * 60)
    
    checkpoint_path = project_root / "checkpoints/behavioral_cloning/dev/surrogate/surrogate_bc_development_epoch_22_level_3_1753298905.pkl"
    checkpoint_data = load_bc_checkpoint(checkpoint_path)
    
    if 'model_params' in checkpoint_data:
        params = checkpoint_data['model_params']
        
        # Check parameter statistics
        print("\nParameter statistics:")
        for key, value in params.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if hasattr(subvalue, 'shape'):
                        param_mean = float(jnp.mean(jnp.abs(subvalue)))
                        param_std = float(jnp.std(subvalue))
                        param_max = float(jnp.max(jnp.abs(subvalue)))
                        print(f"  {key}/{subkey}: mean={param_mean:.4f}, std={param_std:.4f}, max={param_max:.4f}")
                        
                        # Check for issues
                        if param_max < 1e-6:
                            print(f"    ⚠️  Nearly zero parameters!")
                        elif param_max > 100:
                            print(f"    ⚠️  Very large parameters!")
    else:
        print("No model parameters found in checkpoint")


if __name__ == "__main__":
    inspect_model_behavior()
    check_model_parameters()