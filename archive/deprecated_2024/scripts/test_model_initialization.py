#!/usr/bin/env python3
"""
Test model initialization and prediction to find source of astronomical losses.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.random as random

from src.causal_bayes_opt.avici_integration.continuous.factory import (
    create_continuous_parent_set_config,
    create_continuous_parent_set_model
)


def test_model_predictions():
    """Test model initialization and predictions."""
    print("Testing continuous model initialization and predictions")
    
    # Create simple test case
    num_vars = 5
    variable_names = [f"X{i}" for i in range(num_vars)]
    target_variable = "X0"
    target_idx = 0
    
    # Create model configuration
    config = create_continuous_parent_set_config(
        variables=variable_names,
        target_variable=target_variable,
        model_complexity="medium",
        use_attention=True,
        temperature=1.0
    )
    
    print(f"\nModel config created:")
    print(f"  Variables: {variable_names}")
    print(f"  Target: {target_variable}")
    
    # Create model
    model, model_config = create_continuous_parent_set_model(config)
    print(f"\nModel created with config: {model_config}")
    
    # Initialize model
    key = random.PRNGKey(42)
    dummy_data = jnp.ones((100, num_vars, 3))  # [N, d, 3] format
    
    print(f"\nInitializing model with data shape: {dummy_data.shape}")
    
    # Initialize parameters
    try:
        params = model.init(key, dummy_data, target_idx, True)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        output = model.apply(params, key, dummy_data, target_idx, False)
        print(f"Model output type: {type(output)}")
        
        if isinstance(output, dict):
            print(f"Output keys: {list(output.keys())}")
            if 'parent_probabilities' in output:
                probs = output['parent_probabilities']
                print(f"\nParent probabilities:")
                print(f"  Shape: {probs.shape}")
                print(f"  Values: {probs}")
                print(f"  Sum: {jnp.sum(probs)}")
                print(f"  Min/Max: {jnp.min(probs)}/{jnp.max(probs)}")
                
                # Check for issues
                if jnp.any(jnp.isnan(probs)):
                    print("  WARNING: NaN values detected!")
                if jnp.any(probs < 0):
                    print("  WARNING: Negative probabilities!")
                if not jnp.allclose(jnp.sum(probs), 1.0, atol=1e-6):
                    print("  WARNING: Probabilities don't sum to 1!")
        else:
            print(f"Raw output: {output}")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different initializations
    print("\n\nTesting with different random seeds...")
    for seed in [0, 1, 2]:
        key = random.PRNGKey(seed)
        params = model.init(key, dummy_data, target_idx, True)
        output = model.apply(params, key, dummy_data, target_idx, False)
        
        if isinstance(output, dict) and 'parent_probabilities' in output:
            probs = output['parent_probabilities']
            print(f"\nSeed {seed}: sum={jnp.sum(probs):.6f}, "
                  f"min={jnp.min(probs):.6f}, max={jnp.max(probs):.6f}")
    
    # Test KL divergence computation
    print("\n\nTesting KL divergence computation...")
    from src.causal_bayes_opt.training.bc_surrogate_trainer import kl_divergence_loss_jax
    
    # Create target distribution (expert)
    target_probs = jnp.ones(num_vars) / (num_vars - 1)
    target_probs = target_probs.at[target_idx].set(0.0)
    print(f"Target probs: {target_probs}")
    
    # Test with model output
    if isinstance(output, dict) and 'parent_probabilities' in output:
        pred_probs = output['parent_probabilities']
        
        # Ensure same shape
        if pred_probs.shape[0] != target_probs.shape[0]:
            print(f"Shape mismatch: pred={pred_probs.shape}, target={target_probs.shape}")
            # Pad or truncate
            if pred_probs.shape[0] < target_probs.shape[0]:
                pred_probs = jnp.pad(pred_probs, (0, target_probs.shape[0] - pred_probs.shape[0]))
            else:
                pred_probs = pred_probs[:target_probs.shape[0]]
        
        kl_loss = kl_divergence_loss_jax(pred_probs, target_probs)
        print(f"\nKL divergence: {float(kl_loss)}")
        
        if abs(kl_loss) > 1000:
            print("WARNING: Astronomical KL loss detected!")


if __name__ == "__main__":
    test_model_predictions()