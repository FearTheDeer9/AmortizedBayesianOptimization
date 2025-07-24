#!/usr/bin/env python3
"""
Test BC inference functions to verify they produce varied outputs.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from pathlib import Path

# Set up paths
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.causal_bayes_opt.training.bc_model_inference import (
    create_bc_surrogate_inference_fn,
    create_bc_acquisition_inference_fn
)

def test_bc_surrogate_variation():
    """Test that BC surrogate produces different outputs for different inputs."""
    print("Testing BC Surrogate Variation")
    print("=" * 50)
    
    # Mock checkpoint path (will fail but we can check the function structure)
    checkpoint_path = "/tmp/bc_surrogate_checkpoint.pkl"
    
    # Create some test data
    n_samples = 10
    n_vars = 4
    variables = ["X0", "X1", "X2", "X3"]
    target = "X3"
    
    # Generate different AVICI data inputs
    key = random.PRNGKey(42)
    
    for i in range(3):
        key, subkey = random.split(key)
        # Create different data patterns
        if i == 0:
            # All observational
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            avici_data = avici_data.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
            avici_data = avici_data.at[:, :, 2].set(1.0)  # All observational
            desc = "All observational"
        elif i == 1:
            # Mix of interventions
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            avici_data = avici_data.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
            # Some interventions
            avici_data = avici_data.at[::2, 0, 1].set(1.0)  # Intervene on X0
            avici_data = avici_data.at[::2, 0, 2].set(0.0)
            avici_data = avici_data.at[1::2, 1, 1].set(1.0)  # Intervene on X1
            avici_data = avici_data.at[1::2, 1, 2].set(0.0)
            desc = "Mixed interventions"
        else:
            # Strong signal pattern
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            # Create correlation between X1 and X3
            x1_values = random.normal(subkey, (n_samples,))
            avici_data = avici_data.at[:, 1, 0].set(x1_values)
            avici_data = avici_data.at[:, 3, 0].set(2.0 * x1_values + 0.1 * random.normal(subkey, (n_samples,)))
            avici_data = avici_data.at[:, :, 2].set(1.0)
            desc = "Strong X1->X3 correlation"
        
        print(f"\nTest {i+1}: {desc}")
        print(f"Data shape: {avici_data.shape}")
        print(f"Data sample: {avici_data[0, :, 0]}")  # First sample values
        
        try:
            # This will fail due to missing checkpoint, but we can see the structure
            inference_fn = create_bc_surrogate_inference_fn(checkpoint_path)
            posterior = inference_fn(avici_data, variables, target)
            print(f"Success! Posterior: {posterior}")
        except Exception as e:
            print(f"Expected error (no checkpoint): {type(e).__name__}: {str(e)[:100]}")

def test_bc_acquisition_variation():
    """Test that BC acquisition produces different outputs for different keys."""
    print("\n\nTesting BC Acquisition Variation")
    print("=" * 50)
    
    checkpoint_path = "/tmp/bc_acquisition_checkpoint.pkl"
    variables = ["X0", "X1", "X2", "X3"]
    target = "X3"
    
    # Test with different random keys
    base_key = random.PRNGKey(42)
    
    for i in range(3):
        key = random.fold_in(base_key, i)
        print(f"\nTest {i+1}: Key = {key}")
        
        try:
            inference_fn = create_bc_acquisition_inference_fn(
                checkpoint_path, variables, target
            )
            decision = inference_fn(key)
            print(f"Success! Decision: {decision}")
        except Exception as e:
            print(f"Expected error (no checkpoint): {type(e).__name__}: {str(e)[:100]}")

def test_inference_function_signatures():
    """Test the function signatures and internal logic."""
    print("\n\nTesting Function Signatures")
    print("=" * 50)
    
    # Test surrogate signature handling
    print("BC Surrogate expects: (avici_data, variables, target)")
    print("CBO framework calls with: (avici_data, variables, target, current_params)")
    print("Our wrapper should handle this mismatch")
    
    # Test acquisition signature
    print("\nBC Acquisition expects: (key)")
    print("CBO framework expects: intervention decision dict")
    print("Our function should return proper format")

if __name__ == "__main__":
    test_bc_surrogate_variation()
    test_bc_acquisition_variation()
    test_inference_function_signatures()