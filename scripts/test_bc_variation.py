#!/usr/bin/env python3
"""
Test BC models to verify they produce varied outputs with real checkpoints.
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
    """Test BC surrogate with real checkpoints."""
    print("Testing BC Surrogate Variation with Real Checkpoints")
    print("=" * 60)
    
    # Use a real checkpoint
    checkpoint_path = project_root / "checkpoints/behavioral_cloning/dev/surrogate/surrogate_bc_development_epoch_22_level_3_1753298905.pkl"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found, trying different path...")
        # Try alternative paths
        alt_paths = [
            project_root / "checkpoints/bc_surrogate_checkpoint.pkl",
            project_root / "checkpoints/surrogate_bc.pkl"
        ]
        for alt in alt_paths:
            if alt.exists():
                checkpoint_path = alt
                break
        else:
            print("No BC surrogate checkpoint found")
            return
    
    print(f"Using checkpoint: {checkpoint_path.name}")
    
    try:
        # Create inference function
        inference_fn = create_bc_surrogate_inference_fn(str(checkpoint_path))
        print("✓ Successfully loaded BC surrogate model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Test with different inputs
    n_samples = 10
    n_vars = 4
    variables = ["X0", "X1", "X2", "X3"]
    target = "X3"
    
    key = random.PRNGKey(42)
    results = []
    
    print("\nTesting different input patterns:")
    
    for i in range(3):
        key, subkey = random.split(key)
        
        # Create different data patterns
        if i == 0:
            # Random data
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            avici_data = avici_data.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
            avici_data = avici_data.at[:, :, 2].set(1.0)  # All observational
            desc = "All observational (random)"
        elif i == 1:
            # Mix interventions on X0 and X1
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            avici_data = avici_data.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
            # Intervene on X0 for half samples
            avici_data = avici_data.at[:5, 0, 1].set(1.0)
            avici_data = avici_data.at[:5, 0, 2].set(0.0)
            # Intervene on X1 for other half
            avici_data = avici_data.at[5:, 1, 1].set(1.0)
            avici_data = avici_data.at[5:, 1, 2].set(0.0)
            # Rest observational
            mask = jnp.ones((n_samples, n_vars))
            mask = mask.at[:5, 0].set(0)
            mask = mask.at[5:, 1].set(0)
            avici_data = avici_data.at[:, :, 2].set(mask)
            desc = "Mixed interventions (X0, X1)"
        else:
            # Strong correlation between X1 and X3
            avici_data = jnp.zeros((n_samples, n_vars, 3))
            x1_vals = random.normal(subkey, (n_samples,))
            avici_data = avici_data.at[:, 1, 0].set(x1_vals)
            avici_data = avici_data.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
            avici_data = avici_data.at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
            avici_data = avici_data.at[:, :, 2].set(1.0)
            desc = "Strong X1→X3 correlation"
        
        print(f"\nTest {i+1}: {desc}")
        
        try:
            # Get posterior
            posterior = inference_fn(avici_data, variables, target)
            
            # Store results
            result = {
                'parent_set_probs': posterior.parent_set_probs,
                'top_k_sets': posterior.top_k_sets,
                'uncertainty': posterior.uncertainty,
                'metadata': getattr(posterior, 'metadata', {})
            }
            results.append(result)
            
            # Display results
            print(f"  Top parent sets and probabilities:")
            for ps, prob in posterior.top_k_sets[:5]:  # Show top 5
                ps_str = f"{{{', '.join(sorted(ps))}}}" if ps else "∅"
                print(f"    {ps_str}: {prob:.4f}")
            print(f"  Uncertainty: {posterior.uncertainty:.4f}")
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze variation
    if len(results) >= 2:
        print("\nVariation Analysis:")
        # Check if uncertainties differ
        uncertainties = [r['uncertainty'] for r in results]
        if len(set(uncertainties)) == 1 or max(uncertainties) - min(uncertainties) < 0.01:
            print("  ⚠️  WARNING: All tests produced identical uncertainties!")
        else:
            print("  ✓ Different inputs produced different uncertainties")
            print(f"    Range: {min(uncertainties):.4f} - {max(uncertainties):.4f}")
            
        # Check top parent set variations
        top_sets = [tuple((tuple(sorted(ps)), prob) for ps, prob in r['top_k_sets'][:3]) for r in results]
        if len(set(top_sets)) == 1:
            print("  ⚠️  WARNING: All tests produced identical top parent sets!")
        else:
            print("  ✓ Different inputs produced different top parent sets")

def test_bc_acquisition_variation():
    """Test BC acquisition with real checkpoints."""
    print("\n\nTesting BC Acquisition Variation with Real Checkpoints")
    print("=" * 60)
    
    # Use a real checkpoint
    checkpoint_path = project_root / "checkpoints/behavioral_cloning/dev/acquisition/bc_demo_acquisition_epoch_6_level_3_1753299449.pkl"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found, trying different path...")
        # Try alternative paths
        alt_paths = [
            project_root / "checkpoints/bc_acquisition_checkpoint.pkl",
            project_root / "checkpoints/acquisition_bc.pkl"
        ]
        for alt in alt_paths:
            if alt.exists():
                checkpoint_path = alt
                break
        else:
            print("No BC acquisition checkpoint found")
            return
    
    print(f"Using checkpoint: {checkpoint_path.name}")
    
    variables = ["X0", "X1", "X2", "X3"]
    target = "X3"
    
    try:
        # Create inference function
        inference_fn = create_bc_acquisition_inference_fn(
            str(checkpoint_path), variables, target
        )
        print("✓ Successfully loaded BC acquisition model")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    print("\nTesting with different random seeds:")
    
    # Test with different keys
    base_key = random.PRNGKey(42)
    decisions = []
    
    for i in range(5):
        key = random.fold_in(base_key, i)
        
        try:
            decision = inference_fn(key)
            var = list(decision['intervention_variables'])[0]
            val = decision['intervention_values'][0]
            
            print(f"  Seed {i}: Intervene on {var} = {val:.4f}")
            decisions.append((var, val))
            
        except Exception as e:
            print(f"  Error with seed {i}: {e}")
    
    # Analyze variation
    if decisions:
        unique_vars = set(d[0] for d in decisions)
        values = [d[1] for d in decisions]
        value_range = max(values) - min(values)
        value_std = np.std(values)
        
        print(f"\nVariation Analysis:")
        print(f"  Unique variables selected: {unique_vars}")
        print(f"  Value range: {value_range:.4f}")
        print(f"  Value std dev: {value_std:.4f}")
        
        if len(unique_vars) == 1 and value_range < 0.01:
            print("  ⚠️  WARNING: Model produces nearly identical outputs!")
        else:
            print("  ✓ Model produces varied outputs")

if __name__ == "__main__":
    test_bc_surrogate_variation()
    test_bc_acquisition_variation()