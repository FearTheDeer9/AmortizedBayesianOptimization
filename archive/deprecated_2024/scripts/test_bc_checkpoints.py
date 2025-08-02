#!/usr/bin/env python3
"""
Test different BC checkpoints to find ones that produce varied outputs.
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

from src.causal_bayes_opt.training.bc_model_inference import create_bc_surrogate_inference_fn


def test_checkpoint(checkpoint_path, test_name="Test"):
    """Test a single checkpoint with different inputs."""
    print(f"\n{test_name}: {checkpoint_path.name}")
    print("-" * 60)
    
    try:
        # Create inference function
        inference_fn = create_bc_surrogate_inference_fn(str(checkpoint_path))
        
        # Test data setup
        n_samples = 10
        n_vars = 4
        variables = ["X0", "X1", "X2", "X3"]
        target = "X3"
        key = random.PRNGKey(42)
        
        results = []
        
        # Test 1: Random observational data
        key, subkey = random.split(key)
        avici_data1 = jnp.zeros((n_samples, n_vars, 3))
        avici_data1 = avici_data1.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
        avici_data1 = avici_data1.at[:, :, 2].set(1.0)
        
        posterior1 = inference_fn(avici_data1, variables, target)
        results.append(("Random obs", posterior1))
        
        # Test 2: Strong X1->X3 correlation
        key, subkey = random.split(key)
        avici_data2 = jnp.zeros((n_samples, n_vars, 3))
        x1_vals = random.normal(subkey, (n_samples,))
        avici_data2 = avici_data2.at[:, 1, 0].set(x1_vals)
        avici_data2 = avici_data2.at[:, 3, 0].set(2.0 * x1_vals + 0.1 * random.normal(subkey, (n_samples,)))
        avici_data2 = avici_data2.at[:, [0, 2], 0].set(random.normal(subkey, (n_samples, 2)))
        avici_data2 = avici_data2.at[:, :, 2].set(1.0)
        
        posterior2 = inference_fn(avici_data2, variables, target)
        results.append(("X1->X3 corr", posterior2))
        
        # Test 3: Interventional data on X1
        key, subkey = random.split(key)
        avici_data3 = jnp.zeros((n_samples, n_vars, 3))
        avici_data3 = avici_data3.at[:, :, 0].set(random.normal(subkey, (n_samples, n_vars)))
        # Intervene on X1
        avici_data3 = avici_data3.at[:, 1, 1].set(1.0)
        avici_data3 = avici_data3.at[:, 1, 2].set(0.0)
        # Rest observational
        avici_data3 = avici_data3.at[:, [0, 2, 3], 2].set(1.0)
        
        posterior3 = inference_fn(avici_data3, variables, target)
        results.append(("Interv X1", posterior3))
        
        # Display results
        for desc, posterior in results:
            print(f"\n  {desc}:")
            for ps, prob in posterior.top_k_sets[:3]:
                ps_str = f"{{{', '.join(sorted(ps))}}}" if ps else "∅"
                print(f"    {ps_str}: {prob:.4f}")
            print(f"    Uncertainty: {posterior.uncertainty:.4f}")
        
        # Check variation
        uncertainties = [p.uncertainty for _, p in results]
        top_probs = [p.top_k_sets[0][1] for _, p in results]
        
        uncertainty_range = max(uncertainties) - min(uncertainties)
        prob_range = max(top_probs) - min(top_probs)
        
        print(f"\n  Variation:")
        print(f"    Uncertainty range: {uncertainty_range:.4f}")
        print(f"    Top prob range: {prob_range:.4f}")
        
        if uncertainty_range < 0.01 and prob_range < 0.01:
            print("    ❌ No variation detected")
            return False
        else:
            print("    ✅ Model shows variation")
            return True
            
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Test multiple checkpoints to find working ones."""
    print("Testing BC Surrogate Checkpoints")
    print("=" * 60)
    
    checkpoint_dir = project_root / "checkpoints/behavioral_cloning/dev/surrogate"
    
    # Test different epochs and levels
    test_cases = [
        # Earlier epochs (might not have collapsed yet)
        ("Level 2, Epoch 10", "surrogate_bc_development_epoch_10_level_2_1753294174.pkl"),
        ("Level 2, Epoch 11", "surrogate_bc_development_epoch_11_level_2_1753294195.pkl"),
        # Mid training
        ("Level 3, Epoch 20", "surrogate_bc_development_epoch_20_level_3_1753292426.pkl"),
        # Latest (known to be collapsed)
        ("Level 3, Epoch 22", "surrogate_bc_development_epoch_22_level_3_1753298905.pkl"),
    ]
    
    working_checkpoints = []
    
    for test_name, checkpoint_name in test_cases:
        checkpoint_path = checkpoint_dir / checkpoint_name
        if checkpoint_path.exists():
            if test_checkpoint(checkpoint_path, test_name):
                working_checkpoints.append(checkpoint_path)
        else:
            print(f"\n{test_name}: Not found")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Tested {len(test_cases)} checkpoints")
    print(f"Working checkpoints: {len(working_checkpoints)}")
    
    if working_checkpoints:
        print("\nRecommended checkpoints:")
        for cp in working_checkpoints:
            print(f"  - {cp.name}")
    else:
        print("\n⚠️  No working checkpoints found!")
        print("The BC surrogate models appear to have training issues.")


if __name__ == "__main__":
    main()