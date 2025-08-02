#!/usr/bin/env python3
"""Debug BC surrogate inference with different variable counts."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import jax.random as random
from src.causal_bayes_opt.training.bc_model_inference import create_bc_surrogate_inference_fn

# Find a checkpoint
checkpoint_dir = Path("checkpoints/behavioral_cloning/dev/surrogate")
checkpoints = list(checkpoint_dir.glob("*.pkl"))
if not checkpoints:
    print("No checkpoints found!")
    sys.exit(1)

checkpoint_path = str(checkpoints[0])
print(f"Using checkpoint: {checkpoint_path}")

# Create inference function
surrogate_fn = create_bc_surrogate_inference_fn(checkpoint_path, threshold=0.1)

# Test with different variable counts
for n_vars in [3, 4, 5]:
    print(f"\n{'='*60}")
    print(f"Testing with {n_vars} variables")
    print(f"{'='*60}")
    
    # Create dummy data
    key = random.PRNGKey(42)
    data = random.normal(key, (100, n_vars, 3))
    
    # Create variable names
    variables = [f'X{i}' for i in range(n_vars)]
    target = variables[-1]  # Last variable as target
    
    print(f"Variables: {variables}")
    print(f"Target: {target}")
    print(f"Data shape: {data.shape}")
    
    # Try inference
    try:
        result = surrogate_fn(data, variables, target)
        print(f"✅ Success! Result type: {type(result)}")
        if hasattr(result, 'parent_sets'):
            print(f"   Parent sets: {len(result.parent_sets)}")
        if hasattr(result, 'probabilities'):
            print(f"   Probabilities shape: {result.probabilities.shape if hasattr(result.probabilities, 'shape') else len(result.probabilities)}")
    except Exception as e:
        print(f"❌ Failed with error: {e}")
        import traceback
        traceback.print_exc()