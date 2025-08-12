#!/usr/bin/env python3
"""
Test if the model can actually distinguish between different variables
or if they all look the same in the embedding space.
"""

import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from demonstration_to_tensor_fixed import create_bc_training_dataset

def analyze_tensor_differences():
    """Check if different variables have different representations in the 5-channel tensors."""
    
    print("="*80)
    print("TESTING VARIABLE DISTINGUISHABILITY")
    print("="*80)
    
    # Load demonstrations
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    if not demos_path.exists():
        demos_path = Path("expert_demonstrations/raw/raw_demonstrations")
    
    raw_demos = load_demonstrations_from_path(str(demos_path), max_files=10)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations[:5])
        else:
            flat_demos.append(item)
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos[:50], max_trajectory_length=100
    )
    
    print(f"\nAnalyzing {len(all_inputs)} examples")
    print(f"Tensor shape: {all_inputs[0].shape}")  # Should be [T, n_vars, 5]
    
    # Analyze the 5-channel representations
    print("\n" + "="*60)
    print("5-CHANNEL TENSOR STRUCTURE")
    print("="*60)
    print("""
    Channel 0: Values (node features)
    Channel 1: Parent existence 
    Channel 2: Parent values
    Channel 3: Target variable indicator
    Channel 4: Trajectory position
    """)
    
    # Look at specific examples
    for i in range(min(3, len(all_inputs))):
        input_tensor = all_inputs[i]
        label = all_labels[i]
        variables = label.get('variables', [])
        target = list(label.get('targets', []))[0] if label.get('targets') else None
        
        print(f"\n--- Example {i+1} ---")
        print(f"Variables: {variables}")
        print(f"Target: {target}")
        print(f"Tensor shape: {input_tensor.shape}")
        
        # Check if variables have different representations
        if len(variables) >= 3:
            # Look at first timestep
            t = 0
            print(f"\nFirst timestep representations:")
            for j, var in enumerate(variables[:3]):
                var_repr = input_tensor[t, j, :]
                print(f"  {var}: {var_repr}")
            
            # Check if X0, X1, X2 have different patterns
            if 'X0' in variables and 'X1' in variables and 'X2' in variables:
                x0_idx = variables.index('X0')
                x1_idx = variables.index('X1')
                x2_idx = variables.index('X2')
                
                x0_repr = input_tensor[t, x0_idx, :]
                x1_repr = input_tensor[t, x1_idx, :]
                x2_repr = input_tensor[t, x2_idx, :]
                
                # Check differences
                print(f"\nDifferences between variables:")
                print(f"  X0 vs X1: L2 norm = {jnp.linalg.norm(x0_repr - x1_repr):.4f}")
                print(f"  X0 vs X2: L2 norm = {jnp.linalg.norm(x0_repr - x2_repr):.4f}")
                print(f"  X1 vs X2: L2 norm = {jnp.linalg.norm(x1_repr - x2_repr):.4f}")
                
                # Check which channels differ
                for c in range(5):
                    channel_name = ["Values", "Parent exist", "Parent vals", "Target ind", "Traj pos"][c]
                    x0_val = x0_repr[c]
                    x1_val = x1_repr[c]
                    x2_val = x2_repr[c]
                    
                    if abs(x0_val - x1_val) > 0.01 or abs(x0_val - x2_val) > 0.01:
                        print(f"  Channel {c} ({channel_name}): X0={x0_val:.3f}, X1={x1_val:.3f}, X2={x2_val:.3f}")
    
    # Statistical analysis across all examples
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Collect representations for X0, X1, X2 across examples
    x0_reprs = []
    x1_reprs = []
    x2_reprs = []
    
    for i, (input_tensor, label) in enumerate(zip(all_inputs, all_labels)):
        variables = label.get('variables', [])
        
        if 'X0' in variables and 'X1' in variables and 'X2' in variables:
            x0_idx = variables.index('X0')
            x1_idx = variables.index('X1')
            x2_idx = variables.index('X2')
            
            # Take first timestep
            x0_reprs.append(input_tensor[0, x0_idx, :])
            x1_reprs.append(input_tensor[0, x1_idx, :])
            x2_reprs.append(input_tensor[0, x2_idx, :])
    
    if x0_reprs:
        x0_reprs = jnp.array(x0_reprs)
        x1_reprs = jnp.array(x1_reprs)
        x2_reprs = jnp.array(x2_reprs)
        
        print(f"\nCollected {len(x0_reprs)} examples with X0, X1, X2")
        
        # Check variance per channel
        print("\nVariance per channel (how much each channel varies):")
        for c in range(5):
            channel_name = ["Values", "Parent exist", "Parent vals", "Target ind", "Traj pos"][c]
            x0_var = jnp.var(x0_reprs[:, c])
            x1_var = jnp.var(x1_reprs[:, c])
            x2_var = jnp.var(x2_reprs[:, c])
            print(f"  Channel {c} ({channel_name}): X0={x0_var:.4f}, X1={x1_var:.4f}, X2={x2_var:.4f}")
        
        # Check if representations are distinguishable
        print("\nMean differences between variables:")
        mean_x0 = jnp.mean(x0_reprs, axis=0)
        mean_x1 = jnp.mean(x1_reprs, axis=0)
        mean_x2 = jnp.mean(x2_reprs, axis=0)
        
        print(f"  Mean X0: {mean_x0}")
        print(f"  Mean X1: {mean_x1}")
        print(f"  Mean X2: {mean_x2}")
        print(f"  L2(X0-X1): {jnp.linalg.norm(mean_x0 - mean_x1):.4f}")
        print(f"  L2(X0-X2): {jnp.linalg.norm(mean_x0 - mean_x2):.4f}")
        print(f"  L2(X1-X2): {jnp.linalg.norm(mean_x1 - mean_x2):.4f}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    print("""
If the L2 distances between X0, X1, and X2 representations are very small,
then the model cannot distinguish between them based on the input features.

This would explain why:
1. The model resorts to position-based heuristics
2. X2 has poor accuracy (model defaults to X0/X1)
3. Learning plateaus (no signal to improve)

The model needs richer features that capture:
- Causal relationships specific to each variable
- Historical intervention patterns
- Variable-specific properties
""")

if __name__ == "__main__":
    analyze_tensor_differences()