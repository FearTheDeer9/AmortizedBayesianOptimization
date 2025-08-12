#!/usr/bin/env python3
"""
Analyze the distribution of target values in the training data
to understand why value prediction is unstable.
"""

import sys
from pathlib import Path
import numpy as np
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from demonstration_to_tensor_fixed import create_bc_training_dataset

def analyze_value_distribution():
    """Analyze target value distribution to understand instability."""
    
    print("="*80)
    print("TARGET VALUE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Load demonstrations
    demos_path = Path("../expert_demonstrations/raw/raw_demonstrations")
    if not demos_path.exists():
        demos_path = Path("expert_demonstrations/raw/raw_demonstrations")
    
    raw_demos = load_demonstrations_from_path(str(demos_path), max_files=20)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations[:5])
        else:
            flat_demos.append(item)
    
    print(f"Processing {len(flat_demos)} demonstrations")
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    print(f"\nCreated {len(all_labels)} training examples")
    
    # Analyze target values
    all_target_values = []
    value_by_variable = {}
    
    for label in all_labels:
        targets = list(label.get('targets', []))
        if not targets:
            continue
            
        target_var = targets[0]
        target_value = label['values'].get(target_var, None)
        
        if target_value is not None:
            all_target_values.append(target_value)
            
            if target_var not in value_by_variable:
                value_by_variable[target_var] = []
            value_by_variable[target_var].append(target_value)
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL TARGET VALUE STATISTICS")
    print("="*60)
    
    if all_target_values:
        values_array = np.array(all_target_values)
        print(f"Count: {len(values_array)}")
        print(f"Mean: {np.mean(values_array):.4f}")
        print(f"Std: {np.std(values_array):.4f}")
        print(f"Min: {np.min(values_array):.4f}")
        print(f"Max: {np.max(values_array):.4f}")
        print(f"Median: {np.median(values_array):.4f}")
        
        # Check for extreme values
        extreme_low = np.sum(values_array < -10)
        extreme_high = np.sum(values_array > 10)
        print(f"\nExtreme values:")
        print(f"  < -10: {extreme_low} ({extreme_low/len(values_array)*100:.1f}%)")
        print(f"  > 10: {extreme_high} ({extreme_high/len(values_array)*100:.1f}%)")
        
        # Check for NaN/Inf
        nan_count = np.sum(np.isnan(values_array))
        inf_count = np.sum(np.isinf(values_array))
        print(f"\nProblematic values:")
        print(f"  NaN: {nan_count}")
        print(f"  Inf: {inf_count}")
    
    # Per-variable statistics
    print("\n" + "="*60)
    print("PER-VARIABLE VALUE STATISTICS")
    print("="*60)
    
    for var in sorted(value_by_variable.keys()):
        values = np.array(value_by_variable[var])
        print(f"\n{var}:")
        print(f"  Count: {len(values)}")
        print(f"  Mean: {np.mean(values):.4f}")
        print(f"  Std: {np.std(values):.4f}")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Check what happens with loss calculation
    print("\n" + "="*60)
    print("LOSS CALCULATION SIMULATION")
    print("="*60)
    
    # Simulate loss with different scenarios
    import jax.numpy as jnp
    
    # Scenario 1: Normal case
    target_value = 0.5
    predicted_mean = 0.0
    log_std = 0.0  # std = 1.0
    
    std = jnp.exp(log_std)
    loss1 = 0.5 * jnp.log(2 * jnp.pi) + log_std + 0.5 * ((target_value - predicted_mean) / std) ** 2
    print(f"\nNormal case:")
    print(f"  Target: {target_value}, Pred mean: {predicted_mean}, Std: {std}")
    print(f"  Loss: {loss1:.4f}")
    
    # Scenario 2: Large target value
    target_value = 100.0
    loss2 = 0.5 * jnp.log(2 * jnp.pi) + log_std + 0.5 * ((target_value - predicted_mean) / std) ** 2
    print(f"\nLarge target:")
    print(f"  Target: {target_value}, Pred mean: {predicted_mean}, Std: {std}")
    print(f"  Loss: {loss2:.4f}")
    
    # Scenario 3: Very small std (overconfident)
    log_std = -5.0  # std ≈ 0.0067
    std = jnp.exp(log_std)
    target_value = 1.0
    loss3 = 0.5 * jnp.log(2 * jnp.pi) + log_std + 0.5 * ((target_value - predicted_mean) / std) ** 2
    print(f"\nSmall std (overconfident):")
    print(f"  Target: {target_value}, Pred mean: {predicted_mean}, Std: {std}")
    print(f"  Loss: {loss3:.4f}")
    print(f"  MSE term alone: {0.5 * ((target_value - predicted_mean) / std) ** 2:.4f}")
    
    # Scenario 4: Edge case that causes instability
    # Model becomes very confident (small std) but wrong
    log_std = -3.0
    std = jnp.exp(log_std)
    target_value = 10.0  # Far from prediction
    predicted_mean = 0.0
    
    mse_term = 0.5 * ((target_value - predicted_mean) / std) ** 2
    loss4 = 0.5 * jnp.log(2 * jnp.pi) + log_std + mse_term
    print(f"\nInstability case (confident but wrong):")
    print(f"  Target: {target_value}, Pred mean: {predicted_mean}, Std: {std:.6f}")
    print(f"  MSE term: {mse_term:.4f}")
    print(f"  Total loss: {loss4:.4f}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    print("""
The value prediction becomes unstable when:
1. Target values have high variance or extreme values
2. Model becomes overconfident (low std) but wrong
3. The MSE term ((target - pred) / std)^2 explodes

In the training logs, we see the value loss exploding from 1.25 to 77,183.
This suggests the model is:
- Predicting very small std (overconfident)
- But being very wrong about the mean
- Leading to massive MSE terms
""")

if __name__ == "__main__":
    analyze_value_distribution()