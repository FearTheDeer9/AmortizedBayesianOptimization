#!/usr/bin/env python3
"""
Debug the data pipeline to understand why X4 is never predicted correctly.
Examines the full pipeline from demonstration to tensor to prediction.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper

def analyze_x4_examples():
    """Analyze examples where X4 is the target."""
    
    print("="*80)
    print("X4 DATA PIPELINE ANALYSIS")
    print("="*80)
    
    # Load demonstrations
    demos_path = 'expert_demonstrations/raw/raw_demonstrations'
    raw_demos = load_demonstrations_from_path(demos_path, max_files=100)
    
    # Flatten
    flat_demos = []
    for item in raw_demos:
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations)
        else:
            flat_demos.append(item)
    
    print(f"\nLoaded {len(flat_demos)} demonstrations")
    
    # Create dataset
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos[:100], max_trajectory_length=100
    )
    
    print(f"Created {len(all_inputs)} training examples")
    
    # Find X4 examples
    x4_examples = []
    variable_stats = defaultdict(lambda: defaultdict(int))
    scm_type_stats = defaultdict(lambda: defaultdict(int))
    
    for i, label in enumerate(all_labels):
        if 'targets' in label and label['targets']:
            target_var = list(label['targets'])[0]
            
            # Track which variables appear as targets
            variable_stats[target_var]['count'] += 1
            
            # Track SCM type if available
            scm_type = label.get('scm_type', 'unknown')
            scm_type_stats[scm_type][target_var] += 1
            
            if target_var == 'X4':
                x4_examples.append((i, all_inputs[i], label))
    
    print(f"\n{len(x4_examples)} examples have X4 as target")
    
    # Analyze variable distribution
    print("\n" + "="*60)
    print("VARIABLE TARGET DISTRIBUTION")
    print("="*60)
    print("\nVariable | Count | Percentage")
    print("-" * 35)
    
    total = len(all_labels)
    for var in sorted(variable_stats.keys()):
        count = variable_stats[var]['count']
        pct = count / total * 100
        print(f"{var:8s} | {count:5d} | {pct:5.1f}%")
    
    # Analyze SCM type distribution
    print("\n" + "="*60)
    print("VARIABLE DISTRIBUTION BY SCM TYPE")
    print("="*60)
    
    for scm_type in sorted(scm_type_stats.keys()):
        print(f"\n{scm_type} SCM:")
        for var, count in sorted(scm_type_stats[scm_type].items()):
            print(f"  {var}: {count}")
    
    # Deep dive into X4 examples
    if x4_examples:
        print("\n" + "="*60)
        print("DETAILED X4 EXAMPLE ANALYSIS")
        print("="*60)
        
        for idx, (example_idx, input_tensor, label) in enumerate(x4_examples[:5]):
            print(f"\n--- X4 Example {idx+1} (index {example_idx}) ---")
            
            # Print label details
            print("\nLabel contents:")
            print(f"  Targets: {label.get('targets', 'N/A')}")
            print(f"  Variables: {label.get('variables', 'N/A')}")
            print(f"  Target variable: {label.get('target_variable', 'N/A')}")
            print(f"  SCM type: {label.get('scm_type', 'N/A')}")
            
            # Check values
            if 'values' in label and 'X4' in label['values']:
                print(f"  X4 value: {label['values']['X4']}")
            
            # Check variable mapping
            variables = label.get('variables', [])
            if variables:
                print(f"\nVariable mapping:")
                for i, var in enumerate(variables):
                    marker = " <-- X4 is here!" if var == 'X4' else ""
                    print(f"  Index {i}: {var}{marker}")
                
                # Create VariableMapper
                mapper = VariableMapper(
                    variables=variables,
                    target_variable=label.get('target_variable')
                )
                
                # Check X4 index
                try:
                    x4_idx = mapper.get_index('X4')
                    print(f"\nVariableMapper says X4 is at index: {x4_idx}")
                except ValueError as e:
                    print(f"\nVariableMapper error for X4: {e}")
                
                # Check if X4 is in the expected position
                if 'X4' in variables:
                    actual_idx = variables.index('X4')
                    print(f"Actual X4 position in list: {actual_idx}")
                    if actual_idx != 4:
                        print("⚠️ WARNING: X4 is NOT at index 4!")
            
            # Analyze tensor shape and content
            print(f"\nInput tensor shape: {input_tensor.shape}")
            print(f"Tensor dtype: {input_tensor.dtype}")
            
            # Check tensor statistics
            print(f"Tensor stats:")
            print(f"  Min: {np.min(input_tensor):.3f}")
            print(f"  Max: {np.max(input_tensor):.3f}")
            print(f"  Mean: {np.mean(input_tensor):.3f}")
            print(f"  Has NaN: {np.any(np.isnan(input_tensor))}")
            
            # Check first timestep
            if len(input_tensor.shape) == 3:  # [T, n_vars, channels]
                first_step = input_tensor[0]
                print(f"\nFirst timestep (shape {first_step.shape}):")
                for var_idx in range(min(5, first_step.shape[0])):
                    var_name = variables[var_idx] if var_idx < len(variables) else f"idx_{var_idx}"
                    print(f"  {var_name}: {first_step[var_idx, :5]}")  # First 5 channels
    
    # Check for pattern in X4 appearances
    print("\n" + "="*60)
    print("X4 PATTERN ANALYSIS")
    print("="*60)
    
    if x4_examples:
        # Check if X4 appears in specific SCM types
        x4_scm_types = defaultdict(int)
        x4_positions = defaultdict(int)
        
        for _, _, label in x4_examples:
            scm_type = label.get('scm_type', 'unknown')
            x4_scm_types[scm_type] += 1
            
            variables = label.get('variables', [])
            if 'X4' in variables:
                x4_pos = variables.index('X4')
                x4_positions[x4_pos] += 1
        
        print("\nX4 appears in SCM types:")
        for scm_type, count in x4_scm_types.items():
            print(f"  {scm_type}: {count} times")
        
        print("\nX4 position distribution:")
        for pos, count in sorted(x4_positions.items()):
            print(f"  Position {pos}: {count} times")
        
        if len(x4_positions) > 1:
            print("\n⚠️ WARNING: X4 appears at DIFFERENT positions!")
            print("This could explain why the model can't learn it!")
    
    # Check if there's something special about demonstrations with X4
    print("\n" + "="*60)
    print("DEMONSTRATION ANALYSIS")
    print("="*60)
    
    # Look at the original demonstrations
    x4_demo_indices = set()
    for demo_idx, demo in enumerate(flat_demos[:100]):
        # Check if this demonstration has X4 as a target
        if hasattr(demo, 'target_variable'):
            # This is a trajectory
            for step in demo.trajectory:
                if hasattr(step, 'targets') and 'X4' in step.targets:
                    x4_demo_indices.add(demo_idx)
                    break
        elif hasattr(demo, 'targets') and 'X4' in demo.targets:
            x4_demo_indices.add(demo_idx)
    
    print(f"\nDemonstrations containing X4 interventions: {len(x4_demo_indices)}")
    
    if x4_demo_indices:
        # Sample one demonstration
        sample_idx = list(x4_demo_indices)[0]
        sample_demo = flat_demos[sample_idx]
        
        print(f"\nSample demonstration {sample_idx}:")
        print(f"  Type: {type(sample_demo)}")
        if hasattr(sample_demo, 'scm'):
            print(f"  SCM: {sample_demo.scm}")
        if hasattr(sample_demo, 'target_variable'):
            print(f"  Target variable: {sample_demo.target_variable}")

if __name__ == "__main__":
    analyze_x4_examples()