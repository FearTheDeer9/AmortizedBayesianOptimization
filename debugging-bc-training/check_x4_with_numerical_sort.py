#!/usr/bin/env python3
"""
Check why X4 still doesn't appear even with numerical sorting.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path
from src.causal_bayes_opt.data_structures.scm import get_variables

# Import our fixed module
from demonstration_to_tensor_fixed import create_bc_training_dataset
from numerical_sort_utils import numerical_sort_variables

def check_x4_issue():
    print("="*60)
    print("CHECKING X4 WITH NUMERICAL SORTING")
    print("="*60)
    
    # Load a few demonstrations
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
    
    print(f"\nProcessing {len(flat_demos)} demonstrations")
    
    # Check what variables are in the SCMs
    print("\nChecking SCM variables:")
    print("-" * 40)
    
    scm_types = {}
    for i, demo in enumerate(flat_demos[:20]):
        if hasattr(demo, 'scm'):
            variables = list(get_variables(demo.scm))
            n_vars = len(variables)
            
            # Sort both ways
            alpha_sorted = sorted(variables)
            num_sorted = numerical_sort_variables(variables)
            
            if n_vars not in scm_types:
                scm_types[n_vars] = {
                    'alpha': alpha_sorted,
                    'numerical': num_sorted,
                    'count': 0
                }
            scm_types[n_vars]['count'] += 1
            
            # Check trajectory for X4
            if hasattr(demo, 'parent_posterior'):
                trajectory = demo.parent_posterior.get('trajectory', {})
                intervention_sequence = trajectory.get('intervention_sequence', [])
                
                for int_var in intervention_sequence:
                    if isinstance(int_var, tuple):
                        int_var = int_var[0]
                    if 'X4' in str(int_var):
                        print(f"  Demo {i}: Found X4 in trajectory! (SCM has {n_vars} vars)")
    
    print("\nSCM types found:")
    for n_vars, info in sorted(scm_types.items()):
        print(f"\n{n_vars} variables ({info['count']} demos):")
        print(f"  Alpha:     {info['alpha']}")
        print(f"  Numerical: {info['numerical']}")
        if info['alpha'] != info['numerical']:
            print("  ⚠️ Sorting differs!")
    
    # Create dataset with fixed sorting
    print("\n" + "="*60)
    print("CREATING DATASET WITH FIXED SORTING")
    print("="*60)
    
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    print(f"Created {len(all_inputs)} examples")
    
    # Check labels
    variable_counts = {}
    for label in all_labels:
        if 'targets' in label and label['targets']:
            target_var = list(label['targets'])[0]
            variable_counts[target_var] = variable_counts.get(target_var, 0) + 1
    
    print("\nVariable distribution:")
    for var in sorted(variable_counts.keys()):
        count = variable_counts[var]
        print(f"  {var}: {count}")
    
    # Specific check for numbered variables
    print("\nNumbered X variables:")
    for var in sorted(variable_counts.keys()):
        if var.startswith('X') and var[1:].isdigit():
            idx = int(var[1:])
            if idx >= 4:
                print(f"  {var}: {variable_counts[var]} (index would be {idx})")

if __name__ == "__main__":
    check_x4_issue()