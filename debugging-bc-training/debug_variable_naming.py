#!/usr/bin/env python3
"""
Debug why X4 becomes X6/X8 in the data pipeline.
Trace the variable naming from raw demonstrations to tensors.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.data_preprocessing import load_demonstrations_from_path

def trace_variable_naming():
    """Trace how variables are named throughout the pipeline."""
    
    print("="*80)
    print("VARIABLE NAMING INVESTIGATION")
    print("="*80)
    
    # Load raw demonstrations
    demos_path = 'expert_demonstrations/raw/raw_demonstrations'
    raw_demos = load_demonstrations_from_path(demos_path, max_files=20)
    
    print(f"\nLoaded {len(raw_demos)} demonstration files")
    
    # Examine raw demonstration structure
    variable_sets = []
    scm_types = defaultdict(list)
    
    for demo_idx, demo_item in enumerate(raw_demos):
        print(f"\n--- Demonstration File {demo_idx} ---")
        print(f"Type: {type(demo_item)}")
        
        if hasattr(demo_item, 'demonstrations'):
            # This is a collection of demonstrations
            print(f"Contains {len(demo_item.demonstrations)} demonstrations")
            
            for sub_idx, demo in enumerate(demo_item.demonstrations[:3]):  # First 3
                print(f"\n  Sub-demo {sub_idx}:")
                print(f"    Type: {type(demo)}")
                
                # Check SCM info
                if hasattr(demo, 'scm'):
                    scm = demo.scm
                    print(f"    SCM type: {scm}")
                    scm_types[str(scm)].append(demo_idx)
                    
                    # Check variables in SCM
                    if hasattr(scm, 'variables'):
                        vars = scm.variables
                        print(f"    SCM variables: {vars}")
                        variable_sets.append(set(vars))
                    elif hasattr(scm, 'get_variables'):
                        vars = scm.get_variables()
                        print(f"    SCM variables (from method): {vars}")
                        variable_sets.append(set(vars))
                    elif hasattr(scm, 'n_vars'):
                        n = scm.n_vars
                        print(f"    SCM has {n} variables")
                        # Check if variables are named or indexed
                        if hasattr(scm, 'variable_names'):
                            print(f"    Variable names: {scm.variable_names}")
                
                # Check intervention targets
                if hasattr(demo, 'trajectory'):
                    print(f"    Has trajectory with {len(demo.trajectory)} steps")
                    # Sample first step
                    if demo.trajectory:
                        first_step = demo.trajectory[0]
                        if hasattr(first_step, 'targets'):
                            print(f"    First step targets: {first_step.targets}")
                        if hasattr(first_step, 'intervention'):
                            print(f"    First step intervention: {first_step.intervention}")
                
                # Check target variable
                if hasattr(demo, 'target_variable'):
                    print(f"    Target variable: {demo.target_variable}")
                    # THIS IS KEY: The target variable might be using different indexing!
                    if isinstance(demo.target_variable, str):
                        if 'Y' in demo.target_variable:
                            print(f"    Target is Y (outcome variable)")
                        else:
                            # Parse the index
                            if demo.target_variable.startswith('X'):
                                idx = demo.target_variable[1:]
                                print(f"    Target variable index: {idx}")
    
    # Analyze variable naming patterns
    print("\n" + "="*60)
    print("VARIABLE NAMING PATTERNS")
    print("="*60)
    
    if variable_sets:
        print("\nUnique variable sets found:")
        unique_sets = list(set(frozenset(s) for s in variable_sets))
        for i, var_set in enumerate(unique_sets):
            print(f"  Set {i+1}: {sorted(var_set)}")
    
    print("\nSCM type distribution:")
    for scm_type, indices in scm_types.items():
        print(f"  {scm_type}: appears in demos {indices[:5]}...")  # First 5
    
    # Now let's look at how demonstration_to_tensor processes these
    print("\n" + "="*60)
    print("DEMONSTRATION TO TENSOR CONVERSION")
    print("="*60)
    
    from src.causal_bayes_opt.training.demonstration_to_tensor import create_bc_training_dataset
    
    # Flatten demonstrations
    flat_demos = []
    for item in raw_demos[:10]:  # Just first 10 files
        if hasattr(item, 'demonstrations'):
            flat_demos.extend(item.demonstrations[:5])  # First 5 from each
        else:
            flat_demos.append(item)
    
    print(f"\nProcessing {len(flat_demos)} demonstrations")
    
    # Convert to tensors and examine the labels
    all_inputs, all_labels, metadata = create_bc_training_dataset(
        flat_demos, max_trajectory_length=100
    )
    
    print(f"Created {len(all_inputs)} training examples")
    
    # Examine the labels to see how variables are named
    target_variable_names = defaultdict(int)
    intervention_variable_names = defaultdict(int)
    variable_lists = []
    
    for label in all_labels[:50]:  # First 50 examples
        # Check target variable (Y variable)
        if 'target_variable' in label:
            target_variable_names[label['target_variable']] += 1
        
        # Check intervention targets
        if 'targets' in label and label['targets']:
            for target in label['targets']:
                intervention_variable_names[target] += 1
        
        # Check variable list
        if 'variables' in label:
            variable_lists.append(label['variables'])
    
    print("\nTarget variables (Y):")
    for var, count in target_variable_names.items():
        print(f"  {var}: {count}")
    
    print("\nIntervention variables:")
    for var, count in sorted(intervention_variable_names.items()):
        print(f"  {var}: {count}")
    
    print("\nSample variable lists from labels:")
    for i, vars in enumerate(variable_lists[:5]):
        print(f"  Example {i+1}: {vars}")
    
    # Check metadata
    print("\nDataset metadata:")
    for key, value in metadata.items():
        if not isinstance(value, (list, dict)) or len(str(value)) < 100:
            print(f"  {key}: {value}")
    
    # The smoking gun: Check how variables are indexed
    print("\n" + "="*60)
    print("VARIABLE INDEXING ANALYSIS")
    print("="*60)
    
    # Check if there's a pattern where X4 becomes X6 or X8
    for label_idx, label in enumerate(all_labels[:20]):
        if 'variables' in label:
            vars = label['variables']
            if len(vars) > 5 or any(v in vars for v in ['X6', 'X8']):
                print(f"\nExample {label_idx} has unusual variables: {vars}")
                
                # Check what the target was
                if 'targets' in label:
                    print(f"  Targets: {label['targets']}")
                if 'target_variable' in label:
                    print(f"  Target variable (Y): {label['target_variable']}")
                
                # THIS IS THE KEY: Are we using 0-indexed or 1-indexed variables?
                # Or are we accidentally adding the target variable to the count?

if __name__ == "__main__":
    trace_variable_naming()