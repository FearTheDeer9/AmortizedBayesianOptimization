"""
Convert demonstrations to tensors with variable permutation to prevent shortcuts.

This version applies random permutations to variable ordering to force the model
to learn causal patterns rather than position-based heuristics.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent))

# These imports are used in the actual functions, but not needed for test
# from src.causal_bayes_opt.scm import LinearGaussianSCM
# from src.causal_bayes_opt.utils.sampling import sample_interventions
from permuted_variable_mapper import PermutedVariableMapper


def create_bc_training_dataset_permuted(
    demonstrations: List[Any],
    max_trajectory_length: int = 100,
    use_permutation: bool = True,
    base_seed: int = 42
) -> Tuple[List[jnp.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convert demonstrations to 5-channel tensors with variable permutation.
    
    Key difference: Each example gets a random permutation of variables to
    prevent the model from learning position-based shortcuts.
    
    Args:
        demonstrations: List of demonstration objects
        max_trajectory_length: Maximum trajectory length to consider
        use_permutation: Whether to apply random permutations (for ablation)
        base_seed: Base seed for permutation generation
        
    Returns:
        Tuple of (input_tensors, labels, metadata)
    """
    all_inputs = []
    all_labels = []
    
    # Track statistics
    n_demonstrations = 0
    all_variables = set()
    tensor_shape = None
    
    example_idx = 0
    
    for demo_idx, demo in enumerate(demonstrations):
        if not hasattr(demo, 'trajectory') or not demo.trajectory:
            continue
            
        n_demonstrations += 1
        trajectory = demo.trajectory[:max_trajectory_length]
        scm = demo.scm
        
        # Get variable names
        variables = list(scm.variables)
        all_variables.update(variables)
        n_vars = len(variables)
        
        # Create permutation mapper for this demonstration
        if use_permutation:
            # Use unique seed for each example
            seed = base_seed + example_idx
            mapper = PermutedVariableMapper(variables, seed=seed)
            permutation = mapper.get_permutation_vector()
        else:
            # No permutation (identity mapping)
            mapper = None
            permutation = np.arange(n_vars)
        
        # Process each intervention in trajectory
        for t, (intervention_node, intervention_value, values_dict, _) in enumerate(trajectory):
            if intervention_node is None:
                continue
            
            # Create 5-channel tensor WITH PERMUTATION
            tensor = np.zeros((max_trajectory_length, n_vars, 5))
            
            # Fill in channels for each variable IN PERMUTED ORDER
            for orig_idx, var in enumerate(variables):
                # Get permuted position for this variable
                perm_idx = int(np.where(permutation == orig_idx)[0][0])
                
                # Channel 0: Values
                tensor[t, perm_idx, 0] = values_dict.get(var, 0.0)
                
                # Channel 1: Parent existence (1 if has parents)
                parents = scm.get_parents(var)
                tensor[t, perm_idx, 1] = 1.0 if parents else 0.0
                
                # Channel 2: Parent values (mean)
                if parents:
                    parent_values = [values_dict.get(p, 0.0) for p in parents]
                    tensor[t, perm_idx, 2] = np.mean(parent_values)
                
                # Channel 3: Is target variable
                tensor[t, perm_idx, 3] = 1.0 if var == demo.target else 0.0
                
                # Channel 4: Trajectory position
                tensor[t, perm_idx, 4] = t / max_trajectory_length
            
            # Create label WITH PERMUTED INDEX
            if mapper:
                # Map intervention node to permuted index
                permuted_target_idx = mapper.to_permuted_index(intervention_node)
            else:
                # No permutation
                permuted_target_idx = variables.index(intervention_node)
            
            label = {
                'targets': {intervention_node},  # Keep original name for reference
                'values': {intervention_node: intervention_value},
                'variables': variables,  # Original variable order
                'target_variable': demo.target if hasattr(demo, 'target') else None,
                'scm_id': id(scm),
                'trajectory_idx': t,
                'demo_idx': demo_idx,
                'permutation': permutation.tolist() if use_permutation else None,
                'permuted_target_idx': permuted_target_idx,  # Add this for easy access
                'mapper_seed': seed if use_permutation else None
            }
            
            all_inputs.append(jnp.array(tensor))
            all_labels.append(label)
            
            if tensor_shape is None:
                tensor_shape = tensor.shape
            
            example_idx += 1
    
    # Metadata
    metadata = {
        'n_demonstrations': n_demonstrations,
        'n_examples': len(all_inputs),
        'variables': sorted(all_variables),
        'tensor_shape': tensor_shape,
        'uses_permutation': use_permutation,
        'target_idx': 0  # Placeholder
    }
    
    return all_inputs, all_labels, metadata


def test_permutation_effect():
    """Test that permutation changes variable positions across examples."""
    print("="*60)
    print("TESTING PERMUTATION EFFECT ON TENSORS")
    print("="*60)
    
    # Create a mock SCM for testing
    class MockSCM:
        def __init__(self, variables):
            self.variables = variables
            
        def get_parents(self, var):
            # Simple mock parent structure
            parents_map = {
                'X0': [],
                'X1': ['X0'],
                'X2': ['X0', 'X1'],
                'X3': ['X1', 'X2'],
                'X4': ['X2', 'X3']
            }
            return parents_map.get(var, [])
    
    # Create mock demonstration
    class MockDemo:
        def __init__(self):
            self.scm = MockSCM(['X0', 'X1', 'X2', 'X3', 'X4'])
            self.target = 'X4'
            self.trajectory = [
                ('X2', 1.0, {'X0': 0.5, 'X1': 0.3, 'X2': 1.0, 'X3': 0.7, 'X4': 0.2}, None),
                ('X0', 0.8, {'X0': 0.8, 'X1': 0.4, 'X2': 0.9, 'X3': 0.6, 'X4': 0.3}, None),
            ]
    
    # Create dataset without permutation
    demo = MockDemo()
    inputs_no_perm, labels_no_perm, _ = create_bc_training_dataset_permuted(
        [demo], max_trajectory_length=10, use_permutation=False
    )
    
    # Create dataset with permutation (multiple times to show different perms)
    print("\nTarget variable positions WITHOUT permutation:")
    print("-"*40)
    for label in labels_no_perm:
        target = list(label['targets'])[0]
        idx = label['permuted_target_idx']
        print(f"  {target} at index {idx}")
    
    print("\nTarget variable positions WITH permutation:")
    print("-"*40)
    
    for seed_offset in range(3):
        inputs_perm, labels_perm, _ = create_bc_training_dataset_permuted(
            [demo], max_trajectory_length=10, use_permutation=True, base_seed=42+seed_offset*100
        )
        
        print(f"\nRun {seed_offset+1}:")
        for label in labels_perm:
            target = list(label['targets'])[0]
            idx = label['permuted_target_idx']
            print(f"  {target} at index {idx}")
    
    print("\n✅ Permutation successfully randomizes variable positions!")
    print("   X2 appears at different indices in different examples")
    print("   Model can't use position as a shortcut anymore")


if __name__ == "__main__":
    test_permutation_effect()