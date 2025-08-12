"""
Permuted Variable Mapper that randomizes variable positions to prevent
the model from learning size-based shortcuts.

Each training example gets a different random permutation, forcing the model
to learn actual causal patterns rather than position-based heuristics.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PermutedVariableMapper:
    """
    Variable mapper that applies random permutation to break position-based shortcuts.
    
    Key features:
    - Each example gets a unique random permutation
    - Maps between original and permuted variable orders
    - Ensures consistent forward/backward mapping within an example
    """
    
    def __init__(self, original_variables: List[str], seed: Optional[int] = None):
        """
        Initialize permuted variable mapper.
        
        Args:
            original_variables: Original variable names in standard order
            seed: Random seed for this example's permutation (should be unique per example)
        """
        # Sort variables numerically (X0, X1, X2, ..., X10, X11)
        self.original_variables = self._numerical_sort(original_variables)
        self.n_vars = len(self.original_variables)
        
        # Generate random permutation for this example
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Create permutation: permutation[i] tells us which original index goes to position i
        self.permutation = rng.permutation(self.n_vars)
        
        # Create inverse permutation for reverse mapping
        self.inverse_permutation = np.zeros(self.n_vars, dtype=int)
        for i, orig_idx in enumerate(self.permutation):
            self.inverse_permutation[orig_idx] = i
        
        # Create permuted variable list for debugging
        self.permuted_variables = [self.original_variables[i] for i in self.permutation]
        
        # Create name-to-index mappings
        self.original_to_idx = {var: i for i, var in enumerate(self.original_variables)}
        self.permuted_to_idx = {var: i for i, var in enumerate(self.permuted_variables)}
    
    def _numerical_sort(self, variables: List[str]) -> List[str]:
        """Sort variables numerically (X2 before X10)."""
        import re
        
        def sort_key(var: str) -> Tuple[int, int, str]:
            x_match = re.match(r'^X(\d+)', var)
            if x_match:
                return (0, int(x_match.group(1)), var)
            y_match = re.match(r'^Y(\d*)', var)
            if y_match:
                num = int(y_match.group(1)) if y_match.group(1) else 0
                return (1, num, var)
            return (2, 0, var)
        
        return sorted(variables, key=sort_key)
    
    def to_permuted_index(self, original_var_name: str) -> int:
        """
        Convert original variable name to its index in the permuted order.
        
        This is used when preparing training data - we need to know where
        the target variable appears in the permuted tensor.
        
        Args:
            original_var_name: Variable name in original order (e.g., "X2")
            
        Returns:
            Index in permuted order where this variable appears
        """
        # Get original index
        if original_var_name not in self.original_to_idx:
            raise ValueError(f"Variable {original_var_name} not in original variables: {self.original_variables}")
        
        orig_idx = self.original_to_idx[original_var_name]
        
        # Map to permuted position
        permuted_idx = self.inverse_permutation[orig_idx]
        
        return int(permuted_idx)
    
    def from_permuted_index(self, permuted_idx: int) -> str:
        """
        Convert model's prediction (index in permuted order) back to original variable name.
        
        This is used when computing loss - we need to know which original variable
        the model predicted.
        
        Args:
            permuted_idx: Index that model predicted (in permuted order)
            
        Returns:
            Original variable name
        """
        if permuted_idx < 0 or permuted_idx >= self.n_vars:
            raise ValueError(f"Invalid permuted index {permuted_idx}, must be in [0, {self.n_vars})")
        
        # Get original index from permutation
        orig_idx = self.permutation[permuted_idx]
        
        # Get variable name
        return self.original_variables[orig_idx]
    
    def get_permuted_order(self) -> List[str]:
        """Get the permuted variable order for this example."""
        return self.permuted_variables
    
    def get_original_order(self) -> List[str]:
        """Get the original variable order."""
        return self.original_variables
    
    def get_permutation_vector(self) -> np.ndarray:
        """Get the permutation vector for tensor reordering."""
        return self.permutation
    
    def describe_mapping(self) -> str:
        """Get human-readable description of the permutation."""
        mapping_strs = []
        for i in range(self.n_vars):
            orig_var = self.original_variables[i]
            perm_pos = self.inverse_permutation[i]
            mapping_strs.append(f"{orig_var}→pos{perm_pos}")
        return ", ".join(mapping_strs)


def test_permuted_mapper():
    """Test the permuted variable mapper."""
    print("="*60)
    print("TESTING PERMUTED VARIABLE MAPPER")
    print("="*60)
    
    # Test with 5 variables
    variables = ['X0', 'X1', 'X2', 'X3', 'X4']
    
    # Create multiple mappers with different seeds to show different permutations
    print("\nDifferent permutations for same variables:")
    print("-"*40)
    
    for seed in [42, 123, 456]:
        mapper = PermutedVariableMapper(variables, seed=seed)
        print(f"\nSeed {seed}:")
        print(f"  Original: {mapper.get_original_order()}")
        print(f"  Permuted: {mapper.get_permuted_order()}")
        print(f"  Mapping: {mapper.describe_mapping()}")
        
        # Test forward and backward mapping
        test_var = 'X2'
        perm_idx = mapper.to_permuted_index(test_var)
        back_var = mapper.from_permuted_index(perm_idx)
        print(f"  {test_var} → index {perm_idx} → {back_var} ✓")
    
    # Show how this breaks shortcuts
    print("\n" + "="*60)
    print("HOW THIS BREAKS SHORTCUTS")
    print("="*60)
    
    print("\nWithout permutation:")
    print("  X2 is ALWAYS at index 2")
    print("  Model learns: 'position 2 = X2'")
    
    print("\nWith permutation:")
    print("  Example 1: X2 at index 4")
    print("  Example 2: X2 at index 0")
    print("  Example 3: X2 at index 3")
    print("  Model CAN'T use position as shortcut!")
    
    print("\nThis forces the model to learn from:")
    print("  - Causal structure (parent relationships)")
    print("  - Node values and features")
    print("  - Actual intervention patterns")


if __name__ == "__main__":
    test_permuted_mapper()