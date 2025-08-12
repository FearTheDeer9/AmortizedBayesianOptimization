#!/usr/bin/env python3
"""
Test variable mapping to understand why X4 becomes X6/X8.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def test_sorting_issue():
    """Test if sorting causes the issue."""
    
    print("="*60)
    print("VARIABLE SORTING TEST")
    print("="*60)
    
    # Simulate different SCM variable sets
    scm_variable_sets = [
        # 3 variables
        ['Y', 'X0', 'X1'],
        # 5 variables  
        ['Y', 'X0', 'X1', 'X2', 'X3'],
        # 8 variables
        ['Y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
        # 10 variables
        ['Y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'],
        # 12 variables
        ['Y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
    ]
    
    for var_set in scm_variable_sets:
        sorted_vars = sorted(var_set)
        print(f"\nOriginal ({len(var_set)} vars): {var_set}")
        print(f"Sorted: {sorted_vars}")
        
        # Check if X4 exists and where it ends up
        if 'X4' in var_set:
            orig_idx = var_set.index('X4')
            sorted_idx = sorted_vars.index('X4')
            print(f"  X4: position {orig_idx} → {sorted_idx}")
    
    # The issue: When sorting alphabetically, X10 comes before X2!
    print("\n" + "="*60)
    print("ALPHABETICAL SORTING ISSUE")
    print("="*60)
    
    test_vars = ['X0', 'X1', 'X10', 'X11', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y']
    sorted_test = sorted(test_vars)
    
    print(f"Unsorted: {test_vars}")
    print(f"Sorted:   {sorted_test}")
    print("\n⚠️ Notice: X10 and X11 come BEFORE X2 in alphabetical order!")
    
    # This could cause misalignment between SCMs with different numbers of variables
    print("\n" + "="*60)
    print("HYPOTHESIS")
    print("="*60)
    
    print("""
The issue appears to be:
1. Different SCMs have different numbers of variables
2. Variables are sorted alphabetically in demonstration_to_tensor.py
3. For SCMs with >10 variables, X10/X11 sort before X2-X9
4. This causes variable index misalignment

For example:
- 5-var SCM:  ['X0', 'X1', 'X2', 'X3', 'Y'] → X3 is at index 3
- 12-var SCM: ['X0', 'X1', 'X10', 'X11', 'X2', 'X3', ...] → X3 is at index 5!

This would explain why X4 appears as X6 or X8 in the data.
""")

if __name__ == "__main__":
    test_sorting_issue()