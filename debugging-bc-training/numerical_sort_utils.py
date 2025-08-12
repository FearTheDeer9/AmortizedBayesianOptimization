#!/usr/bin/env python3
"""
Numerical sorting utilities to fix variable ordering issues.

This module provides functions to sort variables numerically rather than
alphabetically, ensuring X2 comes before X10, X11, etc.
"""

import re
from typing import List, Tuple, Any


def numerical_sort_key(variable_name: str) -> Tuple[int, int, str]:
    """
    Generate a sort key for numerical sorting of variable names.
    
    Sorting order:
    1. X variables sorted numerically (X0, X1, X2, ..., X10, X11)
    2. Y variable (target)
    3. Other variables alphabetically
    
    Args:
        variable_name: Variable name like 'X0', 'X10', 'Y', etc.
        
    Returns:
        Tuple for sorting: (category, number, name)
        - category: 0 for X variables, 1 for Y, 2 for others
        - number: extracted number for X variables, 0 otherwise
        - name: original name for alphabetical sorting of others
    """
    # Match X variables with numbers
    x_match = re.match(r'^X(\d+)$', variable_name)
    if x_match:
        return (0, int(x_match.group(1)), variable_name)
    
    # Y variable comes after all X variables
    if variable_name == 'Y':
        return (1, 0, variable_name)
    
    # Other variables come last, sorted alphabetically
    return (2, 0, variable_name)


def numerical_sort_variables(variables: List[str]) -> List[str]:
    """
    Sort variables numerically instead of alphabetically.
    
    This ensures X2 comes before X10, fixing the index misalignment issue
    that occurs with alphabetical sorting.
    
    Args:
        variables: List of variable names
        
    Returns:
        Numerically sorted list of variables
        
    Examples:
        >>> numerical_sort_variables(['X0', 'X10', 'X2', 'Y', 'X1'])
        ['X0', 'X1', 'X2', 'X10', 'Y']
        
        >>> numerical_sort_variables(['X0', 'X1', 'X11', 'X2', 'X3', 'Y'])
        ['X0', 'X1', 'X2', 'X3', 'X11', 'Y']
    """
    return sorted(variables, key=numerical_sort_key)


def compare_sorting_methods(variables: List[str]) -> None:
    """
    Compare alphabetical vs numerical sorting for debugging.
    
    Args:
        variables: List of variable names to compare
    """
    alphabetical = sorted(variables)
    numerical = numerical_sort_variables(variables)
    
    print(f"Original:     {variables}")
    print(f"Alphabetical: {alphabetical}")
    print(f"Numerical:    {numerical}")
    
    if alphabetical != numerical:
        print("\n⚠️ Sorting methods differ!")
        print("Differences:")
        for i, (alpha, num) in enumerate(zip(alphabetical, numerical)):
            if alpha != num:
                print(f"  Position {i}: {alpha} (alpha) vs {num} (numerical)")
                
        # Check specific variables
        for var in ['X2', 'X3', 'X4', 'X10', 'X11']:
            if var in variables:
                alpha_idx = alphabetical.index(var)
                num_idx = numerical.index(var)
                if alpha_idx != num_idx:
                    print(f"\n{var}: index {alpha_idx} (alpha) → {num_idx} (numerical)")


def test_numerical_sorting():
    """Test the numerical sorting with various examples."""
    test_cases = [
        # Small SCM (no difference expected)
        ['X0', 'X1', 'X2', 'Y'],
        
        # Medium SCM (no difference expected)
        ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'Y'],
        
        # Large SCM with X10 (difference expected)
        ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'Y'],
        
        # Large SCM with X10 and X11 (major differences)
        ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'Y'],
        
        # Edge case: unsorted input
        ['Y', 'X5', 'X10', 'X1', 'X2', 'X11', 'X0'],
    ]
    
    print("="*60)
    print("NUMERICAL SORTING TESTS")
    print("="*60)
    
    for i, variables in enumerate(test_cases, 1):
        print(f"\nTest Case {i} ({len(variables)} variables):")
        print("-" * 40)
        compare_sorting_methods(variables)


if __name__ == "__main__":
    test_numerical_sorting()
    
    print("\n" + "="*60)
    print("IMPACT ON X4 INDEXING")
    print("="*60)
    
    # Show how X4 indexing changes with different SCM sizes
    scm_sizes = [
        (['X0', 'X1', 'X2', 'X3', 'Y'], "5-var SCM"),
        (['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'], "8-var SCM"),
        (['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'Y'], "11-var SCM"),
        (['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'Y'], "12-var SCM"),
        (['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'Y'], "13-var SCM"),
    ]
    
    for variables, label in scm_sizes:
        alpha_sorted = sorted(variables)
        num_sorted = numerical_sort_variables(variables)
        
        print(f"\n{label}:")
        if 'X4' in variables:
            alpha_idx = alpha_sorted.index('X4')
            num_idx = num_sorted.index('X4')
            print(f"  X4 index: {alpha_idx} (alphabetical) → {num_idx} (numerical)")
            if alpha_idx != num_idx:
                print(f"  ✓ Fixed: X4 now consistently at index {num_idx}")
        else:
            print(f"  X4 not present")