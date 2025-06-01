"""
Parent set enumeration utilities.
"""

import math
from typing import List, FrozenSet
from itertools import combinations


def compute_adaptive_k(n_variables: int, max_parent_size: int = 3) -> int:
    """
    Compute principled number of parent sets to predict.
    
    Args:
        n_variables: Total number of variables in graph
        max_parent_size: Maximum realistic parent set size
        
    Returns:
        Number of parent sets to predict
    """
    # Count possible parent sets up to max_parent_size
    from math import comb
    
    total_possible = sum(
        comb(n_variables - 1, i) 
        for i in range(min(max_parent_size + 1, n_variables))
    )
    
    # Adaptive scaling: min(all possible, practical limit based on graph size)
    practical_limit = min(max(5, int(math.sqrt(n_variables) * 2)), 20)
    
    return min(total_possible, practical_limit)


def enumerate_possible_parent_sets(
    variables: List[str], 
    target_variable: str,
    max_parent_size: int = 3
) -> List[FrozenSet[str]]:
    """
    Enumerate all possible parent sets for a target variable.
    
    Args:
        variables: All variable names in the graph
        target_variable: Target variable name
        max_parent_size: Maximum parent set size to consider
        
    Returns:
        List of all possible parent sets (as frozensets) in deterministic order
    """
    # Get potential parents (all variables except target)
    potential_parents = [v for v in variables if v != target_variable]
    
    # Generate all combinations up to max_parent_size
    parent_sets = []
    
    for size in range(min(max_parent_size + 1, len(potential_parents) + 1)):
        for parent_combo in combinations(potential_parents, size):
            parent_sets.append(frozenset(parent_combo))
    
    # Sort deterministically by size first, then lexicographically
    # This removes bias while maintaining reproducibility
    parent_sets.sort(key=lambda ps: (len(ps), sorted(list(ps))))
    
    return parent_sets
