"""
Graph utility functions for PARENT_SCALE - proper implementation
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Union
import itertools


def create_grid_interventions(
    intervention_range: Dict[str, Tuple[float, float]], 
    get_list_format: bool = True,
    num_points: int = 10
) -> Dict[Tuple[str, ...], List[List[float]]]:
    """
    Create grid of intervention points for variables.
    
    Args:
        intervention_range: Dict mapping variable names to (min, max) ranges
        get_list_format: Whether to return in list format
        num_points: Number of grid points per dimension
        
    Returns:
        Dictionary mapping intervention sets to grid points
    """
    intervention_grid = {}
    variables = list(intervention_range.keys())
    
    # Create grids for individual variables
    for var in variables:
        min_val, max_val = intervention_range[var]
        grid_points = np.linspace(min_val, max_val, num_points)
        
        # Format as list of single-element lists for consistency
        intervention_grid[(var,)] = [[point] for point in grid_points]
    
    # Create grids for pairs of variables (if needed)
    for var1, var2 in itertools.combinations(variables, 2):
        min_val1, max_val1 = intervention_range[var1]
        min_val2, max_val2 = intervention_range[var2]
        
        grid_points1 = np.linspace(min_val1, max_val1, num_points)
        grid_points2 = np.linspace(min_val2, max_val2, num_points)
        
        # Create all combinations
        grid_combinations = []
        for p1 in grid_points1:
            for p2 in grid_points2:
                grid_combinations.append([p1, p2])
        
        intervention_grid[(var1, var2)] = grid_combinations
    
    return intervention_grid