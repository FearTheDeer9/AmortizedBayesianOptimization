#!/usr/bin/env python3
"""
Create SCMs with fixed coefficients for convergence testing.

This module extends VariableSCMFactory to create SCMs with deterministic
coefficients while preserving the graph structure generation logic.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pyrsistent as pyr

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory


class FixedCoefficientSCMFactory(VariableSCMFactory):
    """
    Factory for generating SCMs with fixed coefficients and zero noise.
    
    Extends VariableSCMFactory to override coefficient generation with 
    deterministic values while preserving graph structure logic.
    """
    
    def __init__(self,
                 coefficient_values: Optional[List[float]] = None,
                 noise_scale: float = 0.0,  # Zero noise by default
                 intervention_range: Tuple[float, float] = (-5.0, 5.0),
                 use_target_bounds: bool = False,
                 target_range_multiplier: float = 10.0,
                 seed: int = 42):
        """
        Initialize factory with fixed coefficient values.
        
        Args:
            coefficient_values: List of coefficient values to cycle through
                              Default: [2.0, 1.5, 1.0, -1.5, -1.0, 0.5]
            noise_scale: Noise level (0 for deterministic)
            intervention_range: Range for all intervention values
            use_target_bounds: Whether to restrict target variable range
            target_range_multiplier: Multiplier for target range if unrestricted
            seed: Random seed for structure generation
        """
        # Initialize parent with zero noise and fixed ranges
        super().__init__(
            noise_scale=noise_scale,
            coefficient_range=(-3.0, 3.0),  # Not used, but needed for parent
            intervention_range=intervention_range,
            vary_intervention_ranges=False,  # Keep ranges consistent
            use_output_bounds=False,  # Disable automatic output bounds
            seed=seed
        )
        
        # Set up fixed coefficient values
        if coefficient_values is None:
            # Default nice values
            self.coefficient_values = [2.0, 1.5, 1.0, -1.5, -1.0, 0.5]
        else:
            self.coefficient_values = coefficient_values
        
        self.coefficient_index = 0
        self.use_target_bounds = use_target_bounds
        self.target_range_multiplier = target_range_multiplier
    
    def _generate_coefficient(self) -> float:
        """
        Override to return fixed coefficient values in order.
        Cycles through the list of coefficient values.
        """
        # Get next coefficient from the list
        coeff = self.coefficient_values[self.coefficient_index % len(self.coefficient_values)]
        self.coefficient_index += 1
        return coeff
    
    def reset_coefficient_index(self):
        """Reset coefficient index for reproducible generation."""
        self.coefficient_index = 0
    
    def create_variable_scm(self,
                           num_variables: int,
                           structure_type: str = "mixed",
                           target_variable: Optional[str] = None,
                           edge_density: float = 0.5) -> pyr.PMap:
        """
        Override parent's create_variable_scm to handle target ranges properly.
        
        This ensures the target variable gets the expanded range BEFORE mechanism creation,
        while keeping the parent's intervention_range for non-target variables.
        """
        # Call parent's method to create the SCM structure
        # Keep intervention_range so parents get consistent ranges
        scm = super().create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type,
            target_variable=target_variable,
            edge_density=edge_density
        )
        
        # Now rebuild the SCM with expanded target range only
        if not self.use_target_bounds and self.intervention_range:
            from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
            from src.causal_bayes_opt.data_structures.scm import get_variables, get_edges
            
            # Extract structure from created SCM
            variables = list(get_variables(scm))
            edges = list(get_edges(scm))
            target = scm.get('target')
            metadata = scm.get('metadata', pyr.pmap({}))
            
            # Get coefficients from metadata
            coefficients = metadata.get('coefficients', {})
            
            # Get existing variable ranges from metadata
            existing_ranges = metadata.get('variable_ranges', {})
            
            # Set up variable ranges: keep existing for parents, expand for target
            variable_ranges = {}
            for var in variables:
                if var == target:
                    # Expanded range for target only
                    variable_ranges[var] = (-1000.0, 1000.0)
                else:
                    # Keep the existing range (should be intervention_range)
                    variable_ranges[var] = existing_ranges.get(var, self.intervention_range)
            
            # Set noise scales (zero for deterministic)
            noise_scales = {var: self.noise_scale for var in variables}
            
            # Recreate the SCM with proper ranges
            scm = create_simple_linear_scm(
                variables=variables,
                edges=edges,
                coefficients=coefficients,
                noise_scales=noise_scales,
                target=target,
                variable_ranges=variable_ranges,
                output_bounds=None
            )
            
            # Update metadata with correct ranges
            updated_metadata = metadata.update({'variable_ranges': variable_ranges})
            scm = scm.update({'metadata': updated_metadata})
        
        return scm
    
    def create_fixed_scm(self,
                        num_variables: int,
                        structure_type: str,
                        target_variable: Optional[str] = None,
                        edge_density: float = 0.5,
                        coefficient_pattern: Optional[str] = None) -> Tuple[pyr.PMap, str]:
        """
        Create SCM with fixed coefficients.
        
        Args:
            num_variables: Number of variables
            structure_type: Type of structure ('fork', 'chain', 'scale_free', etc.)
            target_variable: Optional target specification
            edge_density: For random structures
            coefficient_pattern: Pattern for coefficients ('decreasing', 'alternating', 'custom')
        
        Returns:
            Tuple of (SCM, name)
        """
        # Reset coefficient index for reproducibility
        self.reset_coefficient_index()
        
        # Apply coefficient pattern if specified
        if coefficient_pattern == 'decreasing':
            # Coefficients decrease with distance from source
            self.coefficient_values = [2.0, 1.5, 1.0, 0.5, 0.3]
        elif coefficient_pattern == 'alternating':
            # Alternating positive/negative
            self.coefficient_values = [2.0, -1.5, 1.0, -0.5]
        elif coefficient_pattern == 'strong':
            # All strong coefficients
            self.coefficient_values = [2.0, 2.0, 2.0, 2.0]
        elif coefficient_pattern == 'mixed':
            # Mix of strong and weak
            self.coefficient_values = [2.0, 0.5, 1.5, 0.3, 1.0]
        # else use default or provided values
        
        # Create SCM using our overridden method that handles ranges properly
        scm = self.create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type,
            target_variable=target_variable,
            edge_density=edge_density
        )
        
        # Update metadata to indicate fixed coefficients
        metadata = scm.get('metadata', pyr.pmap({}))
        metadata = metadata.update({
            'fixed_coefficients': True,
            'coefficient_pattern': coefficient_pattern or 'custom',
            'noise_scale': self.noise_scale,
            'deterministic': self.noise_scale == 0.0,
            'target_unbounded': not self.use_target_bounds,
            'target_range_multiplier': self.target_range_multiplier if not self.use_target_bounds else 1.0
        })
        scm = scm.update({'metadata': metadata})
        
        # Generate name
        name = f"fixed_{structure_type}_{num_variables}var"
        
        return scm, name


def create_standard_fixed_scms(num_variables: int = 4) -> Dict[str, pyr.PMap]:
    """
    Create a standard set of fixed SCMs for testing.
    
    Args:
        num_variables: Number of variables in each SCM
        
    Returns:
        Dictionary mapping names to SCMs
    """
    factory = FixedCoefficientSCMFactory(
        coefficient_values=[2.0, 1.5, 1.0, -1.5],
        noise_scale=0.0,
        intervention_range=(-5.0, 5.0),
        use_target_bounds=False,  # Allow target to reach full range
        target_range_multiplier=10.0
    )
    
    scms = {}
    
    # Standard structures
    structures = ['fork', 'true_fork', 'chain', 'collider', 'mixed', 'two_layer']
    
    for structure in structures:
        scm, name = factory.create_fixed_scm(
            num_variables=num_variables,
            structure_type=structure
        )
        scms[name] = scm
    
    # Add scale-free with more variables
    if num_variables >= 6:
        scm, name = factory.create_fixed_scm(
            num_variables=num_variables,
            structure_type='scale_free',
            coefficient_pattern='decreasing'
        )
        scms[name] = scm
    
    # Add random DAG
    scm, name = factory.create_fixed_scm(
        num_variables=num_variables,
        structure_type='random',
        edge_density=0.3,
        coefficient_pattern='alternating'
    )
    scms[name] = scm
    
    return scms


def create_large_scale_free(num_variables: int = 50) -> Tuple[pyr.PMap, str]:
    """
    Create a large scale-free network with fixed coefficients.
    
    Args:
        num_variables: Number of variables (default 50)
        
    Returns:
        Tuple of (SCM, name)
    """
    factory = FixedCoefficientSCMFactory(
        # Coefficients decrease with distance from hubs
        coefficient_values=[2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.5, 0.3],
        noise_scale=0.0,
        intervention_range=(-5.0, 5.0),
        use_target_bounds=False,  # Allow target to reach full range
        target_range_multiplier=10.0,
        seed=42
    )
    
    scm, name = factory.create_fixed_scm(
        num_variables=num_variables,
        structure_type='scale_free',
        coefficient_pattern='decreasing'
    )
    
    return scm, name


def create_star_graph_scm(num_nodes: int, max_coefficient: float, seed: int = 42) -> Tuple[pyr.PMap, str]:
    """
    Create a star graph SCM with equally spaced coefficients.
    
    In a star graph, the target variable is the hub connected to all other nodes,
    which are not connected to each other. This creates a pure collider structure
    where all parents directly influence the target.
    
    Args:
        num_nodes: Total number of nodes (including target)
        max_coefficient: Maximum coefficient value (L)
        seed: Random seed for consistency
        
    Returns:
        Tuple of (SCM, name)
    """
    from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
    
    # Create equally spaced coefficients from 0 to max_coefficient
    num_parents = num_nodes - 1
    if num_parents == 1:
        coefficients_list = [max_coefficient]
    else:
        # Equally spaced including endpoints
        coefficients_list = [
            i * max_coefficient / (num_parents - 1) 
            for i in range(num_parents)
        ]
    
    # Variable names
    variables = [f'X{i}' for i in range(num_nodes)]
    target = variables[-1]  # Last variable is target
    parent_vars = variables[:-1]
    
    # Create edges (all parents point to target)
    edges = [(parent, target) for parent in parent_vars]
    
    # Create coefficient dictionary with parent-target pairs
    coefficients = {}
    for i, (parent, tgt) in enumerate(edges):
        coefficients[(parent, tgt)] = coefficients_list[i]
    
    # Zero noise for deterministic behavior
    noise_scales = {var: 0.0 for var in variables}
    
    # Variable ranges: expanded for all to allow proper testing of coefficient magnitudes
    # Parent variables need wider ranges for interventions to explore full effect space
    variable_ranges = {}
    for var in variables:
        if var == target:
            variable_ranges[var] = (-1000.0, 1000.0)  # Expanded range for target
        else:
            variable_ranges[var] = (-10.0, 10.0)  # Wider range for parents to test full coefficient effects
    
    # Create the SCM
    scm = create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales=noise_scales,
        target=target,
        variable_ranges=variable_ranges,
        output_bounds=None
    )
    
    # Add comprehensive metadata
    metadata = scm.get('metadata', pyr.pmap({}))
    metadata = metadata.update({
        'structure_type': 'star',
        'num_variables': num_nodes,
        'num_edges': len(edges),
        'num_parents': num_parents,
        'target_variable': target,
        'coefficients': coefficients,
        'coefficient_spacing': 'equally_spaced',
        'max_coefficient': max_coefficient,
        'optimal_parent': parent_vars[-1],  # Highest coefficient parent
        'variable_ranges': variable_ranges,
        'description': f'Star graph: {num_parents} parents → {target}',
        'deterministic': True,
        'noise_scale': 0.0
    })
    
    scm = scm.update({'metadata': metadata})
    name = f'star_{num_nodes}nodes_L{max_coefficient}'
    
    return scm, name


def test_fixed_factory():
    """Test the fixed coefficient factory."""
    print("Testing Fixed Coefficient SCM Factory")
    print("=" * 60)
    
    # Test small fork
    factory = FixedCoefficientSCMFactory()
    scm, name = factory.create_fixed_scm(
        num_variables=4,
        structure_type='fork'
    )
    
    print(f"\nCreated: {name}")
    metadata = scm.get('metadata', {})
    print(f"Structure: {metadata.get('structure_type')}")
    print(f"Target: {scm.get('target')}")
    print(f"Coefficients: {metadata.get('coefficients')}")
    print(f"Deterministic: {metadata.get('deterministic')}")
    
    # Test large scale-free
    scm_large, name_large = create_large_scale_free(num_variables=20)
    metadata_large = scm_large.get('metadata', {})
    print(f"\nCreated: {name_large}")
    print(f"Number of variables: {metadata_large.get('num_variables')}")
    print(f"Number of edges: {metadata_large.get('num_edges')}")
    print(f"Edge density: {metadata_large.get('edge_density'):.3f}")
    
    # Test standard set
    standard_scms = create_standard_fixed_scms(num_variables=6)
    print(f"\nCreated standard set with {len(standard_scms)} SCMs:")
    for scm_name in standard_scms:
        print(f"  - {scm_name}")


if __name__ == "__main__":
    test_fixed_factory()