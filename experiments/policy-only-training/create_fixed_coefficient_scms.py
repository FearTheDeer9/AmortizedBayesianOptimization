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
        
        # Create SCM using parent's structure generation
        scm = self.create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type,
            target_variable=target_variable,
            edge_density=edge_density
        )
        
        # Adjust target variable range if needed
        if not self.use_target_bounds:
            # Get the target variable
            target = scm.get('target')
            if target:
                # Get current metadata including variable ranges
                metadata = scm.get('metadata', pyr.pmap({}))
                variable_ranges = dict(metadata.get('variable_ranges', {}))
                
                if variable_ranges and target in variable_ranges:
                    # Set target range to unbounded so it won't be clipped
                    # Using very large values instead of inf for better numerical stability
                    variable_ranges[target] = (-1000.0, 1000.0)
                    
                    # Update metadata with new ranges
                    metadata = metadata.update({'variable_ranges': variable_ranges})
                    scm = scm.update({'metadata': metadata})
        
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