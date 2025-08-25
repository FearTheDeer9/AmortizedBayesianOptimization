"""
Variable SCM Factory for Dynamic Size Generation

This module provides factory functions for generating SCMs with configurable
variable counts (3-8 variables) and structure types. Supports reproducible
generation for systematic evaluation across different SCM complexities.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Set
import jax.random as random
import jax.numpy as jnp
import pyrsistent as pyr

from .test_scms import create_simple_linear_scm
from ..data_structures.scm import create_scm, get_variables, get_target
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism

logger = logging.getLogger(__name__)


class VariableSCMFactory:
    """Factory for generating SCMs with configurable variable counts and structures."""
    
    STRUCTURE_TYPES = {
        "fork": "Common effect structure (multiple causes → one effect)",
        "chain": "Sequential causal chain (X1 → X2 → ... → Xn)",
        "collider": "Multiple causes with shared effect",
        "mixed": "Mixed structure with multiple patterns",
        "random": "Random DAG with specified edge density"
    }
    
    def __init__(self, 
                 noise_scale: float = 1.0,
                 coefficient_range: Tuple[float, float] = (-2.0, 2.0),
                 intervention_range: Optional[Tuple[float, float]] = None,
                 vary_intervention_ranges: bool = True,
                 use_output_bounds: bool = True,
                 seed: int = 42):
        """
        Initialize SCM factory.
        
        Args:
            noise_scale: Standard deviation for noise terms
            coefficient_range: Range for edge coefficients
            intervention_range: Default range for intervention values (if None, will vary)
            vary_intervention_ranges: If True, randomly vary ranges per node and SCM
            use_output_bounds: If True, apply size-dependent bounds to mechanism outputs
            seed: Random seed for reproducible generation
        """
        self.noise_scale = noise_scale
        self.coefficient_range = coefficient_range
        self.intervention_range = intervention_range
        self.vary_intervention_ranges = vary_intervention_ranges
        self.use_output_bounds = use_output_bounds
        self.seed = seed
        self.key = random.PRNGKey(seed)
    
    def create_variable_scm(self,
                           num_variables: int,
                           structure_type: str = "mixed",
                           target_variable: Optional[str] = None,
                           edge_density: float = 0.5) -> pyr.PMap:
        """
        Create SCM with specified number of variables and structure.
        
        Args:
            num_variables: Number of variables (3-8)
            structure_type: Type of causal structure
            target_variable: Specific target (None for automatic selection)
            edge_density: Density of edges (for random structures)
            
        Returns:
            Generated SCM with metadata
        """
        if num_variables < 3:
            raise ValueError(f"num_variables must be at least 3, got {num_variables}")
        
        if structure_type not in self.STRUCTURE_TYPES:
            raise ValueError(f"Unknown structure_type: {structure_type}")
        
        # Generate variable names
        variables = [f"X{i}" for i in range(num_variables)]
        
        # Select target variable
        if target_variable is None:
            target_variable = self._select_target(variables, structure_type)
        
        # Generate structure based on type
        if structure_type == "fork":
            edges, coefficients = self._create_fork_structure(variables, target_variable)
        elif structure_type == "chain":
            edges, coefficients = self._create_chain_structure(variables)
            target_variable = variables[-1]  # Chain target is always last
        elif structure_type == "collider":
            edges, coefficients = self._create_collider_structure(variables, target_variable)
        elif structure_type == "mixed":
            edges, coefficients = self._create_mixed_structure(variables, target_variable)
        elif structure_type == "random":
            edges, coefficients = self._create_random_structure(variables, target_variable, edge_density)
        else:
            raise ValueError(f"Structure type {structure_type} not implemented")
        
        # Create noise scales for all variables
        noise_scales = {var: self.noise_scale for var in variables}
        
        # Create UNIFIED variable ranges (for both interventions AND outputs)
        if self.vary_intervention_ranges:
            # Sample different ranges for each variable
            variable_ranges = {}
            for var in variables:
                # Sample range bounds
                self.key, subkey = random.split(self.key)
                # Sample max value between 1.0 and 5.0
                max_val = float(random.uniform(subkey, (), minval=1.0, maxval=5.0))
                # Always use asymmetric ranges to avoid learned biases
                self.key, subkey = random.split(self.key)
                if random.uniform(subkey, ()) < 0.5:  # 50% chance of each asymmetry
                    # Larger negative range
                    self.key, subkey = random.split(self.key)
                    min_val = -float(random.uniform(subkey, (), minval=max_val, maxval=max_val * 2))
                else:
                    # Larger positive range
                    self.key, subkey = random.split(self.key)
                    min_val = -float(random.uniform(subkey, (), minval=0.5, maxval=max_val))
                variable_ranges[var] = (min_val, max_val)
        elif self.intervention_range:
            # Use fixed range for all variables
            variable_ranges = {var: self.intervention_range for var in variables}
        else:
            # Default varying ranges (asymmetric)
            variable_ranges = {}
            for var in variables:
                self.key, subkey = random.split(self.key)
                max_val = float(random.uniform(subkey, (), minval=1.5, maxval=3.0))
                # Make asymmetric by default
                self.key, subkey = random.split(self.key)
                min_val = -float(random.uniform(subkey, (), minval=1.0, maxval=max_val * 1.5))
                variable_ranges[var] = (min_val, max_val)
        
        # If output bounds are enabled, expand ranges based on graph depth
        # This ensures we can still have meaningful dynamics
        if self.use_output_bounds:
            import math
            # Scale factor increases with graph size to allow for value propagation
            bound_scale = math.sqrt(num_variables / 5.0)
            
            # Expand variable ranges by this factor
            expanded_ranges = {}
            for var, (min_val, max_val) in variable_ranges.items():
                expanded_ranges[var] = (min_val * bound_scale, max_val * bound_scale)
            variable_ranges = expanded_ranges
            logger.debug(f"Expanded variable ranges by factor {bound_scale:.2f} for {num_variables} variables")
        
        # Build SCM with unified variable ranges
        scm = create_simple_linear_scm(
            variables=variables,
            edges=edges,
            coefficients=coefficients,
            noise_scales=noise_scales,
            target=target_variable,
            variable_ranges=variable_ranges,
            output_bounds=None  # We handle bounds through variable_ranges now
        )
        
        # Add comprehensive metadata
        metadata = scm.get('metadata', pyr.pmap({}))
        metadata = metadata.update({
            'structure_type': structure_type,
            'num_variables': num_variables,
            'num_edges': len(edges),
            'edge_density': len(edges) / (num_variables * (num_variables - 1) / 2),
            'target_variable': target_variable,
            'coefficients': dict(coefficients),
            'generation_seed': self.seed,
            'factory_version': '1.0'
        })
        
        scm = scm.update({'metadata': metadata})
        
        logger.info(f"Generated {structure_type} SCM: {num_variables} vars, "
                   f"{len(edges)} edges, target={target_variable}")
        
        return scm
    
    def _select_target(self, variables: List[str], structure_type: str) -> str:
        """Select appropriate target variable based on structure type."""
        if structure_type in ["fork", "collider"]:
            # For fork/collider, target should be in the middle range
            mid_idx = len(variables) // 2
            return variables[mid_idx]
        elif structure_type == "chain":
            # For chain, target is always the last variable
            return variables[-1]
        else:
            # For mixed/random, choose randomly from non-first variables
            self.key, subkey = random.split(self.key)
            target_idx = random.randint(subkey, (), 1, len(variables))
            return variables[target_idx]
    
    def _create_fork_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create fork structure: multiple variables → target."""
        edges = []
        coefficients = {}
        
        non_target_vars = [v for v in variables if v != target]
        
        # Connect all non-target variables to target
        for var in non_target_vars:
            edges.append((var, target))
            # Generate random coefficient
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[(var, target)] = float(coeff)
        
        return edges, coefficients
    
    def _create_chain_structure(self, variables: List[str]) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create chain structure: X0 → X1 → X2 → ... → Xn."""
        edges = []
        coefficients = {}
        
        for i in range(len(variables) - 1):
            edge = (variables[i], variables[i + 1])
            edges.append(edge)
            
            # Generate coefficient
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[edge] = float(coeff)
        
        return edges, coefficients
    
    def _create_collider_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create collider structure with additional complexity."""
        edges = []
        coefficients = {}
        
        non_target_vars = [v for v in variables if v != target]
        
        # Main collider: multiple vars → target
        for var in non_target_vars[:min(3, len(non_target_vars))]:
            edges.append((var, target))
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[(var, target)] = float(coeff)
        
        # Add some additional edges for complexity
        remaining_vars = non_target_vars[3:] if len(non_target_vars) > 3 else []
        for i, var in enumerate(remaining_vars):
            if i < len(non_target_vars) - 3:
                # Connect to earlier variable
                parent = non_target_vars[i % 3]
                edges.append((parent, var))
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[(parent, var)] = float(coeff)
        
        return edges, coefficients
    
    def _create_mixed_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create mixed structure combining multiple patterns."""
        edges = []
        coefficients = {}
        
        non_target_vars = [v for v in variables if v != target]
        n_vars = len(variables)
        
        # Pattern 1: Some direct connections to target
        direct_parents = non_target_vars[:min(2, len(non_target_vars))]
        for var in direct_parents:
            edges.append((var, target))
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[(var, target)] = float(coeff)
        
        # Pattern 2: Chain leading to target
        if len(non_target_vars) > 2:
            chain_vars = non_target_vars[2:min(4, len(non_target_vars))]
            for i in range(len(chain_vars) - 1):
                edge = (chain_vars[i], chain_vars[i + 1])
                edges.append(edge)
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[edge] = float(coeff)
            
            # Connect chain end to target
            if chain_vars:
                edges.append((chain_vars[-1], target))
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[(chain_vars[-1], target)] = float(coeff)
        
        # Pattern 3: Additional random connections
        remaining_vars = non_target_vars[4:] if len(non_target_vars) > 4 else []
        for var in remaining_vars:
            # Connect to random earlier variable
            self.key, subkey = random.split(self.key)
            possible_parents = [v for v in variables if v != var and v != target]
            if possible_parents:
                parent_idx = random.randint(subkey, (), 0, len(possible_parents))
                parent = possible_parents[parent_idx]
                
                # Check for cycles (simple check)
                if not self._would_create_cycle(edges, parent, var):
                    edges.append((parent, var))
                    self.key, subkey = random.split(self.key)
                    coeff = random.uniform(subkey, (), 
                                         minval=self.coefficient_range[0], 
                                         maxval=self.coefficient_range[1])
                    coefficients[(parent, var)] = float(coeff)
        
        return edges, coefficients
    
    def _create_random_structure(self, variables: List[str], target: str, edge_density: float) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create random DAG structure with specified edge density."""
        edges = []
        coefficients = {}
        
        n_vars = len(variables)
        max_edges = n_vars * (n_vars - 1) // 2
        target_edges = int(max_edges * edge_density)
        
        # Ensure target has at least one parent
        non_target_vars = [v for v in variables if v != target]
        self.key, subkey = random.split(self.key)
        parent_idx = random.randint(subkey, (), 0, len(non_target_vars))
        parent = non_target_vars[parent_idx]
        edges.append((parent, target))
        
        self.key, subkey = random.split(self.key)
        coeff = random.uniform(subkey, (), 
                             minval=self.coefficient_range[0], 
                             maxval=self.coefficient_range[1])
        coefficients[(parent, target)] = float(coeff)
        
        # Add random edges
        attempts = 0
        while len(edges) < target_edges and attempts < target_edges * 3:
            self.key, subkey1, subkey2 = random.split(self.key, 3)
            
            from_idx = random.randint(subkey1, (), 0, n_vars)
            to_idx = random.randint(subkey2, (), 0, n_vars)
            
            from_var = variables[from_idx]
            to_var = variables[to_idx]
            
            edge = (from_var, to_var)
            
            # Check constraints
            if (from_var != to_var and 
                edge not in edges and 
                not self._would_create_cycle(edges, from_var, to_var)):
                
                edges.append(edge)
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[edge] = float(coeff)
            
            attempts += 1
        
        return edges, coefficients
    
    def _add_output_bounds_to_scm(self, scm: pyr.PMap, edges: List[Tuple[str, str]], 
                                  coefficients: Dict[Tuple[str, str], float],
                                  noise_scales: Dict[str, float],
                                  output_bounds: Tuple[float, float]) -> pyr.PMap:
        """Add output bounds to all mechanisms in an SCM."""
        from ..mechanisms.linear import LinearMechanism, RootMechanism
        from ..data_structures.scm import get_variables, get_mechanisms, create_scm
        
        variables = list(get_variables(scm))
        original_mechanisms = get_mechanisms(scm)
        
        # Create new mechanisms with bounds
        new_mechanisms = {}
        for var in variables:
            # Find parents
            parents = [parent for parent, child in edges if child == var]
            
            if parents:
                # Create bounded linear mechanism
                var_coefficients = {parent: coefficients[(parent, var)] for parent in parents}
                mechanism = LinearMechanism(
                    var_coefficients, 
                    intercept=0.0,  # Default intercept
                    noise_scale=noise_scales[var],
                    output_bounds=output_bounds
                )
            else:
                # Create bounded root mechanism
                mechanism = RootMechanism(
                    mean=0.0,  # Default mean
                    noise_scale=noise_scales[var],
                    output_bounds=output_bounds
                )
            
            new_mechanisms[var] = mechanism
        
        # Create new SCM with bounded mechanisms
        return create_scm(
            variables=get_variables(scm),
            edges=frozenset(edges),
            mechanisms=new_mechanisms,
            target=scm.get('target'),
            metadata=scm.get('metadata', {})
        )
    
    def _would_create_cycle(self, existing_edges: List[Tuple[str, str]], from_var: str, to_var: str) -> bool:
        """Check if adding edge would create cycle (simple DFS check)."""
        # Build adjacency list
        graph = {}
        for edge in existing_edges:
            if edge[0] not in graph:
                graph[edge[0]] = []
            graph[edge[0]].append(edge[1])
        
        # Check if there's a path from to_var to from_var
        def has_path(start: str, end: str, visited: Set[str]) -> bool:
            if start == end:
                return True
            if start in visited:
                return False
            
            visited.add(start)
            for neighbor in graph.get(start, []):
                if has_path(neighbor, end, visited):
                    return True
            return False
        
        return has_path(to_var, from_var, set())
    
    def create_scm_suite(self, 
                        variable_ranges: List[int] = [3, 4, 5, 6, 8],
                        structure_types: List[str] = ["fork", "chain", "collider", "mixed"]) -> Dict[str, pyr.PMap]:
        """
        Create a comprehensive suite of SCMs for testing.
        
        Args:
            variable_ranges: List of variable counts to generate
            structure_types: List of structure types to include
            
        Returns:
            Dictionary mapping names to SCMs
        """
        scm_suite = {}
        
        for num_vars in variable_ranges:
            for structure_type in structure_types:
                name = f"{structure_type}_{num_vars}var"
                scm = self.create_variable_scm(
                    num_variables=num_vars,
                    structure_type=structure_type
                )
                scm_suite[name] = scm
                
                logger.info(f"Added {name} to SCM suite")
        
        logger.info(f"Created SCM suite with {len(scm_suite)} SCMs")
        return scm_suite
    
    def get_random_scm(self, 
                      variable_counts: List[int] = [3, 4, 5, 6],
                      structure_types: List[str] = ["fork", "chain", "collider", "mixed"],
                      edge_density_range: Tuple[float, float] = (0.3, 0.7),
                      name_prefix: str = "random") -> Tuple[str, pyr.PMap]:
        """
        Generate a random SCM by sampling from specified parameter ranges.
        
        Args:
            variable_counts: List of possible variable counts to sample from
            structure_types: List of possible structure types to sample from  
            edge_density_range: Range for edge density (used for random structures)
            name_prefix: Prefix for generated SCM name
            
        Returns:
            Tuple of (scm_name, scm) for consistent interface
        """
        import random as py_random
        
        # Sample configuration
        num_variables = py_random.choice(variable_counts)
        structure_type = py_random.choice(structure_types)
        
        # Sample edge density for random structures
        if structure_type == "random":
            edge_density = py_random.uniform(*edge_density_range)
        else:
            edge_density = 0.5  # Default for structured types
        
        # Generate SCM
        scm = self.create_variable_scm(
            num_variables=num_variables,
            structure_type=structure_type,
            edge_density=edge_density
        )
        
        # Create descriptive name
        scm_name = f"{name_prefix}_{structure_type}_{num_variables}var"
        
        logger.info(f"🎲 Generated random SCM: {scm_name} (density={edge_density:.2f})")
        
        return scm_name, scm


# Convenience functions for backward compatibility
def create_variable_fork_scm(num_variables: int, noise_scale: float = 1.0, 
                            intervention_range: Tuple[float, float] = (-2.0, 2.0),
                            target: Optional[str] = None) -> pyr.PMap:
    """Create fork SCM with variable number of variables."""
    factory = VariableSCMFactory(noise_scale=noise_scale, intervention_range=intervention_range)
    return factory.create_variable_scm(
        num_variables=num_variables,
        structure_type="fork",
        target_variable=target
    )


def create_variable_chain_scm(num_variables: int, noise_scale: float = 1.0,
                             intervention_range: Tuple[float, float] = (-2.0, 2.0)) -> pyr.PMap:
    """Create chain SCM with variable number of variables."""
    factory = VariableSCMFactory(noise_scale=noise_scale, intervention_range=intervention_range)
    return factory.create_variable_scm(
        num_variables=num_variables,
        structure_type="chain"
    )


def create_variable_mixed_scm(num_variables: int, noise_scale: float = 1.0,
                             intervention_range: Tuple[float, float] = (-2.0, 2.0),
                             target: Optional[str] = None) -> pyr.PMap:
    """Create mixed structure SCM with variable number of variables."""
    factory = VariableSCMFactory(noise_scale=noise_scale, intervention_range=intervention_range)
    return factory.create_variable_scm(
        num_variables=num_variables,
        structure_type="mixed",
        target_variable=target
    )


def get_scm_info(scm: pyr.PMap) -> Dict[str, Any]:
    """Extract comprehensive information about an SCM."""
    metadata = scm.get('metadata', pyr.pmap({}))
    variables = list(get_variables(scm))
    target = get_target(scm)
    
    return {
        'num_variables': len(variables),
        'variables': variables,
        'target_variable': target,
        'structure_type': metadata.get('structure_type', 'unknown'),
        'num_edges': metadata.get('num_edges', 0),
        'edge_density': metadata.get('edge_density', 0.0),
        'coefficients': metadata.get('coefficients', {}),
        'generation_info': {
            'seed': metadata.get('generation_seed'),
            'factory_version': metadata.get('factory_version', 'unknown')
        }
    }