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
        "random": "Random DAG with specified edge density",
        "scale_free": "Scale-free graph with hub nodes (power-law degree distribution)",
        "two_layer": "Two-layer hierarchical structure (sources → sinks)"
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
    
    def _shuffle_variables(self, variables: List[str]) -> List[str]:
        """Randomly permute variable list to eliminate positional bias."""
        self.key, subkey = random.split(self.key)
        indices = jnp.arange(len(variables))
        shuffled_indices = random.permutation(subkey, indices)
        return [variables[int(i)] for i in shuffled_indices]
    
    def _random_subset_selection(self, items: List[str], min_size: int, max_size: int) -> List[str]:
        """Randomly select a subset of items."""
        if not items:
            return []
        
        self.key, subkey = random.split(self.key)
        size = random.randint(subkey, (), min_size, min(max_size + 1, len(items) + 1))
        
        self.key, subkey = random.split(self.key)
        indices = random.choice(subkey, len(items), shape=(size,), replace=False)
        return [items[int(i)] for i in indices]
    
    def _random_target_from_non_roots(self, variables: List[str], edges: List[Tuple[str, str]]) -> str:
        """Select random target from variables that have at least one parent."""
        # Find variables that have at least one parent
        children = {child for _, child in edges}
        valid_targets = [v for v in variables if v in children]
        
        if not valid_targets:
            # If no children exist yet, we need to ensure structure creates some
            # For now, return a non-first variable and let structure creation handle it
            self.key, subkey = random.split(self.key)
            target_idx = random.randint(subkey, (), 1, len(variables))
            return variables[target_idx]
        
        self.key, subkey = random.split(self.key)
        target_idx = random.randint(subkey, (), 0, len(valid_targets))
        return valid_targets[target_idx]
    
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
            edges, coefficients, target_variable = self._create_chain_structure(variables, target_variable)
        elif structure_type == "collider":
            edges, coefficients = self._create_collider_structure(variables, target_variable)
        elif structure_type == "mixed":
            edges, coefficients = self._create_mixed_structure(variables, target_variable)
        elif structure_type == "random":
            edges, coefficients = self._create_random_structure(variables, target_variable, edge_density)
        elif structure_type == "scale_free":
            edges, coefficients = self._create_scale_free_structure(variables, target_variable)
        elif structure_type == "two_layer":
            edges, coefficients = self._create_two_layer_structure(variables, target_variable)
        else:
            raise ValueError(f"Structure type {structure_type} not implemented")
        
        # Validate DAG structure
        if not self._validate_dag_structure(edges, variables):
            logger.warning(f"Generated {structure_type} structure is not a valid DAG, attempting to fix...")
            # Remove edges that create cycles
            validated_edges = []
            for edge in edges:
                if not self._would_create_cycle(validated_edges, edge[0], edge[1]):
                    validated_edges.append(edge)
            edges = validated_edges
            logger.info(f"Fixed DAG structure, removed {len(edges) - len(validated_edges)} edges")
        
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
        """Select appropriate target variable based on structure type.
        
        Target must have at least one parent (no root nodes as targets).
        Selection is randomized to avoid positional bias.
        """
        # For structures where we need to pre-determine edges to find valid targets,
        # we'll handle target selection within each structure method.
        # This method now serves as a fallback for random selection.
        
        # Random selection from non-first variables (simple heuristic)
        # Each structure method will override this with proper logic
        self.key, subkey = random.split(self.key)
        target_idx = random.randint(subkey, (), 1, len(variables))
        return variables[target_idx]
    
    def _create_fork_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """
        Create fork structure: multiple variables → target.
        
        Note: In causal terminology, this creates a "collider" structure where
        the target variable has multiple parents (all other variables point to it).
        The name "fork" is kept for backwards compatibility, but this is technically
        a collider pattern where multiple causes affect a common effect (the target).
        
        Structure: X0 → Y ← X1, X2 → Y, etc. (where Y is the target)
        """
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
    
    def _create_chain_structure(self, variables: List[str], target_variable: Optional[str]) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float], str]:
        """Create chain structure with randomized variable ordering and target selection."""
        edges = []
        coefficients = {}
        
        # Shuffle variables to eliminate positional bias
        shuffled_vars = self._shuffle_variables(variables)
        
        # If no target specified, randomly select from non-root positions
        if target_variable is None:
            # Target can be any variable except the first in the shuffled chain
            self.key, subkey = random.split(self.key)
            target_idx = random.randint(subkey, (), 1, len(shuffled_vars))
            target_variable = shuffled_vars[target_idx]
        
        # Find target position in shuffled variables
        target_pos = shuffled_vars.index(target_variable)
        
        # Create chain up to target position
        for i in range(target_pos):
            edge = (shuffled_vars[i], shuffled_vars[i + 1])
            edges.append(edge)
            
            # Generate coefficient
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[edge] = float(coeff)
        
        # If target is not at the end, continue chain after target
        for i in range(target_pos, len(shuffled_vars) - 1):
            edge = (shuffled_vars[i], shuffled_vars[i + 1])
            edges.append(edge)
            
            # Generate coefficient
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[edge] = float(coeff)
        
        return edges, coefficients, target_variable
    
    def _create_collider_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """
        Create collider structure with additional complexity.
        
        This creates a partial collider where a subset of variables (up to 3) point
        to the target, while remaining variables form additional causal chains.
        This is similar to fork but with limited parent count and extra structure.
        
        Structure: Limited collider (max 3 parents) plus additional chains
        """
        edges = []
        coefficients = {}
        
        non_target_vars = [v for v in variables if v != target]
        
        # Main collider: randomly selected vars → target
        collider_parents = self._random_subset_selection(non_target_vars, 1, min(3, len(non_target_vars)))
        for var in collider_parents:
            edges.append((var, target))
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[(var, target)] = float(coeff)
        
        # Add some additional edges for complexity
        remaining_vars = [v for v in non_target_vars if v not in collider_parents]
        for var in remaining_vars:
            # Connect to random available parent (could be collider parent or other remaining var)
            possible_parents = [v for v in variables if v != var and v != target]
            if possible_parents:
                self.key, subkey = random.split(self.key)
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
    
    def _create_mixed_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """Create mixed structure combining multiple patterns."""
        edges = []
        coefficients = {}
        
        non_target_vars = [v for v in variables if v != target]
        n_vars = len(variables)
        
        # Pattern 1: Randomly selected direct connections to target
        direct_parents = self._random_subset_selection(non_target_vars, 1, min(2, len(non_target_vars)))
        for var in direct_parents:
            edges.append((var, target))
            self.key, subkey = random.split(self.key)
            coeff = random.uniform(subkey, (), 
                                 minval=self.coefficient_range[0], 
                                 maxval=self.coefficient_range[1])
            coefficients[(var, target)] = float(coeff)
        
        # Pattern 2: Chain leading to target (if enough variables)
        available_for_chain = [v for v in non_target_vars if v not in direct_parents]
        if len(available_for_chain) > 1:
            chain_vars = self._random_subset_selection(available_for_chain, 2, min(3, len(available_for_chain)))
            
            # Randomly shuffle chain variables to avoid ordering bias
            chain_vars = self._shuffle_variables(chain_vars)
            
            # Create chain connections
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
        
        # Pattern 3: Additional random connections for remaining variables
        used_vars = set(direct_parents + (chain_vars if 'chain_vars' in locals() else []))
        remaining_vars = [v for v in non_target_vars if v not in used_vars]
        for var in remaining_vars:
            # Connect to random available parent
            possible_parents = [v for v in variables if v != var and v != target]
            if possible_parents:
                self.key, subkey = random.split(self.key)
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
    
    def _create_scale_free_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """
        Create scale-free structure using preferential attachment.
        Scale-free networks have hub nodes with many connections.
        """
        edges = []
        coefficients = {}
        n_vars = len(variables)
        
        # Track degree for preferential attachment
        in_degree = {var: 0 for var in variables}
        out_degree = {var: 0 for var in variables}
        
        # Start with a small complete subgraph (seed), excluding target
        non_target_vars = [v for v in variables if v != target]
        seed_size = min(3, len(non_target_vars))
        
        # Randomly select seed variables (excluding target to ensure target has parents)
        seed_vars = self._random_subset_selection(non_target_vars, seed_size, seed_size)
        
        # Create complete subgraph among seed variables
        for i in range(len(seed_vars)):
            for j in range(i + 1, len(seed_vars)):
                # Add forward edge to ensure DAG
                edge = (seed_vars[i], seed_vars[j])
                edges.append(edge)
                
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[edge] = float(coeff)
                
                out_degree[seed_vars[i]] += 1
                in_degree[seed_vars[j]] += 1
        
        # Add remaining nodes using preferential attachment
        remaining_nodes = [v for v in variables if v not in seed_vars]
        for new_node in remaining_nodes:
            
            # Nodes that already exist (seed + previously added)
            existing_nodes = [v for v in variables if v != new_node and (v in seed_vars or in_degree[v] > 0 or out_degree[v] > 0)]
            
            if not existing_nodes:
                continue
                
            # Determine number of edges to add (typically 1-3)
            self.key, subkey = random.split(self.key)
            num_edges = int(random.uniform(subkey, (), minval=1, maxval=min(4, len(existing_nodes) + 1)))
            
            # Calculate attachment probabilities based on degree
            total_degree = sum(in_degree[v] + out_degree[v] + 1 for v in existing_nodes)
            probs = [(in_degree[v] + out_degree[v] + 1) / total_degree for v in existing_nodes]
            
            # Sample nodes to connect (without replacement)
            self.key, subkey = random.split(self.key)
            selected_indices = random.choice(
                subkey, 
                len(existing_nodes),
                shape=(min(num_edges, len(existing_nodes)),),
                p=jnp.array(probs),
                replace=False
            )
            
            for idx in selected_indices:
                parent = existing_nodes[int(idx)]
                # Add edge parent -> new_node
                edge = (parent, new_node)
                edges.append(edge)
                
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[edge] = float(coeff)
                
                out_degree[parent] += 1
                in_degree[new_node] += 1
        
        return edges, coefficients
    
    def _create_two_layer_structure(self, variables: List[str], target: str) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
        """
        Create two-layer hierarchical structure.
        First layer: source nodes (no parents)
        Second layer: sink nodes (receive from first layer)
        """
        edges = []
        coefficients = {}
        n_vars = len(variables)
        
        # Randomly assign variables to two layers, ensuring target is in layer2
        non_target_vars = [v for v in variables if v != target]
        shuffled_non_target = self._shuffle_variables(non_target_vars)
        
        # Split shuffled non-target variables into two layers
        layer1_size = len(shuffled_non_target) // 2
        layer1 = shuffled_non_target[:layer1_size]
        layer2_non_target = shuffled_non_target[layer1_size:]
        
        # Add target to layer2 to ensure it has parents
        layer2 = layer2_non_target + [target]
        
        # Ensure we have at least one node in each layer
        if not layer1 or not layer2:
            # Fall back to chain for very small graphs
            edges, coefficients, _ = self._create_chain_structure(variables, target)
            return edges, coefficients
        
        # Connect layer1 to layer2 with varying density
        for source in layer1:
            # Each source connects to 1-3 sinks
            self.key, subkey = random.split(self.key)
            num_connections = int(random.uniform(subkey, (), minval=1, maxval=min(4, len(layer2) + 1)))
            
            # Sample which sinks to connect to
            self.key, subkey = random.split(self.key)
            selected_sinks = random.choice(
                subkey,
                len(layer2),
                shape=(min(num_connections, len(layer2)),),
                replace=False
            )
            
            for sink_idx in selected_sinks:
                sink = layer2[sink_idx]
                edge = (source, sink)
                edges.append(edge)
                
                self.key, subkey = random.split(self.key)
                coeff = random.uniform(subkey, (), 
                                     minval=self.coefficient_range[0], 
                                     maxval=self.coefficient_range[1])
                coefficients[edge] = float(coeff)
        
        # Optionally add some connections within layer2 for complexity
        if len(layer2) > 2:
            for i in range(len(layer2) - 1):
                # Add edge with 30% probability
                self.key, subkey = random.split(self.key)
                if random.uniform(subkey, ()) < 0.3:
                    edge = (layer2[i], layer2[i + 1])
                    edges.append(edge)
                    
                    self.key, subkey = random.split(self.key)
                    coeff = random.uniform(subkey, (), 
                                         minval=self.coefficient_range[0], 
                                         maxval=self.coefficient_range[1])
                    coefficients[edge] = float(coeff)
        
        return edges, coefficients
    
    def _would_create_cycle(self, existing_edges: List[Tuple[str, str]], from_var: str, to_var: str) -> bool:
        """Check if adding edge would create cycle (simple DFS check)."""
        # Build adjacency list including the proposed new edge
        graph = {}
        for edge in existing_edges:
            if edge[0] not in graph:
                graph[edge[0]] = []
            graph[edge[0]].append(edge[1])
        
        # Add the proposed edge temporarily
        if from_var not in graph:
            graph[from_var] = []
        graph[from_var].append(to_var)
        
        # Check if there's a path from to_var back to from_var (which would create a cycle)
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
    
    def _validate_dag_structure(self, edges: List[Tuple[str, str]], variables: List[str]) -> bool:
        """Validate that the edge list forms a valid DAG."""
        # Build adjacency list
        graph = {var: [] for var in variables}
        for parent, child in edges:
            graph[parent].append(child)
        
        # Perform topological sort using Kahn's algorithm
        in_degree = {var: 0 for var in variables}
        for parent in graph:
            for child in graph[parent]:
                in_degree[child] += 1
        
        queue = [var for var in variables if in_degree[var] == 0]
        sorted_count = 0
        
        while queue:
            node = queue.pop(0)
            sorted_count += 1
            
            for child in graph[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        
        # If we sorted all nodes, it's a DAG
        return sorted_count == len(variables)
    
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
                      structure_types: List[str] = ["fork", "chain", "collider", "mixed", "random", "scale_free", "two_layer"],
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