"""
Immutable Structural Causal Model (SCM) data structure.

This module provides functions for creating and manipulating immutable
Structural Causal Models using pure functional programming principles.
"""

import pyrsistent as pyr
from typing import FrozenSet, Dict, Callable, Optional, Any, Tuple, List, Set
import functools

def create_scm(
    variables: FrozenSet[str],
    edges: FrozenSet[Tuple[str, str]],
    mechanisms: Dict[str, Callable],
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Create a new immutable Structural Causal Model (SCM).
    
    Args:
        variables: Frozen set of variable names
        edges: Frozen set of (parent, child) pairs representing the causal graph
        mechanisms: Dictionary mapping variable names to their mechanism functions
        target: Optional target variable (for optimization tasks)
        metadata: Optional additional metadata
        
    Returns:
        An immutable SCM representation
    """
    # More efficient conversion - avoid multiple type checks
    immutable_variables = pyr.s(*variables) if isinstance(variables, (set, list, tuple)) else variables
    immutable_edges = pyr.s(*edges) if isinstance(edges, (set, list, tuple)) else edges
    
    # Store mechanism names instead of functions to avoid pyrsistent issues
    mechanism_names = pyr.pmap({k: k for k in mechanisms.keys()})
    
    # Create the immutable SCM
    return pyr.m(
        variables=immutable_variables,
        edges=immutable_edges,
        mechanism_names=mechanism_names,
        mechanisms=mechanisms,  # Keep as regular dict for function storage
        target=target,
        metadata=pyr.pmap(metadata or {})
    )

def get_variables(scm: pyr.PMap) -> FrozenSet[str]:
    """Get the set of variables in the SCM."""
    return scm['variables']

def get_edges(scm: pyr.PMap) -> FrozenSet[Tuple[str, str]]:
    """Get the set of edges in the SCM."""
    return scm['edges']

def get_mechanisms(scm: pyr.PMap) -> Dict[str, Callable]:
    """Get the mechanisms dictionary of the SCM."""
    return scm['mechanisms']

def get_target(scm: pyr.PMap) -> Optional[str]:
    """Get the target variable of the SCM."""
    return scm.get('target')


def get_parents(scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """
    Get the parents of a variable in the SCM.
    
    Args:
        scm: The structural causal model
        variable: The variable name to get parents for
        
    Returns:
        A frozen set of parent variable names
    """
    return pyr.s(*(parent for parent, child in scm['edges'] if child == variable))

def get_children(scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """
    Get the children of a variable in the SCM.
    
    Args:
        scm: The structural causal model
        variable: The variable name to get children for
        
    Returns:
        A frozen set of child variable names
    """
    return pyr.s(*(child for parent, child in scm['edges'] if parent == variable))

@functools.lru_cache(maxsize=128)
def get_ancestors(scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """
    Get all ancestors of a variable in the SCM.
    
    Args:
        scm: The structural causal model
        variable: The variable to get ancestors for
        
    Returns:
        A frozen set of ancestor variable names
    """
    # Convert scm to a hashable form for caching
    scm_key = (scm['variables'], scm['edges'])
    return _get_ancestors_impl(scm_key, scm, variable)

def _get_ancestors_impl(scm_key, scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """Internal implementation for get_ancestors."""
    visited = set()
    
    def visit(node):
        parents = get_parents(scm, node)
        for parent in parents:
            if parent not in visited:
                visited.add(parent)
                visit(parent)
    
    visit(variable)
    return pyr.s(*visited)

@functools.lru_cache(maxsize=128)
def get_descendants(scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """
    Get all descendants of a variable in the SCM.
    
    Args:
        scm: The structural causal model
        variable: The variable to get descendants for
        
    Returns:
        A frozen set of descendant variable names
    """
    # Convert scm to a hashable form for caching
    scm_key = (scm['variables'], scm['edges'])
    return _get_descendants_impl(scm_key, scm, variable)

def _get_descendants_impl(scm_key, scm: pyr.PMap, variable: str) -> FrozenSet[str]:
    """Internal implementation for get_descendants."""
    visited = set()
    
    def visit(node):
        children = get_children(scm, node)
        for child in children:
            if child not in visited:
                visited.add(child)
                visit(child)
    
    visit(variable)
    return pyr.s(*visited)

def is_cyclic(scm: pyr.PMap) -> bool:
    """
    Check if the SCM contains cycles.
    
    Args:
        scm: The structural causal model
        
    Returns:
        True if the graph contains cycles, False otherwise
    """
    # Track nodes that are being visited in the current DFS path
    path = set()
    # Track nodes that have been completely visited
    visited = set()
    
    def has_cycle(node):
        if node in path:
            return True
        if node in visited:
            return False
        
        path.add(node)
        
        for child in get_children(scm, node):
            if has_cycle(child):
                return True
        
        path.remove(node)
        visited.add(node)
        return False
    
    # Check each node that hasn't been visited
    variables = get_variables(scm)
    for node in variables:
        if node not in visited and has_cycle(node):
            return True
    
    return False

def topological_sort(scm: pyr.PMap) -> List[str]:
    """
    Return variables in topological order (parents before children).
    
    Args:
        scm: The structural causal model
        
    Returns:
        List of variables in topological order
        
    Raises:
        ValueError: If the graph contains cycles
    """
    if is_cyclic(scm):
        raise ValueError("Cannot perform topological sort on cyclic graph")
    
    # Track visited nodes and result order
    visited = set()
    result = []
    
    def visit(node):
        if node in visited:
            return
        
        # Visit all parents first
        for parent in get_parents(scm, node):
            visit(parent)
        
        visited.add(node)
        result.append(node)
    
    # Visit each node
    for node in get_variables(scm):
        visit(node)
    
    return result

def validate_mechanisms(scm: pyr.PMap) -> bool:
    """
    Check if all variables have defined mechanisms.
    
    Args:
        scm: The structural causal model
        
    Returns:
        True if all variables have mechanisms, False otherwise
    """
    mechanisms = get_mechanisms(scm)
    variables = get_variables(scm)
    return all(var in mechanisms for var in variables)

def validate_edge_consistency(scm: pyr.PMap) -> bool:
    """
    Check if mechanism inputs are consistent with edge definitions.
    
    This validation is limited since Python doesn't easily allow 
    introspection of function parameters. More comprehensive validation
    would need to be implemented within the mechanisms themselves.
    
    Args:
        scm: The structural causal model
        
    Returns:
        True if edges and mechanisms appear consistent, False otherwise
    """
    # For now, just verify that mechanisms exist for variables with parents
    mechanisms = get_mechanisms(scm)
    for var in get_variables(scm):
        if get_parents(scm, var) and var not in mechanisms:
            return False
    return True


def serialize_scm_for_storage(scm: pyr.PMap) -> Dict[str, Any]:
    """
    Serialize an SCM for storage by converting mechanism functions to descriptors.
    
    This enables pickle-safe serialization by storing the "quote" of each mechanism
    (its configuration) rather than the actual function closures.
    
    Args:
        scm: The structural causal model to serialize
        
    Returns:
        Dictionary containing serializable SCM data
    """
    from ..mechanisms.descriptors import descriptors_to_dict
    
    # Get mechanism descriptors for all variables
    mechanism_descriptors = {}
    mechanisms = get_mechanisms(scm)
    
    for var, mechanism_func in mechanisms.items():
        parents = list(get_parents(scm, var))
        
        # Try to recreate mechanism with descriptor
        try:
            if parents:
                # For variables with parents, assume linear mechanism
                # Note: This is a limitation - we need the original parameters
                # In practice, mechanisms should be created with _return_descriptor=True
                from ..mechanisms.linear import create_linear_mechanism
                
                # This is a fallback - in real usage, descriptors should be stored when creating SCMs
                raise ValueError(f"Cannot serialize mechanism for variable '{var}' without descriptor. "
                                f"Use create_linear_mechanism(..., _return_descriptor=True) when creating SCMs for storage.")
            else:
                # Root variable - try to recreate as root mechanism
                from ..mechanisms.linear import create_root_mechanism
                
                raise ValueError(f"Cannot serialize root mechanism for variable '{var}' without descriptor. "
                                f"Use create_root_mechanism(..., _return_descriptor=True) when creating SCMs for storage.")
                                
        except Exception as e:
            # Check if SCM already has descriptors stored
            if hasattr(scm, 'mechanism_descriptors') and var in scm.mechanism_descriptors:
                mechanism_descriptors[var] = scm.mechanism_descriptors[var]
            else:
                raise ValueError(f"Cannot serialize mechanism for variable '{var}': {e}")
    
    # Create serializable SCM dictionary
    serialized_scm = {
        'variables': list(get_variables(scm)),
        'edges': list(get_edges(scm)),
        'mechanism_descriptors': descriptors_to_dict(mechanism_descriptors),
        'target': get_target(scm),
        'metadata': dict(scm.get('metadata', {}))
    }
    
    return serialized_scm


def deserialize_scm_from_storage(serialized_scm: Dict[str, Any]) -> pyr.PMap:
    """
    Deserialize an SCM from storage by recreating mechanism functions from descriptors.
    
    Args:
        serialized_scm: Dictionary containing serialized SCM data
        
    Returns:
        Reconstructed SCM with working mechanism functions
    """
    from ..mechanisms.descriptors import descriptors_from_dict, descriptor_to_mechanism
    
    # Reconstruct basic SCM structure
    variables = frozenset(serialized_scm['variables'])
    edges = frozenset((parent, child) for parent, child in serialized_scm['edges'])
    target = serialized_scm.get('target')
    metadata = serialized_scm.get('metadata', {})
    
    # Recreate mechanism functions from descriptors
    mechanism_descriptors = descriptors_from_dict(serialized_scm['mechanism_descriptors'])
    mechanisms = {}
    
    for var, descriptor in mechanism_descriptors.items():
        mechanisms[var] = descriptor_to_mechanism(descriptor)
    
    # Create new SCM with reconstructed mechanisms
    scm = create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target=target,
        metadata=metadata
    )
    
    # Store descriptors for future serialization
    scm = scm.set('mechanism_descriptors', mechanism_descriptors)
    
    return scm


def create_scm_with_descriptors(
    variables: FrozenSet[str],
    edges: FrozenSet[Tuple[str, str]], 
    mechanism_descriptors: Dict[str, Any],
    target: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Create an SCM with mechanism descriptors for serialization support.
    
    This is the preferred way to create SCMs that will be saved/loaded.
    
    Args:
        variables: Frozen set of variable names
        edges: Frozen set of (parent, child) pairs
        mechanism_descriptors: Dictionary mapping variables to mechanism descriptors
        target: Optional target variable
        metadata: Optional additional metadata
        
    Returns:
        SCM with both mechanism functions and descriptors for serialization
    """
    from ..mechanisms.descriptors import descriptor_to_mechanism
    
    # Create mechanism functions from descriptors
    mechanisms = {}
    for var, descriptor in mechanism_descriptors.items():
        mechanisms[var] = descriptor_to_mechanism(descriptor)
    
    # Create SCM
    scm = create_scm(
        variables=variables,
        edges=edges,
        mechanisms=mechanisms,
        target=target,
        metadata=metadata
    )
    
    # Store descriptors for serialization
    scm = scm.set('mechanism_descriptors', mechanism_descriptors)
    
    return scm