# Immutable Structural Causal Model (SCM) API Reference

## Overview
This module provides functions for creating and manipulating immutable Structural Causal Models using pure functional programming principles. SCMs represent causal relationships between variables and their mechanisms.

## Core Types
- **pyr.PMap**: Persistent (immutable) map from pyrsistent library that represents the SCM structure
- **FrozenSet[str]**: Immutable set of variable names
- **FrozenSet[Tuple[str, str]]**: Immutable set of edges, where each edge is a (parent, child) tuple
- **Dict[str, Callable]**: Dictionary mapping variable names to their mechanism functions

## Core Functions

### create_scm(variables, edges, mechanisms, target=None, metadata=None)
Create a new immutable Structural Causal Model (SCM).

**Parameters:**
- variables: FrozenSet[str] - Set of variable names in the model
- edges: FrozenSet[Tuple[str, str]] - Set of (parent, child) pairs representing the causal graph
- mechanisms: Dict[str, Callable] - Dictionary mapping variable names to their mechanism functions
- target: Optional[str] - Optional target variable (for optimization tasks)
- metadata: Optional[Dict[str, Any]] - Optional additional metadata

**Returns:**
pyr.PMap - An immutable SCM representation

**Example:**
```python
# Create a simple SCM with two variables
import pyrsistent as pyr
from typing import FrozenSet, Dict, Callable, Tuple

# Define variables
variables = pyr.s("X", "Y")

# Define edges (X causes Y)
edges = pyr.s(("X", "Y"))

# Define mechanisms
def x_mechanism():
    return 5  # X is exogenous

def y_mechanism(x):
    return 2 * x  # Y depends on X

mechanisms = {
    "X": x_mechanism,
    "Y": y_mechanism
}

# Create the SCM
scm = create_scm(variables, edges, mechanisms)
```

### get_variables(scm)
Get the set of variables in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
FrozenSet[str] - Set of variable names in the model

**Example:**
```python
variables = get_variables(scm)
print(variables)  # frozenset({'X', 'Y'})
```

### get_edges(scm)
Get the set of edges in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
FrozenSet[Tuple[str, str]] - Set of (parent, child) edges in the causal graph

**Example:**
```python
edges = get_edges(scm)
print(edges)  # frozenset({('X', 'Y')})
```

### get_mechanisms(scm)
Get the mechanisms dictionary of the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
pyr.PMap - Dictionary mapping variable names to their mechanism functions

**Example:**
```python
mechanisms = get_mechanisms(scm)
x_value = mechanisms["X"]()
print(x_value)  # 5
```

### get_parents(scm, variable)
Get the parents of a variable in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- variable: str - The variable name to get parents for

**Returns:**
FrozenSet[str] - A frozen set of parent variable names

**Example:**
```python
parents = get_parents(scm, "Y")
print(parents)  # frozenset({'X'})

parents = get_parents(scm, "X")
print(parents)  # frozenset()
```

### get_children(scm, variable)
Get the children of a variable in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- variable: str - The variable name to get children for

**Returns:**
FrozenSet[str] - A frozen set of child variable names

**Example:**
```python
children = get_children(scm, "X")
print(children)  # frozenset({'Y'})

children = get_children(scm, "Y")
print(children)  # frozenset()
```

### get_ancestors(scm, variable)
Get all ancestors of a variable in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- variable: str - The variable to get ancestors for

**Returns:**
FrozenSet[str] - A frozen set of ancestor variable names

**Example:**
```python
# Create a more complex SCM: A → B → C
variables = pyr.s("A", "B", "C")
edges = pyr.s(("A", "B"), ("B", "C"))
scm = create_scm(variables, edges, {})

ancestors = get_ancestors(scm, "C")
print(ancestors)  # frozenset({'A', 'B'})
```

### get_descendants(scm, variable)
Get all descendants of a variable in the SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- variable: str - The variable to get descendants for

**Returns:**
FrozenSet[str] - A frozen set of descendant variable names

**Example:**
```python
# Using the SCM: A → B → C
descendants = get_descendants(scm, "A")
print(descendants)  # frozenset({'B', 'C'})
```

### is_cyclic(scm)
Check if the SCM contains cycles.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
bool - True if the graph contains cycles, False otherwise

**Example:**
```python
# Acyclic SCM: A → B → C
acyclic_scm = create_scm(
    pyr.s("A", "B", "C"),
    pyr.s(("A", "B"), ("B", "C")),
    {}
)
print(is_cyclic(acyclic_scm))  # False

# Cyclic SCM: A → B → C → A
cyclic_scm = create_scm(
    pyr.s("A", "B", "C"),
    pyr.s(("A", "B"), ("B", "C"), ("C", "A")),
    {}
)
print(is_cyclic(cyclic_scm))  # True
```

### topological_sort(scm)
Return variables in topological order (parents before children).

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
List[str] - List of variables in topological order

**Raises:**
- ValueError: If the graph contains cycles

**Example:**
```python
# SCM: A → B → C and A → C
scm = create_scm(
    pyr.s("A", "B", "C"),
    pyr.s(("A", "B"), ("B", "C"), ("A", "C")),
    {}
)
order = topological_sort(scm)
print(order)  # ['A', 'B', 'C']
```

### validate_mechanisms(scm)
Check if all variables have defined mechanisms.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
bool - True if all variables have mechanisms, False otherwise

**Example:**
```python
# Complete SCM with mechanisms for all variables
complete_scm = create_scm(
    pyr.s("X", "Y"),
    pyr.s(("X", "Y")),
    {
        "X": lambda: 5,
        "Y": lambda x: 2 * x
    }
)
print(validate_mechanisms(complete_scm))  # True

# Incomplete SCM missing a mechanism
incomplete_scm = create_scm(
    pyr.s("X", "Y"),
    pyr.s(("X", "Y")),
    {
        "X": lambda: 5
    }
)
print(validate_mechanisms(incomplete_scm))  # False
```

### validate_edge_consistency(scm)
Check if mechanism inputs are consistent with edge definitions.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
bool - True if edges and mechanisms appear consistent, False otherwise

**Example:**
```python
# Consistent SCM
consistent_scm = create_scm(
    pyr.s("X", "Y"),
    pyr.s(("X", "Y")),
    {
        "X": lambda: 5,
        "Y": lambda x: 2 * x
    }
)
print(validate_edge_consistency(consistent_scm))  # True

# Inconsistent SCM (Y has parent X but no mechanism)
inconsistent_scm = create_scm(
    pyr.s("X", "Y"),
    pyr.s(("X", "Y")),
    {
        "X": lambda: 5
    }
)
print(validate_edge_consistency(inconsistent_scm))  # False
```