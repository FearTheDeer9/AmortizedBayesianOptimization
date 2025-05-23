# ADR 001: Intervention Representation (Revised)

## Context

We need to represent different types of interventions (perfect, imperfect, soft) in a way that:
- Balances functional programming benefits with performance requirements
- Allows for extension to new intervention types
- Clearly separates data from behavior
- Supports efficient comparison and caching
- Integrates well with ML training pipelines

## Decision

Use selective immutability with a function registry pattern for polymorphic behavior:

1. **Intervention specifications**: Immutable (frozen dataclasses) - small, frequently compared objects
2. **Handler functions**: Pure functions registered for each intervention type
3. **Application results**: Efficient approach based on usage (mutable workspaces when needed)
4. **Factory functions**: Pure functions for creating intervention specifications

## Implementation

```python
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Any, Callable, Dict
from functools import singledispatch

@dataclass(frozen=True)
class InterventionSpec:
    """Immutable intervention specification - small and frequently compared"""
    type: str
    variables: frozenset[str]
    parameters: Mapping[str, Any]
    
    def __hash__(self):
        # Enables efficient caching and set operations
        return hash((self.type, self.variables, tuple(sorted(self.parameters.items()))))

# Registry of pure handler functions
INTERVENTION_HANDLERS: Dict[str, Callable] = {}

def register_intervention_handler(intervention_type: str):
    """Decorator to register intervention handlers"""
    def decorator(func):
        INTERVENTION_HANDLERS[intervention_type] = func
        return func
    return decorator

def apply_intervention(scm: SCM, intervention: InterventionSpec) -> SCM:
    """Apply intervention using registered handler"""
    handler = INTERVENTION_HANDLERS[intervention.type]
    return handler(scm, intervention)

# Factory functions for common intervention types
def perfect_intervention(variables: dict[str, Any]) -> InterventionSpec:
    """Create a perfect intervention specification"""
    return InterventionSpec(
        type="perfect",
        variables=frozenset(variables.keys()),
        parameters={"values": variables}
    )

def soft_intervention(variable: str, shift: float, scale: float = 1.0) -> InterventionSpec:
    """Create a soft intervention specification"""
    return InterventionSpec(
        type="soft",
        variables=frozenset([variable]),
        parameters={"shift": shift, "scale": scale}
    )

# Example handler implementations
@register_intervention_handler("perfect")
def apply_perfect_intervention(scm: SCM, intervention: InterventionSpec) -> SCM:
    """Apply perfect intervention - may use mutable workspace for efficiency"""
    values = intervention.parameters["values"]
    # Implementation can use efficient mutable operations internally
    # while maintaining pure function interface
    return scm.with_interventions(values)

@register_intervention_handler("soft")
def apply_soft_intervention(scm: SCM, intervention: InterventionSpec) -> SCM:
    """Apply soft intervention"""
    variable = list(intervention.variables)[0]
    shift = intervention.parameters["shift"]
    scale = intervention.parameters["scale"]
    return scm.with_soft_intervention(variable, shift, scale)
```

## Alternatives Considered

### Class Hierarchy with Inheritance
```python
@dataclass(frozen=True)
class Intervention:
    variables: FrozenSet[str]
    
    def apply(self, scm: SCM) -> SCM:
        raise NotImplementedError
        
@dataclass(frozen=True)
class PerfectIntervention(Intervention):
    values: Mapping[str, Any]
    
    def apply(self, scm: SCM) -> SCM:
        # Implementation
```

**Pros**: Familiar OOP pattern, clear polymorphism  
**Cons**: Less functional, interface changes affect all subclasses, harder to optimize performance

### Switch Statement / Pattern Matching
```python
def apply_intervention(scm, intervention):
    if intervention["type"] == "perfect":
        # Handle perfect intervention
    elif intervention["type"] == "imperfect":
        # Handle imperfect intervention
    # ...
```

**Pros**: Simple, direct  
**Cons**: Not extensible without modifying code, violation of open-closed principle

### Fully Mutable Approach
```python
class MutableIntervention:
    def __init__(self, type: str, variables: set[str]):
        self.type = type
        self.variables = variables
        self.parameters = {}
```

**Pros**: Maximum performance, familiar pattern  
**Cons**: Loses benefits of immutability for caching, comparison, and reasoning

## Consequences

### Positive
- **Extensibility**: New intervention types can be added without modifying existing code
- **Performance**: Intervention specs are lightweight; application can be optimized
- **Caching**: Immutable specs enable efficient caching and memoization
- **Reasoning**: Clear separation of data and behavior
- **Integration**: Works well with both functional and imperative code
- **Composition**: Interventions can be easily combined and compared

### Negative
- **Complexity**: Mixed mutability model requires clear documentation
- **Learning Curve**: Requires understanding of higher-order function patterns
- **Runtime Errors**: Potential for runtime errors if handlers aren't registered
- **Guidelines Needed**: Need explicit guidance about when to use mutable vs immutable approaches

## Implementation Guidelines

1. **Intervention Specifications**: Always immutable - they're small and frequently compared
2. **Handler Functions**: Keep pure for reasoning benefits
3. **Internal Implementation**: Use efficient mutable operations when needed for performance
4. **Caching**: Leverage immutable specs for efficient caching of expensive operations
5. **Testing**: Immutable specs make testing easier and more reliable

## Performance Considerations

- Intervention specs are small objects (typically < 1KB) where immutability overhead is negligible
- Handler functions can use efficient mutable operations internally while maintaining pure interfaces
- Caching benefits of immutable specs often outweigh creation costs
- Critical path (neural network training) isn't affected by intervention representation choice