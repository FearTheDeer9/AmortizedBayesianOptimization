# ADR 001: Intervention Representation

## Context

We need to represent different types of interventions (perfect, imperfect, soft) in a way that:
- Follows functional programming principles
- Allows for extension to new intervention types
- Clearly separates data from behavior
- Maintains immutability

## Decision

Use immutable data structures with a function registry pattern for polymorphic behavior:

1. Represent interventions as immutable maps using pyrsistent
2. Register handler functions for each intervention type
3. Use factory functions to create intervention specifications
4. Apply interventions via a generic function that dispatches to the appropriate handler

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

Pros: Familiar OOP pattern, clear polymorphism
Cons: Less functional, interface changes affect all subclasses

Switch Statement / Pattern Matching
pythondef apply_intervention(scm, intervention):
    if intervention["type"] == "perfect":
        # Handle perfect intervention
    elif intervention["type"] == "imperfect":
        # Handle imperfect intervention
    # ...

Pros: Simple, direct
Cons: Not extensible without modifying code, violation of open-closed principle

Consequences
Positive

New intervention types can be added without modifying existing code
Clear separation of data and behavior
Fully immutable data structures
Composition of interventions becomes straightforward

Negative

Requires understanding of higher-order function patterns
Potential for runtime errors if handlers aren't registered
More complex than simple class hierarchy for newcomers

Implementation
See the function registry pattern implementation in the codebase.