# Interventions API Reference

## Overview
The interventions module provides a flexible, extensible system for applying interventions to structural causal models. It uses a function registry pattern to support different intervention types while maintaining functional programming principles.

The module consists of two main components:
1. **Registry** (`registry.py`) - Core infrastructure for registering and applying intervention handlers
2. **Handlers** (`handlers.py`) - Concrete implementations of specific intervention types

## Design Philosophy
The interventions system follows these principles:
- **Extensibility**: New intervention types can be added without modifying existing code
- **Immutability**: Interventions and SCMs are immutable for safety and reasoning
- **Polymorphism**: Different intervention types are handled through a common interface
- **Validation**: Comprehensive validation at multiple levels

## Core Types

### InterventionSpec
```python
InterventionSpec = pyr.PMap  # Immutable intervention specification
```

An immutable specification containing:
- `type`: str - The intervention type (e.g., 'perfect', 'soft')
- `targets`: FrozenSet[str] - Variables to intervene on
- `values`: pyr.PMap - Intervention values (type-specific)
- `metadata`: pyr.PMap - Optional additional information

### InterventionHandler
```python
InterventionHandler = Callable[[pyr.PMap, pyr.PMap], pyr.PMap]
```
Function signature: `(scm, intervention) -> modified_scm`

## Registry Functions

### register_intervention_handler(intervention_type, handler)
Register a handler function for an intervention type.

**Parameters:**
- intervention_type: str - Identifier for the intervention type
- handler: InterventionHandler - Function to handle this intervention type

**Raises:**
- ValueError: If intervention_type is invalid or handler is not callable

**Example:**
```python
def my_handler(scm, intervention):
    # Apply custom intervention logic
    return modified_scm

register_intervention_handler('my_intervention', my_handler)
```

### apply_intervention(scm, intervention)
Apply an intervention to an SCM using the registered handler.

**Parameters:**
- scm: pyr.PMap - Original structural causal model
- intervention: InterventionSpec - Intervention specification

**Returns:**
pyr.PMap - Modified SCM with intervention applied

**Raises:**
- ValueError: If intervention is invalid or no handler exists

**Example:**
```python
intervention = create_perfect_intervention(
    targets=frozenset(['X']),
    values={'X': 1.0}
)
modified_scm = apply_intervention(scm, intervention)
```

### get_intervention_handler(intervention_type)
Get the registered handler function for an intervention type.

**Parameters:**
- intervention_type: str - Intervention type identifier

**Returns:**
InterventionHandler - The registered handler function

**Raises:**
- ValueError: If no handler is registered for the type

### list_intervention_types()
List all registered intervention types.

**Returns:**
List[str] - Sorted list of intervention type names

**Example:**
```python
types = list_intervention_types()
print(f"Available interventions: {types}")  # ['perfect']
```

### validate_intervention_spec(intervention)
Validate that an intervention specification is well-formed.

**Parameters:**
- intervention: InterventionSpec - Intervention to validate

**Returns:**
bool - True if valid, False otherwise

### validate_intervention_against_scm(scm, intervention)
Validate that an intervention is compatible with a specific SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- intervention: InterventionSpec - Intervention specification

**Returns:**
bool - True if compatible, False otherwise

## Intervention Handlers

### Perfect Intervention

#### perfect_intervention_handler(scm, intervention)
Apply perfect intervention by replacing mechanisms with constants.

A perfect intervention completely replaces the mechanism of target variables with deterministic constant values, breaking all causal links from parents.

**Intervention spec requires:**
- `type`: 'perfect'
- `targets`: FrozenSet[str] - Variables to intervene on
- `values`: Dict[str, Any] - Constant values for each target

**Example:**
```python
# Intervention that sets X=1.0 regardless of parents
intervention = {
    'type': 'perfect',
    'targets': frozenset(['X']),
    'values': {'X': 1.0}
}
```

#### create_perfect_intervention(targets, values, metadata=None)
Factory function to create a perfect intervention specification.

**Parameters:**
- targets: FrozenSet[str] - Variables to intervene on
- values: Dict[str, Any] - Intervention values
- metadata: Optional[Dict[str, Any]] - Additional metadata

**Returns:**
pyr.PMap - Immutable intervention specification

**Example:**
```python
intervention = create_perfect_intervention(
    targets=frozenset(['X', 'Y']),
    values={'X': 1.0, 'Y': 2.0}
)
```

#### create_single_variable_perfect_intervention(variable, value, metadata=None)
Convenience function for single-variable interventions.

**Parameters:**
- variable: str - Variable to intervene on
- value: Any - Intervention value
- metadata: Optional[Dict[str, Any]] - Additional metadata

**Returns:**
pyr.PMap - Intervention specification

**Example:**
```python
intervention = create_single_variable_perfect_intervention('X', 1.0)
```

### Future Intervention Types

#### Imperfect Intervention (Not Implemented)
Interventions with noise, representing imperfect control.

**Planned features:**
- Target values with added noise
- Configurable noise distributions
- Partial mechanism replacement

#### Soft Intervention (Not Implemented)
Interventions that modify but don't replace mechanisms.

**Planned features:**
- Strength parameter (0 = no effect, 1 = perfect)
- Gradual influence on mechanisms
- Preserves some causal relationships

## Usage Patterns

### Basic Intervention Application
```python
from causal_bayes_opt.interventions import (
    create_perfect_intervention,
    apply_intervention
)

# Create intervention
intervention = create_perfect_intervention(
    targets=frozenset(['treatment']),
    values={'treatment': 1.0}
)

# Apply to SCM
treated_scm = apply_intervention(scm, intervention)

# Sample from intervened SCM
outcomes = sample_from_scm(treated_scm, n_samples=100)
```

### Multiple Variable Intervention
```python
# Intervene on multiple variables simultaneously
intervention = create_perfect_intervention(
    targets=frozenset(['X1', 'X2', 'X3']),
    values={'X1': 0.0, 'X2': 1.0, 'X3': 2.0}
)

modified_scm = apply_intervention(scm, intervention)
```

### Custom Intervention Types
```python
def stochastic_intervention_handler(scm, intervention):
    """Custom handler for stochastic interventions."""
    targets = intervention['targets']
    distributions = intervention['distributions']
    
    # Create stochastic mechanisms
    new_mechanisms = get_mechanisms(scm).copy()
    
    for target in targets:
        dist = distributions[target]
        
        def create_stochastic_mechanism(distribution):
            def mechanism(parent_values, noise_key):
                # Sample from distribution
                return sample_from_distribution(distribution, noise_key)
            return mechanism
        
        new_mechanisms[target] = create_stochastic_mechanism(dist)
    
    # Return modified SCM
    return create_scm(
        variables=get_variables(scm),
        edges=get_edges(scm),
        mechanisms=new_mechanisms
    )

# Register custom handler
register_intervention_handler('stochastic', stochastic_intervention_handler)
```

### Validation Before Application
```python
# Validate intervention before applying
if validate_intervention_against_scm(scm, intervention):
    modified_scm = apply_intervention(scm, intervention)
else:
    print("Intervention incompatible with SCM")
```

## Integration with Other Modules

### With Experience Buffer
```python
# Apply intervention and record outcome
intervention = create_perfect_intervention(
    targets=frozenset(['X']),
    values={'X': 5.0}
)

# Execute intervention
outcome = sample_with_intervention(scm, intervention)

# Add to buffer
buffer.add_intervention(intervention, outcome)
```

### With Acquisition Policy
```python
# Policy suggests intervention
suggested_targets, suggested_values = policy.select_intervention(state)

# Create formal intervention
intervention = create_perfect_intervention(
    targets=suggested_targets,
    values=suggested_values
)

# Apply and observe
modified_scm = apply_intervention(scm, intervention)
```

## Error Handling

### Common Errors

**Invalid Targets:**
```python
# Error: Target not in SCM
intervention = create_perfect_intervention(
    targets=frozenset(['NonExistent']),
    values={'NonExistent': 1.0}
)
# Raises: ValueError: Intervention targets not in SCM
```

**Missing Values:**
```python
# Error: Missing value for target
intervention = create_perfect_intervention(
    targets=frozenset(['X', 'Y']),
    values={'X': 1.0}  # Missing Y
)
# Raises: ValueError: Missing values for targets: ['Y']
```

**Unregistered Type:**
```python
# Error: Handler not registered
intervention = pyr.m(type='unknown', targets=frozenset(['X']))
apply_intervention(scm, intervention)
# Raises: ValueError: No handler registered for intervention type 'unknown'
```

## Best Practices

### 1. Always Validate
```python
# Validate before applying
if not validate_intervention_against_scm(scm, intervention):
    raise ValueError("Invalid intervention for this SCM")
```

### 2. Use Factory Functions
```python
# Good: Use factory function
intervention = create_perfect_intervention(...)

# Avoid: Manual construction
intervention = pyr.m(type='perfect', ...)  # Error-prone
```

### 3. Handle Metadata
```python
# Include relevant metadata
intervention = create_perfect_intervention(
    targets=frozenset(['treatment']),
    values={'treatment': 1.0},
    metadata={
        'reason': 'randomized_trial',
        'timestamp': time.time()
    }
)
```

### 4. Immutability
```python
# Interventions are immutable
intervention1 = create_perfect_intervention(...)
intervention2 = intervention1  # Safe - can't be modified
```

## Performance Considerations

- **Handler Lookup**: O(1) - Dictionary lookup
- **Validation**: O(n) where n is number of targets
- **SCM Modification**: O(v) where v is number of variables
- **Memory**: Interventions are lightweight immutable objects

## Future Extensions

1. **Conditional Interventions**: Interventions that depend on current state
2. **Time-Varying Interventions**: Dynamic intervention values
3. **Multi-Stage Interventions**: Sequential intervention protocols
4. **Probabilistic Interventions**: Interventions with uncertainty