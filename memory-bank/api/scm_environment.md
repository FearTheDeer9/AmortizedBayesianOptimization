## File 6: `docs/api/scm_environment.md`

```markdown
# SCM Environment API

## Overview

The SCM environment executes interventions and returns outcomes according to the underlying causal model.

## Core Data Structures

```python
@dataclass(frozen=True)
class SCM:
    variables: FrozenSet[str]
    edges: FrozenSet[Tuple[str, str]]
    target: str
    mechanisms: Mapping[str, Callable]  # Node -> mechanism function
    
@dataclass(frozen=True)
class Sample:
    values: Mapping[str, float]  # Variable name -> value

## Intervention Functions
python# Function registry for intervention handlers
intervention_handlers = {}

def register_intervention_handler(intervention_type: str):
    """Decorator to register intervention handlers in the registry"""
    def decorator(handler_fn):
        intervention_handlers[intervention_type] = handler_fn
        return handler_fn
    return decorator

@register_intervention_handler("perfect")
def apply_perfect_intervention(
    scm: SCM, 
    variables: FrozenSet[str], 
    values: Mapping[str, Any], 
    **_
) -> SCM:
    """
    Apply a perfect intervention to an SCM
    
    Args:
        scm: Original SCM
        variables: Variables to intervene on
        values: Intervention values
        
    Returns:
        New SCM with intervention applied
    """
## Core Functions
pythondef apply_intervention(
    scm: SCM, 
    intervention: InterventionSpec
) -> SCM:
    """
    Apply any intervention to an SCM based on registered handlers
    
    Args:
        scm: Original SCM
        intervention: Intervention specification
        
    Returns:
        New SCM with intervention applied
    """
    handler = intervention_handlers.get(intervention["type"])
    if not handler:
        raise ValueError(f"No handler registered for intervention type: {intervention['type']}")
    
    # Extract parameters and apply handler
    variables = intervention["variables"]
    params = {k: v for k, v in intervention.items() 
              if k not in ["type", "variables"]}
    
    return handler(scm, variables, **params)

def sample_from_scm(
    scm: SCM,
    intervention: Optional[InterventionSpec] = None
) -> Sample:
    """
    Sample from the SCM with optional intervention
    
    Args:
        scm: The structural causal model
        intervention: Optional intervention to apply before sampling
        
    Returns:
        Sample: Observed values for all variables
    """
## Factory Functions
pythondef create_intervention(
    intervention_type: str, 
    variables: FrozenSet[str], 
    **kwargs
) -> InterventionSpec:
    """
    Generic factory function for creating intervention specifications
    
    Args:
        intervention_type: Type of intervention
        variables: Variables to intervene on
        **kwargs: Additional parameters specific to intervention type
        
    Returns:
        InterventionSpec: Immutable intervention specification
    """
    return pyr.m(
        type=intervention_type,
        variables=variables,
        **kwargs
    )

def perfect_intervention(
    variables, 
    values
):
    """
    Create a perfect intervention specification
    
    Args:
        variables: Variables to intervene on
        values: Intervention values
        
    Returns:
        InterventionSpec for a perfect intervention
    """
    return create_intervention(
        "perfect", 
        pyr.freeze(variables), 
        values=pyr.freeze(values)
    )