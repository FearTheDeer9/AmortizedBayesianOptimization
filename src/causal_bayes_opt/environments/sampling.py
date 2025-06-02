"""
Intervention-aware sampling functions.

Provides functions for sampling from SCMs under interventions,
building on the core sampling functionality.
"""

# Standard library imports
import logging
from typing import List, Dict, Tuple, Optional, FrozenSet, Any

# Third-party imports
import jax.random as random
import pyrsistent as pyr

# Local imports
from ..data_structures.sample import (
    create_sample, create_interventional_sample, 
    create_perfect_intervention_sample
)
from ..data_structures.scm import get_variables, topological_sort, get_mechanisms
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.registry import apply_intervention
from ..interventions.handlers import create_perfect_intervention

logger = logging.getLogger(__name__)


def sample_with_intervention(
    scm: pyr.PMap,
    intervention: pyr.PMap, 
    n_samples: int,
    seed: int = 42
) -> List[pyr.PMap]:
    """
    Generate samples from an SCM under intervention.
    
    Args:
        scm: Original structural causal model
        intervention: Intervention specification  
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        List of interventional samples
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> intervention = create_perfect_intervention(['X'], {'X': 1.0})
        >>> samples = sample_with_intervention(scm, intervention, 100)
    """
    # Validate inputs
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError(f"n_samples must be a positive integer, got: {n_samples}")
    
    if not isinstance(scm, pyr.PMap):
        raise ValueError("SCM must be a pyrsistent PMap")
    
    if not isinstance(intervention, pyr.PMap):
        raise ValueError("Intervention must be a pyrsistent PMap")
    
    # Apply intervention to get modified SCM
    try:
        modified_scm = apply_intervention(scm, intervention)
    except Exception as e:
        raise ValueError(f"Failed to apply intervention: {e}") from e
    
    # Sample from the modified SCM
    observational_samples = sample_from_linear_scm(modified_scm, n_samples, seed)
    
    # Convert to interventional samples with proper metadata
    intervention_type = intervention['type']
    intervention_targets = frozenset(intervention['targets'])
    
    interventional_samples = []
    for sample in observational_samples:
        # Create interventional sample with intervention metadata
        interventional_sample = create_interventional_sample(
            values=dict(sample['values']),
            intervention_type=intervention_type,
            targets=intervention_targets,
            metadata={
                'original_intervention': intervention,
                'sample_seed': seed
            }
        )
        interventional_samples.append(interventional_sample)
    
    logger.debug(f"Generated {len(interventional_samples)} interventional samples")
    return interventional_samples


def sample_multiple_interventions(
    scm: pyr.PMap,
    interventions: List[pyr.PMap],
    samples_per_intervention: int,
    seed: int = 42
) -> List[Tuple[pyr.PMap, List[pyr.PMap]]]:
    """
    Generate samples for multiple interventions.
    
    Args:
        scm: Original structural causal model
        interventions: List of intervention specifications
        samples_per_intervention: Number of samples to generate per intervention
        seed: Random seed
        
    Returns:
        List of (intervention, samples) pairs
        
    Example:
        >>> interventions = [
        ...     create_perfect_intervention(['X'], {'X': 0.0}),
        ...     create_perfect_intervention(['X'], {'X': 1.0})
        ... ]
        >>> results = sample_multiple_interventions(scm, interventions, 50)
    """
    if not interventions:
        return []
    
    if not isinstance(samples_per_intervention, int) or samples_per_intervention <= 0:
        raise ValueError("samples_per_intervention must be positive")
    
    # Use different seeds for each intervention to ensure independence
    key = random.PRNGKey(seed)
    results = []
    
    for i, intervention in enumerate(interventions):
        # Generate unique seed for this intervention
        key, subkey = random.split(key)
        intervention_seed = int(subkey[0])
        
        # Sample from this intervention
        samples = sample_with_intervention(
            scm, intervention, samples_per_intervention, intervention_seed
        )
        
        results.append((intervention, samples))
        logger.debug(f"Completed intervention {i+1}/{len(interventions)}")
    
    return results


def generate_mixed_dataset(
    scm: pyr.PMap,
    n_observational: int,
    interventions: List[pyr.PMap],
    samples_per_intervention: int,
    seed: int = 42
) -> Tuple[List[pyr.PMap], List[Tuple[pyr.PMap, List[pyr.PMap]]]]:
    """
    Generate a mixed dataset with observational and interventional data.
    
    Args:
        scm: Original structural causal model
        n_observational: Number of observational samples to generate
        interventions: List of intervention specifications
        samples_per_intervention: Number of samples per intervention
        seed: Random seed
        
    Returns:
        Tuple of (observational_samples, interventional_data)
        
    Example:
        >>> interventions = [create_perfect_intervention(['X'], {'X': 1.0})]
        >>> obs_samples, int_data = generate_mixed_dataset(scm, 100, interventions, 50)
    """
    # Split random seed for independent sampling
    key = random.PRNGKey(seed)
    obs_key, int_key = random.split(key)
    
    # Generate observational data
    observational_samples = []
    if n_observational > 0:
        observational_samples = sample_from_linear_scm(
            scm, n_observational, int(obs_key[0])
        )
    
    # Generate interventional data
    interventional_data = []
    if interventions and samples_per_intervention > 0:
        interventional_data = sample_multiple_interventions(
            scm, interventions, samples_per_intervention, int(int_key[0])
        )
    
    logger.info(f"Generated mixed dataset: {len(observational_samples)} observational, "
                f"{len(interventional_data)} intervention types")
    
    return observational_samples, interventional_data


# Batch utilities
def generate_intervention_batch(
    scm: pyr.PMap,
    intervention_specs: List[Dict[str, Any]],  # List of intervention parameters
    batch_size: int,
    seed: int = 42
) -> List[Tuple[pyr.PMap, List[pyr.PMap]]]:
    """
    Generate a batch of interventions with specified parameters.
    
    Args:
        scm: Original structural causal model
        intervention_specs: List of dictionaries with intervention parameters
        batch_size: Number of samples per intervention
        seed: Random seed
        
    Returns:
        List of (intervention, samples) pairs
        
    Example:
        >>> specs = [
        ...     {'type': 'perfect', 'targets': ['X'], 'values': {'X': 0.0}},
        ...     {'type': 'perfect', 'targets': ['X'], 'values': {'X': 1.0}}
        ... ]
        >>> batch = generate_intervention_batch(scm, specs, 50)
    """
    # Convert specs to intervention objects
    interventions = []
    for spec in intervention_specs:
        if spec.get('type') == 'perfect':
            intervention = create_perfect_intervention(
                targets=frozenset(spec['targets']),
                values=spec['values'],
                metadata=spec.get('metadata')
            )
            interventions.append(intervention)
        else:
            raise ValueError(f"Unsupported intervention type: {spec.get('type')}")
    
    return sample_multiple_interventions(scm, interventions, batch_size, seed)


def generate_random_interventions(
    scm: pyr.PMap,
    n_interventions: int,
    intervention_type: str = "perfect",
    samples_per_intervention: int = 1,
    seed: int = 42
) -> List[Tuple[pyr.PMap, List[pyr.PMap]]]:
    """
    Generate random interventions for exploration.
    
    Args:
        scm: Original structural causal model
        n_interventions: Number of different interventions to generate
        intervention_type: Type of interventions to generate
        samples_per_intervention: Number of samples per intervention
        seed: Random seed
        
    Returns:
        List of (intervention, samples) pairs
        
    Note:
        Currently only supports perfect interventions on single variables.
        Values are sampled from a standard normal distribution.
    """
    if intervention_type != "perfect":
        raise ValueError("Only perfect interventions are currently supported")
    
    variables = list(get_variables(scm))
    if not variables:
        raise ValueError("SCM has no variables")
    
    # Generate random interventions
    key = random.PRNGKey(seed)
    interventions = []
    
    for i in range(n_interventions):
        key, var_key, val_key = random.split(key, 3)
        
        # Randomly select a variable to intervene on
        var_idx = random.randint(var_key, (), 0, len(variables))
        target_var = variables[var_idx]
        
        # Generate random intervention value
        intervention_value = float(random.normal(val_key))
        
        # Create intervention
        intervention = create_perfect_intervention(
            targets=frozenset([target_var]),
            values={target_var: intervention_value},
            metadata={'random_intervention_id': i}
        )
        
        interventions.append(intervention)
    
    # Sample from all interventions
    return sample_multiple_interventions(scm, interventions, samples_per_intervention, seed)


# Utility functions for intervention design
def create_intervention_grid(
    variable: str,
    values: List[Any],
    intervention_type: str = "perfect"
) -> List[pyr.PMap]:
    """
    Create a grid of interventions on a single variable.
    
    Args:
        variable: Name of the variable to intervene on
        values: List of values to try
        intervention_type: Type of intervention
        
    Returns:
        List of intervention specifications
        
    Example:
        >>> interventions = create_intervention_grid('X', [0.0, 0.5, 1.0])
    """
    if intervention_type != "perfect":
        raise ValueError("Only perfect interventions are currently supported")
    
    interventions = []
    for value in values:
        intervention = create_perfect_intervention(
            targets=frozenset([variable]),
            values={variable: value}
        )
        interventions.append(intervention)
    
    return interventions


def create_factorial_interventions(
    variables: List[str],
    values_per_variable: Dict[str, List[Any]],
    intervention_type: str = "perfect"
) -> List[pyr.PMap]:
    """
    Create factorial design of interventions across multiple variables.
    
    Args:
        variables: List of variables to intervene on
        values_per_variable: Dictionary mapping variables to lists of values
        intervention_type: Type of intervention
        
    Returns:
        List of intervention specifications covering all combinations
        
    Example:
        >>> interventions = create_factorial_interventions(
        ...     ['X', 'Y'], 
        ...     {'X': [0, 1], 'Y': [0, 1]}
        ... )
        # Creates 4 interventions: (X=0,Y=0), (X=0,Y=1), (X=1,Y=0), (X=1,Y=1)
    """
    if intervention_type != "perfect":
        raise ValueError("Only perfect interventions are currently supported")
    
    import itertools
    
    # Generate all combinations
    value_combinations = itertools.product(*[values_per_variable[var] for var in variables])
    
    interventions = []
    for combination in value_combinations:
        values_dict = dict(zip(variables, combination))
        
        intervention = create_perfect_intervention(
            targets=frozenset(variables),
            values=values_dict
        )
        interventions.append(intervention)
    
    return interventions


# Convenience functions for common intervention patterns
def sample_do_intervention(
    scm: pyr.PMap,
    variable: str,
    value: Any,
    n_samples: int,
    seed: int = 42
) -> List[pyr.PMap]:
    """
    Convenience function for do(variable = value) interventions.
    
    Args:
        scm: Original structural causal model
        variable: Variable to intervene on
        value: Value to set the variable to
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        List of interventional samples
        
    Example:
        >>> samples = sample_do_intervention(scm, 'X', 1.0, 100)
        # Equivalent to sampling from do(X = 1.0)
    """
    intervention = create_perfect_intervention(
        targets=frozenset([variable]),
        values={variable: value}
    )
    
    return sample_with_intervention(scm, intervention, n_samples, seed)


def compare_intervention_effects(
    scm: pyr.PMap,
    interventions: List[pyr.PMap],
    target_variable: str,
    samples_per_intervention: int = 100,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Compare the effects of different interventions on a target variable.
    
    Args:
        scm: Original structural causal model
        interventions: List of intervention specifications
        target_variable: Variable to analyze effects on
        samples_per_intervention: Number of samples per intervention
        seed: Random seed
        
    Returns:
        Dictionary with intervention effects analysis
        
    Example:
        >>> interventions = create_intervention_grid('X', [0.0, 1.0])
        >>> effects = compare_intervention_effects(scm, interventions, 'Y')
    """
    import statistics
    
    results = {}
    intervention_data = sample_multiple_interventions(
        scm, interventions, samples_per_intervention, seed
    )
    
    for i, (intervention, samples) in enumerate(intervention_data):
        # Extract target variable values
        target_values = [sample['values'][target_variable] for sample in samples]
        
        # Compute statistics
        intervention_desc = f"intervention_{i}"
        if intervention['type'] == 'perfect' and len(intervention['targets']) == 1:
            target = list(intervention['targets'])[0]
            value = intervention['values'][target]
            intervention_desc = f"do({target}={value})"
        
        results[intervention_desc] = {
            'intervention': intervention,
            'target_variable': target_variable,
            'mean': statistics.mean(target_values),
            'std': statistics.stdev(target_values) if len(target_values) > 1 else 0.0,
            'min': min(target_values),
            'max': max(target_values),
            'n_samples': len(target_values)
        }
    
    return results
