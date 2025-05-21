import pyrsistent as pyr
from typing import Dict, Any, Optional, FrozenSet, List, Callable, TypeVar, Mapping

def create_sample(
    values: Dict[str, Any],
    intervention_type: Optional[str] = None,
    intervention_targets: Optional[FrozenSet[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> pyr.PMap:
    """
    Create an immutable Sample representing a single observation or intervention result.
    
    Parameters:
    -----------
    values : Dict[str, Any]
        Mapping from variable names to their values
    intervention_type : Optional[str]
        Type of intervention if this is an interventional sample (e.g., 'perfect', 'imperfect', 'soft')
    intervention_targets : Optional[FrozenSet[str]]
        Set of variables that were intervened upon
    metadata : Optional[Dict[str, Any]]
        Additional metadata about the sample
        
    Returns:
    --------
    pyr.PMap
        An immutable Sample representation
    """
    return pyr.m({
        'values': pyr.m(values),
        'intervention_type': intervention_type,
        'intervention_targets': intervention_targets if intervention_targets is not None else pyr.s(),
        'metadata': pyr.m(metadata) if metadata is not None else pyr.m()
    })

def is_observational(sample: pyr.PMap) -> bool:
    """Check if a sample is observational (not interventional)."""
    return sample['intervention_type'] is None

def is_interventional(sample: pyr.PMap) -> bool:
    """Check if a sample is interventional."""
    return not is_observational(sample)

def get_values(sample: pyr.PMap) -> pyr.PMap:
    """Get the variable values from a sample."""
    return sample['values']

def get_value(sample: pyr.PMap, variable: str) -> Any:
    """
    Get the value of a specific variable from a sample.
    
    Raises KeyError if the variable does not exist in the sample.
    """
    return sample['values'][variable]

def with_value(sample: pyr.PMap, variable: str, value: Any) -> pyr.PMap:
    """Create a new sample with an updated value for a variable."""
    return sample.set('values', sample['values'].set(variable, value))

def get_intervention_type(sample: pyr.PMap) -> Optional[str]:
    """Get the intervention type of a sample."""
    return sample['intervention_type']

def get_intervention_targets(sample: pyr.PMap) -> FrozenSet[str]:
    """Get the set of variables that were intervened upon."""
    return sample['intervention_targets']

def with_metadata(sample: pyr.PMap, key: str, value: Any) -> pyr.PMap:
    """Create a new sample with updated metadata."""
    return sample.set('metadata', sample['metadata'].set(key, value))

def get_metadata(sample: pyr.PMap, key: Optional[str] = None) -> Any:
    """
    Get metadata from a sample.
    
    Parameters:
    -----------
    sample : pyr.PMap
        The sample to get metadata from
    key : Optional[str]
        The specific metadata key to get, or None to get all metadata
        
    Returns:
    --------
    Any
        The metadata value for the given key, or all metadata if key is None
    """
    if key is None:
        return sample['metadata']
    return sample['metadata'][key]

def merge_samples(sample1: pyr.PMap, sample2: pyr.PMap) -> pyr.PMap:
    """
    Merge two samples, combining their values.
    
    Values from sample2 override values from sample1 if there are conflicts.
    Intervention information is preserved from sample1.
    """
    merged_values = dict(sample1['values'])
    merged_values.update(dict(sample2['values']))
    
    return create_sample(
        values=merged_values,
        intervention_type=sample1['intervention_type'],
        intervention_targets=sample1['intervention_targets'],
        metadata=sample1['metadata']
    )

def filter_variables(sample: pyr.PMap, variables: FrozenSet[str]) -> pyr.PMap:
    """Create a new sample containing only the specified variables."""
    filtered_values = {var: val for var, val in sample['values'].items() 
                      if var in variables}
    
    # Filter intervention targets as well
    filtered_targets = sample['intervention_targets'] & variables
    
    return create_sample(
        values=filtered_values,
        intervention_type=sample['intervention_type'],
        intervention_targets=filtered_targets,
        metadata=sample['metadata']
    )

# Batch operations for collections of samples
def create_batch_samples(data_list: List[Dict[str, Any]]) -> List[pyr.PMap]:
    """Create multiple samples from a list of data dictionaries."""
    return [create_sample(values=d) for d in data_list]

def filter_samples_by_condition(
    samples: List[pyr.PMap], 
    condition: Callable[[pyr.PMap], bool]
) -> List[pyr.PMap]:
    """Filter a list of samples based on a condition function."""
    return [s for s in samples if condition(s)]

def get_observational_samples(samples: List[pyr.PMap]) -> List[pyr.PMap]:
    """Filter a list of samples to only include observational samples."""
    return filter_samples_by_condition(samples, is_observational)

def get_interventional_samples(samples: List[pyr.PMap]) -> List[pyr.PMap]:
    """Filter a list of samples to only include interventional samples."""
    return filter_samples_by_condition(samples, is_interventional)

def get_samples_with_intervention_on(
    samples: List[pyr.PMap], 
    variable: str
) -> List[pyr.PMap]:
    """
    Filter a list of samples to only include those with interventions 
    on a specific variable.
    """
    return filter_samples_by_condition(
        samples, 
        lambda s: variable in s['intervention_targets']
    )

def aggregate_variable_values(
    samples: List[pyr.PMap],
    variable: str,
    aggregation_fn: Callable[[List[Any]], Any]
) -> Any:
    """
    Aggregate the values of a specific variable across samples.
    
    Example: aggregate_variable_values(samples, 'Y', statistics.mean)
    """
    values = [get_value(sample, variable) for sample in samples]
    return aggregation_fn(values)