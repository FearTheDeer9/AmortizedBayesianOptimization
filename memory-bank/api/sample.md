# Immutable Causal Samples API Reference

## Overview
This module provides functions for creating and manipulating immutable sample objects in a causal inference context. Samples represent observations or intervention results within a causal framework, allowing for clear distinctions between observational and interventional data.

## Core Types
- **pyr.PMap**: Persistent (immutable) map from pyrsistent library that represents a sample
- **Dict[str, Any]**: Dictionary mapping variable names to their values
- **FrozenSet[str]**: Immutable set of variable names, particularly for intervention targets

## Core Functions

### create_sample(values, intervention_type=None, intervention_targets=None, metadata=None)
Create an immutable Sample representing a single observation or intervention result.

**Parameters:**
- values: Dict[str, Any] - Mapping from variable names to their values
- intervention_type: Optional[str] - Type of intervention if this is an interventional sample (e.g., 'perfect', 'imperfect', 'soft')
- intervention_targets: Optional[FrozenSet[str]] - Set of variables that were intervened upon
- metadata: Optional[Dict[str, Any]] - Additional metadata about the sample

**Returns:**
pyr.PMap - An immutable Sample representation

**Example:**
```python
import pyrsistent as pyr

# Create an observational sample
obs_sample = create_sample({
    "X": 2.5,
    "Y": 7.1,
    "Z": True
})

# Create an interventional sample (do(X=5))
int_sample = create_sample(
    values={"X": 5.0, "Y": 10.2, "Z": False},
    intervention_type="perfect",
    intervention_targets=pyr.s("X")
)
```

### is_observational(sample)
Check if a sample is observational (not interventional).

**Parameters:**
- sample: pyr.PMap - The sample to check

**Returns:**
bool - True if the sample is observational, False otherwise

**Example:**
```python
# Check if sample is observational
is_obs = is_observational(sample)
print(is_obs)  # True for observational samples, False for interventional
```

### is_interventional(sample)
Check if a sample is interventional.

**Parameters:**
- sample: pyr.PMap - The sample to check

**Returns:**
bool - True if the sample is interventional, False otherwise

**Example:**
```python
# Check if sample is interventional
is_int = is_interventional(sample)
print(is_int)  # True for interventional samples, False for observational
```

### get_values(sample)
Get the variable values from a sample.

**Parameters:**
- sample: pyr.PMap - The sample to get values from

**Returns:**
pyr.PMap - Mapping of variable names to their values

**Example:**
```python
# Get all values from a sample
values = get_values(sample)
print(values)  # pyr.m({'X': 2.5, 'Y': 7.1, 'Z': True})
```

### get_value(sample, variable)
Get the value of a specific variable from a sample.

**Parameters:**
- sample: pyr.PMap - The sample to get the value from
- variable: str - The name of the variable to get

**Returns:**
Any - The value of the specified variable

**Raises:**
- KeyError: If the variable does not exist in the sample

**Example:**
```python
# Get a specific variable value
x_value = get_value(sample, "X")
print(x_value)  # 2.5
```

### with_value(sample, variable, value)
Create a new sample with an updated value for a variable.

**Parameters:**
- sample: pyr.PMap - The original sample
- variable: str - The variable to update
- value: Any - The new value for the variable

**Returns:**
pyr.PMap - A new sample with the updated value

**Example:**
```python
# Create a new sample with an updated value
new_sample = with_value(sample, "X", 3.0)
print(get_value(new_sample, "X"))  # 3.0
print(get_value(sample, "X"))  # Original sample remains unchanged: 2.5
```

### get_intervention_type(sample)
Get the intervention type of a sample.

**Parameters:**
- sample: pyr.PMap - The sample to check

**Returns:**
Optional[str] - The intervention type, or None if observational

**Example:**
```python
# Get intervention type
int_type = get_intervention_type(sample)
print(int_type)  # 'perfect' for sample created with perfect intervention, None for observational
```

### get_intervention_targets(sample)
Get the set of variables that were intervened upon.

**Parameters:**
- sample: pyr.PMap - The sample to check

**Returns:**
FrozenSet[str] - Set of variables that were intervened upon

**Example:**
```python
# Get intervention targets
targets = get_intervention_targets(sample)
print(targets)  # frozenset({'X'}) for sample with intervention on X, empty set for observational
```

### with_metadata(sample, key, value)
Create a new sample with updated metadata.

**Parameters:**
- sample: pyr.PMap - The original sample
- key: str - The metadata key to update
- value: Any - The metadata value to set

**Returns:**
pyr.PMap - A new sample with the updated metadata

**Example:**
```python
# Add metadata to a sample
sample_with_meta = with_metadata(sample, "timestamp", "2025-05-21T10:30:00")
```

### get_metadata(sample, key=None)
Get metadata from a sample.

**Parameters:**
- sample: pyr.PMap - The sample to get metadata from
- key: Optional[str] - The specific metadata key to get, or None to get all metadata

**Returns:**
Any - The metadata value for the given key, or all metadata if key is None

**Example:**
```python
# Get specific metadata
timestamp = get_metadata(sample_with_meta, "timestamp")
print(timestamp)  # "2025-05-21T10:30:00"

# Get all metadata
all_meta = get_metadata(sample_with_meta)
print(all_meta)  # pyr.m({'timestamp': '2025-05-21T10:30:00'})
```

### merge_samples(sample1, sample2)
Merge two samples, combining their values.

**Parameters:**
- sample1: pyr.PMap - First sample
- sample2: pyr.PMap - Second sample (values from this sample override sample1 if there are conflicts)

**Returns:**
pyr.PMap - A new merged sample

**Note:**
Intervention information is preserved from sample1.

**Example:**
```python
# Merge two samples
sample1 = create_sample({"X": 1, "Y": 2})
sample2 = create_sample({"Y": 3, "Z": 4})
merged = merge_samples(sample1, sample2)
print(get_values(merged))  # pyr.m({'X': 1, 'Y': 3, 'Z': 4})
```

### filter_variables(sample, variables)
Create a new sample containing only the specified variables.

**Parameters:**
- sample: pyr.PMap - The original sample
- variables: FrozenSet[str] - Set of variables to keep

**Returns:**
pyr.PMap - A new sample with only the specified variables

**Example:**
```python
# Filter a sample to keep only specific variables
sample = create_sample({"X": 1, "Y": 2, "Z": 3})
filtered = filter_variables(sample, pyr.s("X", "Z"))
print(get_values(filtered))  # pyr.m({'X': 1, 'Z': 3})
```

## Batch Operations

### create_batch_samples(data_list)
Create multiple samples from a list of data dictionaries.

**Parameters:**
- data_list: List[Dict[str, Any]] - List of dictionaries with variable values

**Returns:**
List[pyr.PMap] - List of sample objects

**Example:**
```python
# Create multiple samples from a list of data dictionaries
data = [
    {"X": 1, "Y": 2},
    {"X": 3, "Y": 4},
    {"X": 5, "Y": 6}
]
samples = create_batch_samples(data)
print(len(samples))  # 3
```

### filter_samples_by_condition(samples, condition)
Filter a list of samples based on a condition function.

**Parameters:**
- samples: List[pyr.PMap] - List of samples to filter
- condition: Callable[[pyr.PMap], bool] - Function that takes a sample and returns a boolean

**Returns:**
List[pyr.PMap] - Filtered list of samples

**Example:**
```python
# Filter samples where X > 2
samples = create_batch_samples([
    {"X": 1, "Y": 10},
    {"X": 3, "Y": 30},
    {"X": 5, "Y": 50}
])
filtered = filter_samples_by_condition(samples, lambda s: get_value(s, "X") > 2)
print(len(filtered))  # 2
```

### get_observational_samples(samples)
Filter a list of samples to only include observational samples.

**Parameters:**
- samples: List[pyr.PMap] - List of samples

**Returns:**
List[pyr.PMap] - List containing only observational samples

**Example:**
```python
# Get only observational samples
mixed_samples = [
    create_sample({"X": 1, "Y": 2}),  # Observational
    create_sample({"X": 3, "Y": 4}, "perfect", pyr.s("X"))  # Interventional
]
obs_samples = get_observational_samples(mixed_samples)
print(len(obs_samples))  # 1
```

### get_interventional_samples(samples)
Filter a list of samples to only include interventional samples.

**Parameters:**
- samples: List[pyr.PMap] - List of samples

**Returns:**
List[pyr.PMap] - List containing only interventional samples

**Example:**
```python
# Get only interventional samples
mixed_samples = [
    create_sample({"X": 1, "Y": 2}),  # Observational
    create_sample({"X": 3, "Y": 4}, "perfect", pyr.s("X"))  # Interventional
]
int_samples = get_interventional_samples(mixed_samples)
print(len(int_samples))  # 1
```

### get_samples_with_intervention_on(samples, variable)
Filter a list of samples to only include those with interventions on a specific variable.

**Parameters:**
- samples: List[pyr.PMap] - List of samples
- variable: str - Variable name to check for interventions

**Returns:**
List[pyr.PMap] - List of samples with interventions on the specified variable

**Example:**
```python
# Get samples with intervention on a specific variable
samples = [
    create_sample({"X": 1, "Y": 2}),  # Observational
    create_sample({"X": 3, "Y": 4}, "perfect", pyr.s("X")),  # Intervention on X
    create_sample({"X": 5, "Y": 6}, "perfect", pyr.s("Y"))   # Intervention on Y
]
x_int_samples = get_samples_with_intervention_on(samples, "X")
print(len(x_int_samples))  # 1
```

### aggregate_variable_values(samples, variable, aggregation_fn)
Aggregate the values of a specific variable across samples.

**Parameters:**
- samples: List[pyr.PMap] - List of samples to aggregate over
- variable: str - Variable name to aggregate
- aggregation_fn: Callable[[List[Any]], Any] - Function to apply to the list of variable values

**Returns:**
Any - Result of the aggregation function

**Example:**
```python
import statistics

# Calculate the mean of variable Y across samples
samples = create_batch_samples([
    {"X": 1, "Y": 10},
    {"X": 2, "Y": 20},
    {"X": 3, "Y": 30}
])
y_mean = aggregate_variable_values(samples, "Y", statistics.mean)
print(y_mean)  # 20.0
```