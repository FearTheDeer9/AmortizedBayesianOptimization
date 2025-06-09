# Experience Buffer API Reference

## Overview
This module provides a mutable, performance-optimized experience buffer for storing observational and interventional data. The buffer is designed for efficient append-only operations and fast querying, making it ideal for reinforcement learning training loops.

## Design Philosophy
The ExperienceBuffer follows selective mutability principles - it's mutable internally for performance in training loops but provides immutable views for safety. This balances the performance requirements of RL training with the safety benefits of functional programming.

## Core Types
- **Sample**: pyr.PMap - Immutable sample representation (from sample.py)
- **Intervention**: pyr.PMap - Immutable intervention specification
- **BufferStatistics**: @dataclass - Statistics about buffer contents

## BufferStatistics
```python
@dataclass
class BufferStatistics:
    """Statistics about the buffer contents."""
    total_samples: int
    num_observations: int
    num_interventions: int
    unique_variables: int
    unique_intervention_types: int
    unique_intervention_targets: int
    creation_time: float
    last_update_time: float
```

## ExperienceBuffer Class

### Overview
The ExperienceBuffer is a mutable, append-only buffer optimized for:
- Fast appends (O(1))
- Efficient querying by variable sets
- Batch processing for neural networks
- Memory efficiency

### Constructor
```python
def __init__(self)
```
Initialize an empty experience buffer with internal indices for fast queries.

### Core Operations

#### add_observation(sample)
Add an observational sample to the buffer.

**Parameters:**
- sample: Sample - Observational sample to add

**Raises:**
- ValueError: If sample is not observational

**Example:**
```python
buffer = ExperienceBuffer()
obs_sample = create_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0})
buffer.add_observation(obs_sample)
```

#### add_intervention(intervention, outcome)
Add an intervention-outcome pair to the buffer.

**Parameters:**
- intervention: Intervention - Intervention specification
- outcome: Sample - Sample resulting from the intervention

**Raises:**
- ValueError: If outcome is not interventional or inconsistent with intervention

**Example:**
```python
intervention = create_perfect_intervention(
    targets=frozenset(['X']),
    values={'X': 5.0}
)
outcome = create_sample(
    values={'X': 5.0, 'Y': 10.0, 'Z': 15.0},
    intervention_type='perfect',
    intervention_targets=frozenset(['X'])
)
buffer.add_intervention(intervention, outcome)
```

### Query Operations

#### get_observations()
Get all observational samples (returns a copy for safety).

**Returns:**
List[Sample] - Copy of all observational samples

#### get_interventions()
Get all intervention-outcome pairs (returns a copy for safety).

**Returns:**
List[Tuple[Intervention, Sample]] - Copy of all intervention-outcome pairs

#### get_all_samples()
Get all samples (observational + intervention outcomes) combined.

**Returns:**
List[Sample] - All samples in the buffer

**Example:**
```python
all_samples = buffer.get_all_samples()
print(f"Total samples: {len(all_samples)}")
```

### Filtering Operations

#### filter_by_variables(variables)
Create a filtered view of the buffer containing only samples with specified variables.

**Parameters:**
- variables: FrozenSet[str] - Set of variable names to filter by

**Returns:**
ExperienceBuffer - New buffer containing only matching samples

**Example:**
```python
# Get samples containing variables X and Y
filtered = buffer.filter_by_variables(frozenset(['X', 'Y']))
```

#### filter_interventions_by_targets(targets)
Get intervention-outcome pairs that target specific variables.

**Parameters:**
- targets: FrozenSet[str] - Set of target variable names

**Returns:**
List[Tuple[Intervention, Sample]] - Matching intervention-outcome pairs

**Example:**
```python
# Get all interventions on variable X
x_interventions = buffer.filter_interventions_by_targets(frozenset(['X']))
```

#### filter_interventions_by_type(intervention_type)
Get intervention-outcome pairs of a specific type.

**Parameters:**
- intervention_type: str - Type of intervention (e.g., 'perfect', 'soft')

**Returns:**
List[Tuple[Intervention, Sample]] - Matching intervention-outcome pairs

### Batch Processing

#### get_observation_batch(indices)
Get a batch of observational samples by indices.

**Parameters:**
- indices: List[int] - List of observation indices

**Returns:**
List[Sample] - Requested observational samples

**Raises:**
- IndexError: If any index is out of range

#### get_intervention_batch(indices)
Get a batch of intervention-outcome pairs by indices.

**Parameters:**
- indices: List[int] - List of intervention indices

**Returns:**
List[Tuple[Intervention, Sample]] - Requested intervention-outcome pairs

**Raises:**
- IndexError: If any index is out of range

#### batch_iterator(batch_size, include_interventions=True)
Iterate over all samples in batches.

**Parameters:**
- batch_size: int - Size of each batch
- include_interventions: bool - Whether to include intervention outcomes

**Yields:**
List[Sample] - Batches of samples

**Example:**
```python
# Process samples in batches of 32
for batch in buffer.batch_iterator(batch_size=32):
    # Process batch for neural network training
    process_batch(batch)
```

### Statistics and Metadata

#### size()
Get total number of samples (observations + interventions).

**Returns:**
int - Total sample count

#### num_observations()
Get number of observational samples.

**Returns:**
int - Observational sample count

#### num_interventions()
Get number of intervention-outcome pairs.

**Returns:**
int - Intervention count

#### get_variable_coverage()
Get set of all variables that appear in any sample.

**Returns:**
FrozenSet[str] - All variables in the buffer

#### get_intervention_types()
Get set of all intervention types in the buffer.

**Returns:**
FrozenSet[str] - All intervention types

#### get_intervention_targets_coverage()
Get set of all variables that have been intervention targets.

**Returns:**
FrozenSet[str] - All targeted variables

#### get_statistics()
Get comprehensive statistics about the buffer.

**Returns:**
BufferStatistics - Detailed buffer statistics

**Example:**
```python
stats = buffer.get_statistics()
print(f"Total samples: {stats.total_samples}")
print(f"Observations: {stats.num_observations}")
print(f"Interventions: {stats.num_interventions}")
print(f"Unique variables: {stats.unique_variables}")
```

### Validation and Debugging

#### validate_consistency()
Validate internal consistency of the buffer.

**Returns:**
bool - True if buffer is internally consistent

#### summary()
Get a human-readable summary of the buffer contents.

**Returns:**
Dict[str, Any] - Dictionary with summary information

**Example:**
```python
summary = buffer.summary()
print(f"Buffer created at: {summary['created_at']}")
print(f"Variable coverage: {summary['variable_coverage']}")
print(f"Is consistent: {summary['is_consistent']}")
```

### Special Methods

#### __len__()
Support for `len()` operation.

**Example:**
```python
print(f"Buffer size: {len(buffer)}")
```

#### __repr__()
String representation for debugging.

**Example:**
```python
print(buffer)
# ExperienceBuffer(observations=100, interventions=50, variables=5)
```

## Factory Functions

### create_empty_buffer()
Create an empty experience buffer.

**Returns:**
ExperienceBuffer - New empty buffer

**Example:**
```python
buffer = create_empty_buffer()
```

### create_buffer_from_samples(observations, interventions=None)
Create a buffer and populate it with existing samples.

**Parameters:**
- observations: List[Sample] - List of observational samples
- interventions: Optional[List[Tuple[Intervention, Sample]]] - Optional intervention-outcome pairs

**Returns:**
ExperienceBuffer - Populated buffer

**Example:**
```python
# Create buffer from existing data
obs_samples = [create_sample({...}) for _ in range(100)]
int_pairs = [(intervention, outcome) for _ in range(50)]

buffer = create_buffer_from_samples(
    observations=obs_samples,
    interventions=int_pairs
)
```

## Usage Patterns

### Training Loop Integration
```python
# Initialize buffer with observational data
buffer = create_empty_buffer()
for sample in initial_observations:
    buffer.add_observation(sample)

# Training loop
for episode in range(num_episodes):
    # Select intervention based on current buffer
    intervention = select_intervention(buffer)
    
    # Execute intervention and observe outcome
    outcome = execute_intervention(scm, intervention)
    
    # Add to buffer
    buffer.add_intervention(intervention, outcome)
    
    # Batch training
    for batch in buffer.batch_iterator(batch_size=32):
        train_model(batch)
```

### Efficient Querying
```python
# Get all interventions on specific variables
x_interventions = buffer.filter_interventions_by_targets(frozenset(['X']))
y_interventions = buffer.filter_interventions_by_targets(frozenset(['Y']))

# Analyze intervention types
for int_type in buffer.get_intervention_types():
    type_interventions = buffer.filter_interventions_by_type(int_type)
    print(f"{int_type}: {len(type_interventions)} interventions")
```

### Buffer Analysis
```python
# Monitor buffer growth and coverage
stats = buffer.get_statistics()
coverage = buffer.get_variable_coverage()
target_coverage = buffer.get_intervention_targets_coverage()

print(f"Variable coverage: {len(coverage)}/{total_variables}")
print(f"Target coverage: {len(target_coverage)}/{total_variables}")
print(f"Observation/Intervention ratio: {stats.num_observations}/{stats.num_interventions}")
```

## Performance Characteristics

- **Add operations**: O(1) amortized time
- **Query by variables**: O(1) index lookup + O(k) result construction
- **Filter operations**: O(n) where n is number of samples
- **Batch iteration**: O(n) with efficient chunking
- **Memory usage**: O(n) where n is total samples

## Design Decisions

### Why Mutable?
The buffer is mutable internally for performance reasons:
- RL training requires frequent appends
- Neural network training needs efficient batch access
- Index updates would be expensive with immutability

### Safety Features
Despite mutability, the buffer provides safety through:
- Returning copies of internal data
- Validation of inputs
- Consistency checking
- Immutable sample representations

### Indexing Strategy
Multiple indices are maintained for fast queries:
- Observations by variable sets
- Interventions by target variables
- Interventions by type

This trades memory for query speed, which is optimal for RL training scenarios.