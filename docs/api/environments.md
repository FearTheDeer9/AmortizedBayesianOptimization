# Environments API Reference

## Overview
The environments module provides intervention-aware sampling functions that bridge between SCMs, interventions, and samples. It builds on the core sampling functionality to provide utilities for generating data under various intervention scenarios.

## Core Functions

### sample_with_intervention(scm, intervention, n_samples, seed=42)
Generate samples from an SCM under intervention.

**Parameters:**
- scm: pyr.PMap - Original structural causal model
- intervention: pyr.PMap - Intervention specification
- n_samples: int - Number of samples to generate
- seed: int - Random seed for reproducibility

**Returns:**
List[pyr.PMap] - List of interventional samples with proper metadata

**Raises:**
- ValueError: If inputs are invalid or intervention cannot be applied

**Example:**
```python
intervention = create_perfect_intervention(
    targets=frozenset(['X']),
    values={'X': 1.0}
)
samples = sample_with_intervention(scm, intervention, 100)
```

### sample_multiple_interventions(scm, interventions, samples_per_intervention, seed=42)
Generate samples for multiple interventions efficiently.

**Parameters:**
- scm: pyr.PMap - Original structural causal model
- interventions: List[pyr.PMap] - List of intervention specifications
- samples_per_intervention: int - Samples to generate per intervention
- seed: int - Random seed

**Returns:**
List[Tuple[pyr.PMap, List[pyr.PMap]]] - List of (intervention, samples) pairs

**Example:**
```python
interventions = [
    create_perfect_intervention(['X'], {'X': 0.0}),
    create_perfect_intervention(['X'], {'X': 1.0})
]
results = sample_multiple_interventions(scm, interventions, 50)

for intervention, samples in results:
    print(f"Intervention: {intervention['values']}")
    print(f"Generated {len(samples)} samples")
```

### generate_mixed_dataset(scm, n_observational, interventions, samples_per_intervention, seed=42)
Generate a mixed dataset with both observational and interventional data.

**Parameters:**
- scm: pyr.PMap - Original structural causal model
- n_observational: int - Number of observational samples
- interventions: List[pyr.PMap] - List of interventions
- samples_per_intervention: int - Samples per intervention
- seed: int - Random seed

**Returns:**
Tuple[List[pyr.PMap], List[Tuple[pyr.PMap, List[pyr.PMap]]]] - (observational_samples, interventional_data)

**Example:**
```python
interventions = [create_perfect_intervention(['X'], {'X': 1.0})]
obs_samples, int_data = generate_mixed_dataset(
    scm, 
    n_observational=100,
    interventions=interventions,
    samples_per_intervention=50
)
```

## Batch Generation Utilities

### generate_intervention_batch(scm, intervention_specs, batch_size, seed=42)
Generate a batch of interventions from specifications.

**Parameters:**
- scm: pyr.PMap - Original SCM
- intervention_specs: List[Dict[str, Any]] - List of intervention parameters
- batch_size: int - Samples per intervention
- seed: int - Random seed

**Returns:**
List[Tuple[pyr.PMap, List[pyr.PMap]]] - List of (intervention, samples) pairs

**Example:**
```python
specs = [
    {'type': 'perfect', 'targets': ['X'], 'values': {'X': 0.0}},
    {'type': 'perfect', 'targets': ['X'], 'values': {'X': 1.0}}
]
batch = generate_intervention_batch(scm, specs, 50)
```

### generate_random_interventions(scm, n_interventions, intervention_type="perfect", samples_per_intervention=1, seed=42)
Generate random interventions for exploration.

**Parameters:**
- scm: pyr.PMap - Original SCM
- n_interventions: int - Number of different interventions
- intervention_type: str - Type of interventions (currently only "perfect")
- samples_per_intervention: int - Samples per intervention
- seed: int - Random seed

**Returns:**
List[Tuple[pyr.PMap, List[pyr.PMap]]] - Random intervention results

**Note:**
Values are sampled from standard normal distribution. Currently only supports single-variable perfect interventions.

**Example:**
```python
# Generate 10 random interventions
random_results = generate_random_interventions(scm, 10, samples_per_intervention=25)
```

## Intervention Design Utilities

### create_intervention_grid(variable, values, intervention_type="perfect")
Create a grid of interventions on a single variable.

**Parameters:**
- variable: str - Variable to intervene on
- values: List[Any] - Values to try
- intervention_type: str - Type of intervention

**Returns:**
List[pyr.PMap] - List of intervention specifications

**Example:**
```python
# Test effect of X at different levels
interventions = create_intervention_grid('X', [0.0, 0.5, 1.0, 1.5, 2.0])
```

### create_factorial_interventions(variables, values_per_variable, intervention_type="perfect")
Create factorial design of interventions across multiple variables.

**Parameters:**
- variables: List[str] - Variables to intervene on
- values_per_variable: Dict[str, List[Any]] - Values for each variable
- intervention_type: str - Type of intervention

**Returns:**
List[pyr.PMap] - All combinations of interventions

**Example:**
```python
# 2x2 factorial design
interventions = create_factorial_interventions(
    variables=['treatment', 'dose'],
    values_per_variable={
        'treatment': [0, 1],
        'dose': ['low', 'high']
    }
)
# Creates 4 interventions: all combinations
```

## Convenience Functions

### sample_do_intervention(scm, variable, value, n_samples, seed=42)
Convenience function for do(variable = value) interventions.

**Parameters:**
- scm: pyr.PMap - Original SCM
- variable: str - Variable to intervene on
- value: Any - Intervention value
- n_samples: int - Number of samples
- seed: int - Random seed

**Returns:**
List[pyr.PMap] - Interventional samples

**Example:**
```python
# Sample from do(X = 1.0)
samples = sample_do_intervention(scm, 'X', 1.0, 100)
```

### compare_intervention_effects(scm, interventions, target_variable, samples_per_intervention=100, seed=42)
Compare the effects of different interventions on a target variable.

**Parameters:**
- scm: pyr.PMap - Original SCM
- interventions: List[pyr.PMap] - Interventions to compare
- target_variable: str - Variable to analyze effects on
- samples_per_intervention: int - Samples per intervention
- seed: int - Random seed

**Returns:**
Dict[str, Dict[str, Any]] - Analysis results for each intervention

**Example:**
```python
interventions = create_intervention_grid('X', [0.0, 1.0, 2.0])
effects = compare_intervention_effects(scm, interventions, 'Y')

for desc, stats in effects.items():
    print(f"{desc}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

**Output structure:**
```python
{
    'do(X=0.0)': {
        'intervention': <intervention_spec>,
        'target_variable': 'Y',
        'mean': 2.5,
        'std': 0.8,
        'min': 0.9,
        'max': 4.2,
        'n_samples': 100
    },
    ...
}
```

## Usage Patterns

### Basic Intervention Sampling
```python
# Create and apply a single intervention
intervention = create_perfect_intervention(
    targets=frozenset(['treatment']),
    values={'treatment': 1.0}
)

# Sample outcomes
outcomes = sample_with_intervention(scm, intervention, n_samples=100)

# Analyze results
treatment_effects = [s['values']['outcome'] for s in outcomes]
```

### A/B Testing Simulation
```python
# Define treatment conditions
control = create_perfect_intervention(['treatment'], {'treatment': 0})
treatment = create_perfect_intervention(['treatment'], {'treatment': 1})

# Run experiment
results = sample_multiple_interventions(
    scm, 
    interventions=[control, treatment],
    samples_per_intervention=1000
)

# Compare outcomes
control_outcomes = [s['values']['outcome'] for _, samples in results 
                   for s in samples if s['intervention_targets'] == frozenset()]
treatment_outcomes = [s['values']['outcome'] for _, samples in results
                     for s in samples if s['intervention_targets'] == frozenset(['treatment'])]
```

### Dose-Response Analysis
```python
# Create dose levels
doses = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
interventions = create_intervention_grid('dose', doses)

# Generate data
dose_response_data = sample_multiple_interventions(
    scm, interventions, samples_per_intervention=200
)

# Analyze response curve
import numpy as np
dose_levels = []
mean_responses = []

for intervention, samples in dose_response_data:
    dose = intervention['values']['dose']
    responses = [s['values']['response'] for s in samples]
    
    dose_levels.append(dose)
    mean_responses.append(np.mean(responses))
```

### Exploratory Intervention Search
```python
# Random exploration
exploration_data = generate_random_interventions(
    scm,
    n_interventions=50,
    samples_per_intervention=10
)

# Find best intervention
best_intervention = None
best_outcome = float('-inf')

for intervention, samples in exploration_data:
    mean_outcome = np.mean([s['values']['target'] for s in samples])
    
    if mean_outcome > best_outcome:
        best_outcome = mean_outcome
        best_intervention = intervention

print(f"Best intervention: {best_intervention['values']}")
print(f"Mean outcome: {best_outcome}")
```

### Multi-Factor Experiments
```python
# Define factors and levels
factors = ['temperature', 'pressure', 'catalyst']
levels = {
    'temperature': [100, 150, 200],
    'pressure': [1.0, 2.0, 3.0],
    'catalyst': [0, 1]  # absent/present
}

# Create full factorial design
factorial_interventions = create_factorial_interventions(factors, levels)
print(f"Total conditions: {len(factorial_interventions)}")  # 3x3x2 = 18

# Run experiment
results = sample_multiple_interventions(
    scm, factorial_interventions, samples_per_intervention=30
)
```

## Integration Examples

### With Experience Buffer
```python
buffer = ExperienceBuffer()

# Add observational data
obs_samples = sample_from_linear_scm(scm, 100)
for sample in obs_samples:
    buffer.add_observation(sample)

# Add interventional data
interventions = create_intervention_grid('X', [0, 1, 2])
for intervention in interventions:
    outcomes = sample_with_intervention(scm, intervention, 50)
    for outcome in outcomes:
        buffer.add_intervention(intervention, outcome)
```

### With Acquisition Policy
```python
# Policy suggests intervention
state = get_current_state(buffer)
suggested_variable, suggested_value = policy.select_intervention(state)

# Execute intervention
outcomes = sample_do_intervention(
    scm, suggested_variable, suggested_value, n_samples=10
)

# Update buffer with results
intervention = create_perfect_intervention(
    targets=frozenset([suggested_variable]),
    values={suggested_variable: suggested_value}
)
for outcome in outcomes:
    buffer.add_intervention(intervention, outcome)
```

## Performance Considerations

- **Batch sampling**: Use `sample_multiple_interventions()` for efficiency
- **Random seeds**: Different seeds ensure independent samples
- **Memory usage**: Large factorial designs can consume significant memory
- **Parallelization**: Multiple interventions can be sampled in parallel

## Best Practices

### 1. Use Appropriate Sample Sizes
```python
# Observational data: larger samples for baseline
obs_samples = sample_from_linear_scm(scm, n_samples=1000)

# Interventional data: smaller samples per condition
int_samples = sample_with_intervention(scm, intervention, n_samples=50)
```

### 2. Validate Interventions First
```python
# Check intervention validity
if validate_intervention_against_scm(scm, intervention):
    samples = sample_with_intervention(scm, intervention, 100)
else:
    raise ValueError("Invalid intervention for SCM")
```

### 3. Track Intervention Metadata
```python
# Include experiment metadata
intervention = create_perfect_intervention(
    targets=frozenset(['X']),
    values={'X': 1.0},
    metadata={
        'experiment_id': 'exp_001',
        'timestamp': time.time(),
        'purpose': 'dose_response'
    }
)
```

### 4. Handle Edge Cases
```python
# Empty interventions
if not interventions:
    return sample_from_linear_scm(scm, n_samples)  # Fall back to observational

# Zero samples
if n_samples == 0:
    return []  # Return empty list
```

## Limitations

- Currently only supports perfect interventions
- Random interventions use standard normal distribution
- No support for time-varying interventions
- Factorial designs can be memory-intensive for many factors