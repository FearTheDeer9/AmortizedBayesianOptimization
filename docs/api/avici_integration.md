# AVICI Data Format Bridge API Reference

The AVICI Data Format Bridge is a modular system for converting functional SCM Sample objects to AVICI's expected input format with target conditioning support. The system consists of three main modules:

1. **Core Conversion Functions** - Main user-facing conversion functionality
2. **Validation Functions** - Data validation and verification utilities
3. **Analysis and Debugging Utilities** - Analysis tools and debugging support

---

## Core Conversion Functions Module

### Overview
This module contains the main user-facing functions for converting Sample objects to AVICI's expected tensor format with target conditioning. It provides the primary interface for data format conversion.

### Core Types
```python
SampleList = List[pyr.PMap]
VariableOrder = List[str]
AVICIDataBatch = Dict[str, jnp.ndarray]
```

**Constants:**
- `DEFAULT_STANDARDIZATION = "default"` - Default standardization method

### Core Functions

#### samples_to_avici_format(samples, variable_order, target_variable, standardize=True, standardization_type="default")
Convert Sample objects to AVICI's expected input format [N, d, 3].

**Parameters:**
- samples: SampleList - List of Sample objects (observational + interventional)
- variable_order: VariableOrder - Ordered list of variable names for consistent indexing
- target_variable: str - Name of target variable for conditioning
- standardize: bool - Whether to standardize variable values (default: True)
- standardization_type: str - Type of standardization ("default" or "count", default: "default")

**Returns:**
jnp.ndarray - JAX array of shape [N, d, 3] where:
- [:, :, 0] = variable values (standardized if requested)
- [:, :, 1] = intervention indicators (1 if intervened, 0 otherwise)
- [:, :, 2] = target indicators (1 if target variable, 0 otherwise)

**Raises:**
- ValueError: If target_variable not in variable_order
- ValueError: If samples contain variables not in variable_order
- ValueError: If inputs are invalid

**Example:**
```python
# Create sample data
samples = [
    create_observational_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0}),
    create_interventional_sample({'X': 2.0, 'Y': 1.5, 'Z': 4.0}, targets={'X'})
]
variable_order = ['X', 'Y', 'Z']

# Convert to AVICI format with Y as target
data = samples_to_avici_format(samples, variable_order, 'Y')
print(data.shape)  # (2, 3, 3)

# Check target indicators for variable Y (index 1)
print(data[:, 1, 2])  # [1.0, 1.0] - Y is target for all samples
```

#### create_training_batch(scm, samples, target_variable, standardize=True, is_count_data=False)
Create a training batch compatible with AVICI's training pipeline.

**Parameters:**
- scm: pyr.PMap - The structural causal model
- samples: SampleList - List of Sample objects
- target_variable: str - Name of target variable
- standardize: bool - Whether to standardize the data (default: True)
- is_count_data: bool - Whether the data should be treated as count data (default: False)

**Returns:**
AVICIDataBatch - Dictionary containing AVICI-compatible batch data with keys:
- 'x': Input data tensor [N, d, 3]
- 'g': Ground truth adjacency matrix [d, d]
- 'is_count_data': Boolean flag for standardization type
- 'target_variable': Name of target variable
- 'variable_order': List of variable names in order
- 'metadata': Additional batch metadata

**Raises:**
- ValueError: If inputs are invalid

**Example:**
```python
# Create SCM and generate samples
scm = create_linear_scm({'X': [], 'Y': ['X'], 'Z': ['Y']})
samples = sample_from_scm(scm, n_samples=100)

# Create training batch
batch = create_training_batch(scm, samples, target_variable='Z')

print(batch['x'].shape)  # (100, 3, 3)
print(batch['g'].shape)  # (3, 3) - adjacency matrix
print(batch['target_variable'])  # 'Z'
```

---

## Validation Functions Module

### Overview
This module contains validation logic to ensure data conversion preserves information and catches errors early with helpful messages. It provides comprehensive validation for both individual conversions and complete training batches.

### Core Types
```python
SampleList = List[pyr.PMap]
VariableOrder = List[str]
```

### Core Functions

#### validate_data_conversion(original_samples, converted_data, variable_order, target_variable, tolerance=1e-6)
Validate that data conversion preserves all information.

**Parameters:**
- original_samples: SampleList - Original Sample objects
- converted_data: jnp.ndarray - Converted AVICI data tensor [N, d, 3]
- variable_order: VariableOrder - Variable order used in conversion
- target_variable: str - Target variable name
- tolerance: float - Tolerance for numerical comparisons (default: 1e-6)

**Returns:**
bool - True if conversion preserved all information, False otherwise

**Raises:**
- ValueError: If inputs are inconsistent

**Example:**
```python
# Convert samples and validate
samples = [create_observational_sample({'X': 1.0, 'Y': 2.0})]
variable_order = ['X', 'Y']
data = samples_to_avici_format(samples, variable_order, 'Y', standardize=False)

# Validate conversion
is_valid = validate_data_conversion(samples, data, variable_order, 'Y')
print(is_valid)  # True

# Test with corrupted data
corrupted_data = data.at[0, 0, 1].set(1.0)  # Add fake intervention
is_valid = validate_data_conversion(samples, corrupted_data, variable_order, 'Y')
print(is_valid)  # False
```

#### validate_training_batch(batch, expected_n_samples, expected_n_variables, expected_target)
Validate that a training batch has the correct structure and content.

**Parameters:**
- batch: dict - Training batch dictionary
- expected_n_samples: int - Expected number of samples
- expected_n_variables: int - Expected number of variables
- expected_target: str - Expected target variable name

**Returns:**
bool - True if batch is valid, False otherwise

**Example:**
```python
# Create and validate training batch
scm = create_linear_scm({'X': [], 'Y': ['X'], 'Z': ['Y']})
samples = sample_from_scm(scm, n_samples=50)
batch = create_training_batch(scm, samples, 'Z')

# Validate batch structure
is_valid = validate_training_batch(batch, 50, 3, 'Z')
print(is_valid)  # True

# Check required keys are present
print('x' in batch)  # True
print('g' in batch)  # True
print('target_variable' in batch)  # True
```

---

## Analysis and Debugging Utilities Module

### Overview
This module contains functions for analyzing converted data, debugging issues, and providing insights into the conversion process. It offers comprehensive tools for understanding data transformations and troubleshooting conversion problems.

### Core Types
```python
SampleList = List[pyr.PMap]
VariableOrder = List[str]
```

### Core Functions

#### analyze_avici_data(avici_data, variable_order)
Analyze AVICI data tensor and return summary statistics.

**Parameters:**
- avici_data: jnp.ndarray - AVICI data tensor [N, d, 3]
- variable_order: VariableOrder - Variable order used in conversion

**Returns:**
Dict[str, Any] - Dictionary with analysis results including:
- 'shape': Data tensor shape
- 'n_samples': Number of samples
- 'n_variables': Number of variables
- 'values_stats': Statistics for value channel
- 'intervention_stats': Statistics for intervention channel
- 'target_stats': Statistics for target channel

**Example:**
```python
# Create and analyze AVICI data
samples = generate_mixed_samples(n_obs=50, n_int=25)
variable_order = ['X', 'Y', 'Z']
data = samples_to_avici_format(samples, variable_order, 'Z')

# Analyze the data
analysis = analyze_avici_data(data, variable_order)

print(f"Data shape: {analysis['shape']}")
print(f"Samples with interventions: {analysis['intervention_stats']['samples_with_interventions']}")
print(f"Target variable: {analysis['target_stats']['target_variable']}")
```

#### reconstruct_samples_from_avici_data(avici_data, variable_order, target_variable, standardization_params=None)
Reconstruct Sample objects from AVICI data tensor (for validation).

**Parameters:**
- avici_data: jnp.ndarray - AVICI data tensor [N, d, 3]
- variable_order: VariableOrder - Variable order used in conversion
- target_variable: str - Target variable name
- standardization_params: Optional[Dict[str, jnp.ndarray]] - Parameters for reversing standardization

**Returns:**
SampleList - List of reconstructed Sample objects

**Note:**
This function is primarily for validation purposes. Standardization reversal is approximate if standardization_params not provided.

**Example:**
```python
# Convert samples to AVICI format
original_samples = [create_observational_sample({'X': 1.0, 'Y': 2.0})]
variable_order = ['X', 'Y']
avici_data = samples_to_avici_format(original_samples, variable_order, 'Y', standardize=False)

# Reconstruct samples
reconstructed = reconstruct_samples_from_avici_data(avici_data, variable_order, 'Y')

# Compare original and reconstructed
print(original_samples[0]['values'])  # {'X': 1.0, 'Y': 2.0}
print(reconstructed[0]['values'])     # {'X': 1.0, 'Y': 2.0}
```

#### get_variable_order_from_scm(scm)
Get a consistent variable ordering from an SCM.

**Parameters:**
- scm: pyr.PMap - The structural causal model

**Returns:**
VariableOrder - List of variable names in a consistent order

**Note:**
Returns variables in a consistent order. In full implementation, would use topological sorting to respect causal dependencies.

**Example:**
```python
# Create SCM and get variable order
scm = create_linear_scm({
    'X': [],           # No parents
    'Y': ['X'],        # Y depends on X
    'Z': ['X', 'Y']    # Z depends on X and Y
})

# Get consistent variable ordering
variable_order = get_variable_order_from_scm(scm)
print(variable_order)  # ['X', 'Y', 'Z']

# Use for data conversion
samples = sample_from_scm(scm, n_samples=100)
data = samples_to_avici_format(samples, variable_order, target_variable='Z')
```

#### compare_data_conversions(samples, avici_data1, avici_data2, variable_order, labels=None)
Compare two different AVICI data conversions for debugging.

**Parameters:**
- samples: SampleList - Original sample data
- avici_data1: jnp.ndarray - First AVICI conversion
- avici_data2: jnp.ndarray - Second AVICI conversion
- variable_order: VariableOrder - Variable order used
- labels: List[str] - Labels for the two conversions (optional)

**Returns:**
Dict[str, Any] - Dictionary with comparison results including shape matches, channel differences, and individual analyses

**Example:**
```python
# Compare standardized vs non-standardized conversion
samples = generate_test_samples(n_samples=100)
variable_order = ['X', 'Y', 'Z']

data_std = samples_to_avici_format(samples, variable_order, 'Y', standardize=True)
data_raw = samples_to_avici_format(samples, variable_order, 'Y', standardize=False)

# Compare the conversions
comparison = compare_data_conversions(
    samples, data_std, data_raw, variable_order, 
    labels=["Standardized", "Raw"]
)

print(f"Shape match: {comparison['shape_match']}")
print(f"Intervention channels match: {comparison['binary_channel_matches']['interventions']}")
print(f"Values difference: {comparison['channel_differences']['values']:.4f}")
```

#### debug_sample_conversion(sample, sample_idx, avici_data, variable_order, target_variable)
Debug the conversion of a specific sample for detailed inspection.

**Parameters:**
- sample: pyr.PMap - Original Sample object
- sample_idx: int - Index of sample in the batch
- avici_data: jnp.ndarray - Converted AVICI data tensor
- variable_order: VariableOrder - Variable order used in conversion
- target_variable: str - Target variable name

**Returns:**
Dict[str, Any] - Dictionary with detailed debug information including original vs converted values and consistency checks

**Example:**
```python
# Debug specific sample conversion
samples = [create_interventional_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0}, targets={'X'})]
variable_order = ['X', 'Y', 'Z']
data = samples_to_avici_format(samples, variable_order, 'Y')

# Debug the first sample
debug_info = debug_sample_conversion(samples[0], 0, data, variable_order, 'Y')

print(f"Intervention targets match: {debug_info['consistency_checks']['intervention_targets_match']}")
print(f"Target variable match: {debug_info['consistency_checks']['target_variable_match']}")
print(f"Expected interventions: {debug_info['consistency_checks']['expected_interventions']}")
```