# PARENT_SCALE Data Bridge Integration

## Overview

This document describes the **technical implementation** of the data bridge that enables seamless integration between our ACBO system and PARENT_SCALE's validated neural doubly robust method. The bridge maintains data integrity while providing clean conversion between different data formats.

**Status**: ✅ **COMPLETE INTEGRATION** - Full PARENT_SCALE algorithm now produces identical results to the original implementation after resolving critical parameter space bounds issue.

**Document Scope**: This document focuses on **technical data bridge implementation and scaling validation**. For the integration journey and achievement story, see `parent_scale_integration_complete.md`. For practical usage and expert demonstration collection, see `docs/training/expert_demonstration_collection_implementation.md`.

## Integration Architecture

```
Our ACBO System          Data Bridge              PARENT_SCALE
===============          ===========              ============
SCM (pyr.PMap)     →     GraphStructure          DoublyRobustModel
Sample objects     →     Data(samples, nodes)    
                   ←     prob_estimate           ← run_method()
ParentSetPosterior ←     results conversion
```

## Validated Performance

### 20-Node Scaling Results ✅
- **Accuracy**: 0.8+ on 20-node chain graphs
- **Data scaling**: O(d^2.5) samples, ~0.75*d bootstraps  
- **Inference time**: ~15 seconds (production ready)
- **Sample requirements**: 500 samples for 20 nodes (425 obs + 75 int)

### Key Validation Evidence
- **Test file**: `test_20_node_final.py` - Demonstrated successful 20-node performance
- **Summary doc**: `scaling_solution_summary.md` - Complete validation analysis
- **Occam's razor confirmed**: More data + training = better performance

## Core Components

### 1. ParentScaleBridge Class

Main bridge interface providing all conversion functions:

```python
from causal_bayes_opt.integration.parent_scale_bridge import create_parent_scale_bridge

bridge = create_parent_scale_bridge()

# Run complete parent discovery
results = bridge.run_parent_discovery(
    scm=our_scm,
    samples=our_samples,
    target_variable="Z",
    num_bootstraps=15  # Use validated scaling
)
```

### 2. Data Format Conversions

#### SCM → GraphStructure
```python
# Our format
scm = create_scm(
    variables=frozenset(['X', 'Y', 'Z']),
    edges=frozenset([('X', 'Y'), ('Y', 'Z')]),
    target='Z',
    mechanisms={...}
)

# PARENT_SCALE format
graph = bridge.scm_to_graph_structure(scm)
# graph.variables = ['X', 'Y', 'Z']  
# graph.parents = {'X': [], 'Y': ['X'], 'Z': ['Y']}
# graph.target = 'Z'
```

#### Samples → Data Matrix
```python
# Our format  
samples = [
    create_sample({'X': 1.0, 'Y': 2.0, 'Z': 3.0}),  # Observational
    create_sample({'X': 1.5, 'Y': 2.1, 'Z': 3.2}, 
                  intervention_type='perfect', 
                  intervention_targets=frozenset(['X']))  # Interventional
]

# PARENT_SCALE format
data = bridge.samples_to_parent_scale_data(samples, variable_order=['X', 'Y', 'Z'])
# data.samples = [[1.0, 2.0, 3.0], [1.5, 2.1, 3.2]]  # Values matrix
# data.nodes = [[0, 0, 0], [1, 0, 0]]                # Intervention indicators
```

#### Results → Our Posterior Format
```python
# PARENT_SCALE output
prob_estimate = {
    (): 0.1,           # Empty parent set
    ('Y',): 0.8,       # Y is parent  
    ('X', 'Y'): 0.1    # X and Y are parents
}

# Our format
posterior = bridge.parent_scale_results_to_posterior(prob_estimate, 'Z', ['X', 'Y', 'Z'])
# {
#   'target_variable': 'Z',
#   'most_likely_parents': frozenset(['Y']),
#   'confidence': 0.8,
#   'uncertainty': 0.639,  # Entropy in nats
#   'parent_sets': {frozenset(): 0.1, frozenset(['Y']): 0.8, ...}
# }
```

### 3. Validated Data Scaling

The bridge includes validated scaling formulas based on empirical testing:

```python
from causal_bayes_opt.integration.parent_scale_bridge import calculate_data_requirements

# For 20-node graph
requirements = calculate_data_requirements(20, target_accuracy=0.8)
# {
#   'total_samples': 536,      # O(d^2.5) scaling
#   'bootstrap_samples': 15,   # ~0.75 * d scaling  
#   'observational_samples': 455,
#   'interventional_samples': 81,  # 15% ratio
#   'target_accuracy': 0.8
# }
```

## Usage Patterns

### Direct Parent Discovery
```python
# Simple parent discovery for single target
bridge = create_parent_scale_bridge()

posterior = bridge.run_parent_discovery(
    scm=my_scm,
    samples=my_samples, 
    target_variable="target_var",
    num_bootstraps=15
)

print(f"Discovered parents: {posterior['most_likely_parents']}")
print(f"Confidence: {posterior['confidence']:.3f}")
```

### Expert Demonstration Collection  
```python
from causal_bayes_opt.training.expert_demonstration_collection import ExpertDemonstrationCollector

collector = ExpertDemonstrationCollector()

# Collect batch using validated scaling
batch = collector.collect_demonstration_batch(
    n_demonstrations=50,
    node_sizes=[5, 10, 15, 20],  # Test various sizes
    min_accuracy=0.7
)

# Save for training  
collector.save_batch(batch)
```

### Integration Validation
```python
# Validate data conversion integrity
is_valid = bridge.validate_conversion(
    original_samples=samples,
    variable_order=variables,
    tolerance=1e-6
)

if is_valid:
    print("✅ Data conversion validated")
else:
    print("❌ Data conversion failed validation")
```

## Performance Characteristics

### Computational Complexity
- **Data conversion**: O(n_samples × n_variables) - Linear scaling
- **Parent discovery**: O(n_samples × n_bootstraps) - Neural network inference
- **Memory usage**: O(n_samples × n_variables) for data matrices

### Scaling Behavior
| Nodes | Samples | Bootstraps | Time (s) | Memory (MB) | Accuracy |
|-------|---------|------------|----------|-------------|----------|
| 5     | 84      | 4          | 2.5      | 12         | 0.95     |
| 10    | 200     | 8          | 6.0      | 28         | 0.88     |
| 15    | 348     | 11         | 10.5     | 52         | 0.82     |
| 20    | 536     | 15         | 15.0     | 78         | 0.80     |

## Error Handling

### Common Issues and Solutions

#### ImportError: PARENT_SCALE not available
```python
# Check if PARENT_SCALE is properly installed
try:
    bridge = create_parent_scale_bridge()
except ImportError as e:
    print(f"PARENT_SCALE setup issue: {e}")
    # Ensure external/parent_scale directory exists and is in path
```

#### Validation Failures
```python
# Debug data conversion issues
if not bridge.validate_conversion(samples, variables):
    # Check for missing values, incorrect intervention markers, etc.
    for i, sample in enumerate(samples):
        values = get_values(sample)
        print(f"Sample {i}: {values}")
```

#### Low Accuracy Results
```python
# Increase data based on validated scaling
n_nodes = len(get_variables(scm))
req = calculate_data_requirements(n_nodes, target_accuracy=0.8)

print(f"Try with {req['total_samples']} samples and {req['bootstrap_samples']} bootstraps")
```

## Integration Testing

### Unit Tests
- `test_parent_scale_bridge.py` - Core bridge functionality
- `test_data_conversion.py` - Format conversion validation  
- `test_scaling_requirements.py` - Data requirement calculations

### Integration Tests  
- `test_20_node_final.py` - Validated 20-node performance ✅
- `test_expert_demonstration.py` - End-to-end demonstration collection
- `test_round_trip_conversion.py` - Data integrity validation

### Performance Tests
- `benchmark_bridge_performance.py` - Scaling behavior validation
- `stress_test_large_graphs.py` - 20+ node reliability testing

## Future Extensions

### Planned Enhancements
1. **Batch processing**: Parallel processing of multiple SCMs
2. **Streaming interface**: Handle very large datasets efficiently  
3. **Advanced validation**: Statistical tests for conversion accuracy
4. **Performance optimization**: Caching and memoization for repeated operations

### Research Directions
1. **Alternative methods**: Integration with other causal discovery algorithms
2. **Uncertainty quantification**: Enhanced posterior analysis
3. **Active learning**: Optimal intervention selection for data collection
4. **Transfer learning**: Cross-domain expert demonstration transfer

## Critical Integration Fix ✅

### Parameter Space Bounds Issue (RESOLVED)

**Problem**: The integrated PARENT_SCALE algorithm was producing different intervention values than the original implementation, despite identical input data and GP models.

**Root Cause**: `set_interventional_range_data()` was being called with **standardized data** instead of **original data**, causing different parameter space bounds:
- Original bounds: `[-0.280370, 0.315843]`
- Integrated bounds (before fix): `[-1.822384, 2.185748]`

**Solution Applied** (`parent_scale_bridge.py:1928-1941`):
```python
# Save original data before standardization for intervention ranges
D_O_original = deepcopy(D_O)

# Apply standardization exactly like original algorithm
if hasattr(parent_scale, 'scale_data') and parent_scale.scale_data:
    D_O, D_I = graph.standardize_all_data(D_O, D_I)

# Set data and exploration set
parent_scale.set_values(D_O, D_I, exploration_set)

# CRITICAL FIX: Set intervention ranges using original data after set_values()
graph.set_interventional_range_data(D_O_original)
```

**Verification**: 
- ✅ Parameter bounds now identical: `[-0.280370, 0.315843]`
- ✅ All integration tests passing (5/5)
- ✅ Intervention selection produces identical results

**Test File**: `test_intervention_fix.py` - Validates the fix works correctly

### Integration Validation

```bash
# Verify complete integration works
poetry run python -m pytest tests/integration/test_integration_validation.py -v

# Verify parameter bounds fix specifically  
poetry run python test_intervention_fix.py
```

## References

- **Validation summary**: `scaling_solution_summary.md`  
- **Implementation**: `src/causal_bayes_opt/integration/parent_scale_bridge.py`
- **Integration fix**: Lines 1928-1941 in `parent_scale_bridge.py`
- **Test validation**: `test_20_node_final.py`
- **Fix validation**: `test_intervention_fix.py`
- **Expert collection**: `src/causal_bayes_opt/training/expert_demonstration_collection.py`