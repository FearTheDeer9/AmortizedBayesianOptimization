# Complete ACBO Workflow Demo

## Overview

The complete workflow demo (`examples/complete_workflow_demo.py`) provides end-to-end validation of the ACBO framework integration. It demonstrates how all major components work together in a realistic setting while following functional programming principles.

## What It Demonstrates

### Core ACBO Pipeline
1. **SCM Creation**: Fork structure (X → Y ← Z) with linear mechanisms
2. **Data Generation**: Observational sampling from the SCM
3. **Buffer Management**: Storing and querying samples efficiently
4. **AVICI Integration**: Converting samples to AVICI format for posteriors
5. **Acquisition Policy**: Intervention selection using neural networks
6. **Environment Simulation**: Applying interventions and observing outcomes
7. **State Management**: Tracking optimization progress and uncertainty
8. **Reward Computation**: Multi-component verifiable rewards

### Design Principles
- **Pure Functional Programming**: No side effects in core logic
- **Immutable Data Structures**: Using `pyrsistent` and frozen dataclasses
- **Composable Functions**: Small, focused functions that combine cleanly
- **Explicit State Management**: Clear state transitions without hidden mutations
- **Type Safety**: Comprehensive type hints throughout

## Architecture Validation

### Component Integration Testing
The demo validates that all major ACBO components can communicate correctly:

```
SCM → Linear Mechanisms → Environment Sampling
  ↓
Experience Buffer ← Intervention Application ← Acquisition Policy
  ↓                                                ↑
AVICI Data Conversion → Parent Set Posterior → Acquisition State
```

### API Compatibility
- **Data Format Conversion**: Sample → AVICI tensor format
- **Model Interfaces**: Surrogate and acquisition model APIs
- **State Transitions**: AcquisitionState updates with new data
- **Intervention Specification**: Policy output → intervention format

## Usage Examples

### Basic Usage
```python
from examples.complete_workflow_demo import run_complete_workflow_demo

# Run with defaults
results = run_complete_workflow_demo()
print(f"Target improvement: {results['analysis']['improvement']:.3f}")
```

### Custom Configuration
```python
from examples.complete_workflow_demo import DemoConfig, run_complete_workflow_demo

config = DemoConfig(
    n_observational_samples=50,
    n_intervention_steps=10,
    exploration_rate=0.2,
    random_seed=123
)

results = run_complete_workflow_demo(config)
analysis = results['analysis']

print(f"Final best value: {analysis['final_best']:.3f}")
print(f"Uncertainty reduction: {analysis['final_uncertainty']:.2f} bits")
print(f"Intervention types: {analysis['intervention_counts']}")
```

### Component Testing
```python
from examples.complete_workflow_demo import (
    create_demo_environment, 
    create_demo_models,
    DemoConfig
)

# Test individual components
config = DemoConfig()
env = create_demo_environment(config)
models = create_demo_models()

print(f"SCM variables: {env.variables}")
print(f"Target variable: {env.target}")
```

## Expected Behavior

### With Untrained Models
- **Random intervention choices**: Policy should make diverse but unfocused decisions
- **Minimal optimization progress**: Target improvement should be limited
- **Stable uncertainty**: Structure learning should show minimal progress
- **Functional correctness**: All components should integrate without errors

### Performance Characteristics
- **Fast execution**: Should complete in seconds for default configuration
- **Memory efficiency**: Reasonable memory usage for small-scale problems
- **Numerical stability**: No NaN or infinite values in computations
- **Reproducible results**: Same random seed produces identical outputs

## Validation Checklist

When running the demo, verify:

- ✅ **No import errors**: All dependencies load correctly
- ✅ **SCM creation**: Fork structure created with proper mechanisms
- ✅ **Data generation**: Observational samples generated successfully
- ✅ **Buffer operations**: Samples added and retrieved correctly
- ✅ **AVICI conversion**: Data converted to [N, d, 3] tensor format
- ✅ **Model instantiation**: Surrogate and acquisition models created
- ✅ **Intervention loop**: All steps execute without errors
- ✅ **State management**: AcquisitionState updates correctly
- ✅ **Results analysis**: Final analysis completes successfully

## Integration with Development Workflow

### Pre-Training Validation
Use this demo before implementing training pipelines to ensure:
- All component APIs are compatible
- Data flows correctly through the entire system
- Model architectures can be instantiated
- No fundamental integration issues exist

### Regression Testing
Run the demo after major changes to verify:
- Existing functionality still works
- New components integrate correctly
- Performance hasn't degraded significantly
- No new errors or warnings appear

### Development Guidelines
The demo follows project standards:
- **Functional programming principles**: Pure functions, immutable data
- **Clean architecture**: Separation of concerns, dependency injection
- **Scientific computing**: JAX compatibility, numerical stability
- **Code quality**: Type hints, documentation, error handling

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure proper installation
poetry install

# Check package structure
ls -la src/causal_bayes_opt/
```

**Model Creation Failures**
- Demo uses fallback implementations when real models fail
- Check console output for specific error messages
- Verify all dependencies are installed correctly

**Numerical Issues**
- All computations should produce finite values
- Check random seed reproducibility
- Verify JAX installation and backend configuration

### Performance Tips

**For Large-Scale Testing**
- Reduce `n_observational_samples` for faster execution
- Use fewer `n_intervention_steps` for quick validation
- Set explicit `random_seed` for reproducible debugging

**For Development**
- Start with minimal configuration
- Gradually increase complexity
- Use profiling tools for performance analysis
- Monitor memory usage for scaling behavior

## Future Enhancements

Planned improvements to the demo:

1. **Real Model Integration**: Use trained models when available
2. **Comparative Analysis**: Side-by-side comparison with baselines
3. **Visualization**: Plots of optimization progress and uncertainty reduction
4. **Batch Processing**: Demonstrate scaling to multiple SCMs
5. **Error Injection**: Test error handling and recovery mechanisms

This demo serves as both a validation tool and a reference implementation for proper ACBO usage patterns.