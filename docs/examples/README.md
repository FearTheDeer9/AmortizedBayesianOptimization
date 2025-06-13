# ACBO Examples

This directory contains examples demonstrating the usage of the Amortized Causal Bayesian Optimization framework.

## Available Examples

### Complete Workflow Demo (`complete_workflow_demo.py`)

**Purpose**: End-to-end validation of the ACBO pipeline integration

**What it demonstrates**:
- SCM creation with linear mechanisms (fork structure: X → Y ← Z)
- Observational data generation
- AVICI integration for parent set posterior prediction
- Acquisition model for intervention selection
- Complete intervention loop with state updates
- Multi-component reward computation
- Functional programming principles throughout

**Key Features**:
- ✅ **Pure functional design** - No side effects, immutable data structures
- ✅ **Component integration** - Validates all major ACBO components work together
- ✅ **Untrained model testing** - Uses real neural architectures without training
- ✅ **Comprehensive analysis** - Tracks optimization progress and uncertainty reduction

**Usage**:
```python
from examples.complete_workflow_demo import run_complete_workflow_demo, DemoConfig

# Run with default configuration
results = run_complete_workflow_demo()

# Run with custom configuration
config = DemoConfig(
    n_observational_samples=50,
    n_intervention_steps=10,
    exploration_rate=0.1,
    random_seed=42
)
results = run_complete_workflow_demo(config)

# Access results
analysis = results['analysis']
print(f"Target improvement: {analysis['improvement']:.3f}")
print(f"Final uncertainty: {analysis['final_uncertainty']:.2f} bits")
```

**Expected Behavior**:
- **Untrained models**: Should show random/poor intervention choices initially
- **Functional correctness**: All components should integrate without errors
- **Immutable operations**: No side effects or state mutations
- **Realistic fallbacks**: Graceful degradation when real models aren't available

**Architecture Validation**:
- Tests data flow: SCM → Buffer → Surrogate → Acquisition → Environment
- Validates AVICI integration: Sample format conversion and posterior prediction
- Confirms GRPO-compatible data structures work correctly
- Demonstrates verifiable reward computation

## Design Principles Demonstrated

### Functional Programming
- **Pure functions**: All core logic functions have no side effects
- **Immutable data**: Uses `pyrsistent` and frozen dataclasses throughout  
- **Composable operations**: Small, focused functions that can be combined
- **Explicit state management**: State transitions through pure functions

### Clean Architecture
- **Separation of concerns**: Data structures, business logic, and presentation separated
- **Dependency injection**: Models and configuration passed explicitly
- **Error handling**: Graceful fallbacks without throwing exceptions
- **Type safety**: Comprehensive type hints throughout

### Scientific Computing
- **Reproducible results**: Explicit random key management with JAX
- **Numerical stability**: Proper handling of edge cases and invalid values
- **Performance-conscious**: JAX-compatible data structures and operations
- **Validation-friendly**: Easy to verify correctness and debug issues

## Running Examples

### Prerequisites
```bash
# Ensure the project is properly installed
poetry install

# Navigate to project root
cd /path/to/causal_bayes_opt
```

### Command Line Usage
```bash
# Run complete workflow demo
poetry run python examples/complete_workflow_demo.py

# Run with Python module syntax
poetry run python -m examples.complete_workflow_demo
```

### Programmatic Usage
```python
# Import in other code
from examples.complete_workflow_demo import (
    run_complete_workflow_demo,
    DemoConfig,
    create_demo_environment,
    create_demo_models
)

# Use individual components
config = DemoConfig(n_intervention_steps=3)
env = create_demo_environment(config)
models = create_demo_models()
```

## Integration Testing

These examples serve as integration tests for the ACBO framework:

### Component Integration
- **Data Structures** ↔ **Mechanisms**: SCM creation and sampling
- **Mechanisms** ↔ **Environments**: Intervention application
- **Environments** ↔ **AVICI Integration**: Data format conversion
- **AVICI Integration** ↔ **Acquisition**: Posterior → state conversion
- **Acquisition** ↔ **Interventions**: Policy → intervention conversion

### API Compatibility
- All public APIs are exercised in realistic usage patterns
- Parameter passing and return value handling validated
- Error conditions and edge cases tested through fallback implementations

### Performance Characteristics
- JAX compilation compatibility verified
- Memory usage patterns established for large-scale problems
- Numerical stability confirmed across different random seeds

## Future Examples

Planned additions to this directory:

- **Training Pipeline Demo**: Complete training workflow with PARENT_SCALE demonstrations
- **Scaling Analysis**: Performance testing across different graph sizes
- **Multi-Objective Optimization**: Balancing structure learning and target optimization
- **Real-World Case Studies**: Applications to specific domains (economics, biology, etc.)
- **Comparison Baselines**: Side-by-side comparison with other causal discovery methods

## Troubleshooting

### Common Issues

**Import Errors**:
```python
# Ensure proper package installation
poetry install

# Check Python path
import sys
print(sys.path)
```

**JAX Compatibility**:
```python
# Verify JAX installation
import jax
print(jax.devices())

# Check for CUDA/GPU availability if needed
print(jax.default_backend())
```

**Model Initialization**:
- Examples use fallback models by default
- Real model creation may require additional dependencies
- Check `requirements.txt` for complete dependency list

### Performance Tips

**For Large Examples**:
- Use smaller `n_observational_samples` for faster execution
- Reduce `n_intervention_steps` for quicker validation
- Set `random_seed` for reproducible debugging

**For Development**:
- Run with minimal configuration first
- Gradually increase complexity
- Use profiling tools for performance analysis

## Contributing

When adding new examples:

1. **Follow functional programming principles**
2. **Include comprehensive documentation**
3. **Add type hints throughout**
4. **Provide both simple and advanced usage patterns**
5. **Include validation and error handling**
6. **Document expected behavior and performance characteristics**