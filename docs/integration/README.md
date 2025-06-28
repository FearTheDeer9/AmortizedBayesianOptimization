# Integration Documentation

This directory contains documentation for external algorithm integrations within the ACBO framework.

## Available Integrations

### PARENT_SCALE Integration ✅ **COMPLETE**

**Status**: Production-ready with verified algorithmic correctness

The PARENT_SCALE integration enables collection of expert demonstrations from the neural doubly robust causal discovery algorithm for training our amortized models.

**📖 [Complete Integration Guide](parent_scale_integration.md)** - **START HERE**

#### Key Features:
- ✅ **Verified Correctness**: Perfect parent discovery with adequate statistical power
- ✅ **Arbitrary SCM Support**: Works with any causal structure (Chain, Erdos-Renyi, Scale-free, etc.)
- ✅ **Production Ready**: Validated scaling parameters and quality control
- ✅ **Clean Architecture**: Simplified, modular design following functional programming principles

#### Quick Start:
```python
from causal_bayes_opt.integration.parent_scale import (
    run_full_parent_scale_algorithm,
    check_parent_scale_availability
)

# Collect expert demonstration
trajectory = run_full_parent_scale_algorithm(
    scm=my_scm,
    target_variable='X2',
    T=10,
    n_interventional=20,  # Critical: ≥20 for reliable results
    seed=42
)
```

#### Documentation:
- **[PARENT_SCALE Integration Guide](parent_scale_integration.md)** - Complete usage guide
- **[Integration Complete](parent_scale_integration_complete.md)** - Legacy integration journey
- **[Data Bridge](parent_scale_data_bridge.md)** - Technical data conversion details

## Related Documentation

### Training Pipeline
- **[Expert Demonstration Collection](../training/expert_demonstration_collection_implementation.md)** - Implementation details (references new integration)
- **[Multi-Stage Training](../training/multi_stage_training.md)** - Training process overview
- **[Implementation Plan](../training/IMPLEMENTATION_PLAN.md)** - Complete training pipeline plan

### Architecture
- **[Phase 4 Plan](../architecture/PHASE4_CONSOLIDATED_PLAN.md)** - High-level architecture
- **[Module Structure](../architecture/module_structure.md)** - Code organization

### Infrastructure
- **[GPU Cluster Guide](../infrastructure/GPU_CLUSTER_GUIDE.md)** - Cluster setup and usage
- **[Cluster Integration Status](../infrastructure/CLUSTER_INTEGRATION_STATUS.md)** - Infrastructure status

## Integration Development Guidelines

### Adding New Integrations

When adding new external algorithm integrations:

1. **Create dedicated package** under `src/causal_bayes_opt/integration/algorithm_name/`
2. **Follow modular structure**:
   ```
   algorithm_name/
   ├── __init__.py          # Clean public API
   ├── core.py             # Main integration logic
   ├── data_processing.py  # Data conversion utilities
   ├── validation.py       # Quality control
   └── tests/              # Integration tests
   ```
3. **Implement availability check** - graceful degradation when external dependencies missing
4. **Provide comprehensive documentation** following the PARENT_SCALE integration guide template
5. **Validate correctness** - ensure integration produces identical results to original algorithm

### Quality Standards

All integrations must meet these standards:

- ✅ **Functional Programming**: Pure functions, no global state
- ✅ **Graceful Degradation**: Handle missing dependencies cleanly
- ✅ **Comprehensive Testing**: Unit, integration, and correctness tests
- ✅ **Complete Documentation**: Usage guide, API reference, troubleshooting
- ✅ **Performance Validated**: Scaling characteristics and optimization

### Testing Requirements

- **Correctness Tests**: Verify identical behavior to original algorithm
- **Integration Tests**: Test with different data formats and configurations
- **Performance Tests**: Validate scaling and resource usage
- **Quality Control**: Check output quality and consistency

## Future Integrations

Potential future integrations to consider:

- **AVICI Integration**: Direct integration with AVICI for surrogate model baselines
- **UT-IGSP Integration**: Interventional structure learning algorithms
- **CausalML Integration**: Alternative causal inference methods
- **DoWhy Integration**: Causal inference validation and robustness checks

Each integration should follow the established patterns and quality standards demonstrated by the PARENT_SCALE integration.