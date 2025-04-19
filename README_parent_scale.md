# PARENT_SCALE Implementation with the New Graphs Framework

This document describes the PARENT_SCALE algorithm implementation using the new graphs framework.

## Overview

The PARENT_SCALE (Parent Aggregation for Robust Estimation with Novel Target Scaling and Causal Learning Exploration) algorithm is designed for causal optimization in complex systems. This implementation leverages the new graphs framework to provide a clean, maintainable interface.

## Key Files

1. `graphs/parent_scale.py` - Core implementation of the PARENT_SCALE algorithm
2. `examples/parent_scale_demo.py` - Demonstration of how to use the PARENT_SCALE algorithm
3. `examples/parent_scale_wrapper.py` - Simplified wrapper for easier usage

## Implementation Details

The implementation consists of several key components:

- **ReplayBuffer**: A simple class to store data samples
- **DoFunctions**: Class that synthesizes all the do-calculus functions into one
- **PARENT_SCALE**: Main algorithm implementation

The PARENT_SCALE class leverages the CausalEnvironmentAdapter to provide compatibility with both graph structures and NetworkX DiGraphs. It works with observational and interventional data to identify the most probable parent sets for target variables.

## Using the PARENT_SCALE Algorithm

### Basic Usage

```python
from graphs import PARENT_SCALE, CausalEnvironmentAdapter
from graphs.graph_generators import generate_erdos_renyi
from graphs.scm_generators import generate_mixed_scm, sample_observational, sample_interventional

# Create a graph and SCM
graph = generate_erdos_renyi(5, edge_prob=0.3, seed=42)
scm = generate_mixed_scm(graph, seed=42)

# Generate data
obs_data = sample_observational(scm, n_samples=1000, seed=42)
int_data = {}  # Fill with interventional data

# Create exploration set
exploration_set = [(str(node),) for node in range(4)]  # Assuming 5 nodes with target=4

# Create PARENT_SCALE instance
ps = PARENT_SCALE(
    graph=graph,
    nonlinear=True,
    causal_prior=True,
    noiseless=False,
    scale_data=True,
    seed=42
)

# Set data and run algorithm
ps.set_values(obs_data, int_data, exploration_set)
result = ps.run_algorithm(T=10)
```

### Using the Wrapper

For simplified usage, you can use the provided wrapper:

```python
from examples.parent_scale_wrapper import run_parent_scale

# Run with default parameters
result = run_parent_scale()

# Or customize parameters
result = run_parent_scale(
    num_nodes=5,
    edge_prob=0.3,
    mechanism_type="mixed",
    n_obs=1000,
    n_int=10,
    seed=42,
    nonlinear=True,
    scale_data=True
)
```

## Algorithm Features

The PARENT_SCALE algorithm provides:

1. **Causal Structure Learning**: Identifies probable parent sets for target variables
2. **Statistical Inference**: Uses do-calculus for causal effect estimation
3. **Data Scaling**: Normalizes data for more stable inference
4. **Exploration Set Management**: Handles exploration sets for intervention planning

## Future Improvements

Potential improvements to the current implementation:

1. Enhanced SCM modeling for more complex mechanisms
2. Better handling of multi-interventions
3. Integration with advanced acquisition functions for Bayesian optimization
4. Parallelization for handling larger datasets and graphs

## Requirements

- Python 3.7+
- NetworkX
- NumPy
- Matplotlib
