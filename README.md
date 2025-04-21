# Causal Meta-Learning for Bayesian Optimization

A Python library for graph-based causal inference, structural causal models, and meta-learning for causal Bayesian optimization.

## Overview

This library provides tools for working with causal graphs, structural causal models (SCMs), and meta-learning approaches for causal Bayesian optimization. It supports graph-based causal inference, data generation from causal models, and various graph visualization utilities.

## Project Structure

- **causal_meta**: Core package containing all the modules
  - **graph**: Graph classes and utilities for causal modeling
  - **environments**: Environment implementations including SCMs
  - **meta_learning**: Meta-learning implementations for causal tasks
  - **optimization**: Optimization algorithms including Bayesian optimization
  - **inference**: Inference methods for causal discovery and reasoning
  - **discovery**: Causal discovery algorithms
  - **utils**: Utility functions and helpers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/causal_bayes_opt.git
cd causal_bayes_opt

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Import key modules
from causal_meta.graph import CausalGraph
from causal_meta.environments import StructuralCausalModel
import numpy as np

# Create a simple causal graph
graph = CausalGraph()
graph.add_node('X')
graph.add_node('Y')
graph.add_node('Z')
graph.add_edge('X', 'Y')
graph.add_edge('Z', 'Y')

# Create a structural causal model
scm = StructuralCausalModel(causal_graph=graph)
scm.add_variable('X', domain='continuous')
scm.add_variable('Y', domain='continuous')
scm.add_variable('Z', domain='continuous')

# Define structural equations
scm.define_linear_gaussian_equation('X', {}, intercept=0, noise_std=1)
scm.define_linear_gaussian_equation('Z', {}, intercept=0, noise_std=1)
scm.define_linear_gaussian_equation('Y', {'X': 0.5, 'Z': 0.8}, intercept=0, noise_std=0.1)

# Sample data from the SCM
data = scm.sample_data(1000)
print(data.head())

# Perform an intervention
scm.do_intervention('X', 2.0)
interventional_data = scm.sample_data(1000)
print(interventional_data.head())
```

## Features

### Graph Classes

- Base Graph class with directed graph operations
- Directed graph implementation supporting various graph algorithms
- Causal graph with d-separation, Markov blanket, and causal reasoning

### Causal Environments

- Structural Causal Models (SCMs) with various equation types
- Support for both deterministic and probabilistic relationships
- Intervention capabilities including do-operator and counterfactuals

### Graph Generation

- Factory pattern for creating various graph structures
- Support for random, scale-free, and predefined causal graphs

### Visualization

- Graph visualization utilities with customizable layouts
- Edge and node highlighting options
- Path visualization for causal analysis

## Documentation

For detailed documentation and examples, see the Jupyter notebooks in the `notebooks/` directory:

1. **Graph Classes and Operations**: Working with different graph types
2. **Graph Visualization**: Visualization options and customizations
3. **Graph Generation**: Using the factory pattern for graph structures
4. **Causal Environments**: Working with SCMs and causal environments
5. **Example Use Cases**: End-to-end examples combining multiple components

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
