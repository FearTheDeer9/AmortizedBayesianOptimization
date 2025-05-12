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
- **algorithms**: Algorithm implementations
  - **PARENT_SCALE_ACD.py**: Full implementation of Parent-Scaled ACD algorithm
- **demos**: Demonstration scripts
  - **parent_scale_acd_demo.py**: Demo of Parent-Scaled ACD with neural networks

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

## Demo Scripts

### Parent-Scaled ACD Demo

The `parent_scale_acd_demo.py` script demonstrates how to use neural networks for causal discovery with the Parent-Scaled ACD algorithm. This algorithm uses interventions to improve causal structure learning, selecting targets based on their parent count in the currently inferred graph.

#### Current Implementation:

- **Synthetic Data Generation**: Creates random graphs and data for demonstrations
- **Neural Network Models**: Uses AmortizedCausalDiscovery model for structure learning
- **Parent-Count Intervention Selection**: Targets nodes with more inferred parents
- **Intervention Experiments**: Updates causal graph based on interventional data
- **Visualization**: Shows inferred graph before and after interventions
- **Robust Error Handling**: Gracefully handles model loading failures and other errors
- **Integration Options**: Can use either the simplified demo implementation or the full algorithm

#### Running the Demo:

```bash
# Run with default settings
python demos/parent_scale_acd_demo.py

# Run in quick mode (fewer samples, interventions)
python demos/parent_scale_acd_demo.py --quick

# Run with full algorithm implementation
python demos/parent_scale_acd_demo.py --use_full_algorithm

# Customize settings
python demos/parent_scale_acd_demo.py --num_nodes 7 --max_interventions 5
```

#### To Be Implemented:

- **Advanced Acquisition Functions**: Currently uses simple parent count, but could implement Expected Information Gain
- **Comprehensive Metrics**: Need to add precision, recall, and F1 score for structure recovery
- **Benchmark Datasets**: Validate on standard causal discovery benchmarks
- **Improved Visualization**: Add dynamic graph visualization during the intervention process
- **Model Training**: Add functionality to train models from scratch on custom datasets
- **Hyperparameter Tuning**: Allow customization of model architecture and training parameters

## Simplified Causal Discovery Model

We have developed a simplified causal discovery model in `demos/simplified_causal_discovery.py` that can be used with the progressive structure recovery demo. This model:

- Addresses tensor dimension compatibility issues in the original AmortizedCausalDiscovery model
- Provides a clean interface for integrating with EnhancedMAMLForCausalDiscovery
- Supports proper intervention encoding for more effective structure learning
- Works with graphs of different sizes

To use the simplified model:

```bash
# Run the demo with default settings (3 nodes)
python demos/progressive_structure_recovery_demo.py --model-type simplified --visualize

# Test with multiple graph sizes
python demos/progressive_structure_recovery_demo.py --test-multiple-sizes --min-nodes 3 --max-nodes 6 --visualize
```

The implementation includes:

1. `SimplifiedCausalDiscovery`: A lightweight neural model for adjacency matrix prediction
2. `EnhancedMAMLForCausalDiscovery`: Improved MAML wrapper with proper intervention encoding and regularization
3. Testing functionality for multiple graph sizes 

Our results demonstrate that the model can successfully recover simple causal structures with a sufficient number of adaptive interventions.

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

## [Refactor Notice: SCM Node Naming Convention]

**Update (2024-06-12):**
All SCMs, graphs, and interventions must use valid Python identifiers for node names, following the convention: `x0`, `x1`, ..., `xN`.

- This replaces previous conventions using stringified integers (e.g., `'0'`, `'1'`).
- All code, tests, and examples have been updated to use this convention.
- Rationale: Ensures compatibility with dynamic function generation and Python syntax.

### Example
```python
# Creating a graph with 3 nodes
node_names = [f"x{i}" for i in range(3)]
graph = CausalGraph()
for name in node_names:
    graph.add_node(name)

graph.add_edge('x0', 'x1')
graph.add_edge('x1', 'x2')

# Interventions
interventions = {'x1': 5.0}

# SCM variable names
scm.add_variable('x0')
scm.add_variable('x1')
scm.add_variable('x2')
```

---
