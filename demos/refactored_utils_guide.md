# Refactored Utilities Guide

This guide explains how to use the refactored utilities module (`refactored_utils.py`) when creating or modifying demo scripts. The refactored utilities properly leverage components from the `causal_meta` package according to the Component Registry, reducing code duplication and improving maintainability.

## Table of Contents

1. [Overview](#overview)
2. [Import Handling](#import-handling)
3. [Path Management](#path-management)
4. [Tensor Handling](#tensor-handling)
5. [Node Naming](#node-naming)
6. [Graph Handling](#graph-handling)
7. [Model Loading](#model-loading)
8. [Visualization](#visualization)
9. [SCM Conversion](#scm-conversion)
10. [Fallback Implementations](#fallback-implementations)
11. [Complete Example](#complete-example)

## Overview

The refactored utilities module provides improved implementations of common functions used in demo scripts, with proper error handling, logging, and fallback mechanisms. It replaces duplicate implementations with direct use of the official components from the Component Registry.

Key improvements include:

- **Safe Imports**: Graceful handling of import errors with informative messages
- **Tensor Shape Standardization**: Properly standardize tensor shapes for neural network components
- **Graph Handling**: Direct use of CausalGraph instead of custom implementations
- **Model Loading**: Robust model loading with proper error handling and fallbacks
- **Visualization**: Improved graph visualization with consistent styling
- **SCM Conversion**: Proper creation of StructuralCausalModel instances
- **Fallback Implementations**: Graceful degradation when components are not available
- **Logging**: Comprehensive logging for better debugging

## Import Handling

### Safe Import Function

```python
from demos.refactored_utils import safe_import

# Import a component with fallback
GraphEncoder = safe_import('causal_meta.meta_learning.acd_models.GraphEncoder')
```

The `safe_import` function handles import errors gracefully and provides informative error messages. If the import fails, it returns the fallback value (default: None).

### Pre-imported Components

The refactored utilities module pre-imports commonly used components from the `causal_meta` package:

```python
from demos.refactored_utils import (
    CausalGraph,
    GraphFactory,
    StructuralCausalModel,
    AmortizedCausalDiscovery,
    # And many more...
)
```

## Path Management

### Directory Functions

```python
from demos.refactored_utils import get_assets_dir, get_checkpoints_dir

# Get paths to directories
assets_dir = get_assets_dir()
checkpoints_dir = get_checkpoints_dir()

# Save a file in the assets directory
save_path = os.path.join(assets_dir, 'graph_visualization.png')
```

The `get_assets_dir` and `get_checkpoints_dir` functions return paths to the assets and checkpoints directories, creating them if they don't exist.

## Tensor Handling

### Standardize Tensor Shape

```python
from demos.refactored_utils import standardize_tensor_shape

# Standardize for graph encoder
encoder_input = standardize_tensor_shape(
    data, 
    for_component='graph_encoder',
    batch_size=4
)

# Standardize for dynamics decoder
decoder_input = standardize_tensor_shape(
    data, 
    for_component='dynamics_decoder',
    num_nodes=5
)
```

The `standardize_tensor_shape` function transforms tensor shapes based on the target component, with improved error handling and validation.

### Format Interventions

```python
from demos.refactored_utils import format_interventions

# Format interventions for standard dict
interventions = format_interventions({'X_0': 2.0})

# Format interventions for tensor input
tensor_interventions = format_interventions(
    {'X_0': 2.0},
    for_tensor=True,
    num_nodes=5,
    device=device
)
```

The `format_interventions` function standardizes intervention formats, with options for tensor representation needed by neural models.

## Node Naming

```python
from demos.refactored_utils import get_node_name, get_node_id

# Get standardized node name
node_name = get_node_name(0)  # 'X_0'
node_name = get_node_name('variable_1')  # 'X_variable_1'

# Get node ID
node_id = get_node_id('X_2')  # 2
```

The `get_node_name` and `get_node_id` functions ensure consistent node naming across components.

## Graph Handling

### Create Causal Graph from Adjacency Matrix

```python
from demos.refactored_utils import create_causal_graph_from_adjacency

# Create graph from adjacency matrix
adj_matrix = model.infer_causal_graph(data)
graph = create_causal_graph_from_adjacency(
    adj_matrix,
    node_names=['X_0', 'X_1', 'X_2'],
    threshold=0.5
)
```

The `create_causal_graph_from_adjacency` function creates a CausalGraph from an adjacency matrix, with proper handling of probabilistic edges.

### Infer Adjacency Matrix

```python
from demos.refactored_utils import infer_adjacency_matrix

# Infer adjacency matrix from data
adj_matrix = infer_adjacency_matrix(
    model,
    data,
    interventions=None,
    threshold=0.5
)
```

The `infer_adjacency_matrix` function extracts an adjacency matrix from a neural model, with proper handling of different model types.

## Model Loading

```python
from demos.refactored_utils import load_model

# Load model from checkpoint
model = load_model(
    path='example_checkpoints/acd_model.pt',
    model_class=AmortizedCausalDiscovery,
    device='cuda',
    hidden_dim=64,
    input_dim=1
)
```

The `load_model` function loads a model from a checkpoint, with robust error handling and fallback to creating a new model if needed.

## Visualization

### Visualize Graph

```python
from demos.refactored_utils import visualize_graph

# Visualize a graph
visualize_graph(
    graph,
    title='Causal Graph',
    figsize=(8, 6),
    highlight_nodes=['X_0'],
    save_path='assets/graph.png'
)
```

The `visualize_graph` function visualizes a causal graph using the appropriate visualization function based on the graph type.

### Compare Graphs

```python
from demos.refactored_utils import compare_graphs

# Compare two graphs
compare_graphs(
    true_graph,
    inferred_graph,
    left_title='True Graph',
    right_title='Inferred Graph',
    save_path='assets/comparison.png'
)
```

The `compare_graphs` function displays two graphs side by side for comparison.

## SCM Conversion

```python
from demos.refactored_utils import convert_to_structural_equation_model

# Convert graph to SCM
scm = convert_to_structural_equation_model(
    graph,
    noise_scale=0.5
)

# Generate data from SCM
obs_data = scm.sample_data(sample_size=100)
int_data = scm.sample_interventional_data(
    interventions={'X_0': 2.0},
    sample_size=100
)
```

The `convert_to_structural_equation_model` function creates a StructuralCausalModel from a graph, with proper structural equations.

### Intervention Target Selection

```python
from demos.refactored_utils import select_intervention_target_by_parent_count

# Select a node to intervene on
target_node = select_intervention_target_by_parent_count(graph)
```

The `select_intervention_target_by_parent_count` function selects a node to intervene on based on parent count (choosing the node with the most parents).

## Fallback Implementations

The refactored utilities include fallback implementations for when components from `causal_meta` are not available:

- `DummyGraph`: Fallback for CausalGraph
- `DummySCM`: Fallback for StructuralCausalModel
- `fallback_plot_graph`: Fallback for graph visualization

These fallbacks ensure that demo scripts can still run even if some components are not available, with proper warning messages.

## Complete Example

Here's a complete example of using the refactored utilities in a demo script:

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from demos.refactored_utils import (
    # Import required functions
    get_assets_dir,
    get_checkpoints_dir,
    standardize_tensor_shape,
    create_causal_graph_from_adjacency,
    load_model,
    infer_adjacency_matrix,
    visualize_graph,
    compare_graphs,
    format_interventions,
    convert_to_structural_equation_model,
    select_intervention_target_by_parent_count,
    
    # Import pre-imported components
    GraphFactory,
    AmortizedCausalDiscovery
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate synthetic graph
graph = GraphFactory.create_random_dag(num_nodes=5, edge_probability=0.3)

# Convert to SCM
scm = convert_to_structural_equation_model(graph)

# Generate data
obs_data = scm.sample_data(sample_size=100)

# Load model
model = load_model(
    path=f'{get_checkpoints_dir()}/acd_model.pt',
    model_class=AmortizedCausalDiscovery,
    device=device,
    hidden_dim=64,
    input_dim=1
)

# Prepare data for model
obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
encoder_input = standardize_tensor_shape(obs_tensor, for_component='graph_encoder')

# Infer graph structure
adj_matrix = infer_adjacency_matrix(model, encoder_input)

# Create graph from adjacency matrix
inferred_graph = create_causal_graph_from_adjacency(
    adj_matrix.cpu().numpy(),
    threshold=0.5
)

# Compare true and inferred graphs
compare_graphs(
    graph,
    inferred_graph,
    left_title='True Graph',
    right_title='Inferred Graph',
    save_path=f'{get_assets_dir()}/graph_comparison.png'
)

# Select intervention target
target_node = select_intervention_target_by_parent_count(inferred_graph)
print(f"Selected node for intervention: {target_node}")

# Generate interventional data
interventions = {target_node: 2.0}
int_data = scm.sample_interventional_data(interventions, sample_size=100)

# Format interventions for model
tensor_interventions = format_interventions(
    interventions,
    for_tensor=True,
    num_nodes=5,
    device=device
)

# Visualize results
plt.figure(figsize=(10, 6))
visualize_graph(
    inferred_graph,
    title='Inferred Graph with Intervention Target',
    highlight_nodes=[target_node],
    save_path=f'{get_assets_dir()}/intervention_graph.png'
)
plt.show()
```

This example demonstrates how to use the refactored utilities for a complete causal discovery and intervention workflow. 