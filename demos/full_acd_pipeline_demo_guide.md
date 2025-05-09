# Full Amortized Causal Discovery Pipeline Demo Guide

This guide provides detailed documentation for using and understanding the `full_acd_pipeline_demo.py` script, which demonstrates the complete amortized approach to causal discovery, including training, meta-learning adaptation, and intervention selection.

## Overview

The Full ACD Pipeline demo showcases:

1. **Task Family Generation**: Creating families of related causal structures
2. **Synthetic Data Generation**: Generating observational and interventional data from causal models
3. **Neural Model Training**: Training neural networks for causal structure discovery
4. **Meta-Learning**: Adapting models across related causal structures
5. **Intervention Selection**: Optimizing interventions for causal discovery
6. **Performance Evaluation**: Comparing amortized and meta-learning approaches

## Key Components

The demo utilizes the following components from the `causal_meta` package:

- **GraphFactory**: For generating synthetic causal graphs
- **TaskFamily**: For creating families of related causal structures
- **StructuralCausalModel**: For generating data from causal models
- **GraphEncoder**: Neural network for inferring causal structure
- **DynamicsDecoder**: Neural network for predicting interventional outcomes
- **AmortizedCausalDiscovery**: Joint model for structure and dynamics
- **MAMLForCausalDiscovery**: Meta-learning for quick adaptation

## Running the Demo

### Basic Usage

```bash
python demos/full_acd_pipeline_demo.py
```

This runs the demo with default parameters.

### Command-Line Arguments

```bash
python demos/full_acd_pipeline_demo.py [--args]
```

Available arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_nodes` | int | 5 | Number of nodes in the synthetic graphs |
| `--num_samples` | int | 100 | Number of data samples per graph |
| `--task_family_size` | int | 5 | Number of related graphs in the family |
| `--num_meta_train_steps` | int | 5 | Number of meta-training steps |
| `--num_adaptation_steps` | int | 3 | Number of adaptation steps for new tasks |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--visualize` | flag | True | Enable visualization |
| `--quick` | flag | False | Run in quick mode with minimal settings |
| `--pretrained_model_path` | str | `checkpoints/acd_model.pt` | Path to pretrained model |

### Quick Mode

For a faster demonstration with reduced computational requirements:

```bash
python demos/full_acd_pipeline_demo.py --quick
```

Quick mode uses:
- Smaller networks
- Fewer training iterations
- Smaller task families
- Reduced number of meta-learning steps

## Expected Output

### Visualizations

The demo generates several visualizations in the `assets` directory:

1. **Task Family Visualization**: Shows the family of related causal graphs
2. **Training Progress**: Loss curves for graph and dynamics modules
3. **Meta-Learning Progress**: Adaptation curves for new tasks
4. **Intervention Selection**: Visualization of selected interventions
5. **Model Comparison**: Performance comparison between approaches

### Console Output

The demo prints progress information to the console:

- Task family creation details
- Training progress and loss values
- Meta-learning adaptation results
- Interventional results and comparisons
- Performance metrics

### Generated Files

The demo creates:

- `task_family_visualization.png`: Visualization of the task family
- `training_progress.png`: Training loss curves
- `meta_learning_results.png`: Meta-learning adaptation results
- `model_comparison.png`: Performance comparison
- `acd_model.pt`: Trained model checkpoint (if not using pretrained)

## Key Algorithmic Steps

1. **Task Family Generation**
   ```python
   task_family, family_graphs = create_task_family(
       num_nodes=args.num_nodes,
       family_size=args.task_family_size,
       seed=args.seed
   )
   ```

2. **Synthetic Data Generation**
   ```python
   family_obs_data, family_int_data, family_scms = create_synthetic_data(
       task_family=task_family,
       num_samples=args.num_samples
   )
   ```

3. **Model Creation**
   ```python
   model = create_model(
       num_nodes=args.num_nodes,
       device=device
   )
   ```

4. **Training Loop**
   ```python
   train_dataloader = prepare_training_data(
       observational_data=family_obs_data,
       interventional_data=family_int_data,
       scms=family_scms
   )
   
   model = train_model(
       model=model,
       dataloader=train_dataloader,
       num_epochs=20 if not args.quick else 5,
       device=device
   )
   ```

5. **Meta-Learning Setup**
   ```python
   maml = setup_meta_learning(
       model=model,
       device=device
   )
   
   meta_tasks = prepare_meta_training_data(
       observational_data=family_obs_data,
       interventional_data=family_int_data
   )
   ```

6. **Meta-Training**
   ```python
   meta_model = meta_train_model(
       maml=maml,
       meta_tasks=meta_tasks,
       num_epochs=args.num_meta_train_steps,
       device=device
   )
   ```

7. **Intervention Optimization**
   ```python
   amortized_cbo = AmortizedCBO(model=model)
   
   optimal_interventions = amortized_cbo.optimize(
       observational_data=test_obs_data,
       causal_graph=inferred_graph
   )
   ```

8. **Performance Evaluation**
   ```python
   std_results = evaluate_model(
       model=model,
       test_data=(test_obs_data, test_int_data),
       true_graphs=[test_graph]
   )
   
   meta_results = evaluate_model(
       model=meta_model,
       test_data=(test_obs_data, test_int_data),
       true_graphs=[test_graph]
   )
   ```

## Technical Details

### Model Architecture

The demo uses a multi-component neural architecture:

1. **Graph Encoder**:
   - Processes observational data to infer causal structure
   - Uses attention mechanisms and GNN layers
   - Outputs adjacency matrix with edge probabilities

2. **Dynamics Decoder**:
   - Predicts outcomes under interventions
   - Conditions on graph structure and intervention targets
   - Outputs both predictions and uncertainty estimates

3. **Amortized CBO**:
   - Selects optimal interventions for causal discovery
   - Uses acquisition function to balance exploration vs. exploitation
   - Handles both active and passive intervention strategies

### Meta-Learning Implementation

The demo implements Model-Agnostic Meta-Learning (MAML):

- Inner loop adaptation: Quick adaptation to new tasks
- Outer loop optimization: Learning initialization for fast adaptation
- Implementation uses first-order approximation for efficiency

## Understanding Results

### Graph Recovery Metrics

The demo reports several metrics for graph structure recovery:

- **Structural Hamming Distance (SHD)**: Lower is better
- **Precision**: Fraction of predicted edges that are correct
- **Recall**: Fraction of true edges that are recovered
- **F1 Score**: Harmonic mean of precision and recall

### Intervention Outcomes

For intervention prediction:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Negative Log-Likelihood (NLL)**: Probabilistic prediction quality
- **Calibration Error**: How well-calibrated uncertainty estimates are

## Customization Options

### Using Your Own Graphs

To use your own graph structures:

```python
# Create your own graph
my_graph = CausalGraph()
my_graph.add_node("X_0")
my_graph.add_node("X_1")
my_graph.add_edge("X_0", "X_1")

# Create a custom task family
task_family = DummyTaskFamily(
    base_graph=my_graph,
    name="Custom Family"
)

# Proceed with the demo using your graph
```

### Custom SCM Equations

To define your own structural equations:

```python
def custom_equation(**kwargs):
    # Custom structural equation
    parents = kwargs.get('parents', {})
    noise = kwargs.get('noise', 0.0)
    
    # Your equation using parent values
    result = 2.0 * sum(parents.values()) + noise
    return result

# Use in SCM creation
scm = convert_to_structural_equation_model(
    graph=graph,
    node_equations={'X_1': custom_equation}
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size with smaller `--num_samples`
   - Run with `--quick` flag for smaller models
   - Reduce number of nodes with lower `--num_nodes`

2. **NaN Losses During Training**:
   - Check for extreme values in generated data
   - Reduce learning rate in training functions
   - Add gradient clipping to training loops

3. **Visualization Errors**:
   - Run with `--visualize False` if Matplotlib issues occur
   - Ensure proper display configuration for remote environments
   - Check for proper directory permissions

4. **Poor Graph Recovery**:
   - Increase training samples with higher `--num_samples`
   - Adjust `edge_probability` in graph generation
   - Increase training time with more epochs

## Additional Resources

- **Component Registry**: See `memory-bank/component-registry.md` for component details
- **Algorithm Documentation**: See `algorithms/` directory for theoretical background
- **API Documentation**: See function docstrings for detailed API information

## Example Use Cases

1. **Research Experiments**:
   - Benchmarking causal discovery methods
   - Testing intervention selection strategies
   - Evaluating meta-learning for causal tasks

2. **Educational Purposes**:
   - Understanding causal discovery principles
   - Visualizing intervention effects
   - Learning about meta-learning approaches

3. **Method Development**:
   - Testing new neural architectures
   - Implementing custom intervention strategies
   - Developing improved meta-learning algorithms

## Related Demos

- `parent_scale_acd_demo.py`: Focused on parent-scaled intervention selection
- See `examples/` directory for additional targeted examples 