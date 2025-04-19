# Causal Data Generation Examples

This directory contains examples demonstrating the causal data generation framework.

## Main Demo

`causal_data_demo.py` - A comprehensive demonstration of all the features of the causal data generation framework:

- Different graph types (Erdős–Rényi, Scale-Free, Small-World)
- Various mechanism types (Linear, Polynomial, Exponential, Sinusoidal, Mixed)
- Noise configurations (Gaussian, Uniform, Heteroskedastic)
- Interventions and their effects
- Dataset serialization and reproducibility

## Running the Demo

To run the demo:

```bash
# From the project root directory:
python examples/causal_data_demo.py
```

The demo will generate various visualizations in the `plots` directory, including:

- Graph structures
- Mechanism distributions and relationships
- Intervention effects
- Noise distribution effects

## Important Note About Interventions

When using the `sample_interventional` function, make sure to provide the causal graph structure via the `graph` parameter to ensure correct propagation of intervention effects. For example:

```python
# Correct way to sample interventional data:
int_samples = sample_interventional(
    scm=scm,
    node=node_to_intervene,
    value=intervention_value,
    n_samples=1000,
    seed=42,
    graph=graph  # Pass the actual graph
)
```

Without the `graph` parameter, the function falls back to a legacy approach that tries to infer the dependency graph, which may not correctly capture the true causal structure.

## Available Examples

### `causal_data_demo.py`

A comprehensive demo script that showcases the new features:

1. **Mechanism Diversity**: Demonstrates different functional relationships:

   - Linear SCMs
   - Polynomial SCMs
   - Exponential SCMs
   - Sinusoidal SCMs
   - Mixed mechanism types

2. **Noise Configuration**: Shows how to use different noise distributions:

   - Gaussian noise with different standard deviations
   - Uniform noise
   - Heteroskedastic noise (variance depends on parent values)

3. **Intervention Effects**: Demonstrates the effect of interventions with different mechanism types

4. **Serialization & Reproducibility**: Shows how to save and load datasets

## Running the Examples

To run the example scripts, make sure you're in the project root directory and run:

```bash
python examples/causal_data_demo.py
```

The script will create a `plots` directory with visualizations of the generated data.

## Output Directory Structure

After running the examples, you'll see:

- `plots/`: Contains all visualizations

  - `mechanism_*.png`: Visualizations of different mechanism types
  - `noise_*.png`: Visualizations of different noise configurations
  - `intervention_*.png`: Visualizations of intervention effects
  - `causal_graph.png`: Visualization of the causal graph structure

- `plots/saved_data/`: Contains saved datasets
  - `mixed_mechanism_dataset.pkl`: Example of a saved dataset

## Customization

You can modify the example scripts to:

- Change graph types and parameters
- Adjust mechanism parameters
- Try different noise configurations
- Test various intervention strategies
