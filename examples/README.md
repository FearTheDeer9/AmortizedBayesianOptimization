# Examples and Tutorials

This directory contains various examples, tutorials, and demonstration scripts for the `causal_meta` package.

## Core Examples

- **amortized_cbo_workflow.py**: Demonstrates the complete Amortized Causal Bayesian Optimization workflow from data generation to intervention optimization.
- **meta_cbo_workflow.py**: Shows how to use meta-learning to transfer knowledge between related causal optimization tasks.
- **train_gnn_encoder_decoder.py**: Example of training the graph encoder and dynamics decoder neural networks.
- **visualization_example.py**: Demonstrates the visualization utilities for neural causal discovery and optimization results.

## Benchmarking Framework

The benchmarking framework allows you to evaluate and compare different causal discovery and causal Bayesian optimization methods. The following resources provide documentation and guidance:

- **benchmarking_tutorial.md**: Comprehensive tutorial on using the benchmarking framework. This file can be:
  - Read as a markdown document for reference
  - Executed as a Python script (`python benchmarking_tutorial.md`)
  - Converted to a Jupyter notebook (`jupyter nbconvert --to notebook --execute benchmarking_tutorial.md`)

- **benchmark_visualization_guide.md**: Visual guide for interpreting the various visualizations produced by the benchmarking framework, including:
  - Causal discovery benchmark visualizations
  - Causal Bayesian optimization benchmark visualizations
  - Scalability benchmark visualizations
  - Multi-method comparison visualizations

- **run_benchmarks.py**: Example script demonstrating how to create and run benchmarks programmatically.

## Running the Examples

Most examples can be run directly from the command line:

```bash
# Run the Amortized CBO workflow example
python examples/amortized_cbo_workflow.py

# Run the benchmarking example
python examples/run_benchmarks.py --quick

# Run the visualization example
python examples/visualization_example.py
```

Some examples accept command-line arguments for customization. Use the `--help` flag to see available options:

```bash
python examples/run_benchmarks.py --help
```

## Example Datasets and Models

For some examples that require pre-trained models or specific datasets, you may need to download additional files or run the training scripts first. See the individual example files for specific requirements and instructions.

## Contributing New Examples

When contributing new examples, please follow these guidelines:

1. Include clear documentation at the beginning of the file explaining the purpose and usage
2. Add appropriate command-line arguments for customization
3. Include a section in this README file describing the new example
4. Ensure the example runs with default parameters without requiring additional setup
5. Add comments throughout the code explaining key concepts and implementation details 