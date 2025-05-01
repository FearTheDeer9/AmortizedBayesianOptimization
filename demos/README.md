# Amortized Causal Discovery Demos

This directory contains demonstration scripts for the Amortized Causal Discovery and Bayesian Optimization framework. These demos are designed to showcase the capabilities of our implementation so far.

## Available Demos

1. **Parent-Scaled ACD with Neural Networks** (`parent_scale_acd_demo.py`)
   - Demonstrates how to use neural networks as a drop-in replacement for traditional surrogate models in the Parent-Scaled ACD algorithm
   - Uses pre-trained models for quick inference without extensive training time
   - Includes visualizations of the inference process and results comparison

2. **Full Amortized Causal Discovery Pipeline** (`full_acd_pipeline_demo.py`)
   - Shows the complete amortized approach including training and meta-learning adaptation
   - Demonstrates how meta-learning improves performance across related causal structures
   - Includes performance comparisons with and without meta-learning

## Setup Instructions

### Prerequisites

- Python 3.10+
- All dependencies installed as specified in the project's `requirements.txt`
- PyTorch and CUDA (for GPU acceleration, optional but recommended)

### Environment Setup

```bash
# Activate the virtual environment
source .venv/bin/activate  # Unix/Mac
# or
.venv\Scripts\activate  # Windows

# Ensure all dependencies are installed
pip install -e .
```

## Running the Demos

### Parent-Scaled ACD Demo

```bash
python demos/parent_scale_acd_demo.py [--args]
```

Optional arguments:
- `--num_nodes`: Number of nodes in the synthetic graph (default: 10)
- `--num_samples`: Number of data samples (default: 1000)
- `--max_interventions`: Maximum number of interventions (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--visualize`: Enable visualization (default: True)

### Full Amortized Pipeline Demo

```bash
python demos/full_acd_pipeline_demo.py [--args]
```

Optional arguments:
- `--task_family_size`: Number of related tasks in the family (default: 10)
- `--num_nodes`: Number of nodes in the synthetic graphs (default: 10)
- `--num_samples`: Number of samples per graph (default: 1000)
- `--num_meta_train_steps`: Number of meta-training steps (default: 100)
- `--num_adaptation_steps`: Number of adaptation steps for new tasks (default: 10)
- `--seed`: Random seed for reproducibility (default: 42)
- `--visualize`: Enable visualization (default: True)
- `--quick`: Run in quick mode with fewer iterations for faster demonstration (default: True)

## Key Concepts Demonstrated

1. **Neural Network Surrogate Models**
   - Using neural networks instead of traditional Gaussian Process models
   - Handling uncertainty in neural network predictions
   - Scaling to larger graphs than traditional methods

2. **Meta-Learning for Causal Discovery**
   - Transfer learning across related causal structures
   - Few-shot adaptation to new graphs
   - Improved sample efficiency through meta-learning

3. **Amortized Inference**
   - Joint training of graph structure and dynamics models
   - Efficient inference for both graph structure and intervention outcomes
   - Handling interventional data in the neural network framework

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or graph size for larger models
- **Visualization Issues**: Ensure matplotlib is properly installed; try `--visualize False` if problems persist
- **Performance Issues**: Use the `--quick` flag for faster execution with fewer iterations

## Additional Resources

- See `examples/` directory for more detailed examples
- Refer to the main project documentation for API details
- Check `notebooks/` for interactive tutorials 