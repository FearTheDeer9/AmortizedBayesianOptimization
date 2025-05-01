# Amortized Causal Discovery Demos

This directory contains demonstration scripts for the Amortized Causal Discovery and Bayesian Optimization framework. These demos showcase the capabilities of our implementation and provide examples of how to use the various components.

## Available Demos

### 1. **Parent-Scaled ACD with Neural Networks** (`parent_scale_acd_demo.py`)

This demo shows how neural networks can be used for amortized causal discovery with a parent-scaled intervention approach.

**Current Implementation:**
- Uses neural networks for causal structure inference from observational data
- Implements parent-scaled intervention selection based on inferred graph structure
- Performs iterative interventions to refine the causal structure
- Provides comprehensive visualizations throughout the discovery process:
  - True underlying causal graph (ground truth)
  - Initial inferred graph from observational data
  - Updated graphs after each intervention
  - Before/after comparisons for each intervention
  - Final comparison between ground truth and discovered structure
- Color-coded visualizations with clear labels distinguishing between:
  - Ground truth graphs (green nodes)
  - Inferred graphs (blue nodes)
  - Pre-intervention and post-intervention states

**Roadmap to Full PARENT_SCALE_ACD Implementation:**
- Integrate the `update_models` method to refine neural networks with intervention data (current demo only updates the adjacency matrix)
- Add uncertainty estimation for edge predictions (using dropout or ensemble methods)
- Implement proper acquisition function for intervention optimization
- Add intervention cost calculations for more realistic scenarios
- Include intervention outcome prediction mechanism using the dynamics decoder
- Provide performance metrics comparing with other causal discovery methods
- Support for larger graphs and real-world datasets
- Implement the full optimization loop from the `PARENT_SCALE_ACD` algorithm

### 2. **Full Amortized Causal Discovery Pipeline** (`full_acd_pipeline_demo.py`)
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
- `--num_nodes`: Number of nodes in the synthetic graph (default: 5)
- `--num_samples`: Number of data samples (default: 100)
- `--max_interventions`: Maximum number of interventions (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--visualize`: Enable visualization (default: True)
- `--quick`: Run in quick mode with fewer samples and interventions (default: False)
- `--pretrained_model_path`: Path to pretrained model (default: looks in example_checkpoints)

Example for quick testing:
```bash
python demos/parent_scale_acd_demo.py --quick
```

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

2. **Parent-Scaled Intervention Selection**
   - Selecting interventions based on the number of inferred parents
   - Using the parent-count heuristic to prioritize complex relationships
   - Iteratively refining the causal structure with targeted interventions

3. **Amortized Inference for Causal Discovery**
   - Joint training of graph structure and dynamics models
   - Efficient inference for both graph structure and intervention outcomes
   - Handling interventional data in the neural network framework

4. **Meta-Learning for Causal Discovery**
   - Transfer learning across related causal structures
   - Few-shot adaptation to new graphs
   - Improved sample efficiency through meta-learning

## Understanding the Visualization Output

The `parent_scale_acd_demo.py` script produces several types of visualizations:

1. **True Causal Graph**: Shows the ground truth causal structure generated by the synthetic model
2. **Initial Inferred Graph**: The structure inferred from observational data alone
3. **Initial Comparison**: Side-by-side comparison of ground truth vs. initial inference
4. **Intervention Visualizations**: For each intervention:
   - Individual graph showing the structure after intervention
   - Side-by-side comparison of before vs. after intervention
5. **Final Comparison**: Ground truth vs. final inferred structure after all interventions

All visualizations use consistent node positions and clear labeling to make comparisons easier to understand.

## Technical Implementation Details

### Graph Encoder
The graph encoder uses a neural network architecture to process observational data and infer the causal structure as an adjacency matrix. Key components include:
- GNN-based message passing for structure learning
- Attention mechanism for handling variable dependencies
- Sparsity and acyclicity constraints to ensure valid DAGs

### Intervention Selection
The Parent-Scaled ACD algorithm selects intervention targets based on:
- Number of inferred parents (more parents = higher priority)
- Current uncertainty in edge predictions
- Balance between exploration and exploitation

### Full PARENT_SCALE_ACD Algorithm
The complete algorithm (as defined in `algorithms/PARENT_SCALE_ACD.py`) includes:
- Neural surrogate models for both structure and dynamics prediction
- Bayesian optimization for intervention selection
- Uncertainty quantification for robust decision making
- Cost-aware intervention planning

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size or graph size for larger models
- **Visualization Issues**: Ensure matplotlib is properly installed; try `--visualize False` if problems persist
- **Performance Issues**: Use the `--quick` flag for faster execution with fewer iterations

## Additional Resources

- See `examples/` directory for more detailed examples
- Refer to the main project documentation for API details
- Check `notebooks/` for interactive tutorials 