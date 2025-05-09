# Tech Stack

This document lists the primary libraries, frameworks, and tools used or recommended for the Amortized Causal Meta-Learning Framework.

## Core Libraries

-   **Python:** 3.10+
-   **Numerical Computation:**
    -   `numpy`: Fundamental package for scientific computing.
    -   `scipy`: Scientific and technical computing library.
    -   `jax`: High-performance numerical computation and automatic differentiation (optional for specialized models).
-   **Data Handling:**
    -   `pandas`: Data analysis and manipulation tool.
-   **Machine Learning & Statistics:**
    -   `scikit-learn`: General machine learning tools (used for utilities, evaluation metrics).
    -   `statsmodels`: Statistical models, hypothesis testing (for baseline comparisons).
-   **Deep Learning (Primary for Amortized Causal Discovery):**
    -   `torch` (PyTorch): Primary deep learning framework.
    -   `torch_geometric`: Extension library for graph neural networks.
    -   `einops`: Tensor operations made simple (for implementing attention mechanisms).
    -   `pytorch-lightning`: High-level training framework (optional).
    -   `functorch`: Function transforms for JAX-like capabilities in PyTorch.
    -   `higher`: Enables higher-order gradient optimization (for meta-learning).
-   **Causal Inference & Graphs:**
    -   `networkx`: Creating, manipulating, and studying complex networks.
    -   `igraph`: High-performance graph library.
    -   `causaldag`: Tools for representing and learning causal DAGs.
    -   `cdt` (Causal Discovery Toolbox): Library for causal discovery algorithms (baseline comparisons).
    -   `pgmpy`: Library for working with Probabilistic Graphical Models.
    -   Project-specific modules: `causal_meta.graph`, `causal_meta.environments`
-   **Bayesian Optimization & Acquisition Functions:**
    -   `botorch`: Framework for Bayesian optimization (acquisition function implementations).
    -   `dibs`: Deep implicit Bayesian structure learning (potentially useful for comparison).
-   **Meta-Learning & Neural Causality:**
    -   Custom implementation of Amortized Causal Discovery.
    -   Custom implementation of meta-learning for neural causal models.
    -   `jaxtyping`: Type annotations for tensor shapes and dtypes.
-   **Uncertainty Estimation:**
    -   `pyro`: Probabilistic programming framework for implementing Bayesian neural networks.
    -   `torch.distributions`: PyTorch's probability distributions for ensemble methods.
    -   `botorch.posteriors`: Posterior distributions for uncertainty quantification.
    -   `botorch.acquisition`: Acquisition functions that leverage uncertainty.
    -   Custom implementation of uncertainty estimation strategies.
-   **Configuration Management:**
    -   `hydra-core`: Framework for elegantly configuring complex applications.
    -   `omegaconf`: Flexible configuration system (used by Hydra).
    -   `pydantic`: Data validation and settings management using Python type hints.
    -   `yaml`: YAML parser and emitter for Python.

## Development & Workflow Aids

-   **Configuration:**
    -   `hydra-core`: Framework for elegantly configuring complex applications.
    -   `omegaconf`: Flexible configuration system (used by Hydra).
    -   `pydantic`: Data validation and settings management.
    -   `configupdater`: Library for updating configuration files programmatically.
    -   `yaml`: YAML parser and emitter for Python.
-   **Interface Design:**
    -   `abc`: Python's Abstract Base Classes for interface definition.
    -   `typing`: Type hints for interface contracts and validation.
    -   `inspect`: Introspection module for runtime interface checking.
    -   `functools`: Function decorators for interface implementation.
-   **Dependency Injection:**
    -   `dependency_injector`: Container and factories for dependency injection.
    -   Custom dependency injection system.
-   **Testing:**
    -   `pytest`: Framework for writing and running tests.
    -   `pytest-mock`: Mock objects and patching for pytest.
    -   `pytest-cov`: Coverage reports for pytest.
    -   `hypothesis`: Property-based testing for Python.
    -   `pytest-benchmark`: Benchmark timing for pytest.
-   **Code Quality:**
    -   `ruff`: (Recommended) Linter and formatter (replaces `pylint`, `flake8`, `isort`, `black`).
    -   `pylint`: (Alternative) Static code analysis tool.
    -   `mypy`: Static type checker for Python.
    -   `pre-commit`: Git hooks for code quality checks.
-   **Documentation:**
    -   `sphinx`: Documentation generator.
    -   `sphinx-autodoc`: Automatic API documentation.
    -   `sphinx-rtd-theme`: Read the Docs theme for Sphinx.
    -   `nbsphinx`: Jupyter notebook support for Sphinx.
    -   `docstring-parser`: Parse docstrings for documentation.
-   **Progress Bars:**
    -   `tqdm`: Fast, extensible progress bar.
-   **Visualization:**
    -   `matplotlib`: Comprehensive library for creating static, animated, and interactive visualizations.
    -   `seaborn`: (Recommended) Statistical data visualization based on matplotlib.
    -   `networkx.drawing`: Graph visualization.
    -   `plotly`: Interactive visualizations for complex model outputs.
-   **Experiment Tracking:**
    -   `tensorboard`: Visualization tool for tracking training metrics.
    -   `wandb`: (Recommended) Experiment tracking and visualization for deep learning.
-   **Dependency Management:**
    -   `pip` with `requirements.txt` (current).
    -   `poetry` or `rye`: (Recommended) Modern dependency management tools.
-   **Environment Management:**
    -   `venv` (current).
    -   `conda`: (Alternative) Package, dependency, and environment management.

## Hardware Requirements

-   **GPU Support:** Recommended for training neural network models, especially for larger graphs.
-   **Memory:** 16GB+ RAM recommended for larger graph experiments.
-   **Storage:** Additional space for model checkpoints and synthetic datasets.

## Benchmarking Framework

The benchmarking framework integrates several technologies to provide comprehensive evaluation capabilities:

### Core Libraries

- **NumPy/SciPy**: For efficient numerical computation and data manipulation
- **pandas**: For structured data handling and analysis
- **matplotlib/seaborn**: For visualization of benchmark results
- **NetworkX**: For graph operations and metrics
- **PyTorch**: For compatibility with neural model evaluation
- **scikit-learn**: For metrics implementation and analysis

### Performance Monitoring

- **Python's `resource` module**: For memory usage tracking
- **psutil**: For cross-platform system resource monitoring
- **GPUtil**: For GPU memory monitoring when using CUDA devices
- **time/timeit**: For precise timing measurements
- **signal**: For implementing timeouts to handle non-terminating methods

### Statistical Analysis

- **SciPy.stats**: For statistical testing and p-value calculation
- **statsmodels**: For regression analysis in scaling curve fitting
- **numpy.polyfit**: For polynomial fitting to analyze scaling behavior

### Data Management

- **json/pickle**: For serialization of benchmark results and configurations
- **os/pathlib**: For file and directory operations
- **tqdm**: For progress tracking during long benchmark runs
- **yaml**: For configuration file parsing and generation
- **pydantic**: For validating and serializing complex configurations

### Integration Points

- **GraphFactory**: Custom component for test graph generation
- **StructuralCausalModel**: For data generation and intervention simulation
- **AmortizedCausalDiscovery**: For neural model compatibility
- **AmortizedCBO**: For neural optimization methods
- **Interface-based adapters**: For connecting different components

## YAML Configuration System

The YAML Configuration System enables easy experimentation with different model architectures and hyperparameters:

### Components

- **ConfigurationManager**: Central system for loading and validating configurations
- **Schema Validation**: Using Pydantic for configuration validation
- **Default Configurations**: Pre-defined configurations for common use cases
- **Configuration Inheritance**: Support for extending base configurations
- **Command Line Overrides**: Ability to override configuration values from command line
- **Environment Variable Substitution**: Support for environment variables in configurations

### Example Configuration Structure

```yaml
# Example configuration for a GNN-based structure inference model
model:
  type: GraphEncoder
  params:
    hidden_dim: 64
    num_layers: 3
    attention_heads: 4
    dropout: 0.1
    use_batch_norm: true

training:
  optimizer:
    type: Adam
    params:
      lr: 0.001
      weight_decay: 1e-5
  scheduler:
    type: CosineAnnealingLR
    params:
      T_max: 100
  num_epochs: 200
  batch_size: 32
  early_stopping:
    patience: 10
    min_delta: 0.001

evaluation:
  metrics:
    - shd
    - precision
    - recall
    - f1
  test_size: 0.2
  seed: 42
```

This technology stack is designed to support the interface-first refactoring approach, providing tools for dependency injection, comprehensive testing, and flexible configuration. 