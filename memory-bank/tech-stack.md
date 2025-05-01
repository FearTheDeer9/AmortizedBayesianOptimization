# Tech Stack

This document lists the primary libraries, frameworks, and tools used or recommended for the Amortized Causal Bayesian Optimization project.

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

## Development & Workflow Aids

-   **Configuration:**
    -   `hydra-core`: Framework for elegantly configuring complex applications.
    -   `omegaconf`: Flexible configuration system (used by Hydra).
-   **Testing:**
    -   `pytest`: Framework for writing and running tests.
-   **Code Quality:**
    -   `ruff`: (Recommended) Linter and formatter (replaces `pylint`, `flake8`, `isort`, `black`).
    -   `pylint`: (Alternative) Static code analysis tool.
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

*This list has been updated to reflect the project's pivot to Amortized Causal Discovery, emphasizing neural network components and related tools.* 