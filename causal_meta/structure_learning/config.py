"""
Configuration for causal graph structure learning experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    Configuration parameters for causal graph structure learning experiments.

    Contains all the necessary parameters to run a causal graph structure learning
    experiment with a neural network approach, including graph generation, data
    generation, model, and training parameters.

    Attributes:
        num_nodes (int): Number of nodes in the causal graph
        edge_probability (float): Probability of edge between any two nodes
        num_obs_samples (int): Number of observational samples
        num_int_samples (int): Number of interventional samples per intervention
        max_iterations (int): Maximum number of interventions
        hidden_dim (int): Hidden dimension size for neural network
        learning_rate (float): Learning rate for optimizer
        noise_scale (float): Noise scale for linear SCM
        random_seed (Optional[int]): Random seed for reproducibility
        log_interval (int): Interval for logging during training
        device (str): Device to use for computation ('cpu' or 'cuda')
        threshold (float): Threshold for binarizing predicted adjacency matrix
    """

    num_nodes: int = 8
    edge_probability: float = 0.3
    num_obs_samples: int = 200
    num_int_samples: int = 20
    max_iterations: int = 50
    hidden_dim: int = 64
    learning_rate: float = 0.001
    noise_scale: float = 0.1
    random_seed: Optional[int] = 42
    log_interval: int = 5
    device: str = "cpu"
    threshold: float = 0.5

    def __post_init__(self):
        """Validate the config parameters."""
        assert 0 < self.num_nodes, "Number of nodes must be positive"
        assert 0 <= self.edge_probability <= 1, "Edge probability must be between 0 and 1"
        assert 0 < self.num_obs_samples, "Number of observational samples must be positive"
        assert 0 < self.num_int_samples, "Number of interventional samples must be positive"
        assert 0 < self.max_iterations, "Maximum iterations must be positive"
        assert 0 < self.hidden_dim, "Hidden dimension must be positive"
        assert 0 < self.learning_rate, "Learning rate must be positive"
        assert 0 < self.noise_scale, "Noise scale must be positive"
        assert 0 < self.log_interval, "Log interval must be positive"
        assert 0 <= self.threshold <= 1, "Threshold must be between 0 and 1"
        assert self.device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'" 