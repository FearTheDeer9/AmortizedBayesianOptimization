"""
Configuration for causal graph structure learning experiments.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Union


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

    def validate(self) -> Optional[str]:
        """
        Validate the configuration parameters.
        
        Returns:
            Error message string if validation fails, None otherwise.
        """
        if self.num_nodes < 2:
            return "Number of nodes must be at least 2"
        
        if not 0 <= self.edge_probability <= 1:
            return "Edge probability must be between 0 and 1"
        
        if self.noise_scale < 0:
            return "Noise scale must be non-negative"
        
        if self.hidden_dim < 1:
            return "Hidden dimension must be positive"
        
        if self.max_iterations < 1:
            return "Maximum iterations must be positive"
        
        if self.learning_rate <= 0:
            return "Learning rate must be positive"
        
        if self.log_interval < 1:
            return "Log interval must be positive"
        
        if self.threshold < 0 or self.threshold > 1:
            return "Threshold must be between 0 and 1"
        
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            ExperimentConfig instance
        """
        return cls(**config_dict)


@dataclass
class ProgressiveInterventionConfig(ExperimentConfig):
    """
    Configuration for progressive intervention experiments.
    
    This class extends ExperimentConfig with parameters specific to
    progressive intervention experiments, where interventions are
    iteratively selected to improve causal graph structure learning.
    
    Args:
        num_iterations: Number of intervention iterations
        num_obs_samples: Number of observational samples
        num_int_samples: Number of interventional samples per iteration
        acquisition_strategy: Strategy for selecting interventions
            ('uncertainty', 'random', 'information_gain')
        int_budget: Budget constraint for interventions
        int_values: Optional fixed values for interventions
        save_checkpoints: Whether to save model checkpoints
        evaluation_metrics: List of metrics to track during experiment
        output_dir: Directory to save results
        experiment_name: Name of the experiment
        
        # Training parameters
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        epochs: Number of training epochs
        early_stopping_patience: Patience for early stopping
        sparsity_weight: Weight for sparsity regularization
        acyclicity_weight: Weight for acyclicity regularization
        dropout: Dropout rate
        num_layers: Number of layers in neural network
        weight_decay: Weight decay for optimizer
        
        # Anti-bias parameters
        pos_weight: Weight for positive examples in BCE loss
        consistency_weight: Weight for consistency regularization
        edge_prob_bias: Bias for edge probability (positive values encourage edges)
        expected_density: Expected edge density (if None, no density regularization)
        density_weight: Weight for density regularization 
    """
    # Intervention settings
    num_iterations: int = 5
    num_obs_samples: int = 100
    num_int_samples: int = 20
    acquisition_strategy: str = "uncertainty"  # Options: uncertainty, random, information_gain
    int_budget: float = 1.0
    
    # Optional fixed intervention values
    int_values: Optional[List[float]] = None
    
    # Experiment settings
    save_checkpoints: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "shd", "f1"])
    output_dir: str = "results"
    experiment_name: str = "progressive_intervention"
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    sparsity_weight: float = 0.1
    acyclicity_weight: float = 1.0
    dropout: float = 0.1
    num_layers: int = 2
    weight_decay: float = 0.0
    
    # Anti-bias parameters 
    pos_weight: float = 5.0  # Weight for positive examples in BCE loss
    consistency_weight: float = 0.1  # Weight for consistency regularization
    edge_prob_bias: float = 0.1  # Bias for edge probability
    expected_density: Optional[float] = 0.3  # Expected edge density
    density_weight: float = 0.1  # Weight for density regularization
    
    def validate(self) -> Optional[str]:
        """
        Validate the configuration parameters.
        
        Returns:
            Error message string if validation fails, None otherwise.
        """
        # First validate base parameters
        base_validation = super().validate()
        if base_validation is not None:
            return base_validation
        
        # Validate progressive intervention parameters
        if self.num_iterations < 1:
            return "Number of iterations must be positive"
        
        if self.num_obs_samples < 1:
            return "Number of observational samples must be positive"
        
        if self.num_int_samples < 1:
            return "Number of interventional samples must be positive"
        
        valid_strategies = ["uncertainty", "random", "information_gain"]
        if self.acquisition_strategy not in valid_strategies:
            return f"Acquisition strategy must be one of {valid_strategies}"
        
        if self.int_budget <= 0:
            return "Intervention budget must be positive"
            
        # Validate anti-bias parameters
        if self.pos_weight < 0:
            return "Positive weight must be non-negative"
            
        if self.consistency_weight < 0:
            return "Consistency weight must be non-negative"
            
        if self.expected_density is not None and not 0 <= self.expected_density <= 1:
            return "Expected density must be between 0 and 1"
            
        if self.density_weight < 0:
            return "Density weight must be non-negative"
            
        return None
    
    def copy(self) -> 'ProgressiveInterventionConfig':
        """
        Create a copy of the configuration.
        
        Returns:
            New ProgressiveInterventionConfig instance
        """
        return ProgressiveInterventionConfig(**self.to_dict()) 