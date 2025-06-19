"""
Configuration system for acquisition model training with validation and best practices.

This module provides comprehensive configuration management for the acquisition training
pipeline, incorporating 2024 research findings and best practices for GRPO and
verifiable rewards in causal intervention selection.
"""

# Standard library imports
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# Third-party imports
import jax.numpy as jnp
import pyrsistent as pyr

logger = logging.getLogger(__name__)


@dataclass
class VerifiableRewardConfig:
    """Configuration for mathematically verifiable reward components."""
    
    # Reward component weights (sum should be reasonable, individual weights >= 0)
    optimization_weight: float = 1.0  # Target variable improvement
    structure_weight: float = 0.5     # Information gain in structure discovery
    parent_weight: float = 0.3        # Parent intervention guidance  
    exploration_weight: float = 0.1   # Intervention diversity bonus
    
    # Reward computation parameters
    optimization_scale: float = 1.0   # Scale factor for tanh normalization
    structure_threshold: float = 0.1  # Minimum uncertainty reduction for reward
    exploration_decay: float = 0.9    # Decay factor for exploration bonus
    
    # Reward hacking prevention
    max_reward_magnitude: float = 5.0  # Maximum allowable reward value
    min_reward_variance: float = 0.01  # Minimum variance to prevent gaming
    suspicious_reward_threshold: float = 2.0  # Flag suspiciously high rewards
    
    def validate(self) -> bool:
        """Validate reward configuration for common issues."""
        if not all(w >= 0 for w in [self.optimization_weight, self.structure_weight, 
                                   self.parent_weight, self.exploration_weight]):
            logger.error("All reward weights must be non-negative")
            return False
        
        total_weight = (self.optimization_weight + self.structure_weight + 
                       self.parent_weight + self.exploration_weight)
        if total_weight <= 0:
            logger.error("Total reward weight must be positive")
            return False
        
        if self.optimization_scale <= 0:
            logger.error("Optimization scale must be positive")
            return False
        
        if not (0 < self.structure_threshold < 1):
            logger.error("Structure threshold must be in (0, 1)")
            return False
        
        return True
    
    def to_pyrsistent(self) -> pyr.PMap:
        """Convert to pyrsistent format for reward computation."""
        return pyr.m(**{
            'reward_weights': {
                'optimization': self.optimization_weight,
                'structure': self.structure_weight,
                'parent': self.parent_weight,
                'exploration': self.exploration_weight
            },
            'optimization_scale': self.optimization_scale,
            'structure_threshold': self.structure_threshold,
            'exploration_weight': self.exploration_weight,
            'exploration_decay': self.exploration_decay
        })


@dataclass
class EnhancedGRPOConfig:
    """Enhanced GRPO configuration incorporating 2024 research findings."""
    
    # Core GRPO parameters (DeepSeek R1 best practices)
    group_size: int = 64               # Appropriate for intervention selection
    clip_ratio: float = 0.2            # PPO-style clipping
    entropy_coeff: float = 0.01        # Exploration regularization
    kl_penalty_coeff: float = 0.0      # ✅ Remove KL divergence (2024 best practice)
    learning_rate: float = 3e-4        # Standard policy learning rate
    max_grad_norm: float = 1.0         # Gradient clipping for stability
    
    # 2024 enhancements for our use case
    adaptive_advantage_scaling: bool = True   # Handle multi-objective variance
    single_update_per_batch: bool = True      # Training stability
    reward_hacking_detection: bool = True     # Monitor exploitation patterns
    variance_threshold: float = 2.0           # Threshold for adaptive scaling
    
    # Training control
    max_episodes: int = 10000
    evaluation_frequency: int = 500
    early_stopping_patience: int = 5
    convergence_threshold: float = 0.01  # Stop when improvement < threshold
    
    # Memory and performance
    use_jax_compilation: bool = True
    batch_accumulation_steps: int = 1    # For memory-constrained settings
    
    def validate(self) -> bool:
        """Validate GRPO configuration."""
        if self.group_size < 8:
            logger.error("Group size too small for stable GRPO training")
            return False
        
        if not (0.1 <= self.clip_ratio <= 0.5):
            logger.error("Clip ratio should be in [0.1, 0.5]")
            return False
        
        if self.entropy_coeff < 0:
            logger.error("Entropy coefficient must be non-negative")
            return False
        
        if self.kl_penalty_coeff != 0.0:
            logger.warning("KL penalty should be 0.0 for best performance (2024 finding)")
        
        if self.learning_rate <= 0:
            logger.error("Learning rate must be positive")
            return False
        
        return True
    
    def get_memory_optimized_config(self) -> 'EnhancedGRPOConfig':
        """Get memory-optimized version for resource-constrained training."""
        return EnhancedGRPOConfig(
            group_size=min(32, self.group_size),
            clip_ratio=self.clip_ratio,
            entropy_coeff=self.entropy_coeff,
            kl_penalty_coeff=0.0,  # Always 0.0
            learning_rate=self.learning_rate,
            max_grad_norm=self.max_grad_norm,
            adaptive_advantage_scaling=True,
            single_update_per_batch=True,
            reward_hacking_detection=self.reward_hacking_detection,
            max_episodes=self.max_episodes,
            evaluation_frequency=self.evaluation_frequency * 2,  # Less frequent
            early_stopping_patience=self.early_stopping_patience,
            use_jax_compilation=True,
            batch_accumulation_steps=4  # Accumulate gradients
        )


@dataclass
class BehavioralCloningConfig:
    """Configuration for behavioral cloning warm-start phase."""
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Regularization
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    
    # Data augmentation (for intervention data)
    augment_data: bool = False
    augmentation_factor: float = 1.5
    noise_level: float = 0.01
    
    # JAX compilation
    use_jax_compilation: bool = True
    static_argnums: Tuple[int, ...] = (2, 3)
    
    # Evaluation
    accuracy_threshold: float = 0.9   # Target accuracy for BC phase
    
    def validate(self) -> bool:
        """Validate BC configuration."""
        if self.learning_rate <= 0:
            logger.error("Learning rate must be positive")
            return False
        
        if self.batch_size < 1:
            logger.error("Batch size must be positive")
            return False
        
        if not (0 < self.validation_split < 1):
            logger.error("Validation split must be in (0, 1)")
            return False
        
        if self.weight_decay < 0:
            logger.error("Weight decay must be non-negative")
            return False
        
        return True


@dataclass
class PolicyNetworkConfig:
    """Configuration for the acquisition policy network architecture."""
    
    # Network architecture
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8               # For attention mechanisms
    dropout: float = 0.1
    
    # Policy-specific parameters
    exploration_noise: float = 0.1
    variable_selection_temp: float = 1.0
    value_selection_temp: float = 1.0
    
    # Input processing
    max_history_size: int = 100      # Fixed size for consistent compilation
    standardize_inputs: bool = True
    
    def validate(self) -> bool:
        """Validate policy network configuration."""
        if self.hidden_dim < 32:
            logger.error("Hidden dimension too small")
            return False
        
        if self.num_layers < 1:
            logger.error("Must have at least one layer")
            return False
        
        if self.hidden_dim % self.num_heads != 0:
            logger.error("Hidden dimension must be divisible by number of heads")
            return False
        
        if not (0 <= self.dropout < 1):
            logger.error("Dropout must be in [0, 1)")
            return False
        
        return True


@dataclass
class DataConfig:
    """Configuration for training data requirements and processing."""
    
    # Expert demonstration requirements
    min_expert_trajectories: int = 100
    target_expert_trajectories: int = 500
    max_expert_trajectories: int = 2000
    
    # Data quality filters
    min_trajectory_length: int = 5
    max_trajectory_length: int = 50
    filter_incomplete_trajectories: bool = True
    
    # Data augmentation
    augment_expert_data: bool = False
    synthetic_data_ratio: float = 0.0  # Ratio of synthetic to real data
    
    # Validation splits
    validation_split: float = 0.2
    test_split: float = 0.1
    
    def validate(self) -> bool:
        """Validate data configuration."""
        if self.min_expert_trajectories > self.target_expert_trajectories:
            logger.error("Minimum trajectories cannot exceed target")
            return False
        
        if self.target_expert_trajectories > self.max_expert_trajectories:
            logger.error("Target trajectories cannot exceed maximum")
            return False
        
        if self.min_trajectory_length < 1:
            logger.error("Minimum trajectory length must be positive")
            return False
        
        total_split = self.validation_split + self.test_split
        if total_split >= 1.0:
            logger.error("Validation + test split must be < 1.0")
            return False
        
        return True


@dataclass
class TrainingConfig:
    """Complete configuration for acquisition model training pipeline."""
    
    # Component configurations
    bc_config: BehavioralCloningConfig = field(default_factory=BehavioralCloningConfig)
    grpo_config: EnhancedGRPOConfig = field(default_factory=EnhancedGRPOConfig)
    policy_config: PolicyNetworkConfig = field(default_factory=PolicyNetworkConfig)
    reward_config: VerifiableRewardConfig = field(default_factory=VerifiableRewardConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Global training settings
    random_seed: int = 42
    use_gpu: bool = True
    mixed_precision: bool = False     # JAX mixed precision training
    
    # Checkpointing and logging
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1000
    logging_frequency: int = 100
    save_final_model: bool = True
    
    # Experiment tracking
    experiment_name: str = "acquisition_training"
    track_metrics: bool = True
    save_training_curves: bool = True
    
    def validate(self) -> bool:
        """Validate complete training configuration."""
        # Validate all component configs
        if not self.bc_config.validate():
            logger.error("Invalid behavioral cloning configuration")
            return False
        
        if not self.grpo_config.validate():
            logger.error("Invalid GRPO configuration")
            return False
        
        if not self.policy_config.validate():
            logger.error("Invalid policy network configuration")
            return False
        
        if not self.reward_config.validate():
            logger.error("Invalid reward configuration")
            return False
        
        if not self.data_config.validate():
            logger.error("Invalid data configuration")
            return False
        
        # Cross-component validation
        if self.grpo_config.group_size > self.data_config.target_expert_trajectories:
            logger.warning("GRPO group size larger than target trajectory count")
        
        return True
    
    def get_production_config(self) -> 'TrainingConfig':
        """Get production-ready configuration with conservative settings."""
        production_config = TrainingConfig(
            bc_config=BehavioralCloningConfig(
                learning_rate=5e-4,  # More conservative
                epochs=100,          # More epochs
                early_stopping_patience=15
            ),
            grpo_config=EnhancedGRPOConfig(
                group_size=64,
                max_episodes=20000,  # More episodes
                evaluation_frequency=250,  # More frequent evaluation
                early_stopping_patience=10
            ),
            policy_config=PolicyNetworkConfig(
                hidden_dim=256,      # Larger network
                num_layers=6,        # Deeper network
                dropout=0.05         # Less dropout
            ),
            reward_config=self.reward_config,  # Keep reward config
            data_config=DataConfig(
                target_expert_trajectories=1000,  # More data
                validation_split=0.15,            # Less validation, more training
                test_split=0.05
            ),
            save_checkpoints=True,
            checkpoint_frequency=500,
            logging_frequency=50
        )
        
        return production_config
    
    def get_debug_config(self) -> 'TrainingConfig':
        """Get debug configuration for fast iteration."""
        debug_config = TrainingConfig(
            bc_config=BehavioralCloningConfig(
                epochs=5,
                batch_size=8,
                early_stopping_patience=3
            ),
            grpo_config=EnhancedGRPOConfig(
                group_size=16,
                max_episodes=100,
                evaluation_frequency=25,
                early_stopping_patience=3
            ),
            policy_config=PolicyNetworkConfig(
                hidden_dim=64,
                num_layers=2,
                num_heads=4
            ),
            reward_config=self.reward_config,
            data_config=DataConfig(
                min_expert_trajectories=10,
                target_expert_trajectories=20,
                max_expert_trajectories=50
            ),
            save_checkpoints=False,
            logging_frequency=10
        )
        
        return debug_config


# Configuration factory functions

def create_standard_config() -> TrainingConfig:
    """Create standard training configuration with proven defaults."""
    return TrainingConfig()


def create_high_performance_config() -> TrainingConfig:
    """Create high-performance configuration for production use."""
    return TrainingConfig().get_production_config()


def create_memory_efficient_config() -> TrainingConfig:
    """Create memory-efficient configuration for resource-constrained environments."""
    config = TrainingConfig()
    config.grpo_config = config.grpo_config.get_memory_optimized_config()
    config.bc_config.batch_size = 16
    config.policy_config.hidden_dim = 64
    config.policy_config.num_layers = 3
    return config


def validate_config_compatibility(config: TrainingConfig) -> List[str]:
    """
    Validate configuration compatibility and return list of warnings.
    
    Args:
        config: Training configuration to validate
        
    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []
    
    # Check BC -> GRPO transition compatibility
    if config.bc_config.accuracy_threshold > 0.95 and config.grpo_config.learning_rate > 1e-3:
        warnings.append("High BC accuracy threshold with high GRPO learning rate may cause instability")
    
    # Check reward weights vs GRPO settings
    total_reward_weight = (config.reward_config.optimization_weight + 
                          config.reward_config.structure_weight +
                          config.reward_config.parent_weight + 
                          config.reward_config.exploration_weight)
    
    if total_reward_weight > 3.0 and config.grpo_config.clip_ratio < 0.15:
        warnings.append("High reward weights with low clip ratio may cause training instability")
    
    # Check data requirements
    if (config.data_config.target_expert_trajectories < 200 and 
        config.grpo_config.max_episodes > 5000):
        warnings.append("Low expert data with many GRPO episodes may lead to overfitting")
    
    # Check network capacity
    if (config.policy_config.hidden_dim < 128 and 
        config.data_config.target_expert_trajectories > 1000):
        warnings.append("Small network with large dataset may underfit")
    
    return warnings


def get_recommended_config_for_problem_size(n_variables: int, n_expert_trajectories: int) -> TrainingConfig:
    """
    Get recommended configuration based on problem characteristics.
    
    Args:
        n_variables: Number of variables in the causal graphs
        n_expert_trajectories: Available expert trajectories
        
    Returns:
        Recommended training configuration
    """
    # Scale configuration based on problem size
    if n_variables <= 5:
        # Small problems
        hidden_dim = 64
        num_layers = 3
        group_size = 32
    elif n_variables <= 10:
        # Medium problems  
        hidden_dim = 128
        num_layers = 4
        group_size = 64
    else:
        # Large problems
        hidden_dim = 256
        num_layers = 6
        group_size = 128
    
    # Scale training based on data availability
    if n_expert_trajectories < 100:
        # Limited data - be conservative
        bc_epochs = 100
        grpo_episodes = 5000
        learning_rate = 1e-4
    elif n_expert_trajectories < 500:
        # Moderate data
        bc_epochs = 50
        grpo_episodes = 10000
        learning_rate = 3e-4
    else:
        # Abundant data
        bc_epochs = 30
        grpo_episodes = 15000
        learning_rate = 5e-4
    
    return TrainingConfig(
        bc_config=BehavioralCloningConfig(
            learning_rate=learning_rate,
            epochs=bc_epochs
        ),
        grpo_config=EnhancedGRPOConfig(
            group_size=group_size,
            max_episodes=grpo_episodes,
            learning_rate=learning_rate
        ),
        policy_config=PolicyNetworkConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
    )