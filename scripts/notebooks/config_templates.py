#!/usr/bin/env python3
"""
Configuration Templates for GRPO Training and Evaluation

Provides standard configurations that ensure consistency between
training and evaluation, especially for optimization direction.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig


@dataclass
class TrainingModeConfig:
    """Configuration for different training modes."""
    episodes_per_scm: int
    episode_length: int
    learning_rate: float
    training_duration_minutes: int
    num_scms: int
    description: str
    
    # GRPO-specific settings
    group_size: int = 64
    interventions_per_state: int = 8
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.1  # Increased to prevent posterior collapse


# Standard training mode configurations
TRAINING_MODES = {
    "QUICK": TrainingModeConfig(
        episodes_per_scm=3,
        episode_length=8,
        learning_rate=0.001,
        training_duration_minutes=5,
        num_scms=32,
        description="Fast testing and development",
        group_size=32,
        interventions_per_state=4
    ),
    "STANDARD": TrainingModeConfig(
        episodes_per_scm=5,
        episode_length=10,
        learning_rate=0.001,
        training_duration_minutes=10,
        num_scms=48,
        description="Standard training run",
        group_size=64,
        interventions_per_state=8
    ),
    "FULL": TrainingModeConfig(
        episodes_per_scm=8,
        episode_length=12,
        learning_rate=0.001,
        training_duration_minutes=15,
        num_scms=64,
        description="Production-quality training",
        group_size=64,
        interventions_per_state=8
    ),
    "PRECISION": TrainingModeConfig(
        episodes_per_scm=15,
        episode_length=15,
        learning_rate=0.0005,
        training_duration_minutes=30,
        num_scms=128,
        description="Maximum quality training",
        group_size=128,
        interventions_per_state=16
    )
}


@dataclass
class ObjectiveConfig:
    """Configuration for different training objectives."""
    optimization_direction: str  # "MINIMIZE" or "MAXIMIZE"
    reward_weights: Dict[str, float]
    description: str
    
    # Additional settings for handling demonstrations
    convert_demonstrations: bool = False  # Whether to convert demo data
    demonstration_source: str = "PARENT_SCALE"  # Source of demonstrations


# Standard objective configurations
OBJECTIVE_CONFIGS = {
    "TARGET_MINIMIZE": ObjectiveConfig(
        optimization_direction="MINIMIZE",
        reward_weights={
            'optimization': 0.8,    # High weight on target minimization
            'discovery': 0.1,       # Low weight on structure discovery
            'efficiency': 0.1       # Low weight on efficiency
        },
        description="Minimize target variable (like PARENT_SCALE)",
        convert_demonstrations=False,  # PARENT_SCALE already minimizes
        demonstration_source="PARENT_SCALE"
    ),
    "TARGET_MAXIMIZE": ObjectiveConfig(
        optimization_direction="MAXIMIZE",
        reward_weights={
            'optimization': 0.8,    # High weight on target maximization
            'discovery': 0.1,       # Low weight on structure discovery
            'efficiency': 0.1       # Low weight on efficiency
        },
        description="Maximize target variable",
        convert_demonstrations=True,  # Need to convert PARENT_SCALE demos
        demonstration_source="PARENT_SCALE"
    ),
    "STRUCTURE_FOCUSED": ObjectiveConfig(
        optimization_direction="MAXIMIZE",  # Structure metrics are maximized
        reward_weights={
            'optimization': 0.2,    # Low weight on target
            'discovery': 0.6,       # High weight on structure discovery
            'efficiency': 0.2       # Medium weight on efficiency
        },
        description="Focus on structure discovery with some target optimization",
        convert_demonstrations=True,
        demonstration_source="PARENT_SCALE"
    ),
    "BALANCED": ObjectiveConfig(
        optimization_direction="MAXIMIZE",
        reward_weights={
            'optimization': 0.5,    # Balanced target optimization
            'discovery': 0.3,       # Medium structure discovery
            'efficiency': 0.2       # Medium efficiency
        },
        description="Balance all objectives",
        convert_demonstrations=True,
        demonstration_source="PARENT_SCALE"
    )
}


def create_training_config(
    mode: str = "QUICK",
    objective: str = "TARGET_MINIMIZE",
    random_seed: int = 42,
    checkpoint_dir: str = "checkpoints/grpo_training",
    custom_settings: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Create a complete training configuration.
    
    Args:
        mode: Training mode (QUICK, STANDARD, FULL, PRECISION)
        objective: Training objective (TARGET_MINIMIZE, TARGET_MAXIMIZE, etc.)
        random_seed: Random seed for reproducibility
        checkpoint_dir: Directory for saving checkpoints
        custom_settings: Optional custom settings to override defaults
        
    Returns:
        Complete configuration for training
    """
    if mode not in TRAINING_MODES:
        raise ValueError(f"Invalid training mode: {mode}. Choose from {list(TRAINING_MODES.keys())}")
    
    if objective not in OBJECTIVE_CONFIGS:
        raise ValueError(f"Invalid objective: {objective}. Choose from {list(OBJECTIVE_CONFIGS.keys())}")
    
    mode_config = TRAINING_MODES[mode]
    objective_config = OBJECTIVE_CONFIGS[objective]
    
    # Build configuration dictionary
    config_dict = {
        'mode': mode,
        'objective': objective,
        'seed': random_seed,
        
        # Optimization settings
        'optimization': {
            'direction': objective_config.optimization_direction,
            'target_baseline': 0.0,  # Can be customized
            'convert_demonstrations': objective_config.convert_demonstrations,
            'demonstration_source': objective_config.demonstration_source
        },
        
        # Training settings
        'training': {
            'n_episodes': mode_config.num_scms * mode_config.episodes_per_scm,
            'episode_length': mode_config.episode_length,
            'learning_rate': mode_config.learning_rate,
            'gamma': 0.99,
            'max_intervention_value': 2.0,
            
            # Reward weights
            'reward_weights': objective_config.reward_weights,
            
            # Architecture
            'architecture': {
                'hidden_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'key_size': 32,
                'widening_factor': 4,
                'dropout': 0.1,
                'policy_intermediate_dim': None
            },
            
            # State configuration
            'state_config': {
                'max_history_size': 100,
                'num_channels': 5,
                'standardize_values': True,
                'include_temporal_features': True
            },
            
            # GRPO configuration
            'grpo_config': {
                'group_size': mode_config.group_size,
                'interventions_per_state': mode_config.interventions_per_state,
                'clip_ratio': mode_config.clip_ratio,
                'entropy_coeff': mode_config.entropy_coeff,
                'kl_penalty_coeff': 0.0,
                'max_grad_norm': 1.0,
                'scale_rewards': True
            }
        },
        
        # Experiment settings
        'experiment': {
            # SCM generation (trainer expects it here)
            'scm_generation': {
                'num_scms': mode_config.num_scms,
                'variable_range': [3, 6],
                'structure_types': ['fork', 'chain', 'collider', 'mixed'],
                'noise_scale': 1.0,
                'edge_density_range': [0.3, 0.7],
                'use_variable_factory': True,  # Add this flag the trainer checks
                'rotation_frequency': 5,  # Add expected fields
                'fallback_scms': ['fork_3var', 'chain_3var', 'collider_3var']
            },
            'training_duration_minutes': mode_config.training_duration_minutes,
            'checkpoint_frequency': 50,
            'validation_frequency': 10,
            'early_stopping_patience': 5
        },
        
        # Logging
        'logging': {
            'checkpoint_dir': checkpoint_dir,
            'level': 'INFO',
            'save_frequency': 50,
            'wandb': {'enabled': False}  # Can be enabled if needed
        }
    }
    
    # Apply custom settings
    if custom_settings:
        config_dict = _deep_update(config_dict, custom_settings)
    
    return OmegaConf.create(config_dict)


def create_evaluation_config(
    checkpoint_path: str,
    num_test_scms: int = 10,
    runs_per_method: int = 3,
    intervention_budget: int = 10,
    random_seed: int = 42,
    output_dir: str = "results/evaluation",
    custom_settings: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Create evaluation configuration.
    
    Args:
        checkpoint_path: Path to checkpoint to evaluate
        num_test_scms: Number of test SCMs
        runs_per_method: Number of runs per method
        intervention_budget: Budget for interventions
        random_seed: Random seed
        output_dir: Output directory for results
        custom_settings: Optional custom settings
        
    Returns:
        Complete configuration for evaluation
    """
    config_dict = {
        'checkpoint_path': checkpoint_path,
        'seed': random_seed,
        
        # Test parameters
        'test_params': {
            'num_test_scms': num_test_scms,
            'runs_per_method': runs_per_method,
            'intervention_budget': intervention_budget,
            'scm_variable_range': [3, 6],
            'structure_types': ['fork', 'chain', 'collider', 'mixed']
        },
        
        # Baseline methods to compare against
        'baselines': {
            'random': True,
            'parent_scale': True,
            'uncertainty_sampling': True,
            'expected_improvement': True
        },
        
        # Output settings
        'output': {
            'dir': output_dir,
            'save_trajectories': True,
            'save_plots': True,
            'plot_format': 'png',
            'plot_dpi': 300
        },
        
        # Visualization
        'visualization': {
            'show_trajectories': True,
            'show_rankings': True,
            'show_objective_comparison': True,
            'interactive': False  # Set to True for Jupyter
        }
    }
    
    # Apply custom settings
    if custom_settings:
        config_dict = _deep_update(config_dict, custom_settings)
    
    return OmegaConf.create(config_dict)


def _deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep update dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


# Quick access functions for common configurations
def get_quick_minimize_config(**kwargs) -> DictConfig:
    """Get configuration for quick minimization training."""
    return create_training_config(mode="QUICK", objective="TARGET_MINIMIZE", **kwargs)


def get_quick_maximize_config(**kwargs) -> DictConfig:
    """Get configuration for quick maximization training."""
    return create_training_config(mode="QUICK", objective="TARGET_MAXIMIZE", **kwargs)


def get_production_minimize_config(**kwargs) -> DictConfig:
    """Get configuration for production minimization training."""
    return create_training_config(mode="FULL", objective="TARGET_MINIMIZE", **kwargs)


def get_production_maximize_config(**kwargs) -> DictConfig:
    """Get configuration for production maximization training."""
    return create_training_config(mode="FULL", objective="TARGET_MAXIMIZE", **kwargs)