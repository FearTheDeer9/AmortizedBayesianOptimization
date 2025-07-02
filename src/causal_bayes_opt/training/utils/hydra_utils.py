"""
Hydra utility functions for configuration management and validation.
"""

import logging
from typing import Any, Optional, Dict, List
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def validate_hydra_config(config: DictConfig) -> List[str]:
    """
    Validate Hydra configuration and return list of issues.
    
    Args:
        config: Hydra configuration to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Validate required sections
    required_sections = ['training', 'paths', 'curriculum']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")
    
    # Validate paths
    if 'paths' in config:
        paths_config = config.paths
        required_dirs = ['base_dir', 'checkpoint_base_dir', 'results_base_dir']
        
        for dir_key in required_dirs:
            if dir_key in paths_config:
                try:
                    Path(paths_config[dir_key]).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {dir_key}: {e}")
    
    # Validate training configuration
    if 'training' in config:
        training_config = config.training
        
        # Check learning rate
        if 'learning_rate' in training_config:
            lr = training_config.learning_rate
            if lr <= 0:
                issues.append("Learning rate must be positive")
        
        # Check training steps
        if 'n_training_steps' in training_config:
            steps = training_config.n_training_steps
            if steps <= 0:
                issues.append("Training steps must be positive")
    
    # Validate curriculum configuration
    if 'curriculum' in config:
        curriculum_config = config.curriculum
        
        # Check difficulty levels
        if 'difficulty_levels' in curriculum_config:
            levels = curriculum_config.difficulty_levels
            if not levels or len(levels) == 0:
                issues.append("At least one difficulty level must be specified")
    
    # Validate WandB configuration if enabled
    if 'logging' in config and 'wandb' in config.logging:
        wandb_config = config.logging.wandb
        if wandb_config.get('enabled', False):
            if not wandb_config.get('project'):
                issues.append("WandB project name required when WandB is enabled")
    
    return issues


def get_config_value(config: DictConfig, key_path: str, default: Any = None) -> Any:
    """
    Get value from nested configuration using dot notation.
    
    Args:
        config: Hydra configuration
        key_path: Dot-separated path (e.g., 'training.learning_rate')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    try:
        keys = key_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def override_config_value(config: DictConfig, key_path: str, value: Any) -> None:
    """
    Override configuration value using dot notation.
    
    Args:
        config: Hydra configuration to modify
        key_path: Dot-separated path (e.g., 'training.learning_rate')
        value: New value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value


def log_config_to_wandb(config: DictConfig) -> None:
    """
    Log configuration to WandB if enabled.
    
    Args:
        config: Hydra configuration to log
    """
    try:
        from .wandb_setup import is_wandb_enabled, log_metrics
        
        if is_wandb_enabled():
            # Convert config to flat dict for better WandB visualization
            flat_config = flatten_config(config)
            log_metrics({"config/" + k: v for k, v in flat_config.items()})
            logger.info("Configuration logged to WandB")
    except Exception as e:
        logger.warning(f"Failed to log config to WandB: {e}")


def flatten_config(config: DictConfig, parent_key: str = "", separator: str = ".") -> Dict[str, Any]:
    """
    Flatten nested configuration into a flat dictionary.
    
    Args:
        config: Configuration to flatten
        parent_key: Parent key prefix
        separator: Separator for nested keys
        
    Returns:
        Flattened configuration dictionary
    """
    items = []
    
    if isinstance(config, DictConfig):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = config
    
    for key, value in config_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_config(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def create_experiment_name(config: DictConfig) -> str:
    """
    Create descriptive experiment name from configuration.
    
    Args:
        config: Hydra configuration
        
    Returns:
        Generated experiment name
    """
    components = []
    
    # Add training type
    training_type = get_config_value(config, 'training.architecture.level', 'unknown')
    components.append(training_type)
    
    # Add experiment type
    exp_name = get_config_value(config, 'experiment.name', '')
    if exp_name:
        components.append(exp_name.replace('_experiment', ''))
    
    # Add difficulty if single level
    difficulty_levels = get_config_value(config, 'curriculum.difficulty_levels', [])
    if len(difficulty_levels) == 1:
        components.append(difficulty_levels[0])
    
    # Add steps
    steps = get_config_value(config, 'training.n_training_steps', 0)
    if steps > 0:
        components.append(f"{steps}steps")
    
    return "_".join(components) if components else "training"


def save_config_summary(config: DictConfig, output_path: str) -> None:
    """
    Save configuration summary to file.
    
    Args:
        config: Hydra configuration
        output_path: Path to save summary
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Configuration Summary\n\n")
            f.write(OmegaConf.to_yaml(config))
        
        logger.info(f"Configuration summary saved to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to save config summary: {e}")