"""
Configuration utilities for the causal_meta library.

This module contains utilities for loading and managing configuration using Hydra.
"""
import os
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf


def get_config_path() -> str:
    """
    Get the path to the config directory.

    Returns:
        str: Path to the config directory
    """
    # Get the path relative to the project root
    base_path = os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, "configs")


def load_config(config_name: str = "config") -> DictConfig:
    """
    Load configuration from the config directory.

    Args:
        config_name: Name of the config file (without extension)

    Returns:
        DictConfig: Configuration object
    """
    config_path = get_config_path()
    config_file = os.path.join(config_path, f"{config_name}.yaml")

    # Ensure the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Load the configuration using OmegaConf
    return OmegaConf.load(config_file)


def save_config(config: DictConfig, config_name: str) -> None:
    """
    Save a configuration to the config directory.

    Args:
        config: Configuration object
        config_name: Name of the config file (without extension)
    """
    config_path = get_config_path()
    config_file = os.path.join(config_path, f"{config_name}.yaml")

    # Save the configuration
    with open(config_file, 'w') as f:
        f.write(OmegaConf.to_yaml(config))


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations, with the override config taking precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration that overrides the base

    Returns:
        DictConfig: Merged configuration
    """
    return OmegaConf.merge(base_config, override_config)


def get_default_config() -> DictConfig:
    """
    Get the default configuration.

    Returns:
        DictConfig: Default configuration
    """
    return load_config("config")


def update_config_from_args(config: DictConfig, args_dict: Dict[str, Any]) -> DictConfig:
    """
    Update configuration from a dictionary of arguments.

    Args:
        config: Configuration to update
        args_dict: Dictionary of arguments

    Returns:
        DictConfig: Updated configuration
    """
    # Convert args dict to OmegaConf
    args_conf = OmegaConf.create(args_dict)

    # Merge with the base config
    return OmegaConf.merge(config, args_conf)
