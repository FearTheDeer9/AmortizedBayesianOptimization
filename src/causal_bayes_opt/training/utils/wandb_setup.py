"""
Centralized WandB setup and management for Hydra-based training.

This module provides unified WandB configuration that replaces scattered
wandb.init() calls throughout the training components.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class WandBManager:
    """Centralized WandB management with Hydra integration."""
    
    def __init__(self):
        self.run = None
        self.enabled = False
        
    def setup(self, config: DictConfig, experiment_name: Optional[str] = None) -> Optional[Any]:
        """
        Setup WandB from Hydra configuration.
        
        Args:
            config: Hydra configuration containing wandb settings
            experiment_name: Optional experiment name override
            
        Returns:
            WandB run object if enabled, None otherwise
        """
        try:
            import wandb
        except ImportError:
            logger.warning("WandB not available, disabling logging")
            self.enabled = False
            return None
        
        wandb_config = config.get('logging', {}).get('wandb', {})
        self.enabled = wandb_config.get('enabled', False)
        
        if not self.enabled:
            logger.info("WandB logging disabled")
            return None
        
        try:
            # Extract WandB configuration
            project = wandb_config.get('project', 'causal-bayes-opt')
            entity = wandb_config.get('entity', None)
            tags = list(wandb_config.get('tags', []))
            group = wandb_config.get('group', None)
            notes = wandb_config.get('notes', '')
            
            # Generate run name
            run_name = wandb_config.get('name', None)
            if run_name is None:
                run_name = experiment_name or f"training_{int(time.time())}"
            
            # Add configuration-based tags
            if hasattr(config, 'training') and hasattr(config.training, 'architecture'):
                tags.append(f"arch_{config.training.architecture.get('level', 'unknown')}")
            if hasattr(config, 'experiment'):
                tags.extend(config.experiment.get('tags', []))
            
            # Initialize WandB
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                tags=tags,
                group=group,
                notes=notes,
                config=OmegaConf.to_container(config, resolve=True),
                reinit=True  # Allow multiple inits in same session
            )
            
            logger.info(f"WandB initialized: {wandb.run.url}")
            return self.run
            
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}, disabling logging")
            self.enabled = False
            return None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB if enabled."""
        if not self.enabled or self.run is None:
            return
        
        try:
            import wandb
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file", name: Optional[str] = None) -> None:
        """Log artifact to WandB if enabled."""
        if not self.enabled or self.run is None:
            return
        
        try:
            import wandb
            artifact_name = name or Path(artifact_path).stem
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
            logger.info(f"Logged artifact: {artifact_path}")
        except Exception as e:
            logger.warning(f"WandB artifact logging failed: {e}")
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self.enabled and self.run is not None:
            try:
                import wandb
                wandb.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.warning(f"WandB cleanup failed: {e}")
            finally:
                self.run = None
                self.enabled = False


# Global WandB manager instance
_wandb_manager = WandBManager()


def setup_wandb_from_config(
    config: DictConfig, 
    experiment_name: Optional[str] = None
) -> Optional[Any]:
    """
    Setup WandB from Hydra configuration using global manager.
    
    Args:
        config: Hydra configuration
        experiment_name: Optional experiment name
        
    Returns:
        WandB run object if enabled, None otherwise
    """
    return _wandb_manager.setup(config, experiment_name)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics using global WandB manager."""
    _wandb_manager.log(metrics, step)


def log_artifact(artifact_path: str, artifact_type: str = "file", name: Optional[str] = None) -> None:
    """Log artifact using global WandB manager."""
    _wandb_manager.log_artifact(artifact_path, artifact_type, name)


def finish_wandb() -> None:
    """Finish WandB run using global manager."""
    _wandb_manager.finish()


def is_wandb_enabled() -> bool:
    """Check if WandB is enabled."""
    return _wandb_manager.enabled