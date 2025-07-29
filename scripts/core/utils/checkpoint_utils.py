"""
Checkpoint Utilities for ACBO Framework

Provides unified checkpoint management for all model types (GRPO, BC, baselines).
Handles saving, loading, and version compatibility.
"""

import pickle
import json
import gzip
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
from datetime import datetime
import jax
import jax.numpy as jnp
import numpy as onp

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Unified checkpoint management for ACBO models."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize checkpoint manager.
        
        Args:
            base_dir: Base directory for checkpoints
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_grpo_checkpoint(
        self,
        policy_params: Any,
        config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        episode: int,
        optimization_direction: str = "MINIMIZE"
    ) -> Path:
        """
        Save GRPO checkpoint with all necessary data.
        
        Args:
            policy_params: JAX policy parameters
            config: Training configuration
            training_metrics: Training history and metrics
            episode: Current episode number
            optimization_direction: MINIMIZE or MAXIMIZE
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"grpo_{optimization_direction.lower()}_ep{episode}_{timestamp}"
        checkpoint_dir = self.base_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save policy parameters
        policy_path = checkpoint_dir / "policy_params.pkl"
        with open(policy_path, 'wb') as f:
            pickle.dump(policy_params, f)
            
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # Save training metrics
        metrics_path = checkpoint_dir / "training_metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(training_metrics, f)
            
        # Save metadata
        metadata = {
            "model_type": "grpo",
            "optimization_direction": optimization_direction,
            "episode": episode,
            "timestamp": timestamp,
            "checkpoint_version": "1.0"
        }
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved GRPO checkpoint to {checkpoint_dir}")
        return checkpoint_dir
        
    def save_bc_checkpoint(
        self,
        model_params: Any,
        model_type: str,  # "surrogate" or "acquisition"
        config: Dict[str, Any],
        training_metrics: Dict[str, Any],
        epoch: int
    ) -> Path:
        """
        Save BC checkpoint.
        
        Args:
            model_params: Model parameters
            model_type: "surrogate" or "acquisition"
            config: Training configuration
            training_metrics: Training history
            epoch: Current epoch
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"bc_{model_type}_ep{epoch}_{timestamp}"
        checkpoint_dir = self.base_dir / checkpoint_name
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model parameters
        params_path = checkpoint_dir / f"{model_type}_params.pkl"
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)
            
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        # Save training metrics
        metrics_path = checkpoint_dir / "training_metrics.pkl"
        with open(metrics_path, 'wb') as f:
            pickle.dump(training_metrics, f)
            
        # Save metadata
        metadata = {
            "model_type": f"bc_{model_type}",
            "epoch": epoch,
            "timestamp": timestamp,
            "checkpoint_version": "1.0"
        }
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved BC {model_type} checkpoint to {checkpoint_dir}")
        return checkpoint_dir
        
    def load_grpo_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load GRPO checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary containing:
                - policy_params: Model parameters
                - config: Training config
                - training_metrics: Training history
                - metadata: Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Verify model type
        if metadata["model_type"] != "grpo":
            raise ValueError(f"Expected GRPO checkpoint, got {metadata['model_type']}")
            
        # Load policy parameters
        policy_path = checkpoint_path / "policy_params.pkl"
        with open(policy_path, 'rb') as f:
            policy_params = pickle.load(f)
            
        # Load config
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Load training metrics
        metrics_path = checkpoint_path / "training_metrics.pkl"
        if metrics_path.exists():
            with open(metrics_path, 'rb') as f:
                training_metrics = pickle.load(f)
        else:
            training_metrics = {}
            
        logger.info(f"Loaded GRPO checkpoint from {checkpoint_path}")
        
        return {
            "policy_params": policy_params,
            "config": config,
            "training_metrics": training_metrics,
            "metadata": metadata
        }
        
    def load_bc_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load BC checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary containing model data
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Determine model type
        model_type = metadata["model_type"]
        if not model_type.startswith("bc_"):
            raise ValueError(f"Expected BC checkpoint, got {model_type}")
            
        # Extract specific type (surrogate or acquisition)
        specific_type = model_type.replace("bc_", "")
        
        # Load model parameters
        params_path = checkpoint_path / f"{specific_type}_params.pkl"
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
            
        # Load config
        config_path = checkpoint_path / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Load training metrics
        metrics_path = checkpoint_path / "training_metrics.pkl"
        if metrics_path.exists():
            with open(metrics_path, 'rb') as f:
                training_metrics = pickle.load(f)
        else:
            training_metrics = {}
            
        logger.info(f"Loaded BC {specific_type} checkpoint from {checkpoint_path}")
        
        return {
            "model_params": model_params,
            "model_type": specific_type,
            "config": config,
            "training_metrics": training_metrics,
            "metadata": metadata
        }
        
    def find_latest_checkpoint(self, model_type: str = "grpo") -> Optional[Path]:
        """
        Find the latest checkpoint for a given model type.
        
        Args:
            model_type: Type of model ("grpo", "bc_surrogate", "bc_acquisition")
            
        Returns:
            Path to latest checkpoint or None
        """
        pattern = f"{model_type}_*"
        checkpoints = list(self.base_dir.glob(pattern))
        
        if not checkpoints:
            return None
            
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]
        
    def list_checkpoints(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Args:
            model_type: Optional filter by model type
            
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = []
        
        for checkpoint_dir in self.base_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
                
            metadata_path = checkpoint_dir / "metadata.json"
            if not metadata_path.exists():
                continue
                
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                if model_type and metadata.get("model_type") != model_type:
                    continue
                    
                info = {
                    "path": checkpoint_dir,
                    "name": checkpoint_dir.name,
                    "model_type": metadata.get("model_type"),
                    "timestamp": metadata.get("timestamp"),
                    "episode": metadata.get("episode"),
                    "epoch": metadata.get("epoch")
                }
                checkpoints.append(info)
                
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_dir}: {e}")
                
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints


def convert_params_for_jax(params: Any) -> Any:
    """
    Convert parameters to JAX arrays if needed.
    
    Args:
        params: Model parameters (possibly numpy)
        
    Returns:
        Parameters as JAX arrays
    """
    if isinstance(params, dict):
        return {k: convert_params_for_jax(v) for k, v in params.items()}
    elif isinstance(params, list):
        return [convert_params_for_jax(v) for v in params]
    elif isinstance(params, tuple):
        return tuple(convert_params_for_jax(v) for v in params)
    elif isinstance(params, onp.ndarray):
        return jnp.array(params)
    else:
        return params


def extract_model_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract basic information about a checkpoint without fully loading it.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    
    info = {
        "exists": checkpoint_path.exists(),
        "path": str(checkpoint_path)
    }
    
    if not checkpoint_path.exists():
        return info
        
    # Try to load metadata
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                info.update(metadata)
        except Exception as e:
            info["metadata_error"] = str(e)
            
    # Check for model files
    info["has_policy_params"] = (checkpoint_path / "policy_params.pkl").exists()
    info["has_surrogate_params"] = (checkpoint_path / "surrogate_params.pkl").exists()
    info["has_acquisition_params"] = (checkpoint_path / "acquisition_params.pkl").exists()
    info["has_config"] = (checkpoint_path / "config.json").exists()
    
    return info