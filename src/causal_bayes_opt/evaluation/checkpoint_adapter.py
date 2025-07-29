"""
Checkpoint Adapter for Evaluators

This module provides adapters to make new simplified trainer checkpoints
compatible with existing evaluators.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

logger = logging.getLogger(__name__)


class CheckpointAdapter:
    """Adapter to convert between checkpoint formats."""
    
    @staticmethod
    def load_grpo_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load GRPO checkpoint and adapt to evaluator format.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            
        Returns:
            Adapted checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Handle different path formats
        if checkpoint_path.is_dir():
            # Look for checkpoint files in directory
            pkl_files = list(checkpoint_path.glob("*.pkl"))
            if pkl_files:
                checkpoint_file = pkl_files[0]
            else:
                # Try final_checkpoint.pkl
                checkpoint_file = checkpoint_path / "final_checkpoint.pkl"
                if not checkpoint_file.exists():
                    raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        else:
            checkpoint_file = checkpoint_path
            
        # Load checkpoint
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
            
        # Adapt format for evaluator
        adapted = {
            "policy_params": checkpoint.get("params", checkpoint.get("policy_params")),
            "config": checkpoint.get("config", {}),
            "optimization_direction": checkpoint.get("config", {}).get("optimization_direction", "MINIMIZE"),
            "metadata": checkpoint.get("metadata", {})
        }
        
        # Add architecture info if available
        if "architecture_level" in checkpoint.get("config", {}):
            adapted["config"]["architecture"] = {
                "level": checkpoint["config"]["architecture_level"],
                "type": "enhanced"
            }
            
        logger.info(f"Loaded GRPO checkpoint from {checkpoint_file}")
        return adapted
        
    @staticmethod
    def load_bc_checkpoint(checkpoint_path: Path, expected_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load BC checkpoint and adapt to evaluator format.
        
        Args:
            checkpoint_path: Path to checkpoint file
            expected_type: Expected model type ("surrogate" or "acquisition")
            
        Returns:
            Adapted checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            
        model_type = checkpoint.get("model_type", checkpoint.get("metadata", {}).get("model_type"))
        
        # Verify type if specified
        if expected_type and model_type != expected_type:
            logger.warning(f"Expected {expected_type} model but got {model_type}")
            
        # Adapt format
        adapted = {
            "model_params": checkpoint.get("params", checkpoint.get("model_params")),
            "model_type": model_type,
            "config": checkpoint.get("config", {}),
            "metadata": checkpoint.get("metadata", {})
        }
        
        logger.info(f"Loaded BC {model_type} checkpoint from {checkpoint_path}")
        return adapted
        
    @staticmethod
    def create_grpo_evaluator_checkpoint(policy_params: Any, config: Dict[str, Any]) -> str:
        """
        Create a temporary checkpoint file in the format expected by GRPOEvaluator.
        
        Args:
            policy_params: Policy parameters
            config: Configuration
            
        Returns:
            Path to temporary checkpoint file
        """
        import tempfile
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="grpo_eval_"))
        
        # Save checkpoint.pkl in expected format
        checkpoint_data = {
            "policy_params": policy_params,
            "policy_config": config,
            "training_config": config,  # GRPOEvaluator expects this
            "optimization_direction": config.get("optimization_direction", "MINIMIZE")
        }
        
        checkpoint_file = temp_dir / "checkpoint.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)
            
        logger.info(f"Created temporary checkpoint at {temp_dir}")
        return str(temp_dir)