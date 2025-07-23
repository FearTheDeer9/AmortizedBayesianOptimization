#!/usr/bin/env python3
"""
Checkpoint Utilities for GRPO Notebooks

Provides utilities for managing, loading, and converting checkpoints
with proper optimization direction handling.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from .base_components import (
    CheckpointMetadata, OptimizationConfig, NotebookError
)

logger = logging.getLogger(__name__)


def list_checkpoints(checkpoint_dir: Path, 
                    filter_by: Optional[Dict[str, Any]] = None) -> List[CheckpointMetadata]:
    """
    List all available checkpoints with optional filtering.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        filter_by: Optional filters (e.g., {'optimization_direction': 'MINIMIZE'})
        
    Returns:
        List of checkpoint metadata objects
    """
    checkpoints = []
    
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return checkpoints
    
    # Look for checkpoint directories
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Look for metadata
            metadata_paths = [
                item / "metadata.json",
                item.parent / f"{item.name}_metadata.json"
            ]
            
            for metadata_path in metadata_paths:
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        
                        checkpoint = CheckpointMetadata.from_dict(data, item)
                        
                        # Apply filters
                        if filter_by:
                            match = True
                            for key, value in filter_by.items():
                                if key == 'optimization_direction':
                                    if checkpoint.optimization_config.direction != value:
                                        match = False
                                        break
                                elif key == 'training_mode':
                                    if checkpoint.training_config.get('mode') != value:
                                        match = False
                                        break
                            
                            if match:
                                checkpoints.append(checkpoint)
                        else:
                            checkpoints.append(checkpoint)
                        
                        break
                        
                    except Exception as e:
                        logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
    
    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
    
    return checkpoints


def find_checkpoint_by_name(checkpoint_dir: Path, name: str) -> Optional[CheckpointMetadata]:
    """Find a specific checkpoint by name."""
    checkpoints = list_checkpoints(checkpoint_dir)
    
    for checkpoint in checkpoints:
        if checkpoint.name == name:
            return checkpoint
    
    return None


def load_checkpoint_params(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load actual model parameters from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dictionary of model parameters or None if not found
    """
    # Look for common checkpoint file patterns
    possible_files = [
        checkpoint_path / "checkpoint.pkl",
        checkpoint_path / "model_params.pkl",
        checkpoint_path / "policy_params.pkl",
        checkpoint_path / "params.pkl"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    params = pickle.load(f)
                return params
            except Exception as e:
                logger.warning(f"Failed to load params from {file_path}: {e}")
    
    # Also check for .pt files (PyTorch format)
    pt_files = list(checkpoint_path.glob("*.pt"))
    if pt_files:
        logger.info(f"Found PyTorch checkpoint files, but loading not implemented")
    
    return None


def convert_checkpoint_optimization(
    checkpoint_metadata: CheckpointMetadata,
    new_direction: str,
    new_checkpoint_dir: Path
) -> CheckpointMetadata:
    """
    Convert a checkpoint to a different optimization direction.
    
    This creates a new checkpoint with converted metadata. The actual
    model parameters remain the same, but the optimization configuration
    is updated to reflect the new direction.
    
    Args:
        checkpoint_metadata: Original checkpoint metadata
        new_direction: New optimization direction ("MINIMIZE" or "MAXIMIZE")
        new_checkpoint_dir: Directory for new checkpoint
        
    Returns:
        New checkpoint metadata with converted optimization
    """
    if new_direction not in ["MINIMIZE", "MAXIMIZE"]:
        raise ValueError(f"Invalid optimization direction: {new_direction}")
    
    # Create new optimization config
    new_opt_config = OptimizationConfig(
        direction=new_direction,
        target_baseline=checkpoint_metadata.optimization_config.target_baseline
    )
    
    # Generate new checkpoint name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_mode = checkpoint_metadata.training_config.get('mode', 'unknown').lower()
    new_name = f"grpo_{original_mode}_{new_direction.lower()}_converted_{timestamp}"
    new_path = new_checkpoint_dir / new_name
    
    # Create new metadata
    new_metadata = CheckpointMetadata(
        name=new_name,
        path=new_path,
        optimization_config=new_opt_config,
        training_config=checkpoint_metadata.training_config.copy(),
        training_results=checkpoint_metadata.training_results.copy(),
        timestamp=timestamp,
        grpo_version=checkpoint_metadata.grpo_version
    )
    
    # Add conversion info
    new_metadata.training_config['conversion_info'] = {
        'original_checkpoint': checkpoint_metadata.name,
        'original_direction': checkpoint_metadata.optimization_config.direction,
        'converted_at': timestamp,
        'converted_to': new_direction
    }
    
    # Create new checkpoint directory
    new_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model files if they exist
    if checkpoint_metadata.path.exists():
        for item in checkpoint_metadata.path.iterdir():
            if item.is_file() and item.suffix in ['.pkl', '.pt', '.pth']:
                shutil.copy2(item, new_path / item.name)
    
    # Save new metadata
    metadata_path = new_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(new_metadata.to_dict(), f, indent=2)
    
    logger.info(f"Created converted checkpoint: {new_name}")
    logger.info(f"  Original: {checkpoint_metadata.optimization_config.direction}")
    logger.info(f"  Converted: {new_direction}")
    
    return new_metadata


def validate_checkpoint_compatibility(
    checkpoint1: CheckpointMetadata,
    checkpoint2: CheckpointMetadata
) -> Tuple[bool, List[str]]:
    """
    Check if two checkpoints are compatible for comparison.
    
    Args:
        checkpoint1: First checkpoint
        checkpoint2: Second checkpoint
        
    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    warnings = []
    
    # Check version compatibility
    if checkpoint1.grpo_version != checkpoint2.grpo_version:
        warnings.append(
            f"Version mismatch: {checkpoint1.grpo_version} vs {checkpoint2.grpo_version}"
        )
    
    # Check training configuration compatibility
    config1 = checkpoint1.training_config
    config2 = checkpoint2.training_config
    
    # Check SCM configuration
    scm_config1 = config1.get('scm_config', {})
    scm_config2 = config2.get('scm_config', {})
    
    if scm_config1.get('variable_range') != scm_config2.get('variable_range'):
        warnings.append(
            f"Variable range mismatch: {scm_config1.get('variable_range')} vs "
            f"{scm_config2.get('variable_range')}"
        )
    
    # Check reward weights (important for fair comparison)
    weights1 = config1.get('reward_weights', {})
    weights2 = config2.get('reward_weights', {})
    
    if weights1 != weights2:
        warnings.append(
            f"Reward weight mismatch - this may affect comparison fairness"
        )
    
    # Different optimization directions is expected for COMPARE_OBJECTIVES mode
    if checkpoint1.optimization_config.direction != checkpoint2.optimization_config.direction:
        warnings.append(
            f"Different optimization directions: {checkpoint1.optimization_config.direction} vs "
            f"{checkpoint2.optimization_config.direction} (expected for objective comparison)"
        )
    
    # Determine compatibility
    is_compatible = len(warnings) == 0 or (
        len(warnings) == 1 and "optimization directions" in warnings[0]
    )
    
    return is_compatible, warnings


def create_checkpoint_summary_table(checkpoints: List[CheckpointMetadata]) -> str:
    """
    Create a formatted summary table of checkpoints.
    
    Args:
        checkpoints: List of checkpoint metadata
        
    Returns:
        Formatted string table
    """
    if not checkpoints:
        return "No checkpoints found."
    
    # Prepare table data
    headers = ["Name", "Optimization", "Mode", "Episodes", "Duration", "Date"]
    rows = []
    
    for ckpt in checkpoints:
        name = ckpt.name[:30] + "..." if len(ckpt.name) > 30 else ckpt.name
        opt_dir = ckpt.optimization_config.direction
        mode = ckpt.training_config.get('mode', 'unknown')
        episodes = ckpt.training_results.get('episodes_completed', 'N/A')
        duration = ckpt.training_results.get('duration_minutes', 0)
        duration_str = f"{duration:.1f}m" if isinstance(duration, (int, float)) else "N/A"
        
        # Parse timestamp
        try:
            if ckpt.timestamp and ckpt.timestamp != 'unknown':
                if len(ckpt.timestamp) == 15:  # Format: 20250722_120000
                    dt = datetime.strptime(ckpt.timestamp, "%Y%m%d_%H%M%S")
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = ckpt.timestamp[:10]
            else:
                date_str = "unknown"
        except:
            date_str = ckpt.timestamp
        
        rows.append([name, opt_dir, mode, str(episodes), duration_str, date_str])
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    # Build table
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in rows:
        row_line = " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths))
        lines.append(row_line)
    
    return "\n".join(lines)


def get_latest_checkpoint(
    checkpoint_dir: Path,
    optimization_direction: Optional[str] = None
) -> Optional[CheckpointMetadata]:
    """
    Get the most recent checkpoint, optionally filtered by optimization direction.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        optimization_direction: Optional filter for optimization direction
        
    Returns:
        Most recent checkpoint or None if not found
    """
    filters = {}
    if optimization_direction:
        filters['optimization_direction'] = optimization_direction
    
    checkpoints = list_checkpoints(checkpoint_dir, filter_by=filters)
    
    if checkpoints:
        return checkpoints[0]  # Already sorted by timestamp
    
    return None