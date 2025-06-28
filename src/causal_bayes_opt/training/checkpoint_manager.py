"""
Comprehensive checkpoint management for ACBO training.

This module provides robust, versioned checkpointing with metadata,
integrity verification, and automatic cleanup following functional principles.
"""

import jax
import jax.numpy as jnp
import json
import pickle
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import pyrsistent as pyr
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Immutable checkpoint metadata."""
    checkpoint_id: str
    timestamp: float
    stage: str
    version: str
    file_size_bytes: int
    content_hash: str
    training_step: int
    error_count: int
    recovery_count: int
    git_commit: Optional[str] = None
    config_hash: Optional[str] = None
    user_notes: str = ""


@dataclass(frozen=True)
class CheckpointInfo:
    """Immutable checkpoint information."""
    path: Path
    metadata: CheckpointMetadata
    is_valid: bool
    verification_status: str


@dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpoint management."""
    max_checkpoints: int = 10
    auto_cleanup: bool = True
    compression_enabled: bool = True
    integrity_verification: bool = True
    backup_to_remote: bool = False
    save_frequency_steps: int = 100
    save_on_error: bool = True


class CheckpointManager:
    """
    Manages training checkpoints with versioning, metadata, and integrity verification.
    
    Features:
    - Versioned checkpoints with metadata
    - Automatic cleanup of old checkpoints
    - Integrity verification with checksums
    - Atomic saves (write to temp, then move)
    - Robust error handling
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        config: Optional[CheckpointConfig] = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or CheckpointConfig()
        
        # Create checkpoint directory structure
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.checkpoint_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.temp_dir = self.checkpoint_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized CheckpointManager at {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        state: Any,
        checkpoint_name: str,
        stage: str = "unknown",
        user_notes: str = ""
    ) -> CheckpointInfo:
        """
        Save training state with comprehensive metadata.
        
        Args:
            state: Training state to save
            checkpoint_name: Name for the checkpoint
            stage: Current training stage
            user_notes: Optional user notes
            
        Returns:
            CheckpointInfo with save details
        """
        timestamp = time.time()
        checkpoint_id = f"{checkpoint_name}_{int(timestamp)}"
        
        # Create checkpoint and metadata paths
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        metadata_path = self.metadata_dir / f"{checkpoint_id}.json"
        temp_checkpoint_path = self.temp_dir / f"{checkpoint_id}_temp.pkl"
        
        try:
            # Atomic save: write to temp file first
            with open(temp_checkpoint_path, 'wb') as f:
                if self.config.compression_enabled:
                    import gzip
                    with gzip.open(f, 'wb') as gz_f:
                        pickle.dump(state, gz_f)
                else:
                    pickle.dump(state, f)
            
            # Calculate file size and hash
            file_size = temp_checkpoint_path.stat().st_size
            content_hash = self._calculate_file_hash(temp_checkpoint_path)
            
            # Extract metadata from state
            training_step = getattr(state, 'training_step', 0)
            error_count = getattr(state.error_context, 'total_errors', 0) if hasattr(state, 'error_context') else 0
            recovery_count = getattr(state, 'recovery_count', 0)
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                stage=stage,
                version="1.0",  # Could be made configurable
                file_size_bytes=file_size,
                content_hash=content_hash,
                training_step=training_step,
                error_count=error_count,
                recovery_count=recovery_count,
                user_notes=user_notes
            )
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Atomic move from temp to final location
            shutil.move(str(temp_checkpoint_path), str(checkpoint_path))
            
            # Verify the saved checkpoint
            is_valid, verification_status = self._verify_checkpoint(checkpoint_path, metadata)
            
            checkpoint_info = CheckpointInfo(
                path=checkpoint_path,
                metadata=metadata,
                is_valid=is_valid,
                verification_status=verification_status
            )
            
            logger.info(f"💾 Saved checkpoint {checkpoint_id} ({file_size} bytes)")
            
            # Auto-cleanup if enabled
            if self.config.auto_cleanup:
                self._cleanup_old_checkpoints()
            
            return checkpoint_info
            
        except Exception as e:
            # Cleanup temp file on error
            if temp_checkpoint_path.exists():
                temp_checkpoint_path.unlink()
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path, CheckpointInfo]
    ) -> Any:
        """
        Load checkpoint with integrity verification.
        
        Args:
            checkpoint_path: Path to checkpoint or CheckpointInfo
            
        Returns:
            Loaded training state
        """
        if isinstance(checkpoint_path, CheckpointInfo):
            path = checkpoint_path.path
            metadata = checkpoint_path.metadata
        else:
            path = Path(checkpoint_path)
            metadata = self._load_metadata(path)
        
        # Verify checkpoint integrity if enabled
        if self.config.integrity_verification:
            is_valid, status = self._verify_checkpoint(path, metadata)
            if not is_valid:
                raise RuntimeError(f"Checkpoint verification failed: {status}")
        
        try:
            with open(path, 'rb') as f:
                if self.config.compression_enabled and metadata.file_size_bytes != path.stat().st_size:
                    # Assume compressed if file sizes don't match
                    import gzip
                    with gzip.open(f, 'rb') as gz_f:
                        state = pickle.load(gz_f)
                else:
                    state = pickle.load(f)
            
            logger.info(f"📂 Loaded checkpoint from {path}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            raise
    
    def list_checkpoints(self, stage: Optional[str] = None) -> List[CheckpointInfo]:
        """
        List available checkpoints, optionally filtered by stage.
        
        Args:
            stage: Optional stage filter
            
        Returns:
            List of CheckpointInfo sorted by timestamp (newest first)
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                metadata = self._load_metadata(checkpoint_file)
                
                # Filter by stage if specified
                if stage and metadata.stage != stage:
                    continue
                
                is_valid, verification_status = self._verify_checkpoint(checkpoint_file, metadata)
                
                checkpoint_info = CheckpointInfo(
                    path=checkpoint_file,
                    metadata=metadata,
                    is_valid=is_valid,
                    verification_status=verification_status
                )
                checkpoints.append(checkpoint_info)
                
            except Exception as e:
                logger.warning(f"Skipping invalid checkpoint {checkpoint_file}: {e}")
                continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.metadata.timestamp, reverse=True)
        return checkpoints
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[CheckpointInfo]:
        """Get the most recent checkpoint, optionally filtered by stage."""
        checkpoints = self.list_checkpoints(stage)
        return checkpoints[0] if checkpoints else None
    
    def delete_checkpoint(self, checkpoint_info: CheckpointInfo) -> None:
        """Delete a checkpoint and its metadata."""
        try:
            # Delete checkpoint file
            if checkpoint_info.path.exists():
                checkpoint_info.path.unlink()
            
            # Delete metadata file
            metadata_path = self.metadata_dir / f"{checkpoint_info.metadata.checkpoint_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"🗑️ Deleted checkpoint {checkpoint_info.metadata.checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_info.metadata.checkpoint_id}: {e}")
            raise
    
    def _load_metadata(self, checkpoint_path: Path) -> CheckpointMetadata:
        """Load metadata for a checkpoint."""
        checkpoint_id = checkpoint_path.stem
        metadata_path = self.metadata_dir / f"{checkpoint_id}.json"
        
        if not metadata_path.exists():
            # Create minimal metadata for legacy checkpoints
            return CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                timestamp=checkpoint_path.stat().st_mtime,
                stage="unknown",
                version="legacy",
                file_size_bytes=checkpoint_path.stat().st_size,
                content_hash="unknown",
                training_step=0,
                error_count=0,
                recovery_count=0
            )
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        return CheckpointMetadata(**metadata_dict)
    
    def _verify_checkpoint(
        self,
        checkpoint_path: Path,
        metadata: CheckpointMetadata
    ) -> Tuple[bool, str]:
        """Verify checkpoint integrity."""
        if not self.config.integrity_verification:
            return True, "verification_disabled"
        
        try:
            # Check file exists
            if not checkpoint_path.exists():
                return False, "file_not_found"
            
            # Check file size
            actual_size = checkpoint_path.stat().st_size
            if actual_size != metadata.file_size_bytes:
                return False, f"size_mismatch: expected {metadata.file_size_bytes}, got {actual_size}"
            
            # Check content hash (skip for legacy checkpoints)
            if metadata.content_hash != "unknown":
                actual_hash = self._calculate_file_hash(checkpoint_path)
                if actual_hash != metadata.content_hash:
                    return False, f"hash_mismatch: expected {metadata.content_hash}, got {actual_hash}"
            
            return True, "verified"
            
        except Exception as e:
            return False, f"verification_error: {e}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the maximum limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) > self.config.max_checkpoints:
            # Keep the most recent checkpoints, delete the rest
            to_delete = checkpoints[self.config.max_checkpoints:]
            
            for checkpoint_info in to_delete:
                try:
                    self.delete_checkpoint(checkpoint_info)
                except Exception as e:
                    logger.warning(f"Failed to cleanup checkpoint {checkpoint_info.metadata.checkpoint_id}: {e}")


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    config: Optional[CheckpointConfig] = None
) -> CheckpointManager:
    """Create a CheckpointManager with default configuration."""
    return CheckpointManager(checkpoint_dir, config)


__all__ = [
    'CheckpointMetadata',
    'CheckpointInfo',
    'CheckpointConfig',
    'CheckpointManager',
    'create_checkpoint_manager'
]