"""
Comprehensive checkpoint management for ACBO modular training system.

This module provides robust, versioned checkpointing with metadata,
integrity verification, and automatic cleanup following functional principles.
Enhanced for modular 3-component training system with curriculum support.
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

# Import checkpoint dataclasses
from .checkpoint_dataclasses import (
    TrainingCheckpoint, SurrogateCheckpoint, AcquisitionCheckpoint, JointCheckpoint
)

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


@dataclass(frozen=True)
class ComponentCheckpointMetadata:
    """Enhanced metadata for component-specific checkpoints."""
    checkpoint_id: str
    component_type: str
    difficulty_level: str
    training_step: int
    timestamp: float
    performance_metrics: Dict[str, float]
    config_hash: str
    file_size_bytes: int
    content_hash: str
    tags: List[str] = None
    experiment_name: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            object.__setattr__(self, 'tags', [])


@dataclass(frozen=True)
class CheckpointCollection:
    """Collection of related checkpoints across components."""
    collection_id: str
    experiment_name: str
    surrogate_checkpoints: Dict[str, str] = None  # difficulty -> checkpoint_path
    acquisition_checkpoints: Dict[str, str] = None
    joint_checkpoints: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    created_timestamp: float = None
    
    def __post_init__(self):
        if self.surrogate_checkpoints is None:
            object.__setattr__(self, 'surrogate_checkpoints', {})
        if self.acquisition_checkpoints is None:
            object.__setattr__(self, 'acquisition_checkpoints', {})
        if self.joint_checkpoints is None:
            object.__setattr__(self, 'joint_checkpoints', {})
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if self.created_timestamp is None:
            object.__setattr__(self, 'created_timestamp', time.time())


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


class ModularCheckpointManager:
    """
    Enhanced checkpoint manager for modular 3-component training system.
    
    Provides component-specific checkpoint management with curriculum support,
    checkpoint collections, and independent component training coordination.
    """
    
    def __init__(self, base_dir: str = "checkpoints"):
        self.base_dir = Path(base_dir)
        
        # Create component-specific directories
        self.surrogate_dir = self.base_dir / "surrogate"
        self.acquisition_dir = self.base_dir / "acquisition"
        self.joint_dir = self.base_dir / "joint"
        self.collections_dir = self.base_dir / "collections"
        self.metadata_dir = self.base_dir / "metadata"
        
        for directory in [self.surrogate_dir, self.acquisition_dir, self.joint_dir, 
                         self.collections_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint registries
        self.checkpoint_registry = {}
        self.collection_registry = {}
        
        # Load existing registries
        self._load_registries()
        
        logger.info(f"Initialized ModularCheckpointManager with base directory: {self.base_dir}")
    
    def save_component_checkpoint(
        self, 
        checkpoint: TrainingCheckpoint,
        experiment_name: str = "",
        tags: Optional[List[str]] = None,
        keep_best_n: int = 5
    ) -> str:
        """
        Save component checkpoint with enhanced metadata.
        
        Args:
            checkpoint: Training checkpoint to save
            experiment_name: Name of the experiment
            tags: Optional tags for organization
            keep_best_n: Number of best checkpoints to keep per difficulty
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint ID
        timestamp = int(time.time())
        checkpoint_id = f"{checkpoint.component_type}_{checkpoint.difficulty_level}_{timestamp}"
        
        # Determine save directory
        if checkpoint.component_type == "surrogate":
            checkpoint_path = self.surrogate_dir / f"{checkpoint_id}.pkl"
        elif checkpoint.component_type == "acquisition":
            checkpoint_path = self.acquisition_dir / f"{checkpoint_id}.pkl"
        elif checkpoint.component_type == "joint":
            checkpoint_path = self.joint_dir / f"{checkpoint_id}.pkl"
        else:
            raise ValueError(f"Unknown component type: {checkpoint.component_type}")
        
        # Update checkpoint path
        checkpoint.checkpoint_path = str(checkpoint_path)
        
        try:
            # Save checkpoint data
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Calculate file metadata
            file_size = checkpoint_path.stat().st_size
            content_hash = self._calculate_file_hash(checkpoint_path)
            
            # Create enhanced metadata
            metadata = ComponentCheckpointMetadata(
                checkpoint_id=checkpoint_id,
                component_type=checkpoint.component_type,
                difficulty_level=checkpoint.difficulty_level,
                training_step=checkpoint.training_step,
                timestamp=checkpoint.timestamp,
                performance_metrics=checkpoint.performance_metrics,
                config_hash=self._compute_config_hash(checkpoint.config),
                file_size_bytes=file_size,
                content_hash=content_hash,
                tags=tags or [],
                experiment_name=experiment_name
            )
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{checkpoint_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            
            # Update registry
            registry_key = f"{checkpoint.component_type}_{checkpoint.difficulty_level}"
            if registry_key not in self.checkpoint_registry:
                self.checkpoint_registry[registry_key] = []
            
            self.checkpoint_registry[registry_key].append({
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": str(checkpoint_path),
                "metadata_path": str(metadata_path),
                "performance_metrics": checkpoint.performance_metrics,
                "timestamp": checkpoint.timestamp,
                "experiment_name": experiment_name
            })
            
            # Keep only best N checkpoints
            self._cleanup_component_checkpoints(registry_key, keep_best_n)
            
            # Save updated registry
            self._save_registries()
            
            logger.info(f"Saved {checkpoint.component_type} checkpoint {checkpoint_id} for {checkpoint.difficulty_level}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            # Cleanup partial saves
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise
    
    def load_component_checkpoint(self, checkpoint_path: str) -> TrainingCheckpoint:
        """Load component checkpoint from path."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            logger.info(f"Loaded {checkpoint.component_type} checkpoint from {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            raise
    
    def find_best_component_checkpoint(
        self, 
        component_type: str, 
        difficulty_level: str,
        metric: str = "auto"
    ) -> Optional[str]:
        """
        Find best checkpoint for component and difficulty.
        
        Args:
            component_type: 'surrogate', 'acquisition', or 'joint'
            difficulty_level: Curriculum difficulty level
            metric: Metric to optimize ('auto' for component-specific default)
            
        Returns:
            Path to best checkpoint or None
        """
        registry_key = f"{component_type}_{difficulty_level}"
        
        if registry_key not in self.checkpoint_registry:
            return None
        
        checkpoints = self.checkpoint_registry[registry_key]
        if not checkpoints:
            return None
        
        # Determine metric
        if metric == "auto":
            if component_type == "surrogate":
                metric = "validation_accuracy"
            elif component_type == "acquisition":
                metric = "final_avg_reward"
            else:
                metric = "joint_final_reward"
        
        # Find best checkpoint
        best_checkpoint = None
        best_score = float('-inf')
        
        for checkpoint_info in checkpoints:
            metrics = checkpoint_info.get("performance_metrics", {})
            score = metrics.get(metric, float('-inf'))
            
            if score > best_score:
                best_score = score
                best_checkpoint = checkpoint_info
        
        return best_checkpoint["checkpoint_path"] if best_checkpoint else None
    
    def create_checkpoint_collection(
        self,
        experiment_name: str,
        surrogate_paths: Optional[Dict[str, str]] = None,
        acquisition_paths: Optional[Dict[str, str]] = None,
        joint_paths: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a collection of related checkpoints across components."""
        collection_id = f"{experiment_name}_{int(time.time())}"
        
        collection = CheckpointCollection(
            collection_id=collection_id,
            experiment_name=experiment_name,
            surrogate_checkpoints=surrogate_paths or {},
            acquisition_checkpoints=acquisition_paths or {},
            joint_checkpoints=joint_paths or {},
            metadata=metadata or {}
        )
        
        # Save collection
        collection_path = self.collections_dir / f"{collection_id}.json"
        with open(collection_path, 'w') as f:
            json.dump(asdict(collection), f, indent=2)
        
        # Update registry
        self.collection_registry[collection_id] = {
            "collection_path": str(collection_path),
            "experiment_name": experiment_name,
            "created_timestamp": collection.created_timestamp
        }
        
        self._save_registries()
        
        logger.info(f"Created checkpoint collection {collection_id}")
        return collection_id
    
    def load_checkpoint_collection(self, collection_id: str) -> CheckpointCollection:
        """Load checkpoint collection by ID."""
        if collection_id not in self.collection_registry:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection_path = self.collection_registry[collection_id]["collection_path"]
        
        with open(collection_path, 'r') as f:
            collection_data = json.load(f)
        
        return CheckpointCollection(**collection_data)
    
    def list_component_checkpoints(
        self, 
        component_type: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        experiment_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available component checkpoints with filtering."""
        all_checkpoints = []
        
        for registry_key, checkpoints in self.checkpoint_registry.items():
            key_component, key_difficulty = registry_key.split('_', 1)
            
            # Apply filters
            if component_type and key_component != component_type:
                continue
            if difficulty_level and key_difficulty != difficulty_level:
                continue
            
            for checkpoint_info in checkpoints:
                if experiment_name and checkpoint_info.get("experiment_name") != experiment_name:
                    continue
                
                checkpoint_info = checkpoint_info.copy()
                checkpoint_info["component_type"] = key_component
                checkpoint_info["difficulty_level"] = key_difficulty
                all_checkpoints.append(checkpoint_info)
        
        # Sort by timestamp (newest first)
        all_checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_checkpoints
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get statistics about component checkpoints."""
        stats = {
            "total_checkpoints": 0,
            "by_component": {},
            "by_difficulty": {},
            "collections": len(self.collection_registry)
        }
        
        for registry_key, checkpoints in self.checkpoint_registry.items():
            component, difficulty = registry_key.split('_', 1)
            
            stats["total_checkpoints"] += len(checkpoints)
            
            if component not in stats["by_component"]:
                stats["by_component"][component] = 0
            stats["by_component"][component] += len(checkpoints)
            
            if difficulty not in stats["by_difficulty"]:
                stats["by_difficulty"][difficulty] = 0
            stats["by_difficulty"][difficulty] += len(checkpoints)
        
        return stats
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]  # Shortened for display
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _cleanup_component_checkpoints(self, registry_key: str, keep_best_n: int):
        """Clean up old checkpoints for specific component/difficulty."""
        checkpoints = self.checkpoint_registry.get(registry_key, [])
        
        if len(checkpoints) <= keep_best_n:
            return
        
        # Sort by performance metric
        component_type = registry_key.split('_')[0]
        if component_type == "surrogate":
            metric = "validation_accuracy"
        elif component_type == "acquisition":
            metric = "final_avg_reward"
        else:
            metric = "joint_final_reward"
        
        # Sort by performance (best first)
        checkpoints_with_scores = []
        for checkpoint_info in checkpoints:
            score = checkpoint_info.get("performance_metrics", {}).get(metric, float('-inf'))
            checkpoints_with_scores.append((score, checkpoint_info))
        
        checkpoints_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Keep best N
        to_keep = [info for _, info in checkpoints_with_scores[:keep_best_n]]
        to_remove = [info for _, info in checkpoints_with_scores[keep_best_n:]]
        
        # Remove old checkpoint files
        for checkpoint_info in to_remove:
            try:
                checkpoint_path = Path(checkpoint_info["checkpoint_path"])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                
                metadata_path = Path(checkpoint_info["metadata_path"])
                if metadata_path.exists():
                    metadata_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_info['checkpoint_id']}: {e}")
        
        # Update registry
        self.checkpoint_registry[registry_key] = to_keep
    
    def _load_registries(self):
        """Load registries from disk."""
        registry_path = self.base_dir / "component_registry.json"
        collection_registry_path = self.base_dir / "collection_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    self.checkpoint_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load component registry: {e}")
                self.checkpoint_registry = {}
        
        if collection_registry_path.exists():
            try:
                with open(collection_registry_path, 'r') as f:
                    self.collection_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load collection registry: {e}")
                self.collection_registry = {}
    
    def _save_registries(self):
        """Save registries to disk."""
        registry_path = self.base_dir / "component_registry.json"
        collection_registry_path = self.base_dir / "collection_registry.json"
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.checkpoint_registry, f, indent=2)
            
            with open(collection_registry_path, 'w') as f:
                json.dump(self.collection_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registries: {e}")


def create_checkpoint_manager(
    checkpoint_dir: Union[str, Path],
    config: Optional[CheckpointConfig] = None
) -> CheckpointManager:
    """Create a CheckpointManager with default configuration."""
    return CheckpointManager(checkpoint_dir, config)


def create_modular_checkpoint_manager(base_dir: str = "checkpoints") -> ModularCheckpointManager:
    """Create a ModularCheckpointManager for component training."""
    return ModularCheckpointManager(base_dir)


__all__ = [
    'CheckpointMetadata',
    'CheckpointInfo',
    'CheckpointConfig',
    'CheckpointManager',
    'ComponentCheckpointMetadata',
    'CheckpointCollection',
    'ModularCheckpointManager',
    'create_checkpoint_manager',
    'create_modular_checkpoint_manager'
]