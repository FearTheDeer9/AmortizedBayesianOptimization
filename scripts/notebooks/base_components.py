#!/usr/bin/env python3
"""
Base Components for Modular GRPO Notebooks

Provides reusable utilities for training and evaluation notebooks that:
- Eliminate silent failures through explicit error handling
- Enable independent cell execution through checkpoint management
- Support both minimization and maximization objectives
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pyrsistent as pyr
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as onp

# Import new interfaces
from .pipeline_interfaces import (
    OptimizationConfig as NewOptimizationConfig,
    CheckpointInterface, TrajectoryData, MetricNames
)

logger = logging.getLogger(__name__)


class NotebookError(Exception):
    """Base exception for notebook errors - ensures no silent failures."""
    pass


class CheckpointSaveError(NotebookError):
    """Raised when checkpoint saving fails."""
    pass


class CheckpointNotFoundError(NotebookError):
    """Raised when a required checkpoint cannot be found."""
    pass


class ConfigurationError(NotebookError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(NotebookError):
    """Raised when data loading or generation fails."""
    pass


@dataclass
class OptimizationConfig:
    """Configuration for optimization direction and related settings."""
    direction: str  # "MINIMIZE" or "MAXIMIZE"
    target_baseline: float = 0.0  # Baseline value for target
    
    def __post_init__(self):
        if self.direction not in ["MINIMIZE", "MAXIMIZE"]:
            raise ConfigurationError(f"Invalid optimization direction: {self.direction}")
    
    @property
    def is_minimizing(self) -> bool:
        return self.direction == "MINIMIZE"
    
    def convert_for_maximization(self, value: float) -> float:
        """Convert value to maximization format if needed."""
        if self.is_minimizing:
            return -value + self.target_baseline
        return value
    
    def convert_from_maximization(self, value: float) -> float:
        """Convert back from maximization format if needed."""
        if self.is_minimizing:
            return -(value - self.target_baseline)
        return value
    
    def format_improvement(self, value: float) -> str:
        """Format value with appropriate direction indicator."""
        if self.is_minimizing:
            return f"{value:.4f} (↓ better)"
        return f"{value:.4f} (↑ better)"


@dataclass
class CheckpointMetadata:
    """Metadata for a training checkpoint."""
    name: str
    path: Path
    optimization_config: OptimizationConfig
    training_config: Dict[str, Any]
    training_results: Dict[str, Any]
    timestamp: str
    grpo_version: str = "2.0"  # Version for compatibility checking
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], path: Path) -> "CheckpointMetadata":
        """Create from dictionary, handling legacy formats."""
        # Extract optimization config
        opt_config = data.get("optimization_config")
        if opt_config is None:
            # Legacy checkpoint - infer from training config
            logger.warning("Legacy checkpoint detected - inferring optimization direction")
            # Default to MAXIMIZE for legacy checkpoints
            opt_config = OptimizationConfig(direction="MAXIMIZE")
        elif isinstance(opt_config, dict):
            opt_config = OptimizationConfig(**opt_config)
        
        # Extract timestamp - check multiple locations
        timestamp = data.get("timestamp", "unknown")
        if timestamp == "unknown" and "training_results" in data:
            # Check if timestamp is nested in training_results
            timestamp = data["training_results"].get("timestamp", "unknown")
        
        return cls(
            name=data.get("name", path.name),
            path=path,
            optimization_config=opt_config,
            training_config=data.get("training_config", {}),
            training_results=data.get("training_results", {}),
            timestamp=timestamp,
            grpo_version=data.get("grpo_version", "1.0")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "name": self.name,
            "path": str(self.path),
            "optimization_config": asdict(self.optimization_config),
            "training_config": self.training_config,
            "training_results": self.training_results,
            "timestamp": self.timestamp,
            "grpo_version": self.grpo_version
        }


class CheckpointManager:
    """Manages checkpoint loading, saving, and discovery."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint_interface(self, checkpoint_path: Union[str, Path]) -> CheckpointInterface:
        """Load checkpoint using new standardized interface."""
        checkpoint_path = Path(checkpoint_path)
        try:
            return CheckpointInterface.load_from_path(checkpoint_path)
        except FileNotFoundError:
            # Try legacy loading and convert
            logger.info(f"Attempting legacy checkpoint loading for {checkpoint_path}")
            legacy_metadata = self._load_legacy_checkpoint(checkpoint_path)
            return self._convert_legacy_to_interface(legacy_metadata)
    
    def save_checkpoint_interface(self, checkpoint: CheckpointInterface, 
                                model_data: Any = None) -> Path:
        """Save checkpoint using new standardized interface."""
        # Create directory
        checkpoint.path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        checkpoint.save_metadata()
        
        # Save model data if provided
        if model_data is not None:
            model_path = checkpoint.path / checkpoint.model_params_file
            try:
                import pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                checkpoint.has_model_params = True
                # Update metadata to reflect model params saved
                checkpoint.save_metadata()
                logger.info(f"Saved model parameters: {model_path}")
            except Exception as e:
                logger.error(f"Failed to save model parameters: {e}")
                raise CheckpointSaveError(f"Failed to save model parameters: {e}")
        
        logger.info(f"Saved checkpoint with interface: {checkpoint.path}")
        return checkpoint.path
    
    def find_checkpoint_interfaces(self) -> List[CheckpointInterface]:
        """Find all checkpoints and return as standardized interfaces."""
        checkpoint_paths = self._find_checkpoint_files()
        
        interfaces = []
        for path in checkpoint_paths:
            try:
                interface = self.load_checkpoint_interface(path)
                interfaces.append(interface)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path}: {e}")
        
        return interfaces
    
    def _find_checkpoint_files(self) -> List[Path]:
        """Find all valid checkpoint directories."""
        valid_checkpoints = []
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                # Check for required files
                has_metadata = (item / "metadata.json").exists()
                has_pkl = any(item.glob("*.pkl"))
                
                if has_metadata or has_pkl:
                    valid_checkpoints.append(item)
        
        return sorted(valid_checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
    
    def _convert_legacy_to_interface(self, legacy_metadata: CheckpointMetadata) -> CheckpointInterface:
        """Convert legacy CheckpointMetadata to new CheckpointInterface."""
        # Extract optimization config
        if hasattr(legacy_metadata, 'optimization_config') and legacy_metadata.optimization_config:
            opt_config = legacy_metadata.optimization_config
        else:
            # Default to MAXIMIZE for legacy checkpoints
            logger.warning(f"No optimization config in legacy checkpoint, defaulting to MAXIMIZE")
            opt_config = NewOptimizationConfig(direction="MAXIMIZE")
        
        # Extract training info
        training_config = legacy_metadata.training_config or {}
        training_results = legacy_metadata.training_results or {}
        
        # Create interface
        return CheckpointInterface(
            name=legacy_metadata.name,
            path=Path(legacy_metadata.path),
            optimization_config=opt_config,
            timestamp=legacy_metadata.timestamp,
            training_mode=training_config.get('mode', 'UNKNOWN'),
            training_episodes=training_results.get('episodes_completed', 0),
            reward_weights=training_config.get('reward_weights', {}),
            final_performance=training_results.get('final_performance', {}),
            training_duration_minutes=training_results.get('duration_minutes', 0.0),
            success=training_results.get('success', True),
            has_model_params=True  # Assume true for legacy
        )
    
    def _load_legacy_checkpoint(self, checkpoint_path: Path) -> CheckpointMetadata:
        """Load legacy checkpoint metadata."""
        # Find metadata
        metadata_paths = [
            checkpoint_path / "metadata.json",
            checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json"
        ]
        
        metadata = None
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    metadata = CheckpointMetadata.from_dict(data, checkpoint_path)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load legacy metadata from {metadata_path}: {e}")
        
        if metadata is None:
            raise CheckpointNotFoundError(f"No metadata found for checkpoint: {checkpoint_path}")
        
        return metadata
    
    def discover_checkpoints(self) -> List[CheckpointMetadata]:
        """Discover all available checkpoints with metadata."""
        checkpoints = []
        
        # Look for checkpoint directories
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
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
                            checkpoints.append(checkpoint)
                            break
                        except Exception as e:
                            logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        return checkpoints
    
    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints - alias for discover_checkpoints."""
        return self.discover_checkpoints()
    
    def is_complete(self, checkpoint: CheckpointMetadata) -> bool:
        """Check if checkpoint has both metadata and model parameters."""
        metadata_exists = (checkpoint.path / "metadata.json").exists()
        model_exists = (checkpoint.path / "checkpoint.pkl").exists()
        return metadata_exists and model_exists
    
    def is_loadable(self, checkpoint: CheckpointMetadata) -> bool:
        """Verify checkpoint can actually be loaded."""
        if not self.is_complete(checkpoint):
            return False
        
        try:
            # Try to load the checkpoint file
            checkpoint_file = checkpoint.path / "checkpoint.pkl"
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            # Basic validation - should have policy_params
            return 'policy_params' in data if isinstance(data, dict) else True
        except Exception as e:
            logger.warning(f"Checkpoint {checkpoint.name} not loadable: {e}")
            return False
    
    def find_usable_checkpoints(self, optimization_direction: str = None) -> List[CheckpointMetadata]:
        """Find all complete, loadable checkpoints matching criteria."""
        all_checkpoints = self.discover_checkpoints()
        usable = []
        
        for checkpoint in all_checkpoints:
            # Check if complete and loadable
            if not self.is_complete(checkpoint):
                logger.debug(f"Skipping incomplete checkpoint: {checkpoint.name}")
                continue
            
            if not self.is_loadable(checkpoint):
                logger.debug(f"Skipping unloadable checkpoint: {checkpoint.name}")
                continue
            
            # Check optimization direction if specified
            if optimization_direction:
                if not hasattr(checkpoint, 'optimization_config') or not checkpoint.optimization_config:
                    logger.debug(f"Skipping checkpoint without optimization config: {checkpoint.name}")
                    continue
                
                if checkpoint.optimization_config.direction != optimization_direction:
                    logger.debug(f"Skipping checkpoint with wrong direction {checkpoint.optimization_config.direction}: {checkpoint.name}")
                    continue
            
            usable.append(checkpoint)
        
        return usable
    
    def find_best_checkpoint(self, criteria: Dict[str, Any] = None) -> Optional[CheckpointMetadata]:
        """Find the best checkpoint matching criteria."""
        if criteria is None:
            criteria = {}
        
        # Get base set of usable checkpoints
        candidates = self.find_usable_checkpoints(
            optimization_direction=criteria.get('optimization_direction')
        )
        
        if not candidates:
            return None
        
        # Filter by training mode if specified
        if 'training_mode' in criteria:
            candidates = [
                c for c in candidates 
                if c.training_config.get('mode', '').upper() == criteria['training_mode'].upper()
            ]
        
        # Filter by objective if specified  
        if 'objective' in criteria:
            candidates = [
                c for c in candidates
                if c.training_config.get('objective', '').upper() == criteria['objective'].upper()
            ]
        
        if not candidates:
            return None
        
        # Return most recent (already sorted by timestamp)
        return candidates[0]
    
    def validate_checkpoint(self, checkpoint: CheckpointMetadata) -> Dict[str, Any]:
        """Comprehensive checkpoint validation with detailed report."""
        validation = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'metadata_exists': False,
            'model_exists': False,
            'is_loadable': False,
            'has_optimization_config': False,
            'optimization_direction': None
        }
        
        # Check metadata
        metadata_path = checkpoint.path / "metadata.json"
        validation['metadata_exists'] = metadata_path.exists()
        if not validation['metadata_exists']:
            validation['issues'].append("Missing metadata.json file")
            validation['is_valid'] = False
        
        # Check model file
        model_path = checkpoint.path / "checkpoint.pkl"
        validation['model_exists'] = model_path.exists()
        if not validation['model_exists']:
            validation['issues'].append("Missing checkpoint.pkl file")
            validation['is_valid'] = False
        
        # Check if loadable
        if validation['model_exists']:
            validation['is_loadable'] = self.is_loadable(checkpoint)
            if not validation['is_loadable']:
                validation['issues'].append("Checkpoint file exists but cannot be loaded")
                validation['is_valid'] = False
        
        # Check optimization config
        if hasattr(checkpoint, 'optimization_config') and checkpoint.optimization_config:
            validation['has_optimization_config'] = True
            validation['optimization_direction'] = checkpoint.optimization_config.direction
        else:
            validation['warnings'].append("No optimization configuration found")
        
        return validation
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Tuple[Any, CheckpointMetadata]:
        """Load checkpoint and its metadata."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Find metadata
        metadata_paths = [
            checkpoint_path / "metadata.json",
            checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json"
        ]
        
        metadata = None
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                    metadata = CheckpointMetadata.from_dict(data, checkpoint_path)
                    break
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        if metadata is None:
            raise CheckpointNotFoundError(f"No metadata found for checkpoint: {checkpoint_path}")
        
        # Load actual checkpoint files
        # This would be implemented based on your checkpoint format
        # For now, return None for the checkpoint data
        checkpoint_data = None
        
        return checkpoint_data, metadata
    
    def save_checkpoint(self, 
                       checkpoint_data: Any,
                       metadata: CheckpointMetadata,
                       checkpoint_name: Optional[str] = None) -> Path:
        """Save checkpoint with metadata."""
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            opt_dir = metadata.optimization_config.direction.lower()
            checkpoint_name = f"grpo_{opt_dir}_{timestamp}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Update metadata path
        metadata.path = checkpoint_path
        metadata.name = checkpoint_name
        
        # Save metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Also save in parent directory for compatibility
        parent_metadata_path = self.checkpoint_dir / f"{checkpoint_name}_metadata.json"
        with open(parent_metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Save actual checkpoint data
        if checkpoint_data is not None:
            checkpoint_file = checkpoint_path / "checkpoint.pkl"
            try:
                import pickle
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                logger.info(f"Saved checkpoint data: {checkpoint_file}")
                
                # Also save policy_params.pkl for Phase 2 compatibility
                if 'policy_params' in checkpoint_data and 'policy_config' in checkpoint_data:
                    policy_file = checkpoint_path / "policy_params.pkl"
                    
                    # Ensure architecture config has variable_agnostic flag
                    policy_config = checkpoint_data['policy_config'].copy()
                    if 'architecture' not in policy_config:
                        policy_config['architecture'] = {}
                    # Default to True for enriched policies
                    if 'variable_agnostic' not in policy_config['architecture']:
                        policy_config['architecture']['variable_agnostic'] = True
                    
                    policy_data = {
                        'policy_params': checkpoint_data['policy_params'],
                        'policy_config': policy_config,
                        'enriched_architecture': True,
                        'episode': checkpoint_data.get('training_metrics', {}).get('episodes_completed', 0),
                        'is_final': True
                    }
                    with open(policy_file, 'wb') as f:
                        pickle.dump(policy_data, f)
                    logger.info(f"Saved policy params separately: {policy_file}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint data: {e}")
                raise CheckpointSaveError(f"Failed to save checkpoint data: {e}")
        else:
            logger.warning(f"No checkpoint data provided - only metadata saved")
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        logger.info(f"Optimization: {metadata.optimization_config.direction}")
        
        return checkpoint_path


class SCMGenerator:
    """Generate SCMs for training/testing with explicit error handling."""
    
    def __init__(self, factory=None):
        self.factory = factory
        if factory is None:
            # Import here to avoid circular dependencies
            try:
                from causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
                self.factory = VariableSCMFactory(noise_scale=1.0, coefficient_range=(-2.0, 2.0))
            except ImportError as e:
                raise DataError(f"Failed to import SCM factory: {e}")
    
    def generate_balanced_scms(self,
                              num_scms: int,
                              variable_range: Tuple[int, int] = (3, 6),
                              structure_types: List[str] = None,
                              seed: int = 42) -> Tuple[List[Any], List[Dict]]:
        """Generate balanced set of SCMs with metadata."""
        if structure_types is None:
            structure_types = ['fork', 'chain', 'collider', 'mixed']
        
        scms = []
        metadata = []
        
        # Calculate distribution
        n_structure_types = len(structure_types)
        n_var_sizes = variable_range[1] - variable_range[0] + 1
        scms_per_config = num_scms // (n_structure_types * n_var_sizes)
        remaining = num_scms % (n_structure_types * n_var_sizes)
        
        # Generate SCMs
        key = random.PRNGKey(seed)
        
        try:
            for structure_type in structure_types:
                for n_vars in range(variable_range[0], variable_range[1] + 1):
                    n_instances = scms_per_config + (1 if remaining > 0 else 0)
                    remaining = max(0, remaining - 1)
                    
                    for instance in range(n_instances):
                        if len(scms) >= num_scms:
                            break
                        
                        key, subkey = random.split(key)
                        
                        scm = self.factory.create_variable_scm(
                            num_variables=n_vars,
                            structure_type=structure_type,
                            target_variable=None,
                            edge_density=0.5
                        )
                        
                        scms.append(scm)
                        
                        # Import here to avoid circular dependencies
                        from causal_bayes_opt.data_structures.scm import get_variables, get_target, get_edges
                        
                        metadata.append({
                            'structure_type': structure_type,
                            'n_variables': n_vars,
                            'target': get_target(scm),
                            'n_edges': len(get_edges(scm)),
                            'variables': list(get_variables(scm)),
                            'instance': instance
                        })
        except Exception as e:
            raise DataError(f"Failed to generate SCMs: {e}")
        
        logger.info(f"Generated {len(scms)} SCMs")
        logger.info(f"Distribution: {self._summarize_distribution(metadata)}")
        
        return scms, metadata
    
    def _summarize_distribution(self, metadata: List[Dict]) -> Dict[str, Any]:
        """Summarize SCM distribution."""
        structure_counts = {}
        variable_counts = {}
        
        for meta in metadata:
            struct = meta['structure_type']
            n_vars = meta['n_variables']
            structure_counts[struct] = structure_counts.get(struct, 0) + 1
            variable_counts[n_vars] = variable_counts.get(n_vars, 0) + 1
        
        return {
            'structure_types': structure_counts,
            'variable_counts': variable_counts,
            'total': len(metadata)
        }


def validate_environment() -> Dict[str, Any]:
    """Validate the notebook environment and return system info."""
    info = {
        'jax_devices': str(jax.devices()),
        'jax_backend': jax.default_backend(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check JAX configuration
    try:
        jax.config.update("jax_enable_x64", True)
        info['jax_x64_enabled'] = True
    except Exception as e:
        raise ConfigurationError(f"Failed to configure JAX: {e}")
    
    return info


def format_results_summary(results: Dict[str, Any], optimization_config: OptimizationConfig) -> str:
    """Format results summary with appropriate optimization direction."""
    summary = []
    summary.append(f"Optimization: {optimization_config.direction}")
    summary.append("")
    
    if 'final_value' in results:
        final_value = results['final_value']
        formatted = optimization_config.format_improvement(final_value)
        summary.append(f"Final target value: {formatted}")
    
    if 'improvement' in results:
        improvement = results['improvement']
        if optimization_config.is_minimizing:
            summary.append(f"Reduction from baseline: {improvement:.4f}")
        else:
            summary.append(f"Improvement from baseline: {improvement:.4f}")
    
    return "\n".join(summary)