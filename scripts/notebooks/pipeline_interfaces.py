#!/usr/bin/env python3
"""
Pipeline Interface Contracts for GRPO Training and Evaluation

This module defines standardized interfaces to ensure clean communication
between training and evaluation components, eliminating the need for
"ugly patches" and complex fallback logic.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


# Standard metric names - single source of truth
class MetricNames:
    """Standardized metric names used throughout the pipeline."""
    # Performance metrics
    TARGET_VALUE = "target_value"
    TARGET_IMPROVEMENT = "target_improvement"  # Positive = improvement (handles both min/max)
    TARGET_REDUCTION = "target_reduction"  # Deprecated - use TARGET_IMPROVEMENT
    
    # Structure learning metrics
    F1_SCORE = "f1_score"
    SHD = "shd"  # Structural Hamming Distance
    TRUE_PARENT_LIKELIHOOD = "true_parent_likelihood"
    STRUCTURE_ACCURACY = "structure_accuracy"
    
    # Efficiency metrics
    SAMPLE_EFFICIENCY = "sample_efficiency"
    INTERVENTION_COUNT = "intervention_count"
    CONVERGENCE_STEPS = "convergence_steps"
    INTERVENTION_EFFICIENCY = "intervention_efficiency"
    
    # Uncertainty metrics
    UNCERTAINTY = "uncertainty"
    UNCERTAINTY_REDUCTION = "uncertainty_reduction"
    
    # Trajectory suffixes
    FINAL = "_final"
    MEAN = "_mean"
    STD = "_std"
    TRAJECTORY = "_trajectory"


@dataclass
class OptimizationConfig:
    """Configuration for optimization direction and related settings."""
    direction: str  # "MINIMIZE" or "MAXIMIZE"
    target_baseline: float = 0.0  # Baseline value for target
    
    def __post_init__(self):
        if self.direction not in ["MINIMIZE", "MAXIMIZE"]:
            raise ValueError(f"Invalid optimization direction: {self.direction}")
    
    @property
    def is_minimizing(self) -> bool:
        return self.direction == "MINIMIZE"
    
    def compute_improvement(self, initial: float, final: float) -> float:
        """
        Compute improvement value (always positive = better).
        
        For minimization: improvement = initial - final
        For maximization: improvement = final - initial
        """
        if self.is_minimizing:
            return initial - final
        else:
            return final - initial
    
    def format_value(self, value: float) -> str:
        """Format value with appropriate direction indicator."""
        if self.is_minimizing:
            return f"{value:.4f} (↓ better)"
        return f"{value:.4f} (↑ better)"


@dataclass
class TrajectoryData:
    """Standardized trajectory data structure for consistent plotting."""
    steps: List[int] = field(default_factory=list)
    target_values: List[float] = field(default_factory=list)
    f1_scores: List[float] = field(default_factory=list)
    shd_values: List[float] = field(default_factory=list)
    true_parent_likelihood: List[float] = field(default_factory=list)
    uncertainty_bits: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate that all arrays have the same length."""
        # Create a mapping of field names to their lengths for better error messages
        field_lengths = {
            'steps': len(self.steps),
            'target_values': len(self.target_values),
            'f1_scores': len(self.f1_scores),
            'shd_values': len(self.shd_values),
            'true_parent_likelihood': len(self.true_parent_likelihood),
            'uncertainty_bits': len(self.uncertainty_bits)
        }
        
        # Filter out empty lists
        non_empty_fields = {name: length for name, length in field_lengths.items() if length > 0}
        
        if non_empty_fields:
            # Get unique lengths
            unique_lengths = set(non_empty_fields.values())
            
            # Allow small differences (e.g., 15 vs 16) as these might be due to initial vs final measurements
            if len(unique_lengths) > 1:
                min_length = min(unique_lengths)
                max_length = max(unique_lengths)
                
                # If the difference is more than 1, it's a real problem
                if max_length - min_length > 1:
                    error_msg = f"Trajectory arrays have inconsistent lengths: {field_lengths}"
                    raise ValueError(error_msg)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary format."""
        return {
            'steps': self.steps,
            'target_values': self.target_values,
            'f1_scores': self.f1_scores,
            'shd_values': self.shd_values,
            'true_parent_likelihood': self.true_parent_likelihood,
            'uncertainty_bits': self.uncertainty_bits
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryData":
        """Create from dictionary, handling missing fields gracefully."""
        return cls(
            steps=data.get('steps', []),
            target_values=data.get('target_values', data.get('target_progress', [])),
            f1_scores=data.get('f1_scores', []),
            shd_values=data.get('shd_values', data.get('shd_progress', [])),
            true_parent_likelihood=data.get('true_parent_likelihood', 
                                          data.get('parent_probability', [])),
            uncertainty_bits=data.get('uncertainty_bits', 
                                    data.get('uncertainty_progress', []))
        )
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Extract final values from trajectories."""
        metrics = {}
        
        if self.target_values:
            metrics[MetricNames.TARGET_VALUE + MetricNames.FINAL] = self.target_values[-1]
        
        if self.f1_scores:
            metrics[MetricNames.F1_SCORE + MetricNames.FINAL] = self.f1_scores[-1]
        
        if self.shd_values:
            metrics[MetricNames.SHD + MetricNames.FINAL] = self.shd_values[-1]
        
        if self.true_parent_likelihood:
            metrics[MetricNames.TRUE_PARENT_LIKELIHOOD + MetricNames.FINAL] = \
                self.true_parent_likelihood[-1]
        
        if self.uncertainty_bits:
            metrics[MetricNames.UNCERTAINTY + MetricNames.FINAL] = self.uncertainty_bits[-1]
        
        return metrics


@dataclass
class MethodResult:
    """Standardized result format for a single method run."""
    method_name: str
    run_idx: int
    scm_idx: int
    
    # Core results
    final_target_value: float
    intervention_count: int
    
    # Optional trajectory data
    trajectory: Optional[TrajectoryData] = None
    
    # Structure learning results (None for non-learning methods)
    final_marginals: Optional[Dict[str, float]] = None
    structure_metrics: Optional[Dict[str, float]] = None
    
    # Additional metadata
    optimization_config: Optional[OptimizationConfig] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    
    def compute_improvement(self, baseline: float) -> float:
        """Compute improvement from baseline (positive = better)."""
        if self.optimization_config:
            return self.optimization_config.compute_improvement(
                baseline, self.final_target_value
            )
        else:
            # Default to maximization if not specified
            return self.final_target_value - baseline
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        result = {
            'method_name': self.method_name,
            'run_idx': self.run_idx,
            'scm_idx': self.scm_idx,
            'final_target_value': self.final_target_value,
            'intervention_count': self.intervention_count,
            'execution_time': self.execution_time
        }
        
        if self.trajectory:
            result['trajectory'] = self.trajectory.to_dict()
            result['final_metrics'] = self.trajectory.get_final_metrics()
        
        if self.final_marginals is not None:
            result['final_marginals'] = self.final_marginals
        
        if self.structure_metrics is not None:
            result['structure_metrics'] = self.structure_metrics
        
        if self.optimization_config:
            result['optimization_config'] = asdict(self.optimization_config)
        
        if self.error:
            result['error'] = self.error
        
        return result
    
    @classmethod
    def from_legacy_result(cls, result: Dict[str, Any], 
                          method_name: str, run_idx: int, scm_idx: int) -> "MethodResult":
        """
        Create from legacy result format with multiple fallbacks.
        
        This handles the various formats currently in use.
        """
        # Extract final target value with fallbacks
        final_target = result.get('final_target_value',
                                result.get('final_best',
                                         result.get('best_outcome', 0.0)))
        
        # Extract intervention count
        intervention_count = result.get('intervention_count',
                                      len(result.get('learning_history', [])))
        
        # Extract trajectory data if available
        trajectory = None
        detailed_results = result.get('detailed_results', {})
        
        # Check multiple possible locations for trajectory data
        if any(key in detailed_results for key in ['target_progress', 'f1_scores', 'shd_values']):
            trajectory = TrajectoryData.from_dict(detailed_results)
        elif any(key in result for key in ['target_progress', 'f1_scores', 'shd_values']):
            trajectory = TrajectoryData.from_dict(result)
        
        # Extract structure metrics
        structure_metrics = None
        if 'structure_metrics' in result:
            structure_metrics = result['structure_metrics']
        elif trajectory and (trajectory.f1_scores or trajectory.shd_values):
            structure_metrics = trajectory.get_final_metrics()
        
        # Extract optimization config if available
        opt_config = None
        if 'optimization_config' in result:
            opt_data = result['optimization_config']
            if isinstance(opt_data, dict):
                opt_config = OptimizationConfig(**opt_data)
        
        return cls(
            method_name=method_name,
            run_idx=run_idx,
            scm_idx=scm_idx,
            final_target_value=final_target,
            intervention_count=intervention_count,
            trajectory=trajectory,
            final_marginals=result.get('final_marginal_probs'),
            structure_metrics=structure_metrics,
            optimization_config=opt_config,
            execution_time=result.get('execution_time', 0.0),
            error=result.get('error')
        )


@dataclass
class CheckpointInterface:
    """Standardized checkpoint format for training/evaluation compatibility."""
    # Required fields
    name: str
    path: Path
    optimization_config: OptimizationConfig
    timestamp: str
    
    # Training configuration
    training_mode: str  # "QUICK", "STANDARD", "FULL"
    training_episodes: int
    reward_weights: Dict[str, float]
    
    # Training results
    final_performance: Dict[str, float]
    training_duration_minutes: float
    success: bool
    
    # Model data
    has_model_params: bool = False
    model_params_file: Optional[str] = "checkpoint.pkl"
    
    # Version for compatibility
    interface_version: str = "1.0"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate checkpoint completeness."""
        issues = []
        
        # Check required files exist
        if not self.path.exists():
            issues.append(f"Checkpoint path does not exist: {self.path}")
        
        metadata_path = self.path / "metadata.json"
        if not metadata_path.exists():
            issues.append("Missing metadata.json file")
        
        if self.has_model_params:
            model_path = self.path / self.model_params_file
            if not model_path.exists():
                issues.append(f"Missing model params file: {self.model_params_file}")
        
        # Validate optimization config
        if self.optimization_config.direction not in ["MINIMIZE", "MAXIMIZE"]:
            issues.append(f"Invalid optimization direction: {self.optimization_config.direction}")
        
        return len(issues) == 0, issues
    
    def save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        metadata_path = self.path / "metadata.json"
        self.path.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'name': self.name,
            'path': str(self.path),
            'optimization_config': asdict(self.optimization_config),
            'timestamp': self.timestamp,
            'training_mode': self.training_mode,
            'training_episodes': self.training_episodes,
            'reward_weights': self.reward_weights,
            'final_performance': self.final_performance,
            'training_duration_minutes': self.training_duration_minutes,
            'success': self.success,
            'has_model_params': self.has_model_params,
            'model_params_file': self.model_params_file,
            'interface_version': self.interface_version
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved checkpoint metadata: {metadata_path}")
    
    @classmethod
    def load_from_path(cls, checkpoint_path: Path) -> "CheckpointInterface":
        """Load checkpoint interface from path."""
        metadata_path = checkpoint_path / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Handle optimization config
        opt_data = data.get('optimization_config', {})
        opt_config = OptimizationConfig(
            direction=opt_data.get('direction', 'MAXIMIZE'),
            target_baseline=opt_data.get('target_baseline', 0.0)
        )
        
        # Extract timestamp - check multiple locations
        timestamp = data.get("timestamp", "unknown")
        if timestamp == "unknown" and "training_results" in data:
            # Check if timestamp is nested in training_results
            timestamp = data["training_results"].get("timestamp", "unknown")
        
        return cls(
            name=data['name'],
            path=Path(data['path']),
            optimization_config=opt_config,
            timestamp=timestamp,
            training_mode=data.get('training_mode', 'UNKNOWN'),
            training_episodes=data.get('training_episodes', 0),
            reward_weights=data.get('reward_weights', {}),
            final_performance=data.get('final_performance', {}),
            training_duration_minutes=data.get('training_duration_minutes', 0.0),
            success=data.get('success', True),
            has_model_params=data.get('has_model_params', False),
            model_params_file=data.get('model_params_file', 'checkpoint.pkl'),
            interface_version=data.get('interface_version', '1.0')
        )


@dataclass
class EvaluationResults:
    """Standardized evaluation results format."""
    # Metadata
    evaluation_timestamp: str
    checkpoint_name: str
    optimization_config: OptimizationConfig
    
    # Configuration
    num_scms: int
    runs_per_method: int
    intervention_budget: int
    
    # Method results
    method_results: Dict[str, List[MethodResult]]  # method_name -> list of runs
    
    # Aggregated statistics
    summary_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical comparisons
    pairwise_comparisons: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    total_duration_minutes: float = 0.0
    
    def save_to_file(self, filepath: Path) -> None:
        """Save results to JSON file with standard name."""
        # Convert to serializable format
        data = {
            'evaluation_timestamp': self.evaluation_timestamp,
            'checkpoint_name': self.checkpoint_name,
            'optimization_config': asdict(self.optimization_config),
            'num_scms': self.num_scms,
            'runs_per_method': self.runs_per_method,
            'intervention_budget': self.intervention_budget,
            'method_results': {
                method: [r.to_dict() for r in runs]
                for method, runs in self.method_results.items()
            },
            'summary_statistics': self.summary_statistics,
            'pairwise_comparisons': self.pairwise_comparisons,
            'total_duration_minutes': self.total_duration_minutes
        }
        
        # Always save as comparison_results.json for compatibility
        if filepath.is_dir():
            filepath = filepath / "comparison_results.json"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved evaluation results: {filepath}")