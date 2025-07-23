#!/usr/bin/env python3
"""
Interface Adapters for Legacy Format Conversion

This module provides adapters to convert between legacy data formats
and the new standardized interfaces, ensuring backward compatibility
while moving toward cleaner interfaces.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from .pipeline_interfaces import (
    MetricNames, OptimizationConfig, TrajectoryData, 
    MethodResult, CheckpointInterface, EvaluationResults
)
from .base_components import CheckpointMetadata

logger = logging.getLogger(__name__)


class LegacyMetricAdapter:
    """Adapter for converting legacy metric names to standardized ones."""
    
    # Mapping of legacy names to standard names
    LEGACY_TO_STANDARD = {
        # Performance metrics
        'final_best': MetricNames.TARGET_VALUE,
        'best_outcome': MetricNames.TARGET_VALUE,
        'reduction': MetricNames.TARGET_IMPROVEMENT,
        'improvement': MetricNames.TARGET_IMPROVEMENT,
        'target_reduction': MetricNames.TARGET_IMPROVEMENT,
        
        # Structure metrics
        'f1': MetricNames.F1_SCORE,
        'f1_score_final': MetricNames.F1_SCORE + MetricNames.FINAL,
        'parent_probability': MetricNames.TRUE_PARENT_LIKELIHOOD,
        
        # Trajectory names
        'target_progress': 'target_values',
        'f1_progress': 'f1_scores',
        'shd_progress': 'shd_values',
        'uncertainty_progress': 'uncertainty_bits',
        'parent_prob_progress': 'true_parent_likelihood'
    }
    
    @classmethod
    def convert_metrics(cls, legacy_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert legacy metric names to standard ones."""
        converted = {}
        
        for key, value in legacy_metrics.items():
            # Check if this is a legacy name
            if key in cls.LEGACY_TO_STANDARD:
                new_key = cls.LEGACY_TO_STANDARD[key]
                converted[new_key] = value
            else:
                # Keep as is
                converted[key] = value
        
        return converted
    
    @classmethod
    def extract_trajectory_data(cls, result: Dict[str, Any]) -> Optional[TrajectoryData]:
        """Extract trajectory data from various legacy formats."""
        # Check multiple possible locations
        sources = [
            result.get('detailed_results', {}),
            result,
            result.get('trajectory_data', {}),
            result.get('learning_curves', {})
        ]
        
        trajectory_data = {}
        
        for source in sources:
            if not source:
                continue
            
            # Look for trajectory arrays
            for legacy_key, standard_key in cls.LEGACY_TO_STANDARD.items():
                if legacy_key in source and isinstance(source[legacy_key], list):
                    trajectory_data[standard_key] = source[legacy_key]
            
            # Also check for standard names
            standard_names = ['target_values', 'f1_scores', 'shd_values', 
                            'true_parent_likelihood', 'uncertainty_bits']
            for name in standard_names:
                if name in source and isinstance(source[name], list):
                    trajectory_data[name] = source[name]
        
        # Create trajectory if we found any data
        if trajectory_data:
            # Ensure we have steps
            if 'steps' not in trajectory_data:
                # Infer from longest array
                max_len = max(len(v) for v in trajectory_data.values() if isinstance(v, list))
                trajectory_data['steps'] = list(range(max_len))
            
            try:
                return TrajectoryData(**trajectory_data)
            except Exception as e:
                logger.warning(f"Failed to create TrajectoryData: {e}")
                return None
        
        return None


class CheckpointAdapter:
    """Adapter for converting between legacy and new checkpoint formats."""
    
    @classmethod
    def from_legacy_metadata(cls, legacy_metadata: CheckpointMetadata) -> CheckpointInterface:
        """Convert legacy CheckpointMetadata to new CheckpointInterface."""
        # Extract optimization config
        if hasattr(legacy_metadata, 'optimization_config') and legacy_metadata.optimization_config:
            opt_config = legacy_metadata.optimization_config
        else:
            # Default to MAXIMIZE for legacy checkpoints
            logger.warning(f"No optimization config in legacy checkpoint, defaulting to MAXIMIZE")
            opt_config = OptimizationConfig(direction="MAXIMIZE")
        
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
    
    @classmethod
    def to_legacy_metadata(cls, checkpoint: CheckpointInterface) -> CheckpointMetadata:
        """Convert new CheckpointInterface to legacy CheckpointMetadata."""
        return CheckpointMetadata(
            name=checkpoint.name,
            path=checkpoint.path,
            optimization_config=checkpoint.optimization_config,
            training_config={
                'mode': checkpoint.training_mode,
                'reward_weights': checkpoint.reward_weights
            },
            training_results={
                'episodes_completed': checkpoint.training_episodes,
                'duration_minutes': checkpoint.training_duration_minutes,
                'final_performance': checkpoint.final_performance,
                'success': checkpoint.success
            },
            timestamp=checkpoint.timestamp
        )
    
    @classmethod
    def find_checkpoint_files(cls, checkpoint_dir: Path) -> List[Path]:
        """Find all valid checkpoint directories."""
        valid_checkpoints = []
        
        for item in checkpoint_dir.iterdir():
            if item.is_dir():
                # Check for required files
                has_metadata = (item / "metadata.json").exists()
                has_pkl = any(item.glob("*.pkl"))
                
                if has_metadata or has_pkl:
                    valid_checkpoints.append(item)
        
        return sorted(valid_checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)


class ResultsAdapter:
    """Adapter for converting evaluation results between formats."""
    
    @classmethod
    def convert_acbo_results(cls, acbo_results: Dict[str, Any], 
                           checkpoint_name: str,
                           optimization_config: OptimizationConfig) -> EvaluationResults:
        """Convert ACBO comparison results to standardized format."""
        # Extract metadata
        execution_metadata = acbo_results.get('execution_metadata', {})
        
        # Convert method results
        method_results = {}
        raw_results = acbo_results.get('experiment_results', {})
        
        for method_name, method_data in raw_results.items():
            if 'runs' not in method_data:
                continue
            
            converted_runs = []
            for run_idx, run in enumerate(method_data['runs']):
                for scm_idx, scm_result in enumerate(run.get('scm_results', [])):
                    # Convert using legacy adapter
                    method_result = MethodResult.from_legacy_result(
                        scm_result,
                        method_name=method_name,
                        run_idx=run_idx,
                        scm_idx=scm_idx
                    )
                    # Add optimization config
                    method_result.optimization_config = optimization_config
                    converted_runs.append(method_result)
            
            if converted_runs:
                method_results[method_name] = converted_runs
        
        # Create evaluation results
        return EvaluationResults(
            evaluation_timestamp=execution_metadata.get('timestamp', ''),
            checkpoint_name=checkpoint_name,
            optimization_config=optimization_config,
            num_scms=execution_metadata.get('scms_tested', 0),
            runs_per_method=execution_metadata.get('runs_per_method', 0),
            intervention_budget=execution_metadata.get('intervention_budget', 0),
            method_results=method_results,
            summary_statistics=acbo_results.get('statistical_analysis', {}).get('summary_statistics', {}),
            pairwise_comparisons=acbo_results.get('statistical_analysis', {}).get('pairwise_comparisons', {}),
            total_duration_minutes=execution_metadata.get('total_time', 0) / 60
        )
    
    @classmethod
    def extract_plot_data(cls, evaluation_results: EvaluationResults) -> Dict[str, Any]:
        """Extract plot-ready data from evaluation results."""
        plot_data = {}
        
        for method_name, runs in evaluation_results.method_results.items():
            # Group by run and aggregate trajectories
            trajectories_by_run = {}
            
            for result in runs:
                if result.trajectory:
                    run_key = f"run_{result.run_idx}"
                    if run_key not in trajectories_by_run:
                        trajectories_by_run[run_key] = []
                    trajectories_by_run[run_key].append(result.trajectory)
            
            if trajectories_by_run:
                # Aggregate across runs
                aggregated = cls._aggregate_trajectories(list(trajectories_by_run.values()))
                plot_data[method_name] = aggregated
        
        return plot_data
    
    @classmethod
    def _aggregate_trajectories(cls, trajectory_groups: List[List[TrajectoryData]]) -> Dict[str, Any]:
        """Aggregate trajectory data across multiple runs."""
        import numpy as np
        
        # Flatten all trajectories
        all_trajectories = []
        for group in trajectory_groups:
            all_trajectories.extend(group)
        
        if not all_trajectories:
            return {}
        
        # Find max length
        max_length = max(len(t.steps) for t in all_trajectories)
        
        # Initialize aggregated arrays
        aggregated = {
            'steps': list(range(max_length)),
            'target_mean': [],
            'target_std': [],
            'f1_mean': [],
            'f1_std': [],
            'shd_mean': [],
            'shd_std': []
        }
        
        # Compute statistics at each step
        for i in range(max_length):
            # Target values
            target_vals = [t.target_values[i] for t in all_trajectories 
                         if i < len(t.target_values)]
            if target_vals:
                aggregated['target_mean'].append(float(np.mean(target_vals)))
                aggregated['target_std'].append(float(np.std(target_vals)))
            
            # F1 scores
            f1_vals = [t.f1_scores[i] for t in all_trajectories 
                     if i < len(t.f1_scores)]
            if f1_vals:
                aggregated['f1_mean'].append(float(np.mean(f1_vals)))
                aggregated['f1_std'].append(float(np.std(f1_vals)))
            
            # SHD values
            shd_vals = [t.shd_values[i] for t in all_trajectories 
                      if i < len(t.shd_values)]
            if shd_vals:
                aggregated['shd_mean'].append(float(np.mean(shd_vals)))
                aggregated['shd_std'].append(float(np.std(shd_vals)))
        
        return aggregated


def ensure_standard_output_location(results: Dict[str, Any], output_dir: Path) -> Path:
    """Ensure results are saved in the standard location with standard name."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard filename
    results_file = output_dir / "comparison_results.json"
    
    # Save with proper formatting
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved results to standard location: {results_file}")
    return results_file