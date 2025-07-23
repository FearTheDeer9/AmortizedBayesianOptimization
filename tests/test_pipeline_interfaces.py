#!/usr/bin/env python3
"""
Tests for Pipeline Interface Contracts

Verify that the standardized interfaces work correctly and provide
clean communication between training and evaluation components.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Import the interfaces
from scripts.notebooks.pipeline_interfaces import (
    MetricNames, OptimizationConfig, TrajectoryData, 
    MethodResult, CheckpointInterface, EvaluationResults
)
from scripts.notebooks.interface_adapters import (
    LegacyMetricAdapter, CheckpointAdapter, ResultsAdapter
)


class TestOptimizationConfig:
    """Test optimization configuration handling."""
    
    def test_minimization_config(self):
        config = OptimizationConfig(direction="MINIMIZE")
        assert config.is_minimizing
        assert config.compute_improvement(5.0, 3.0) == 2.0  # Positive improvement
        assert "↓ better" in config.format_value(3.0)
    
    def test_maximization_config(self):
        config = OptimizationConfig(direction="MAXIMIZE")
        assert not config.is_minimizing
        assert config.compute_improvement(3.0, 5.0) == 2.0  # Positive improvement
        assert "↑ better" in config.format_value(5.0)
    
    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="Invalid optimization direction"):
            OptimizationConfig(direction="INVALID")


class TestTrajectoryData:
    """Test trajectory data standardization."""
    
    def test_trajectory_creation(self):
        trajectory = TrajectoryData(
            steps=[0, 1, 2],
            target_values=[1.0, 2.0, 3.0],
            f1_scores=[0.1, 0.5, 0.8],
            shd_values=[2, 1, 0]
        )
        
        assert len(trajectory.steps) == 3
        assert trajectory.target_values[-1] == 3.0
        
        final_metrics = trajectory.get_final_metrics()
        assert final_metrics[MetricNames.TARGET_VALUE + MetricNames.FINAL] == 3.0
        assert final_metrics[MetricNames.F1_SCORE + MetricNames.FINAL] == 0.8
    
    def test_from_legacy_dict(self):
        legacy_data = {
            'target_progress': [1.0, 2.0, 3.0],
            'f1_progress': [0.1, 0.5, 0.8],
            'uncertainty_progress': [1.0, 0.5, 0.1]
        }
        
        trajectory = TrajectoryData.from_dict(legacy_data)
        assert trajectory.target_values == [1.0, 2.0, 3.0]
        assert trajectory.uncertainty_bits == [1.0, 0.5, 0.1]
    
    def test_inconsistent_lengths_validation(self):
        with pytest.raises(ValueError, match="inconsistent lengths"):
            TrajectoryData(
                steps=[0, 1],
                target_values=[1.0, 2.0, 3.0],  # Different length
                f1_scores=[0.1, 0.5]
            )


class TestMethodResult:
    """Test method result standardization."""
    
    def test_method_result_creation(self):
        result = MethodResult(
            method_name="Test Method",
            run_idx=0,
            scm_idx=0,
            final_target_value=5.0,
            intervention_count=10
        )
        
        assert result.method_name == "Test Method"
        assert result.final_target_value == 5.0
        assert result.intervention_count == 10
    
    def test_from_legacy_result(self):
        legacy_result = {
            'final_best': 5.0,
            'learning_history': [{'step': i} for i in range(10)],
            'detailed_results': {
                'target_progress': [1.0, 3.0, 5.0],
                'f1_scores': [0.1, 0.5, 0.8]
            }
        }
        
        result = MethodResult.from_legacy_result(
            legacy_result, "Legacy Method", 0, 0
        )
        
        assert result.method_name == "Legacy Method"
        assert result.final_target_value == 5.0
        assert result.intervention_count == 10
        assert result.trajectory is not None
        assert len(result.trajectory.target_values) == 3


class TestCheckpointInterface:
    """Test checkpoint interface standardization."""
    
    def test_checkpoint_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint"
            
            checkpoint = CheckpointInterface(
                name="test_checkpoint",
                path=checkpoint_path,
                optimization_config=OptimizationConfig(direction="MINIMIZE"),
                timestamp=datetime.now().isoformat(),
                training_mode="QUICK",
                training_episodes=100,
                reward_weights={"optimization": 0.8, "discovery": 0.2},
                final_performance={"target_value": -2.5},
                training_duration_minutes=10.0,
                success=True
            )
            
            # Save metadata
            checkpoint.save_metadata()
            
            # Validate
            is_valid, issues = checkpoint.validate()
            # Should fail because no model file exists yet
            assert not is_valid
            assert "Missing checkpoint.pkl file" in str(issues)
            
            # Create dummy model file
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            (checkpoint_path / "checkpoint.pkl").write_text("dummy")
            checkpoint.has_model_params = True
            
            # Should pass now
            is_valid, issues = checkpoint.validate()
            assert is_valid
    
    def test_load_from_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint"
            checkpoint_path.mkdir(parents=True)
            
            # Create metadata file
            metadata = {
                'name': 'test_checkpoint',
                'path': str(checkpoint_path),
                'optimization_config': {'direction': 'MAXIMIZE', 'target_baseline': 0.0},
                'timestamp': '2023-01-01T00:00:00',
                'training_mode': 'STANDARD',
                'training_episodes': 200,
                'reward_weights': {'optimization': 0.7, 'discovery': 0.3},
                'final_performance': {'target_value': 3.2},
                'training_duration_minutes': 25.0,
                'success': True,
                'has_model_params': False,
                'interface_version': '1.0'
            }
            
            with open(checkpoint_path / "metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            # Load checkpoint
            loaded = CheckpointInterface.load_from_path(checkpoint_path)
            
            assert loaded.name == "test_checkpoint"
            assert loaded.optimization_config.direction == "MAXIMIZE"
            assert loaded.training_mode == "STANDARD"
            assert loaded.training_episodes == 200


class TestLegacyMetricAdapter:
    """Test conversion from legacy metric formats."""
    
    def test_metric_name_conversion(self):
        legacy_metrics = {
            'final_best': 5.0,
            'reduction': 2.0,
            'f1': 0.8,
            'target_progress': [1.0, 3.0, 5.0]
        }
        
        converted = LegacyMetricAdapter.convert_metrics(legacy_metrics)
        
        assert converted[MetricNames.TARGET_VALUE] == 5.0
        assert converted[MetricNames.TARGET_IMPROVEMENT] == 2.0
        assert converted[MetricNames.F1_SCORE] == 0.8
        assert converted['target_values'] == [1.0, 3.0, 5.0]
    
    def test_trajectory_extraction(self):
        result = {
            'detailed_results': {
                'target_progress': [1.0, 3.0, 5.0],
                'f1_scores': [0.1, 0.5, 0.8],
                'shd_values': [2, 1, 0]
            }
        }
        
        trajectory = LegacyMetricAdapter.extract_trajectory_data(result)
        
        assert trajectory is not None
        assert trajectory.target_values == [1.0, 3.0, 5.0]
        assert trajectory.f1_scores == [0.1, 0.5, 0.8]
        assert trajectory.shd_values == [2, 1, 0]


class TestEvaluationResults:
    """Test evaluation results standardization."""
    
    def test_evaluation_results_creation(self):
        opt_config = OptimizationConfig(direction="MINIMIZE")
        
        # Create some method results
        method1_results = [
            MethodResult("Method1", 0, 0, -2.0, 5),
            MethodResult("Method1", 1, 0, -1.5, 6)
        ]
        
        method2_results = [
            MethodResult("Method2", 0, 0, -1.0, 8),
            MethodResult("Method2", 1, 0, -0.5, 9)
        ]
        
        evaluation = EvaluationResults(
            evaluation_timestamp="2023-01-01T00:00:00",
            checkpoint_name="test_checkpoint",
            optimization_config=opt_config,
            num_scms=1,
            runs_per_method=2,
            intervention_budget=10,
            method_results={
                "Method1": method1_results,
                "Method2": method2_results
            }
        )
        
        assert len(evaluation.method_results) == 2
        assert len(evaluation.method_results["Method1"]) == 2
        assert evaluation.optimization_config.direction == "MINIMIZE"
    
    def test_save_to_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            opt_config = OptimizationConfig(direction="MAXIMIZE")
            evaluation = EvaluationResults(
                evaluation_timestamp="2023-01-01T00:00:00",
                checkpoint_name="test_checkpoint",
                optimization_config=opt_config,
                num_scms=1,
                runs_per_method=1,
                intervention_budget=5,
                method_results={},
                total_duration_minutes=10.0
            )
            
            evaluation.save_to_file(output_dir)
            
            results_file = output_dir / "comparison_results.json"
            assert results_file.exists()
            
            # Load and verify
            with open(results_file) as f:
                data = json.load(f)
            
            assert data['checkpoint_name'] == "test_checkpoint"
            assert data['optimization_config']['direction'] == "MAXIMIZE"
            assert data['total_duration_minutes'] == 10.0


if __name__ == "__main__":
    pytest.main([__file__])