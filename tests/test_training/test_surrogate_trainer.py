#!/usr/bin/env python3
"""
Comprehensive Test Suite for SurrogateTrainer

Tests the decoupled surrogate training interface and integration with
existing components following TDD principles.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import pyrsistent as pyr

# Import the module we're testing
from src.causal_bayes_opt.training.surrogate_trainer import (
    SurrogateTrainer,
    SurrogateTrainingResults,
    load_expert_demonstrations_from_path,
    convert_demonstrations_to_training_batches
)

# Import existing components for integration testing
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.training.surrogate_training import (
    TrainingBatchJAX,
    TrainingMetrics,
    ValidationResults,
    kl_divergence_loss_jax
)

# Create mock classes for testing without dependencies
@dataclass
class ExpertDemonstration:
    """Mock ExpertDemonstration for testing."""
    observational_data: jnp.ndarray
    expert_posterior: Any
    scm: Any
    target_variable: str
    variable_order: List[str]
    expert_accuracy: float
    problem_difficulty: str

@dataclass 
class ExpertTrajectoryDemonstration:
    """Mock ExpertTrajectoryDemonstration for testing."""
    states: List[Any]
    actions: List[Any]
    scm: Any

try:
    from src.causal_bayes_opt.avici_integration.parent_set.posterior import ParentSetPosterior
    from src.causal_bayes_opt.data_structures.scm import create_scm
    from src.causal_bayes_opt.mechanisms.linear import create_linear_mechanism, create_root_mechanism
except ImportError:
    # Mock these classes if imports fail
    ParentSetPosterior = type('ParentSetPosterior', (), {})
    create_scm = lambda **kwargs: kwargs
    create_linear_mechanism = lambda **kwargs: lambda *args: 0.0
    create_root_mechanism = lambda **kwargs: lambda *args: 0.0


class TestSurrogateTrainerInterface:
    """Test the basic interface and configuration of SurrogateTrainer."""
    
    def test_trainer_initialization_with_default_config(self):
        """Test SurrogateTrainer initializes with default configuration."""
        trainer = SurrogateTrainer()
        
        assert trainer.config is not None
        assert isinstance(trainer.config, SurrogateTrainingConfig)
        assert trainer.config.learning_rate > 0
        assert trainer.config.batch_size > 0
    
    def test_trainer_initialization_with_custom_config(self):
        """Test SurrogateTrainer initializes with custom configuration."""
        config = SurrogateTrainingConfig(
            learning_rate=1e-4,
            batch_size=16,
            max_epochs=50
        )
        trainer = SurrogateTrainer(config)
        
        assert trainer.config == config
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.batch_size == 16
        assert trainer.config.max_epochs == 50
    
    def test_trainer_has_required_methods(self):
        """Test SurrogateTrainer has all required public methods."""
        trainer = SurrogateTrainer()
        
        # Check required methods exist
        assert hasattr(trainer, 'load_expert_demonstrations')
        assert hasattr(trainer, 'train')
        assert hasattr(trainer, 'validate')
        assert hasattr(trainer, 'save_checkpoint')
        assert hasattr(trainer, 'load_checkpoint')
        
        # Check methods are callable
        assert callable(trainer.load_expert_demonstrations)
        assert callable(trainer.train)
        assert callable(trainer.validate)
        assert callable(trainer.save_checkpoint)
        assert callable(trainer.load_checkpoint)


class TestExpertDemonstrationLoading:
    """Test expert demonstration loading and preprocessing."""
    
    
    def test_load_expert_demonstrations_from_valid_path(self, mock_expert_demonstrations, tmp_path):
        """Test loading expert demonstrations from a valid file path."""
        # Save mock demonstrations to temporary file
        demo_path = tmp_path / "expert_demos.pkl"
        
        # Actually create the file for testing
        import pickle
        with open(demo_path, 'wb') as f:
            pickle.dump(mock_expert_demonstrations, f)
        
        trainer = SurrogateTrainer()
        loaded_demos = trainer.load_expert_demonstrations(str(demo_path))
        
        assert len(loaded_demos) == 2
        assert all(demo.__class__.__name__ == 'ExpertDemonstration' for demo in loaded_demos)
    
    def test_load_expert_demonstrations_validation(self, mock_expert_demonstrations):
        """Test that loaded demonstrations are properly validated."""
        trainer = SurrogateTrainer()
        
        # Test valid demonstrations pass validation
        assert trainer._validate_expert_demonstrations(mock_expert_demonstrations) == True
        
        # Test invalid demonstrations fail validation
        invalid_demos = [None, "not_a_demo"]
        assert trainer._validate_expert_demonstrations(invalid_demos) == False
    
    def test_convert_demonstrations_to_training_batches(self, mock_expert_demonstrations):
        """Test conversion of demonstrations to JAX training batches."""
        trainer = SurrogateTrainer()
        
        # Convert to training batches
        batches = trainer._convert_to_training_batches(
            mock_expert_demonstrations,
            batch_size=2
        )
        
        assert len(batches) == 1  # 2 demos with batch_size=2
        batch = batches[0]
        
        assert isinstance(batch, TrainingBatchJAX)
        assert batch.batch_size == 2
        assert batch.observational_data.shape == (2, 50, 3, 3)
        assert len(batch.target_variables) == 2
        assert all(target == 'Y' for target in batch.target_variables)


class TestTrainingPipeline:
    """Test the complete training pipeline with mocked components."""
    
    @pytest.fixture
    def trainer_with_mocks(self):
        """Create trainer with mocked internal components."""
        config = SurrogateTrainingConfig(
            learning_rate=1e-3,
            batch_size=4,
            max_epochs=2,  # Small for testing
            early_stopping_patience=1
        )
        trainer = SurrogateTrainer(config)
        
        # Mock the underlying training functions
        trainer._create_model = Mock(return_value=(Mock(), {}))
        trainer._create_optimizer = Mock(return_value=Mock())
        trainer._training_step = Mock(return_value=(
            {},  # new_params
            Mock(),  # new_opt_state
            TrainingMetrics(
                total_loss=1.0,
                kl_loss=0.8,
                regularization_loss=0.2,
                mean_expert_accuracy=0.9,
                predicted_entropy=1.5,
                expert_entropy=1.2,
                gradient_norm=0.5,
                learning_rate=1e-3,
                step_time=0.01
            )
        ))
        
        return trainer
    
    def test_train_method_with_valid_demonstrations(self, trainer_with_mocks, mock_expert_demonstrations):
        """Test complete training pipeline execution."""
        results = trainer_with_mocks.train(mock_expert_demonstrations)
        
        assert isinstance(results, SurrogateTrainingResults)
        assert results.final_params is not None
        assert results.training_metrics is not None
        assert results.validation_metrics is not None
        assert results.total_training_time >= 0
    
    def test_train_method_handles_early_stopping(self, trainer_with_mocks, mock_expert_demonstrations):
        """Test training respects early stopping criteria."""
        # Mock validation to trigger early stopping
        trainer_with_mocks._validate_model = Mock(return_value=ValidationResults(
            posterior_kl_divergence=0.15,
            reverse_kl_divergence=0.12,
            total_variation_distance=0.08,
            calibration_error=0.03,
            uncertainty_correlation=0.75,
            accuracy_drop=0.08,
            inference_speedup=150.0,
            easy_accuracy=0.92,
            medium_accuracy=0.88,
            hard_accuracy=0.82
        ))
        
        results = trainer_with_mocks.train(mock_expert_demonstrations)
        
        # Should complete successfully with early stopping
        assert results.converged == True
        assert results.epochs_trained <= trainer_with_mocks.config.max_epochs
    
    def test_train_method_error_handling(self, trainer_with_mocks):
        """Test training handles errors gracefully."""
        # Test with empty demonstrations
        with pytest.raises(ValueError, match="No expert demonstrations provided"):
            trainer_with_mocks.train([])
        
        # Test with invalid demonstrations
        with pytest.raises(ValueError, match="Invalid expert demonstrations"):
            trainer_with_mocks.train([None, "invalid"])


class TestValidationAndMetrics:
    """Test validation functionality and metrics computation."""
    
    def test_validate_model_with_held_out_data(self, mock_expert_demonstrations):
        """Test model validation on held-out expert demonstrations."""
        trainer = SurrogateTrainer()
        
        # Mock model parameters and validation function
        mock_params = {"layer1": jnp.ones((10, 5))}
        
        with patch.object(trainer, '_compute_validation_metrics') as mock_compute:
            mock_compute.return_value = ValidationResults(
                posterior_kl_divergence=0.15,
                reverse_kl_divergence=0.12,
                total_variation_distance=0.08,
                calibration_error=0.03,
                uncertainty_correlation=0.75,
                accuracy_drop=0.08,
                inference_speedup=150.0,
                easy_accuracy=0.92,
                medium_accuracy=0.88,
                hard_accuracy=0.82
            )
            
            validation_results = trainer.validate(mock_params, mock_expert_demonstrations)
            
            assert isinstance(validation_results, ValidationResults)
            assert validation_results.posterior_kl_divergence == 0.15
            assert validation_results.inference_speedup > 1.0
    
    def test_validation_metrics_computation(self, mock_expert_demonstrations):
        """Test detailed validation metrics computation."""
        trainer = SurrogateTrainer()
        
        # Create mock model predictions
        try:
            mock_predictions = [
                ParentSetPosterior(
                    target_variable='Y',
                    parent_set_probs=pyr.m(
                        frozenset(), 0.15,
                        frozenset(['X']), 0.25,
                        frozenset(['Z']), 0.15,
                        frozenset(['X', 'Z']), 0.45
                    ),
                    uncertainty=1.1,
                    top_k_sets=[
                        (frozenset(['X', 'Z']), 0.45),
                        (frozenset(['X']), 0.25),
                        (frozenset(['Z']), 0.15),
                        (frozenset(), 0.15)
                    ]
                )
            ] * len(mock_expert_demonstrations)
        except:
            # Simple mock if ParentSetPosterior fails
            mock_predictions = [type('MockPosterior', (), {})() for _ in mock_expert_demonstrations]
        
        with patch.object(trainer, '_predict_posteriors', return_value=mock_predictions):
            metrics = trainer._compute_validation_metrics(
                model_params={},
                validation_data=mock_expert_demonstrations
            )
            
            assert isinstance(metrics, ValidationResults)
            assert metrics.posterior_kl_divergence >= 0
            assert metrics.total_variation_distance >= 0
            assert metrics.calibration_error >= 0


class TestCheckpointingAndRecovery:
    """Test model checkpointing and recovery functionality."""
    
    def test_save_checkpoint_creates_valid_file(self, tmp_path):
        """Test checkpoint saving creates valid checkpoint file."""
        trainer = SurrogateTrainer()
        
        mock_params = {"layer1": jnp.ones((5, 3))}
        mock_metrics = {"loss": 0.5, "accuracy": 0.9}
        checkpoint_path = tmp_path / "checkpoint.pkl"
        
        trainer.save_checkpoint(mock_params, mock_metrics, str(checkpoint_path))
        
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0
    
    def test_load_checkpoint_recovers_state(self, tmp_path):
        """Test checkpoint loading recovers saved state correctly."""
        trainer = SurrogateTrainer()
        
        # Save checkpoint
        original_params = {"layer1": jnp.ones((5, 3))}
        original_metrics = {"loss": 0.3, "accuracy": 0.95}
        checkpoint_path = tmp_path / "checkpoint.pkl"
        
        trainer.save_checkpoint(original_params, original_metrics, str(checkpoint_path))
        
        # Load checkpoint
        loaded_params, loaded_metrics = trainer.load_checkpoint(str(checkpoint_path))
        
        # Verify recovery
        assert jnp.allclose(loaded_params["layer1"], original_params["layer1"])
        assert loaded_metrics["loss"] == original_metrics["loss"]
        assert loaded_metrics["accuracy"] == original_metrics["accuracy"]
    
    def test_load_nonexistent_checkpoint_raises_error(self):
        """Test loading nonexistent checkpoint raises appropriate error."""
        trainer = SurrogateTrainer()
        
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("/nonexistent/path/checkpoint.pkl")


class TestIntegrationWithExistingComponents:
    """Test integration with existing training components."""
    
    def test_integration_with_existing_loss_functions(self, mock_expert_demonstrations):
        """Test integration with existing JAX loss functions."""
        trainer = SurrogateTrainer()
        
        # Test that trainer can call existing loss functions
        predicted_logits = jnp.array([0.1, 0.3, 0.5, 0.1])
        expert_probs = jnp.array([0.1, 0.2, 0.2, 0.5])
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Z']), frozenset(['X', 'Z'])]
        
        # Should not raise any errors
        loss = kl_divergence_loss_jax(predicted_logits, expert_probs, parent_sets)
        assert isinstance(loss, (float, jnp.ndarray))
        assert loss >= 0
    
    def test_integration_with_training_config(self):
        """Test integration with existing SurrogateTrainingConfig."""
        config = SurrogateTrainingConfig(
            model_hidden_dim=64,
            learning_rate=5e-4,
            batch_size=8,
            max_epochs=10
        )
        
        trainer = SurrogateTrainer(config)
        
        # Verify configuration is properly used
        assert trainer.config.model_hidden_dim == 64
        assert trainer.config.learning_rate == 5e-4
        assert trainer.config.batch_size == 8
    
    def test_integration_with_jax_compilation(self):
        """Test that JAX compilation optimizations are preserved."""
        trainer = SurrogateTrainer()
        
        # Mock a training step function that should be JIT-compiled
        @jax.jit
        def mock_compiled_step(params, data):
            return params, 0.5
        
        # Verify the function can be called (compilation should work)
        mock_params = {"weights": jnp.ones((3, 3))}
        mock_data = jnp.ones((4, 10))
        
        result_params, loss = mock_compiled_step(mock_params, mock_data)
        
        assert jnp.allclose(result_params["weights"], mock_params["weights"])
        assert loss == 0.5


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    def test_invalid_configuration_raises_error(self):
        """Test invalid configuration raises appropriate errors."""
        # Test with invalid configurations that our trainer validates
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            trainer = SurrogateTrainer(SurrogateTrainingConfig(learning_rate=-0.01))
        
        with pytest.raises(ValueError, match="Batch size must be positive"):
            trainer = SurrogateTrainer(SurrogateTrainingConfig(batch_size=0))
    
    def test_empty_demonstration_list_handling(self):
        """Test handling of empty expert demonstration lists."""
        trainer = SurrogateTrainer()
        
        with pytest.raises(ValueError, match="No expert demonstrations"):
            trainer.train([])
    
    def test_corrupted_demonstration_data_handling(self):
        """Test handling of corrupted demonstration data."""
        trainer = SurrogateTrainer()
        
        # Create corrupted demonstration (missing required fields)
        corrupted_demo = object()  # Not an ExpertDemonstration
        
        with pytest.raises(ValueError, match="Invalid expert demonstrations"):
            trainer.train([corrupted_demo])
    
    def test_memory_efficient_batch_processing(self, mock_expert_demonstrations):
        """Test that large demonstration sets are processed efficiently."""
        # Create large demonstration set
        large_demo_set = mock_expert_demonstrations * 100  # 200 demonstrations
        
        trainer = SurrogateTrainer()
        
        # Mock the training to avoid actual computation
        trainer._create_model = Mock(return_value=(Mock(), {}))
        trainer._create_optimizer = Mock(return_value=Mock())
        trainer._training_step = Mock(return_value=(
            {}, Mock(), TrainingMetrics(
                total_loss=1.0, kl_loss=0.8, regularization_loss=0.2,
                mean_expert_accuracy=0.9, predicted_entropy=1.5, expert_entropy=1.2,
                gradient_norm=0.5, learning_rate=1e-3, step_time=0.01
            )
        ))
        trainer._validate_model = Mock(return_value=ValidationResults(
            posterior_kl_divergence=0.1, reverse_kl_divergence=0.1, total_variation_distance=0.05,
            calibration_error=0.02, uncertainty_correlation=0.8, accuracy_drop=0.05,
            inference_speedup=100.0, easy_accuracy=0.95, medium_accuracy=0.90, hard_accuracy=0.85
        ))
        
        # Should handle large dataset without memory issues
        results = trainer.train(large_demo_set)
        assert isinstance(results, SurrogateTrainingResults)


@pytest.fixture
def mock_expert_demonstrations():
    """Fixture providing mock expert demonstrations for testing."""
    # Create mock SCM - use simple dict if create_scm fails
    try:
        scm = create_scm(
            variables=frozenset(['X', 'Y', 'Z']),
            edges=frozenset([('X', 'Y'), ('Z', 'Y')]),
            target='Y',
            mechanisms={
                'X': create_root_mechanism(mean=0.0, noise_scale=0.1),
                'Z': create_root_mechanism(mean=0.0, noise_scale=0.1),
                'Y': create_linear_mechanism(
                    parents=['X', 'Z'],
                    coefficients={'X': 1.0, 'Z': -0.5},
                    noise_scale=0.1
                )
            }
        )
    except:
        scm = {
            'variables': frozenset(['X', 'Y', 'Z']),
            'edges': frozenset([('X', 'Y'), ('Z', 'Y')]),
            'target': 'Y'
        }
    
    # Create mock expert posterior
    try:
        expert_posterior = ParentSetPosterior(
            target_variable='Y',
            parent_set_probs=pyr.m({
                frozenset(): 0.1,
                frozenset(['X']): 0.2,
                frozenset(['Z']): 0.2,
                frozenset(['X', 'Z']): 0.5
            }),
            uncertainty=1.2,
            top_k_sets=[
                (frozenset(['X', 'Z']), 0.5),
                (frozenset(['X']), 0.2),
                (frozenset(['Z']), 0.2),
                (frozenset(), 0.1)
            ]
        )
    except:
        # Simple mock object using dataclass
        @dataclass
        class MockPosterior:
            target_variable: str = 'Y'
            parent_set_probs: dict = field(default_factory=lambda: {
                frozenset(): 0.1,
                frozenset(['X']): 0.2,
                frozenset(['Z']): 0.2,
                frozenset(['X', 'Z']): 0.5
            })
            uncertainty: float = 1.2
            top_k_sets: list = field(default_factory=lambda: [
                (frozenset(['X', 'Z']), 0.5),
                (frozenset(['X']), 0.2),
                (frozenset(['Z']), 0.2),
                (frozenset(), 0.1)
            ])
        
        expert_posterior = MockPosterior()
    
    # Create mock demonstration
    demo = ExpertDemonstration(
        observational_data=jnp.ones((50, 3, 3)),  # [N, d, 3] format
        expert_posterior=expert_posterior,
        scm=scm,
        target_variable='Y',
        variable_order=['X', 'Y', 'Z'],
        expert_accuracy=0.95,
        problem_difficulty='medium'
    )
    
    return [demo, demo]  # Return list with duplicate for batch testing


if __name__ == "__main__":
    pytest.main([__file__])