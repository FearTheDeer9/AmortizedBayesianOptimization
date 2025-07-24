#!/usr/bin/env python3
"""
Comprehensive tests for base_trainer.py

Following TDD approach with complete test coverage for the base trainer functionality.
"""

import pytest
from unittest.mock import Mock, patch
import time

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.training.base_trainer import (
    BaseBCTrainer,
    TrainingConfig,
    TrainingState,
    TrainingMetrics,
    ValidationMetrics
)


# Concrete implementation for testing
class MockBCTrainer(BaseBCTrainer):
    """Mock implementation of BaseBCTrainer for testing."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.init_calls = 0
        self.forward_calls = 0
        self.loss_calls = 0
        self.accuracy_calls = 0
    
    def _initialize_model_params(self, rng_key: jax.Array, sample_input):
        self.init_calls += 1
        return pyr.pmap({'weights': jnp.ones((2, 2)), 'bias': jnp.zeros(2)})
    
    def _forward_pass(self, params, inputs, rng_key):
        self.forward_calls += 1
        return {'predictions': jnp.array([0.8, 0.2])}
    
    def _compute_loss(self, params, batch, rng_key):
        self.loss_calls += 1
        # Mock loss that decreases over time for testing
        base_loss = 1.0 - (self.loss_calls * 0.1)
        loss = max(0.1, base_loss)  # Ensure positive loss
        metrics = {'accuracy': 0.8 + (self.loss_calls * 0.02)}
        return loss, metrics
    
    def _compute_accuracy(self, predictions, targets):
        self.accuracy_calls += 1
        return 0.85
    
    def _get_loss_function(self):
        """Return mock loss function for testing."""
        loss_calls = self.loss_calls
        
        def mock_loss_fn(model_apply_fn, params, batch, rng_key):
            nonlocal loss_calls
            loss_calls += 1
            # Mock loss that decreases over time for testing
            base_loss = 1.0 - (loss_calls * 0.1)
            loss = max(0.1, base_loss)  # Ensure positive loss
            metrics = {'accuracy': 0.8 + (loss_calls * 0.02)}
            return loss, metrics
        
        return mock_loss_fn
    
    def _get_model_apply_function(self):
        """Return mock model apply function for testing."""
        def mock_apply_fn(params, rng_key, *args, **kwargs):
            return {'predictions': jnp.array([0.8, 0.2])}
        return mock_apply_fn


class TestTrainingConfig:
    """Test TrainingConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.max_epochs == 100
        assert config.min_epochs == 5
        assert config.early_stopping_patience == 10
        assert config.validation_frequency == 5
        assert config.gradient_clip_norm == 1.0
        assert config.weight_decay == 1e-4
        assert config.use_jax_compilation == True
        assert config.random_seed == 42
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=16,
            max_epochs=50,
            early_stopping_patience=5
        )
        
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.max_epochs == 50
        assert config.early_stopping_patience == 5


class TestTrainingMetrics:
    """Test TrainingMetrics functionality."""
    
    def test_training_metrics_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.8,
            learning_rate=1e-3,
            epoch=10,
            step=100,
            elapsed_time=30.0
        )
        
        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.8
        assert metrics.learning_rate == 1e-3
        assert metrics.epoch == 10
        assert metrics.step == 100
        assert metrics.elapsed_time == 30.0
    
    def test_training_metrics_immutable(self):
        """Test that training metrics are immutable."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.8,
            learning_rate=1e-3,
            epoch=10,
            step=100,
            elapsed_time=30.0
        )
        
        with pytest.raises(AttributeError):
            metrics.loss = 0.3  # Should raise error due to frozen=True


class TestValidationMetrics:
    """Test ValidationMetrics functionality."""
    
    def test_validation_metrics_creation(self):
        """Test creating validation metrics."""
        metrics = ValidationMetrics(
            loss=0.3,
            accuracy=0.9,
            improvement=0.1,
            best_loss=0.2,
            epochs_without_improvement=0
        )
        
        assert metrics.loss == 0.3
        assert metrics.accuracy == 0.9
        assert metrics.improvement == 0.1
        assert metrics.best_loss == 0.2
        assert metrics.epochs_without_improvement == 0


class TestTrainingState:
    """Test TrainingState functionality."""
    
    def test_training_state_creation(self):
        """Test creating training state."""
        model_params = {'weights': jnp.ones((2, 2))}
        optimizer_state = {'step': 0}
        
        state = TrainingState(
            model_params=model_params,
            optimizer_state=optimizer_state,
            epoch=5,
            step=50
        )
        
        assert state.epoch == 5
        assert state.step == 50
        assert state.best_loss == float('inf')
        assert state.epochs_without_improvement == 0
        assert len(state.training_metrics) == 0
        assert len(state.validation_metrics) == 0
    
    def test_training_state_immutable_updates(self):
        """Test immutable updates to training state."""
        state = TrainingState(
            model_params={},
            optimizer_state={},
            epoch=1,
            step=10
        )
        
        # Update epoch
        new_state = state.replace(epoch=2)
        
        assert state.epoch == 1  # Original unchanged
        assert new_state.epoch == 2  # New state updated
        assert state is not new_state  # Different objects
    
    def test_training_state_metrics_append(self):
        """Test appending metrics to training state."""
        state = TrainingState(
            model_params={},
            optimizer_state={},
            epoch=1,
            step=10
        )
        
        metrics = TrainingMetrics(
            loss=0.5, accuracy=0.8, learning_rate=1e-3,
            epoch=1, step=10, elapsed_time=30.0
        )
        
        new_state = state.append_training_metric(metrics)
        
        assert len(state.training_metrics) == 0
        assert len(new_state.training_metrics) == 1
        assert new_state.training_metrics[0] == metrics


class TestBaseBCTrainer:
    """Test BaseBCTrainer functionality."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = TrainingConfig(learning_rate=1e-4)
        trainer = MockBCTrainer(config)
        
        assert trainer.config == config
        assert trainer._rng_key is not None
        assert trainer._optimizer is not None
        assert trainer._training_start_time is None
    
    def test_initialize_training_state(self):
        """Test training state initialization."""
        config = TrainingConfig()
        trainer = MockBCTrainer(config)
        
        sample_input = jnp.ones((5, 3))
        state = trainer.initialize_training_state(sample_input)
        
        assert isinstance(state, TrainingState)
        assert trainer.init_calls == 1
        assert state.epoch == 0
        assert state.step == 0
        assert 'model_type' in state.metadata
        assert state.metadata['model_type'] == 'MockBCTrainer'
    
    def test_predict(self):
        """Test prediction functionality."""
        config = TrainingConfig()
        trainer = MockBCTrainer(config)
        
        sample_input = jnp.ones((5, 3))
        state = trainer.initialize_training_state(sample_input)
        
        prediction = trainer.predict(state, sample_input)
        
        assert trainer.forward_calls == 1
        assert 'predictions' in prediction
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        config = TrainingConfig(use_jax_compilation=False)  # Disable JIT for testing
        trainer = MockBCTrainer(config)
        
        sample_input = jnp.ones((5, 3))
        state = trainer.initialize_training_state(sample_input)
        
        # Create mock training batches
        train_batches = [
            jnp.ones((2, 3)),
            jnp.ones((2, 3)),
            jnp.ones((2, 3))
        ]
        
        new_state = trainer.train_epoch(state, train_batches)
        
        assert new_state.epoch == 1
        assert new_state.step == 3  # 3 batches processed
        assert len(new_state.training_metrics) == 3
        assert trainer.loss_calls == 3
        assert trainer._training_start_time is not None
    
    def test_train_epoch_with_validation(self):
        """Test training epoch with validation."""
        config = TrainingConfig(
            validation_frequency=1,  # Validate every epoch
            use_jax_compilation=False
        )
        trainer = MockBCTrainer(config)
        
        sample_input = jnp.ones((5, 3))
        state = trainer.initialize_training_state(sample_input)
        
        train_batches = [jnp.ones((2, 3))]
        val_batches = [jnp.ones((2, 3))]
        
        new_state = trainer.train_epoch(state, train_batches, val_batches)
        
        assert new_state.epoch == 1
        assert len(new_state.validation_metrics) == 1
        assert new_state.best_loss < float('inf')
    
    def test_early_stopping_criteria(self):
        """Test early stopping logic."""
        config = TrainingConfig(
            min_epochs=2,
            early_stopping_patience=3
        )
        trainer = MockBCTrainer(config)
        
        # Create state that meets early stopping criteria
        state = TrainingState(
            model_params={},
            optimizer_state={},
            epoch=5,  # Above min_epochs
            step=50,
            epochs_without_improvement=4  # Above patience
        )
        
        assert trainer.should_stop_early(state) == True
        
        # Create state that doesn't meet criteria
        state_no_stop = state.replace(epochs_without_improvement=2)
        assert trainer.should_stop_early(state_no_stop) == False
        
        # Create state below min_epochs
        state_min_epochs = state.replace(epoch=1)
        assert trainer.should_stop_early(state_min_epochs) == False
    
    def test_fit_basic(self):
        """Test basic fit functionality."""
        config = TrainingConfig(
            max_epochs=3,
            use_jax_compilation=False
        )
        trainer = MockBCTrainer(config)
        
        train_data = [
            jnp.ones((2, 3)),
            jnp.ones((2, 3))
        ]
        
        final_state = trainer.fit(train_data)
        
        assert isinstance(final_state, TrainingState)
        assert final_state.epoch <= 3
        assert len(final_state.training_metrics) > 0
    
    def test_fit_with_validation(self):
        """Test fit with validation data."""
        config = TrainingConfig(
            max_epochs=2,
            validation_frequency=1,
            use_jax_compilation=False
        )
        trainer = MockBCTrainer(config)
        
        train_data = [jnp.ones((2, 3))]
        val_data = [jnp.ones((2, 3))]
        
        final_state = trainer.fit(train_data, val_data)
        
        assert len(final_state.validation_metrics) > 0
        assert final_state.best_loss < float('inf')
    
    def test_fit_early_stopping(self):
        """Test fit with early stopping."""
        config = TrainingConfig(
            max_epochs=10,
            min_epochs=1,
            early_stopping_patience=2,
            validation_frequency=1,
            use_jax_compilation=False
        )
        
        # Create trainer that produces no improvement
        class NoImprovementTrainer(MockBCTrainer):
            def _compute_loss(self, params, batch, rng_key):
                # Always return same loss (no improvement)
                return 1.0, {'accuracy': 0.8}
        
        trainer = NoImprovementTrainer(config)
        
        train_data = [jnp.ones((2, 3))]
        val_data = [jnp.ones((2, 3))]
        
        final_state = trainer.fit(train_data, val_data)
        
        # Should stop early due to no improvement
        assert final_state.epoch < 10
        assert final_state.epochs_without_improvement >= 2
    
    def test_fit_empty_training_data(self):
        """Test fit with empty training data raises error."""
        config = TrainingConfig()
        trainer = MockBCTrainer(config)
        
        with pytest.raises(ValueError, match="Training data cannot be empty"):
            trainer.fit([])
    
    def test_get_training_summary(self):
        """Test training summary generation."""
        config = TrainingConfig(
            max_epochs=2,
            use_jax_compilation=False
        )
        trainer = MockBCTrainer(config)
        
        train_data = [jnp.ones((2, 3))]
        final_state = trainer.fit(train_data)
        
        summary = trainer.get_training_summary(final_state)
        
        assert 'total_epochs' in summary
        assert 'total_steps' in summary
        assert 'best_loss' in summary
        assert 'final_training_loss' in summary
        assert 'config' in summary
        assert summary['total_epochs'] == final_state.epoch
        assert summary['total_steps'] == final_state.step
    
    def test_loss_decreases_over_training(self):
        """Test that loss actually decreases during training."""
        config = TrainingConfig(
            max_epochs=5,
            use_jax_compilation=False
        )
        trainer = MockBCTrainer(config)
        
        train_data = [jnp.ones((2, 3)) for _ in range(3)]
        final_state = trainer.fit(train_data)
        
        # Check that loss generally decreases
        training_metrics = list(final_state.training_metrics)
        
        # First few losses should be higher than last few
        early_losses = [m.loss for m in training_metrics[:3]]
        late_losses = [m.loss for m in training_metrics[-3:]]
        
        avg_early_loss = sum(early_losses) / len(early_losses)
        avg_late_loss = sum(late_losses) / len(late_losses)
        
        assert avg_late_loss < avg_early_loss, "Loss should decrease over training"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])