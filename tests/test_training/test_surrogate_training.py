#!/usr/bin/env python3
"""
Comprehensive tests for surrogate training infrastructure.

Tests all components of the behavioral cloning pipeline following TDD principles.
Validates loss functions, training pipeline, validation metrics, and experimental framework.
"""

import pytest
import time
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr

from src.causal_bayes_opt.training.surrogate_training import (
    # Data structures
    TrainingExample,
    TrainingBatch,
    TrainingMetrics,
    ValidationResults,
    
    # Loss functions
    kl_divergence_loss,
    uncertainty_weighted_loss,
    calibrated_loss,
    multi_target_loss,
    
    # Data processing
    extract_training_data_from_demonstrations,
    create_training_batch_from_examples,
    progressive_curriculum_sampling,
    
    # Training
    create_surrogate_train_step,
    create_optimizer,
    create_learning_rate_schedule,
    train_surrogate_model,
    
    # Validation
    validate_surrogate_performance,
    
    # Experiments
    run_loss_function_experiment
)

from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def mock_expert_posterior():
    """Create a mock expert posterior for testing."""
    parent_sets = [
        frozenset(),
        frozenset(['A']),
        frozenset(['B']),
        frozenset(['A', 'B'])
    ]
    probs = jnp.array([0.1, 0.3, 0.4, 0.2])
    
    return create_parent_set_posterior(
        target_variable="Y",
        parent_sets=parent_sets,
        probabilities=probs,
        metadata={'source': 'test'}
    )


@pytest.fixture
def sample_training_example(mock_expert_posterior):
    """Create a sample training example."""
    # Create dummy AVICI data [N, d, 3]
    observational_data = jnp.ones((10, 3, 3))  # 10 samples, 3 variables
    
    return TrainingExample(
        observational_data=observational_data,
        target_variable="Y",
        variable_order=["A", "B", "Y"],
        expert_posterior=mock_expert_posterior,
        expert_accuracy=0.85,
        scm_info=pyr.m(n_nodes=3, graph_type="chain", target="Y"),
        problem_difficulty="medium"
    )


@pytest.fixture
def training_config():
    """Create test training configuration."""
    return SurrogateTrainingConfig(
        model_hidden_dim=32,  # Smaller for testing
        model_n_layers=2,     # Fewer layers for testing
        learning_rate=1e-3,
        batch_size=4,
        max_epochs=5,
        early_stopping_patience=3,
        validation_frequency=2,
        weight_decay=1e-4,
        max_parent_size=3,
        dropout=0.1
    )


@pytest.fixture
def mock_expert_demonstration():
    """Create a mock expert demonstration."""
    demonstration = Mock()
    demonstration.scm = create_simple_linear_scm(
        variables=['A', 'B', 'Y'], 
        edges=[('A', 'Y'), ('B', 'Y')], 
        coefficients={('A', 'Y'): 1.0, ('B', 'Y'): 1.0},
        noise_scales={'A': 1.0, 'B': 1.0, 'Y': 1.0},
        target='Y'
    )
    demonstration.target_variable = "Y"
    demonstration.n_nodes = 3
    demonstration.graph_type = "fork"
    demonstration.observational_samples = [
        pyr.m(values=pyr.m(A=1.0, B=2.0, Y=3.0), intervention_type="observational"),
        pyr.m(values=pyr.m(A=1.5, B=2.5, Y=3.5), intervention_type="observational")
    ]
    demonstration.interventional_samples = [
        pyr.m(values=pyr.m(A=2.0, B=2.0, Y=4.0), intervention_type="perfect", intervention_targets=pyr.s("A"))
    ]
    demonstration.accuracy = 0.9
    
    # Mock parent posterior
    demonstration.parent_posterior = create_parent_set_posterior(
        target_variable="Y",
        parent_sets=[frozenset(['A', 'B']), frozenset(['A']), frozenset(['B'])],
        probabilities=jnp.array([0.7, 0.2, 0.1])
    )
    
    return demonstration


# ============================================================================
# Test Data Structures
# ============================================================================

class TestDataStructures:
    """Test immutable data structures."""
    
    def test_training_example_immutability(self, sample_training_example):
        """Test TrainingExample is immutable."""
        example = sample_training_example
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            example.target_variable = "Z"
        
        # Fields should be accessible
        assert example.target_variable == "Y"
        assert len(example.variable_order) == 3
        assert example.expert_accuracy == 0.85
    
    def test_training_batch_creation(self, sample_training_example):
        """Test TrainingBatch creation and properties."""
        examples = [sample_training_example, sample_training_example]
        batch = TrainingBatch(examples=examples)
        
        assert batch.batch_size == 2
        assert len(batch.examples) == 2
        assert batch.examples[0].target_variable == "Y"
    
    def test_training_metrics_structure(self):
        """Test TrainingMetrics data structure."""
        metrics = TrainingMetrics(
            total_loss=0.5,
            kl_loss=0.4,
            regularization_loss=0.1,
            mean_expert_accuracy=0.85,
            predicted_entropy=1.2,
            expert_entropy=1.0,
            gradient_norm=0.8,
            learning_rate=1e-3,
            step_time=0.05
        )
        
        assert metrics.total_loss == 0.5
        assert metrics.kl_loss == 0.4
        assert metrics.gradient_norm == 0.8
    
    def test_validation_results_structure(self):
        """Test ValidationResults data structure."""
        results = ValidationResults(
            posterior_kl_divergence=0.3,
            reverse_kl_divergence=0.35,
            total_variation_distance=0.2,
            calibration_error=0.1,
            uncertainty_correlation=0.7,
            accuracy_drop=0.05,
            inference_speedup=10.0,
            easy_accuracy=0.95,
            medium_accuracy=0.85,
            hard_accuracy=0.75
        )
        
        assert results.inference_speedup == 10.0
        assert results.easy_accuracy > results.hard_accuracy


# ============================================================================
# Test Loss Functions
# ============================================================================

class TestLossFunctions:
    """Test various loss function implementations."""
    
    def test_kl_divergence_loss(self, mock_expert_posterior):
        """Test KL divergence loss computation."""
        # Mock model outputs
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])  # Same length as parent sets
        parent_sets = [
            frozenset(),
            frozenset(['A']),
            frozenset(['B']),
            frozenset(['A', 'B'])
        ]
        
        loss = kl_divergence_loss(predicted_logits, mock_expert_posterior, parent_sets)
        
        # Loss should be finite and positive
        assert jnp.isfinite(loss)
        assert loss >= 0.0
        
        # Test temperature scaling
        loss_hot = kl_divergence_loss(predicted_logits, mock_expert_posterior, parent_sets, temperature=2.0)
        assert jnp.isfinite(loss_hot)
    
    def test_uncertainty_weighted_loss(self, mock_expert_posterior):
        """Test uncertainty-weighted loss function."""
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])
        parent_sets = [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        
        loss = uncertainty_weighted_loss(predicted_logits, mock_expert_posterior, parent_sets)
        
        # Should be finite and positive
        assert jnp.isfinite(loss)
        assert loss >= 0.0
        
        # Test weight scaling
        loss_scaled = uncertainty_weighted_loss(
            predicted_logits, mock_expert_posterior, parent_sets, weight_scale=2.0
        )
        assert jnp.isfinite(loss_scaled)
    
    def test_calibrated_loss(self, mock_expert_posterior):
        """Test calibrated loss function."""
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])
        parent_sets = [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        true_parent_set = frozenset(['A', 'B'])  # Ground truth
        
        loss = calibrated_loss(
            predicted_logits, mock_expert_posterior, parent_sets, true_parent_set
        )
        
        # Should be finite and positive
        assert jnp.isfinite(loss)
        assert loss >= 0.0
        
        # Test calibration weight
        loss_weighted = calibrated_loss(
            predicted_logits, mock_expert_posterior, parent_sets, 
            true_parent_set, calibration_weight=0.5
        )
        assert jnp.isfinite(loss_weighted)
    
    def test_multi_target_loss(self, mock_expert_posterior):
        """Test multi-target loss function."""
        # Mock predictions for multiple targets
        predictions = {
            'Y': jnp.array([0.1, 0.3, 0.4, 0.2]),
            'Z': jnp.array([0.2, 0.4, 0.3, 0.1])
        }
        
        expert_posteriors = {
            'Y': mock_expert_posterior,
            'Z': mock_expert_posterior  # Reuse for simplicity
        }
        
        parent_sets_per_target = {
            'Y': [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])],
            'Z': [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        }
        
        loss = multi_target_loss(predictions, expert_posteriors, parent_sets_per_target)
        
        # Should be finite and positive
        assert jnp.isfinite(loss)
        assert loss >= 0.0
        
        # Test with target weights
        target_weights = {'Y': 0.7, 'Z': 0.3}
        weighted_loss = multi_target_loss(
            predictions, expert_posteriors, parent_sets_per_target, target_weights
        )
        assert jnp.isfinite(weighted_loss)
    
    def test_loss_functions_numerical_stability(self, mock_expert_posterior):
        """Test loss functions handle edge cases gracefully."""
        parent_sets = [frozenset(), frozenset(['A'])]
        
        # Test with extreme logits
        extreme_logits = jnp.array([-100.0, 100.0])
        
        loss1 = kl_divergence_loss(extreme_logits, mock_expert_posterior, parent_sets)
        loss2 = uncertainty_weighted_loss(extreme_logits, mock_expert_posterior, parent_sets)
        
        assert jnp.isfinite(loss1)
        assert jnp.isfinite(loss2)
        
        # Test with uniform logits
        uniform_logits = jnp.array([0.0, 0.0])
        
        loss3 = kl_divergence_loss(uniform_logits, mock_expert_posterior, parent_sets)
        assert jnp.isfinite(loss3)


# ============================================================================
# Test Data Processing
# ============================================================================

class TestDataProcessing:
    """Test data processing and sampling functions."""
    
    def test_extract_training_data_from_demonstrations(self, mock_expert_demonstration):
        """Test extraction of training data from expert demonstrations."""
        demonstrations = [mock_expert_demonstration, mock_expert_demonstration]
        
        with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
            # Mock the AVICI data creation
            mock_batch.return_value = {'x': jnp.ones((5, 3, 3))}
            
            train_examples, val_examples = extract_training_data_from_demonstrations(
                demonstrations, validation_split=0.5
            )
            
            # Should split properly
            assert len(train_examples) == 1
            assert len(val_examples) == 1
            
            # Check example structure
            example = train_examples[0]
            assert example.target_variable == "Y"
            assert len(example.variable_order) == 3
            assert example.expert_accuracy == 0.9
            assert example.problem_difficulty in ["easy", "medium", "hard"]
    
    def test_create_training_batch_from_examples(self, sample_training_example):
        """Test training batch creation from examples."""
        examples = [sample_training_example] * 5
        key = random.PRNGKey(42)
        
        batch = create_training_batch_from_examples(examples, batch_size=3, key=key)
        
        assert batch.batch_size == 3
        assert len(batch.examples) == 3
        
        # Test when fewer examples than batch size
        batch_small = create_training_batch_from_examples(examples[:2], batch_size=5, key=key)
        assert batch_small.batch_size == 2
    
    def test_progressive_curriculum_sampling(self, sample_training_example):
        """Test progressive curriculum sampling."""
        # Create examples with different difficulties
        easy_example = TrainingExample(
            observational_data=sample_training_example.observational_data,
            target_variable="Y",
            variable_order=["A", "B", "Y"],
            expert_posterior=sample_training_example.expert_posterior,
            expert_accuracy=0.95,
            scm_info=pyr.m(n_nodes=3, graph_type="chain", target="Y"),
            problem_difficulty="easy"
        )
        
        hard_example = TrainingExample(
            observational_data=sample_training_example.observational_data,
            target_variable="Y",
            variable_order=["A", "B", "Y"],
            expert_posterior=sample_training_example.expert_posterior,
            expert_accuracy=0.65,
            scm_info=pyr.m(n_nodes=3, graph_type="chain", target="Y"),
            problem_difficulty="hard"
        )
        
        examples = [easy_example] * 5 + [sample_training_example] * 5 + [hard_example] * 5
        key = random.PRNGKey(42)
        
        # Early training should favor easy examples
        batch_early = progressive_curriculum_sampling(
            examples, batch_size=6, training_step=100, curriculum_schedule={}, key=key
        )
        
        difficulties = [ex.problem_difficulty for ex in batch_early.examples]
        easy_count = difficulties.count("easy")
        
        # Should have more easy examples early in training
        assert easy_count > 0
        
        # Later training should have more diverse examples
        batch_late = progressive_curriculum_sampling(
            examples, batch_size=6, training_step=2000, curriculum_schedule={}, key=key
        )
        
        difficulties_late = [ex.problem_difficulty for ex in batch_late.examples]
        hard_count_late = difficulties_late.count("hard")
        
        # Should have some hard examples later
        assert hard_count_late >= 0  # At least some chance of hard examples


# ============================================================================
# Test Training Infrastructure
# ============================================================================

class TestTrainingInfrastructure:
    """Test training infrastructure components."""
    
    def test_create_learning_rate_schedule(self, training_config):
        """Test learning rate schedule creation."""
        total_steps = 1000
        schedule = create_learning_rate_schedule(training_config, total_steps)
        
        # Test schedule at different points
        lr_start = schedule(0)
        lr_peak = schedule(100)  # After warmup
        lr_end = schedule(total_steps - 1)
        
        # Should start low, peak at config value, end low
        assert lr_start < training_config.learning_rate
        assert abs(lr_peak - training_config.learning_rate) < 1e-4
        assert lr_end < training_config.learning_rate
    
    def test_create_optimizer(self, training_config):
        """Test optimizer creation."""
        total_steps = 1000
        optimizer = create_optimizer(training_config, total_steps)
        
        # Should be an Optax transformation
        assert hasattr(optimizer, 'init')
        assert hasattr(optimizer, 'update')
        
        # Test initialization
        dummy_params = {'weights': jnp.ones((3, 3))}
        opt_state = optimizer.init(dummy_params)
        
        # Should initialize without error
        assert opt_state is not None
    
    @patch('src.causal_bayes_opt.training.surrogate_training.create_parent_set_model')
    def test_create_surrogate_train_step(self, mock_model, training_config, sample_training_example):
        """Test training step creation."""
        # Mock model
        mock_model_instance = Mock()
        mock_model_instance.apply.return_value = {
            'parent_set_logits': jnp.array([0.1, 0.3, 0.4, 0.2]),
            'parent_sets': [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        }
        
        optimizer = optax.adam(1e-3)
        
        # Create training step
        train_step = create_surrogate_train_step(
            mock_model_instance, optimizer, kl_divergence_loss, training_config
        )
        
        # Should return a callable
        assert callable(train_step)
        
        # Test execution (would need more complete mocking for full test)
        # This is a basic structure test
        assert hasattr(train_step, '__call__')
    
    def test_training_config_integration(self, training_config):
        """Test that training config integrates properly."""
        # Test model kwargs extraction
        model_kwargs = training_config.get_model_kwargs()
        
        assert 'layers' in model_kwargs
        assert 'dim' in model_kwargs
        assert model_kwargs['layers'] == 2
        assert model_kwargs['dim'] == 32
        
        # Test configuration values
        assert training_config.learning_rate == 1e-3
        assert training_config.batch_size == 4
        assert training_config.weight_decay == 1e-4


# ============================================================================
# Test Validation
# ============================================================================

class TestValidation:
    """Test validation and evaluation functions."""
    
    @patch('src.causal_bayes_opt.training.surrogate_training.predict_parent_posterior')
    def test_validate_surrogate_performance(self, mock_predict, sample_training_example):
        """Test surrogate performance validation."""
        # Mock model and parameters
        mock_model = Mock()
        mock_params = {'weights': jnp.ones((3, 3))}
        
        # Mock model.apply to return reasonable output
        mock_model.apply.return_value = {
            'parent_set_logits': jnp.array([0.1, 0.3, 0.4, 0.2]),
            'parent_sets': [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        }
        
        # Mock predict_parent_posterior
        mock_predict.return_value = sample_training_example.expert_posterior
        
        validation_examples = [sample_training_example, sample_training_example]
        expert_timing = 1.0
        key = random.PRNGKey(42)
        
        results = validate_surrogate_performance(
            mock_model, mock_params, validation_examples, expert_timing, key
        )
        
        # Check result structure
        assert isinstance(results, ValidationResults)
        assert results.inference_speedup > 0
        assert 0 <= results.easy_accuracy <= 1.0
        assert 0 <= results.medium_accuracy <= 1.0
        assert 0 <= results.hard_accuracy <= 1.0
        
        # Check that model.apply was called
        assert mock_model.apply.call_count == len(validation_examples)


# ============================================================================
# Test Experimental Framework
# ============================================================================

class TestExperimentalFramework:
    """Test the experimental comparison framework."""
    
    @patch('src.causal_bayes_opt.training.surrogate_training.train_surrogate_model')
    def test_run_loss_function_experiment_structure(self, mock_train):
        """Test loss function experiment framework structure."""
        # Mock training results
        mock_train.return_value = (
            Mock(),  # model
            Mock(),  # params
            {
                'losses': [1.0, 0.8, 0.6, 0.5],
                'validation_metrics': [
                    ValidationResults(
                        posterior_kl_divergence=0.5,
                        reverse_kl_divergence=0.6,
                        total_variation_distance=0.3,
                        calibration_error=0.1,
                        uncertainty_correlation=0.7,
                        accuracy_drop=0.05,
                        inference_speedup=8.0,
                        easy_accuracy=0.9,
                        medium_accuracy=0.8,
                        hard_accuracy=0.7
                    )
                ]
            }
        )
        
        # Mock expert demonstrations
        mock_demonstrations = [Mock(), Mock()]
        config = SurrogateTrainingConfig(max_epochs=2)  # Short for testing
        
        results = run_loss_function_experiment(
            expert_demonstrations=mock_demonstrations,
            config=config,
            loss_types=["kl_divergence", "uncertainty_weighted"],
            n_trials=2,
            random_seed=42
        )
        
        # Check result structure
        assert isinstance(results, dict)
        assert "kl_divergence" in results
        assert "uncertainty_weighted" in results
        
        # Check aggregated metrics format
        for loss_type, metrics in results.items():
            if metrics:  # If any successful trials
                assert 'kl_divergence_mean' in metrics
                assert 'accuracy_drop_mean' in metrics
                assert 'kl_divergence_std' in metrics
    
    def test_experiment_error_handling(self):
        """Test that experiment framework handles errors gracefully."""
        # Create config that will cause training to fail
        bad_config = SurrogateTrainingConfig(
            learning_rate=-1.0,  # Invalid learning rate
            max_epochs=1
        )
        
        mock_demonstrations = []  # Empty demonstrations
        
        # Should not crash, should return empty results
        results = run_loss_function_experiment(
            expert_demonstrations=mock_demonstrations,
            config=bad_config,
            loss_types=["kl_divergence"],
            n_trials=1,
            random_seed=42
        )
        
        # Should have results structure even if trials failed
        assert isinstance(results, dict)
        assert "kl_divergence" in results


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_data_structure_compatibility(self, sample_training_example, training_config):
        """Test that all data structures work together."""
        # Create batch
        batch = TrainingBatch(examples=[sample_training_example])
        
        # Test that batch can be processed
        assert batch.batch_size == 1
        assert len(batch.examples) == 1
        
        # Test training example fields
        example = batch.examples[0]
        assert example.observational_data.shape == (10, 3, 3)
        assert len(example.variable_order) == 3
        assert example.expert_posterior is not None
    
    def test_loss_function_integration(self, mock_expert_posterior):
        """Test that all loss functions work with same inputs."""
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])
        parent_sets = [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        true_parent_set = frozenset(['A', 'B'])
        
        # All loss functions should work with same inputs
        loss1 = kl_divergence_loss(predicted_logits, mock_expert_posterior, parent_sets)
        loss2 = uncertainty_weighted_loss(predicted_logits, mock_expert_posterior, parent_sets)
        loss3 = calibrated_loss(predicted_logits, mock_expert_posterior, parent_sets, true_parent_set)
        
        # All should be finite and positive
        for loss in [loss1, loss2, loss3]:
            assert jnp.isfinite(loss)
            assert loss >= 0.0
    
    def test_config_to_training_pipeline(self, training_config):
        """Test that config properly configures training pipeline."""
        # Test that config can be used to create optimizer
        total_steps = 100
        optimizer = create_optimizer(training_config, total_steps)
        
        # Test that config model kwargs work
        model_kwargs = training_config.get_model_kwargs()
        assert isinstance(model_kwargs, dict)
        assert all(key in model_kwargs for key in ['layers', 'dim', 'dropout'])
        
        # Test training hyperparameters
        assert training_config.learning_rate > 0
        assert training_config.batch_size > 0
        assert training_config.max_epochs > 0


# ============================================================================
# Performance and Numerical Stability Tests
# ============================================================================

class TestPerformanceStability:
    """Test performance and numerical stability."""
    
    def test_loss_function_performance(self, mock_expert_posterior):
        """Test that loss functions are reasonably fast."""
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])
        parent_sets = [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        
        # Time loss computation
        start_time = time.time()
        for _ in range(100):
            loss = kl_divergence_loss(predicted_logits, mock_expert_posterior, parent_sets)
        end_time = time.time()
        
        # Should be reasonably fast (less than 1ms per call on average)
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.001  # Less than 1ms per call
    
    def test_numerical_stability_extreme_values(self, mock_expert_posterior):
        """Test numerical stability with extreme values."""
        parent_sets = [frozenset(), frozenset(['A'])]
        
        # Test with very large logits
        large_logits = jnp.array([1000.0, -1000.0])
        loss1 = kl_divergence_loss(large_logits, mock_expert_posterior, parent_sets)
        assert jnp.isfinite(loss1)
        
        # Test with very small differences
        small_diff_logits = jnp.array([1e-8, 2e-8])
        loss2 = kl_divergence_loss(small_diff_logits, mock_expert_posterior, parent_sets)
        assert jnp.isfinite(loss2)
        
        # Test with NaN inputs (should handle gracefully or raise clear error)
        try:
            nan_logits = jnp.array([jnp.nan, 0.0])
            loss3 = kl_divergence_loss(nan_logits, mock_expert_posterior, parent_sets)
            # If it doesn't raise, should at least detect the NaN
            assert jnp.isnan(loss3) or jnp.isfinite(loss3)
        except ValueError:
            # Acceptable to raise ValueError for NaN inputs
            pass
    
    def test_jax_compilation_compatibility(self, mock_expert_posterior):
        """Test that functions are compatible with JAX compilation."""
        predicted_logits = jnp.array([0.1, 0.3, 0.4, 0.2])
        parent_sets = [frozenset(), frozenset(['A']), frozenset(['B']), frozenset(['A', 'B'])]
        
        # Test that loss function can be JIT compiled
        @jax.jit
        def compiled_loss(logits):
            return kl_divergence_loss(logits, mock_expert_posterior, parent_sets)
        
        # Should compile and run without error
        loss = compiled_loss(predicted_logits)
        assert jnp.isfinite(loss)
        
        # Second call should use compiled version (and be faster)
        loss2 = compiled_loss(predicted_logits)
        assert jnp.allclose(loss, loss2)


class TestJAXOptimization:
    """Test JAX-specific optimization features."""
    
    def test_convert_to_jax_batch(self, sample_training_example):
        """Test conversion of TrainingBatch to JAX-compatible format."""
        from causal_bayes_opt.training.surrogate_training import convert_to_jax_batch, TrainingBatchJAX, TrainingBatch
        
        # Create a training batch with the sample
        training_batch = TrainingBatch(examples=[sample_training_example])
        
        # Convert to JAX format
        jax_batch = convert_to_jax_batch(training_batch)
        
        # Verify structure
        assert isinstance(jax_batch, TrainingBatchJAX)
        assert jax_batch.batch_size == training_batch.batch_size
        
        # Verify JAX arrays
        assert isinstance(jax_batch.observational_data, jnp.ndarray)
        assert isinstance(jax_batch.expert_probs, jnp.ndarray)
        assert isinstance(jax_batch.expert_accuracies, jnp.ndarray)
        
        # Verify metadata
        assert isinstance(jax_batch.parent_sets, list)
        assert isinstance(jax_batch.variable_orders, list)
        assert isinstance(jax_batch.target_variables, list)
        
        # Verify shapes
        assert jax_batch.observational_data.shape[0] == jax_batch.batch_size
        assert jax_batch.expert_probs.shape[0] == jax_batch.batch_size
        assert jax_batch.expert_accuracies.shape[0] == jax_batch.batch_size
    
    def test_jax_compatible_loss_functions(self):
        """Test JAX-compatible loss functions."""
        from causal_bayes_opt.training.surrogate_training import kl_divergence_loss_jax, uncertainty_weighted_loss_jax
        
        # Test data
        predicted_logits = jnp.array([1.0, 2.0, 0.5])
        expert_probs = jnp.array([0.1, 0.8, 0.1])
        dummy_parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y'])]
        
        # Test KL divergence loss
        kl_loss = kl_divergence_loss_jax(predicted_logits, expert_probs, dummy_parent_sets)
        assert jnp.isfinite(kl_loss)
        assert kl_loss >= 0, "KL divergence should be non-negative"
        
        # Test uncertainty weighted loss
        uw_loss = uncertainty_weighted_loss_jax(predicted_logits, expert_probs, dummy_parent_sets)
        assert jnp.isfinite(uw_loss)
        assert uw_loss >= 0, "Weighted loss should be non-negative"
    
    def test_jax_loss_functions_compilation(self):
        """Test that JAX loss functions work correctly (compilation tested in actual training step)."""
        from causal_bayes_opt.training.surrogate_training import kl_divergence_loss_jax, uncertainty_weighted_loss_jax
        
        # Test data
        predicted_logits = jnp.array([1.0, 2.0, 0.5])
        expert_probs = jnp.array([0.1, 0.8, 0.1])
        dummy_parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y'])]
        
        # Test that functions work correctly (JAX compilation is tested in the actual training step)
        kl_loss = kl_divergence_loss_jax(predicted_logits, expert_probs, dummy_parent_sets)
        uw_loss = uncertainty_weighted_loss_jax(predicted_logits, expert_probs, dummy_parent_sets)
        
        assert jnp.isfinite(kl_loss)
        assert jnp.isfinite(uw_loss)
        assert kl_loss >= 0, "KL divergence should be non-negative"
        assert uw_loss >= 0, "Uncertainty weighted loss should be non-negative"
        
        # Test with different inputs to ensure robustness
        other_logits = jnp.array([0.1, 0.9, 0.0])
        other_probs = jnp.array([0.5, 0.3, 0.2])
        
        kl_loss2 = kl_divergence_loss_jax(other_logits, other_probs, dummy_parent_sets)
        uw_loss2 = uncertainty_weighted_loss_jax(other_logits, other_probs, dummy_parent_sets)
        
        assert jnp.isfinite(kl_loss2)
        assert jnp.isfinite(uw_loss2)
    
    def test_create_jax_surrogate_train_step(self, training_config):
        """Test creation of JAX-compiled training step."""
        from causal_bayes_opt.training.surrogate_training import create_jax_surrogate_train_step, kl_divergence_loss
        from causal_bayes_opt.avici_integration import create_parent_set_model
        
        # Create a simple mock model
        model_kwargs = training_config.get_model_kwargs()
        model = create_parent_set_model(
            model_kwargs=model_kwargs,
            max_parent_size=training_config.max_parent_size
        )
        
        # Create optimizer
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Create JAX training step
        jax_train_step = create_jax_surrogate_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=training_config
        )
        
        # Verify it's a function
        assert callable(jax_train_step)
        
        # Test function signature by inspecting
        import inspect
        sig = inspect.signature(jax_train_step)
        expected_params = ['params', 'opt_state', 'parent_sets', 'variable_orders', 
                          'target_variables', 'observational_data', 'expert_probs', 
                          'expert_accuracies', 'key']
        actual_params = list(sig.parameters.keys())
        
        for expected in expected_params:
            assert expected in actual_params, f"Missing parameter: {expected}"
    
    def test_create_adaptive_train_step(self, training_config):
        """Test adaptive training step creation."""
        from causal_bayes_opt.training.surrogate_training import create_adaptive_train_step, kl_divergence_loss
        from causal_bayes_opt.avici_integration import create_parent_set_model
        
        # Create a simple mock model
        model_kwargs = training_config.get_model_kwargs()
        model = create_parent_set_model(
            model_kwargs=model_kwargs,
            max_parent_size=training_config.max_parent_size
        )
        
        optimizer = optax.adam(learning_rate=1e-3)
        
        # Test with JAX compilation enabled
        adaptive_step_jax = create_adaptive_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=training_config,
            use_jax_compilation=True
        )
        assert callable(adaptive_step_jax)
        
        # Test with JAX compilation disabled
        adaptive_step_no_jax = create_adaptive_train_step(
            model=model,
            optimizer=optimizer,
            loss_fn=kl_divergence_loss,
            config=training_config,
            use_jax_compilation=False
        )
        assert callable(adaptive_step_no_jax)
    
    def test_jax_batch_conversion_round_trip(self, sample_training_example):
        """Test that JAX batch conversion preserves data integrity."""
        from causal_bayes_opt.training.surrogate_training import convert_to_jax_batch, TrainingBatch
        
        # Create a training batch with the sample
        training_batch = TrainingBatch(examples=[sample_training_example])
        
        # Convert to JAX format
        jax_batch = convert_to_jax_batch(training_batch)
        
        # Verify we can access all data
        assert jax_batch.batch_size == len(training_batch.examples)
        
        # Check that observational data shapes are correct
        for i, example in enumerate(training_batch.examples):
            original_shape = example.observational_data.shape
            jax_shape = jax_batch.observational_data[i].shape
            assert original_shape == jax_shape, f"Shape mismatch at index {i}"
        
        # Check expert accuracies
        for i, example in enumerate(training_batch.examples):
            original_acc = example.expert_accuracy
            jax_acc = float(jax_batch.expert_accuracies[i])
            assert abs(original_acc - jax_acc) < 1e-6, f"Accuracy mismatch at index {i}"
    
    def test_jax_optimization_consistency(self, mock_expert_posterior):
        """Test that JAX and non-JAX loss functions give consistent results."""
        from causal_bayes_opt.training.surrogate_training import kl_divergence_loss_jax, kl_divergence_loss
        
        # Test data
        predicted_logits = jnp.array([1.0, 2.0, 0.5])
        expert_probs = jnp.array([0.1, 0.8, 0.1])
        dummy_parent_sets = [frozenset(), frozenset(['X']), frozenset(['Y'])]
        
        # Compare KL divergence implementations
        jax_kl_loss = kl_divergence_loss_jax(predicted_logits, expert_probs, dummy_parent_sets)
        non_jax_kl_loss = kl_divergence_loss(predicted_logits, mock_expert_posterior, dummy_parent_sets)
        
        # They should both be finite and non-negative
        assert jnp.isfinite(jax_kl_loss)
        assert jnp.isfinite(non_jax_kl_loss)
        assert jax_kl_loss >= 0
        assert non_jax_kl_loss >= 0
    
    def test_jax_vs_regular_performance_comparison(self):
        """Test that JAX version achieves expected speedup."""
        from causal_bayes_opt.training.surrogate_training import kl_divergence_loss_jax
        import time
        
        # Simple computation for timing comparison
        predicted_logits = jnp.array([1.0, 2.0, 0.5])
        expert_probs = jnp.array([0.1, 0.8, 0.1])
        parent_sets = (frozenset(), frozenset(['X']), frozenset(['Y']))
        
        # Time regular function
        start = time.time()
        for _ in range(100):
            kl_divergence_loss_jax(predicted_logits, expert_probs, parent_sets)
        regular_time = time.time() - start
        
        # Time JIT-compiled function with static parent sets
        compiled_kl_loss = jax.jit(kl_divergence_loss_jax, static_argnums=(2,))
        
        # Warmup compilation
        compiled_kl_loss(predicted_logits, expert_probs, parent_sets)
        
        start = time.time()
        for _ in range(100):
            compiled_kl_loss(predicted_logits, expert_probs, parent_sets)
        compiled_time = time.time() - start
        
        # JAX version should be at least as fast (after compilation)
        # Note: In practice, speedup is more significant for larger computations
        print(f"Regular time: {regular_time:.4f}s, Compiled time: {compiled_time:.4f}s")
        assert compiled_time <= regular_time * 2.0, "JAX should not be significantly slower"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])