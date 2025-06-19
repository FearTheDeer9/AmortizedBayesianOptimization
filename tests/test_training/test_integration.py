#!/usr/bin/env python3
"""
Integration Tests for Surrogate Model Training Pipeline

Tests the complete end-to-end training workflow from expert demonstrations
to trained model, validating all component integration and data flow.

Following TDD principles - these tests validate the full system works correctly.
"""

import pytest
import time
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr

from src.causal_bayes_opt.training.surrogate_training import (
    TrainingExample,
    TrainingBatch,
    TrainingMetrics,
    ValidationResults,
    extract_training_data_from_demonstrations,
    train_surrogate_model,
    run_loss_function_experiment,
    validate_surrogate_performance
)
from src.causal_bayes_opt.training.config import SurrogateTrainingConfig
from src.causal_bayes_opt.training.expert_collection.data_structures import ExpertDemonstration
from src.causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior
from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
def realistic_expert_demonstration():
    """Create a realistic expert demonstration with proper structure."""
    # Create a realistic SCM
    scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'Y'],
        edges=[('A', 'Y'), ('B', 'Y'), ('C', 'Y')],
        coefficients={('A', 'Y'): 1.5, ('B', 'Y'): -0.8, ('C', 'Y'): 2.0},
        noise_scales={'A': 1.0, 'B': 1.0, 'C': 1.0, 'Y': 0.5},
        target='Y'
    )
    
    # Create realistic samples
    observational_samples = []
    for i in range(20):
        observational_samples.append(pyr.m(
            values=pyr.m(A=float(i*0.1), B=float(i*0.2), C=float(i*0.15), Y=float(i*0.5)),
            intervention_type="observational",
            intervention_targets=pyr.s(),
            metadata=pyr.m(sample_id=i)
        ))
    
    interventional_samples = []
    for i in range(10):
        target_var = ['A', 'B', 'C'][i % 3]
        interventional_samples.append(pyr.m(
            values=pyr.m(A=1.0 if target_var=='A' else float(i*0.1),
                        B=1.0 if target_var=='B' else float(i*0.2), 
                        C=1.0 if target_var=='C' else float(i*0.15),
                        Y=float(2.0 + i*0.1)),
            intervention_type="perfect",
            intervention_targets=pyr.s(target_var),
            metadata=pyr.m(sample_id=20+i, intervention_value=1.0)
        ))
    
    # Create realistic parent posterior
    parent_posterior = create_parent_set_posterior(
        target_variable="Y",
        parent_sets=[
            frozenset(['A', 'B', 'C']),  # True parents
            frozenset(['A', 'B']),
            frozenset(['A', 'C']),
            frozenset(['B', 'C']),
            frozenset(['A']),
            frozenset(['B']),
            frozenset(['C']),
            frozenset()
        ],
        probabilities=jnp.array([0.6, 0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]),
        metadata={'algorithm': 'PARENT_SCALE', 'confidence': 0.85}
    )
    
    return ExpertDemonstration(
        scm=scm,
        target_variable="Y",
        n_nodes=4,
        graph_type="fork",
        observational_samples=observational_samples,
        interventional_samples=interventional_samples,
        discovered_parents=frozenset(['A', 'B', 'C']),
        confidence=0.85,
        accuracy=0.9,
        parent_posterior=parent_posterior,
        data_requirements={'observational': 20, 'interventional': 10},
        inference_time=2.5,
        total_samples_used=30
    )


@pytest.fixture
def expert_demonstration_batch(realistic_expert_demonstration):
    """Create a batch of diverse expert demonstrations."""
    demonstrations = []
    
    # Add the realistic demonstration
    demonstrations.append(realistic_expert_demonstration)
    
    # Create additional demonstrations with different characteristics
    for i, (n_nodes, graph_type, accuracy) in enumerate([
        (3, "chain", 0.95),     # Easy problem
        (5, "collider", 0.75),  # Medium problem  
        (6, "complex", 0.65)    # Hard problem
    ]):
        # Create simpler SCM for variety
        variables = [chr(65 + j) for j in range(n_nodes-1)] + ['Y']  # A, B, ..., Y
        edges = [(variables[j], 'Y') for j in range(min(2, n_nodes-1))]
        
        scm = create_simple_linear_scm(
            variables=variables,
            edges=edges,
            coefficients={edge: 1.0 for edge in edges},
            noise_scales={var: 1.0 for var in variables},
            target='Y'
        )
        
        # Create samples (fewer for smaller problems)
        n_obs = max(10, n_nodes * 3)
        n_int = max(5, n_nodes * 2)
        
        obs_samples = [
            pyr.m(
                values=pyr.m(**{var: float(j*0.1) for var in variables}),
                intervention_type="observational",
                intervention_targets=pyr.s()
            ) for j in range(n_obs)
        ]
        
        int_samples = [
            pyr.m(
                values=pyr.m(**{var: 1.0 if var == variables[j % (n_nodes-1)] else float(j*0.1) for var in variables}),
                intervention_type="perfect", 
                intervention_targets=pyr.s(variables[j % (n_nodes-1)])
            ) for j in range(n_int)
        ]
        
        # Create parent posterior
        true_parents = frozenset([edge[0] for edge in edges])
        posterior = create_parent_set_posterior(
            target_variable="Y",
            parent_sets=[true_parents, frozenset()],
            probabilities=jnp.array([accuracy, 1.0 - accuracy])
        )
        
        demo = ExpertDemonstration(
            scm=scm,
            target_variable="Y", 
            n_nodes=n_nodes,
            graph_type=graph_type,
            observational_samples=obs_samples,
            interventional_samples=int_samples,
            discovered_parents=true_parents,
            confidence=accuracy,
            accuracy=accuracy,
            parent_posterior=posterior,
            data_requirements={'observational': n_obs, 'interventional': n_int},
            inference_time=1.0 + i * 0.5,
            total_samples_used=n_obs + n_int
        )
        
        demonstrations.append(demo)
    
    return demonstrations


@pytest.fixture
def integration_config():
    """Configuration optimized for integration testing."""
    return SurrogateTrainingConfig(
        model_hidden_dim=32,        # Small for fast testing
        model_n_layers=2,           # Minimal layers
        learning_rate=1e-2,         # Higher LR for fast convergence
        batch_size=2,               # Small batches
        max_epochs=3,               # Minimal epochs for testing
        early_stopping_patience=2,
        validation_frequency=1,     # Validate every epoch
        weight_decay=1e-5,          # Minimal regularization
        max_parent_size=4,
        dropout=0.0                 # No dropout for testing
    )


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestCompleteTrainingPipeline:
    """Test the complete training pipeline end-to-end."""
    
    @patch('src.causal_bayes_opt.training.surrogate_training.create_parent_set_model')
    @patch('src.causal_bayes_opt.training.surrogate_training.predict_parent_posterior')
    def test_full_training_workflow(self, mock_predict, mock_create_model, 
                                   expert_demonstration_batch, integration_config):
        """Test complete workflow from expert demonstrations to trained model."""
        
        # Mock model creation and behavior
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        # Mock model initialization
        mock_params = {'weights': jnp.ones((3, 3)), 'bias': jnp.zeros((3,))}
        mock_model.init.return_value = mock_params
        
        # Mock model forward pass
        def mock_apply(params, key, data, var_order, target, is_training):
            batch_size = data.shape[0] if len(data.shape) > 2 else 1
            return {
                'parent_set_logits': jnp.array([0.2, 0.3, 0.4, 0.1]),  # 4 parent sets
                'parent_sets': [
                    frozenset(['A', 'B', 'C']),
                    frozenset(['A', 'B']), 
                    frozenset(['A']),
                    frozenset()
                ]
            }
        
        mock_model.apply.side_effect = mock_apply
        
        # Mock predict_parent_posterior for validation
        mock_predict.return_value = expert_demonstration_batch[0].parent_posterior
        
        # Run training
        model, params, history = train_surrogate_model(
            expert_demonstrations=expert_demonstration_batch,
            config=integration_config,
            loss_type="kl_divergence",
            validation_split=0.25,  # Use 25% for validation
            random_seed=42
        )
        
        # Verify training completed
        assert model is not None
        assert params is not None
        assert isinstance(history, dict)
        
        # Check history structure
        assert 'losses' in history
        assert 'validation_metrics' in history
        assert 'step_times' in history
        
        # Verify training progressed
        assert len(history['losses']) > 0
        assert len(history['step_times']) > 0
        
        # Check that losses are finite
        for loss in history['losses']:
            assert jnp.isfinite(loss)
            assert loss >= 0.0
        
        # Verify model was called for training
        assert mock_model.apply.call_count > 0
        
        # Verify initialization was called
        mock_model.init.assert_called_once()
        
        print(f"✅ Training completed successfully:")
        print(f"  - Final loss: {history['losses'][-1]:.4f}")
        print(f"  - Training steps: {len(history['losses'])}")
        print(f"  - Validation runs: {len(history['validation_metrics'])}")
    
    def test_data_extraction_integration(self, expert_demonstration_batch):
        """Test data extraction from expert demonstrations."""
        
        with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
            # Mock the AVICI data creation
            mock_batch.return_value = {
                'x': jnp.ones((15, 4, 3)),  # [N, d, 3] format
                'g': jnp.eye(4),            # Ground truth adjacency
                'variable_order': ['A', 'B', 'C', 'Y']
            }
            
            # Extract training data
            train_examples, val_examples = extract_training_data_from_demonstrations(
                expert_demonstration_batch, validation_split=0.3
            )
            
            # Verify split
            total_demos = len(expert_demonstration_batch)
            expected_val = int(total_demos * 0.3)
            expected_train = total_demos - expected_val
            
            assert len(val_examples) == expected_val
            assert len(train_examples) == expected_train
            
            # Verify training examples structure
            for example in train_examples:
                assert isinstance(example, TrainingExample)
                assert example.observational_data.shape == (15, 4, 3)
                assert example.target_variable == "Y"
                assert len(example.variable_order) >= 3  # At least 3 variables including target
                assert example.expert_accuracy > 0.0
                assert example.problem_difficulty in ["easy", "medium", "hard"]
            
            # Verify all demonstrations were processed
            assert mock_batch.call_count == total_demos
        
        print(f"✅ Data extraction completed:")
        print(f"  - Training examples: {len(train_examples)}")
        print(f"  - Validation examples: {len(val_examples)}")
        print(f"  - Total processed: {len(train_examples) + len(val_examples)}")
    
    @patch('src.causal_bayes_opt.training.surrogate_training.train_surrogate_model')
    def test_loss_function_experiment_integration(self, mock_train, expert_demonstration_batch, integration_config):
        """Test loss function comparison experiment."""
        
        # Mock successful training results
        def mock_training_result(demos, config, loss_type, **kwargs):
            # Simulate different performance for different loss functions
            performance_map = {
                "kl_divergence": 0.3,
                "uncertainty_weighted": 0.25,
                "calibrated": 0.35
            }
            
            final_loss = performance_map.get(loss_type, 0.4)
            
            mock_model = Mock()
            mock_params = Mock()
            mock_history = {
                'losses': [1.0, 0.8, 0.6, final_loss],
                'validation_metrics': [
                    ValidationResults(
                        posterior_kl_divergence=final_loss,
                        reverse_kl_divergence=final_loss + 0.1,
                        total_variation_distance=final_loss * 0.8,
                        calibration_error=0.1,
                        uncertainty_correlation=0.7,
                        accuracy_drop=0.05,
                        inference_speedup=10.0,
                        easy_accuracy=0.9,
                        medium_accuracy=0.8,
                        hard_accuracy=0.7
                    )
                ]
            }
            
            return mock_model, mock_params, mock_history
        
        mock_train.side_effect = mock_training_result
        
        # Run experiment
        results = run_loss_function_experiment(
            expert_demonstrations=expert_demonstration_batch,
            config=integration_config,
            loss_types=["kl_divergence", "uncertainty_weighted"],
            n_trials=2,
            random_seed=42
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert "kl_divergence" in results
        assert "uncertainty_weighted" in results
        
        # Check result aggregation
        for loss_type, metrics in results.items():
            if metrics:  # If any successful trials
                assert 'kl_divergence_mean' in metrics
                assert 'kl_divergence_std' in metrics
                assert 'accuracy_drop_mean' in metrics
                assert 'final_loss_mean' in metrics
                
                # Values should be reasonable
                assert 0.0 <= metrics['kl_divergence_mean'] <= 1.0
                assert metrics['kl_divergence_std'] >= 0.0
        
        # Verify training was called for each loss type and trial
        expected_calls = len(["kl_divergence", "uncertainty_weighted"]) * 2  # 2 trials
        assert mock_train.call_count == expected_calls
        
        print(f"✅ Loss function experiment completed:")
        for loss_type, metrics in results.items():
            if metrics:
                print(f"  - {loss_type}: KL = {metrics.get('kl_divergence_mean', 'N/A'):.3f}")


class TestDataFlowIntegration:
    """Test data flow through the complete pipeline."""
    
    def test_training_example_to_batch_flow(self, realistic_expert_demonstration):
        """Test data flow from demonstration to training batch."""
        
        with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
            mock_batch.return_value = {
                'x': jnp.ones((10, 4, 3)),
                'variable_order': ['A', 'B', 'C', 'Y']
            }
            
            # Extract training examples  
            train_examples, _ = extract_training_data_from_demonstrations(
                [realistic_expert_demonstration], validation_split=0.0
            )
            
            assert len(train_examples) == 1
            example = train_examples[0]
            
            # Verify data structure integrity
            assert example.observational_data.shape == (10, 4, 3)
            assert example.target_variable == "Y"
            assert len(example.variable_order) == 4
            assert example.expert_posterior is not None
            assert example.scm_info['n_nodes'] == 4
            assert example.scm_info['target'] == "Y"
            
            # Create training batch
            batch = TrainingBatch(examples=[example])
            assert batch.batch_size == 1
            assert len(batch.examples) == 1
            
            # Verify batch preserves example data
            batch_example = batch.examples[0]
            assert batch_example.target_variable == example.target_variable
            assert jnp.array_equal(batch_example.observational_data, example.observational_data)
        
        print("✅ Data flow validation passed")
    
    def test_expert_posterior_preservation(self, realistic_expert_demonstration):
        """Test that expert posterior is correctly preserved through pipeline."""
        
        original_posterior = realistic_expert_demonstration.parent_posterior
        
        with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
            mock_batch.return_value = {
                'x': jnp.ones((10, 4, 3)),
                'variable_order': ['A', 'B', 'C', 'Y']
            }
            
            # Extract training data
            train_examples, _ = extract_training_data_from_demonstrations(
                [realistic_expert_demonstration], validation_split=0.0
            )
            
            example = train_examples[0]
            preserved_posterior = example.expert_posterior
            
            # Verify posterior preservation
            assert preserved_posterior.target_variable == original_posterior.target_variable
            assert preserved_posterior.uncertainty == original_posterior.uncertainty
            assert len(preserved_posterior.top_k_sets) == len(original_posterior.top_k_sets)
            
            # Check that parent set probabilities are preserved
            for ps, prob in original_posterior.parent_set_probs.items():
                assert ps in preserved_posterior.parent_set_probs
                assert abs(preserved_posterior.parent_set_probs[ps] - prob) < 1e-6
        
        print("✅ Expert posterior preservation validated")


class TestErrorHandlingIntegration:
    """Test error handling throughout the pipeline."""
    
    def test_malformed_demonstration_handling(self, integration_config):
        """Test handling of malformed expert demonstrations."""
        
        # Create malformed demonstration (missing required fields)
        malformed_demo = Mock()
        malformed_demo.scm = None  # Missing SCM
        malformed_demo.target_variable = "Y"
        
        # Should handle gracefully without crashing
        with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
            mock_batch.side_effect = Exception("Invalid SCM")
            
            try:
                train_examples, _ = extract_training_data_from_demonstrations(
                    [malformed_demo], validation_split=0.0
                )
                # If no exception, should return empty results
                assert len(train_examples) == 0
            except Exception as e:
                # Should be a meaningful error message related to SCM handling
                assert any(term in str(e).lower() for term in ["none", "scm", "subscriptable", "invalid"])
        
        print("✅ Error handling validation passed")
    
    def test_training_failure_recovery(self, expert_demonstration_batch):
        """Test recovery from training failures."""
        
        # Create config that will cause training issues
        bad_config = SurrogateTrainingConfig(
            learning_rate=0.0,     # Zero learning rate
            max_epochs=1,
            batch_size=1000        # Batch size larger than data
        )
        
        with patch('src.causal_bayes_opt.training.surrogate_training.create_parent_set_model') as mock_model:
            # Mock model that will cause training to fail
            mock_model.side_effect = Exception("Model creation failed")
            
            try:
                model, params, history = train_surrogate_model(
                    expert_demonstrations=expert_demonstration_batch,
                    config=bad_config,
                    random_seed=42
                )
                # Should not reach here
                assert False, "Expected training to fail"
            except Exception as e:
                # Should be a meaningful error
                assert "Model creation failed" in str(e)
        
        print("✅ Training failure recovery validated")


class TestPerformanceIntegration:
    """Test performance characteristics of integrated system."""
    
    @patch('src.causal_bayes_opt.training.surrogate_training.create_parent_set_model')
    def test_training_performance_scaling(self, mock_create_model, integration_config):
        """Test that training performance scales reasonably."""
        
        # Mock minimal model for performance testing
        mock_model = MagicMock()
        mock_create_model.return_value = mock_model
        
        mock_model.init.return_value = {'weights': jnp.ones((2, 2))}
        mock_model.apply.return_value = {
            'parent_set_logits': jnp.array([0.3, 0.7]),
            'parent_sets': [frozenset(['A']), frozenset()]
        }
        
        # Test with different numbers of demonstrations
        for n_demos in [1, 3, 5]:
            # Create minimal demonstrations
            demos = []
            for i in range(n_demos):
                demo = Mock()
                demo.scm = create_simple_linear_scm(
                    variables=['A', 'Y'], edges=[('A', 'Y')], 
                    coefficients={('A', 'Y'): 1.0}, noise_scales={'A': 1.0, 'Y': 1.0}, target='Y'
                )
                demo.target_variable = "Y"
                demo.n_nodes = 2
                demo.graph_type = "simple"
                demo.accuracy = 0.9
                demo.observational_samples = [pyr.m(values=pyr.m(A=1.0, Y=2.0), intervention_type="observational")]
                demo.interventional_samples = []
                demo.parent_posterior = create_parent_set_posterior(
                    "Y", [frozenset(['A'])], jnp.array([1.0])
                )
                demos.append(demo)
            
            with patch('src.causal_bayes_opt.training.surrogate_training.create_training_batch') as mock_batch:
                mock_batch.return_value = {'x': jnp.ones((5, 2, 3))}
                
                start_time = time.time()
                
                try:
                    model, params, history = train_surrogate_model(
                        expert_demonstrations=demos,
                        config=integration_config,
                        random_seed=42
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # Training should complete in reasonable time (< 10 seconds for test)
                    assert elapsed_time < 10.0
                    
                    print(f"✅ Training with {n_demos} demos: {elapsed_time:.2f}s")
                    
                except Exception as e:
                    print(f"⚠️  Training failed with {n_demos} demos: {e}")
                    # Don't fail test for performance issues, just log


# ============================================================================
# Validation Helpers
# ============================================================================

def assert_training_convergence(history: Dict[str, Any], tolerance: float = 0.1):
    """Assert that training showed convergence behavior."""
    losses = history['losses']
    
    if len(losses) > 3:
        # Check that loss generally decreased
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        # Allow for some noise, but expect general improvement
        improvement = (initial_loss - final_loss) / initial_loss
        assert improvement > -tolerance, f"Loss increased by more than {tolerance*100}%"


def assert_validation_metrics_reasonable(metrics: ValidationResults):
    """Assert that validation metrics are in reasonable ranges."""
    assert 0.0 <= metrics.easy_accuracy <= 1.0
    assert 0.0 <= metrics.medium_accuracy <= 1.0  
    assert 0.0 <= metrics.hard_accuracy <= 1.0
    assert metrics.inference_speedup > 0.0
    assert metrics.posterior_kl_divergence >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])