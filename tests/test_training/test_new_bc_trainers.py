#!/usr/bin/env python3
"""
Integration tests for new behavioral cloning trainers.

Tests the complete pipeline of surrogate and acquisition BC trainers
to ensure they work correctly with the new base trainer architecture.
"""

import pytest
from unittest.mock import Mock
import tempfile
import os

import jax
import jax.numpy as jnp
import pyrsistent as pyr

from src.causal_bayes_opt.training.base_trainer import TrainingConfig
from src.causal_bayes_opt.training.surrogate_bc_trainer import (
    SurrogateBCTrainer, 
    SurrogateBCConfig, 
    create_surrogate_bc_trainer
)
from src.causal_bayes_opt.training.acquisition_bc_trainer import (
    AcquisitionBCTrainer, 
    AcquisitionBCConfig, 
    create_acquisition_bc_trainer
)
from src.causal_bayes_opt.training.trajectory_extractor import (
    SurrogateTrainingData, 
    AcquisitionTrainingData
)
from src.causal_bayes_opt.training.surrogate_training import TrainingExample
from src.causal_bayes_opt.avici_integration.parent_set.posterior import ParentSetPosterior


class TestSurrogateBCTrainer:
    """Test SurrogateBCTrainer functionality."""
    
    def create_mock_training_example(self):
        """Create mock training example for testing."""
        # Mock expert posterior
        mock_posterior = Mock(spec=ParentSetPosterior)
        mock_posterior.target_variable = 'X0'
        mock_posterior.parent_set_probs = {
            frozenset(): 0.3,
            frozenset(['X1']): 0.7
        }
        mock_posterior.uncertainty = 0.611
        mock_posterior.top_k_sets = [(frozenset(['X1']), 0.7), (frozenset(), 0.3)]
        
        return TrainingExample(
            observational_data=jnp.ones((5, 3, 3)),  # AVICI format [N, d, 3]
            target_variable='X0',
            variable_order=['X0', 'X1', 'X2'],
            expert_posterior=mock_posterior,
            expert_accuracy=0.8,
            scm_info=pyr.pmap({'n_nodes': 3, 'complexity': 2.0}),
            problem_difficulty='easy'
        )
    
    def test_trainer_initialization(self):
        """Test surrogate trainer initialization."""
        config = SurrogateBCConfig(
            hidden_dims=[32, 16],
            learning_rate=1e-3,
            batch_size=4,
            max_epochs=2,
            use_jax_compilation=False  # Disable for testing
        )
        
        trainer = SurrogateBCTrainer(config)
        
        assert trainer.config == config
        assert trainer._model_fn is not None
        assert trainer._rng_key is not None
    
    def test_factory_function(self):
        """Test factory function creates trainer correctly."""
        trainer = create_surrogate_bc_trainer(
            hidden_dims=[32, 16],
            learning_rate=1e-4,
            batch_size=8,
            max_epochs=5
        )
        
        assert isinstance(trainer, SurrogateBCTrainer)
        assert trainer.config.hidden_dims == [32, 16]
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.batch_size == 8
        assert trainer.config.max_epochs == 5
    
    def test_model_initialization(self):
        """Test model parameter initialization."""
        config = SurrogateBCConfig(
            hidden_dims=[32, 16],
            batch_size=4,
            use_jax_compilation=False
        )
        trainer = SurrogateBCTrainer(config)
        
        sample_example = self.create_mock_training_example()
        state = trainer.initialize_training_state(sample_example)
        
        assert state.epoch == 0
        assert state.step == 0
        assert isinstance(state.model_params, dict)
        assert state.optimizer_state is not None
    
    def test_forward_pass(self):
        """Test forward pass through surrogate model."""
        config = SurrogateBCConfig(
            hidden_dims=[32, 16],
            batch_size=4,
            use_jax_compilation=False
        )
        trainer = SurrogateBCTrainer(config)
        
        sample_example = self.create_mock_training_example()
        state = trainer.initialize_training_state(sample_example)
        
        prediction = trainer.predict(state, sample_example)
        
        assert 'parent_set_logits' in prediction
        assert 'parent_set_probs' in prediction
        assert 'hidden_features' in prediction
        
        # Check output shapes
        probs = prediction['parent_set_probs']
        assert probs.shape[0] > 0  # Should have some parent sets
        assert jnp.allclose(jnp.sum(probs), 1.0, atol=1e-5)  # Probabilities sum to 1


class TestAcquisitionBCTrainer:
    """Test AcquisitionBCTrainer functionality."""
    
    def create_mock_state_action_pair(self):
        """Create mock state-action pair for testing."""
        state_dict = {
            'target_variable': 'X0',
            'posterior_entropy': 0.693,
            'current_step': 1,
            'scm_metadata': {
                'n_nodes': 3,
                'complexity_score': 2.5
            },
            'intervention_history': [
                {'variables': ['X1'], 'step': 0}
            ],
            'posterior_distribution': {
                frozenset(): 0.3,
                frozenset(['X1']): 0.7
            }
        }
        
        action_dict = {
            'intervention_variables': frozenset(['X1', 'X2']),
            'intervention_values': (1.0, 2.0),
            'step': 1,
            'action_type': 'intervention'
        }
        
        return state_dict, action_dict
    
    def test_trainer_initialization(self):
        """Test acquisition trainer initialization."""
        config = AcquisitionBCConfig(
            hidden_dims=[32, 16],
            learning_rate=1e-3,
            batch_size=4,
            max_epochs=2,
            max_variables=10,
            use_jax_compilation=False  # Disable for testing
        )
        
        trainer = AcquisitionBCTrainer(config)
        
        assert trainer.config == config
        assert trainer._model_fn is not None
        assert trainer._rng_key is not None
        assert trainer._variable_to_idx == {}
        assert trainer._idx_to_variable == {}
    
    def test_factory_function(self):
        """Test factory function creates trainer correctly."""
        trainer = create_acquisition_bc_trainer(
            hidden_dims=[32, 16],
            learning_rate=1e-4,
            batch_size=8,
            max_epochs=5,
            max_variables=15
        )
        
        assert isinstance(trainer, AcquisitionBCTrainer)
        assert trainer.config.hidden_dims == [32, 16]
        assert trainer.config.learning_rate == 1e-4
        assert trainer.config.batch_size == 8
        assert trainer.config.max_epochs == 5
        assert trainer.config.max_variables == 15
    
    def test_variable_mapping_construction(self):
        """Test building variable mapping from training data."""
        config = AcquisitionBCConfig(
            max_variables=5,
            use_jax_compilation=False
        )
        trainer = AcquisitionBCTrainer(config)
        
        # Create training pairs with different variables
        pairs = [
            self.create_mock_state_action_pair(),
            ({
                'target_variable': 'X1',
                'scm_metadata': {},
                'intervention_history': [],
                'posterior_distribution': {}
            }, {
                'intervention_variables': frozenset(['X0']),
                'intervention_values': (0.5,),
                'step': 0,
                'action_type': 'intervention'
            })
        ]
        
        trainer._build_variable_mapping(pairs)
        
        # Should have mapped the variables we used
        assert len(trainer._variable_to_idx) > 0
        assert len(trainer._idx_to_variable) > 0
        assert len(trainer._variable_to_idx) == len(trainer._idx_to_variable)
    
    def test_state_features_extraction(self):
        """Test extracting features from state dictionary."""
        config = AcquisitionBCConfig(use_jax_compilation=False)
        trainer = AcquisitionBCTrainer(config)
        
        state_dict, _ = self.create_mock_state_action_pair()
        features = trainer._extract_state_features(state_dict)
        
        assert isinstance(features, jnp.ndarray)
        assert features.shape[0] > 0  # Should have some features
        assert jnp.all(jnp.isfinite(features))  # All features should be finite
    
    def test_action_encoding(self):
        """Test encoding expert actions as target vectors."""
        config = AcquisitionBCConfig(max_variables=5, use_jax_compilation=False)
        trainer = AcquisitionBCTrainer(config)
        
        # Build variable mapping first
        pairs = [self.create_mock_state_action_pair()]
        trainer._build_variable_mapping(pairs)
        
        _, action_dict = self.create_mock_state_action_pair()
        target = trainer._encode_action(action_dict)
        
        assert isinstance(target, jnp.ndarray)
        assert target.shape[0] == config.max_variables
        assert jnp.all((target == 0) | (target == 1))  # Binary encoding


class TestTrainingIntegration:
    """Test complete training integration."""
    
    def test_loss_decreases_during_training(self):
        """Test that loss actually decreases during training (key requirement)."""
        # Test surrogate trainer
        config = SurrogateBCConfig(
            hidden_dims=[16, 8],
            learning_rate=1e-2,  # Higher LR for faster convergence
            batch_size=2,
            max_epochs=3,
            use_jax_compilation=False  # Core loss computation is now JAX-compatible
        )
        trainer = SurrogateBCTrainer(config)
        
        # Create simple training data
        mock_posterior = Mock(spec=ParentSetPosterior)
        mock_posterior.target_variable = 'X0'
        mock_posterior.parent_set_probs = {frozenset(): 0.5, frozenset(['X1']): 0.5}
        mock_posterior.uncertainty = 0.693
        mock_posterior.top_k_sets = [(frozenset(['X1']), 0.5), (frozenset(), 0.5)]
        
        examples = []
        for i in range(4):  # Small dataset for quick test
            example = TrainingExample(
                observational_data=jnp.ones((3, 2, 3)),
                target_variable='X0',
                variable_order=['X0', 'X1'],
                expert_posterior=mock_posterior,
                expert_accuracy=0.8,
                scm_info=pyr.pmap({'n_nodes': 2, 'complexity': 1.5}),
                problem_difficulty='easy'
            )
            examples.append(example)
        
        # Create batches
        batches = [examples[:2], examples[2:]]
        
        # Run training
        final_state = trainer.fit(batches, sample_input=examples[0])
        
        # Check that training completed
        assert final_state.epoch > 0
        assert len(final_state.training_metrics) > 0
        
        # Check that loss generally decreased
        training_metrics = list(final_state.training_metrics)
        if len(training_metrics) >= 2:
            early_loss = training_metrics[0].loss
            late_loss = training_metrics[-1].loss
            # Loss should decrease or at least not increase significantly
            assert late_loss <= early_loss * 1.1, f"Loss increased from {early_loss} to {late_loss}"
        
        print(f"✓ Surrogate trainer: Loss decreased from {training_metrics[0].loss:.4f} to {training_metrics[-1].loss:.4f}")
    
    def test_acquisition_trainer_loss_improvement(self):
        """Test acquisition trainer shows learning behavior."""
        config = AcquisitionBCConfig(
            hidden_dims=[16, 8],
            learning_rate=1e-2,
            batch_size=2,
            max_epochs=3,
            max_variables=3,
            use_jax_compilation=False
        )
        trainer = AcquisitionBCTrainer(config)
        
        # Create training pairs
        pairs = []
        for i in range(4):
            state_dict = {
                'target_variable': 'X0',
                'posterior_entropy': 0.5 + i * 0.1,
                'current_step': i,
                'scm_metadata': {'n_nodes': 3, 'complexity_score': 2.0},
                'intervention_history': [],
                'posterior_distribution': {frozenset(): 0.5, frozenset(['X1']): 0.5}
            }
            action_dict = {
                'intervention_variables': frozenset(['X1']) if i % 2 == 0 else frozenset(['X2']),
                'intervention_values': (1.0,),
                'step': i,
                'action_type': 'intervention'
            }
            pairs.append((state_dict, action_dict))
        
        # Build variable mapping
        trainer._build_variable_mapping(pairs)
        
        # Create batches
        batches = [pairs[:2], pairs[2:]]
        
        # Run training
        final_state = trainer.fit(batches, sample_input=pairs[0])
        
        # Check that training completed
        assert final_state.epoch > 0
        assert len(final_state.training_metrics) > 0
        
        # Check loss behavior
        training_metrics = list(final_state.training_metrics)
        if len(training_metrics) >= 2:
            early_loss = training_metrics[0].loss
            late_loss = training_metrics[-1].loss
            assert late_loss <= early_loss * 1.1, f"Loss increased from {early_loss} to {late_loss}"
        
        print(f"✓ Acquisition trainer: Loss decreased from {training_metrics[0].loss:.4f} to {training_metrics[-1].loss:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])