"""
Tests for Modular Training Components

This module tests the refactored, modular training components that follow
CLAUDE.md principles of single responsibility and immutable state.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from omegaconf import DictConfig

import jax.random as random
import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.training.modular_trainer import (
    TrainingMetrics, PolicyFactory, SCMRotationManager,
    StateConverter, CheckpointManager, MetricsCollector
)


class TestTrainingMetrics:
    """Test immutable training metrics dataclass."""
    
    def test_metrics_immutability(self):
        """Test that metrics are immutable."""
        metrics = TrainingMetrics(
            episode=5,
            mean_reward=1.5,
            structure_accuracy=0.8,
            optimization_improvement=0.3,
            policy_loss=0.1,
            value_loss=0.05,
            scm_type="fork"
        )
        
        # Should be frozen (immutable)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            metrics.episode = 10
    
    def test_metrics_creation(self):
        """Test metrics creation with valid values."""
        metrics = TrainingMetrics(
            episode=1,
            mean_reward=2.0,
            structure_accuracy=0.9,
            optimization_improvement=0.5,
            policy_loss=0.2,
            value_loss=0.1,
            scm_type="chain"
        )
        
        assert metrics.episode == 1
        assert metrics.mean_reward == 2.0
        assert metrics.scm_type == "chain"


class TestSCMRotationManager:
    """Test SCM rotation management."""
    
    def test_scm_rotation_creation(self):
        """Test SCM rotation creation."""
        config = DictConfig({
            'seed': 42,
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 4],
                    'structure_types': ['fork', 'chain'],
                    'rotation_frequency': 3
                }
            }
        })
        
        with patch('causal_bayes_opt.training.modular_trainer.VariableSCMFactory') as mock_factory_cls:
            # Setup mock factory
            mock_factory = Mock()
            # Need enough SCMs for all combinations: 2 variables (3,4) * 2 structures = 4 SCMs
            mock_scms = [
                pyr.m(variables={'X', 'Y', 'Z'}, target='Z'),        # 3var fork
                pyr.m(variables={'A', 'B', 'C'}, target='C'),        # 3var chain
                pyr.m(variables={'P', 'Q', 'R', 'S'}, target='S'),   # 4var fork
                pyr.m(variables={'W', 'X', 'Y', 'Z'}, target='Z')    # 4var chain
            ]
            mock_factory.create_variable_scm.side_effect = mock_scms
            mock_factory_cls.return_value = mock_factory
            
            manager = SCMRotationManager(config)
            
            # Should have created SCMs
            assert len(manager.scm_rotation) == 4  # 2 variables * 2 structures
            assert manager.max_variables == 4  # Max from mock SCMs
    
    def test_get_current_scm(self):
        """Test SCM selection based on episode."""
        config = DictConfig({
            'seed': 42,
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 3],
                    'structure_types': ['fork'],
                    'rotation_frequency': 2
                }
            }
        })
        
        with patch('causal_bayes_opt.training.modular_trainer.VariableSCMFactory'):
            manager = SCMRotationManager(config)
            
            # Mock the rotation
            manager.scm_rotation = [
                ('scm_0', pyr.m()),
                ('scm_1', pyr.m()),
                ('scm_2', pyr.m())
            ]
            
            # Test rotation with frequency 2
            assert manager.get_current_scm(0)[0] == 'scm_0'
            assert manager.get_current_scm(1)[0] == 'scm_0'
            assert manager.get_current_scm(2)[0] == 'scm_1'
            assert manager.get_current_scm(3)[0] == 'scm_1'
            assert manager.get_current_scm(4)[0] == 'scm_2'
            assert manager.get_current_scm(5)[0] == 'scm_2'
            assert manager.get_current_scm(6)[0] == 'scm_0'  # Wrap around


class TestStateConverter:
    """Test state conversion to enriched representation."""
    
    def test_state_converter_initialization(self):
        """Test state converter initializes correctly."""
        config = DictConfig({
            'training': {
                'state_config': {
                    'max_history_size': 50,
                    'standardize_values': True,
                    'include_temporal_features': True
                }
            }
        })
        
        converter = StateConverter(config, max_variables=4)
        
        assert converter.max_variables == 4
        assert converter.history_builder.max_history_size == 50
        assert converter.history_builder.support_variable_scms is True
    
    def test_variable_count_mismatch_padding(self):
        """Test padding when current variables < target variables."""
        config = DictConfig({
            'training': {
                'state_config': {
                    'max_history_size': 20
                }
            }
        })
        
        converter = StateConverter(config, max_variables=5)
        
        # Create input with 3 variables, target is 5
        input_tensor = jnp.ones((20, 3, 10))
        
        result = converter._handle_variable_count_mismatch(input_tensor, 3, 5)
        
        assert result.shape == (20, 5, 10)
        # Original data preserved
        assert jnp.allclose(result[:, :3, :], input_tensor)
        # Padding is zeros
        assert jnp.allclose(result[:, 3:, :], 0.0)
    
    def test_variable_count_mismatch_truncation(self):
        """Test truncation when current variables > target variables."""
        config = DictConfig({
            'training': {
                'state_config': {
                    'max_history_size': 20
                }
            }
        })
        
        converter = StateConverter(config, max_variables=3)
        
        # Create input with 5 variables, target is 3
        input_tensor = jnp.arange(20 * 5 * 10).reshape(20, 5, 10)
        
        result = converter._handle_variable_count_mismatch(input_tensor, 5, 3)
        
        assert result.shape == (20, 3, 10)
        # First 3 variables preserved
        assert jnp.allclose(result, input_tensor[:, :3, :])


class TestCheckpointManager:
    """Test checkpoint management."""
    
    def test_checkpoint_manager_initialization(self):
        """Test checkpoint manager setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DictConfig({
                'logging': {
                    'checkpoint_dir': temp_dir
                }
            })
            
            manager = CheckpointManager(config)
            
            assert manager.checkpoint_dir == Path(temp_dir)
            assert manager.checkpoint_dir.exists()
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DictConfig({
                'logging': {'checkpoint_dir': temp_dir},
                'training': {'learning_rate': 0.001}
            })
            
            manager = CheckpointManager(config)
            
            # Mock parameters and config
            params = {'weights': jnp.ones((4, 8))}
            policy_config = {'architecture': {'hidden_dim': 128}}
            
            metrics = TrainingMetrics(
                episode=10, mean_reward=1.5, structure_accuracy=0.8,
                optimization_improvement=0.3, policy_loss=0.1,
                value_loss=0.05, scm_type="fork"
            )
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                params, policy_config, episode=10, metrics=metrics
            )
            
            # Verify checkpoint was saved
            assert checkpoint_path.exists()
            checkpoint_file = checkpoint_path / "checkpoint.pkl"
            assert checkpoint_file.exists()
            
            # Verify checkpoint content
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            
            assert data['episode'] == 10
            assert data['enriched_architecture'] is True
            assert 'metrics' in data
            assert data['metrics']['mean_reward'] == 1.5


class TestMetricsCollector:
    """Test metrics collection and analysis."""
    
    def test_metrics_collector_immutability(self):
        """Test that metrics collector is immutable."""
        collector = MetricsCollector()
        
        metrics = TrainingMetrics(
            episode=1, mean_reward=1.0, structure_accuracy=0.7,
            optimization_improvement=0.2, policy_loss=0.1,
            value_loss=0.05, scm_type="fork"
        )
        
        # Adding metrics returns new collector
        new_collector = collector.add_metrics(metrics)
        
        # Original collector unchanged
        assert len(collector.metrics_history) == 0
        # New collector has the metrics
        assert len(new_collector.metrics_history) == 1
        assert new_collector.get_latest_metrics() == metrics
    
    def test_performance_analysis(self):
        """Test performance analysis computation."""
        collector = MetricsCollector()
        
        # Add multiple metrics
        metrics_list = [
            TrainingMetrics(1, 1.0, 0.7, 0.1, 0.1, 0.05, "fork"),
            TrainingMetrics(2, 1.5, 0.8, 0.2, 0.08, 0.04, "chain"),
            TrainingMetrics(3, 2.0, 0.9, 0.3, 0.06, 0.03, "collider")
        ]
        
        for metrics in metrics_list:
            collector = collector.add_metrics(metrics)
        
        # Analyze performance
        analysis = collector.analyze_performance(total_time=10.0)
        
        assert analysis['total_episodes'] == 3
        assert analysis['training_time'] == 10.0
        assert analysis['final_reward'] == 2.0
        assert analysis['mean_reward'] == 1.5
        assert analysis['final_accuracy'] == 0.9
        assert analysis['reward_improvement'] == 1.0  # 2.0 - 1.0
        assert analysis['episodes_per_second'] == 0.3  # 3/10


class TestModularTrainerIntegration:
    """Test integration between modular components."""
    
    def test_components_work_together(self):
        """Test that components integrate correctly."""
        config = DictConfig({
            'seed': 42,
            'training': {
                'learning_rate': 0.001,
                'architecture': {
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'num_heads': 4,
                    'key_size': 16,
                    'widening_factor': 2,
                    'dropout': 0.1
                },
                'state_config': {
                    'max_history_size': 20,
                    'standardize_values': True
                }
            },
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 3],
                    'structure_types': ['fork'],
                    'rotation_frequency': 5
                }
            },
            'logging': {'checkpoint_dir': '/tmp/test'}
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.logging.checkpoint_dir = temp_dir
            
            with patch('causal_bayes_opt.training.modular_trainer.VariableSCMFactory') as mock_factory_cls:
                # Setup mocks
                mock_factory = Mock()
                mock_factory.create_variable_scm.return_value = pyr.m(
                    variables={'X', 'Y', 'Z'}, target='Z'
                )
                mock_factory_cls.return_value = mock_factory
                
                # Create components
                scm_manager = SCMRotationManager(config)
                state_converter = StateConverter(config, scm_manager.max_variables)
                checkpoint_manager = CheckpointManager(config)
                metrics_collector = MetricsCollector()
                
                # Test that they work together
                assert scm_manager.max_variables == 3
                assert state_converter.max_variables == 3
                assert checkpoint_manager.checkpoint_dir.exists()
                
                # Test metrics flow
                metrics = TrainingMetrics(1, 1.0, 0.8, 0.2, 0.1, 0.05, "fork")
                metrics_collector = metrics_collector.add_metrics(metrics)
                
                assert metrics_collector.get_latest_metrics() == metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])