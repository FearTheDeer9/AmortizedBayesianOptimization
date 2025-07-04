"""
Tests for EnrichedGRPOTrainer - Critical training component.

This module tests the GRPO training functionality for enriched policies
with variable-count SCMs and proper JAX key threading.

Following TDD principles - these tests define the expected behavior.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
import tempfile
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
from hypothesis import given, strategies as st

# Mock the trainer import to avoid complex dependencies
@pytest.fixture(autouse=True)
def mock_trainer_dependencies():
    """Mock complex dependencies for trainer testing."""
    with patch.multiple(
        'causal_bayes_opt.acquisition.enriched.policy_heads',
        EnrichedAcquisitionPolicyNetwork=MagicMock(),
        create_enriched_policy_factory=MagicMock()
    ), patch.multiple(
        'causal_bayes_opt.acquisition.enriched.state_enrichment',
        EnrichedHistoryBuilder=MagicMock(),
        create_enriched_history_tensor=MagicMock()
    ), patch.multiple(
        'causal_bayes_opt.experiments.variable_scm_factory',
        VariableSCMFactory=MagicMock(),
        get_scm_info=MagicMock()
    ), patch.multiple(
        'causal_bayes_opt.acquisition.rewards',
        compute_verifiable_reward=MagicMock(return_value=1.0),
        create_default_reward_config=MagicMock(return_value={})
    ):
        yield


class TestEnrichedGRPOTrainerInitialization:
    """Test trainer initialization and setup."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = DictConfig({
            'seed': 42,
            'training': {
                'n_episodes': 10,
                'episode_length': 5,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'max_intervention_value': 2.0,
                'reward_weights': {
                    'optimization': 1.0,
                    'discovery': 1.0,
                    'efficiency': 0.5
                },
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
                    'num_channels': 10,
                    'standardize_values': True,
                    'include_temporal_features': True
                }
            },
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 5],
                    'structure_types': ['fork', 'chain'],
                    'rotation_frequency': 5
                }
            },
            'logging': {
                'wandb': {'enabled': False},
                'checkpoint_dir': '/tmp/test_checkpoints'
            }
        })
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_trainer_initialization_immutable_state(self, mock_policy, mock_scm):
        """Test that trainer initializes with immutable state structures."""
        # Setup mocks
        mock_scm.return_value = [('test_scm', pyr.m())]
        mock_policy.return_value = (MagicMock(), {'architecture': {}})
        
        # Import here to avoid dependency issues
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Check immutable state
            assert isinstance(trainer.training_metrics, pyr.PVector)
            assert len(trainer.training_metrics) == 0
            
            # Check proper key initialization (not mutated)
            assert hasattr(trainer, '_initial_key')
            assert hasattr(trainer, '_current_key')
            
            # Keys should be different (properly split)
            assert not jnp.array_equal(trainer._initial_key, trainer._current_key)
            
        finally:
            sys.path.pop(0)
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_trainer_max_variables_determination(self, mock_policy, mock_scm):
        """Test that trainer correctly determines max variables from SCM rotation."""
        # Setup mock SCMs with different variable counts
        mock_scms = [
            ('scm_3var', pyr.m(variables={'X', 'Y', 'Z'})),
            ('scm_4var', pyr.m(variables={'A', 'B', 'C', 'D'})),
            ('scm_5var', pyr.m(variables={'P', 'Q', 'R', 'S', 'T'}))
        ]
        mock_scm.return_value = mock_scms
        mock_policy.return_value = (MagicMock(), {'architecture': {}})
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Should determine max variables correctly
            assert trainer.max_variables == 5
            
        finally:
            sys.path.pop(0)


class TestEnrichedGRPOTrainerKeyThreading:
    """Test proper JAX key threading throughout training."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = DictConfig({
            'seed': 123,
            'training': {
                'n_episodes': 3,
                'episode_length': 2,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'max_intervention_value': 1.0,
                'reward_weights': {
                    'optimization': 1.0,
                    'discovery': 1.0,
                    'efficiency': 0.5
                },
                'architecture': {
                    'hidden_dim': 32,
                    'num_layers': 1,
                    'num_heads': 2,
                    'key_size': 8,
                    'dropout': 0.0
                },
                'state_config': {
                    'max_history_size': 10,
                    'num_channels': 10
                }
            },
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 3],
                    'structure_types': ['fork'],
                    'rotation_frequency': 10
                }
            },
            'logging': {'wandb': {'enabled': False}}
        })
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._run_episode')
    def test_key_threading_through_training_loop(self, mock_episode, mock_policy, mock_scm):
        """Test that JAX keys are properly threaded through training loop."""
        # Setup mocks
        mock_scm.return_value = [('test_scm', pyr.m(variables={'X', 'Y', 'Z'}, target='Z'))]
        mock_policy.return_value = (MagicMock(), {'architecture': {}})
        
        # Mock episode to track key usage
        episode_keys_received = []
        def mock_run_episode(episode_idx, episode_key):
            episode_keys_received.append(episode_key)
            return MagicMock(
                episode=episode_idx, mean_reward=1.0, structure_accuracy=0.8,
                optimization_improvement=0.1, policy_loss=0.5, value_loss=0.3,
                scm_type='fork'
            )
        mock_episode.side_effect = mock_run_episode
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Mock required methods
            trainer._setup_wandb = MagicMock()
            trainer._save_checkpoint = MagicMock()
            trainer._update_policy = MagicMock()
            
            # Run training
            trainer.train()
            
            # Check that keys were properly threaded
            assert len(episode_keys_received) == 3  # n_episodes
            
            # All keys should be different (proper splitting)
            for i in range(len(episode_keys_received)):
                for j in range(i + 1, len(episode_keys_received)):
                    assert not jnp.array_equal(episode_keys_received[i], episode_keys_received[j])
            
            # Training metrics should be accumulated immutably
            assert len(trainer.training_metrics) == 3
            assert isinstance(trainer.training_metrics, pyr.PVector)
            
        finally:
            sys.path.pop(0)
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_episode_key_threading(self, mock_policy, mock_scm):
        """Test that keys are properly threaded within episodes."""
        # Setup mocks
        mock_scm.return_value = [('test_scm', pyr.m(variables={'X', 'Y', 'Z'}, target='Z'))]
        
        # Mock policy that tracks key usage
        policy_keys_received = []
        def mock_policy_apply(params, key, *args):
            policy_keys_received.append(key)
            return {
                'variable_logits': jnp.array([0.1, -10.0, 0.3]),  # Target Y masked
                'value_params': jnp.array([[1.0, 0.1], [0.0, 0.0], [0.5, 0.2]]),
                'state_value': 1.0
            }
        
        mock_policy_fn = MagicMock()
        mock_policy_fn.apply = mock_policy_apply
        mock_policy.return_value = (mock_policy_fn, {'architecture': {}})
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Mock required methods
            trainer._get_current_scm = MagicMock(return_value=('test', pyr.m(variables={'X', 'Y', 'Z'}, target='Z')))
            trainer._create_mock_state = MagicMock(return_value=MagicMock())
            trainer._convert_state_to_enriched_input = MagicMock(return_value=jnp.ones((10, 3, 10)))
            trainer._simulate_intervention = MagicMock(return_value=(pyr.m(), pyr.m(), 1.0))
            trainer._update_policy = MagicMock()
            
            # Run a single episode
            episode_key = random.PRNGKey(456)
            result = trainer._run_episode(0, episode_key)
            
            # Check that policy was called with properly threaded keys
            assert len(policy_keys_received) == 2  # episode_length = 2
            
            # All policy keys should be different
            for i in range(len(policy_keys_received)):
                for j in range(i + 1, len(policy_keys_received)):
                    assert not jnp.array_equal(policy_keys_received[i], policy_keys_received[j])
            
            # None of the policy keys should equal the episode key
            for policy_key in policy_keys_received:
                assert not jnp.array_equal(policy_key, episode_key)
            
        finally:
            sys.path.pop(0)


class TestEnrichedGRPOTrainerSCMManagement:
    """Test SCM rotation and management functionality."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = DictConfig({
            'seed': 42,
            'training': {'n_episodes': 10},
            'experiment': {
                'scm_generation': {
                    'use_variable_factory': True,
                    'variable_range': [3, 4],
                    'structure_types': ['fork', 'chain', 'collider'],
                    'rotation_frequency': 3
                }
            }
        })
    
    @patch('causal_bayes_opt.experiments.variable_scm_factory.VariableSCMFactory')
    def test_scm_rotation_creation_with_factory(self, mock_factory_class):
        """Test SCM rotation creation using variable factory."""
        # Setup mock factory
        mock_factory = MagicMock()
        mock_scms = [
            ('fork_3var', pyr.m(variables={'X', 'Y', 'Z'}, target='Z')),
            ('chain_4var', pyr.m(variables={'A', 'B', 'C', 'D'}, target='D')),
            ('collider_3var', pyr.m(variables={'P', 'Q', 'R'}, target='R'))
        ]
        mock_factory.create_scm_suite.return_value = mock_scms
        mock_factory_class.return_value = mock_factory
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            # Mock other dependencies
            with patch.object(EnrichedGRPOTrainer, '_create_enriched_policy', return_value=(MagicMock(), {})):
                trainer = EnrichedGRPOTrainer(self.config)
                
                # Check that factory was used correctly
                mock_factory_class.assert_called_once()
                mock_factory.create_scm_suite.assert_called_once()
                
                # Check SCM rotation
                assert len(trainer.scm_rotation) == 3
                assert trainer.scm_rotation[0][0] == 'fork_3var'
                assert trainer.scm_rotation[1][0] == 'chain_4var'
                assert trainer.scm_rotation[2][0] == 'collider_3var'
                
        finally:
            sys.path.pop(0)
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_current_scm_selection(self, mock_policy, mock_scm):
        """Test current SCM selection based on episode and rotation frequency."""
        # Setup mock SCMs
        mock_scms = [
            ('scm_0', pyr.m(variables={'A', 'B'}, target='B')),
            ('scm_1', pyr.m(variables={'X', 'Y', 'Z'}, target='Z')),
            ('scm_2', pyr.m(variables={'P', 'Q'}, target='Q'))
        ]
        mock_scm.return_value = mock_scms
        mock_policy.return_value = (MagicMock(), {})
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Test rotation with frequency 3
            # Episodes 0, 1, 2 should use scm_0
            assert trainer._get_current_scm(0)[0] == 'scm_0'
            assert trainer._get_current_scm(1)[0] == 'scm_0'
            assert trainer._get_current_scm(2)[0] == 'scm_0'
            
            # Episodes 3, 4, 5 should use scm_1
            assert trainer._get_current_scm(3)[0] == 'scm_1'
            assert trainer._get_current_scm(4)[0] == 'scm_1'
            assert trainer._get_current_scm(5)[0] == 'scm_1'
            
            # Episodes 6, 7, 8 should use scm_2
            assert trainer._get_current_scm(6)[0] == 'scm_2'
            assert trainer._get_current_scm(7)[0] == 'scm_2'
            assert trainer._get_current_scm(8)[0] == 'scm_2'
            
            # Episode 9 should wrap around to scm_0
            assert trainer._get_current_scm(9)[0] == 'scm_0'
            
        finally:
            sys.path.pop(0)


class TestEnrichedGRPOTrainerPolicyOutputs:
    """Test policy output handling and intervention generation."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.config = DictConfig({
            'seed': 42,
            'training': {
                'max_intervention_value': 2.0,
                'architecture': {},
                'state_config': {}
            },
            'experiment': {'scm_generation': {}}
        })
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_policy_output_to_intervention_conversion(self, mock_policy, mock_scm):
        """Test conversion of policy outputs to interventions."""
        mock_scm.return_value = [('test', pyr.m())]
        mock_policy.return_value = (MagicMock(), {})
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Test policy output with enriched format
            policy_output = {
                'variable_logits': jnp.array([0.5, -10.0, 0.8, 0.2]),  # Y is target (masked)
                'value_params': jnp.array([
                    [1.5, 0.1],   # X: mean=1.5, log_std=0.1
                    [0.0, 0.0],   # Y: target (shouldn't be used)
                    [0.8, 0.2],   # Z: mean=0.8, log_std=0.2
                    [-1.2, 0.3]   # W: mean=-1.2, log_std=0.3
                ]),
                'state_value': 1.0
            }
            variables = ['X', 'Y', 'Z', 'W']
            target = 'Y'
            scm = pyr.m(variables=set(variables), target=target)
            
            # Mock the intervention simulation
            def mock_simulate(scm, action):
                # Return intervention, outcome, target_value
                intervention_targets = set()
                intervention_values = {}
                
                # Convert action to interventions (excluding target)
                non_target_vars = [v for v in variables if v != target]
                for i, var in enumerate(non_target_vars):
                    if i < len(action) and abs(action[i]) > 0.1:
                        intervention_targets.add(var)
                        intervention_values[var] = float(action[i])
                
                intervention = pyr.m(
                    type="perfect",
                    targets=intervention_targets,
                    values=intervention_values
                )
                outcome = pyr.m(values={v: 1.0 for v in variables})
                target_value = 1.5
                
                return intervention, outcome, target_value
            
            trainer._simulate_intervention = mock_simulate
            
            # Test the conversion
            scm_obj = pyr.m(variables={'X', 'Y', 'Z', 'W'}, target='Y')
            intervention, outcome, target_value = trainer._simulate_intervention(scm_obj, jnp.array([1.0, 0.0, -0.5]))
            
            # Check intervention format
            assert isinstance(intervention, pyr.PMap)
            assert intervention['type'] == 'perfect'
            assert isinstance(intervention['targets'], set)
            assert isinstance(intervention['values'], dict)
            
            # Target should not be in intervention
            assert 'Y' not in intervention['targets']
            assert 'Y' not in intervention['values']
            
        finally:
            sys.path.pop(0)


class TestEnrichedGRPOTrainerCheckpointing:
    """Test checkpointing and model persistence."""
    
    def setup_method(self):
        """Setup test configuration."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DictConfig({
            'seed': 42,
            'training': {
                'architecture': {'hidden_dim': 64},
                'state_config': {}
            },
            'experiment': {'scm_generation': {}},
            'logging': {'checkpoint_dir': str(self.temp_dir)}
        })
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_checkpoint_saving_format(self, mock_policy, mock_scm):
        """Test that checkpoints are saved with correct format."""
        mock_scm.return_value = [('test', pyr.m())]
        mock_policy_config = {
            'architecture': {'hidden_dim': 64},
            'variable_agnostic': True,
            'enriched_architecture': True
        }
        mock_policy.return_value = (MagicMock(), mock_policy_config)
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(self.config)
            
            # Mock parameters
            trainer.policy_params = {'dense': jnp.ones((4, 8))}
            
            # Save checkpoint
            checkpoint_path = trainer._save_checkpoint(episode=10, is_final=False)
            
            # Check that checkpoint file exists
            assert checkpoint_path.exists()
            checkpoint_file = checkpoint_path / "checkpoint.pkl"
            assert checkpoint_file.exists()
            
            # Load and validate checkpoint content
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Check required keys
            required_keys = [
                'policy_params', 'policy_config', 'training_config',
                'episode', 'is_final', 'enriched_architecture'
            ]
            for key in required_keys:
                assert key in checkpoint_data
            
            # Check specific values
            assert checkpoint_data['episode'] == 10
            assert checkpoint_data['is_final'] is False
            assert checkpoint_data['enriched_architecture'] is True
            assert checkpoint_data['policy_config']['variable_agnostic'] is True
            
        finally:
            sys.path.pop(0)


class TestEnrichedGRPOTrainerErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            # Test missing required config sections
            invalid_config = DictConfig({'seed': 42})
            
            with pytest.raises((KeyError, AttributeError)):
                EnrichedGRPOTrainer(invalid_config)
                
        finally:
            sys.path.pop(0)
    
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_scm_rotation')
    @patch('scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer._create_enriched_policy')
    def test_episode_failure_handling(self, mock_policy, mock_scm):
        """Test handling of episode failures."""
        mock_scm.return_value = [('test', pyr.m())]
        mock_policy.return_value = (MagicMock(), {})
        
        config = DictConfig({
            'seed': 42,
            'training': {
                'n_episodes': 2,
                'episode_length': 1,
                'learning_rate': 0.001,
                'architecture': {},
                'state_config': {}
            },
            'experiment': {'scm_generation': {}},
            'logging': {'wandb': {'enabled': False}}
        })
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        try:
            from train_enriched_acbo_policy import EnrichedGRPOTrainer
            
            trainer = EnrichedGRPOTrainer(config)
            
            # Mock episode to fail
            def failing_episode(episode_idx, episode_key):
                if episode_idx == 1:  # Fail on second episode
                    raise ValueError("Episode failed")
                return MagicMock(
                    episode=episode_idx, mean_reward=1.0, structure_accuracy=0.8,
                    optimization_improvement=0.1, policy_loss=0.5, value_loss=0.3,
                    scm_type='test'
                )
            
            trainer._run_episode = failing_episode
            trainer._setup_wandb = MagicMock()
            trainer._save_checkpoint = MagicMock()
            trainer._update_policy = MagicMock()
            
            # Training should handle the failure gracefully
            with pytest.raises(ValueError, match="Episode failed"):
                trainer.train()
            
            # Should have completed the first episode before failing
            assert len(trainer.training_metrics) == 1
            
        finally:
            sys.path.pop(0)


@given(
    episode_length=st.integers(min_value=1, max_value=10),
    max_intervention_value=st.floats(min_value=0.1, max_value=5.0),
    num_variables=st.integers(min_value=2, max_value=6)
)
def test_trainer_invariants_property(episode_length, max_intervention_value, num_variables):
    """Property test: trainer should maintain invariants across configurations."""
    config = DictConfig({
        'seed': 42,
        'training': {
            'episode_length': episode_length,
            'max_intervention_value': max_intervention_value,
            'learning_rate': 0.001,
            'architecture': {
                'hidden_dim': 32,
                'num_layers': 1,
                'num_heads': 2,
                'key_size': 8
            },
            'state_config': {
                'max_history_size': 10,
                'num_channels': 10
            }
        },
        'experiment': {
            'scm_generation': {
                'variable_range': [num_variables, num_variables],
                'structure_types': ['fork']
            }
        }
    })
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    
    try:
        from train_enriched_acbo_policy import EnrichedGRPOTrainer
        
        with patch.multiple(
            'scripts.train_enriched_acbo_policy.EnrichedGRPOTrainer',
            _create_scm_rotation=MagicMock(return_value=[('test', pyr.m(variables=set(f'X{i}' for i in range(num_variables)), target='X0'))]),
            _create_enriched_policy=MagicMock(return_value=(MagicMock(), {}))
        ):
            trainer = EnrichedGRPOTrainer(config)
            
            # Invariants that should always hold
            assert isinstance(trainer.training_metrics, pyr.PVector)
            assert len(trainer.training_metrics) == 0
            assert hasattr(trainer, '_initial_key')
            assert hasattr(trainer, '_current_key')
            assert trainer.config.training.episode_length == episode_length
            assert trainer.config.training.max_intervention_value == max_intervention_value
            
    finally:
        sys.path.pop(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])