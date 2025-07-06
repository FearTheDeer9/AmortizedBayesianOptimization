"""
Tests for GRPO-TensorBackedAcquisitionState Integration

Integration tests validating that GRPO operations work correctly with the unified
tensor-state architecture, focusing on vmap compatibility and loss computation.

Following TDD principles: These tests are written FIRST and should FAIL initially.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from typing import Dict, List, Any

# Import will fail initially - this is expected for TDD
try:
    from causal_bayes_opt.jax_native.state import TensorBackedAcquisitionState
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    TensorBackedAcquisitionState = None

from causal_bayes_opt.jax_native.config import create_test_config, create_jax_config
from causal_bayes_opt.acquisition.grpo import GRPOConfig, _compute_grpo_loss
from causal_bayes_opt.interventions.handlers import create_perfect_intervention


# Helper function for test configs
def create_test_config_with_params(n_vars: int, target_idx: int):
    """Create test config with specified parameters."""
    var_names = [f'X{i}' for i in range(n_vars)]
    target_name = var_names[target_idx]
    return create_jax_config(var_names, target_name)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestGRPOVmapCompatibility:
    """Test JAX vmap operations with TensorBackedAcquisitionState."""
    
    def test_vmap_over_tensor_states(self):
        """Test that GRPO can vmap over TensorBackedAcquisitionState objects."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        # Create batch of states
        states = [
            TensorBackedAcquisitionState.create_empty(config)
            for _ in range(5)
        ]
        
        def extract_tensor_data(state):
            """Extract tensor data for policy network."""
            # This should work with vmap
            return {
                'best_value': state.best_value,
                'uncertainty': state.uncertainty_bits,
                'step': state.current_step
            }
        
        # vmap should work without errors
        vmapped_extract = jax.vmap(extract_tensor_data)
        results = vmapped_extract(states)
        
        assert results['best_value'].shape == (5,)
        assert results['uncertainty'].shape == (5,)
        assert results['step'].shape == (5,)
    
    def test_grpo_forward_pass_batch(self):
        """Test GRPO-style forward pass over batch of tensor states."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        # Create states with different properties
        states = []
        for i in range(4):
            state = TensorBackedAcquisitionState.create_empty(config)
            # Modify state properties for testing (would be done through proper methods)
            states.append(state)
        
        def mock_policy_forward(state):
            """Mock policy network forward pass."""
            # Should be able to access tensor data efficiently
            features = state.mechanism_features  # [n_vars, feature_dim]
            return {
                'variable_logits': jnp.ones(config.n_vars),
                'value_params': jnp.ones((config.n_vars, 2))
            }
        
        # Should work with vmap
        vmapped_policy = jax.vmap(mock_policy_forward)
        policy_outputs = vmapped_policy(states)
        
        assert policy_outputs['variable_logits'].shape == (4, config.n_vars)
        assert policy_outputs['value_params'].shape == (4, config.n_vars, 2)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestGRPOLossComputation:
    """Test GRPO loss computation with tensor-backed states."""
    
    def test_grpo_loss_with_tensor_states(self):
        """Test that GRPO loss computation works with TensorBackedAcquisitionState."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        grpo_config = GRPOConfig(group_size=4, learning_rate=1e-3)
        
        # Create GRPO batch data
        states = [
            TensorBackedAcquisitionState.create_empty(config)
            for _ in range(grpo_config.group_size)
        ]
        
        # Mock interventions and rewards
        actions = [
            create_perfect_intervention(targets={'X0'}, values={'X0': 1.0})
            for _ in range(grpo_config.group_size)
        ]
        
        rewards = jnp.array([0.5, 0.8, 0.3, 0.9])
        old_log_probs = jnp.array([-1.2, -0.8, -1.5, -0.7])
        
        batch_data = {
            'states': states,
            'actions': actions, 
            'rewards': rewards,
            'old_log_probs': old_log_probs
        }
        
        # Mock policy network for testing
        def mock_policy_network(params, state, is_training=False):
            """Mock policy network that works with tensor states."""
            return {
                'variable_logits': jnp.array([2.0, -10.0, 1.0]),  # Avoid target (idx=1)
                'value_params': jnp.array([[0.0, -1.0], [-10.0, -10.0], [1.0, -0.5]])
            }
        
        # Mock parameters
        params = {'mock': 'params'}
        
        # GRPO loss computation should work
        total_loss, loss_info = _compute_grpo_loss(
            params, batch_data, mock_policy_network, grpo_config
        )
        
        assert jnp.isfinite(total_loss)
        assert 'policy_loss' in loss_info
        assert 'entropy_loss' in loss_info
        assert 'mean_reward' in loss_info
    
    def test_grpo_advantage_computation(self):
        """Test GRPO advantage computation with tensor states."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        # Create states with different quality scores
        states = []
        rewards = [0.1, 0.9, 0.4, 0.7]  # Different reward levels
        
        for reward in rewards:
            state = TensorBackedAcquisitionState.create_empty(config)
            # In real implementation, state would reflect the reward
            states.append(state)
        
        rewards_array = jnp.array(rewards)
        
        # Group baseline computation (core GRPO)
        group_baseline = jnp.mean(rewards_array)
        advantages = rewards_array - group_baseline
        
        # Advantages should sum to approximately zero
        assert jnp.abs(jnp.sum(advantages)) < 1e-6
        
        # Higher rewards should have positive advantages
        assert advantages[1] > 0  # reward=0.9 > mean
        assert advantages[0] < 0  # reward=0.1 < mean


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestGRPOBatchCollection:
    """Test GRPO batch collection with tensor states."""
    
    def test_collect_same_state_batch(self):
        """Test collecting GRPO batch using 'Same State, Different Interventions' strategy."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        grpo_config = GRPOConfig(
            group_size=3, 
            interventions_per_state=3,
            learning_rate=1e-3
        )
        
        # Base state for sampling multiple interventions
        base_state = TensorBackedAcquisitionState.create_empty(config)
        
        # Mock the batch collection (would call actual GRPO function)
        def mock_collect_same_state_batch():
            """Mock GRPO batch collection."""
            return {
                'states': [base_state] * grpo_config.interventions_per_state,
                'actions': [
                    create_perfect_intervention(targets={'X0'}, values={'X0': float(i)})
                    for i in range(grpo_config.interventions_per_state)
                ],
                'rewards': jnp.array([0.3, 0.7, 0.5]),
                'old_log_probs': jnp.array([-1.1, -0.9, -1.2])
            }
        
        batch = mock_collect_same_state_batch()
        
        # Validate batch structure
        assert len(batch['states']) == grpo_config.interventions_per_state
        assert len(batch['actions']) == grpo_config.interventions_per_state
        assert batch['rewards'].shape == (grpo_config.interventions_per_state,)
        assert batch['old_log_probs'].shape == (grpo_config.interventions_per_state,)
        
        # All states should be from same base context
        for state in batch['states']:
            assert isinstance(state, TensorBackedAcquisitionState)
            assert state.config == config
    
    def test_reward_computation_with_tensor_states(self):
        """Test reward computation using tensor state interface."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        # Before and after states
        state_before = TensorBackedAcquisitionState.create_empty(config)
        state_after = TensorBackedAcquisitionState.create_empty(config)
        
        # Mock intervention and outcome
        intervention = create_perfect_intervention(targets={'X0'}, values={'X0': 1.5})
        outcome = pyr.m(
            intervention_targets={'X0'},
            values={'X0': 1.5, 'X1': 2.0, 'X2': 3.0}
        )
        
        # Should be able to compute rewards using state interface
        # (This would call the actual reward computation function)
        def mock_compute_reward(state_before, intervention, outcome, state_after):
            """Mock reward computation using state interface."""
            # Access state properties needed for rewards
            _ = state_before.buffer
            _ = state_before.posterior
            _ = state_before.best_value
            _ = state_after.best_value
            _ = state_after.step - state_before.step  # Step progression
            
            return 0.75  # Mock reward
        
        reward = mock_compute_reward(state_before, intervention, outcome, state_after)
        assert isinstance(reward, (int, float))
        assert jnp.isfinite(reward)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestActionLogProbabilityComputation:
    """Test action log probability computation with tensor states."""
    
    def test_log_prob_with_tensor_state(self):
        """Test computing action log probabilities with tensor states."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Mock policy output
        policy_output = {
            'variable_logits': jnp.array([1.0, -10.0, 0.5]),  # Avoid target (idx=1)
            'value_params': jnp.array([[0.0, -1.0], [-10.0, -10.0], [1.0, -0.5]])
        }
        
        # Test intervention
        intervention = create_perfect_intervention(targets={'X0'}, values={'X0': 0.8})
        
        # Should be able to compute log probability
        # (This would call the actual log probability function)
        def mock_compute_log_prob(policy_output, intervention, state):
            """Mock log probability computation."""
            # Should be able to access variable ordering from state
            variable_coverage = state.buffer.get_variable_coverage()
            
            # Variable selection log prob (categorical)
            var_log_probs = jax.nn.log_softmax(policy_output['variable_logits'])
            var_idx = 0  # X0 intervention
            var_log_prob = var_log_probs[var_idx]
            
            # Value selection log prob (Gaussian)
            mean, log_std = policy_output['value_params'][var_idx]
            value = 0.8
            val_log_prob = -0.5 * ((value - mean) / jnp.exp(log_std)) ** 2
            val_log_prob -= 0.5 * jnp.log(2 * jnp.pi) + log_std
            
            return var_log_prob + val_log_prob
        
        log_prob = mock_compute_log_prob(policy_output, intervention, state)
        assert jnp.isfinite(log_prob)
        assert isinstance(log_prob, jnp.ndarray)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestGRPOIntegrationEnd2End:
    """End-to-end integration tests for GRPO with tensor states."""
    
    def test_full_grpo_update_cycle(self):
        """Test complete GRPO update cycle with tensor states."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        grpo_config = GRPOConfig(group_size=4, learning_rate=1e-3)
        
        # Mock components for end-to-end test
        def create_mock_batch():
            """Create mock GRPO batch."""
            states = [
                TensorBackedAcquisitionState.create_empty(config)
                for _ in range(grpo_config.group_size)
            ]
            
            actions = [
                create_perfect_intervention(targets={'X0'}, values={'X0': float(i * 0.5)})
                for i in range(grpo_config.group_size)
            ]
            
            return {
                'states': states,
                'actions': actions,
                'rewards': jnp.array([0.2, 0.8, 0.4, 0.9]),
                'old_log_probs': jnp.array([-1.0, -0.8, -1.2, -0.6])
            }
        
        def mock_policy_network(params, state, is_training=False):
            """Mock policy network."""
            return {
                'variable_logits': jnp.array([1.5, -10.0, 0.8]),
                'value_params': jnp.array([[0.2, -0.8], [-10.0, -10.0], [0.5, -0.6]])
            }
        
        # Create batch and run GRPO update
        batch = create_mock_batch()
        params = {'test': 'params'}
        
        # GRPO loss computation (core test)
        total_loss, loss_info = _compute_grpo_loss(
            params, batch, mock_policy_network, grpo_config
        )
        
        # Validate results
        assert jnp.isfinite(total_loss)
        assert total_loss > 0  # Loss should be positive
        
        # Check loss components
        assert 'policy_loss' in loss_info
        assert 'entropy_loss' in loss_info
        assert 'mean_reward' in loss_info
        assert 'group_baseline' in loss_info
        
        # Group baseline should equal mean reward
        expected_baseline = jnp.mean(batch['rewards'])
        assert jnp.abs(loss_info['group_baseline'] - expected_baseline) < 1e-6
    
    def test_grpo_parameter_updates(self):
        """Test that GRPO actually updates parameters (integration with optax)."""
        import optax
        
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        grpo_config = GRPOConfig(group_size=2, learning_rate=1e-3)
        
        # Mock parameters and optimizer
        params = {'linear': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        optimizer = optax.adam(learning_rate=grpo_config.learning_rate)
        opt_state = optimizer.init(params)
        
        # Mock gradient computation
        def mock_loss_fn(params):
            """Mock loss function that returns gradients."""
            return 2.0, {'policy_loss': 1.5, 'entropy_loss': 0.5}
        
        # Test gradient computation and parameter update
        (loss_value, loss_info), grads = jax.value_and_grad(mock_loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Parameters should change
        param_diff = jnp.mean(jnp.abs(new_params['linear'] - params['linear']))
        assert param_diff > 1e-8  # Parameters should actually update
        
        assert jnp.isfinite(loss_value)
        assert loss_value > 0


if __name__ == "__main__":
    # Run tests that should fail initially (TDD)
    pytest.main([__file__, "-v"])