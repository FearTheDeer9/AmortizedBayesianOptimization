"""
Simple GRPO Validation Tests with Basic Networks.

These tests verify that the GRPO training loop works correctly with simple
policy networks before testing with enhanced architectures. This establishes
a baseline that the core training logic is sound.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
from unittest.mock import Mock, patch
import pyrsistent as pyr

from src.causal_bayes_opt.training.grpo_core import create_grpo_update_fn
from src.causal_bayes_opt.training.reward_component_validation import (
    create_component_test_configs, validate_all_components
)
from src.causal_bayes_opt.acquisition.rewards import create_default_reward_config


class SimpleTestPolicyNetwork(hk.Module):
    """Simple policy network for testing GRPO training."""
    
    def __init__(self, num_variables: int = 4, hidden_dim: int = 32, name: str = "simple_policy"):
        super().__init__(name=name)
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
    
    def __call__(self, state_tensor: jnp.ndarray, is_training: bool = False) -> dict:
        """
        Simple policy forward pass.
        
        Args:
            state_tensor: [T, n_vars, channels] state representation
            is_training: Whether in training mode
            
        Returns:
            Dictionary with policy outputs
        """
        # Simple processing: take mean across time and channels
        current_state = jnp.mean(state_tensor, axis=(0, 2))  # [n_vars]
        
        # Simple MLP
        x = hk.Linear(self.hidden_dim, name="hidden")(current_state)
        x = jax.nn.relu(x)
        
        # Policy head: variable selection logits
        variable_logits = hk.Linear(self.num_variables, name="variable_logits")(x)
        
        # Value head: state value estimation
        value_estimate = hk.Linear(1, name="value")(x)
        value_estimate = jnp.squeeze(value_estimate)
        
        # Dummy value parameters for intervention values
        value_params = hk.Linear(self.num_variables * 2, name="value_params")(x)
        value_params = jnp.reshape(value_params, (self.num_variables, 2))  # [n_vars, 2] for mean/std
        
        return {
            'intervention_logits': variable_logits,
            'value_params': value_params,
            'value_estimate': value_estimate
        }


@pytest.fixture
def simple_policy_function():
    """Create a simple policy function for testing."""
    def policy_fn(state_tensor: jnp.ndarray, is_training: bool = False):
        network = SimpleTestPolicyNetwork(num_variables=4)
        return network(state_tensor, is_training=is_training)
    
    return hk.transform(policy_fn)


@pytest.fixture
def simple_grpo_config():
    """Create simple GRPO configuration for testing."""
    return type('GRPOConfig', (), {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_loss_coeff': 0.5,
        'entropy_coeff': 0.01,
        'max_grad_norm': 1.0,
        'group_size': 4
    })()


@pytest.fixture
def mock_experience_batch():
    """Create mock experience batch for testing."""
    batch_size = 8
    n_vars = 4
    max_history = 10
    channels = 5
    
    return {
        'states': jnp.zeros((batch_size, max_history, n_vars, channels)),
        'actions': jnp.zeros((batch_size, n_vars)),  # Variable selection
        'rewards': jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        'dones': jnp.array([False] * (batch_size - 1) + [True]),
        'values': jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        'log_probs': jnp.array([-0.5] * batch_size)
    }


class TestSimpleGRPOValidation:
    """Test suite for simple GRPO validation."""
    
    def test_simple_policy_network_forward_pass(self, simple_policy_function):
        """Test that simple policy network forward pass works."""
        key = jax.random.PRNGKey(42)
        
        # Create dummy input
        batch_size = 2
        max_history = 10
        n_vars = 4
        channels = 5
        dummy_input = jnp.zeros((max_history, n_vars, channels))
        
        # Initialize and run forward pass
        params = simple_policy_function.init(key, dummy_input)
        outputs = simple_policy_function.apply(params, key, dummy_input)
        
        # Check output structure
        assert 'intervention_logits' in outputs
        assert 'value_params' in outputs  
        assert 'value_estimate' in outputs
        
        # Check output shapes
        assert outputs['intervention_logits'].shape == (n_vars,)
        assert outputs['value_params'].shape == (n_vars, 2)
        assert outputs['value_estimate'].shape == ()
        
        # Check values are finite
        assert jnp.all(jnp.isfinite(outputs['intervention_logits']))
        assert jnp.all(jnp.isfinite(outputs['value_params']))
        assert jnp.isfinite(outputs['value_estimate'])
    
    def test_grpo_update_function_creation(self, simple_policy_function, simple_grpo_config):
        """Test that GRPO update function can be created with simple policy."""
        optimizer = optax.adam(learning_rate=simple_grpo_config.learning_rate)
        
        # Create GRPO update function
        grpo_update_fn = create_grpo_update_fn(
            policy_fn=simple_policy_function,
            optimizer=optimizer,
            config=simple_grpo_config
        )
        
        assert callable(grpo_update_fn)
    
    def test_grpo_update_step(self, simple_policy_function, simple_grpo_config, mock_experience_batch):
        """Test that GRPO update step runs without errors."""
        key = jax.random.PRNGKey(42)
        optimizer = optax.adam(learning_rate=simple_grpo_config.learning_rate)
        
        # Initialize policy parameters
        dummy_state = jnp.zeros((10, 4, 5))
        params = simple_policy_function.init(key, dummy_state)
        opt_state = optimizer.init(params)
        
        # Create GRPO update function
        grpo_update_fn = create_grpo_update_fn(
            policy_fn=simple_policy_function,
            optimizer=optimizer,
            config=simple_grpo_config
        )
        
        # Run update step
        update_result = grpo_update_fn(params, opt_state, mock_experience_batch)
        
        # Check that update result has expected structure
        assert hasattr(update_result, 'params') or 'params' in update_result
        assert hasattr(update_result, 'opt_state') or 'opt_state' in update_result
        
        # Check that parameters changed (learning occurred)
        if hasattr(update_result, 'params'):
            new_params = update_result.params
        else:
            new_params = update_result['params']
        
        # At least some parameters should have changed
        param_changed = False
        for key in params:
            if not jnp.allclose(params[key], new_params[key], atol=1e-6):
                param_changed = True
                break
        
        assert param_changed, "GRPO update should change policy parameters"
    
    def test_grpo_learning_progression(self, simple_policy_function, simple_grpo_config):
        """Test that GRPO shows learning progression over multiple steps."""
        key = jax.random.PRNGKey(42)
        optimizer = optax.adam(learning_rate=simple_grpo_config.learning_rate)
        
        # Initialize
        dummy_state = jnp.zeros((10, 4, 5))
        params = simple_policy_function.init(key, dummy_state)
        opt_state = optimizer.init(params)
        
        # Create GRPO update function
        grpo_update_fn = create_grpo_update_fn(
            policy_fn=simple_policy_function,
            optimizer=optimizer,
            config=simple_grpo_config
        )
        
        # Create progressive experience batches with increasing rewards
        def create_experience_batch(step):
            batch_size = 8
            base_reward = 0.1 * step  # Increasing rewards
            return {
                'states': jnp.zeros((batch_size, 10, 4, 5)),
                'actions': jnp.zeros((batch_size, 4)),
                'rewards': jnp.array([base_reward + 0.1 * i for i in range(batch_size)]),
                'dones': jnp.array([False] * (batch_size - 1) + [True]),
                'values': jnp.array([base_reward - 0.05 + 0.1 * i for i in range(batch_size)]),
                'log_probs': jnp.array([-0.5] * batch_size)
            }
        
        # Run multiple update steps
        policy_losses = []
        value_losses = []
        
        for step in range(5):
            batch = create_experience_batch(step)
            update_result = grpo_update_fn(params, opt_state, batch)
            
            # Extract new parameters and optimizer state
            if hasattr(update_result, 'params'):
                params = update_result.params
                opt_state = update_result.opt_state
                policy_loss = getattr(update_result, 'policy_loss', 0.0)
                value_loss = getattr(update_result, 'value_loss', 0.0)
            else:
                params = update_result['params']
                opt_state = update_result['opt_state']
                policy_loss = update_result.get('policy_loss', 0.0)
                value_loss = update_result.get('value_loss', 0.0)
            
            policy_losses.append(float(policy_loss))
            value_losses.append(float(value_loss))
        
        # Check that learning occurred (losses should be finite and meaningful)
        assert all(jnp.isfinite(loss) for loss in policy_losses), "Policy losses should be finite"
        assert all(jnp.isfinite(loss) for loss in value_losses), "Value losses should be finite"
        
        # At least some variation in losses (not all identical)
        policy_variance = jnp.var(jnp.array(policy_losses))
        assert policy_variance > 1e-8, "Policy losses should show some variation during learning"
    
    def test_grpo_with_different_reward_configs(self, simple_policy_function, simple_grpo_config):
        """Test GRPO behavior with different reward component configurations."""
        key = jax.random.PRNGKey(42)
        optimizer = optax.adam(learning_rate=simple_grpo_config.learning_rate)
        
        # Get test configurations
        test_configs = create_component_test_configs()
        
        # Initialize policy
        dummy_state = jnp.zeros((10, 4, 5))
        params = simple_policy_function.init(key, dummy_state)
        
        grpo_update_fn = create_grpo_update_fn(
            policy_fn=simple_policy_function,
            optimizer=optimizer,
            config=simple_grpo_config
        )
        
        # Test with optimization-only vs structure-only configurations
        opt_only_batch = {
            'states': jnp.zeros((4, 10, 4, 5)),
            'actions': jnp.zeros((4, 4)),
            'rewards': jnp.array([1.0, 0.5, 0.8, 0.3]),  # High optimization rewards
            'dones': jnp.array([False, False, False, True]),
            'values': jnp.array([0.9, 0.4, 0.7, 0.2]),
            'log_probs': jnp.array([-0.3, -0.5, -0.4, -0.6])
        }
        
        struct_only_batch = {
            'states': jnp.zeros((4, 10, 4, 5)),
            'actions': jnp.zeros((4, 4)),
            'rewards': jnp.array([0.2, 0.8, 0.1, 0.6]),  # Different structure rewards
            'dones': jnp.array([False, False, False, True]),
            'values': jnp.array([0.1, 0.7, 0.0, 0.5]),
            'log_probs': jnp.array([-0.4, -0.3, -0.7, -0.2])
        }
        
        # Run updates with different batches
        opt_state = optimizer.init(params)
        
        opt_result = grpo_update_fn(params, opt_state, opt_only_batch)
        struct_result = grpo_update_fn(params, opt_state, struct_only_batch)
        
        # Both should complete without errors
        assert opt_result is not None
        assert struct_result is not None
        
        # Parameter updates should be different (different reward signals)
        if hasattr(opt_result, 'params'):
            opt_params = opt_result.params
            struct_params = struct_result.params
        else:
            opt_params = opt_result['params']
            struct_params = struct_result['params']
        
        # Check that updates produced different results
        params_differ = False
        for key in opt_params:
            if not jnp.allclose(opt_params[key], struct_params[key], atol=1e-5):
                params_differ = True
                break
        
        assert params_differ, "Different reward signals should produce different parameter updates"
    
    def test_grpo_handles_zero_rewards(self, simple_policy_function, simple_grpo_config):
        """Test that GRPO handles zero/negative rewards gracefully."""
        key = jax.random.PRNGKey(42)
        optimizer = optax.adam(learning_rate=simple_grpo_config.learning_rate)
        
        dummy_state = jnp.zeros((10, 4, 5))
        params = simple_policy_function.init(key, dummy_state)
        opt_state = optimizer.init(params)
        
        grpo_update_fn = create_grpo_update_fn(
            policy_fn=simple_policy_function,
            optimizer=optimizer,
            config=simple_grpo_config
        )
        
        # Create batch with zero/negative rewards
        zero_reward_batch = {
            'states': jnp.zeros((4, 10, 4, 5)),
            'actions': jnp.zeros((4, 4)),
            'rewards': jnp.array([0.0, 0.0, -0.1, -0.2]),
            'dones': jnp.array([False, False, False, True]),
            'values': jnp.array([0.1, 0.0, -0.1, -0.2]),
            'log_probs': jnp.array([-0.5, -0.5, -0.5, -0.5])
        }
        
        # Should not crash with zero/negative rewards
        result = grpo_update_fn(params, opt_state, zero_reward_batch)
        assert result is not None
        
        # Check that parameters are still finite
        if hasattr(result, 'params'):
            new_params = result.params
        else:
            new_params = result['params']
        
        for param_key, param_value in new_params.items():
            assert jnp.all(jnp.isfinite(param_value)), f"Parameter {param_key} should remain finite"


if __name__ == "__main__":
    pytest.main([__file__])