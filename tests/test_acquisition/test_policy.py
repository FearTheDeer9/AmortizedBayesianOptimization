"""
Comprehensive test suite for Policy Network with Alternating Attention.

Tests all components of the policy network including:
1. AlternatingAttentionEncoder architecture
2. AcquisitionPolicyNetwork functionality
3. Integration with existing infrastructure
4. Edge cases and error handling
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import pyrsistent as pyr
from typing import Dict, Any

from causal_bayes_opt.acquisition.policy import (
    PolicyConfig,
    AlternatingAttentionEncoder,
    AcquisitionPolicyNetwork,
    create_acquisition_policy,
    sample_intervention_from_policy,
    compute_action_log_probability,
    compute_policy_entropy,
    analyze_policy_output,
    validate_policy_output,
)

from causal_bayes_opt.acquisition.state import (
    AcquisitionState,
    create_acquisition_state,
)

from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.avici_integration.parent_set import create_parent_set_posterior
from causal_bayes_opt.interventions.handlers import create_perfect_intervention


class TestPolicyConfig:
    """Test policy configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PolicyConfig()
        
        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.exploration_noise == 0.1
        assert config.variable_selection_temp == 1.0
        assert config.value_selection_temp == 1.0
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        config = PolicyConfig(
            hidden_dim=256,
            num_layers=6,
            num_heads=16,
            dropout=0.2
        )
        
        assert config.hidden_dim == 256
        assert config.num_layers == 6
        assert config.num_heads == 16
        assert config.dropout == 0.2


class TestAlternatingAttentionEncoder:
    """Test the alternating attention encoder architecture."""
    
    @pytest.fixture
    def sample_history(self):
        """Create sample history tensor for testing."""
        # [n_samples=5, n_vars=3, features=3]
        return jnp.array([
            [[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [2.0, 0.0, 1.0]],  # Sample 1
            [[1.5, 0.0, 0.0], [0.3, 0.0, 0.0], [1.8, 1.0, 1.0]],  # Sample 2
            [[0.8, 1.0, 0.0], [0.9, 0.0, 0.0], [2.2, 0.0, 1.0]],  # Sample 3
            [[1.2, 0.0, 0.0], [0.1, 1.0, 0.0], [1.9, 0.0, 1.0]],  # Sample 4
            [[0.9, 0.0, 0.0], [0.7, 0.0, 0.0], [2.1, 1.0, 1.0]],  # Sample 5
        ])
    
    def test_encoder_initialization(self):
        """Test encoder can be initialized with different configurations."""
        import haiku as hk
        
        def encoder_fn(history):
            encoder = AlternatingAttentionEncoder(
                num_layers=2,
                num_heads=4,
                hidden_dim=64,
                dropout=0.1
            )
            return encoder(history, is_training=True)
        
        encoder_transform = hk.transform(encoder_fn)
        
        # Test with sample input
        history = jnp.ones((3, 4, 3))  # [3 samples, 4 vars, 3 features]
        key = jax.random.PRNGKey(42)
        
        params = encoder_transform.init(key, history)
        output = encoder_transform.apply(params, key, history)
        
        # Check output shape: [n_vars, hidden_dim]
        assert output.shape == (4, 64)
        assert jnp.all(jnp.isfinite(output))
    
    def test_encoder_output_shape(self, sample_history):
        """Test encoder produces correct output shape."""
        import haiku as hk
        
        def encoder_fn(history):
            encoder = AlternatingAttentionEncoder(hidden_dim=128)
            return encoder(history, is_training=False)
        
        encoder_transform = hk.transform(encoder_fn)
        key = jax.random.PRNGKey(42)
        
        params = encoder_transform.init(key, sample_history)
        output = encoder_transform.apply(params, key, sample_history)
        
        n_vars = sample_history.shape[1]
        expected_shape = (n_vars, 128)
        
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))
    
    def test_encoder_training_vs_inference(self, sample_history):
        """Test encoder behaves differently in training vs inference mode."""
        import haiku as hk
        
        def encoder_fn(history, is_training):
            encoder = AlternatingAttentionEncoder(dropout=0.5)  # High dropout for testing
            return encoder(history, is_training=is_training)
        
        encoder_transform = hk.transform(encoder_fn)
        key = jax.random.PRNGKey(42)
        
        params = encoder_transform.init(key, sample_history, True)
        
        # Training mode (with dropout)
        output_train = encoder_transform.apply(params, key, sample_history, True)
        
        # Inference mode (no dropout)
        output_inference = encoder_transform.apply(params, key, sample_history, False)
        
        # Outputs should be different due to dropout
        assert not jnp.allclose(output_train, output_inference, atol=1e-6)
        assert jnp.all(jnp.isfinite(output_train))
        assert jnp.all(jnp.isfinite(output_inference))


class TestAcquisitionPolicyNetwork:
    """Test the complete acquisition policy network."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock acquisition state for testing."""
        # Create mock buffer
        buffer = ExperienceBuffer()
        
        # Add some observational samples
        for i in range(3):
            sample = create_sample(
                values={'X': float(i), 'Y': float(i + 1), 'Z': float(i + 2)}
            )
            buffer.add_observation(sample)
        
        # Add some interventional samples
        for i in range(2):
            intervention = create_perfect_intervention(
                targets=frozenset(['X']),
                values={'X': float(i + 10)}
            )
            outcome = create_sample(
            values={'X': float(i + 10), 'Y': float(i + 5), 'Z': float(i + 7)},
            intervention_type='perfect',
                intervention_targets=frozenset(['X'])
        )
            buffer.add_intervention(intervention, outcome)
        
        # Create mock posterior
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset(), frozenset(['X']), frozenset(['Z'])],
            probabilities=jnp.array([0.2, 0.6, 0.2]),
            metadata={'test': True}
        )
        
        return AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=5.0,
            current_target='Y',
            step=10,
            metadata=pyr.m(**{'test_state': True})
        )
    
    def test_policy_network_initialization(self, mock_state):
        """Test policy network can be initialized and run."""
        config = PolicyConfig(hidden_dim=64, num_layers=2)
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, True)
        output = policy_transform.apply(params, key, mock_state, True)
        
        # Check output structure
        assert isinstance(output, dict)
        assert 'variable_logits' in output
        assert 'value_params' in output
        assert 'state_value' in output
        
        # Check shapes
        n_vars = len(mock_state.buffer.get_variable_coverage())
        assert output['variable_logits'].shape == (n_vars,)
        assert output['value_params'].shape == (n_vars, 2)
        assert output['state_value'].shape == ()
        
        # Check that non-target variables have finite logits
        variable_order = sorted(mock_state.buffer.get_variable_coverage())
        target_idx = variable_order.index(mock_state.current_target)
        
        # Target variable should be -inf, others should be finite
        assert output['variable_logits'][target_idx] == -jnp.inf
        non_target_mask = jnp.arange(len(variable_order)) != target_idx
        assert jnp.all(jnp.isfinite(output['variable_logits'][non_target_mask]))
        assert jnp.all(jnp.isfinite(output['value_params']))
        assert jnp.isfinite(output['state_value'])
    
    def test_policy_output_validation(self, mock_state):
        """Test policy output validation function."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        # Valid output should pass validation
        assert validate_policy_output(output, mock_state)
        
        # Invalid outputs should fail validation  
        invalid_output1 = {**output, 'variable_logits': jnp.array([1.0, jnp.nan, 2.0])}
        assert not validate_policy_output(invalid_output1, mock_state)
        
        invalid_output2 = {k: v for k, v in output.items() if k != 'state_value'}
        assert not validate_policy_output(invalid_output2, mock_state)
    
    def test_target_variable_masking(self, mock_state):
        """Test that target variable is properly masked from intervention selection."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        variable_order = sorted(mock_state.buffer.get_variable_coverage())
        target_idx = variable_order.index(mock_state.current_target)
        
        # Target variable should have very low logit (masked with -inf)
        assert output['variable_logits'][target_idx] == -jnp.inf
    
    def test_intervention_sampling(self, mock_state):
        """Test intervention sampling from policy output."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        # Sample intervention
        sample_key = jax.random.PRNGKey(123)
        intervention = sample_intervention_from_policy(output, mock_state, sample_key, config)
        
        # Check intervention structure
        assert isinstance(intervention, pyr.PMap)
        assert 'type' in intervention
        assert 'targets' in intervention
        assert 'values' in intervention
        
        # Check intervention validity
        assert intervention['type'] == 'perfect'
        assert len(intervention['targets']) == 1
        target_var = list(intervention['targets'])[0]
        assert target_var != mock_state.current_target  # Shouldn't target the optimization target
        assert target_var in intervention['values']
        assert isinstance(intervention['values'][target_var], float)
    
    def test_log_probability_computation(self, mock_state):
        """Test log probability computation for actions."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        # Create a test intervention
        intervention = create_perfect_intervention(
            targets=frozenset(['X']),
            values={'X': 1.5}
        )
        
        # Compute log probability
        log_prob = compute_action_log_probability(output, intervention, mock_state, config)
        
        assert jnp.isfinite(log_prob)
        assert isinstance(log_prob.item(), float)
    
    def test_entropy_computation(self, mock_state):
        """Test policy entropy computation."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        entropy = compute_policy_entropy(output, config)
        
        assert jnp.isfinite(entropy)
        assert entropy > 0  # Entropy should be positive
    
    def test_policy_analysis(self, mock_state):
        """Test policy output analysis functionality."""
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, mock_state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, mock_state, False)
        output = policy_transform.apply(params, key, mock_state, False)
        
        analysis = analyze_policy_output(output, mock_state, top_k=2)
        
        # Check analysis structure
        assert 'state_value_estimate' in analysis
        assert 'top_variables' in analysis
        assert 'variable_selection_entropy' in analysis
        assert 'state_context' in analysis
        
        # Check top variables
        assert len(analysis['top_variables']) == 2
        for var_info in analysis['top_variables']:
            assert 'variable' in var_info
            assert 'logit' in var_info
            assert 'intervention_mean' in var_info
            assert 'intervention_std' in var_info


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""
    
    def test_empty_buffer_handling(self):
        """Test policy behavior with minimal data."""
        # Create minimal buffer with just one sample
        buffer = ExperienceBuffer()
        sample = create_sample(
        values={'X': 1.0, 'Y': 2.0}
        )
        buffer.add_observation(sample)
        
        # Create minimal posterior
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset()],
            probabilities=jnp.array([1.0]),
            metadata={'minimal': True}
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=0,
            metadata=pyr.m()
        )
        
        # Test policy can handle minimal state
        config = PolicyConfig(hidden_dim=32, num_layers=1)
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
    
    def test_single_variable_state(self):
        """Test policy behavior with only target variable."""
        buffer = ExperienceBuffer()
        sample = create_sample(
        values={'Y': 2.0}  # Only target variable
        )
        buffer.add_observation(sample)
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=[frozenset()],
            probabilities=jnp.array([1.0]),
            metadata={'single_var': True}
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=2.0,
            current_target='Y',
            step=0,
            metadata=pyr.m()
        )
        
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        # Should handle gracefully even with no valid intervention targets
        assert validate_policy_output(output, state)
        
        # All variable logits should be -inf (only target variable exists)
        assert jnp.all(output['variable_logits'] == -jnp.inf)
    
    def test_large_state_scalability(self):
        """Test policy scales to larger state spaces."""
        # Create larger buffer with many variables
        buffer = ExperienceBuffer()
        n_vars = 10
        var_names = [f'X{i}' for i in range(n_vars)] + ['Y']  # Y is target
        
        # Add observational samples
        for i in range(20):
            values = {var: float(i + j) for j, var in enumerate(var_names)}
            sample = create_sample(values=values)
            buffer.add_observation(sample)
        
        # Add interventional samples
        for i in range(10):
            target_var = f'X{i % n_vars}'
            intervention = create_perfect_intervention(
                targets=frozenset([target_var]),
                values={target_var: float(i + 100)}
            )
            values = {var: float(i + j + 50) for j, var in enumerate(var_names)}
            values[target_var] = float(i + 100)  # Intervention value
            outcome = create_sample(
                values=values, 
                intervention_type='perfect',
                intervention_targets=frozenset([target_var])
            )
            buffer.add_intervention(intervention, outcome)
        
        # Create posterior with multiple parent sets
        parent_sets = [frozenset(), frozenset(['X0']), frozenset(['X1']), frozenset(['X0', 'X1'])]
        probs = jnp.array([0.25, 0.35, 0.25, 0.15])
        
        posterior = create_parent_set_posterior(
            target_variable='Y',
            parent_sets=parent_sets,
            probabilities=probs,
            metadata={'large_state': True}
        )
        
        state = AcquisitionState(
            posterior=posterior,
            buffer=buffer,
            best_value=100.0,
            current_target='Y',
            step=25,
            metadata=pyr.m()
        )
        
        # Test policy handles large state
        config = PolicyConfig()
        policy_transform = create_acquisition_policy(config, state)
        
        key = jax.random.PRNGKey(42)
        params = policy_transform.init(key, state, False)
        output = policy_transform.apply(params, key, state, False)
        
        assert validate_policy_output(output, state)
        assert output['variable_logits'].shape[0] == len(var_names)  # All variables


def test_policy_determinism():
    """Test that policy is deterministic given same inputs and random key."""
    # This test ensures reproducibility
    
    # Create consistent test state
    buffer = ExperienceBuffer()
    for i in range(3):
        sample = create_sample(
            values={'X': float(i), 'Y': float(i + 1), 'Z': float(i + 2)}
        )
        buffer.add_observation(sample)
    
    posterior = create_parent_set_posterior(
        target_variable='Y',
        parent_sets=[frozenset(['X'])],
        probabilities=jnp.array([1.0]),
        metadata={}
    )
    
    state = AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=3.0,
        current_target='Y',
        step=5,
        metadata=pyr.m()
    )
    
    config = PolicyConfig()
    policy_transform = create_acquisition_policy(config, state)
    
    # Same key should produce same results
    key = jax.random.PRNGKey(42)
    
    # Initialize once
    params = policy_transform.init(key, state, False)
    
    # Run multiple times with same key
    output1 = policy_transform.apply(params, key, state, False)
    output2 = policy_transform.apply(params, key, state, False)
    
    # Should be identical
    assert jnp.allclose(output1['variable_logits'], output2['variable_logits'])
    assert jnp.allclose(output1['value_params'], output2['value_params'])
    assert jnp.allclose(output1['state_value'], output2['state_value'])


def test_error_handling():
    """Test error handling in policy components."""
    
    # Test validation with malformed inputs
    mock_output = {
        'variable_logits': jnp.array([1.0, 2.0]),
        'value_params': jnp.array([[1.0, 0.5], [2.0, 0.3]]),
        'state_value': jnp.array(1.5)
    }
    
    # Create minimal valid state for validation
    buffer = ExperienceBuffer()
    sample = create_sample(values={'X': 1.0, 'Y': 2.0})
    buffer.add_observation(sample)
    
    posterior = create_parent_set_posterior(
        target_variable='Y',
        parent_sets=[frozenset()],
        probabilities=jnp.array([1.0]),
        metadata={}
    )
    
    state = AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=2.0,
        current_target='Y',
        step=0,
        metadata=pyr.m()
    )
    
    # Should pass validation
    assert validate_policy_output(mock_output, state)
    
    # Test with NaN values
    invalid_output = {**mock_output, 'state_value': jnp.array(jnp.nan)}
    assert not validate_policy_output(invalid_output, state)


if __name__ == "__main__":
    # Quick smoke test
    print("Running basic policy network smoke test...")
    
    # Create minimal test case
    buffer = ExperienceBuffer()
    sample = create_sample(
        values={'X': 1.0, 'Y': 2.0, 'Z': 3.0}
    )
    buffer.add_observation(sample)
    
    posterior = create_parent_set_posterior(
        target_variable='Y',
        parent_sets=[frozenset(['X'])],
        probabilities=jnp.array([1.0]),
        metadata={}
    )
    
    state = AcquisitionState(
        posterior=posterior,
        buffer=buffer,
        best_value=2.0,
        current_target='Y',
        step=0,
        metadata=pyr.m()
    )
    
    # Test policy creation and execution
    config = PolicyConfig()
    policy_transform = create_acquisition_policy(config, state)
    
    key = jax.random.PRNGKey(42)
    params = policy_transform.init(key, state, False)
    output = policy_transform.apply(params, key, state, False)
    
    print(f"✓ Policy output shape check: {output['variable_logits'].shape}")
    print(f"✓ Validation passed: {validate_policy_output(output, state)}")
    
    # Check that target variable is masked and others are finite
    variable_order = sorted(state.buffer.get_variable_coverage())
    target_idx = variable_order.index(state.current_target)
    non_target_mask = jnp.arange(len(variable_order)) != target_idx
    print(f"✓ Target variable masked: {output['variable_logits'][target_idx] == -jnp.inf}")
    print(f"✓ Non-target variables finite: {jnp.all(jnp.isfinite(output['variable_logits'][non_target_mask]))}")
    
    # Test intervention sampling
    intervention = sample_intervention_from_policy(output, state, key, config)
    print(f"✓ Intervention sampled: {intervention['type']} on {list(intervention['targets'])}")
    
    print("\nBasic smoke test passed! ✓")
