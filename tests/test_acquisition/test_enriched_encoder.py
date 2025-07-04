"""
Tests for EnrichedTransformerEncoder architecture.

This module tests the new enriched context architecture that replaces
the flawed post-transformer feature concatenation approach.

Key components tested:
1. ContextEnrichmentBuilder: Converts AcquisitionState to enriched tensor input
2. EnrichedTransformerEncoder: Processes 10-channel enriched input
3. EnrichedAcquisitionPolicyNetwork: Complete policy with enriched encoder
4. Integration with existing AcquisitionState interface
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import pyrsistent as pyr
from hypothesis import given, strategies as st
from typing import Dict, Any

# Test imports - these will be implemented
from causal_bayes_opt.acquisition.enriched_encoder import (
    EnrichedTransformerEncoder,
    ContextEnrichmentBuilder, 
    EnrichedAcquisitionPolicyNetwork
)
from causal_bayes_opt.acquisition.context_enrichment import (
    build_enriched_history,
    ENRICHED_CHANNEL_COUNT,
    EnrichmentConfig
)

# Existing imports for test setup
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.acquisition.policy import PolicyConfig
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.scm import create_simple_scm
from tests.conftest import create_mock_acquisition_state


class TestContextEnrichmentBuilder:
    """Test context enrichment functionality."""
    
    def test_enriched_history_shape(self):
        """Test that enriched history has correct shape."""
        # Setup
        state = create_mock_acquisition_state(n_vars=5, buffer_size=20)
        builder = ContextEnrichmentBuilder(max_history_size=100)
        
        # Action
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Assertions
        assert enriched_history.shape == (100, 5, ENRICHED_CHANNEL_COUNT)
        assert ENRICHED_CHANNEL_COUNT == 10  # As per design
        assert jnp.all(jnp.isfinite(enriched_history))
    
    def test_enriched_history_channels(self):
        """Test that each channel contains expected information."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=10)
        builder = ContextEnrichmentBuilder(max_history_size=50)
        
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Channel 0: standardized values - should have reasonable range
        values_channel = enriched_history[:, :, 0]
        assert jnp.all(jnp.abs(values_channel) <= 10.0)  # Reasonable standardized range
        
        # Channel 1: intervention indicators - should be 0 or 1  
        intervention_channel = enriched_history[:, :, 1]
        assert jnp.all((intervention_channel == 0) | (intervention_channel == 1))
        
        # Channel 2: target indicators - should be 0 or 1
        target_channel = enriched_history[:, :, 2]
        assert jnp.all((target_channel == 0) | (target_channel == 1))
        
        # Channels 3-9: contextual information - should be finite
        for channel_idx in range(3, 10):
            context_channel = enriched_history[:, :, channel_idx]
            assert jnp.all(jnp.isfinite(context_channel))
    
    @given(
        n_vars=st.integers(min_value=2, max_value=10),
        buffer_size=st.integers(min_value=5, max_value=50),
        max_history_size=st.integers(min_value=20, max_value=200)
    )
    def test_enriched_history_property_based(self, n_vars, buffer_size, max_history_size):
        """Property-based test for enriched history generation."""
        state = create_mock_acquisition_state(n_vars=n_vars, buffer_size=buffer_size)
        builder = ContextEnrichmentBuilder(max_history_size=max_history_size)
        
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Properties that should always hold
        assert enriched_history.shape == (max_history_size, n_vars, 10)
        assert jnp.all(jnp.isfinite(enriched_history))
        
        # Intervention and target indicators should be binary
        assert jnp.all((enriched_history[:, :, 1] == 0) | (enriched_history[:, :, 1] == 1))
        assert jnp.all((enriched_history[:, :, 2] == 0) | (enriched_history[:, :, 2] == 1))
    
    def test_empty_buffer_handling(self):
        """Test graceful handling of empty buffers."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=0)
        builder = ContextEnrichmentBuilder(max_history_size=50)
        
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Should return zeros with correct shape
        assert enriched_history.shape == (50, 3, 10)
        assert jnp.allclose(enriched_history, 0.0)
    
    def test_temporal_ordering_preserved(self):
        """Test that temporal ordering is preserved in enriched history."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=20)
        builder = ContextEnrichmentBuilder(max_history_size=50)
        
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Most recent samples should be at the end (higher indices)
        # This tests that temporal structure is maintained
        values_channel = enriched_history[:, :, 0]
        
        # Should have non-zero values in recent timesteps
        recent_timesteps = values_channel[-20:, :, :]  # Last 20 timesteps
        assert not jnp.allclose(recent_timesteps, 0.0)


class TestEnrichedTransformerEncoder:
    """Test the enriched transformer encoder."""
    
    def test_encoder_initialization(self):
        """Test encoder can be initialized with enriched input."""
        def encoder_fn(enriched_history):
            encoder = EnrichedTransformerEncoder(
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                dropout=0.1
            )
            return encoder(enriched_history, is_training=False)
        
        transform = hk.transform(encoder_fn)
        
        # Test with enriched input shape
        enriched_history = jnp.zeros((50, 5, 10))  # [history, vars, enriched_channels]
        key = random.PRNGKey(42)
        
        params = transform.init(key, enriched_history)
        output = transform.apply(params, key, enriched_history)
        
        # Should output variable-level embeddings
        assert output.shape == (5, 64)  # [n_vars, hidden_dim]
        assert jnp.all(jnp.isfinite(output))
    
    def test_encoder_handles_different_input_sizes(self):
        """Test encoder handles different variable counts."""
        def encoder_fn(enriched_history):
            encoder = EnrichedTransformerEncoder(hidden_dim=32)
            return encoder(enriched_history, is_training=False)
        
        transform = hk.transform(encoder_fn)
        key = random.PRNGKey(42)
        
        # Test with different variable counts
        for n_vars in [2, 5, 10]:
            enriched_history = jnp.zeros((50, n_vars, 10))
            params = transform.init(key, enriched_history)
            output = transform.apply(params, key, enriched_history)
            
            assert output.shape == (n_vars, 32)
            assert jnp.all(jnp.isfinite(output))
    
    def test_encoder_training_vs_inference(self):
        """Test encoder behaves differently in training vs inference."""
        def encoder_fn(enriched_history, is_training):
            encoder = EnrichedTransformerEncoder(dropout=0.5)  # High dropout for testing
            return encoder(enriched_history, is_training=is_training)
        
        transform = hk.transform(encoder_fn)
        enriched_history = jnp.ones((50, 5, 10))
        key = random.PRNGKey(42)
        
        params = transform.init(key, enriched_history, is_training=True)
        
        # Training and inference should give different outputs due to dropout
        train_output = transform.apply(params, key, enriched_history, is_training=True)
        inference_output = transform.apply(params, key, enriched_history, is_training=False)
        
        # Outputs should be different due to dropout
        assert not jnp.allclose(train_output, inference_output, rtol=1e-6)
    
    def test_encoder_attention_over_enriched_channels(self):
        """Test that encoder can attend over enriched channel information."""
        def encoder_fn(enriched_history):
            encoder = EnrichedTransformerEncoder(
                hidden_dim=128,
                num_layers=3,
                num_heads=8
            )
            return encoder(enriched_history, is_training=False)
        
        transform = hk.transform(encoder_fn)
        key = random.PRNGKey(42)
        
        # Create input with structured pattern across channels
        enriched_history = jnp.zeros((50, 5, 10))
        
        # Channel 0: Linear increasing pattern
        enriched_history = enriched_history.at[:, :, 0].set(
            jnp.linspace(0, 1, 50)[:, None]
        )
        
        # Channel 3: Step function pattern  
        enriched_history = enriched_history.at[:, :, 3].set(
            (jnp.arange(50) > 25)[:, None].astype(jnp.float32)
        )
        
        params = transform.init(key, enriched_history)
        output = transform.apply(params, key, enriched_history)
        
        # Output should capture patterns across channels
        assert output.shape == (5, 128)
        assert jnp.all(jnp.isfinite(output))
        
        # Different variables should have different representations
        # (since they get different channel patterns)
        var_similarities = jnp.corrcoef(output)
        assert not jnp.allclose(var_similarities, 1.0)


class TestEnrichedAcquisitionPolicyNetwork:
    """Test the complete enriched policy network."""
    
    def test_policy_network_initialization(self):
        """Test enriched policy network can be initialized."""
        state = create_mock_acquisition_state(n_vars=5, buffer_size=20)
        config = PolicyConfig(hidden_dim=64, num_layers=2)
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        
        params = transform.init(key, state)
        output = transform.apply(params, key, state)
        
        # Should have same output format as original policy
        assert 'variable_logits' in output
        assert 'value_params' in output
        assert 'state_value' in output
        
        n_vars = len(state.buffer.get_variable_coverage())
        assert output['variable_logits'].shape == (n_vars,)
        assert output['value_params'].shape == (n_vars, 2)
        assert output['state_value'].shape == ()
    
    def test_policy_eliminates_feature_concatenation(self):
        """Test that new policy doesn't use hand-crafted feature concatenation."""
        state = create_mock_acquisition_state(n_vars=5, buffer_size=20)
        config = PolicyConfig(hidden_dim=64)
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        
        params = transform.init(key, state)
        
        # The new policy should have fewer parameters than the old one
        # since it doesn't need feature concatenation MLPs
        total_params = sum(p.size for p in jax.tree_leaves(params))
        
        # Should be reasonable number of parameters (not bloated by feature engineering)
        assert total_params < 1_000_000  # Reasonable upper bound
        assert total_params > 10_000     # Reasonable lower bound
    
    def test_policy_output_validity(self):
        """Test that policy outputs are valid and well-formed."""
        state = create_mock_acquisition_state(n_vars=4, buffer_size=15)
        config = PolicyConfig()
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        
        params = transform.init(key, state)
        output = transform.apply(params, key, state)
        
        # Variable logits should be finite (except for target masking)
        variable_logits = output['variable_logits']
        finite_logits = jnp.isfinite(variable_logits)
        assert jnp.sum(finite_logits) >= 1  # At least one non-target variable
        
        # Value parameters should be finite
        value_params = output['value_params']
        assert jnp.all(jnp.isfinite(value_params))
        
        # State value should be finite scalar
        state_value = output['state_value']
        assert jnp.isfinite(state_value)
        assert state_value.ndim == 0
    
    def test_target_variable_masking_preserved(self):
        """Test that target variable masking still works with enriched encoder."""
        state = create_mock_acquisition_state(n_vars=4, buffer_size=15)
        target_var = list(state.buffer.get_variable_coverage())[0]
        state = state.set('current_target', target_var)
        
        config = PolicyConfig()
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        
        params = transform.init(key, state)
        output = transform.apply(params, key, state)
        
        variable_logits = output['variable_logits']
        variable_order = sorted(state.buffer.get_variable_coverage())
        target_idx = variable_order.index(target_var)
        
        # Target variable should be masked (-inf)
        assert variable_logits[target_idx] == -jnp.inf
        
        # Other variables should have finite logits
        other_indices = [i for i in range(len(variable_order)) if i != target_idx]
        assert jnp.all(jnp.isfinite(variable_logits[other_indices]))


class TestTemporalIntegration:
    """Test temporal integration capabilities of enriched encoder."""
    
    def test_encoder_learns_temporal_patterns(self):
        """Test that encoder can learn from temporal evolution of context."""
        # Create state with temporal pattern in uncertainty
        state = create_mock_acquisition_state(n_vars=3, buffer_size=30)
        
        # Modify state to have evolving uncertainty pattern
        evolving_probs = {}
        variables = sorted(state.buffer.get_variable_coverage())
        for i, var in enumerate(variables):
            # Uncertainty increases over time for variable 0, decreases for others
            if i == 0:
                evolving_probs[var] = 0.8  # High uncertainty
            else:
                evolving_probs[var] = 0.2  # Low uncertainty
        
        state = state.set('marginal_parent_probs', pyr.pmap(evolving_probs))
        
        builder = ContextEnrichmentBuilder(max_history_size=50)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Check that uncertainty information is captured
        uncertainty_channel = enriched_history[:, :, 3]  # Marginal probabilities
        assert not jnp.allclose(uncertainty_channel, uncertainty_channel[0])
    
    def test_mechanism_confidence_evolution(self):
        """Test that mechanism confidence evolution is captured."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=20)
        
        # Set different mechanism confidence levels
        variables = sorted(state.buffer.get_variable_coverage())
        confidence_map = {
            variables[0]: 0.9,  # High confidence
            variables[1]: 0.5,  # Medium confidence  
            variables[2]: 0.1   # Low confidence
        }
        state = state.set('mechanism_confidence', pyr.pmap(confidence_map))
        
        builder = ContextEnrichmentBuilder(max_history_size=50)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Check mechanism confidence channel
        confidence_channel = enriched_history[:, :, 5]
        
        # Should have different values for different variables
        var_confidences = confidence_channel[-1, :]  # Most recent timestep
        assert not jnp.allclose(var_confidences, var_confidences[0])


class TestJAXCompatibility:
    """Test JAX compilation and performance improvements."""
    
    def test_enriched_encoder_is_jax_compilable(self):
        """Test that enriched encoder compiles properly with JAX."""
        @jax.jit
        def compiled_encoder_fn(enriched_history):
            def encoder_fn(enriched_history):
                encoder = EnrichedTransformerEncoder(hidden_dim=32)
                return encoder(enriched_history, is_training=False)
            
            transform = hk.transform(encoder_fn)
            key = random.PRNGKey(42)
            params = transform.init(key, enriched_history)
            return transform.apply(params, key, enriched_history)
        
        enriched_history = jnp.ones((20, 3, 10))
        
        # Should compile and run without errors
        output = compiled_encoder_fn(enriched_history)
        assert output.shape == (3, 32)
        assert jnp.all(jnp.isfinite(output))
    
    def test_policy_network_compilation(self):
        """Test that complete policy network is JAX-compilable."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=10)
        
        @jax.jit
        def compiled_policy_fn(state):
            def policy_fn(state):
                policy = EnrichedAcquisitionPolicyNetwork(hidden_dim=32)
                return policy(state, is_training=False)
            
            transform = hk.transform(policy_fn)
            key = random.PRNGKey(42)
            params = transform.init(key, state)
            return transform.apply(params, key, state)
        
        # Should compile and produce valid outputs
        output = compiled_policy_fn(state)
        
        assert 'variable_logits' in output
        assert 'value_params' in output
        assert 'state_value' in output


class TestBackwardCompatibility:
    """Test interface compatibility with existing code."""
    
    def test_policy_config_compatibility(self):
        """Test that PolicyConfig interface is preserved."""
        config = PolicyConfig(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
        
        state = create_mock_acquisition_state(n_vars=4, buffer_size=15)
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        
        # Should work with existing PolicyConfig
        params = transform.init(key, state)
        output = transform.apply(params, key, state)
        
        # Output format should be identical to old policy
        assert isinstance(output, dict)
        assert set(output.keys()) == {'variable_logits', 'value_params', 'state_value'}
    
    def test_acquisition_state_interface_preserved(self):
        """Test that AcquisitionState interface requirements are preserved."""
        state = create_mock_acquisition_state(n_vars=5, buffer_size=20)
        
        # Should work with existing AcquisitionState without modification
        builder = ContextEnrichmentBuilder(max_history_size=50)
        enriched_history, _ = builder.build_enriched_history(state)
        
        assert enriched_history.shape == (50, 5, 10)
        assert jnp.all(jnp.isfinite(enriched_history))


# Performance benchmarks
class TestPerformanceImprovements:
    """Test that new architecture provides performance benefits."""
    
    def test_parameter_count_reduction(self):
        """Test that new architecture has fewer parameters due to eliminated feature engineering."""
        state = create_mock_acquisition_state(n_vars=5, buffer_size=20)
        config = PolicyConfig(hidden_dim=128, num_layers=3)
        
        def enriched_policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers
            )
            return policy(state, is_training=False)
        
        transform = hk.transform(enriched_policy_fn)
        key = random.PRNGKey(42)
        
        params = transform.init(key, state)
        param_count = sum(p.size for p in jax.tree_leaves(params))
        
        # Should have reasonable parameter count
        # (Without exact comparison to old architecture, we verify it's not bloated)
        assert param_count < 500_000  # Reasonable upper bound
        print(f"Enriched policy parameter count: {param_count}")
    
    def test_gradient_flow_cleanliness(self):
        """Test that gradients flow cleanly through enriched architecture."""
        state = create_mock_acquisition_state(n_vars=3, buffer_size=10)
        
        def loss_fn(params, state, key):
            def policy_fn(state):
                policy = EnrichedAcquisitionPolicyNetwork(hidden_dim=32)
                return policy(state, is_training=True)
            
            transform = hk.transform(policy_fn)
            output = transform.apply(params, key, state)
            
            # Simple loss for gradient testing
            return jnp.sum(output['variable_logits']**2) + jnp.sum(output['value_params']**2)
        
        def policy_fn(state):
            policy = EnrichedAcquisitionPolicyNetwork(hidden_dim=32)
            return policy(state, is_training=True)
        
        transform = hk.transform(policy_fn)
        key = random.PRNGKey(42)
        params = transform.init(key, state)
        
        # Compute gradients
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(params, state, key)
        
        # All gradients should be finite
        finite_grads = jax.tree_map(lambda x: jnp.all(jnp.isfinite(x)), gradients)
        all_finite = jax.tree_reduce(lambda x, y: x and y, finite_grads, True)
        assert all_finite, "Some gradients are not finite"
        
        # Gradients should not be all zeros (indicating learning is possible)
        grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), gradients)
        total_grad_norm = jax.tree_reduce(lambda x, y: x + y, grad_norms, 0.0)
        assert total_grad_norm > 1e-6, "Gradients are too small - possible vanishing gradient"