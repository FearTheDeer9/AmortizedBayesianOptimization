"""
Tests for global standardization fix in enriched history building.

This module tests the fix for the GRPO collapse issue where per-variable
standardization was removing natural diversity between variables.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from typing import List, Dict

from src.causal_bayes_opt.acquisition.enriched.state_enrichment import (
    EnrichedHistoryBuilder,
    create_enriched_history_tensor
)
from src.causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from src.causal_bayes_opt.jax_native.state import create_tensor_backed_state_from_scm
from src.causal_bayes_opt.jax_native.sample_buffer import add_sample_jax


class TestGlobalStandardization:
    """Test suite for global standardization in enriched history building."""
    
    def test_standardization_preserves_diversity(self):
        """Test that global standardization preserves natural variable diversity."""
        # Create SCM with variables at different scales
        variables = ['X0', 'X1', 'X2', 'X3', 'X4']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3'), ('X3', 'X4')],
            coefficients={
                ('X0', 'X1'): 2.0,    # X1 will be ~2x scale of X0
                ('X1', 'X2'): 1.5,    # X2 will be ~3x scale of X0  
                ('X2', 'X3'): 1.2,    # X3 will be ~3.6x scale of X0
                ('X3', 'X4'): 1.1     # X4 will be ~4x scale of X0
            },
            noise_scales={var: 0.1 for var in variables},
            target='X2'
        )
        
        # Generate state with samples
        key = random.PRNGKey(42)
        state = create_tensor_backed_state_from_scm(scm, step=0)
        
        # Add samples with natural scale differences
        n_vars = len(variables)
        for i in range(50):
            key, subkey = random.split(key)
            
            # Create values at different scales matching the SCM structure
            values = jnp.array([
                float(random.normal(subkey, shape=())),           # X0: base scale
                float(random.normal(subkey, shape=()) * 2.0),    # X1: 2x scale
                float(random.normal(subkey, shape=()) * 3.0),    # X2: 3x scale  
                float(random.normal(subkey, shape=()) * 3.6),    # X3: 3.6x scale
                float(random.normal(subkey, shape=()) * 4.0)     # X4: 4x scale
            ])
            
            intervention_mask = jnp.zeros(n_vars, dtype=bool)
            target_value = values[2]  # X2 is target
            
            new_buffer = add_sample_jax(state.sample_buffer, values, 
                                       intervention_mask, target_value)
            from dataclasses import replace
            state = replace(state, sample_buffer=new_buffer)
        
        # Create enriched history with global standardization
        builder = EnrichedHistoryBuilder(standardize_values=True)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Extract standardized values from last timestep
        variable_values = enriched_history[-1, :, 0]  # Last time, all vars, channel 0
        
        # Compute diversity metrics
        var_means = jnp.mean(variable_values)
        var_stds = jnp.std(variable_values)
        diversity_score = float(var_stds)
        
        # With global standardization, variables should have different means
        # reflecting their natural scale differences
        assert diversity_score > 0.1, f"Diversity score too low: {diversity_score}"
        
        # Check that variables are not all identical
        unique_values = jnp.unique(variable_values)
        assert len(unique_values) == n_vars, "Variables have identical values after standardization"
    
    def test_no_standardization_option(self):
        """Test that standardization can be disabled."""
        variables = ['X0', 'X1', 'X2']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2')],
            coefficients={('X0', 'X1'): 1.5, ('X1', 'X2'): 1.5},
            noise_scales={var: 0.1 for var in variables},
            target='X2'
        )
        
        state = create_tensor_backed_state_from_scm(scm, step=0)
        
        # Add a sample
        key = random.PRNGKey(42)
        values = random.normal(key, shape=(3,))
        intervention_mask = jnp.zeros(3, dtype=bool)
        target_value = values[2]
        
        new_buffer = add_sample_jax(state.sample_buffer, values, 
                                   intervention_mask, target_value)
        from dataclasses import replace
        state = replace(state, sample_buffer=new_buffer)
        
        # Test with standardization disabled
        builder = EnrichedHistoryBuilder(standardize_values=False)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Values should be raw (not standardized)
        raw_values = enriched_history[-1, :, 0]
        
        # Should approximately match the original values
        assert jnp.allclose(raw_values, values, atol=1e-5)
    
    def test_embedding_diversity_with_global_standardization(self):
        """Test that global standardization improves embedding diversity."""
        from src.causal_bayes_opt.acquisition.enriched.enriched_policy import (
            EnrichedAttentionEncoder
        )
        import haiku as hk
        
        # Create test SCM
        variables = ['X0', 'X1', 'X2', 'X3']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3')],
            coefficients={
                ('X0', 'X1'): 2.0,
                ('X1', 'X2'): 1.5,
                ('X2', 'X3'): 1.2
            },
            noise_scales={var: 0.1 for var in variables},
            target='X2'
        )
        
        # Create state with samples
        key = random.PRNGKey(42)
        state = create_tensor_backed_state_from_scm(scm, step=0)
        
        # Add diverse samples
        n_vars = len(variables)
        for i in range(20):
            key, subkey = random.split(key)
            values = random.normal(subkey, (n_vars,)) * jnp.array([1.0, 2.0, 3.0, 3.6])
            intervention_mask = jnp.zeros(n_vars, dtype=bool)
            target_value = values[2]
            
            new_buffer = add_sample_jax(state.sample_buffer, values, 
                                       intervention_mask, target_value)
            from dataclasses import replace
            state = replace(state, sample_buffer=new_buffer)
        
        # Create enriched history
        enriched_history, _ = create_enriched_history_tensor(state)
        
        # Encode with attention encoder
        def encode_fn(history):
            encoder = EnrichedAttentionEncoder(
                num_layers=2,
                num_heads=4,
                hidden_dim=128,
                dropout=0.0
            )
            n_vars = history.shape[1]
            target_mask = jnp.zeros(n_vars)
            intervention_mask = jnp.zeros(n_vars)
            return encoder(history, is_training=False,
                          target_mask=target_mask,
                          intervention_mask=intervention_mask)
        
        encode = hk.transform(encode_fn)
        key, subkey = random.split(key)
        params = encode.init(subkey, enriched_history)
        embeddings = encode.apply(params, subkey, enriched_history)
        
        # Compute embedding similarity
        n_vars = embeddings.shape[0]
        norms = jnp.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        similarity_matrix = jnp.dot(normalized, normalized.T)
        
        # Extract upper triangle similarities
        upper_indices = jnp.triu_indices(n_vars, k=1)
        mean_similarity = float(jnp.mean(similarity_matrix[upper_indices]))
        
        # With global standardization, similarity should be lower (more diversity)
        assert mean_similarity < 0.95, f"Embedding similarity too high: {mean_similarity}"


class TestBootstrapSurrogateIntegration:
    """Test suite for bootstrap surrogate with global standardization."""
    
    def test_bootstrap_with_global_standardization(self):
        """Test that bootstrap surrogate works correctly with global standardization."""
        variables = ['X0', 'X1', 'X2', 'X3']
        scm = create_simple_linear_scm(
            variables=variables,
            edges=[('X0', 'X1'), ('X1', 'X2'), ('X2', 'X3')],
            coefficients={
                ('X0', 'X1'): 1.5,
                ('X1', 'X2'): 1.5,
                ('X2', 'X3'): 1.5
            },
            noise_scales={var: 0.1 for var in variables},
            target='X2'
        )
        
        # Create state with bootstrap surrogate
        state = create_tensor_backed_state_from_scm(
            scm, step=0, use_bootstrap_surrogate=True
        )
        
        # Create enriched history
        enriched_history, _ = create_enriched_history_tensor(state)
        
        # Check that parent probability channel exists and has diversity
        parent_prob_channel = enriched_history[:, :, 3]  # Channel 3
        channel_variance = float(jnp.var(parent_prob_channel))
        
        # Bootstrap should provide some diversity in parent probabilities
        assert channel_variance > 0.01, f"Parent probability variance too low: {channel_variance}"
        
        # But the main diversity should come from value channel with global standardization
        value_channel = enriched_history[:, :, 0]  # Channel 0
        value_variance = float(jnp.var(value_channel))
        
        # Value channel should have meaningful variance
        assert value_variance > 0.1, f"Value channel variance too low: {value_variance}"


def test_configuration_integration():
    """Test that global standardization is properly integrated in configuration."""
    # This would test configuration files once we update them
    # For now, just verify the default behavior
    builder = EnrichedHistoryBuilder()
    assert builder.standardize_values == True  # Default should be True
    
    # Verify the standardization method is global
    # This is implicitly tested by the diversity preservation tests above


if __name__ == "__main__":
    # Run tests manually for debugging
    test_suite = TestGlobalStandardization()
    test_suite.test_standardization_preserves_diversity()
    test_suite.test_no_standardization_option()
    test_suite.test_embedding_diversity_with_global_standardization()
    
    bootstrap_suite = TestBootstrapSurrogateIntegration()
    bootstrap_suite.test_bootstrap_with_global_standardization()
    
    test_configuration_integration()
    
    print("✅ All tests passed!")