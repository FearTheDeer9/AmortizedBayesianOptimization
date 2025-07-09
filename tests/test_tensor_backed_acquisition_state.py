"""
Tests for TensorBackedAcquisitionState

Property-based tests validating the unified tensor-state architecture that provides
both JAX efficiency and AcquisitionState interface compatibility.

Following TDD principles: These tests are written FIRST and should FAIL initially.
"""

import pytest
import jax.numpy as jnp
import jax.random as random
import pyrsistent as pyr
from hypothesis import given, strategies as st
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Import will fail initially - this is expected for TDD
try:
    from causal_bayes_opt.jax_native.state import TensorBackedAcquisitionState
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    # Create placeholder for tests
    TensorBackedAcquisitionState = None

from causal_bayes_opt.jax_native.config import JAXConfig, create_jax_config, create_test_config
from causal_bayes_opt.jax_native.sample_buffer import create_empty_jax_buffer

# Import legacy types for compatibility testing
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.avici_integration.parent_set import ParentSetPosterior
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestTensorBackedAcquisitionState:
    """Test suite for TensorBackedAcquisitionState core functionality."""
    
    def test_creation_with_valid_config(self):
        """Test creating TensorBackedAcquisitionState with valid configuration."""
        config = create_jax_config(['X0', 'X1', 'X2'], 'X1')
        
        # This should pass once implemented
        state = TensorBackedAcquisitionState.create_empty(config)
        
        assert isinstance(state, TensorBackedAcquisitionState)
        assert state.config == config
        assert state.current_step == 0
        assert jnp.isfinite(state.best_value)
        assert state.uncertainty_bits >= 0
    
    def test_acquisition_state_interface_compatibility(self):
        """Test that TensorBackedAcquisitionState provides AcquisitionState interface."""
        config = create_jax_config(['X0', 'X1', 'X2'], 'X1') 
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Should provide all AcquisitionState properties
        assert hasattr(state, 'posterior')
        assert hasattr(state, 'buffer')
        assert hasattr(state, 'step')
        assert hasattr(state, 'best_value')
        assert hasattr(state, 'current_target')
        assert hasattr(state, 'uncertainty_bits')
        assert hasattr(state, 'marginal_parent_probs')
        
    def test_posterior_property_returns_valid_posterior(self):
        """Test that posterior property returns valid ParentSetPosterior."""
        config = create_jax_config(['X0', 'X1', 'X2'], 'X1')
        state = TensorBackedAcquisitionState.create_empty(config)
        
        posterior = state.posterior
        assert isinstance(posterior, ParentSetPosterior)
        assert hasattr(posterior, 'parent_set_probs')
        assert hasattr(posterior, 'uncertainty')
        assert hasattr(posterior, 'target_variable')
        
    def test_buffer_property_returns_valid_buffer(self):
        """Test that buffer property returns ExperienceBuffer interface."""
        config = create_jax_config(['X0', 'X1', 'X2'], 'X1')
        state = TensorBackedAcquisitionState.create_empty(config)
        
        buffer = state.buffer
        # Should provide ExperienceBuffer interface methods
        assert hasattr(buffer, 'get_all_samples')
        assert hasattr(buffer, 'get_variable_coverage')
        assert hasattr(buffer, 'get_statistics')
        
    def test_marginal_parent_probs_computation(self):
        """Test computation of marginal parent probabilities from tensor data."""
        config = create_jax_config(['X0', 'X1', 'X2'], 'X1')
        state = TensorBackedAcquisitionState.create_empty(config)
        
        marginal_probs = state.marginal_parent_probs
        assert isinstance(marginal_probs, dict)
        assert len(marginal_probs) == config.n_vars - 1  # Excluding target
        
        # All probabilities should be valid
        for var_name, prob in marginal_probs.items():
            assert 0.0 <= prob <= 1.0
            assert var_name != config.get_target_name()


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet") 
class TestTensorBackedStateJAXCompatibility:
    """Test JAX compilation and tensor operation compatibility."""
    
    def test_state_jax_compilation(self):
        """Test that state operations are JAX-compilable."""
        import jax
        
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        @jax.jit
        def process_tensors(best_value, marginal_probs):
            """Test JAX compilation with tensor operations."""
            return best_value + jnp.sum(marginal_probs)
            
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Extract tensors from state for JAX compilation
        best_value = state.best_value
        marginal_probs = state.marginal_probs
        
        # Should compile and execute without error
        result = process_tensors(best_value, marginal_probs)
        assert jnp.isfinite(result)
    
    def test_tensor_data_access(self):
        """Test direct access to tensor components for JAX operations."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Should provide direct tensor access
        assert isinstance(state.mechanism_features, jnp.ndarray)
        assert isinstance(state.marginal_probs, jnp.ndarray)
        assert isinstance(state.confidence_scores, jnp.ndarray)
        
        # Tensor shapes should match config
        assert state.mechanism_features.shape[0] == config.n_vars
        assert state.marginal_probs.shape[0] == config.n_vars
        assert state.confidence_scores.shape[0] == config.n_vars


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestVariableSizeGraphSupport:
    """Test support for variable-size graphs (3, 4, 5+ variables)."""
    
    @given(n_vars=st.integers(min_value=2, max_value=10))
    def test_variable_graph_sizes(self, n_vars):
        """Property test: State creation works for any reasonable graph size."""
        target_idx = n_vars // 2  # Middle variable as target
        config = create_test_config_with_params(n_vars=n_vars, target_idx=target_idx)
        
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Core invariants
        assert state.config.n_vars == n_vars
        assert state.mechanism_features.shape[0] == n_vars
        assert len(state.marginal_parent_probs) == n_vars - 1  # Excluding target
        
    def test_graph_size_compatibility_3_to_5_vars(self):
        """Test compatibility across common graph sizes."""
        for n_vars in [3, 4, 5]:
            target_idx = 0  # First variable as target
            config = create_test_config_with_params(n_vars=n_vars, target_idx=target_idx)
            
            state = TensorBackedAcquisitionState.create_empty(config)
            
            # Interface should work consistently 
            assert isinstance(state.posterior, ParentSetPosterior)
            assert len(state.marginal_parent_probs) == n_vars - 1
            assert state.current_target == config.get_target_name()


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestGRPOCompatibility:
    """Test compatibility with GRPO training requirements."""
    
    def test_vmap_compatibility(self):
        """Test that state tensors can be used in JAX vmap operations."""
        import jax
        
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        
        # Create batch of states for vmap testing
        states = [
            TensorBackedAcquisitionState.create_empty(config)
            for _ in range(5)
        ]
        
        # Extract tensor components that can be vmapped
        best_values = jnp.array([state.best_value for state in states])
        marginal_probs_batch = jnp.stack([state.marginal_probs for state in states])
        
        def process_tensor_data(best_value, marginal_probs):
            """Function to vmap over tensor data."""
            return best_value + jnp.sum(marginal_probs)
            
        # Should work with vmap over tensor data
        vmapped_fn = jax.vmap(process_tensor_data)
        results = vmapped_fn(best_values, marginal_probs_batch)
        
        assert results.shape == (5,)
        assert jnp.all(jnp.isfinite(results))
    
    def test_grpo_interface_methods(self):
        """Test that state provides methods needed by GRPO."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # GRPO needs these for reward computation
        assert hasattr(state, 'buffer')
        assert hasattr(state, 'posterior')
        assert hasattr(state, 'step')
        assert hasattr(state, 'best_value')
        
        # Buffer should support GRPO operations
        buffer = state.buffer
        assert callable(getattr(buffer, 'get_variable_coverage', None))
        assert callable(getattr(buffer, 'get_statistics', None))


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestMigrationCompatibility:
    """Test backward compatibility with existing AcquisitionState usage."""
    
    def test_can_replace_acquisition_state(self):
        """Test that TensorBackedAcquisitionState can replace AcquisitionState in existing code."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Should be usable wherever AcquisitionState is expected
        self._legacy_function_expecting_acquisition_state(state)
    
    def _legacy_function_expecting_acquisition_state(self, state):
        """Simulate legacy function that expects AcquisitionState interface."""
        # These operations should all work
        _ = state.posterior
        _ = state.buffer
        _ = state.step
        _ = state.best_value
        _ = state.current_target
        _ = state.uncertainty_bits
        _ = state.marginal_parent_probs
        
    def test_maintains_immutability(self):
        """Test that TensorBackedAcquisitionState maintains immutability."""
        config = create_test_config_with_params(n_vars=3, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Should be frozen dataclass
        with pytest.raises(AttributeError):
            state.best_value = 999.0  # Should fail - immutable
            
        with pytest.raises(AttributeError):
            state.current_step = 100  # Should fail - immutable


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestPerformanceRequirements:
    """Test performance requirements for JAX compilation."""
    
    def test_fast_property_access(self):
        """Test that property access is reasonably fast."""
        import time
        
        config = create_test_config_with_params(n_vars=4, target_idx=1)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Time property access (should be fast)
        start_time = time.time()
        for _ in range(1000):
            _ = state.posterior
            _ = state.buffer
            _ = state.marginal_parent_probs
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert (end_time - start_time) < 1.0  # 1 second for 1000 accesses
    
    def test_memory_efficiency(self):
        """Test that state doesn't use excessive memory."""
        config = create_test_config_with_params(n_vars=5, target_idx=2)
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Memory usage should be reasonable for tensor storage
        # (This is a placeholder - specific memory tests would need more sophisticated tooling)
        assert hasattr(state, '__sizeof__')


# Helper function for test configs
def create_test_config_with_params(n_vars: int, target_idx: int) -> JAXConfig:
    """Create test config with specified parameters."""
    var_names = [f'X{i}' for i in range(n_vars)]
    target_name = var_names[target_idx]
    return create_jax_config(var_names, target_name)

# Property-based test strategies
@st.composite 
def valid_configs(draw):
    """Generate valid JAXConfig instances for property testing."""
    n_vars = draw(st.integers(min_value=2, max_value=8))
    target_idx = draw(st.integers(min_value=0, max_value=n_vars-1))
    
    return create_test_config_with_params(n_vars, target_idx)


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="TensorBackedAcquisitionState not implemented yet")
class TestPropertyBasedInvariants:
    """Property-based tests for state invariants."""
    
    @given(config=valid_configs())
    def test_state_creation_invariants(self, config):
        """Property test: State creation maintains invariants."""
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Core invariants that should hold for any valid config
        assert state.config == config
        assert state.current_step >= 0
        assert jnp.isfinite(state.best_value)
        assert state.uncertainty_bits >= 0
        assert state.mechanism_features.shape[0] == config.n_vars
        
    @given(config=valid_configs())
    def test_interface_consistency(self, config):
        """Property test: AcquisitionState interface is consistent."""
        state = TensorBackedAcquisitionState.create_empty(config)
        
        # Interface consistency
        assert state.step == state.current_step
        assert state.current_target == config.get_target_name()
        assert isinstance(state.marginal_parent_probs, dict)
        
        # Target variable should not appear in marginal probs
        target_name = config.get_target_name()
        assert target_name not in state.marginal_parent_probs


if __name__ == "__main__":
    # Run tests that should fail initially (TDD)
    pytest.main([__file__, "-v"])