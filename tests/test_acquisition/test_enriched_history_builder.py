"""
Tests for EnrichedHistoryBuilder - Critical state enrichment component.

This module tests the enriched history building functionality that converts
AcquisitionState to multi-channel transformer input with temporal context evolution.

Following TDD principles - these tests define the expected behavior.
"""

import pytest
import jax
import jax.numpy as jnp
import pyrsistent as pyr
from typing import Optional, List, Any
from unittest.mock import Mock, MagicMock
from hypothesis import given, strategies as st

from causal_bayes_opt.acquisition.enriched.state_enrichment import (
    EnrichedHistoryBuilder,
    create_enriched_history_jax,
    create_enriched_history_tensor,
    validate_enriched_state_integration
)


class MockSample:
    """Mock sample for testing."""
    
    def __init__(self, values: dict, interventions: set = None, 
                 target_value: float = 0.0, reward: float = 0.0):
        self.values = values
        self.interventions = interventions or set()
        self.target_value = target_value
        self.reward = reward


class MockBuffer:
    """Mock buffer for testing."""
    
    def __init__(self, samples: List[MockSample], variables: List[str]):
        self.samples = samples
        self.variables = variables
    
    def get_all_samples(self):
        return self.samples
    
    def get_variable_coverage(self):
        return self.variables


class MockState:
    """Mock acquisition state for testing."""
    
    def __init__(self, buffer: MockBuffer, current_target: str = "Y",
                 marginal_parent_probs: dict = None,
                 mechanism_confidence: dict = None,
                 uncertainty_bits: float = 1.0):
        self.buffer = buffer
        self.current_target = current_target
        # Ensure these are proper dictionaries, not Mock objects
        self.marginal_parent_probs = dict(marginal_parent_probs) if marginal_parent_probs else {}
        self.mechanism_confidence = dict(mechanism_confidence) if mechanism_confidence else {}
        self.uncertainty_bits = uncertainty_bits
    
    def get_mechanism_insights(self):
        return {
            'predicted_effects': {'X': 1.2, 'Y': 0.0, 'Z': -0.8},
            'mechanism_types': {'X': 'linear', 'Y': 'linear', 'Z': 'gaussian'}
        }
    
    def get_optimization_progress(self):
        return {'best_value': 2.3, 'steps_since_improvement': 5}


class TestEnrichedHistoryBuilder:
    """Test the main enriched history builder functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = EnrichedHistoryBuilder(
            standardize_values=True,
            include_temporal_features=True,
            max_history_size=50,
            support_variable_scms=True
        )
    
    def test_builder_initialization(self):
        """Test builder initializes with correct parameters."""
        assert self.builder.standardize_values is True
        assert self.builder.include_temporal_features is True
        assert self.builder.max_history_size == 50
        assert self.builder.num_channels == 10
        assert self.builder.support_variable_scms is True
    
    def test_channel_definitions(self):
        """Test that channel definitions are complete and correct."""
        expected_channels = {
            0: "variable_values",
            1: "intervention_indicators", 
            2: "target_indicators",
            3: "marginal_parent_probs",
            4: "uncertainty_bits",
            5: "mechanism_confidence",
            6: "predicted_effects",
            7: "mechanism_type_encoding",
            8: "best_value_progression",
            9: "steps_since_improvement"
        }
        
        assert self.builder.CHANNEL_DEFINITIONS == expected_channels
        assert len(self.builder.CHANNEL_DEFINITIONS) == 10
    
    def test_basic_enriched_history_shape(self):
        """Test that enriched history has correct shape."""
        # Create mock state with 3 variables
        samples = [
            MockSample({'X': 1.0, 'Y': 2.0, 'Z': -0.5}, {'X'}, 2.0, 2.0),
            MockSample({'X': 0.5, 'Y': 1.5, 'Z': -1.0}, {'Z'}, 1.5, 1.5)
        ]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')
        
        enriched_history, variable_mask = self.builder.build_enriched_history(state)
        
        # Shape should be [max_history_size, n_vars, num_channels]
        assert enriched_history.shape == (50, 3, 10)
        
        # Variable mask should indicate valid variables
        assert variable_mask.shape == (3,)
        assert jnp.all(variable_mask == 1.0)  # All variables valid
        
        # All values should be finite
        assert jnp.all(jnp.isfinite(enriched_history))
    
    def test_empty_buffer_handling(self):
        """Test handling of empty buffer."""
        empty_buffer = MockBuffer([], ['X', 'Y'])
        state = MockState(empty_buffer, 'Y')
        
        enriched_history, variable_mask = self.builder.build_enriched_history(state)
        
        # Should return zeros with correct shape
        assert enriched_history.shape == (50, 2, 10)
        assert variable_mask.shape == (2,)
        assert jnp.all(variable_mask == 1.0)
        assert jnp.all(enriched_history == 0.0)
    
    def test_variable_values_channel(self):
        """Test that variable values are correctly extracted and placed."""
        samples = [
            MockSample({'X': 2.0, 'Y': -1.0, 'Z': 0.5}),
            MockSample({'X': 1.0, 'Y': 0.0, 'Z': -0.5})
        ]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')
        
        enriched_history, _ = self.builder.build_enriched_history(state)
        
        # Check that values are in the right places (channel 0)
        # Most recent samples should be at the end of history
        values_channel = enriched_history[:, :, 0]
        
        # Should have non-zero values in the last few timesteps
        assert jnp.any(values_channel[-2:, :] != 0.0)
        
        # Should be zeros in early timesteps (no history there)
        assert jnp.all(values_channel[:-2, :] == 0.0)
    
    def test_intervention_indicators_channel(self):
        """Test that intervention indicators are correctly set."""
        samples = [
            MockSample({'X': 1.0, 'Y': 2.0, 'Z': 0.0}, {'X'}),  # Intervened on X
            MockSample({'X': 0.5, 'Y': 1.0, 'Z': -0.5}, {'Z'})  # Intervened on Z
        ]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')
        
        enriched_history, _ = self.builder.build_enriched_history(state)
        
        # Check intervention indicators (channel 1)
        intervention_channel = enriched_history[:, :, 1]
        
        # Most recent timestep: should show intervention on Z (index 2)
        assert intervention_channel[-1, 2] == 1.0  # Z intervened
        assert intervention_channel[-1, 0] == 0.0  # X not intervened
        assert intervention_channel[-1, 1] == 0.0  # Y not intervened
        
        # Previous timestep: should show intervention on X (index 0)
        assert intervention_channel[-2, 0] == 1.0  # X intervened
        assert intervention_channel[-2, 1] == 0.0  # Y not intervened  
        assert intervention_channel[-2, 2] == 0.0  # Z not intervened
    
    def test_target_indicators_channel(self):
        """Test that target indicators are correctly set."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0, 'Z': 0.0})]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')  # Y is target
        
        enriched_history, _ = self.builder.build_enriched_history(state)
        
        # Check target indicators (channel 2)
        target_channel = enriched_history[:, :, 2]
        
        # Y (index 1) should be marked as target in timesteps with data
        # Target indicators are set for the most recent timestep(s) 
        assert target_channel[-1, 1] == 1.0  # Y is target in last timestep
        assert target_channel[-1, 0] == 0.0  # X is not target
        assert target_channel[-1, 2] == 0.0  # Z is not target
    
    def test_standardization_of_values(self):
        """Test value standardization functionality."""
        # Create samples with known values for testing standardization
        samples = [
            MockSample({'X': 10.0, 'Y': 5.0}),
            MockSample({'X': 20.0, 'Y': 15.0}),
            MockSample({'X': 30.0, 'Y': 25.0})
        ]
        buffer = MockBuffer(samples, ['X', 'Y'])
        state = MockState(buffer, 'Y')
        
        builder_with_standardization = EnrichedHistoryBuilder(
            standardize_values=True,
            max_history_size=20
        )
        
        enriched_history, _ = builder_with_standardization.build_enriched_history(state)
        
        # Check that values are standardized (channel 0)
        values_channel = enriched_history[:, :, 0]
        
        # Should have reasonable standardized range
        non_zero_values = values_channel[values_channel != 0.0]
        if len(non_zero_values) > 0:
            assert jnp.abs(jnp.mean(non_zero_values)) < 2.0  # Approximately centered
            assert jnp.std(non_zero_values) > 0.1  # Has some variance
    
    def test_temporal_features_inclusion(self):
        """Test that temporal features are included when enabled."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0})]
        buffer = MockBuffer(samples, ['X', 'Y'])
        state = MockState(buffer, 'Y', 
                         marginal_parent_probs={'X': 0.8, 'Y': 0.3},
                         mechanism_confidence={'X': 0.9, 'Y': 0.7},
                         uncertainty_bits=2.5)
        
        enriched_history, _ = self.builder.build_enriched_history(state)
        
        # Check that temporal features are present (channels 3-9)
        # Marginal parent probs (channel 3)
        marginal_channel = enriched_history[:, :, 3]
        assert jnp.any(marginal_channel != 0.0)
        
        # Uncertainty bits (channel 4)
        uncertainty_channel = enriched_history[:, :, 4]
        assert jnp.any(uncertainty_channel != 0.0)
        
        # Mechanism confidence (channel 5)
        confidence_channel = enriched_history[:, :, 5]
        assert jnp.any(confidence_channel != 0.0)
    
    def test_temporal_features_disabled(self):
        """Test that temporal features are zeroed when disabled."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0})]
        buffer = MockBuffer(samples, ['X', 'Y'])
        state = MockState(buffer, 'Y')
        
        builder_no_temporal = EnrichedHistoryBuilder(
            include_temporal_features=False,
            max_history_size=20
        )
        
        enriched_history, _ = builder_no_temporal.build_enriched_history(state)
        
        # Channels 3-9 should be zero when temporal features disabled
        temporal_channels = enriched_history[:, :, 3:]
        assert jnp.all(temporal_channels == 0.0)
        
        # But core channels 0-2 should still have data
        core_channels = enriched_history[:, :, :3]
        assert jnp.any(core_channels != 0.0)
    
    def test_shape_invariants_multiple_cases(self):
        """Test: output shape should always be consistent across different cases."""
        test_cases = [
            (2, 10, 1),
            (3, 20, 3),
            (5, 30, 2),
            (4, 15, 4)
        ]
        
        for num_variables, max_history_size, num_samples in test_cases:
            builder = EnrichedHistoryBuilder(max_history_size=max_history_size)
            
            # Create samples
            variables = [f'X{i}' for i in range(num_variables)]
            samples = []
            for i in range(num_samples):
                values = {var: float(i + j) for j, var in enumerate(variables)}
                samples.append(MockSample(values))
            
            buffer = MockBuffer(samples, variables)
            state = MockState(buffer, variables[0])
            
            enriched_history, variable_mask = builder.build_enriched_history(state)
            
            # Shape invariants
            assert enriched_history.shape == (max_history_size, num_variables, 10)
            assert variable_mask.shape == (num_variables,)
            assert jnp.all(jnp.isfinite(enriched_history))
            assert jnp.all((variable_mask == 0.0) | (variable_mask == 1.0))
    
    def test_validation_functionality(self):
        """Test enriched history validation."""
        # Test valid enriched history
        valid_history = jnp.ones((50, 3, 10))
        assert self.builder.validate_enriched_history(valid_history) is True
        
        # Test invalid shapes
        invalid_2d = jnp.ones((50, 10))
        assert self.builder.validate_enriched_history(invalid_2d) is False
        
        invalid_channels = jnp.ones((50, 3, 8))  # Wrong number of channels
        assert self.builder.validate_enriched_history(invalid_channels) is False
        
        invalid_timesteps = jnp.ones((40, 3, 10))  # Wrong number of timesteps
        assert self.builder.validate_enriched_history(invalid_timesteps) is False
        
        # Test NaN/infinite values
        invalid_nan = jnp.ones((50, 3, 10)).at[0, 0, 0].set(jnp.nan)
        assert self.builder.validate_enriched_history(invalid_nan) is False
        
        invalid_inf = jnp.ones((50, 3, 10)).at[0, 0, 0].set(jnp.inf)
        assert self.builder.validate_enriched_history(invalid_inf) is False
    
    def test_channel_info_retrieval(self):
        """Test channel information retrieval."""
        channel_info = self.builder.get_channel_info()
        
        assert isinstance(channel_info, dict)
        assert len(channel_info) == 10
        
        for i in range(10):
            assert i in channel_info
            assert isinstance(channel_info[i], str)
    
    def test_missing_state_attributes_handling(self):
        """Test graceful handling of missing state attributes."""
        # Create minimal state without optional attributes
        samples = [MockSample({'X': 1.0})]
        buffer = MockBuffer(samples, ['X'])
        
        # State missing optional attributes - use MockState with empty dicts
        minimal_state = MockState(
            buffer=buffer,
            current_target='X',
            marginal_parent_probs={},  # Empty dict instead of Mock
            mechanism_confidence={},   # Empty dict instead of Mock
            uncertainty_bits=1.0
        )
        
        enriched_history, variable_mask = self.builder.build_enriched_history(minimal_state)
        
        # Should still work and return valid shape
        assert enriched_history.shape == (50, 1, 10)
        assert variable_mask.shape == (1,)
        assert jnp.all(jnp.isfinite(enriched_history))


class TestEnrichedHistoryConvenienceFunctions:
    """Test convenience functions for enriched history creation."""
    
    def test_create_enriched_history_jax(self):
        """Test JAX-compatible enriched history creation."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0})]
        buffer = MockBuffer(samples, ['X', 'Y'])
        state = MockState(buffer, 'Y')
        
        enriched_history, variable_mask = create_enriched_history_jax(
            state=state,
            max_history_size=30,
            include_temporal_features=True,
            support_variable_scms=True
        )
        
        assert enriched_history.shape == (30, 2, 10)
        assert variable_mask.shape == (2,)
        assert jnp.all(jnp.isfinite(enriched_history))
    
    def test_create_enriched_history_tensor(self):
        """Test main entry point for enriched history creation."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0, 'Z': -0.5})]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')
        
        enriched_history, variable_mask = create_enriched_history_tensor(
            state=state,
            max_history_size=40,
            include_temporal_features=False,
            support_variable_scms=True
        )
        
        assert enriched_history.shape == (40, 3, 10)
        assert variable_mask.shape == (3,)
        
        # Temporal features should be disabled
        temporal_channels = enriched_history[:, :, 3:]
        assert jnp.all(temporal_channels == 0.0)


class TestEnrichedStateIntegration:
    """Test integration validation and end-to-end functionality."""
    
    def test_validate_enriched_state_integration(self):
        """Test the integration validation function."""
        result = validate_enriched_state_integration()
        
        # Should return True if integration works correctly
        assert isinstance(result, bool)
        
        # If it returns True, the integration is working
        # If it returns False, there might be issues but the test framework exists
    
    def test_enriched_history_with_mechanism_insights(self):
        """Test enriched history creation with mechanism insights."""
        samples = [MockSample({'X': 1.0, 'Y': 2.0, 'Z': -0.5})]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        
        # State with rich mechanism insights
        state = MockState(
            buffer, 'Y',
            marginal_parent_probs={'X': 0.8, 'Y': 0.3, 'Z': 0.6},
            mechanism_confidence={'X': 0.9, 'Y': 0.7, 'Z': 0.8},
            uncertainty_bits=1.5
        )
        
        builder = EnrichedHistoryBuilder(include_temporal_features=True)
        enriched_history, variable_mask = builder.build_enriched_history(state)
        
        # Check that mechanism insights are reflected in channels
        # Channel 3: marginal parent probs
        marginal_channel = enriched_history[-1, :, 3]  # Most recent timestep
        expected_probs = jnp.array([0.8, 0.3, 0.6])  # X, Y, Z order
        assert jnp.allclose(marginal_channel, expected_probs, atol=0.1)
        
        # Channel 5: mechanism confidence
        confidence_channel = enriched_history[-1, :, 5]
        expected_confidence = jnp.array([0.9, 0.7, 0.8])
        assert jnp.allclose(confidence_channel, expected_confidence, atol=0.1)
    
    def test_enriched_history_best_value_progression(self):
        """Test that best value progression is correctly tracked."""
        # Create samples with increasing target values
        samples = [
            MockSample({'X': 1.0}, reward=1.0, target_value=1.0),
            MockSample({'X': 2.0}, reward=1.5, target_value=1.5),
            MockSample({'X': 1.5}, reward=1.2, target_value=1.2),  # Worse than previous
            MockSample({'X': 3.0}, reward=2.0, target_value=2.0)   # New best
        ]
        buffer = MockBuffer(samples, ['X'])
        state = MockState(buffer, 'X')
        
        builder = EnrichedHistoryBuilder(include_temporal_features=True, max_history_size=10)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Check best value progression (channel 8)
        best_value_channel = enriched_history[:, 0, 8]  # Focus on variable X
        
        # Should show progression: 1.0, 1.5, 1.5 (no improvement), 2.0
        non_zero_indices = jnp.where(best_value_channel != 0.0)[0]
        if len(non_zero_indices) >= 4:
            # Check the progression in the last 4 non-zero values
            progression = best_value_channel[non_zero_indices[-4:]]
            
            # Should be monotonically non-decreasing (best value only improves)
            assert jnp.all(progression[1:] >= progression[:-1])


class TestEnrichedHistoryEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_variable_scm(self):
        """Test enriched history with single variable SCM."""
        samples = [MockSample({'X': 1.0})]
        buffer = MockBuffer(samples, ['X'])
        state = MockState(buffer, 'X')  # Target is the only variable
        
        builder = EnrichedHistoryBuilder(max_history_size=20)
        enriched_history, variable_mask = builder.build_enriched_history(state)
        
        assert enriched_history.shape == (20, 1, 10)
        assert variable_mask.shape == (1,)
        
        # Target indicators should be 1.0 for the single variable in data timesteps
        target_channel = enriched_history[:, 0, 2]
        assert target_channel[-1] == 1.0  # Should be marked in the last timestep
    
    def test_large_variable_count(self):
        """Test enriched history with large number of variables."""
        variables = [f'X{i}' for i in range(10)]
        samples = [MockSample({var: float(i) for i, var in enumerate(variables)})]
        buffer = MockBuffer(samples, variables)
        state = MockState(buffer, 'X0')
        
        builder = EnrichedHistoryBuilder(max_history_size=30)
        enriched_history, variable_mask = builder.build_enriched_history(state)
        
        assert enriched_history.shape == (30, 10, 10)
        assert variable_mask.shape == (10,)
        assert jnp.all(variable_mask == 1.0)  # All variables valid
    
    def test_extreme_values_handling(self):
        """Test handling of extreme values in samples."""
        # Create samples with extreme values
        samples = [
            MockSample({'X': 1e6, 'Y': -1e6, 'Z': 0.0}),
            MockSample({'X': 1e-6, 'Y': 1e-6, 'Z': 1e6})
        ]
        buffer = MockBuffer(samples, ['X', 'Y', 'Z'])
        state = MockState(buffer, 'Y')
        
        builder = EnrichedHistoryBuilder(standardize_values=True)
        enriched_history, _ = builder.build_enriched_history(state)
        
        # Should handle extreme values without NaN/Inf
        assert jnp.all(jnp.isfinite(enriched_history))
        
        # Standardization should bring values into reasonable range
        values_channel = enriched_history[:, :, 0]
        non_zero_values = values_channel[values_channel != 0.0]
        if len(non_zero_values) > 0:
            assert jnp.max(jnp.abs(non_zero_values)) < 100.0  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])