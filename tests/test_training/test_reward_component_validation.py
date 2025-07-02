"""
Tests for reward component validation framework.

These tests verify that the component validation system works correctly
and can properly isolate and test individual reward components.
"""

import pytest
import jax.numpy as jnp
import pyrsistent as pyr
from unittest.mock import Mock, patch

from src.causal_bayes_opt.training.reward_component_validation import (
    create_component_test_configs,
    validate_single_component,
    validate_all_components,
    run_component_isolation_test,
    ComponentValidationResult,
    RewardValidationSuite
)
from src.causal_bayes_opt.acquisition.rewards import create_default_reward_config
from src.causal_bayes_opt.acquisition.state import AcquisitionState


@pytest.fixture
def mock_states():
    """Create mock states for testing."""
    # Mock state before
    state_before = Mock(spec=AcquisitionState)
    state_before.current_target = "Y"
    state_before.best_value = 0.0
    state_before.posterior = Mock()
    state_before.buffer = Mock()
    state_before.buffer.samples = []
    
    # Mock state after
    state_after = Mock(spec=AcquisitionState)
    state_after.current_target = "Y"
    state_after.best_value = 1.0
    state_after.posterior = Mock()
    
    return state_before, state_after


@pytest.fixture
def mock_intervention_and_outcome():
    """Create mock intervention and outcome."""
    intervention = pyr.m(
        type='perfect',
        targets=frozenset(['X']),
        values={'X': 1.0}
    )
    
    outcome = pyr.m(
        values={'X': 1.0, 'Y': 1.5},
        intervention_targets=frozenset(['X'])
    )
    
    return intervention, outcome


def test_create_component_test_configs():
    """Test that component test configs are created correctly."""
    configs = create_component_test_configs()
    
    # Check that all expected configs exist
    expected_configs = [
        'no_optimization', 'no_structure', 'no_parent', 'no_exploration',
        'balanced', 'optimization_only', 'structure_only'
    ]
    
    for config_name in expected_configs:
        assert config_name in configs
        assert isinstance(configs[config_name], pyr.PMap)
    
    # Check that no_optimization config has optimization weight = 0
    no_opt_config = configs['no_optimization']
    weights = no_opt_config['reward_weights']
    assert weights['optimization'] == 0.0
    assert weights['structure'] > 0.0  # Other components should be non-zero
    
    # Check that balanced config has equal optimization and structure weights
    balanced_config = configs['balanced']
    balanced_weights = balanced_config['reward_weights']
    assert balanced_weights['optimization'] == balanced_weights['structure'] == 1.0


@patch('src.causal_bayes_opt.training.reward_component_validation.compute_verifiable_reward')
def test_validate_single_component(mock_compute_reward, mock_states, mock_intervention_and_outcome):
    """Test validation of a single component."""
    state_before, state_after = mock_states
    intervention, outcome = mock_intervention_and_outcome
    
    # Mock reward computation to return different values for enabled/disabled
    def mock_reward_side_effect(sb, i, o, sa, config):
        # Return higher reward when optimization component is enabled
        weights = config['reward_weights']
        base_reward = 0.5
        if weights['optimization'] > 0:
            base_reward += 0.3  # Component contribution
        
        result = Mock()
        result.total_reward = base_reward
        result.optimization_reward = 0.3 if weights['optimization'] > 0 else 0.0
        result.structure_discovery_reward = 0.2
        result.parent_intervention_reward = 0.1
        result.exploration_bonus = 0.05
        return result
    
    mock_compute_reward.side_effect = mock_reward_side_effect
    
    # Test optimization component validation
    result = validate_single_component(
        'optimization', state_before, intervention, outcome, state_after, tolerance=0.1
    )
    
    assert isinstance(result, ComponentValidationResult)
    assert result.component_name == 'optimization'
    assert result.enabled_reward > result.disabled_reward
    assert result.component_contribution > 0.1  # Should exceed tolerance
    assert result.validation_passed is True
    
    # Test with tolerance too high (should fail)
    result_high_tolerance = validate_single_component(
        'optimization', state_before, intervention, outcome, state_after, tolerance=0.5
    )
    assert result_high_tolerance.validation_passed is False


@patch('src.causal_bayes_opt.training.reward_component_validation.validate_single_component')
def test_validate_all_components(mock_validate_single, mock_states, mock_intervention_and_outcome):
    """Test validation of all components."""
    state_before, state_after = mock_states
    intervention, outcome = mock_intervention_and_outcome
    
    # Mock individual component validations
    def mock_validation_side_effect(component_name, *args, **kwargs):
        # Critical components (optimization, structure) pass, others may fail
        passed = component_name in ['optimization', 'structure']
        return ComponentValidationResult(
            component_name=component_name,
            enabled_reward=1.0,
            disabled_reward=0.7,
            component_contribution=0.3 if passed else 0.05,
            validation_passed=passed,
            metadata={}
        )
    
    mock_validate_single.side_effect = mock_validation_side_effect
    
    # Test complete validation
    suite = validate_all_components(
        state_before, intervention, outcome, state_after, tolerance=0.1
    )
    
    assert isinstance(suite, RewardValidationSuite)
    assert suite.optimization_result.validation_passed is True
    assert suite.structure_result.validation_passed is True
    assert suite.overall_validation_passed is True  # Critical components passed
    
    # Verify all components were tested
    assert mock_validate_single.call_count == 4
    tested_components = [call[0][0] for call in mock_validate_single.call_args_list]
    assert set(tested_components) == {'optimization', 'structure', 'parent', 'exploration'}


def test_run_component_isolation_test(mock_states, mock_intervention_and_outcome):
    """Test component isolation testing."""
    state_before, state_after = mock_states
    intervention, outcome = mock_intervention_and_outcome
    
    # Create test scenarios
    scenarios = [
        (state_before, intervention, outcome, state_after),
        (state_before, intervention, outcome, state_after)
    ]
    
    # Mock reward computer that returns increasing rewards
    def mock_reward_computer(sb, i, o, sa, config):
        result = Mock()
        # Simulate increasing trend for optimization component
        weights = config['reward_weights']
        if weights['optimization'] > 0:
            result.total_reward = 0.5 + len(scenarios) * 0.1  # Increasing
        else:
            result.total_reward = 0.3  # Stable without optimization
        return result
    
    # Test isolation of optimization component
    isolation_result = run_component_isolation_test(
        mock_reward_computer, scenarios, 'optimization', expected_behavior='increase'
    )
    
    assert isolation_result['component_name'] == 'optimization'
    assert 'test_passed' in isolation_result
    assert 'rewards' in isolation_result
    assert len(isolation_result['rewards']) == len(scenarios)
    
    # Test with insufficient scenarios
    single_scenario = [scenarios[0]]
    isolation_result_insufficient = run_component_isolation_test(
        mock_reward_computer, single_scenario, 'optimization'
    )
    assert isolation_result_insufficient['test_passed'] is False
    assert 'Insufficient scenarios' in isolation_result_insufficient['reason']


def test_component_validation_error_handling(mock_states, mock_intervention_and_outcome):
    """Test that validation handles errors gracefully."""
    state_before, state_after = mock_states
    intervention, outcome = mock_intervention_and_outcome
    
    # Test with invalid component name
    with patch('src.causal_bayes_opt.training.reward_component_validation.compute_verifiable_reward') as mock_compute:
        mock_compute.side_effect = Exception("Test error")
        
        result = validate_single_component(
            'invalid_component', state_before, intervention, outcome, state_after
        )
        
        assert result.validation_passed is False
        assert 'error' in result.metadata
        assert result.component_contribution == 0.0


def test_config_weights_are_correct():
    """Test that generated configs have correct weight values."""
    configs = create_component_test_configs()
    
    # Test optimization_only config
    opt_only = configs['optimization_only']
    opt_weights = opt_only['reward_weights']
    assert opt_weights['optimization'] == 1.0
    assert opt_weights['structure'] == 0.0
    assert opt_weights['parent'] == 0.0
    assert opt_weights['exploration'] == 0.0
    
    # Test structure_only config
    struct_only = configs['structure_only']
    struct_weights = struct_only['reward_weights']
    assert struct_weights['optimization'] == 0.0
    assert struct_weights['structure'] == 1.0
    assert struct_weights['parent'] == 0.0
    assert struct_weights['exploration'] == 0.0
    
    # Test no_optimization config preserves other weights
    no_opt = configs['no_optimization']
    no_opt_weights = no_opt['reward_weights']
    assert no_opt_weights['optimization'] == 0.0
    assert no_opt_weights['structure'] == 0.5  # Original weight preserved
    assert no_opt_weights['parent'] == 0.3    # Original weight preserved
    assert no_opt_weights['exploration'] == 0.1  # Original weight preserved


if __name__ == "__main__":
    pytest.main([__file__])