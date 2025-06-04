"""
Test suite for Multi-Component Verifiable Rewards.

Tests the reward system for dual-objective ACBO using proper pytest structure
and integration with existing ACBO infrastructure.
"""

import pytest
import jax.numpy as jnp
import pyrsistent as pyr
from unittest.mock import Mock

# Import the rewards system
from causal_bayes_opt.acquisition.rewards import (
    RewardComponents,
    compute_verifiable_reward,
    analyze_reward_trends,
    validate_reward_config,
    create_default_reward_config,
    _compute_optimization_reward,
    _compute_structure_discovery_reward,
    _compute_parent_intervention_reward,
    _compute_exploration_bonus,
)

# Import supporting infrastructure
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.avici_integration.parent_set.posterior import create_parent_set_posterior


class TestRewardComponents:
    """Test the RewardComponents dataclass."""
    
    def test_creation(self):
        """Test basic RewardComponents creation."""
        components = RewardComponents(
            optimization_reward=0.5,
            structure_discovery_reward=0.3,
            parent_intervention_reward=0.2,
            exploration_bonus=0.1,
            total_reward=1.1
        )
        
        assert components.optimization_reward == 0.5
        assert components.structure_discovery_reward == 0.3
        assert components.parent_intervention_reward == 0.2
        assert components.exploration_bonus == 0.1
        assert components.total_reward == 1.1
    
    def test_validation(self):
        """Test validation of reward components."""
        # Valid components should work
        RewardComponents(
            optimization_reward=0.5,
            structure_discovery_reward=0.3,
            parent_intervention_reward=0.2,
            exploration_bonus=0.1,
            total_reward=1.1
        )
        
        # Invalid components should fail
        with pytest.raises(ValueError, match="must be finite"):
            RewardComponents(
                optimization_reward=float('inf'),
                structure_discovery_reward=0.3,
                parent_intervention_reward=0.2,
                exploration_bonus=0.1,
                total_reward=1.1
            )
    
    def test_summary(self):
        """Test reward summary generation."""
        components = RewardComponents(
            optimization_reward=0.4,
            structure_discovery_reward=0.3,
            parent_intervention_reward=0.2,
            exploration_bonus=0.1,
            total_reward=1.0
        )
        
        summary = components.summary()
        
        assert summary['total_reward'] == 1.0
        assert summary['optimization_component'] == 0.4
        assert summary['structure_component'] == 0.3
        assert abs(summary['optimization_fraction'] - 0.4) < 1e-6


class TestOptimizationReward:
    """Test optimization reward computation."""
    
    @pytest.fixture
    def mock_state(self):
        """Create a mock acquisition state."""
        state = Mock()
        state.best_value = 5.0
        return state
    
    def test_improvement(self, mock_state):
        """Test reward for target variable improvement."""
        outcome = create_sample({'Y': 7.0, 'X': 1.0})
        reward = _compute_optimization_reward(mock_state, outcome, 'Y')
        
        assert reward > 0  # Should be positive for improvement
        assert -1 <= reward <= 1  # Should be bounded by tanh
    
    def test_deterioration(self, mock_state):
        """Test reward for target variable deterioration."""
        outcome = create_sample({'Y': 3.0, 'X': 1.0})
        reward = _compute_optimization_reward(mock_state, outcome, 'Y')
        
        assert reward < 0  # Should be negative for deterioration
        assert -1 <= reward <= 1  # Should be bounded by tanh
    
    def test_missing_target(self, mock_state):
        """Test behavior when target variable is missing."""
        outcome = create_sample({'X': 1.0})  # No Y
        reward = _compute_optimization_reward(mock_state, outcome, 'Y')
        
        assert reward == 0.0  # Should return 0 when target is missing


class TestStructureDiscoveryReward:
    """Test structure discovery reward computation."""
    
    @pytest.fixture
    def posterior_high_uncertainty(self):
        """Create a posterior with high uncertainty."""
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Z'])]
        probs = jnp.array([0.4, 0.35, 0.25])  # High uncertainty
        return create_parent_set_posterior('Y', parent_sets, probs)
    
    @pytest.fixture
    def posterior_low_uncertainty(self):
        """Create a posterior with low uncertainty."""
        parent_sets = [frozenset(), frozenset(['X']), frozenset(['Z'])]
        probs = jnp.array([0.9, 0.08, 0.02])  # Low uncertainty
        return create_parent_set_posterior('Y', parent_sets, probs)
    
    def test_uncertainty_reduction(self, posterior_high_uncertainty, posterior_low_uncertainty):
        """Test reward for uncertainty reduction."""
        reward = _compute_structure_discovery_reward(
            posterior_high_uncertainty, posterior_low_uncertainty
        )
        
        assert reward > 0  # Should be positive for uncertainty reduction
        assert 0 <= reward <= 1  # Should be bounded [0, 1]
    
    def test_uncertainty_increase(self, posterior_low_uncertainty, posterior_high_uncertainty):
        """Test reward when uncertainty increases."""
        reward = _compute_structure_discovery_reward(
            posterior_low_uncertainty, posterior_high_uncertainty
        )
        
        assert reward == 0.0  # Should be zero (clipped) for uncertainty increase
    
    def test_target_mismatch(self, posterior_high_uncertainty):
        """Test behavior with mismatched target variables."""
        # Create posterior with different target
        parent_sets = [frozenset(), frozenset(['Y'])]
        probs = jnp.array([0.3, 0.7])
        posterior_different = create_parent_set_posterior('X', parent_sets, probs)
        
        reward = _compute_structure_discovery_reward(
            posterior_high_uncertainty, posterior_different
        )
        
        assert reward == 0.0  # Should return 0 for mismatched targets


class TestParentInterventionReward:
    """Test parent intervention reward computation."""
    
    def test_high_probability_parent(self):
        """Test reward for intervening on likely parents."""
        intervention = pyr.m(**{
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 5.0}
        })
        marginal_probs = {'X': 0.8, 'Z': 0.2}
        
        reward = _compute_parent_intervention_reward(intervention, marginal_probs)
        
        assert abs(reward - 0.8) < 1e-6  # Should equal the marginal probability
    
    def test_non_perfect_intervention(self):
        """Test that non-perfect interventions get zero reward."""
        intervention = pyr.m(**{
            'type': 'soft',
            'targets': frozenset(['X']),
            'values': {'X': 5.0}
        })
        marginal_probs = {'X': 0.8}
        
        reward = _compute_parent_intervention_reward(intervention, marginal_probs)
        
        assert reward == 0.0  # Should be zero for non-perfect interventions


class TestExplorationBonus:
    """Test exploration bonus computation."""
    
    @pytest.fixture
    def empty_buffer(self):
        """Create a mock buffer with no interventions."""
        buffer = Mock()
        buffer.get_interventions.return_value = []
        return buffer
    
    def test_first_intervention(self, empty_buffer):
        """Test exploration bonus for first intervention."""
        intervention = pyr.m(**{
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 5.0}
        })
        
        bonus = _compute_exploration_bonus(intervention, empty_buffer, weight=0.2)
        
        assert bonus == 0.2  # Should get full bonus for first intervention


class TestConfiguration:
    """Test reward configuration functionality."""
    
    def test_default_config(self):
        """Test creating default reward configuration."""
        config = create_default_reward_config()
        
        assert 'reward_weights' in config
        weights = config['reward_weights']
        assert weights['optimization'] == 1.0
        assert weights['structure'] == 0.5
        assert weights['parent'] == 0.3
        assert weights['exploration'] == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        valid_config = create_default_reward_config()
        assert validate_reward_config(valid_config) is True
        
        # Invalid config
        invalid_config = pyr.m(**{
            'reward_weights': {
                'optimization': 1.0,
                'structure': 0.5
                # Missing required weights
            }
        })
        assert validate_reward_config(invalid_config) is False
    
    def test_custom_config(self):
        """Test creating custom reward configuration."""
        config = create_default_reward_config(
            optimization_weight=2.0,
            structure_weight=0.3
        )
        
        weights = config['reward_weights']
        assert weights['optimization'] == 2.0
        assert weights['structure'] == 0.3


class TestRewardTrends:
    """Test reward trends analysis functionality."""
    
    @pytest.fixture
    def reward_history(self):
        """Create synthetic reward history for testing."""
        history = []
        for i in range(10):
            components = RewardComponents(
                optimization_reward=0.1 * i + 0.5,  # Increasing trend
                structure_discovery_reward=0.3 - 0.01 * i,  # Decreasing trend
                parent_intervention_reward=0.2,
                exploration_bonus=0.1,
                total_reward=1.0
            )
            history.append(components)
        return history
    
    def test_trends_analysis(self, reward_history):
        """Test reward trends analysis."""
        trends = analyze_reward_trends(reward_history, window_size=5)
        
        assert 'optimization_trend' in trends
        assert 'structure_trend' in trends
        assert trends['optimization_trend'] > 0  # Should have positive trend
        assert trends['structure_trend'] < 0    # Should have negative trend
        assert trends['n_samples'] == 10
    
    def test_empty_history(self):
        """Test trends analysis with empty history."""
        trends = analyze_reward_trends([])
        assert trends == {}
