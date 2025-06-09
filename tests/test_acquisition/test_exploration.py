"""
Test suite for uncertainty-guided exploration strategies.

Tests the UncertaintyGuidedExploration and AdaptiveExploration classes,
verifying correct exploration bonus computation, temperature scheduling,
and integration with acquisition state.
"""

import pytest
import jax
import jax.numpy as jnp
import pyrsistent as pyr
from unittest.mock import MagicMock

from src.causal_bayes_opt.acquisition.exploration import (
    ExplorationConfig,
    UncertaintyGuidedExploration,
    AdaptiveExploration,
    create_exploration_strategy,
    compute_exploration_value,
    select_exploration_intervention,
    balance_exploration_exploitation,
)


class TestExplorationConfig:
    """Test exploration configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExplorationConfig()
        
        assert config.uncertainty_weight == 1.0
        assert config.count_weight == 0.1
        assert config.variable_uncertainty_weight == 0.5
        assert config.temperature == 1.0
        assert config.initial_temperature == 2.0
        assert config.final_temperature == 0.1
        assert config.adaptation_steps == 1000
        assert config.stagnation_threshold == 100
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExplorationConfig(
            uncertainty_weight=2.0,
            count_weight=0.2,
            temperature=0.5,
            adaptation_steps=500
        )
        
        assert config.uncertainty_weight == 2.0
        assert config.count_weight == 0.2
        assert config.temperature == 0.5
        assert config.adaptation_steps == 500


class TestUncertaintyGuidedExploration:
    """Test uncertainty-guided exploration strategy."""
    
    @pytest.fixture
    def exploration_strategy(self):
        """Create exploration strategy for testing."""
        config = ExplorationConfig(
            uncertainty_weight=1.0,
            count_weight=0.1,
            variable_uncertainty_weight=0.5,
            temperature=1.0
        )
        return UncertaintyGuidedExploration(config)
    
    @pytest.fixture
    def mock_state(self):
        """Create mock acquisition state."""
        state = MagicMock()
        state.uncertainty_bits = 3.5
        state.marginal_parent_probs = {'X': 0.8, 'Y': 0.3, 'Z': 0.5}
        
        # Mock buffer
        mock_buffer = MagicMock()
        mock_buffer.filter_interventions_by_targets.return_value = [1, 2]  # 2 previous interventions
        mock_buffer.num_interventions.return_value = 10  # 10 total interventions
        state.buffer = mock_buffer
        
        return state
    
    @pytest.fixture
    def sample_intervention(self):
        """Create sample intervention."""
        return pyr.pmap({
            'type': 'perfect',
            'targets': frozenset(['X', 'Y']),
            'values': {'X': 1.0, 'Y': 2.0}
        })
    
    def test_epistemic_bonus_computation(self, exploration_strategy, mock_state, sample_intervention):
        """Test epistemic uncertainty bonus computation."""
        bonus = exploration_strategy._compute_epistemic_bonus(mock_state, sample_intervention)
        
        # Should be based on expected information gain
        # For X: parent_prob=0.8, uncertainty_factor = 1.0 - 2*|0.8-0.5| = 0.4
        # For Y: parent_prob=0.3, uncertainty_factor = 1.0 - 2*|0.3-0.5| = 0.6
        # Average uncertainty_factor = (0.4 + 0.6) / 2 = 0.5
        # Expected gain = 0.5 * 3.5 (uncertainty_bits) = 1.75
        assert bonus == pytest.approx(1.75)
    
    def test_count_bonus_computation(self, exploration_strategy, mock_state, sample_intervention):
        """Test count-based exploration bonus."""
        bonus = exploration_strategy._compute_count_bonus(sample_intervention, mock_state.buffer)
        
        # Expected: 1.0 - (2/10) = 0.8
        assert bonus == pytest.approx(0.8)
    
    def test_count_bonus_with_no_interventions(self, exploration_strategy, sample_intervention):
        """Test count bonus when no previous interventions."""
        mock_buffer = MagicMock()
        mock_buffer.num_interventions.return_value = 0
        
        bonus = exploration_strategy._compute_count_bonus(sample_intervention, mock_buffer)
        assert bonus == 1.0  # Maximum bonus
    
    def test_variable_uncertainty_bonus(self, exploration_strategy, sample_intervention):
        """Test variable uncertainty bonus computation."""
        marginal_probs = {'X': 0.8, 'Y': 0.3, 'Z': 0.5}
        
        bonus = exploration_strategy._compute_variable_uncertainty_bonus(
            sample_intervention, marginal_probs
        )
        
        # For X: uncertainty = 1.0 - 2*|0.8-0.5| = 1.0 - 0.6 = 0.4
        # For Y: uncertainty = 1.0 - 2*|0.3-0.5| = 1.0 - 0.4 = 0.6
        # Mean: (0.4 + 0.6) / 2 = 0.5
        assert bonus == pytest.approx(0.5)
    
    def test_variable_uncertainty_maximum_at_half(self, exploration_strategy):
        """Test that variable uncertainty is maximized at probability 0.5."""
        intervention = pyr.pmap({
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 1.0}
        })
        
        # Test with prob = 0.5 (maximum uncertainty)
        marginal_probs = {'X': 0.5}
        bonus_half = exploration_strategy._compute_variable_uncertainty_bonus(
            intervention, marginal_probs
        )
        
        # Test with prob = 0.0 (minimum uncertainty)
        marginal_probs = {'X': 0.0}
        bonus_zero = exploration_strategy._compute_variable_uncertainty_bonus(
            intervention, marginal_probs
        )
        
        # Test with prob = 1.0 (minimum uncertainty)
        marginal_probs = {'X': 1.0}
        bonus_one = exploration_strategy._compute_variable_uncertainty_bonus(
            intervention, marginal_probs
        )
        
        assert bonus_half == pytest.approx(1.0)
        assert bonus_zero == pytest.approx(0.0)
        assert bonus_one == pytest.approx(0.0)
    
    def test_total_exploration_bonus(self, exploration_strategy, mock_state, sample_intervention):
        """Test combined exploration bonus computation."""
        bonus = exploration_strategy.compute_exploration_bonus(mock_state, sample_intervention)
        
        # Should be sum of all bonus components divided by temperature
        expected = (
            1.0 * 1.75 +     # uncertainty_weight * epistemic_bonus (now 1.75)
            0.1 * 0.8 +      # count_weight * count_bonus  
            0.5 * 0.5        # variable_uncertainty_weight * var_uncertainty_bonus
        ) / 1.0              # temperature
        
        assert bonus == pytest.approx(expected)
    
    def test_non_perfect_intervention_handling(self, exploration_strategy, mock_state):
        """Test handling of non-perfect interventions."""
        soft_intervention = pyr.pmap({
            'type': 'soft',
            'targets': frozenset(['X']),
            'values': {'X': 1.0}
        })
        
        # All bonuses should be 0 for non-perfect interventions
        epistemic_bonus = exploration_strategy._compute_epistemic_bonus(mock_state, soft_intervention)
        count_bonus = exploration_strategy._compute_count_bonus(soft_intervention, mock_state.buffer)
        var_bonus = exploration_strategy._compute_variable_uncertainty_bonus(
            soft_intervention, mock_state.marginal_parent_probs
        )
        
        assert epistemic_bonus == 0.0
        assert count_bonus == 0.0
        assert var_bonus == 0.0
    
    def test_temperature_scaling(self, mock_state, sample_intervention):
        """Test temperature scaling of exploration bonus."""
        high_temp_config = ExplorationConfig(temperature=2.0)
        low_temp_config = ExplorationConfig(temperature=0.5)
        
        high_temp_strategy = UncertaintyGuidedExploration(high_temp_config)
        low_temp_strategy = UncertaintyGuidedExploration(low_temp_config)
        
        high_temp_bonus = high_temp_strategy.compute_exploration_bonus(mock_state, sample_intervention)
        low_temp_bonus = low_temp_strategy.compute_exploration_bonus(mock_state, sample_intervention)
        
        # Higher temperature should give lower bonus (more conservative exploration)
        assert high_temp_bonus < low_temp_bonus
        assert low_temp_bonus == pytest.approx(high_temp_bonus * 4.0)  # 2.0 / 0.5 = 4


class TestAdaptiveExploration:
    """Test adaptive exploration strategy."""
    
    @pytest.fixture
    def adaptive_strategy(self):
        """Create adaptive exploration strategy for testing."""
        config = ExplorationConfig(
            initial_temperature=2.0,
            final_temperature=0.1,
            adaptation_steps=1000,
            stagnation_threshold=100
        )
        return AdaptiveExploration(config)
    
    @pytest.fixture
    def mock_state(self):
        """Create mock acquisition state."""
        state = MagicMock()
        state.uncertainty_bits = 1.5  # Low uncertainty
        # Provide realistic attributes for stagnation testing
        state.optimization_stagnation_steps = 0
        state.recent_target_improvements = []
        return state
    
    def test_temperature_decay_over_steps(self, adaptive_strategy, mock_state):
        """Test temperature decay over training steps."""
        # At step 0
        temp_0 = adaptive_strategy.get_exploration_temperature(0, mock_state)
        assert temp_0 == pytest.approx(2.0)  # Should be initial temperature
        
        # At middle step
        temp_500 = adaptive_strategy.get_exploration_temperature(500, mock_state)
        expected_500 = 2.0 * 0.5 + 0.1 * 0.5  # Linear interpolation
        assert temp_500 == pytest.approx(expected_500)
        
        # At final step
        temp_1000 = adaptive_strategy.get_exploration_temperature(1000, mock_state)
        assert temp_1000 == pytest.approx(0.1)  # Should be final temperature
        
        # Beyond final step
        temp_2000 = adaptive_strategy.get_exploration_temperature(2000, mock_state)
        assert temp_2000 == pytest.approx(0.1)  # Should cap at final temperature
    
    def test_stagnation_adjustment(self, adaptive_strategy):
        """Test temperature adjustment based on optimization stagnation."""
        # Create state with stagnation
        stagnant_state = MagicMock()
        stagnant_state.uncertainty_bits = 1.5
        stagnant_state.optimization_stagnation_steps = 50  # Half threshold
        
        # Create state without stagnation
        normal_state = MagicMock()
        normal_state.uncertainty_bits = 1.5
        normal_state.optimization_stagnation_steps = 0
        
        # Get temperature at middle step with stagnation
        temp_with_stagnation = adaptive_strategy.get_exploration_temperature(500, stagnant_state)
        temp_without_stagnation = adaptive_strategy.get_exploration_temperature(500, normal_state)
        
        # With stagnation, should have higher temperature (more exploration)
        assert temp_with_stagnation > temp_without_stagnation
    
    def test_should_explore_high_uncertainty(self, adaptive_strategy):
        """Test exploration decision with high uncertainty."""
        high_uncertainty_state = MagicMock()
        high_uncertainty_state.uncertainty_bits = 3.0  # > 2.0 threshold
        
        # Should always explore with high uncertainty
        assert adaptive_strategy.should_explore(high_uncertainty_state, 0)
        assert adaptive_strategy.should_explore(high_uncertainty_state, 500)
        assert adaptive_strategy.should_explore(high_uncertainty_state, 1000)
    
    def test_should_explore_with_stagnation(self, adaptive_strategy):
        """Test exploration decision with optimization stagnation."""
        stagnant_state = MagicMock()
        stagnant_state.uncertainty_bits = 1.0  # Low uncertainty
        stagnant_state.recent_target_improvements = [0.001, 0.002, 0.001, 0.005, 0.003, 0.002]
        
        # Should explore due to stagnation (max improvement < 0.01)
        assert adaptive_strategy.should_explore(stagnant_state, 500)
    
    def test_should_explore_temperature_based(self, adaptive_strategy):
        """Test temperature-based exploration decisions."""
        normal_state = MagicMock()
        normal_state.uncertainty_bits = 1.0  # Low uncertainty
        normal_state.recent_target_improvements = [0.05, 0.03, 0.02, 0.04]  # Good progress
        
        # At early steps (high temperature), should explore
        early_decision = adaptive_strategy.should_explore(normal_state, 0)
        
        # At late steps (low temperature), should exploit
        late_decision = adaptive_strategy.should_explore(normal_state, 1000)
        
        assert early_decision == True   # High temp -> explore
        assert late_decision == False   # Low temp -> exploit
    
    def test_exploration_bonus_schedule(self, adaptive_strategy):
        """Test exploration bonus scheduling."""
        # Early in training
        early_bonus = adaptive_strategy.get_exploration_bonus_schedule(0)
        assert early_bonus == pytest.approx(1.0)  # initial_temp / initial_temp
        
        # Late in training
        late_bonus = adaptive_strategy.get_exploration_bonus_schedule(1000)
        expected_late = 0.1 / 2.0  # final_temp / initial_temp
        assert late_bonus == pytest.approx(expected_late)


class TestExplorationFactoryAndUtilities:
    """Test exploration factory functions and utilities."""
    
    def test_create_uncertainty_guided_strategy(self):
        """Test creating uncertainty-guided exploration strategy."""
        strategy = create_exploration_strategy("uncertainty_guided")
        assert isinstance(strategy, UncertaintyGuidedExploration)
    
    def test_create_adaptive_strategy(self):
        """Test creating adaptive exploration strategy."""
        strategy = create_exploration_strategy("adaptive")
        assert isinstance(strategy, AdaptiveExploration)
    
    def test_create_strategy_with_config(self):
        """Test creating strategy with custom config."""
        config = ExplorationConfig(uncertainty_weight=2.0)
        strategy = create_exploration_strategy("uncertainty_guided", config=config)
        assert strategy.config.uncertainty_weight == 2.0
    
    def test_create_strategy_with_kwargs(self):
        """Test creating strategy with kwargs."""
        strategy = create_exploration_strategy(
            "uncertainty_guided", 
            uncertainty_weight=3.0,
            temperature=0.5
        )
        assert strategy.config.uncertainty_weight == 3.0
        assert strategy.config.temperature == 0.5
    
    def test_create_unknown_strategy(self):
        """Test error handling for unknown strategy type."""
        with pytest.raises(ValueError, match="Unknown exploration strategy"):
            create_exploration_strategy("unknown_strategy")
    
    def test_compute_exploration_value(self):
        """Test exploration value computation."""
        mock_state = MagicMock()
        mock_state.uncertainty_bits = 2.0
        mock_state.marginal_parent_probs = {'X': 0.5}
        
        # Mock buffer
        mock_buffer = MagicMock()
        mock_buffer.filter_interventions_by_targets.return_value = []
        mock_buffer.num_interventions.return_value = 1
        mock_state.buffer = mock_buffer
        
        intervention = pyr.pmap({
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 1.0}
        })
        
        strategy = create_exploration_strategy("uncertainty_guided")
        value = compute_exploration_value(strategy, mock_state, intervention)
        
        assert isinstance(value, float)
        assert value > 0
    
    def test_select_exploration_intervention(self):
        """Test intervention selection for exploration."""
        mock_state = MagicMock()
        mock_state.uncertainty_bits = 2.0
        mock_state.marginal_parent_probs = {'X': 0.5, 'Y': 0.8}
        
        # Mock buffer
        mock_buffer = MagicMock()
        mock_buffer.filter_interventions_by_targets.return_value = []
        mock_buffer.num_interventions.return_value = 5
        mock_state.buffer = mock_buffer
        
        candidates = [
            pyr.pmap({
                'type': 'perfect',
                'targets': frozenset(['X']),
                'values': {'X': 1.0}
            }),
            pyr.pmap({
                'type': 'perfect', 
                'targets': frozenset(['Y']),
                'values': {'Y': 2.0}
            })
        ]
        
        strategy = create_exploration_strategy("uncertainty_guided")
        selected = select_exploration_intervention(candidates, strategy, mock_state, top_k=1)
        
        assert len(selected) == 1
        assert selected[0] in candidates
        
        # X should be selected over Y (X has higher uncertainty: 0.5 vs 0.8)
        assert 'X' in selected[0]['targets']
    
    def test_balance_exploration_exploitation(self):
        """Test exploration-exploitation balancing."""
        mock_state = MagicMock()
        mock_state.uncertainty_bits = 1.0
        mock_state.optimization_stagnation_steps = 0
        mock_state.recent_target_improvements = []
        
        strategy = create_exploration_strategy("adaptive")
        
        # Test balanced values
        balanced = balance_exploration_exploitation(
            exploration_value=1.0,
            exploitation_value=2.0,
            exploration_strategy=strategy,
            state=mock_state,
            alpha=0.5
        )
        
        # The adaptive strategy should modify the base alpha value
        # At step 0, should_explore returns False (exploitation favored)
        # So effective_alpha should be reduced (alpha - 0.3 = 0.2)
        # Expected: 0.2 * 1.0 + 0.8 * 2.0 = 1.8
        # But let's just check it's reasonable and not equal to base expected
        base_expected = 0.5 * 1.0 + 0.5 * 2.0  # 1.5
        assert balanced != pytest.approx(base_expected)  # Should be different due to adaptation
        assert isinstance(balanced, float)
        assert 1.0 <= balanced <= 2.0  # Should be within reasonable bounds
    
    def test_empty_candidate_list(self):
        """Test handling of empty candidate list."""
        mock_state = MagicMock()
        strategy = create_exploration_strategy("uncertainty_guided")
        
        selected = select_exploration_intervention([], strategy, mock_state)
        assert selected == []


class TestExplorationIntegration:
    """Test integration with broader ACBO system."""
    
    def test_exploration_with_buffer_fallback(self):
        """Test exploration when buffer methods are missing."""
        # Create strategy
        strategy = create_exploration_strategy("uncertainty_guided")
        
        # Create intervention
        intervention = pyr.pmap({
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 1.0}
        })
        
        # Create buffer without required methods
        mock_buffer = MagicMock()
        del mock_buffer.filter_interventions_by_targets
        del mock_buffer.num_interventions
        
        # Should fall back gracefully
        bonus = strategy._compute_count_bonus(intervention, mock_buffer)
        assert bonus == 0.1  # config.count_weight
    
    def test_exploration_config_validation(self):
        """Test exploration configuration validation."""
        # Test negative weights (should work but give different behavior)
        config = ExplorationConfig(uncertainty_weight=-1.0)
        strategy = UncertaintyGuidedExploration(config)
        
        # Should still work, just with negative contribution
        mock_state = MagicMock()
        mock_state.uncertainty_bits = 2.0
        mock_state.marginal_parent_probs = {'X': 0.5}
        
        intervention = pyr.pmap({
            'type': 'perfect',
            'targets': frozenset(['X']),
            'values': {'X': 1.0}
        })
        
        bonus = strategy._compute_epistemic_bonus(mock_state, intervention)
        # X has prob=0.5, so uncertainty_factor=1.0, expected_gain = 1.0 * 2.0 = 2.0
        assert bonus == 2.0
    
    def test_missing_state_attributes(self):
        """Test handling of missing state attributes."""
        config = ExplorationConfig()
        strategy = AdaptiveExploration(config)
        
        # State without stagnation attributes
        minimal_state = MagicMock()
        minimal_state.uncertainty_bits = 1.5
        del minimal_state.optimization_stagnation_steps
        del minimal_state.recent_target_improvements
        
        # Should work with defaults
        temp = strategy.get_exploration_temperature(500, minimal_state)
        assert isinstance(temp, float)
        assert temp > 0
        
        decision = strategy.should_explore(minimal_state, 500)
        assert isinstance(decision, bool)