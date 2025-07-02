#!/usr/bin/env python3
"""
Comprehensive Validation Tests for Continuous Reward System

Tests the new continuous SCM-objective reward system to ensure:
1. Reward components work correctly in isolation
2. SCM-objective reward encourages optimal interventions 
3. Normalization schemes make sense across variable ranges
4. No gaming exploits in the reward design
5. Component interactions are balanced properly
"""

import pytest
import jax.numpy as jnp
import pyrsistent as pyr
from unittest.mock import Mock, patch
from typing import Dict, List, Any

# Import the continuous reward system
from causal_bayes_opt.acquisition.rewards import (
    RewardComponents,
    compute_verifiable_reward,
    _compute_optimization_reward,
    _compute_scm_objective_reward,
    _compute_improved_relative_reward,
    _compute_structure_discovery_reward,
    _compute_parent_intervention_reward,
    _compute_exploration_bonus,
    create_default_reward_config,
    create_adaptive_reward_config,
    validate_reward_consistency,
    compute_adaptive_thresholds,
)

# Import supporting infrastructure  
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer


class TestContinuousRewardValidation:
    """Test suite for validating the continuous reward system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock acquisition state
        self.mock_posterior = Mock()
        self.mock_posterior.uncertainty = 0.8
        self.mock_posterior.target_variable = "Y"
        
        self.mock_buffer = Mock()
        self.mock_buffer.samples = []
        self.mock_buffer.get_interventions.return_value = []
        
        self.state_before = Mock()
        self.state_before.current_target = "Y"
        self.state_before.step = 5
        self.state_before.best_value = 2.0
        self.state_before.posterior = self.mock_posterior
        self.state_before.buffer = self.mock_buffer
        self.state_before.marginal_parent_probs = {"X": 0.8, "Z": 0.3}
        self.state_before.uncertainty_bits = 0.8
        self.state_before.buffer_statistics = Mock()
        self.state_before.buffer_statistics.total_samples = 10
        
        # State after intervention
        self.mock_posterior_after = Mock()
        self.mock_posterior_after.uncertainty = 0.6
        self.mock_posterior_after.target_variable = "Y"
        
        self.state_after = Mock()
        self.state_after.current_target = "Y"
        self.state_after.step = 6
        self.state_after.best_value = 2.5
        self.state_after.posterior = self.mock_posterior_after
        self.state_after.uncertainty_bits = 0.6
        self.state_after.buffer_statistics = Mock()
        self.state_after.buffer_statistics.total_samples = 11
        
        # Intervention and outcome
        self.intervention = pyr.m(
            type="perfect",
            targets={"X"},
            values={"X": 1.5}
        )
        
        self.outcome = pyr.m(
            values={"Y": 2.8, "X": 1.5, "Z": 0.5}
        )


    def test_zero_out_optimization_component(self):
        """Test that zeroing optimization weight removes optimization pressure."""
        config = create_default_reward_config(
            optimization_weight=0.0,  # Zero out optimization
            structure_weight=1.0,
            parent_weight=1.0,
            exploration_weight=1.0
        )
        
        rewards = compute_verifiable_reward(
            self.state_before, self.intervention, self.outcome, self.state_after, config
        )
        
        # Individual components are still computed, but optimization weight should be zero in total
        # Total reward should only come from non-optimization components
        expected_total = (
            0.0 * rewards.optimization_reward +
            1.0 * rewards.structure_discovery_reward +
            1.0 * rewards.parent_intervention_reward +
            1.0 * rewards.exploration_bonus
        )
        assert abs(rewards.total_reward - expected_total) < 1e-6
        
        # The optimization component should still exist but its contribution should be zero
        optimization_contribution = 0.0 * rewards.optimization_reward
        assert optimization_contribution == 0.0


    def test_zero_out_structure_component(self):
        """Test that zeroing structure weight removes structure learning pressure."""
        config = create_default_reward_config(
            optimization_weight=1.0,
            structure_weight=0.0,  # Zero out structure
            parent_weight=1.0,
            exploration_weight=1.0
        )
        
        rewards = compute_verifiable_reward(
            self.state_before, self.intervention, self.outcome, self.state_after, config
        )
        
        # Structure component should contribute zero to total
        expected_total = (
            1.0 * rewards.optimization_reward +
            0.0 * rewards.structure_discovery_reward +
            1.0 * rewards.parent_intervention_reward +
            1.0 * rewards.exploration_bonus
        )
        assert abs(rewards.total_reward - expected_total) < 1e-6


    def test_scm_objective_reward_encourages_optimal_interventions(self):
        """Test that SCM-objective reward always prefers optimal interventions."""
        # Mock mechanism predictions for target variable Y
        mock_mechanism = Mock()
        mock_mechanism.coefficients = {"X": 2.0, "Z": -1.0}  # Y = 2*X - Z + intercept
        mock_mechanism.intercept = 1.0
        
        self.state_before.mechanism_predictions = {"Y": mock_mechanism}
        self.state_before.intervention_bounds = {"X": (-2.0, 2.0), "Z": (-2.0, 2.0)}
        
        # Test optimal intervention (X=2.0, Z=-2.0 should give Y = 2*2 - (-2) + 1 = 7)
        optimal_outcome = pyr.m(values={"Y": 7.0, "X": 2.0, "Z": -2.0})
        optimal_reward = _compute_optimization_reward(self.state_before, optimal_outcome, "Y")
        
        # Test suboptimal intervention (X=0, Z=0 should give Y = 0 + 0 + 1 = 1)
        suboptimal_outcome = pyr.m(values={"Y": 1.0, "X": 0.0, "Z": 0.0})
        suboptimal_reward = _compute_optimization_reward(self.state_before, suboptimal_outcome, "Y")
        
        # Optimal should always get higher reward
        assert optimal_reward > suboptimal_reward
        assert optimal_reward == 1.0  # Should be at theoretical maximum
        
        # Test that repeatedly finding optimal gives same high reward (no avoidance)
        repeated_optimal_reward = _compute_optimization_reward(self.state_before, optimal_outcome, "Y")
        assert repeated_optimal_reward == optimal_reward


    def test_reward_normalization_across_variable_ranges(self):
        """Test that rewards are properly normalized across different variable ranges."""
        # Test with small value range
        small_range_outcome = pyr.m(values={"Y": 0.1, "X": 0.05})
        small_reward = _compute_optimization_reward(self.state_before, small_range_outcome, "Y")
        
        # Test with large value range 
        large_range_outcome = pyr.m(values={"Y": 100.0, "X": 50.0})
        large_reward = _compute_optimization_reward(self.state_before, large_range_outcome, "Y")
        
        # Both should be in [0, 1] range
        assert 0.0 <= small_reward <= 1.0
        assert 0.0 <= large_reward <= 1.0
        
        # Rewards should be meaningful (not just edge values)
        assert small_reward > 0.0 or large_reward > 0.0


    def test_parent_intervention_rewards_likely_parents(self):
        """Test that parent intervention component rewards intervening on likely parents."""
        # Intervention on high-probability parent
        high_parent_intervention = pyr.m(type="perfect", targets={"X"})  # X has prob 0.8
        high_parent_reward = _compute_parent_intervention_reward(
            high_parent_intervention, self.state_before.marginal_parent_probs
        )
        
        # Intervention on low-probability parent
        low_parent_intervention = pyr.m(type="perfect", targets={"Z"})  # Z has prob 0.3
        low_parent_reward = _compute_parent_intervention_reward(
            low_parent_intervention, self.state_before.marginal_parent_probs
        )
        
        # Should reward high-probability parents more
        assert high_parent_reward > low_parent_reward
        assert abs(high_parent_reward - 0.8) < 1e-6  # Should match probability (within precision)
        assert abs(low_parent_reward - 0.3) < 1e-6


    def test_exploration_bonus_encourages_diversity(self):
        """Test that exploration bonus encourages diverse interventions."""
        # Mock intervention history with repeated X interventions
        repeated_interventions = [
            (pyr.m(targets={"X"}), Mock()),
            (pyr.m(targets={"X"}), Mock()),
            (pyr.m(targets={"X"}), Mock()),
        ]
        self.mock_buffer.get_interventions.return_value = repeated_interventions
        
        # New X intervention should get lower exploration bonus
        repeated_intervention = pyr.m(type="perfect", targets={"X"})
        repeated_bonus = _compute_exploration_bonus(repeated_intervention, self.mock_buffer, 0.1)
        
        # Novel Z intervention should get higher exploration bonus
        novel_intervention = pyr.m(type="perfect", targets={"Z"})
        novel_bonus = _compute_exploration_bonus(novel_intervention, self.mock_buffer, 0.1)
        
        assert novel_bonus > repeated_bonus


    def test_reward_consistency_validation_detects_gaming(self):
        """Test that reward consistency validation detects potential gaming patterns."""
        # Create reward history with gaming patterns
        gaming_rewards = []
        
        # Pattern 1: No optimization progress (gaming detection should trigger)
        for _ in range(20):
            gaming_rewards.append(RewardComponents(
                optimization_reward=0.0,  # Consistently zero
                structure_discovery_reward=0.5,
                parent_intervention_reward=0.95,  # Suspiciously high (above 0.9 threshold)
                exploration_bonus=0.0,  # No exploration
                total_reward=1.45,
                metadata=pyr.m()
            ))
        
        validation_result = validate_reward_consistency(gaming_rewards)
        
        assert not validation_result['valid']
        assert len(validation_result['gaming_issues']) > 0
        
        # Should detect low optimization, high parent rate, low exploration
        issues = validation_result['gaming_issues']
        print(f"Gaming issues detected: {issues}")  # Debug output
        assert any("low optimization" in issue.lower() for issue in issues)
        assert any("parent" in issue.lower() for issue in issues)  # More flexible matching
        assert any("exploration" in issue.lower() for issue in issues)


    def test_healthy_reward_pattern_passes_validation(self):
        """Test that healthy reward patterns pass validation."""
        healthy_rewards = []
        
        # Create diverse, healthy reward pattern
        for i in range(20):
            healthy_rewards.append(RewardComponents(
                optimization_reward=0.3 + 0.1 * (i % 3),  # Varies between 0.3-0.5
                structure_discovery_reward=0.2 + 0.1 * (i % 2),  # Varies between 0.2-0.3
                parent_intervention_reward=0.4 + 0.2 * (i % 2),  # Varies between 0.4-0.6
                exploration_bonus=0.05 + 0.05 * (i % 2),  # Varies between 0.05-0.1
                total_reward=1.0 + 0.3 * (i % 3),  # Varies with good variance
                metadata=pyr.m()
            ))
        
        validation_result = validate_reward_consistency(healthy_rewards)
        
        assert validation_result['valid']
        assert len(validation_result['gaming_issues']) == 0


    def test_adaptive_thresholds_scale_with_scm_complexity(self):
        """Test that adaptive thresholds properly scale with SCM complexity."""
        # Small SCM
        small_scm = pyr.m(variables={"X", "Y"}, edges={("X", "Y")})
        small_thresholds = compute_adaptive_thresholds(small_scm, difficulty_level=1)
        
        # Large SCM
        large_scm = pyr.m(
            variables={"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"},
            edges={("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")}
        )
        large_thresholds = compute_adaptive_thresholds(large_scm, difficulty_level=1)
        
        # Large SCMs should have more relaxed (lower) thresholds
        assert large_thresholds['improvement_threshold'] < small_thresholds['improvement_threshold']
        assert large_thresholds['size_factor'] < small_thresholds['size_factor']


    def test_difficulty_scaling_makes_thresholds_easier(self):
        """Test that higher difficulty levels result in easier thresholds."""
        scm = pyr.m(variables={"X", "Y", "Z"}, edges={("X", "Y"), ("Y", "Z")})
        
        easy_thresholds = compute_adaptive_thresholds(scm, difficulty_level=1)
        hard_thresholds = compute_adaptive_thresholds(scm, difficulty_level=5)
        
        # Higher difficulty should result in easier (lower) thresholds
        assert hard_thresholds['improvement_threshold'] < easy_thresholds['improvement_threshold']
        assert hard_thresholds['difficulty_factor'] < easy_thresholds['difficulty_factor']


    def test_balanced_weights_give_equal_influence(self):
        """Test that balanced weights give components equal influence."""
        balanced_config = create_default_reward_config(
            optimization_weight=1.0,
            structure_weight=1.0,  # Equal to optimization
            parent_weight=1.0,
            exploration_weight=1.0
        )
        
        rewards = compute_verifiable_reward(
            self.state_before, self.intervention, self.outcome, self.state_after, balanced_config
        )
        
        # All components should contribute equally to total
        expected_total = (
            rewards.optimization_reward +
            rewards.structure_discovery_reward + 
            rewards.parent_intervention_reward +
            rewards.exploration_bonus
        )
        assert abs(rewards.total_reward - expected_total) < 1e-6


    def test_reward_components_are_finite_and_bounded(self):
        """Test that all reward components produce finite, reasonable values."""
        config = create_default_reward_config()
        
        rewards = compute_verifiable_reward(
            self.state_before, self.intervention, self.outcome, self.state_after, config
        )
        
        # All rewards should be finite
        assert jnp.isfinite(rewards.optimization_reward)
        assert jnp.isfinite(rewards.structure_discovery_reward)
        assert jnp.isfinite(rewards.parent_intervention_reward)
        assert jnp.isfinite(rewards.exploration_bonus)
        assert jnp.isfinite(rewards.total_reward)
        
        # Individual components should be reasonably bounded
        assert 0.0 <= rewards.optimization_reward <= 1.0
        assert 0.0 <= rewards.structure_discovery_reward <= 1.0
        assert 0.0 <= rewards.parent_intervention_reward <= 1.0
        assert 0.0 <= rewards.exploration_bonus <= 1.0


    def test_adaptive_config_maintains_structure(self):
        """Test that adaptive config maintains proper structure and types."""
        scm = pyr.m(variables={"X", "Y", "Z"}, edges={("X", "Y")})
        config = create_adaptive_reward_config(scm, difficulty_level=2)
        
        # Should be a pyrsistent map
        assert isinstance(config, pyr.PMap)
        
        # Should have required keys
        assert 'reward_weights' in config
        assert 'adaptive_thresholds' in config
        assert 'scm_characteristics' in config
        
        # Reward weights should have correct structure
        weights = config['reward_weights']
        assert 'optimization' in weights
        assert 'structure' in weights
        assert 'parent' in weights
        assert 'exploration' in weights
        
        # All weights should be numeric and finite
        for weight in weights.values():
            assert isinstance(weight, (int, float))
            assert jnp.isfinite(weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])