#!/usr/bin/env python3
"""
Reward Normalization Tests

Tests to ensure reward components are properly normalized across different
variable ranges, SCM complexities, and intervention scenarios.
"""

import pytest
import jax.numpy as jnp
import pyrsistent as pyr
from unittest.mock import Mock
from typing import List, Dict, Any, Tuple

# Import the continuous reward system
from causal_bayes_opt.acquisition.rewards import (
    RewardComponents,
    compute_verifiable_reward,
    _compute_optimization_reward,
    _compute_scm_objective_reward,
    _compute_improved_relative_reward,
    create_default_reward_config,
    compute_adaptive_thresholds,
)


class TestRewardNormalization:
    """Test suite for reward normalization across different ranges and scenarios."""
    
    def setup_method(self):
        """Set up normalization test scenarios."""
        self.config = create_default_reward_config()
        
        # Base mock state template
        self.mock_posterior = Mock()
        self.mock_posterior.uncertainty = 0.5
        self.mock_posterior.target_variable = "Y"
        
        self.mock_buffer = Mock()
        self.mock_buffer.samples = []
        self.mock_buffer.get_interventions.return_value = []


    def create_test_state(self, best_value: float = 0.0, step: int = 5) -> Mock:
        """Create a test acquisition state with specified parameters."""
        state = Mock()
        state.current_target = "Y"
        state.step = step
        state.best_value = best_value
        state.posterior = self.mock_posterior
        state.buffer = self.mock_buffer
        state.marginal_parent_probs = {"X": 0.5, "Z": 0.3}
        state.uncertainty_bits = 0.5
        state.buffer_statistics = Mock()
        state.buffer_statistics.total_samples = 20
        return state


    def test_normalization_across_tiny_ranges(self):
        """Test reward normalization with very small variable ranges."""
        # Tiny range scenario (e.g., values around 0.001)
        state_before = self.create_test_state(best_value=0.001)
        state_after = self.create_test_state(best_value=0.0012, step=6)
        
        intervention = pyr.m(type="perfect", targets={"X"})
        tiny_outcome = pyr.m(values={"Y": 0.0015, "X": 0.0008, "Z": 0.0005})
        
        rewards = compute_verifiable_reward(
            state_before, intervention, tiny_outcome, state_after, self.config
        )
        
        # All rewards should be in [0, 1] range despite tiny values
        assert 0.0 <= rewards.optimization_reward <= 1.0
        assert 0.0 <= rewards.structure_discovery_reward <= 1.0
        assert 0.0 <= rewards.parent_intervention_reward <= 1.0
        assert 0.0 <= rewards.exploration_bonus <= 1.0
        assert jnp.isfinite(rewards.total_reward)


    def test_normalization_across_huge_ranges(self):
        """Test reward normalization with very large variable ranges."""
        # Huge range scenario (e.g., values in millions)
        state_before = self.create_test_state(best_value=1e6)
        state_after = self.create_test_state(best_value=1.2e6, step=6)
        
        intervention = pyr.m(type="perfect", targets={"X"})
        huge_outcome = pyr.m(values={"Y": 1.5e6, "X": 8e5, "Z": 5e5})
        
        rewards = compute_verifiable_reward(
            state_before, intervention, huge_outcome, state_after, self.config
        )
        
        # All rewards should be in [0, 1] range despite huge values
        assert 0.0 <= rewards.optimization_reward <= 1.0
        assert 0.0 <= rewards.structure_discovery_reward <= 1.0
        assert 0.0 <= rewards.parent_intervention_reward <= 1.0
        assert 0.0 <= rewards.exploration_bonus <= 1.0
        assert jnp.isfinite(rewards.total_reward)


    def test_normalization_across_negative_ranges(self):
        """Test reward normalization with negative variable ranges."""
        # Negative range scenario
        state_before = self.create_test_state(best_value=-100.0)
        state_after = self.create_test_state(best_value=-80.0, step=6)
        
        intervention = pyr.m(type="perfect", targets={"X"})
        negative_outcome = pyr.m(values={"Y": -60.0, "X": -90.0, "Z": -110.0})
        
        rewards = compute_verifiable_reward(
            state_before, intervention, negative_outcome, state_after, self.config
        )
        
        # All rewards should be in [0, 1] range despite negative values
        assert 0.0 <= rewards.optimization_reward <= 1.0
        assert 0.0 <= rewards.structure_discovery_reward <= 1.0
        assert 0.0 <= rewards.parent_intervention_reward <= 1.0
        assert 0.0 <= rewards.exploration_bonus <= 1.0
        assert jnp.isfinite(rewards.total_reward)


    def test_scm_objective_normalization_different_mechanisms(self):
        """Test SCM-objective reward normalization across different mechanism types."""
        test_cases = [
            # Small coefficients
            {"coefficients": {"X": 0.1, "Z": -0.05}, "intercept": 0.5, "bounds": (-10, 10)},
            # Large coefficients  
            {"coefficients": {"X": 100.0, "Z": -50.0}, "intercept": 1000.0, "bounds": (-1, 1)},
            # Mixed coefficients
            {"coefficients": {"X": 1e6, "Z": -1e-6}, "intercept": 0.0, "bounds": (-1e3, 1e3)},
            # Zero intercept
            {"coefficients": {"X": 2.0, "Z": -1.0}, "intercept": 0.0, "bounds": (-5, 5)},
        ]
        
        for i, case in enumerate(test_cases):
            # Set up mechanism
            mock_mechanism = Mock()
            mock_mechanism.coefficients = case["coefficients"]
            mock_mechanism.intercept = case["intercept"]
            
            state = self.create_test_state()
            state.mechanism_predictions = {"Y": mock_mechanism}
            state.intervention_bounds = {
                "X": (case["bounds"][0], case["bounds"][1]),
                "Z": (case["bounds"][0], case["bounds"][1])
            }
            
            # Test different target values
            test_values = [
                case["bounds"][0],  # Minimum
                case["bounds"][1],  # Maximum  
                (case["bounds"][0] + case["bounds"][1]) / 2,  # Middle
            ]
            
            for target_val in test_values:
                outcome = pyr.m(values={"Y": target_val})
                reward = _compute_optimization_reward(state, outcome, "Y")
                
                # Should always be normalized to [0, 1]
                assert 0.0 <= reward <= 1.0, f"Case {i}, value {target_val}: reward {reward}"
                assert jnp.isfinite(reward), f"Case {i}, value {target_val}: non-finite reward"


    def test_relative_reward_fallback_normalization(self):
        """Test that relative reward fallback is properly normalized."""
        # Create scenario where SCM-objective reward fails (no mechanism predictions)
        state = self.create_test_state(best_value=5.0)
        # Don't set mechanism_predictions to trigger fallback
        
        # Test various target values
        test_outcomes = [
            {"Y": 0.0},   # Much worse than best
            {"Y": 4.0},   # Slightly worse
            {"Y": 5.0},   # Same as best
            {"Y": 6.0},   # Slightly better
            {"Y": 10.0},  # Much better
        ]
        
        for outcome_vals in test_outcomes:
            outcome = pyr.m(values=outcome_vals)
            reward = _compute_optimization_reward(state, outcome, "Y")
            
            # Should be normalized to [0, 1] even in fallback mode
            assert 0.0 <= reward <= 1.0, f"Outcome {outcome_vals}: reward {reward}"
            assert jnp.isfinite(reward), f"Outcome {outcome_vals}: non-finite reward"


    def test_normalization_consistency_across_scales(self):
        """Test that normalization is consistent when scaling all values."""
        base_case = {
            "best_value": 1.0,
            "outcome_value": 2.0,
            "coefficients": {"X": 1.0},
            "intercept": 0.0,
            "bounds": (-2.0, 2.0)
        }
        
        scale_factors = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        
        rewards = []
        for scale in scale_factors:
            # Scale everything by the same factor
            mock_mechanism = Mock()
            mock_mechanism.coefficients = {k: v * scale for k, v in base_case["coefficients"].items()}
            mock_mechanism.intercept = base_case["intercept"] * scale
            
            state = self.create_test_state(best_value=base_case["best_value"] * scale)
            state.mechanism_predictions = {"Y": mock_mechanism}
            state.intervention_bounds = {
                "X": (base_case["bounds"][0] * scale, base_case["bounds"][1] * scale)
            }
            
            outcome = pyr.m(values={"Y": base_case["outcome_value"] * scale})
            reward = _compute_optimization_reward(state, outcome, "Y")
            rewards.append(reward)
        
        # Rewards should be similar across different scales (within reasonable tolerance)
        # Some variance is expected across extreme scales, but shouldn't be excessive
        reward_std = float(jnp.std(jnp.array(rewards)))
        assert reward_std < 0.3, f"High variance across scales: {rewards}, std={reward_std}"
        
        # All rewards should still be in valid range
        for i, reward in enumerate(rewards):
            assert 0.0 <= reward <= 1.0, f"Scale {scale_factors[i]}: reward {reward} out of bounds"


    def test_normalization_with_zero_ranges(self):
        """Test normalization behavior when variable ranges are zero."""
        # Create mechanism where optimal equals worst case (zero range)
        mock_mechanism = Mock()
        mock_mechanism.coefficients = {}  # No variables affect target
        mock_mechanism.intercept = 5.0
        
        state = self.create_test_state()
        state.mechanism_predictions = {"Y": mock_mechanism}
        state.intervention_bounds = {}
        
        # Any outcome should give same reward since no improvement possible
        outcome = pyr.m(values={"Y": 5.0})  # Matches intercept
        reward = _compute_optimization_reward(state, outcome, "Y")
        
        # Should handle gracefully (likely give max reward since no improvement possible)
        assert jnp.isfinite(reward)
        assert 0.0 <= reward <= 1.0


    def test_adaptive_thresholds_normalization(self):
        """Test that adaptive thresholds maintain proper normalization."""
        # Test different SCM complexities
        scm_configs = [
            # Simple SCM
            pyr.m(variables={"X", "Y"}, edges={("X", "Y")}),
            # Medium SCM
            pyr.m(variables={"A", "B", "C", "D"}, edges={("A", "B"), ("B", "C"), ("C", "D")}),
            # Complex SCM
            pyr.m(
                variables={"X1", "X2", "X3", "X4", "X5", "Y"},
                edges={("X1", "X3"), ("X2", "X3"), ("X3", "Y"), ("X4", "Y"), ("X5", "Y")}
            )
        ]
        
        for difficulty in [1, 3, 5]:
            for scm in scm_configs:
                thresholds = compute_adaptive_thresholds(scm, difficulty_level=difficulty)
                
                # All thresholds should be positive and reasonable
                assert thresholds['improvement_threshold'] > 0
                assert thresholds['diversity_threshold'] > 0
                assert thresholds['improvement_threshold'] <= 1.0  # Should be reasonable
                assert thresholds['diversity_threshold'] <= 1.0
                
                # Should be finite
                assert jnp.isfinite(thresholds['improvement_threshold'])
                assert jnp.isfinite(thresholds['diversity_threshold'])


    def test_normalization_edge_cases(self):
        """Test normalization in edge cases."""
        edge_cases = [
            # Zero best value
            {"best_value": 0.0, "outcome_value": 0.0},
            # Identical values
            {"best_value": 1.0, "outcome_value": 1.0},
            # Very close values
            {"best_value": 1.0, "outcome_value": 1.0 + 1e-10},
            # Large difference
            {"best_value": 1.0, "outcome_value": 1e10},
        ]
        
        for case in edge_cases:
            state = self.create_test_state(best_value=case["best_value"])
            outcome = pyr.m(values={"Y": case["outcome_value"]})
            
            reward = _compute_optimization_reward(state, outcome, "Y")
            
            # Should handle all edge cases gracefully
            assert jnp.isfinite(reward), f"Non-finite reward for case {case}"
            assert 0.0 <= reward <= 1.0, f"Out of bounds reward for case {case}: {reward}"


    def test_cross_component_normalization_consistency(self):
        """Test that different reward components have consistent normalization."""
        # Create scenario with known bounds for all components
        state_before = self.create_test_state()
        state_after = self.create_test_state(step=6)
        
        intervention = pyr.m(type="perfect", targets={"X"})
        outcome = pyr.m(values={"Y": 1.0, "X": 0.5, "Z": 0.2})
        
        rewards = compute_verifiable_reward(
            state_before, intervention, outcome, state_after, self.config
        )
        
        # All individual components should be in [0, 1]
        components = [
            rewards.optimization_reward,
            rewards.structure_discovery_reward, 
            rewards.parent_intervention_reward,
            rewards.exploration_bonus
        ]
        
        for i, component in enumerate(components):
            assert 0.0 <= component <= 1.0, f"Component {i} out of bounds: {component}"
            assert jnp.isfinite(component), f"Component {i} not finite: {component}"
        
        # Total reward should be sensible given weights
        weights = self.config['reward_weights']
        expected_max = (
            weights['optimization'] * 1.0 +
            weights['structure'] * 1.0 +
            weights['parent'] * 1.0 +
            weights['exploration'] * 1.0
        )
        
        assert rewards.total_reward <= expected_max + 1e-6, "Total reward exceeds theoretical maximum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])