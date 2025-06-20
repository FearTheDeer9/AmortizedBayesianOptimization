#!/usr/bin/env python3
"""
Tests for Simple Verifiable Rewards System

Comprehensive test suite for the binary ground-truth reward system
designed for GRPO training with anti-gaming properties.
"""

import pytest
import pyrsistent as pyr
from unittest.mock import patch

from causal_bayes_opt.acquisition.verifiable_rewards import (
    target_improvement_reward,
    true_parent_intervention_reward,
    exploration_diversity_reward,
    compute_simple_verifiable_reward,
    validate_reward_consistency,
    create_reward_config,
    compute_adaptive_thresholds,
    create_adaptive_reward_config,
    compute_verifiable_reward_simple,
    SimpleRewardComponents
)


class TestTargetImprovementReward:
    """Test binary target improvement reward function."""
    
    def test_improvement_above_threshold(self):
        """Should return 1.0 when improvement exceeds threshold."""
        reward = target_improvement_reward(
            outcome_value=5.5,
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 1.0
    
    def test_improvement_below_threshold(self):
        """Should return 0.0 when improvement is below threshold."""
        reward = target_improvement_reward(
            outcome_value=5.05,
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 0.0
    
    def test_no_improvement(self):
        """Should return 0.0 when there's no improvement."""
        reward = target_improvement_reward(
            outcome_value=5.0,
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 0.0
    
    def test_negative_improvement(self):
        """Should return 0.0 when performance decreases."""
        reward = target_improvement_reward(
            outcome_value=4.5,
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 0.0
    
    def test_exact_threshold(self):
        """Should return 0.0 when improvement exactly equals threshold."""
        reward = target_improvement_reward(
            outcome_value=5.1,
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 0.0  # Must be > threshold, not >=
    
    def test_non_finite_values(self):
        """Should return 0.0 for non-finite values."""
        import jax.numpy as jnp
        
        # Test NaN outcome
        reward = target_improvement_reward(
            outcome_value=float('nan'),
            current_best=5.0,
            improvement_threshold=0.1
        )
        assert reward == 0.0
        
        # Test infinite current_best
        reward = target_improvement_reward(
            outcome_value=5.5,
            current_best=float('inf'),
            improvement_threshold=0.1
        )
        assert reward == 0.0


class TestTrueParentInterventionReward:
    """Test binary true parent intervention reward function."""
    
    def test_intervention_on_true_parent(self):
        """Should return 1.0 when intervening on true parent."""
        reward = true_parent_intervention_reward(
            intervention_targets={'X'},
            true_parents={'X', 'Y'},
            target_variable='Z'
        )
        assert reward == 1.0
    
    def test_intervention_on_multiple_true_parents(self):
        """Should return 1.0 when intervening on any true parent."""
        reward = true_parent_intervention_reward(
            intervention_targets={'X', 'Y'},
            true_parents={'X', 'Y'},
            target_variable='Z'
        )
        assert reward == 1.0
    
    def test_intervention_on_non_parent(self):
        """Should return 0.0 when intervening on non-parent."""
        reward = true_parent_intervention_reward(
            intervention_targets={'Z'},
            true_parents={'X', 'Y'},
            target_variable='T'
        )
        assert reward == 0.0
    
    def test_intervention_on_target_variable(self):
        """Should return 0.0 when intervening on target itself."""
        reward = true_parent_intervention_reward(
            intervention_targets={'T'},
            true_parents={'X', 'Y'},
            target_variable='T'
        )
        assert reward == 0.0
    
    def test_mixed_intervention(self):
        """Should return 1.0 if any target is a true parent."""
        reward = true_parent_intervention_reward(
            intervention_targets={'X', 'Z'},  # X is true parent, Z is not
            true_parents={'X', 'Y'},
            target_variable='T'
        )
        assert reward == 1.0
    
    def test_empty_intervention_targets(self):
        """Should return 0.0 for empty intervention targets."""
        reward = true_parent_intervention_reward(
            intervention_targets=set(),
            true_parents={'X', 'Y'},
            target_variable='T'
        )
        assert reward == 0.0
    
    def test_empty_true_parents(self):
        """Should return 0.0 when there are no true parents."""
        reward = true_parent_intervention_reward(
            intervention_targets={'X'},
            true_parents=set(),
            target_variable='T'
        )
        assert reward == 0.0


class TestExplorationDiversityReward:
    """Test binary exploration diversity reward function."""
    
    def test_first_intervention(self):
        """Should return 1.0 for first intervention on variable."""
        reward = exploration_diversity_reward(
            intervention_targets={'X'},
            previous_interventions=[],
            diversity_threshold=3
        )
        assert reward == 1.0
    
    def test_under_threshold_frequency(self):
        """Should return 1.0 when frequency is under threshold."""
        previous = [{'Y'}, {'Z'}, {'Y'}]  # X never intervened
        reward = exploration_diversity_reward(
            intervention_targets={'X'},
            previous_interventions=previous,
            diversity_threshold=3
        )
        assert reward == 1.0
    
    def test_at_threshold_frequency(self):
        """Should return 0.0 when frequency reaches threshold."""
        previous = [{'X'}, {'Y'}, {'X'}]  # X intervened 2 times
        reward = exploration_diversity_reward(
            intervention_targets={'X'},
            previous_interventions=previous,
            diversity_threshold=2
        )
        assert reward == 0.0  # Would be 3rd time, >= threshold
    
    def test_above_threshold_frequency(self):
        """Should return 0.0 when frequency exceeds threshold."""
        previous = [{'X'}, {'Y'}, {'X'}, {'Z'}, {'X'}]  # X intervened 3 times
        reward = exploration_diversity_reward(
            intervention_targets={'X'},
            previous_interventions=previous,
            diversity_threshold=3
        )
        assert reward == 0.0
    
    def test_overlapping_interventions(self):
        """Should count overlapping interventions correctly."""
        previous = [{'X', 'Y'}, {'Z'}, {'X'}]  # X appears in 2 interventions
        reward = exploration_diversity_reward(
            intervention_targets={'X'},
            previous_interventions=previous,
            diversity_threshold=3
        )
        assert reward == 1.0  # Count is 2 < 3
    
    def test_empty_intervention_targets(self):
        """Should return 0.0 for empty intervention targets."""
        reward = exploration_diversity_reward(
            intervention_targets=set(),
            previous_interventions=[{'X'}, {'Y'}],
            diversity_threshold=3
        )
        assert reward == 0.0
    
    def test_multi_variable_intervention(self):
        """Should handle multi-variable interventions correctly."""
        previous = [{'X'}, {'Y'}, {'X', 'Z'}]  # X overlaps with 2 interventions
        reward = exploration_diversity_reward(
            intervention_targets={'X', 'Y'},  # Overlaps with all 3 previous
            previous_interventions=previous,
            diversity_threshold=4
        )
        assert reward == 1.0  # Count is 3 < 4


class TestComputeSimpleVerifiableReward:
    """Test main reward computation function."""
    
    def test_all_components_positive(self):
        """Should combine all reward components correctly."""
        reward = compute_simple_verifiable_reward(
            intervention_targets={'X'},
            outcome_value=5.5,           # Improvement of 0.5 > 0.1
            current_best=5.0,
            true_parents={'X', 'Y'},     # X is true parent
            target_variable='Z',
            previous_interventions=[],    # First intervention on X
            weights={'target_improvement': 2.0, 'true_parent': 1.0, 'exploration': 0.5}
        )
        
        assert reward.target_improvement == 1.0
        assert reward.true_parent_intervention == 1.0
        assert reward.exploration_diversity == 1.0
        assert reward.total_reward == 3.5  # 2.0*1 + 1.0*1 + 0.5*1
    
    def test_mixed_components(self):
        """Should handle mixed positive/zero components."""
        reward = compute_simple_verifiable_reward(
            intervention_targets={'Z'},   # Not a true parent
            outcome_value=5.05,          # Small improvement < 0.1
            current_best=5.0,
            true_parents={'X', 'Y'},
            target_variable='T',
            previous_interventions=[],
            weights={'target_improvement': 2.0, 'true_parent': 1.0, 'exploration': 0.5}
        )
        
        assert reward.target_improvement == 0.0      # No meaningful improvement
        assert reward.true_parent_intervention == 0.0 # Not true parent
        assert reward.exploration_diversity == 1.0   # First intervention
        assert reward.total_reward == 0.5            # Only exploration component
    
    def test_default_weights(self):
        """Should use sensible default weights."""
        reward = compute_simple_verifiable_reward(
            intervention_targets={'X'},
            outcome_value=5.5,
            current_best=5.0,
            true_parents={'X'},
            target_variable='Z',
            previous_interventions=[]
        )
        
        # Default weights: target=2.0, parent=1.0, exploration=0.5
        assert reward.total_reward == 3.5
    
    def test_summary_method(self):
        """Should provide comprehensive summary."""
        reward = compute_simple_verifiable_reward(
            intervention_targets={'X'},
            outcome_value=5.5,
            current_best=5.0,
            true_parents={'X', 'Y'},
            target_variable='Z',
            previous_interventions=[]
        )
        
        summary = reward.summary()
        
        assert summary['total_reward'] == 3.5
        assert summary['target_variable'] == 'Z'
        assert summary['improvement_amount'] == 0.5
        assert summary['intervened_on_true_parent'] == True
        assert 'X' in summary['intervention_targets']
    
    def test_input_validation(self):
        """Should handle various input types correctly."""
        # Test with lists instead of sets
        reward = compute_simple_verifiable_reward(
            intervention_targets=['X'],  # List instead of set
            outcome_value=5.5,
            current_best=5.0,
            true_parents=['X', 'Y'],     # List instead of set
            target_variable='Z',
            previous_interventions=[]
        )
        
        assert reward.target_improvement == 1.0
        assert reward.true_parent_intervention == 1.0


class TestValidateRewardConsistency:
    """Test reward consistency validation and gaming detection."""
    
    def test_healthy_reward_pattern(self):
        """Should validate healthy reward patterns."""
        # Create diverse reward history
        rewards = []
        for i in range(50):
            rewards.append(SimpleRewardComponents(
                target_improvement=1.0 if i % 3 == 0 else 0.0,
                true_parent_intervention=1.0 if i % 4 == 0 else 0.0,
                exploration_diversity=1.0 if i % 2 == 0 else 0.0,
                total_reward=2.5 if i % 5 == 0 else 1.0,
                intervention_targets={'X'},
                target_variable='Y',
                outcome_value=5.0,
                current_best=4.0,
                true_parents={'X'}
            ))
        
        validation = validate_reward_consistency(rewards)
        
        assert validation['valid'] == True
        assert len(validation['gaming_issues']) == 0
        assert 0.1 <= validation['component_rates']['target_improvement'] <= 0.9
    
    def test_gaming_detection_no_target_improvement(self):
        """Should detect when target improvement rate is too low."""
        # All rewards have no target improvement
        rewards = []
        for i in range(50):
            rewards.append(SimpleRewardComponents(
                target_improvement=0.0,  # Always 0
                true_parent_intervention=1.0,
                exploration_diversity=1.0,
                total_reward=1.5,
                intervention_targets={'X'},
                target_variable='Y',
                outcome_value=5.0,
                current_best=4.0,
                true_parents={'X'}
            ))
        
        validation = validate_reward_consistency(rewards)
        
        assert validation['valid'] == False
        assert any('target improvement rate' in issue for issue in validation['gaming_issues'])
    
    def test_gaming_detection_too_many_parent_interventions(self):
        """Should detect suspiciously high parent intervention rate."""
        rewards = []
        for i in range(50):
            rewards.append(SimpleRewardComponents(
                target_improvement=1.0,
                true_parent_intervention=1.0,  # Always 1 (suspicious)
                exploration_diversity=1.0,
                total_reward=3.5,
                intervention_targets={'X'},
                target_variable='Y',
                outcome_value=5.0,
                current_best=4.0,
                true_parents={'X'}
            ))
        
        validation = validate_reward_consistency(rewards)
        
        assert validation['valid'] == False
        assert any('parent intervention rate' in issue for issue in validation['gaming_issues'])
    
    def test_gaming_detection_no_exploration(self):
        """Should detect when exploration diversity is too low."""
        rewards = []
        for i in range(50):
            rewards.append(SimpleRewardComponents(
                target_improvement=1.0,
                true_parent_intervention=1.0,
                exploration_diversity=0.0,  # Always 0 (mode collapse)
                total_reward=3.0,
                intervention_targets={'X'},
                target_variable='Y',
                outcome_value=5.0,
                current_best=4.0,
                true_parents={'X'}
            ))
        
        validation = validate_reward_consistency(rewards)
        
        assert validation['valid'] == False
        assert any('exploration diversity' in issue for issue in validation['gaming_issues'])
    
    def test_gaming_detection_low_variance(self):
        """Should detect suspiciously low reward variance."""
        rewards = []
        for i in range(50):
            rewards.append(SimpleRewardComponents(
                target_improvement=1.0,
                true_parent_intervention=1.0,
                exploration_diversity=1.0,
                total_reward=3.5,  # Always exactly the same (suspicious)
                intervention_targets={'X'},
                target_variable='Y',
                outcome_value=5.0,
                current_best=4.0,
                true_parents={'X'}
            ))
        
        validation = validate_reward_consistency(rewards)
        
        assert validation['valid'] == False
        assert any('variance' in issue for issue in validation['gaming_issues'])
    
    def test_empty_history(self):
        """Should handle empty reward history gracefully."""
        validation = validate_reward_consistency([])
        
        assert validation['valid'] == True
        assert 'No reward history' in validation['warning']


class TestCreateRewardConfig:
    """Test reward configuration creation and validation."""
    
    def test_default_config(self):
        """Should create valid default configuration."""
        config = create_reward_config()
        
        assert config['weights']['target_improvement'] == 2.0
        assert config['weights']['true_parent'] == 1.0
        assert config['weights']['exploration'] == 0.5
        assert config['improvement_threshold'] == 0.1
        assert config['diversity_threshold'] == 3
    
    def test_custom_config(self):
        """Should create custom configuration."""
        config = create_reward_config(
            target_improvement_weight=3.0,
            true_parent_weight=0.5,
            exploration_weight=1.0,
            improvement_threshold=0.2,
            diversity_threshold=5
        )
        
        assert config['weights']['target_improvement'] == 3.0
        assert config['weights']['true_parent'] == 0.5
        assert config['weights']['exploration'] == 1.0
        assert config['improvement_threshold'] == 0.2
        assert config['diversity_threshold'] == 5
    
    def test_negative_weight_validation(self):
        """Should reject negative weights."""
        with pytest.raises(ValueError, match="non-negative"):
            create_reward_config(target_improvement_weight=-1.0)
    
    def test_invalid_threshold_validation(self):
        """Should reject invalid thresholds."""
        with pytest.raises(ValueError, match="positive"):
            create_reward_config(improvement_threshold=0.0)
        
        with pytest.raises(ValueError, match="positive"):
            create_reward_config(diversity_threshold=0)


class TestComputeVerifiableRewardSimple:
    """Test convenience function for integration with existing interfaces."""
    
    def test_pyrsistent_integration(self):
        """Should work with pyrsistent data structures."""
        # Create pyrsistent data structures
        intervention = pyr.m(**{
            'type': 'perfect',
            'targets': {'X'}
        })
        
        outcome = pyr.m(**{
            'values': {'Y': 5.5, 'X': 2.0},
            'intervention_type': 'perfect'
        })
        
        scm = pyr.m(**{
            'target': 'Y',
            'edges': frozenset([('X', 'Y')])
        })
        
        previous_interventions = [
            pyr.m(**{'targets': {'Z'}})
        ]
        
        reward = compute_verifiable_reward_simple(
            intervention=intervention,
            outcome=outcome,
            scm=scm,
            current_best=5.0,
            previous_interventions=previous_interventions
        )
        
        assert reward.target_improvement == 1.0  # 5.5 > 5.0 + 0.1
        assert reward.true_parent_intervention == 1.0  # X is true parent
        assert reward.exploration_diversity == 1.0  # X not in previous
        assert reward.total_reward == 3.5
    
    def test_missing_target_error(self):
        """Should raise error when SCM missing target."""
        intervention = pyr.m(**{'targets': {'X'}})
        outcome = pyr.m(**{'values': {'Y': 5.5}})
        scm = pyr.m(**{})  # Missing target
        
        with pytest.raises(ValueError, match="target variable"):
            compute_verifiable_reward_simple(
                intervention=intervention,
                outcome=outcome,
                scm=scm,
                current_best=5.0,
                previous_interventions=[]
            )
    
    def test_missing_outcome_value_error(self):
        """Should raise error when outcome missing target value."""
        intervention = pyr.m(**{'targets': {'X'}})
        outcome = pyr.m(**{'values': {'Z': 5.5}})  # Missing Y value
        scm = pyr.m(**{'target': 'Y'})
        
        with pytest.raises(ValueError, match="Outcome must contain value"):
            compute_verifiable_reward_simple(
                intervention=intervention,
                outcome=outcome,
                scm=scm,
                current_best=5.0,
                previous_interventions=[]
            )
    
    def test_list_to_set_conversion(self):
        """Should convert lists to sets automatically."""
        intervention = pyr.m(**{
            'targets': ['X', 'Y']  # List instead of set
        })
        
        outcome = pyr.m(**{
            'values': {'Z': 5.5}
        })
        
        scm = pyr.m(**{
            'target': 'Z',
            'edges': frozenset([('X', 'Z'), ('Y', 'Z')])
        })
        
        reward = compute_verifiable_reward_simple(
            intervention=intervention,
            outcome=outcome,
            scm=scm,
            current_best=5.0,
            previous_interventions=[]
        )
        
        assert len(reward.intervention_targets) == 2
        assert 'X' in reward.intervention_targets
        assert 'Y' in reward.intervention_targets


class TestAdaptiveThresholds:
    """Test adaptive threshold computation for different SCM characteristics."""
    
    def test_small_scm_thresholds(self):
        """Should use base thresholds for small SCMs."""
        scm = pyr.m(**{
            'variables': {'X', 'Y', 'Z'},  # 3 variables
            'edges': frozenset([('X', 'Y'), ('Y', 'Z')])  # 2 edges
        })
        
        thresholds = compute_adaptive_thresholds(scm)
        
        # Small SCM should use base thresholds
        assert thresholds['improvement_threshold'] == 0.1  # Base threshold
        assert thresholds['diversity_threshold'] == 3      # Base threshold
        assert thresholds['size_factor'] == 1.0
    
    def test_medium_scm_thresholds(self):
        """Should use relaxed thresholds for medium SCMs."""
        scm = pyr.m(**{
            'variables': {f'X{i}' for i in range(8)},  # 8 variables
            'edges': frozenset([(f'X{i}', f'X{i+1}') for i in range(7)])  # 7 edges
        })
        
        thresholds = compute_adaptive_thresholds(scm)
        
        # Medium SCM should have easier thresholds
        assert thresholds['improvement_threshold'] < 0.1
        assert thresholds['diversity_threshold'] <= 3
        assert thresholds['size_factor'] == 0.7
    
    def test_large_scm_thresholds(self):
        """Should use very relaxed thresholds for large SCMs."""
        scm = pyr.m(**{
            'variables': {f'X{i}' for i in range(15)},  # 15 variables
            'edges': frozenset([(f'X{i}', f'X{i+1}') for i in range(14)])  # 14 edges
        })
        
        thresholds = compute_adaptive_thresholds(scm)
        
        # Large SCM should have very relaxed thresholds
        assert thresholds['improvement_threshold'] <= 0.05
        assert thresholds['diversity_threshold'] <= 2
        assert thresholds['size_factor'] == 0.5
    
    def test_dense_graph_adjustment(self):
        """Should adjust thresholds for dense graphs."""
        scm = pyr.m(**{
            'variables': {'X', 'Y', 'Z', 'W'},  # 4 variables
            'edges': frozenset([('X', 'Y'), ('X', 'Z'), ('Y', 'Z'), ('Z', 'W'), ('X', 'W')])  # 5 edges (dense)
        })
        
        thresholds = compute_adaptive_thresholds(scm)
        
        # Dense graph should have density adjustment
        assert thresholds['density_factor'] == 0.8  # Harder threshold
        assert thresholds['improvement_threshold'] < 0.1  # Adjusted down
    
    def test_difficulty_level_adjustment(self):
        """Should adjust thresholds based on curriculum difficulty."""
        scm = pyr.m(**{
            'variables': {'X', 'Y', 'Z'},
            'edges': frozenset([('X', 'Y')])
        })
        
        # Test different difficulty levels
        thresholds_1 = compute_adaptive_thresholds(scm, difficulty_level=1)
        thresholds_3 = compute_adaptive_thresholds(scm, difficulty_level=3)
        thresholds_5 = compute_adaptive_thresholds(scm, difficulty_level=5)
        
        # Higher difficulty should have easier thresholds
        assert thresholds_1['improvement_threshold'] > thresholds_3['improvement_threshold']
        assert thresholds_3['improvement_threshold'] > thresholds_5['improvement_threshold']
        
        # Check difficulty factors
        assert thresholds_1['difficulty_factor'] == 1.0
        assert thresholds_3['difficulty_factor'] == 0.8
        assert thresholds_5['difficulty_factor'] == 0.6
    
    def test_adaptive_config_creation(self):
        """Should create adaptive configuration correctly."""
        scm = pyr.m(**{
            'variables': {'X', 'Y', 'Z', 'W'},  # 4 variables
            'edges': frozenset([('X', 'Y'), ('Y', 'Z')])
        })
        
        config = create_adaptive_reward_config(scm, difficulty_level=2)
        
        # Should have all standard config components
        assert 'weights' in config
        assert 'improvement_threshold' in config
        assert 'diversity_threshold' in config
        
        # Should have adaptive metadata
        assert 'adaptive_thresholds' in config
        assert 'scm_characteristics' in config
        
        # Check SCM characteristics
        assert config['scm_characteristics']['n_variables'] == 4
        assert config['scm_characteristics']['n_edges'] == 2
        assert config['scm_characteristics']['difficulty_level'] == 2
        
        # Should use adaptive thresholds
        assert config['improvement_threshold'] < 0.1  # Easier than base


class TestAdaptiveRewardIntegration:
    """Test integration of adaptive rewards with existing system."""
    
    def test_adaptive_vs_static_config(self):
        """Should show difference between adaptive and static configurations."""
        # Large SCM that should trigger adaptations
        scm = pyr.m(**{
            'variables': {f'X{i}' for i in range(12)},  # 12 variables
            'edges': frozenset([(f'X{i}', f'X{i+1}') for i in range(11)])
        })
        
        static_config = create_reward_config()
        adaptive_config = create_adaptive_reward_config(scm, difficulty_level=3)
        
        # Adaptive should have easier thresholds for large SCM
        assert adaptive_config['improvement_threshold'] < static_config['improvement_threshold']
        assert adaptive_config['diversity_threshold'] <= static_config['diversity_threshold']
    
    def test_adaptive_config_with_simple_reward(self):
        """Should work with compute_simple_verifiable_reward function."""
        scm = pyr.m(**{
            'variables': {'X', 'Y', 'Z'},
            'edges': frozenset([('X', 'Y')])
        })
        
        adaptive_config = create_adaptive_reward_config(scm, difficulty_level=1)
        
        # Should work with reward computation
        reward = compute_simple_verifiable_reward(
            intervention_targets={'X'},
            outcome_value=5.05,  # Small improvement
            current_best=5.0,
            true_parents={'X'},
            target_variable='Y',
            previous_interventions=[],
            weights=adaptive_config['weights'],
            improvement_threshold=adaptive_config['improvement_threshold'],
            diversity_threshold=adaptive_config['diversity_threshold']
        )
        
        # With adaptive thresholds, small improvement might still get reward
        assert isinstance(reward.total_reward, float)
        assert reward.total_reward >= 0


if __name__ == "__main__":
    pytest.main([__file__])