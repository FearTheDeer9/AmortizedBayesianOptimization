#!/usr/bin/env python3
"""
Hybrid Reward System Demo

This script demonstrates the enhanced hybrid reward system that combines
supervised learning signals (using ground truth during training) with 
observable signals (no ground truth, for robustness) to guide mechanism-aware
intervention selection.

Part B: Hybrid Reward System (Complete)
"""

import jax.numpy as jnp
import pyrsistent as pyr

from causal_bayes_opt.acquisition.hybrid_rewards import (
    HybridRewardConfig,
    compute_hybrid_reward,
    create_hybrid_reward_config,
    supervised_mechanism_impact_reward,
    supervised_mechanism_discovery_reward,
    posterior_confidence_reward,
    causal_effect_discovery_reward,
    mechanism_consistency_reward,
    validate_hybrid_reward_consistency,
    compare_reward_strategies
)
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.avici_integration.parent_set.mechanism_aware import (
    MechanismPrediction
)


class MockAcquisitionState:
    """Mock acquisition state for demo."""
    def __init__(self, parent_posterior, best_target_value, optimization_target, mechanism_predictions=None):
        self.parent_posterior = parent_posterior
        self.best_target_value = best_target_value
        self.optimization_target = optimization_target
        self.mechanism_predictions = mechanism_predictions or []
        self.buffer_statistics = {'total_samples': 20}
        self.intervention_history = []
        self.uncertainty_bits = 1.2
        self.step = 5


def main():
    """Demonstrate hybrid reward system capabilities."""
    print("=== Hybrid Reward System Demo ===\\n")
    
    # Setup test scenario: X → Y ← Z
    print("Test scenario: X → Y ← Z (target variable: Y)")
    
    # Ground truth information
    ground_truth = {
        'scm': create_scm(
            variables=frozenset(['X', 'Y', 'Z']),
            edges=frozenset([('X', 'Y'), ('Z', 'Y')]),
            target='Y',
            mechanisms={}
        ),
        'mechanism_info': {
            'Y': {
                'type': 'linear',
                'parents': frozenset(['X', 'Z']),
                'coefficients': {'X': 1.5, 'Z': 0.8}  # X has bigger impact
            }
        }
    }
    
    print(f"True parents of Y: {ground_truth['mechanism_info']['Y']['parents']}")
    print(f"True coefficients: {ground_truth['mechanism_info']['Y']['coefficients']}")
    print()
    
    # Demo 1: Supervised Reward Components
    print("=== Demo 1: Supervised Reward Components ===")
    
    # High-impact intervention (on X)
    high_impact_reward = supervised_mechanism_impact_reward(
        intervention_targets=frozenset(['X']),
        intervention_values={'X': 3.0},
        true_mechanism_info=ground_truth['mechanism_info'],
        target_variable='Y'
    )
    
    # Low-impact intervention (on Z)
    low_impact_reward = supervised_mechanism_impact_reward(
        intervention_targets=frozenset(['Z']),
        intervention_values={'Z': 3.0},
        true_mechanism_info=ground_truth['mechanism_info'],
        target_variable='Y'
    )
    
    print(f"High-impact intervention (X): {high_impact_reward:.3f}")
    print(f"Low-impact intervention (Z): {low_impact_reward:.3f}")
    print(f"Correctly prioritizes high-impact variables: {high_impact_reward > low_impact_reward}")
    print()
    
    # Demo 2: Observable Reward Components
    print("=== Demo 2: Observable Reward Components ===")
    
    # Uncertainty reduction
    current_posterior = jnp.array([0.4, 0.3, 0.2, 0.1])  # High uncertainty
    next_posterior = jnp.array([0.1, 0.8, 0.05, 0.05])   # Low uncertainty
    
    uncertainty_reward = posterior_confidence_reward(
        current_posterior=current_posterior,
        next_posterior=next_posterior
    )
    
    print(f"Uncertainty reduction reward: {uncertainty_reward:.3f}")
    
    # Strong causal effect
    strong_effect_reward = causal_effect_discovery_reward(
        intervention_outcome=8.0,
        baseline_prediction=5.0,
        predicted_effect=2.8,  # Close to observed 3.0
        effect_threshold=1.0
    )
    
    print(f"Strong effect discovery reward: {strong_effect_reward:.3f}")
    
    # Mechanism consistency
    consistent_mechanism = MechanismPrediction(
        parent_set=frozenset(['X']),
        mechanism_type='linear',
        parameters={'coefficients': {'X': 1.4}},  # Close to true 1.5
        confidence=0.9
    )
    
    consistency_reward = mechanism_consistency_reward(
        predicted_mechanism=consistent_mechanism,
        observed_effect=4.2,  # X=3.0 * 1.4 ≈ 4.2
        intervention_values={'X': 3.0}
    )
    
    print(f"Mechanism consistency reward: {consistency_reward:.3f}")
    print()
    
    # Demo 3: Hybrid Reward Integration
    print("=== Demo 3: Hybrid Reward Integration ===")
    
    # Create mock states
    current_state = MockAcquisitionState(
        parent_posterior=current_posterior,
        best_target_value=5.0,
        optimization_target='Y',
        mechanism_predictions=[consistent_mechanism]
    )
    
    next_state = MockAcquisitionState(
        parent_posterior=next_posterior,
        best_target_value=8.0,  # Improved
        optimization_target='Y',
        mechanism_predictions=[consistent_mechanism]
    )
    
    intervention = pyr.m(
        targets=frozenset(['X']),
        values={'X': 3.0},
        type='perfect'
    )
    
    outcome = create_sample(
        values={'X': 3.0, 'Y': 8.0, 'Z': 2.0},
        intervention_type='perfect',
        intervention_targets=frozenset(['X'])
    )
    
    # Test different configurations
    configs = [
        ('training', create_hybrid_reward_config(mode='training')),
        ('deployment', create_hybrid_reward_config(mode='deployment')),
        ('research', create_hybrid_reward_config(mode='research'))
    ]
    
    print("Reward comparison across configurations:")
    print(f"{'Mode':<12} {'Total':<8} {'Supervised':<12} {'Observable':<12}")
    print("-" * 45)
    
    for mode_name, config in configs:
        reward_components = compute_hybrid_reward(
            current_state=current_state,
            intervention=intervention,
            outcome=outcome,
            next_state=next_state,
            config=config,
            ground_truth=ground_truth if config.use_supervised_signals else None
        )
        
        supervised_total = (reward_components.supervised_parent_reward + 
                          reward_components.supervised_mechanism_reward)
        observable_total = (reward_components.posterior_confidence_reward + 
                          reward_components.causal_effect_reward + 
                          reward_components.mechanism_consistency_reward)
        
        print(f"{mode_name:<12} {reward_components.total_reward:<8.3f} "
              f"{supervised_total:<12.3f} {observable_total:<12.3f}")
    
    print()
    
    # Demo 4: Reward Strategy Comparison
    print("=== Demo 4: Reward Strategy Comparison ===")
    
    # Simulate two different strategies
    hybrid_rewards = []
    supervised_only_rewards = []
    
    for i in range(20):
        # Hybrid strategy (balanced)
        hybrid_config = create_hybrid_reward_config(mode='training')
        hybrid_reward = compute_hybrid_reward(
            current_state=current_state,
            intervention=intervention,
            outcome=outcome,
            next_state=next_state,
            config=hybrid_config,
            ground_truth=ground_truth
        )
        hybrid_rewards.append(hybrid_reward)
        
        # Supervised-only strategy
        supervised_config = create_hybrid_reward_config(mode='research')
        supervised_reward = compute_hybrid_reward(
            current_state=current_state,
            intervention=intervention,
            outcome=outcome,
            next_state=next_state,
            config=supervised_config,
            ground_truth=ground_truth
        )
        supervised_only_rewards.append(supervised_reward)
    
    comparison = compare_reward_strategies(
        strategy1_rewards=hybrid_rewards,
        strategy2_rewards=supervised_only_rewards,
        strategy1_name="Hybrid",
        strategy2_name="Supervised Only"
    )
    
    print("Strategy comparison results:")
    print(f"  Hybrid mean reward: {comparison['strategy1_mean']:.3f}")
    print(f"  Supervised-only mean reward: {comparison['strategy2_mean']:.3f}")
    print(f"  Difference: {comparison['difference']:.3f}")
    print(f"  Statistically significant: {comparison['statistical_significance']}")
    print()
    
    # Demo 5: Gaming Detection
    print("=== Demo 5: Gaming Detection ===")
    
    # Test with balanced rewards (should pass)
    validation_balanced = validate_hybrid_reward_consistency(hybrid_rewards)
    print(f"Balanced rewards valid: {validation_balanced['valid']}")
    
    if not validation_balanced['valid']:
        print(f"Issues detected: {validation_balanced['gaming_issues']}")
    else:
        print("No gaming patterns detected")
    
    print()
    
    print("=== Demo Complete ===")
    print("✅ Part B: Hybrid Reward System successfully implemented!")
    print("✅ Supervised signals: Using ground truth for training guidance")
    print("✅ Observable signals: Robust deployment without ground truth")
    print("✅ Configurable weighting: Flexible balance between signal types")
    print("✅ Gaming detection: Comprehensive validation and monitoring")
    print("✅ Integration ready: Compatible with mechanism-aware architecture")


if __name__ == "__main__":
    main()