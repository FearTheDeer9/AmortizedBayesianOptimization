#!/usr/bin/env python3
"""
Direct test of information gain reward computation.
Shows how rewards differ when surrogate posteriors are provided.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.acquisition.clean_rewards import (
    compute_clean_reward, compute_information_gain_reward
)
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample
from src.causal_bayes_opt.experiments.test_scms import create_fork_test_scm
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm


def test_info_gain_computation():
    """Test information gain reward computation with different entropy scenarios."""
    
    print("="*60)
    print("TESTING INFORMATION GAIN REWARD COMPUTATION")
    print("="*60)
    print()
    
    # Create test SCM and buffer
    scm = create_fork_test_scm()
    buffer = ExperienceBuffer()
    
    # Add some observational data
    obs_samples = sample_from_linear_scm(scm, 50, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Test intervention
    intervention = {
        'targets': frozenset(['X']),
        'values': {'X': 1.0}
    }
    
    # Create outcome sample
    outcome = create_sample(
        values={'X': 1.0, 'Y': 0.5, 'Z': -0.3},
        intervention_type='perfect',
        intervention_targets=frozenset(['X'])
    )
    
    # Test 1: No surrogate (no posteriors)
    print("Test 1: Without surrogate (no information gain)")
    reward_no_surrogate = compute_clean_reward(
        buffer_before=buffer,
        intervention=intervention,
        outcome=outcome,
        target_variable='Y',
        config={
            'optimization_direction': 'MINIMIZE',
            'weights': {
                'target': 0.7,
                'diversity': 0.1,
                'exploration': 0.1,
                'info_gain': 0.1
            }
        }
    )
    
    print(f"  Total reward: {reward_no_surrogate['total']:.4f}")
    print(f"  Target reward: {reward_no_surrogate['target']:.4f}")
    print(f"  Info gain reward: {reward_no_surrogate['info_gain']:.4f}")
    print()
    
    # Test 2: With surrogate (simulated entropy reduction)
    print("Test 2: With surrogate (high information gain)")
    
    # Simulate high entropy before (uncertain)
    posterior_before = {
        'entropy': 2.0,  # High uncertainty
        'marginal_parent_probs': {'X': 0.5, 'Z': 0.5}
    }
    
    # Simulate low entropy after (more certain)
    posterior_after = {
        'entropy': 0.5,  # Much more certain
        'marginal_parent_probs': {'X': 0.9, 'Z': 0.1}
    }
    
    reward_with_surrogate = compute_clean_reward(
        buffer_before=buffer,
        intervention=intervention,
        outcome=outcome,
        target_variable='Y',
        config={
            'optimization_direction': 'MINIMIZE',
            'weights': {
                'target': 0.7,
                'diversity': 0.1,
                'exploration': 0.1,
                'info_gain': 0.1
            }
        },
        posterior_before=posterior_before,
        posterior_after=posterior_after
    )
    
    print(f"  Total reward: {reward_with_surrogate['total']:.4f}")
    print(f"  Target reward: {reward_with_surrogate['target']:.4f}")
    print(f"  Info gain reward: {reward_with_surrogate['info_gain']:.4f}")
    print()
    
    # Test 3: Low information gain scenario
    print("Test 3: Low information gain")
    
    # Similar entropy before and after
    posterior_before_low = {
        'entropy': 1.0,
        'marginal_parent_probs': {'X': 0.7, 'Z': 0.3}
    }
    
    posterior_after_low = {
        'entropy': 0.95,  # Only slight reduction
        'marginal_parent_probs': {'X': 0.72, 'Z': 0.28}
    }
    
    reward_low_info = compute_clean_reward(
        buffer_before=buffer,
        intervention=intervention,
        outcome=outcome,
        target_variable='Y',
        config={
            'optimization_direction': 'MINIMIZE',
            'weights': {
                'target': 0.7,
                'diversity': 0.1,
                'exploration': 0.1,
                'info_gain': 0.1
            }
        },
        posterior_before=posterior_before_low,
        posterior_after=posterior_after_low
    )
    
    print(f"  Total reward: {reward_low_info['total']:.4f}")
    print(f"  Target reward: {reward_low_info['target']:.4f}")
    print(f"  Info gain reward: {reward_low_info['info_gain']:.4f}")
    print()
    
    # Test 4: High weight on information gain
    print("Test 4: High weight on information gain (0.5)")
    
    reward_high_weight = compute_clean_reward(
        buffer_before=buffer,
        intervention=intervention,
        outcome=outcome,
        target_variable='Y',
        config={
            'optimization_direction': 'MINIMIZE',
            'weights': {
                'target': 0.3,
                'diversity': 0.1,
                'exploration': 0.1,
                'info_gain': 0.5  # High weight on info gain
            }
        },
        posterior_before=posterior_before,
        posterior_after=posterior_after
    )
    
    print(f"  Total reward: {reward_high_weight['total']:.4f}")
    print(f"  Target reward: {reward_high_weight['target']:.4f}")
    print(f"  Info gain reward: {reward_high_weight['info_gain']:.4f}")
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print()
    print("Reward differences due to information gain:")
    print(f"  No surrogate total: {reward_no_surrogate['total']:.4f}")
    print(f"  With surrogate (high IG) total: {reward_with_surrogate['total']:.4f}")
    print(f"  Difference: {reward_with_surrogate['total'] - reward_no_surrogate['total']:.4f}")
    print()
    print(f"  Low info gain total: {reward_low_info['total']:.4f}")
    print(f"  Difference from no surrogate: {reward_low_info['total'] - reward_no_surrogate['total']:.4f}")
    print()
    print(f"  High IG weight total: {reward_high_weight['total']:.4f}")
    print(f"  Difference from low weight: {reward_high_weight['total'] - reward_with_surrogate['total']:.4f}")
    
    # Test the raw information gain function
    print("\n" + "="*60)
    print("RAW INFORMATION GAIN COMPUTATION")
    print("="*60)
    print()
    
    # Test different entropy reductions
    test_cases = [
        (2.0, 1.0, "Medium reduction"),
        (2.0, 0.5, "Large reduction"),
        (1.0, 0.95, "Small reduction"),
        (1.0, 1.0, "No reduction"),
        (0.5, 1.0, "Negative (should be 0)")
    ]
    
    for entropy_before, entropy_after, description in test_cases:
        info_gain_reward = compute_information_gain_reward(
            {'entropy': entropy_before},
            {'entropy': entropy_after}
        )
        raw_gain = max(0, entropy_before - entropy_after)
        print(f"{description}:")
        print(f"  Entropy: {entropy_before:.2f} -> {entropy_after:.2f}")
        print(f"  Raw gain: {raw_gain:.2f}")
        print(f"  Reward: {info_gain_reward:.4f}")
        print()


if __name__ == "__main__":
    test_info_gain_computation()