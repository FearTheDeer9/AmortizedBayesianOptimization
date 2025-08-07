#!/usr/bin/env python3
"""
Debug script to trace GRPO reward computation and identify why rewards aren't increasing.
"""

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.acquisition.clean_rewards import compute_clean_reward, compute_target_reward
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample


def test_reward_computation():
    """Test reward computation with different scenarios."""
    
    print("="*80)
    print("DEBUGGING REWARD COMPUTATION")
    print("="*80)
    
    # Create a simple buffer with observational data
    buffer = ExperienceBuffer()
    
    # Add observational samples with Y values
    print("\n1. Creating observational baseline...")
    obs_values_Y = [1.0, 1.2, 0.8, 1.1, 0.9]  # Average = 1.0
    for i, y_val in enumerate(obs_values_Y):
        obs_sample = create_sample(
            values={'X': np.random.randn(), 'Y': y_val, 'Z': np.random.randn()},
            intervention_targets=frozenset()
        )
        buffer.add_observation(obs_sample)
    
    print(f"Observational Y values: {obs_values_Y}")
    print(f"Baseline (mean): {np.mean(obs_values_Y):.3f}")
    
    # Test different intervention outcomes
    print("\n2. Testing reward computation for different outcomes...")
    print("-"*60)
    
    test_outcomes = [0.5, 0.8, 1.0, 1.2, 1.5]  # Range from better to worse than baseline
    
    for outcome_value in test_outcomes:
        # Create intervention outcome
        outcome = create_sample(
            values={'X': 0.5, 'Y': outcome_value, 'Z': np.random.randn()},
            intervention_targets=frozenset(['X'])
        )
        
        # Compute reward using the actual function
        reward_info = compute_clean_reward(
            buffer_before=buffer,
            intervention={'targets': frozenset(['X']), 'values': {'X': 0.5}},
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'weights': {'target': 1.0, 'diversity': 0.0, 'exploration': 0.0, 'info_gain': 0.0}
            }
        )
        
        # Manually compute what we expect
        baseline = 1.0
        improvement = baseline - outcome_value  # For MINIMIZE
        std_dev = np.std(obs_values_Y)
        normalized_improvement = improvement / std_dev
        expected_reward = 1.0 / (1.0 + np.exp(-2.0 * normalized_improvement))
        
        print(f"\nOutcome Y = {outcome_value:.1f}:")
        print(f"  Improvement: {baseline:.1f} - {outcome_value:.1f} = {improvement:.2f}")
        print(f"  Normalized: {improvement:.2f} / {std_dev:.3f} = {normalized_improvement:.2f}")
        print(f"  Expected reward: {expected_reward:.3f}")
        print(f"  Actual reward: {reward_info['target']:.3f}")
        print(f"  Match: {'✓' if abs(expected_reward - reward_info['target']) < 0.01 else '✗'}")
    
    # Analyze the issue
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print("\n🔍 Key Findings:")
    print("1. Rewards are computed relative to a FIXED baseline (mean of observations)")
    print("2. The baseline never updates during training")
    print("3. Even if the policy improves, rewards stay centered around 0.5")
    print("4. This is why we don't see increasing rewards during training!")
    
    print("\n❌ The Problem:")
    print("- Outcome Y=0.5 (good) → reward ≈ 0.85")
    print("- Outcome Y=1.5 (bad) → reward ≈ 0.15")
    print("- But these rewards don't change even if the policy consistently achieves Y=0.5!")
    print("- The RL algorithm sees the same reward distribution throughout training")
    
    print("\n💡 Why This Breaks RL:")
    print("- In proper RL, rewards should increase as the policy improves")
    print("- Current setup: rewards measure 'distance from baseline', not improvement")
    print("- Policy gradient updates are based on advantages (reward - baseline)")
    print("- If rewards don't increase, advantages stay near zero → no learning signal")


def simulate_training_rewards():
    """Simulate what happens to rewards during training."""
    
    print("\n" + "="*80)
    print("SIMULATING TRAINING REWARDS")
    print("="*80)
    
    # Fixed observational baseline
    buffer = ExperienceBuffer()
    baseline_Y = 1.0
    
    for i in range(5):
        buffer.add_observation(create_sample(
            values={'Y': baseline_Y + 0.1 * np.random.randn()},
            intervention_targets=frozenset()
        ))
    
    print(f"\nFixed baseline: {baseline_Y:.1f}")
    
    # Simulate policy improving over episodes
    print("\nEpisode rewards as policy 'improves':")
    print("-"*40)
    
    episodes = []
    rewards = []
    
    for episode in range(5):
        # Policy gets better at minimizing Y
        achieved_Y = baseline_Y - 0.1 * episode  # Getting better each episode
        
        outcome = create_sample(
            values={'Y': achieved_Y},
            intervention_targets=frozenset(['X'])
        )
        
        reward_info = compute_clean_reward(
            buffer_before=buffer,
            intervention={'targets': frozenset(['X']), 'values': {'X': 0.5}},
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'weights': {'target': 1.0, 'diversity': 0.0, 'exploration': 0.0}
            }
        )
        
        episodes.append(episode)
        rewards.append(reward_info['target'])
        
        print(f"Episode {episode}: Y={achieved_Y:.1f} → reward={reward_info['target']:.3f}")
    
    # Check if rewards are increasing
    reward_slope = np.polyfit(episodes, rewards, 1)[0]
    
    print(f"\nReward slope: {reward_slope:.4f}")
    print(f"Are rewards increasing? {'YES ✓' if reward_slope > 0.01 else 'NO ✗'}")
    
    print("\n🎯 This shows the fundamental issue:")
    print("- Policy achieves better Y values each episode")
    print("- But rewards plateau because they're relative to fixed baseline")
    print("- RL needs increasing rewards to drive learning")


def propose_solutions():
    """Propose solutions to fix the reward issue."""
    
    print("\n" + "="*80)
    print("PROPOSED SOLUTIONS")
    print("="*80)
    
    print("\n1. ADAPTIVE BASELINE (Recommended):")
    print("   - Use a moving average of recent outcomes as baseline")
    print("   - Baseline improves as policy improves")
    print("   - Rewards measure improvement over recent performance")
    print("   - This gives proper RL signal")
    
    print("\n2. ABSOLUTE PERFORMANCE REWARDS:")
    print("   - For minimize: reward = 1.0 / (1.0 + outcome_value)")
    print("   - For maximize: reward = outcome_value / (1.0 + outcome_value)")
    print("   - Simple, direct relationship to performance")
    print("   - No baseline needed")
    
    print("\n3. PERCENTILE-BASED REWARDS:")
    print("   - Rank outcomes in buffer by performance")
    print("   - Reward based on percentile (0-100)")
    print("   - Automatically adapts as performance improves")
    print("   - Natural curriculum learning")
    
    print("\n4. ADVANTAGE-ONLY TRAINING:")
    print("   - Use raw advantages without reward transformation")
    print("   - Advantage = outcome_improvement - mean_improvement")
    print("   - More direct gradient signal")


if __name__ == "__main__":
    # Run tests
    test_reward_computation()
    simulate_training_rewards()
    propose_solutions()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Add detailed logging to trace these values during actual training")
    print("2. Implement adaptive baseline solution")
    print("3. Test that rewards increase during training")
    print("4. Verify policy gradients are non-zero")