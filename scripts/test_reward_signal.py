#!/usr/bin/env python3
"""
Test script to verify that reward signals behave correctly for RL.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.acquisition.clean_rewards import compute_clean_reward, compute_target_reward
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample


def test_current_reward_function():
    """Test how current reward function behaves as policy improves."""
    
    print("="*80)
    print("TESTING CURRENT REWARD FUNCTION")
    print("="*80)
    
    # Setup: Create buffer with observational baseline
    buffer = ExperienceBuffer()
    baseline_Y = 1.0
    
    # Add observational data
    for _ in range(100):
        buffer.add_observation(create_sample(
            values={'Y': baseline_Y + 0.2 * np.random.randn()},
            intervention_targets=frozenset()
        ))
    
    print(f"\nBaseline Y value: {baseline_Y:.1f}")
    
    # Simulate policy improving over time
    episodes = []
    rewards = []
    y_values = []
    
    print("\nSimulating policy improvement over 50 episodes:")
    print("-"*50)
    
    for episode in range(50):
        # Policy gradually learns to minimize Y
        # Early: Y ~ 1.0, Late: Y ~ 0.3
        progress = episode / 50
        achieved_Y = baseline_Y - 0.7 * progress + 0.1 * np.random.randn()
        
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
        y_values.append(achieved_Y)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Y={achieved_Y:.2f}, Reward={reward_info['target']:.3f}")
    
    # Analyze results
    reward_slope = np.polyfit(episodes, rewards, 1)[0]
    y_slope = np.polyfit(episodes, y_values, 1)[0]
    
    print(f"\nResults:")
    print(f"  Y-value slope: {y_slope:.4f} (negative = improving)")
    print(f"  Reward slope: {reward_slope:.6f}")
    print(f"  Are rewards increasing? {'YES ✓' if reward_slope > 0.001 else 'NO ✗'}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Y values over time
    ax1.scatter(episodes, y_values, alpha=0.5, s=10)
    ax1.plot(episodes, np.polyfit(episodes, y_values, 1)[1] + np.polyfit(episodes, y_values, 1)[0] * np.array(episodes), 'r--', label='Trend')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Y Value (lower is better)')
    ax1.set_title('Task Performance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rewards over time
    ax2.scatter(episodes, rewards, alpha=0.5, s=10)
    ax2.plot(episodes, np.polyfit(episodes, rewards, 1)[1] + np.polyfit(episodes, rewards, 1)[0] * np.array(episodes), 'r--', label='Trend')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.set_title('RL Reward Signal Over Time')
    ax2.axhline(y=0.5, color='k', linestyle=':', alpha=0.5, label='Neutral (Y=baseline)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('current_reward_behavior.png', dpi=150)
    print(f"\nPlot saved to: current_reward_behavior.png")
    
    return reward_slope


def test_adaptive_baseline_reward():
    """Test reward function with adaptive baseline."""
    
    print("\n" + "="*80)
    print("TESTING ADAPTIVE BASELINE REWARD")
    print("="*80)
    
    # Track recent outcomes for adaptive baseline
    recent_outcomes = []
    window_size = 20
    
    episodes = []
    rewards = []
    baselines = []
    
    print("\nSimulating with adaptive baseline:")
    print("-"*50)
    
    for episode in range(50):
        # Policy improves over time
        progress = episode / 50
        achieved_Y = 1.0 - 0.7 * progress + 0.1 * np.random.randn()
        
        # Compute adaptive baseline
        if len(recent_outcomes) > 5:
            adaptive_baseline = np.mean(recent_outcomes[-window_size:])
        else:
            adaptive_baseline = 1.0  # Initial estimate
        
        # Compute reward relative to adaptive baseline
        improvement = adaptive_baseline - achieved_Y  # For MINIMIZE
        reward = 1.0 / (1.0 + np.exp(-2.0 * improvement))
        
        # Update recent outcomes
        recent_outcomes.append(achieved_Y)
        
        episodes.append(episode)
        rewards.append(reward)
        baselines.append(adaptive_baseline)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Y={achieved_Y:.2f}, Baseline={adaptive_baseline:.2f}, Reward={reward:.3f}")
    
    # Analyze
    reward_slope = np.polyfit(episodes, rewards, 1)[0]
    
    print(f"\nResults with adaptive baseline:")
    print(f"  Reward slope: {reward_slope:.6f}")
    print(f"  Are rewards increasing? {'YES ✓' if reward_slope > 0.001 else 'NO ✗'}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, 'b-', alpha=0.7, label='Rewards')
    plt.plot(episodes, baselines, 'g--', alpha=0.7, label='Adaptive Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Adaptive Baseline Reward System')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('adaptive_baseline_rewards.png', dpi=150)
    print(f"\nPlot saved to: adaptive_baseline_rewards.png")
    
    return reward_slope


def test_absolute_performance_reward():
    """Test absolute performance-based rewards."""
    
    print("\n" + "="*80)
    print("TESTING ABSOLUTE PERFORMANCE REWARD")
    print("="*80)
    
    episodes = []
    rewards = []
    
    print("\nSimulating with absolute performance rewards:")
    print("-"*50)
    
    for episode in range(50):
        # Policy improves over time
        progress = episode / 50
        achieved_Y = 1.0 - 0.7 * progress + 0.1 * np.random.randn()
        
        # Absolute performance reward for minimization
        # Maps Y values to rewards: Y=0 → reward≈1, Y=1 → reward=0.5, Y=∞ → reward→0
        reward = 1.0 / (1.0 + achieved_Y)
        
        episodes.append(episode)
        rewards.append(reward)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Y={achieved_Y:.2f}, Reward={reward:.3f}")
    
    # Analyze
    reward_slope = np.polyfit(episodes, rewards, 1)[0]
    
    print(f"\nResults with absolute performance rewards:")
    print(f"  Reward slope: {reward_slope:.6f}")
    print(f"  Are rewards increasing? {'YES ✓' if reward_slope > 0.001 else 'NO ✗'}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, rewards, alpha=0.5, s=20)
    plt.plot(episodes, np.polyfit(episodes, rewards, 1)[1] + np.polyfit(episodes, rewards, 1)[0] * np.array(episodes), 
             'r--', linewidth=2, label=f'Trend (slope={reward_slope:.4f})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Absolute Performance Reward System')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('absolute_performance_rewards.png', dpi=150)
    print(f"\nPlot saved to: absolute_performance_rewards.png")
    
    return reward_slope


def compare_reward_functions():
    """Compare different reward functions."""
    
    print("\n" + "="*80)
    print("COMPARISON OF REWARD FUNCTIONS")
    print("="*80)
    
    # Test all approaches
    current_slope = test_current_reward_function()
    adaptive_slope = test_adaptive_baseline_reward()
    absolute_slope = test_absolute_performance_reward()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nReward Function Slopes (higher is better for RL):")
    print(f"1. Current (fixed baseline):    {current_slope:.6f} {'✗' if current_slope < 0.001 else '✓'}")
    print(f"2. Adaptive baseline:           {adaptive_slope:.6f} {'✗' if adaptive_slope < 0.001 else '✓'}")
    print(f"3. Absolute performance:        {absolute_slope:.6f} {'✗' if absolute_slope < 0.001 else '✓'}")
    
    print("\n🔍 Analysis:")
    print("- Current system: Rewards plateau as policy improves (bad for RL)")
    print("- Adaptive baseline: Maintains learning signal throughout training")
    print("- Absolute performance: Direct relationship between performance and reward")
    
    print("\n✅ Recommendation:")
    if adaptive_slope > current_slope and adaptive_slope > absolute_slope:
        print("Use ADAPTIVE BASELINE - provides consistent learning signal")
    elif absolute_slope > current_slope:
        print("Use ABSOLUTE PERFORMANCE - simple and effective")
    else:
        print("Current system needs fixing - rewards must increase with performance!")


if __name__ == "__main__":
    compare_reward_functions()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Implement chosen reward function in clean_rewards.py")
    print("2. Test with actual GRPO training")
    print("3. Verify rewards increase during training")
    print("4. Check that policy gradients are non-zero")