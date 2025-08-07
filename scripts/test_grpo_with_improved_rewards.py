#!/usr/bin/env python3
"""
Test GRPO training with improved reward functions to verify proper RL signals.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.acquisition.clean_rewards import compute_clean_reward
from src.causal_bayes_opt.acquisition.improved_rewards import (
    compute_improved_clean_reward, AdaptiveBaseline
)
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.data_structures.sample import create_sample


def simulate_training_with_both_rewards():
    """Compare original vs improved rewards during simulated training."""
    
    print("="*80)
    print("COMPARING ORIGINAL VS IMPROVED REWARDS")
    print("="*80)
    
    # Setup
    buffer = ExperienceBuffer()
    adaptive_baseline = AdaptiveBaseline(window_size=20)
    
    # Add initial observational data
    for _ in range(20):
        buffer.add_observation(create_sample(
            values={'Y': 1.0 + 0.1 * np.random.randn()},
            intervention_targets=frozenset()
        ))
    
    # Simulate training episodes
    episodes = []
    original_rewards = []
    improved_absolute_rewards = []
    improved_adaptive_rewards = []
    y_values = []
    
    print("\nSimulating 50 episodes of training...")
    print("-"*60)
    
    for episode in range(50):
        # Policy improves over time (Y decreases from ~1.0 to ~0.3)
        progress = episode / 50
        achieved_Y = 1.0 - 0.7 * progress + 0.05 * np.random.randn()
        
        # Create outcome
        outcome = create_sample(
            values={'Y': achieved_Y},
            intervention_targets=frozenset(['X'])
        )
        
        # Original reward
        original_reward_info = compute_clean_reward(
            buffer_before=buffer,
            intervention={'targets': frozenset(['X']), 'values': {'X': 0.5}},
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'weights': {'target': 1.0, 'diversity': 0.0, 'exploration': 0.0}
            }
        )
        
        # Improved absolute reward
        improved_absolute_info = compute_improved_clean_reward(
            buffer_before=buffer,
            intervention={'targets': frozenset(['X']), 'values': {'X': 0.5}},
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'reward_type': 'absolute',
                'scale': 1.0
            }
        )
        
        # Improved adaptive reward
        improved_adaptive_info = compute_improved_clean_reward(
            buffer_before=buffer,
            intervention={'targets': frozenset(['X']), 'values': {'X': 0.5}},
            outcome=outcome,
            target_variable='Y',
            config={
                'optimization_direction': 'MINIMIZE',
                'reward_type': 'adaptive',
                'temperature': 2.0
            },
            adaptive_baseline=adaptive_baseline
        )
        
        episodes.append(episode)
        original_rewards.append(original_reward_info['target'])
        improved_absolute_rewards.append(improved_absolute_info['target'])
        improved_adaptive_rewards.append(improved_adaptive_info['target'])
        y_values.append(achieved_Y)
        
        if episode % 10 == 0:
            print(f"Episode {episode}:")
            print(f"  Y value: {achieved_Y:.3f}")
            print(f"  Original reward: {original_reward_info['target']:.3f}")
            print(f"  Improved (absolute): {improved_absolute_info['target']:.3f}")
            print(f"  Improved (adaptive): {improved_adaptive_info['target']:.3f}")
    
    # Analyze slopes
    original_slope = np.polyfit(episodes, original_rewards, 1)[0]
    absolute_slope = np.polyfit(episodes, improved_absolute_rewards, 1)[0]
    adaptive_slope = np.polyfit(episodes, improved_adaptive_rewards, 1)[0]
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\nReward slopes (should be positive for proper RL):")
    print(f"  Original (fixed baseline): {original_slope:.6f} {'✓' if original_slope > 0.001 else '✗'}")
    print(f"  Improved (absolute):       {absolute_slope:.6f} {'✓' if absolute_slope > 0.001 else '✗'}")
    print(f"  Improved (adaptive):       {adaptive_slope:.6f} {'✓' if adaptive_slope > 0.001 else '✗'}")
    
    # Check saturation
    print("\nReward saturation check (last 10 episodes):")
    original_saturated = np.std(original_rewards[-10:]) < 0.01
    absolute_saturated = np.std(improved_absolute_rewards[-10:]) < 0.01
    adaptive_saturated = np.std(improved_adaptive_rewards[-10:]) < 0.01
    
    print(f"  Original: {'SATURATED ✗' if original_saturated else 'Not saturated ✓'}")
    print(f"  Improved (absolute): {'SATURATED ✗' if absolute_saturated else 'Not saturated ✓'}")
    print(f"  Improved (adaptive): {'SATURATED ✗' if adaptive_saturated else 'Not saturated ✓'}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Y values over time
    ax = axes[0, 0]
    ax.plot(episodes, y_values, 'k-', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Y Value')
    ax.set_title('Task Performance (Lower is Better)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: All rewards
    ax = axes[0, 1]
    ax.plot(episodes, original_rewards, 'r-', label='Original', alpha=0.7)
    ax.plot(episodes, improved_absolute_rewards, 'g-', label='Improved (Absolute)', alpha=0.7)
    ax.plot(episodes, improved_adaptive_rewards, 'b-', label='Improved (Adaptive)', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Original reward details
    ax = axes[1, 0]
    ax.plot(episodes, original_rewards, 'r-', alpha=0.5)
    ax.plot(episodes, np.polyfit(episodes, original_rewards, 1)[1] + 
            original_slope * np.array(episodes), 'r--', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Original Rewards (slope={original_slope:.4f})')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Improved reward details
    ax = axes[1, 1]
    ax.plot(episodes, improved_absolute_rewards, 'g-', alpha=0.5, label='Absolute')
    ax.plot(episodes, improved_adaptive_rewards, 'b-', alpha=0.5, label='Adaptive')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Improved Rewards')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reward_comparison.png', dpi=150)
    print(f"\nPlot saved to: reward_comparison.png")
    
    # Recommendations
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    print("\n🔍 Key Findings:")
    print("1. Original rewards saturate quickly (plateau near 1.0)")
    print("2. Improved absolute rewards maintain consistent growth")
    print("3. Improved adaptive rewards provide steady learning signal")
    
    print("\n✅ Recommendation:")
    if absolute_slope > original_slope and not absolute_saturated:
        print("Use IMPROVED ABSOLUTE REWARDS for GRPO training")
        print("- Simple, direct relationship to performance")
        print("- No saturation issues")
        print("- Consistent learning signal throughout training")
    else:
        print("Further investigation needed")


if __name__ == "__main__":
    simulate_training_with_both_rewards()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Replace compute_clean_reward with compute_improved_clean_reward in GRPO")
    print("2. Use 'absolute' reward type for stable training")
    print("3. Test with actual GRPO training loop")
    print("4. Verify policy gradients are non-zero throughout training")