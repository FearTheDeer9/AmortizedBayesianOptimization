#!/usr/bin/env python3
"""
Simple test to show reward improvement with GRPO fixes.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

print("SIMPLE GRPO REWARD IMPROVEMENT TEST")
print("="*60)

# Create fork SCM
scm = create_fork_scm()

# Train for 50 episodes
trainer = UnifiedGRPOTrainer(
    learning_rate=3e-4,
    n_episodes=50,
    episode_length=10,
    batch_size=16,
    architecture_level="baseline",
    optimization_direction="MINIMIZE",
    seed=42,
    use_surrogate=False,
    checkpoint_dir="checkpoints/simple_reward_test",
    reward_weights={
        'optimization': 1.0,  # Only target reward
        'discovery': 0.0,
        'efficiency': 0.0,
        'info_gain': 0.0
    }
)

print("\nTraining for 50 episodes (this may take a minute)...")
print("Target: Y (minimize)")
print("Causal parents: X, Z")
print("-"*60)

# Capture episode rewards
episode_rewards = []

# Hook into trainer to capture rewards (hacky but works)
original_log = trainer._log_episode_metrics

def capture_rewards(episode_idx, metrics, elapsed_time):
    if 'mean_reward' in metrics:
        # Since minimizing, lower is better
        episode_rewards.append(metrics['mean_reward'])
        # Print progress every 10 episodes
        if episode_idx % 10 == 0:
            print(f"Episode {episode_idx}: reward = {metrics['mean_reward']:.3f}")
    return original_log(episode_idx, metrics, elapsed_time)

trainer._log_episode_metrics = capture_rewards

# Train
metrics = trainer.train({"fork": scm})

print("\n" + "="*60)
print("RESULTS")
print("="*60)

if len(episode_rewards) > 10:
    # Since we're minimizing, improvement means rewards decrease
    early_rewards = episode_rewards[:10]
    late_rewards = episode_rewards[-10:]
    
    early_avg = np.mean(early_rewards)
    late_avg = np.mean(late_rewards)
    
    # For minimization: improvement = early - late (positive if improving)
    improvement = early_avg - late_avg
    improvement_pct = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
    
    print(f"\nEarly episodes (0-9) average: {early_avg:.3f}")
    print(f"Late episodes (last 10) average: {late_avg:.3f}")
    print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
    
    # Calculate slope (negative slope is good for minimization)
    episodes = np.arange(len(episode_rewards))
    slope = np.polyfit(episodes, episode_rewards, 1)[0]
    
    print(f"\nLinear trend slope: {slope:.6f}")
    
    if slope < 0:
        print("✅ NEGATIVE SLOPE = IMPROVING! (minimizing target)")
    else:
        print("❌ Positive/flat slope - not improving")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, 'b-', alpha=0.5, label='Episode rewards')
    
    # Add smoothed curve
    window = 5
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), smoothed, 'r-', linewidth=2, label='Smoothed')
    
    # Add trend line
    plt.plot(episodes, slope * episodes + np.polyfit(episodes, episode_rewards, 1)[1], 
             'g--', linewidth=2, label=f'Trend (slope={slope:.4f})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward (lower is better)')
    plt.title('GRPO Learning Progress - Fork SCM')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('simple_reward_progress.png', dpi=150)
    print(f"\nPlot saved to: simple_reward_progress.png")
    
    # Show intervention diversity
    print("\n" + "-"*60)
    print("INTERVENTION ANALYSIS")
    print("-"*60)
    print("Check the training output above for:")
    print("1. Variable diversity (not stuck on one variable)")
    print("2. Exploration in early episodes")
    print("3. Focus on causal parents in later episodes")
    
else:
    print("Not enough episodes to analyze trends")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("With the fixes:")
print("- Exploration noise enables trying different variables")
print("- Non-intervention baseline provides learning signal")
print("- Policy can learn which interventions help")
print("- Rewards should improve (decrease) over time")