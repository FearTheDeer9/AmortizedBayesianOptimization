#!/usr/bin/env python3
"""
Final comprehensive test for GRPO reward improvement.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

# Set up logging to capture all episodes
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('final_reward_test.log'),
        logging.StreamHandler()
    ]
)

print("="*80)
print("FINAL GRPO REWARD SLOPE TEST")
print("="*80)
print("\nObjective: Test if GRPO shows improving rewards after fixes")
print("Expected: Negative slope (decreasing rewards) for minimization")
print("="*80)

# Create SCM
scm = create_fork_scm()
print("\nSCM: Fork (X -> Y <- Z)")
print("Target: Y (minimize)")
print("Optimal: Intervene on X or Z (causal parents)")

# Configure trainer with more episodes
trainer = UnifiedGRPOTrainer(
    learning_rate=3e-4,
    n_episodes=100,  # More episodes for clear trend
    episode_length=20,  # Longer episodes for stable estimates
    batch_size=32,
    architecture_level="baseline",
    optimization_direction="MINIMIZE",
    seed=42,
    use_surrogate=False,
    checkpoint_dir="checkpoints/final_slope_test",
    reward_weights={
        'optimization': 0.9,  # Mostly target
        'discovery': 0.1,     # Small discovery bonus
        'efficiency': 0.0,
        'info_gain': 0.0
    }
)

print(f"\nTraining for 100 episodes...")
print("This will take 2-3 minutes...")
print("-"*60)

# Train
metrics = trainer.train({"fork": scm})

print("\n" + "="*80)
print("EXTRACTING RESULTS")
print("="*80)

# Extract rewards from log file
rewards = []
with open('final_reward_test.log', 'r') as f:
    for line in f:
        if "Episode" in line and "mean_reward=" in line:
            # Parse: Episode X: mean_reward=Y
            parts = line.split("mean_reward=")
            if len(parts) > 1:
                reward = float(parts[1].split(",")[0])
                rewards.append(reward)

print(f"\nExtracted {len(rewards)} episode rewards")

if len(rewards) >= 20:
    rewards = np.array(rewards)
    episodes = np.arange(len(rewards))
    
    # Calculate statistics
    early_rewards = rewards[:20]
    late_rewards = rewards[-20:]
    
    early_avg = np.mean(early_rewards)
    early_std = np.std(early_rewards)
    late_avg = np.mean(late_rewards)
    late_std = np.std(late_rewards)
    
    # Linear regression
    coeffs = np.polyfit(episodes, rewards, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nEarly episodes (0-19):")
    print(f"  Average: {early_avg:.4f} ± {early_std:.4f}")
    
    print(f"\nLate episodes (last 20):")
    print(f"  Average: {late_avg:.4f} ± {late_std:.4f}")
    
    # For minimization: improvement = early - late
    improvement = early_avg - late_avg
    improvement_pct = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
    
    print(f"\nChange: {late_avg - early_avg:+.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement_pct:+.1f}%)")
    
    print(f"\nLinear regression:")
    print(f"  Slope: {slope:.6f}")
    print(f"  R² score: {1 - np.var(rewards - (slope * episodes + intercept)) / np.var(rewards):.3f}")
    
    print("\n" + "-"*60)
    if slope < -0.0001:
        print("✅ SUCCESS! Negative slope - rewards improving over time!")
        print("   The GRPO fixes are working!")
    elif slope > 0.0001:
        print("❌ Positive slope - rewards getting worse")
        print("   This suggests the fixes may need tuning")
    else:
        print("➖ Essentially flat - no clear improvement")
        print("   May need more episodes or parameter tuning")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Raw rewards with trend
    ax = axes[0, 0]
    ax.scatter(episodes, rewards, alpha=0.5, s=10, c='blue', label='Episode rewards')
    ax.plot(episodes, slope * episodes + intercept, 'r-', linewidth=2, 
            label=f'Trend (slope={slope:.4f})')
    
    # Moving average
    window = 10
    if len(rewards) > window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], moving_avg, 'g-', linewidth=2, alpha=0.8, label='Moving avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (lower is better)')
    ax.set_title('Episode Rewards with Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Binned averages
    ax = axes[0, 1]
    n_bins = 10
    bin_size = len(rewards) // n_bins
    bin_means = []
    bin_stds = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(rewards)
        bin_rewards = rewards[start:end]
        bin_means.append(np.mean(bin_rewards))
        bin_stds.append(np.std(bin_rewards) / np.sqrt(len(bin_rewards)))  # SEM
    
    x = np.arange(n_bins)
    ax.bar(x, bin_means, yerr=bin_stds, capsize=5, alpha=0.7, color='skyblue')
    
    # Trend on bins
    bin_slope = np.polyfit(x, bin_means, 1)[0]
    ax.plot(x, np.poly1d(np.polyfit(x, bin_means, 1))(x), 'r--', linewidth=2)
    
    ax.set_xlabel('Training Decile')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Rewards by Training Phase (bin slope={bin_slope:.4f})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(n_bins)])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Early vs Late comparison
    ax = axes[1, 0]
    data = [early_rewards, late_rewards]
    positions = [0, 1]
    bp = ax.boxplot(data, positions=positions, patch_artist=True, 
                    labels=['Early\n(0-19)', 'Late\n(80-99)'])
    
    colors = ['lightcoral', 'lightgreen'] if improvement > 0 else ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Reward Distribution')
    ax.set_title(f'Early vs Late Episodes (diff={improvement:.3f})')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative average
    ax = axes[1, 1]
    cumulative_avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    ax.plot(episodes, cumulative_avg, 'b-', linewidth=2)
    ax.axhline(y=early_avg, color='r', linestyle='--', alpha=0.5, label='Early avg')
    ax.axhline(y=late_avg, color='g', linestyle='--', alpha=0.5, label='Late avg')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Average Reward')
    ax.set_title('Cumulative Average Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_reward_slope_analysis.png', dpi=150)
    print(f"\nDetailed plots saved to: final_reward_slope_analysis.png")
    
    # Summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if slope < -0.0001 and improvement > 0.01:
        print("🎉 The GRPO fixes are working!")
        print("- Policy explores different variables")
        print("- Non-intervention baseline provides learning signal")
        print("- Rewards improve over time")
    else:
        print("The fixes enable exploration and learning signals, but:")
        print("- May need more training episodes")
        print("- Could benefit from hyperparameter tuning")
        print("- Consider enabling surrogate for info gain rewards")
        
else:
    print("\n❌ Not enough episodes found in log")
    print("Check if training completed successfully")

print("\n" + "="*80)
print("To further improve results:")
print("1. Train longer (200+ episodes)")
print("2. Tune exploration_noise (try 0.1, 0.2, 0.5)")
print("3. Enable surrogate (use_surrogate=True)")
print("4. Adjust learning rate (try 1e-3, 1e-4)")
print("="*80)