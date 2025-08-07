#!/usr/bin/env python3
"""
Test GRPO training with improved reward functions to verify rewards increase.
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

# Configure logging to capture reward details
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('improved_grpo_test.log'),
        logging.StreamHandler()
    ]
)

print("="*80)
print("TESTING GRPO WITH IMPROVED REWARDS")
print("="*80)
print("\nObjective: Verify that rewards increase during training")
print("Expected: Positive reward slope throughout training")
print("="*80)

# Create simple fork SCM
scm = create_fork_scm()
print("\nSCM: Fork (X -> Y <- Z)")
print("Target: Y (minimize)")
print("With improved absolute rewards:")
print("- Y=2.0 → reward≈0.33")
print("- Y=1.0 → reward=0.50")
print("- Y=0.5 → reward≈0.67")
print("- Y=0.1 → reward≈0.91")

# Configure trainer
trainer = UnifiedGRPOTrainer(
    learning_rate=3e-4,
    n_episodes=50,
    episode_length=20,
    batch_size=32,
    architecture_level="baseline",
    optimization_direction="MINIMIZE",
    seed=42,
    use_surrogate=False,
    checkpoint_dir="checkpoints/improved_reward_test",
    reward_weights={
        'optimization': 1.0,  # Only target rewards to test core functionality
        'discovery': 0.0,
        'efficiency': 0.0,
        'info_gain': 0.0
    }
)

print(f"\nTraining for 50 episodes...")
print("Watch for:")
print("1. [IMPROVED REWARD] messages showing reward computation")
print("2. [REWARD TREND] messages showing improvement")
print("3. Episode rewards that increase over time")
print("-"*60)

# Train
metrics = trainer.train({"fork": scm})

# Extract rewards from log
print("\n" + "="*80)
print("ANALYZING RESULTS")
print("="*80)

rewards = []
with open('improved_grpo_test.log', 'r') as f:
    for line in f:
        if "Episode" in line and "mean_reward=" in line and "current_scm=" in line:
            # Parse: Episode X: mean_reward=Y
            parts = line.split("mean_reward=")
            if len(parts) > 1:
                reward_str = parts[1].split(",")[0]
                try:
                    reward = float(reward_str)
                    rewards.append(reward)
                except ValueError:
                    pass

print(f"\nExtracted {len(rewards)} episode rewards")

if len(rewards) >= 20:
    rewards = np.array(rewards)
    episodes = np.arange(len(rewards))
    
    # Calculate statistics
    early_rewards = rewards[:10]
    late_rewards = rewards[-10:]
    
    early_avg = np.mean(early_rewards)
    late_avg = np.mean(late_rewards)
    
    # Linear regression
    slope = np.polyfit(episodes, rewards, 1)[0]
    
    print("\nRESULTS:")
    print(f"  Early episodes (0-9) average: {early_avg:.4f}")
    print(f"  Late episodes (last 10) average: {late_avg:.4f}")
    print(f"  Change: {late_avg - early_avg:+.4f}")
    print(f"  Linear slope: {slope:.6f}")
    
    print("\n" + "-"*60)
    if slope > 0.001:
        print("✅ SUCCESS! Rewards are increasing during training!")
        print("   The improved reward function is working correctly.")
    elif slope > 0:
        print("✓ Positive slope detected, but small.")
        print("  May need more episodes or tuning.")
    else:
        print("❌ No positive slope detected.")
        print("  Check the log for issues.")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(episodes, rewards, alpha=0.5, s=20, label='Episode rewards')
    
    # Add trend line
    trend = np.poly1d(np.polyfit(episodes, rewards, 1))
    plt.plot(episodes, trend(episodes), 'r--', linewidth=2, 
             label=f'Trend (slope={slope:.4f})')
    
    # Add moving average
    if len(rewards) > 10:
        window = 10
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 'g-', 
                linewidth=2, alpha=0.8, label='Moving avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Mean Reward')
    plt.title('GRPO Training with Improved Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('improved_grpo_rewards.png', dpi=150)
    print(f"\nPlot saved to: improved_grpo_rewards.png")
    
else:
    print("\nNot enough episodes found in log.")
    print("Check if training completed successfully.")

# Look for specific reward values in log
print("\n" + "="*80)
print("SAMPLE REWARD COMPUTATIONS")
print("="*80)

improved_rewards = []
with open('improved_grpo_test.log', 'r') as f:
    for line in f:
        if "[IMPROVED REWARD]" in line:
            # Extract outcome and reward values
            if "Outcome:" in line and "Target reward:" in line:
                try:
                    outcome = float(line.split("Outcome:")[1].split(",")[0])
                    reward = float(line.split("Target reward:")[1].strip())
                    improved_rewards.append((outcome, reward))
                except:
                    pass

if improved_rewards:
    print("\nSample improved reward computations:")
    for i, (outcome, reward) in enumerate(improved_rewards[:5]):
        print(f"  Y={outcome:.3f} → reward={reward:.3f}")
    
    # Check if rewards follow expected pattern
    outcomes = [o for o, r in improved_rewards]
    rewards_list = [r for o, r in improved_rewards]
    
    # For minimization with absolute rewards, lower Y should give higher reward
    if len(outcomes) > 1:
        correlation = np.corrcoef(outcomes, rewards_list)[0, 1]
        print(f"\nCorrelation between Y and reward: {correlation:.3f}")
        if correlation < -0.5:
            print("✅ Good! Lower Y values produce higher rewards (minimization)")
        else:
            print("⚠️  Unexpected correlation for minimization task")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("The improved reward function should provide:")
print("1. Rewards that increase as the policy improves")
print("2. No saturation (rewards don't plateau at 1.0)")
print("3. Direct relationship between performance and reward")
print("\nCheck the plots and logs to verify these properties!")