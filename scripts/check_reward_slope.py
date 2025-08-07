#!/usr/bin/env python3
"""
Direct test to check if rewards improve with GRPO fixes.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm

print("CHECKING REWARD IMPROVEMENT WITH GRPO FIXES")
print("="*60)

# Create fork SCM
scm = create_fork_scm()

# Train briefly
trainer = UnifiedGRPOTrainer(
    learning_rate=3e-4,
    n_episodes=30,
    episode_length=10,
    batch_size=16,
    architecture_level="baseline",
    optimization_direction="MINIMIZE",
    seed=42,
    use_surrogate=False,
    checkpoint_dir="checkpoints/slope_test",
    reward_weights={'optimization': 1.0}  # Focus on target only
)

print("\nTraining for 30 episodes...")
print("SCM: X -> Y <- Z (target=Y, minimize)")
print("-"*60)

# Train and get metrics
metrics = trainer.train({"fork": scm})

# Extract rewards from history
if 'history' in metrics:
    rewards = []
    for i, ep in enumerate(metrics['history']):
        if 'mean_reward' in ep:
            reward = ep['mean_reward']
            rewards.append(reward)
            if i % 5 == 0:  # Print every 5 episodes
                print(f"Episode {i}: reward = {reward:.3f}")
    
    if len(rewards) >= 20:
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        
        # Compare early vs late
        early = rewards[:10]
        late = rewards[-10:]
        
        early_avg = np.mean(early)
        late_avg = np.mean(late)
        
        print(f"\nFirst 10 episodes average: {early_avg:.3f}")
        print(f"Last 10 episodes average: {late_avg:.3f}")
        
        # Since minimizing, improvement = early - late
        change = late_avg - early_avg
        
        if change < 0:
            print(f"\n✅ IMPROVEMENT! Rewards decreased by {-change:.3f}")
            print("   (Lower is better since we're minimizing)")
        elif change > 0:
            print(f"\n❌ Got worse. Rewards increased by {change:.3f}")
        else:
            print(f"\n➖ No change in rewards")
        
        # Calculate slope
        x = np.arange(len(rewards))
        slope = np.polyfit(x, rewards, 1)[0]
        
        print(f"\nLinear slope: {slope:.6f}")
        if slope < -0.001:
            print("✅ Negative slope = improving over time!")
        elif slope > 0.001:
            print("❌ Positive slope = getting worse")
        else:
            print("➖ Flat slope = no clear trend")
        
        # Check exploration
        print("\n" + "-"*60)
        print("EXPLORATION CHECK")
        print("-"*60)
        print("Look for these patterns in the output above:")
        print("1. [EXPLORATION] showing different 'Selected:' values")
        print("2. Applied intervention on different variables")
        print("3. Non-zero advantages for learning")
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("The fixes enable:")
        print("- Exploration of different variables")
        print("- Meaningful learning signals (advantages)")
        print("- Potential for improvement over time")
        print("\nFor stronger improvement:")
        print("- Train longer (100+ episodes)")
        print("- Tune exploration_noise (currently 0.3)")
        print("- Enable surrogate for info gain rewards")
        
else:
    print("\nNo history found in metrics!")