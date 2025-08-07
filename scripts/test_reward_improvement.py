#!/usr/bin/env python3
"""
Test if GRPO rewards improve over episodes (positive slope).
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm, create_chain_scm


def test_reward_improvement(scm_name="fork", n_episodes=100):
    """Test if rewards improve over training episodes."""
    
    print(f"\n{'='*80}")
    print(f"Testing Reward Improvement on {scm_name.upper()} SCM")
    print(f"{'='*80}")
    
    # Create SCM
    if scm_name == "fork":
        scm = create_fork_scm()  # X -> Y <- Z
        print("Structure: X -> Y <- Z")
        print("Target: Y (trying to minimize)")
        print("Optimal strategy: Intervene on X or Z (causal parents)")
    else:
        scm = create_chain_scm()  # X -> Y -> Z
        print("Structure: X -> Y -> Z")
        print("Target: Z (trying to minimize)")
        print("Optimal strategy: Intervene on Y (direct parent)")
    
    # Train with longer episodes for stable learning
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=n_episodes,
        episode_length=20,  # Longer episodes for better estimates
        batch_size=32,      # Larger batch for stable gradients
        architecture_level="baseline",
        optimization_direction="MINIMIZE",
        seed=42,
        use_surrogate=False,
        checkpoint_dir=f"checkpoints/reward_test_{scm_name}",
        reward_weights={
            'optimization': 1.0,  # Focus purely on target optimization
            'discovery': 0.0,
            'efficiency': 0.0,
            'info_gain': 0.0
        }
    )
    
    print(f"\nTraining for {n_episodes} episodes...")
    metrics = trainer.train({scm_name: scm})
    
    # Extract episode rewards
    if 'history' in metrics:
        episode_rewards = []
        for ep in metrics['history']:
            if 'mean_reward' in ep:
                # Since we're minimizing, negate for intuitive "improvement"
                episode_rewards.append(-ep['mean_reward'])
        
        if len(episode_rewards) > 10:
            # Calculate trend
            episodes = np.arange(len(episode_rewards))
            
            # Fit linear regression
            A = np.vstack([episodes, np.ones(len(episodes))]).T
            slope, intercept = np.linalg.lstsq(A, episode_rewards, rcond=None)[0]
            
            # Calculate moving average
            window = min(10, len(episode_rewards) // 5)
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            
            # Early vs late comparison
            early_avg = np.mean(episode_rewards[:20])
            late_avg = np.mean(episode_rewards[-20:])
            improvement = late_avg - early_avg
            improvement_pct = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
            
            print(f"\n📊 RESULTS:")
            print(f"Episodes trained: {len(episode_rewards)}")
            print(f"Linear regression slope: {slope:.6f}")
            print(f"Early episodes (0-19) avg: {early_avg:.3f}")
            print(f"Late episodes (last 20) avg: {late_avg:.3f}")
            print(f"Absolute improvement: {improvement:.3f}")
            print(f"Percentage improvement: {improvement_pct:.1f}%")
            
            if slope > 0:
                print(f"\n✅ POSITIVE SLOPE! Rewards are improving over time.")
            else:
                print(f"\n❌ Negative/flat slope. Rewards not improving.")
            
            # Plot results
            plt.figure(figsize=(12, 6))
            
            # Plot 1: Raw rewards with trend
            plt.subplot(1, 2, 1)
            plt.scatter(episodes, episode_rewards, alpha=0.5, s=10, label='Episode rewards')
            plt.plot(episodes, slope * episodes + intercept, 'r-', linewidth=2, label=f'Trend (slope={slope:.4f})')
            if len(moving_avg) > 0:
                plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'g-', linewidth=2, label='Moving avg')
            plt.xlabel('Episode')
            plt.ylabel('Reward (negated for minimization)')
            plt.title(f'Reward Progression - {scm_name.upper()} SCM')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Binned averages
            plt.subplot(1, 2, 2)
            n_bins = min(10, len(episode_rewards) // 10)
            if n_bins > 2:
                bin_size = len(episode_rewards) // n_bins
                bin_avgs = []
                bin_centers = []
                for i in range(n_bins):
                    start = i * bin_size
                    end = (i + 1) * bin_size if i < n_bins - 1 else len(episode_rewards)
                    bin_avgs.append(np.mean(episode_rewards[start:end]))
                    bin_centers.append((start + end) / 2)
                
                plt.bar(range(n_bins), bin_avgs, width=0.8)
                plt.xlabel('Training Phase (bins)')
                plt.ylabel('Average Reward')
                plt.title('Average Reward by Training Phase')
                plt.xticks(range(n_bins), [f'{int(c)}' for c in bin_centers])
                
                # Add trend line
                if len(bin_avgs) > 2:
                    z = np.polyfit(range(n_bins), bin_avgs, 1)
                    p = np.poly1d(z)
                    plt.plot(range(n_bins), p(range(n_bins)), "r--", linewidth=2, label=f'Trend')
                    plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'reward_improvement_{scm_name}.png', dpi=150)
            print(f"\nPlots saved to: reward_improvement_{scm_name}.png")
            
            return slope, improvement_pct
    
    return None, None


def main():
    """Test reward improvement on different SCMs."""
    
    print("="*80)
    print("TESTING GRPO REWARD IMPROVEMENT")
    print("="*80)
    print("\nThis test checks if the GRPO fixes lead to improving rewards over time.")
    print("We expect to see:")
    print("1. Positive slope in reward curve")
    print("2. Higher rewards in later episodes")
    print("3. Policy learning to focus on causal parents")
    
    # Test on different SCMs
    results = {}
    
    for scm_name in ["fork", "chain"]:
        slope, improvement = test_reward_improvement(scm_name, n_episodes=100)
        if slope is not None:
            results[scm_name] = {
                'slope': slope,
                'improvement': improvement
            }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Reward Improvement Results")
    print("="*80)
    
    for scm_name, res in results.items():
        print(f"\n{scm_name.upper()} SCM:")
        print(f"  Slope: {res['slope']:.6f} {'✅' if res['slope'] > 0 else '❌'}")
        print(f"  Improvement: {res['improvement']:.1f}% {'✅' if res['improvement'] > 5 else '❌'}")
    
    # Overall assessment
    all_positive = all(res['slope'] > 0 for res in results.values())
    all_improving = all(res['improvement'] > 5 for res in results.values())
    
    print("\n" + "="*80)
    if all_positive and all_improving:
        print("🎉 SUCCESS! All SCMs show positive reward slopes and improvement!")
        print("The GRPO fixes are working - the policy is learning!")
    elif all_positive:
        print("✅ Good! All SCMs show positive slopes (learning happening)")
        print("Some improvement percentages may be small due to noise.")
    else:
        print("⚠️  Mixed results. Some SCMs may need more episodes or tuning.")
        print("Check the plots for detailed analysis.")
    
    print("\nNext steps:")
    print("1. Try longer training (n_episodes=200) for clearer trends")
    print("2. Test with use_surrogate=True for better discovery rewards")
    print("3. Tune exploration_noise (currently 0.3) for optimal exploration/exploitation")


if __name__ == "__main__":
    main()