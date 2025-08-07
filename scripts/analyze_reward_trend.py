#!/usr/bin/env python3
"""
Analyze reward trends from GRPO training logs.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_log_file(log_path):
    """Extract and analyze rewards from a GRPO training log."""
    
    print(f"Analyzing log file: {log_path}")
    print("="*60)
    
    # Read log file
    try:
        with open(log_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Log file not found: {log_path}")
        return
    
    # Extract episode rewards
    pattern = r"Episode (\d+): mean_reward=([\d.-]+)"
    matches = re.findall(pattern, content)
    
    if not matches:
        print("No episode rewards found in log.")
        return
    
    episodes = []
    rewards = []
    for ep, reward in matches:
        episodes.append(int(ep))
        rewards.append(float(reward))
    
    print(f"\nFound {len(rewards)} episodes")
    
    if len(rewards) < 10:
        print("Not enough episodes for trend analysis")
        return
    
    # Analysis
    rewards = np.array(rewards)
    
    # Early vs late
    n_early = min(10, len(rewards) // 4)
    n_late = min(10, len(rewards) // 4)
    
    early_rewards = rewards[:n_early]
    late_rewards = rewards[-n_late:]
    
    early_avg = np.mean(early_rewards)
    late_avg = np.mean(late_rewards)
    
    print(f"\nFirst {n_early} episodes average: {early_avg:.4f}")
    print(f"Last {n_late} episodes average: {late_avg:.4f}")
    
    # For minimization, improvement = early - late
    improvement = early_avg - late_avg
    improvement_pct = (improvement / abs(early_avg)) * 100 if early_avg != 0 else 0
    
    print(f"\nChange: {late_avg - early_avg:+.4f}")
    if improvement > 0:
        print(f"✅ IMPROVEMENT: Rewards decreased by {improvement:.4f} ({improvement_pct:.1f}%)")
    else:
        print(f"❌ No improvement or got worse")
    
    # Calculate trend
    x = np.arange(len(rewards))
    coeffs = np.polyfit(x, rewards, 1)
    slope = coeffs[0]
    
    print(f"\nLinear regression slope: {slope:.6f}")
    if slope < -0.0001:
        print("✅ Negative slope - rewards improving over time!")
    elif slope > 0.0001:
        print("❌ Positive slope - rewards getting worse")
    else:
        print("➖ Essentially flat - no clear trend")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Raw rewards with trend
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, 'b-', alpha=0.5, marker='o', markersize=3, label='Episode rewards')
    
    # Add trend line
    trend_line = np.poly1d(coeffs)
    plt.plot(episodes, trend_line(x), 'r--', linewidth=2, label=f'Trend (slope={slope:.4f})')
    
    # Add moving average
    if len(rewards) > 10:
        window = min(10, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(episodes[window-1:], moving_avg, 'g-', linewidth=2, label='Moving avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward (lower is better)')
    plt.title('GRPO Learning Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Binned averages
    plt.subplot(1, 2, 2)
    n_bins = min(10, max(2, len(rewards) // 5))
    bin_size = len(rewards) // n_bins
    
    bin_means = []
    bin_stds = []
    bin_centers = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(rewards)
        bin_rewards = rewards[start:end]
        bin_means.append(np.mean(bin_rewards))
        bin_stds.append(np.std(bin_rewards))
        bin_centers.append((start + end) / 2)
    
    plt.bar(range(n_bins), bin_means, yerr=bin_stds, capsize=5, alpha=0.7)
    plt.xlabel('Training Phase')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Training Phase')
    plt.xticks(range(n_bins), [f'Ep {int(c)}' for c in bin_centers], rotation=45)
    
    # Add trend line on bins
    if n_bins > 2:
        bin_trend = np.polyfit(range(n_bins), bin_means, 1)
        bin_line = np.poly1d(bin_trend)
        plt.plot(range(n_bins), bin_line(range(n_bins)), 'r--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('reward_trend_analysis.png', dpi=150)
    print(f"\nPlot saved to: reward_trend_analysis.png")
    
    return slope, improvement_pct


def main():
    """Analyze reward trends from training logs."""
    
    print("REWARD TREND ANALYSIS")
    print("="*80)
    print("\nThis script analyzes GRPO training logs to check for improvement.")
    print("Looking for:")
    print("- Negative slope (rewards decreasing over time)")
    print("- Lower average rewards in later episodes")
    print("="*80)
    
    # Try to find recent log files
    import glob
    log_files = glob.glob('grpo_improvement_test.log') + \
                glob.glob('checkpoints/*/training.log') + \
                glob.glob('*.log')
    
    if log_files:
        print(f"\nFound {len(log_files)} log file(s)")
        for log_file in log_files[:3]:  # Analyze up to 3 most recent
            print("\n" + "-"*60)
            analyze_log_file(log_file)
    else:
        print("\nNo log files found.")
        print("\nTo generate logs, run:")
        print("  python scripts/verify_grpo_improvements.py")
        print("  or")
        print("  python scripts/test_grpo_learning.py")
        
        # Create example data to show what we expect
        print("\n" + "="*60)
        print("EXPECTED BEHAVIOR WITH FIXES")
        print("="*60)
        
        # Simulate improving rewards
        np.random.seed(42)
        episodes = np.arange(50)
        # Start high, decrease with noise
        rewards = 0.8 - 0.01 * episodes + 0.1 * np.random.randn(50)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, rewards, 'b-', alpha=0.5, marker='o', markersize=3)
        
        # Add trend
        trend = np.polyfit(episodes, rewards, 1)
        plt.plot(episodes, np.poly1d(trend)(episodes), 'r--', linewidth=2, 
                label=f'Expected trend (slope≈-0.01)')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward (lower is better)')
        plt.title('Expected GRPO Behavior with Fixes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('expected_reward_trend.png', dpi=150)
        
        print("\nWith exploration and proper baselines, we expect:")
        print("- Negative slope (around -0.001 to -0.01)")
        print("- Decreasing rewards over episodes")
        print("- Some noise due to exploration")
        print("\nExpected trend plot saved to: expected_reward_trend.png")


if __name__ == "__main__":
    main()