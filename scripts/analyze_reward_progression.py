#!/usr/bin/env python3
"""
Analyze reward progression during GRPO training to see if the policy is learning.
"""

import sys
from pathlib import Path
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def extract_rewards_from_log(log_file: str = None, log_text: str = None):
    """Extract reward information from training logs."""
    if log_file:
        with open(log_file, 'r') as f:
            text = f.read()
    else:
        text = log_text
    
    # Extract episode rewards
    episode_pattern = r"Episode (\d+): mean_reward=([\d.]+)"
    episode_matches = re.findall(episode_pattern, text)
    
    episodes = []
    mean_rewards = []
    
    for episode, reward in episode_matches:
        episodes.append(int(episode))
        mean_rewards.append(float(reward))
    
    # Extract individual intervention rewards
    reward_pattern = r"Total reward: ([\d.]+)"
    all_rewards = [float(r) for r in re.findall(reward_pattern, text)]
    
    # Extract reward components
    target_pattern = r"Target reward: ([\d.]+)"
    info_gain_pattern = r"Info gain reward: ([\d.]+)"
    
    target_rewards = [float(r) for r in re.findall(target_pattern, text)]
    info_gain_rewards = [float(r) for r in re.findall(info_gain_pattern, text)]
    
    return {
        'episodes': episodes,
        'mean_rewards': mean_rewards,
        'all_rewards': all_rewards,
        'target_rewards': target_rewards,
        'info_gain_rewards': info_gain_rewards
    }


def run_training_with_more_episodes():
    """Run GRPO training with more episodes to see learning progression."""
    print("Running GRPO training with 50 episodes to analyze learning...")
    print("=" * 80)
    
    cmd = [
        sys.executable,
        "scripts/train_acbo_methods.py",
        "--method", "grpo_with_surrogate",
        "--episodes", "50",  # More episodes to see progression
        "--batch_size", "8",
        "--hidden_dim", "64",
        "--scm_type", "fork",
        "--min_vars", "3",
        "--max_vars", "3",
        "--checkpoint_dir", "checkpoints/test_reward_progression",
        "--seed", "456"
    ]
    
    # Run and capture output - also capture stderr where logs go
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Logs typically go to stderr, not stdout
    full_output = result.stdout + result.stderr
    
    # Extract rewards from output
    rewards_data = extract_rewards_from_log(log_text=full_output)
    
    return rewards_data, full_output


def plot_reward_progression(rewards_data):
    """Create plots showing reward progression."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mean reward per episode
    if rewards_data['episodes'] and rewards_data['mean_rewards']:
        ax = axes[0, 0]
        ax.plot(rewards_data['episodes'], rewards_data['mean_rewards'], 'b-o')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Mean Reward per Episode')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(rewards_data['episodes']) > 1:
            z = np.polyfit(rewards_data['episodes'], rewards_data['mean_rewards'], 1)
            p = np.poly1d(z)
            ax.plot(rewards_data['episodes'], p(rewards_data['episodes']), 
                   'r--', alpha=0.5, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax.legend()
    
    # 2. All intervention rewards over time
    if rewards_data['all_rewards']:
        ax = axes[0, 1]
        ax.plot(rewards_data['all_rewards'], alpha=0.5)
        ax.set_xlabel('Intervention Number')
        ax.set_ylabel('Total Reward')
        ax.set_title('All Intervention Rewards')
        ax.grid(True, alpha=0.3)
        
        # Add moving average
        window = min(20, len(rewards_data['all_rewards']) // 5)
        if window > 1:
            ma = np.convolve(rewards_data['all_rewards'], 
                            np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards_data['all_rewards'])), 
                   ma, 'r-', linewidth=2, label=f'{window}-intervention MA')
            ax.legend()
    
    # 3. Target rewards distribution
    if rewards_data['target_rewards']:
        ax = axes[1, 0]
        ax.hist(rewards_data['target_rewards'], bins=30, alpha=0.7, 
               color='green', edgecolor='black')
        ax.set_xlabel('Target Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Target Reward Distribution')
        ax.axvline(np.mean(rewards_data['target_rewards']), 
                  color='red', linestyle='--', 
                  label=f'Mean: {np.mean(rewards_data["target_rewards"]):.3f}')
        ax.legend()
    
    # 4. Info gain rewards over time
    if rewards_data['info_gain_rewards']:
        ax = axes[1, 1]
        ax.plot(rewards_data['info_gain_rewards'], alpha=0.5)
        ax.set_xlabel('Intervention Number')
        ax.set_ylabel('Info Gain Reward')
        ax.set_title('Information Gain Rewards')
        ax.grid(True, alpha=0.3)
        
        # Check if info gain is always the same (indicating no learning)
        unique_gains = len(set(rewards_data['info_gain_rewards']))
        ax.text(0.02, 0.98, f'Unique values: {unique_gains}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('checkpoints/test_reward_progression/reward_analysis.png', dpi=150)
    print("\nSaved plot to: checkpoints/test_reward_progression/reward_analysis.png")
    
    return fig


def analyze_learning_indicators(rewards_data):
    """Analyze indicators of whether the policy is learning."""
    print("\n" + "=" * 80)
    print("LEARNING ANALYSIS")
    print("=" * 80)
    
    if rewards_data['mean_rewards']:
        rewards = rewards_data['mean_rewards']
        
        # 1. Trend analysis
        if len(rewards) > 1:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            slope = z[0]
            print(f"\n1. Reward Trend:")
            print(f"   - Slope: {slope:.6f}")
            print(f"   - Interpretation: ", end="")
            if slope > 0.001:
                print("POSITIVE - Policy appears to be improving!")
            elif slope < -0.001:
                print("NEGATIVE - Policy is getting worse")
            else:
                print("FLAT - No clear improvement")
        
        # 2. Variance analysis
        print(f"\n2. Reward Stability:")
        print(f"   - Std deviation: {np.std(rewards):.4f}")
        print(f"   - Coefficient of variation: {np.std(rewards)/np.mean(rewards):.4f}")
        
        # 3. Compare early vs late performance
        if len(rewards) >= 10:
            early = np.mean(rewards[:5])
            late = np.mean(rewards[-5:])
            improvement = ((late - early) / early) * 100
            print(f"\n3. Early vs Late Performance:")
            print(f"   - First 5 episodes: {early:.4f}")
            print(f"   - Last 5 episodes: {late:.4f}")
            print(f"   - Improvement: {improvement:.1f}%")
    
    # 4. Info gain analysis
    if rewards_data['info_gain_rewards']:
        info_gains = rewards_data['info_gain_rewards']
        print(f"\n4. Information Gain Analysis:")
        print(f"   - Mean info gain: {np.mean(info_gains):.4f}")
        print(f"   - Unique values: {len(set(info_gains))}")
        print(f"   - Non-zero ratio: {sum(g > 0 for g in info_gains) / len(info_gains):.2f}")


if __name__ == "__main__":
    # Option 1: Analyze existing logs
    log_file = None  # Set to path if you have a log file
    
    if log_file and Path(log_file).exists():
        print(f"Analyzing existing log file: {log_file}")
        with open(log_file, 'r') as f:
            rewards_data = extract_rewards_from_log(log_text=f.read())
    else:
        # Option 2: Run new training
        rewards_data, full_log = run_training_with_more_episodes()
        
        # Save log for future analysis
        Path("checkpoints/test_reward_progression").mkdir(parents=True, exist_ok=True)
        with open("checkpoints/test_reward_progression/training.log", 'w') as f:
            f.write(full_log)
        print("\nSaved training log to: checkpoints/test_reward_progression/training.log")
    
    # Create visualizations
    if rewards_data['mean_rewards']:
        plot_reward_progression(rewards_data)
        analyze_learning_indicators(rewards_data)
    else:
        print("\nNo reward data found in logs!")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)