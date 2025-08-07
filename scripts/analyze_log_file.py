#!/usr/bin/env python3
"""
Analyze reward progression from a saved log file.
Usage: python analyze_log_file.py <log_file_path>
"""

import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def extract_rewards_from_log(log_file):
    """Extract reward information from training logs."""
    with open(log_file, 'r') as f:
        text = f.read()
    
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
    info_gain_pattern = r"info_gain_reward=([\d.]+)"
    
    target_rewards = [float(r) for r in re.findall(target_pattern, text)]
    info_gain_rewards = [float(r) for r in re.findall(info_gain_pattern, text)]
    
    return {
        'episodes': episodes,
        'mean_rewards': mean_rewards,
        'all_rewards': all_rewards,
        'target_rewards': target_rewards,
        'info_gain_rewards': info_gain_rewards
    }


def analyze_and_plot(log_file):
    """Analyze log file and create visualizations."""
    print(f"Analyzing log file: {log_file}")
    print("=" * 80)
    
    rewards_data = extract_rewards_from_log(log_file)
    
    # Print summary statistics
    print(f"\nFound data for {len(rewards_data['episodes'])} episodes")
    print(f"Total interventions: {len(rewards_data['all_rewards'])}")
    
    if rewards_data['mean_rewards']:
        print(f"\nMean rewards per episode:")
        for ep, reward in zip(rewards_data['episodes'], rewards_data['mean_rewards']):
            print(f"  Episode {ep}: {reward:.4f}")
        
        # Analyze trend
        if len(rewards_data['mean_rewards']) > 1:
            z = np.polyfit(range(len(rewards_data['mean_rewards'])), 
                          rewards_data['mean_rewards'], 1)
            slope = z[0]
            
            print(f"\nLearning trend:")
            print(f"  Slope: {slope:.6f}")
            print(f"  Interpretation: ", end="")
            if slope > 0.001:
                print("POSITIVE - Policy is improving! ✓")
            elif slope < -0.001:
                print("NEGATIVE - Policy is degrading ✗")
            else:
                print("FLAT - No clear trend")
            
            # Compare early vs late
            if len(rewards_data['mean_rewards']) >= 10:
                early = np.mean(rewards_data['mean_rewards'][:5])
                late = np.mean(rewards_data['mean_rewards'][-5:])
                improvement = ((late - early) / early) * 100 if early > 0 else 0
                
                print(f"\nEarly vs Late performance:")
                print(f"  First 5 episodes: {early:.4f}")
                print(f"  Last 5 episodes: {late:.4f}")
                print(f"  Change: {improvement:+.1f}%")
    
    if rewards_data['info_gain_rewards']:
        print(f"\nInfo gain analysis:")
        print(f"  Total info gain calculations: {len(rewards_data['info_gain_rewards'])}")
        print(f"  Mean info gain: {np.mean(rewards_data['info_gain_rewards']):.4f}")
        print(f"  Non-zero ratio: {sum(g > 0 for g in rewards_data['info_gain_rewards']) / len(rewards_data['info_gain_rewards']):.2f}")
    
    # Create plots if we have data
    if rewards_data['mean_rewards']:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(rewards_data['episodes'], rewards_data['mean_rewards'], 'b-o')
        plt.xlabel('Episode')
        plt.ylabel('Mean Reward')
        plt.title('Mean Reward per Episode')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(rewards_data['episodes']) > 1:
            z = np.polyfit(rewards_data['episodes'], rewards_data['mean_rewards'], 1)
            p = np.poly1d(z)
            plt.plot(rewards_data['episodes'], p(rewards_data['episodes']), 
                    'r--', alpha=0.5, label=f'Trend: slope={z[0]:.4f}')
            plt.legend()
        
        plt.subplot(2, 1, 2)
        if rewards_data['all_rewards']:
            plt.plot(rewards_data['all_rewards'], alpha=0.3, label='Individual rewards')
            
            # Add moving average
            window = min(20, len(rewards_data['all_rewards']) // 5)
            if window > 1:
                ma = np.convolve(rewards_data['all_rewards'], 
                                np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(rewards_data['all_rewards'])), 
                        ma, 'r-', linewidth=2, label=f'{window}-intervention MA')
            plt.xlabel('Intervention Number')
            plt.ylabel('Reward')
            plt.title('All Intervention Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(log_file).parent / 'reward_analysis.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to: {output_path}")
        plt.close()
    else:
        print("\nNo episode data found for plotting!")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        # Try default locations
        default_paths = [
            "checkpoints/test_reward_progression/training.log",
            "checkpoints/test_timeout/training.log",
            "training.log"
        ]
        
        for path in default_paths:
            if Path(path).exists():
                print(f"Using default log file: {path}")
                analyze_and_plot(path)
                break
        else:
            print("Usage: python analyze_log_file.py <log_file_path>")
            print("No log file provided and no default files found.")
            sys.exit(1)
    else:
        log_file = sys.argv[1]
        if not Path(log_file).exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)
        
        analyze_and_plot(log_file)