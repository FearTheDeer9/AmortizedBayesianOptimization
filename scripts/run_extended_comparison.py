#!/usr/bin/env python3
"""
Run extended architecture comparison with detailed monitoring.

This script runs the comparison and provides real-time feedback on:
- Embedding statistics (checking for uniform embeddings)
- Reward progression
- Policy behavior (which variables are being selected)
"""

import subprocess
import sys
import re
from datetime import datetime

def main():
    print("="*80)
    print("EXTENDED ARCHITECTURE COMPARISON - 200 EPISODES")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    print("\nMonitoring for:")
    print("- [EMBEDDING WARNING/STATS] - Embedding variance issues")
    print("- [LOGIT WARNING] - Policy discrimination issues")
    print("- [REWARD TREND] - Learning progress")
    print("- Episode progress and reward values")
    print("="*80)
    print()
    
    # Run the comparison script
    process = subprocess.Popen(
        [sys.executable, 'scripts/compare_policy_architectures.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Track key metrics
    embedding_warnings = []
    logit_warnings = []
    reward_trends = []
    episode_count = 0
    
    # Process output line by line
    for line in process.stdout:
        # Skip matplotlib debug output
        if 'findfont' in line or 'matplotlib' in line:
            continue
            
        # Track embedding warnings
        if '[EMBEDDING WARNING]' in line:
            embedding_warnings.append(line.strip())
            print(f"⚠️  {line.strip()}")
        elif '[EMBEDDING STATS]' in line:
            print(f"📊 {line.strip()}")
            
        # Track logit warnings
        elif '[LOGIT WARNING]' in line:
            if len(logit_warnings) < 5 or episode_count % 50 == 0:  # Limit spam
                print(f"⚠️  {line.strip()}")
            logit_warnings.append(line.strip())
            
        # Track reward trends
        elif '[REWARD TREND]' in line:
            reward_trends.append(line.strip())
            print(f"📈 {line.strip()}")
            
        # Track episode progress
        elif 'Episode' in line and 'mean_reward=' in line:
            match = re.search(r'Episode (\d+).*mean_reward=([\d.-]+)', line)
            if match:
                episode_count = int(match.group(1))
                reward = float(match.group(2))
                
                # Show progress every 10 episodes
                if episode_count % 10 == 0:
                    progress = episode_count / 200 * 100
                    print(f"Progress: {progress:5.1f}% | Episode {episode_count} | Reward: {reward:.4f}")
                    
        # Show architecture switches
        elif 'Training with' in line and 'architecture' in line:
            print(f"\n{'='*60}")
            print(line.strip())
            print('='*60)
            
        # Show final summary
        elif 'COMPARISON SUMMARY' in line:
            print(f"\n{line.strip()}")
        elif 'Winner:' in line or 'Average Improvement:' in line or 'Overhead:' in line:
            print(line.strip())
            
        # Show completion
        elif 'saved to' in line:
            print(f"✅ {line.strip()}")
    
    # Wait for completion
    process.wait()
    
    # Summary
    print("\n" + "="*80)
    print("MONITORING SUMMARY")
    print("="*80)
    
    print(f"\nEmbedding Warnings: {len(embedding_warnings)}")
    if embedding_warnings:
        print("First warning:", embedding_warnings[0])
        print("Last warning:", embedding_warnings[-1])
    
    print(f"\nLogit Warnings: {len(logit_warnings)}")
    
    print(f"\nReward Trends Logged: {len(reward_trends)}")
    if reward_trends:
        print("Last trend:", reward_trends[-1])
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()