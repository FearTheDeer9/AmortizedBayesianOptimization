#!/usr/bin/env python3
"""
Integration test for GRPO training with pre-trained surrogate.

This test:
1. Trains GRPO without surrogate (baseline)
2. Trains GRPO with pre-trained BC surrogate
3. Compares the rewards to verify information gain is being computed
"""

import logging
import subprocess
import json
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import reward logger to capture detailed rewards
from src.causal_bayes_opt.acquisition.clean_rewards import compute_clean_reward


class RewardLogger:
    """Capture reward computations for analysis."""
    def __init__(self):
        self.rewards = []
        self.original_fn = compute_clean_reward
        
    def __enter__(self):
        # Monkey patch the reward function
        def logged_compute_clean_reward(*args, **kwargs):
            result = self.original_fn(*args, **kwargs)
            self.rewards.append({
                'total': result['total'],
                'target': result['target'],
                'diversity': result['diversity'],
                'exploration': result['exploration'],
                'info_gain': result.get('info_gain', 0.0),
                'has_surrogate': kwargs.get('posterior_before') is not None
            })
            return result
        
        import src.causal_bayes_opt.acquisition.clean_rewards as rewards_module
        rewards_module.compute_clean_reward = logged_compute_clean_reward
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original function
        import src.causal_bayes_opt.acquisition.clean_rewards as rewards_module
        rewards_module.compute_clean_reward = self.original_fn


def run_training_test():
    """Run the integration test."""
    logger.info("="*60)
    logger.info("GRPO WITH PRE-TRAINED SURROGATE INTEGRATION TEST")
    logger.info("="*60)
    
    # Paths
    bc_surrogate_checkpoint = Path("checkpoints/validation/bc_surrogate_final")
    test_checkpoint_dir = Path("checkpoints/test_integration")
    test_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if BC surrogate exists
    if not bc_surrogate_checkpoint.exists():
        logger.error(f"BC surrogate checkpoint not found at {bc_surrogate_checkpoint}")
        logger.info("Please run scripts/train_focused.sh first to train the BC surrogate")
        return
    
    # Test parameters
    episodes = 30  # Quick test
    batch_size = 8
    
    # 1. Train GRPO without surrogate (baseline)
    logger.info("\n1. Training GRPO WITHOUT surrogate (baseline)...")
    cmd_no_surrogate = [
        "poetry", "run", "python", "scripts/train_acbo_methods.py",
        "--method", "grpo",
        "--episodes", str(episodes),
        "--batch_size", str(batch_size),
        "--learning_rate", "3e-4",
        "--scm_type", "mixed",
        "--checkpoint_dir", str(test_checkpoint_dir / "no_surrogate"),
        "--seed", "42"
    ]
    
    # Capture rewards for no-surrogate training
    logger.info(f"Running: {' '.join(cmd_no_surrogate)}")
    with RewardLogger() as reward_logger_no_surrogate:
        result = subprocess.run(cmd_no_surrogate, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
    
    rewards_no_surrogate = reward_logger_no_surrogate.rewards
    logger.info(f"✓ Captured {len(rewards_no_surrogate)} reward computations")
    
    # 2. Train GRPO with pre-trained surrogate
    logger.info("\n2. Training GRPO WITH pre-trained surrogate...")
    cmd_with_surrogate = [
        "poetry", "run", "python", "scripts/train_acbo_methods.py",
        "--method", "grpo",
        "--episodes", str(episodes),
        "--batch_size", str(batch_size),
        "--learning_rate", "3e-4",
        "--scm_type", "mixed",
        "--use_surrogate",
        "--surrogate_checkpoint", str(bc_surrogate_checkpoint),
        "--checkpoint_dir", str(test_checkpoint_dir / "with_surrogate"),
        "--seed", "43"
    ]
    
    # Capture rewards for with-surrogate training
    logger.info(f"Running: {' '.join(cmd_with_surrogate)}")
    with RewardLogger() as reward_logger_with_surrogate:
        result = subprocess.run(cmd_with_surrogate, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            return
    
    rewards_with_surrogate = reward_logger_with_surrogate.rewards
    logger.info(f"✓ Captured {len(rewards_with_surrogate)} reward computations")
    
    # 3. Analyze results
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS")
    logger.info("="*60)
    
    # Extract statistics
    def get_reward_stats(rewards_list):
        if not rewards_list:
            return {}
        
        total_rewards = [r['total'] for r in rewards_list]
        info_gains = [r['info_gain'] for r in rewards_list]
        
        return {
            'count': len(rewards_list),
            'mean_total': np.mean(total_rewards),
            'std_total': np.std(total_rewards),
            'mean_info_gain': np.mean(info_gains),
            'std_info_gain': np.std(info_gains),
            'max_info_gain': np.max(info_gains),
            'num_with_info_gain': sum(1 for ig in info_gains if ig > 0)
        }
    
    stats_no_surrogate = get_reward_stats(rewards_no_surrogate)
    stats_with_surrogate = get_reward_stats(rewards_with_surrogate)
    
    logger.info("\nWithout Surrogate:")
    logger.info(f"  Reward computations: {stats_no_surrogate.get('count', 0)}")
    logger.info(f"  Mean total reward: {stats_no_surrogate.get('mean_total', 0):.4f} ± {stats_no_surrogate.get('std_total', 0):.4f}")
    logger.info(f"  Mean info gain: {stats_no_surrogate.get('mean_info_gain', 0):.4f}")
    logger.info(f"  Rewards with info gain > 0: {stats_no_surrogate.get('num_with_info_gain', 0)}")
    
    logger.info("\nWith Pre-trained Surrogate:")
    logger.info(f"  Reward computations: {stats_with_surrogate.get('count', 0)}")
    logger.info(f"  Mean total reward: {stats_with_surrogate.get('mean_total', 0):.4f} ± {stats_with_surrogate.get('std_total', 0):.4f}")
    logger.info(f"  Mean info gain: {stats_with_surrogate.get('mean_info_gain', 0):.4f}")
    logger.info(f"  Max info gain: {stats_with_surrogate.get('max_info_gain', 0):.4f}")
    logger.info(f"  Rewards with info gain > 0: {stats_with_surrogate.get('num_with_info_gain', 0)}")
    
    # Check if info gain is being computed
    if stats_with_surrogate.get('mean_info_gain', 0) > stats_no_surrogate.get('mean_info_gain', 0):
        logger.info("\n✅ SUCCESS: Information gain rewards are being computed with surrogate!")
        logger.info(f"   Info gain increase: {stats_with_surrogate['mean_info_gain'] - stats_no_surrogate['mean_info_gain']:.4f}")
    else:
        logger.warning("\n⚠️  WARNING: No significant information gain detected")
    
    # 4. Create visualization
    if rewards_no_surrogate and rewards_with_surrogate:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Total rewards comparison
        ax = axes[0, 0]
        x = ['No Surrogate', 'With Surrogate']
        means = [stats_no_surrogate['mean_total'], stats_with_surrogate['mean_total']]
        stds = [stats_no_surrogate['std_total'], stats_with_surrogate['std_total']]
        ax.bar(x, means, yerr=stds, capsize=10)
        ax.set_ylabel('Mean Total Reward')
        ax.set_title('Total Reward Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Info gain comparison
        ax = axes[0, 1]
        info_means = [stats_no_surrogate['mean_info_gain'], stats_with_surrogate['mean_info_gain']]
        ax.bar(x, info_means)
        ax.set_ylabel('Mean Information Gain')
        ax.set_title('Information Gain Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Reward components breakdown (with surrogate)
        ax = axes[1, 0]
        if rewards_with_surrogate:
            components = ['target', 'diversity', 'exploration', 'info_gain']
            component_means = [
                np.mean([r[comp] for r in rewards_with_surrogate])
                for comp in components
            ]
            ax.bar(components, component_means)
            ax.set_ylabel('Mean Reward')
            ax.set_title('Reward Components (With Surrogate)')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Info gain distribution
        ax = axes[1, 1]
        info_gains_with = [r['info_gain'] for r in rewards_with_surrogate]
        info_gains_without = [r['info_gain'] for r in rewards_no_surrogate]
        ax.hist(info_gains_without, bins=20, alpha=0.5, label='No Surrogate')
        ax.hist(info_gains_with, bins=20, alpha=0.5, label='With Surrogate')
        ax.set_xlabel('Information Gain')
        ax.set_ylabel('Count')
        ax.set_title('Information Gain Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grpo_pretrained_surrogate_test.png', dpi=150)
        logger.info(f"\n✓ Saved visualization to grpo_pretrained_surrogate_test.png")
        plt.close()
    
    # 5. Save detailed results
    results = {
        'test_config': {
            'episodes': episodes,
            'batch_size': batch_size,
            'bc_surrogate_checkpoint': str(bc_surrogate_checkpoint)
        },
        'stats_no_surrogate': stats_no_surrogate,
        'stats_with_surrogate': stats_with_surrogate,
        'success': stats_with_surrogate.get('mean_info_gain', 0) > stats_no_surrogate.get('mean_info_gain', 0)
    }
    
    with open('grpo_pretrained_surrogate_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Saved detailed results to grpo_pretrained_surrogate_test_results.json")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    if results['success']:
        logger.info("✅ Integration test PASSED!")
        logger.info("   - Pre-trained surrogate successfully loaded")
        logger.info("   - Information gain rewards are being computed")
        logger.info("   - Total rewards include information gain component")
    else:
        logger.info("❌ Integration test FAILED")
        logger.info("   - Check if surrogate is being loaded correctly")
        logger.info("   - Verify reward weight configuration")


if __name__ == "__main__":
    run_training_test()