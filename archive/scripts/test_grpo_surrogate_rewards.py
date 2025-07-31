#!/usr/bin/env python3
"""
Test script to compare GRPO training with and without a pre-trained surrogate.
This will show evidence of different reward computations when information gain is included.
"""

import logging
import sys
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.test_scms import create_fork_test_scm, create_collider_test_scm
from src.causal_bayes_opt.training.continuous_surrogate_integration import create_surrogate_fn_wrapper

# Set up logging to see reward details
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pretrained_surrogate(checkpoint_path):
    """Load a pre-trained surrogate model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    # Try to find the checkpoint file
    if checkpoint_path.is_dir():
        checkpoint_file = checkpoint_path / "checkpoint.pkl"
    else:
        checkpoint_file = checkpoint_path
    
    if not checkpoint_file.exists():
        logger.warning(f"Surrogate checkpoint not found at {checkpoint_file}")
        return None, None, None
    
    logger.info(f"Loading surrogate from {checkpoint_file}")
    
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Extract surrogate components if available
    if 'surrogate_net' in checkpoint and 'surrogate_params' in checkpoint:
        return (
            checkpoint['surrogate_net'],
            checkpoint['surrogate_params'],
            lambda params, opt_state, buffer, target: (params, opt_state, {'loss': 0.0})  # Dummy update
        )
    else:
        logger.warning("No surrogate found in checkpoint")
        return None, None, None


def train_and_compare(episodes=20, batch_size=8):
    """Train GRPO with and without surrogate and compare rewards."""
    
    # Create test SCMs
    scms = [create_fork_test_scm(), create_collider_test_scm()]
    
    # Store results
    results = {
        'no_surrogate': {'rewards': [], 'info_gains': []},
        'with_surrogate': {'rewards': [], 'info_gains': []}
    }
    
    # 1. Train without surrogate
    logger.info("\n" + "="*60)
    logger.info("TRAINING WITHOUT SURROGATE")
    logger.info("="*60)
    
    trainer_no_surrogate = create_unified_grpo_trainer(
        learning_rate=3e-4,
        n_episodes=episodes,
        episode_length=10,
        batch_size=batch_size,
        use_surrogate=False,
        seed=42
    )
    
    # Capture reward logs
    original_compute_clean_reward = __import__(
        'src.causal_bayes_opt.acquisition.clean_rewards',
        fromlist=['compute_clean_reward']
    ).compute_clean_reward
    
    def logged_compute_clean_reward(*args, **kwargs):
        result = original_compute_clean_reward(*args, **kwargs)
        results['no_surrogate']['rewards'].append(result)
        return result
    
    # Monkey patch for logging
    import src.causal_bayes_opt.acquisition.clean_rewards as rewards_module
    rewards_module.compute_clean_reward = logged_compute_clean_reward
    
    training_results_no_surrogate = trainer_no_surrogate.train(scms)
    
    # 2. Train with surrogate
    logger.info("\n" + "="*60)
    logger.info("TRAINING WITH PRE-TRAINED SURROGATE")
    logger.info("="*60)
    
    # Reset results capture
    results['with_surrogate'] = {'rewards': [], 'info_gains': []}
    
    def logged_compute_clean_reward_with_surrogate(*args, **kwargs):
        result = original_compute_clean_reward(*args, **kwargs)
        results['with_surrogate']['rewards'].append(result)
        return result
    
    rewards_module.compute_clean_reward = logged_compute_clean_reward_with_surrogate
    
    # Try to load pre-trained surrogate
    surrogate_checkpoint_path = "checkpoints/validation/bc_surrogate_final"
    net, params, _ = load_pretrained_surrogate(surrogate_checkpoint_path)
    
    if net is not None and params is not None:
        logger.info("Successfully loaded pre-trained surrogate")
        
        # Create trainer with surrogate
        trainer_with_surrogate = create_unified_grpo_trainer(
            learning_rate=3e-4,
            n_episodes=episodes,
            episode_length=10,
            batch_size=batch_size,
            use_surrogate=True,
            reward_weights={
                'optimization': 0.5,  # Reduced to make room for info gain
                'discovery': 0.1,
                'efficiency': 0.1,
                'info_gain': 0.3      # Significant weight on information gain
            },
            seed=43
        )
        
        # Override with pre-trained surrogate
        trainer_with_surrogate.surrogate_net = net
        trainer_with_surrogate.surrogate_params = params
        
        # Create prediction function wrapper
        trainer_with_surrogate.surrogate_predict_fn = create_surrogate_fn_wrapper(net, params)
        
        training_results_with_surrogate = trainer_with_surrogate.train(scms)
    else:
        logger.warning("Could not load surrogate, training with fresh surrogate")
        trainer_with_surrogate = create_unified_grpo_trainer(
            learning_rate=3e-4,
            n_episodes=episodes,
            episode_length=10,
            batch_size=batch_size,
            use_surrogate=True,
            reward_weights={
                'optimization': 0.5,
                'discovery': 0.1,
                'efficiency': 0.1,
                'info_gain': 0.3
            },
            seed=43
        )
        training_results_with_surrogate = trainer_with_surrogate.train(scms)
    
    # Restore original function
    rewards_module.compute_clean_reward = original_compute_clean_reward
    
    # 3. Analyze and visualize results
    logger.info("\n" + "="*60)
    logger.info("ANALYZING RESULTS")
    logger.info("="*60)
    
    # Extract reward components
    def extract_reward_stats(rewards_list):
        if not rewards_list:
            return {}
        
        stats = {
            'total': [r['total'] for r in rewards_list],
            'target': [r['target'] for r in rewards_list],
            'diversity': [r['diversity'] for r in rewards_list],
            'exploration': [r['exploration'] for r in rewards_list],
            'info_gain': [r.get('info_gain', 0.0) for r in rewards_list]
        }
        
        return {
            'mean_total': np.mean(stats['total']),
            'mean_target': np.mean(stats['target']),
            'mean_diversity': np.mean(stats['diversity']),
            'mean_exploration': np.mean(stats['exploration']),
            'mean_info_gain': np.mean(stats['info_gain']),
            'num_rewards': len(rewards_list)
        }
    
    no_surrogate_stats = extract_reward_stats(results['no_surrogate']['rewards'])
    with_surrogate_stats = extract_reward_stats(results['with_surrogate']['rewards'])
    
    logger.info("\nWithout Surrogate:")
    logger.info(f"  Mean total reward: {no_surrogate_stats.get('mean_total', 0):.4f}")
    logger.info(f"  Mean target reward: {no_surrogate_stats.get('mean_target', 0):.4f}")
    logger.info(f"  Mean info gain reward: {no_surrogate_stats.get('mean_info_gain', 0):.4f}")
    logger.info(f"  Number of rewards computed: {no_surrogate_stats.get('num_rewards', 0)}")
    
    logger.info("\nWith Surrogate:")
    logger.info(f"  Mean total reward: {with_surrogate_stats.get('mean_total', 0):.4f}")
    logger.info(f"  Mean target reward: {with_surrogate_stats.get('mean_target', 0):.4f}")
    logger.info(f"  Mean info gain reward: {with_surrogate_stats.get('mean_info_gain', 0):.4f}")
    logger.info(f"  Number of rewards computed: {with_surrogate_stats.get('num_rewards', 0)}")
    
    # Plot comparison
    if results['no_surrogate']['rewards'] and results['with_surrogate']['rewards']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Total rewards over time
        ax = axes[0, 0]
        no_surrogate_totals = [r['total'] for r in results['no_surrogate']['rewards']]
        with_surrogate_totals = [r['total'] for r in results['with_surrogate']['rewards']]
        
        ax.plot(no_surrogate_totals, label='No Surrogate', alpha=0.7)
        ax.plot(with_surrogate_totals, label='With Surrogate', alpha=0.7)
        ax.set_title('Total Rewards Over Time')
        ax.set_xlabel('Reward Computation')
        ax.set_ylabel('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Info gain component
        ax = axes[0, 1]
        no_surrogate_info = [r.get('info_gain', 0) for r in results['no_surrogate']['rewards']]
        with_surrogate_info = [r.get('info_gain', 0) for r in results['with_surrogate']['rewards']]
        
        ax.plot(no_surrogate_info, label='No Surrogate', alpha=0.7)
        ax.plot(with_surrogate_info, label='With Surrogate', alpha=0.7)
        ax.set_title('Information Gain Rewards')
        ax.set_xlabel('Reward Computation')
        ax.set_ylabel('Info Gain Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Reward components breakdown
        ax = axes[1, 0]
        components = ['target', 'diversity', 'exploration', 'info_gain']
        no_surrogate_means = [
            np.mean([r.get(comp, 0) for r in results['no_surrogate']['rewards']])
            for comp in components
        ]
        with_surrogate_means = [
            np.mean([r.get(comp, 0) for r in results['with_surrogate']['rewards']])
            for comp in components
        ]
        
        x = np.arange(len(components))
        width = 0.35
        
        ax.bar(x - width/2, no_surrogate_means, width, label='No Surrogate')
        ax.bar(x + width/2, with_surrogate_means, width, label='With Surrogate')
        ax.set_xticks(x)
        ax.set_xticklabels(components)
        ax.set_title('Mean Reward Components')
        ax.set_ylabel('Mean Reward')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Episode rewards from training
        ax = axes[1, 1]
        if 'all_metrics' in training_results_no_surrogate:
            no_surrogate_episode_rewards = [m['mean_reward'] for m in training_results_no_surrogate['all_metrics']]
            ax.plot(no_surrogate_episode_rewards, label='No Surrogate', marker='o', markersize=4)
        
        if 'all_metrics' in training_results_with_surrogate:
            with_surrogate_episode_rewards = [m['mean_reward'] for m in training_results_with_surrogate['all_metrics']]
            ax.plot(with_surrogate_episode_rewards, label='With Surrogate', marker='s', markersize=4)
        
        ax.set_title('Training Episode Mean Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Mean Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grpo_surrogate_reward_comparison.png', dpi=150)
        logger.info(f"\nSaved comparison plot to grpo_surrogate_reward_comparison.png")
        plt.close()
    
    # Save detailed results
    detailed_results = {
        'no_surrogate': {
            'stats': no_surrogate_stats,
            'num_episodes': len(training_results_no_surrogate.get('all_metrics', [])),
            'final_mean_reward': training_results_no_surrogate.get('final_metrics', {}).get('mean_reward', 0)
        },
        'with_surrogate': {
            'stats': with_surrogate_stats,
            'num_episodes': len(training_results_with_surrogate.get('all_metrics', [])),
            'final_mean_reward': training_results_with_surrogate.get('final_metrics', {}).get('mean_reward', 0)
        }
    }
    
    with open('grpo_surrogate_comparison_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Saved detailed results to grpo_surrogate_comparison_results.json")
    
    return results


if __name__ == "__main__":
    logger.info("Starting GRPO surrogate reward comparison test...")
    train_and_compare(episodes=20, batch_size=8)