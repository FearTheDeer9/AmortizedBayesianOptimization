"""Target-only reward configuration for sanity check."""

from .base_config import get_base_config, merge_configs


def get_config():
    """Get target-only configuration.
    
    This config uses ONLY target value improvement as the reward signal.
    Purpose: Verify if the policy can learn to optimize values without
    conflicting signals from info gain or direct parent rewards.
    """
    base = get_base_config()
    
    override = {
        # CRITICAL: Enable value loss computation
        'use_grpo_rewards': True,
        
        'policy_loss_weights': {
            'absolute_target': 1.0,  # Only target matters
            'information_gain': 0.0,  # No info gain reward
        },
        
        # Override GRPO rewards to focus on target
        'grpo_reward_weights': {
            'target_delta': 1.0,
            'info_gain': 0.0,
            'direct_parent': 0.0
        },
        
        # Slightly larger group for better value exploration
        'grpo_config': {
            'group_size': 6,  # More candidates for value diversity
            'entropy_coefficient': 0.15,  # Slightly more exploration
            'clip_ratio': 0.2
        },
        
        'checkpoint_dir': 'experiments/joint-grpo-target-training/results/target_only',
        'experiment_name': 'target_only_sanity_check'
    }
    
    return merge_configs(base, override)