"""Balanced reward configuration (current production settings)."""

from .base_config import get_base_config, merge_configs


def get_config():
    """Get balanced configuration.
    
    Current production configuration for baseline comparison.
    Purpose: Baseline to compare against modified reward schemes.
    """
    base = get_base_config()
    
    override = {
        'policy_loss_weights': {
            'absolute_target': 0.7,  # Current production weight
            'information_gain': 0.3,
        },
        
        'grpo_reward_weights': {
            'target_delta': 0.5,
            'info_gain': 0.3,
            'direct_parent': 0.2
        },
        
        'checkpoint_dir': 'experiments/joint-grpo-target-training/results/balanced',
        'experiment_name': 'balanced_baseline'
    }
    
    return merge_configs(base, override)