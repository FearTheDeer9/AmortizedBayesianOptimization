"""Target-heavy reward configuration."""

from .base_config import get_base_config, merge_configs


def get_config():
    """Get target-heavy configuration.
    
    Strong focus on target optimization with minimal structure guidance.
    Purpose: Test if reducing (but not eliminating) conflicting signals helps.
    """
    base = get_base_config()
    
    override = {
        'policy_loss_weights': {
            'absolute_target': 0.8,  # Strong target focus
            'information_gain': 0.2,  # Light structure guidance
        },
        
        'grpo_reward_weights': {
            'target_delta': 0.8,
            'info_gain': 0.1,
            'direct_parent': 0.1
        },
        
        'checkpoint_dir': 'experiments/joint-grpo-target-training/results/target_heavy',
        'experiment_name': 'target_heavy_80_10_10'
    }
    
    return merge_configs(base, override)