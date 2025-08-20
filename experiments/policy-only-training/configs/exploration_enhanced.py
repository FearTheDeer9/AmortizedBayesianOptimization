"""Enhanced exploration configuration."""

from .base_config import get_base_config, merge_configs


def get_config():
    """Get exploration-enhanced configuration.
    
    Larger group size and diversity bonus for better value exploration.
    Purpose: Test if poor exploration is limiting learning.
    """
    base = get_base_config()
    
    override = {
        'policy_loss_weights': {
            'absolute_target': 0.6,
            'information_gain': 0.3,
            'value_diversity': 0.1,  # New: reward value diversity
        },
        
        'grpo_reward_weights': {
            'target_delta': 0.5,
            'info_gain': 0.2,
            'direct_parent': 0.2,
            'value_diversity': 0.1  # Bonus for exploring different values
        },
        
        'grpo_config': {
            'group_size': 8,  # Double the candidates
            'entropy_coefficient': 0.2,  # More exploration
            'clip_ratio': 0.2,
            'value_diversity_bonus': True,  # Enable diversity tracking
        },
        
        'checkpoint_dir': 'experiments/joint-grpo-target-training/results/exploration_enhanced',
        'experiment_name': 'exploration_enhanced_group8'
    }
    
    return merge_configs(base, override)