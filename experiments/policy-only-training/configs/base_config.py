"""Base configuration shared across all experiments."""

def get_base_config():
    """Get base configuration for GRPO experiments."""
    return {
        # Episode settings
        'max_episodes': 20,
        'obs_per_episode': 20,
        'max_interventions': 20,  # Longer episodes for within-episode analysis
        
        # Phase control (policy-only training)
        'episodes_per_phase': 1000,  # High to prevent phase switching
        'use_surrogate': True,  # Use for guidance but don't train
        'use_replay_buffer': False,
        
        # CRITICAL: Enable value loss computation
        'use_grpo_rewards': True,
        
        # Model architecture
        'hidden_dim': 64,
        'learning_rate': 1e-4,  # Reduced for stability
        'batch_size': 4,
        'architecture': {
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'key_size': 16,
            'dropout': 0.0
        },
        
        # GRPO settings (can be overridden)
        'grpo_config': {
            'group_size': 4,  # Number of candidates per intervention
            'entropy_coefficient': 0.1,
            'clip_ratio': 0.2
        },
        
        # Joint training settings
        'joint_training': {
            'initial_phase': 'policy',  # Start with policy
            'adaptive': {
                'use_performance_rotation': False  # No adaptive switching
            }
        },
        
        # Logging and checkpointing
        'verbose': True,
        'checkpoint_freq': 5,
        'enable_within_episode_analysis': True,
    }


def merge_configs(base, override):
    """Recursively merge override config into base config."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result