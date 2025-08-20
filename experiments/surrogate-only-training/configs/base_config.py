"""Base configuration for surrogate-only experiments."""

def get_base_config():
    """Get base configuration for surrogate training experiments."""
    return {
        # Episode settings
        'max_episodes': 5,
        'obs_per_episode': 20,
        'max_interventions': 75,  # Long episodes for learning
        
        # Model architecture (surrogate)
        'surrogate_config': {
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 8,
            'key_size': 32,
            'dropout': 0.1,
            'encoder_type': 'node_feature',
            'attention_type': 'pairwise'
        },
        
        # Training settings
        'surrogate_lr': 3e-4,
        'batch_size': 32,
        'gradient_clip': 1.0,
        'weight_decay': 1e-4,
        
        # Policy settings (random, no training)
        'use_random_policy': True,
        'policy_config': {
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'key_size': 16,
            'dropout': 0.0
        },
        
        # Metrics and logging
        'track_metrics_per_step': True,
        'verbose': True,
        'checkpoint_freq': 10,
        'save_trajectories': True,
        
        # Experiment settings
        'seed': 42,
        'device': 'cpu'  # or 'gpu' if available
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