"""Base configuration for joint ACBO training."""

def get_config():
    """Get base configuration for joint training."""
    return {
        # Model initialization
        'policy_checkpoint': None,  # Path or None for from-scratch
        'surrogate_checkpoint': None,  # Path or None for from-scratch
        
        # Architecture (when training from scratch)
        'policy_architecture': 'simple_permutation_invariant',
        'surrogate_architecture': 'avici_style',
        'hidden_dim': 128,
        'num_layers': 8,
        'num_heads': 8,
        'key_size': 32,
        'dropout': 0.1,
        
        # GRPO reward weights (fully configurable)
        'grpo_reward_config': {
            'target_improvement_weight': 0.7,
            'parent_accuracy_weight': 0.2,
            'info_gain_weight': 0.1,
            'exploration_bonus': 0.0
        },
        
        # Alternating schedule
        'policy_episodes_per_phase': 5,
        'surrogate_steps_per_phase': 1000,
        'f1_rotation_threshold': 0.9,
        
        # SCM generation
        'scm_generation': {
            'min_vars': 3,
            'max_vars': 30,
            'generator_type': 'diverse'
        },
        
        # Training settings
        'max_episodes': 200,
        'obs_per_episode': 10,
        'max_interventions': 30,
        'use_surrogate': True,
        'use_grpo_rewards': True,
        'use_replay_buffer': True,
        
        # Optimization
        'learning_rate': 5e-4,  # Policy learning rate
        'surrogate_lr': 1e-4,    # Surrogate learning rate
        'batch_size': 32,
        
        # GRPO specific
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.001,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Fixed exploration
        'use_fixed_std': True,
        'fixed_std': 0.5,
        
        # Logging
        'checkpoint_dir': 'experiments/joint-training/checkpoints',
        'log_every': 10,
        'save_every': 50,
        'verbose': False,
        
        # General
        'seed': 42
    }


def get_pretrained_config():
    """Configuration for training with pretrained models."""
    config = get_config()
    
    # Example paths - update with actual checkpoint paths
    config['policy_checkpoint'] = 'experiments/policy-only-training/checkpoints/diverse_fixed_3to10/joint_ep2/policy.pkl'
    config['surrogate_checkpoint'] = 'experiments/surrogate-only-training/checkpoints/avici_runs/avici_style_20250818_161941/best_model.pkl'
    
    # Adjust training for fine-tuning
    config['learning_rate'] = 1e-4  # Lower LR for fine-tuning
    config['surrogate_lr'] = 5e-5   # Lower LR for fine-tuning
    
    return config


def get_quick_test_config():
    """Configuration for quick testing."""
    config = get_config()
    
    config['max_episodes'] = 2
    config['policy_episodes_per_phase'] = 1
    config['surrogate_steps_per_phase'] = 100
    config['scm_generation']['max_vars'] = 10
    config['verbose'] = True
    
    return config