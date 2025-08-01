#!/usr/bin/env python3
"""
Clean integration test using trainer's built-in metrics instead of monkey patching.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from omegaconf import DictConfig

from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.scm import get_target, get_variables
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint

def test_grpo_surrogate_clean():
    """Test GRPO-surrogate integration without monkey patching."""
    
    print("=" * 60)
    print("Clean GRPO-Surrogate Integration Test")
    print("=" * 60)
    
    # Load surrogate
    checkpoint = load_checkpoint(Path("checkpoints/bc_surrogate_final"))
    net, params = create_model_from_checkpoint(checkpoint)
    pretrained_surrogate = {'net': net, 'params': params}
    
    # Create two trainers - with and without surrogate
    config_base = DictConfig({
        'seed': 42,
        'max_episodes': 1,
        'n_variables_range': [3, 3],
        'obs_per_episode': 50,
        'max_interventions': 10,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'use_surrogate': False,
        'reward_weights': {
            'optimization': 0.7,
            'discovery': 0.2,
            'efficiency': 0.1,
            'info_gain': 0.0
        }
    })
    
    config_surrogate = config_base.copy()
    config_surrogate['use_surrogate'] = True
    config_surrogate['reward_weights']['info_gain'] = 0.4
    config_surrogate['reward_weights']['optimization'] = 0.3
    
    trainer_base = create_unified_grpo_trainer(config_base)
    trainer_surrogate = create_unified_grpo_trainer(config_surrogate, pretrained_surrogate=pretrained_surrogate)
    
    # Run on same SCM
    scm = create_fork_scm(noise_scale=1.0)
    
    from jax import random
    key = random.PRNGKey(42)
    
    # Run episodes
    print("\n1. Running WITHOUT surrogate (baseline)")
    key, subkey = random.split(key)
    metrics_base = trainer_base._run_grpo_episode(0, scm, "fork", subkey)
    print(f"   Mean reward: {metrics_base['mean_reward']:.3f}")
    
    print("\n2. Running WITH surrogate (info gain active)")
    key, subkey = random.split(key)
    metrics_surrogate = trainer_surrogate._run_grpo_episode(0, scm, "fork", subkey)
    print(f"   Mean reward: {metrics_surrogate['mean_reward']:.3f}")
    
    # Compare
    print("\n" + "=" * 40)
    print("ANALYSIS")
    print("=" * 40)
    
    reward_diff = metrics_surrogate['mean_reward'] - metrics_base['mean_reward']
    print(f"Reward difference: {reward_diff:+.3f}")
    
    if abs(reward_diff) < 0.01:
        print("⚠️  Rewards are nearly identical - info gain might not be contributing")
    else:
        print("✓ Rewards differ, suggesting info gain is active")
    
    # Check if surrogate is actually being used
    print(f"\nSurrogate configuration:")
    print(f"  use_surrogate: {trainer_surrogate.use_surrogate}")
    print(f"  info_gain weight: {trainer_surrogate.reward_weights['info_gain']}")
    print(f"  surrogate_predict_fn exists: {trainer_surrogate.surrogate_predict_fn is not None}")

if __name__ == "__main__":
    test_grpo_surrogate_clean()