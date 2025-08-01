#!/usr/bin/env python3
"""
Test GRPO-Surrogate integration focusing on positive information gain scenarios.

This test creates a scenario where information gain should be positive by:
1. Starting with high uncertainty (random initial buffer)
2. Making targeted interventions that should reduce uncertainty
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from omegaconf import DictConfig

# Patch before imports
captured_rewards = []

def capture_reward_details(buffer_before, intervention, outcome, target_variable, 
                         config=None, posterior_before=None, posterior_after=None):
    """Wrapper to capture reward computation details."""
    from src.causal_bayes_opt.acquisition import clean_rewards
    original = clean_rewards._original_compute_clean_reward
    
    result = original(buffer_before, intervention, outcome, target_variable, 
                     config, posterior_before, posterior_after)
    
    details = {
        'intervention': intervention,
        'reward_components': result,
        'has_posteriors': posterior_before is not None and posterior_after is not None,
        'posterior_before': posterior_before,
        'posterior_after': posterior_after
    }
    
    if posterior_before and posterior_after:
        details['entropy_before'] = posterior_before.get('entropy', -1)
        details['entropy_after'] = posterior_after.get('entropy', -1)
        details['info_gain_reward'] = result.get('info_gain', 0)
    
    captured_rewards.append(details)
    return result

# Patch early
import src.causal_bayes_opt.acquisition.clean_rewards
src.causal_bayes_opt.acquisition.clean_rewards._original_compute_clean_reward = src.causal_bayes_opt.acquisition.clean_rewards.compute_clean_reward
src.causal_bayes_opt.acquisition.clean_rewards.compute_clean_reward = capture_reward_details

# Now imports
from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm, create_chain_scm
from src.causal_bayes_opt.data_structures.scm import get_target, get_variables
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint

def test_positive_information_gain():
    """Test scenarios that should produce positive information gain."""
    
    print("=" * 60)
    print("Testing Positive Information Gain Scenarios")
    print("=" * 60)
    
    # Load pre-trained surrogate
    surrogate_path = Path("checkpoints/bc_surrogate_final")
    if not surrogate_path.exists():
        print("❌ BC surrogate checkpoint not found.")
        return
    
    checkpoint = load_checkpoint(surrogate_path)
    net, params = create_model_from_checkpoint(checkpoint)
    pretrained_surrogate = {'net': net, 'params': params}
    print(f"✓ Loaded surrogate")
    
    # Create GRPO trainer with high info gain weight
    config = DictConfig({
        'seed': 42,
        'max_episodes': 1,
        'n_variables_range': [4, 4],
        'obs_per_episode': 50,
        'max_interventions': 10,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'use_surrogate': True,
        'reward_weights': {
            'optimization': 0.2,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.6  # High weight for info gain
        }
    })
    
    trainer = create_unified_grpo_trainer(config, pretrained_surrogate=pretrained_surrogate)
    print("✓ Created GRPO trainer with info gain weight = 0.6")
    
    # Test 1: Chain SCM (clear parent relationships)
    print("\n" + "=" * 40)
    print("Test 1: Chain SCM")
    print("=" * 40)
    
    captured_rewards.clear()
    scm = create_chain_scm(chain_length=4, noise_scale=1.0)
    variables = list(get_variables(scm))
    target = get_target(scm)
    print(f"Created chain: {' → '.join(variables)}, target={target}")
    
    from jax import random
    key = random.PRNGKey(123)
    metrics = trainer._run_grpo_episode(0, scm, "chain_4", key)
    
    print(f"\nEpisode completed: mean_reward={metrics['mean_reward']:.3f}")
    analyze_information_gains()
    
    # Test 2: Fork SCM with multiple interventions
    print("\n" + "=" * 40)
    print("Test 2: Fork SCM (Extended)")
    print("=" * 40)
    
    captured_rewards.clear()
    
    # Run multiple episodes to see learning
    print("\nRunning 3 episodes to observe learning...")
    for episode in range(3):
        key = random.PRNGKey(456 + episode)
        metrics = trainer._run_grpo_episode(episode, scm, "fork_extended", key)
        print(f"Episode {episode+1}: mean_reward={metrics['mean_reward']:.3f}")
    
    print("\nAnalysis across all episodes:")
    analyze_information_gains()

def analyze_information_gains():
    """Analyze captured information gains."""
    info_gain_rewards = [r for r in captured_rewards if r['has_posteriors']]
    
    if not info_gain_rewards:
        print("❌ No rewards with information gain captured")
        return
    
    print(f"\n✓ Captured {len(info_gain_rewards)} rewards with information gain")
    
    # Compute statistics
    info_gains = []
    positive_gains = 0
    
    for r in info_gain_rewards:
        entropy_before = r['entropy_before']
        entropy_after = r['entropy_after']
        info_gain = entropy_before - entropy_after
        info_gains.append(info_gain)
        
        if info_gain > 0:
            positive_gains += 1
    
    print(f"\nInformation Gain Statistics:")
    print(f"  Positive gains: {positive_gains}/{len(info_gains)} ({positive_gains/len(info_gains)*100:.1f}%)")
    print(f"  Average gain: {np.mean(info_gains):+.3f} nats")
    print(f"  Max gain: {np.max(info_gains):+.3f} nats")
    print(f"  Min gain: {np.min(info_gains):+.3f} nats")
    
    # Show top 3 positive gains
    sorted_gains = sorted(zip(info_gains, info_gain_rewards), reverse=True)
    print(f"\nTop information gains:")
    for i, (gain, r) in enumerate(sorted_gains[:3]):
        if gain <= 0:
            break
        print(f"  {i+1}. Intervention on {r['intervention']['targets']}: {gain:+.3f} nats")
        print(f"     Entropy: {r['entropy_before']:.3f} → {r['entropy_after']:.3f}")


if __name__ == "__main__":
    test_positive_information_gain()