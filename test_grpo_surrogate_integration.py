#!/usr/bin/env python3
"""
Test GRPO-Surrogate integration with information gain rewards.

This test verifies that:
1. Surrogate predictions change after interventions
2. Information gain is computed correctly
3. Rewards incorporate information gain properly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import numpy as np
from omegaconf import DictConfig

# Patch before imports to catch all uses
captured_rewards = []

def capture_reward_details(buffer_before, intervention, outcome, target_variable, 
                         config=None, posterior_before=None, posterior_after=None):
    """Wrapper to capture reward computation details."""
    print(f"\n*** REWARD FUNCTION CALLED ***")
    print(f"  posterior_before: {posterior_before is not None}")
    print(f"  posterior_after: {posterior_after is not None}")
    
    # Import the original here to avoid circular import
    from src.causal_bayes_opt.acquisition import clean_rewards
    original = clean_rewards._original_compute_clean_reward
    
    result = original(buffer_before, intervention, outcome, target_variable, 
                     config, posterior_before, posterior_after)
    
    # Capture details
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
    
    # Log details
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"\nReward computation #{len(captured_rewards)}:")
    logger.info(f"  Intervention: {intervention}")
    logger.info(f"  Has posteriors: {details['has_posteriors']}")
    logger.info(f"  Total reward: {result['total']:.3f}")
    
    return result

# Patch early before any imports
import src.causal_bayes_opt.acquisition.clean_rewards
src.causal_bayes_opt.acquisition.clean_rewards._original_compute_clean_reward = src.causal_bayes_opt.acquisition.clean_rewards.compute_clean_reward
src.causal_bayes_opt.acquisition.clean_rewards.compute_clean_reward = capture_reward_details

# Now do the rest of the imports
from src.causal_bayes_opt.training.unified_grpo_trainer import create_unified_grpo_trainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.scm import get_target, get_variables
from src.causal_bayes_opt.utils.checkpoint_utils import load_checkpoint, create_model_from_checkpoint

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_grpo_surrogate_integration():
    """Test that information gain rewards work correctly with real surrogate."""
    
    print("=" * 60)
    print("Testing GRPO-Surrogate Integration")
    print("=" * 60)
    
    # 1. Load pre-trained surrogate
    surrogate_path = Path("checkpoints/bc_surrogate_final")
    if not surrogate_path.exists():
        print("❌ BC surrogate checkpoint not found. Please train it first.")
        return
    
    print(f"\n1. Loading pre-trained surrogate from {surrogate_path}")
    try:
        checkpoint = load_checkpoint(surrogate_path)
        net, params = create_model_from_checkpoint(checkpoint)
        pretrained_surrogate = {'net': net, 'params': params}
        print(f"✓ Loaded surrogate with architecture: {checkpoint['architecture']}")
    except Exception as e:
        print(f"❌ Failed to load surrogate: {e}")
        return
    
    # 2. Create GRPO trainer with surrogate
    print("\n2. Creating GRPO trainer with surrogate")
    config = DictConfig({
        'seed': 42,
        'max_episodes': 1,
        'n_variables_range': [3, 3],
        'obs_per_episode': 50,
        'max_interventions': 5,
        'batch_size': 2,
        'learning_rate': 3e-4,
        'use_surrogate': True,
        'reward_weights': {
            'optimization': 0.4,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.4  # High weight to test info gain
        }
    })
    
    trainer = create_unified_grpo_trainer(config, pretrained_surrogate=pretrained_surrogate)
    print("✓ Created GRPO trainer with info gain weight = 0.4")
    
    # 3. Create a simple test SCM
    print("\n3. Creating test SCM (fork structure)")
    scm = create_fork_scm(noise_scale=1.0)
    variables = list(get_variables(scm))
    target = get_target(scm)
    print(f"✓ Created fork SCM: variables={variables}, target={target}")
    
    # 4. Run one episode and capture reward details
    print("\n4. Running one GRPO episode")
    print("-" * 40)
    
    # Run one episode
    from jax import random
    key = random.PRNGKey(42)
    
    # Debug: check if trainer has surrogate
    print(f"\nDebug info:")
    print(f"  trainer.use_surrogate = {trainer.use_surrogate}")
    print(f"  trainer.surrogate_predict_fn is not None = {trainer.surrogate_predict_fn is not None}")
    print(f"  trainer.reward_weights = {trainer.reward_weights}")
    
    metrics = trainer._run_grpo_episode(0, scm, "test_fork", key)
    
    print("\n" + "-" * 40)
    print(f"Episode completed: mean_reward={metrics['mean_reward']:.3f}")
    
    # 5. Analyze captured rewards
    print("\n5. Analysis of Information Gain Rewards")
    print("-" * 40)
    
    info_gain_rewards = [r for r in captured_rewards if r['has_posteriors']]
    
    if not info_gain_rewards:
        print("❌ No rewards with posterior information were captured!")
        print(f"   Total rewards captured: {len(captured_rewards)}")
        for i, r in enumerate(captured_rewards):
            print(f"   Reward {i+1}: has_posteriors={r['has_posteriors']}")
        return
    
    print(f"✓ Captured {len(info_gain_rewards)} rewards with information gain")
    
    # Check if entropy changes
    entropy_changes = []
    for i, r in enumerate(info_gain_rewards):
        entropy_before = r['entropy_before']
        entropy_after = r['entropy_after']
        info_gain = entropy_before - entropy_after
        info_gain_reward = r['info_gain_reward']
        
        entropy_changes.append(info_gain)
        
        print(f"\nIntervention {i+1}: {r['intervention']['targets']}")
        print(f"  Entropy: {entropy_before:.3f} → {entropy_after:.3f}")
        print(f"  Information gain: {info_gain:+.3f}")
        print(f"  Info gain reward: {info_gain_reward:.3f}")
        
        # Verify reward calculation
        expected_reward = 1.0 / (1.0 + np.exp(-4.0 * info_gain))
        if abs(info_gain_reward - expected_reward) > 0.001:
            print(f"  ❌ Reward mismatch! Expected: {expected_reward:.3f}")
        else:
            print(f"  ✓ Reward calculation correct")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if len(entropy_changes) > 0:
        avg_info_gain = np.mean(entropy_changes)
        print(f"✓ Average information gain: {avg_info_gain:+.3f}")
        
        if avg_info_gain > 0:
            print("✓ Surrogate is learning from interventions (entropy decreasing)")
        elif avg_info_gain < 0:
            print("⚠️  Surrogate entropy is increasing (might be early in training)")
        else:
            print("⚠️  No information gain on average")
            
        # Check if rewards are being used
        total_rewards = [r['reward_components']['total'] for r in info_gain_rewards]
        info_gain_contribution = [r['reward_components']['info_gain'] * 0.4 for r in info_gain_rewards]
        
        print(f"\n✓ Info gain contribution to total reward: "
              f"{np.mean(info_gain_contribution) / np.mean(total_rewards) * 100:.1f}%")


if __name__ == "__main__":
    test_grpo_surrogate_integration()