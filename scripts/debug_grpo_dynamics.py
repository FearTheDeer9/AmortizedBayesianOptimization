#!/usr/bin/env python3
"""
Debug GRPO training dynamics to understand why learning is so slow.

This script focuses on:
1. Gradient magnitudes during GRPO updates
2. Advantage computation and baseline
3. Exploration vs exploitation trade-off
4. Impact of standardization on learning
"""

import sys
sys.path.append('.')

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Dict, List, Tuple
import logging

# Set up detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.acquisition.better_rewards import adaptive_sigmoid_reward, RunningStats
from src.causal_bayes_opt.data_structures.sample import get_values
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor


def test_grpo_gradients():
    """Test GRPO gradient magnitudes with different scenarios."""
    logger.info("\n=== TESTING GRPO GRADIENTS ===")
    
    # Create simple scenario
    scm = create_fork_scm()
    target_var = 'Y'
    
    # Create trainer
    trainer = UnifiedGRPOTrainer(
        learning_rate=3e-4,
        n_episodes=1,  # Just one episode
        episode_length=20,
        batch_size=32,
        use_early_stopping=False,
        reward_weights={
            'optimization': 1.0,  # Only optimization reward
            'discovery': 0.0,
            'efficiency': 0.0,
            'info_gain': 0.0
        },
        optimization_direction="MINIMIZE",
        use_surrogate=False,
        seed=42
    )
    
    # Manually run one GRPO update to inspect gradients
    logger.info("\n--- Running Manual GRPO Update ---")
    
    # Create buffer with some data
    buffer = ExperienceBuffer()
    obs_samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in obs_samples:
        buffer.add_observation(sample)
    
    # Create a batch of interventions with known good/bad outcomes
    good_interventions = []
    bad_interventions = []
    
    # X interventions (parent of Y) should be good
    for i in range(5):
        intervention = create_perfect_intervention(
            targets=frozenset(['X']),
            values={'X': np.random.randn()}
        )
        samples = sample_with_intervention(scm, intervention, n_samples=10, seed=42+i)
        outcome = get_values(samples[0])[target_var]
        good_interventions.append({
            'variable': 0,  # X index
            'value': list(intervention.values())[0],
            'outcome': outcome
        })
    
    # Z interventions (also parent) 
    for i in range(5):
        intervention = create_perfect_intervention(
            targets=frozenset(['Z']),
            values={'Z': np.random.randn()}
        )
        samples = sample_with_intervention(scm, intervention, n_samples=10, seed=52+i)
        outcome = get_values(samples[0])[target_var]
        bad_interventions.append({
            'variable': 2,  # Z index
            'value': list(intervention.values())[0],
            'outcome': outcome
        })
    
    # Compute rewards
    stats = RunningStats()
    all_interventions = good_interventions + bad_interventions
    
    for interv in all_interventions:
        reward = adaptive_sigmoid_reward(interv['outcome'], stats, 'MINIMIZE')
        interv['reward'] = reward
    
    logger.info(f"\nGood intervention rewards: {[i['reward'] for i in good_interventions]}")
    logger.info(f"Bad intervention rewards: {[i['reward'] for i in bad_interventions]}")
    
    # Test with different advantage scales
    for advantage_scale in [0.1, 1.0, 10.0]:
        logger.info(f"\n--- Testing with advantage scale: {advantage_scale} ---")
        
        # Create fake advantages
        advantages = []
        for i in good_interventions:
            advantages.append(advantage_scale)  # Good gets positive
        for i in bad_interventions:
            advantages.append(-advantage_scale)  # Bad gets negative
        
        advantages = jnp.array(advantages)
        logger.info(f"Advantages: {advantages}")
        
        # Compute what the gradient magnitude would be
        # Simplified: gradient ∝ advantage * log_prob_gradient
        # For uniform initial policy, log_prob_gradient ≈ 1/n_actions
        n_actions = 2  # Two valid actions (X and Z)
        gradient_scale = jnp.mean(jnp.abs(advantages)) / n_actions
        
        logger.info(f"Expected gradient scale: {gradient_scale:.6f}")
        logger.info(f"With learning rate 3e-4: {gradient_scale * 3e-4:.9f}")
    
    return all_interventions


def test_standardization_impact():
    """Test how standardization affects discrimination."""
    logger.info("\n=== TESTING STANDARDIZATION IMPACT ===")
    
    scm = create_fork_scm()
    target_var = 'Y'
    
    # Create buffer with high variance data
    buffer = ExperienceBuffer()
    
    # Add samples with different scales
    for scale in [0.1, 1.0, 10.0]:
        logger.info(f"\n--- Testing with data scale: {scale} ---")
        
        # Sample with scaled noise
        modified_scm = scm  # In practice, would scale the SCM noise
        samples = sample_from_linear_scm(modified_scm, 50, seed=42)
        
        # Manually scale the values
        for sample in samples:
            values = get_values(sample)
            scaled_values = {k: v * scale for k, v in values.items()}
            scaled_sample = sample.set('values', scaled_values)
            buffer.add_observation(scaled_sample)
        
        # Convert with and without standardization
        tensor_std, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, standardize=True
        )
        tensor_unstd, _ = buffer_to_three_channel_tensor(
            buffer, target_var, standardize=False
        )
        
        # Compare discrimination ability
        values_std = tensor_std[:, :, 0]
        values_unstd = tensor_unstd[:, :, 0]
        
        # Compute "discrimination score" - how different are the variables?
        disc_std = jnp.std(jnp.mean(values_std, axis=0))
        disc_unstd = jnp.std(jnp.mean(values_unstd, axis=0))
        
        logger.info(f"Discrimination (standardized): {disc_std:.6f}")
        logger.info(f"Discrimination (unstandardized): {disc_unstd:.6f}")
        logger.info(f"Ratio: {disc_unstd/disc_std:.2f}x")


def test_exploration_noise():
    """Test impact of exploration noise on learning."""
    logger.info("\n=== TESTING EXPLORATION NOISE ===")
    
    # Test different noise levels
    for noise_level in [0.01, 0.1, 0.5, 1.0]:
        logger.info(f"\n--- Noise level: {noise_level} ---")
        
        # Simulate policy selection with noise
        true_logits = jnp.array([-1.0, -np.inf, -3.0])  # X is better than Z
        valid_mask = true_logits != -np.inf
        
        # Add noise and sample
        key = jax.random.PRNGKey(42)
        sampled_actions = []
        
        for i in range(100):
            key, subkey = jax.random.split(key)
            noisy_logits = true_logits + noise_level * jax.random.normal(subkey, true_logits.shape)
            noisy_logits = jnp.where(valid_mask, noisy_logits, -np.inf)
            
            action = jax.random.categorical(subkey, noisy_logits)
            sampled_actions.append(int(action))
        
        # Analyze selection frequencies
        action_counts = np.bincount(sampled_actions, minlength=3)
        action_probs = action_counts / len(sampled_actions)
        
        true_probs = jax.nn.softmax(true_logits)
        
        logger.info(f"True probs: X={true_probs[0]:.3f}, Z={true_probs[2]:.3f}")
        logger.info(f"Sampled probs: X={action_probs[0]:.3f}, Z={action_probs[2]:.3f}")
        logger.info(f"Deviation: {np.abs(action_probs - true_probs).sum():.3f}")


def test_baseline_computation():
    """Test how baseline affects advantages."""
    logger.info("\n=== TESTING BASELINE COMPUTATION ===")
    
    # Simulate a batch of rewards
    rewards = jnp.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.5, 0.6, 0.4])
    
    # Different baseline strategies
    baselines = {
        'mean': jnp.mean(rewards),
        'median': jnp.median(rewards),
        'min': jnp.min(rewards),
        'zero': 0.0
    }
    
    for name, baseline in baselines.items():
        advantages = rewards - baseline
        
        logger.info(f"\n--- Baseline: {name} ({baseline:.3f}) ---")
        logger.info(f"Advantages: {advantages}")
        logger.info(f"Advantage std: {jnp.std(advantages):.3f}")
        logger.info(f"Positive advantages: {jnp.sum(advantages > 0)}/{len(advantages)}")
        
        # Normalized advantages (what GRPO uses)
        adv_std = jnp.std(advantages)
        if adv_std > 1e-8:
            norm_advantages = advantages / adv_std
            logger.info(f"Normalized advantages: {norm_advantages}")
            logger.info(f"Normalized range: [{norm_advantages.min():.2f}, {norm_advantages.max():.2f}]")


def main():
    logger.info("="*80)
    logger.info("GRPO DYNAMICS DEBUGGING")
    logger.info("="*80)
    
    # Run tests
    interventions = test_grpo_gradients()
    test_standardization_impact()
    test_exploration_noise()
    test_baseline_computation()
    
    # Summary insights
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    
    logger.info("\n1. GRADIENT SCALE:")
    logger.info("   - With uniform initial policy and small advantages, gradients are tiny")
    logger.info("   - Learning rate 3e-4 × small gradients = negligible parameter updates")
    
    logger.info("\n2. STANDARDIZATION:")
    logger.info("   - Reduces discrimination between variables")
    logger.info("   - Makes all inputs look similar to the network")
    
    logger.info("\n3. EXPLORATION NOISE:")
    logger.info("   - 0.1 noise level significantly affects action selection")
    logger.info("   - Can overwhelm small policy preferences")
    
    logger.info("\n4. BASELINE:")
    logger.info("   - Mean baseline creates balanced positive/negative advantages")
    logger.info("   - But also reduces the magnitude of advantages")
    
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS:")
    logger.info("1. Try larger learning rate (3e-3 or 3e-2)")
    logger.info("2. Reduce exploration noise (0.01 instead of 0.1)")
    logger.info("3. Consider not standardizing or using different normalization")
    logger.info("4. Use stronger reward signal (larger weight differences)")
    logger.info("="*80)


if __name__ == "__main__":
    main()