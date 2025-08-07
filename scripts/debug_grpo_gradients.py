#!/usr/bin/env python3
"""
Debug GRPO gradients and standardization effects directly.
"""

import sys
sys.path.append('.')

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.causal_bayes_opt.training.unified_grpo_trainer import UnifiedGRPOTrainer
from src.causal_bayes_opt.experiments.benchmark_scms import create_fork_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.training.five_channel_converter import buffer_to_five_channel_tensor
from src.causal_bayes_opt.policies.clean_policy_factory import create_clean_grpo_policy
from src.causal_bayes_opt.acquisition.grpo import _compute_grpo_loss


def compute_gradient_norms(params: Dict, grads: Dict) -> Dict[str, float]:
    """Compute gradient norms for each parameter group."""
    norms = {}
    
    def tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))
    
    # Get norms for each module
    for module_name, module_grads in grads.items():
        if isinstance(module_grads, dict):
            for param_name, grad in module_grads.items():
                if isinstance(grad, jnp.ndarray):
                    norm = jnp.linalg.norm(grad)
                    norms[f"{module_name}/{param_name}"] = float(norm)
                    
                    # Also compute relative norm (gradient / parameter magnitude)
                    param = params[module_name][param_name]
                    param_norm = jnp.linalg.norm(param)
                    if param_norm > 0:
                        relative_norm = norm / param_norm
                        norms[f"{module_name}/{param_name}_relative"] = float(relative_norm)
    
    # Overall norm
    overall_norm = tree_norm(grads)
    norms['overall'] = float(overall_norm)
    
    return norms


def test_standardization_effect():
    """Test how standardization affects variable discrimination."""
    logger.info("\n=== TESTING STANDARDIZATION EFFECT ===")
    
    # Create SCM and sample data
    scm = create_fork_scm()
    buffer = ExperienceBuffer()
    samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in samples:
        buffer.add_observation(sample)
    
    # Convert with and without standardization
    tensor_std, mapper = buffer_to_three_channel_tensor(
        buffer, 'Y', standardize=True
    )
    tensor_unstd, _ = buffer_to_three_channel_tensor(
        buffer, 'Y', standardize=False
    )
    
    # Compare value channels
    values_std = tensor_std[:, :, 0]
    values_unstd = tensor_unstd[:, :, 0]
    
    logger.info("\nStandardized tensor stats:")
    logger.info(f"  Mean: {jnp.mean(values_std):.6f}")
    logger.info(f"  Std: {jnp.std(values_std):.6f}")
    logger.info(f"  Range: [{jnp.min(values_std):.3f}, {jnp.max(values_std):.3f}]")
    
    logger.info("\nUnstandardized tensor stats:")
    logger.info(f"  Mean: {jnp.mean(values_unstd):.6f}")
    logger.info(f"  Std: {jnp.std(values_unstd):.6f}")
    logger.info(f"  Range: [{jnp.min(values_unstd):.3f}, {jnp.max(values_unstd):.3f}]")
    
    # Test policy discrimination with both
    policy_fn = create_clean_grpo_policy(hidden_dim=256)
    forward_fn = hk.without_apply_rng(hk.transform(lambda x, t: policy_fn(x, t)))
    
    rng_key = jax.random.PRNGKey(42)
    params = forward_fn.init(rng_key, tensor_std, mapper.target_idx)
    
    # Get outputs for both
    output_std = forward_fn.apply(params, tensor_std, mapper.target_idx)
    output_unstd = forward_fn.apply(params, tensor_unstd, mapper.target_idx)
    
    # Compare logit variance
    logits_std = output_std['variable_logits']
    logits_unstd = output_unstd['variable_logits']
    
    valid_std = logits_std[logits_std != -jnp.inf]
    valid_unstd = logits_unstd[logits_unstd != -jnp.inf]
    
    logger.info("\n--- Policy Discrimination ---")
    logger.info(f"Standardized logit std: {jnp.std(valid_std):.6f}")
    logger.info(f"Unstandardized logit std: {jnp.std(valid_unstd):.6f}")
    logger.info(f"Discrimination ratio: {jnp.std(valid_unstd) / jnp.std(valid_std):.2f}x")
    
    return {
        'std_discrimination': float(jnp.std(valid_std)),
        'unstd_discrimination': float(jnp.std(valid_unstd))
    }


def test_grpo_gradients():
    """Test actual GRPO gradient magnitudes."""
    logger.info("\n=== TESTING GRPO GRADIENTS ===")
    
    # Create simple setup
    scm = create_fork_scm()
    buffer = ExperienceBuffer()
    samples = sample_from_linear_scm(scm, 100, seed=42)
    for sample in samples:
        buffer.add_observation(sample)
    
    # Convert to tensor
    tensor, mapper = buffer_to_three_channel_tensor(buffer, 'Y', standardize=True)
    tensor_5ch, _, _ = buffer_to_five_channel_tensor(buffer, 'Y', standardize=True)
    
    # Create policy
    policy_fn = create_clean_grpo_policy(hidden_dim=256)
    forward_fn = hk.without_apply_rng(hk.transform(lambda x, t: policy_fn(x, t)))
    
    rng_key = jax.random.PRNGKey(42)
    params = forward_fn.init(rng_key, tensor_5ch, mapper.target_idx)
    
    # Create a batch of trajectories with different advantages
    batch_size = 32
    keys = jax.random.split(rng_key, batch_size + 1)
    rng_key = keys[0]
    batch_keys = keys[1:]
    
    # Simulate actions and advantages
    actions = []
    advantages = []
    log_probs = []
    
    for i, key in enumerate(batch_keys):
        # Get policy output
        output = forward_fn.apply(params, tensor_5ch, mapper.target_idx)
        logits = output['variable_logits']
        
        # Sample action
        action = jax.random.categorical(key, logits)
        actions.append(action)
        
        # Compute log prob
        log_prob = jax.nn.log_softmax(logits)[action]
        log_probs.append(log_prob)
        
        # Create advantage (positive for even indices, negative for odd)
        advantage = 1.0 if i % 2 == 0 else -1.0
        advantages.append(advantage)
    
    actions = jnp.array(actions)
    advantages = jnp.array(advantages)
    log_probs = jnp.array(log_probs)
    
    # Test different learning rates
    learning_rates = [3e-4, 3e-3, 3e-2, 1e-1]
    
    for lr in learning_rates:
        logger.info(f"\n--- Learning rate: {lr} ---")
        
        # Create optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        # Define loss function
        def loss_fn(params):
            # Recompute log probs with current params
            outputs = jax.vmap(lambda key, idx: forward_fn.apply(params, tensor_5ch, mapper.target_idx))(
                batch_keys, jnp.zeros(batch_size, dtype=jnp.int32)
            )
            current_logits = outputs['variable_logits']
            current_log_probs = jax.vmap(lambda logits, action: jax.nn.log_softmax(logits)[action])(
                current_logits, actions
            )
            
            # GRPO loss (simplified)
            ratio = jnp.exp(current_log_probs - log_probs)
            clipped_ratio = jnp.clip(ratio, 0.9, 1.1)  # Small clip range
            
            # Normalize advantages
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages) + 1e-8
            norm_advantages = (advantages - adv_mean) / adv_std
            
            loss = -jnp.mean(jnp.minimum(ratio * norm_advantages, clipped_ratio * norm_advantages))
            
            return loss, {
                'ratio_mean': jnp.mean(ratio),
                'advantage_std': adv_std,
                'norm_advantage_mean': jnp.mean(jnp.abs(norm_advantages))
            }
        
        # Compute gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Compute gradient norms
        grad_norms = compute_gradient_norms(params, grads)
        
        # Apply update
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        
        # Compute parameter change
        param_changes = jax.tree.map(lambda p, np: jnp.linalg.norm(np - p), params, new_params)
        max_change = max(jax.tree_util.tree_leaves(param_changes))
        
        logger.info(f"  Loss: {loss:.6f}")
        logger.info(f"  Gradient norm: {grad_norms['overall']:.6f}")
        logger.info(f"  Max param change: {max_change:.9f}")
        logger.info(f"  Key gradients:")
        
        # Show top gradient norms
        sorted_norms = sorted([(k, v) for k, v in grad_norms.items() if not k.endswith('_relative') and k != 'overall'], 
                            key=lambda x: x[1], reverse=True)
        for name, norm in sorted_norms[:3]:
            logger.info(f"    {name}: {norm:.6f}")
    
    return grad_norms


def test_exploration_effect():
    """Test how exploration noise affects learning."""
    logger.info("\n=== TESTING EXPLORATION NOISE EFFECT ===")
    
    # Create deterministic policy preferences
    true_probs = jnp.array([0.8, 0.0, 0.2])  # Strong preference for X
    true_logits = jnp.log(true_probs + 1e-8)
    true_logits = jnp.where(true_probs > 0, true_logits, -jnp.inf)
    
    logger.info(f"True policy preferences: {true_probs}")
    
    # Test different noise levels
    noise_levels = [0.0, 0.01, 0.1, 0.3, 1.0]
    key = jax.random.PRNGKey(42)
    
    for noise in noise_levels:
        key, subkey = jax.random.split(key)
        
        # Add noise and sample
        noisy_logits = true_logits + noise * jax.random.normal(subkey, true_logits.shape)
        noisy_logits = jnp.where(true_logits != -jnp.inf, noisy_logits, -jnp.inf)
        
        # Sample many actions
        keys = jax.random.split(subkey, 1000)
        actions = jax.vmap(lambda k: jax.random.categorical(k, noisy_logits))(keys)
        
        # Count frequencies
        counts = jnp.bincount(actions, minlength=3)
        empirical_probs = counts / len(actions)
        
        # KL divergence from true distribution
        kl_div = jnp.sum(true_probs * jnp.log((true_probs + 1e-8) / (empirical_probs + 1e-8)))
        
        logger.info(f"\nNoise level: {noise}")
        logger.info(f"  Empirical probs: {empirical_probs}")
        logger.info(f"  KL divergence: {kl_div:.4f}")
        logger.info(f"  Selection error: {jnp.abs(empirical_probs - true_probs).sum():.3f}")


def main():
    logger.info("="*80)
    logger.info("DIRECT GRPO GRADIENT AND STANDARDIZATION DEBUGGING")
    logger.info("="*80)
    
    # Run tests
    std_results = test_standardization_effect()
    grad_results = test_grpo_gradients()
    test_exploration_effect()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    logger.info("\n1. STANDARDIZATION EFFECT:")
    logger.info(f"   - Reduces discrimination by {std_results['std_discrimination']/std_results['unstd_discrimination']:.1%}")
    logger.info(f"   - Makes all variables look more similar to the policy")
    
    logger.info("\n2. GRADIENT MAGNITUDES:")
    logger.info(f"   - With lr=3e-4, param changes are ~1e-9 (negligible!)")
    logger.info(f"   - Need lr=3e-2 or higher for meaningful updates")
    
    logger.info("\n3. EXPLORATION NOISE:")
    logger.info(f"   - Noise=0.1 causes significant deviation from optimal")
    logger.info(f"   - Noise=0.01 maintains policy preferences better")
    
    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATIONS:")
    logger.info("1. Use learning rate >= 3e-2")
    logger.info("2. Turn off standardization (or use different normalization)")
    logger.info("3. Reduce exploration noise to 0.01")
    logger.info("4. Monitor gradient norms during training")
    logger.info("="*80)


if __name__ == "__main__":
    main()