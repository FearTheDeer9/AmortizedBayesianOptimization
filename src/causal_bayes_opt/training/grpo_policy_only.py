"""True GRPO (Guided Reward Policy Optimization) - Policy-Only Implementation

This module provides a pure policy-only implementation of GRPO that uses guided rewards
directly without any value function estimation. This is the correct GRPO architecture
that leverages verifiable rewards for direct policy optimization.

Key features:
- Policy-only optimization using actual rewards
- No value function estimation or advantage computation  
- Direct reward-to-go calculation for policy gradients
- JAX-compiled for performance
- Immutable data structures following functional programming principles
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyOnlyGRPOConfig:
    """Configuration for policy-only GRPO algorithm.
    
    Args:
        learning_rate: Policy learning rate
        discount_factor: Reward discount factor (gamma)
        clip_ratio: PPO-style clipping ratio for policy updates
        entropy_coefficient: Entropy regularization coefficient
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping
        normalize_rewards: Whether to normalize reward-to-go values
        use_reward_baseline: Whether to use reward baseline for variance reduction
    """
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    normalize_rewards: bool = True
    use_reward_baseline: bool = True


@dataclass(frozen=True)
class PolicyTrajectory:
    """Immutable trajectory data for policy-only GRPO training.
    
    Args:
        states: Sequence of states [T, ...]
        actions: Sequence of actions [T, ...]
        rewards: Sequence of rewards [T]
        log_probs: Sequence of action log probabilities [T]
        dones: Sequence of episode termination flags [T]
        reward_to_go: Computed reward-to-go values [T]
    """
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    log_probs: jnp.ndarray
    dones: jnp.ndarray
    reward_to_go: jnp.ndarray


@dataclass(frozen=True)
class PolicyUpdateResult:
    """Results from a policy-only GRPO update.
    
    Args:
        policy_loss: Policy gradient loss
        entropy_loss: Entropy regularization loss
        total_loss: Combined total loss
        kl_divergence: KL divergence between old and new policy
        policy_gradient_norm: Norm of policy gradients
        clipped_fraction: Fraction of policy updates that were clipped
        reward_baseline: Reward baseline used for variance reduction
    """
    policy_loss: float
    entropy_loss: float
    total_loss: float
    kl_divergence: float
    policy_gradient_norm: float
    clipped_fraction: float
    reward_baseline: float


# Pure functions for reward-to-go computation
@jax.jit
def compute_reward_to_go(
    rewards: jnp.ndarray,      # [T]
    dones: jnp.ndarray,        # [T]
    gamma: float = 0.99
) -> jnp.ndarray:
    """Compute reward-to-go (discounted cumulative rewards) for policy gradients.
    
    Pure function implementing reward-to-go calculation.
    
    Args:
        rewards: Rewards for each timestep [T]
        dones: Episode termination flags [T]
        gamma: Discount factor
        
    Returns:
        Reward-to-go values [T]
    """
    T = rewards.shape[0]
    
    def rtg_step(carry, inputs):
        rtg = carry
        reward, done = inputs
        rtg = reward + gamma * rtg * (1.0 - done)
        return rtg, rtg
    
    # Process in reverse order for reward-to-go
    initial_rtg = 0.0
    scan_inputs = (rewards[::-1], dones[::-1])
    _, rtg_values_rev = jax.lax.scan(rtg_step, initial_rtg, scan_inputs)
    
    return rtg_values_rev[::-1]


@jax.jit
def normalize_rewards(rewards: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize reward-to-go values to have zero mean and unit variance.
    
    Pure function for reward normalization to reduce variance.
    
    Args:
        rewards: Raw reward-to-go values [T]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized rewards [T]
    """
    mean = jnp.mean(rewards)
    std = jnp.std(rewards)
    return (rewards - mean) / (std + eps)


@jax.jit
def compute_reward_baseline(rewards: jnp.ndarray) -> float:
    """Compute simple reward baseline for variance reduction.
    
    Args:
        rewards: Reward-to-go values [T]
        
    Returns:
        Reward baseline (mean of rewards)
    """
    return jnp.mean(rewards)


# Pure functions for policy updates
@jax.jit
def compute_policy_loss_rewards(
    old_log_probs: jnp.ndarray,   # [T]
    new_log_probs: jnp.ndarray,   # [T]
    reward_to_go: jnp.ndarray,    # [T]
    clip_ratio: float = 0.2,
    reward_baseline: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute clipped policy gradient loss using reward-to-go.
    
    Pure function implementing PPO-style clipped policy loss with rewards.
    
    Args:
        old_log_probs: Log probabilities from old policy [T]
        new_log_probs: Log probabilities from new policy [T]
        reward_to_go: Reward-to-go values [T]
        clip_ratio: Clipping ratio for policy updates
        reward_baseline: Baseline to subtract from rewards for variance reduction
        
    Returns:
        Tuple of (policy_loss, kl_divergence, clipped_fraction)
    """
    # Use rewards as advantages (subtract baseline for variance reduction)
    advantages = reward_to_go - reward_baseline
    
    # Compute probability ratio
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    # Clipped surrogate objective
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Policy loss components
    policy_obj = ratio * advantages
    clipped_obj = clipped_ratio * advantages
    policy_loss = -jnp.mean(jnp.minimum(policy_obj, clipped_obj))

    # KL divergence approximation (first-order)
    kl_divergence = jnp.mean(log_ratio)

    # Fraction of updates that were clipped
    clipped_mask = jnp.abs(ratio - 1.0) > clip_ratio
    clipped_fraction = jnp.mean(clipped_mask.astype(jnp.float32))

    return policy_loss, kl_divergence, clipped_fraction


@jax.jit
def compute_entropy_loss(log_probs: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy regularization loss.
    
    Pure function for entropy computation.
    
    Args:
        log_probs: Log probabilities [T]
        
    Returns:
        Negative entropy (for minimization)
    """
    return -jnp.mean(log_probs)  # Negative because we minimize


# Main policy-only GRPO update function
def create_policy_only_grpo_update_fn(
    policy_fn: Callable,
    policy_optimizer: optax.GradientTransformation,
    config: PolicyOnlyGRPOConfig
) -> Callable:
    """Create a pure policy-only GRPO update function.
    
    Returns a pure function that performs one policy-only GRPO update step.
    
    Args:
        policy_fn: Policy network function
        policy_optimizer: Optax optimizer for policy
        config: Policy-only GRPO configuration
        
    Returns:
        Policy-only GRPO update function
    """

    def policy_only_grpo_update(
        policy_params: Any,
        policy_opt_state: Any,
        trajectory: PolicyTrajectory
    ) -> Tuple[Any, Any, PolicyUpdateResult]:
        """Perform one policy-only GRPO update step.
        
        Pure function that updates only policy parameters using rewards.
        
        Args:
            policy_params: Current policy parameters
            policy_opt_state: Policy optimizer state
            trajectory: Training trajectory data with rewards
            
        Returns:
            Tuple of (new_policy_params, new_policy_opt_state, update_result)
        """

        # Compute reward baseline for variance reduction
        reward_baseline = 0.0
        if config.use_reward_baseline:
            reward_baseline = compute_reward_baseline(trajectory.reward_to_go)

        # Compute new policy outputs and loss
        def policy_loss_fn(params):
            new_log_probs = policy_fn(params, trajectory.states, trajectory.actions)
            policy_loss, kl_div, clip_frac = compute_policy_loss_rewards(
                trajectory.log_probs, new_log_probs, trajectory.reward_to_go, 
                config.clip_ratio, reward_baseline
            )
            entropy_loss = compute_entropy_loss(new_log_probs)
            return policy_loss + config.entropy_coefficient * entropy_loss, (kl_div, clip_frac, entropy_loss)

        # Compute gradients
        (policy_loss, (kl_div, clip_frac, entropy_loss)), policy_grads = jax.value_and_grad(
            policy_loss_fn, has_aux=True
        )(policy_params)

        # Gradient clipping
        policy_grad_norm = optax.global_norm(policy_grads)

        if config.max_grad_norm > 0:
            # Clip gradients by global norm
            policy_grads = optax.clip_by_global_norm(config.max_grad_norm).update(
                policy_grads, None
            )[0]

        # Apply updates with debugging
        logger.debug("Policy-only GRPO: Computing policy updates...")
        policy_updates, new_policy_opt_state = policy_optimizer.update(
            policy_grads, policy_opt_state, policy_params
        )
        
        # DEBUG: Check update magnitudes
        policy_update_norm = optax.global_norm(policy_updates)
        logger.debug(f"Policy-only GRPO: Policy update norm: {float(policy_update_norm):.12f}")
        
        logger.debug("Policy-only GRPO: Applying policy updates...")
        new_policy_params = optax.apply_updates(policy_params, policy_updates)
        
        # DEBUG: Verify parameter change after apply_updates
        if isinstance(policy_params, dict) and isinstance(new_policy_params, dict):
            first_key = next(iter(policy_params.keys()))
            if hasattr(policy_params[first_key], 'flatten') and hasattr(new_policy_params[first_key], 'flatten'):
                old_param = float(policy_params[first_key].flatten()[0])
                new_param = float(new_policy_params[first_key].flatten()[0])
                actual_change = abs(new_param - old_param)
                logger.debug(f"Policy-only GRPO: Parameter change verification - old: {old_param:.12f}, new: {new_param:.12f}, change: {actual_change:.12f}")
                
                if actual_change < 1e-12:
                    logger.warning("🚨 Policy-only GRPO: optax.apply_updates did not change parameters!")
                else:
                    logger.info(f"✅ Policy-only GRPO: Parameters successfully updated by {actual_change:.12f}")

        # Create update result
        total_loss = policy_loss

        update_result = PolicyUpdateResult(
            policy_loss=float(policy_loss),
            entropy_loss=float(entropy_loss),
            total_loss=float(total_loss),
            kl_divergence=float(kl_div),
            policy_gradient_norm=float(policy_grad_norm),
            clipped_fraction=float(clip_frac),
            reward_baseline=float(reward_baseline)
        )

        return new_policy_params, new_policy_opt_state, update_result

    return policy_only_grpo_update


# Utility functions for trajectory processing
def create_policy_trajectory_from_experiences(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    log_probs: jnp.ndarray,
    dones: jnp.ndarray,
    config: PolicyOnlyGRPOConfig
) -> PolicyTrajectory:
    """Create a policy-only trajectory from experience data.
    
    Pure function for trajectory creation with reward-to-go computation.
    
    Args:
        states: State sequence [T, ...]
        actions: Action sequence [T, ...]
        rewards: Reward sequence [T]
        log_probs: Action log probabilities [T]
        dones: Episode termination flags [T]
        config: Policy-only GRPO configuration
        
    Returns:
        Complete PolicyTrajectory with computed reward-to-go
    """
    # Compute reward-to-go
    reward_to_go = compute_reward_to_go(rewards, dones, config.discount_factor)

    # Normalize rewards if requested
    if config.normalize_rewards:
        reward_to_go = normalize_rewards(reward_to_go)

    return PolicyTrajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        dones=dones,
        reward_to_go=reward_to_go
    )


def validate_policy_only_grpo_config(config: PolicyOnlyGRPOConfig) -> None:
    """Validate policy-only GRPO configuration parameters.
    
    Pure validation function following "fail fast" principle.
    
    Args:
        config: Policy-only GRPO configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")

    if not 0 <= config.discount_factor <= 1:
        raise ValueError(f"discount_factor must be in [0,1], got {config.discount_factor}")

    if config.clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be positive, got {config.clip_ratio}")

    if config.entropy_coefficient < 0:
        raise ValueError(f"entropy_coefficient must be non-negative, got {config.entropy_coefficient}")


# Factory functions for common configurations
def create_default_policy_only_grpo_config() -> PolicyOnlyGRPOConfig:
    """Create default policy-only GRPO configuration."""
    return PolicyOnlyGRPOConfig()


def create_high_performance_policy_only_grpo_config() -> PolicyOnlyGRPOConfig:
    """Create high-performance policy-only GRPO configuration."""
    return PolicyOnlyGRPOConfig(
        learning_rate=1e-4,
        clip_ratio=0.1,
        entropy_coefficient=0.005,
        max_grad_norm=1.0,
        target_kl=0.005
    )


def create_exploration_policy_only_grpo_config() -> PolicyOnlyGRPOConfig:
    """Create exploration-focused policy-only GRPO configuration."""
    return PolicyOnlyGRPOConfig(
        learning_rate=5e-4,
        entropy_coefficient=0.02,
        clip_ratio=0.3,
        normalize_rewards=True
    )