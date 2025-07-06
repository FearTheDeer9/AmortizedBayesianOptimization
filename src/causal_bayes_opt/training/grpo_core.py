"""⚠️  DEPRECATED: PPO-contaminated GRPO implementation.

This module contains an INCORRECT implementation that mixes PPO and GRPO concepts.

🚫 PROBLEMS WITH THIS IMPLEMENTATION:
- Contains value functions (GRPO should be policy-only)
- Uses GAE (Generalized Advantage Estimation) 
- Has value loss computation
- Implements PPO-style trajectories with returns/advantages

✅ USE THE CORRECT IMPLEMENTATION INSTEAD:
    from causal_bayes_opt.acquisition.grpo import (
        GRPOConfig, GRPOUpdate, create_grpo_trainer
    )

The correct implementation in acquisition/grpo.py:
- Policy-only updates (no value functions)
- Uses group mean as baseline 
- Follows DeepSeek GRPO paper correctly
- Includes sample reuse and open-r1 enhancements

⚠️  THIS FILE WILL BE REMOVED IN A FUTURE RELEASE
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import optax

logger = logging.getLogger(__name__)

# Issue deprecation warning when module is imported
warnings.warn(
    "⚠️  DEPRECATED: Using PPO-contaminated GRPO implementation! "
    "Use 'from causal_bayes_opt.acquisition.grpo import ...' instead. "
    "This module mixes PPO concepts (value functions, GAE) with GRPO and "
    "will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)


@dataclass(frozen=True)
class GRPOConfig:
    """Configuration for GRPO algorithm.
    
    Args:
        learning_rate: Policy learning rate
        value_learning_rate: Value function learning rate
        discount_factor: Reward discount factor (gamma)
        gae_lambda: GAE lambda parameter for advantage estimation
        clip_ratio: PPO-style clipping ratio for policy updates
        entropy_coefficient: Entropy regularization coefficient
        value_loss_coefficient: Value function loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        target_kl: Target KL divergence for early stopping
        normalize_advantages: Whether to normalize advantages
        use_gae: Whether to use Generalized Advantage Estimation
    """
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    normalize_advantages: bool = True
    use_gae: bool = True


@dataclass(frozen=True)
class GRPOTrajectory:
    """Immutable trajectory data for GRPO training.
    
    Args:
        states: Sequence of states [T, ...]
        actions: Sequence of actions [T, ...]
        rewards: Sequence of rewards [T]
        values: Sequence of value estimates [T]
        log_probs: Sequence of action log probabilities [T]
        dones: Sequence of episode termination flags [T]
        advantages: Computed advantages [T] (computed separately)
        returns: Computed returns [T] (computed separately)
    """
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    values: jnp.ndarray
    log_probs: jnp.ndarray
    dones: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


@dataclass(frozen=True)
class GRPOUpdateResult:
    """Results from a GRPO policy update.
    
    Args:
        policy_loss: Policy gradient loss
        value_loss: Value function loss
        entropy_loss: Entropy regularization loss
        total_loss: Combined total loss
        kl_divergence: KL divergence between old and new policy
        policy_gradient_norm: Norm of policy gradients
        value_gradient_norm: Norm of value gradients
        clipped_fraction: Fraction of policy updates that were clipped
        explained_variance: Explained variance of value function
    """
    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    kl_divergence: float
    policy_gradient_norm: float
    value_gradient_norm: float
    clipped_fraction: float
    explained_variance: float


# Pure functions for advantage estimation
@jax.jit
def compute_gae_advantages(
    rewards: jnp.ndarray,      # [T]
    values: jnp.ndarray,       # [T+1] (includes bootstrap value)
    dones: jnp.ndarray,        # [T]
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.
    
    Pure function implementing GAE algorithm for advantage estimation.
    
    Args:
        rewards: Rewards for each timestep [T]
        values: Value estimates [T+1] including bootstrap value
        dones: Episode termination flags [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages [T], returns [T])
    """
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)

    # GAE computation with scan for efficiency
    def gae_step(carry, inputs):
        gae, next_value = carry
        reward, value, done, next_non_terminal = inputs

        delta = reward + gamma * next_value * next_non_terminal - value
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        return (gae, value), gae

    # Prepare inputs for scan (reverse order for GAE)
    scan_inputs = (
        rewards[::-1],
        values[:-1][::-1],
        dones[::-1],
        (1.0 - dones[::-1])  # next_non_terminal
    )

    initial_carry = (0.0, values[-1])  # (initial_gae, bootstrap_value)
    _, advantages_rev = jax.lax.scan(gae_step, initial_carry, scan_inputs)

    advantages = advantages_rev[::-1]
    returns = advantages + values[:-1]

    return advantages, returns


@jax.jit
def compute_simple_advantages(
    rewards: jnp.ndarray,      # [T]
    values: jnp.ndarray,       # [T+1]
    dones: jnp.ndarray,        # [T]
    gamma: float = 0.99
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute simple advantage estimates without GAE.
    
    Pure function for basic advantage estimation.
    
    Args:
        rewards: Rewards for each timestep [T]
        values: Value estimates [T+1] including bootstrap value  
        dones: Episode termination flags [T]
        gamma: Discount factor
        
    Returns:
        Tuple of (advantages [T], returns [T])
    """
    next_values = values[1:]
    next_non_terminal = 1.0 - dones

    # Simple TD(0) advantage: r + gamma * V(s') - V(s)
    advantages = rewards + gamma * next_values * next_non_terminal - values[:-1]

    # Compute returns via discounted reward sum
    def return_step(carry, inputs):
        ret = carry
        reward, done = inputs
        ret = reward + gamma * ret * (1.0 - done)
        return ret, ret

    initial_return = 0.0
    scan_inputs = (rewards[::-1], dones[::-1])
    _, returns_rev = jax.lax.scan(return_step, initial_return, scan_inputs)
    returns = returns_rev[::-1]

    return advantages, returns


@jax.jit
def normalize_advantages(advantages: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Normalize advantages to have zero mean and unit variance.
    
    Pure function for advantage normalization.
    
    Args:
        advantages: Raw advantage estimates [T]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantages [T]
    """
    mean = jnp.mean(advantages)
    std = jnp.std(advantages)
    return (advantages - mean) / (std + eps)


# Pure functions for policy updates
@jax.jit
def compute_policy_loss(
    old_log_probs: jnp.ndarray,   # [T]
    new_log_probs: jnp.ndarray,   # [T]
    advantages: jnp.ndarray,      # [T]
    clip_ratio: float = 0.2
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute clipped policy gradient loss.
    
    Pure function implementing PPO-style clipped policy loss.
    
    Args:
        old_log_probs: Log probabilities from old policy [T]
        new_log_probs: Log probabilities from new policy [T]
        advantages: Advantage estimates [T]
        clip_ratio: Clipping ratio for policy updates
        
    Returns:
        Tuple of (policy_loss, kl_divergence, clipped_fraction)
    """
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


def compute_value_loss(
    predicted_values: jnp.ndarray,  # [T]
    target_returns: jnp.ndarray,    # [T]
    old_values: jnp.ndarray,        # [T]
    clip_ratio: float = 0.2,
    use_clipping: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute value function loss with optional clipping.
    
    Pure function for value function loss computation.
    
    Args:
        predicted_values: Current value predictions [T]
        target_returns: Target return values [T]
        old_values: Previous value predictions [T]
        clip_ratio: Clipping ratio for value updates
        use_clipping: Whether to use value clipping
        
    Returns:
        Tuple of (value_loss, explained_variance)
    """
    value_errors = target_returns - predicted_values

    if use_clipping:
        # Clipped value loss (similar to PPO)
        clipped_values = old_values + jnp.clip(
            predicted_values - old_values, -clip_ratio, clip_ratio
        )
        clipped_errors = target_returns - clipped_values

        value_loss = jnp.mean(jnp.maximum(
            value_errors ** 2,
            clipped_errors ** 2
        ))
    else:
        # Simple MSE loss
        value_loss = jnp.mean(value_errors ** 2)

    # Explained variance
    returns_var = jnp.var(target_returns)
    explained_var = 1.0 - jnp.var(value_errors) / (returns_var + 1e-8)

    return value_loss, explained_var


@jax.jit
def compute_value_loss_jit(
    predicted_values: jnp.ndarray,  # [T]
    target_returns: jnp.ndarray,    # [T]
    old_values: jnp.ndarray,        # [T]
    clip_ratio: float = 0.2
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compiled value loss computation without clipping.
    
    Pure function for value function loss computation.
    
    Args:
        predicted_values: Current value predictions [T]
        target_returns: Target return values [T]
        old_values: Previous value predictions [T]
        clip_ratio: Clipping ratio for value updates
        
    Returns:
        Tuple of (value_loss, explained_variance)
    """
    value_errors = target_returns - predicted_values

    # Simple MSE loss (no conditional for JAX compatibility)
    value_loss = jnp.mean(value_errors ** 2)

    # Explained variance
    returns_var = jnp.var(target_returns)
    explained_var = 1.0 - jnp.var(value_errors) / (returns_var + 1e-8)

    return value_loss, explained_var


@jax.jit
def compute_value_loss_clipped_jit(
    predicted_values: jnp.ndarray,  # [T]
    target_returns: jnp.ndarray,    # [T]
    old_values: jnp.ndarray,        # [T]
    clip_ratio: float = 0.2
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX-compiled value loss computation with clipping.
    
    Pure function for value function loss computation.
    
    Args:
        predicted_values: Current value predictions [T]
        target_returns: Target return values [T]
        old_values: Previous value predictions [T]
        clip_ratio: Clipping ratio for value updates
        
    Returns:
        Tuple of (value_loss, explained_variance)
    """
    value_errors = target_returns - predicted_values

    # Clipped value loss (similar to PPO)
    clipped_values = old_values + jnp.clip(
        predicted_values - old_values, -clip_ratio, clip_ratio
    )
    clipped_errors = target_returns - clipped_values

    value_loss = jnp.mean(jnp.maximum(
        value_errors ** 2,
        clipped_errors ** 2
    ))

    # Explained variance
    returns_var = jnp.var(target_returns)
    explained_var = 1.0 - jnp.var(value_errors) / (returns_var + 1e-8)

    return value_loss, explained_var


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


# Main GRPO update function
def create_grpo_update_fn(
    policy_fn: Callable,
    value_fn: Callable,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    config: GRPOConfig
) -> Callable:
    """Create a pure GRPO update function.
    
    Returns a pure function that performs one GRPO update step.
    
    Args:
        policy_fn: Policy network function
        value_fn: Value network function  
        policy_optimizer: Optax optimizer for policy
        value_optimizer: Optax optimizer for value function
        config: GRPO configuration
        
    Returns:
        GRPO update function
    """

    def grpo_update(
        policy_params: Any,
        value_params: Any,
        policy_opt_state: Any,
        value_opt_state: Any,
        trajectory: GRPOTrajectory
    ) -> Tuple[Any, Any, Any, Any, GRPOUpdateResult]:
        """Perform one GRPO update step.
        
        Pure function that updates policy and value function parameters.
        
        Args:
            policy_params: Current policy parameters
            value_params: Current value parameters
            policy_opt_state: Policy optimizer state
            value_opt_state: Value optimizer state
            trajectory: Training trajectory data
            
        Returns:
            Tuple of (new_policy_params, new_value_params, 
                     new_policy_opt_state, new_value_opt_state, update_result)
        """

        # Compute new policy outputs
        def policy_loss_fn(params):
            new_log_probs = policy_fn(params, trajectory.states, trajectory.actions)
            policy_loss, kl_div, clip_frac = compute_policy_loss(
                trajectory.log_probs, new_log_probs, trajectory.advantages, config.clip_ratio
            )
            entropy_loss = compute_entropy_loss(new_log_probs)
            return policy_loss + config.entropy_coefficient * entropy_loss, (kl_div, clip_frac, entropy_loss)

        # Compute new value outputs
        def value_loss_fn(params):
            new_values = value_fn(params, trajectory.states)
            value_loss, explained_var = compute_value_loss_jit(
                new_values, trajectory.returns, trajectory.values, config.clip_ratio
            )
            return value_loss, explained_var

        # Compute gradients
        (policy_loss, (kl_div, clip_frac, entropy_loss)), policy_grads = jax.value_and_grad(
            policy_loss_fn, has_aux=True
        )(policy_params)

        (value_loss, explained_var), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(value_params)

        # Gradient clipping
        policy_grad_norm = optax.global_norm(policy_grads)
        value_grad_norm = optax.global_norm(value_grads)

        if config.max_grad_norm > 0:
            # Clip gradients by global norm
            policy_grads = optax.clip_by_global_norm(config.max_grad_norm).update(
                policy_grads, None
            )[0]
            value_grads = optax.clip_by_global_norm(config.max_grad_norm).update(
                value_grads, None
            )[0]

        # Apply updates with debugging
        logger.debug("GRPO: Computing policy updates...")
        policy_updates, new_policy_opt_state = policy_optimizer.update(
            policy_grads, policy_opt_state, policy_params
        )
        
        # DEBUG: Check update magnitudes
        policy_update_norm = optax.global_norm(policy_updates)
        logger.debug(f"GRPO: Policy update norm: {float(policy_update_norm):.12f}")
        
        # DEBUG: Log sample updates
        if isinstance(policy_updates, dict):
            first_key = next(iter(policy_updates.keys()))
            if hasattr(policy_updates[first_key], 'flatten'):
                sample_update = float(policy_updates[first_key].flatten()[0])
                logger.debug(f"GRPO: Sample policy update value: {sample_update:.12f}")
        
        logger.debug("GRPO: Applying policy updates...")
        new_policy_params = optax.apply_updates(policy_params, policy_updates)
        
        # DEBUG: Verify parameter change after apply_updates
        if isinstance(policy_params, dict) and isinstance(new_policy_params, dict):
            first_key = next(iter(policy_params.keys()))
            if hasattr(policy_params[first_key], 'flatten') and hasattr(new_policy_params[first_key], 'flatten'):
                old_param = float(policy_params[first_key].flatten()[0])
                new_param = float(new_policy_params[first_key].flatten()[0])
                actual_change = abs(new_param - old_param)
                logger.debug(f"GRPO: Parameter change verification - old: {old_param:.12f}, new: {new_param:.12f}, change: {actual_change:.12f}")
                
                if actual_change < 1e-12:
                    logger.warning("🚨 GRPO: optax.apply_updates did not change parameters!")
                else:
                    logger.info(f"✅ GRPO: Parameters successfully updated by {actual_change:.12f}")

        logger.debug("GRPO: Computing value updates...")
        value_updates, new_value_opt_state = value_optimizer.update(
            value_grads, value_opt_state, value_params
        )
        new_value_params = optax.apply_updates(value_params, value_updates)

        # Create update result
        total_loss = policy_loss + config.value_loss_coefficient * value_loss

        update_result = GRPOUpdateResult(
            policy_loss=float(policy_loss),
            value_loss=float(value_loss),
            entropy_loss=float(entropy_loss),
            total_loss=float(total_loss),
            kl_divergence=float(kl_div),
            policy_gradient_norm=float(policy_grad_norm),
            value_gradient_norm=float(value_grad_norm),
            clipped_fraction=float(clip_frac),
            explained_variance=float(explained_var)
        )

        return (new_policy_params, new_value_params,
                new_policy_opt_state, new_value_opt_state, update_result)

    return grpo_update


# Utility functions for trajectory processing
def create_trajectory_from_experiences(
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    log_probs: jnp.ndarray,
    dones: jnp.ndarray,
    bootstrap_value: float,
    config: GRPOConfig
) -> GRPOTrajectory:
    """Create a GRPO trajectory from experience data.
    
    Pure function for trajectory creation with advantage computation.
    
    Args:
        states: State sequence [T, ...]
        actions: Action sequence [T, ...]
        rewards: Reward sequence [T]
        values: Value estimates [T]
        log_probs: Action log probabilities [T]
        dones: Episode termination flags [T]
        bootstrap_value: Value estimate for final state
        config: GRPO configuration
        
    Returns:
        Complete GRPOTrajectory with computed advantages and returns
    """
    # Append bootstrap value for advantage computation
    extended_values = jnp.concatenate([values, jnp.array([bootstrap_value])])

    # Compute advantages based on configuration
    if config.use_gae:
        advantages, returns = compute_gae_advantages(
            rewards, extended_values, dones, config.discount_factor, config.gae_lambda
        )
    else:
        advantages, returns = compute_simple_advantages(
            rewards, extended_values, dones, config.discount_factor
        )

    # Normalize advantages if requested
    if config.normalize_advantages:
        advantages = normalize_advantages(advantages)

    return GRPOTrajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        values=values,
        log_probs=log_probs,
        dones=dones,
        advantages=advantages,
        returns=returns
    )


def validate_grpo_config(config: GRPOConfig) -> None:
    """Validate GRPO configuration parameters.
    
    Pure validation function following "fail fast" principle.
    
    Args:
        config: GRPO configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")

    if config.value_learning_rate <= 0:
        raise ValueError(f"value_learning_rate must be positive, got {config.value_learning_rate}")

    if not 0 <= config.discount_factor <= 1:
        raise ValueError(f"discount_factor must be in [0,1], got {config.discount_factor}")

    if not 0 <= config.gae_lambda <= 1:
        raise ValueError(f"gae_lambda must be in [0,1], got {config.gae_lambda}")

    if config.clip_ratio <= 0:
        raise ValueError(f"clip_ratio must be positive, got {config.clip_ratio}")

    if config.entropy_coefficient < 0:
        raise ValueError(f"entropy_coefficient must be non-negative, got {config.entropy_coefficient}")

    if config.value_loss_coefficient < 0:
        raise ValueError(f"value_loss_coefficient must be non-negative, got {config.value_loss_coefficient}")


# Factory functions for common configurations
def create_default_grpo_config() -> GRPOConfig:
    """Create default GRPO configuration."""
    return GRPOConfig()


def create_high_performance_grpo_config() -> GRPOConfig:
    """Create high-performance GRPO configuration."""
    return GRPOConfig(
        learning_rate=1e-4,
        value_learning_rate=5e-4,
        clip_ratio=0.1,
        entropy_coefficient=0.005,
        max_grad_norm=1.0,
        target_kl=0.005
    )


def create_exploration_grpo_config() -> GRPOConfig:
    """Create exploration-focused GRPO configuration."""
    return GRPOConfig(
        learning_rate=5e-4,
        entropy_coefficient=0.02,
        gae_lambda=0.9,
        clip_ratio=0.3,
        normalize_advantages=True
    )
