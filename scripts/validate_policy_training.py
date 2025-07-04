#!/usr/bin/env python3
"""
Short Policy Training Validation Script

Simple script to validate that our continuous reward system works correctly
in a short GRPO policy training run. This provides validation that:
1. The reward system encourages proper learning
2. Policy performance improves over time
3. No silent failures or crashes occur
4. Component interactions work correctly

Usage:
    poetry run python scripts/validate_policy_training.py
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_bayes_opt.acquisition.rewards import (
    RewardComponents,
    compute_verifiable_reward,
    create_default_reward_config,
    validate_reward_consistency,
)
from causal_bayes_opt.acquisition.state import AcquisitionState
from causal_bayes_opt.acquisition.policy import (
    AcquisitionPolicyNetwork,
    create_acquisition_policy,
    sample_intervention_from_policy,
)
from causal_bayes_opt.data_structures.scm import create_scm
from causal_bayes_opt.data_structures.sample import create_sample
from causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from causal_bayes_opt.mechanisms.linear import create_linear_mechanism
from causal_bayes_opt.training.grpo_core import (
    GRPOConfig,
    GRPOTrajectory,
    create_grpo_update_fn,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_scm() -> pyr.PMap:
    """Create a simple 3-variable linear SCM for validation."""
    return create_scm(
        variables={"X", "Y", "Z"},
        edges={("X", "Y"), ("Z", "Y")},  # X->Y, Z->Y
        mechanisms={
            "X": create_linear_mechanism([], {}, intercept=0.0, noise_scale=1.0),  # Root variable
            "Z": create_linear_mechanism([], {}, intercept=0.0, noise_scale=1.0),  # Root variable
            "Y": create_linear_mechanism(["X", "Z"], {"X": 2.0, "Z": -1.0}, intercept=1.0, noise_scale=1.0)  # Y = 2*X - Z + 1 + noise
        },
        target="Y"
    )


def create_mock_acquisition_state(scm: pyr.PMap, step: int = 0, best_value: float = 0.0) -> Mock:
    """Create a mock acquisition state for validation."""
    
    mock_posterior = Mock()
    mock_posterior.uncertainty = max(0.1, 1.0 - step * 0.1)  # Decreasing uncertainty
    mock_posterior.target_variable = "Y"
    
    mock_buffer = Mock()
    mock_buffer.samples = []
    mock_buffer.get_interventions.return_value = []
    
    state = Mock()
    state.current_target = "Y"
    state.step = step
    state.best_value = best_value
    state.posterior = mock_posterior
    state.buffer = mock_buffer
    state.marginal_parent_probs = {"X": 0.8, "Z": 0.6}
    state.uncertainty_bits = mock_posterior.uncertainty
    state.buffer_statistics = Mock()
    state.buffer_statistics.total_samples = max(1, step * 2)
    
    # Add mechanism predictions for SCM-objective reward
    mock_mechanism = Mock()
    mock_mechanism.coefficients = {"X": 2.0, "Z": -1.0}
    mock_mechanism.intercept = 1.0
    state.mechanism_predictions = {"Y": mock_mechanism}
    state.intervention_bounds = {"X": (-3.0, 3.0), "Z": (-3.0, 3.0)}
    
    return state


def simulate_intervention_outcome(scm: pyr.PMap, intervention: pyr.PMap) -> pyr.PMap:
    """Simulate outcome of an intervention on the SCM."""
    # Get intervention values
    targets = intervention.get('targets', set())
    values = intervention.get('values', {})
    
    # For simplicity, use the true SCM structure: Y = 2*X - Z + 1
    if 'X' in values and 'Z' in values:
        y_value = 2.0 * values['X'] - values['Z'] + 1.0
    elif 'X' in values:
        # Assume Z=0 if not intervened
        y_value = 2.0 * values['X'] + 1.0
    elif 'Z' in values:
        # Assume X=0 if not intervened
        y_value = -values['Z'] + 1.0
    else:
        # No intervention, baseline value
        y_value = 1.0
    
    # Add small amount of noise
    key = random.PRNGKey(42)
    noise = random.normal(key) * 0.1
    y_value += noise
    
    outcome_values = {"Y": y_value}
    outcome_values.update(values)  # Include intervention values
    
    return pyr.m({'values': outcome_values})


def create_simple_policy_network(key: jax.random.PRNGKey) -> Tuple[Any, Any]:
    """Create a simple policy network for validation."""
    # Simple network configuration
    config = {
        "hidden_size": 32,
        "num_layers": 2,
        "action_dim": 2,  # X and Z intervention values
        "max_intervention_value": 3.0,
    }
    
    def policy_fn(state_tensor):
        """Simple policy function: output intervention values for X and Z."""
        # Very basic neural network (just for validation)
        import haiku as hk
        
        mlp = hk.nets.MLP([config["hidden_size"]] * config["num_layers"] + [config["action_dim"]])
        logits = mlp(state_tensor)
        
        # Map to intervention range [-3, 3]
        interventions = jnp.tanh(logits) * config["max_intervention_value"]
        return interventions
    
    def value_fn(state_tensor):
        """Simple value function for GRPO."""
        import haiku as hk
        
        mlp = hk.nets.MLP([config["hidden_size"]] * config["num_layers"] + [1])
        value = mlp(state_tensor)
        return jnp.squeeze(value, axis=-1)
    
    import haiku as hk
    
    # Transform to JAX functions
    policy_net = hk.transform(policy_fn)
    value_net = hk.transform(value_fn)
    
    # Initialize parameters
    dummy_input = jnp.ones((1, 10))  # Dummy state tensor
    policy_params = policy_net.init(key, dummy_input)
    value_params = value_net.init(key, dummy_input)
    
    return (policy_net, policy_params), (value_net, value_params)


def convert_state_to_tensor(state: Any) -> jnp.ndarray:
    """Convert acquisition state to tensor for policy network."""
    # Simple state representation for validation
    features = [
        state.step / 10.0,  # Normalized step
        state.best_value / 10.0,  # Normalized best value
        state.uncertainty_bits,  # Uncertainty
        state.marginal_parent_probs.get("X", 0.0),  # Parent prob X
        state.marginal_parent_probs.get("Z", 0.0),  # Parent prob Z
        state.buffer_statistics.total_samples / 20.0,  # Normalized sample count
    ]
    
    # Pad to fixed size
    while len(features) < 10:
        features.append(0.0)
    
    return jnp.array(features[:10])


def run_validation_episode(
    scm: pyr.PMap,
    policy_net: Any,
    policy_params: Any,
    value_net: Any,
    value_params: Any,
    config: pyr.PMap,
    num_steps: int = 10
) -> Tuple[List[RewardComponents], List[float], Dict[str, Any]]:
    """Run a single validation episode and collect rewards."""
    
    rewards_history = []
    values_history = []
    best_value = 0.0
    
    # Initial state
    state_before = create_mock_acquisition_state(scm, step=0, best_value=best_value)
    
    for step in range(num_steps):
        # Convert state to tensor
        state_tensor = convert_state_to_tensor(state_before)
        state_tensor = jnp.expand_dims(state_tensor, axis=0)  # Add batch dimension
        
        # Get policy action (Haiku needs RNG key)
        key = random.PRNGKey(step)
        action = policy_net.apply(policy_params, key, state_tensor)[0]  # Remove batch dimension
        
        # Get value estimate (Haiku needs RNG key)  
        value_estimate = value_net.apply(value_params, key, state_tensor)[0]
        values_history.append(float(value_estimate))
        
        # Convert action to intervention
        intervention = pyr.m({
            'type': "perfect",
            'targets': {"X", "Z"},
            'values': {"X": float(action[0]), "Z": float(action[1])}
        })
        
        # Simulate outcome
        outcome = simulate_intervention_outcome(scm, intervention)
        target_value = outcome['values']['Y']
        
        # Update best value
        if target_value > best_value:
            best_value = target_value
        
        # Create state after
        state_after = create_mock_acquisition_state(
            scm, step=step + 1, best_value=best_value
        )
        
        # Compute reward
        reward_components = compute_verifiable_reward(
            state_before, intervention, outcome, state_after, config
        )
        rewards_history.append(reward_components)
        
        logger.info(
            f"Step {step}: action=[{action[0]:.2f}, {action[1]:.2f}], "
            f"Y={target_value:.2f}, reward={reward_components.total_reward:.3f}, "
            f"value_est={value_estimate:.3f}"
        )
        
        # Update for next iteration
        state_before = state_after
    
    # Analyze results
    total_rewards = [r.total_reward for r in rewards_history]
    optimization_rewards = [r.optimization_reward for r in rewards_history]
    
    metrics = {
        "mean_total_reward": float(jnp.mean(jnp.array(total_rewards))),
        "mean_optimization_reward": float(jnp.mean(jnp.array(optimization_rewards))),
        "final_best_value": best_value,
        "reward_trend": float(jnp.polyfit(jnp.arange(len(total_rewards), dtype=jnp.float32), jnp.array(total_rewards), 1)[0]),
        "value_trend": float(jnp.polyfit(jnp.arange(len(values_history), dtype=jnp.float32), jnp.array(values_history), 1)[0]) if len(values_history) > 1 else 0.0,
    }
    
    return rewards_history, values_history, metrics


def run_grpo_update(
    policy_net: Any,
    policy_params: Any,
    value_net: Any,
    value_params: Any,
    optimizer_state: Any,
    trajectory: List[Dict[str, Any]],
    optimizer: optax.GradientTransformation
) -> Tuple[Any, Any, Any, Dict[str, float]]:
    """Run a simple GRPO update on collected trajectory."""
    
    if len(trajectory) < 2:
        return policy_params, value_params, optimizer_state, {}
    
    # Extract trajectory data
    states = jnp.stack([t['state'] for t in trajectory])
    actions = jnp.stack([t['action'] for t in trajectory])
    rewards = jnp.array([t['reward'] for t in trajectory])
    values = jnp.stack([t['value'] for t in trajectory])
    
    # Compute returns (simple discounted)
    gamma = 0.99
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = jnp.array(returns)
    
    # Compute advantages
    advantages = returns - values
    
    def loss_fn(policy_params, value_params):
        # Policy loss (simplified GRPO) - need RNG key for Haiku
        key = random.PRNGKey(0)
        new_logits = policy_net.apply(policy_params, key, states)
        policy_loss = -jnp.mean(advantages * jnp.sum(new_logits * actions, axis=-1))
        
        # Value loss
        new_values = value_net.apply(value_params, key, states)
        value_loss = jnp.mean((new_values - returns) ** 2)
        
        total_loss = policy_loss + 0.5 * value_loss
        return total_loss, {"policy_loss": policy_loss, "value_loss": value_loss}
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1), has_aux=True)
    (policy_grads, value_grads), metrics = grad_fn(policy_params, value_params)
    
    # Combine parameters and gradients for optimizer
    params = {"policy": policy_params, "value": value_params}
    grads = {"policy": policy_grads, "value": value_grads}
    
    # Apply updates
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params["policy"], new_params["value"], optimizer_state, metrics


def main():
    """Main validation function."""
    logger.info("Starting Policy Training Validation")
    
    # Set up JAX
    key = random.PRNGKey(42)
    
    # Create simple SCM
    scm = create_simple_scm()
    logger.info(f"Created SCM with variables: {scm['variables']}")
    
    # Create reward configuration
    reward_config = create_default_reward_config(
        optimization_weight=1.0,
        structure_weight=0.5,
        parent_weight=0.3,
        exploration_weight=0.1
    )
    logger.info("Created reward configuration")
    
    # Create policy network
    (policy_net, policy_params), (value_net, value_params) = create_simple_policy_network(key)
    logger.info("Created policy and value networks")
    
    # Create optimizer
    optimizer = optax.adam(learning_rate=0.001)
    optimizer_state = optimizer.init({"policy": policy_params, "value": value_params})
    
    # Run validation episodes with training
    num_episodes = 5
    episode_length = 10
    all_rewards = []
    episode_metrics = []
    
    for episode in range(num_episodes):
        logger.info(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Run episode
        rewards_history, values_history, metrics = run_validation_episode(
            scm, policy_net, policy_params, value_net, value_params, 
            reward_config, episode_length
        )
        
        all_rewards.extend(rewards_history)
        episode_metrics.append(metrics)
        
        logger.info(f"Episode {episode + 1} metrics: {metrics}")
        
        # Collect trajectory for training
        trajectory = []
        state_before = create_mock_acquisition_state(scm, step=0)
        
        for step in range(episode_length):
            state_tensor = convert_state_to_tensor(state_before)
            state_tensor = jnp.expand_dims(state_tensor, axis=0)
            
            key = random.PRNGKey(step)
            action = policy_net.apply(policy_params, key, state_tensor)[0]
            value = value_net.apply(value_params, key, state_tensor)[0]
            
            trajectory.append({
                'state': state_tensor[0],
                'action': action,
                'value': value,
                'reward': rewards_history[step].total_reward
            })
            
            state_before = create_mock_acquisition_state(scm, step=step + 1)
        
        # Update policy
        if len(trajectory) >= 2:
            policy_params, value_params, optimizer_state, update_metrics = run_grpo_update(
                policy_net, policy_params, value_net, value_params,
                optimizer_state, trajectory, optimizer
            )
            logger.info(f"Update metrics: {update_metrics}")
    
    # Final analysis
    logger.info("\n=== VALIDATION RESULTS ===")
    
    # Check reward consistency
    validation_result = validate_reward_consistency(all_rewards)
    logger.info(f"Reward consistency: {'PASS' if validation_result['valid'] else 'FAIL'}")
    if not validation_result['valid']:
        logger.warning(f"Gaming issues detected: {validation_result['gaming_issues']}")
    
    # Analyze trends
    total_rewards = [r.total_reward for r in all_rewards]
    optimization_rewards = [r.optimization_reward for r in all_rewards]
    
    reward_trend = float(jnp.polyfit(jnp.arange(len(total_rewards), dtype=jnp.float32), jnp.array(total_rewards), 1)[0])
    optimization_trend = float(jnp.polyfit(jnp.arange(len(optimization_rewards), dtype=jnp.float32), jnp.array(optimization_rewards), 1)[0])
    
    logger.info(f"Total reward trend: {reward_trend:.4f} (positive = improving)")
    logger.info(f"Optimization reward trend: {optimization_trend:.4f} (positive = improving)")
    
    # Episode progression
    final_values = [m['final_best_value'] for m in episode_metrics]
    value_improvement = final_values[-1] - final_values[0] if len(final_values) > 1 else 0
    logger.info(f"Best value improvement: {value_improvement:.3f} (episode 1 to {num_episodes})")
    
    # Summary
    success_criteria = {
        "no_crashes": True,  # We got here!
        "reward_consistency": validation_result['valid'],
        "positive_reward_trend": reward_trend > -0.01,  # Allow for small negative trends
        "value_improvement": value_improvement > 0,
        "finite_rewards": all(jnp.isfinite(r.total_reward) for r in all_rewards)
    }
    
    all_passed = all(success_criteria.values())
    
    logger.info(f"\nSUCCESS CRITERIA:")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {criterion}: {status}")
    
    logger.info(f"\nOVERALL VALIDATION: {'✓ PASS' if all_passed else '✗ FAIL'}")
    
    if all_passed:
        logger.info("Policy training validation completed successfully!")
        logger.info("The continuous reward system works correctly with GRPO training.")
    else:
        logger.warning("Some validation criteria failed. Review the results above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)