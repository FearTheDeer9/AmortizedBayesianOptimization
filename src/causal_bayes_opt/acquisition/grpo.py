"""Group Relative Policy Optimization (GRPO) for causal intervention selection.

GRPO is well-suited for our multi-objective setting because:
- Group-based advantages handle multi-objective variance well  
- NO value network needed (key GRPO innovation)
- Uses group mean as baseline instead of learned value function
- More stable training with conflicting reward components

Based on DeepSeek's GRPO paper: https://arxiv.org/abs/2402.03300
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax
import pyrsistent as pyr

from ..environments.sampling import sample_with_intervention
from ..interventions.handlers import create_perfect_intervention
from .policy import compute_action_log_probability, sample_intervention_from_policy
from .rewards import compute_verifiable_reward
from .state import AcquisitionState
from .services import create_acquisition_state
from .trajectory import TrajectoryBuffer


@dataclass
class GRPOConfig:
    """Configuration for GRPO algorithm following DeepSeek recommendations."""
    
    # Group sampling (DeepSeek uses 64-256, start with 64)
    group_size: int = 64
    
    # Loss coefficients (no value loss in pure GRPO!)
    clip_ratio: float = 0.2
    entropy_coeff: float = 0.01
    kl_penalty_coeff: float = 0.0  # Updated default based on recent findings
    
    # Optimization
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    
    # Open-r1 enhancements
    num_iterations: int = 4  # Sample reuse iterations
    scale_rewards: bool = True  # Configurable advantage scaling


@dataclass  
class GRPOUpdate:
    """Results from a GRPO update step."""
    
    policy_loss: float
    entropy_loss: float 
    kl_penalty: float
    total_loss: float
    grad_norm: float
    
    # Group statistics (key GRPO diagnostics)
    group_baseline: float
    mean_reward: float
    reward_std: float
    mean_advantage: float
    advantage_std: float
    
    # Policy diagnostics
    mean_entropy: float
    approx_kl: float


def create_grpo_trainer(
    policy_network: Any,
    config: GRPOConfig
) -> Tuple[Callable, Callable]:
    """
    Create GRPO training infrastructure.
    
    GRPO eliminates value networks and uses group statistics as baseline.
    
    Returns:
        (grpo_update_step, optimizer_init) tuple
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate)
    )
    
    @jax.jit
    def grpo_update_step(
        params: Any,
        opt_state: Any,
        batch_data: Dict[str, Any]
    ) -> Tuple[Any, Any, GRPOUpdate]:
        """Single GRPO update step - no value network needed."""
        
        def loss_fn(params):
            return _compute_grpo_loss(params, batch_data, policy_network, config)
        
        # Compute loss and gradients
        (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Apply updates
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads)
        
        return new_params, new_opt_state, GRPOUpdate(
            policy_loss=loss_info['policy_loss'],
            entropy_loss=loss_info['entropy_loss'],
            kl_penalty=loss_info['kl_penalty'],
            total_loss=loss_value,
            grad_norm=grad_norm,
            group_baseline=loss_info['group_baseline'],
            mean_reward=loss_info['mean_reward'],
            reward_std=loss_info['reward_std'],
            mean_advantage=loss_info['mean_advantage'],
            advantage_std=loss_info['advantage_std'],
            mean_entropy=loss_info['mean_entropy'],
            approx_kl=loss_info['approx_kl']
        )
    
    return grpo_update_step, optimizer.init


def _compute_grpo_loss(
    params: Any,
    batch_data: Dict[str, Any],
    policy_network: Any,
    config: GRPOConfig
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute GRPO loss using group-based advantage estimation.
    
    Pure GRPO: NO value network, uses group mean as baseline.
    Each sample is a single (state, action, reward) tuple.
    """
    # Extract batch data - these are lists/arrays of individual samples
    states = batch_data['states']  # List of AcquisitionState objects
    actions = batch_data['actions']  # List of intervention objects  
    rewards = batch_data['rewards']  # [group_size] array
    old_log_probs = batch_data['old_log_probs']  # [group_size] array
    
    # Vectorized forward pass through policy for all states
    # We need to use vmap to handle AcquisitionState objects efficiently
    def single_forward(state):
        return policy_network.apply(params, state, is_training=False)
    
    policy_outputs = jax.vmap(single_forward)(states)
    
    # Vectorized computation of new log probabilities
    def single_log_prob(output, action):
        return compute_action_log_probability(output, action)
    
    new_log_probs = jax.vmap(single_log_prob)(policy_outputs, actions)
    
    # Vectorized entropy computation
    def single_entropy(output):
        return _compute_policy_entropy(output)
    
    entropies = jax.vmap(single_entropy)(policy_outputs)
    
    # GRPO core: Use group mean as baseline (NO value network!)
    group_baseline = jnp.mean(rewards)
    advantages = rewards - group_baseline
    
    # Configurable advantage scaling (open-r1 enhancement)
    if config.scale_rewards:
        # Normalize advantages by group standard deviation (GRPO best practice)
        advantages = advantages / (jnp.std(advantages) + 1e-8)
    
    # Policy loss: clipped probability ratio
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    
    # Entropy loss: encourage exploration  
    entropy_loss = -jnp.mean(entropies)
    
    # Exact KL divergence: KL(π_old || π_new) = E[log(π_old) - log(π_new)]
    # This is the standard policy gradient KL divergence formula
    kl_divergence = jnp.mean(old_log_probs - new_log_probs)
    
    # KL penalty with threshold (standard practice: penalize when KL > target)
    kl_target = 0.01  # Standard threshold for policy divergence
    kl_penalty = config.kl_penalty_coeff * jnp.maximum(0.0, kl_divergence - kl_target)
    
    # Total loss: only policy + entropy + KL (NO value loss!)
    total_loss = policy_loss + config.entropy_coeff * entropy_loss + kl_penalty
    
    loss_info = {
        'policy_loss': policy_loss,
        'entropy_loss': entropy_loss, 
        'kl_penalty': kl_penalty,
        'group_baseline': group_baseline,
        'mean_reward': jnp.mean(rewards),
        'reward_std': jnp.std(rewards),
        'mean_advantage': jnp.mean(advantages),
        'advantage_std': jnp.std(advantages),
        'mean_entropy': jnp.mean(entropies),
        'approx_kl': kl_divergence
    }
    
    return total_loss, loss_info


def _compute_policy_entropy(policy_output: Dict[str, jnp.ndarray]) -> float:
    """Compute entropy of the policy for exploration regularization."""
    variable_logits = policy_output['variable_logits']
    value_params = policy_output['value_params']
    
    # Variable selection entropy (categorical distribution)
    var_probs = jax.nn.softmax(variable_logits)
    var_entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
    
    # Value distribution entropy (Gaussian)
    # For Gaussian: H = 0.5 * log(2 * pi * e * sigma^2) = log(sigma) + 0.5 * log(2 * pi * e)
    log_stds = value_params[:, 1]
    value_entropy = jnp.mean(log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    
    return var_entropy + value_entropy


def collect_grpo_batch(
    policy_network: Any,
    params: Any,
    states: List[AcquisitionState],
    scms: List[pyr.PMap],
    surrogate_model: Any,
    surrogate_params: Any,
    config: GRPOConfig,
    reward_config: pyr.PMap,
    key: jax.Array,
    reward_scaling: str = "none",
    reward_clip_value: float = 10.0
) -> Dict[str, Any]:
    """
    Collect a batch of (state, action, reward) tuples for GRPO training.
    
    Each sample is a single intervention, not a trajectory.
    States and SCMs can differ (allows diverse training).
    
    Args:
        states: List of initial states (can be different)
        scms: List of SCMs (can be different) 
        ... other args
    
    Returns:
        Dict with 'states', 'actions', 'rewards', 'old_log_probs'
    """
    if len(states) != len(scms):
        raise ValueError("Number of states must match number of SCMs")
    
    if len(states) < config.group_size:
        # Repeat states/scms to reach group size
        states = (states * ((config.group_size // len(states)) + 1))[:config.group_size]
        scms = (scms * ((config.group_size // len(scms)) + 1))[:config.group_size]
    else:
        # Take first group_size elements
        states = states[:config.group_size]
        scms = scms[:config.group_size]
    
    keys = jax.random.split(key, config.group_size)
    
    # Lists to store batch data
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_log_probs = []
    
    # Generate group of single intervention samples
    for i in range(config.group_size):
        current_state = states[i]
        current_scm = scms[i]
        
        # Sample intervention from policy
        policy_output = policy_network.apply(params, current_state, is_training=False)
        intervention = sample_intervention_from_policy(
            policy_output, current_state, keys[i]
        )
        
        # Apply intervention and get outcome
        outcome = sample_with_intervention(
            current_scm, intervention, n_samples=1, key=keys[i]
        )[0]
        
        # Create next state
        new_buffer = current_state.buffer.copy()
        new_buffer.add_intervention(intervention, outcome)
        
        next_state = create_acquisition_state(
            scm=current_scm,
            buffer=new_buffer,
            surrogate_model=surrogate_model,
            surrogate_params=surrogate_params,
            target_variable=current_state.current_target,
            step=current_state.step + 1
        )
        
        # Compute reward using our verifiable reward function
        reward_components = compute_verifiable_reward(
            current_state, intervention, outcome, next_state, reward_config
        )
        
        # Compute action log probability  
        log_prob = compute_action_log_probability(policy_output, intervention)
        
        # Store sample
        batch_states.append(current_state)
        batch_actions.append(intervention)
        batch_rewards.append(reward_components.total_reward)
        batch_log_probs.append(log_prob)
    
    # Apply reward scaling if specified (open-r1 enhancement)
    rewards_array = jnp.array(batch_rewards)
    if reward_scaling == "tanh":
        rewards_array = jnp.tanh(rewards_array / reward_clip_value) * reward_clip_value
    elif reward_scaling == "clip":
        rewards_array = jnp.clip(rewards_array, -reward_clip_value, reward_clip_value)
    # "none" or any other value: no scaling applied
    
    return {
        'states': batch_states,  # List of AcquisitionState objects
        'actions': batch_actions,  # List of intervention objects
        'rewards': rewards_array,
        'old_log_probs': jnp.array(batch_log_probs)
    }


def create_grpo_batch_from_samples(
    samples: List[Tuple[AcquisitionState, pyr.PMap, float, float]],
    config: GRPOConfig
) -> Dict[str, Any]:
    """
    Create a GRPO training batch from pre-collected samples.
    
    Args:
        samples: List of (state, action, reward, old_log_prob) tuples
        config: GRPO configuration
    
    Returns:
        Batch dict ready for GRPO training
    """
    if len(samples) < config.group_size:
        raise ValueError(
            f"Need at least {config.group_size} samples, got {len(samples)}"
        )
    
    # Take exactly group_size samples
    selected_samples = samples[:config.group_size]
    
    # Extract components
    states = [sample[0] for sample in selected_samples]
    actions = [sample[1] for sample in selected_samples]
    rewards = jnp.array([sample[2] for sample in selected_samples])
    old_log_probs = jnp.array([sample[3] for sample in selected_samples])
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'old_log_probs': old_log_probs
    }


def create_grpo_batch_from_buffer(
    buffer: TrajectoryBuffer,
    config: GRPOConfig,
    key: jax.Array
) -> Dict[str, Any]:
    """
    Create a GRPO batch by sampling from a trajectory buffer.
    
    Samples individual (state, action, reward) tuples randomly.
    """
    # Sample from buffer
    sampled_steps = buffer.sample_batch(config.group_size, key)
    
    # Convert to GRPO format
    states = [step.state for step in sampled_steps]
    actions = [step.action for step in sampled_steps] 
    rewards = jnp.array([step.reward for step in sampled_steps])
    
    # Extract log probabilities from metadata if available
    old_log_probs = []
    for step in sampled_steps:
        if 'log_prob' in step.metadata:
            old_log_probs.append(step.metadata['log_prob'])
        else:
            raise ValueError("TrajectoryStep must include log_prob in metadata for GRPO")
    
    old_log_probs = jnp.array(old_log_probs)
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'old_log_probs': old_log_probs
    }


# ============================================================================
# Open-R1 Enhanced Sample Reuse System
# ============================================================================

@dataclass
class SampleReuseManager:
    """Manages sample reuse for open-r1 enhanced GRPO training."""
    
    current_samples: List[Tuple[AcquisitionState, pyr.PMap, float, float]]
    reuse_iteration: int
    max_iterations: int
    
    def should_collect_new_samples(self, required_size: int) -> bool:
        """Determine if new samples should be collected."""
        # Collect new samples if:
        # 1. No samples exist yet
        # 2. Completed all reuse iterations  
        # 3. Not enough samples for required size
        
        if not self.current_samples:
            return True
            
        if self.reuse_iteration >= self.max_iterations:
            return True
            
        if len(self.current_samples) < required_size:
            return True
            
        return False
    
    def update_for_reuse(self) -> 'SampleReuseManager':
        """Create updated manager for next reuse iteration."""
        return SampleReuseManager(
            current_samples=self.current_samples,
            reuse_iteration=self.reuse_iteration + 1,
            max_iterations=self.max_iterations
        )
    
    def reset_with_new_samples(
        self, 
        new_samples: List[Tuple[AcquisitionState, pyr.PMap, float, float]]
    ) -> 'SampleReuseManager':
        """Reset manager with fresh samples."""
        return SampleReuseManager(
            current_samples=new_samples,
            reuse_iteration=0,
            max_iterations=self.max_iterations
        )


def collect_grpo_batch_with_reuse(
    policy_network: Any,
    params: Any,
    states: List[AcquisitionState],
    scms: List[pyr.PMap],
    surrogate_model: Any,
    surrogate_params: Any,
    config: GRPOConfig,
    reuse_manager: SampleReuseManager,
    reward_config: pyr.PMap,
    key: jax.Array,
    reward_scaling: str = "none",
    reward_clip_value: float = 10.0
) -> Tuple[Dict[str, Any], SampleReuseManager]:
    """
    Collect GRPO batch with sample reuse (open-r1 enhancement).
    
    Returns:
        (batch_data, updated_reuse_manager)
    """
    if reuse_manager.should_collect_new_samples(config.group_size):
        # Collect fresh samples
        batch = collect_grpo_batch(
            policy_network, params, states, scms, surrogate_model,
            surrogate_params, config, reward_config, key,
            reward_scaling, reward_clip_value
        )
        
        # Create new samples for reuse
        new_samples = list(zip(
            batch['states'],
            batch['actions'],
            batch['rewards'].tolist(),
            batch['old_log_probs'].tolist()
        ))
        
        updated_manager = reuse_manager.reset_with_new_samples(new_samples)
        
    else:
        # Reuse existing samples
        batch = create_grpo_batch_from_samples(
            reuse_manager.current_samples, config
        )
        
        updated_manager = reuse_manager.update_for_reuse()
    
    return batch, updated_manager


def create_sample_reuse_manager(config: GRPOConfig) -> SampleReuseManager:
    """Create a sample reuse manager with config parameters."""
    return SampleReuseManager(
        current_samples=[],
        reuse_iteration=0,
        max_iterations=config.num_iterations
    )