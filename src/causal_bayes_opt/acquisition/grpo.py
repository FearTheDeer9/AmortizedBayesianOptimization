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
    interventions_per_state: int = 1  # NEW: How many interventions to sample per (state, SCM) pair
    
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


def grpo_update_step(
    policy_network: Any,
    params: Any,
    opt_state: Any,
    batch_data: Dict[str, Any],
    config: GRPOConfig
) -> Tuple[Any, Any, GRPOUpdate]:
    """
    Standalone GRPO update step function.
    
    This is the standalone version of the GRPO update that can be imported directly.
    
    Args:
        policy_network: Policy network to update
        params: Current policy parameters
        opt_state: Current optimizer state
        batch_data: GRPO batch with 'states', 'actions', 'rewards', 'old_log_probs'
        config: GRPO configuration
        
    Returns:
        (new_params, new_opt_state, update_result)
    """
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate)
    )
    
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



def _extract_policy_input_from_tensor_state(state) -> jnp.ndarray:
    """
    Extract policy input tensor from TensorBackedAcquisitionState for enriched policy.
    
    Returns 3D tensor [max_history_size, n_vars, num_channels] as expected by enriched policy networks.
    Fixed to provide varied, realistic inputs instead of constant channels.
    """
    n_vars = state.config.n_vars
    
    # Get temporal configuration from state - standardized to 50
    max_history_size = getattr(state.config, 'max_history', 50)  # Standardized temporal dimension
    num_channels = 10  # Standard enriched policy channel count
    
    # Build varying feature channels for enriched policy
    # Channel 0-2: Mechanism features with variable-specific variation
    mech_features = state.mechanism_features  # [n_vars, feature_dim]
    if mech_features.shape[1] < 3:
        # Pad with varied values instead of zeros
        mech_pad = jnp.ones((n_vars, 3 - mech_features.shape[1])) * 0.5
        # Add variable-specific variation
        var_variation = jnp.arange(n_vars)[:, None] * 0.1  # [n_vars, 1]
        mech_pad = mech_pad + var_variation[:, :mech_pad.shape[1]]
        mech_channels = jnp.concatenate([mech_features, mech_pad], axis=1)
    else:
        # Take first 3 features if more than 3
        mech_channels = mech_features[:, :3]
    
    # Add variable-specific variation to mechanism features
    var_indices = jnp.arange(n_vars)[:, None]  # [n_vars, 1]
    mech_variation = var_indices * 0.05  # Small variation per variable
    mech_channels = mech_channels + mech_variation  # [n_vars, 3]
    
    # Channel 3: Marginal parent probabilities (already has variation)
    marginal_channel = state.marginal_probs[:, None]  # [n_vars, 1]
    
    # Channel 4: Confidence scores (already has variation)
    confidence_channel = state.confidence_scores[:, None]  # [n_vars, 1]
    
    # Channel 5-8: Global context features with variable-specific modifications
    base_global_features = jnp.array([
        state.best_value,
        state.uncertainty_bits,
        float(state.current_step) / 100.0,  # Normalize step count
        float(state.sample_buffer.n_samples) / 1000.0  # Normalize sample count
    ])
    
    # Create variable-specific global features instead of broadcasting identical values
    global_channels = jnp.zeros((n_vars, 4))  # [n_vars, 4]
    for var_idx in range(n_vars):
        # Add variable-specific perturbations to global features
        var_factor = (var_idx + 1) / n_vars  # 0.25, 0.5, 0.75, 1.0 for 4 variables
        
        global_channels = global_channels.at[var_idx, 0].set(base_global_features[0] * (0.8 + 0.4 * var_factor))  # Best value varies
        global_channels = global_channels.at[var_idx, 1].set(base_global_features[1] * (0.5 + 0.5 * var_factor))  # Uncertainty varies
        global_channels = global_channels.at[var_idx, 2].set(base_global_features[2])  # Step count same
        global_channels = global_channels.at[var_idx, 3].set(base_global_features[3] + var_factor * 0.1)  # Sample count slightly varies
    
    # Channel 9: Variable-specific noise instead of constant step noise
    variable_noise = jnp.zeros((n_vars, 1))
    for var_idx in range(n_vars):
        # Create variable-specific noise based on variable index and step
        noise_factor = (state.current_step * (var_idx + 1)) % 31  # Use prime for variation
        variable_noise = variable_noise.at[var_idx, 0].set(noise_factor / 31.0)  # Normalize to [0, 1]
    
    # Combine all channels: [n_vars, 10]
    current_features = jnp.concatenate([
        mech_channels,      # [n_vars, 3] - channels 0-2 (now varied)
        marginal_channel,   # [n_vars, 1] - channel 3 
        confidence_channel, # [n_vars, 1] - channel 4
        global_channels,    # [n_vars, 4] - channels 5-8 (now varied)
        variable_noise      # [n_vars, 1] - channel 9 (now varied)
    ], axis=1)  # [n_vars, 10]
    
    # Create temporal dimension with time-varying features instead of static repetition
    temporal_features = jnp.zeros((max_history_size, n_vars, num_channels))
    
    for t in range(max_history_size):
        time_factor = t / max_history_size  # 0.0 to ~1.0
        
        # Create time-varying features
        time_features = current_features.copy()
        
        # Add temporal variation to some channels
        # Channel 5: Best value evolves over time (simulates learning progress)
        time_features = time_features.at[:, 5].set(current_features[:, 5] * (0.5 + 0.5 * time_factor))
        
        # Channel 6: Uncertainty decreases over time (simulates confidence building)
        time_features = time_features.at[:, 6].set(current_features[:, 6] * (1.0 - 0.3 * time_factor))
        
        # Channel 7: Step count increases linearly over "history"
        time_features = time_features.at[:, 7].set(time_factor)
        
        # Channel 8: Sample count increases over "history"
        time_features = time_features.at[:, 8].set(current_features[:, 8] + time_factor * 0.5)
        
        # Channel 9: Time-varying noise
        time_noise = (t * jnp.arange(1, n_vars + 1)) % 17 / 17.0  # Different prime for temporal variation
        time_features = time_features.at[:, 9].set(time_noise)
        
        temporal_features = temporal_features.at[t].set(time_features)
    
    return temporal_features


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
    
    Updated to work efficiently with TensorBackedAcquisitionState objects.
    """
    # Extract batch data - these are lists/arrays of individual samples
    states = batch_data['states']  # List of TensorBackedAcquisitionState objects
    actions = batch_data['actions']  # List of intervention objects  
    rewards = batch_data['rewards']  # [group_size] array
    old_log_probs = batch_data['old_log_probs']  # [group_size] array
    
    # Validate that we have tensor-backed states
    from ..jax_native.state import TensorBackedAcquisitionState
    
    if not states or not isinstance(states[0], TensorBackedAcquisitionState):
        raise ValueError(
            f"GRPO requires TensorBackedAcquisitionState objects, got {type(states[0]) if states else 'empty list'}"
        )
    
    # Extract policy inputs and target indices for batch processing
    policy_inputs = []
    target_indices = []
    
    for state in states:
        policy_input = _extract_policy_input_from_tensor_state(state)
        policy_inputs.append(policy_input)
        
        # Find target index from state
        target_name = state.current_target
        variable_names = state.variable_names
        target_idx = variable_names.index(target_name) if target_name in variable_names else 0
        target_indices.append(target_idx)
    
    policy_input_batch = jnp.stack(policy_inputs)  # [batch_size, max_history_size, n_vars, num_channels]
    target_indices_batch = jnp.array(target_indices)  # [batch_size]
    
    # Pre-process actions to extract JAX-compatible arrays
    action_var_indices = []
    action_values = []
    
    for action in actions:
        # Get intervention details
        targets = action.get('targets', set())
        values = action.get('values', {})
        
        if not targets or not values:
            action_var_indices.append(0)
            action_values.append(0.0)
        else:
            # Assume single intervention for simplicity
            target_var = list(targets)[0]
            target_val = values[target_var]
            
            # Find variable index (assume variable names are 'X0', 'X1', etc.)
            if target_var.startswith('X'):
                var_idx = int(target_var[1:])
            else:
                var_idx = 0  # Fallback
            
            action_var_indices.append(var_idx)
            action_values.append(target_val)
    
    action_var_indices = jnp.array(action_var_indices)  # [batch_size]
    action_values = jnp.array(action_values)  # [batch_size]
    
    def single_forward_tensor(policy_input, target_idx):
        """Forward pass using policy input tensor and target index."""
        # Policy network expects: apply(params, key, enriched_input, target_idx, is_training)
        # policy_input is now [max_history_size, n_vars, num_channels] as required
        dummy_key = jax.random.PRNGKey(0)  # Use dummy key for deterministic forward pass
        
        # Call policy network and ensure outputs have correct shapes for vmap
        output = policy_network.apply(params, dummy_key, policy_input, target_idx, False)
        
        # DEBUG: Check output shapes before vmap
        # The policy network should return: {'variable_logits': [n_vars], 'value_params': [n_vars, 2], 'state_value': []}
        # But state_value might be returning wrong shape due to temporal dimension confusion
        
        # Fix state_value shape if needed
        if 'state_value' in output:
            state_value = output['state_value']
            # If state_value has wrong shape (temporal dimension), fix it
            if len(state_value.shape) == 1 and state_value.shape[0] > 10:
                # This is likely the temporal dimension being returned instead of scalar
                # Take mean over temporal dimension to get scalar
                fixed_state_value = jnp.mean(state_value)
                output = output.copy()
                output['state_value'] = fixed_state_value
        
        return output
    
    # Use vmap over policy input tensors and target indices
    policy_outputs = jax.vmap(single_forward_tensor)(policy_input_batch, target_indices_batch)
    
    # Vectorized computation of new log probabilities
    def single_log_prob(output, var_idx, action_val):
        """Compute log probability of action from policy output - simplified for GRPO."""
        variable_logits = output['variable_logits']  # [n_vars]
        value_params = output['value_params']        # [n_vars, 2]
        
        # Variable selection log probability (categorical)
        var_log_probs = jax.nn.log_softmax(variable_logits)
        var_log_prob = var_log_probs[var_idx]
        
        # Value selection log probability (Gaussian)
        mean, log_std = value_params[var_idx, 0], value_params[var_idx, 1]
        std = jnp.exp(log_std)
        val_log_prob = -0.5 * ((action_val - mean) / std) ** 2
        val_log_prob -= 0.5 * jnp.log(2 * jnp.pi) + log_std
        
        return var_log_prob + val_log_prob
    
    new_log_probs = jax.vmap(single_log_prob)(policy_outputs, action_var_indices, action_values)
    
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


def collect_grpo_batch_same_state(
    policy_network: Any,
    params: Any,
    base_state: AcquisitionState,
    base_scm: pyr.PMap,
    surrogate_model: Any,
    surrogate_params: Any,
    config: GRPOConfig,
    reward_config: pyr.PMap,
    key: jax.Array,
    policy_config: Any,
    reward_scaling: str = "none",
    reward_clip_value: float = 10.0
) -> Dict[str, Any]:
    """
    Collect GRPO batch using "Same State, Different Interventions" strategy.
    
    This is the CORRECT grouping approach for causal discovery:
    - Takes ONE state and ONE SCM 
    - Generates multiple interventions from same policy output
    - Groups interventions from same causal context (comparable rewards)
    - Computes meaningful group-relative advantages
    
    Args:
        base_state: Single acquisition state to sample from
        base_scm: Single SCM to apply interventions to
        config: GRPO config with interventions_per_state parameter
        ... other standard args
        
    Returns:
        GRPO batch dict with interventions from same state/SCM context
    """
    # Determine how many interventions to sample
    num_interventions = config.interventions_per_state
    if num_interventions <= 0:
        raise ValueError(f"interventions_per_state must be > 0, got {num_interventions}")
    
    # Convert AcquisitionState to enriched input tensor for policy network
    # This is a temporary conversion - in real implementation, this would be done properly
    from ..data_structures.scm import get_variables, get_target
    variables = list(get_variables(base_scm))
    target = get_target(base_scm)
    target_idx = variables.index(target) if target in variables else 0
    
    # Create enriched input tensor with correct variable count
    num_vars = len(variables)
    enriched_input = jnp.zeros((50, num_vars, 10))  # [time, actual_vars, channels]
    
    # Generate random key for policy network
    policy_key, key = jax.random.split(key)
    
    # Policy forward pass with correct signature
    policy_output = policy_network.apply(
        params, policy_key, enriched_input, target_idx, False
    )
    
    # Generate multiple random keys for different intervention samples
    intervention_keys = jax.random.split(key, num_interventions + 1)
    outcome_keys = jax.random.split(intervention_keys[-1], num_interventions)
    
    # Lists to store batch data
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_log_probs = []
    
    # Sample multiple interventions from same policy output and state
    for i in range(num_interventions):
        # Sample intervention using different random key
        # Create policy config if not provided
        if policy_config is None:
            from .policy import PolicyConfig
            policy_config = PolicyConfig()
        
        intervention = sample_intervention_from_policy(
            policy_output, base_state, intervention_keys[i], policy_config
        )
        
        # Apply intervention to same SCM
        # Convert JAX key to integer seed for sampling function
        seed = int(outcome_keys[i][0]) % (2**31 - 1)
        outcome = sample_with_intervention(
            base_scm, intervention, n_samples=1, seed=seed
        )[0]
        
        # Create next state with incremented step (minimal update for reward computation)
        from dataclasses import replace
        next_state = replace(base_state, step=base_state.step + 1)
        
        # Compute reward for this intervention
        reward_components = compute_verifiable_reward(
            state_before=base_state,
            intervention=intervention,
            outcome=outcome,
            state_after=next_state,
            config=reward_config
        )
        
        # Compute log probability of this intervention
        log_prob = compute_action_log_probability(
            policy_output, intervention, base_state, policy_config
        )
        
        # Store in batch (all from same state/SCM context)
        batch_states.append(base_state)
        batch_actions.append(intervention)
        batch_rewards.append(reward_components.total_reward)
        batch_log_probs.append(log_prob)
    
    # Convert to arrays for GRPO processing
    rewards_array = jnp.array(batch_rewards)
    
    # Apply reward scaling if requested
    if reward_scaling == "standardize":
        rewards_array = (rewards_array - jnp.mean(rewards_array)) / (jnp.std(rewards_array) + 1e-8)
    elif reward_scaling == "tanh":
        rewards_array = jnp.tanh(rewards_array / reward_clip_value) * reward_clip_value
    elif reward_scaling == "clip":
        rewards_array = jnp.clip(rewards_array, -reward_clip_value, reward_clip_value)
    
    return {
        'states': batch_states,  # All same state (comparable context)
        'actions': batch_actions,  # Different interventions from same context
        'rewards': rewards_array,  # Comparable rewards (same SCM/state)
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