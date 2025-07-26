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



def _project_embeddings_to_causal_importance(embeddings: jnp.ndarray) -> jnp.ndarray:
    """
    Project high-dimensional node embeddings to scalar causal importance scores.
    
    Uses PCA-like projection focusing on the highest variance dimensions,
    which typically capture the most important structural information.
    
    Args:
        embeddings: Node embeddings [n_vars, embed_dim] from surrogate model
        
    Returns:
        Causal importance scores [n_vars] in range [0, 1]
    """
    if embeddings.shape[1] >= 3:
        # Use first 3 dimensions with decreasing weights
        # These dimensions contain the highest variance in learned representations
        importance_weights = jnp.array([0.5, 0.3, 0.2])
        importance_scores = jnp.sum(embeddings[:, :3] * importance_weights, axis=1)
    else:
        # Fallback: simple mean for lower-dimensional embeddings
        importance_scores = jnp.mean(embeddings, axis=1)
    
    # Normalize to [0, 1] range using sigmoid-like transformation
    # This handles both positive and negative embedding values
    importance_scores = jax.nn.sigmoid(importance_scores)
    
    return importance_scores


def _project_embeddings_to_3d_features(embeddings: jnp.ndarray) -> jnp.ndarray:
    """
    Project high-dimensional node embeddings to 3D features preserving maximum information.
    
    Uses learned projection weights that focus on the most informative dimensions
    while preserving relative differences between variables.
    
    Args:
        embeddings: Node embeddings [n_vars, embed_dim] from surrogate model
        
    Returns:
        3D projected features [n_vars, 3] with normalized components
    """
    embed_dim = embeddings.shape[1]
    n_vars = embeddings.shape[0]
    
    if embed_dim >= 128:
        # For full 128D embeddings, use learned projection patterns
        # These indices are chosen to capture different aspects of causal structure
        
        # Dimension 1: Structural connectivity (early dimensions often encode graph properties)
        structural_indices = jnp.array([0, 1, 4, 8, 16])
        structural_weights = jnp.array([0.4, 0.3, 0.15, 0.1, 0.05])
        feature_1 = jnp.sum(embeddings[:, structural_indices] * structural_weights, axis=1)
        
        # Dimension 2: Causal depth (middle dimensions often encode hierarchical relationships)
        depth_indices = jnp.array([32, 48, 64, 80, 96])
        depth_weights = jnp.array([0.3, 0.25, 0.2, 0.15, 0.1])
        feature_2 = jnp.sum(embeddings[:, depth_indices] * depth_weights, axis=1)
        
        # Dimension 3: Variable-specific properties (later dimensions encode unique characteristics)
        specific_indices = jnp.array([100, 108, 116, 120, 124])
        specific_weights = jnp.array([0.25, 0.25, 0.2, 0.15, 0.15])
        feature_3 = jnp.sum(embeddings[:, specific_indices] * specific_weights, axis=1)
        
    elif embed_dim >= 64:
        # For smaller embeddings, adapt the projection
        structural_indices = jnp.array([0, 1, 4, 8])
        structural_weights = jnp.array([0.4, 0.3, 0.2, 0.1])
        feature_1 = jnp.sum(embeddings[:, structural_indices] * structural_weights, axis=1)
        
        depth_indices = jnp.array([16, 24, 32, 40])
        depth_weights = jnp.array([0.4, 0.3, 0.2, 0.1])
        feature_2 = jnp.sum(embeddings[:, depth_indices] * depth_weights, axis=1)
        
        specific_indices = jnp.array([48, 52, 56, 60])
        specific_weights = jnp.array([0.4, 0.3, 0.2, 0.1])
        feature_3 = jnp.sum(embeddings[:, specific_indices] * specific_weights, axis=1)
        
    else:
        # For very small embeddings, just use first 3 dimensions or repeat
        if embed_dim >= 3:
            feature_1 = embeddings[:, 0]
            feature_2 = embeddings[:, 1]
            feature_3 = embeddings[:, 2]
        elif embed_dim == 2:
            feature_1 = embeddings[:, 0]
            feature_2 = embeddings[:, 1]
            feature_3 = (embeddings[:, 0] + embeddings[:, 1]) / 2.0
        else:
            # embed_dim == 1
            feature_1 = embeddings[:, 0]
            feature_2 = embeddings[:, 0] * 0.5
            feature_3 = embeddings[:, 0] * -0.5
    
    # Stack into 3D features
    features_3d = jnp.stack([feature_1, feature_2, feature_3], axis=1)  # [n_vars, 3]
    
    # Normalize each dimension to [0, 1] range using tanh (preserves relative differences)
    features_3d = jax.nn.tanh(features_3d) * 0.5 + 0.5
    
    return features_3d


def _create_adaptive_exploration_schedule(current_step: int, n_vars: int, training_progress: float = 0.0) -> jnp.ndarray:
    """
    Create adaptive exploration factor that considers both training step and progress.
    
    Implements a more sophisticated schedule that adapts based on actual training
    progress rather than just step count.
    
    Args:
        current_step: Current training step
        n_vars: Number of variables (for broadcasting)
        training_progress: Training progress indicator [0, 1] (0=just started, 1=converged)
        
    Returns:
        Exploration factors [n_vars] that decrease with progress
    """
    # Base exponential decay schedule
    step_decay_rate = 0.005  # Slower decay than original
    step_exploration = jnp.exp(-step_decay_rate * current_step)
    
    # Progress-based adaptation
    progress_exploration = 1.0 - training_progress
    
    # Combine step and progress signals
    # Early training: dominated by step decay
    # Later training: dominated by progress signal
    step_weight = jnp.exp(-current_step / 200.0)  # Decreases over time
    progress_weight = 1.0 - step_weight
    
    combined_exploration = (
        step_weight * step_exploration + 
        progress_weight * progress_exploration
    )
    
    # Ensure minimum exploration to maintain some stochasticity
    min_exploration = 0.02  # Lower minimum for more focused behavior
    exploration_factor = jnp.maximum(combined_exploration, min_exploration)
    
    # Add small variable-specific variation to break symmetry
    var_indices = jnp.arange(n_vars)
    var_noise = 0.01 * jnp.sin(var_indices * 0.5)  # Small sinusoidal variation
    variable_factors = exploration_factor + var_noise
    
    # Clip to valid range
    variable_factors = jnp.clip(variable_factors, min_exploration, 1.0)
    
    return variable_factors


def _create_exploration_schedule(current_step: int, n_vars: int) -> jnp.ndarray:
    """
    Create exploration factor that decreases over training steps.
    
    Implements a schedule that encourages exploration early in training
    and gradually reduces it as the policy learns.
    
    Args:
        current_step: Current training step
        n_vars: Number of variables (for broadcasting)
        
    Returns:
        Exploration factors [n_vars] that decrease over time
    """
    # Exponential decay: starts at 1.0, decays to ~0.1 over 1000 steps
    decay_rate = 0.01
    base_exploration = jnp.exp(-decay_rate * current_step)
    
    # Ensure minimum exploration to maintain some stochasticity
    min_exploration = 0.05
    exploration_factor = jnp.maximum(base_exploration, min_exploration)
    
    # Broadcast to all variables
    return jnp.full(n_vars, exploration_factor)


def _extract_policy_input_from_tensor_state(state) -> jnp.ndarray:
    """
    Extract simplified policy input tensor from TensorBackedAcquisitionState.
    
    Phase 3: Simplified Input Extraction - Uses 5 meaningful channels derived from 
    actual surrogate model learned features instead of artificial variation.
    
    Returns 3D tensor [max_history_size, n_vars, 5] with interpretable channels:
    - Channel 0: Causal importance (projected from node embeddings)
    - Channel 1: Parent probability (from surrogate predictions)
    - Channel 2: Confidence score (prediction uncertainty)
    - Channel 3: Global exploration factor (training progress)
    - Channel 4: Temporal position (intervention history)
    """
    n_vars = state.config.n_vars
    max_history_size = getattr(state.config, 'max_history', 100)  # Use 100 as default to match training config
    num_channels = 5  # Simplified to 5 interpretable channels
    
    # Channel 0: Causal Importance - Enhanced projection from 128D node embeddings
    node_embeddings = state.mechanism_features  # [n_vars, 128] from surrogate
    causal_importance = _project_embeddings_to_causal_importance(node_embeddings)
    
    # Channel 1: Parent Probability - Direct from surrogate predictions
    parent_probs = state.marginal_probs  # [n_vars] - already normalized
    
    # Channel 2: Confidence Score - Direct from surrogate uncertainty
    confidence_scores = state.confidence_scores  # [n_vars] - already normalized
    
    # Channel 3: Adaptive Exploration Factor - Consider training progress
    # Get training progress from state if available, otherwise use step-based estimate
    training_progress = getattr(state, 'training_progress', 0.0)
    exploration_factor = _create_adaptive_exploration_schedule(
        current_step=state.current_step,
        n_vars=n_vars,
        training_progress=training_progress
    )
    
    # Channel 4: Temporal Position - Enhanced intervention history indicator
    # Use actual sample count as proxy for temporal position with better scaling
    sample_density = float(state.sample_buffer.n_samples) / 1000.0
    temporal_base = jnp.clip(sample_density, 0.0, 1.0)
    
    # Add variable-specific temporal variation based on intervention patterns
    # This helps the policy understand which variables have been intervened on more
    var_indices = jnp.arange(n_vars)
    temporal_variation = 0.1 * jnp.sin(var_indices * 0.3 + temporal_base * 2.0)
    temporal_position = jnp.clip(temporal_base + temporal_variation, 0.0, 1.0)
    
    # Combine channels: [n_vars, 5]
    current_features = jnp.stack([
        causal_importance,   # Channel 0: Enhanced causal importance from embeddings
        parent_probs,        # Channel 1: Surrogate parent probabilities
        confidence_scores,   # Channel 2: Prediction confidence
        exploration_factor,  # Channel 3: Adaptive exploration signal
        temporal_position    # Channel 4: Enhanced temporal progress indicator
    ], axis=1)  # [n_vars, 5]
    
    # Create temporal history: Use real intervention history from sample buffer
    temporal_features = _extract_temporal_history_features(
        current_features=current_features,
        sample_buffer=state.sample_buffer,
        max_history_size=max_history_size,
        n_vars=n_vars,
        num_channels=num_channels
    )
    
    return temporal_features


def _extract_temporal_history_features(
    current_features: jnp.ndarray,
    sample_buffer,
    max_history_size: int,
    n_vars: int,
    num_channels: int
) -> jnp.ndarray:
    """
    Extract temporal history features from sample buffer.
    
    Creates a temporal sequence showing how intervention patterns and learned
    features have evolved over the recent history.
    
    Args:
        current_features: Current timestep features [n_vars, num_channels]
        sample_buffer: Sample buffer containing intervention history
        max_history_size: Maximum number of temporal steps
        n_vars: Number of variables
        num_channels: Number of feature channels
        
    Returns:
        Temporal features [max_history_size, n_vars, num_channels]
    """
    # Get recent intervention history from sample buffer
    recent_interventions = _get_recent_interventions(sample_buffer, max_history_size)
    
    if not recent_interventions:
        # No history available - use current features for all timesteps
        return jnp.broadcast_to(
            current_features[None, :, :], 
            (max_history_size, n_vars, num_channels)
        )
    
    # Create temporal sequence
    temporal_sequence = []
    
    for t in range(max_history_size):
        if t < len(recent_interventions):
            # Use actual historical intervention data
            intervention_data = recent_interventions[-(t+1)]  # Most recent first
            timestep_features = _create_features_from_intervention_data(
                intervention_data=intervention_data,
                current_features=current_features,
                timestep=t,
                n_vars=n_vars,
                num_channels=num_channels
            )
        else:
            # Beyond available history - decay the oldest available data
            decay_factor = 0.8 ** (t - len(recent_interventions) + 1)
            oldest_features = temporal_sequence[-1] if temporal_sequence else current_features
            timestep_features = oldest_features * decay_factor
            
            # Add small noise to break perfect symmetry
            noise = 0.01 * jax.random.normal(
                jax.random.PRNGKey(t), 
                (n_vars, num_channels)
            )
            timestep_features = timestep_features + noise
        
        temporal_sequence.append(timestep_features)
    
    # Stack into temporal tensor [max_history_size, n_vars, num_channels]
    # Note: sequence is built newest-to-oldest, so reverse for oldest-to-newest ordering
    temporal_features = jnp.stack(temporal_sequence[::-1], axis=0)
    
    return temporal_features


def _get_recent_interventions(sample_buffer, max_history_size: int) -> list:
    """
    Extract recent interventions from sample buffer.
    
    Args:
        sample_buffer: Sample buffer with intervention history
        max_history_size: Maximum number of interventions to extract
        
    Returns:
        List of recent intervention data dictionaries
    """
    if not hasattr(sample_buffer, 'samples') or not sample_buffer.samples:
        return []
    
    # Get recent samples (most recent first)
    recent_samples = sample_buffer.samples[-max_history_size:]
    
    intervention_data = []
    for sample in recent_samples:
        # Extract intervention information from sample
        data = {
            'intervention_target': getattr(sample, 'intervention_target', None),
            'intervention_value': getattr(sample, 'intervention_value', 0.0),
            'outcome_data': getattr(sample, 'outcome_data', None),
            'timestamp': getattr(sample, 'timestamp', 0),
            'reward': getattr(sample, 'reward', 0.0)
        }
        intervention_data.append(data)
    
    return intervention_data


def _create_features_from_intervention_data(
    intervention_data: dict,
    current_features: jnp.ndarray,
    timestep: int,
    n_vars: int,
    num_channels: int
) -> jnp.ndarray:
    """
    Create feature representation for a specific historical intervention.
    
    Args:
        intervention_data: Dictionary with intervention information
        current_features: Current timestep features as baseline [n_vars, num_channels]
        timestep: Temporal offset (0=most recent, higher=older)
        n_vars: Number of variables
        num_channels: Number of feature channels
        
    Returns:
        Features for this timestep [n_vars, num_channels]
    """
    # Start with decayed current features
    temporal_decay = 0.9 ** timestep
    features = current_features * temporal_decay
    
    # Modify features based on historical intervention
    intervention_target = intervention_data.get('intervention_target')
    intervention_value = intervention_data.get('intervention_value', 0.0)
    reward = intervention_data.get('reward', 0.0)
    
    if intervention_target is not None:
        try:
            # Find target variable index
            if isinstance(intervention_target, str):
                # Handle string variable names like 'X0', 'X1', etc.
                if intervention_target.startswith('X'):
                    target_idx = int(intervention_target[1:])
                else:
                    target_idx = 0  # Fallback
            else:
                target_idx = int(intervention_target)
            
            # Ensure target_idx is valid
            target_idx = max(0, min(target_idx, n_vars - 1))
            
            # Modify features for the intervened variable
            # Channel 0: Increase causal importance for intervened variable
            intervention_boost = 0.2 * jnp.clip(abs(intervention_value), 0.0, 1.0)
            features = features.at[target_idx, 0].add(intervention_boost)
            
            # Channel 1: Adjust parent probability based on intervention outcome
            reward_signal = jnp.clip(reward / 10.0, -0.3, 0.3)  # Normalize reward
            features = features.at[target_idx, 1].add(reward_signal)
            
            # Channel 2: Reduce confidence for intervened variable (uncertainty from intervention)
            confidence_reduction = 0.1 * timestep  # More reduction for older interventions
            features = features.at[target_idx, 2].add(-confidence_reduction)
            
            # Channel 3: Mark intervention in exploration signal
            features = features.at[target_idx, 3].set(0.8 - 0.1 * timestep)
            
            # Channel 4: Set temporal position marker
            temporal_marker = 1.0 - 0.1 * timestep
            features = features.at[target_idx, 4].set(temporal_marker)
            
        except (ValueError, IndexError):
            # Invalid target index - use current features with decay
            pass
    
    # Ensure all values stay in reasonable ranges
    features = jnp.clip(features, 0.0, 1.0)
    
    return features


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
    
    # Get variable names from the first state (all states in batch have same variables)
    variable_names = states[0].variable_names if states else []
    
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
            
            # Find variable index using the actual variable names list
            if target_var in variable_names:
                var_idx = variable_names.index(target_var)
            else:
                # Fallback: try old 'X0' pattern for backwards compatibility
                if target_var.startswith('X') and len(target_var) > 1:
                    try:
                        var_idx = int(target_var[1:])
                    except ValueError:
                        var_idx = 0  # Default fallback
                else:
                    var_idx = 0  # Default fallback
            
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
    # Note: This is a temporary dummy tensor for policy initialization - should use actual max_history_size
    enriched_input = jnp.zeros((100, num_vars, 5))  # [time, actual_vars, channels] - 5-channel system
    
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