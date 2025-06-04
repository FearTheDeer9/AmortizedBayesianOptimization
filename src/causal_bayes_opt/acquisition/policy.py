"""
Policy Network with Alternating Attention for ACBO Acquisition.

This module implements a sophisticated policy network that combines alternating attention
transformers (following CAASL architecture) with dual-objective design for both
optimization and structure learning in causal discovery.

Key Components:
1. AlternatingAttentionEncoder: Transformer that alternates attention over samples and variables
2. AcquisitionPolicyNetwork: Two-headed policy (variable selection + value parameters)
3. Integration with rich AcquisitionState from Phase 1&2

Architecture decisions based on CAASL insights and dual-objective requirements.
"""

# Standard library imports
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Third-party imports
import jax
import jax.numpy as jnp
import haiku as hk
import pyrsistent as pyr

# Local imports
from .state import AcquisitionState
from ..interventions.registry import apply_intervention
from ..data_structures.sample import get_values, get_intervention_targets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for the acquisition policy network."""
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    exploration_noise: float = 0.1
    variable_selection_temp: float = 1.0
    value_selection_temp: float = 1.0


class AlternatingAttentionEncoder(hk.Module):
    """
    Alternating attention transformer encoder for intervention data.
    
    Applies self-attention alternately over samples and variables to properly encode
    permutation symmetries in intervention data, following CAASL architecture.
    
    This design handles the fact that intervention data has natural symmetries:
    - Samples can be permuted arbitrarily
    - Variables can be permuted (with consistent ordering)
    
    The alternating pattern ensures both symmetries are properly captured.
    """
    
    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 name: str = "AlternatingAttentionEncoder"):
        super().__init__(name=name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

    def __call__(self, 
                 history: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, 3]
                 is_training: bool = True) -> jnp.ndarray:
        """
        Apply alternating attention following CAASL architecture.
        
        Args:
            history: Intervention history in [MAX_HISTORY_SIZE, n_vars, 3] format where:
                     [:, :, 0] = standardized variable values
                     [:, :, 1] = intervention indicators (1 if intervened, 0 otherwise)  
                     [:, :, 2] = target indicators (1 if target variable, 0 otherwise)
                     Note: History is padded/truncated to fixed size for parameter consistency
            is_training: Training mode flag for dropout
            
        Returns:
            State embedding of shape [n_vars, hidden_dim]
        """
        dropout_rate = self.dropout if is_training else 0.0
        
        # Input projection: [MAX_HISTORY_SIZE, n_vars, 3] -> [MAX_HISTORY_SIZE, n_vars, hidden_dim]
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(history)
        
        # Apply alternating attention layers
        for layer_idx in range(self.num_layers):
            # Attention over samples (axis=0) - each variable attends over its samples
            x = self._sample_attention_layer(x, dropout_rate, f"sample_attn_{layer_idx}")
            
            # Attention over variables (axis=1) - each sample attends over variables
            x = self._variable_attention_layer(x, dropout_rate, f"var_attn_{layer_idx}")
        
        # Max pooling over samples dimension to get variable-level representations
        # [MAX_HISTORY_SIZE, n_vars, hidden_dim] -> [n_vars, hidden_dim]
        state_embedding = jnp.max(x, axis=0)
        
        return state_embedding
    
    @hk.transparent
    def _sample_attention_layer(self, 
                               x: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, hidden_dim]
                               dropout_rate: float,
                               layer_name: str) -> jnp.ndarray:
        """Apply self-attention over samples dimension."""
        with hk.experimental.name_scope(layer_name):
            n_samples, n_vars, hidden_dim = x.shape
            
            # Transpose for attention: [n_vars, MAX_HISTORY_SIZE, hidden_dim]
            x_transposed = jnp.transpose(x, (1, 0, 2))
            
            # Process each variable independently - attend over its samples
            attended_vars = []
            
            for var_idx in range(n_vars):
                var_samples = x_transposed[var_idx]  # [MAX_HISTORY_SIZE, hidden_dim]
                
                # Layer norm before attention (pre-norm architecture)
                var_samples_norm = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True
                )(var_samples)
                
                # Multi-head self-attention over samples for this variable
                attn_output = hk.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_size=hidden_dim // self.num_heads,
                    w_init_scale=2.0,
                    model_size=hidden_dim
                )(var_samples_norm, var_samples_norm, var_samples_norm)
                
                # Residual connection
                var_attended = var_samples + hk.dropout(hk.next_rng_key(), dropout_rate, attn_output)
                attended_vars.append(var_attended)
            
            # Stack back: [n_vars, MAX_HISTORY_SIZE, hidden_dim]
            x_attended = jnp.stack(attended_vars, axis=0)
            
            # Feed-forward network for each variable
            x_ff_list = []
            for var_idx in range(n_vars):
                var_data = x_attended[var_idx]  # [MAX_HISTORY_SIZE, hidden_dim]
                
                # Layer norm
                var_norm = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True
                )(var_data)
                
                # Feed-forward
                var_ff = hk.Sequential([
                    hk.Linear(4 * hidden_dim, w_init=self.w_init),
                    jax.nn.relu,
                    hk.Linear(hidden_dim, w_init=self.w_init),
                ])(var_norm)
                
                # Residual connection with dropout
                var_out = var_data + hk.dropout(hk.next_rng_key(), dropout_rate, var_ff)
                x_ff_list.append(var_out)
            
            x_ff = jnp.stack(x_ff_list, axis=0)  # [n_vars, MAX_HISTORY_SIZE, hidden_dim]
            
            # Transpose back: [MAX_HISTORY_SIZE, n_vars, hidden_dim]
            return jnp.transpose(x_ff, (1, 0, 2))
    
    @hk.transparent
    def _variable_attention_layer(self,
                                 x: jnp.ndarray,  # [MAX_HISTORY_SIZE, n_vars, hidden_dim]
                                 dropout_rate: float,
                                 layer_name: str) -> jnp.ndarray:
        """Apply self-attention over variables dimension."""
        with hk.experimental.name_scope(layer_name):
            n_samples, n_vars, hidden_dim = x.shape
            
            # Process each sample independently - attend over variables
            attended_samples = []
            
            for sample_idx in range(n_samples):
                sample_vars = x[sample_idx]  # [n_vars, hidden_dim]
                
                # Layer norm before attention
                sample_vars_norm = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True
                )(sample_vars)
                
                # Multi-head self-attention over variables for this sample
                attn_output = hk.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_size=hidden_dim // self.num_heads,
                    w_init_scale=2.0,
                    model_size=hidden_dim
                )(sample_vars_norm, sample_vars_norm, sample_vars_norm)
                
                # Residual connection
                sample_attended = sample_vars + hk.dropout(hk.next_rng_key(), dropout_rate, attn_output)
                attended_samples.append(sample_attended)
            
            # Stack back: [MAX_HISTORY_SIZE, n_vars, hidden_dim]
            x_attended = jnp.stack(attended_samples, axis=0)
            
            # Feed-forward network for each sample
            x_ff_list = []
            for sample_idx in range(n_samples):
                sample_data = x_attended[sample_idx]  # [n_vars, hidden_dim]
                
                # Layer norm
                sample_norm = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True
                )(sample_data)
                
                # Feed-forward
                sample_ff = hk.Sequential([
                    hk.Linear(4 * hidden_dim, w_init=self.w_init),
                    jax.nn.relu,
                    hk.Linear(hidden_dim, w_init=self.w_init),
                ])(sample_norm)
                
                # Residual connection with dropout
                sample_out = sample_data + hk.dropout(hk.next_rng_key(), dropout_rate, sample_ff)
                x_ff_list.append(sample_out)
            
            return jnp.stack(x_ff_list, axis=0)  # [MAX_HISTORY_SIZE, n_vars, hidden_dim]


class AcquisitionPolicyNetwork(hk.Module):
    """
    Two-headed policy network for intervention selection with alternating attention.
    
    Combines proven transformer architecture with dual-objective design for 
    optimization + structure learning. The two heads are:
    1. Variable selection head: Which variable to intervene on (uses uncertainty info)
    2. Value selection head: What value to intervene with (uses optimization context)
    
    This design leverages our rich uncertainty infrastructure from Phase 2.
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 name: str = "AcquisitionPolicyNetwork"):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")

    def __call__(self, state: AcquisitionState, is_training: bool = True) -> Dict[str, jnp.ndarray]:
        """
        Forward pass for intervention selection.
        
        Args:
            state: Current acquisition state with rich context
            is_training: Training mode flag
            
        Returns:
            Dictionary containing:
            - 'variable_logits': [n_vars] - Which variable to intervene on
            - 'value_params': [n_vars, 2] - (mean, log_std) for intervention values
            - 'state_value': [] - State value estimate (for GRPO baseline)
        """
        # Convert state to history format for transformer input
        history = self._state_to_history_format(state)  # [MAX_HISTORY_SIZE, n_vars, 3]
        
        # Encode with alternating attention
        encoder = AlternatingAttentionEncoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        )
        
        # Get state embedding: [n_vars, hidden_dim]
        state_embedding = encoder(history, is_training)
        
        # Variable selection head (leverages uncertainty information)
        variable_logits = self._variable_selection_head(state_embedding, state)
        
        # Value selection head (leverages optimization context)
        value_params = self._value_selection_head(state_embedding, state)
        
        # State value estimation (for GRPO baseline)
        state_value = self._state_value_head(state_embedding, state)
        
        return {
            'variable_logits': variable_logits,
            'value_params': value_params,
            'state_value': state_value
        }

    @hk.transparent
    def _state_to_history_format(self, state: AcquisitionState) -> jnp.ndarray:
        """
        Convert AcquisitionState to standardized history format for transformer input.
        
        IMPORTANT: Returns a FIXED SIZE array to ensure Haiku parameter compatibility
        when buffer size changes during training.
        
        Returns:
            Array of shape [MAX_HISTORY_SIZE, n_vars, 3] where:
            [:, :, 0] = standardized variable values
            [:, :, 1] = intervention indicators (1 if intervened, 0 otherwise)
            [:, :, 2] = target indicators (1 if target variable, 0 otherwise)
        """
        # Fixed maximum history size to ensure consistent transformer input shape
        MAX_HISTORY_SIZE = 100  
        
        # Get all samples from buffer
        all_samples = state.buffer.get_all_samples()
        
        # Get variable ordering from buffer
        variable_order = sorted(state.buffer.get_variable_coverage())
        n_vars = len(variable_order)
        
        if not all_samples or n_vars == 0:
            # Handle empty buffer case - return zeros with fixed shape
            return jnp.zeros((MAX_HISTORY_SIZE, max(1, n_vars), 3))
        
        n_actual_samples = len(all_samples)
        
        # Take most recent samples if we have too many, pad if too few
        if n_actual_samples > MAX_HISTORY_SIZE:
            # Take the most recent MAX_HISTORY_SIZE samples
            samples_to_use = all_samples[-MAX_HISTORY_SIZE:]
            n_used_samples = MAX_HISTORY_SIZE
        else:
            # Use all samples and pad the rest
            samples_to_use = all_samples
            n_used_samples = n_actual_samples
        
        # Initialize arrays with fixed size
        values_array = jnp.zeros((MAX_HISTORY_SIZE, n_vars))
        intervention_array = jnp.zeros((MAX_HISTORY_SIZE, n_vars))
        target_array = jnp.zeros((MAX_HISTORY_SIZE, n_vars))
        
        # Fill arrays for actual samples
        for sample_idx, sample in enumerate(samples_to_use):
            sample_values = get_values(sample)
            
            # Fill variable values
            for var_idx, var_name in enumerate(variable_order):
                if var_name in sample_values:
                    values_array = values_array.at[sample_idx, var_idx].set(
                        float(sample_values[var_name])
                    )
                
                # Mark target variable
                if var_name == state.current_target:
                    target_array = target_array.at[sample_idx, var_idx].set(1.0)
            
            # Fill intervention indicators
            intervention_targets = get_intervention_targets(sample)
            for var_idx, var_name in enumerate(variable_order):
                if var_name in intervention_targets:
                    intervention_array = intervention_array.at[sample_idx, var_idx].set(1.0)
        
        # Standardize values (zero mean, unit variance per variable)
        # Only use the actual samples for computing statistics
        if n_used_samples > 1:
            actual_values = values_array[:n_used_samples, :]
            values_std = jnp.std(actual_values, axis=0, keepdims=True) + 1e-8
            values_mean = jnp.mean(actual_values, axis=0, keepdims=True)
        else:
            # Single sample case - just use unit scaling
            values_std = jnp.ones((1, n_vars))
            values_mean = jnp.zeros((1, n_vars))
        
        # Apply standardization to the full array (padded entries will remain 0)
        values_standardized = (values_array - values_mean) / values_std
        
        # Stack into final format: [MAX_HISTORY_SIZE, n_vars, 3]
        history = jnp.stack([values_standardized, intervention_array, target_array], axis=2)
        
        return history

    @hk.transparent
    def _variable_selection_head(self, 
                                state_emb: jnp.ndarray,  # [n_vars, hidden_dim]
                                state: AcquisitionState) -> jnp.ndarray:
        """
        Select which variable to intervene on using uncertainty information.
        
        Leverages rich uncertainty information from ParentSetPosterior rather than
        simple thresholding approaches used in pure structure learning.
        """
        n_vars = state_emb.shape[0]
        
        # Get marginal parent probabilities as uncertainty features
        variable_order = sorted(state.buffer.get_variable_coverage())
        marginal_probs = []
        
        for var_name in variable_order:
            prob = state.marginal_parent_probs.get(var_name, 0.0)
            marginal_probs.append(prob)
        
        marginal_probs = jnp.array(marginal_probs)  # [n_vars]
        
        # Compute uncertainty features (high uncertainty = prob near 0.5)
        uncertainty_features = 1.0 - 2.0 * jnp.abs(marginal_probs - 0.5)  # [n_vars]
        
        # Add global context features
        uncertainty_bits = jnp.full((n_vars,), state.uncertainty_bits)
        best_value_feat = jnp.full((n_vars,), state.best_value)
        step_feat = jnp.full((n_vars,), float(state.step))
        
        # Combine all features: [n_vars, hidden_dim + 4]
        context_features = jnp.stack([
            marginal_probs, uncertainty_features, uncertainty_bits, 
            best_value_feat, step_feat
        ], axis=1)  # [n_vars, 5]
        
        combined_features = jnp.concatenate([state_emb, context_features], axis=1)
        
        # MLP for variable selection with batch norm for stability
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(combined_features)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(x)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        
        # Output logits
        variable_logits = hk.Linear(1, w_init=self.w_init)(x).squeeze(-1)  # [n_vars]
        
        # Mask out target variable (can't intervene on target)
        target_mask = jnp.array([
            1.0 if var != state.current_target else -jnp.inf
            for var in variable_order
        ])
        
        variable_logits = variable_logits + target_mask
        
        return variable_logits

    @hk.transparent
    def _value_selection_head(self,
                             state_emb: jnp.ndarray,  # [n_vars, hidden_dim]
                             state: AcquisitionState) -> jnp.ndarray:
        """
        Select intervention values using optimization context.
        
        Returns parameters for normal distribution over intervention values,
        incorporating current best value and optimization progress.
        """
        n_vars = state_emb.shape[0]
        
        # Add optimization context features
        best_value_feature = jnp.full((n_vars, 1), state.best_value)
        
        # Get optimization progress metrics
        opt_progress = state.get_optimization_progress()
        progress_features = jnp.full((n_vars, 4), jnp.array([
            opt_progress['improvement_from_start'],
            opt_progress['recent_improvement'],
            opt_progress['optimization_rate'],
            opt_progress['stagnation_steps']
        ]))
        
        # Combine features: [n_vars, hidden_dim + 5]
        augmented_features = jnp.concatenate([
            state_emb, best_value_feature, progress_features
        ], axis=1)
        
        # MLP for value parameters
        x = hk.Linear(self.hidden_dim, w_init=self.w_init)(augmented_features)
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(x)
        x = jax.nn.relu(x)
        
        # Output mean and log_std for each variable: [n_vars, 2]
        value_params = hk.Linear(2, w_init=self.w_init)(x)
        
        # Split into means and log_stds
        means = value_params[:, 0]
        log_stds = jnp.clip(value_params[:, 1], -2.0, 2.0)  # Reasonable variance range
        
        return jnp.stack([means, log_stds], axis=1)  # [n_vars, 2]

    @hk.transparent
    def _state_value_head(self, 
                         state_emb: jnp.ndarray,  # [n_vars, hidden_dim]
                         state: AcquisitionState) -> jnp.ndarray:
        """State value estimation for GRPO baseline."""
        # Global pooling over variables to get state-level representation
        global_features = jnp.mean(state_emb, axis=0)  # [hidden_dim]
        
        # Add scalar state features
        state_features = jnp.array([
            state.best_value,
            state.uncertainty_bits,
            float(state.step),
            float(state.buffer_statistics.total_samples),
            float(state.buffer_statistics.num_interventions)
        ])
        
        # Combine: [hidden_dim + 5]
        combined = jnp.concatenate([global_features, state_features])
        
        # MLP for state value
        x = hk.Linear(self.hidden_dim // 2, w_init=self.w_init)(combined)
        x = jax.nn.relu(x)
        
        x = hk.Linear(self.hidden_dim // 4, w_init=self.w_init)(x)
        x = jax.nn.relu(x)
        
        state_value = hk.Linear(1, w_init=self.w_init)(x).squeeze(-1)  # []
        
        return state_value


# Factory functions
def create_acquisition_policy(
    config: PolicyConfig,
    example_state: AcquisitionState
) -> hk.Transformed:
    """
    Create and initialize acquisition policy network.
    
    Args:
        config: Policy configuration
        example_state: Example state for initialization
        
    Returns:
        Transformed Haiku model ready for training
    """
    def policy_fn(state: AcquisitionState, is_training: bool = True):
        policy = AcquisitionPolicyNetwork(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        return policy(state, is_training)
    
    return hk.transform(policy_fn)


def sample_intervention_from_policy(
    policy_output: Dict[str, jnp.ndarray],
    state: AcquisitionState,
    key: jax.Array,
    config: PolicyConfig
) -> pyr.PMap:
    """
    Sample intervention from policy network output.
    
    Args:
        policy_output: Output from policy network forward pass
        state: Current acquisition state
        key: JAX random key
        config: Policy configuration
        
    Returns:
        Intervention specification ready for application
    """
    variable_logits = policy_output['variable_logits']
    value_params = policy_output['value_params']
    
    # Sample variable to intervene on
    var_key, val_key = jax.random.split(key)
    
    # Add exploration noise to variable selection
    noisy_logits = variable_logits + config.exploration_noise * jax.random.normal(
        var_key, variable_logits.shape
    )
    
    # Temperature-scaled sampling for variable selection
    scaled_logits = noisy_logits / config.variable_selection_temp
    selected_var_idx = jax.random.categorical(var_key, scaled_logits)
    
    # Sample intervention value for selected variable
    mean, log_std = value_params[selected_var_idx]
    std = jnp.exp(log_std) * config.value_selection_temp
    intervention_value = mean + std * jax.random.normal(val_key)
    
    # Get variable names from state
    variable_order = sorted(state.buffer.get_variable_coverage())
    selected_var = variable_order[selected_var_idx]
    
    # Create intervention specification using existing framework
    from ..interventions.handlers import create_perfect_intervention
    
    return create_perfect_intervention(
        targets=frozenset([selected_var]),
        values={selected_var: float(intervention_value)}
    )


def compute_action_log_probability(
    policy_output: Dict[str, jnp.ndarray],
    intervention: pyr.PMap,
    state: AcquisitionState,
    config: PolicyConfig
) -> jnp.ndarray:
    """
    Compute log probability of an action under the current policy.
    
    Required for GRPO training to compute policy gradients.
    
    Args:
        policy_output: Output from policy network
        intervention: Intervention that was taken
        state: State when intervention was selected
        config: Policy configuration
        
    Returns:
        Log probability of the intervention under current policy
    """
    variable_logits = policy_output['variable_logits']
    value_params = policy_output['value_params']
    
    # Extract intervention details
    targets = intervention['targets']
    values = intervention['values']
    
    if len(targets) != 1:
        # This policy only supports single-variable interventions
        return jnp.array(-jnp.inf)
    
    target_var = list(targets)[0]
    target_value = values[target_var]
    
    # Get variable index
    variable_order = sorted(state.buffer.get_variable_coverage())
    try:
        var_idx = variable_order.index(target_var)
    except ValueError:
        # Variable not in state
        return jnp.array(-jnp.inf)
    
    # Variable selection log probability
    scaled_logits = variable_logits / config.variable_selection_temp
    var_log_probs = jax.nn.log_softmax(scaled_logits)
    var_log_prob = var_log_probs[var_idx]
    
    # Value selection log probability (normal distribution)
    mean, log_std = value_params[var_idx]
    std = jnp.exp(log_std) * config.value_selection_temp
    
    # Log probability under normal distribution
    val_log_prob = -0.5 * jnp.log(2 * jnp.pi) - log_std - jnp.log(config.value_selection_temp)
    val_log_prob -= 0.5 * ((target_value - mean) / std) ** 2
    
    return var_log_prob + val_log_prob


def compute_policy_entropy(
    policy_output: Dict[str, jnp.ndarray],
    config: PolicyConfig
) -> jnp.ndarray:
    """
    Compute entropy of the policy distribution for exploration bonus.
    
    Args:
        policy_output: Output from policy network
        config: Policy configuration
        
    Returns:
        Total entropy of the policy distribution
    """
    variable_logits = policy_output['variable_logits']
    value_params = policy_output['value_params']
    
    # Variable selection entropy (handle -inf values for masked variables)
    scaled_logits = variable_logits / config.variable_selection_temp
    
    # Mask out -inf values before computing entropy
    finite_mask = jnp.isfinite(scaled_logits)
    if jnp.sum(finite_mask) == 0:
        # All variables are masked - no entropy
        var_entropy = 0.0
    else:
        # Only compute entropy over non-masked variables
        finite_logits = jnp.where(finite_mask, scaled_logits, -jnp.inf)
        var_probs = jax.nn.softmax(finite_logits)
        log_probs = jax.nn.log_softmax(finite_logits)
        # Only include finite probabilities in entropy calculation
        finite_probs = jnp.where(finite_mask, var_probs, 0.0)
        finite_log_probs = jnp.where(finite_mask, log_probs, 0.0)
        var_entropy = -jnp.sum(finite_probs * finite_log_probs)
    
    # Value selection entropy (mean of individual normal entropies)
    log_stds = value_params[:, 1]
    value_entropies = 0.5 * jnp.log(2 * jnp.pi * jnp.e) + log_stds + jnp.log(config.value_selection_temp)
    avg_value_entropy = jnp.mean(value_entropies)
    
    return var_entropy + avg_value_entropy


# Utility functions for debugging and analysis
def analyze_policy_output(
    policy_output: Dict[str, jnp.ndarray],
    state: AcquisitionState,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Analyze policy output for debugging and interpretation.
    
    Args:
        policy_output: Output from policy network
        state: Current acquisition state
        top_k: Number of top variables to show
        
    Returns:
        Human-readable analysis of policy decisions
    """
    variable_logits = policy_output['variable_logits']
    value_params = policy_output['value_params']
    state_value = policy_output['state_value']
    
    variable_order = sorted(state.buffer.get_variable_coverage())
    
    # Top-k variable preferences
    top_indices = jnp.argsort(variable_logits)[::-1][:top_k]
    top_variables = []
    
    for idx in top_indices:
        var_name = variable_order[idx]
        logit = float(variable_logits[idx])
        mean, log_std = value_params[idx]
        std = jnp.exp(log_std)
        
        top_variables.append({
            'variable': var_name,
            'logit': logit,
            'intervention_mean': float(mean),
            'intervention_std': float(std),
            'marginal_parent_prob': state.marginal_parent_probs.get(var_name, 0.0)
        })
    
    return {
        'state_value_estimate': float(state_value),
        'top_variables': top_variables,
        'variable_selection_entropy': float(compute_policy_entropy(policy_output, PolicyConfig()).item()),
        'state_context': {
            'step': state.step,
            'uncertainty_bits': state.uncertainty_bits,
            'best_value': state.best_value,
            'buffer_size': state.buffer_statistics.total_samples,
            "num_variables": len(state.buffer.get_variable_coverage())
        }
    }


def validate_policy_output(
    policy_output: Dict[str, jnp.ndarray],
    state: AcquisitionState
) -> bool:
    """
    Validate that policy output is well-formed.
    
    Args:
        policy_output: Output to validate
        state: Associated state
        
    Returns:
        True if output is valid, False otherwise
    """
    try:
        required_keys = {'variable_logits', 'value_params', 'state_value'}
        if not all(key in policy_output for key in required_keys):
            logger.error(f"Missing required keys in policy output: {required_keys}")
            return False
        
        variable_logits = policy_output['variable_logits']
        value_params = policy_output['value_params']
        state_value = policy_output['state_value']
        
        n_vars = len(state.buffer.get_variable_coverage())
        
        # Check shapes
        if variable_logits.shape != (n_vars,):
            logger.error(f"Invalid variable_logits shape: {variable_logits.shape}, expected ({n_vars},)")
            return False
        
        if value_params.shape != (n_vars, 2):
            logger.error(f"Invalid value_params shape: {value_params.shape}, expected ({n_vars}, 2)")
            return False
        
        if state_value.shape != ():
            logger.error(f"Invalid state_value shape: {state_value.shape}, expected ()")
            return False
        
        # Check for NaN values (Inf is allowed for target variable masking)
        if jnp.any(jnp.isnan(variable_logits)):
            logger.error("NaN values in variable_logits")
            return False
        
        # Check that we have valid intervention targets (unless only target variable exists)
        n_vars = len(state.buffer.get_variable_coverage())
        n_finite_logits = jnp.sum(jnp.isfinite(variable_logits))
        
        if n_vars == 1:
            # Edge case: only target variable exists, all logits should be -inf
            if n_finite_logits > 0:
                logger.error("Expected all logits to be -inf when only target variable exists")
                return False
        else:
            # Normal case: at least one non-target variable should have finite logit
            if n_finite_logits == 0:
                logger.error("All variable logits are masked (-inf) - no valid intervention targets")
                return False
        
        if not jnp.all(jnp.isfinite(value_params)):
            logger.error("Non-finite values in value_params")
            return False
        
        if not jnp.isfinite(state_value):
            logger.error("Non-finite state_value")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating policy output: {e}")
        return False
