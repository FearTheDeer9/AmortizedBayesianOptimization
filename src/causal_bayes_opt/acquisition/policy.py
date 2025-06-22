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
from .jax_utils import (
    create_target_mask_jax,
    create_mechanism_features_vectorized,
    apply_target_mask_jit,
    prepare_samples_for_history_jax
)
# JAX-native vectorized components
from .vectorized_attention import (
    VectorizedAcquisitionPolicyNetwork,
    create_vectorized_acquisition_policy
)
from .state_tensor_converter import (
    convert_acquisition_state_to_tensors,
    apply_target_mask_to_logits,
    TensorizedAcquisitionState
)
from .state_enhanced import (
    EnhancedAcquisitionState,
    create_enhanced_acquisition_state,
    upgrade_acquisition_state_to_enhanced
)
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
        
        # Use JAX-compatible preparation function
        # Note: This function is called outside the JAX-compiled region,
        # so it can still use Python loops for data preparation
        history = prepare_samples_for_history_jax(
            all_samples,
            variable_order,
            state.current_target,
            MAX_HISTORY_SIZE
        )
        
        return history

    @hk.transparent
    def _variable_selection_head(self, 
                                state_emb: jnp.ndarray,  # [n_vars, hidden_dim]
                                state: AcquisitionState) -> jnp.ndarray:
        """
        Select which variable to intervene on using uncertainty and mechanism information.
        
        Enhanced for mechanism-aware intervention selection (Architecture Enhancement Pivot - Part C):
        - Leverages uncertainty information from ParentSetPosterior
        - Uses mechanism confidence and predicted effects
        - Prioritizes high-impact variables with uncertain mechanisms
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
        
        # Get mechanism confidence features (Architecture Enhancement Pivot - Part C)
        mechanism_confidence_features = []
        mechanism_insights = state.get_mechanism_insights()
        
        for var_name in variable_order:
            confidence = state.mechanism_confidence.get(var_name, 0.0)
            mechanism_confidence_features.append(confidence)
        
        mechanism_confidence_features = jnp.array(mechanism_confidence_features)  # [n_vars]
        
        # Get predicted effect magnitude features
        predicted_effects = []
        for var_name in variable_order:
            effect = mechanism_insights['predicted_effects'].get(var_name, 0.0)
            predicted_effects.append(abs(effect))  # Use magnitude for prioritization
        
        predicted_effects = jnp.array(predicted_effects)  # [n_vars]
        
        # High-impact variable indicators
        high_impact_indicators = []
        for var_name in variable_order:
            is_high_impact = 1.0 if var_name in mechanism_insights['high_impact_variables'] else 0.0
            high_impact_indicators.append(is_high_impact)
        
        high_impact_indicators = jnp.array(high_impact_indicators)  # [n_vars]
        
        # Get mechanism type features (Architecture Enhancement Pivot - Part C)
        mechanism_type_features = []
        mechanism_types = mechanism_insights.get('mechanism_types', {})
        
        for var_name in variable_order:
            mech_type = mechanism_types.get(var_name, 'unknown')
            # One-hot encoding for mechanism types (linear=1, polynomial=2, gaussian=3, neural=4, unknown=0)
            type_encoding = {
                'linear': 1.0, 'polynomial': 2.0, 'gaussian': 3.0, 'neural': 4.0
            }.get(mech_type, 0.0)
            mechanism_type_features.append(type_encoding)
        
        mechanism_type_features = jnp.array(mechanism_type_features)  # [n_vars]
        
        # Add global context features
        uncertainty_bits = jnp.full((n_vars,), state.uncertainty_bits)
        best_value_feat = jnp.full((n_vars,), state.best_value)
        step_feat = jnp.full((n_vars,), float(state.step))
        
        # Combine all features: [n_vars, hidden_dim + 9] (enhanced with mechanism type features)
        context_features = jnp.stack([
            marginal_probs, uncertainty_features, uncertainty_bits, 
            best_value_feat, step_feat,
            mechanism_confidence_features, predicted_effects, high_impact_indicators,
            mechanism_type_features  # New: mechanism type encoding
        ], axis=1)  # [n_vars, 9]
        
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
        
        # Mask out target variable (can't intervene on target) - JAX-compatible
        target_mask = create_target_mask_jax(variable_order, state.current_target)
        variable_logits = apply_target_mask_jit(variable_logits, target_mask)
        
        return variable_logits

    @hk.transparent
    def _value_selection_head(self,
                             state_emb: jnp.ndarray,  # [n_vars, hidden_dim]
                             state: AcquisitionState) -> jnp.ndarray:
        """
        Select intervention values using optimization context and mechanism predictions.
        
        Enhanced for mechanism-aware value selection (Architecture Enhancement Pivot - Part C):
        - Uses predicted effect magnitudes to scale intervention values
        - Incorporates mechanism confidence for uncertainty-based exploration
        - Returns parameters for normal distribution over intervention values
        """
        n_vars = state_emb.shape[0]
        variable_order = sorted(state.buffer.get_variable_coverage())
        mechanism_insights = state.get_mechanism_insights()
        
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
        
        # Get mechanism-based features for value scaling (JAX-compatible)
        mechanism_features = create_mechanism_features_vectorized(
            variable_order,
            mechanism_insights,
            state.mechanism_confidence
        )  # [n_vars, 3]
        
        # Combine features: [n_vars, hidden_dim + 8] (enhanced with mechanism type scaling)
        augmented_features = jnp.concatenate([
            state_emb, best_value_feature, progress_features, mechanism_features
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
    example_state: AcquisitionState,
    use_vectorized: bool = True
) -> hk.Transformed:
    """
    Create and initialize acquisition policy network.
    
    Args:
        config: Policy configuration
        example_state: Example state for initialization
        use_vectorized: Whether to use JAX-native vectorized implementation
        
    Returns:
        Transformed Haiku model ready for training
    """
    if use_vectorized:
        # Use JAX-native vectorized implementation
        example_tensor_data = convert_acquisition_state_to_tensors(example_state)
        return create_vectorized_acquisition_policy(config, example_tensor_data)
    else:
        # Use legacy implementation (with for loops)
        def policy_fn(state: AcquisitionState, is_training: bool = True):
            policy = AcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            return policy(state, is_training)
        
        return hk.transform(policy_fn)


def create_jax_native_policy(
    config: PolicyConfig,
    example_state: AcquisitionState
) -> hk.Transformed:
    """
    Create JAX-native vectorized policy network.
    
    This function specifically creates the high-performance vectorized version
    that eliminates all Python loops and is fully JAX-compilable.
    
    Args:
        config: Policy configuration  
        example_state: Example state for tensor format determination
        
    Returns:
        JAX-compiled Haiku model
    """
    example_tensor_data = convert_acquisition_state_to_tensors(example_state)
    return create_vectorized_acquisition_policy(config, example_tensor_data)


def sample_intervention_from_policy(
    policy_output: Dict[str, jnp.ndarray],
    state,  # AcquisitionState or tensor data
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
    
    # Get variable names from state (handle both formats)
    if hasattr(state, 'buffer'):
        # Legacy AcquisitionState format
        variable_order = sorted(state.buffer.get_variable_coverage())
    elif isinstance(state, dict) and 'variable_order' in state:
        # Tensor data format
        variable_order = state['variable_order']
    else:
        raise ValueError("State must be AcquisitionState or tensor data dict")
    
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


def run_vectorized_policy_inference(
    policy_apply_fn,
    params,
    state: AcquisitionState,
    key: jax.Array,
    is_training: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    High-level function for running vectorized policy inference.
    
    This function handles the complete pipeline:
    1. Convert AcquisitionState to tensor format
    2. Run vectorized policy network
    3. Apply target masking
    4. Return results
    
    Args:
        policy_apply_fn: Policy network apply function
        params: Policy network parameters
        state: AcquisitionState object
        key: JAX random key
        is_training: Training mode flag
        
    Returns:
        Policy outputs with target masking applied
    """
    # Convert state to tensor format
    tensor_data = convert_acquisition_state_to_tensors(state)
    
    # Run vectorized policy network
    policy_output = policy_apply_fn(params, key, tensor_data, is_training)
    
    # Apply target masking to variable logits
    if tensor_data['target_idx'] >= 0:
        masked_logits = apply_target_mask_to_logits(
            policy_output['variable_logits'], 
            tensor_data['target_idx']
        )
        policy_output['variable_logits'] = masked_logits
    
    return policy_output


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


# Enhanced Factory Functions with Tensor-Based State Management

def create_enhanced_acquisition_policy(
    config: PolicyConfig,
    example_state: AcquisitionState,
    use_tensor_ops: bool = True,
    force_vectorized: bool = True
) -> hk.Transformed:
    """
    Create enhanced acquisition policy with tensor-based state management.
    
    This factory function creates a policy network that can work with both
    the original AcquisitionState and the new EnhancedAcquisitionState with
    tensor operations for maximum performance.
    
    Args:
        config: Policy configuration
        example_state: Example state for initialization
        use_tensor_ops: Whether to use tensor operations in state management
        force_vectorized: Whether to force JAX-native vectorized policy network
        
    Returns:
        Transformed Haiku model optimized for JAX compilation
    """
    if force_vectorized:
        # Convert to enhanced state for tensor operations
        if not isinstance(example_state, EnhancedAcquisitionState):
            enhanced_state = upgrade_acquisition_state_to_enhanced(
                example_state, use_tensor_ops=use_tensor_ops
            )
        else:
            enhanced_state = example_state
        
        # Get tensor input format for vectorized policy
        example_tensor_data = enhanced_state.get_tensor_input_for_policy()
        return create_vectorized_acquisition_policy(config, example_tensor_data)
    else:
        # Use legacy implementation with optional tensor ops in state
        def policy_fn(state, is_training: bool = True):
            # Auto-upgrade state if needed
            if not isinstance(state, EnhancedAcquisitionState):
                state = upgrade_acquisition_state_to_enhanced(state, use_tensor_ops=use_tensor_ops)
            
            policy = AcquisitionPolicyNetwork(
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            return policy(state, is_training)
        
        return hk.transform(policy_fn)


def create_fully_jax_native_policy(
    config: PolicyConfig,
    example_state: AcquisitionState
) -> hk.Transformed:
    """
    Create fully JAX-native policy with complete tensor-based architecture.
    
    This function creates the highest-performance version that:
    1. Uses EnhancedAcquisitionState with tensor operations
    2. Uses VectorizedAcquisitionPolicyNetwork with vmap operations
    3. Eliminates all Python loops and dictionary operations
    4. Enables full JAX compilation with maximum speedup
    
    Args:
        config: Policy configuration
        example_state: Example state for tensor format determination
        
    Returns:
        Fully JAX-compiled policy network for maximum performance
    """
    # Ensure we have enhanced state with tensor operations
    if not isinstance(example_state, EnhancedAcquisitionState):
        enhanced_state = upgrade_acquisition_state_to_enhanced(
            example_state, use_tensor_ops=True
        )
    else:
        enhanced_state = example_state
        enhanced_state.enable_tensor_operations(True)
    
    # Get tensor input format
    example_tensor_data = enhanced_state.get_tensor_input_for_policy()
    
    # Use vectorized policy network
    return create_vectorized_acquisition_policy(config, example_tensor_data)


def run_enhanced_policy_inference(
    policy_apply_fn,
    params,
    state: AcquisitionState,
    key: jax.Array,
    is_training: bool = False,
    use_tensor_ops: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Run policy inference with enhanced state management and tensor operations.
    
    This function provides the complete pipeline for high-performance inference:
    1. Auto-upgrade state to EnhancedAcquisitionState if needed
    2. Use tensor operations for all state computations
    3. Run JAX-compiled policy network
    4. Apply proper target masking
    
    Args:
        policy_apply_fn: Policy network apply function
        params: Policy network parameters
        state: AcquisitionState (will be upgraded if needed)
        key: JAX random key
        is_training: Training mode flag
        use_tensor_ops: Whether to use tensor operations
        
    Returns:
        Policy outputs with enhanced tensor-based processing
    """
    # Auto-upgrade state if needed
    if not isinstance(state, EnhancedAcquisitionState):
        enhanced_state = upgrade_acquisition_state_to_enhanced(state, use_tensor_ops=use_tensor_ops)
    else:
        enhanced_state = state
        if use_tensor_ops:
            enhanced_state.enable_tensor_operations(True)
    
    # Get tensor input
    tensor_data = enhanced_state.get_tensor_input_for_policy()
    
    # Run policy network
    policy_output = policy_apply_fn(params, key, tensor_data, is_training)
    
    # Apply target masking
    if tensor_data['target_idx'] >= 0:
        masked_logits = apply_target_mask_to_logits(
            policy_output['variable_logits'], 
            tensor_data['target_idx']
        )
        policy_output['variable_logits'] = masked_logits
    
    return policy_output


def sample_intervention_from_enhanced_policy(
    policy_output: Dict[str, jnp.ndarray],
    state: EnhancedAcquisitionState,
    key: jax.Array,
    config: PolicyConfig
) -> pyr.PMap:
    """
    Sample intervention from enhanced policy using tensor operations.
    
    Args:
        policy_output: Output from enhanced policy network
        state: EnhancedAcquisitionState with tensor capabilities
        key: JAX random key
        config: Policy configuration
        
    Returns:
        Intervention specification
    """
    # Use tensor data for variable ordering
    tensor_data = state.get_tensor_input_for_policy()
    
    # Delegate to existing sampling function with tensor data
    return sample_intervention_from_policy(policy_output, tensor_data, key, config)


# Auto-Selection Factory Function

def create_auto_acquisition_policy(
    config: PolicyConfig,
    example_state: AcquisitionState,
    performance_mode: str = "balanced"
) -> hk.Transformed:
    """
    Auto-select the best policy implementation based on performance requirements.
    
    This function automatically chooses between different implementations based
    on the performance mode and automatically handles state upgrades.
    
    Args:
        config: Policy configuration
        example_state: Example state for initialization
        performance_mode: Performance mode selection:
            - "maximum": Fully JAX-native with tensor operations (highest performance)
            - "balanced": Enhanced state with vectorized policy (good performance + compatibility)
            - "compatible": Legacy implementation with enhanced state (maximum compatibility)
            
    Returns:
        Optimally configured policy network
    """
    if performance_mode == "maximum":
        logger.info("Creating maximum performance JAX-native policy")
        return create_fully_jax_native_policy(config, example_state)
    
    elif performance_mode == "balanced":
        logger.info("Creating balanced enhanced policy with tensor operations")
        return create_enhanced_acquisition_policy(
            config, example_state, use_tensor_ops=True, force_vectorized=True
        )
    
    elif performance_mode == "compatible":
        logger.info("Creating compatible enhanced policy with legacy fallback")
        return create_enhanced_acquisition_policy(
            config, example_state, use_tensor_ops=True, force_vectorized=False
        )
    
    else:
        raise ValueError(f"Unknown performance mode: {performance_mode}. "
                        f"Choose from 'maximum', 'balanced', or 'compatible'")


# Migration Utilities

def validate_tensor_policy_compatibility(
    policy_apply_fn,
    params,
    example_state: AcquisitionState,
    key: jax.Array
) -> bool:
    """
    Validate that a policy works correctly with tensor operations.
    
    Args:
        policy_apply_fn: Policy apply function to test
        params: Policy parameters
        example_state: Example state for testing
        key: JAX random key
        
    Returns:
        True if policy works correctly with tensor operations
    """
    try:
        # Test with enhanced state
        enhanced_state = upgrade_acquisition_state_to_enhanced(example_state, use_tensor_ops=True)
        
        # Test tensor input creation
        tensor_data = enhanced_state.get_tensor_input_for_policy()
        
        # Test policy inference
        policy_output = policy_apply_fn(params, key, tensor_data, is_training=False)
        
        # Validate output format
        required_keys = ['variable_logits', 'value_params', 'state_value']
        if not all(key in policy_output for key in required_keys):
            logger.error(f"Missing required keys in policy output: {required_keys}")
            return False
        
        # Test intervention sampling
        intervention = sample_intervention_from_enhanced_policy(
            policy_output, enhanced_state, key, PolicyConfig()
        )
        
        if not isinstance(intervention, pyr.PMap):
            logger.error("Intervention sampling failed to return PMap")
            return False
        
        logger.info("Tensor policy compatibility validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Tensor policy compatibility validation failed: {e}")
        return False
