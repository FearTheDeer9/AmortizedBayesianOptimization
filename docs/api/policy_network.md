# Acquisition Policy Network API Reference

## Overview
The Acquisition Policy Network module provides neural networks for intelligent intervention selection using alternating attention transformers. The policy networks are designed for dual-objective optimization, balancing structure learning and target variable optimization through a two-headed architecture.

## Core Types

### PolicyConfig
```python
@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for acquisition policy networks."""
    
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.0
    ffn_multiplier: int = 4
    layer_norm_eps: float = 1e-6
    use_bias: bool = True
```

Configuration object controlling policy network architecture and training behavior.

### AlternatingAttentionEncoder
```python
class AlternatingAttentionEncoder(hk.Module):
    """Alternating attention transformer for intervention data processing."""
    
    def __call__(self, 
                 history: jnp.ndarray,  # [n_samples, n_vars, 2]
                 is_training: bool = True) -> jnp.ndarray:
        """Apply alternating attention over samples and variables."""
```

Transformer encoder that alternates attention over samples and variables to properly encode permutation symmetries in intervention data.

## Core Functions

### create_acquisition_policy(config, example_state)
Create and initialize an acquisition policy network.

**Parameters:**
- `config: PolicyConfig` - Network configuration parameters
- `example_state: AcquisitionState` - Example state for parameter initialization

**Returns:**
`hk.Transformed` - Haiku transformed function for policy network

**Example:**
```python
config = PolicyConfig(
    hidden_dim=128,
    num_layers=6,
    num_heads=8,
    dropout=0.1
)

# Create example state for initialization
example_state = create_acquisition_state(samples, posterior, target, variables)

# Create policy network
policy_network = create_acquisition_policy(config, example_state)

# Initialize parameters
key = jax.random.PRNGKey(42)
params = policy_network.init(key, example_state)
```

### sample_intervention_from_policy(policy_output, state, key, exploration_noise=0.1)
Sample an intervention from policy network output.

**Parameters:**
- `policy_output: Dict[str, jnp.ndarray]` - Output from policy network forward pass
- `state: AcquisitionState` - Current state for context
- `key: jax.Array` - Random key for sampling
- `exploration_noise: float` - Exploration noise level (default: 0.1)

**Returns:**
`Sample` - Sampled intervention

**Example:**
```python
# Forward pass through policy
policy_output = policy_network.apply(params, state)

# Sample intervention
key = jax.random.PRNGKey(42)
intervention = sample_intervention_from_policy(
    policy_output=policy_output,
    state=state,
    key=key,
    exploration_noise=0.1
)

print(f"Selected variable: {list(intervention['targets'])[0]}")
print(f"Intervention value: {intervention['values']}")
```

### compute_action_log_probability(policy_output, intervention, state)
Compute log probability of an intervention under current policy.

**Parameters:**
- `policy_output: Dict[str, jnp.ndarray]` - Policy network output
- `intervention: Sample` - Intervention to evaluate
- `state: AcquisitionState` - State context

**Returns:**
`float` - Log probability of intervention

**Example:**
```python
# Compute log probability of intervention
log_prob = compute_action_log_probability(
    policy_output=policy_output,
    intervention=intervention,
    state=state
)

print(f"Log probability: {log_prob:.3f}")
print(f"Probability: {jnp.exp(log_prob):.3f}")
```

### compute_policy_entropy(policy_output)
Compute entropy of policy output for regularization.

**Parameters:**
- `policy_output: Dict[str, jnp.ndarray]` - Policy network output

**Returns:**
`float` - Policy entropy

**Example:**
```python
entropy = compute_policy_entropy(policy_output)
print(f"Policy entropy: {entropy:.3f}")

# Higher entropy indicates more exploration
if entropy > 2.0:
    print("High entropy - policy is exploring")
else:
    print("Low entropy - policy is exploiting")
```

## Policy Network Architecture

### Two-Headed Design
The policy network uses a two-headed architecture optimized for dual objectives:

```python
class AcquisitionPolicyNetwork(hk.Module):
    """Two-headed policy network for intervention selection."""
    
    def __call__(self, state: AcquisitionState, is_training: bool = True):
        """
        Returns:
            {
                'variable_logits': [n_vars] - Variable selection logits
                'value_params': [n_vars, 2] - (mean, log_std) for intervention values
            }
        """
```

### Variable Selection Head
Selects which variable to intervene on using uncertainty information:

```python
def _variable_selection_head(self, state_emb, state):
    """Select intervention target using marginal parent probabilities."""
    # Incorporate uncertainty features
    marginal_probs = jnp.array(list(state.marginal_parent_probs.values()))
    uncertainty_features = jnp.expand_dims(marginal_probs, axis=1)
    
    # Combined features for selection
    combined_features = jnp.concatenate([state_emb, uncertainty_features], axis=1)
    
    # MLP for variable selection
    variable_logits = self._variable_mlp(combined_features)
    return variable_logits
```

### Value Selection Head
Determines intervention values using optimization context:

```python
def _value_selection_head(self, state_emb, state):
    """Select intervention values using optimization context."""
    # Add optimization context
    best_value_feature = jnp.full((state_emb.shape[0], 1), state.best_target_value)
    augmented_features = jnp.concatenate([state_emb, best_value_feature], axis=1)
    
    # Output mean and log_std for Normal distribution
    value_params = self._value_mlp(augmented_features)  # [n_vars, 2]
    return value_params
```

## Analysis and Debugging Functions

### analyze_policy_output(policy_output, state, variable_order)
Analyze policy network output for insights and debugging.

**Parameters:**
- `policy_output: Dict[str, jnp.ndarray]` - Policy network output
- `state: AcquisitionState` - Current state
- `variable_order: List[str]` - Variable names in order

**Returns:**
`Dict[str, Any]` - Analysis including:
- `'variable_preferences'`: Which variables policy prefers
- `'value_distributions'`: Intervention value distributions
- `'entropy_breakdown'`: Entropy by component
- `'uncertainty_alignment'`: How well policy aligns with uncertainty

**Example:**
```python
analysis = analyze_policy_output(policy_output, state, ['X', 'Y', 'Z'])

print("Variable preferences:")
for var, pref in analysis['variable_preferences'].items():
    print(f"  {var}: {pref:.3f}")

print(f"Total entropy: {analysis['entropy_breakdown']['total']:.3f}")
print(f"Uncertainty alignment: {analysis['uncertainty_alignment']:.3f}")
```

### validate_policy_output(policy_output, state)
Validate policy output for correctness and numerical stability.

**Parameters:**
- `policy_output: Dict[str, jnp.ndarray]` - Policy output to validate
- `state: AcquisitionState` - State context

**Returns:**
`Dict[str, bool]` - Validation results

**Example:**
```python
validation = validate_policy_output(policy_output, state)

if not validation['logits_finite']:
    print("Warning: Non-finite logits detected")
if not validation['probabilities_normalized']:
    print("Warning: Probabilities don't sum to 1")
if not validation['value_params_reasonable']:
    print("Warning: Unreasonable value parameters")
```

## Alternating Attention Architecture

### Core Concept
The alternating attention encoder applies self-attention alternately over samples and variables:

```python
def __call__(self, history: jnp.ndarray, is_training: bool = True):
    """Process intervention history with alternating attention."""
    # Input: [n_samples, n_vars, 2]
    x = self._input_projection(history)
    
    for layer in range(self.num_layers):
        # Attention over samples (for each variable)
        x = self._sample_attention_layer(x, is_training, f"sample_{layer}")
        
        # Attention over variables (for each sample)  
        x = self._variable_attention_layer(x, is_training, f"var_{layer}")
    
    # Global pooling: [n_samples, n_vars, hidden] -> [n_vars, hidden]
    state_embedding = jnp.max(x, axis=0)
    return state_embedding
```

### Benefits
- **Permutation Invariance**: Handles variable and sample reordering
- **Rich Interactions**: Models complex dependencies in intervention data
- **Proven Architecture**: Based on successful CAASL design
- **Scalability**: Efficient attention computation with linear layers

## Integration Patterns

### With GRPO Training
```python
# Policy network in GRPO training loop
def grpo_update_step(params, opt_state, batch_data):
    states = batch_data['states']
    
    # Vectorized policy forward pass
    policy_outputs = jax.vmap(policy_network.apply, in_axes=(None, 0))(params, states)
    
    # Compute log probabilities for actions
    log_probs = jax.vmap(compute_action_log_probability)(
        policy_outputs, batch_data['actions'], states
    )
    
    # Continue with GRPO loss computation...
```

### With Exploration Strategies
```python
# Policy with exploration guidance
def select_intervention_with_exploration(policy, params, state, exploration_strategy, key):
    # Get base policy output
    policy_output = policy.apply(params, state)
    
    # Add exploration bonuses
    exploration_bonuses = exploration_strategy.compute_bonuses(state)
    
    # Modify logits based on exploration
    modified_logits = policy_output['variable_logits'] + exploration_bonuses
    modified_output = {**policy_output, 'variable_logits': modified_logits}
    
    # Sample intervention
    return sample_intervention_from_policy(modified_output, state, key)
```

### With Reward Computation
```python
# Policy evaluation for reward computation
def evaluate_intervention_quality(policy, params, state, intervention):
    policy_output = policy.apply(params, state)
    
    # Check alignment with policy preferences
    log_prob = compute_action_log_probability(policy_output, intervention, state)
    entropy = compute_policy_entropy(policy_output)
    
    return {
        'policy_likelihood': jnp.exp(log_prob),
        'policy_entropy': entropy,
        'exploration_quality': log_prob / entropy if entropy > 0 else 0.0
    }
```

## Performance Optimization

### JAX Compilation
```python
# Compile policy functions for speed
@jax.jit
def compiled_policy_forward(params, state):
    return policy_network.apply(params, state)

@jax.jit 
def compiled_intervention_sampling(policy_output, state, key):
    return sample_intervention_from_policy(policy_output, state, key)
```

### Batch Processing
```python
# Vectorized policy evaluation
batch_policy_outputs = jax.vmap(policy_network.apply, in_axes=(None, 0))(
    params, batch_states
)

batch_interventions = jax.vmap(sample_intervention_from_policy, in_axes=(0, 0, 0))(
    batch_policy_outputs, batch_states, batch_keys
)
```

## Key Design Principles

### Dual-Objective Awareness
- Variable selection head uses structural uncertainty (marginal parent probabilities)
- Value selection head incorporates optimization context (current best value)
- Architecture balances exploration and exploitation naturally

### Architectural Soundness
- Based on proven alternating attention design from CAASL
- Proper handling of permutation symmetries in intervention data
- Efficient transformer implementation with linear complexity

### Training Stability
- Gradient clipping and normalization for stable training
- Proper initialization schemes for transformer components
- Regularization through dropout and entropy penalties

## Common Usage Patterns

### Basic Policy Usage
```python
# Initialize policy
config = PolicyConfig(hidden_dim=128, num_layers=4)
policy = create_acquisition_policy(config, example_state)
params = policy.init(key, example_state)

# Select intervention
policy_output = policy.apply(params, current_state)
intervention = sample_intervention_from_policy(policy_output, current_state, key)
```

### Training Integration
```python
# Policy in training loop
for step in range(training_steps):
    # Collect batch of experiences
    batch = collect_policy_batch(policy, params, states, scms, key)
    
    # Update policy parameters
    params, opt_state, update_info = grpo_update_step(params, opt_state, batch)
    
    # Monitor training progress
    if step % 100 == 0:
        entropy = jnp.mean([compute_policy_entropy(out) for out in batch['policy_outputs']])
        print(f"Step {step}: Policy entropy = {entropy:.3f}")
```

### Analysis and Debugging
```python
# Analyze policy behavior
analysis = analyze_policy_output(policy_output, state, variable_order)
validation = validate_policy_output(policy_output, state)

if not all(validation.values()):
    print("Policy output validation failed:")
    for check, passed in validation.items():
        if not passed:
            print(f"  {check}: FAILED")
```