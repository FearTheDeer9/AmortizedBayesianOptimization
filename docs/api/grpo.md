# Enhanced GRPO Algorithm API Reference

## Overview
This module provides an enhanced implementation of Group Relative Policy Optimization (GRPO) with open-r1 features for training acquisition policies in causal Bayesian optimization. The implementation eliminates value networks by using group-based advantage estimation, making it particularly suitable for multi-objective reinforcement learning.

## Core Types

### GRPOConfig
```python
@dataclass
class GRPOConfig:
    """Enhanced GRPO configuration with open-r1 features."""
    
    # Core GRPO parameters
    group_size: int = 64                # DeepSeek recommendation (not 8!)
    clip_ratio: float = 0.2             # Policy gradient clipping
    entropy_coeff: float = 0.01         # Entropy regularization
    kl_penalty_coeff: float = 0.0       # Updated default (open-r1)
    max_grad_norm: float = 1.0          # Gradient clipping
    learning_rate: float = 3e-4         # Adam learning rate
    
    # Open-r1 enhancements
    num_iterations: int = 4             # Sample reuse iterations
    scale_rewards: bool = True          # Configurable advantage scaling
```

Enhanced configuration supporting modern GRPO training techniques.

### GRPOUpdate
```python
@dataclass
class GRPOUpdate:
    """Results from GRPO update step with comprehensive diagnostics."""
    
    # Loss components
    policy_loss: float
    entropy_loss: float
    kl_penalty: float
    total_loss: float
    grad_norm: float
    
    # Group statistics (key GRPO diagnostics)
    group_baseline: float               # Group mean baseline
    mean_reward: float                  # Average reward
    reward_std: float                   # Reward standard deviation
    mean_advantage: float               # Average advantage (should be ~0)
    advantage_std: float                # Advantage std (should be ~1)
    
    # Policy diagnostics
    mean_entropy: float                 # Average policy entropy
    approx_kl: float                    # Approximate KL divergence
```

Comprehensive update information including all key GRPO diagnostics.

### SampleReuseManager
```python
@dataclass(frozen=True)
class SampleReuseManager:
    """Manages sample reuse for training efficiency (open-r1 feature)."""
    
    current_samples: List[Tuple[Any, Any, float, float]]
    reuse_iteration: int
    max_iterations: int
    
    def should_collect_new_samples(self, required_size: int) -> bool:
        """Determine if new samples should be collected."""
    
    def update_for_reuse(self) -> 'SampleReuseManager':
        """Update iteration counter for reuse."""
    
    def reset_with_new_samples(self, new_samples: List) -> 'SampleReuseManager':
        """Reset with fresh samples."""
```

Manages sample reuse for improved training efficiency.

## Core Functions

### create_grpo_trainer(policy_network, config)
Create GRPO training infrastructure with JAX compilation.

**Parameters:**
- `policy_network: Any` - Policy network (AcquisitionPolicyNetwork)
- `config: GRPOConfig` - GRPO configuration

**Returns:**
- `Tuple[Callable, Callable]` - (grpo_update_step, optimizer_init) functions

**Example:**
```python
# Create enhanced GRPO trainer
config = GRPOConfig(
    group_size=64,           # DeepSeek recommendation
    num_iterations=4,        # Sample reuse
    scale_rewards=True,      # Configurable scaling
    kl_penalty_coeff=0.0     # Open-r1 default
)

update_step, optimizer_init = create_grpo_trainer(policy_network, config)

# Initialize training
params = policy_network.init(key, dummy_state)
opt_state = optimizer_init(params)

# Training step
batch = collect_grpo_batch(...)
new_params, new_opt_state, update_info = update_step(params, opt_state, batch)

print(f"Policy loss: {update_info.policy_loss:.3f}")
print(f"Group baseline: {update_info.group_baseline:.3f}")
print(f"KL divergence: {update_info.approx_kl:.6f}")
```

### collect_grpo_batch(policy_network, params, states, scms, surrogate_model, surrogate_params, config, reward_config, key, reward_scaling='tanh', reward_clip_value=10.0)
Collect experience batch for GRPO training with enhanced features.

**Parameters:**
- `policy_network: Any` - Policy network for intervention selection
- `params: Any` - Current policy parameters
- `states: List[AcquisitionState]` - Initial states (can be diverse)
- `scms: List[pyr.PMap]` - SCMs for environment simulation
- `surrogate_model: Any` - Surrogate model for posterior updates
- `surrogate_params: Any` - Surrogate model parameters
- `config: GRPOConfig` - GRPO configuration
- `reward_config: pyr.PMap` - Reward computation configuration
- `key: jax.Array` - Random key
- `reward_scaling: str` - Reward scaling method ('tanh', 'clip', 'none')
- `reward_clip_value: float` - Clipping value for rewards

**Returns:**
- `Dict[str, Any]` - Training batch with keys:
  - `'states'`: List of AcquisitionState objects
  - `'actions'`: List of intervention objects  
  - `'rewards'`: JAX array of shape [group_size]
  - `'old_log_probs'`: JAX array of shape [group_size]

**Example:**
```python
# Collect diverse training batch
states = [create_acquisition_state(...) for _ in range(10)]
scms = [create_random_scm(...) for _ in range(10)]

batch = collect_grpo_batch(
    policy_network=policy_network,
    params=params,
    states=states,
    scms=scms,
    surrogate_model=surrogate_model,
    surrogate_params=surrogate_params,
    config=config,
    reward_config=reward_config,
    key=key,
    reward_scaling='tanh',
    reward_clip_value=10.0
)

print(f"Batch size: {len(batch['states'])}")
print(f"Mean reward: {jnp.mean(batch['rewards']):.3f}")
```

### collect_grpo_batch_with_reuse(policy_network, params, states, scms, surrogate_model, surrogate_params, config, reward_config, sample_reuse_manager, key)
Collect GRPO batch with sample reuse for efficiency.

**Parameters:**
- All parameters from `collect_grpo_batch`
- `sample_reuse_manager: SampleReuseManager` - Manages sample reuse

**Returns:**
- `Tuple[Dict[str, Any], SampleReuseManager]` - (batch, updated_manager)

**Example:**
```python
# Initialize sample reuse
config = GRPOConfig(num_iterations=4)
manager = create_sample_reuse_manager(config)

# Training loop with sample reuse
for step in range(training_steps):
    batch, manager = collect_grpo_batch_with_reuse(
        policy_network, params, states, scms,
        surrogate_model, surrogate_params, config,
        reward_config, manager, keys[step]
    )
    
    # Update policy
    params, opt_state, update_info = update_step(params, opt_state, batch)
    
    if step % 50 == 0:
        print(f"Step {step}: Reuse iteration {manager.reuse_iteration}")
```

### create_sample_reuse_manager(config)
Create sample reuse manager for training efficiency.

**Parameters:**
- `config: GRPOConfig` - Configuration with reuse parameters

**Returns:**
- `SampleReuseManager` - Initialized reuse manager

**Example:**
```python
manager = create_sample_reuse_manager(config)

# Check if new samples needed
if manager.should_collect_new_samples(config.group_size):
    # Collect fresh samples
    samples = collect_new_samples(...)
    manager = manager.reset_with_new_samples(samples)
else:
    # Reuse existing samples
    manager = manager.update_for_reuse()
```

## Enhanced Features

### Sample Reuse (Open-r1)
Efficient training through sample reuse:

```python
# Sample reuse workflow
def training_step_with_reuse(manager, config):
    if manager.should_collect_new_samples(config.group_size):
        # Collect fresh samples
        samples = expensive_sample_collection(...)
        manager = manager.reset_with_new_samples(samples)
        print("Collected fresh samples")
    else:
        # Reuse existing samples
        samples = manager.current_samples
        manager = manager.update_for_reuse()
        print(f"Reusing samples (iteration {manager.reuse_iteration})")
    
    return samples, manager
```

**Benefits:**
- Reduces computational cost of sample collection
- Maintains training stability through repeated use
- Configurable reuse iterations (typically 4)
- Automatic refresh when reuse limit reached

### Configurable Reward Scaling
Multiple reward scaling options:

```python
def apply_reward_scaling(rewards, method='tanh', clip_value=10.0):
    """Apply different reward scaling methods."""
    if method == 'tanh':
        return jnp.tanh(rewards / clip_value) * clip_value
    elif method == 'clip':
        return jnp.clip(rewards, -clip_value, clip_value)
    elif method == 'none':
        return rewards
    else:
        raise ValueError(f"Unknown scaling method: {method}")
```

### Configurable Advantage Scaling
Control advantage normalization:

```python
def compute_advantages(rewards, config):
    """Compute advantages with configurable scaling."""
    group_baseline = jnp.mean(rewards)
    advantages = rewards - group_baseline
    
    if config.scale_rewards:
        # Standard deviation normalization
        advantages = advantages / (jnp.std(advantages) + 1e-8)
    # else: use raw advantages without normalization
    
    return advantages
```

**Use Cases:**
- `scale_rewards=True`: Standard for most applications
- `scale_rewards=False`: When reward scales are already appropriate
- Helps with numerical stability and convergence

### Zero KL Penalty Default
Updated default based on recent findings:

```python
# Old default (problematic)
old_config = GRPOConfig(kl_penalty_coeff=0.1)

# New default (open-r1 recommendation)
new_config = GRPOConfig(kl_penalty_coeff=0.0)
```

**Rationale:**
- Recent research shows KL penalties can harm performance
- Group-based advantages provide sufficient regularization
- Zero penalty allows more aggressive policy updates
- Can still be manually set if needed

## Algorithm Implementation

### Core GRPO Loss
Literature-compliant implementation:

```python
def _compute_grpo_loss(params, batch_data, policy_network, config):
    """Compute GRPO loss with enhanced features."""
    states = batch_data['states']
    actions = batch_data['actions']
    rewards = batch_data['rewards']
    old_log_probs = batch_data['old_log_probs']
    
    # Forward pass through policy
    policy_outputs = jax.vmap(policy_network.apply, in_axes=(None, 0))(params, states)
    new_log_probs = compute_action_log_probs(policy_outputs, actions)
    
    # Group-based advantages (core GRPO innovation)
    group_baseline = jnp.mean(rewards)
    advantages = rewards - group_baseline
    
    # Configurable advantage scaling
    if config.scale_rewards:
        advantages = advantages / (jnp.std(advantages) + 1e-8)
    
    # Policy loss with clipping
    ratio = jnp.exp(new_log_probs - old_log_probs)
    clipped_ratio = jnp.clip(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))
    
    # Entropy regularization
    entropies = jax.vmap(compute_policy_entropy)(policy_outputs)
    entropy_loss = -jnp.mean(entropies)
    
    # KL penalty (typically 0.0)
    kl_divergence = jnp.mean(old_log_probs - new_log_probs)
    kl_penalty = config.kl_penalty_coeff * jnp.maximum(0.0, kl_divergence - 0.01)
    
    # Total loss (NO value network components)
    total_loss = policy_loss + config.entropy_coeff * entropy_loss + kl_penalty
    
    return total_loss, {
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
```

### Key Features
- **No Value Network**: Uses group mean as baseline
- **Group-based Advantages**: Stable for multi-objective rewards
- **Standard Deviation Normalization**: Numerical stability
- **Exact KL Divergence**: Proper KL(π_old || π_new) formula
- **Comprehensive Diagnostics**: All key training metrics

## Integration Patterns

### With Training Pipeline
```python
def enhanced_grpo_training_loop(config, initial_states, scms):
    """Training loop with enhanced GRPO features."""
    # Initialize components
    update_step, optimizer_init = create_grpo_trainer(policy_network, config)
    params = initialize_policy_parameters()
    opt_state = optimizer_init(params)
    manager = create_sample_reuse_manager(config)
    
    for step in range(config.total_steps):
        # Collect batch with reuse
        batch, manager = collect_grpo_batch_with_reuse(
            policy_network, params, initial_states, scms,
            surrogate_model, surrogate_params, config,
            reward_config, manager, keys[step]
        )
        
        # Update policy
        params, opt_state, update_info = update_step(params, opt_state, batch)
        
        # Monitor training
        if step % 100 == 0:
            log_training_progress(step, update_info, manager)
```

### With Curriculum Learning
```python
def curriculum_enhanced_grpo(curriculum_schedule):
    """GRPO training with curriculum learning."""
    for difficulty in curriculum_schedule:
        # Adjust configuration for difficulty
        config = adjust_grpo_config_for_difficulty(difficulty)
        
        # Generate appropriate SCMs
        scms = generate_scms_for_difficulty(difficulty)
        states = generate_initial_states(scms)
        
        # Train with enhanced GRPO
        enhanced_grpo_training_loop(config, states, scms)
```

### With Exploration Strategies
```python
def grpo_with_exploration(exploration_strategy):
    """Combine GRPO with exploration strategies."""
    def enhanced_batch_collection():
        # Base batch collection
        batch = collect_grpo_batch(...)
        
        # Add exploration bonuses
        exploration_bonuses = exploration_strategy.compute_bonuses(batch['states'])
        enhanced_rewards = batch['rewards'] + exploration_bonuses
        
        return {**batch, 'rewards': enhanced_rewards}
    
    return enhanced_batch_collection
```

## Performance Optimizations

### JAX Compilation
```python
# Compile key functions for performance
@jax.jit
def compiled_grpo_update(params, opt_state, batch_data):
    return update_step(params, opt_state, batch_data)

@jax.jit
def compiled_batch_collection(policy_params, states, keys):
    return jax.vmap(collect_single_sample)(policy_params, states, keys)
```

### Memory Efficiency
```python
# Efficient batch processing
def memory_efficient_grpo_update(params, opt_state, large_batch, chunk_size=16):
    """Process large batches in chunks for memory efficiency."""
    total_loss = 0.0
    total_info = {}
    
    for i in range(0, len(large_batch['states']), chunk_size):
        chunk = extract_batch_chunk(large_batch, i, chunk_size)
        params, opt_state, update_info = update_step(params, opt_state, chunk)
        
        # Accumulate results
        total_loss += update_info.total_loss
        accumulate_update_info(total_info, update_info)
    
    return params, opt_state, average_update_info(total_info)
```

## Troubleshooting

### Common Issues

**High Advantage Variance**
```python
if update_info.advantage_std > 3.0:
    print("High advantage variance detected")
    print("Solutions:")
    print("- Increase group_size")
    print("- Check reward scaling")
    print("- Verify diverse initial states")
```

**Policy Collapse**
```python
if update_info.mean_entropy < 0.1:
    print("Policy collapse detected")
    print("Solutions:")
    print("- Reduce learning_rate")
    print("- Increase entropy_coeff")
    print("- Check KL penalty")
```

**Training Instability**
```python
if update_info.grad_norm > 5.0:
    print("High gradient norm detected")
    print("Solutions:")
    print("- Reduce max_grad_norm")
    print("- Check reward scaling")
    print("- Verify numerical stability")
```

### Debugging Tools
```python
def debug_grpo_update(update_info):
    """Debug GRPO update for issues."""
    checks = {
        'advantages_normalized': abs(update_info.mean_advantage) < 0.1,
        'entropy_reasonable': 0.1 < update_info.mean_entropy < 3.0,
        'kl_bounded': update_info.approx_kl < 0.1,
        'gradients_stable': update_info.grad_norm < 2.0
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    return all(checks.values())
```

## Key Design Principles

### Literature Compliance
- Follows DeepSeek GRPO formulation exactly
- Group size of 64+ as recommended
- Proper advantage normalization by standard deviation
- Exact KL divergence computation

### Open-r1 Enhancements
- Sample reuse for training efficiency
- Configurable reward and advantage scaling
- Zero KL penalty default based on recent research
- Enhanced configuration validation

### Multi-Objective Optimization
- Group-based advantages handle conflicting rewards
- No value network complexity
- Stable training for dual objectives
- Comprehensive diagnostic information

### Training Efficiency
- Sample reuse reduces computational cost
- JAX compilation for performance
- Memory-efficient batch processing
- Robust numerical stability