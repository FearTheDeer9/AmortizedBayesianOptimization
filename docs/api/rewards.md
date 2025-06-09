# Verifiable Rewards API Reference

## Overview
The Verifiable Rewards module provides multi-component reward computation for dual-objective causal Bayesian optimization. The system balances structure learning and target optimization through decomposed, verifiable reward components that require no human feedback.

## Core Types

### RewardComponents
```python
@dataclass(frozen=True)
class RewardComponents:
    """Decomposed reward components for dual-objective learning."""
    
    optimization_reward: float        # Target variable improvement
    structure_discovery_reward: float # Information gain in structure learning
    parent_intervention_reward: float # Bonus for intervening on likely parents
    exploration_bonus: float          # Diversity encouragement
    total_reward: float              # Weighted combination
    metadata: Dict[str, Any]         # Additional context
```

Immutable reward decomposition showing how different objectives contribute to the total reward signal.

## Core Functions

### compute_verifiable_reward(state_before, intervention, outcome, state_after, config)
Compute comprehensive reward for a causal intervention.

**Parameters:**
- `state_before: AcquisitionState` - State before intervention
- `intervention: Sample` - Intervention that was applied
- `outcome: Sample` - Observed outcome from intervention
- `state_after: AcquisitionState` - State after incorporating outcome
- `config: Dict[str, Any]` - Reward configuration parameters

**Returns:**
`RewardComponents` - Decomposed reward breakdown

**Example:**
```python
# Configure reward weights
reward_config = {
    'target_variable': 'Y',
    'reward_weights': {
        'optimization': 1.0,    # Primary objective
        'structure': 0.5,       # Secondary objective
        'parent': 0.3,         # Learning guidance
        'exploration': 0.1     # Diversity maintenance
    },
    'optimization_baseline': 0.0,
    'normalization_method': 'tanh'
}

# Compute reward for intervention
reward_components = compute_verifiable_reward(
    state_before=old_state,
    intervention=intervention,
    outcome=outcome,
    state_after=new_state,
    config=reward_config
)

print(f"Optimization reward: {reward_components.optimization_reward:.3f}")
print(f"Structure reward: {reward_components.structure_discovery_reward:.3f}")
print(f"Total reward: {reward_components.total_reward:.3f}")
```

### create_default_reward_config(target_variable, optimization_baseline=0.0)
Create default reward configuration for dual-objective optimization.

**Parameters:**
- `target_variable: str` - Variable being optimized
- `optimization_baseline: float` - Baseline value for improvement measurement

**Returns:**
`Dict[str, Any]` - Default reward configuration

**Example:**
```python
config = create_default_reward_config(
    target_variable='Y',
    optimization_baseline=0.0
)

print("Default weights:")
for component, weight in config['reward_weights'].items():
    print(f"  {component}: {weight}")
```

### validate_reward_config(config)
Validate reward configuration for correctness.

**Parameters:**
- `config: Dict[str, Any]` - Reward configuration to validate

**Returns:**
`Dict[str, bool]` - Validation results

**Example:**
```python
validation = validate_reward_config(config)

if not all(validation.values()):
    print("Configuration validation failed:")
    for check, passed in validation.items():
        if not passed:
            print(f"  {check}: FAILED")
else:
    print("Configuration is valid")
```

## Reward Component Details

### Optimization Reward
Measures improvement in the target variable value:

```python
def _compute_optimization_reward(state_before, outcome, target_variable, baseline=0.0):
    """Reward based on target variable improvement."""
    target_value = outcome['values'][target_variable]
    current_best = state_before.best_target_value
    
    # Improvement over current best
    improvement = target_value - max(current_best, baseline)
    
    # Tanh normalization for bounded rewards
    return float(jnp.tanh(improvement))
```

**Key Features:**
- Measures actual improvement in target variable
- Normalized using tanh for stability
- Compares against both baseline and current best
- Critical for optimization objectives

### Structure Discovery Reward
Rewards information gain about causal structure:

```python
def _compute_structure_discovery_reward(posterior_before, posterior_after):
    """Reward based on uncertainty reduction in posterior."""
    if posterior_before.target_variable != posterior_after.target_variable:
        return 0.0
    
    # Information gain = uncertainty reduction
    uncertainty_reduction = posterior_before.uncertainty - posterior_after.uncertainty
    
    # Normalize by maximum possible reduction
    max_uncertainty = posterior_before.uncertainty
    if max_uncertainty > 0:
        normalized_gain = uncertainty_reduction / max_uncertainty
        return float(jnp.clip(normalized_gain, 0.0, 1.0))
    else:
        return 0.0
```

**Key Features:**
- Based on information theory (uncertainty reduction)
- Uses posterior entropy from ParentSetPosterior
- Normalized to [0, 1] range for stability
- Encourages interventions that reduce structural uncertainty

### Parent Intervention Reward
Provides bonus for intervening on likely causal parents:

```python
def _compute_parent_intervention_reward(intervention, marginal_parent_probs):
    """Reward for intervening on variables likely to be parents."""
    if intervention['type'] != 'perfect':
        return 0.0
    
    targets = intervention['targets']
    if not targets:
        return 0.0
    
    # Average parent probability of intervened variables
    target_probs = [marginal_parent_probs.get(var, 0.0) for var in targets]
    avg_parent_prob = sum(target_probs) / len(target_probs)
    
    return float(avg_parent_prob)
```

**Key Features:**
- Uses marginal parent probabilities from posterior
- Guides policy toward informative interventions
- Helps with structure learning during training
- Reduces random exploration

### Exploration Bonus
Encourages diverse intervention strategies:

```python
def _compute_exploration_bonus(intervention, buffer, weight=0.1):
    """Bonus for exploring under-sampled intervention types."""
    targets = intervention['targets']
    
    # Count previous interventions on these targets
    previous_count = len(buffer.filter_interventions_by_targets(targets))
    total_interventions = buffer.num_interventions()
    
    if total_interventions == 0:
        return weight
    
    # Inverse frequency bonus
    frequency = previous_count / total_interventions
    bonus = weight * (1.0 - frequency)
    
    return float(jnp.clip(bonus, 0.0, weight))
```

**Key Features:**
- Prevents mode collapse in intervention selection
- Based on intervention frequency in buffer
- Inversely proportional to how often targets were used
- Configurable weight for exploration intensity

## Reward Analysis Functions

### analyze_reward_trends(reward_history)
Analyze reward trends over training for insights and debugging.

**Parameters:**
- `reward_history: List[RewardComponents]` - History of reward components

**Returns:**
`Dict[str, Any]` - Analysis including:
- `'component_trends'`: Trend direction for each component
- `'correlation_matrix'`: Correlations between components
- `'dominance_analysis'`: Which components dominate over time
- `'balance_metrics'`: How well objectives are balanced

**Example:**
```python
# Analyze reward trends over training
analysis = analyze_reward_trends(reward_history)

print("Component trends:")
for component, trend in analysis['component_trends'].items():
    direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
    print(f"  {component}: {direction} {abs(trend):.3f}")

print(f"Optimization dominance: {analysis['dominance_analysis']['optimization']:.2%}")
print(f"Structure dominance: {analysis['dominance_analysis']['structure']:.2%}")
```

### decompose_reward_sources(reward_components, state_before, state_after)
Decompose reward sources for detailed analysis.

**Parameters:**
- `reward_components: RewardComponents` - Computed rewards
- `state_before: AcquisitionState` - Pre-intervention state
- `state_after: AcquisitionState` - Post-intervention state

**Returns:**
`Dict[str, Any]` - Detailed breakdown of reward sources

**Example:**
```python
breakdown = decompose_reward_sources(reward_components, old_state, new_state)

print("Reward breakdown:")
print(f"  Target improvement: {breakdown['target_improvement']:.3f}")
print(f"  Uncertainty reduction: {breakdown['uncertainty_reduction']:.3f}")
print(f"  Parent targeting accuracy: {breakdown['parent_accuracy']:.3f}")
print(f"  Exploration novelty: {breakdown['exploration_novelty']:.3f}")
```

## Configuration Options

### Reward Weights
Control the relative importance of different objectives:

```python
reward_weights = {
    'optimization': 1.0,     # Target variable improvement (primary)
    'structure': 0.5,        # Structure learning (secondary)
    'parent': 0.3,          # Parent intervention bonus (guidance)
    'exploration': 0.1      # Exploration diversity (regularization)
}
```

### Normalization Methods
Different approaches for reward normalization:

```python
normalization_options = {
    'tanh': 'Bounded sigmoid normalization',
    'clip': 'Hard clipping to [-1, 1]',
    'standardize': 'Z-score normalization',
    'none': 'No normalization applied'
}
```

### Exploration Settings
Configure exploration bonus computation:

```python
exploration_config = {
    'exploration_weight': 0.1,          # Base exploration bonus
    'frequency_decay': 'inverse',        # How frequency affects bonus
    'novelty_threshold': 0.05,          # Threshold for "novel" interventions
    'diversity_metric': 'targets'       # What constitutes diversity
}
```

## Integration Patterns

### With GRPO Training
```python
# Reward computation in GRPO batch collection
def collect_grpo_batch_with_rewards(policy, params, states, scms, reward_config, key):
    batch_rewards = []
    
    for state, scm in zip(states, scms):
        # Select and apply intervention
        intervention = select_intervention(policy, params, state, key)
        outcome = apply_intervention(scm, intervention)
        
        # Update state and compute reward
        new_state = update_state(state, intervention, outcome)
        reward_components = compute_verifiable_reward(
            state, intervention, outcome, new_state, reward_config
        )
        
        batch_rewards.append(reward_components.total_reward)
    
    return jnp.array(batch_rewards)
```

### With Training Pipeline
```python
# Adaptive reward weights during training
def adaptive_reward_weighting(step, initial_weights, adaptation_schedule):
    """Adjust reward weights based on training progress."""
    progress = min(step / adaptation_schedule['total_steps'], 1.0)
    
    # Gradually shift from exploration to exploitation
    exploration_weight = initial_weights['exploration'] * (1.0 - progress)
    optimization_weight = initial_weights['optimization'] * (1.0 + 0.5 * progress)
    
    return {
        **initial_weights,
        'exploration': exploration_weight,
        'optimization': optimization_weight
    }
```

### With Curriculum Learning
```python
# Curriculum-based reward adjustment
def curriculum_reward_config(difficulty_level, base_config):
    """Adjust reward configuration based on curriculum difficulty."""
    if difficulty_level == 'easy':
        # More guidance for structure learning
        weights = {**base_config['reward_weights'], 'parent': 0.5}
    elif difficulty_level == 'hard':
        # Focus on optimization performance
        weights = {**base_config['reward_weights'], 'optimization': 1.5}
    else:
        weights = base_config['reward_weights']
    
    return {**base_config, 'reward_weights': weights}
```

## Performance Considerations

### Computational Complexity
- Optimization reward: O(1) - simple value comparison
- Structure discovery: O(1) - pre-computed posterior entropies  
- Parent intervention: O(k) - k is number of intervention targets
- Exploration bonus: O(n) - n is number of previous interventions

### Memory Usage
- RewardComponents: Small fixed-size dataclass
- Reward history: Linear in number of training steps
- Configuration: Negligible memory footprint

### Numerical Stability
- All rewards normalized to reasonable ranges
- Tanh normalization prevents extreme values
- Clipping ensures bounded rewards
- Special handling for edge cases (zero variance, empty posteriors)

## Common Usage Patterns

### Basic Reward Computation
```python
# Standard dual-objective reward
config = create_default_reward_config('Y')
reward_components = compute_verifiable_reward(
    state_before, intervention, outcome, state_after, config
)

print(f"Total reward: {reward_components.total_reward:.3f}")
```

### Custom Reward Configuration
```python
# Custom weights for specific application
custom_config = {
    'target_variable': 'Y',
    'reward_weights': {
        'optimization': 2.0,    # Prioritize optimization
        'structure': 0.2,       # Reduce structure learning
        'parent': 0.1,         # Minimal guidance
        'exploration': 0.05    # Low exploration
    },
    'normalization_method': 'clip',
    'exploration_weight': 0.05
}

reward_components = compute_verifiable_reward(
    state_before, intervention, outcome, state_after, custom_config
)
```

### Training Analysis
```python
# Analyze reward trends during training
reward_history = []
for step in range(training_steps):
    # ... training step ...
    reward_history.append(reward_components)
    
    if step % 100 == 0:
        analysis = analyze_reward_trends(reward_history[-100:])
        print(f"Step {step} - Balance: {analysis['balance_metrics']['total_balance']:.3f}")
```

## Key Design Principles

### Verifiability
- All rewards computed from observable quantities
- No human feedback or subjective evaluation required
- Deterministic and reproducible across runs
- Clear mathematical basis for each component

### Multi-Objective Balance
- Explicit trade-off control through configurable weights
- Components measured in compatible scales
- Natural balance between exploration and exploitation
- Adaptable to different problem requirements

### Training Stability
- Bounded reward ranges prevent gradient explosion
- Smooth reward surfaces aid optimization
- Proper normalization handles different scales
- Robust to edge cases and numerical issues