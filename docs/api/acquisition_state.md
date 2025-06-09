# Acquisition State API Reference

## Overview
The Acquisition State module provides rich state representations for reinforcement learning-based intervention selection. The `AcquisitionState` dataclass combines structural uncertainty from parent set posteriors with optimization progress tracking, enabling intelligent dual-objective decision making.

## Core Types

### AcquisitionState
```python
@dataclass(frozen=True)
class AcquisitionState:
    """Rich state representation for dual-objective RL-based acquisition."""
    
    # Core state components
    parent_posterior: ParentSetPosterior
    buffer_statistics: BufferStatistics  
    optimization_target: str
    best_target_value: float
    intervention_history: List[Sample]
    
    # Derived uncertainty metrics
    uncertainty_bits: float
    marginal_parent_probs: Dict[str, float]
    
    # Progress tracking
    step: int
    metadata: Dict[str, Any]
```

An immutable state representation that combines:
- **Structural uncertainty** from ParentSetPosterior 
- **Optimization progress** tracking target variable improvement
- **Experience statistics** from ExperienceBuffer
- **Intervention history** for pattern recognition

## Core Functions

### create_acquisition_state(samples, parent_posterior, target_variable, variable_order)
Create an AcquisitionState from current data and posterior predictions.

**Parameters:**
- `samples: List[Sample]` - Current observational and interventional data
- `parent_posterior: ParentSetPosterior` - Posterior over parent sets for target
- `target_variable: str` - Variable being optimized
- `variable_order: List[str]` - Consistent variable ordering

**Returns:**
`AcquisitionState` - Immutable state representation

**Example:**
```python
# Create state from current data
samples = [obs_sample1, int_sample2, ...]
posterior = predict_parent_posterior(model, params, data, vars, target)

state = create_acquisition_state(
    samples=samples,
    parent_posterior=posterior, 
    target_variable='Y',
    variable_order=['X', 'Y', 'Z']
)

print(f"Uncertainty: {state.uncertainty_bits:.2f} bits")
print(f"Best target value: {state.best_target_value:.3f}")
```

### update_state_with_intervention(state, intervention, outcome, new_posterior)
Create new state after applying an intervention and observing outcome.

**Parameters:**
- `state: AcquisitionState` - Current state
- `intervention: Sample` - Intervention that was applied
- `outcome: Sample` - Observed outcome from intervention
- `new_posterior: ParentSetPosterior` - Updated posterior after new data

**Returns:**
`AcquisitionState` - New state incorporating the intervention result

**Example:**
```python
# Apply intervention and update state
intervention = create_perfect_intervention(targets={'X'}, values={'X': 2.0})
outcome = sample_with_intervention(scm, intervention)[0]
new_posterior = predict_parent_posterior(model, params, updated_data, vars, target)

new_state = update_state_with_intervention(
    state=current_state,
    intervention=intervention,
    outcome=outcome, 
    new_posterior=new_posterior
)

# Check improvement
improvement = new_state.best_target_value - current_state.best_target_value
print(f"Target improvement: {improvement:.3f}")
```

### get_state_uncertainty_bits(state)
Extract uncertainty in bits from the state.

**Parameters:**
- `state: AcquisitionState` - State to analyze

**Returns:**
`float` - Uncertainty in bits (entropy / log(2))

**Example:**
```python
uncertainty = get_state_uncertainty_bits(state)
print(f"Current uncertainty: {uncertainty:.2f} bits")

# High uncertainty suggests more exploration needed
if uncertainty > 2.0:
    print("High uncertainty - prioritize exploration")
```

### get_state_optimization_progress(state)
Compute optimization progress metrics from state.

**Parameters:**
- `state: AcquisitionState` - State to analyze

**Returns:**
`Dict[str, float]` - Progress metrics including:
- `'best_value'`: Current best target value
- `'improvement'`: Improvement from baseline
- `'stagnation_steps'`: Steps without improvement
- `'progress_rate'`: Rate of improvement

**Example:**
```python
progress = get_state_optimization_progress(state)
print(f"Best value: {progress['best_value']:.3f}")
print(f"Improvement: {progress['improvement']:.3f}")
print(f"Stagnation: {progress['stagnation_steps']} steps")
```

### get_state_marginal_probabilities(state, variables)
Get marginal parent probabilities for specified variables.

**Parameters:**
- `state: AcquisitionState` - State containing posterior
- `variables: List[str]` - Variables to get probabilities for

**Returns:**
`Dict[str, float]` - Mapping from variable to marginal parent probability

**Example:**
```python
variables = ['X', 'Y', 'Z']
marginals = get_state_marginal_probabilities(state, variables)

for var, prob in marginals.items():
    print(f"P({var} is parent of {state.optimization_target}) = {prob:.3f}")
    
# Find most uncertain variable (prob closest to 0.5)
uncertainties = {var: 1.0 - 2.0 * abs(prob - 0.5) 
                for var, prob in marginals.items()}
most_uncertain = max(uncertainties, key=uncertainties.get)
print(f"Most uncertain variable: {most_uncertain}")
```

## State Analysis Functions

### analyze_state_composition(state)
Analyze the composition and quality of current state.

**Parameters:**
- `state: AcquisitionState` - State to analyze

**Returns:**
`Dict[str, Any]` - Analysis including:
- `'data_composition'`: Breakdown of observational vs interventional data
- `'uncertainty_breakdown'`: Sources of uncertainty
- `'optimization_metrics'`: Progress tracking
- `'exploration_coverage'`: Intervention diversity

**Example:**
```python
analysis = analyze_state_composition(state)

print("Data composition:")
for key, value in analysis['data_composition'].items():
    print(f"  {key}: {value}")
    
print(f"Uncertainty breakdown: {analysis['uncertainty_breakdown']}")
print(f"Exploration coverage: {analysis['exploration_coverage']:.2%}")
```

### compare_states(state_before, state_after)
Compare two states to analyze intervention effects.

**Parameters:**
- `state_before: AcquisitionState` - State before intervention
- `state_after: AcquisitionState` - State after intervention

**Returns:**
`Dict[str, float]` - Comparison metrics including:
- `'uncertainty_change'`: Change in uncertainty (bits)
- `'optimization_improvement'`: Change in target value
- `'information_gain'`: Information gained about structure
- `'exploration_progress'`: Progress in exploration coverage

**Example:**
```python
comparison = compare_states(old_state, new_state)

print(f"Uncertainty change: {comparison['uncertainty_change']:.3f} bits")
print(f"Target improvement: {comparison['optimization_improvement']:.3f}")
print(f"Information gain: {comparison['information_gain']:.3f}")

# Assess intervention quality
if comparison['information_gain'] > 0.1:
    print("Good intervention - significant information gain")
```

## Integration with Other Components

### With ExperienceBuffer
```python
# Create state from buffer
buffer = ExperienceBuffer()
# ... add samples to buffer ...

buffer_stats = buffer.get_statistics()
samples = buffer.get_samples()

state = create_acquisition_state(
    samples=samples,
    parent_posterior=posterior,
    target_variable='Y',
    variable_order=['X', 'Y', 'Z']
)
```

### With Policy Networks
```python
# Use state as input to policy
policy_output = policy_network.apply(params, state)
intervention = sample_intervention_from_policy(policy_output, state, key)
```

### With Reward Computation
```python
# Use state in reward computation
reward_components = compute_verifiable_reward(
    state_before=old_state,
    intervention=intervention,
    outcome=outcome,
    state_after=new_state,
    config=reward_config
)
```

## Key Design Principles

### Immutability
- All state objects are immutable for thread safety and predictability
- State updates create new objects rather than modifying existing ones
- Enables efficient caching and memoization of derived properties

### Rich Context
- Combines multiple information sources into single coherent representation
- Provides both raw data and derived metrics for decision making
- Supports both optimization and structure learning objectives

### Efficient Computation
- Pre-computes expensive derived properties on creation
- Provides fast access to frequently needed metrics
- Optimized for use in training loops and policy networks

## Performance Considerations

- State creation is O(n) in number of samples for statistics computation
- Marginal probability computation is O(k) in number of parent sets
- Uncertainty computation is O(1) - pre-computed from posterior
- Memory usage scales with intervention history length (configurable)

## Common Usage Patterns

### Training Loop Integration
```python
for step in range(training_steps):
    # Create current state
    state = create_acquisition_state(samples, posterior, target, variables)
    
    # Policy selects intervention
    intervention = select_intervention(policy, state, key)
    
    # Apply intervention and observe outcome
    outcome = apply_intervention_and_observe(scm, intervention)
    
    # Update state and continue
    new_posterior = update_posterior(model, samples + [outcome])
    new_state = update_state_with_intervention(state, intervention, outcome, new_posterior)
```

### Analysis and Debugging
```python
# Analyze state quality
analysis = analyze_state_composition(state)
print(f"Data quality: {analysis['data_composition']}")

# Track optimization progress
progress = get_state_optimization_progress(state)
if progress['stagnation_steps'] > 10:
    print("Consider increasing exploration")

# Monitor uncertainty reduction
if state.uncertainty_bits < 0.5:
    print("Low uncertainty - focus on optimization")
```