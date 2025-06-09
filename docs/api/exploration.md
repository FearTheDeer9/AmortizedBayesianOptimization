# Exploration Strategies API Reference

## Overview
This module provides uncertainty-guided exploration strategies for the ACBO acquisition model. It implements both static uncertainty-guided exploration and adaptive exploration that balances between structure discovery and optimization objectives based on training progress and optimization stagnation.

## Core Types

### ExplorationConfig
```python
@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies."""
    
    # Uncertainty-guided exploration weights
    uncertainty_weight: float = 1.0         # Weight for overall posterior uncertainty
    count_weight: float = 0.1               # Weight for count-based exploration
    variable_uncertainty_weight: float = 0.5 # Weight for variable-specific uncertainty
    temperature: float = 1.0                # Temperature for exploration scaling
    
    # Adaptive exploration parameters
    initial_temperature: float = 2.0        # Starting temperature (high exploration)
    final_temperature: float = 0.1          # Final temperature (low exploration)
    adaptation_steps: int = 1000            # Steps to decay from initial to final
    stagnation_threshold: int = 100         # Steps before considering optimization stagnant
```

## Core Classes

### UncertaintyGuidedExploration
```python
class UncertaintyGuidedExploration:
    """
    Exploration strategy leveraging rich uncertainty infrastructure.
    
    Uses ParentSetPosterior uncertainty plus optimization progress 
    to guide exploration vs. exploitation decisions.
    """
```

Static exploration strategy that computes exploration bonuses for interventions based on:
- Expected information gain from the intervention
- Count-based frequency of previous interventions
- Variable-specific uncertainty in parent status

#### Methods

##### compute_exploration_bonus(state, candidate_intervention)
Compute total exploration bonus for a candidate intervention.

**Parameters:**
- `state: AcquisitionState` - Current acquisition state with uncertainty information
- `candidate_intervention: pyr.PMap` - Intervention to evaluate

**Returns:**
- `float` - Exploration bonus value (higher = more exploration incentive)

**Example:**
```python
config = ExplorationConfig(uncertainty_weight=1.0, temperature=1.0)
exploration = UncertaintyGuidedExploration(config)

# Create intervention
intervention = pyr.pmap({
    'type': 'perfect',
    'targets': frozenset(['X', 'Y']),
    'values': {'X': 1.0, 'Y': 2.0}
})

# Compute bonus
bonus = exploration.compute_exploration_bonus(state, intervention)
```

**Key Algorithm:**
- **Epistemic Bonus**: Predicts expected information gain based on intervening on variables with uncertain parent status (maximized at probability ~0.5)
- **Count Bonus**: Inversely proportional to intervention frequency `(1 - frequency)`
- **Variable Uncertainty**: Bonus for variables with marginal parent probabilities near 0.5
- **Temperature Scaling**: Final bonus divided by temperature for exploration control

### AdaptiveExploration
```python
class AdaptiveExploration:
    """
    Adaptive exploration balancing optimization and structure discovery.
    
    Adapts exploration based on both optimization progress and 
    structural uncertainty - our dual-objective advantage.
    """
```

Adaptive exploration that adjusts temperature based on training progress and optimization stagnation.

#### Methods

##### get_exploration_temperature(step, state)
Get current exploration temperature based on training progress and state.

**Parameters:**
- `step: int` - Current training step
- `state: AcquisitionState` - Current acquisition state

**Returns:**
- `float` - Exploration temperature (higher = more exploration)

**Algorithm:**
```python
# Linear decay from initial to final temperature
base_progress = min(step / adaptation_steps, 1.0)

# Adjust for optimization stagnation
if optimization is stagnating:
    increase temperature (more exploration)

temperature = initial_temp * (1 - progress) + final_temp * progress
```

##### should_explore(state, step)
Binary decision whether to prioritize exploration vs exploitation.

**Parameters:**
- `state: AcquisitionState` - Current acquisition state
- `step: int` - Current training step

**Returns:**
- `bool` - True if should prioritize exploration, False for exploitation

**Decision Logic:**
1. Always explore if uncertainty > 2.0 bits
2. Explore if optimization has stagnated (max improvement < 0.01 in last 5 steps)
3. Otherwise, use temperature-based probability

##### get_exploration_bonus_schedule(step)
Get exploration bonus multiplier based on training progress.

**Parameters:**
- `step: int` - Current training step

**Returns:**
- `float` - Exploration bonus multiplier (0 to 1)

## Factory Functions

### create_exploration_strategy(strategy_type, config=None, **kwargs)
Create exploration strategy instance.

**Parameters:**
- `strategy_type: str` - Type of strategy ("uncertainty_guided" or "adaptive")
- `config: Optional[ExplorationConfig]` - Configuration object
- `**kwargs` - Additional configuration parameters

**Returns:**
- Exploration strategy instance

**Example:**
```python
# Create with default config
strategy = create_exploration_strategy("uncertainty_guided")

# Create with custom config
config = ExplorationConfig(uncertainty_weight=2.0, temperature=0.5)
strategy = create_exploration_strategy("adaptive", config=config)

# Create with kwargs
strategy = create_exploration_strategy(
    "uncertainty_guided",
    uncertainty_weight=1.5,
    count_weight=0.2
)
```

## Utility Functions

### compute_exploration_value(exploration_strategy, state, intervention, step=0)
Compute exploration value for an intervention using given strategy.

**Parameters:**
- `exploration_strategy` - Strategy instance
- `state: AcquisitionState` - Current state
- `intervention: pyr.PMap` - Candidate intervention
- `step: int` - Current training step (for adaptive strategies)

**Returns:**
- `float` - Exploration value

### select_exploration_intervention(candidates, exploration_strategy, state, step=0, top_k=1)
Select top-k interventions for exploration from candidates.

**Parameters:**
- `candidates: List[pyr.PMap]` - List of candidate interventions
- `exploration_strategy` - Strategy to use
- `state: AcquisitionState` - Current state
- `step: int` - Current training step
- `top_k: int` - Number of interventions to select

**Returns:**
- `List[pyr.PMap]` - Top-k interventions ranked by exploration value

**Example:**
```python
# Generate candidate interventions
candidates = [
    create_perfect_intervention(['X'], {'X': 1.0}),
    create_perfect_intervention(['Y'], {'Y': 2.0}),
    create_perfect_intervention(['Z'], {'Z': 3.0})
]

# Select best exploration intervention
strategy = create_exploration_strategy("uncertainty_guided")
best = select_exploration_intervention(
    candidates, strategy, state, top_k=1
)[0]
```

### balance_exploration_exploitation(exploration_value, exploitation_value, exploration_strategy, state, step=0, alpha=0.5)
Balance exploration and exploitation values.

**Parameters:**
- `exploration_value: float` - Value from exploration perspective
- `exploitation_value: float` - Value from exploitation perspective
- `exploration_strategy` - Strategy for adaptive weighting
- `state: AcquisitionState` - Current state
- `step: int` - Current training step
- `alpha: float` - Base weighting between exploration (α) and exploitation (1-α)

**Returns:**
- `float` - Combined value balancing both objectives

**Algorithm:**
```python
# Adaptive weighting based on should_explore decision
if should_explore:
    effective_alpha = min(1.0, alpha + 0.3)  # Increase exploration
else:
    effective_alpha = max(0.0, alpha - 0.3)  # Increase exploitation

return effective_alpha * exploration_value + (1 - effective_alpha) * exploitation_value
```

## Key Algorithms

### Expected Information Gain (Epistemic Bonus)
The epistemic bonus now correctly predicts intervention-specific information gain:

```python
def _compute_epistemic_bonus(state, intervention):
    total_expected_gain = 0.0
    for var in intervention_targets:
        parent_prob = state.marginal_parent_probs.get(var, 0.0)
        
        # Maximum information gain when prob ~= 0.5
        uncertainty_factor = 1.0 - 2.0 * abs(parent_prob - 0.5)
        
        # Scale by overall posterior uncertainty
        expected_gain = uncertainty_factor * state.uncertainty_bits
        total_expected_gain += expected_gain
    
    return total_expected_gain / len(intervention_targets)
```

### Variable Uncertainty Calculation
Variables with marginal parent probabilities near 0.5 are most informative:

```python
# For each variable in intervention
uncertainty = 1.0 - 2.0 * abs(prob - 0.5)

# Examples:
# prob = 0.5 -> uncertainty = 1.0 (maximum)
# prob = 0.0 -> uncertainty = 0.0 (minimum)
# prob = 1.0 -> uncertainty = 0.0 (minimum)
# prob = 0.3 -> uncertainty = 0.6
```

### Temperature Scheduling
Adaptive temperature decay with stagnation adjustment:

```python
# Base decay over training
base_progress = min(step / adaptation_steps, 1.0)

# Reduce progress (increase temperature) if stagnating
if stagnation_steps > 0:
    stagnation_bonus = min(stagnation_steps / threshold, 0.5)
    base_progress = max(0.0, base_progress - stagnation_bonus)

# Linear interpolation
temperature = initial_temp * (1 - base_progress) + final_temp * base_progress
```

## Integration with ACBO

### State Requirements
The exploration strategies expect `AcquisitionState` to provide:
- `uncertainty_bits: float` - Overall posterior uncertainty in bits
- `marginal_parent_probs: Dict[str, float]` - Marginal parent probabilities per variable
- `buffer: ExperienceBuffer` - For count-based exploration
- `optimization_stagnation_steps: int` - (Optional) For adaptive exploration
- `recent_target_improvements: List[float]` - (Optional) For stagnation detection

### Usage in Training Pipeline
```python
# Initialize exploration strategy
exploration = create_exploration_strategy(
    "adaptive",
    initial_temperature=2.0,
    final_temperature=0.1
)

# During training
for step in range(num_steps):
    # Get current state
    state = create_acquisition_state(...)
    
    # Generate candidate interventions
    candidates = generate_intervention_candidates(...)
    
    # Use exploration to select intervention
    if exploration.should_explore(state, step):
        # Pure exploration
        intervention = select_exploration_intervention(
            candidates, exploration, state, step
        )[0]
    else:
        # Balance exploration and exploitation
        intervention = select_best_with_exploration_bonus(
            candidates, exploitation_scores, exploration, state, step
        )
```

## Performance Characteristics

- **Computational Complexity**: O(n) for n candidate interventions
- **Memory Usage**: Minimal - only stores configuration
- **Integration Overhead**: Negligible compared to neural network inference

## Design Rationale

### Fixed Design Flaws
1. **Epistemic Bonus**: Now correctly computes intervention-specific expected information gain rather than using state-only uncertainty
2. **Temperature Scheduling**: Removed hacky dummy state, now computes directly from step and config

### Key Innovations
1. **Expected Information Gain**: Variables with uncertain parent status (prob ~0.5) provide maximum information when intervened upon
2. **Stagnation Detection**: Automatically increases exploration when optimization plateaus
3. **Dual-Objective Balance**: Adaptive strategies aware of both structure learning and optimization progress

## Examples

### Basic Uncertainty-Guided Exploration
```python
# Setup
config = ExplorationConfig(
    uncertainty_weight=1.0,
    count_weight=0.1,
    temperature=1.0
)
exploration = UncertaintyGuidedExploration(config)

# Evaluate intervention
bonus = exploration.compute_exploration_bonus(state, intervention)
```

### Adaptive Exploration with Stagnation
```python
# Setup adaptive exploration
adaptive = AdaptiveExploration(
    initial_temperature=2.0,
    final_temperature=0.1,
    adaptation_steps=1000
)

# Check exploration decision
if adaptive.should_explore(state, step=500):
    print("Exploring due to high uncertainty or stagnation")
else:
    print("Exploiting - optimization is progressing well")
```

### Multi-Intervention Ranking
```python
# Rank multiple interventions by exploration value
strategy = create_exploration_strategy("uncertainty_guided")
top_3 = select_exploration_intervention(
    candidates, strategy, state, top_k=3
)
```

## Troubleshooting

### Low Exploration
- Increase `uncertainty_weight` or `temperature`
- Check if `marginal_parent_probs` are being computed correctly
- Verify that `uncertainty_bits` reflects true posterior uncertainty

### Over-Exploration
- Decrease `initial_temperature` or increase `adaptation_steps`
- Reduce `uncertainty_weight` to rely more on count-based exploration
- Check stagnation detection thresholds

### Stagnation Not Detected
- Ensure `optimization_stagnation_steps` is being updated in state
- Verify `recent_target_improvements` contains actual improvement values
- Adjust `stagnation_threshold` based on problem characteristics