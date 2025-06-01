# ADR 003: Verifiable Rewards for GRPO

## Context

Reinforcement learning with LLMs often relies on human feedback or reward models that can be noisy, subjective, or expensive. Our causal discovery and optimization domain offers a unique opportunity: we can define objective, verifiable rewards that don't require human annotation.

## Decision

Implement a verifiable reward mechanism for GRPO that combines structure discovery and optimization objectives.

## Reward Components

1. **Structure Discovery Reward**: Based on information gain in the posterior over parent sets
   - Higher reward when uncertainty about true parents decreases
   - Can be directly calculated from posterior distributions

2. **Optimization Reward**: Based on improvement in the target variable value
   - Higher reward when interventions lead to better target values
   - Directly observable from SCM outcomes

3. **Parent Intervention Reward**: During training, directly rewards intervening on true parents
   - Provides a reliable learning signal when ground truth is available
   - Can be approximated using the posterior during deployment

## Reward Formula

The combined reward is a weighted sum:
```
R(s_t, a_t, s_{t+1}) = α * R_opt(s_t, a_t, s_{t+1}) + β * R_struct(s_t, a_t, s_{t+1}) + γ * R_parent(s_t, a_t)
```

Where:
- `R_opt` measures optimization improvement
- `R_struct` measures information gain in structural understanding
- `R_parent` rewards actions intervening on true parents
- `α`, `β`, `γ` are hyperparameters controlling the relative importance

## Consequences

### Positive
- Eliminates need for human feedback or learned reward models
- Provides more reliable, consistent training signal
- Allows for principled multi-objective optimization
- Enables faster convergence with less data

### Negative
- Requires domain knowledge to design appropriate rewards
- May need careful balancing of different reward components
- Structure discovery reward may be less informative with very large graphs