## File 5: `docs/api/acquisition_model.md`

```markdown
# Acquisition Model API

## Overview

The acquisition model selects interventions based on the current posterior distribution, using Group Relative Policy Optimization (GRPO) for reinforcement learning.

## Core Data Structures

```python
@dataclass(frozen=True)
class State:
    posterior: ParentSetPosterior
    buffer: ExperienceBuffer
    best_value: float
    
@dataclass(frozen=True)
class AcquisitionParams:
    """Immutable parameters for acquisition model"""
    policy_params: Any
## Core Functions
### policy_network
pythondef policy_network(
    state: State,
    params: AcquisitionParams
) -> Distribution[Intervention]:
    """
    Output distribution over interventions given current state
    
    Args:
        state: Current state including posterior and history
        params: Model parameters for policy network
        
    Returns:
        Distribution over possible interventions
    """
### compute_reward
pythondef compute_reward(
    state: State,
    intervention: Intervention,
    outcome: Sample,
    next_posterior: ParentSetPosterior,
    true_parent_set: Optional[FrozenSet[str]] = None
) -> float:
    """
    Compute reward based on information gain and outcome improvement
    
    Args:
        state: Current state before intervention
        intervention: Selected intervention
        outcome: Observed outcome from intervention
        next_posterior: Updated posterior after intervention
        true_parent_set: Optional ground truth (for training)
        
    Returns:
        reward: Combined reward for structure discovery and optimization
    """
### update_acquisition
pythondef update_acquisition(
    params: AcquisitionParams,
    trajectories: List[Tuple[State, Intervention, float, State]],
    learning_rate: float
) -> AcquisitionParams:
    """
    Update acquisition model using GRPO
    
    Args:
        params: Current model parameters
        trajectories: Collected state-action-reward-state transitions
        learning_rate: Learning rate for parameter updates
        
    Returns:
        AcquisitionParams: Updated model parameters
    """
### GRPO Implementation
pythondef compute_grpo_advantage(
    rewards: List[float]
) -> List[float]:
    """
    Compute advantages using group relative normalization
    
    Args:
        rewards: List of rewards for a group of interventions
        
    Returns:
        advantages: Normalized advantages for each intervention
    """
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards) + 1e-8  # Avoid division by zero
    
    return [(r - mean_reward) / std_reward for r in rewards]