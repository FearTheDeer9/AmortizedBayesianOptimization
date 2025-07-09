"""
Training Phase Management

This module manages the three-phase training approach:
1. Bootstrap: Use SCM structure knowledge (steps 0 to N)
2. Transition: Gradually switch to trained surrogate (steps N to N+M)  
3. Trained: Use fully trained surrogate model (steps N+M+)

Provides configuration and scheduling utilities for smooth phase transitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
import jax.numpy as jnp


class TrainingPhase(Enum):
    """Training phase enumeration."""
    BOOTSTRAP = "bootstrap"
    TRANSITION = "transition"
    TRAINED = "trained"


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for three-phase training approach."""
    bootstrap_steps: int = 100
    transition_steps: int = 50
    exploration_noise_start: float = 0.5
    exploration_noise_end: float = 0.1
    transition_schedule: str = "linear"  # "linear", "cosine", "sigmoid"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.bootstrap_steps < 0:
            raise ValueError("bootstrap_steps must be non-negative")
        if self.transition_steps < 0:
            raise ValueError("transition_steps must be non-negative")
        if not (0.0 <= self.exploration_noise_start <= 1.0):
            raise ValueError("exploration_noise_start must be in [0, 1]")
        if not (0.0 <= self.exploration_noise_end <= 1.0):
            raise ValueError("exploration_noise_end must be in [0, 1]")
        if self.transition_schedule not in ["linear", "cosine", "sigmoid"]:
            raise ValueError("transition_schedule must be 'linear', 'cosine', or 'sigmoid'")
    
    @property
    def total_steps(self) -> int:
        """Total steps before fully trained phase."""
        return self.bootstrap_steps + self.transition_steps


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap surrogate features."""
    structure_encoding_dim: int = 128
    use_graph_distance: bool = True
    use_structural_priors: bool = True
    noise_schedule: str = "exponential_decay"  # "exponential_decay", "linear_decay", "cosine_decay"
    min_noise_factor: float = 0.1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.structure_encoding_dim <= 0:
            raise ValueError("structure_encoding_dim must be positive")
        if self.noise_schedule not in ["exponential_decay", "linear_decay", "cosine_decay"]:
            raise ValueError("noise_schedule must be 'exponential_decay', 'linear_decay', or 'cosine_decay'")
        if not (0.0 <= self.min_noise_factor <= 1.0):
            raise ValueError("min_noise_factor must be in [0, 1]")


def get_current_phase(step: int, config: PhaseConfig) -> TrainingPhase:
    """
    Determine current training phase based on step number.
    
    Args:
        step: Current training step
        config: Phase configuration
        
    Returns:
        Current training phase
        
    Example:
        >>> config = PhaseConfig(bootstrap_steps=100, transition_steps=50)
        >>> get_current_phase(50, config)
        TrainingPhase.BOOTSTRAP
        >>> get_current_phase(125, config)
        TrainingPhase.TRANSITION
        >>> get_current_phase(200, config)
        TrainingPhase.TRAINED
    """
    if step < config.bootstrap_steps:
        return TrainingPhase.BOOTSTRAP
    elif step < config.bootstrap_steps + config.transition_steps:
        return TrainingPhase.TRANSITION
    else:
        return TrainingPhase.TRAINED


def should_transition_phase(
    current_step: int, 
    config: PhaseConfig,
    surrogate_ready: bool = False
) -> bool:
    """
    Determine if we should transition to the next phase.
    
    Args:
        current_step: Current training step
        config: Phase configuration
        surrogate_ready: Whether surrogate model is ready for transition
        
    Returns:
        True if should transition to next phase
    """
    current_phase = get_current_phase(current_step, config)
    
    if current_phase == TrainingPhase.BOOTSTRAP:
        # Transition to TRANSITION phase when bootstrap period ends
        return current_step >= config.bootstrap_steps
    elif current_phase == TrainingPhase.TRANSITION:
        # Transition to TRAINED phase when transition period ends AND surrogate is ready
        transition_complete = current_step >= config.bootstrap_steps + config.transition_steps
        return transition_complete and surrogate_ready
    else:
        # Already in TRAINED phase
        return False


def compute_exploration_factor(
    step: int, 
    config: PhaseConfig,
    bootstrap_config: BootstrapConfig
) -> float:
    """
    Compute exploration factor for bootstrap surrogate features.
    
    During bootstrap phase, exploration factor decreases from high to low values,
    allowing the policy to explore initially but become more structured over time.
    
    Args:
        step: Current training step
        config: Phase configuration
        bootstrap_config: Bootstrap configuration
        
    Returns:
        Exploration factor in [min_noise_factor, exploration_noise_start]
        
    Example:
        >>> config = PhaseConfig(bootstrap_steps=100, exploration_noise_start=0.5)
        >>> bootstrap_config = BootstrapConfig(min_noise_factor=0.1)
        >>> compute_exploration_factor(0, config, bootstrap_config)
        0.5
        >>> compute_exploration_factor(100, config, bootstrap_config)
        0.1
    """
    if step >= config.bootstrap_steps:
        # Past bootstrap phase - minimum noise
        return bootstrap_config.min_noise_factor
    
    # Progress through bootstrap phase
    progress = step / max(1, config.bootstrap_steps)
    
    # Apply noise schedule
    if bootstrap_config.noise_schedule == "exponential_decay":
        # Exponential decay: fast initial drop, then slow
        decay_factor = jnp.exp(-3.0 * progress)  # -3.0 gives reasonable decay curve
    elif bootstrap_config.noise_schedule == "linear_decay":
        # Linear decay
        decay_factor = 1.0 - progress
    elif bootstrap_config.noise_schedule == "cosine_decay":
        # Cosine decay: smooth S-curve
        decay_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    else:
        # Default to linear
        decay_factor = 1.0 - progress
    
    # Interpolate between start and end noise levels
    noise_range = config.exploration_noise_start - bootstrap_config.min_noise_factor
    exploration_factor = bootstrap_config.min_noise_factor + noise_range * decay_factor
    
    return float(jnp.clip(exploration_factor, bootstrap_config.min_noise_factor, config.exploration_noise_start))


def compute_transition_weight(
    step: int, 
    config: PhaseConfig
) -> Dict[str, float]:
    """
    Compute interpolation weights for bootstrap → trained transition.
    
    Args:
        step: Current training step
        config: Phase configuration
        
    Returns:
        Dictionary with 'bootstrap_weight' and 'trained_weight' (sum to 1.0)
        
    Example:
        >>> config = PhaseConfig(bootstrap_steps=100, transition_steps=50)
        >>> weights = compute_transition_weight(125, config)  # Mid-transition
        >>> weights
        {'bootstrap_weight': 0.5, 'trained_weight': 0.5}
    """
    if step < config.bootstrap_steps:
        # Pure bootstrap phase
        return {'bootstrap_weight': 1.0, 'trained_weight': 0.0}
    elif step >= config.bootstrap_steps + config.transition_steps:
        # Pure trained phase
        return {'bootstrap_weight': 0.0, 'trained_weight': 1.0}
    else:
        # Transition phase - interpolate
        transition_progress = (step - config.bootstrap_steps) / max(1, config.transition_steps)
        
        # Apply transition schedule
        if config.transition_schedule == "linear":
            trained_weight = transition_progress
        elif config.transition_schedule == "cosine":
            # Smooth S-curve transition
            trained_weight = 0.5 * (1.0 - jnp.cos(jnp.pi * transition_progress))
        elif config.transition_schedule == "sigmoid":
            # Sigmoid transition (slower at start/end, faster in middle)
            sigmoid_input = 12.0 * (transition_progress - 0.5)  # Scale to [-6, 6]
            trained_weight = 1.0 / (1.0 + jnp.exp(-sigmoid_input))
        else:
            # Default to linear
            trained_weight = transition_progress
        
        trained_weight = float(jnp.clip(trained_weight, 0.0, 1.0))
        bootstrap_weight = 1.0 - trained_weight
        
        return {
            'bootstrap_weight': bootstrap_weight,
            'trained_weight': trained_weight
        }


def get_phase_metadata(
    step: int, 
    config: PhaseConfig,
    bootstrap_config: BootstrapConfig,
    surrogate_ready: bool = False
) -> Dict[str, Any]:
    """
    Get comprehensive metadata about current training phase.
    
    Args:
        step: Current training step
        config: Phase configuration
        bootstrap_config: Bootstrap configuration  
        surrogate_ready: Whether surrogate model is ready
        
    Returns:
        Dictionary with phase information and transition parameters
    """
    current_phase = get_current_phase(step, config)
    exploration_factor = compute_exploration_factor(step, config, bootstrap_config)
    transition_weights = compute_transition_weight(step, config)
    
    metadata = {
        'step': step,
        'phase': current_phase.value,
        'exploration_factor': exploration_factor,
        'bootstrap_weight': transition_weights['bootstrap_weight'],
        'trained_weight': transition_weights['trained_weight'],
        'surrogate_ready': surrogate_ready,
        'should_transition': should_transition_phase(step, config, surrogate_ready),
        'bootstrap_progress': min(1.0, step / max(1, config.bootstrap_steps)),
        'transition_progress': max(0.0, min(1.0, (step - config.bootstrap_steps) / max(1, config.transition_steps))),
        'total_progress': min(1.0, step / max(1, config.total_steps))
    }
    
    # Add phase-specific metadata
    if current_phase == TrainingPhase.BOOTSTRAP:
        metadata.update({
            'steps_remaining_in_phase': max(0, config.bootstrap_steps - step),
            'next_phase': TrainingPhase.TRANSITION.value
        })
    elif current_phase == TrainingPhase.TRANSITION:
        metadata.update({
            'steps_remaining_in_phase': max(0, config.bootstrap_steps + config.transition_steps - step),
            'next_phase': TrainingPhase.TRAINED.value if surrogate_ready else 'waiting_for_surrogate'
        })
    else:
        metadata.update({
            'steps_remaining_in_phase': 0,
            'next_phase': 'none'
        })
    
    return metadata


def validate_phase_transition(
    old_phase: TrainingPhase,
    new_phase: TrainingPhase,
    step: int,
    config: PhaseConfig
) -> bool:
    """
    Validate that a phase transition is valid.
    
    Args:
        old_phase: Previous training phase
        new_phase: New training phase
        step: Current step
        config: Phase configuration
        
    Returns:
        True if transition is valid
        
    Raises:
        ValueError: If transition is invalid
    """
    # Check that phases follow proper sequence
    valid_transitions = {
        TrainingPhase.BOOTSTRAP: [TrainingPhase.TRANSITION],
        TrainingPhase.TRANSITION: [TrainingPhase.TRAINED],
        TrainingPhase.TRAINED: []  # No transitions from trained phase
    }
    
    if new_phase not in valid_transitions.get(old_phase, []):
        raise ValueError(f"Invalid phase transition: {old_phase.value} → {new_phase.value}")
    
    # Check step-based constraints
    expected_phase = get_current_phase(step, config)
    if new_phase != expected_phase:
        raise ValueError(
            f"Step {step} should be in {expected_phase.value} phase, not {new_phase.value}"
        )
    
    return True