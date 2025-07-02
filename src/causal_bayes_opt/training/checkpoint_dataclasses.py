"""
Checkpoint dataclasses for modular training system.

This module contains the checkpoint dataclass definitions extracted from
the deprecated modular_components.py file. These dataclasses are used
by the checkpoint management system for serializing training state.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class TrainingCheckpoint:
    """Base class for training checkpoints."""
    component_type: str
    difficulty_level: str
    training_step: int
    timestamp: float
    config: Dict[str, Any]
    performance_metrics: Dict[str, float]
    checkpoint_path: str


@dataclass
class SurrogateCheckpoint(TrainingCheckpoint):
    """Checkpoint for surrogate model training."""
    surrogate_params: Any
    surrogate_config: Dict[str, Any]
    expert_demo_stats: Dict[str, Any]
    
    def __post_init__(self):
        self.component_type = "surrogate"


@dataclass
class AcquisitionCheckpoint(TrainingCheckpoint):
    """Checkpoint for acquisition model training."""
    policy_params: Any
    optimizer_state: Any
    policy_config: Dict[str, Any]
    training_history: List[Dict[str, Any]]
    is_surrogate_aware: bool = False
    surrogate_checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        self.component_type = "acquisition"


@dataclass
class JointCheckpoint(TrainingCheckpoint):
    """Checkpoint for joint training."""
    surrogate_params: Any
    policy_params: Any
    surrogate_optimizer_state: Any
    policy_optimizer_state: Any
    joint_config: Dict[str, Any]
    joint_training_history: List[Dict[str, Any]]
    
    def __post_init__(self):
        self.component_type = "joint"


# Export checkpoint classes
__all__ = [
    'TrainingCheckpoint',
    'SurrogateCheckpoint',
    'AcquisitionCheckpoint',
    'JointCheckpoint'
]