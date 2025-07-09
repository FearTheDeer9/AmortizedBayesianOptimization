"""
Surrogate Model Integration Module

This module provides principled integration between surrogate models and the policy network,
solving the information loss problem in the Surrogate → Policy pipeline.

Key components:
- Bootstrap surrogate features from SCM structure
- Training phase management (Bootstrap → Transition → Trained)
- Structure encoding utilities for meaningful variable differentiation
- Transition scheduling between phases

This replaces the problematic constant default values with learned causal knowledge.
"""

from .bootstrap import (
    create_bootstrap_surrogate_features,
    BootstrapSurrogateOutputs,
    BootstrapConfig
)

from .structure_encoding import (
    encode_causal_structure,
    compute_structural_parent_probabilities
)

from .phase_manager import (
    PhaseConfig,
    TrainingPhase,
    get_current_phase,
    should_transition_phase
)

from .transition import (
    MixedSurrogateOutputs,
    TrainedSurrogateOutputs,
    get_mixed_surrogate_features,
    create_trained_surrogate_features
)

from .utils import (
    project_embeddings_to_mechanism_features,
    generate_surrogate_features
)

__all__ = [
    # Bootstrap
    'create_bootstrap_surrogate_features',
    'BootstrapSurrogateOutputs', 
    'BootstrapConfig',
    
    # Structure encoding
    'encode_causal_structure',
    'compute_structural_parent_probabilities',
    
    # Phase management
    'PhaseConfig',
    'TrainingPhase',
    'get_current_phase',
    'should_transition_phase',
    
    # Transition
    'MixedSurrogateOutputs',
    'TrainedSurrogateOutputs', 
    'get_mixed_surrogate_features',
    'create_trained_surrogate_features',
    
    # Utils
    'project_embeddings_to_mechanism_features',
    'generate_surrogate_features'
]