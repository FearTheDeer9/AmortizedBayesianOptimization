"""
Acquisition module for ACBO.

This module provides the acquisition model using Group Relative Policy Optimization (GRPO)
for intelligent intervention selection based on structural uncertainty and optimization progress.
"""

from .state import (
    AcquisitionState,
    create_acquisition_state,
    update_state_with_intervention,
    get_state_uncertainty_bits,
    get_state_optimization_progress,
    get_state_marginal_probabilities,
)

from .trajectory import (
    TrajectoryStep,
    TrajectoryBuffer,
    create_trajectory_buffer,
    create_trajectory_step,
    migrate_experience_to_trajectory_buffer,
)

from .policy import (
    PolicyConfig,
    AlternatingAttentionEncoder,
    AcquisitionPolicyNetwork,
    create_acquisition_policy,
    sample_intervention_from_policy,
    compute_action_log_probability,
    compute_policy_entropy,
    analyze_policy_output,
    validate_policy_output,
)

from .rewards import (
    RewardComponents,
    compute_verifiable_reward,
    analyze_reward_trends,
    validate_reward_config,
    create_default_reward_config,
)

from .grpo import (
    GRPOConfig,
    GRPOUpdate,
    SampleReuseManager,
    create_grpo_trainer,
    collect_grpo_batch,
    collect_grpo_batch_with_reuse,
    create_grpo_batch_from_samples,
    create_grpo_batch_from_buffer,
    create_sample_reuse_manager,
)

from .exploration import (
    ExplorationConfig,
    UncertaintyGuidedExploration,
    AdaptiveExploration,
    create_exploration_strategy,
    compute_exploration_value,
    select_exploration_intervention,
    balance_exploration_exploitation,
)

__all__ = [
    # Core state representation
    'AcquisitionState',
    'create_acquisition_state', 
    'update_state_with_intervention',
    
    # State utilities
    'get_state_uncertainty_bits',
    'get_state_optimization_progress',
    'get_state_marginal_probabilities',
    
    # Trajectory storage for RL training
    'TrajectoryStep',
    'TrajectoryBuffer',
    'create_trajectory_buffer',
    'create_trajectory_step',
    'migrate_experience_to_trajectory_buffer',
    
    # Policy network with alternating attention
    'PolicyConfig',
    'AlternatingAttentionEncoder',
    'AcquisitionPolicyNetwork',
    'create_acquisition_policy',
    'sample_intervention_from_policy',
    'compute_action_log_probability',
    'compute_policy_entropy',
    'analyze_policy_output',
    'validate_policy_output',
    
    # Multi-component verifiable rewards
    'RewardComponents',
    'compute_verifiable_reward',
    'analyze_reward_trends',
    'validate_reward_config',
    'create_default_reward_config',
    
    # GRPO algorithm for RL training (enhanced with open-r1 features)
    'GRPOConfig',
    'GRPOUpdate',
    'SampleReuseManager',
    'create_grpo_trainer',
    'collect_grpo_batch',
    'collect_grpo_batch_with_reuse',
    'create_grpo_batch_from_samples',
    'create_grpo_batch_from_buffer',
    'create_sample_reuse_manager',
    
    # Exploration strategies
    'ExplorationConfig',
    'UncertaintyGuidedExploration',
    'AdaptiveExploration',
    'create_exploration_strategy',
    'compute_exploration_value',
    'select_exploration_intervention',
    'balance_exploration_exploitation',
]
