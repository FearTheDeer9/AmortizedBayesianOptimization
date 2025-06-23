"""
Acquisition module for ACBO.

This module provides the acquisition model using Group Relative Policy Optimization (GRPO)
for intelligent intervention selection based on structural uncertainty and optimization progress.
"""

from .state import (
    AcquisitionState,
)

from .services import (
    create_acquisition_state,
    update_state_with_intervention,
    create_acquisition_state_from_samples,
    validate_state_inputs,
    compute_state_delta,
)

from .utilities import (
    get_state_uncertainty_bits,
    get_state_optimization_progress,
    get_state_marginal_probabilities,
    get_state_exploration_coverage,
    get_state_best_value,
    get_state_target_variable,
    get_state_step,
    get_state_buffer_size,
    get_most_likely_parents,
    is_state_highly_uncertain,
    has_state_improved,
    compute_information_gain,
    get_state_summary_compact,
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

from .verifiable_rewards import (
    SimpleRewardComponents,
    target_improvement_reward,
    true_parent_intervention_reward,
    exploration_diversity_reward,
    compute_simple_verifiable_reward,
    validate_reward_consistency,
    create_reward_config,
    compute_adaptive_thresholds,
    create_adaptive_reward_config,
    compute_verifiable_reward_simple,
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

from .reward_rubric import (
    RewardComponent,
    CausalRewardRubric,
    RewardResult,
    create_training_rubric,
    create_deployment_rubric,
    create_ablation_rubric,
)

__all__ = [
    # Core state representation
    'AcquisitionState',
    
    # State creation and updating services
    'create_acquisition_state', 
    'update_state_with_intervention',
    'create_acquisition_state_from_samples',
    'validate_state_inputs',
    'compute_state_delta',
    
    # State utilities (pure functions)
    'get_state_uncertainty_bits',
    'get_state_optimization_progress',
    'get_state_marginal_probabilities',
    'get_state_exploration_coverage',
    'get_state_best_value',
    'get_state_target_variable',
    'get_state_step',
    'get_state_buffer_size',
    'get_most_likely_parents',
    'is_state_highly_uncertain',
    'has_state_improved',
    'compute_information_gain',
    'get_state_summary_compact',
    
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
    
    # Multi-component verifiable rewards (original system)
    'RewardComponents',
    'compute_verifiable_reward',
    'analyze_reward_trends',
    'validate_reward_config',
    'create_default_reward_config',
    
    # Simple ground-truth verifiable rewards (new binary system)
    'SimpleRewardComponents',
    'target_improvement_reward',
    'true_parent_intervention_reward',
    'exploration_diversity_reward',
    'compute_simple_verifiable_reward',
    'validate_reward_consistency',
    'create_reward_config',
    'compute_adaptive_thresholds',
    'create_adaptive_reward_config',
    'compute_verifiable_reward_simple',
    
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
    
    # Reward rubric system (modular reward composition)
    'RewardComponent',
    'CausalRewardRubric',
    'RewardResult',
    'create_training_rubric',
    'create_deployment_rubric',
    'create_ablation_rubric',
]
