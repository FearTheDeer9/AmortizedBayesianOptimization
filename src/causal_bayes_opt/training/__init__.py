"""
Training Infrastructure for ACBO

This module implements complete training infrastructure for the ACBO system,
including expert demonstration collection, surrogate training, and acquisition
model training using functional programming principles and 2024 research findings.

Key Components:
- Configuration system with comprehensive validation
- Expert demonstration collection using PARENT_SCALE  
- Surrogate model training with JAX optimization (250-3,386x speedup)
- Acquisition model training with enhanced GRPO and verifiable rewards
- Training pipelines following functional programming principles
"""

# Configuration system
from .config import (
    TrainingConfig,
    GRPOTrainingConfig,
    RewardTrainingConfig,
    ExplorationTrainingConfig,
    SurrogateTrainingConfig,
    EvaluationConfig,
    create_training_config,
    create_default_training_config,
    validate_training_config
)

# Surrogate training system
from .surrogate_training import (
    TrainingExample,
    TrainingBatch,
    TrainingBatchJAX,
    TrainingMetrics,
    ValidationResults,
    kl_divergence_loss,
    uncertainty_weighted_loss,
    calibrated_loss,
    multi_target_loss,
    kl_divergence_loss_jax,
    uncertainty_weighted_loss_jax,
    convert_to_jax_batch,
    create_jax_surrogate_train_step,
    create_adaptive_train_step,
    train_surrogate_model,
    run_loss_function_experiment,
    validate_surrogate_performance
)

# Decoupled surrogate training interface
try:
    from .surrogate_trainer import (
        SurrogateTrainer,
        SurrogateTrainingResults,
        load_expert_demonstrations_from_path,
        convert_demonstrations_to_training_batches
    )
    _SURROGATE_TRAINER_AVAILABLE = True
except ImportError:
    _SURROGATE_TRAINER_AVAILABLE = False

# Acquisition training system
try:
    from .acquisition_training import (
        AcquisitionGRPOConfig,
        BehavioralCloningConfig,
        AcquisitionTrainingConfig,
        TrainingResults,
        train_acquisition_model,
        behavioral_cloning_phase,
        grpo_fine_tuning_phase
    )
    from .acquisition_config import (
        VerifiableRewardConfig,
        EnhancedGRPOConfig,
        PolicyNetworkConfig,
        DataConfig,
        TrainingConfig as AcquisitionTrainingConfigComplete,
        create_standard_config,
        create_high_performance_config,
        create_memory_efficient_config,
        validate_config_compatibility,
        get_recommended_config_for_problem_size
    )
    _ACQUISITION_TRAINING_AVAILABLE = True
except ImportError:
    _ACQUISITION_TRAINING_AVAILABLE = False

# Expert demonstration collection (optional import to avoid dependency issues)
try:
    from .expert_demonstration_collection import (
        ExpertDemonstration,
        ExpertTrajectoryDemonstration,
        DemonstrationBatch,
        ExpertDemonstrationCollector,
        collect_expert_demonstrations_main
    )
    _EXPERT_COLLECTION_AVAILABLE = True
except ImportError:
    _EXPERT_COLLECTION_AVAILABLE = False

__all__ = [
    # Configuration system
    "TrainingConfig",
    "GRPOTrainingConfig", 
    "RewardTrainingConfig",
    "ExplorationTrainingConfig",
    "SurrogateTrainingConfig",
    "EvaluationConfig",
    "create_training_config",
    "create_default_training_config",
    "validate_training_config"
]

# Add surrogate training to exports
__all__.extend([
    # Surrogate training
    "TrainingExample",
    "TrainingBatch",
    "TrainingBatchJAX",
    "TrainingMetrics", 
    "ValidationResults",
    "kl_divergence_loss",
    "uncertainty_weighted_loss",
    "calibrated_loss",
    "multi_target_loss",
    "kl_divergence_loss_jax",
    "uncertainty_weighted_loss_jax",
    "convert_to_jax_batch",
    "create_jax_surrogate_train_step",
    "create_adaptive_train_step",
    "train_surrogate_model",
    "run_loss_function_experiment",
    "validate_surrogate_performance"
])

# Add decoupled surrogate trainer if available
if _SURROGATE_TRAINER_AVAILABLE:
    __all__.extend([
        "SurrogateTrainer",
        "SurrogateTrainingResults",
        "load_expert_demonstrations_from_path",
        "convert_demonstrations_to_training_batches"
    ])

# Add acquisition training if available
if _ACQUISITION_TRAINING_AVAILABLE:
    __all__.extend([
        # Acquisition training
        "AcquisitionGRPOConfig",
        "BehavioralCloningConfig", 
        "AcquisitionTrainingConfig",
        "TrainingResults",
        "train_acquisition_model",
        "behavioral_cloning_phase",
        "grpo_fine_tuning_phase",
        # Acquisition configuration
        "VerifiableRewardConfig",
        "EnhancedGRPOConfig",
        "PolicyNetworkConfig",
        "DataConfig",
        "AcquisitionTrainingConfigComplete",
        "create_standard_config",
        "create_high_performance_config", 
        "create_memory_efficient_config",
        "validate_config_compatibility",
        "get_recommended_config_for_problem_size"
    ])

# Async training infrastructure (Phase 2.2)
try:
    from .async_training import (
        AsyncTrainingConfig,
        TrainingBatch as AsyncTrainingBatch,
        TrainingProgress,
        AsyncTrainingManager,
        create_async_training_manager,
        estimate_batch_memory_usage,
        optimize_batch_size_for_memory
    )
    from .diversity_monitor import (
        DiversityMetrics,
        DiversityAlert,
        DiversityMonitor,
        create_diversity_monitor
    )
    __all__.extend([
        # Async training
        "AsyncTrainingConfig",
        "AsyncTrainingBatch",
        "TrainingProgress",
        "AsyncTrainingManager",
        "create_async_training_manager",
        "estimate_batch_memory_usage",
        "optimize_batch_size_for_memory",
        # Diversity monitoring
        "DiversityMetrics",
        "DiversityAlert",
        "DiversityMonitor",
        "create_diversity_monitor"
    ])
    _ASYNC_TRAINING_AVAILABLE = True
except ImportError:
    _ASYNC_TRAINING_AVAILABLE = False

# Add expert collection if available
if _EXPERT_COLLECTION_AVAILABLE:
    __all__.extend([
        "ExpertDemonstration",
        "ExpertTrajectoryDemonstration",
        "DemonstrationBatch",
        "ExpertDemonstrationCollector", 
        "collect_expert_demonstrations_main"
    ])

# GRPO Core Algorithm (Phase 2.2)
try:
    from .grpo_core import (
        GRPOConfig,
        GRPOTrajectory,
        GRPOUpdateResult,
        compute_gae_advantages,
        compute_simple_advantages,
        normalize_advantages,
        compute_policy_loss,
        compute_value_loss,
        compute_value_loss_jit,
        compute_value_loss_clipped_jit,
        compute_entropy_loss,
        create_grpo_update_fn,
        create_trajectory_from_experiences,
        validate_grpo_config,
        create_default_grpo_config,
        create_high_performance_grpo_config,
        create_exploration_grpo_config,
    )
    __all__.extend([
        # GRPO Core Algorithm
        "GRPOConfig",
        "GRPOTrajectory", 
        "GRPOUpdateResult",
        "compute_gae_advantages",
        "compute_simple_advantages",
        "normalize_advantages",
        "compute_policy_loss",
        "compute_value_loss",
        "compute_value_loss_jit",
        "compute_value_loss_clipped_jit",
        "compute_entropy_loss",
        "create_grpo_update_fn",
        "create_trajectory_from_experiences",
        "validate_grpo_config",
        "create_default_grpo_config",
        "create_high_performance_grpo_config",
        "create_exploration_grpo_config",
    ])
    _GRPO_CORE_AVAILABLE = True
except ImportError:
    _GRPO_CORE_AVAILABLE = False

# Experience Management System (Phase 2.2)
try:
    from .experience_management import (
        ExperienceConfig,
        Experience,
        ExperienceBatch,
        ExperienceManager,
        SumTree,
        create_experience_manager,
        create_high_capacity_experience_manager,
        create_memory_efficient_experience_manager,
    )
    __all__.extend([
        # Experience Management
        "ExperienceConfig",
        "Experience",
        "ExperienceBatch",
        "ExperienceManager",
        "SumTree",
        "create_experience_manager",
        "create_high_capacity_experience_manager",
        "create_memory_efficient_experience_manager",
    ])
    _EXPERIENCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    _EXPERIENCE_MANAGEMENT_AVAILABLE = False

# GRPO Configuration System (Phase 2.2)
try:
    from .grpo_config import (
        TrainingMode,
        OptimizationLevel,
        PolicyNetworkConfig,
        ValueNetworkConfig,
        CurriculumConfig,
        AdaptiveConfig,
        CheckpointingConfig,
        LoggingConfig,
        ComprehensiveGRPOConfig,
        validate_comprehensive_grpo_config,
        create_standard_grpo_config,
        create_research_grpo_config,
        create_production_grpo_config,
        create_debug_grpo_config,
        get_recommended_config_for_problem_size,
    )
    __all__.extend([
        # GRPO Configuration System
        "TrainingMode",
        "OptimizationLevel",
        "PolicyNetworkConfig",
        "ValueNetworkConfig",
        "CurriculumConfig",
        "AdaptiveConfig",
        "CheckpointingConfig",
        "LoggingConfig",
        "ComprehensiveGRPOConfig",
        "validate_comprehensive_grpo_config",
        "create_standard_grpo_config",
        "create_research_grpo_config",
        "create_production_grpo_config",
        "create_debug_grpo_config",
        "get_recommended_config_for_problem_size",
    ])
    _GRPO_CONFIG_AVAILABLE = True
except ImportError:
    _GRPO_CONFIG_AVAILABLE = False

# GRPO Training Manager (Phase 2.2)
try:
    from .grpo_training_manager import (
        TrainingStep,
        TrainingSession,
        GRPOTrainingManager,
        create_grpo_training_manager,
        create_debug_training_manager,
        create_production_training_manager,
    )
    __all__.extend([
        # GRPO Training Manager
        "TrainingStep",
        "TrainingSession",
        "GRPOTrainingManager",
        "create_grpo_training_manager",
        "create_debug_training_manager",
        "create_production_training_manager",
    ])
    _GRPO_TRAINING_MANAGER_AVAILABLE = True
except ImportError:
    _GRPO_TRAINING_MANAGER_AVAILABLE = False

# Master training orchestrator and curriculum learning
try:
    from .master_trainer import (
        TrainingState,
        MasterTrainingResults,
        MasterTrainer,
        create_master_trainer,
        run_complete_acbo_training
    )
    from .curriculum import (
        DifficultyLevel,
        CurriculumStage,
        CurriculumConfig,
        CurriculumManager,
        create_default_curriculum_config,
        create_curriculum_manager,
        generate_curriculum_scms
    )
    __all__.extend([
        # Master trainer
        "TrainingState",
        "MasterTrainingResults", 
        "MasterTrainer",
        "create_master_trainer",
        "run_complete_acbo_training",
        # Curriculum learning
        "DifficultyLevel",
        "CurriculumStage",
        "CurriculumConfig", 
        "CurriculumManager",
        "create_default_curriculum_config",
        "create_curriculum_manager",
        "generate_curriculum_scms"
    ])
    _MASTER_TRAINER_AVAILABLE = True
except ImportError as e:
    _MASTER_TRAINER_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Master trainer not available: {e}")