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

# Add expert collection if available
if _EXPERT_COLLECTION_AVAILABLE:
    __all__.extend([
        "ExpertDemonstration",
        "ExpertTrajectoryDemonstration",
        "DemonstrationBatch",
        "ExpertDemonstrationCollector", 
        "collect_expert_demonstrations_main"
    ])

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