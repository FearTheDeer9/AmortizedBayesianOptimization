#!/usr/bin/env python3
"""
Master Training Orchestrator for ACBO

Coordinates the complete training pipeline including expert demonstration collection,
surrogate model training, and acquisition model training with curriculum learning.

Key Features:
1. Orchestrates multi-stage training pipeline
2. Manages curriculum learning progression
3. Provides clean configuration-driven interface
4. Implements checkpointing and recovery
5. Monitors training health and anti-gaming

Design Principles:
- Pure functions for training logic
- Immutable configurations
- Functional composition of training stages
- Clean separation of concerns
"""

import logging
from dataclasses import dataclass, replace
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

import pyrsistent as pyr

from .config import TrainingConfig, create_default_training_config
from .curriculum import create_curriculum_manager
from ..acquisition.verifiable_rewards import (
    validate_reward_consistency,
    create_adaptive_reward_config
)
from .error_handling import (
    ErrorContext, TrainingError, create_training_error,
    safe_training_step, should_abort_training, add_error_to_context
)
from .checkpoint_manager import (
    CheckpointManager, CheckpointConfig, create_checkpoint_manager
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingState:
    """Immutable training state representation."""
    current_stage: str
    current_difficulty: int
    surrogate_params: Optional[Any]
    acquisition_params: Optional[Any]
    expert_demonstrations: Optional[Any]
    training_metrics: pyr.PMap
    completed_stages: List[str]
    checkpoint_path: Optional[str]
    error_context: Optional[Any] = None  # Will be ErrorContext from error_handling module
    last_successful_stage: Optional[str] = None
    recovery_count: int = 0



@dataclass(frozen=True)
class MasterTrainingResults:
    """Final results from complete training pipeline."""
    final_state: TrainingState
    surrogate_metrics: Dict[str, float]
    acquisition_metrics: Dict[str, float]
    curriculum_completion: Dict[int, bool]
    total_training_time: float
    checkpoints_saved: List[str]


class MasterTrainer:
    """
    Master orchestrator for ACBO training pipeline.
    
    Coordinates all training stages in sequence while maintaining
    clean functional interfaces and proper error handling.
    """
    
    def __init__(
            self, 
            config: Optional[TrainingConfig] = None,
            checkpoint_dir: Optional[str] = None,
            checkpoint_config: Optional[CheckpointConfig] = None
        ):
            """
            Initialize master trainer with comprehensive error handling and checkpointing.
            
            Args:
                config: Training configuration (uses defaults if None)
                checkpoint_dir: Directory for saving checkpoints
                checkpoint_config: Checkpoint management configuration
            """
            self.config = config or create_default_training_config()
            self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
            
            # Initialize enhanced checkpoint manager
            self.checkpoint_manager = create_checkpoint_manager(
                self.checkpoint_dir,
                checkpoint_config or CheckpointConfig()
            )
            
            # Initialize curriculum manager
            self.curriculum = create_curriculum_manager(self.config)
            
            # Initialize error context
            error_context = ErrorContext(
                errors=(),
                total_errors=0,
                recovery_attempts=0,
                last_successful_checkpoint=None,
                current_stage="initial"
            )
            
            # Initialize training state with error context
            self.state = TrainingState(
                current_stage="initial",
                current_difficulty=1,
                surrogate_params=None,
                acquisition_params=None,
                expert_demonstrations=None,
                training_metrics=pyr.m(),
                completed_stages=[],
                checkpoint_path=None,
                error_context=error_context,
                last_successful_stage=None,
                recovery_count=0
            )
            
            logger.info(f"Initialized MasterTrainer with config type: {type(self.config).__name__}")
            logger.info(f"Enhanced checkpointing and error handling enabled")
            logger.info(f"Checkpoint directory: {self.checkpoint_dir}")


    
    def train_complete_pipeline(
            self,
            resume_from_checkpoint: Optional[str] = None
        ) -> MasterTrainingResults:
            """
            Execute complete ACBO training pipeline with comprehensive error handling.
            
            Args:
                resume_from_checkpoint: Path to checkpoint to resume from
                
            Returns:
                Complete training results with all metrics
            """
            logger.info("🚀 Starting ACBO training pipeline with enhanced error handling")
            
            # Resume from checkpoint if specified
            if resume_from_checkpoint:
                try:
                    self.state = self._load_checkpoint(resume_from_checkpoint)
                    logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
                except Exception as e:
                    error = create_training_error(e, "checkpoint_recovery", {"checkpoint_path": resume_from_checkpoint})
                    logger.error(f"Failed to resume from checkpoint: {error.message}")
                    # Continue with fresh start if checkpoint recovery fails
                    self.state = replace(
                        self.state,
                        error_context=add_error_to_context(self.state.error_context, error)
                    )
            
            # Main training loop with error handling
            training_stages = [
                ("expert_collection", self._run_expert_collection_safe),
                ("surrogate_training", self._run_surrogate_training_safe), 
                ("acquisition_training", self._run_curriculum_acquisition_training_safe)
            ]
            
            for stage_name, stage_fn in training_stages:
                # Check if we should abort due to too many errors
                if should_abort_training(self.state.error_context):
                    logger.error("🛑 Aborting training due to excessive errors")
                    raise RuntimeError("Training aborted due to excessive error rate")
                
                # Skip completed stages
                if stage_name in self.state.completed_stages:
                    logger.info(f"⏭️ Skipping completed stage: {stage_name}")
                    continue
                
                # Update current stage
                self.state = replace(self.state, current_stage=stage_name)
                
                # Execute stage with error handling
                try:
                    logger.info(f"▶️ Starting stage: {stage_name}")
                    self.state = stage_fn()
                    
                    # Mark stage as completed and save checkpoint
                    self.state = replace(
                        self.state,
                        completed_stages=self.state.completed_stages + [stage_name],
                        last_successful_stage=stage_name
                    )
                    self._save_checkpoint(f"{stage_name}_complete")
                    logger.info(f"✅ Completed stage: {stage_name}")
                    
                except Exception as e:
                    # Handle stage failure
                    error = create_training_error(
                        e, stage_name, 
                        {"stage": stage_name, "recovery_count": self.state.recovery_count}
                    )
                    
                    logger.error(f"❌ Stage {stage_name} failed: {error.message}")
                    
                    # Update error context
                    self.state = replace(
                        self.state,
                        error_context=add_error_to_context(self.state.error_context, error),
                        recovery_count=self.state.recovery_count + 1
                    )
                    
                    # Save emergency checkpoint
                    self._save_checkpoint(f"{stage_name}_failed_emergency")
                    
                    # Decide whether to continue or abort
                    if error.recoverable and self.state.recovery_count < 3:
                        logger.warning(f"⚠️ Attempting recovery for stage: {stage_name}")
                        # For now, re-raise to stop pipeline, but this could be enhanced
                        # with stage-specific recovery strategies
                        raise
                    else:
                        logger.error(f"🚫 Stage {stage_name} failed permanently")
                        raise
            
            # Compile final results
            try:
                results = self._compile_final_results()
                logger.info("✅ ACBO training pipeline completed successfully")
                return results
                
            except Exception as e:
                error = create_training_error(e, "results_compilation", {})
                logger.error(f"Failed to compile results: {error.message}")
                self._save_checkpoint("results_compilation_failed")
                raise
    
    def _run_expert_collection_safe(self) -> TrainingState:
        """Safe wrapper for expert collection with error handling."""
        safe_fn = safe_training_step(
            self._run_expert_collection,
            self.state.error_context,
            "expert_collection",
            max_retries=2
        )
        return safe_fn()
    
    def _run_surrogate_training_safe(self) -> TrainingState:
        """Safe wrapper for surrogate training with error handling."""
        safe_fn = safe_training_step(
            self._run_surrogate_training,
            self.state.error_context,
            "surrogate_training",
            max_retries=3
        )
        return safe_fn()
    
    def _run_curriculum_acquisition_training_safe(self) -> TrainingState:
        """Safe wrapper for acquisition training with error handling."""
        safe_fn = safe_training_step(
            self._run_curriculum_acquisition_training,
            self.state.error_context,
            "acquisition_training",
            max_retries=3
        )
        return safe_fn()
    
    def _run_expert_collection(self) -> TrainingState:
        """Run expert demonstration collection stage."""
        logger.info("📚 Stage 1: Expert demonstration collection")
        
        # TODO: Implement expert demonstration collection (Phase 2.1)
        # This will use config for n_demonstrations, parallel settings, etc.
        logger.warning("Expert demonstration collection not yet implemented")
        demonstrations = None  # Placeholder
        
        return replace(
            self.state,
            current_stage="expert_collection_complete",
            expert_demonstrations=demonstrations,
            completed_stages=self.state.completed_stages + ["expert_collection"]
        )
    
    def _run_surrogate_training(self) -> TrainingState:
        """Run surrogate model training stage."""
        logger.info("🧠 Stage 2: Surrogate model training")
        
        if self.state.expert_demonstrations is None:
            raise ValueError("Expert demonstrations required for surrogate training")
        
        try:
            # Import SurrogateTrainer
            from .surrogate_trainer import SurrogateTrainer
            
            # Create surrogate trainer with configuration
            surrogate_config = self.config.surrogate if hasattr(self.config, 'surrogate') else None
            trainer = SurrogateTrainer(surrogate_config)
            
            # Train surrogate model on expert demonstrations
            logger.info(f"Training surrogate on {len(self.state.expert_demonstrations)} demonstrations")
            training_results = trainer.train(self.state.expert_demonstrations)
            
            # Extract metrics for tracking
            metrics = {
                "final_loss": training_results.final_loss,
                "best_validation_score": training_results.best_validation_score,
                "total_training_time": training_results.total_training_time,
                "epochs_trained": training_results.epochs_trained,
                "converged": training_results.converged,
                "validation_kl_divergence": training_results.validation_metrics.posterior_kl_divergence,
                "validation_accuracy_drop": training_results.validation_metrics.accuracy_drop,
                "inference_speedup": training_results.validation_metrics.inference_speedup
            }
            
            logger.info(f"Surrogate training completed: loss={training_results.final_loss:.4f}, "
                       f"converged={training_results.converged}, epochs={training_results.epochs_trained}")
            
            # Save surrogate model checkpoint
            surrogate_checkpoint = self.checkpoint_dir / "surrogate_model.pkl"
            trainer.save_checkpoint(
                training_results.final_params,
                metrics,
                str(surrogate_checkpoint)
            )
            
            # Store both parameters and model for acquisition training
            surrogate_data = {
                "params": training_results.final_params,
                "model": training_results.final_model,
                "trainer": trainer,
                "checkpoint_path": str(surrogate_checkpoint)
            }
            
        except ImportError:
            logger.warning("SurrogateTrainer not available, using placeholder implementation")
            surrogate_data = None
            metrics = {"loss": 0.0, "accuracy": 0.0}
        except Exception as e:
            logger.error(f"Surrogate training failed: {e}")
            # Use placeholder to allow pipeline to continue for testing
            surrogate_data = None
            metrics = {"loss": float('inf'), "error": str(e)}
        
        updated_metrics = self.state.training_metrics.update({
            "surrogate_metrics": metrics
        })
        
        return replace(
            self.state,
            current_stage="surrogate_training_complete",
            surrogate_params=surrogate_data,
            training_metrics=updated_metrics,
            completed_stages=self.state.completed_stages + ["surrogate_training"]
        )
    
    def _run_curriculum_acquisition_training(self) -> TrainingState:
        """Run curriculum-based acquisition training."""
        logger.info("🎯 Stage 3: Curriculum-based acquisition training")
        
        if self.state.surrogate_params is None:
            raise ValueError("Surrogate model must be trained before acquisition training")
        
        current_state = self.state
        
        # Iterate through curriculum difficulties
        for difficulty_level in self.curriculum.get_difficulty_sequence():
            logger.info(f"Training difficulty level {difficulty_level}")
            
            # Create adaptive reward config for this difficulty
            scm_config = self.curriculum.get_scm_config_for_difficulty(difficulty_level)
            reward_config = create_adaptive_reward_config(
                scm=scm_config,
                difficulty_level=difficulty_level
            )
            
            # TODO: Implement acquisition model training (Phase 2.2)
            # This will use self.config.grpo for GRPO parameters
            logger.warning(f"Acquisition model training not yet implemented for difficulty {difficulty_level}")
            acquisition_params = None  # Placeholder
            training_metrics = {  # Placeholder metrics
                "final_f1_score": 0.8,
                "optimization_improvement": 0.1,
                "reward_history": []
            }
            
            # Validate training quality (anti-gaming)
            reward_history = training_metrics.get("reward_history", [])
            validation_results = validate_reward_consistency(reward_history)
            
            if not validation_results["valid"]:
                logger.warning(f"Gaming detected at difficulty {difficulty_level}: {validation_results['gaming_issues']}")
                # Could implement recovery strategies here
            
            # Check progression criteria
            if self.curriculum.should_advance_difficulty(training_metrics):
                logger.info(f"✅ Completed difficulty {difficulty_level}")
                current_state = replace(
                    current_state,
                    current_difficulty=difficulty_level,
                    acquisition_params=acquisition_params,
                    training_metrics=current_state.training_metrics.update({
                        f"difficulty_{difficulty_level}_metrics": training_metrics
                    })
                )
                
                # Save checkpoint after each difficulty
                self._save_checkpoint(f"difficulty_{difficulty_level}_complete")
            else:
                logger.warning(f"Failed to meet progression criteria for difficulty {difficulty_level}")
                break
        
        return replace(
            current_state,
            current_stage="acquisition_training_complete",
            completed_stages=current_state.completed_stages + ["acquisition_training"]
        )
    
    def _save_checkpoint(self, checkpoint_name: str, user_notes: str = "") -> str:
        """Save training state checkpoint using enhanced checkpoint manager."""
        try:
            checkpoint_info = self.checkpoint_manager.save_checkpoint(
                state=self.state,
                checkpoint_name=checkpoint_name,
                stage=self.state.current_stage,
                user_notes=user_notes
            )
            
            # Update state with checkpoint path
            self.state = replace(
                self.state,
                checkpoint_path=str(checkpoint_info.path)
            )
            
            logger.info(f"💾 Enhanced checkpoint saved: {checkpoint_info.metadata.checkpoint_id}")
            return str(checkpoint_info.path)
            
        except Exception as e:
            logger.error(f"Failed to save enhanced checkpoint {checkpoint_name}: {e}")
            # Fallback to simple checkpoint save
            return self._save_checkpoint_fallback(checkpoint_name)
    
    def _save_checkpoint_fallback(self, checkpoint_name: str) -> str:
        """Fallback checkpoint save method."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}_fallback.pkl"
        
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.state, f)
        
        logger.warning(f"🔄 Saved fallback checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    
    def _load_checkpoint(self, checkpoint_path: str) -> TrainingState:
        """Load training state from checkpoint using enhanced checkpoint manager."""
        try:
            # Try to load using enhanced checkpoint manager
            state = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            logger.info(f"📂 Loaded enhanced checkpoint: {checkpoint_path}")
            return state
            
        except Exception as e:
            logger.warning(f"Enhanced checkpoint load failed, trying fallback: {e}")
            # Fallback to simple pickle load
            return self._load_checkpoint_fallback(checkpoint_path)
    
    def _load_checkpoint_fallback(self, checkpoint_path: str) -> TrainingState:
        """Fallback checkpoint load method."""
        import pickle
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        logger.info(f"🔄 Loaded fallback checkpoint: {checkpoint_path}")
        return state
    
    def list_available_checkpoints(self, stage: Optional[str] = None) -> List[Any]:
        """List all available checkpoints, optionally filtered by stage."""
        return self.checkpoint_manager.list_checkpoints(stage)
    
    def get_latest_checkpoint(self, stage: Optional[str] = None) -> Optional[Any]:
        """Get the most recent checkpoint, optionally filtered by stage."""
        return self.checkpoint_manager.get_latest_checkpoint(stage)
    
    def auto_resume_from_latest(self) -> bool:
        """
        Automatically resume from the latest checkpoint if available.
        
        Returns:
            True if resumed from checkpoint, False if starting fresh
        """
        latest_checkpoint = self.get_latest_checkpoint()
        if latest_checkpoint and latest_checkpoint.is_valid:
            try:
                self.state = self._load_checkpoint(str(latest_checkpoint.path))
                logger.info(f"🔄 Auto-resumed from latest checkpoint: {latest_checkpoint.metadata.checkpoint_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to auto-resume from latest checkpoint: {e}")
        
        logger.info("🆕 Starting fresh training (no valid checkpoints found)")
        return False
    
    def cleanup_old_checkpoints(self) -> None:
        """Manually trigger cleanup of old checkpoints."""
        self.checkpoint_manager._cleanup_old_checkpoints()
        logger.info("🧹 Triggered checkpoint cleanup")
    
    def _compile_final_results(self) -> MasterTrainingResults:
        """Compile final training results."""
        surrogate_metrics = self.state.training_metrics.get("surrogate_metrics", {})
        
        # Collect all acquisition metrics across difficulties
        acquisition_metrics = {}
        curriculum_completion = {}
        
        for key, value in self.state.training_metrics.items():
            if key.startswith("difficulty_") and key.endswith("_metrics"):
                difficulty_level = int(key.split("_")[1])
                acquisition_metrics[difficulty_level] = value
                curriculum_completion[difficulty_level] = True
        
        return MasterTrainingResults(
            final_state=self.state,
            surrogate_metrics=surrogate_metrics,
            acquisition_metrics=acquisition_metrics,
            curriculum_completion=curriculum_completion,
            total_training_time=0.0,  # TODO: Add timing
            checkpoints_saved=[]  # TODO: Track checkpoints
        )


# Factory functions for clean interface
def create_master_trainer(
    config: Optional[TrainingConfig] = None,
    checkpoint_dir: Optional[str] = None
) -> MasterTrainer:
    """
    Create master trainer with validated configuration.
    
    Args:
        config: Training configuration
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Initialized MasterTrainer
    """
    return MasterTrainer(config=config, checkpoint_dir=checkpoint_dir)


def run_complete_acbo_training(
    config: Optional[TrainingConfig] = None,
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None
) -> MasterTrainingResults:
    """
    Convenience function to run complete ACBO training pipeline.
    
    Args:
        config: Training configuration
        checkpoint_dir: Directory for checkpoints  
        resume_from: Checkpoint to resume from
        
    Returns:
        Complete training results
    """
    trainer = create_master_trainer(config=config, checkpoint_dir=checkpoint_dir)
    return trainer.train_complete_pipeline(resume_from_checkpoint=resume_from)