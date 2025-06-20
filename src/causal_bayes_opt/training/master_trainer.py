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

import pyrsistent as pyr

from .config import TrainingConfig, create_default_training_config
from .curriculum import create_curriculum_manager
from ..acquisition.verifiable_rewards import (
    validate_reward_consistency,
    create_adaptive_reward_config
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
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize master trainer.
        
        Args:
            config: Training configuration (uses defaults if None)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.config = config or create_default_training_config()
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize curriculum manager
        self.curriculum = create_curriculum_manager(self.config)
        
        # Initialize training state
        self.state = TrainingState(
            current_stage="initial",
            current_difficulty=1,
            surrogate_params=None,
            acquisition_params=None,
            expert_demonstrations=None,
            training_metrics=pyr.m(),
            completed_stages=[],
            checkpoint_path=None
        )
        
        logger.info(f"Initialized MasterTrainer with config type: {type(self.config).__name__}")
    
    def train_complete_pipeline(
        self,
        resume_from_checkpoint: Optional[str] = None
    ) -> MasterTrainingResults:
        """
        Execute complete ACBO training pipeline.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Complete training results with all metrics
        """
        logger.info("🚀 Starting ACBO training pipeline")
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.state = self._load_checkpoint(resume_from_checkpoint)
            logger.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
        
        try:
            # Stage 1: Expert demonstration collection
            if "expert_collection" not in self.state.completed_stages:
                self.state = self._run_expert_collection()
                self._save_checkpoint("expert_collection_complete")
            
            # Stage 2: Surrogate training
            if "surrogate_training" not in self.state.completed_stages:
                self.state = self._run_surrogate_training()
                self._save_checkpoint("surrogate_training_complete")
            
            # Stage 3: Curriculum-based acquisition training
            if "acquisition_training" not in self.state.completed_stages:
                self.state = self._run_curriculum_acquisition_training()
                self._save_checkpoint("acquisition_training_complete")
            
            # Compile final results
            results = self._compile_final_results()
            
            logger.info("✅ ACBO training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            # Save emergency checkpoint
            self._save_checkpoint("emergency_checkpoint")
            raise
    
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
            logger.warning("No expert demonstrations available, using placeholder")
        
        # TODO: Implement surrogate training (Phase 2.1)
        # This will use self.config.surrogate for training parameters
        logger.warning("Surrogate model training not yet implemented")
        surrogate_params = None  # Placeholder
        metrics = {"loss": 0.0, "accuracy": 0.0}  # Placeholder metrics
        
        updated_metrics = self.state.training_metrics.update({
            "surrogate_metrics": metrics
        })
        
        return replace(
            self.state,
            current_stage="surrogate_training_complete",
            surrogate_params=surrogate_params,
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
    
    def _save_checkpoint(self, checkpoint_name: str) -> str:
        """Save training state checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        # Save state using pickle for simplicity
        import pickle
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(self.state, f)
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        return str(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str) -> TrainingState:
        """Load training state from checkpoint."""
        import pickle
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        logger.info(f"📂 Loaded checkpoint: {checkpoint_path}")
        return state
    
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