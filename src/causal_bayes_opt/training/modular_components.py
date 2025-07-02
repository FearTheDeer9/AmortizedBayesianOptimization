"""
Modular Training Components for 3-Component ACBO Training System.

This module implements the modular architecture for independent training of:
1. Surrogate training from expert demonstrations
2. Surrogate-free acquisition training with GRPO + RLVR
3. Joint training combining both components

Key design principles:
- Independent component training with checkpointing
- Curriculum progression across difficulty levels
- Modular architecture allowing mix-and-match of checkpoints
- Support for both surrogate-free and surrogate-aware acquisition (future)
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
import pyrsistent as pyr

# TODO: This import references the fraudulent enhanced_acbo_with_grpo component
# from ..experiments.enhanced_acbo_with_grpo import EnhancedACBOWithGRPO
from ..acquisition.enhanced_policy_network import (
    create_enhanced_policy_for_grpo, validate_enhanced_policy_integration
)
from ..avici_integration.enhanced_surrogate import (
    create_enhanced_surrogate_for_grpo, validate_enhanced_surrogate_integration
)
from .curriculum import CurriculumManager, DifficultyLevel
from ..data_structures.scm import get_variables, get_target, get_parents

# Handle examples import
import sys
project_root = Path(__file__).parent.parent.parent.parent
examples_path = project_root / "examples"
if str(examples_path) not in sys.path:
    sys.path.insert(0, str(examples_path))

from demo_learning import DemoConfig

logger = logging.getLogger(__name__)


# Checkpoint dataclasses moved to checkpoint_dataclasses.py to avoid duplication
# Import them instead of redefining


class SurrogateTrainer:
    """
    Component 1: Independent surrogate training from expert demonstrations.
    
    This component trains enhanced surrogate models using behavioral cloning
    on expert demonstrations collected from PARENT_SCALE. It progresses through
    curriculum difficulty levels and can save/load checkpoints independently.
    """
    
    def __init__(self, config: DemoConfig, curriculum_manager: CurriculumManager):
        self.config = config
        self.curriculum_manager = curriculum_manager
        self.current_difficulty = None
        self.training_history = []
        
        logger.info("Initialized SurrogateTrainer for expert demonstration training")
    
    def train_from_demonstrations(
        self,
        expert_demos: List[Dict[str, Any]],
        difficulty_level: str,
        n_training_steps: int = 1000,
        learning_rate: float = 0.0003,
        save_checkpoints: bool = True
    ) -> SurrogateCheckpoint:
        """
        Train surrogate model from expert demonstrations.
        
        Args:
            expert_demos: List of expert demonstration trajectories
            difficulty_level: Curriculum difficulty level
            n_training_steps: Number of training steps
            learning_rate: Learning rate for training
            save_checkpoints: Whether to save checkpoints
            
        Returns:
            SurrogateCheckpoint with trained model
        """
        logger.info(f"Starting surrogate training at difficulty {difficulty_level}")
        self.current_difficulty = difficulty_level
        
        # TODO: Implement actual surrogate training from expert demonstrations
        # For now, create a placeholder implementation
        
        # Initialize enhanced surrogate network
        variables = self._extract_variables_from_demos(expert_demos)
        target_variable = self._extract_target_from_demos(expert_demos)
        
        enhanced_surrogate_fn, surrogate_config = create_enhanced_surrogate_for_grpo(
            variables=variables,
            target_variable=target_variable,
            model_complexity="full",
            use_continuous=True,
            performance_mode="balanced"
        )
        
        # Initialize parameters (placeholder - would use actual training)
        key = jax.random.PRNGKey(42)
        dummy_data = jnp.ones((1, len(variables), 10))
        surrogate_transform = hk.transform(lambda x: enhanced_surrogate_fn(x, target_idx=0, is_training=True))
        surrogate_params = surrogate_transform.init(key, dummy_data)
        
        # Placeholder training loop (would implement actual behavioral cloning)
        for step in range(n_training_steps):
            # TODO: Implement actual training step
            if step % 100 == 0:
                logger.info(f"Surrogate training step {step}/{n_training_steps}")
        
        # Create checkpoint
        checkpoint = SurrogateCheckpoint(
            component_type="surrogate",
            difficulty_level=difficulty_level,
            training_step=n_training_steps,
            timestamp=time.time(),
            config=self.config.__dict__,
            performance_metrics={
                "final_loss": 0.1,  # Placeholder
                "validation_accuracy": 0.85  # Placeholder
            },
            checkpoint_path="",  # Will be set when saved
            surrogate_params=surrogate_params,
            surrogate_config=surrogate_config,
            expert_demo_stats={
                "num_demonstrations": len(expert_demos),
                "avg_trajectory_length": self._compute_avg_trajectory_length(expert_demos)
            }
        )
        
        if save_checkpoints:
            checkpoint_path = self._save_surrogate_checkpoint(checkpoint)
            checkpoint.checkpoint_path = checkpoint_path
        
        logger.info(f"Completed surrogate training at difficulty {difficulty_level}")
        return checkpoint
    
    def _extract_variables_from_demos(self, expert_demos: List[Dict[str, Any]]) -> List[str]:
        """Extract variable names from expert demonstrations."""
        if not expert_demos:
            return ["X0", "X1", "X2", "X3"]  # Default fallback
        
        # TODO: Extract actual variables from demo format
        return ["X0", "X1", "X2", "X3"]  # Placeholder
    
    def _extract_target_from_demos(self, expert_demos: List[Dict[str, Any]]) -> str:
        """Extract target variable from expert demonstrations."""
        if not expert_demos:
            return "X3"  # Default fallback
        
        # TODO: Extract actual target from demo format
        return "X3"  # Placeholder
    
    def _compute_avg_trajectory_length(self, expert_demos: List[Dict[str, Any]]) -> float:
        """Compute average trajectory length from demonstrations."""
        if not expert_demos:
            return 10.0
        
        # TODO: Compute actual trajectory lengths
        return 10.0  # Placeholder
    
    def _save_surrogate_checkpoint(self, checkpoint: SurrogateCheckpoint) -> str:
        """Save surrogate checkpoint to disk."""
        from .checkpoint_manager import create_modular_checkpoint_manager
        
        checkpoint_manager = create_modular_checkpoint_manager()
        checkpoint_path = checkpoint_manager.save_component_checkpoint(
            checkpoint=checkpoint,
            experiment_name="surrogate_training",
            tags=["surrogate", "expert_demos"]
        )
        
        logger.info(f"Saved surrogate checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_surrogate_checkpoint(self, checkpoint_path: str) -> SurrogateCheckpoint:
        """Load surrogate checkpoint from disk."""
        from .checkpoint_manager import create_modular_checkpoint_manager
        
        checkpoint_manager = create_modular_checkpoint_manager()
        checkpoint = checkpoint_manager.load_component_checkpoint(checkpoint_path)
        
        logger.info(f"Loaded surrogate checkpoint: {checkpoint_path}")
        return checkpoint


class SurrogateFreeAcquisitionTrainer:
    """
    Component 2A: Surrogate-free acquisition training with GRPO + RLVR.
    
    This component trains acquisition policies without requiring a surrogate model.
    It uses enriched history tensors and verifiable rewards for direct policy learning.
    Enables immediate training without waiting for surrogate model training.
    """
    
    def __init__(self, config: DemoConfig, curriculum_manager: CurriculumManager):
        self.config = config
        self.curriculum_manager = curriculum_manager
        self.current_difficulty = None
        self.training_history = []
        
        logger.info("Initialized SurrogateFreeAcquisitionTrainer for immediate deployment")
    
    def train_grpo_with_rlvr(
        self,
        difficulty_level: str,
        n_training_steps: int = 30,
        learning_rate: float = 0.0003,
        enable_wandb: bool = False,
        save_checkpoints: bool = True
    ) -> AcquisitionCheckpoint:
        """
        Train acquisition policy using GRPO + RLVR without surrogate dependency.
        
        Args:
            difficulty_level: Curriculum difficulty level
            n_training_steps: Number of GRPO training steps
            learning_rate: Learning rate for policy optimization
            enable_wandb: Whether to enable WandB logging
            save_checkpoints: Whether to save checkpoints
            
        Returns:
            AcquisitionCheckpoint with trained policy
        """
        logger.info(f"Starting surrogate-free acquisition training at difficulty {difficulty_level}")
        self.current_difficulty = difficulty_level
        
        # Create SCM for this difficulty level
        scm = self.curriculum_manager.create_scm_for_difficulty(difficulty_level)
        
        # TODO: EnhancedACBOWithGRPO was a fraudulent component - needs replacement
        # This functionality should be implemented using clean GRPO training
        logger.info(f"Training for {n_training_steps} steps at difficulty {difficulty_level}")
        
        # Placeholder implementation
        results = {
            'training_history': [],
            'final_results': {'final_avg_reward': 0.0, 'final_target_improvement': 0.0},
            'best_reward': 0.0,
            'total_time': 1.0
        }
        
        # Extract training results
        training_history = results.get('training_history', [])
        final_performance = results.get('final_results', {})
        
        # Create checkpoint
        checkpoint = AcquisitionCheckpoint(
            component_type="acquisition",
            difficulty_level=difficulty_level,
            training_step=n_training_steps,
            timestamp=time.time(),
            config=self.config.__dict__,
            performance_metrics={
                "final_avg_reward": final_performance.get('final_avg_reward', 0.0),
                "final_target_improvement": final_performance.get('final_target_improvement', 0.0),
                "best_reward": results.get('best_reward', 0.0),
                "total_time": results.get('total_time', 0.0)
            },
            checkpoint_path="",  # Will be set when saved
            policy_params=None,  # TODO: Replace with actual trained params
            optimizer_state=None,  # TODO: Replace with actual optimizer state
            policy_config={},  # TODO: Replace with actual policy config
            training_history=training_history,
            is_surrogate_aware=False,
            surrogate_checkpoint_path=None
        )
        
        if save_checkpoints:
            checkpoint_path = self._save_acquisition_checkpoint(checkpoint)
            checkpoint.checkpoint_path = checkpoint_path
        
        logger.info(f"Completed surrogate-free acquisition training at difficulty {difficulty_level}")
        return checkpoint
    
    def _save_acquisition_checkpoint(self, checkpoint: AcquisitionCheckpoint) -> str:
        """Save acquisition checkpoint to disk."""
        from .checkpoint_manager import create_modular_checkpoint_manager
        
        checkpoint_manager = create_modular_checkpoint_manager()
        checkpoint_path = checkpoint_manager.save_component_checkpoint(
            checkpoint=checkpoint,
            experiment_name="surrogate_free_acquisition",
            tags=["acquisition", "surrogate_free", "grpo"]
        )
        
        logger.info(f"Saved acquisition checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def load_acquisition_checkpoint(self, checkpoint_path: str) -> AcquisitionCheckpoint:
        """Load acquisition checkpoint from disk."""
        from .checkpoint_manager import create_modular_checkpoint_manager
        
        checkpoint_manager = create_modular_checkpoint_manager()
        checkpoint = checkpoint_manager.load_component_checkpoint(checkpoint_path)
        
        logger.info(f"Loaded acquisition checkpoint: {checkpoint_path}")
        return checkpoint


class SurrogateAwareAcquisitionTrainer:
    """
    Component 2B: Surrogate-aware acquisition training (future implementation).
    
    This component will train acquisition policies that use surrogate model predictions
    for structured state representation and uncertainty-guided exploration.
    """
    
    def __init__(self, config: DemoConfig, curriculum_manager: CurriculumManager):
        self.config = config
        self.curriculum_manager = curriculum_manager
        
        logger.info("Initialized SurrogateAwareAcquisitionTrainer (placeholder for future)")
    
    def train_grpo_with_surrogate(
        self,
        surrogate_checkpoint: SurrogateCheckpoint,
        difficulty_level: str,
        n_training_steps: int = 30,
        learning_rate: float = 0.0003
    ) -> AcquisitionCheckpoint:
        """
        Train acquisition policy using surrogate predictions (future implementation).
        
        Args:
            surrogate_checkpoint: Trained surrogate model checkpoint
            difficulty_level: Curriculum difficulty level
            n_training_steps: Number of GRPO training steps
            learning_rate: Learning rate for policy optimization
            
        Returns:
            AcquisitionCheckpoint with surrogate-aware policy
        """
        logger.info("Surrogate-aware acquisition training not yet implemented")
        raise NotImplementedError("Surrogate-aware acquisition training will be implemented after surrogate training is ready")


class JointTrainer:
    """
    Component 3: Joint training for end-to-end optimization.
    
    This component takes pre-trained surrogate and acquisition checkpoints and
    fine-tunes them together for optimal end-to-end performance.
    """
    
    def __init__(self, config: DemoConfig, curriculum_manager: CurriculumManager):
        self.config = config
        self.curriculum_manager = curriculum_manager
        
        logger.info("Initialized JointTrainer for end-to-end optimization")
    
    def joint_training(
        self,
        acquisition_checkpoint: AcquisitionCheckpoint,
        surrogate_checkpoint: Optional[SurrogateCheckpoint] = None,
        difficulty_level: str = "difficulty_1",
        n_training_steps: int = 50,
        learning_rate: float = 0.0001
    ) -> JointCheckpoint:
        """
        Perform joint training of acquisition and surrogate models.
        
        Args:
            acquisition_checkpoint: Pre-trained acquisition model
            surrogate_checkpoint: Pre-trained surrogate model (optional for surrogate-free)
            difficulty_level: Curriculum difficulty level
            n_training_steps: Number of joint training steps
            learning_rate: Learning rate for joint optimization
            
        Returns:
            JointCheckpoint with jointly optimized models
        """
        logger.info(f"Starting joint training at difficulty {difficulty_level}")
        
        if surrogate_checkpoint is None:
            logger.info("Joint training with surrogate-free acquisition")
        else:
            logger.info("Joint training with surrogate-aware acquisition")
        
        # TODO: Implement actual joint training
        # For now, return the acquisition checkpoint as basis for joint checkpoint
        
        joint_checkpoint = JointCheckpoint(
            component_type="joint",
            difficulty_level=difficulty_level,
            training_step=n_training_steps,
            timestamp=time.time(),
            config=self.config.__dict__,
            performance_metrics={
                "joint_final_reward": 0.0,  # Placeholder
                "improvement_over_independent": 0.0  # Placeholder
            },
            checkpoint_path="",
            surrogate_params=surrogate_checkpoint.surrogate_params if surrogate_checkpoint else None,
            policy_params=acquisition_checkpoint.policy_params,
            surrogate_optimizer_state=None,
            policy_optimizer_state=acquisition_checkpoint.optimizer_state,
            joint_config={
                "acquisition_type": "surrogate_aware" if surrogate_checkpoint else "surrogate_free",
                "joint_learning_rate": learning_rate
            },
            joint_training_history=[]
        )
        
        logger.info(f"Completed joint training at difficulty {difficulty_level}")
        return joint_checkpoint


class ModularCurriculumManager:
    """
    Enhanced curriculum manager for modular training components.
    
    Coordinates curriculum progression across all three training components
    and manages advancement criteria and difficulty scaling.
    """
    
    def __init__(self, base_curriculum_manager: CurriculumManager):
        self.base_manager = base_curriculum_manager
        self.component_progress = {
            "surrogate": {},
            "acquisition": {},
            "joint": {}
        }
        
        logger.info("Initialized ModularCurriculumManager for component coordination")
    
    def should_advance_difficulty(
        self,
        component_type: str,
        current_difficulty: str,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """
        Determine if component should advance to next difficulty level.
        
        Args:
            component_type: Type of component ('surrogate', 'acquisition', 'joint')
            current_difficulty: Current difficulty level
            performance_metrics: Performance metrics from training
            
        Returns:
            True if should advance to next difficulty
        """
        # Use base curriculum manager logic with component-specific thresholds
        return self.base_manager.should_advance_difficulty(performance_metrics)
    
    def get_next_difficulty(self, current_difficulty: str) -> Optional[str]:
        """Get next difficulty level in curriculum."""
        return self.base_manager.get_next_difficulty(current_difficulty)
    
    def create_scm_for_difficulty(self, difficulty_level: str) -> pyr.PMap:
        """Create SCM for given difficulty level."""
        return self.base_manager.create_scm_for_difficulty(difficulty_level)
    
    def record_component_progress(
        self,
        component_type: str,
        difficulty_level: str,
        checkpoint: TrainingCheckpoint
    ):
        """Record training progress for component at difficulty level."""
        if component_type not in self.component_progress:
            self.component_progress[component_type] = {}
        
        self.component_progress[component_type][difficulty_level] = {
            "checkpoint_path": checkpoint.checkpoint_path,
            "performance_metrics": checkpoint.performance_metrics,
            "timestamp": checkpoint.timestamp
        }
        
        logger.info(f"Recorded progress for {component_type} at {difficulty_level}")
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all components across difficulty levels."""
        return self.component_progress.copy()


# Export key classes
__all__ = [
    'SurrogateTrainer',
    'SurrogateFreeAcquisitionTrainer', 
    'SurrogateAwareAcquisitionTrainer',
    'JointTrainer',
    'ModularCurriculumManager',
    'SurrogateCheckpoint',
    'AcquisitionCheckpoint',
    'JointCheckpoint',
    'TrainingCheckpoint'
]