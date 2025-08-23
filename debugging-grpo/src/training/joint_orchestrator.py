"""
Joint training orchestrator - coordinates between policy and surrogate training.

This class handles the orchestration logic that was inappropriately mixed into 
UnifiedGRPOTrainer. It manages:
- Phase switching (policy vs surrogate)
- SCM rotation and management
- Training coordination between different trainers
- Cross-phase progress tracking
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from collections import deque

import numpy as np
import jax.random as random

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingConfig:
    """Configuration for joint training orchestration."""
    # Phase control
    episodes_per_phase: int = 5
    initial_phase: str = "policy"  # "policy" or "surrogate"
    
    # SCM management
    rotation_episodes: int = 20
    max_episodes_per_scm: int = 50
    
    # Training episodes
    max_episodes: int = 100
    
    # Adaptive rotation
    use_performance_rotation: bool = False
    plateau_patience: int = 10


@dataclass 
class PhaseMetrics:
    """Metrics for a single training phase."""
    episode: int
    phase: str
    mean_reward: float = 0.0
    target_improvement: float = 0.0
    loss: float = 0.0
    f1_score: float = 0.0


class JointOrchestrator:
    """
    Orchestrates joint training between policy and surrogate trainers.
    
    Uses composition instead of inheritance for clean separation of concerns.
    """
    
    def __init__(self, 
                 policy_trainer: Any,  # Pure GRPO trainer
                 surrogate_trainer: Optional[Any] = None,  # AVICI trainer
                 config: Optional[JointTrainingConfig] = None):
        """
        Initialize joint orchestrator.
        
        Args:
            policy_trainer: Pure GRPO trainer for policy episodes
            surrogate_trainer: Optional AVICI trainer for surrogate episodes  
            config: Joint training configuration
        """
        self.policy_trainer = policy_trainer
        self.surrogate_trainer = surrogate_trainer
        self.config = config or JointTrainingConfig()
        
        # Phase control
        self.current_phase = self.config.initial_phase
        self.phase_episode_count = 0
        
        # SCM management
        self.current_scm = None
        self.current_scm_idx = 0
        self.episodes_on_current_scm = 0
        
        # Training tracking
        self.policy_episodes = 0
        self.surrogate_episodes = 0
        self.joint_metrics_history = []
        self.target_progression_history = []
        
        # Initialize RNG
        self.rng_key = random.PRNGKey(42)
        
        logger.info(f"Initialized JointOrchestrator with {self.config.initial_phase} phase")
    
    def train(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> Dict[str, Any]:
        """
        Run joint training with proper orchestration.
        
        Args:
            scms: Training SCMs in various formats
            
        Returns:
            Combined training results
        """
        logger.info("="*70)
        logger.info("Starting Joint ACBO Training with Proper Orchestration")
        logger.info(f"  Max episodes: {self.config.max_episodes}")
        logger.info(f"  Episodes per phase: {self.config.episodes_per_phase}")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Prepare SCMs using orchestrator logic
        scm_rotation = self._prepare_scms(scms)
        logger.info(f"Training with {len(scm_rotation)} SCMs")
        
        episode_metrics = []
        
        # Main training loop
        for episode in range(self.config.max_episodes):
            # Check phase switching
            if self._should_switch_phase():
                self._switch_phase()
            
            # Check SCM rotation 
            if self._should_rotate_scm():
                self._rotate_scm(scm_rotation)
            
            # Get current SCM
            scm_name, scm = scm_rotation[self.current_scm_idx]
            self.current_scm = scm
            
            # Run episode based on current phase
            self.rng_key, episode_key = random.split(self.rng_key)
            
            if self.current_phase == 'policy':
                # Policy phase: delegate to GRPO trainer
                metrics = self._run_policy_episode(episode, scm, scm_name, episode_key)
                metrics['training_phase'] = 'policy'
                self.policy_episodes += 1
            else:
                # Surrogate phase: delegate to AVICI trainer
                if self.surrogate_trainer is not None:
                    metrics = self._run_surrogate_episode(episode, scm, scm_name, episode_key)
                    metrics['training_phase'] = 'surrogate'
                    self.surrogate_episodes += 1
                else:
                    # Skip surrogate phase if no trainer available
                    logger.warning("Surrogate phase requested but no surrogate trainer provided")
                    metrics = {'episode': episode, 'mean_reward': 0.0, 'training_phase': 'surrogate_skipped'}
            
            episode_metrics.append(metrics)
            self.joint_metrics_history.append(self._create_joint_metrics(metrics))
            
            # Track target progression
            if 'target_values' in metrics and metrics['target_values']:
                self.target_progression_history.append({
                    'episode': episode,
                    'target_values': metrics['target_values'],
                    'phase': self.current_phase,
                    'scm': scm_name
                })
            
            self.phase_episode_count += 1
            self.episodes_on_current_scm += 1
        
        # Prepare results
        return {
            'training_time': time.time() - start_time,
            'final_metrics': episode_metrics[-1] if episode_metrics else {},
            'all_metrics': episode_metrics,
            'policy_episodes': self.policy_episodes,
            'surrogate_episodes': self.surrogate_episodes,
            'target_progression': self.target_progression_history,
            'joint_metrics': self.joint_metrics_history
        }
    
    def _prepare_scms(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> List[Tuple[str, Any]]:
        """Convert various SCM formats to standard list of (name, scm) tuples."""
        if isinstance(scms, list):
            if scms and isinstance(scms[0], tuple) and len(scms[0]) == 2:
                return scms
            else:
                return [(f"scm_{i}", scm) for i, scm in enumerate(scms)]
        elif isinstance(scms, dict):
            return list(scms.items())
        elif callable(scms):
            generated = []
            for i in range(10):
                scm = scms()
                generated.append((f"generated_{i}", scm))
            return generated
        else:
            raise ValueError(f"Unsupported SCM format: {type(scms)}")
    
    def _should_switch_phase(self) -> bool:
        """Check if we should switch training phases."""
        return self.phase_episode_count >= self.config.episodes_per_phase
    
    def _switch_phase(self):
        """Switch between policy and surrogate training phases."""
        old_phase = self.current_phase
        self.current_phase = 'surrogate' if self.current_phase == 'policy' else 'policy'
        self.phase_episode_count = 0
        
        logger.info(f"Phase switch: {old_phase} -> {self.current_phase}")
    
    def _should_rotate_scm(self) -> bool:
        """Check if we should rotate to the next SCM."""
        if self.config.use_performance_rotation:
            # Implement performance-based rotation logic
            return self._check_performance_plateau()
        else:
            # Simple episode-based rotation
            return self.episodes_on_current_scm >= self.config.rotation_episodes
    
    def _rotate_scm(self, scm_rotation: List[Tuple[str, Any]]):
        """Rotate to the next SCM."""
        old_idx = self.current_scm_idx
        self.current_scm_idx = (self.current_scm_idx + 1) % len(scm_rotation)
        self.episodes_on_current_scm = 0
        
        old_name = scm_rotation[old_idx][0]
        new_name = scm_rotation[self.current_scm_idx][0]
        logger.info(f"SCM rotation: {old_name} -> {new_name}")
    
    def _check_performance_plateau(self) -> bool:
        """Check if performance has plateaued (for adaptive rotation)."""
        if len(self.joint_metrics_history) < self.config.plateau_patience:
            return False
        
        # Check if recent performance is not improving
        recent_rewards = [m.mean_reward for m in self.joint_metrics_history[-self.config.plateau_patience:]]
        if len(set(recent_rewards)) == 1:  # All same reward
            return True
            
        # Check if trend is flat
        trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
        return abs(trend) < 0.01  # Very small improvement
    
    def _run_policy_episode(self, episode: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """Run policy training episode - delegates to GRPO trainer."""
        return self.policy_trainer._run_grpo_episode(episode, scm, scm_name, key)
    
    def _run_surrogate_episode(self, episode: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """Run surrogate training episode - delegates to AVICI trainer."""
        if self.surrogate_trainer is None:
            return {'episode': episode, 'mean_reward': 0.0}
        
        # This would call AVICI training logic
        # For now, return dummy metrics
        return {
            'episode': episode,
            'mean_reward': 0.0,
            'surrogate_loss': 0.1,
            'f1_score': 0.5
        }
    
    def _create_joint_metrics(self, episode_metrics: Dict[str, Any]) -> PhaseMetrics:
        """Create joint metrics from episode results."""
        return PhaseMetrics(
            episode=episode_metrics.get('episode', 0),
            phase=episode_metrics.get('training_phase', self.current_phase),
            mean_reward=episode_metrics.get('mean_reward', 0.0),
            target_improvement=episode_metrics.get('target_improvement', 0.0),
            loss=episode_metrics.get('loss', 0.0),
            f1_score=episode_metrics.get('structure_metrics', {}).get('f1_score', 0.0)
        )