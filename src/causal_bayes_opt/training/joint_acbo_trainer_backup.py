"""
Joint ACBO Trainer for coordinated policy and surrogate learning.

This module implements a GAN-like training approach where policy and surrogate
models are trained alternately to optimize their collaborative performance.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn
import optax
import haiku as hk
from omegaconf import DictConfig
import pyrsistent as pyr

# Import parent class
from .unified_grpo_trainer import UnifiedGRPOTrainer, compute_param_change

# Import trajectory replay buffer (to be created)
from .trajectory_replay_buffer import TrajectoryReplayBuffer, Trajectory

# Import adaptive SCM generator (to be created)
from .adaptive_scm_generator import AdaptiveSCMGenerator

# Import existing components
from .curriculum_factory import SCMCurriculumFactory
from .surrogate_bc_trainer import SurrogateBCTrainer
from .joint_losses import (
    compute_absolute_target_loss,
    compute_information_gain,
    compute_joint_policy_loss,
    compute_trajectory_info_gain,
    validate_loss_computation
)
from ..acquisition.grpo import grpo_update_step, GRPOConfig
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.scm import get_variables, get_target, get_parents
from ..data_structures.sample import get_values
from ..utils.checkpoint_utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingMetrics:
    """Metrics for tracking joint training performance."""
    episode: int
    phase: str  # "policy" or "surrogate"
    policy_loss: float = 0.0
    surrogate_loss: float = 0.0
    mean_reward: float = 0.0
    f1_score: float = 0.0
    target_improvement: float = 0.0
    information_gain: float = 0.0
    episodes_in_phase: int = 0
    total_episodes: int = 0


@dataclass
class PhasePerformance:
    """Track performance for each training phase."""
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_f1_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_improvements: deque = field(default_factory=lambda: deque(maxlen=10))
    episodes_completed: int = 0
    is_plateaued: bool = False
    
    def update(self, loss: float = None, f1: float = None, improvement: float = None):
        """Update performance metrics."""
        if loss is not None:
            self.recent_losses.append(loss)
        if f1 is not None:
            self.recent_f1_scores.append(f1)
        if improvement is not None:
            self.recent_improvements.append(improvement)
        self.episodes_completed += 1
        
        # Check for plateau (no improvement in last 5 episodes)
        if len(self.recent_losses) >= 5:
            recent = list(self.recent_losses)[-5:]
            if max(recent) - min(recent) < 0.01:  # Very small change
                self.is_plateaued = True
            else:
                self.is_plateaued = False


class JointACBOTrainer(UnifiedGRPOTrainer):
    """
    Joint trainer for policy and surrogate models with alternating updates.
    
    This trainer extends UnifiedGRPOTrainer to add:
    - Alternating training phases
    - Trajectory replay buffer
    - Adaptive SCM management
    - Joint performance tracking
    """
    
    def __init__(self,
                 config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
                 **kwargs):
        """
        Initialize joint ACBO trainer.
        
        Args:
            config: Training configuration
            **kwargs: Additional parameters passed to parent
        """
        print("DEBUG: JointACBOTrainer.__init__ called")
        import sys
        sys.stdout.flush()
        
        logger.info("Initializing JointACBOTrainer...")
        
        # Store config for override methods
        self.config = config
        
        # Initialize parent class (UnifiedGRPOTrainer)
        print("DEBUG: About to call super().__init__")
        sys.stdout.flush()
        logger.info("Calling parent UnifiedGRPOTrainer.__init__...")
        super().__init__(config=config, **kwargs)
        print("DEBUG: super().__init__ returned")
        sys.stdout.flush()
        logger.info("Parent initialization complete")
        
        # Ensure GRPO is initialized from parent
        if not hasattr(self, 'grpo_update'):
            self._initialize_grpo()
        
        # Joint training specific configuration
        joint_config = config.get('joint_training', {}) if config else {}
        
        # Override max_episodes from config if provided
        if config and 'max_episodes' in config:
            self.max_episodes = config['max_episodes']
        
        # Phase control
        self.episodes_per_phase = config.get('episodes_per_phase', 5) if config else 5
        logger.info(f"JointACBOTrainer set episodes_per_phase={self.episodes_per_phase} from config")
        # Also check if it's in joint_config
        if 'episodes_per_phase' in joint_config:
            logger.warning(f"Found episodes_per_phase in joint_config: {joint_config['episodes_per_phase']} (not using)")
        self.current_phase = joint_config.get('initial_phase', 'policy')
        self.phase_episode_count = 0
        
        # Performance tracking
        self.policy_performance = PhasePerformance()
        self.surrogate_performance = PhasePerformance()
        self.joint_metrics_history = []
        
        # Trajectory replay buffer
        replay_config = joint_config.get('replay', {})
        self.replay_buffer = TrajectoryReplayBuffer(
            capacity=replay_config.get('capacity', 100),
            prioritize_recent=replay_config.get('prioritize_recent', True)
        )
        self.replay_batch_size = replay_config.get('batch_size', 4)
        self.min_replay_size = replay_config.get('min_size', 10)
        self.use_replay = replay_config.get('enabled', True)
        
        # SCM management
        scm_config = joint_config.get('scm_management', {})
        self.use_curriculum = scm_config.get('use_curriculum', True)
        self.rotation_episodes = scm_config.get('rotation_episodes', 20)
        self.convergence_f1_threshold = scm_config.get('convergence_f1_threshold', 0.9)
        
        # Initialize SCM generator (will be set in train())
        self.scm_generator = None
        self.current_scm = None
        self.current_scm_metadata = {}
        self.episodes_on_current_scm = 0
        
        # Loss weights
        loss_weights = joint_config.get('loss_weights', {})
        self.policy_loss_weights = loss_weights.get('policy', {
            'absolute_target': 0.7,
            'information_gain': 0.3
        })
        
        # Adaptive features
        adaptive_config = joint_config.get('adaptive', {})
        self.use_adaptive_weights = adaptive_config.get('use_adaptive_weights', False)
        self.use_performance_rotation = adaptive_config.get('use_performance_rotation', True)
        
        # Debug and logging
        self.log_every = joint_config.get('log_every', 10)
        self.validate_every = joint_config.get('validate_every', 50)
        self.save_checkpoints = joint_config.get('save_checkpoints', True)
        
        # Initialize surrogate trainer for alternating updates
        self._initialize_surrogate_trainer()
        
        logger.info(f"Initialized JointACBOTrainer with:")
        logger.info(f"  Episodes per phase: {self.episodes_per_phase}")
        logger.info(f"  Replay buffer capacity: {self.replay_buffer.capacity}")
        logger.info(f"  Use curriculum: {self.use_curriculum}")
        logger.info(f"  Use adaptive weights: {self.use_adaptive_weights}")
        
    def _initialize_surrogate_trainer(self):
        """Initialize a separate surrogate trainer for alternating updates."""
        # We'll use the existing surrogate components from parent class
        # but add a separate trainer for the alternating phase
        if self.use_surrogate:
            self.surrogate_trainer = SurrogateBCTrainer(
                hidden_dim=self.config.get('surrogate_hidden_dim', 128),
                num_layers=self.config.get('surrogate_layers', 4),
                num_heads=self.config.get('surrogate_heads', 8),
                learning_rate=self.config.get('surrogate_lr', 1e-3),
                batch_size=32,
                max_epochs=10,  # Few epochs per phase
                seed=self.seed
            )
            # Initialize with existing params if available
            # The parent class uses 'surrogate_net' and we also have 'surrogate_model' 
            # Both should be available from parent initialization
            if self.surrogate_params is not None:
                self.surrogate_trainer.model_params = self.surrogate_params
                if hasattr(self, 'surrogate_net'):
                    self.surrogate_trainer.net = self.surrogate_net
                elif hasattr(self, 'surrogate_model'):
                    self.surrogate_trainer.net = self.surrogate_model
                    # Also create alias for compatibility
                    self.surrogate_net = self.surrogate_model
            logger.info("Initialized separate surrogate trainer for alternating updates")
    
    def train(self,
              scms: Union[List[Any], Dict[str, Any], Callable[[], Any], SCMCurriculumFactory],
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Train policy and surrogate jointly with alternating updates.
        
        Args:
            scms: Training SCMs or curriculum factory
            eval_scms: Optional evaluation SCMs
            
        Returns:
            Training results with joint metrics
        """
        print("DEBUG: train() method called")
        import sys
        sys.stdout.flush()
        
        logger.info("\n" + "="*70)
        logger.info("Starting Joint ACBO Training")
        logger.info("="*70)
        logger.info(f"  Max episodes: {self.max_episodes}")
        logger.info(f"  Current phase: {self.current_phase}")
        logger.info(f"  Episodes per phase: {self.episodes_per_phase}")
        logger.info("="*70 + "\n")
        
        print("DEBUG: About to save initial checkpoints")
        sys.stdout.flush()
        
        # Save initial models for before/after comparison
        logger.info("Saving initial model checkpoints for evaluation...")
        self._save_joint_checkpoint(episode=0)
        logger.info("✓ Initial checkpoints saved\n")
        
        print("DEBUG: Initial checkpoints saved")
        sys.stdout.flush()
        
        start_time = time.time()
        
        # Initialize SCM generator
        logger.info(f"About to initialize SCM generator with type: {type(scms)}")
        if isinstance(scms, SCMCurriculumFactory):
            logger.info("Creating AdaptiveSCMGenerator with curriculum factory")
            self.scm_generator = AdaptiveSCMGenerator(curriculum_factory=scms)
            logger.info("Using curriculum-based SCM generation")
        elif callable(scms):
            # Direct use of generator - no wrapping needed!
            logger.info("Using provided generator directly (no wrapping)")
            self.scm_generator = scms
            logger.info(f"Generator type: {type(scms).__name__}")
            # Check if it has the attributes we might need
            if hasattr(scms, 'min_vars') and hasattr(scms, 'max_vars'):
                logger.info(f"Generator range: {scms.min_vars}-{scms.max_vars} variables")
        else:
            # Create simple generator from fixed SCMs
            logger.info(f"Creating AdaptiveSCMGenerator with {len(scms) if isinstance(scms, list) else 'unknown'} fixed SCMs")
            self.scm_generator = AdaptiveSCMGenerator(fixed_scms=scms)
            logger.info(f"Using fixed SCM rotation with {len(self.scm_generator.scm_list)} SCMs")
        
        # Get initial SCM
        logger.info("Requesting initial SCM from generator...")
        if hasattr(self.scm_generator, 'request_new_scm'):
            # AdaptiveSCMGenerator style
            self.current_scm, self.current_scm_metadata = self.scm_generator.request_new_scm()
        else:
            # DiverseSCMGenerator style - just call it
            self.current_scm = self.scm_generator()
            self.current_scm_metadata = self.current_scm.get('metadata', {})
        logger.info(f"Got initial SCM: {self.current_scm.name if hasattr(self.current_scm, 'name') else 'unknown'}")
        self.episodes_on_current_scm = 0
        
        # Initialize frozen parameters
        self._freeze_models()
        
        # Main training loop
        episode_metrics = []
        logger.info(f"Starting main training loop for {self.max_episodes} episodes...")
        
        for episode in range(self.max_episodes):
            self.episode_count = episode
            
            # Print immediate progress
            print(f"\nEpisode {episode}/{self.max_episodes} | Phase: {self.current_phase} | phase_count={self.phase_episode_count}/{self.episodes_per_phase}")
            sys.stdout.flush()
            
            # Check if we should switch training phase
            if self._should_switch_phase():
                self._switch_training_phase()
                print(f"  → Phase switched to: {self.current_phase}")
                sys.stdout.flush()
            
            # Check if we should rotate SCM
            if self._should_rotate_scm():
                self._rotate_scm(episode_metrics)
            
            # Run collaborative episode (both phases use same method)
            scm_name = self.current_scm.name if hasattr(self.current_scm, 'name') else f"SCM-{episode%3}"
            print(f"  Running episode with SCM: {scm_name}")
            sys.stdout.flush()
            
            # Print model diagnostics at start of episode
            self._print_model_diagnostics("START", episode)
            
            metrics = self._run_collaborative_episode(episode)
            
            # Print model diagnostics at end of episode
            self._print_model_diagnostics("END", episode)
            
            # Print immediate results
            reward_str = f"{metrics.get('mean_reward', 0):.3f}" if self.current_phase == 'policy' else "N/A (surrogate phase)"
            print(f"  → Reward: {reward_str}, Info gain: {metrics.get('mean_info_gain', 0):.4f}, Target: {metrics.get('target_improvement', 0):.3f}")
            sys.stdout.flush()
            
            # Update performance tracking
            self._update_performance_tracking(metrics)
            
            # Add to metrics history
            episode_metrics.append(metrics)
            self.joint_metrics_history.append(self._create_joint_metrics(metrics))
            
            # Store trajectory in replay buffer if enabled
            if self.use_replay and hasattr(metrics, 'trajectory'):
                self._store_trajectory(metrics['trajectory'], metrics)
            
            # Logging
            if episode % self.log_every == 0:
                self._log_progress(episode, episode_metrics)
            
            # Validation
            if episode % self.validate_every == 0:
                self._validate_training(episode)
            
            # Save checkpoints
            if self.save_checkpoints and episode % 100 == 0:
                self._save_joint_checkpoint(episode)
            
            # Check for early stopping
            if self._check_convergence(episode_metrics):
                logger.info(f"Converged at episode {episode}")
                break
        
        # Final logging and results
        training_time = time.time() - start_time
        results = self._prepare_results(episode_metrics, training_time)
        
        # === FINAL TARGET VALUE SUMMARY ===
        if episode_metrics:
            print("\n" + "="*70)
            print("🎯 TARGET VALUE IMPROVEMENT SUMMARY")
            print("="*70)
            
            # Get first and last episode data
            first_episode = episode_metrics[0]
            last_episode = episode_metrics[-1]
            
            # Extract target values from all interventions
            first_targets = first_episode.get('all_target_values', [])
            last_targets = last_episode.get('all_target_values', [])
            
            if first_targets and last_targets:
                # Calculate averages
                first_avg = np.mean(first_targets)
                first_std = np.std(first_targets)
                last_avg = np.mean(last_targets)
                last_std = np.std(last_targets)
                
                # Calculate improvement (for minimization)
                improvement = first_avg - last_avg
                percent_improvement = (improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
                
                print(f"\n📊 Episode 1 (Initial Performance):")
                print(f"   Average target value: {first_avg:.4f} ± {first_std:.4f}")
                print(f"   Best target value:    {min(first_targets):.4f}")
                print(f"   Worst target value:   {max(first_targets):.4f}")
                print(f"   Interventions:        {len(first_targets)}")
                
                print(f"\n📊 Episode {len(episode_metrics)} (Final Performance):")
                print(f"   Average target value: {last_avg:.4f} ± {last_std:.4f}")
                print(f"   Best target value:    {min(last_targets):.4f}")
                print(f"   Worst target value:   {max(last_targets):.4f}")
                print(f"   Interventions:        {len(last_targets)}")
                
                print(f"\n✨ Improvement:")
                print(f"   Absolute: {improvement:.4f} (lower is better for minimization)")
                print(f"   Relative: {percent_improvement:.1f}%")
                
                # Check if improvement is significant
                if improvement > 0.1:  # Threshold for significance
                    print(f"   ✅ Policy successfully learned to minimize target!")
                elif improvement < -0.1:
                    print(f"   ⚠️ Target got worse - policy may need more training or different hyperparameters")
                else:
                    print(f"   ➖ No significant change - consider longer training")
                
                # Show trajectory over all episodes
                print(f"\n📈 Target Value Trajectory (episode averages):")
                trajectory_points = [
                    0,
                    len(episode_metrics)//4,
                    len(episode_metrics)//2,
                    3*len(episode_metrics)//4,
                    len(episode_metrics)-1
                ]
                for i in trajectory_points:
                    if i < len(episode_metrics):
                        ep_targets = episode_metrics[i].get('all_target_values', [])
                        if ep_targets:
                            ep_avg = np.mean(ep_targets)
                            ep_std = np.std(ep_targets)
                            print(f"   Episode {i+1:3d}: {ep_avg:.4f} ± {ep_std:.4f}")
                
                # Show best single intervention across all episodes
                all_best_targets = []
                for ep_metrics in episode_metrics:
                    ep_targets = ep_metrics.get('all_target_values', [])
                    if ep_targets:
                        all_best_targets.append(min(ep_targets))
                
                if all_best_targets:
                    overall_best = min(all_best_targets)
                    best_episode = all_best_targets.index(overall_best) + 1
                    print(f"\n🎆 Best single intervention:")
                    print(f"   Value: {overall_best:.4f} (achieved in episode {best_episode})")
            else:
                print("   ⚠️ No target values tracked - check episode execution")
            
            print("="*70)
            
            # === WITHIN-EPISODE IMPROVEMENT SUMMARY ===
            print("\n" + "="*70)
            print("📈 WITHIN-EPISODE LEARNING SUMMARY")
            print("="*70)
            print("\nThis shows how the policy improves WITHIN each episode on the SAME SCM:")
            print("(Comparing first 3 vs last 3 interventions in each episode)\n")
            
            within_episode_improvements = []
            for ep_metrics in episode_metrics:
                if 'within_episode' in ep_metrics:
                    we_data = ep_metrics['within_episode']
                    episode_num = ep_metrics['episode']
                    scm_name = ep_metrics.get('scm_name', 'Unknown')
                    improvement = we_data['improvement']
                    percent = we_data['percent_improvement']
                    
                    within_episode_improvements.append(improvement)
                    
                    print(f"Episode {episode_num:2d} (SCM: {scm_name:12s}): "
                          f"First 3={we_data['first_3_avg']:.3f} → Last 3={we_data['last_3_avg']:.3f} | "
                          f"Improvement: {improvement:+.3f} ({percent:+.1f}%)")
            
            if within_episode_improvements:
                avg_within_improvement = np.mean(within_episode_improvements)
                std_within_improvement = np.std(within_episode_improvements)
                positive_episodes = sum(1 for imp in within_episode_improvements if imp > 0)
                
                print(f"\n📊 Aggregate Within-Episode Statistics:")
                print(f"   Average improvement: {avg_within_improvement:+.4f} ± {std_within_improvement:.4f}")
                print(f"   Episodes with improvement: {positive_episodes}/{len(within_episode_improvements)} "
                      f"({positive_episodes/len(within_episode_improvements)*100:.1f}%)")
                
                if avg_within_improvement > 0:
                    print(f"   ✅ Policy consistently learns within episodes!")
                elif avg_within_improvement < -0.01:
                    print(f"   ⚠️  Policy tends to get worse within episodes - may need different approach")
                else:
                    print(f"   ➖ No consistent within-episode learning pattern")
                
                # Find best within-episode improvement
                if within_episode_improvements:
                    best_idx = np.argmax(within_episode_improvements)
                    best_improvement = within_episode_improvements[best_idx]
                    best_ep = episode_metrics[best_idx]['episode']
                    best_scm = episode_metrics[best_idx].get('scm_name', 'Unknown')
                    
                    print(f"\n🌟 Best within-episode improvement:")
                    print(f"   Episode {best_ep} (SCM: {best_scm}): {best_improvement:+.4f}")
            else:
                print("   ⚠️ Not enough interventions per episode for within-episode analysis")
                print("   (Need at least 6 interventions per episode)")
            
            print("="*70)
        
        # Save final models
        final_episode = len(episode_metrics)
        logger.info(f"\nSaving final model checkpoints (episode {final_episode})...")
        self._save_joint_checkpoint(episode=final_episode)
        
        logger.info("\n" + "="*70)
        logger.info("Joint Training Complete")
        logger.info(f"  Total time: {training_time:.2f}s")
        logger.info(f"  Final phase: {self.current_phase}")
        logger.info(f"  Episodes completed: {episode}")
        logger.info("="*70)
        
        # Print evaluation instructions
        logger.info("\n📊 To evaluate model improvement, run:")
        logger.info(f"python scripts/evaluate_acbo_methods_v2.py \\")
        logger.info(f"  --register_policy initial checkpoints/joint_ep0/policy.pkl \\")
        logger.info(f"  --register_policy final checkpoints/joint_ep{final_episode}/policy.pkl \\")
        logger.info(f"  --register_surrogate initial checkpoints/joint_ep0/surrogate.pkl \\")
        logger.info(f"  --register_surrogate final checkpoints/joint_ep{final_episode}/surrogate.pkl \\")
        logger.info(f"  --evaluate_pairs initial initial \\")
        logger.info(f"  --evaluate_pairs final final \\")
        logger.info(f"  --n_interventions 20 \\")
        logger.info(f"  --output_dir joint_evaluation_results")
        
        return results
    
    def _should_switch_phase(self) -> bool:
        """Determine if we should switch training phase."""
        # DEBUG: Log the actual values being compared
        logger.info(f"DEBUG _should_switch_phase: phase_episode_count={self.phase_episode_count}, episodes_per_phase={self.episodes_per_phase}")
        
        # Basic: switch after fixed number of episodes
        if self.phase_episode_count >= self.episodes_per_phase:
            logger.warning(f"PHASE SWITCH TRIGGERED: {self.phase_episode_count} >= {self.episodes_per_phase}")
            return True
        
        # Advanced: switch based on performance
        if self.use_performance_rotation:
            if self.current_phase == 'policy' and self.policy_performance.is_plateaued:
                if not self.surrogate_performance.is_plateaued:
                    return True
            elif self.current_phase == 'surrogate' and self.surrogate_performance.is_plateaued:
                if not self.policy_performance.is_plateaued:
                    return True
        
        return False
    
    def _switch_training_phase(self):
        """Switch between policy and surrogate training."""
        old_phase = self.current_phase
        self.current_phase = 'surrogate' if self.current_phase == 'policy' else 'policy'
        self.phase_episode_count = 0
        
        # Freeze models for new phase
        self._freeze_models()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Switching training phase: {old_phase} -> {self.current_phase}")
        logger.info(f"{'='*50}\n")
    
    def _should_rotate_scm(self) -> bool:
        """Determine if we should get a new SCM."""
        # Check episodes on current SCM
        if self.episodes_on_current_scm >= self.rotation_episodes:
            return True
        
        # Check convergence on current SCM
        if self.surrogate_performance.recent_f1_scores:
            recent_f1 = np.mean(list(self.surrogate_performance.recent_f1_scores))
            if recent_f1 > self.convergence_f1_threshold:
                return True
        
        # Check if both models plateaued
        if self.policy_performance.is_plateaued and self.surrogate_performance.is_plateaued:
            return True
        
        return False
    
    def _rotate_scm(self, episode_metrics: List[Dict]):
        """Rotate to a new SCM."""
        # Get performance metrics for current SCM
        if episode_metrics:
            recent_metrics = episode_metrics[-min(10, len(episode_metrics)):]
            performance = {
                'mean_reward': np.mean([m.get('mean_reward', 0) for m in recent_metrics]),
                'mean_f1': np.mean([m.get('structure_metrics', {}).get('f1_score', 0) 
                                   for m in recent_metrics])
            }
        else:
            performance = None
        
        # Request new SCM
        old_scm_name = self.current_scm_metadata.get('name', 'unknown')
        if hasattr(self.scm_generator, 'request_new_scm'):
            # AdaptiveSCMGenerator style
            self.current_scm, self.current_scm_metadata = self.scm_generator.request_new_scm(performance)
        else:
            # DiverseSCMGenerator style - just call it
            self.current_scm = self.scm_generator()
            self.current_scm_metadata = self.current_scm.get('metadata', {})
        self.episodes_on_current_scm = 0
        
        new_scm_name = self.current_scm_metadata.get('name', 'unknown')
        logger.info(f"Rotated SCM: {old_scm_name} -> {new_scm_name}")
    
    def _run_collaborative_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Run a collaborative episode where both models contribute to trajectory.
        
        Both policy and surrogate phases use this method:
        - Policy phase: Policy learns via GRPO, surrogate frozen
        - Surrogate phase: Policy frozen, surrogate learns
        
        The key is that both models always participate in data collection.
        """
        # Get frozen parameters based on phase
        if self.current_phase == 'policy':
            surrogate_params = self.frozen_surrogate_params if hasattr(self, 'frozen_surrogate_params') else self.surrogate_params
            policy_params = self.policy_params  # Trainable
        else:  # surrogate phase
            policy_params = self.frozen_policy_params if hasattr(self, 'frozen_policy_params') else self.policy_params
            surrogate_params = self.surrogate_params  # Trainable
        
        # Initialize collaborative buffer
        collab_buffer = ExperienceBuffer()
        variables = list(get_variables(self.current_scm))
        target_var = get_target(self.current_scm)
        true_parents = set(get_parents(self.current_scm, target_var))
        
        # Sample observational data
        self.rng_key, obs_key = random.split(self.rng_key)
        from ..mechanisms.linear import sample_from_linear_scm
        obs_samples = sample_from_linear_scm(self.current_scm, self.obs_per_episode, seed=int(obs_key[0]))
        
        # Add observations to buffer
        for sample in obs_samples:
            collab_buffer.add_observation(sample)
        
        # Track metrics
        all_rewards = []
        all_info_gains = []
        all_target_values = []
        
        # Track reward component contributions for episode summary
        component_contributions = {
            'target_delta': [],
            'info_gain': [],
            'direct_parent': []
        }
        
        # Debug buffer at start
        if episode_idx % 5 == 0:
            print(f"\n  📦 Buffer state at episode {episode_idx} start:")
            obs_count = len(collab_buffer.get_observations())
            int_count = len(collab_buffer.get_interventions())
            # Use same logic as buffer extraction
            samples_with_posteriors_initial = collab_buffer.get_all_samples_with_posteriors()
            post_count = len(samples_with_posteriors_initial)
            print(f"     Observations: {obs_count}")
            print(f"     Interventions: {int_count}")
            print(f"     Total samples with posteriors: {post_count}")
            # Count how many actually have non-None posteriors
            non_none_posteriors = sum(1 for _, p in samples_with_posteriors_initial if p is not None)
            if non_none_posteriors > 0:
                print(f"     Non-None posteriors: {non_none_posteriors}")
        
        # Collaborative intervention loop
        for step in range(self.max_interventions):
            self.rng_key, step_key = random.split(self.rng_key)
            
            if episode_idx % 5 == 0:
                print(f"\n  🔄 Intervention {step+1}/{self.max_interventions}:")
            
            # === STEP 1: SURROGATE CONTRIBUTION ===
            # Import here so it's available in both branches
            from .three_channel_converter import buffer_to_three_channel_tensor
            
            # Compute posterior with current surrogate state
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                # Get tensor from buffer
                tensor_3ch, mapper_3ch = buffer_to_three_channel_tensor(
                    collab_buffer, target_var, max_history_size=100, standardize=True
                )
                
                if episode_idx % 5 == 0:
                    # Validate tensor has correct timesteps
                    actual_timesteps = 0
                    for t in range(tensor_3ch.shape[0]):
                        if jnp.any(tensor_3ch[t, :, :] != 0):
                            actual_timesteps += 1
                    expected_timesteps = len(collab_buffer.get_observations()) + len(collab_buffer.get_interventions())
                    print(f"     Tensor validation: shape={tensor_3ch.shape}, "
                          f"actual_timesteps={actual_timesteps}, expected={expected_timesteps}")
                
                # Use appropriate params (frozen or trainable)
                # The surrogate_predict_fn uses internal params, not passed ones
                # We need to temporarily set params if different
                current_posterior = self.surrogate_predict_fn(
                    tensor_3ch, target_var, variables
                )
            else:
                # No surrogate - use uniform posterior
                from .five_channel_converter import create_uniform_posterior
                current_posterior = create_uniform_posterior(variables, target_var)
            
            # === STEP 2: POLICY CONTRIBUTION ===
            if self.current_phase == 'policy':
                # Show current posterior beliefs that policy will use (for validation)
                if episode_idx <= 5 and step == 0 and current_posterior:  # First step of first few episodes
                    if 'marginal_parent_probs' in current_posterior:
                        probs = current_posterior['marginal_parent_probs']
                        print(f"\n  🧠 Policy Input - Current Posterior Beliefs:")
                        for var, parents in list(probs.items())[:3]:
                            # Check if parents is a dict (expected) or something else
                            if isinstance(parents, dict) and parents:
                                sorted_parents = sorted(parents.items(), key=lambda x: x[1], reverse=True)
                                if sorted_parents:
                                    top_parent = sorted_parents[0]
                                    print(f"     {var}: Most likely parent = {top_parent[0]} (prob={top_parent[1]:.3f})")
                                    if len(sorted_parents) > 1:
                                        print(f"         Other candidates: {sorted_parents[1:][:2]}")
                            elif isinstance(parents, (int, float)):
                                # Handle case where it's a single probability value
                                print(f"     {var}: Single prob value = {parents:.3f}")
                            else:
                                print(f"     {var}: Unexpected format: {type(parents)}")
                
                # Full GRPO: Generate batch and learn
                candidates = self._generate_grpo_batch_with_info_gain(
                    collab_buffer, current_posterior, target_var, 
                    variables, self.current_scm, policy_params, 
                    surrogate_params, step_key
                )
                
                # Compute GRPO advantages and select best
                best_candidate = self._select_best_grpo_intervention(candidates)
                
                # Update policy parameters using GRPO (only in policy phase)
                self._update_policy_with_grpo(candidates)
                
                # Use selected intervention
                selected_intervention = best_candidate['intervention']
                selected_outcome = best_candidate['outcome']
                selected_posterior = best_candidate['posterior_after']
                
                # Track metrics
                all_rewards.append(best_candidate['reward'])
                all_info_gains.append(best_candidate.get('reward_components', {}).get('info_gain', 0.0))
                
                # Validate GRPO reward calculation (for debugging)
                if episode_idx <= 10 or episode_idx % 10 == 0:
                    # Extract all reward information from best candidate
                    raw_components = best_candidate.get('reward_components_raw', {})
                    scaled_components = best_candidate.get('reward_components', {})
                    
                    # Get target value from outcome
                    from ..data_structures.sample import get_values as get_sample_values
                    target_val = get_sample_values(selected_outcome).get(target_var, 0.0)
                    
                    # Get weights being used
                    weights = self.policy_loss_weights if hasattr(self, 'policy_loss_weights') else {
                        'target_delta': 0.5,
                        'information_gain': 0.3,
                        'direct_parent': 0.2
                    }
                    
                    print(f"\n     💰 GRPO Reward Breakdown (Step {step+1}):")
                    print(f"        Selected: {best_candidate['variable']} = {best_candidate['value']:.3f}")
                    
                    print(f"\n        === RAW VALUES ===")
                    print(f"        Target outcome: {raw_components.get('target_value', target_val):.4f}")
                    print(f"        Info gain (entropy reduction): {raw_components.get('info_gain_raw', 0.0):.4f}")
                    print(f"        Direct parent: {raw_components.get('direct_parent', 0.0):.1f}")
                    
                    print(f"\n        === SCALED COMPONENTS ===")
                    target_delta = scaled_components.get('target_delta', 0.0)
                    info_gain_scaled = scaled_components.get('info_gain', 0.0)
                    direct_parent = scaled_components.get('direct_parent', 0.0)
                    
                    print(f"        Target delta: {target_delta:.4f} (weight: {weights.get('target_delta', 0.5):.2f})")
                    print(f"          → Contribution: {weights.get('target_delta', 0.5) * target_delta:.4f}")
                    print(f"        Info gain: {info_gain_scaled:.4f} (weight: {weights.get('information_gain', 0.3):.2f})")
                    print(f"          → Contribution: {weights.get('information_gain', 0.3) * info_gain_scaled:.4f}")
                    print(f"        Direct parent: {direct_parent:.1f} (weight: {weights.get('direct_parent', 0.2):.2f})")
                    print(f"          → Contribution: {weights.get('direct_parent', 0.2) * direct_parent:.4f}")
                    
                    print(f"\n        === FINAL REWARD ===")
                    print(f"        Total: {best_candidate['reward']:.4f}")
                    
                    # Add interpretation
                    dominant_component = 'target'
                    max_contrib = abs(weights.get('target_delta', 0.5) * target_delta)
                    if abs(weights.get('information_gain', 0.3) * info_gain_scaled) > max_contrib:
                        dominant_component = 'info_gain'
                        max_contrib = abs(weights.get('information_gain', 0.3) * info_gain_scaled)
                    if abs(weights.get('direct_parent', 0.2) * direct_parent) > max_contrib:
                        dominant_component = 'parent'
                    print(f"        Dominant: {dominant_component}")
                    
                    # Track contributions for episode summary
                    component_contributions['target_delta'].append(
                        weights.get('target_delta', 0.5) * target_delta
                    )
                    component_contributions['info_gain'].append(
                        weights.get('information_gain', 0.3) * info_gain_scaled
                    )
                    component_contributions['direct_parent'].append(
                        weights.get('direct_parent', 0.2) * direct_parent
                    )
                    if len(all_rewards) > 1:
                        print(f"        Running avg reward: {np.mean(all_rewards):.4f}")
                
            else:  # surrogate phase
                # Simple collection: Use frozen policy to select single intervention
                selected_intervention, selected_outcome = self._collect_single_intervention(
                    collab_buffer, current_posterior, target_var, 
                    variables, self.current_scm, policy_params, step_key
                )
                
                # Compute new posterior after intervention for consistency
                temp_buffer = ExperienceBuffer()
                
                # Copy observations
                for obs in collab_buffer.get_observations():
                    temp_buffer.add_observation(obs)
                
                # Copy existing interventions
                for old_intervention, old_sample in collab_buffer.get_interventions():
                    temp_buffer.add_intervention(old_intervention, old_sample)
                
                # Add new intervention
                temp_buffer.add_intervention(selected_intervention, selected_outcome)
                
                tensor_3ch_new, mapper_3ch_new = buffer_to_three_channel_tensor(
                    temp_buffer, target_var, max_history_size=100, standardize=True
                )
                selected_posterior = self.surrogate_predict_fn(
                    tensor_3ch_new, target_var, variables
                )
                
                # Show what surrogate predicts (for validation)
                if episode_idx <= 5 and step <= 1 and selected_posterior:
                    if 'marginal_parent_probs' in selected_posterior:
                        probs = selected_posterior['marginal_parent_probs']
                        print(f"\n  🔮 Surrogate Prediction - Updated Posterior:")
                        true_parents = list(get_parents(self.current_scm, target_var))
                        if target_var in probs and probs[target_var]:
                            sorted_parents = sorted(probs[target_var].items(), key=lambda x: x[1], reverse=True)
                            if sorted_parents:
                                top_parent = sorted_parents[0]
                                is_correct = top_parent[0] in true_parents
                                correct_marker = "✅" if is_correct else "❌"
                                print(f"     Target {target_var}: Predicted parent = {top_parent[0]} (prob={top_parent[1]:.3f}) {correct_marker}")
                                print(f"     True parents: {true_parents}")
                
                # Compute info gain for tracking
                info_gain = compute_information_gain(current_posterior, selected_posterior, target_var)
                all_info_gains.append(info_gain)
            
            # === STEP 3: COMMIT TO BUFFER ===
            # Track buffer state before adding
            before_interventions = len(collab_buffer.get_interventions())
            samples_before = collab_buffer.get_all_samples_with_posteriors()
            before_posteriors = len(samples_before)
            
            # Store intervention details for validation (before adding to buffer)
            if episode_idx <= 10 or episode_idx % 5 == 0:
                int_targets = selected_intervention.get('targets', frozenset())
                int_values = selected_intervention.get('values', {})
                print(f"\n     📝 Storing Intervention (Step {step+1}):")
                print(f"        Targets: {int_targets}")
                print(f"        Values: {int_values}")
            
            # Store complete timestep with posterior
            collab_buffer.add_intervention(
                selected_intervention, 
                selected_outcome,
                posterior=selected_posterior
            )
            
            # Validate buffer update (more frequent for debugging)
            if episode_idx <= 10 or episode_idx % 5 == 0:  # More verbose for first episodes
                after_interventions = len(collab_buffer.get_interventions())
                # Use the same logic as buffer extraction
                samples_with_posteriors = collab_buffer.get_all_samples_with_posteriors()
                after_posteriors = len(samples_with_posteriors)
                
                # Check exactly one intervention added
                interventions_added = after_interventions - before_interventions
                posteriors_added = after_posteriors - before_posteriors
                
                print(f"\n     📊 Buffer Validation (Step {step+1}):")
                print(f"        Interventions: {before_interventions} → {after_interventions} (added: {interventions_added})")
                print(f"        Posteriors: {before_posteriors} → {after_posteriors} (added: {posteriors_added})")
                
                # Validate correct number added
                if interventions_added != 1:
                    print(f"        ⚠️ WARNING: Expected 1 intervention, got {interventions_added}")
                if posteriors_added != 1:
                    print(f"        ⚠️ WARNING: Expected 1 posterior, got {posteriors_added}")
                
                # Verify intervention write-read consistency
                last_intervention, last_outcome = collab_buffer.get_interventions()[-1]
                stored_targets = last_intervention.get('targets', frozenset())
                stored_values = last_intervention.get('values', {})
                
                if stored_targets == int_targets and stored_values == int_values:
                    print(f"        ✅ Intervention write-read verified! Targets: {stored_targets}")
                else:
                    print(f"        ⚠️ Intervention write-read mismatch!")
                    print(f"           Wrote targets: {int_targets}, Read: {stored_targets}")
                    print(f"           Wrote values: {int_values}, Read: {stored_values}")
                
                # Check posterior content and write-read consistency
                if selected_posterior:
                    if 'marginal_parent_probs' in selected_posterior:
                        probs = selected_posterior['marginal_parent_probs']
                        n_vars_with_probs = len(probs)
                        print(f"        Posterior has probs for {n_vars_with_probs} variables: {list(probs.keys())[:3]}...")
                        
                        # Verify write-read consistency
                        last_sample, last_posterior = samples_with_posteriors[-1]
                        if last_posterior and 'marginal_parent_probs' in last_posterior:
                            # Check if the stored posterior matches what we just added
                            stored_probs = last_posterior['marginal_parent_probs']
                            if stored_probs == probs:
                                print(f"        ✅ Posterior write-read verified!")
                            else:
                                print(f"        ⚠️ Posterior write-read mismatch!")
                                # Show first difference
                                for var in probs:
                                    if var in stored_probs and stored_probs[var] != probs[var]:
                                        print(f"           Variable {var}: wrote {probs[var]}, read {stored_probs[var]}")
                                        break
                    else:
                        print(f"        ⚠️ Posterior missing 'marginal_parent_probs'")
                else:
                    print(f"        ⚠️ No posterior attached!")
                
                # Validate tensor reconstruction
                if step > 0:  # After first intervention
                    try:
                        from ..training.five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
                        test_tensor, test_mapper, _ = buffer_to_five_channel_tensor_with_posteriors(
                            collab_buffer, target_var, max_history_size=100, standardize=True
                        )
                        print(f"        ✅ Tensor reconstruction successful: shape {test_tensor.shape}")
                        # Check tensor timesteps match buffer
                        expected_timesteps = len(collab_buffer.get_observations()) + len(collab_buffer.get_interventions())
                        actual_timesteps = test_tensor.shape[0]  # This will be max_history_size
                        # The tensor uses a sliding window of the most recent samples
                        if expected_timesteps > actual_timesteps:
                            print(f"        Buffer: {expected_timesteps} samples → Tensor uses last {actual_timesteps} (sliding window)")
                        else:
                            print(f"        Buffer: {expected_timesteps} samples, Tensor capacity: {actual_timesteps}")
                            
                        # Verify posterior is embedded in channel 3 (parent probs)
                        parent_prob_channel = test_tensor[:, :, 3]  # Channel 3 is parent probs
                        non_zero_probs = jnp.sum(parent_prob_channel != 0.0)
                        print(f"        Parent prob channel has {non_zero_probs} non-zero entries")
                    except Exception as e:
                        print(f"        ❌ Tensor reconstruction failed: {e}")
            
            # Track target value
            from ..data_structures.sample import get_values as get_sample_values
            target_value = get_sample_values(selected_outcome).get(target_var, 0.0)
            all_target_values.append(target_value)
        
        # === STEP 4: TRAIN SURROGATE (if in surrogate phase) ===
        if self.current_phase == 'surrogate' and self.use_surrogate:
            # Train surrogate on the collected buffer
            surrogate_loss = self._train_surrogate_on_buffer(collab_buffer, self.current_scm)
        else:
            surrogate_loss = 0.0
        
        # === STEP 5: EPISODE SUMMARY ===
        # Print reward component summary for the episode
        if component_contributions['target_delta']:  # If we tracked components
            print(f"\n  📊 Episode {episode_idx} Reward Component Summary:")
            print(f"     Average contributions per intervention:")
            
            avg_target = np.mean(component_contributions['target_delta'])
            avg_info = np.mean(component_contributions['info_gain'])
            avg_parent = np.mean(component_contributions['direct_parent'])
            total_avg = avg_target + avg_info + avg_parent
            
            print(f"       Target delta:  {avg_target:+.4f} ({abs(avg_target/total_avg)*100:.1f}%)")
            print(f"       Info gain:     {avg_info:+.4f} ({abs(avg_info/total_avg)*100:.1f}%)")
            print(f"       Direct parent: {avg_parent:+.4f} ({abs(avg_parent/total_avg)*100:.1f}%)")
            print(f"       Total average: {total_avg:+.4f}")
            
            # Identify which component is driving learning
            dominant = 'target delta'
            if abs(avg_info) > abs(avg_target) and abs(avg_info) > abs(avg_parent):
                dominant = 'info gain'
            elif abs(avg_parent) > abs(avg_target) and abs(avg_parent) > abs(avg_info):
                dominant = 'direct parent'
            print(f"\n     💡 Dominant driver: {dominant}")
        
        # === STEP 6: COMPUTE METRICS ===
        # Compute structure learning metrics
        structure_metrics = {}
        if self.use_surrogate and selected_posterior:
            structure_metrics = self._compute_structure_metrics(collab_buffer)
        
        # Prepare results
        metrics = {
            'episode': episode_idx,
            'training_phase': self.current_phase,
            'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'mean_info_gain': float(np.mean(all_info_gains)) if all_info_gains else 0.0,
            'information_gain': float(np.mean(all_info_gains)) if all_info_gains else 0.0,  # For compatibility
            'total_info_gain': float(np.sum(all_info_gains)) if all_info_gains else 0.0,
            'initial_target': float(all_target_values[0]) if all_target_values else 0.0,
            'final_target': float(all_target_values[-1]) if all_target_values else 0.0,
            'target_improvement': float(all_target_values[-1] - all_target_values[0]) if all_target_values else 0.0,
            'all_target_values': all_target_values,  # Store all raw target values for analysis
            'structure_metrics': structure_metrics,
            'surrogate_loss': surrogate_loss,
            'trajectory': collab_buffer,  # Store for replay
            'n_interventions': len(all_target_values),
            'scm_name': self.current_scm_metadata.get('name', 'Unknown-SCM')  # Track which SCM
        }
        
        # === WITHIN-EPISODE ANALYSIS ===
        if len(all_target_values) >= 6:  # Need at least 6 interventions for first 3 vs last 3
            first_3 = all_target_values[:3]
            last_3 = all_target_values[-3:]
            
            first_avg = np.mean(first_3)
            first_std = np.std(first_3)
            last_avg = np.mean(last_3)
            last_std = np.std(last_3)
            
            # For minimization, improvement means last < first
            within_episode_improvement = first_avg - last_avg
            percent_improvement = (within_episode_improvement / abs(first_avg)) * 100 if first_avg != 0 else 0
            
            # Store within-episode metrics
            metrics['within_episode'] = {
                'first_3_avg': first_avg,
                'first_3_std': first_std,
                'last_3_avg': last_avg,
                'last_3_std': last_std,
                'improvement': within_episode_improvement,
                'percent_improvement': percent_improvement
            }
            
            # Print within-episode analysis
            print(f"\n  📈 Episode {episode_idx} Within-Episode Analysis (SCM: {metrics['scm_name']}):")
            print(f"     First 3 interventions: {first_avg:.4f} ± {first_std:.4f}")
            print(f"     Last 3 interventions:  {last_avg:.4f} ± {last_std:.4f}")
            print(f"     Within-episode improvement: {within_episode_improvement:+.4f} ({percent_improvement:+.1f}%)")
            
            if within_episode_improvement > 0:
                print(f"     ✅ Policy improved within this episode!")
            elif within_episode_improvement < -0.01:
                print(f"     ⚠️  Policy got worse within this episode")
            else:
                print(f"     ➖ No significant change within episode")
        
        # Update performance tracking
        self._update_performance_tracking(metrics)
        self.phase_episode_count += 1
        logger.info(f"DEBUG: Incremented phase_episode_count to {self.phase_episode_count} (episodes_per_phase={self.episodes_per_phase})")
        self.episodes_on_current_scm += 1
        
        # Increment debug counter
        if hasattr(self, '_debug_episode_count'):
            self._debug_episode_count += 1
        
        # Track intervention patterns for diagnostics
        self._track_intervention_patterns(episode_idx, collab_buffer, target_var)
        
        return metrics
    
    def _generate_grpo_batch_with_info_gain(self, buffer, current_posterior, target_var, 
                                           variables, scm, policy_params, 
                                           surrogate_params, key) -> List[Dict]:
        """
        Generate GRPO batch with information gain computation.
        
        This computes hypothetical posteriors for each candidate intervention,
        which doubles the surrogate forward passes but provides info gain rewards.
        """
        candidates = []
        
        # Convert buffer to 5-channel tensor once
        from .five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
        tensor_5ch, mapper, _ = buffer_to_five_channel_tensor_with_posteriors(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Generate batch of candidate interventions
        for i in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # Sample intervention from policy
            key, policy_key = random.split(step_key)
            policy_output = self.policy_fn.apply(
                policy_params, policy_key, tensor_5ch, mapper.target_idx
            )
            
            # Sample variable selection
            var_logits = policy_output['variable_logits']
            key, var_key = random.split(step_key)
            selected_var_idx = random.categorical(var_key, var_logits)
            selected_var = mapper.get_name(int(selected_var_idx))
            
            # Sample intervention value
            value_params = policy_output['value_params']
            mean = value_params[selected_var_idx, 0]
            log_std = value_params[selected_var_idx, 1]
            std = jnp.exp(log_std)
            key, val_key = random.split(step_key)
            raw_intervention_value = mean + std * random.normal(val_key)
            
            # Soft tanh-based mapping to variable range (preserves gradients)
            var_ranges = scm.get('metadata', {}).get('variable_ranges', {})
            if selected_var in var_ranges:
                min_val, max_val = var_ranges[selected_var]
            else:
                # Default range if not specified
                min_val, max_val = -10.0, 10.0
            
            # Map to range using tanh (smooth, differentiable)
            # Scale factor controls how quickly we approach boundaries
            scale_factor = 2.0  # Adjust for different mapping behavior
            range_center = (max_val + min_val) / 2
            range_radius = (max_val - min_val) / 2
            intervention_value = range_center + range_radius * jnp.tanh(raw_intervention_value / scale_factor)
            
            # Create intervention
            from ..interventions.handlers import create_perfect_intervention
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            # Apply intervention to get outcome
            from ..environments.sampling import sample_with_intervention
            key, sample_key = random.split(step_key)
            outcome_samples = sample_with_intervention(
                scm, intervention, n_samples=1, seed=int(sample_key[0])
            )
            outcome = outcome_samples[0] if outcome_samples else None
            
            # Create hypothetical buffer for new posterior
            # Since buffer doesn't have copy(), create new and add samples
            hypo_buffer = ExperienceBuffer()
            
            # Copy observations
            for obs in buffer.get_observations():
                hypo_buffer.add_observation(obs)
            
            # Copy existing interventions
            for old_intervention, old_sample in buffer.get_interventions():
                hypo_buffer.add_intervention(old_intervention, old_sample)
            
            # Add new hypothetical intervention
            hypo_buffer.add_intervention(intervention, outcome)
            
            # Compute hypothetical posterior (EXPENSIVE but valuable)
            from .three_channel_converter import buffer_to_three_channel_tensor
            hypo_tensor, hypo_mapper = buffer_to_three_channel_tensor(
                hypo_buffer, target_var, max_history_size=100, standardize=True
            )
            
            hypo_posterior = self.surrogate_predict_fn(
                hypo_tensor, target_var, variables
            )
            
            # Compute reward components (will be scaled after all candidates)
            from ..data_structures.sample import get_values as get_sample_values
            target_value = get_sample_values(outcome).get(target_var, 0.0)
            
            # Component 2: Information gain (raw)
            info_gain_raw = compute_information_gain(current_posterior, hypo_posterior, target_var)
            
            # Component 3: Direct parent reward (binary)
            # Use the standard SCM interface to get true parents
            from ..data_structures.scm import get_parents
            try:
                # Standard method that should work with all SCMs
                true_parents_set = set(get_parents(scm, target_var))
                true_parents = list(true_parents_set)
            except Exception as e:
                # Fallback methods if standard method fails
                true_parents = []
                if hasattr(scm, 'get_parent_names'):
                    true_parents = scm.get_parent_names(target_var)
                elif hasattr(scm, 'parents'):
                    parents_dict = scm.parents if isinstance(scm.parents, dict) else {}
                    true_parents = parents_dict.get(target_var, [])
                elif hasattr(scm, 'get_parents'):
                    true_parents = list(scm.get_parents(target_var))
                
                if i == 0:  # Log error once
                    logger.warning(f"Failed to get parents using standard method: {e}")
            
            # Check if selected variable is a true parent
            is_direct_parent = 1.0 if selected_var in true_parents else 0.0
            
            # Enhanced debug for parent matching
            if i == 0 and not hasattr(self, '_parent_debug_done'):
                print(f"    DEBUG Parent Detection:")
                print(f"      Target variable: '{target_var}'")
                print(f"      Selected variable: '{selected_var}'")
                print(f"      True parents: {true_parents}")
                print(f"      Is direct parent: {is_direct_parent}")
                print(f"      SCM type: {type(scm).__name__}")
                self._parent_debug_done = True
            
            # Store raw components for group-based scaling
            reward_components_raw = {
                'target_value': target_value,
                'info_gain_raw': info_gain_raw,
                'direct_parent': is_direct_parent
            }
            
            # Debug raw components (before scaling)
            if i == 0 and not hasattr(self, '_debug_episode_count'):
                self._debug_episode_count = 0
            
            # Compute log probability for GRPO
            # Variable selection log prob
            log_probs_var = jax.nn.log_softmax(var_logits)
            log_prob_var = log_probs_var[selected_var_idx]
            
            # Value selection log prob (Gaussian)
            log_prob_val = -0.5 * ((intervention_value - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            
            # Total log prob
            log_prob = log_prob_var + log_prob_val
            
            candidates.append({
                'intervention': intervention,
                'outcome': outcome,
                'posterior_before': current_posterior,
                'posterior_after': hypo_posterior,
                'reward_components_raw': reward_components_raw,  # Raw components
                'log_prob': float(log_prob),
                'variable': selected_var,
                'value': float(intervention_value),
                'policy_output': policy_output,  # Store for GRPO update
                'tensor_5ch': tensor_5ch,  # Store tensor for GRPO update
                'target_idx': mapper.target_idx,  # Store target index
                'mapper': mapper  # Store mapper for variable conversion
            })
        
        # Now scale rewards using group statistics (GRPO best practice)
        candidates = self._scale_rewards_group_based(candidates, target_var)
        
        return candidates
    
    def _scale_rewards_group_based(self, candidates: List[Dict], target_var: str) -> List[Dict]:
        """
        Scale rewards using group statistics per GRPO best practices.
        
        Components:
        1. Target delta: Scale using group mean/std to [-1, 1]
        2. Information gain: Scale to [-1, 1] 
        3. Direct parent: Binary (0 or 1)
        """
        # Extract raw values from all candidates
        target_values = [c['reward_components_raw']['target_value'] for c in candidates]
        info_gains = [c['reward_components_raw']['info_gain_raw'] for c in candidates]
        
        # Compute group statistics for target values
        group_mean = jnp.mean(jnp.array(target_values))
        group_std = jnp.std(jnp.array(target_values))
        
        # Compute info gain statistics
        info_mean = jnp.mean(jnp.array(info_gains))
        info_std = jnp.std(jnp.array(info_gains))
        
        # Get reward weights
        weights = self.policy_loss_weights if hasattr(self, 'policy_loss_weights') else {
            'target_delta': 0.5,
            'information_gain': 0.3,
            'direct_parent': 0.2
        }
        
        # Scale each candidate's rewards
        for i, c in enumerate(candidates):
            raw = c['reward_components_raw']
            
            # Component 1: Target reward (raw value, let GRPO normalize)
            # Fix double normalization: use raw target value, not normalized delta
            # For minimization: negative target values are good (positive reward)
            target_reward = -raw['target_value']  # Negate for minimization
            # Mild clipping for stability but preserve signal strength
            target_reward = jnp.clip(target_reward, -10.0, 10.0)
            
            # Component 2: Information gain (scale to [-1, 1])
            if info_std > 1e-8:
                # Use proper normalization with epsilon
                info_gain_scaled = (raw['info_gain_raw'] - info_mean) / (info_std + 1e-8)
                info_gain_scaled = jnp.clip(info_gain_scaled, -1.0, 1.0)
            else:
                info_gain_scaled = raw['info_gain_raw']  # Use raw if no variance
            
            # Component 3: Direct parent (already binary)
            direct_parent = raw['direct_parent']
            
            # Combine with weights
            total_reward = (
                weights.get('target_delta', 0.5) * target_reward +  # Using raw reward now
                weights.get('information_gain', 0.3) * info_gain_scaled +
                weights.get('direct_parent', 0.2) * direct_parent
            )
            
            # Store scaled components
            c['reward'] = float(total_reward)
            c['reward_components'] = {
                'target_delta': float(target_reward),  # Actually target_reward now
                'info_gain': float(info_gain_scaled),
                'direct_parent': float(direct_parent)
            }
            
            # Debug first candidate of first few episodes
            if i == 0 and hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 5:
                print(f"\n    🔍 GRPO Reward Components (Candidate {i+1}):")
                print(f"       Variable: {c['variable']}, Value: {c['value']:.3f}")
                print(f"       Raw target value: {raw['target_value']:.3f}")
                print(f"       Target reward (negated): {target_reward:.3f} (weight: {weights.get('target_delta', 0.5)})")
                print(f"       Info gain: {info_gain_scaled:.3f} (weight: {weights.get('information_gain', 0.3)})")
                print(f"       Direct parent: {direct_parent:.1f} (weight: {weights.get('direct_parent', 0.2)})")
                print(f"       Total reward: {total_reward:.3f}")
                print(f"       → GRPO will normalize this in advantages")
                if i == 0:  # Print true parents only once
                    # Get true parents from SCM for debugging
                    from ..data_structures.scm import get_parents
                    if hasattr(self, 'current_scm'):
                        true_parents_debug = list(get_parents(self.current_scm, target_var))
                        print(f"       True parents of {target_var}: {true_parents_debug}")
        
        return candidates
    
    def _validate_reward_components(self, sample_candidates: List[Dict]):
        """Validate reward components for debugging."""
        print(f"\n    🔍 Validating Reward Components (first {len(sample_candidates)} candidates):")
        for i, c in enumerate(sample_candidates):
            raw = c.get('reward_components_raw', {})
            scaled = c.get('reward_components', {})
            print(f"       Candidate {i+1} - Var: {c['variable']}, Val: {c.get('value', 0):.3f}")
            print(f"         Raw: target={raw.get('target_value', 0):.3f}, "
                  f"info_gain={raw.get('info_gain_raw', 0):.3f}, "
                  f"parent={raw.get('direct_parent', 0):.1f}")
            print(f"         Scaled: target_delta={scaled.get('target_delta', 0):.3f}, "
                  f"info_gain={scaled.get('info_gain', 0):.3f}, "
                  f"parent={scaled.get('direct_parent', 0):.1f}")
            print(f"         Total reward: {c.get('reward', 0):.3f}")
    
    def _analyze_gradient_flow(self, grpo_metrics, param_change, total_change):
        """Analyze gradient flow for debugging."""
        print(f"\n    🔬 Gradient Flow Analysis:")
        print(f"       Policy loss: {grpo_metrics.policy_loss:.4f}")
        print(f"       Gradient norm: {grpo_metrics.grad_norm:.6f}")
        print(f"       Total param change: {total_change:.6f}")
        
        # Check if gradients are flowing
        if grpo_metrics.grad_norm < 1e-6:
            print(f"       ⚠️ WARNING: Very small gradients! Policy may not be learning.")
        elif grpo_metrics.grad_norm > 10.0:
            print(f"       ⚠️ WARNING: Large gradients! Consider reducing learning rate.")
        else:
            print(f"       ✓ Gradient norm in healthy range")
        
        # Check parameter updates
        if total_change < 1e-6:
            print(f"       ⚠️ WARNING: Parameters barely changing! Check learning rate or loss.")
        elif total_change > 1.0:
            print(f"       ⚠️ WARNING: Large parameter changes! May cause instability.")
        else:
            print(f"       ✓ Parameter updates in healthy range")
        
        # Analyze individual param changes
        param_norms = jax.tree.leaves(param_change)
        if param_norms:
            max_change = max(param_norms)
            min_change = min(param_norms)
            print(f"       Param change range: [{min_change:.6f}, {max_change:.6f}]")
    
    def _select_best_grpo_intervention(self, candidates: List[Dict]) -> Dict:
        """Select best intervention based on GRPO advantages."""
        # Compute advantages with proper normalization
        rewards = jnp.array([c['reward'] for c in candidates])
        baseline = jnp.mean(rewards)
        reward_std = jnp.std(rewards)
        
        # Use normalized advantages for selection (consistent with update)
        if reward_std > 1e-8:
            advantages = (rewards - baseline) / reward_std
        else:
            advantages = rewards - baseline
        
        # Debug advantage calculation with validation
        if hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 5:
            print(f"\n    📊 GRPO Advantage Calculation:")
            print(f"       Rewards: {[f'{float(r):.3f}' for r in rewards]}")
            print(f"       Baseline (mean): {baseline:.3f}")
            print(f"       Reward std: {reward_std:.3f}")
            print(f"       Normalized advantages: {[f'{float(a):.3f}' for a in advantages]}")
            print(f"       Advantage mean: {jnp.mean(advantages):.3f} (should be ~0)")
            print(f"       Advantage std: {jnp.std(advantages):.3f} (should be ~1 if normalized)")
            print(f"       Variables chosen: {[c['variable'] for c in candidates]}")
            values_str = ', '.join([f"{c['value']:.3f}" for c in candidates])
            print(f"       Values chosen: [{values_str}]")
            
            # Validate reward components
            self._validate_reward_components(candidates[:3])
        
        # Select best
        best_idx = jnp.argmax(advantages)
        
        if hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 5:
            print(f"       Best candidate: idx={best_idx}, var={candidates[best_idx]['variable']}, "
                  f"val={candidates[best_idx]['value']:.3f}, advantage={advantages[best_idx]:.3f}")
        
        # Return best candidate with target_value included for tracking
        best_candidate = dict(candidates[best_idx])  # Make a copy
        if 'reward_components_raw' in best_candidate:
            best_candidate['target_value'] = best_candidate['reward_components_raw'].get('target_value', float('nan'))
        
        return best_candidate
    
    def _update_policy_with_grpo(self, candidates: List[Dict]):
        """Update policy parameters using GRPO on the candidate batch."""
        if self.current_phase != 'policy':
            logger.debug(f"Skipping policy update - current phase is {self.current_phase}")
            return  # Only update in policy phase
        
        logger.info(f"Updating policy with {len(candidates)} candidates")
        
        # Extract data for GRPO update
        rewards = jnp.array([c['reward'] for c in candidates])
        old_log_probs = jnp.array([c['log_prob'] for c in candidates])
        
        # Compute advantages with proper GRPO normalization
        # Standard GRPO: advantages = (rewards - mean) / (std + ε)
        baseline = jnp.mean(rewards)
        reward_std = jnp.std(rewards)
        advantages = (rewards - baseline) / (reward_std + 1e-8)
        
        logger.debug(f"GRPO update with {len(candidates)} candidates, "
                    f"mean reward: {baseline:.3f}, reward std: {reward_std:.3f}, "
                    f"normalized advantage std: {jnp.std(advantages):.3f}")
        
        # Debug GRPO update details
        if hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 5:
            print(f"\n    🎯 GRPO Update Details:")
            print(f"       Num candidates: {len(candidates)}")
            print(f"       Mean reward: {baseline:.3f}")
            print(f"       Reward std: {reward_std:.3f}")
            print(f"       Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
            print(f"       Normalized advantage mean: {jnp.mean(advantages):.3f} (should be ~0)")
            print(f"       Normalized advantage std: {jnp.std(advantages):.3f} (should be ~1)")
            print(f"       Advantage range: [{advantages.min():.3f}, {advantages.max():.3f}]")
        
        # Format batch for parent's GRPO update function
        # The parent expects: states, actions, rewards, old_log_probs, advantages
        states = jnp.stack([c.get('tensor_5ch', jnp.zeros((10, 5, 5))) for c in candidates])
        
        # In enhanced mode (use_grpo_rewards=True), actions should be list of dicts
        # In standard mode, it would be dict with 'variables' key
        # Need to convert variable names to indices
        if self.use_grpo_rewards:
            # Enhanced mode format - expects variable indices not names
            # Get mapper from first candidate if available
            if candidates and 'tensor_5ch' in candidates[0]:
                # Need to get the mapper to convert names to indices
                from .five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
                # Use the mapper stored in candidates or create new one
                mapper = candidates[0].get('mapper')
                if mapper is None:
                    # Fallback: use variable order from first candidate
                    variables = sorted(set([c['variable'] for c in candidates]))
                    var_to_idx = {var: i for i, var in enumerate(variables)}
                else:
                    var_to_idx = mapper.name_to_idx if hasattr(mapper, 'name_to_idx') else {}
                
                actions = [{'variable': var_to_idx.get(c['variable'], 0), 'value': c['value']} 
                          for c in candidates]
            else:
                # Fallback if no tensor available
                actions = [{'variable': 0, 'value': c['value']} for c in candidates]
        else:
            # Standard mode format - also needs indices
            variables = sorted(set([c['variable'] for c in candidates]))
            var_to_idx = {var: i for i, var in enumerate(variables)}
            actions = {'variables': [var_to_idx.get(c['variable'], 0) for c in candidates]}
        
        grpo_batch = {
            'states': states,  # Required: [batch_size, T, n_vars, 5]
            'actions': actions,  # Required: format depends on mode
            'rewards': rewards,  # Required: reward array
            'old_log_probs': old_log_probs,  # Required: old log probs
            'advantages': advantages,  # Optional but we provide it
            'target_idx': candidates[0].get('target_idx', 0) if candidates else 0
        }
        
        # Use parent's GRPO update directly
        old_params = self.policy_params
        
        # Ensure optimizer state exists
        if not hasattr(self, 'optimizer_state'):
            logger.warning("No optimizer_state found, initializing now")
            if hasattr(self, 'optimizer'):
                self.optimizer_state = self.optimizer.init(self.policy_params)
            else:
                logger.error("No optimizer found! Cannot update policy")
                return
        
        try:
            self.policy_params, self.optimizer_state, grpo_metrics = self.grpo_update(
                self.policy_params,
                self.optimizer_state,
                grpo_batch
            )
        except Exception as e:
            logger.error(f"GRPO update failed: {e}")
            return
        
        # Check if parameters actually changed
        param_change = jax.tree.map(
            lambda old, new: jnp.linalg.norm(new - old),
            old_params, self.policy_params
        )
        total_change = sum(jax.tree.leaves(param_change))
        
        logger.info(f"Policy updated - loss: {grpo_metrics.policy_loss:.4f}, "
                   f"grad norm: {grpo_metrics.grad_norm:.6f}, "
                   f"param change: {total_change:.6f}")
        
        # Detailed gradient flow diagnostics
        if hasattr(self, '_debug_episode_count') and self._debug_episode_count <= 5:
            self._analyze_gradient_flow(grpo_metrics, param_change, total_change)
    
    def _collect_single_intervention(self, buffer, current_posterior, target_var, 
                                    variables, scm, policy_params, key):
        """Collect a single intervention using frozen policy (for surrogate phase)."""
        # Convert buffer to tensor
        from .five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
        tensor_5ch, mapper, _ = buffer_to_five_channel_tensor_with_posteriors(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Get policy action
        key, policy_key = random.split(key)
        policy_output = self.policy_fn.apply(
            policy_params, policy_key, tensor_5ch, mapper.target_idx
        )
        
        # Sample intervention
        var_logits = policy_output['variable_logits']
        key, var_key = random.split(key)
        selected_var_idx = random.categorical(var_key, var_logits)
        selected_var = mapper.get_name(int(selected_var_idx))
        
        # Sample value
        value_params = policy_output['value_params']
        mean = value_params[selected_var_idx, 0]
        log_std = value_params[selected_var_idx, 1]
        std = jnp.exp(log_std)
        key, val_key = random.split(key)
        raw_intervention_value = mean + std * random.normal(val_key)
        
        # Soft tanh-based mapping to variable range (preserves gradients)
        var_ranges = scm.get('metadata', {}).get('variable_ranges', {})
        if selected_var in var_ranges:
            min_val, max_val = var_ranges[selected_var]
        else:
            # Default range if not specified
            min_val, max_val = -10.0, 10.0
        
        # Map to range using tanh (smooth, differentiable)
        scale_factor = 2.0  # Same as in hypothetical evaluation
        range_center = (max_val + min_val) / 2
        range_radius = (max_val - min_val) / 2
        intervention_value = range_center + range_radius * jnp.tanh(raw_intervention_value / scale_factor)
        
        # Create and apply intervention
        from ..interventions.handlers import create_perfect_intervention
        from ..environments.sampling import sample_with_intervention
        
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        key, sample_key = random.split(key)
        outcome_samples = sample_with_intervention(
            scm, intervention, n_samples=1, seed=int(sample_key[0])
        )
        outcome = outcome_samples[0] if outcome_samples else None
        
        return intervention, outcome
    
    def _train_surrogate_on_buffer(self, buffer: ExperienceBuffer, scm) -> float:
        """Train surrogate model on collected buffer using ground truth from SCM."""
        if not self.use_surrogate:
            logger.warning("Surrogate training skipped - use_surrogate is False")
            return 0.0
        
        logger.info(f"Training surrogate on buffer with {buffer.size()} samples")
        
        # Validate interventions in buffer before surrogate training
        interventions = buffer.get_interventions()
        if interventions and len(interventions) <= 5:  # Show first few for validation
            print(f"\n  🔬 Surrogate sees {len(interventions)} interventions:")
            for i, (intervention, outcome) in enumerate(interventions[:3]):
                targets = intervention.get('targets', frozenset())
                values = intervention.get('values', {})
                print(f"     Intervention {i+1}: Targets={targets}, Values={values}")
        
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var))
        variables = list(get_variables(scm))
        
        logger.info(f"Ground truth - Target: {target_var}, Parents: {true_parents}")
        
        # Create ground truth parent probabilities
        true_parent_probs = jnp.zeros(len(variables))
        for i, var in enumerate(variables):
            if var in true_parents and var != target_var:
                true_parent_probs = true_parent_probs.at[i].set(1.0)
            else:
                true_parent_probs = true_parent_probs.at[i].set(0.0)
        
        # Convert buffer to tensor
        from .three_channel_converter import buffer_to_three_channel_tensor
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Initialize optimizer if not present
        if not hasattr(self, 'surrogate_optimizer'):
            self.surrogate_optimizer = optax.adam(1e-3)
            self.surrogate_opt_state = self.surrogate_optimizer.init(self.surrogate_params)
        
        # Define supervised loss function
        def surrogate_loss_fn(params):
            # Forward pass
            self.rng_key, net_key = random.split(self.rng_key)
            predictions = self.surrogate_net.apply(
                params, net_key, tensor, mapper.target_idx, True  # is_training=True
            )
            pred_probs = predictions['parent_probabilities']
            
            # Binary cross-entropy loss against ground truth
            epsilon = 1e-10
            bce_loss = -jnp.mean(
                true_parent_probs * jnp.log(pred_probs + epsilon) +
                (1 - true_parent_probs) * jnp.log(1 - pred_probs + epsilon)
            )
            
            # Add L2 regularization on non-target predictions
            non_target_mask = jnp.ones_like(pred_probs)
            non_target_mask = non_target_mask.at[mapper.target_idx].set(0.0)
            l2_penalty = 0.01 * jnp.sum((pred_probs * non_target_mask) ** 2)
            
            return bce_loss + l2_penalty
        
        # Compute gradients and update
        old_params = self.surrogate_params
        loss, grads = jax.value_and_grad(surrogate_loss_fn)(self.surrogate_params)
        
        # Gradient clipping
        grad_norm = optax.global_norm(grads)
        max_grad_norm = 1.0
        grads = jax.lax.cond(
            grad_norm > max_grad_norm,
            lambda g: jax.tree.map(lambda x: x * max_grad_norm / grad_norm, g),
            lambda g: g,
            grads
        )
        
        # Update parameters
        updates, self.surrogate_opt_state = self.surrogate_optimizer.update(
            grads, self.surrogate_opt_state, self.surrogate_params
        )
        self.surrogate_params = optax.apply_updates(self.surrogate_params, updates)
        
        # Check parameter change
        param_change = jax.tree.map(
            lambda old, new: jnp.linalg.norm(new - old),
            old_params, self.surrogate_params
        )
        total_change = sum(jax.tree.leaves(param_change))
        
        # Compute F1 score for logging
        # Get predictions with updated params
        self.rng_key, eval_key = random.split(self.rng_key)
        eval_predictions = self.surrogate_net.apply(
            self.surrogate_params, eval_key, tensor, mapper.target_idx, False
        )
        eval_probs = eval_predictions['parent_probabilities']
        
        # Threshold predictions
        predicted_parents = eval_probs > 0.5
        true_parent_binary = true_parent_probs > 0.5
        
        # Compute F1
        tp = jnp.sum(predicted_parents & true_parent_binary)
        fp = jnp.sum(predicted_parents & ~true_parent_binary)
        fn = jnp.sum(~predicted_parents & true_parent_binary)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        logger.info(f"Surrogate updated - loss: {float(loss):.4f}, "
                   f"grad norm: {float(grad_norm):.6f}, "
                   f"param change: {float(total_change):.6f}, "
                   f"F1: {float(f1):.3f}")
        
        return float(loss)
    
    def _print_model_diagnostics(self, stage: str, episode: int):
        """Print diagnostic information during actual training episodes."""
        # Only print diagnostics every few episodes to avoid clutter
        if episode % 5 != 0 and stage == "START":
            return
            
        from ..data_structures.scm import get_variables, get_target, get_parents
        
        # Get SCM info
        variables = list(get_variables(self.current_scm))
        target_var = get_target(self.current_scm)
        true_parents = list(get_parents(self.current_scm, target_var))
        
        print(f"\n  === Episode {episode} {stage} ===")
        print(f"  SCM Target: {target_var}, True parents: {true_parents}")
        print(f"  Current training phase: {self.current_phase}")
        
        # Show which model is being trained
        if self.current_phase == 'policy':
            print(f"  📚 Policy is LEARNING (surrogate frozen)")
        else:
            print(f"  🔬 Surrogate is LEARNING (policy frozen)")
        
        print(f"  ===========================\n")
        sys.stdout.flush()
    
    def _freeze_models(self):
        """Freeze appropriate model parameters based on current phase."""
        if self.current_phase == 'policy':
            # Freeze surrogate for policy training
            if self.surrogate_params is not None:
                self.frozen_surrogate_params = self.surrogate_params.copy()
                logger.debug("Froze surrogate parameters for policy phase")
        else:  # surrogate phase
            # Freeze policy for surrogate training
            if self.policy_params is not None:
                self.frozen_policy_params = self.policy_params.copy()
                logger.debug("Froze policy parameters for surrogate phase")
    
    def _train_surrogate_on_trajectory(self, buffer: ExperienceBuffer) -> float:
        """Train surrogate model on collected trajectory."""
        if not self.use_surrogate:
            return 0.0
        
        # Convert buffer to training data for surrogate (surrogate uses 3 channels)
        from .three_channel_converter import buffer_to_three_channel_tensor
        target_var = get_target(self.current_scm)
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Get true parent labels
        true_parents = set(get_parents(self.current_scm, target_var))
        parent_labels = jnp.array([
            1.0 if mapper.get_name(i) in true_parents else 0.0
            for i in range(len(mapper.variables))
            if mapper.get_name(i) != target_var
        ])
        
        # Simple gradient step on surrogate
        # (In full implementation, would batch with replay data)
        loss = self._compute_surrogate_loss(
            self.surrogate_params,
            tensor,
            parent_labels,
            target_var,
            mapper.variables
        )
        
        # Update surrogate parameters
        # (Simplified - in practice would use proper optimizer)
        if self.surrogate_update_fn:
            self.surrogate_params, self.surrogate_opt_state, _ = \
                self.surrogate_update_fn(
                    self.surrogate_params,
                    self.surrogate_opt_state,
                    buffer,
                    target_var
                )
        
        return float(loss)
    
    def _compute_surrogate_loss(self, params, tensor, labels, target_var, variables):
        """Compute BCE loss for surrogate."""
        # Convert target var name to index
        target_idx = variables.index(target_var) if isinstance(target_var, str) else target_var
        
        # Get predictions
        predictions = self.surrogate_predict_fn(tensor, target_idx, variables)
        
        # Extract marginal probabilities
        if 'marginal_parent_probs' in predictions:
            probs = predictions['marginal_parent_probs']
            pred_array = jnp.array([
                probs.get(var, 0.0) for var in variables if var != target_var
            ])
            
            # Binary cross-entropy
            eps = 1e-8
            loss = -jnp.mean(
                labels * jnp.log(pred_array + eps) + 
                (1 - labels) * jnp.log(1 - pred_array + eps)
            )
            return loss
        
        return 0.0
    
    def _compute_structure_metrics(self, buffer: ExperienceBuffer) -> Dict[str, float]:
        """Compute structure learning metrics."""
        if not self.use_surrogate:
            return {}
        
        # Get predictions and true structure
        target_var = get_target(self.current_scm)
        true_parents = set(get_parents(self.current_scm, target_var))
        
        # Convert buffer to tensor
        from .three_channel_converter import buffer_to_three_channel_tensor
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Convert target var name to index
        target_idx = mapper.variables.index(target_var) if isinstance(target_var, str) else target_var
        
        # Get predictions
        predictions = self.surrogate_predict_fn(tensor, target_idx, mapper.variables)
        
        if 'marginal_parent_probs' in predictions:
            probs = predictions['marginal_parent_probs']
            
            # Compute F1 score (simplified - threshold at 0.5)
            predicted_parents = {
                var for var, prob in probs.items() 
                if prob > 0.5 and var != target_var
            }
            
            if true_parents:
                tp = len(predicted_parents & true_parents)
                fp = len(predicted_parents - true_parents)
                fn = len(true_parents - predicted_parents)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                f1 = 1.0 if len(predicted_parents) == 0 else 0.0
            
            return {
                'f1_score': f1,
                'n_predicted': len(predicted_parents),
                'n_true': len(true_parents)
            }
        
        return {}
    
    def _track_intervention_patterns(self, episode_idx: int, buffer: ExperienceBuffer, target_var: str):
        """
        Track intervention patterns to diagnose learning.
        Compares first 3 episodes vs last 3 episodes.
        """
        if not hasattr(self, 'intervention_history'):
            self.intervention_history = []
        
        # Extract intervention data from buffer
        interventions = buffer.get_interventions()
        episode_data = {
            'episode': episode_idx,
            'target': target_var,
            'interventions': [],
            'values': []
        }
        
        for intervention, _ in interventions:
            targets = intervention.get('targets', frozenset())
            values = intervention.get('values', {})
            for var in targets:
                episode_data['interventions'].append(var)
                episode_data['values'].append(values.get(var, 0.0))
        
        self.intervention_history.append(episode_data)
        
        # Analyze patterns every 10 episodes
        if episode_idx > 0 and episode_idx % 10 == 0:
            self._analyze_intervention_trends()
    
    def _analyze_intervention_trends(self):
        """Analyze intervention patterns comparing early vs late episodes."""
        if len(self.intervention_history) < 6:
            return
        
        # Get first 3 and last 3 episodes
        first_3 = self.intervention_history[:3]
        last_3 = self.intervention_history[-3:]
        
        # Analyze variable selection
        first_vars = []
        last_vars = []
        first_values = []
        last_values = []
        
        for ep in first_3:
            first_vars.extend(ep['interventions'])
            first_values.extend(ep['values'])
        
        for ep in last_3:
            last_vars.extend(ep['interventions'])
            last_values.extend(ep['values'])
        
        # Count variable frequencies
        from collections import Counter
        first_var_counts = Counter(first_vars)
        last_var_counts = Counter(last_vars)
        
        # Compute statistics
        first_mean_value = np.mean(np.abs(first_values)) if first_values else 0
        last_mean_value = np.mean(np.abs(last_values)) if last_values else 0
        
        print("\n" + "="*60)
        print("📊 INTERVENTION PATTERN ANALYSIS")
        print("="*60)
        
        print("\n🔍 Variable Selection Distribution:")
        print("   First 3 episodes:")
        for var, count in first_var_counts.most_common():
            pct = 100 * count / len(first_vars) if first_vars else 0
            print(f"      {var}: {count}/{len(first_vars)} ({pct:.1f}%)")
        
        print("   Last 3 episodes:")
        for var, count in last_var_counts.most_common():
            pct = 100 * count / len(last_vars) if last_vars else 0
            print(f"      {var}: {count}/{len(last_vars)} ({pct:.1f}%)")
        
        print(f"\n📈 Intervention Value Statistics:")
        print(f"   First 3 episodes - Mean |value|: {first_mean_value:.3f}")
        print(f"   Last 3 episodes  - Mean |value|: {last_mean_value:.3f}")
        print(f"   Change: {last_mean_value - first_mean_value:+.3f}")
        
        # Check if learning to select parents
        target = self.intervention_history[-1]['target']
        true_parents = self.current_scm.get_parent_names(target) if hasattr(self.current_scm, 'get_parent_names') else []
        
        if true_parents:
            first_parent_pct = sum(1 for v in first_vars if v in true_parents) / len(first_vars) * 100 if first_vars else 0
            last_parent_pct = sum(1 for v in last_vars if v in true_parents) / len(last_vars) * 100 if last_vars else 0
            
            print(f"\n🎯 Parent Selection Rate:")
            print(f"   True parents: {true_parents}")
            print(f"   First 3 episodes: {first_parent_pct:.1f}%")
            print(f"   Last 3 episodes:  {last_parent_pct:.1f}%")
            print(f"   Improvement: {last_parent_pct - first_parent_pct:+.1f}%")
            
            if last_parent_pct > first_parent_pct + 20:
                print("   ✅ Policy is learning to select true parents!")
            elif last_parent_pct > first_parent_pct:
                print("   📈 Some improvement in parent selection")
            else:
                print("   ⚠️ No clear improvement in parent selection")
        
        if last_mean_value > first_mean_value + 0.5:
            print("\n✅ Intervention values becoming more extreme (good!)")
        elif last_mean_value > first_mean_value:
            print("\n📈 Slight increase in intervention magnitude")
        else:
            print("\n⚠️ Intervention values not becoming more extreme")
        
        print("="*60 + "\n")
    
    def _update_performance_tracking(self, metrics: Dict[str, Any]):
        """Update performance tracking for current phase."""
        if self.current_phase == 'policy':
            self.policy_performance.update(
                loss=metrics.get('loss', 0.0),
                improvement=metrics.get('target_improvement', 0.0)
            )
        else:
            self.surrogate_performance.update(
                loss=metrics.get('surrogate_loss', 0.0),
                f1=metrics.get('structure_metrics', {}).get('f1_score', 0.0)
            )
    
    def _create_joint_metrics(self, episode_metrics: Dict[str, Any]) -> JointTrainingMetrics:
        """Create joint training metrics."""
        return JointTrainingMetrics(
            episode=self.episode_count,
            phase=self.current_phase,
            policy_loss=episode_metrics.get('loss', 0.0),
            surrogate_loss=episode_metrics.get('surrogate_loss', 0.0),
            mean_reward=episode_metrics.get('mean_reward', 0.0),
            f1_score=episode_metrics.get('structure_metrics', {}).get('f1_score', 0.0),
            target_improvement=episode_metrics.get('target_improvement', 0.0),
            information_gain=episode_metrics.get('information_gain', 0.0),
            episodes_in_phase=self.phase_episode_count,
            total_episodes=self.episode_count
        )
    
    def _store_trajectory(self, trajectory: ExperienceBuffer, metrics: Dict[str, Any]):
        """Store trajectory in replay buffer."""
        performance_delta = metrics.get('target_improvement', 0.0)
        self.replay_buffer.add_trajectory(
            trajectory,
            self.current_scm_metadata,
            performance_delta
        )
    
    def _log_progress(self, episode: int, episode_metrics: List[Dict]):
        """Log training progress."""
        recent = episode_metrics[-min(10, len(episode_metrics)):]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode} | Phase: {self.current_phase}")
        logger.info(f"  Episodes in phase: {self.phase_episode_count}/{self.episodes_per_phase}")
        logger.info(f"  Current SCM: {self.current_scm_metadata.get('name', 'unknown')}")
        logger.info(f"  Episodes on SCM: {self.episodes_on_current_scm}")
        
        if self.current_phase == 'policy':
            mean_reward = np.mean([m.get('mean_reward', 0) for m in recent])
            logger.info(f"  Mean reward: {mean_reward:.4f}")
            if self.policy_performance.recent_losses:
                logger.info(f"  Policy loss: {self.policy_performance.recent_losses[-1]:.4f}")
        else:
            if self.surrogate_performance.recent_f1_scores:
                logger.info(f"  F1 score: {self.surrogate_performance.recent_f1_scores[-1]:.4f}")
            if self.surrogate_performance.recent_losses:
                logger.info(f"  Surrogate loss: {self.surrogate_performance.recent_losses[-1]:.4f}")
        
        logger.info(f"  Replay buffer size: {len(self.replay_buffer)}")
        logger.info(f"  Policy plateaued: {self.policy_performance.is_plateaued}")
        logger.info(f"  Surrogate plateaued: {self.surrogate_performance.is_plateaued}")
        logger.info(f"{'='*60}\n")
    
    def _validate_training(self, episode: int):
        """Validate training correctness."""
        checks = {
            'both_models_have_params': (
                self.policy_params is not None and 
                (not self.use_surrogate or self.surrogate_params is not None)
            ),
            'phase_switching_works': self.phase_episode_count <= self.episodes_per_phase,
            'scm_rotation_works': self.episodes_on_current_scm <= self.rotation_episodes,
            'metrics_recorded': len(self.joint_metrics_history) > 0,
            'replay_buffer_working': not self.use_replay or len(self.replay_buffer) >= 0
        }
        
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            logger.warning(f"Validation failed at episode {episode}: {failed}")
        else:
            logger.debug(f"Validation passed at episode {episode}")
    
    def _save_joint_checkpoint(self, episode: int):
        """Save both models as standard checkpoints that can be loaded independently."""
        checkpoint_dir = Path(self.checkpoint_dir) / f'joint_ep{episode}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy in standard format that create_grpo_acquisition can load
        policy_path = checkpoint_dir / 'policy.pkl'
        save_checkpoint(
            path=policy_path,
            params=self.policy_params,
            architecture={
                'hidden_dim': self.hidden_dim,
                'num_layers': 4,
                'num_heads': 8,
                'key_size': 32,
                'dropout': 0.1,
                'architecture_type': 'permutation_invariant'
            },
            model_type='policy',
            model_subtype='grpo',
            training_config={
                'learning_rate': self.learning_rate,
                'joint_trained': True,
                'episode': episode,
                'phase': self.current_phase
            },
            metadata={
                'trainer': 'JointACBOTrainer',
                'total_episodes': episode,
                'phase_at_save': self.current_phase
            },
            metrics={
                'episodes_trained': self.policy_performance.episodes_completed,
                'is_plateaued': self.policy_performance.is_plateaued
            }
        )
        
        # Save surrogate in standard format that create_bc_surrogate can load
        surrogate_path = checkpoint_dir / 'surrogate.pkl'
        save_checkpoint(
            path=surrogate_path,
            params=self.surrogate_params,
            architecture={
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'key_size': 32,
                'dropout': 0.1,
                'encoder_type': 'node_feature'
            },
            model_type='surrogate',
            model_subtype='continuous_parent_set',
            training_config={
                'learning_rate': 1e-3,
                'joint_trained': True,
                'episode': episode,
                'phase': self.current_phase
            },
            metadata={
                'trainer': 'JointACBOTrainer',
                'total_episodes': episode,
                'phase_at_save': self.current_phase
            },
            metrics={
                'episodes_trained': self.surrogate_performance.episodes_completed,
                'is_plateaued': self.surrogate_performance.is_plateaued,
                'recent_f1': np.mean(list(self.surrogate_performance.recent_f1_scores)) 
                            if self.surrogate_performance.recent_f1_scores else 0.0
            }
        )
        
        logger.info(f"Saved standard checkpoints to {checkpoint_dir}/")
    
    def _check_convergence(self, episode_metrics: List[Dict]) -> bool:
        """Check if training has converged."""
        if len(episode_metrics) < 50:
            return False
        
        # Check if both models have plateaued for a while
        if (self.policy_performance.is_plateaued and 
            self.surrogate_performance.is_plateaued and
            self.policy_performance.episodes_completed > 20 and
            self.surrogate_performance.episodes_completed > 20):
            
            # Check if we've achieved good performance
            recent_f1 = np.mean(list(self.surrogate_performance.recent_f1_scores)) \
                       if self.surrogate_performance.recent_f1_scores else 0
            if recent_f1 > 0.9:
                return True
        
        return False
    
    def _prepare_results(self, episode_metrics: List[Dict], training_time: float) -> Dict[str, Any]:
        """Prepare final training results."""
        results = {
            'training_time': training_time,
            'total_episodes': len(episode_metrics),
            'final_phase': self.current_phase,
            'policy_params': self.policy_params,
            'surrogate_params': self.surrogate_params,
            'episode_metrics': episode_metrics,
            'joint_metrics': self.joint_metrics_history,
            'replay_buffer_size': len(self.replay_buffer),
            'policy_performance': {
                'episodes_trained': self.policy_performance.episodes_completed,
                'final_plateaued': self.policy_performance.is_plateaued
            },
            'surrogate_performance': {
                'episodes_trained': self.surrogate_performance.episodes_completed,
                'final_plateaued': self.surrogate_performance.is_plateaued,
                'final_f1': np.mean(list(self.surrogate_performance.recent_f1_scores))
                          if self.surrogate_performance.recent_f1_scores else 0
            }
        }
        
        return results