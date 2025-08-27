"""
Simplified Joint ACBO Trainer with clean phase alternation.

This trainer provides a clean, simple implementation of joint training that:
1. Uses UnifiedGRPOTrainer's proven GRPO implementation (no custom reimplementation)
2. Alternates between policy and surrogate training phases
3. No complex parameter "freezing" - JAX handles this naturally
4. Clean delegation instead of overriding everything
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from collections import deque

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from omegaconf import DictConfig

# Import parent class - use its proven GRPO implementation
from .unified_grpo_trainer import UnifiedGRPOTrainer

# SCM and data structures
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.scm import get_variables, get_target, get_parents
from ..data_structures.sample import get_values
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..environments.sampling import sample_with_intervention

# Converters
from .three_channel_converter import buffer_to_three_channel_tensor
from .five_channel_converter import (
    buffer_to_five_channel_tensor_with_posteriors,
    create_uniform_posterior
)

# Utilities
from ..utils.checkpoint_utils import save_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class JointTrainingMetrics:
    """Simple metrics for joint training."""
    episode: int
    phase: str  # "policy" or "surrogate"
    mean_reward: float = 0.0
    surrogate_loss: float = 0.0
    target_improvement: float = 0.0
    f1_score: float = 0.0


class JointACBOTrainer(UnifiedGRPOTrainer):
    """
    Simplified joint trainer using proven UnifiedGRPOTrainer implementation.
    
    Key principles:
    1. Delegate GRPO training to parent's working implementation
    2. Simple phase alternation without complex parameter freezing
    3. Clean separation of concerns
    """
    
    def __init__(self, config: Optional[Union[DictConfig, Dict[str, Any]]] = None, **kwargs):
        """Initialize joint trainer with minimal additional complexity."""
        logger.info("Initializing simplified JointACBOTrainer...")
        
        # Initialize parent (UnifiedGRPOTrainer) - this gives us working GRPO
        super().__init__(config=config, **kwargs)
        
        # Joint training configuration
        joint_config = config.get('joint_training', {}) if config else {}
        
        # Phase control - simple alternation
        self.episodes_per_phase = config.get('episodes_per_phase', 5) if config else 5
        self.current_phase = joint_config.get('initial_phase', 'policy')
        self.phase_episode_count = 0
        
        # SCM management
        self.current_scm = None
        self.episodes_on_current_scm = 0
        self.rotation_episodes = joint_config.get('rotation_episodes', 20)
        
        # New rotation configuration
        self.rotate_after_episode = config.get('rotate_after_episode', True) if config else True
        self.convergence_patience = config.get('convergence_patience', 3) if config else 3
        self.convergence_threshold = config.get('convergence_threshold', 0.9) if config else 0.9
        self.enable_early_rotation = config.get('enable_early_rotation', True) if config else True
        
        # Convergence tracking
        self.recent_interventions = []  # Track recent (var, quantile) selections
        self.consecutive_same_count = 0
        self.last_intervention = None
        self.convergence_detected = False
        
        # Simple performance tracking
        self.policy_episodes = 0
        self.surrogate_episodes = 0
        self.joint_metrics_history = []
        
        # Track target progression across all episodes
        self.target_progression_history = []
        
        # Initialize surrogate optimizer here (like AVICI) to persist state
        self._initialize_surrogate_optimizer()
        
        # DEBUG: Print reward configuration
        print(f"\n{'='*70}")
        print(f"REWARD CONFIGURATION")
        print(f"{'='*70}")
        if config and 'reward_weights' in config:
            weights = config['reward_weights']
            print(f"Composite reward weights:")
            print(f"  - Target weight: {weights.get('target', 1.0)}")
            print(f"  - Parent bonus: {weights.get('parent', 0.0)}")
            print(f"  - Info gain: {weights.get('info_gain', 0.0)}")
        else:
            print(f"Using standard target-based reward")
        
        if config and 'grpo_config' in config:
            grpo = config['grpo_config']
            print(f"\nGRPO Configuration:")
            print(f"  - Entropy coefficient: {grpo.get('entropy_coefficient', 0.01)}")
            print(f"  - Group size: {grpo.get('group_size', 10)}")
        print(f"{'='*70}\n")
        
        logger.info(f"Initialized JointACBOTrainer:")
        logger.info(f"  Episodes per phase: {self.episodes_per_phase}")
        logger.info(f"  Initial phase: {self.current_phase}")
    
    def _initialize_surrogate_optimizer(self):
        """Initialize surrogate optimizer with proper state management."""
        if self.use_surrogate:
            logger.info("Initializing surrogate optimizer (AdamW + gradient clipping)")
            # Use AVICI-style optimizer: AdamW with weight decay and cosine schedule
            schedule = optax.cosine_decay_schedule(
                init_value=1e-3,
                decay_steps=1000,  # Will be updated based on episodes
                alpha=0.1
            )
            self.surrogate_optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),  # Gradient clipping
                optax.adamw(learning_rate=schedule, weight_decay=0.01)
            )
            # Initialize state when surrogate params are available
            self.surrogate_opt_state = None  # Will be set when surrogate is initialized
            logger.info("✅ Surrogate optimizer configured")
        else:
            self.surrogate_optimizer = None
            self.surrogate_opt_state = None
            logger.info("Surrogate disabled - no optimizer needed")
        
    def train(self, scms: Union[List[Any], Dict[str, Any], Callable[[], Any]], 
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Joint training with phase alternation.
        
        Simple logic:
        - Policy phase: Use parent's proven GRPO implementation
        - Surrogate phase: Generate single intervention + train surrogate with hard labels
        """
        logger.info("\n" + "="*70)
        logger.info("Starting Simplified Joint ACBO Training")
        logger.info(f"  Max episodes: {self.max_episodes}")
        logger.info(f"  Episodes per phase: {self.episodes_per_phase}")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Prepare SCMs using parent's logic
        scm_rotation = self._prepare_scms(scms)
        logger.info(f"Training with {len(scm_rotation)} SCMs")
        
        current_scm_idx = 0
        episode_metrics = []
        
        # Determine starting episode (for checkpoint continuation)
        start_episode = getattr(self, 'start_episode', 0)
        end_episode = start_episode + self.max_episodes
        logger.info(f"Training from episode {start_episode} to {end_episode}")
        
        # Main training loop
        for episode in range(start_episode, end_episode):
            self.episode_count = episode
            
            # Check time limit if specified
            if self.config and self.config.get('wall_clock_timeout_minutes'):
                elapsed_minutes = (time.time() - start_time) / 60.0
                if elapsed_minutes >= self.config['wall_clock_timeout_minutes']:
                    logger.info(f"\nTime limit reached ({self.config['wall_clock_timeout_minutes']} minutes), stopping training")
                    logger.info(f"Completed {episode} episodes in {elapsed_minutes:.1f} minutes")
                    break
            
            # Check phase switching
            if self._should_switch_phase():
                self._switch_phase()
            
            # Check SCM rotation
            if self._should_rotate_scm():
                current_scm_idx = (current_scm_idx + 1) % len(scm_rotation)
                self.episodes_on_current_scm = 0
                self._reset_convergence_tracking()  # Reset tracking for new SCM
            
            # Get current SCM
            scm_name, scm = scm_rotation[current_scm_idx]
            self.current_scm = scm
            
            # Run episode based on current phase
            self.rng_key, episode_key = random.split(self.rng_key)
            
            if self.current_phase == 'policy':
                # Policy phase: Multiple interventions per episode with GRPO learning
                metrics = self._run_policy_episode_with_interventions(episode, scm, scm_name, episode_key)
                metrics['training_phase'] = 'policy'
                self.policy_episodes += 1
            else:
                # Surrogate training phase
                metrics = self._run_surrogate_episode(episode, scm, scm_name, episode_key)
                metrics['training_phase'] = 'surrogate'
                self.surrogate_episodes += 1
            
            episode_metrics.append(metrics)
            self.joint_metrics_history.append(self._create_joint_metrics(metrics))
            
            # Track target progression across episodes
            if 'target_values' in metrics and metrics['target_values']:
                episode_targets = {
                    'episode': episode,
                    'target_values': metrics['target_values'],
                    'best_target': metrics['best_target'],
                    'improvement': metrics['target_improvement']
                }
                self.target_progression_history.append(episode_targets)
            
            self.phase_episode_count += 1
            self.episodes_on_current_scm += 1
            
            # Logging
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Phase={self.current_phase}, "
                          f"SCM={scm_name}, Reward={metrics.get('mean_reward', 0):.3f}")
        
        # Prepare results
        training_time = time.time() - start_time
        results = {
            'training_time': training_time,
            'final_metrics': episode_metrics[-1] if episode_metrics else {},
            'all_metrics': episode_metrics,
            'policy_params': self.policy_params,
            'surrogate_params': self.surrogate_params,
            'episode_metrics': episode_metrics,  # For compatibility
            'joint_metrics': self.joint_metrics_history,
            'policy_episodes': self.policy_episodes,
            'surrogate_episodes': self.surrogate_episodes
        }
        
        logger.info(f"\nJoint training completed in {training_time:.2f}s")
        logger.info(f"Policy episodes: {self.policy_episodes}, Surrogate episodes: {self.surrogate_episodes}")
        
        # CROSS-EPISODE TARGET PROGRESSION ANALYSIS
        if self.target_progression_history:
            print(f"\n{'='*70}")
            print(f"🎯 TARGET PROGRESSION ACROSS ALL EPISODES")
            print(f"{'='*70}")
            
            all_episode_bests = [ep['best_target'] for ep in self.target_progression_history]
            all_episode_improvements = [ep['improvement'] for ep in self.target_progression_history]
            
            print(f"\nBest target per episode: {[f'{v:.3f}' for v in all_episode_bests]}")
            print(f"Within-episode improvements: {[f'{v:+.3f}' for v in all_episode_improvements]}")
            
            if len(all_episode_bests) >= 2:
                overall_trend = all_episode_bests[-1] - all_episode_bests[0]
                print(f"\nOverall progression: {all_episode_bests[0]:.3f} → {all_episode_bests[-1]:.3f} ({overall_trend:+.3f})")
                
                if overall_trend < -0.1:
                    print(f"✅ LEARNING ACROSS EPISODES! Target values improving over time")
                elif overall_trend > 0.1:
                    print(f"⚠️ Target values getting worse across episodes")
                else:
                    print(f"➖ No clear learning trend across episodes")
                
                # Best single intervention across all episodes
                global_best = min(all_episode_bests)
                best_episode = all_episode_bests.index(global_best)
                print(f"\nBest single target: {global_best:.3f} (achieved in episode {best_episode})")
        
        return results
    
    def _should_switch_phase(self) -> bool:
        """Simple phase switching after fixed episodes."""
        return self.phase_episode_count >= self.episodes_per_phase
    
    def _switch_phase(self):
        """Switch training phase."""
        old_phase = self.current_phase
        self.current_phase = 'surrogate' if self.current_phase == 'policy' else 'policy'
        self.phase_episode_count = 0
        
        logger.info(f"Phase switched: {old_phase} -> {self.current_phase}")
    
    def _check_convergence(self, var_idx: int, quantile_idx: int, probability: float) -> bool:
        """
        Check if we've converged on a single intervention.
        
        Returns True if the same (var, quantile) has been selected
        consecutively with high probability.
        """
        current_intervention = (var_idx, quantile_idx)
        
        print(f"    🔄 Convergence check: current={current_intervention}, last={self.last_intervention}")
        print(f"    📊 Probability={probability:.3f}, threshold={self.convergence_threshold:.3f}")
        print(f"    🔢 Consecutive count={self.consecutive_same_count}, patience={self.convergence_patience}")
        
        # Check if this is a high-confidence selection
        if probability >= self.convergence_threshold:
            if self.last_intervention == current_intervention:
                self.consecutive_same_count += 1
                print(f"    ✅ Same intervention! Count now: {self.consecutive_same_count}")
                
                if self.consecutive_same_count >= self.convergence_patience:
                    print(f"  🎯 CONVERGENCE: Intervention {current_intervention} selected "
                          f"{self.consecutive_same_count} times with >{self.convergence_threshold:.1%} probability")
                    return True
            else:
                # Different intervention, reset counter
                print(f"    🔀 Different intervention, resetting count from {self.consecutive_same_count} to 1")
                self.consecutive_same_count = 1
                self.last_intervention = current_intervention
        else:
            # Low probability, don't count
            print(f"    ⚠️ Probability {probability:.3f} below threshold {self.convergence_threshold:.3f}, resetting")
            self.consecutive_same_count = 0
            self.last_intervention = None
        
        return False
    
    def _reset_convergence_tracking(self):
        """Reset convergence tracking for new SCM."""
        self.recent_interventions = []
        self.consecutive_same_count = 0
        self.last_intervention = None
        self.convergence_detected = False
    
    def _should_rotate_scm(self) -> bool:
        """
        Check if we should rotate to a new SCM.
        
        Rotates if:
        1. Episode-based rotation is enabled and we've finished an episode
        2. Early rotation is enabled and convergence was detected
        3. Falls back to old behavior if rotate_after_episode is False
        """
        # Check for convergence-based early rotation
        if self.enable_early_rotation and self.convergence_detected:
            print(f"  🔄 EARLY ROTATION: Convergence detected!")
            self.convergence_detected = False  # Reset flag
            return True
        
        # Check for episode-based rotation
        if self.rotate_after_episode:
            # Rotate after every episode (when we've done interventions)
            should_rotate = self.episodes_on_current_scm >= 1
            if should_rotate:
                print(f"  🔄 EPISODE ROTATION: Completed episode on current SCM")
            return should_rotate
        
        # Fall back to old behavior (rotate after fixed number of episodes)
        return self.episodes_on_current_scm >= self.rotation_episodes
    
    def _run_surrogate_episode(self, episode_idx: int, scm, scm_name: str, key) -> Dict[str, Any]:
        """
        Surrogate training episode with AVICI-style data volume and debugging.
        
        Enhanced logic:
        1. Generate many observations (100) + interventions (20) like AVICI
        2. Track prediction progression throughout episode
        3. Train surrogate with hard labels
        """
        print(f"\\n{'='*70}")
        print(f"🧠 SURROGATE EPISODE {episode_idx} - {scm_name}")
        print(f"{'='*70}")
        
        # DEBUG: Check optimizer state at episode start
        if hasattr(self, 'surrogate_opt_state') and self.surrogate_opt_state is not None:
            opt_state_norm = jnp.sqrt(sum(jnp.sum(jnp.array(s)**2) for s in jax.tree.leaves(self.surrogate_opt_state) if hasattr(s, 'shape')))
            print(f"🔧 EPISODE START - Optimizer state norm: {opt_state_norm:.6f}")
        else:
            print(f"🔧 EPISODE START - Optimizer state: None")
        
        # Check surrogate params at episode start  
        if hasattr(self, 'surrogate_params') and self.surrogate_params is not None:
            param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(self.surrogate_params)))
            print(f"🎯 EPISODE START - Surrogate params norm: {param_norm:.6f}")
        else:
            print(f"🎯 EPISODE START - Surrogate params: None")
        
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        true_parents = set(get_parents(scm, target_var))
        
        print(f"📋 SCM Info: target={target_var}, true_parents={true_parents}")
        
        # Initialize buffer with observations (INCREASED VOLUME)
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        
        # Add observations
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        print(f"📊 Added {len(obs_samples)} observations to buffer")
        
        # Get initial predictions BEFORE any interventions
        initial_predictions = self._get_surrogate_predictions(buffer, target_var, variables)
        print(f"🔮 INITIAL predictions: {initial_predictions}")
        
        # Generate MULTIPLE interventions (like AVICI: 200 interventions)
        intervention_results = []
        
        for intervention_idx in range(self.max_interventions):
            print(f"\\n--- Intervention {intervention_idx+1}/{self.max_interventions} ---")
            
            # Generate intervention using policy
            key, intervention_key = random.split(key)
            intervention, outcome = self._generate_single_intervention(
                buffer, target_var, variables, intervention_key
            )
            
            # Add intervention to buffer
            buffer.add_intervention(intervention, outcome)
            
            # Get predictions AFTER this intervention
            post_predictions = self._get_surrogate_predictions(buffer, target_var, variables)
            print(f"  After intervention: {post_predictions}")
            
            intervention_results.append({
                'intervention': intervention,
                'outcome': outcome,
                'predictions': post_predictions
            })
        
        print(f"\\n📈 PREDICTION PROGRESSION:")
        print(f"  Initial: {initial_predictions}")
        print(f"  Final: {intervention_results[-1]['predictions'] if intervention_results else 'None'}")
        
        # Train surrogate with hard labels on full buffer (LOTS of data now)
        print(f"\\n🎯 Training surrogate on buffer with {buffer.size()} samples")
        surrogate_loss = self._train_surrogate_with_hard_labels(buffer, scm)
        
        # Get FINAL predictions after training
        final_predictions = self._get_surrogate_predictions(buffer, target_var, variables)
        print(f"🏁 FINAL predictions after training: {final_predictions}")
        
        # DEBUG: Check optimizer state at episode end
        print(f"\\n🔧 EPISODE END DEBUG:")
        if hasattr(self, 'surrogate_opt_state') and self.surrogate_opt_state is not None:
            opt_state_norm = jnp.sqrt(sum(jnp.sum(jnp.array(s)**2) for s in jax.tree.leaves(self.surrogate_opt_state) if hasattr(s, 'shape')))
            print(f"  Optimizer state norm: {opt_state_norm:.6f}")
        else:
            print(f"  Optimizer state: None")
        
        # Check surrogate params at episode end  
        if hasattr(self, 'surrogate_params') and self.surrogate_params is not None:
            param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(self.surrogate_params)))
            print(f"  Surrogate params norm: {param_norm:.6f}")
        else:
            print(f"  Surrogate params: None")
        
        # Compute metrics
        target_value = get_values(intervention_results[-1]['outcome']).get(target_var, 0.0) if intervention_results else 0.0
        structure_metrics = self._compute_simple_structure_metrics(scm)
        
        return {
            'episode': episode_idx,
            'mean_reward': 0.0,  # Not relevant for surrogate phase
            'surrogate_loss': surrogate_loss,
            'target_improvement': 0.0,  # Could compute if needed
            'structure_metrics': structure_metrics,
            'n_variables': len(variables),
            'scm_type': scm_name,
            'prediction_progression': {
                'initial': initial_predictions,
                'final': final_predictions,
                'num_interventions': len(intervention_results)
            }
        }
    
    def _get_surrogate_predictions(self, buffer, target_var, variables):
        """Get current surrogate predictions for debugging."""
        if not self.use_surrogate:
            return {}
        
        try:
            tensor, mapper = buffer_to_three_channel_tensor(
                buffer, target_var, max_history_size=100, standardize=True
            )
            
            self.rng_key, pred_key = random.split(self.rng_key)
            predictions = self.surrogate_net.apply(
                self.surrogate_params, pred_key, tensor, mapper.target_idx, variables
            )
            
            if 'parent_probabilities' in predictions:
                probs = predictions['parent_probabilities']
                return {
                    var: float(probs[i]) for i, var in enumerate(mapper.variables) 
                    if var != target_var
                }
            else:
                return {}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_single_intervention(self, buffer, target_var, variables, key):
        """Generate single intervention using policy (no gradients)."""
        # Convert buffer to tensor
        tensor, mapper, _ = buffer_to_five_channel_tensor_with_posteriors(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        # Get policy action (no gradients computed here)
        key, policy_key = random.split(key)
        policy_output = self.policy_fn.apply(
            self.policy_params, policy_key, tensor, mapper.target_idx
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
        intervention_value = mean + std * random.normal(val_key)
        
        # Create and apply intervention
        intervention = create_perfect_intervention(
            targets=frozenset([selected_var]),
            values={selected_var: float(intervention_value)}
        )
        
        key, sample_key = random.split(key)
        outcome_samples = sample_with_intervention(
            self.current_scm, intervention, n_samples=1, seed=int(sample_key[0])
        )
        outcome = outcome_samples[0] if outcome_samples else None
        
        return intervention, outcome
    
    def _train_surrogate_with_hard_labels(self, buffer, scm) -> float:
        """Train surrogate with hard labels from ground truth SCM."""
        if not self.use_surrogate:
            return 0.0
        
        print(f"\n{'='*60}")
        print(f"🧠 SURROGATE TRAINING DEBUG")
        print(f"{'='*60}")
        
        target_var = get_target(scm)
        true_parents = set(get_parents(scm, target_var))
        variables = list(get_variables(scm))
        
        print(f"📋 INPUT VALIDATION:")
        print(f"  Buffer size: {buffer.size()}")
        print(f"  Target var: {target_var}")
        print(f"  True parents: {true_parents}")
        print(f"  Variables: {variables}")
        
        # Convert buffer to tensor
        tensor, mapper = buffer_to_three_channel_tensor(
            buffer, target_var, max_history_size=100, standardize=True
        )
        
        print(f"\n📊 TENSOR CONVERSION:")
        print(f"  Tensor shape: {tensor.shape}")
        print(f"  Mapper variables: {mapper.variables}")
        print(f"  Mapper target_idx: {mapper.target_idx}")
        print(f"  Mapper name_to_idx: {mapper.name_to_idx}")
        
        # Create hard labels (0 or 1)
        true_labels = jnp.array([
            1.0 if mapper.get_name(i) in true_parents and mapper.get_name(i) != target_var else 0.0
            for i in range(len(mapper.variables))
        ])
        
        print(f"\n🏷️  LABEL CREATION:")
        print(f"  True labels shape: {true_labels.shape}")
        print(f"  True labels: {true_labels}")
        print(f"  Label breakdown:")
        for i in range(len(mapper.variables)):
            var_name = mapper.get_name(i)
            label = true_labels[i]
            is_parent = var_name in true_parents
            is_target = var_name == target_var
            print(f"    {var_name}: label={label:.1f}, is_parent={is_parent}, is_target={is_target}")
        
        # First, do a test forward pass for debugging (no gradients)
        print(f"\\n🔮 TEST FORWARD PASS (no gradients):")
        self.rng_key, test_key = random.split(self.rng_key)
        test_predictions = self.surrogate_net.apply(
            self.surrogate_params, test_key, tensor, mapper.target_idx, variables
        )
        
        print(f"  Raw predictions keys: {list(test_predictions.keys())}")
        
        # Extract probabilities for debugging
        if 'parent_probabilities' in test_predictions:
            test_pred_probs = test_predictions['parent_probabilities']
            print(f"  Using 'parent_probabilities' format")
            print(f"  Parent probabilities shape: {test_pred_probs.shape}")
            print(f"  Parent probabilities: {test_pred_probs}")
        else:
            # Fallback - extract from marginal_parent_probs
            marginal_probs = test_predictions.get('marginal_parent_probs', {})
            print(f"  Using 'marginal_parent_probs' format")
            print(f"  Marginal probs keys: {list(marginal_probs.keys()) if marginal_probs else 'None'}")
            test_pred_probs = jnp.array([
                marginal_probs.get(var, 0.0) for var in variables if var != target_var
            ])
            print(f"  Extracted pred_probs shape: {test_pred_probs.shape}")
            print(f"  Extracted pred_probs: {test_pred_probs}")
        
        print(f"\\n📊 LOSS COMPUTATION PREVIEW:")
        print(f"  pred_probs range: [{jnp.min(test_pred_probs):.4f}, {jnp.max(test_pred_probs):.4f}]")
        print(f"  true_labels: {true_labels}")
        print(f"  pred_probs: {test_pred_probs}")
        
        # Verify dimensions match
        if test_pred_probs.shape != true_labels.shape:
            print(f"  ❌ DIMENSION MISMATCH: pred_probs {test_pred_probs.shape} vs true_labels {true_labels.shape}")
        else:
            print(f"  ✅ Dimensions match: {test_pred_probs.shape}")
        
        # Define loss function (no print statements inside for JAX tracing)
        def surrogate_loss_fn(params):
            # Forward pass with gradients
            rng_key, net_key = random.split(self.rng_key)
            predictions = self.surrogate_net.apply(
                params, net_key, tensor, mapper.target_idx, variables
            )
            
            # Get parent probabilities
            if 'parent_probabilities' in predictions:
                pred_probs = predictions['parent_probabilities']
            else:
                # Fallback - extract from marginal_parent_probs
                marginal_probs = predictions.get('marginal_parent_probs', {})
                pred_probs = jnp.array([
                    marginal_probs.get(var, 0.0) for var in variables if var != target_var
                ])
            
            # Binary cross-entropy loss
            eps = 1e-8
            positive_term = true_labels * jnp.log(pred_probs + eps)
            negative_term = (1 - true_labels) * jnp.log(1 - pred_probs + eps)
            loss = -jnp.mean(positive_term + negative_term)
            
            return loss
        
        # Use pre-initialized optimizer (AVICI-style state management)
        print(f"\\n⚙️  OPTIMIZER SETUP:")
        if self.surrogate_opt_state is None:
            print(f"  Initializing optimizer state for surrogate parameters")
            self.surrogate_opt_state = self.surrogate_optimizer.init(self.surrogate_params)
            print(f"  ✅ Optimizer state initialized")
        else:
            print(f"  Using persistent optimizer state")
        
        # (Old parameter debugging removed for cleaner output)
        
        print(f"\\n🎯 GRADIENT COMPUTATION:")
        loss, grads = jax.value_and_grad(surrogate_loss_fn)(self.surrogate_params)
        
        # Essential gradient checks only
        total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
        print(f"  Total gradient norm: {total_grad_norm:.3f}")
        
        if total_grad_norm < 1e-8:
            print(f"  ❌ Gradients are essentially zero!")
        elif total_grad_norm > 100:
            print(f"  ⚠️ Gradients exploding: {total_grad_norm:.1f}")
        else:
            print(f"  ✅ Gradient norms reasonable")
        
        print(f"\\n🔄 PARAMETER UPDATE:")
        # Store old parameters for comparison
        old_param_total = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(self.surrogate_params)))
        
        # Apply update
        updates, self.surrogate_opt_state = self.surrogate_optimizer.update(
            grads, self.surrogate_opt_state, self.surrogate_params
        )
        
        # Check update magnitude
        total_update_norm = jnp.sqrt(sum(jnp.sum(u**2) for u in jax.tree.leaves(updates)))
        print(f"  Update norm: {total_update_norm:.6f}")
        
        self.surrogate_params = optax.apply_updates(self.surrogate_params, updates)
        
        # Verify parameters actually changed
        new_param_total = jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree.leaves(self.surrogate_params)))
        param_change = jnp.abs(new_param_total - old_param_total)
        
        print(f"  Parameter change magnitude: {param_change:.6f}")
        
        if param_change < 1e-8:
            print(f"  ❌ Parameters didn't change!")
        else:
            print(f"  ✅ Parameters updated successfully")
        
        return float(loss)
    
    def _compute_simple_structure_metrics(self, scm) -> Dict[str, float]:
        """Compute simple F1 score for structure learning with debugging."""
        if not self.use_surrogate:
            return {}
        
        print(f"\\n📈 STRUCTURE METRICS DEBUG:")
        
        target_var = get_target(scm)
        true_parents = set(get_parents(scm, target_var))
        
        print(f"  Target var: {target_var}")
        print(f"  True parents: {true_parents}")
        
        # Create dummy buffer for prediction
        dummy_buffer = ExperienceBuffer()
        dummy_samples = sample_from_linear_scm(scm, 10, seed=42)
        for sample in dummy_samples:
            dummy_buffer.add_observation(sample)
        
        tensor, mapper = buffer_to_three_channel_tensor(
            dummy_buffer, target_var, max_history_size=100, standardize=True
        )
        
        print(f"  Dummy tensor shape: {tensor.shape}")
        print(f"  Mapper variables: {mapper.variables}")
        
        # Get predictions
        self.rng_key, pred_key = random.split(self.rng_key)
        predictions = self.surrogate_net.apply(
            self.surrogate_params, pred_key, tensor, mapper.target_idx, mapper.variables
        )
        
        print(f"  Prediction keys: {list(predictions.keys())}")
        
        # Extract probabilities with debugging
        if 'marginal_parent_probs' in predictions:
            probs = predictions['marginal_parent_probs']
            print(f"  Marginal parent probs: {probs}")
            
            # Debug prediction extraction
            target_probs = probs.get(target_var, {})
            print(f"  Target '{target_var}' probs: {target_probs}")
            
            predicted_parents = {
                var for var, prob in target_probs.items() 
                if prob > 0.5 and var != target_var
            }
            
            print(f"  Thresholded predictions (>0.5): {predicted_parents}")
        elif 'parent_probabilities' in predictions:
            pred_probs = predictions['parent_probabilities']
            print(f"  Parent probabilities: {pred_probs}")
            
            # Map back to variable names
            predicted_parents = set()
            for i, prob in enumerate(pred_probs):
                if prob > 0.5:
                    var_name = mapper.get_name(i)
                    if var_name != target_var:
                        predicted_parents.add(var_name)
            
            print(f"  Thresholded predictions (>0.5): {predicted_parents}")
        else:
            print(f"  ❌ No recognized prediction format!")
            predicted_parents = set()
        
        # Compute F1 with debugging
        print(f"\\n📊 F1 COMPUTATION:")
        print(f"  True parents: {true_parents}")
        print(f"  Predicted parents: {predicted_parents}")
        
        if true_parents:
            tp = len(predicted_parents & true_parents)
            fp = len(predicted_parents - true_parents)
            fn = len(true_parents - predicted_parents)
            
            print(f"  True Positives: {tp}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1: {f1:.3f}")
        else:
            f1 = 1.0 if len(predicted_parents) == 0 else 0.0
            print(f"  No true parents case, F1: {f1:.3f}")
        
        return {'f1_score': f1}
    
    def _create_joint_metrics(self, episode_metrics: Dict[str, Any]) -> JointTrainingMetrics:
        """Create joint training metrics."""
        return JointTrainingMetrics(
            episode=episode_metrics['episode'],
            phase=self.current_phase,
            mean_reward=episode_metrics.get('mean_reward', 0.0),
            surrogate_loss=episode_metrics.get('surrogate_loss', 0.0),
            target_improvement=episode_metrics.get('target_improvement', 0.0),
            f1_score=episode_metrics.get('structure_metrics', {}).get('f1_score', 0.0)
        )
    
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save joint checkpoint in compatible format."""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save policy checkpoint
        policy_path = checkpoint_dir / 'policy.pkl'
        save_checkpoint(
            path=policy_path,
            params=self.policy_params,
            architecture={
                'hidden_dim': getattr(self, 'hidden_dim', 256),
                'architecture_type': getattr(self, 'policy_architecture', 'permutation_invariant')
            },
            model_type='policy',
            model_subtype='grpo',
            training_config={
                'learning_rate': self.learning_rate,
                'joint_trained': True,
                'policy_episodes': self.policy_episodes,
                'surrogate_episodes': self.surrogate_episodes
            },
            metadata={'trainer': 'SimplifiedJointACBOTrainer'}
        )
        
        # Save surrogate checkpoint
        if self.use_surrogate:
            surrogate_path = checkpoint_dir / 'surrogate.pkl'
            save_checkpoint(
                path=surrogate_path,
                params=self.surrogate_params,
                architecture={
                    'hidden_dim': 128,
                    'num_layers': 4,
                    'num_heads': 8
                },
                model_type='surrogate',
                model_subtype='continuous_parent_set',
                training_config={'learning_rate': 1e-3, 'joint_trained': True},
                metadata={'trainer': 'SimplifiedJointACBOTrainer'}
            )
        
        logger.info(f"Saved joint checkpoints to {checkpoint_dir}/")
    
    def _run_policy_episode_with_interventions(self, episode_idx: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """
        Run policy episode with multiple interventions.
        
        This is the correct episode structure:
        - Episode: Multiple interventions on same SCM
        - Intervention: Generate GRPO candidates → update policy → add best to buffer
        """
        # Get SCM info
        from ..data_structures.scm import get_variables, get_target, get_parents
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        true_parents = list(get_parents(scm, target_var))
        
        # DEBUG: Print SCM details at episode start
        print(f"\n{'='*70}")
        print(f"📊 EPISODE {episode_idx} - SCM DETAILS")
        print(f"{'='*70}")
        print(f"  Target: {target_var}")
        print(f"  Parents: {true_parents if true_parents else 'None (root variable)'}")
        print(f"  All variables: {variables}")
        
        # Try to get coefficients if available
        try:
            from ..experiments.variable_scm_factory import get_scm_info
            scm_info = get_scm_info(scm)
            if 'coefficients' in scm_info:
                print(f"  Coefficients: {scm_info['coefficients']}")
        except:
            pass
        print(f"{'='*70}\n")
        
        # Initialize buffer with observations
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        
        # Add initial posterior for observations
        if self.use_surrogate and hasattr(self, '_get_surrogate_predictions'):
            # Get initial posterior
            temp_buffer = ExperienceBuffer()
            for sample in obs_samples:
                temp_buffer.add_observation(sample)
            initial_tensor, mapper = buffer_to_three_channel_tensor(
                temp_buffer, target_var, max_history_size=100, standardize=True
            )
            initial_posterior = self._get_surrogate_predictions(temp_buffer, target_var, variables)
        else:
            initial_posterior = create_uniform_posterior(variables, target_var)
        
        # Add observations with posterior
        for sample in obs_samples:
            buffer.add_observation(sample, posterior=initial_posterior)
        
        initial_buffer_size = buffer.size()
        logger.info(f"\nEpisode {episode_idx} starting: buffer initialized with {initial_buffer_size} observations")
        
        # INTERVENTION LOOP - this is what was missing!
        intervention_metrics = []
        all_rewards = []
        all_target_values = []  # Track target progression
        
        for intervention_idx in range(self.max_interventions):
            print(f"\n{'='*50}")
            print(f"INTERVENTION {intervention_idx+1}/{self.max_interventions}")
            print(f"{'='*50}")
            
            # Use parent's proven GRPO implementation for single intervention
            key, intervention_key = random.split(key)
            
            # Run single GRPO intervention (generates candidates, updates policy, returns best)
            single_result = super()._run_single_grpo_intervention(
                buffer, scm, target_var, variables, intervention_key
            )
            
            # Check for convergence if we have quantile architecture info
            print(f"\n🔍 CONVERGENCE CHECK (Intervention {intervention_idx + 1}):")
            print(f"  single_result keys: {list(single_result.keys())}")
            
            if 'best_intervention' in single_result:
                best_int = single_result['best_intervention']
                print(f"  best_intervention keys: {list(best_int.keys())}")
                
                if 'debug_info' in best_int and best_int['debug_info']:
                    debug_info = best_int['debug_info']
                    print(f"  debug_info keys: {list(debug_info.keys())}")
                    print(f"  debug_info content: {debug_info}")
                    
                    if 'selected_var_idx' in debug_info and 'selected_quantile' in debug_info:
                        var_idx = debug_info['selected_var_idx']
                        quantile_idx = debug_info['selected_quantile']
                        probability = debug_info.get('selection_probability', 0.0)
                        
                        print(f"  📊 Extracted: var_idx={var_idx}, quantile_idx={quantile_idx}, prob={probability:.3f}")
                        
                        # Check for convergence
                        if self._check_convergence(var_idx, quantile_idx, probability):
                            self.convergence_detected = True
                            print(f"  ⚡ Setting convergence flag for early rotation")
                            break  # Exit intervention loop early
                    else:
                        print(f"  ❌ Missing required keys in debug_info")
                else:
                    print(f"  ❌ No debug_info in best_intervention")
            else:
                print(f"  ❌ No best_intervention in single_result")
            
            # Add best intervention to buffer
            if 'best_intervention' in single_result and single_result['best_intervention']['outcome'] is not None:
                buffer.add_intervention(
                    single_result['best_intervention']['intervention'],
                    single_result['best_intervention']['outcome'],
                    posterior=single_result['best_intervention'].get('posterior')
                )
                
                # Log buffer growth
                current_size = buffer.size()
                print(f"Buffer progression: {current_size-1} -> {current_size}")
                
                # Track target value for this intervention
                outcome = single_result['best_intervention']['outcome']
                if outcome:
                    target_value = get_values(outcome).get(target_var, 0.0)
                    all_target_values.append(target_value)
                    print(f"Selected intervention TARGET: {target_value:.3f}")
                
                # Track metrics
                intervention_metrics.append(single_result)
                all_rewards.extend(single_result.get('candidate_rewards', []))
        
        # Episode summary
        final_buffer_size = buffer.size()
        total_interventions_added = final_buffer_size - initial_buffer_size
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_idx} COMPLETE")
        print(f"{'='*60}")
        print(f"Total interventions: {len(intervention_metrics)}")
        print(f"Buffer growth: {initial_buffer_size} -> {final_buffer_size} (+{total_interventions_added})")
        print(f"Mean reward across all interventions: {np.mean(all_rewards) if all_rewards else 0:.3f}")
        
        # TARGET PROGRESSION ANALYSIS within episode
        if all_target_values:
            print(f"\n📈 TARGET PROGRESSION (within episode):")
            print(f"  Values: {[f'{v:.3f}' for v in all_target_values]}")
            print(f"  Best (lowest): {min(all_target_values):.3f}")
            print(f"  Worst (highest): {max(all_target_values):.3f}")
            print(f"  Trend: {all_target_values[0]:.3f} → {all_target_values[-1]:.3f} ({all_target_values[-1] - all_target_values[0]:+.3f})")
            
            # Check if improving within episode (for minimization)
            improvement = all_target_values[0] - all_target_values[-1]
            if improvement > 0.1:
                print(f"  ✅ IMPROVING within episode! ({improvement:+.3f})")
            elif improvement < -0.1:
                print(f"  ⚠️ Getting worse within episode ({improvement:+.3f})")
            else:
                print(f"  ➖ No clear trend within episode ({improvement:+.3f})")
        
        # DEBUG: Print surrogate predictions at end of episode
        if self.use_surrogate and buffer:
            try:
                final_tensor, mapper = buffer_to_three_channel_tensor(
                    buffer, target_var, max_history_size=100, standardize=True
                )
                surrogate_out = self.surrogate_predict_fn(final_tensor, target_var, variables)
                
                if 'parent_probs' in surrogate_out:
                    print(f"\n🔮 SURROGATE PREDICTIONS (End of Episode):")
                    probs = surrogate_out['parent_probs']
                    for i, var in enumerate(variables):
                        if var != target_var and i < len(probs):
                            prob = float(probs[i])
                            is_parent = "✓" if var in true_parents else ""
                            print(f"  {var}: {prob:.3f} {is_parent}")
                    print()
            except Exception as e:
                print(f"  Could not get surrogate predictions: {e}")
        
        return {
            'episode': episode_idx,
            'mean_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
            'n_interventions': len(intervention_metrics),
            'buffer_growth': total_interventions_added,
            'intervention_metrics': intervention_metrics,
            'structure_metrics': {},  # Could compute if needed
            'n_variables': len(variables),
            'scm_type': scm_name,
            'target_values': all_target_values,  # Store for cross-episode analysis
            'best_target': min(all_target_values) if all_target_values else 0.0,
            'target_improvement': (all_target_values[0] - all_target_values[-1]) if len(all_target_values) >= 2 else 0.0
        }