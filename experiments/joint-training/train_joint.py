#!/usr/bin/env python3
"""
Joint ACBO Training with alternating policy and surrogate updates.
Combines patterns from train_grpo_diverse_fixed.py and train_avici_style.py.
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import haiku as hk
import optax
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.causal_bayes_opt.training.joint_acbo_trainer import JointACBOTrainer
from src.causal_bayes_opt.training.diverse_scm_generator import DiverseSCMGenerator
from src.causal_bayes_opt.utils.checkpoint_utils import save_checkpoint, load_checkpoint, create_model_from_checkpoint
from src.causal_bayes_opt.experiments.variable_scm_factory import VariableSCMFactory
from src.causal_bayes_opt.data_structures.scm import get_parents, get_variables, get_target
from src.causal_bayes_opt.data_structures.sample import get_values, create_sample
from src.causal_bayes_opt.interventions.handlers import create_perfect_intervention, clip_intervention_values
from src.causal_bayes_opt.environments.sampling import sample_with_intervention
from src.causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from src.causal_bayes_opt.data_structures.buffer import ExperienceBuffer
from src.causal_bayes_opt.training.three_channel_converter import buffer_to_three_channel_tensor
from src.causal_bayes_opt.utils.variable_mapping import VariableMapper
from src.causal_bayes_opt.avici_integration.continuous.model import ContinuousParentSetPredictionModel
from src.causal_bayes_opt.policies.simple_permutation_invariant_policy import create_simple_permutation_invariant_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def compute_gradient_norm(grads):
    """Compute L2 norm of gradients."""
    if grads is None:
        return 0.0
    leaves = jax.tree.leaves(grads)
    if not leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum(g**2) for g in leaves if g is not None)))


def compute_param_change(before_params, after_params):
    """Compute L2 norm of parameter change."""
    if before_params is None or after_params is None:
        return 0.0
    before_leaves = jax.tree.leaves(before_params)
    after_leaves = jax.tree.leaves(after_params)
    if not before_leaves or not after_leaves:
        return 0.0
    return float(jnp.sqrt(sum(jnp.sum((b-a)**2) for b, a in zip(before_leaves, after_leaves))))


def log_learning_health(grad_norm, param_change, loss_delta=None):
    """Log whether learning appears healthy."""
    health_status = []
    
    # Check gradient health
    if grad_norm < 1e-8:
        logger.warning(f"  ⚠️  Vanishing gradients: {grad_norm:.2e}")
        health_status.append("vanishing_grads")
    elif grad_norm > 100:
        logger.warning(f"  ⚠️  Exploding gradients: {grad_norm:.2e}")
        health_status.append("exploding_grads")
    else:
        logger.info(f"  ✓ Healthy gradient norm: {grad_norm:.6f}")
        health_status.append("healthy_grads")
    
    # Check parameter update health
    if param_change < 1e-8:
        logger.warning(f"  ⚠️  No parameter updates: {param_change:.2e}")
        health_status.append("frozen_params")
    elif param_change > 1.0:
        logger.warning(f"  ⚠️  Large parameter change: {param_change:.2e}")
        health_status.append("unstable_params")
    else:
        logger.info(f"  ✓ Healthy parameter update: {param_change:.6f}")
        health_status.append("healthy_params")
    
    # Check loss change if provided
    if loss_delta is not None:
        if abs(loss_delta) < 1e-8:
            logger.warning(f"  ⚠️  Loss not changing: {loss_delta:.2e}")
            health_status.append("stuck_loss")
        else:
            direction = "improving" if loss_delta < 0 else "worsening"
            logger.info(f"  ✓ Loss {direction}: {loss_delta:.6f}")
            health_status.append(f"loss_{direction}")
    
    return health_status


# DiverseSCMGenerator moved to src.causal_bayes_opt.training.diverse_scm_generator
# Import it from there instead of duplicating


def generate_diverse_graph_batch(rng_key, batch_size: int = 32, 
                                min_vars: int = 3, max_vars: int = 100) -> List[Dict]:
    """Generate diverse graph types (from train_avici_style.py)."""
    graphs = []
    
    for i in range(batch_size):
        rng_key, subkey = random.split(rng_key)
        
        num_vars = int(random.uniform(subkey, minval=min_vars, maxval=max_vars + 1))
        
        rng_key, type_key = random.split(rng_key)
        graph_type_idx = random.choice(type_key, 5)
        
        if graph_type_idx == 0:
            # Erdos-Renyi with varying edge density
            rng_key, density_key = random.split(rng_key)
            edges_per_var = random.uniform(density_key, minval=1.0, maxval=3.0)
            edge_density = min(edges_per_var / (num_vars - 1), 0.5)
            structure = 'random'
        elif graph_type_idx == 1:
            structure = 'chain'
            edge_density = 1.0 / (num_vars - 1) if num_vars > 1 else 0.0
        elif graph_type_idx == 2:
            structure = 'fork'
            edge_density = 0.3
        elif graph_type_idx == 3:
            structure = 'collider'
            edge_density = 0.3
        else:
            structure = 'mixed'
            edge_density = 0.25
        
        graphs.append({
            'num_vars': num_vars,
            'structure': structure,
            'edge_density': edge_density
        })
    
    return graphs


def load_models(policy_checkpoint: Optional[str] = None, 
                surrogate_checkpoint: Optional[str] = None,
                config: Dict[str, Any] = None,
                rng_key: jax.random.PRNGKey = None) -> Tuple:
    """Load pretrained models or initialize from scratch."""
    
    logger.info("\n" + "="*70)
    logger.info("LOADING/INITIALIZING MODELS")
    logger.info("="*70)
    
    # Policy loading/initialization
    if policy_checkpoint and Path(policy_checkpoint).exists():
        logger.info(f"Loading policy from: {policy_checkpoint}")
        checkpoint = load_checkpoint(policy_checkpoint)
        policy_net, policy_params = create_model_from_checkpoint(checkpoint)
        logger.info(f"  ✓ Loaded {checkpoint['model_subtype']} policy")
    else:
        logger.info("Initializing policy from scratch")
        hidden_dim = config.get('hidden_dim', 128)
        
        # Create dummy data for initialization
        dummy_buffer = ExperienceBuffer()
        dummy_values = {f'X{i}': 0.0 for i in range(10)}
        dummy_sample = create_sample(dummy_values, intervention_type=None)
        dummy_buffer.add_observation(dummy_sample)
        dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
        
        # Initialize policy
        policy_key, surrogate_key = random.split(rng_key)
        policy_fn = create_simple_permutation_invariant_policy(hidden_dim)
        policy_net = hk.without_apply_rng(hk.transform(policy_fn))
        policy_params = policy_net.init(policy_key, dummy_tensor, 0)
        logger.info(f"  ✓ Initialized simple_permutation_invariant policy")
    
    # Surrogate loading/initialization
    if surrogate_checkpoint and Path(surrogate_checkpoint).exists():
        logger.info(f"Loading surrogate from: {surrogate_checkpoint}")
        
        with open(surrogate_checkpoint, 'rb') as f:
            ckpt = pickle.load(f)
        
        if isinstance(ckpt, dict) and 'params' in ckpt:
            # Old format from train_avici_style.py
            surrogate_params = ckpt['params']
            logger.info("  Loaded old format checkpoint, initializing network...")
            
            # Initialize network architecture
            hidden_dim = config.get('hidden_dim', 128)
            num_layers = config.get('num_layers', 8)
            num_heads = config.get('num_heads', 8)
            key_size = config.get('key_size', 32)
            dropout = config.get('dropout', 0.1)
            
            def surrogate_fn(x, target_idx, is_training):
                model = ContinuousParentSetPredictionModel(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    key_size=key_size,
                    dropout=dropout
                )
                return model(x, target_idx, is_training)
            
            surrogate_net = hk.transform(surrogate_fn)
            logger.info(f"  ✓ Loaded surrogate with F1={ckpt.get('avg_f1', 'unknown')}")
        else:
            # Try new checkpoint_utils format
            checkpoint = load_checkpoint(surrogate_checkpoint)
            surrogate_net, surrogate_params = create_model_from_checkpoint(checkpoint)
            logger.info(f"  ✓ Loaded {checkpoint['model_subtype']} surrogate")
    else:
        logger.info("Initializing surrogate from scratch")
        hidden_dim = config.get('hidden_dim', 128)
        num_layers = config.get('num_layers', 8)
        num_heads = config.get('num_heads', 8)
        key_size = config.get('key_size', 32)
        dropout = config.get('dropout', 0.1)
        
        # Create dummy data for initialization
        dummy_buffer = ExperienceBuffer()
        dummy_values = {f'X{i}': 0.0 for i in range(10)}
        dummy_sample = create_sample(dummy_values, intervention_type=None)
        dummy_buffer.add_observation(dummy_sample)
        dummy_tensor, _ = buffer_to_three_channel_tensor(dummy_buffer, 'X0')
        
        def surrogate_fn(x, target_idx, is_training):
            model = ContinuousParentSetPredictionModel(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                key_size=key_size,
                dropout=dropout
            )
            return model(x, target_idx, is_training)
        
        surrogate_net = hk.transform(surrogate_fn)
        surrogate_params = surrogate_net.init(surrogate_key, dummy_tensor, 0, True)
        logger.info(f"  ✓ Initialized AVICI-style surrogate")
    
    logger.info("="*70 + "\n")
    
    return policy_net, policy_params, surrogate_net, surrogate_params


class JointTrainer(JointACBOTrainer):
    """Extended trainer with checkpoint loading and custom alternating logic."""
    
    def __init__(self, config, policy_checkpoint=None, surrogate_checkpoint=None):
        logger.info("Initializing JointTrainer...")
        
        # Load pretrained models if provided
        if policy_checkpoint or surrogate_checkpoint:
            rng_key = random.PRNGKey(config.get('seed', 42))
            policy_net, policy_params, surrogate_net, surrogate_params = load_models(
                policy_checkpoint, surrogate_checkpoint, config, rng_key
            )
            
            # Store in config for parent class
            config['pretrained_policy_params'] = policy_params
            config['pretrained_policy_net'] = policy_net
            config['pretrained_surrogate_params'] = surrogate_params
            config['pretrained_surrogate_net'] = surrogate_net
        
        # Configure alternating phases
        config['joint_training'] = config.get('joint_training', {})
        config['joint_training']['episodes_per_phase'] = config.get('policy_episodes_per_phase', 5)
        config['joint_training']['initial_phase'] = 'policy'
        
        # Initialize parent
        super().__init__(config=config)
        
        # Override with pretrained models if available
        if policy_checkpoint or surrogate_checkpoint:
            if policy_checkpoint:
                self.policy_params = config['pretrained_policy_params']
                self.policy_fn = config['pretrained_policy_net']
            if surrogate_checkpoint:
                self.surrogate_params = config['pretrained_surrogate_params']
                self.surrogate_net = config['pretrained_surrogate_net']
                self.surrogate_model = config['pretrained_surrogate_net']
        
        # Set up training parameters
        self.policy_episodes_per_phase = config.get('policy_episodes_per_phase', 5)
        self.surrogate_steps_per_phase = config.get('surrogate_steps_per_phase', 1000)
        self.f1_rotation_threshold = config.get('f1_rotation_threshold', 0.9)
        
        # GRPO reward weights
        self.grpo_reward_config = config.get('grpo_reward_config', {
            'target_improvement_weight': 0.7,
            'parent_accuracy_weight': 0.2,
            'info_gain_weight': 0.1,
            'exploration_bonus': 0.0
        })
        
        # SCM generator
        scm_config = config.get('scm_generation', {})
        self.scm_generator = DiverseSCMGenerator(
            min_vars=scm_config.get('min_vars', 3),
            max_vars=scm_config.get('max_vars', 30),
            seed=config.get('seed', 42)
        )
        self.scm_generator_callable = self.scm_generator
        
        # Tracking metrics
        self.policy_phase_metrics = []
        self.surrogate_phase_metrics = []
        self.current_surrogate_f1 = 0.0
        
        # Debug logging flags (from config)
        self.log_gradients = config.get('log_gradients', False)
        self.log_params = config.get('log_params', False)
        self.log_freq = config.get('log_freq', 10)
        self.debug_mode = config.get('debug', False)
        
        # If debug mode, enable all logging
        if self.debug_mode:
            self.log_gradients = True
            self.log_params = True
            self.log_freq = 1
            logger.info("  🐛 Debug mode enabled - full logging active")
        
        # Initialize optimizers for surrogate training
        self.surrogate_optimizer = optax.adamw(
            learning_rate=config.get('surrogate_lr', 0.0001),
            weight_decay=0.01
        )
        self.surrogate_opt_state = self.surrogate_optimizer.init(self.surrogate_params)
        
        logger.info(f"JointTrainer initialized with:")
        logger.info(f"  Policy episodes per phase: {self.policy_episodes_per_phase}")
        logger.info(f"  Surrogate steps per phase: {self.surrogate_steps_per_phase}")
        logger.info(f"  F1 rotation threshold: {self.f1_rotation_threshold}")
        logger.info(f"  GRPO reward config: {self.grpo_reward_config}")
    
    def _compute_grpo_rewards_with_tracking(self, candidates, target_var, scm, buffer, variables):
        """Compute rewards with configurable weights (from train_grpo_diverse_fixed.py)."""
        rewards = []
        
        weights = self.grpo_reward_config
        true_parents = set(get_parents(scm, target_var))
        
        # Track reward statistics
        all_target_values = []
        all_rewards = []
        parent_selections = 0
        
        for i, candidate in enumerate(candidates):
            # Target minimization component
            target_component = -candidate['target_value'] * weights.get('target_improvement_weight', 0.7)
            
            # Parent selection component
            is_parent = 1.0 if candidate['variable'] in true_parents else 0.0
            parent_component = is_parent * weights.get('parent_accuracy_weight', 0.2)
            if is_parent:
                parent_selections += 1
            
            # Information gain component (if using surrogate)
            info_component = 0.0
            if self.use_surrogate and 'info_gain' in candidate:
                info_component = candidate['info_gain'] * weights.get('info_gain_weight', 0.1)
            
            # Exploration bonus (optional)
            explore_component = weights.get('exploration_bonus', 0.0)
            
            # Total reward
            total_reward = target_component + parent_component + info_component + explore_component
            rewards.append(total_reward)
            all_target_values.append(candidate['target_value'])
            all_rewards.append(total_reward)
            
            # Enhanced debugging for all candidates if debug mode
            if self.debug_mode and i < 5:  # Show first 5 in debug mode
                logger.info(f"  📊 Intervention {i}: {candidate['variable']}={candidate['value']:.3f}")
                logger.info(f"     → Target value: {candidate['target_value']:.3f}")
                logger.info(f"     → Reward: {total_reward:.3f} (target:{target_component:.3f}, "
                           f"parent:{parent_component:.3f}, info:{info_component:.3f})")
                if 'advantage' in candidate:
                    logger.info(f"     → Advantage: {candidate['advantage']:.3f}")
        
        # Log batch statistics
        if self.debug_mode:
            logger.info(f"\n  📈 GRPO Batch Statistics:")
            logger.info(f"     Mean reward: {np.mean(all_rewards):.3f}")
            logger.info(f"     Reward std: {np.std(all_rewards):.3f}")
            logger.info(f"     Best reward: {np.max(all_rewards):.3f}")
            logger.info(f"     Worst reward: {np.min(all_rewards):.3f}")
            logger.info(f"     Parent selections: {parent_selections}/{len(candidates)} "
                       f"({100*parent_selections/len(candidates):.1f}%)")
            logger.info(f"     Mean target value: {np.mean(all_target_values):.3f}")
        
        return np.array(rewards)
    
    def run_surrogate_training_phase(self):
        """Run surrogate training phase (adapted from train_avici_style.py)."""
        logger.info("\n" + "="*50)
        logger.info("SURROGATE TRAINING PHASE")
        logger.info("="*50)
        
        best_f1 = 0.0
        factory = VariableSCMFactory(seed=self.seed)
        
        # Store last F1 for reporting
        self.last_surrogate_f1 = 0.0
        
        # Track metrics for phase summary
        phase_losses = []
        phase_f1_scores = []
        phase_grad_norms = []
        phase_param_changes = []
        
        # Use tqdm for progress if not in debug mode
        iterator = tqdm(range(self.surrogate_steps_per_phase), desc="Surrogate training") if not self.debug_mode else range(self.surrogate_steps_per_phase)
        
        for step in iterator:
            # Generate diverse batch of graphs
            rng_key = random.PRNGKey(self.seed + step)
            # Get max_vars from generator if available, otherwise use default
            max_vars_limit = 100
            if hasattr(self.scm_generator, 'max_vars'):
                max_vars_limit = min(100, self.scm_generator.max_vars * 3)
            
            graph_configs = generate_diverse_graph_batch(
                rng_key, batch_size=32, 
                min_vars=3, max_vars=max_vars_limit
            )
            
            # Create SCMs from configs
            scm_batch = []
            for config in graph_configs:
                scm = factory.create_variable_scm(
                    num_variables=config['num_vars'],
                    structure_type=config['structure'],
                    edge_density=config['edge_density']
                )
                scm_batch.append(scm)
            
            # Train on batch
            total_loss = 0.0
            total_f1 = 0.0
            num_scms = 0
            
            for scm in scm_batch:
                loss, f1, grad_norm, param_change = self._train_surrogate_on_scm(scm, rng_key)
                total_loss += loss
                total_f1 += f1
                num_scms += 1
                rng_key = random.split(rng_key)[0]
            
            avg_loss = total_loss / max(num_scms, 1)
            avg_f1 = total_f1 / max(num_scms, 1)
            self.current_surrogate_f1 = avg_f1
            
            # Track metrics
            phase_losses.append(avg_loss)
            phase_f1_scores.append(avg_f1)
            self.last_surrogate_f1 = avg_f1  # Store for reporting
            if grad_norm is not None:
                phase_grad_norms.append(grad_norm)
            if param_change is not None:
                phase_param_changes.append(param_change)
            
            # Log progress with enhanced metrics
            should_log = (step % self.log_freq == 0) or (step == 0) or self.debug_mode
            if should_log:
                if not self.debug_mode and hasattr(iterator, 'write'):
                    # Use tqdm.write for clean output with progress bar
                    iterator.write(f"  Step {step}: Loss={avg_loss:.4f}, F1={avg_f1:.4f}, "
                                 f"Grad={grad_norm:.6f}, Param Δ={param_change:.6f}")
                else:
                    logger.info(f"  Step {step}/{self.surrogate_steps_per_phase}: "
                               f"Loss={avg_loss:.4f}, F1={avg_f1:.4f}")
                    if self.log_gradients:
                        logger.info(f"     Gradient norm: {grad_norm:.6f}")
                    if self.log_params:
                        logger.info(f"     Param change: {param_change:.6f}")
                    
                    # Log learning health in debug mode
                    if self.debug_mode and step > 0:
                        prev_loss = phase_losses[-2] if len(phase_losses) > 1 else avg_loss
                        loss_delta = avg_loss - prev_loss
                        log_learning_health(grad_norm, param_change, loss_delta)
            
            # Check for rotation
            if avg_f1 > self.f1_rotation_threshold:
                logger.info(f"  🎯 F1 {avg_f1:.3f} > threshold {self.f1_rotation_threshold}, "
                           f"rotating SCM and ending surrogate phase early")
                self._rotate_scm([])
                break
            
            # Track best F1
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                logger.info(f"  📈 New best F1: {best_f1:.4f}")
        
        # Log phase summary
        if phase_losses:
            logger.info(f"\n  📈 SURROGATE PHASE SUMMARY:")
            logger.info(f"     Best F1: {best_f1:.4f}")
            logger.info(f"     Final F1: {phase_f1_scores[-1]:.4f}")
            logger.info(f"     F1 improvement: {phase_f1_scores[-1] - phase_f1_scores[0]:.4f}")
            logger.info(f"     Mean loss: {np.mean(phase_losses):.4f}")
            logger.info(f"     Loss reduction: {phase_losses[0] - phase_losses[-1]:.4f}")
            
            if phase_grad_norms:
                logger.info(f"     Mean gradient norm: {np.mean(phase_grad_norms):.6f}")
                logger.info(f"     Gradient norm range: [{np.min(phase_grad_norms):.6f}, {np.max(phase_grad_norms):.6f}]")
            
            if phase_param_changes:
                logger.info(f"     Mean param change: {np.mean(phase_param_changes):.6f}")
                logger.info(f"     Total param distance: {np.sum(phase_param_changes):.6f}")
        
        logger.info(f"Surrogate phase complete. Best F1: {best_f1:.4f}")
        logger.info("="*50 + "\n")
    
    def _train_surrogate_on_scm(self, scm, rng_key):
        """Train surrogate on a single SCM (adapted from train_avici_style.py)."""
        variables = get_variables(scm)
        target_var = get_target(scm)
        mapper = VariableMapper(variables, target_variable=target_var)
        target_idx = mapper.target_idx
        true_parents = get_parents(scm, target_var)
        
        # Store params before update for tracking
        before_params = jax.tree.map(lambda x: x.copy() if hasattr(x, 'copy') else x, self.surrogate_params)
        
        # Generate training data
        buffer = ExperienceBuffer()
        
        # Observational data
        rng_key, obs_key = random.split(rng_key)
        obs_seed = int(obs_key[0]) % 1000000
        samples = sample_from_linear_scm(scm, n_samples=600, seed=obs_seed)
        for sample in samples:
            buffer.add_observation(sample)
        
        # Interventional data
        for _ in range(200):
            rng_key, action_key, int_key, post_key = random.split(rng_key, 4)
            
            # Random intervention
            valid_mask = jnp.ones(len(variables)).at[target_idx].set(0)
            valid_indices = jnp.where(valid_mask)[0]
            selected_idx = random.choice(action_key, valid_indices)
            selected_var = mapper.get_name(int(selected_idx))
            
            if selected_var == target_var:
                continue
                
            intervention_value = random.normal(int_key) * 2.0
            
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            post_data = sample_with_intervention(scm, intervention, 1, seed=int(post_key[0]))
            if post_data:
                buffer.add_intervention(intervention, post_data[0])
        
        # Compute loss and gradients
        def loss_fn(params):
            tensor, _ = buffer_to_three_channel_tensor(buffer, target_var)
            predictions = self.surrogate_net.apply(params, rng_key, tensor, target_idx, True)
            
            if 'parent_probabilities' in predictions:
                pred_probs = predictions['parent_probabilities']
            else:
                raw_logits = predictions.get('attention_logits', jnp.zeros(len(variables)))
                pred_probs = jax.nn.sigmoid(raw_logits)
            
            pred_probs = jnp.clip(pred_probs, 1e-7, 1 - 1e-7)
            
            # Create ground truth labels
            labels = []
            for i, var in enumerate(variables):
                if i != target_idx:
                    label = 1.0 if var in true_parents else 0.0
                    labels.append(label)
                else:
                    labels.append(0.0)
            
            labels = jnp.array(labels)
            
            # Binary cross-entropy loss
            bce_loss = -(labels * jnp.log(pred_probs) + (1 - labels) * jnp.log(1 - pred_probs))
            return jnp.mean(bce_loss), (pred_probs, labels)
        
        (loss, (pred_probs, labels)), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.surrogate_params)
        
        # Compute gradient norm
        grad_norm = compute_gradient_norm(grads)
        
        # Update parameters
        updates, self.surrogate_opt_state = self.surrogate_optimizer.update(
            grads, self.surrogate_opt_state, self.surrogate_params
        )
        self.surrogate_params = optax.apply_updates(self.surrogate_params, updates)
        
        # Compute parameter change
        param_change = compute_param_change(before_params, self.surrogate_params)
        
        # Compute F1 score and other metrics
        predictions = pred_probs > 0.5
        tp = jnp.sum(predictions * labels)
        fp = jnp.sum(predictions * (1 - labels))
        fn = jnp.sum((1 - predictions) * labels)
        tn = jnp.sum((1 - predictions) * (1 - labels))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Log detailed metrics in debug mode
        if self.debug_mode and np.random.random() < 0.1:  # Log 10% of SCMs in debug
            logger.info(f"    📊 SCM Training Metrics:")
            logger.info(f"       Loss: {float(loss):.6f}")
            logger.info(f"       F1: {float(f1):.4f}, Precision: {float(precision):.4f}, Recall: {float(recall):.4f}")
            logger.info(f"       Accuracy: {float(accuracy):.4f}")
            logger.info(f"       Gradient norm: {grad_norm:.6f}")
            logger.info(f"       Param change: {param_change:.6f}")
        
        return float(loss), float(f1), grad_norm, param_change
    
    def _run_collaborative_episode(self, episode_idx):
        """Override to handle alternating phases."""
        
        # Check if we need to switch phases
        if self.phase_episode_count >= self.episodes_per_phase:
            # Switch phase
            if self.current_phase == 'policy':
                self.current_phase = 'surrogate'
                self.phase_episode_count = 0
                logger.info(f"\n🔄 Switching to SURROGATE phase at episode {episode_idx}")
            else:  # surrogate phase ending
                self.current_phase = 'policy'
                self.phase_episode_count = 0
                logger.info(f"🔄 Switching to POLICY phase at episode {episode_idx}")
        
        # Execute the current phase
        if self.current_phase == 'policy':
            # Run policy training episode
            result = super()._run_collaborative_episode(episode_idx)
            self.phase_episode_count += 1
            return result
        else:  # surrogate phase
            # Run surrogate training for this "episode slot"
            logger.info(f"Running surrogate training (episode {episode_idx})...")
            self.run_surrogate_training_phase()
            self.phase_episode_count += 1
            
            # Return minimal metrics for surrogate phase
            return {
                'phase': 'surrogate',
                'episode': episode_idx,
                'surrogate_steps': self.surrogate_steps_per_phase,
                'mean_f1': getattr(self, 'last_surrogate_f1', 0.0)
            }


def create_config(args) -> Dict[str, Any]:
    """Create configuration from command line arguments."""
    
    config = {
        # Model initialization
        'policy_checkpoint': args.policy_checkpoint,
        'surrogate_checkpoint': args.surrogate_checkpoint,
        
        # Architecture (when training from scratch)
        'policy_architecture': 'simple_permutation_invariant',
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': 8,
        'key_size': 32,
        'dropout': 0.1,
        
        # GRPO reward weights (configurable)
        'grpo_reward_config': {
            'target_improvement_weight': args.target_weight,
            'parent_accuracy_weight': args.parent_weight,
            'info_gain_weight': args.info_weight,
            'exploration_bonus': args.exploration_bonus
        },
        
        # Alternating schedule
        'policy_episodes_per_phase': args.policy_episodes,
        'surrogate_steps_per_phase': args.surrogate_steps,
        'f1_rotation_threshold': args.f1_threshold,
        'episodes_per_phase': args.policy_episodes,  # For JointACBOTrainer
        
        # SCM generation
        'scm_generation': {
            'min_vars': args.min_vars,
            'max_vars': args.max_vars,
            'generator_type': 'diverse'
        },
        
        # Training settings
        'max_episodes': args.episodes,
        'obs_per_episode': 10,
        'max_interventions': 30,
        'use_surrogate': not args.no_surrogate,
        'use_grpo_rewards': True,
        'use_fixed_std': True,
        'fixed_std': 0.5,
        'learning_rate': args.learning_rate,
        'surrogate_lr': args.surrogate_lr,
        'batch_size': 32,
        'seed': args.seed,
        
        # GRPO configuration
        'grpo_config': {
            'group_size': 10,
            'entropy_coefficient': 0.001,
            'clip_ratio': 0.2,
            'gradient_clip': 1.0,
            'ppo_epochs': 4,
            'normalize_advantages': True
        },
        
        # Logging and debugging
        'checkpoint_dir': 'experiments/joint-training/checkpoints',
        'verbose': args.verbose,
        'debug': args.debug,
        'log_gradients': args.log_gradients,
        'log_params': args.log_params,
        'log_freq': args.log_freq,
        'log_every': 10,
        'save_every': 50
    }
    
    return config


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Joint ACBO Training")
    
    # Model checkpoints
    parser.add_argument('--policy-checkpoint', type=str, default=None,
                       help='Path to pretrained policy checkpoint (optional)')
    parser.add_argument('--surrogate-checkpoint', type=str, default=None,
                       help='Path to pretrained surrogate checkpoint (optional)')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=200,
                       help='Total number of training episodes')
    parser.add_argument('--policy-episodes', type=int, default=5,
                       help='Policy episodes per phase')
    parser.add_argument('--surrogate-steps', type=int, default=1000,
                       help='Surrogate training steps per phase')
    parser.add_argument('--f1-threshold', type=float, default=0.9,
                       help='F1 threshold for SCM rotation')
    
    # GRPO reward weights
    parser.add_argument('--target-weight', type=float, default=0.7,
                       help='Weight for target improvement')
    parser.add_argument('--parent-weight', type=float, default=0.2,
                       help='Weight for parent accuracy')
    parser.add_argument('--info-weight', type=float, default=0.1,
                       help='Weight for information gain')
    parser.add_argument('--exploration-bonus', type=float, default=0.0,
                       help='Exploration bonus')
    
    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for models')
    parser.add_argument('--num-layers', type=int, default=8,
                       help='Number of layers for surrogate')
    
    # SCM generation
    parser.add_argument('--min-vars', type=int, default=3,
                       help='Minimum number of variables')
    parser.add_argument('--max-vars', type=int, default=30,
                       help='Maximum number of variables')
    
    # Learning rates
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Policy learning rate')
    parser.add_argument('--surrogate-lr', type=float, default=1e-4,
                       help='Surrogate learning rate')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-surrogate', action='store_true',
                       help='Disable surrogate (pure GRPO)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with 2 episodes')
    
    # Debug and logging flags
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with full logging')
    parser.add_argument('--log-gradients', action='store_true',
                       help='Log gradient norms')
    parser.add_argument('--log-params', action='store_true',
                       help='Log parameter changes')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Logging frequency (default: 10)')
    
    args = parser.parse_args()
    
    # Override for quick test
    if args.quick_test:
        args.episodes = 2
        args.policy_episodes = 1
        args.surrogate_steps = 100
        logger.info("Running quick test mode")
    
    # Create configuration
    logger.info("\n" + "="*70)
    logger.info("JOINT ACBO TRAINING")
    logger.info("="*70)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Policy checkpoint: {args.policy_checkpoint or 'None (from scratch)'}")
    logger.info(f"Surrogate checkpoint: {args.surrogate_checkpoint or 'None (from scratch)'}")
    logger.info(f"Alternating: {args.policy_episodes} policy episodes, {args.surrogate_steps} surrogate steps")
    logger.info(f"GRPO weights: target={args.target_weight}, parent={args.parent_weight}, info={args.info_weight}")
    logger.info(f"SCM variables: {args.min_vars}-{args.max_vars}")
    logger.info("="*70 + "\n")
    
    config = create_config(args)
    
    # Initialize trainer
    logger.info("Initializing JointTrainer...")
    trainer = JointTrainer(
        config=config,
        policy_checkpoint=args.policy_checkpoint,
        surrogate_checkpoint=args.surrogate_checkpoint
    )
    logger.info("✅ Trainer initialized\n")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70 + "\n")
    
    try:
        # Train with the SCM generator
        results = trainer.train(scms=trainer.scm_generator)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("experiments/joint-training/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"joint_training_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\n✅ Training complete! Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())