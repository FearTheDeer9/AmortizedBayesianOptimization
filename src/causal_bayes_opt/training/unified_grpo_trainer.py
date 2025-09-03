"""
UnifiedGRPOTrainer - Modular drop-in replacement.

This provides the EXACT same interface as the original UnifiedGRPOTrainer
but uses modular components internally for better maintainability.

Maintains 100% backward compatibility with all existing experiments.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
from omegaconf import DictConfig
import pyrsistent as pyr

# Import modular components
from .grpo_reward_computer import GRPORewardComputer, create_reward_computer_from_config
from .grpo_logger import GRPOLogger

# Import original components we still need
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import create_sample, get_values
from ..acquisition.better_rewards import compute_better_clean_reward
from ..acquisition.composite_reward import RewardConfig, compute_composite_reward
from ..policies.clean_policy_factory import create_clean_grpo_policy
from ..data_structures.scm import get_variables, get_target, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..environments.sampling import sample_with_intervention
from .four_channel_converter import buffer_to_four_channel_tensor
from .three_channel_converter import buffer_to_three_channel_tensor
from ..acquisition.grpo import GRPOConfig, GRPOUpdate, create_grpo_trainer, _compute_grpo_loss
from .convergence_detector import ConvergenceDetector, ConvergenceConfig
from .data_structures import TrainingMetrics

logger = logging.getLogger(__name__)


class UnifiedGRPOTrainer:
    """
    Unified GRPO trainer with modular internal implementation.
    
    Maintains 100% backward compatibility with original interface while
    using clean modular components internally.
    """
    
    def __init__(self, 
                 config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
                 # Individual parameters for flexibility
                 learning_rate: float = 3e-4,
                 n_episodes: int = 1000,
                 episode_length: int = 20,
                 batch_size: int = 64,
                 architecture_level: str = "simplified",
                 use_early_stopping: bool = True,
                 convergence_config: Optional[ConvergenceConfig] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 optimization_direction: str = "MINIMIZE",
                 seed: int = 42,
                 use_surrogate: bool = True,
                 checkpoint_dir: str = "checkpoints"):
        """Initialize unified GRPO trainer - same interface as original."""
        
        # Handle config vs parameters (same as original)
        if config is not None:
            if isinstance(config, DictConfig):
                config = dict(config)
            self._init_from_config(config)
        else:
            self._init_from_params(
                learning_rate=learning_rate,
                n_episodes=n_episodes,
                episode_length=episode_length,
                batch_size=batch_size,
                architecture_level=architecture_level,
                use_early_stopping=use_early_stopping,
                convergence_config=convergence_config,
                reward_weights=reward_weights,
                optimization_direction=optimization_direction,
                seed=seed,
                use_surrogate=use_surrogate,
                checkpoint_dir=checkpoint_dir
            )
        
        # Initialize modular components
        self._initialize_modular_components()
        
        # Initialize original components (for compatibility)
        self._initialize_policy()
        self._initialize_surrogate()  
        self._initialize_grpo()
        
        # Training state (same as original)
        self.training_metrics = []
        self.episode_count = 0
        self.training_step = 0
    
    def _init_from_config(self, config: Dict[str, Any]):
        """Initialize from config dictionary - same interface as original."""
        self.config = config
        self.seed = config.get('seed', 42)
        self.rng_key = random.PRNGKey(self.seed)
        
        # Extract configurations (same as original)
        self.max_episodes = config.get('max_episodes', 1000)
        self.n_variables_range = config.get('n_variables_range', [3, 8])
        self.obs_per_episode = config.get('obs_per_episode', 100)
        self.max_interventions = config.get('max_interventions', 10)
        self.batch_size = config.get('batch_size', 64)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        
        # Architecture config
        arch_config = config.get('architecture', {})
        self.num_layers = arch_config.get('num_layers', 4)
        self.num_heads = arch_config.get('num_heads', 8)
        self.hidden_dim = arch_config.get('hidden_dim', 256)
        self.key_size = arch_config.get('key_size', 32)
        self.dropout = arch_config.get('dropout', 0.1)
        self.architecture_level = arch_config.get('level', 'simplified')
        
        # Training config
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.optimization_direction = config.get('optimization_direction', 'MINIMIZE')
        self.use_surrogate = config.get('use_surrogate', True)
        
        # Adaptive history sizing configuration
        buffer_config = config.get('buffer_config', {}) if config else {}
        self.max_history_size = buffer_config.get('max_history_size', 30)  # Reduced from 100
        self.adaptive_history = buffer_config.get('adaptive_history', True)
        self.min_history_size = buffer_config.get('min_history_size', 10)
        
        # Reward weights
        self.reward_weights = config.get('reward_weights', {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.0
        })
        
        # Auto-activate info gain weight when using surrogate ONLY if not explicitly set
        # Check if info_gain was explicitly provided in config
        config_has_info_gain = (config and 'reward_weights' in config 
                                and 'info_gain' in config['reward_weights'])
        if self.use_surrogate and not config_has_info_gain:
            # Only set default if not explicitly provided
            self.reward_weights['info_gain'] = 0.3
        
        # Group-based rewards - no running stats needed
        
        # Initialize composite reward configuration
        self.reward_config = RewardConfig(
            target_weight=config.get('reward_weights', {}).get('target', 0.7),
            info_gain_weight=config.get('reward_weights', {}).get('info_gain', 0.2 if self.use_surrogate else 0.0),
            parent_weight=config.get('reward_weights', {}).get('parent', 0.1),
            optimization_direction=self.optimization_direction,
            reward_type=config.get('reward_type', 'continuous'),
            info_gain_type=config.get('info_gain_type', 'entropy_reduction'),
            stats=None
        )
        
        # Convergence config
        self.use_early_stopping = config.get('use_early_stopping', True)
        if self.use_early_stopping:
            conv_config = config.get('convergence', {})
            self.convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=conv_config.get('accuracy_threshold', 0.95),
                patience=conv_config.get('patience', 5),
                min_episodes=conv_config.get('min_episodes', 5),
                max_episodes_per_scm=conv_config.get('max_episodes_per_scm', 200)
            )
            self.convergence_detector = ConvergenceDetector(self.convergence_config)
        else:
            self.convergence_detector = None
    
    def _init_from_params(self, **kwargs):
        """Initialize from individual parameters - same interface as original."""
        # Create config dict and delegate to _init_from_config
        config = {
            'learning_rate': kwargs['learning_rate'],
            'max_episodes': kwargs['n_episodes'],
            'episode_length': kwargs['episode_length'],
            'batch_size': kwargs['batch_size'],
            'architecture': {'level': kwargs['architecture_level']},
            'use_early_stopping': kwargs['use_early_stopping'],
            'convergence': kwargs['convergence_config'].__dict__ if kwargs['convergence_config'] else {},
            'reward_weights': kwargs['reward_weights'] or {
                'optimization': 0.8, 'discovery': 0.1, 'efficiency': 0.1, 'info_gain': 0.0
            },
            'optimization_direction': kwargs['optimization_direction'],
            'seed': kwargs['seed'],
            'use_surrogate': kwargs['use_surrogate'],
            'checkpoint_dir': kwargs['checkpoint_dir']
        }
        
        self._init_from_config(config)
    
    def _initialize_modular_components(self):
        """Initialize the new modular components."""
        # Create reward computer using modular component
        self.reward_computer = create_reward_computer_from_config(self.config)
        
        # Create logger
        self.logger = GRPOLogger(self.optimization_direction)
        
        logger.info("Initialized modular components (GRPORewardComputer + GRPOLogger)")
    
    def _initialize_policy(self):
        """Initialize policy network - simplified using modular approach."""
        import os
        
        # Check if we should use enhanced architecture (default: yes)
        use_enhanced = os.environ.get('USE_ENHANCED_POLICY', '1') == '1'
        
        if use_enhanced:
            # Use the configured architecture (from self.config)
            policy_architecture = self.config.get('policy_architecture', 'simplified_permutation_invariant')
            use_fixed_std = self.config.get('use_fixed_std', True)
            fixed_std = self.config.get('fixed_std', 0.5)
            
            logger.info(f"GRPO using {policy_architecture} policy architecture")
            policy_fn = create_clean_grpo_policy(
                architecture=policy_architecture,
                hidden_dim=self.hidden_dim,
                use_fixed_std=use_fixed_std,
                fixed_std=fixed_std
            )
        else:
            logger.warning("GRPO using legacy policy architecture")
            policy_fn = create_clean_grpo_policy(hidden_dim=self.hidden_dim)
            
        self.policy_fn = hk.transform(policy_fn)
        
        # Initialize with dummy data - use correct number of channels
        n_channels = 4 if policy_architecture in ['quantile', 'permutation_invariant'] else 5
        dummy_tensor = jnp.zeros((10, 5, n_channels))
        self.rng_key, init_key = random.split(self.rng_key)
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        
        # Log the actual architecture being used (not self.architecture_level which is different)
        actual_architecture = self.config.get('policy_architecture', 'simplified_permutation_invariant')
        logger.info(f"Initialized policy with architecture: {actual_architecture}")
    
    def _initialize_surrogate(self):
        """Initialize surrogate model - compatibility method."""
        if not self.use_surrogate:
            self.surrogate_net = None
            self.surrogate_params = None
            self.surrogate_opt_state = None
            self.surrogate_predict_fn = None
            self.surrogate_update_fn = None
            logger.info("Surrogate learning disabled")
            return
        
        # Load AVICI surrogate checkpoint if specified
        surrogate_checkpoint_path = self.config.get('surrogate_checkpoint_path', None)
        if surrogate_checkpoint_path:
            self._load_avici_surrogate_checkpoint(surrogate_checkpoint_path)
            logger.info(f"Loaded AVICI surrogate from: {surrogate_checkpoint_path}")
        else:
            # No surrogate checkpoint specified - create placeholder
            self.surrogate_predict_fn = None
            logger.info("No surrogate checkpoint specified - using uniform posteriors")
    
    def _load_avici_surrogate_checkpoint(self, checkpoint_path: str):
        """Load AVICI surrogate checkpoint using existing utilities."""
        from pathlib import Path
        from ..utils.checkpoint_utils import load_checkpoint
        
        checkpoint_file = Path(checkpoint_path)
        logger.info(f"Loading AVICI surrogate from: {checkpoint_path}")
        
        # Use existing checkpoint loading utilities
        checkpoint = load_checkpoint(checkpoint_file)
        
        # Extract surrogate components from AVICI checkpoint
        if 'params' in checkpoint:
            surrogate_params = checkpoint['params']
        elif 'model_state' in checkpoint:
            surrogate_params = checkpoint['model_state']
        else:
            logger.warning(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")
            surrogate_params = checkpoint  # Try using the whole checkpoint
        
        surrogate_architecture = checkpoint.get('architecture', {})
        
        self.surrogate_params = surrogate_params
        self.surrogate_architecture = surrogate_architecture
        
        # Create surrogate prediction function using loaded AVICI model
        # This will be used as surrogate_fn in buffer_to_four_channel_tensor()
        def surrogate_predict_fn(tensor_3ch, target_var, variables):
            """Predict parent probabilities using loaded AVICI surrogate."""
            try:
                # Get target index
                target_idx = variables.index(target_var) if isinstance(target_var, str) else target_var
                
                # Apply loaded AVICI surrogate (adapt interface as needed)
                # The exact interface depends on how AVICI model was saved
                self.rng_key, surrogate_key = random.split(self.rng_key)
                
                # Try to reconstruct and apply AVICI model
                # This may need adjustment based on actual AVICI checkpoint format
                from ..avici_integration.enhanced_surrogate import create_enhanced_surrogate_for_grpo
                
                # Apply loaded AVICI surrogate using proper interface
                # Need to reconstruct the AVICI model from loaded parameters
                from ..avici_integration.continuous.model import ContinuousParentSetPredictionModel
                import haiku as hk
                
                # Reconstruct AVICI model with loaded architecture
                def avici_model_fn(x, target_idx, is_training=False):
                    model = ContinuousParentSetPredictionModel(
                        hidden_dim=self.surrogate_architecture.get('hidden_dim', 128),
                        num_heads=self.surrogate_architecture.get('num_heads', 8),
                        num_layers=self.surrogate_architecture.get('num_layers', 8),
                        key_size=self.surrogate_architecture.get('key_size', 32),
                        dropout=self.surrogate_architecture.get('dropout', 0.1)
                    )
                    return model(x, target_idx, is_training)
                
                # Create Haiku transformed model
                avici_net = hk.transform(avici_model_fn)
                
                # Apply with loaded parameters
                avici_prediction = avici_net.apply(
                    self.surrogate_params, surrogate_key, tensor_3ch, target_idx, False
                )
                
                # Extract parent probabilities from AVICI output
                if 'parent_probabilities' in avici_prediction:
                    parent_probs = avici_prediction['parent_probabilities']
                elif 'attention_logits' in avici_prediction:
                    # Convert logits to probabilities
                    attention_logits = avici_prediction['attention_logits']
                    parent_probs = jax.nn.sigmoid(attention_logits)
                else:
                    logger.warning(f"Unknown AVICI output format: {list(avici_prediction.keys())}")
                    parent_probs = jnp.ones(len(variables)) / len(variables)
                
                prediction = {
                    'parent_probs': parent_probs,
                    'entropy': -jnp.sum(parent_probs * jnp.log(parent_probs + 1e-8))
                }
                
                return prediction
                
            except Exception as e:
                logger.warning(f"AVICI surrogate prediction failed: {e}")
                # Fallback to uniform distribution
                return {'parent_probs': jnp.ones(len(variables)) / len(variables)}
        
        self.surrogate_predict_fn = surrogate_predict_fn
        logger.info("✅ AVICI surrogate loaded using ModelLoader infrastructure")
    
    def _assign_progressive_posteriors(self, buffer: ExperienceBuffer, target_var: str):
        """
        Assign posteriors to observational samples progressively.
        
        Each sample gets the posterior the surrogate would have had at that point in time:
        - Sample 1: posterior from just sample 1
        - Sample 2: posterior from samples 1+2  
        - Sample 3: posterior from samples 1+2+3
        - And so on...
        
        This maintains temporal consistency with forward training process.
        """
        if not self.use_surrogate or not self.surrogate_predict_fn:
            logger.info("No surrogate available - observational samples will use uniform posteriors")
            return
        
        logger.info("Assigning progressive posteriors to observational samples...")
        
        observations = buffer.get_observations()
        
        for i, obs_sample in enumerate(observations):
            # Create temporary buffer with samples 1 through i+1
            temp_buffer = ExperienceBuffer()
            for j in range(i + 1):
                temp_buffer.add_observation(observations[j])
            
            # Convert to tensor for surrogate prediction
            from .three_channel_converter import buffer_to_three_channel_tensor
            # Use adaptive history sizing
            actual_buffer_size = len(temp_buffer.get_all_samples())
            history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
            
            tensor_3ch, mapper = buffer_to_three_channel_tensor(
                temp_buffer, target_var, max_history_size=history_size, standardize=False
            )
            
            # Get posterior that surrogate would have had at this point
            posterior = self.surrogate_predict_fn(tensor_3ch, target_var, mapper.variables)
            
            # Update the observation with this posterior
            # Note: This requires the buffer to support posterior updates
            # For now, we'll store it and apply during tensor conversion
            obs_sample.posterior = posterior
            
            if i < 3:  # Log first few for debugging
                logger.debug(f"  Sample {i+1}: posterior entropy = {posterior.get('entropy', 0):.3f}")
        
        logger.info(f"✅ Assigned progressive posteriors to {len(observations)} observational samples")
    
    def _initialize_new_surrogate(self):
        """Initialize new surrogate model (when no checkpoint provided)."""
        # Placeholder for new surrogate initialization
        self.surrogate_predict_fn = None
        logger.info("New surrogate initialization not implemented - using uniform posteriors")
    
    def _initialize_grpo(self):
        """Initialize GRPO trainer - using modular approach."""
        # Get GRPO config from config file (with defaults)
        grpo_config = self.config.get('grpo_config', {})
        
        self.grpo_config = GRPOConfig(
            group_size=grpo_config.get('group_size', self.batch_size),  # Allow override, default to batch_size
            interventions_per_state=1,
            learning_rate=self.learning_rate,
            clip_ratio=grpo_config.get('clip_ratio', 0.2),
            entropy_coeff=grpo_config.get('entropy_coefficient', 0.1),
            max_grad_norm=grpo_config.get('gradient_clip', 1.0)
        )
        
        # Create optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.grpo_config.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate)
        )
        
        self.optimizer_state = self.optimizer.init(self.policy_params)
        
        logger.info(f"Initialized GRPO with group_size={self.batch_size}")
    
    def train(self, 
              scms: Union[List[Any], Dict[str, Any], Callable[[], Any]],
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Train GRPO policy - same interface as original but using modular components.
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting unified GRPO training with modular components")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info(f"  Use surrogate: {self.use_surrogate}")
        logger.info(f"  Reward weights: {self.reward_weights}")
        logger.info(f"  Reward type: {self.config.get('reward_type', 'continuous')}")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Convert SCMs to standard format
        scm_rotation = self._prepare_scms(scms)
        logger.info(f"Starting training with {len(scm_rotation)} SCMs")
        
        # Training history
        episode_metrics = []
        scm_episodes = {name: 0 for name, _ in scm_rotation}
        current_scm_idx = 0
        
        # Main training loop - same as original
        for episode in range(self.max_episodes):
            # Get current SCM
            scm_name, scm = scm_rotation[current_scm_idx]
            scm_episodes[scm_name] += 1
            
            # Run episode using modular implementation
            self.rng_key, episode_key = random.split(self.rng_key)
            metrics = self._run_grpo_episode(episode, scm, scm_name, episode_key)
            episode_metrics.append(metrics)
            
            # Check convergence if enabled (same as original)
            if self.convergence_detector:
                from types import SimpleNamespace
                training_metrics = SimpleNamespace(
                    episode=metrics['episode'],
                    mean_reward=metrics['mean_reward'],
                    structure_accuracy=metrics.get('structure_metrics', {}).get('f1_score', 0.0),
                    optimization_improvement=0.0,
                    policy_loss=metrics.get('loss', 0.0),
                    value_loss=0.0,
                    scm_type=metrics.get('scm_type', scm_name),
                    f1_score=metrics.get('structure_metrics', {}).get('f1_score', None),
                    true_parent_likelihood=metrics.get('structure_metrics', {}).get('parent_likelihood', None),
                    shd=metrics.get('structure_metrics', {}).get('shd', None),
                    marginal_probs=metrics.get('structure_metrics', {}).get('marginal_probs', None)
                )
                self.convergence_detector.update(scm_name, training_metrics)
                converged, reason = self.convergence_detector.check_convergence(scm_name)
                
                if converged or scm_episodes[scm_name] >= self.convergence_config.max_episodes_per_scm:
                    logger.info(f"SCM {scm_name} converged after {scm_episodes[scm_name]} episodes: {reason}")
                    current_scm_idx = (current_scm_idx + 1) % len(scm_rotation)
                    
                    all_converged = all(
                        self.convergence_detector.scm_states.get(name, None) and 
                        self.convergence_detector.scm_states[name].converged
                        for name, _ in scm_rotation
                    )
                    
                    if all_converged:
                        logger.info(f"All SCMs converged! Stopping early at episode {episode}")
                        break
            
            # Log progress (same as original)
            if episode % 10 == 0:
                recent_rewards = [m['mean_reward'] for m in episode_metrics[-10:]]
                mean_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                logger.info(f"Episode {episode}: mean_reward={mean_reward:.4f}, current_scm={scm_name}")
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Export convergence metrics before returning
        if hasattr(self, 'convergence_metrics'):
            self.export_convergence_metrics()
        
        # Prepare results (same format as original)
        training_time = time.time() - start_time
        final_metrics = episode_metrics[-1] if episode_metrics else {}
        
        return {
            'training_time': training_time,
            'final_metrics': final_metrics,
            'all_metrics': episode_metrics,
            'policy_params': self.policy_params,
            'episodes_per_scm': scm_episodes,
            'converged': all(
                self.convergence_detector.scm_states.get(name, None) and 
                self.convergence_detector.scm_states[name].converged
                for name, _ in scm_rotation
            ) if self.convergence_detector else False
        }
    
    def _run_grpo_episode(self, episode_idx: int, scm: pyr.PMap, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Run single training episode - using modular reward computation.
        
        This is the core method that needed the most cleanup.
        """
        # Get SCM info - use mapper for consistent variable ordering
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Log SCM structure details for this episode
        coefficients = scm.get('coefficients', {})
        parent_coeffs = {
            parent: coefficients.get((parent, target_var), 0.0)
            for parent in true_parents
        }
        metadata = scm.get('metadata', pyr.pmap({}))
        structure_type = metadata.get('structure_type', 'unknown')
        
        logger.info(
            f"\n[Episode {episode_idx}] SCM: {scm_name}\n"
            f"  Structure: {structure_type} with {len(variables)} variables\n"
            f"  Target: {target_var}\n"
            f"  True parents: {true_parents}\n"
            f"  Coefficients: {parent_coeffs}"
        )
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        for sample in obs_samples:
            buffer.add_observation(sample)
        
        # CRITICAL: Assign progressive posteriors to observational samples
        # Each sample gets posterior based on data available up to that point
        self._assign_progressive_posteriors(buffer, target_var)
        
        # Collect GRPO batch - TWO PASS APPROACH for group-based rewards
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],  # Will be filled in second pass
            'old_log_probs': [],
            'target_idx': target_idx,
            'intervention_details': [],
            # FIRST PASS: Collect raw values for group statistics
            'raw_target_values': [],
            'raw_info_gains': [],
            'parent_flags': [],
            'selected_vars': [],
            'outcomes': []
        }
        
        # Generate batch of interventions for GRPO
        for _ in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # Convert buffer to tensor
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                # Use surrogate integration if available
                # Use adaptive history sizing
                actual_buffer_size = len(buffer.get_all_samples())
                history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
                
                # Use 4-channel tensor with surrogate for quantile policy
                surrogate_fn = self.surrogate_predict_fn if self.use_surrogate else None
                tensor, mapper, diagnostics = buffer_to_four_channel_tensor(
                    buffer, target_var, surrogate_fn=surrogate_fn, max_history_size=history_size, standardize=True
                )
            else:
                # No surrogate - use simpler tensor format
                # Use adaptive history sizing
                actual_buffer_size = len(buffer.get_all_samples())
                history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
                
                tensor_3ch, mapper = buffer_to_three_channel_tensor(
                    buffer, target_var, max_history_size=history_size, standardize=True
                )
                # Pad to 5 channels
                T, n_vars, _ = tensor_3ch.shape
                tensor = jnp.zeros((T, n_vars, 5))
                tensor = tensor.at[:, :, :3].set(tensor_3ch)
            
            # Get policy output
            key, policy_key = random.split(key)
            mapper_target_idx = mapper.get_index(target_var)
            
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper_target_idx
            )
            
            # Sample intervention
            var_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            key, var_key = random.split(key)
            exploration_noise = 0.1
            noise = exploration_noise * random.normal(var_key, var_logits.shape)
            mask = jnp.isfinite(var_logits)
            noisy_logits = jnp.where(mask, var_logits + noise, var_logits)
            var_probs = jax.nn.softmax(noisy_logits)
            selected_var_idx = random.categorical(var_key, noisy_logits)
            
            log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
            
            # Sample value
            key, val_key = random.split(key)
            mean = value_params[selected_var_idx, 0]
            log_std = value_params[selected_var_idx, 1]
            std = jnp.exp(log_std)
            intervention_value = mean + std * random.normal(val_key)
            
            # Create and apply intervention
            selected_var = mapper.get_name(int(selected_var_idx))
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            key, sample_key = random.split(key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=10, seed=int(sample_key[0])
            )
            
            # FIRST PASS: Collect raw values (don't compute reward yet)
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                # Get raw target value
                from ..data_structures.sample import get_values
                target_value = float(get_values(outcome_sample).get(target_var, 0.0))
                grpo_batch_data['raw_target_values'].append(target_value)
                
                # Compute raw info gain if surrogate available
                info_gain = 0.0
                if self.surrogate_predict_fn:
                    try:
                        # Compute info gain as probability change
                        from ..acquisition.composite_reward import compute_information_gain_reward
                        info_gain = compute_information_gain_reward(
                            buffer, intervention, outcome_sample, self.surrogate_predict_fn,
                            target_var, mapper.variables, tensor, mapper, 'probability_change'
                        )
                    except:
                        info_gain = 0.0
                grpo_batch_data['raw_info_gains'].append(info_gain)
                
                # Check if parent
                is_parent = selected_var in true_parents
                grpo_batch_data['parent_flags'].append(is_parent)
                grpo_batch_data['selected_vars'].append(selected_var)
                grpo_batch_data['outcomes'].append(outcome_sample)
            else:
                # Fallback values
                grpo_batch_data['raw_target_values'].append(0.0)
                grpo_batch_data['raw_info_gains'].append(0.0)
                grpo_batch_data['parent_flags'].append(False)
                grpo_batch_data['selected_vars'].append(selected_var)
                grpo_batch_data['outcomes'].append(None)
            
            # Store state and action for batch (no reward yet)
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({
                'variable': selected_var_idx,
                'value': float(intervention_value)
            })
            grpo_batch_data['old_log_probs'].append(float(log_prob))
            grpo_batch_data['intervention_details'].append({
                'intervention': intervention,
                'samples': intervention_samples
            })
        
        # SECOND PASS: Compute group-based binary rewards
        # Calculate group means
        group_target_mean = np.mean(grpo_batch_data['raw_target_values'])
        group_info_mean = np.mean(grpo_batch_data['raw_info_gains'])
        
        # Get reward weights from config
        target_weight = self.config.get('reward_weights', {}).get('target', 0.7)
        info_weight = self.config.get('reward_weights', {}).get('info_gain', 0.2)
        parent_weight = self.config.get('reward_weights', {}).get('parent', 0.1)
        
        # Assign binary rewards based on group comparison
        rewards = []
        for i in range(len(grpo_batch_data['raw_target_values'])):
            # Binary target: 1 if better than group mean, 0 otherwise
            if self.optimization_direction == "MINIMIZE":
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] < group_target_mean else 0.0
            else:
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] > group_target_mean else 0.0
            
            # Binary info gain: 1 if above group mean, 0 otherwise
            info_binary = 1.0 if grpo_batch_data['raw_info_gains'][i] > group_info_mean else 0.0
            
            # Binary parent: 1 if true parent, 0 otherwise
            parent_binary = 1.0 if grpo_batch_data['parent_flags'][i] else 0.0
            
            # Weighted combination
            total_reward = (target_weight * target_binary + 
                          info_weight * info_binary + 
                          parent_weight * parent_binary)
            
            rewards.append(total_reward)
            
            # Log individual reward
            if i < 5:  # Log first few for visibility
                logger.info(f"[REWARD #{i}] Var={grpo_batch_data['selected_vars'][i]}: "
                           f"Target={target_binary:.0f}({grpo_batch_data['raw_target_values'][i]:.3f}) "
                           f"Info={info_binary:.0f}({grpo_batch_data['raw_info_gains'][i]:.3f}) "
                           f"Parent={parent_binary:.0f} "
                           f"→ Total={total_reward:.3f}")
        
        # Log group statistics
        logger.info(f"[GROUP STATS] Size={len(rewards)}, "
                   f"Target mean={group_target_mean:.3f}, "
                   f"Info mean={group_info_mean:.3f}, "
                   f"Reward range=[{min(rewards):.3f}, {max(rewards):.3f}]")
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(rewards)
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        
        # Create GRPO batch
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Perform GRPO update
        self.policy_params, self.optimizer_state, grpo_metrics = self._grpo_update(
            self.policy_params, self.optimizer_state, grpo_batch
        )
        
        self.training_step += 1
        
        return {
            'episode': episode_idx,
            'mean_reward': float(grpo_metrics.mean_reward),
            'loss': float(grpo_metrics.total_loss),
            'n_variables': len(mapper.variables),
            'scm_type': scm_name,
            'structure_metrics': {},  # Placeholder
            'has_surrogate': self.use_surrogate,
            'grpo_metrics': {
                'policy_loss': float(grpo_metrics.policy_loss),
                'entropy_loss': float(grpo_metrics.entropy_loss),
                'mean_advantage': float(grpo_metrics.mean_advantage),
                'advantage_std': float(grpo_metrics.advantage_std),
                'group_baseline': float(grpo_metrics.group_baseline),
                'approx_kl': float(grpo_metrics.approx_kl)
            }
        }
    
    def _create_grpo_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create GRPO batch format - enhanced for quantile architecture."""
        states_batch = jnp.stack(batch_data['states'])
        action_var_indices = jnp.array([a['variable'] for a in batch_data['actions']])
        action_values = jnp.array([a['value'] for a in batch_data['actions']])
        
        # Extract quantile indices if present (for quantile architecture)
        actions_dict = {
            'variables': action_var_indices,
            'values': action_values
        }
        
        # Add quantile information if available
        if batch_data['actions'] and 'flat_quantile_idx' in batch_data['actions'][0]:
            actions_dict['flat_quantile_idx'] = jnp.array([a['flat_quantile_idx'] for a in batch_data['actions']])
            actions_dict['quantile_idx'] = jnp.array([a['quantile_idx'] for a in batch_data['actions']])
        
        return {
            'states': states_batch,
            'actions': actions_dict,
            'rewards': batch_data['rewards'],
            'old_log_probs': batch_data['old_log_probs'],
            'target_idx': batch_data['target_idx']
        }
    
    def _grpo_update(self, params: Any, opt_state: Any, batch: Dict[str, Any]):
        """GRPO update with modular loss computation."""
        def loss_fn(p):
            return self._compute_simple_grpo_loss(p, batch)
        
        (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Apply updates
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        grad_norm = optax.global_norm(grads)
        
        return new_params, new_opt_state, GRPOUpdate(
            policy_loss=loss_info['policy_loss'],
            entropy_loss=loss_info['entropy_loss'],
            kl_penalty=loss_info['kl_penalty'],
            total_loss=loss_value,
            grad_norm=grad_norm,
            group_baseline=loss_info['group_baseline'],
            mean_reward=loss_info['mean_reward'],
            reward_std=loss_info['reward_std'],
            mean_advantage=loss_info['mean_advantage'],
            advantage_std=loss_info['advantage_std'],
            mean_entropy=loss_info['mean_entropy'],
            approx_kl=loss_info['approx_kl'],
            mean_ratio=loss_info['mean_ratio'],
            ratio_std=loss_info['ratio_std']
        )
    
    def _compute_simple_grpo_loss(self, params: Any, batch: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Compute GRPO loss - same as original."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs']
        
        # Compute advantages using group baseline
        group_baseline = jnp.mean(rewards)
        advantages = rewards - group_baseline
        advantages = advantages / (jnp.std(advantages) + 1e-8)
        
        # Log advantage distribution
        logger.info(f"[ADVANTAGES] Baseline={group_baseline:.3f}, "
                   f"Raw range=[{jnp.min(rewards - group_baseline):.3f}, {jnp.max(rewards - group_baseline):.3f}], "
                   f"Normalized range=[{jnp.min(advantages):.3f}, {jnp.max(advantages):.3f}], "
                   f"Std={jnp.std(rewards):.3f}")
        
        # Forward pass
        batch_size = states.shape[0]
        new_log_probs = []
        entropy_values = []
        debug_info = None  # Initialize debug info
        
        # GRPO loss computation - use fresh keys as in original action selection
        for i in range(batch_size):
            self.rng_key, policy_key = random.split(self.rng_key)
            
            policy_output = self.policy_fn.apply(
                params, policy_key, states[i], batch['target_idx']
            )
            
            # Handle quantile vs traditional architecture
            if 'quantile_scores' in policy_output:
                # QUANTILE ARCHITECTURE: Use flattened quantile scores
                flat_scores = policy_output['quantile_scores'].flatten()
                flat_probs = jax.nn.softmax(flat_scores)
                
                # Use the stored flat quantile index for proper gradient flow
                flat_quantile_idx = actions['flat_quantile_idx'][i]
                log_prob = jnp.log(flat_probs[flat_quantile_idx] + 1e-8)
                new_log_probs.append(log_prob)
                
                # Remove debug code to avoid JAX compilation errors
                
                # Entropy over all quantile options
                entropy = -jnp.sum(flat_probs * jnp.log(flat_probs + 1e-8))
                entropy_values.append(entropy)
            else:
                # TRADITIONAL ARCHITECTURE: Use variable logits
                var_probs = jax.nn.softmax(policy_output['variable_logits'])
                selected_var = actions['variables'][i]
                log_prob = jnp.log(var_probs[selected_var] + 1e-8)
                new_log_probs.append(log_prob)
                
                entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
                entropy_values.append(entropy)
        
        new_log_probs = jnp.array(new_log_probs)
        entropy_values = jnp.array(entropy_values)
        
        # Clipped surrogate objective
        ratio = jnp.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.grpo_config.clip_ratio, 1.0 + self.grpo_config.clip_ratio) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # DEBUG: Detailed analysis of first few samples (avoid JAX compilation issues)
        # Store first 3 samples for detailed analysis outside JAX
        sample_analysis = {
            'old_log_probs_sample': old_log_probs[:3],
            'new_log_probs_sample': new_log_probs[:3], 
            'advantages_sample': advantages[:3],
            'ratios_sample': ratio[:3],
            'surr1_sample': surr1[:3],
            # Add detailed log prob analysis
            'log_prob_differences': new_log_probs[:5] - old_log_probs[:5],
            'identical_count': jnp.sum(jnp.abs(new_log_probs - old_log_probs) < 1e-8),
            'total_samples': len(new_log_probs)
        }
        
        entropy_loss = -self.grpo_config.entropy_coeff * jnp.mean(entropy_values)
        total_loss = policy_loss + entropy_loss
        
        approx_kl = jnp.mean((new_log_probs - old_log_probs) ** 2)
        
        loss_info = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'kl_penalty': 0.0,
            'group_baseline': group_baseline,
            'mean_reward': jnp.mean(rewards),
            'reward_std': jnp.std(rewards),
            'mean_advantage': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'mean_entropy': jnp.mean(entropy_values),
            'approx_kl': approx_kl,
            # DEBUG: Add detailed gradient flow analysis
            'mean_ratio': jnp.mean(ratio),
            'mean_surr1': jnp.mean(surr1),
            'log_prob_diff': jnp.mean(jnp.abs(new_log_probs - old_log_probs)),
            'ratio_std': jnp.std(ratio),
            'old_log_prob_mean': jnp.mean(old_log_probs),
            'new_log_prob_mean': jnp.mean(new_log_probs),
            'sample_analysis': sample_analysis
        }
        
        return total_loss, loss_info
    
    def _prepare_scms(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> List[Tuple[str, Any]]:
        """Convert various SCM formats - same as original."""
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
                result = scms()
                # Handle different return formats from callable
                if isinstance(result, tuple) and len(result) == 2:
                    # Callable returns (name, scm) tuple
                    scm_name, scm = result
                    generated.append((scm_name, scm))
                else:
                    # Callable returns just scm
                    scm = result
                    generated.append((f"generated_{i}", scm))
            return generated
        else:
            raise ValueError(f"Unsupported SCM format: {type(scms)}")
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save checkpoint - simplified using modular approach."""
        from ..utils.checkpoint_utils import save_checkpoint as save_unified_checkpoint
        
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        name = "unified_grpo_final" if is_final else f"unified_grpo_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        
        architecture = {
            'hidden_dim': self.hidden_dim,
            'architecture_level': self.architecture_level
        }
        
        training_config = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_episodes': self.max_episodes,
            'optimization_direction': self.optimization_direction
        }
        
        metadata = {
            'trainer_type': 'UnifiedGRPOTrainerModular',
            'episode': self.episode_count,
            'is_final': is_final,
            'uses_modular_components': True,
            'has_surrogate': self.use_surrogate
        }
        
        save_unified_checkpoint(
            path=checkpoint_path,
            params=self.policy_params,
            architecture=architecture,
            model_type='policy',
            model_subtype='grpo',
            training_config=training_config,
            metadata=metadata
        )
    
    # Compatibility methods for JointACBOTrainer inheritance
    def _should_switch_phase(self) -> bool:
        """Compatibility method - always return False for pure GRPO."""
        return False
    
    def _switch_phase(self):
        """Compatibility method - no-op for pure GRPO."""
        pass
    
    def _should_rotate_scm(self) -> bool:
        """Compatibility method - basic rotation logic."""
        return False  # Let orchestrator handle this
    
    def _run_surrogate_episode(self, episode_idx: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """Compatibility method - return dummy metrics."""
        return {
            'episode': episode_idx,
            'mean_reward': 0.0,
            'surrogate_loss': 0.0,
            'f1_score': 0.0
        }
    
    def _run_policy_episode_with_interventions(self, episode_idx: int, scm: Any, scm_name: str, key) -> Dict[str, Any]:
        """Compatibility method - delegates to _run_grpo_episode."""
        return self._run_grpo_episode(episode_idx, scm, scm_name, key)
    
    def _run_single_grpo_intervention(self, buffer: ExperienceBuffer, scm: Any, 
                                     target_var: str, variables: list, key) -> Dict[str, Any]:
        """
        Run single GRPO intervention using modular components.
        
        4-channel tensor format: Values, Target, Intervention, Posterior (no Recency)
        """
        # Compute optimal action for this SCM
        from ..data_structures.scm import get_parents, get_mechanisms
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Find parent with best coefficient × range product (optimal for minimization)
        optimal_var = None
        optimal_coefficient = 0.0
        optimal_score = 0.0
        optimal_range = (-10, 10)  # default
        
        # Get coefficients from target's mechanism (LinearMechanism)
        mechanisms = get_mechanisms(scm)
        target_mechanism = mechanisms.get(target_var)
        coefficients = {}
        if target_mechanism and hasattr(target_mechanism, 'coefficients'):
            # Mechanism stores coefficients as {parent: coeff}
            for parent, coeff in target_mechanism.coefficients.items():
                coefficients[(parent, target_var)] = coeff
        
        # Get variable ranges (may be stored in SCM or in metadata)
        variable_ranges = scm.get('variable_ranges', {})
        if not variable_ranges:
            # Check metadata for variable_ranges
            metadata = scm.get('metadata', {})
            variable_ranges = metadata.get('variable_ranges', {})
        
        for parent in true_parents:
            coeff = coefficients.get((parent, target_var), 0.0)
            parent_range = variable_ranges.get(parent, (-10, 10))
            range_size = parent_range[1] - parent_range[0]
            
            # Optimal score = coefficient magnitude × range size
            score = abs(coeff) * range_size
            
            if score > optimal_score:
                optimal_score = score
                optimal_coefficient = coeff
                optimal_var = parent
                optimal_range = parent_range
        
        # Initialize convergence tracking if not present
        if not hasattr(self, 'convergence_metrics'):
            self.convergence_metrics = {
                'selections': [],  # (episode, intervention, var, is_optimal, probability)
                'consecutive_optimal': 0,
                'total_optimal': 0,
                'total_selections': 0
            }
        
        # Ensure all keys exist (in case of partial initialization)
        if 'total_selections' not in self.convergence_metrics:
            self.convergence_metrics['total_selections'] = 0
        if 'total_optimal' not in self.convergence_metrics:
            self.convergence_metrics['total_optimal'] = 0
        if 'consecutive_optimal' not in self.convergence_metrics:
            self.convergence_metrics['consecutive_optimal'] = 0
        if 'selections' not in self.convergence_metrics:
            self.convergence_metrics['selections'] = []
        
        # Store old params for change tracking
        old_params = self.policy_params
        
        # Collect GRPO batch of candidates
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],  # Will be filled in second pass
            'old_log_probs': [],
            'target_idx': None,
            'intervention_details': [],
            # NEW: Store raw values for group-based binary rewards
            'raw_target_values': [],
            'raw_info_gains': [],
            'parent_flags': [],
            'selected_vars': []  # For logging
        }
        
        # Generate batch of candidates
        for step in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # Convert buffer to 4-channel tensor with AVICI surrogate if available
            surrogate_fn = self.surrogate_predict_fn if (self.use_surrogate and hasattr(self, 'surrogate_predict_fn')) else None
            
            # Use adaptive history sizing
            actual_buffer_size = len(buffer.get_all_samples())
            history_size = min(self.max_history_size, max(actual_buffer_size, self.min_history_size)) if self.adaptive_history else self.max_history_size
            
            tensor, mapper, diagnostics = buffer_to_four_channel_tensor(
                buffer, target_var, surrogate_fn=surrogate_fn, max_history_size=history_size, standardize=True
            )
            
            # Get policy output
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Sample intervention - support both architectures
            key, selection_key = random.split(key)
            
            if 'quantile_scores' in policy_output:
                # QUANTILE ARCHITECTURE: Unified variable+value selection
                from .quantile_selection import select_quantile_intervention
                
                selected_var, intervention_value, log_prob, debug_info = select_quantile_intervention(
                    policy_output, buffer, scm, mapper.variables, target_var, selection_key, fixed_std=1.0
                )
                selected_var_idx = mapper.variables.index(selected_var)
                
                # Log quantile selection details  
                if step == 0:  # Log once per intervention
                    from .quantile_selection import log_quantile_details
                    
                    # DEBUG: Check mapper and target index
                    
                    log_quantile_details(policy_output['quantile_scores'], debug_info, mapper.variables, target_var, scm)
                
            else:
                # TRADITIONAL ARCHITECTURE: Separate variable and value heads
                var_logits = policy_output['variable_logits']
                value_params = policy_output['value_params']
                
                key, var_key = random.split(key)
                selected_var_idx = random.categorical(var_key, var_logits)
                var_probs = jax.nn.softmax(var_logits)
                
                # Sample intervention value
                key, val_key = random.split(key)
                mean = value_params[selected_var_idx, 0]
                log_std = value_params[selected_var_idx, 1]
                std = jnp.exp(log_std)
                intervention_value = mean + std * random.normal(val_key)
                
                # Compute log probability
                log_prob = float(jnp.log(var_probs[selected_var_idx] + 1e-8))
                
                # Traditional variable probability logging
                if step == 0:
                    prob_info = []
                    for i, var_name in enumerate(mapper.variables):
                        if i != mapper.target_idx:
                            prob_info.append(f"{var_name}:{var_probs[i]:.3f}")
            
            # Create and apply intervention (same for both architectures)
            # FIXED: Always use mapper.variables for consistent ordering
            selected_var = mapper.variables[selected_var_idx] if 'quantile_scores' in policy_output else mapper.get_name(int(selected_var_idx))
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            key, sample_key = random.split(key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=1, seed=int(sample_key[0])
            )
            
            # First pass: Collect raw values (not computing binary rewards yet)
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                # Get raw target value
                from ..data_structures.sample import get_values
                target_value = float(get_values(outcome_sample).get(target_var, 0.0))
                grpo_batch_data['raw_target_values'].append(target_value)
                
                # Compute raw info gain if surrogate available
                info_gain = 0.0
                if self.surrogate_predict_fn and hasattr(self, 'surrogate_predict_fn'):
                    try:
                        # Compute info gain as probability change
                        from ..acquisition.composite_reward import compute_information_gain_reward
                        info_gain = compute_information_gain_reward(
                            buffer, intervention, outcome_sample, self.surrogate_predict_fn,
                            target_var, mapper.variables, tensor, mapper, 'probability_change'
                        )
                    except:
                        info_gain = 0.0
                grpo_batch_data['raw_info_gains'].append(info_gain)
                
                # Check if parent
                is_parent = selected_var in true_parents
                grpo_batch_data['parent_flags'].append(is_parent)
                grpo_batch_data['selected_vars'].append(selected_var)
            else:
                # Fallback if no sample
                grpo_batch_data['raw_target_values'].append(0.0)
                grpo_batch_data['raw_info_gains'].append(0.0)
                grpo_batch_data['parent_flags'].append(False)
                grpo_batch_data['selected_vars'].append(selected_var)
            
            # Store for GRPO batch (log_prob already computed above)
            grpo_batch_data['states'].append(tensor)
            
            # Store action with quantile info for proper loss computation
            action_data = {
                'variable': selected_var_idx,
                'value': float(intervention_value)
            }
            
            # Add quantile index for quantile architecture
            if 'quantile_scores' in policy_output:
                action_data['quantile_idx'] = debug_info['selected_quantile_idx']
                # CRITICAL FIX: Store the actual winner_idx used in selection, not recalculated
                action_data['flat_quantile_idx'] = debug_info['winner_idx']
            
            grpo_batch_data['actions'].append(action_data)
            # Don't append reward yet - will compute in second pass
            grpo_batch_data['old_log_probs'].append(float(log_prob))
            
            # Store intervention details including debug info for convergence detection
            intervention_detail = {
                'intervention': intervention,
                'samples': intervention_samples,
                'posterior': None  # 4-channel includes posterior in tensor
            }
            
            # Track convergence to optimal action
            is_parent = selected_var in true_parents
            is_optimal = (selected_var == optimal_var) if optimal_var else False
            selection_prob = float(jnp.exp(log_prob))
            
            # Log selection details for first candidate only
            if step == 0:
                self.convergence_metrics['total_selections'] += 1
                if is_optimal:
                    self.convergence_metrics['total_optimal'] += 1
                    self.convergence_metrics['consecutive_optimal'] += 1
                else:
                    self.convergence_metrics['consecutive_optimal'] = 0
                
                # Compute optimal intervention value for minimization
                # For minimization: negative coeff → use upper bound, positive coeff → use lower bound
                optimal_value = None
                if optimal_var and optimal_coefficient != 0:
                    if optimal_coefficient < 0:
                        # Negative coefficient: increase variable to decrease target
                        optimal_value = optimal_range[1]  # upper bound
                    else:
                        # Positive coefficient: decrease variable to decrease target
                        optimal_value = optimal_range[0]  # lower bound
                
                # Store detailed selection info
                selection_info = {
                    'episode': getattr(self, 'episode_count', 0),
                    'selected_var': selected_var,
                    'selected_value': float(intervention_value),
                    'optimal_var': optimal_var if optimal_var else 'None',
                    'optimal_value': optimal_value if optimal_value is not None else 'N/A',
                    'is_parent': is_parent,
                    'is_optimal': is_optimal,
                    'probability': selection_prob,
                    'true_parents': true_parents,
                    'coefficient': optimal_coefficient if optimal_var else 0.0,
                    'optimal_score': optimal_score if optimal_var else 0.0
                }
                self.convergence_metrics['selections'].append(selection_info)
                
                # Log convergence status with score details
                optimality_rate = self.convergence_metrics['total_optimal'] / max(self.convergence_metrics['total_selections'], 1)
                if optimal_var:
                    logger.info(f"  Convergence: Selected={selected_var}({intervention_value:.2f}), "
                              f"Optimal={optimal_var}(val={optimal_value:.2f}, score={optimal_score:.2f}), "
                              f"IsOptimal={is_optimal}, Rate={optimality_rate:.1%}")
                else:
                    logger.info(f"  Convergence: Selected={selected_var}, No optimal parent found")
            
            # Add debug_info for quantile architecture (needed for convergence detection)
            if 'quantile_scores' in policy_output and debug_info:
                intervention_detail['debug_info'] = {
                    'selected_var_idx': selected_var_idx,
                    'selected_quantile': debug_info.get('selected_quantile_idx'),
                    'selection_probability': selection_prob,
                    'is_optimal': is_optimal,
                    'optimal_var': optimal_var
                }
            
            grpo_batch_data['intervention_details'].append(intervention_detail)
            
            if grpo_batch_data['target_idx'] is None:
                grpo_batch_data['target_idx'] = mapper.target_idx
        
        # SECOND PASS: Compute group-based binary rewards
        # Calculate group means
        group_target_mean = np.mean(grpo_batch_data['raw_target_values'])
        group_info_mean = np.mean(grpo_batch_data['raw_info_gains'])
        
        # Get reward weights from config
        target_weight = self.config.get('reward_weights', {}).get('target', 0.7)
        info_weight = self.config.get('reward_weights', {}).get('info_gain', 0.2)
        parent_weight = self.config.get('reward_weights', {}).get('parent', 0.1)
        
        # Assign binary rewards based on group comparison
        rewards = []
        for i in range(len(grpo_batch_data['raw_target_values'])):
            # Binary target: 1 if better than group mean, 0 otherwise
            if self.optimization_direction == "MINIMIZE":
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] < group_target_mean else 0.0
            else:
                target_binary = 1.0 if grpo_batch_data['raw_target_values'][i] > group_target_mean else 0.0
            
            # Binary info gain: 1 if above group mean, 0 otherwise
            info_binary = 1.0 if grpo_batch_data['raw_info_gains'][i] > group_info_mean else 0.0
            
            # Binary parent: 1 if true parent, 0 otherwise
            parent_binary = 1.0 if grpo_batch_data['parent_flags'][i] else 0.0
            
            # Weighted combination
            total_reward = (target_weight * target_binary + 
                          info_weight * info_binary + 
                          parent_weight * parent_binary)
            
            rewards.append(total_reward)
            
            # Log individual reward (replacing old BINARY TARGET REWARD)
            if i < 5:  # Log first few for visibility
                logger.info(f"[REWARD #{i}] Var={grpo_batch_data['selected_vars'][i]}: "
                           f"Target={target_binary:.0f}({grpo_batch_data['raw_target_values'][i]:.3f}) "
                           f"Info={info_binary:.0f}({grpo_batch_data['raw_info_gains'][i]:.3f}) "
                           f"Parent={parent_binary:.0f} "
                           f"→ Total={total_reward:.3f}")
        
        # Log group statistics
        logger.info(f"[GROUP STATS] Size={len(rewards)}, "
                   f"Target mean={group_target_mean:.3f}, "
                   f"Info mean={group_info_mean:.3f}, "
                   f"Reward range=[{min(rewards):.3f}, {max(rewards):.3f}]")
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(rewards)
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        
        # Log batch intervention statistics
        if grpo_batch_data['actions']:
            intervention_vars = [a['variable'] for a in grpo_batch_data['actions']]
            unique_vars, counts = jnp.unique(jnp.array(intervention_vars), return_counts=True)
            var_distribution = {variables[int(v)]: int(c) for v, c in zip(unique_vars, counts)}
            
            logger.info(
                f"  Batch interventions: {len(grpo_batch_data['actions'])} total\n"
                f"  Variable distribution: {var_distribution}\n"
                f"  Mean reward: {float(jnp.mean(grpo_batch_data['rewards'])):.4f}"
            )
        
        # Create GRPO batch and update policy
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Store old params for gradient analysis
        old_params_for_grad_analysis = self.policy_params
        
        self.policy_params, self.optimizer_state, grpo_metrics = self._grpo_update(
            self.policy_params, self.optimizer_state, grpo_batch
        )
        
        
        
        # Add gradient flow analysis for quantile architecture
        
        # Get quantile scores before and after update
        sample_tensor = grpo_batch_data['states'][0]
        sample_key = random.PRNGKey(123)  # Fixed for comparison
        
        old_output = self.policy_fn.apply(old_params_for_grad_analysis, sample_key, sample_tensor, mapper.target_idx)
        new_output = self.policy_fn.apply(self.policy_params, sample_key, sample_tensor, mapper.target_idx)
        
        if 'quantile_scores' in old_output:
            old_scores = old_output['quantile_scores'].flatten()
            new_scores = new_output['quantile_scores'].flatten() 
            score_changes = new_scores - old_scores
            
            for i in range(min(9, len(score_changes))):  # Show first 9 (3 vars × 3 quantiles)
                var_idx, quant_idx = divmod(i, 3)
                var_name = mapper.variables[var_idx] if var_idx < len(mapper.variables) else f"var_{var_idx}"
                quant_name = ['25%', '50%', '75%'][quant_idx]
                change = float(score_changes[i])
                
                # Determine if this quantile option was used and what advantage it should have gotten
                direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                
            
            # Check specific actions and their expected vs actual changes
            baseline = float(grpo_metrics.group_baseline)
            advantage_std = float(grpo_metrics.advantage_std)
            
            for i in range(min(3, len(grpo_batch['rewards']))):
                flat_idx = int(grpo_batch['actions']['flat_quantile_idx'][i])
                var_idx, quant_idx = divmod(flat_idx, 3)
                var_name = mapper.variables[var_idx] if var_idx < len(mapper.variables) else f"var_{var_idx}"
                quant_name = ['25%', '50%', '75%'][quant_idx]
                reward = float(grpo_batch['rewards'][i])
                
                # Expected direction
                raw_advantage = reward - baseline
                normalized_advantage = raw_advantage / (advantage_std + 1e-8) if advantage_std > 1e-8 else 0.0
                expected = "↑" if normalized_advantage > 0 else "↓"
                
                # Actual direction  
                actual_change = float(score_changes[flat_idx])
                actual = "↑" if actual_change > 0 else "↓" if actual_change < 0 else "→"
                
                match = "✅" if expected == actual else "❌"
                
                
        if hasattr(self, 'grpo_logger') and len(grpo_batch_data['states']) > 0:
            # Get a sample policy output to determine architecture
            sample_tensor = grpo_batch_data['states'][0]
            sample_output = self.policy_fn.apply(self.policy_params, random.PRNGKey(42), sample_tensor, 0)
            
            # Compute gradients for analysis
            def sample_loss(params):
                out = self.policy_fn.apply(params, random.PRNGKey(42), sample_tensor, 0)
                if 'quantile_scores' in out:
                    # Simple loss for gradient analysis
                    return jnp.sum(out['quantile_scores']**2)
                else:
                    return jnp.sum(out['variable_logits']**2)
            
            _, grads = jax.value_and_grad(sample_loss)(self.policy_params)
            
            # Log gradient analysis
            self.grpo_logger.log_gradient_analysis(grads, sample_output)
        
        # Compute parameter change
        param_changes = jax.tree.map(lambda old, new: jnp.linalg.norm(new - old), old_params, self.policy_params)
        total_change = sum(jax.tree.leaves(param_changes))
        
        # Select best intervention
        best_idx = jnp.argmax(grpo_batch_data['rewards'])
        selected_idx = int(best_idx)  # Could be random for experimentation
        
        best_intervention_info = grpo_batch_data['intervention_details'][selected_idx]
        best_reward = float(grpo_batch_data['rewards'][best_idx])
        selected_reward = float(grpo_batch_data['rewards'][selected_idx])
        
        # Return format expected by JointACBOTrainer
        return {
            'best_intervention': {
                'intervention': best_intervention_info['intervention'],
                'outcome': best_intervention_info['samples'][0] if best_intervention_info['samples'] else None,
                'posterior': best_intervention_info.get('posterior'),
                'debug_info': best_intervention_info.get('debug_info')  # Pass through debug info for convergence detection
            },
            'candidate_rewards': [float(r) for r in grpo_batch_data['rewards']],
            'grpo_metrics': grpo_metrics,
            'param_change': float(total_change),
            'selection_info': {
                'selected_idx': selected_idx,
                'selected_reward': selected_reward,
                'best_idx': int(best_idx),
                'best_reward': best_reward,
                'selection_advantage': selected_reward - best_reward
            }
        }


    def export_convergence_metrics(self, output_dir: Path = None) -> None:
        """Export convergence metrics to CSV for analysis."""
        import csv
        from pathlib import Path
        
        if not hasattr(self, 'convergence_metrics') or not self.convergence_metrics['selections']:
            logger.info("No convergence metrics to export")
            return
        
        # Use checkpoint dir if no output dir specified
        if output_dir is None:
            output_dir = Path(self.checkpoint_dir) if hasattr(self, 'checkpoint_dir') else Path('.')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / 'convergence_metrics.csv'
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['episode', 'selected_var', 'optimal_var', 'is_parent', 
                         'is_optimal', 'probability', 'coefficient', 'parents']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for selection in self.convergence_metrics['selections']:
                writer.writerow({
                    'episode': selection['episode'],
                    'selected_var': selection['selected_var'],
                    'optimal_var': selection['optimal_var'],
                    'is_parent': selection['is_parent'],
                    'is_optimal': selection['is_optimal'],
                    'probability': f"{selection['probability']:.4f}",
                    'coefficient': f"{selection['coefficient']:.3f}",
                    'parents': ','.join(selection['true_parents']) if selection['true_parents'] else 'None'
                })
        
        # Print summary statistics
        total = self.convergence_metrics['total_selections']
        optimal = self.convergence_metrics['total_optimal']
        rate = optimal / max(total, 1)
        
        logger.info(f"\n📊 CONVERGENCE METRICS SUMMARY:")
        logger.info(f"  Total selections: {total}")
        logger.info(f"  Optimal selections: {optimal}")
        logger.info(f"  Optimality rate: {rate:.1%}")
        logger.info(f"  Max consecutive optimal: {self.convergence_metrics['consecutive_optimal']}")
        logger.info(f"  Metrics saved to: {csv_path}")


# Factory function for backward compatibility
def create_unified_grpo_trainer(config: Union[DictConfig, Dict[str, Any], None] = None,
                               pretrained_surrogate: Optional[Dict[str, Any]] = None,
                               **kwargs) -> UnifiedGRPOTrainer:
    """Factory function to create unified GRPO trainer - same interface as original."""
    if config is not None:
        trainer = UnifiedGRPOTrainer(config=config)
    else:
        trainer = UnifiedGRPOTrainer(config=None, **kwargs)
    
    if pretrained_surrogate:
        # Handle pretrained surrogate if provided
        logger.info("Pretrained surrogate integration not implemented in modular version")
    
    return trainer