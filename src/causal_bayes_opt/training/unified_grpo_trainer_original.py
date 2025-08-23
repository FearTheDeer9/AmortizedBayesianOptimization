"""
Unified GRPO trainer combining the best of clean and simplified implementations.

This module provides a comprehensive GRPO training implementation that:
1. Uses true GRPO algorithm with batch advantages (from acquisition/grpo.py)
2. Supports flexible SCM input formats (from simplified)
3. Uses 3-channel tensor format (from clean)
4. Includes convergence detection (from simplified)
5. Integrates surrogate learning (from clean)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import haiku as hk
from omegaconf import DictConfig
import pyrsistent as pyr

# Core imports
from .three_channel_converter import buffer_to_three_channel_tensor
from .five_channel_converter import (
    buffer_to_five_channel_tensor,
    buffer_to_five_channel_tensor_with_posteriors,
    create_uniform_posterior
)
from .four_channel_converter import buffer_to_four_channel_tensor
from ..data_structures.buffer import ExperienceBuffer
from ..data_structures.sample import create_sample, get_values
from ..acquisition.better_rewards import compute_better_clean_reward, RunningStats
from ..acquisition.composite_reward import compute_composite_reward, RewardConfig
from ..acquisition.clean_rewards import compute_clean_reward
from ..acquisition.grpo_rewards import (
    compute_grpo_reward, compute_group_advantages, 
    GRPORewardComponents, analyze_reward_distribution
)
from ..policies.clean_policy_factory import create_clean_grpo_policy
from ..data_structures.scm import get_variables, get_target, get_parents
from ..mechanisms.linear import sample_from_linear_scm
from ..interventions.handlers import create_perfect_intervention
from ..environments.sampling import sample_with_intervention
from .continuous_surrogate_integration import (
    create_continuous_learnable_surrogate,
    compute_posterior_from_buffer_continuous,
    compute_structure_metrics_continuous
)

# Import true GRPO algorithm
from ..acquisition.grpo import (
    GRPOConfig, GRPOUpdate, create_grpo_trainer, _compute_grpo_loss
)

# Import convergence detection
from .convergence_detector import ConvergenceDetector, ConvergenceConfig
from .data_structures import TrainingMetrics

logger = logging.getLogger(__name__)

# Global tracking for test scripts to access (from enhanced trainer)
ENHANCED_TARGET_VALUES = []


def compute_param_change(old_params, new_params):
    """Compute magnitude of parameter changes."""
    import jax.tree_util as tree
    
    # Flatten parameters
    old_flat, _ = tree.tree_flatten(old_params)
    new_flat, _ = tree.tree_flatten(new_params)
    
    # Compute total change
    total_change = 0.0
    total_magnitude = 0.0
    
    for old_p, new_p in zip(old_flat, new_flat):
        change = jnp.sum((new_p - old_p) ** 2)
        magnitude = jnp.sum(old_p ** 2)
        total_change += change
        total_magnitude += magnitude
    
    total_change = jnp.sqrt(total_change)
    total_magnitude = jnp.sqrt(total_magnitude)
    
    # Relative change
    relative_change = total_change / (total_magnitude + 1e-8)
    
    return {
        'total': float(total_change),
        'relative': float(relative_change),
        'magnitude': float(total_magnitude)
    }


class UnifiedGRPOTrainer:
    """
    Unified GRPO trainer with true GRPO algorithm and all key features.
    
    This trainer combines:
    - True GRPO with batch advantages (not REINFORCE)
    - Flexible SCM input (list, dict, callable)
    - 3-channel tensor format
    - Convergence detection with early stopping
    - Surrogate learning integration
    - Proper reward computation
    """
    
    def __init__(self, 
                 # Can accept either DictConfig or individual parameters
                 config: Optional[Union[DictConfig, Dict[str, Any]]] = None,
                 # Individual parameters for flexibility
                 learning_rate: float = 3e-4,
                 n_episodes: int = 1000,
                 episode_length: int = 20,
                 batch_size: int = 64,  # GRPO group size
                 architecture_level: str = "simplified",
                 # Convergence
                 use_early_stopping: bool = True,
                 convergence_config: Optional[ConvergenceConfig] = None,
                 # Reward weights
                 reward_weights: Optional[Dict[str, float]] = None,
                 # Optimization
                 optimization_direction: str = "MINIMIZE",
                 # Other
                 seed: int = 42,
                 use_surrogate: bool = True,
                 checkpoint_dir: str = "checkpoints",
                 policy_architecture: Optional[str] = None,
                 **kwargs):
        """
        Initialize unified GRPO trainer.
        
        Supports both config-based and parameter-based initialization.
        """
        
        # Handle config vs parameters
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
                checkpoint_dir=checkpoint_dir,
                policy_architecture=policy_architecture,
                **kwargs
            )
        
        # Ensure convergence_config exists (for backward compatibility)
        if not hasattr(self, 'convergence_config'):
            self.convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=0.95,
                patience=5,
                min_episodes=5,
                max_episodes_per_scm=200
            )
        
        # Initialize components
        self._initialize_policy()
        self._initialize_surrogate()
        self._initialize_grpo()
        
        # Training state
        self.training_metrics = []
        self.episode_count = 0
        self.training_step = 0
        
        # Track target values for performance monitoring
        self.target_value_history = []
        self.best_target_values = {}  # Best value per SCM
        self.initial_target_values = {}  # Initial value per SCM
        
    def _init_from_config(self, config: Dict[str, Any]):
        """Initialize from config dictionary."""
        self.config = config
        self.seed = config.get('seed', 42)
        self.rng_key = random.PRNGKey(self.seed)
        
        # Extract key configurations
        self.max_episodes = config.get('max_episodes', 20)  # REDUCED FROM 1000
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
        
        # Policy architecture override (from command line)
        self.policy_architecture = config.get('policy_architecture', None)
        
        # Fixed std configuration
        self.use_fixed_std = config.get('use_fixed_std', False)
        self.fixed_std = config.get('fixed_std', 0.5)
        
        # Training config
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.optimization_direction = config.get('optimization_direction', 'MINIMIZE')
        self.use_surrogate = config.get('use_surrogate', True)
        
        # Reward weights
        self.reward_weights = config.get('reward_weights', {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.0  # Default to 0, activated when use_surrogate=True
        })
        
        # Auto-activate info gain weight when using surrogate
        if self.use_surrogate and self.reward_weights.get('info_gain', 0) == 0:
            self.reward_weights['info_gain'] = 0.3
        
        # Initialize running stats for reward normalization FIRST
        self.reward_stats = RunningStats(window_size=1000)
        
        # Initialize composite reward configuration
        self.reward_config = RewardConfig(
            target_weight=config.get('reward_weights', {}).get('target', 0.7),
            info_gain_weight=config.get('reward_weights', {}).get('info_gain', 0.2 if self.use_surrogate else 0.0),
            parent_weight=config.get('reward_weights', {}).get('parent', 0.1),
            optimization_direction=self.optimization_direction,
            reward_type=config.get('reward_type', 'continuous'),
            stats=self.reward_stats
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
        
        # Running stats already initialized earlier
        
        # Enhanced GRPO parameters (from enhanced trainer)
        # Get group_size from grpo_config if present, otherwise from root config
        grpo_config = config.get('grpo_config', {})
        self.group_size = grpo_config.get('group_size', config.get('group_size', 4))  # Number of samples per state
        self.use_grpo_rewards = config.get('use_grpo_rewards', False)  # Default to False for backward compat
        self.grpo_reward_config = config.get('grpo_reward_config', {
            'reward_weights': {
                'variable_selection': 0.5,
                'value_selection': 0.5,
                'parent_bonus': 0.3,
                'improvement_bonus': 0.2,
                'structure_discovery': 0.3 if self.use_surrogate else 0.0
            },
            'improvement_threshold': 0.1,
            'reward_type': 'squared'  # Default to squared (best from our tests)
        })
        
        # Initialize tracking for enhanced mode
        if self.use_grpo_rewards:
            self.reward_history = []
            self.gradient_history = []
            self.param_change_history = []
            self.component_reward_history = {
                'target_improvement': [],
                'parent_intervention': [],
                'value_optimization': [],
                'structure_discovery': [],
                'total': []
            }
            
    def _init_from_params(self, **kwargs):
        """Initialize from individual parameters."""
        self.learning_rate = kwargs['learning_rate']
        self.max_episodes = kwargs['n_episodes']
        self.episode_length = kwargs['episode_length']
        self.batch_size = kwargs['batch_size']
        self.architecture_level = kwargs['architecture_level']
        self.policy_architecture = kwargs.get('policy_architecture', None)
        self.optimization_direction = kwargs['optimization_direction']
        self.seed = kwargs['seed']
        self.use_surrogate = kwargs['use_surrogate']
        self.checkpoint_dir = kwargs['checkpoint_dir']
        
        self.rng_key = random.PRNGKey(self.seed)
        
        # Store policy architecture if provided
        if 'policy_architecture' in kwargs:
            self.policy_architecture = kwargs['policy_architecture']
        
        # Default ranges
        self.n_variables_range = [3, 8]
        self.obs_per_episode = 100
        self.max_interventions = self.episode_length
        
        # Architecture defaults based on level
        if self.architecture_level == "baseline":
            self.hidden_dim = 128
            self.num_layers = 2
            self.num_heads = 4
        elif self.architecture_level == "simplified":
            self.hidden_dim = 256
            self.num_layers = 4
            self.num_heads = 8
        else:  # full
            self.hidden_dim = 512
            self.num_layers = 6
            self.num_heads = 16
        
        self.key_size = 32
        self.dropout = 0.1
        
        # Reward weights
        self.reward_weights = kwargs['reward_weights'] or {
            'optimization': 0.8,
            'discovery': 0.1,
            'efficiency': 0.1,
            'info_gain': 0.0  # Default to 0, activated when use_surrogate=True
        }
        
        # Auto-activate info gain weight when using surrogate
        if self.use_surrogate and self.reward_weights.get('info_gain', 0) == 0:
            self.reward_weights['info_gain'] = 0.3
        
        # Initialize composite reward configuration
        self.reward_config = RewardConfig(
            target_weight=kwargs.get('reward_weights', {}).get('target', 0.7),
            info_gain_weight=kwargs.get('reward_weights', {}).get('info_gain', 0.2 if self.use_surrogate else 0.0),
            parent_weight=kwargs.get('reward_weights', {}).get('parent', 0.1),
            optimization_direction=self.optimization_direction,
            reward_type=kwargs.get('reward_type', 'continuous'),
            stats=self.reward_stats
        )
        
        # Convergence detection
        self.use_early_stopping = kwargs['use_early_stopping']
        
        # Always set convergence_config for SCM rotation
        conv_config = kwargs.get('convergence_config')
        if isinstance(conv_config, dict):
            # Convert dict to ConvergenceConfig, filtering out 'enabled' if present
            config_args = {k: v for k, v in conv_config.items() if k != 'enabled'}
            self.convergence_config = ConvergenceConfig(**config_args)
        elif conv_config:
            self.convergence_config = conv_config
        else:
            self.convergence_config = ConvergenceConfig(
                structure_accuracy_threshold=0.95,
                patience=5,
                min_episodes=5,
                max_episodes_per_scm=200
            )
        
        if self.use_early_stopping:
            self.convergence_detector = ConvergenceDetector(self.convergence_config)
        else:
            self.convergence_detector = None
        
        # Running stats already initialized earlier
        
        # Enhanced GRPO parameters (from enhanced trainer)
        self.group_size = kwargs.get('group_size', 4)
        self.use_grpo_rewards = kwargs.get('use_grpo_rewards', False)  # Default to False for backward compat
        self.grpo_reward_config = kwargs.get('grpo_reward_config', {
            'reward_weights': {
                'variable_selection': 0.5,
                'value_selection': 0.5,
                'parent_bonus': 0.3,
                'improvement_bonus': 0.2,
                'structure_discovery': 0.3 if self.use_surrogate else 0.0
            },
            'improvement_threshold': 0.1,
            'reward_type': 'squared'  # Default to squared (best from our tests)
        })
        
        # Initialize tracking for enhanced mode
        if self.use_grpo_rewards:
            self.reward_history = []
            self.gradient_history = []
            self.param_change_history = []
            self.component_reward_history = {
                'target_improvement': [],
                'parent_intervention': [],
                'value_optimization': [],
                'structure_discovery': [],
                'total': []
            }
            
        # Create minimal config for compatibility
        self.config = {
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'use_surrogate': self.use_surrogate,
            'checkpoint_dir': self.checkpoint_dir
        }
        
    def _initialize_policy(self):
        """Initialize policy network using shared factory."""
        logger.info("Starting _initialize_policy()...")
        
        # Determine policy architecture - default to permutation_invariant for best performance
        # Priority: explicit policy_architecture > architecture_level > default
        if hasattr(self, 'policy_architecture') and self.policy_architecture is not None:
            policy_architecture = self.policy_architecture
            logger.info(f"Using explicitly set policy_architecture: {policy_architecture}")
        elif hasattr(self, 'architecture_level') and self.architecture_level is not None:
            # Map architecture_level to policy architecture
            if self.architecture_level == "attention":
                policy_architecture = "attention"
            elif self.architecture_level == "simplified" or self.architecture_level == "simple":
                policy_architecture = "simple"  # Only use simple if explicitly requested
            else:
                # Default for any other architecture_level
                policy_architecture = "permutation_invariant"
            logger.info(f"Mapped architecture_level '{self.architecture_level}' to '{policy_architecture}'")
        else:
            # Default when nothing is specified
            policy_architecture = "permutation_invariant"
            logger.info("Using default architecture: permutation_invariant")
            
        logger.info(f"Initializing policy with architecture: {policy_architecture}")
        logger.info(f"About to call create_clean_grpo_policy...")
        
        # Log configuration source
        if policy_architecture == "permutation_invariant":
            logger.info("✓ Using permutation_invariant_alternating_policy (BEST architecture)")
        else:
            logger.warning(f"⚠ Using {policy_architecture} instead of recommended permutation_invariant")
        
        # Check if we should use fixed std
        use_fixed_std = getattr(self, 'use_fixed_std', False)
        fixed_std = getattr(self, 'fixed_std', 0.5)
        
        policy_fn = create_clean_grpo_policy(
            hidden_dim=self.hidden_dim,
            architecture=policy_architecture,
            use_fixed_std=use_fixed_std,
            fixed_std=fixed_std
        )
        logger.info(f"create_clean_grpo_policy returned with {policy_architecture} architecture")
        
        self.policy_fn = hk.transform(policy_fn)
        logger.info("hk.transform completed")
        
        # Initialize with dummy data - now 5 channels
        dummy_tensor = jnp.zeros((10, 5, 5))  # [T=10, n_vars=5, channels=5]
        self.rng_key, init_key = random.split(self.rng_key)
        logger.info("About to initialize policy params...")
        self.policy_params = self.policy_fn.init(init_key, dummy_tensor, 0)
        logger.info("Policy params initialized")
        
        logger.info(f"Initialized policy with architecture: {policy_architecture}")
        
    def _initialize_surrogate(self):
        """Initialize surrogate model for structure learning."""
        if not self.use_surrogate:
            self.surrogate_net = None
            self.surrogate_params = None
            self.surrogate_opt_state = None
            self.surrogate_predict_fn = None
            self.surrogate_update_fn = None
            logger.info("Surrogate learning disabled")
            return
            
        # Initialize learnable surrogate
        self.rng_key, surrogate_key = random.split(self.rng_key)
        
        # Determine max variables from config
        max_vars = max(self.n_variables_range) if isinstance(self.n_variables_range, list) else 10
        
        (self.surrogate_net, 
         self.surrogate_params, 
         self.surrogate_opt_state,
         self.surrogate_predict_fn,
         self.surrogate_update_fn) = create_continuous_learnable_surrogate(
            n_variables=max_vars,
            key=surrogate_key,
            learning_rate=self.config.get('surrogate_lr', 1e-3),
            hidden_dim=self.config.get('surrogate_hidden_dim', 128),
            num_layers=self.config.get('surrogate_layers', 4),
            num_heads=self.config.get('surrogate_heads', 8),
            encoder_type=self.config.get('encoder_type', 'node_feature')
        )
        
        logger.info("Initialized learnable surrogate model")
        
    def _initialize_grpo(self):
        """Initialize true GRPO trainer with batch advantages."""
        # GRPO config with proper defaults
        # Use group_size from enhanced mode if available, otherwise use batch_size
        group_size = self.group_size if hasattr(self, 'group_size') else self.batch_size
        
        # Get GRPO config parameters
        grpo_config = getattr(self, 'config', {}).get('grpo_config', {})
        ppo_epochs = grpo_config.get('ppo_epochs', 4)
        # Lower entropy coefficient for less exploration, more exploitation
        entropy_coeff = grpo_config.get('entropy_coefficient', 0.001 if self.use_grpo_rewards else 0.01)
        clip_ratio = grpo_config.get('clip_ratio', 0.2)
        gradient_clip = grpo_config.get('gradient_clip', 1.0)
        
        self.grpo_config = GRPOConfig(
            group_size=group_size,
            interventions_per_state=1,
            learning_rate=self.learning_rate,
            clip_ratio=clip_ratio,
            entropy_coeff=entropy_coeff,
            max_grad_norm=gradient_clip,
            ppo_epochs=ppo_epochs
        )
        
        # Create optimizer with gradient clipping and learning rate schedule
        optimizer_chain = []
        if gradient_clip is not None and gradient_clip > 0:
            optimizer_chain.append(optax.clip_by_global_norm(gradient_clip))
        
        # Add cosine decay schedule for better exploration/exploitation balance
        # Decay from initial LR to 10% over training
        total_steps = getattr(self, 'max_episodes', 1000) * 10  # Approximate steps
        lr_schedule = optax.cosine_decay_schedule(
            init_value=self.learning_rate,
            decay_steps=total_steps,
            alpha=0.1  # Final LR will be 10% of initial
        )
        optimizer_chain.append(optax.adam(learning_rate=lr_schedule))
        self.optimizer = optax.chain(*optimizer_chain)
        
        # Initialize optimizer state
        self.optimizer_state = self.optimizer.init(self.policy_params)
        
        # Create custom GRPO update function
        def grpo_update(params, opt_state, batch):
            """GRPO update with PPO-style multiple epochs."""
            # Store metrics from all epochs
            all_losses = []
            all_grad_norms = []
            all_approx_kls = []
            
            current_params = params
            current_opt_state = opt_state
            
            # Perform multiple PPO epochs on the same batch
            for epoch in range(ppo_epochs):
                # Compute loss and gradients
                def loss_fn(p):
                    return self._compute_simple_grpo_loss(p, batch)
                
                (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
                
                # Apply updates
                updates, current_opt_state = self.optimizer.update(grads, current_opt_state, current_params)
                current_params = optax.apply_updates(current_params, updates)
                
                # Track metrics
                all_losses.append(loss_value)
                all_grad_norms.append(optax.global_norm(grads))
                all_approx_kls.append(loss_info['approx_kl'])
                
                # Early stopping if KL divergence too large (PPO best practice)
                if loss_info['approx_kl'] > 0.02:  # Standard PPO threshold
                    logger.debug(f"Early stopping at epoch {epoch+1}/{ppo_epochs} due to KL={loss_info['approx_kl']:.4f}")
                    break
            
            # Use metrics from final epoch for reporting
            return current_params, current_opt_state, GRPOUpdate(
                policy_loss=loss_info['policy_loss'],
                entropy_loss=loss_info['entropy_loss'],
                kl_penalty=loss_info['kl_penalty'],
                total_loss=loss_value,
                grad_norm=all_grad_norms[-1],
                group_baseline=loss_info['group_baseline'],
                mean_reward=loss_info['mean_reward'],
                reward_std=loss_info['reward_std'],
                mean_advantage=loss_info['mean_advantage'],
                advantage_std=loss_info['advantage_std'],
                mean_entropy=loss_info['mean_entropy'],
                approx_kl=all_approx_kls[-1],
                mean_ratio=loss_info.get('mean_ratio', 1.0),
                ratio_std=loss_info.get('ratio_std', 0.0)
            )
        
        # Use enhanced update function if in enhanced mode
        if self.use_grpo_rewards:
            self._create_enhanced_grpo_update()
        else:
            self.grpo_update = grpo_update
        
        logger.info(f"Initialized {'enhanced' if self.use_grpo_rewards else 'standard'} GRPO with group_size={group_size}, entropy_coeff={self.grpo_config.entropy_coeff}")
    
    def _create_enhanced_grpo_update(self):
        """Create enhanced GRPO update function with PPO epochs and debugging."""
        from types import SimpleNamespace
        import jax
        import optax
        
        # Get PPO epochs from config
        grpo_config = getattr(self, 'config', {}).get('grpo_config', {})
        ppo_epochs = grpo_config.get('ppo_epochs', 4)
        
        def enhanced_grpo_update(params, opt_state, batch):
            """Enhanced GRPO update with PPO-style multiple epochs."""
            # Store metrics from all epochs
            all_losses = []
            all_grad_norms = []
            all_approx_kls = []
            all_loss_infos = []
            
            current_params = params
            current_opt_state = opt_state
            
            # Perform multiple PPO epochs on the same batch
            for epoch in range(ppo_epochs):
                # Compute loss and gradients with our enhanced loss function
                def loss_fn(p):
                    return self._compute_simple_grpo_loss(p, batch)
                
                (loss_value, loss_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
                
                # Apply updates
                updates, current_opt_state = self.optimizer.update(grads, current_opt_state, current_params)
                current_params = optax.apply_updates(current_params, updates)
                
                # Track metrics
                all_losses.append(loss_value)
                all_grad_norms.append(optax.global_norm(grads))
                all_approx_kls.append(loss_info['approx_kl'])
                all_loss_infos.append(loss_info)
                
                # Early stopping if KL divergence too large (PPO best practice)
                if loss_info['approx_kl'] > 0.02:  # Standard PPO threshold
                    logger.debug(f"Enhanced: Early stopping at epoch {epoch+1}/{ppo_epochs} due to KL={loss_info['approx_kl']:.4f}")
                    break
            
            # Use metrics from final epoch for reporting
            final_loss_info = all_loss_infos[-1]
            
            # Create enhanced metrics with all debugging info from final_loss_info
            enhanced_metrics = SimpleNamespace(
                # Standard fields
                policy_loss=final_loss_info['policy_loss'],
                entropy_loss=final_loss_info['entropy_loss'],
                kl_penalty=final_loss_info['kl_penalty'],
                total_loss=all_losses[-1],
                grad_norm=all_grad_norms[-1],
                group_baseline=final_loss_info['group_baseline'],
                mean_reward=final_loss_info['mean_reward'],
                reward_std=final_loss_info['reward_std'],
                mean_advantage=final_loss_info['mean_advantage'],
                advantage_std=final_loss_info['advantage_std'],
                mean_entropy=final_loss_info['mean_entropy'],
                approx_kl=all_approx_kls[-1],
                # Enhanced debugging fields
                mean_ratio=final_loss_info.get('mean_ratio', 1.0),
                ratio_std=final_loss_info.get('ratio_std', 0.0),
                mean_log_prob_change=final_loss_info.get('mean_log_prob_change', 0.0),
                surr1_mean=final_loss_info.get('surr1_mean', 0.0),
                surr2_mean=final_loss_info.get('surr2_mean', 0.0),
                surr_min_mean=final_loss_info.get('surr_min_mean', 0.0),
                clip_fraction=final_loss_info.get('clip_fraction', 0.0),
                positive_advantages=final_loss_info.get('positive_advantages', 0),
                negative_advantages=final_loss_info.get('negative_advantages', 0),
                # Diagnostic fields
                loss_terms_sum=final_loss_info.get('loss_terms_sum', 0.0),
                loss_terms_mean=final_loss_info.get('loss_terms_mean', 0.0),
                log_prob_variance=final_loss_info.get('log_prob_variance', 0.0),
                unique_log_probs=final_loss_info.get('unique_log_probs', 0),
                # PPO epoch tracking
                ppo_epochs_completed=len(all_losses)
            )
            
            return current_params, current_opt_state, enhanced_metrics
        
        self.grpo_update = enhanced_grpo_update
        
    def _override_surrogate(self, pretrained_components: Dict[str, Any]):
        """Override surrogate with pre-trained components."""
        if 'net' in pretrained_components and 'params' in pretrained_components:
            self.surrogate_net = pretrained_components['net']
            self.surrogate_params = pretrained_components['params']
            
            # Recreate prediction and update functions with pre-trained model
            from .continuous_surrogate_integration import create_surrogate_fn_wrapper, create_update_fn_wrapper
            self.surrogate_predict_fn = create_surrogate_fn_wrapper(
                self.surrogate_net, 
                self.surrogate_params
            )
            
            # Create optimizer for the pre-trained model
            optimizer = optax.adam(learning_rate=self.config.get('surrogate_lr', 1e-3))
            self.surrogate_opt_state = optimizer.init(self.surrogate_params)
            
            # Create update function
            self.surrogate_update_fn = create_update_fn_wrapper(
                self.surrogate_net,
                optimizer
            )
            
            logger.info("Successfully overrode surrogate with pre-trained model")
        else:
            logger.warning("Pre-trained components missing 'net' or 'params', using fresh surrogate")
        
    def train(self, 
              scms: Union[List[Any], Dict[str, Any], Callable[[], Any]],
              eval_scms: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Train GRPO policy on provided SCMs.
        
        Args:
            scms: Training SCMs - can be:
                - List of SCMs to rotate through
                - Dict mapping names to SCMs
                - Callable that generates SCMs on demand
                - SCMCurriculumFactory for curriculum learning
            eval_scms: Optional separate evaluation set
            
        Returns:
            Training results and metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("Starting unified GRPO training with true GRPO algorithm")
        logger.info(f"  Episodes: {self.max_episodes}")
        logger.info(f"  Use surrogate: {self.use_surrogate}")
        logger.info(f"  Reward weights: {self.reward_weights}")
        if self.use_surrogate:
            logger.info(f"  Surrogate loaded: {self.surrogate_params is not None}")
            logger.info(f"  Surrogate predict_fn available: {self.surrogate_predict_fn is not None}")
            logger.info(f"  Surrogate update_fn available: {self.surrogate_update_fn is not None}")
        logger.info(f"{'='*70}\n")
        start_time = time.time()
        
        # Check if using curriculum factory
        from .curriculum_factory import SCMCurriculumFactory
        is_curriculum = isinstance(scms, SCMCurriculumFactory)
        
        # Convert SCMs to standard format or use curriculum
        if is_curriculum:
            logger.info("Using curriculum learning factory")
            curriculum_factory = scms
            scm_rotation = None  # Not used in curriculum mode
            scm_episodes = {}
        else:
            scm_rotation = self._prepare_scms(scms)
            logger.info(f"Starting training with {len(scm_rotation)} SCMs")
            scm_episodes = {name: 0 for name, _ in scm_rotation}
            current_scm_idx = 0
        
        # Training history
        episode_metrics = []
        embedding_stats = []  # Track embedding statistics
        self._embedding_stats_temp = embedding_stats  # Make accessible to episode method
        curriculum_info = []  # Track curriculum progression
        
        # Main training loop
        for episode in range(self.max_episodes):
            # Get current SCM
            if is_curriculum:
                # Get performance metrics from last episode
                perf_metrics = episode_metrics[-1] if episode_metrics else None
                
                # Get next SCM from curriculum
                scm, curr_metadata = curriculum_factory.get_next_scm(perf_metrics)
                scm_name = f"level_{curr_metadata['curriculum_level']}_{curr_metadata['curriculum_stage']}"
                
                # Track curriculum info
                curriculum_info.append(curr_metadata)
                
                # Log curriculum progress
                if episode % 10 == 0:
                    logger.info(f"Curriculum: Level {curr_metadata['curriculum_level']} "
                              f"({curr_metadata['curriculum_stage']}), "
                              f"Episodes at level: {curr_metadata['episodes_at_level']}")
            else:
                # Original rotation logic
                scm_name, scm = scm_rotation[current_scm_idx]
                scm_episodes[scm_name] += 1
                
                # Check for rotation based on max_episodes_per_scm
                if scm_episodes[scm_name] >= self.convergence_config.max_episodes_per_scm:
                    logger.info(f"SCM {scm_name} reached max episodes ({scm_episodes[scm_name]})")
                    # Rotate to next SCM
                    current_scm_idx = (current_scm_idx + 1) % len(scm_rotation)
            
            # Run episode with GRPO batch collection
            self.rng_key, episode_key = random.split(self.rng_key)
            metrics = self._run_grpo_episode(episode, scm, scm_name, episode_key)
            episode_metrics.append(metrics)
            
            # Check convergence if enabled
            if self.convergence_detector:
                # Create a simple dataclass-like object for convergence detector
                from types import SimpleNamespace
                training_metrics = SimpleNamespace(
                    episode=metrics['episode'],
                    mean_reward=metrics['mean_reward'],
                    structure_accuracy=metrics.get('structure_metrics', {}).get('f1_score', 0.0),
                    optimization_improvement=0.0,  # Not tracked in this implementation
                    policy_loss=metrics.get('loss', 0.0),
                    value_loss=0.0,  # Not tracked separately
                    scm_type=metrics.get('scm_type', scm_name),
                    f1_score=metrics.get('structure_metrics', {}).get('f1_score', None),
                    true_parent_likelihood=metrics.get('structure_metrics', {}).get('parent_likelihood', None),
                    shd=metrics.get('structure_metrics', {}).get('shd', None),
                    marginal_probs=metrics.get('structure_metrics', {}).get('marginal_probs', None)
                )
                self.convergence_detector.update(scm_name, training_metrics)
                converged, reason = self.convergence_detector.check_convergence(scm_name)
                
                if converged:
                    logger.info(f"SCM {scm_name} converged after {scm_episodes[scm_name]} episodes: {reason}")
                    
                    # Check if all SCMs have converged
                    all_converged = all(
                        self.convergence_detector.scm_states.get(name, None) and 
                        self.convergence_detector.scm_states[name].converged
                        for name, _ in scm_rotation
                    )
                    
                    if all_converged:
                        logger.info(f"All SCMs converged! Stopping early at episode {episode}")
                        break
            
            # Log progress
            if episode % 10 == 0:
                recent_rewards = [m['mean_reward'] for m in episode_metrics[-10:]]
                mean_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                logger.info(f"Episode {episode}: mean_reward={mean_reward:.4f}, "
                          f"current_scm={scm_name}")
                
                # Report target value improvement
                if scm_name in self.best_target_values:
                    improvement = self.best_target_values[scm_name] - self.initial_target_values[scm_name]
                    logger.info(f"  Target value - Initial: {self.initial_target_values[scm_name]:.3f}, "
                               f"Best: {self.best_target_values[scm_name]:.3f}, "
                               f"Improvement: {improvement:.3f}")
                
                # Track reward trend
                if episode > 0 and episode % 10 == 0:
                    early_rewards = [m['mean_reward'] for m in episode_metrics[:min(10, len(episode_metrics))]]
                    recent_rewards = [m['mean_reward'] for m in episode_metrics[-10:]]
                    early_avg = sum(early_rewards) / len(early_rewards) if early_rewards else 0
                    recent_avg = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
                    trend = recent_avg - early_avg
                    logger.info(
                        f"[REWARD TREND] Episode {episode}: "
                        f"Early avg: {early_avg:.3f}, Recent avg: {recent_avg:.3f}, "
                        f"Trend: {trend:+.3f} ({'↑' if trend > 0 else '↓' if trend < 0 else '→'})"
                    )
                
                # Log GRPO-specific metrics if available
                if 'grpo_metrics' in metrics:
                    gm = metrics['grpo_metrics']
                    logger.info(f"  GRPO: advantage={gm.get('mean_advantage', 0):.3f}, "
                              f"baseline={gm.get('group_baseline', 0):.3f}")
        
        # Save final checkpoint
        self._save_checkpoint(is_final=True)
        
        # Log final target value summary
        if self.target_value_history:
            logger.info("\n" + "="*70)
            logger.info("TARGET VALUE IMPROVEMENT SUMMARY")
            logger.info("="*70)
            
            # Calculate overall statistics
            total_improvements = []
            for scm_name in self.best_target_values:
                if scm_name in self.initial_target_values:
                    improvement = self.best_target_values[scm_name] - self.initial_target_values[scm_name]
                    total_improvements.append(improvement)
                    logger.info(f"\n{scm_name}:")
                    logger.info(f"  Initial target value: {self.initial_target_values[scm_name]:.4f}")
                    logger.info(f"  Best target value:    {self.best_target_values[scm_name]:.4f}")
                    # Handle division by zero for percentage calculation
                    if abs(self.initial_target_values[scm_name]) > 1e-6:
                        pct_improvement = abs(improvement/self.initial_target_values[scm_name]*100)
                        logger.info(f"  Improvement:          {improvement:.4f} ({pct_improvement:.1f}%)")
                    else:
                        logger.info(f"  Improvement:          {improvement:.4f} (from zero baseline)")
            
            if total_improvements:
                avg_improvement = sum(total_improvements) / len(total_improvements)
                logger.info(f"\nOverall Statistics:")
                logger.info(f"  Average improvement: {avg_improvement:.4f}")
                logger.info(f"  Best improvement:    {min(total_improvements):.4f}")  # Min because we minimize
                logger.info(f"  Worst improvement:   {max(total_improvements):.4f}")
                
                # Track how many SCMs showed improvement
                improved_count = sum(1 for imp in total_improvements if imp < 0)
                logger.info(f"  SCMs improved:       {improved_count}/{len(total_improvements)} ({improved_count/len(total_improvements)*100:.1f}%)")
            
            # Plot target value progression over episodes
            if len(self.target_value_history) > 10:
                episodes = [entry['episode'] for entry in self.target_value_history]
                values = [entry['target_value'] for entry in self.target_value_history]
                
                # Group by episode and average
                from collections import defaultdict
                episode_values = defaultdict(list)
                for ep, val in zip(episodes, values):
                    episode_values[ep].append(val)
                
                # Calculate running average
                sorted_episodes = sorted(episode_values.keys())
                running_avg = []
                window = []
                window_size = 10
                
                for ep in sorted_episodes:
                    window.extend(episode_values[ep])
                    if len(window) > window_size * 8:  # 8 samples per episode (group_size)
                        window = window[-window_size * 8:]
                    running_avg.append(sum(window) / len(window))
                
                # Log trend
                if len(running_avg) > 1:
                    trend = "improving" if running_avg[-1] < running_avg[0] else "worsening"
                    logger.info(f"\nTarget Value Trend: {trend}")
                    logger.info(f"  Start avg: {running_avg[0]:.4f}")
                    logger.info(f"  End avg:   {running_avg[-1]:.4f}")
                    logger.info(f"  Change:    {running_avg[-1] - running_avg[0]:.4f}")
            
            logger.info("="*70 + "\n")
        
        # Log probability evolution summary
        if hasattr(self, 'probability_history') and self.probability_history:
            logger.info("\n" + "="*70)
            logger.info("PROBABILITY EVOLUTION SUMMARY")
            logger.info("="*70)
            
            # Group by SCM
            from collections import defaultdict
            scm_probs = defaultdict(list)
            for record in self.probability_history:
                scm_probs[record['scm']].append(record)
            
            for scm_name, records in scm_probs.items():
                if len(records) > 1:
                    logger.info(f"\n{scm_name}:")
                    first = records[0]
                    last = records[-1]
                    logger.info(f"  Episodes: {first['episode']} -> {last['episode']}")
                    logger.info(f"  Max-Min diff: {first['max_min']:.4f} -> {last['max_min']:.4f}")
                    logger.info(f"  Entropy: {first['entropy']:.3f} -> {last['entropy']:.3f}")
                    
                    # Check for sudden changes
                    sudden_changes = []
                    for i in range(1, len(records)):
                        change = records[i]['max_min'] - records[i-1]['max_min']
                        if abs(change) > 0.5:
                            sudden_changes.append((records[i]['episode'], change))
                    
                    if sudden_changes:
                        logger.info(f"  SUDDEN CHANGES detected at episodes: {sudden_changes}")
        
        # Prepare results
        training_time = time.time() - start_time
        final_metrics = episode_metrics[-1] if episode_metrics else {}
        
        # Add target value metrics to final metrics
        if self.target_value_history:
            final_metrics['target_value_metrics'] = {
                'best_values': self.best_target_values,
                'initial_values': self.initial_target_values,
                'improvements': {
                    scm: self.best_target_values[scm] - self.initial_target_values[scm]
                    for scm in self.best_target_values 
                    if scm in self.initial_target_values
                },
                'history_length': len(self.target_value_history)
            }
        
        results = {
            'training_time': training_time,
            'final_metrics': final_metrics,
            'all_metrics': episode_metrics,
            'policy_params': self.policy_params,
            'embedding_stats': embedding_stats,
            'target_value_history': self.target_value_history
        }
        
        # Add curriculum-specific results
        if is_curriculum:
            results['curriculum_info'] = curriculum_info
            results['curriculum_summary'] = curriculum_factory.get_summary()
            results['final_level'] = curriculum_info[-1]['curriculum_level'] if curriculum_info else 1
        else:
            results['episodes_per_scm'] = scm_episodes
            results['converged'] = all(
                self.convergence_detector.scm_states.get(name, None) and 
                self.convergence_detector.scm_states[name].converged
                for name, _ in scm_rotation
            ) if self.convergence_detector and scm_rotation else False
            
        return results
    
    def _run_grpo_episode(self, episode_idx: int, scm: pyr.PMap, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Run GRPO episode - dispatches to enhanced or standard version based on use_grpo_rewards."""
        if self.use_grpo_rewards:
            # For now, use standard episode with enhanced rewards enabled
            # TODO: Add full enhanced episode implementation
            return self._run_standard_grpo_episode(episode_idx, scm, scm_name, key)
        else:
            return self._run_standard_grpo_episode(episode_idx, scm, scm_name, key)
    
    def _run_standard_grpo_episode(self, episode_idx: int, scm: pyr.PMap, scm_name: str, key: jax.random.PRNGKey) -> Dict[str, Any]:
        """Run single training episode with GRPO batch collection."""
        # Get SCM info
        variables = list(get_variables(scm))
        target_var = get_target(scm)
        target_idx = variables.index(target_var)
        true_parents = list(get_parents(scm, target_var)) if hasattr(scm, 'edges') else []
        
        # Track training progress
        if episode_idx % 10 == 0:  # Less frequent logging
            logger.info(f"\n{'='*70}")
            logger.info(f"EPISODE {episode_idx} - {scm_name}")
            logger.info(f"Target: {target_var}, True parents: {true_parents}")
            logger.info(f"Training step: {self.training_step}")
        
        # Initialize probability tracking if not exists
        if not hasattr(self, 'probability_history'):
            self.probability_history = []
        
        # Initialize buffer with observational data
        buffer = ExperienceBuffer()
        key, obs_key = random.split(key)
        
        # Sample observational data
        obs_samples = sample_from_linear_scm(scm, self.obs_per_episode, seed=int(obs_key[0]))
        
        # Compute initial posterior if we have a surrogate, otherwise use uniform
        initial_posterior = None
        if self.use_surrogate and self.surrogate_predict_fn is not None:
            # Create initial tensor from observations to get posterior
            temp_buffer = ExperienceBuffer()
            for sample in obs_samples:
                temp_buffer.add_observation(sample)
            initial_tensor, initial_mapper = buffer_to_three_channel_tensor(
                temp_buffer, target_var, max_history_size=100, standardize=False
            )
            initial_posterior = self.surrogate_predict_fn(initial_tensor, target_var, initial_mapper.variables)
        else:
            # No surrogate - use uniform posterior
            from ..training.five_channel_converter import create_uniform_posterior
            initial_posterior = create_uniform_posterior(variables, target_var)
        
        # Add observations with the computed posterior
        for sample in obs_samples:
            buffer.add_observation(sample, posterior=initial_posterior)
        
        # MULTIPLE INTERVENTIONS PER EPISODE
        for intervention_idx in range(self.max_interventions):
            logger.info(f"\n--- Intervention {intervention_idx+1}/{self.max_interventions} ---")
            
            # Collect GRPO batch for this intervention
            grpo_batch_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'old_log_probs': [],
                'target_idx': None,  # Will be set from mapper
                'intervention_details': []  # Store full intervention info for best selection
            }
            
            # Generate batch of candidates for this intervention
            for step in range(self.grpo_config.group_size):
                key, step_key = random.split(key)
            
            # Get posterior before intervention for info gain calculation
            posterior_before = None
            
            # Convert buffer to 5-channel tensor using stored posteriors
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                # Use the new function that uses stored posteriors
                from ..training.five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
                tensor, mapper, diagnostics = buffer_to_five_channel_tensor_with_posteriors(
                    buffer, target_var, 
                    max_history_size=100, 
                    standardize=False,
                    use_uniform_for_missing=True
                )
                
                # Log surrogate diagnostics
                logger.debug(
                    f"Tensor conversion with stored posteriors: "
                    f"n_samples={diagnostics['n_samples']}, "
                    f"n_with_posteriors={diagnostics['n_with_posteriors']}"
                )
                
                # Compute current posterior for this step
                # (This will be stored with the intervention samples)
                tensor_3ch = tensor[:, :, :3]  # Extract first 3 channels for surrogate
                posterior_before = self.surrogate_predict_fn(tensor_3ch, target_var, mapper.variables)
            else:
                # No surrogate - use buffer_to_five_channel_tensor with None surrogate
                tensor, mapper, diagnostics = buffer_to_five_channel_tensor(
                    buffer, target_var, 
                    surrogate_fn=None,  # No surrogate function
                    max_history_size=100, 
                    standardize=False
                )
                posterior_before = None
            
            # Monitor embedding statistics
            if episode_idx % 10 == 0 and step == 0:  # Check first sample every 10 episodes
                # Analyze tensor values (channel 0 - the actual values)
                value_channel = tensor[:, :, 0]
                # Compute statistics across variables and time
                var_stds = jnp.std(value_channel, axis=0)  # Std per variable across time
                time_stds = jnp.std(value_channel, axis=1)  # Std per timestep across variables
                overall_std = jnp.std(value_channel)
                
                embedding_stat = {
                    'episode': episode_idx,
                    'mean_std': float(jnp.mean(var_stds)),
                    'max_std': float(jnp.max(var_stds)),
                    'min_std': float(jnp.min(var_stds)),
                    'overall_std': float(overall_std),
                    'mean_time_std': float(jnp.mean(time_stds))
                }
                # Only append if embedding_stats exists (when called from train())
                if hasattr(self, '_embedding_stats_temp'):
                    self._embedding_stats_temp.append(embedding_stat)
                
                # Log warning if embeddings are too uniform
                if overall_std < 0.01:
                    logger.warning(
                        f"[EMBEDDING WARNING] Episode {episode_idx}: Very low variance in embeddings! "
                        f"Overall std: {overall_std:.6f}, Mean var std: {jnp.mean(var_stds):.6f}"
                    )
                elif episode_idx % 50 == 0:
                    logger.info(
                        f"[EMBEDDING STATS] Episode {episode_idx}: "
                        f"Overall std: {overall_std:.4f}, "
                        f"Mean var std: {jnp.mean(var_stds):.4f}, "
                        f"Range: [{jnp.min(value_channel):.2f}, {jnp.max(value_channel):.2f}]"
                    )
            
            # Get policy output with 5-channel input
            key, policy_key = random.split(key)
            
            # DEBUG: Analyze tensor before policy call
            if episode_idx % 10 == 0 and step == 0:
                # Get raw values from buffer to compare
                all_samples = buffer.get_all_samples()
                if all_samples:
                    last_sample = all_samples[-1]
                    raw_values = get_values(last_sample)
                    logger.info(f"\n[RAW VALUES] Episode {episode_idx}:")
                    logger.info(f"  Last sample raw values: {raw_values}")
                    logger.info(f"  Variables in buffer: {sorted(raw_values.keys())}")
                    raw_vals_list = list(raw_values.values())
                    logger.info(f"  Raw value range: [{min(raw_vals_list):.3f}, {max(raw_vals_list):.3f}]")
                    logger.info(f"  Raw value std: {jnp.std(jnp.array(raw_vals_list)):.3f}")
                
                logger.info(f"\n[TENSOR DEBUG] Episode {episode_idx}, Step {step}:")
                logger.info(f"  Tensor shape: {tensor.shape}")
                logger.info(f"  Mapper variables: {mapper.variables}")
                logger.info(f"  Target: {target_var} (idx: {mapper.target_idx})")
                
                # Analyze each channel
                for ch_idx, ch_name in enumerate(['Values', 'Target', 'Interv', 'Probs', 'Recency']):
                    channel = tensor[:, :, ch_idx]
                    logger.info(f"  Channel {ch_idx} ({ch_name}):")
                    logger.info(f"    Mean: {jnp.mean(channel):.3f}, Std: {jnp.std(channel):.3f}")
                    logger.info(f"    Last timestep: {channel[-1]}")
                
                # Check if variables look different
                value_channel = tensor[:, :, 0]
                var_means = jnp.mean(value_channel, axis=0)  # Mean per variable
                logger.info(f"  Variable value means: {var_means}")
                logger.info(f"  Variable value std between vars: {jnp.std(var_means):.3f}")
            
            # CRITICAL: Use mapper's target index, not original SCM index!
            # The mapper sorts variables, so indices change
            policy_output = self.policy_fn.apply(
                self.policy_params, policy_key, tensor, mapper.target_idx
            )
            
            # Sample intervention
            var_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            # Sample variable with exploration noise
            key, var_key = random.split(key)
            
            # Add exploration noise to enable trying different variables
            exploration_noise = 0.1  # Reduced noise for better exploitation
            noisy_logits = var_logits + exploration_noise * random.normal(var_key, var_logits.shape)
            
            # Sample from noisy logits
            selected_var_idx = random.categorical(var_key, noisy_logits)
            
            # Compute probabilities from original logits for GRPO loss
            var_probs = jax.nn.softmax(var_logits)
            
            # PRINCIPLED LOGGING: Track probabilities and input data
            if step == 0:  # Log once per GRPO batch
                # Log probabilities for each variable
                prob_info = []
                for i, var_name in enumerate(mapper.variables):
                    if i != mapper.target_idx:
                        prob_info.append(f"{var_name}:{var_probs[i]:.3f}")
                logger.info(f"  Variable probabilities: {' '.join(prob_info)}")
                
                # Compute differentiation metrics
                valid_probs = var_probs[var_probs > 0]
                prob_std = float(jnp.std(valid_probs))
                max_min_diff = float(jnp.max(valid_probs) - jnp.min(valid_probs))
                entropy = -float(jnp.sum(valid_probs * jnp.log(valid_probs + 1e-8)))
                
                logger.info(f"  Prob metrics - Std: {prob_std:.4f}, Max-Min: {max_min_diff:.4f}, Entropy: {entropy:.3f}")
                
                # Store probability history for tracking changes
                prob_record = {
                    'episode': episode_idx,
                    'scm': scm_name,
                    'target': target_var,
                    'probs': {var_name: float(var_probs[i]) for i, var_name in enumerate(mapper.variables) if i != mapper.target_idx},
                    'std': prob_std,
                    'max_min': max_min_diff,
                    'entropy': entropy
                }
                self.probability_history.append(prob_record)
                
                # Check for sudden changes in probabilities
                if len(self.probability_history) > 1:
                    prev_record = self.probability_history[-2]
                    if prev_record['scm'] == scm_name:  # Same SCM
                        prev_max_min = prev_record['max_min']
                        change = max_min_diff - prev_max_min
                        if abs(change) > 0.5:  # Sudden large change
                            logger.warning(f"  SUDDEN PROBABILITY CHANGE: {prev_max_min:.4f} -> {max_min_diff:.4f} (change: {change:+.4f})")
                
                # Log input tensor info for consistency check
                logger.info(f"  Input tensor shape: {tensor.shape}")
                logger.info(f"  Buffer size: {buffer.size()}")
                
                # Check consistency: tensor should grow by 1 sample each intervention
                if hasattr(self, '_last_buffer_size'):
                    size_diff = buffer.size() - self._last_buffer_size
                    if size_diff != 1 and step > 0:
                        logger.warning(f"  WARNING: Buffer size changed by {size_diff} (expected 1)")
                self._last_buffer_size = buffer.size()
            
            # DEBUG: Print detailed info about variable selection
            if self.training_step % 10 == 0 and step == 0:  # First step every 10 training steps
                logger.info(f"\n[DEBUG] Episode {episode_idx}, Step {step}:")
                logger.info(f"  SCM: {scm_name}, Target: {target_var}")
                logger.info(f"  Variable mapping: {mapper.name_to_idx}")
                logger.info(f"  Target index: {mapper.target_idx}")
                logger.info(f"  Logits: {var_logits}")
                logger.info(f"  Probs: {var_probs}")
                
                # Track probability differentiation
                probs_array = jnp.array(var_probs)
                # Exclude target (which is masked as 0)
                valid_probs = probs_array[probs_array > 0]
                prob_std = jnp.std(valid_probs) if len(valid_probs) > 1 else 0.0
                prob_max_diff = jnp.max(valid_probs) - jnp.min(valid_probs) if len(valid_probs) > 1 else 0.0
                logger.info(f"  Prob differentiation - Std: {prob_std:.4f}, Max-Min: {prob_max_diff:.4f}")
                
                logger.info(f"  Selected idx: {selected_var_idx}")
                selected_var_name = mapper.get_name(int(selected_var_idx))
                logger.info(f"  Selected var: {selected_var_name}")
                if selected_var_name == target_var:
                    logger.warning(f"  WARNING: Selected target variable!")
                
                # Check tensor values
                current_values = tensor[0, :, 0]  # Most recent values
                logger.info(f"  Current values:")
                for var, idx in mapper.name_to_idx.items():
                    logger.info(f"    {var}: {current_values[idx]:.3f}")
                
                # Check if embeddings are distinguishable
                embeddings_std = float(jnp.std(current_values))
                logger.info(f"  Embedding std: {embeddings_std:.3f}")
                
                # Analyze why this variable was chosen
                logit_values = [float(var_logits[i]) if float(var_logits[i]) != float('-inf') else -999 for i in range(len(var_logits))]
                max_other_logit = max(logit_values[i] for i in range(len(logit_values)) if i != int(selected_var_idx))
                logit_diff = logit_values[int(selected_var_idx)] - max_other_logit
                logger.info(f"  Logit advantage of selected var: {logit_diff:.3f}")
                
                # Check relationship to target
                if scm_name == 'fork' and target_var == 'Y':
                    logger.info(f"  Fork SCM: X,Z are parents of Y")
                elif scm_name == 'chain' and target_var == 'X2':
                    logger.info(f"  Chain SCM: X0->X1->X2")
                elif scm_name == 'collider' and target_var == 'Z':
                    logger.info(f"  Collider SCM: X,Y are parents of Z")
                
                # Check if we selected a parent
                is_parent = selected_var_name in true_parents
                logger.info(f"  Selected variable is parent: {is_parent}")
                if not is_parent and len(true_parents) > 0:
                    logger.warning(f"  WARNING: Selected non-parent {selected_var_name}, true parents are {true_parents}")
            
            # Track discrimination metrics if requested
            if hasattr(self, '_track_discrimination') and self._track_discrimination and step == 0:
                # Compute logit spread (excluding masked values)
                valid_logits = var_logits[var_logits != -jnp.inf]
                logit_spread = float(jnp.std(valid_logits)) if len(valid_logits) > 1 else 0.0
                
                # Compute embedding variance
                embedding_variance = float(jnp.var(tensor[0, :, 0]))  # Variance of current values
                
                # Check if selected variable is a direct parent
                is_direct_parent = mapper.get_name(int(selected_var_idx)) in true_parents
                
                # Get logit for each parent vs non-parent
                parent_logits = []
                non_parent_logits = []
                for var_name, var_idx in mapper.name_to_idx.items():
                    if var_name != target_var:  # Exclude target
                        logit_val = float(var_logits[var_idx])
                        if var_name in true_parents:
                            parent_logits.append(logit_val)
                        else:
                            non_parent_logits.append(logit_val)
                
                # Compute average logit advantage for parents
                parent_logit_advantage = 0.0
                if parent_logits and non_parent_logits:
                    parent_logit_advantage = np.mean(parent_logits) - np.mean(non_parent_logits)
                
                # Store discrimination metric
                from scripts.test_grpo_with_logged_values import DISCRIMINATION_METRICS
                DISCRIMINATION_METRICS.append({
                    'episode': episode_idx,
                    'scm': scm_name,
                    'logit_spread': logit_spread,
                    'embedding_variance': embedding_variance,
                    'buffer_size': buffer.size(),
                    'selected_var': mapper.get_name(int(selected_var_idx)),
                    'target_var': target_var,
                    'is_direct_parent': is_direct_parent,
                    'parent_logit_advantage': parent_logit_advantage,
                    'true_parents': true_parents
                })
            
            # Log exploration behavior and check for uniform logits
            if self.training_step % 50 == 0:
                logit_std = float(jnp.std(var_logits[var_logits != -jnp.inf]))
                logger.debug(
                    f"Variable selection - logits: {var_logits}, "
                    f"selected: {selected_var_idx}, "
                    f"probs: {var_probs}, "
                    f"logit_std: {logit_std:.4f}"
                )
                
                # Warn if logits are too uniform (excluding masked values)
                if logit_std < 0.1:
                    logger.warning(
                        f"[LOGIT WARNING] Low variance in logits: std={logit_std:.4f}, "
                        f"may indicate poor discrimination between variables"
                    )
            
            # Compute log probability for GRPO
            log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
            
            # Sample value
            key, val_key = random.split(key)
            mean = value_params[selected_var_idx, 0]
            log_std = value_params[selected_var_idx, 1]
            std = jnp.exp(log_std)
            intervention_value = mean + std * random.normal(val_key)
            
            # Create and apply intervention
            selected_var = mapper.get_name(int(selected_var_idx))
            
            # DEBUG: Log intervention details
            if self.training_step % 10 == 0 and step == 0:
                logger.info(f"  Intervention: {selected_var} = {intervention_value:.3f}")
            
            intervention = create_perfect_intervention(
                targets=frozenset([selected_var]),
                values={selected_var: float(intervention_value)}
            )
            
            # Sample with intervention
            key, sample_key = random.split(key)
            intervention_samples = sample_with_intervention(
                scm, intervention, n_samples=10, seed=int(sample_key[0])
            )
            
            # Store intervention details for later (before computing reward)
            # Get posterior before intervention
            posterior_before = None
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                # Compute posterior BEFORE intervention
                logger.debug(
                    f"Computing posterior before intervention - "
                    f"target: {target_var}, buffer size: {buffer.size()}"
                )
                posterior_before = self.surrogate_predict_fn(
                    tensor[:, :, :3], target_var, mapper.variables
                )
                logger.debug(
                    f"Posterior entropy before: {posterior_before.get('entropy', 0):.3f}"
                )
            else:
                # No surrogate - use uniform posterior (0.5 probability for each potential parent)
                from ..training.five_channel_converter import create_uniform_posterior
                posterior_before = create_uniform_posterior(mapper.variables, target_var, probability=0.5)
            
            # Store intervention details for buffer update after GRPO
            intervention_info = {
                'intervention': intervention,
                'samples': intervention_samples,
                'posterior': posterior_before
            }
            
            # Compute reward
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                # Calculate posterior_after for reward computation
                # We need to simulate what it would be WITHOUT actually adding to buffer
                posterior_after = None
                if self.use_surrogate and self.surrogate_predict_fn is not None:
                    # Create a temporary buffer copy to simulate the intervention
                    temp_buffer = ExperienceBuffer()
                    # Copy existing samples
                    for sample in buffer.get_all_samples():
                        if hasattr(sample, 'intervention'):
                            temp_buffer.add_intervention(
                                sample.intervention, 
                                sample, 
                                posterior=sample.posterior if hasattr(sample, 'posterior') else None
                            )
                        else:
                            temp_buffer.add_observation(
                                sample,
                                posterior=sample.posterior if hasattr(sample, 'posterior') else None
                            )
                    
                    # Add the current intervention to temp buffer
                    for sample in intervention_samples:
                        temp_buffer.add_intervention(intervention, sample, posterior=posterior_before)
                    
                    # Update surrogate with temp buffer to get what posterior would be
                    temp_surrogate_params = self.surrogate_params
                    temp_surrogate_opt_state = self.surrogate_opt_state
                    temp_surrogate_params, temp_surrogate_opt_state, _ = \
                        self.surrogate_update_fn(
                            temp_surrogate_params,
                            temp_surrogate_opt_state,
                            temp_buffer,
                            target_var
                        )
                    
                    # Create tensor from temp buffer for posterior_after
                    tensor_after, mapper_after = buffer_to_three_channel_tensor(
                        temp_buffer, target_var, max_history_size=100, standardize=False
                    )
                    
                    # Compute posterior after with temp surrogate
                    # Create a temporary predict function with updated params
                    # Need to provide RNG key for Haiku transformed function
                    self.rng_key, surrogate_key = random.split(self.rng_key)
                    # Convert target_var to index for surrogate
                    target_idx_after = mapper_after.variables.index(target_var) if isinstance(target_var, str) else target_var
                    posterior_after = self.surrogate_net.apply(
                        temp_surrogate_params,
                        surrogate_key,  # RNG key required as second argument
                        tensor_after, 
                        target_idx_after, 
                        mapper_after.variables
                    )
                    
                    logger.debug(
                        f"Simulated posterior after intervention - "
                        f"entropy before: {posterior_before.get('entropy', 0):.3f}, "
                        f"entropy after: {posterior_after.get('entropy', 0):.3f}"
                    )
                else:
                    logger.debug("No surrogate model available for info gain")
                
                # Map reward weights to clean reward format
                clean_weights = {
                    'target': self.reward_weights.get('optimization', 0.8),
                    'diversity': self.reward_weights.get('discovery', 0.1),
                    'exploration': self.reward_weights.get('efficiency', 0.1),
                    'info_gain': self.reward_weights.get('info_gain', 0.3)  # Add info gain weight
                }
                
                # Debug logging for reward computation
                logger.debug(
                    f"Computing reward for intervention on {selected_var}:"
                    f"  - Intervention value: {float(intervention_value):.3f}\n"
                    f"  - Target variable: {target_var}\n"
                    f"  - Optimization direction: {self.optimization_direction}"
                )
                
                # Use configurable composite reward system
                reward_info = compute_composite_reward(
                    intervention=intervention,
                    outcome_sample=outcome_sample,
                    buffer=buffer,
                    scm=scm,
                    target_variable=target_var,
                    variables=variables,
                    surrogate_predict_fn=self.surrogate_predict_fn if self.use_surrogate else None,
                    config=self.reward_config,
                    tensor_5ch=tensor,
                    mapper=mapper,
                    reward_type=self.config.get('reward_type', 'continuous'),
                    stats=self.reward_stats if hasattr(self, 'reward_stats') else None
                )
                
                # Log reward details
                outcome_values = get_values(outcome_sample)
                logger.debug(
                    f"Reward computed -"
                    f"  - Outcome Y value: {outcome_values.get(target_var, 'N/A')}\n"
                    f"  - Target reward: {reward_info['target']:.3f}\n"
                    f"  - Diversity reward: {reward_info['diversity']:.3f}\n"
                    f"  - Total reward: {reward_info['total']:.3f}"
                )
                
                reward = reward_info['total']
                
                # Track target value
                target_val = outcome_values.get(target_var, 0.0)
                
                # Store initial value for this SCM if first time
                scm_name = scm.name if hasattr(scm, 'name') else f"scm_{id(scm)}"
                if scm_name not in self.initial_target_values:
                    # Get initial value before any interventions
                    initial_samples = sample_from_linear_scm(scm, n_samples=100, seed=self.seed)
                    initial_values = [s.get(target_var, 0.0) for s in initial_samples]
                    self.initial_target_values[scm_name] = float(np.mean(initial_values))
                    self.best_target_values[scm_name] = self.initial_target_values[scm_name]
                
                # Update best value
                if self.optimization_direction == "MINIMIZE":
                    if target_val < self.best_target_values[scm_name]:
                        self.best_target_values[scm_name] = target_val
                else:
                    if target_val > self.best_target_values[scm_name]:
                        self.best_target_values[scm_name] = target_val
                
                # Store in history
                self.target_value_history.append({
                    'episode': self.episode_count,
                    'step': step,
                    'scm': scm_name,
                    'target_value': target_val,
                    'best_so_far': self.best_target_values[scm_name],
                    'initial': self.initial_target_values[scm_name],
                    'improvement': self.best_target_values[scm_name] - self.initial_target_values[scm_name]
                })
                
                # DEBUG: Log target value trend
                if self.training_step % 10 == 0 and step == 0:
                    logger.info(f"  Target value after intervention: {target_val:.3f}")
                    logger.info(f"  Best target so far: {self.best_target_values[scm_name]:.3f}")
                    logger.info(f"  Improvement from initial: {self.best_target_values[scm_name] - self.initial_target_values[scm_name]:.3f}")
                    logger.info(f"  Reward received: {reward:.3f}")
                    
                    # Analyze if intervention helped
                    if self.optimization_direction == "MINIMIZE":
                        if target_val < 0:
                            logger.info(f"  ✓ Good intervention (negative target)")
                        else:
                            logger.warning(f"  ✗ Bad intervention (positive target)")
                            # Log which variable was intervened on
                            logger.warning(f"    Intervened on {selected_var} = {intervention_value:.3f}")
            else:
                reward = 0.0
            
            # Store for GRPO batch
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({
                'variable': selected_var_idx,
                'value': float(intervention_value)
            })
            grpo_batch_data['rewards'].append(reward)
            grpo_batch_data['old_log_probs'].append(float(log_prob))
            grpo_batch_data['intervention_details'].append(intervention_info)
            
            # Store mapper's target index (same for all samples in batch)
            if grpo_batch_data['target_idx'] is None:
                grpo_batch_data['target_idx'] = mapper.target_idx
        
        # Convert to arrays
        grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        
        # Create GRPO batch with proper format
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Add buffer and target info for non-intervention baseline
        grpo_batch['buffer'] = buffer
        grpo_batch['target_variable'] = target_var
        grpo_batch['target_idx'] = grpo_batch_data['target_idx']  # Pass mapper's target index
        
        # Store old params for comparison
        old_params = self.policy_params
        
        # Log before update
        print(f"\n{'='*60}")
        print(f"[GRPO UPDATE] Episode {episode_idx}, Intervention {len(buffer.get_interventions())}")
        print(f"{'='*60}")
        
        # Log batch statistics
        print(f"\n📊 GRPO Batch Statistics:")
        print(f"  Group size: {len(grpo_batch['interventions'])}")
        print(f"  Rewards shape: {grpo_batch['rewards'].shape}")
        print(f"  Rewards range: [{jnp.min(grpo_batch['rewards']):.4f}, {jnp.max(grpo_batch['rewards']):.4f}]")
        print(f"  Mean reward: {jnp.mean(grpo_batch['rewards']):.4f}")
        
        # Log advantages if available
        if 'advantages' in grpo_batch:
            print(f"  Advantages range: [{jnp.min(grpo_batch['advantages']):.4f}, {jnp.max(grpo_batch['advantages']):.4f}]")
            print(f"  Mean advantage: {jnp.mean(grpo_batch['advantages']):.4f}")
        
        # Perform GRPO update
        print(f"\n🔄 Performing GRPO update...")
        self.policy_params, self.optimizer_state, grpo_metrics = self.grpo_update(
            self.policy_params,
            self.optimizer_state,
            grpo_batch
        )
        
        # Always check if parameters actually changed (not just every 50 episodes)
        param_changes = jax.tree.map(
            lambda old, new: jnp.linalg.norm(new - old),
            old_params, self.policy_params
        )
        change_leaves, _ = jax.tree_util.tree_flatten(param_changes)
        total_change = jnp.sum(jnp.array(change_leaves))
        max_change = jnp.max(jnp.array(change_leaves))
        mean_change = jnp.mean(jnp.array(change_leaves))
        
        print(f"\n✅ Update Complete:")
        print(f"  Gradient norm: {grpo_metrics.grad_norm:.6f}")
        print(f"  Policy loss: {grpo_metrics.policy_loss:.4f}")
        print(f"  Value loss: {grpo_metrics.value_loss:.4f}")
        print(f"  Mean advantage: {grpo_metrics.mean_advantage:.4f}")
        
        print(f"\n🔧 Parameter Changes:")
        print(f"  Total change (sum of norms): {total_change:.6f}")
        print(f"  Mean change: {mean_change:.6f}")
        print(f"  Max change: {max_change:.6f}")
        
        if total_change < 1e-8:
            print(f"  ⚠️ WARNING: NO PARAMETERS CHANGED!")
            print(f"     Possible causes:")
            print(f"     - Learning rate too small (current: {self.learning_rate})")
            print(f"     - Gradients are zero (norm: {grpo_metrics.grad_norm:.6f})")
            print(f"     - Optimizer issue")
        elif total_change < 1e-6:
            print(f"  ⚠️ WARNING: Very small parameter changes")
        else:
            print(f"  ✅ Parameters updated successfully")
        
        # Log specific parameter changes for key layers
        if episode_idx % 10 == 0:  # More frequent detailed logging
            print(f"\n📍 Key Layer Changes:")
            for key, change in param_changes.items():
                if 'var_mlp_output' in key or 'val_mlp_output' in key or 'input_projection' in key:
                    if hasattr(change, 'shape'):
                        print(f"    {key}: norm={jnp.linalg.norm(change):.6f}, shape={change.shape}")
        
        self.training_step += 1
        
        # Only process GRPO batch data if it exists (i.e., we're in GRPO/policy phase)
        if 'grpo_batch_data' in locals() and grpo_batch_data is not None:
            # After GRPO update, add only the BEST intervention to buffer
            # This maintains a consistent causal history
            best_idx = jnp.argmax(grpo_batch_data['rewards'])
            best_intervention_info = grpo_batch_data['intervention_details'][int(best_idx)]
            
            # Get buffer size before adding
            buffer_size_before = buffer.size()
            
            # Add only ONE sample from the best intervention to maintain clear causal attribution
            # This ensures the model sees a consistent history without noise confusion
            if best_intervention_info['samples']:
                # Take the first sample (could also randomly select one)
                single_sample = best_intervention_info['samples'][0]
                buffer.add_intervention(
                    best_intervention_info['intervention'], 
                    single_sample, 
                    posterior=best_intervention_info['posterior']
                )
            
            # Verify buffer grew correctly
            buffer_size_after = buffer.size()
            samples_added = buffer_size_after - buffer_size_before
            
            # Get the last sample from buffer for verification
            last_buffer_sample = buffer.get_all_samples()[-1] if buffer.get_all_samples() else None
            last_sample_values = get_values(last_buffer_sample) if last_buffer_sample else {}
            last_target_value = last_sample_values.get(target_var, None)
            
            # Log which intervention was selected with detailed info
            best_reward = grpo_batch_data['rewards'][best_idx]
            best_action = grpo_batch_data['actions'][int(best_idx)]
            best_var_name = mapper.get_name(int(best_action['variable']))
            
            logger.info(
                f"\n[BEST INTERVENTION SELECTED] Episode {episode_idx}:"
            )
            logger.info(
                f"  Selected intervention {best_idx+1}/{self.grpo_config.group_size}: "
                f"{best_var_name} = {best_action['value']:.3f}"
            )
            logger.info(f"  Best reward: {best_reward:.3f}")
            logger.info(f"  Buffer size: {buffer_size_before} -> {buffer_size_after} ({samples_added} samples added)")
            logger.info(f"  Last buffer target value: {last_target_value:.3f}" if last_target_value is not None else "  Last buffer target value: None")
            
            # Compare with best intervention's target value
            best_intervention_sample = best_intervention_info['samples'][0] if best_intervention_info['samples'] else None
            if best_intervention_sample:
                best_intervention_values = get_values(best_intervention_sample)
                best_intervention_target = best_intervention_values.get(target_var, None)
                logger.info(f"  Best intervention target value: {best_intervention_target:.3f}" if best_intervention_target is not None else "  Best intervention target value: None")
                
                # Sanity check
                if last_target_value is not None and best_intervention_target is not None:
                    if abs(last_target_value - best_intervention_target) < 0.001:
                        logger.info("  ✓ SANITY CHECK PASSED: Buffer contains correct intervention")
                    else:
                        logger.warning(f"  ✗ SANITY CHECK FAILED: Buffer value {last_target_value:.3f} != Best intervention {best_intervention_target:.3f}")
        
        # Now update surrogate if using one
        if self.use_surrogate and self.surrogate_predict_fn is not None:
            self.surrogate_params, self.surrogate_opt_state, update_metrics = \
                self.surrogate_update_fn(
                    self.surrogate_params,
                    self.surrogate_opt_state,
                    buffer,  # Buffer now includes only the best intervention
                    target_var
                )
            logger.debug(f"Surrogate updated with best intervention. Loss: {update_metrics.get('loss', 0):.4f}")
        
        # Compute structure metrics if surrogate is available
        structure_metrics = {}
        if self.use_surrogate and true_parents and posterior_before:
            # Use the last posterior (after all interventions)
            metrics = compute_structure_metrics_continuous(posterior_before, true_parents)
            structure_metrics = metrics
        
        # Log target value trend every 5 episodes (per SCM!)
        if episode_idx % 5 == 0 and self.target_value_history:
            # Get history ONLY for this specific SCM
            scm_history = [h for h in self.target_value_history if h['scm'] == scm_name]
            if len(scm_history) >= 2:
                recent_targets = [h['target_value'] for h in scm_history[-5:]]  # Last 5 for this SCM
                initial_value = scm_history[0]['target_value']
                current_value = scm_history[-1]['target_value']
                trend = current_value - initial_value
                logger.info(
                    f"\n[TARGET TREND] Episode {episode_idx}, SCM {scm_name}: "
                    f"Initial: {initial_value:.2f}, Current: {current_value:.2f}, "
                    f"Trend: {trend:+.3f} {'↓' if trend < 0 else '↑'}"
                )
        
        return {
            'episode': episode_idx,
            'mean_reward': float(grpo_metrics.mean_reward),
            'loss': float(grpo_metrics.total_loss),
            'n_variables': len(variables),
            'scm_type': scm_name,
            'structure_metrics': structure_metrics,
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
    
    def _prepare_scms(self, scms: Union[List[Any], Dict[str, Any], Callable]) -> List[Tuple[str, Any]]:
        """Convert various SCM formats to standard list of (name, scm) tuples."""
        if isinstance(scms, list):
            # Check if it's already a list of tuples
            if scms and isinstance(scms[0], tuple) and len(scms[0]) == 2:
                # Already in (name, scm) format
                return scms
            else:
                # Generate names for unnamed SCMs
                return [(f"scm_{i}", scm) for i, scm in enumerate(scms)]
        elif isinstance(scms, dict):
            # Use provided names
            return list(scms.items())
        elif callable(scms):
            # Generate SCMs on demand
            generated = []
            for i in range(10):  # Default to 10 SCMs
                scm = scms()
                generated.append((f"generated_{i}", scm))
            return generated
        else:
            raise ValueError(f"Unsupported SCM format: {type(scms)}")
    
    def _create_grpo_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create GRPO batch in format expected by GRPO loss computation."""
        # Stack tensors
        states_batch = jnp.stack(batch_data['states'])  # [batch_size, T, n_vars, 5]
        
        # Extract action indices and values
        action_var_indices = jnp.array([a['variable'] for a in batch_data['actions']])
        action_values = jnp.array([a['value'] for a in batch_data['actions']])
        
        # Rewards and old log probs are already arrays
        
        return {
            'states': states_batch,
            'actions': {
                'variables': action_var_indices,
                'values': action_values
            },
            'rewards': batch_data['rewards'],
            'old_log_probs': batch_data['old_log_probs']
        }
    
    def _compute_simple_grpo_loss(self, params: Any, batch: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute GRPO loss - supports both standard and enhanced modes.
        
        Enhanced mode (use_grpo_rewards=True): 
        - Uses pre-computed group advantages
        - Complete log probabilities (variable + value)
        - Enhanced debugging metrics
        
        Standard mode (use_grpo_rewards=False):
        - Computes advantages from baselines
        - Variable-only log probabilities
        - Basic metrics
        """
        states = batch['states']  # [batch_size, T, n_vars, 5]
        actions = batch['actions']
        rewards = batch['rewards']
        old_log_probs = batch['old_log_probs']
        
        # Use pre-computed advantages in enhanced mode, compute them in standard mode
        if self.use_grpo_rewards and 'advantages' in batch:
            # Enhanced mode: use pre-computed group advantages
            advantages = batch['advantages']
            group_baseline = jnp.mean(rewards)  # Still need for metrics
        else:
            # PROPER GRPO: Use within-group baseline
            # The group baseline is the mean of rewards within this batch
            # This compares actions taken from the SAME state
            group_baseline = jnp.mean(rewards)
            
            # Advantages = how much better than average in this group
            advantages = rewards - group_baseline
            
            logger.debug(
                f"[GRPO BASELINE] Group mean: {group_baseline:.3f}, "
                f"Rewards range: [{jnp.min(rewards):.3f}, {jnp.max(rewards):.3f}]"
            )
        
        # Normalize advantages for stability (only in standard mode)
        adv_std = jnp.std(advantages)
        if adv_std > 1e-8:
            advantages = advantages / adv_std
        
        # Forward pass to get current log probs
        batch_size = states.shape[0]
        new_log_probs = []
        entropy_values = []
        
        # Track individual components for enhanced debugging
        log_prob_changes = [] if self.use_grpo_rewards else None
        
        for i in range(batch_size):
            # Get policy output for this state
            self.rng_key, policy_key = random.split(self.rng_key)
            
            # Use the target index from the batch (from mapper)
            target_idx = batch.get('target_idx', states[i].shape[1] - 1)  # Fallback to last var
            
            policy_output = self.policy_fn.apply(
                params, policy_key, states[i], target_idx
            )
            
            # Compute log prob for selected action
            var_probs = jax.nn.softmax(policy_output['variable_logits'])
            
            # Check action format to determine mode
            if isinstance(actions, list) and len(actions) > i and isinstance(actions[i], dict):
                # Enhanced mode format: list of dicts with 'variable' and 'value'
                selected_var = actions[i]['variable']
                
                if self.use_grpo_rewards:
                    # Compute COMPLETE log probability
                    log_prob_var = jnp.log(var_probs[selected_var] + 1e-8)
                    
                    # Get value distribution parameters
                    value_params = policy_output['value_params']
                    mean = value_params[selected_var, 0]
                    log_std = value_params[selected_var, 1]
                    std = jnp.exp(log_std)
                    
                    # Compute log prob of the actual value under the Gaussian
                    actual_value = actions[i]['value']
                    log_prob_value = -0.5 * ((actual_value - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
                    
                    # Complete log probability
                    log_prob = log_prob_var + log_prob_value
                    new_log_probs.append(log_prob)
                    
                    # Track change from old log prob
                    if log_prob_changes is not None:
                        log_prob_changes.append(log_prob - old_log_probs[i])
                else:
                    # Just variable log prob
                    log_prob = jnp.log(var_probs[selected_var] + 1e-8)
                    new_log_probs.append(log_prob)
            else:
                # Standard mode format: dict with 'variables' key
                selected_var = actions['variables'][i]
                log_prob = jnp.log(var_probs[selected_var] + 1e-8)
                new_log_probs.append(log_prob)
            
            # Compute entropy
            entropy = -jnp.sum(var_probs * jnp.log(var_probs + 1e-8))
            entropy_values.append(entropy)
        
        new_log_probs = jnp.array(new_log_probs)
        entropy_values = jnp.array(entropy_values)
        
        # Compute ratio for PPO-style clipping
        log_ratio = new_log_probs - old_log_probs
        ratio = jnp.exp(log_ratio)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = jnp.clip(ratio, 1.0 - self.grpo_config.clip_ratio, 1.0 + self.grpo_config.clip_ratio) * advantages
        
        # Track which surrogate is used (for enhanced debugging)
        surr_min = jnp.minimum(surr1, surr2)
        policy_loss = -jnp.mean(surr_min)
        
        # Track how many samples are clipped (enhanced mode)
        if self.use_grpo_rewards:
            clipped_mask = jnp.abs(ratio - 1.0) > self.grpo_config.clip_ratio
            clip_fraction = jnp.mean(clipped_mask)
        else:
            clip_fraction = 0.0
        
        # Entropy loss (negative for maximization)
        entropy_loss = -self.grpo_config.entropy_coeff * jnp.mean(entropy_values)
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Compute diagnostics
        approx_kl = jnp.mean((new_log_probs - old_log_probs) ** 2)
        
        # Determine baseline value for reporting
        # group_baseline already computed above in both modes
        
        loss_info = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'kl_penalty': 0.0,  # Not using KL penalty
            'group_baseline': group_baseline,
            'mean_reward': jnp.mean(rewards),
            'reward_std': jnp.std(rewards),
            'mean_advantage': jnp.mean(advantages),
            'advantage_std': jnp.std(advantages),
            'mean_entropy': jnp.mean(entropy_values),
            'approx_kl': approx_kl,
            'mean_ratio': jnp.mean(ratio),
            'ratio_std': jnp.std(ratio),
            'clip_fraction': clip_fraction
        }
        
        # Add enhanced debugging info if in enhanced mode
        if self.use_grpo_rewards:
            loss_info.update({
                'mean_log_prob_change': jnp.mean(jnp.array(log_prob_changes)) if log_prob_changes else 0.0,
                'surr1_mean': jnp.mean(surr1),
                'surr2_mean': jnp.mean(surr2),
                'surr_min_mean': jnp.mean(surr_min),
                'positive_advantages': jnp.sum(advantages > 0),
                'negative_advantages': jnp.sum(advantages < 0),
                # Diagnostic info
                'loss_terms_sum': jnp.sum(-surr_min),
                'loss_terms_mean': jnp.mean(-surr_min),
                'log_prob_variance': jnp.var(new_log_probs),
                'unique_log_probs': jnp.unique(new_log_probs).shape[0]
            })
        
        # Add loss components for external diagnostics
        loss_info.update({
            'total_loss_value': total_loss,
            'policy_loss_value': policy_loss,
            'entropy_loss_value': entropy_loss
        })
        
        return total_loss, loss_info
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint using unified format."""
        logger.info(f"[CHECKPOINT] Starting checkpoint save (is_final={is_final})")
        from ..utils.checkpoint_utils import save_checkpoint as save_unified_checkpoint
        
        checkpoint_dir = Path(self.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[CHECKPOINT] Checkpoint directory: {checkpoint_dir}")
        
        name = "unified_grpo_final" if is_final else f"unified_grpo_ep{self.episode_count}"
        checkpoint_path = checkpoint_dir / name
        logger.info(f"[CHECKPOINT] Saving to: {checkpoint_path}")
        
        # Save policy checkpoint
        policy_architecture = {
            'hidden_dim': self.config.get('architecture', {}).get('hidden_dim', 256),
            'architecture_level': self.architecture_level,
            'architecture_type': self.policy_architecture if hasattr(self, 'policy_architecture') else 'alternating_attention'
        }
        
        training_config = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_episodes': self.max_episodes,
            'optimization_direction': self.optimization_direction
        }
        
        metadata = {
            'trainer_type': 'UnifiedGRPOTrainer',
            'episode': self.episode_count,
            'is_final': is_final,
            'uses_true_grpo': True,
            'has_surrogate': self.use_surrogate
        }
        
        metrics = {
            'mean_reward': self.training_metrics[-1].mean_reward if self.training_metrics else 0.0,
            'episode': self.episode_count
        }
        
        try:
            save_unified_checkpoint(
                path=checkpoint_path,
                params=self.policy_params,
                architecture=policy_architecture,
                model_type='policy',
                model_subtype='grpo',
                training_config=training_config,
                metadata=metadata,
                metrics=metrics
            )
            logger.info(f"[CHECKPOINT] ✓ Policy checkpoint saved successfully")
        except Exception as e:
            logger.error(f"[CHECKPOINT] ✗ Failed to save policy checkpoint: {e}")
            logger.exception("Full traceback:")
        
        # Save surrogate separately if enabled
        if self.use_surrogate and self.surrogate_params is not None:
            surrogate_path = checkpoint_path / 'surrogate'
            
            # Get surrogate architecture from config
            surrogate_architecture = {
                'hidden_dim': self.config.get('surrogate_hidden_dim', 128),
                'num_layers': self.config.get('surrogate_layers', 4),
                'num_heads': self.config.get('surrogate_heads', 8),
                'key_size': self.config.get('surrogate_hidden_dim', 128) // self.config.get('surrogate_heads', 8),
                'dropout': 0.1
            }
            
            save_unified_checkpoint(
                path=surrogate_path,
                params=self.surrogate_params,
                architecture=surrogate_architecture,
                model_type='surrogate',
                model_subtype='continuous_parent_set',
                metadata={'parent_policy': name}
            )
            
            logger.info(f"Saved surrogate checkpoint to {surrogate_path}")
        
    def save_checkpoint(self, path: Path, results: Dict[str, Any]) -> None:
        """Save training checkpoint using unified format."""
        from ..utils.checkpoint_utils import save_checkpoint as save_unified_checkpoint
        
        # Extract architecture
        config = self.config if hasattr(self, 'config') else {}
        architecture = {
            'hidden_dim': config.get('architecture', {}).get('hidden_dim', 256),
            'architecture_level': self.architecture_level,
            'architecture_type': self.policy_architecture if hasattr(self, 'policy_architecture') else 'alternating_attention'
        }
        
        # Training configuration
        training_config = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_episodes': self.max_episodes,
            'optimization_direction': self.optimization_direction
        }
        
        # Metadata
        metadata = {
            'trainer_type': 'UnifiedGRPOTrainer',
            'uses_true_grpo': True,
            'converged': results.get('converged', False),
            'has_surrogate': self.use_surrogate
        }
        
        save_unified_checkpoint(
            path=path,
            params=results.get('policy_params', self.policy_params),
            architecture=architecture,
            model_type='policy',
            model_subtype='grpo',
            training_config=training_config,
            metadata=metadata,
            metrics=results.get('all_metrics', {})
        )
    
    def _run_single_grpo_intervention(self, buffer: ExperienceBuffer, scm: Any, 
                                      target_var: str, variables: list, key) -> Dict[str, Any]:
        """
        Run single GRPO intervention: generate candidates, update policy, return best.
        
        KEEPS all the essential logging:
        - Candidate reward breakdown
        - GRPO advantages  
        - Variable probabilities
        - Parameter changes
        - Buffer validation
        """
        # Collect GRPO batch
        grpo_batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'old_log_probs': [],
            'target_idx': None,
            'intervention_details': []
        }
        
        # Generate batch of candidates
        for step in range(self.grpo_config.group_size):
            key, step_key = random.split(key)
            
            # Convert buffer to 5-channel tensor using stored posteriors
            if self.use_surrogate and self.surrogate_predict_fn is not None:
                from ..training.five_channel_converter import buffer_to_five_channel_tensor_with_posteriors
                tensor, mapper, diagnostics = buffer_to_five_channel_tensor_with_posteriors(
                    buffer, target_var, max_history_size=100, standardize=False, use_uniform_for_missing=True
                )
                tensor_3ch = tensor[:, :, :3]
                posterior_before = self.surrogate_predict_fn(tensor_3ch, target_var, mapper.variables)
            else:
                # Check if policy expects 4 channels (simple_permutation_invariant variants)
                if hasattr(self, 'policy_architecture') and 'permutation_invariant' in self.policy_architecture:
                    # Use 4-channel converter for permutation invariant policies
                    tensor, mapper, diagnostics = buffer_to_four_channel_tensor(
                        buffer, target_var, surrogate_fn=None, max_history_size=100, standardize=False
                    )
                else:
                    # Use 5-channel converter for other policies
                    tensor, mapper, diagnostics = buffer_to_five_channel_tensor(
                        buffer, target_var, surrogate_fn=None, max_history_size=100, standardize=False
                    )
                posterior_before = None
            
            # Get policy output
            key, policy_key = random.split(key)
            policy_output = self.policy_fn.apply(self.policy_params, policy_key, tensor, mapper.target_idx)
            
            # Sample intervention
            var_logits = policy_output['variable_logits']
            value_params = policy_output['value_params']
            
            key, var_key = random.split(key)
            exploration_noise = 0.1
            noisy_logits = var_logits + exploration_noise * random.normal(var_key, var_logits.shape)
            selected_var_idx = random.categorical(var_key, noisy_logits)
            var_probs = jax.nn.softmax(var_logits)
            
            # KEEP: Variable probability logging 
            if step == 0:  # Log once per intervention
                prob_info = []
                for i, var_name in enumerate(mapper.variables):
                    if i != mapper.target_idx:
                        prob_info.append(f"{var_name}:{var_probs[i]:.3f}")
                print(f"  Variable probabilities: {' '.join(prob_info)}")
                
                # DIAGNOSTIC: Buffer state analysis
                print(f"\n🔍 BUFFER STATE ANALYSIS:")
                print(f"  Buffer total size: {buffer.size()}")
                print(f"  Observations: {len(buffer.get_observations())}")
                print(f"  Interventions: {len(buffer.get_interventions())}")
                
                # Show last few samples from buffer
                all_samples = buffer.get_all_samples()
                print(f"\n📋 BUFFER CONTENT (last 3 samples):")
                for i, sample in enumerate(all_samples[-3:]):
                    sample_values = get_values(sample)
                    print(f"    Sample {len(all_samples)-3+i+1}: {dict(sample_values)}")
                
                # CRITICAL: Analyze tensor construction
                print(f"\n🧮 TENSOR CONSTRUCTION ANALYSIS:")
                print(f"  Tensor shape: {tensor.shape}")
                print(f"  Mapper variables: {mapper.variables}")
                print(f"  Target variable: {target_var} (index: {mapper.target_idx})")
                
                # Check each channel content
                for ch_idx, ch_name in enumerate(['Values', 'Target', 'Interv', 'Probs', 'Recency']):
                    channel = tensor[:, :, ch_idx]
                    print(f"  Channel {ch_idx} ({ch_name}):")
                    print(f"    Shape: {channel.shape}")
                    print(f"    Mean: {jnp.mean(channel):.6f}")
                    print(f"    Std: {jnp.std(channel):.6f}")
                    print(f"    Range: [{jnp.min(channel):.3f}, {jnp.max(channel):.3f}]")
                    
                    # Show last timestep for current state
                    last_timestep = channel[-1, :]  # Most recent values
                    print(f"    Last timestep: {last_timestep}")
                    
                    # Check for suspicious patterns
                    if ch_name == 'Values':
                        # Are all variables distinguishable?
                        var_means = jnp.mean(channel, axis=0)  # Mean per variable
                        var_std_across_vars = jnp.std(var_means)
                        print(f"    Variable distinguishability: {var_std_across_vars:.6f}")
                        
                        if var_std_across_vars < 0.01:
                            print(f"    ❌ CRITICAL: Variables look identical!")
                        else:
                            print(f"    ✅ Variables are distinguishable")
                    
                    elif ch_name == 'Probs':
                        # Are posteriors actually varying?
                        unique_vals = jnp.unique(channel)
                        print(f"    Unique probability values: {len(unique_vals)}")
                        if len(unique_vals) <= 2:
                            print(f"    ❌ CRITICAL: Posteriors not varying! All: {unique_vals}")
                        else:
                            print(f"    ✅ Posteriors show variation: {unique_vals[:5]}...")
                
                # Check if tensor is actually informative
                total_variation = jnp.std(tensor)
                print(f"\n📊 OVERALL TENSOR ANALYSIS:")
                print(f"  Total variation (std): {total_variation:.6f}")
                print(f"  Non-zero elements: {jnp.sum(tensor != 0.0)}/{tensor.size}")
                print(f"  Informative ratio: {jnp.sum(tensor != 0.0)/tensor.size:.3f}")
                
                if total_variation < 0.1:
                    print(f"  ❌ CRITICAL: Tensor has very low variation - may not be informative!")
                elif jnp.sum(tensor != 0.0)/tensor.size < 0.3:
                    print(f"  ⚠️ WARNING: Tensor is mostly zeros - check data population")
                else:
                    print(f"  ✅ Tensor appears to have sufficient information")
            
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
            
            # Compute reward with detailed logging
            outcome_sample = intervention_samples[0] if intervention_samples else None
            if outcome_sample:
                reward_info = compute_composite_reward(
                    intervention=intervention,
                    outcome_sample=outcome_sample,
                    buffer=buffer,
                    scm=scm,
                    target_variable=target_var,
                    variables=variables,
                    surrogate_predict_fn=self.surrogate_predict_fn if self.use_surrogate else None,
                    config=self.reward_config,
                    tensor_5ch=tensor,
                    mapper=mapper,
                    reward_type=self.config.get('reward_type', 'continuous'),
                    stats=self.reward_stats
                )
                reward = reward_info['total']
            else:
                reward = 0.0
            
            # Store for GRPO batch
            log_prob = jnp.log(var_probs[selected_var_idx] + 1e-8)
            grpo_batch_data['states'].append(tensor)
            grpo_batch_data['actions'].append({'variable': selected_var_idx, 'value': float(intervention_value)})
            grpo_batch_data['rewards'].append(reward)
            grpo_batch_data['old_log_probs'].append(float(log_prob))
            grpo_batch_data['intervention_details'].append({
                'intervention': intervention,
                'samples': intervention_samples,
                'posterior': posterior_before
            })
            
            if grpo_batch_data['target_idx'] is None:
                grpo_batch_data['target_idx'] = mapper.target_idx
        
        # KEEP: Detailed candidate logging with reward breakdown
        self._log_candidates_with_rewards(grpo_batch_data, mapper, target_var, scm)
        
        # GRPO update with comprehensive diagnostics
        grpo_batch_data['rewards'] = jnp.array(grpo_batch_data['rewards'])
        grpo_batch_data['old_log_probs'] = jnp.array(grpo_batch_data['old_log_probs'])
        grpo_batch = self._create_grpo_batch(grpo_batch_data)
        
        # Store old params and policy output for comparison
        old_params = self.policy_params
        
        # Capture old policy output for evolution analysis
        old_policy_output = self.policy_fn.apply(old_params, random.PRNGKey(42), tensor, mapper.target_idx)
        old_var_logits = old_policy_output['variable_logits']
        old_var_probs = jax.nn.softmax(old_var_logits)
        
        print(f"\n🔬 PRE-UPDATE DIAGNOSTICS:")
        print(f"  Old var logits: {old_var_logits}")
        print(f"  Old var probs: {old_var_probs}")
        
        # Perform GRPO update with detailed gradient analysis
        print(f"\n🔬 DETAILED GRADIENT FLOW ANALYSIS:")
        
        # Mathematical flow tracing
        rewards_array = grpo_batch_data['rewards']
        baseline = jnp.mean(rewards_array)
        advantages = rewards_array - baseline
        
        print(f"  📊 Mathematical Flow:")
        print(f"    Rewards: {[f'{float(r):.3f}' for r in rewards_array]}")
        print(f"    Baseline: {baseline:.3f}")
        print(f"    Advantages: {[f'{float(a):+.3f}' for a in advantages]}")
        print(f"    Advantage signal strength: {jnp.std(advantages):.6f}")
        
        # Capture gradients during update
        def grpo_loss_fn(params):
            return self._compute_simple_grpo_loss(params, grpo_batch)
        
        old_params_copy = jax.tree.map(lambda x: x.copy(), self.policy_params)
        (loss_val, loss_info), grads = jax.value_and_grad(grpo_loss_fn, has_aux=True)(self.policy_params)
        
        # Analyze gradient distribution (handle nested structure)
        grad_norms = {}
        var_head_total = 0.0
        val_head_total = 0.0
        other_total = 0.0
        
        # Flatten gradient tree to handle nested dictionaries
        flat_grads = jax.tree_util.tree_flatten_with_path(grads)[0]
        
        for path_info, grad in flat_grads:
            # Convert path to string
            param_path = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_info)
            
            # Skip if grad is not a valid array
            if not hasattr(grad, 'shape'):
                continue
                
            norm = jnp.linalg.norm(grad)
            grad_norms[param_path] = float(norm)
            
            # Categorize by network component
            if 'var_mlp' in param_path or 'variable' in param_path:
                var_head_total += norm
            elif 'val_mlp' in param_path or 'value' in param_path:
                val_head_total += norm
            else:
                other_total += norm
        
        total_grad_norm = var_head_total + val_head_total + other_total
        
        print(f"  📊 Gradient Flow Distribution:")
        print(f"    Variable head: {var_head_total:.6f} ({var_head_total/total_grad_norm:.1%})")
        print(f"    Value head: {val_head_total:.6f} ({val_head_total/total_grad_norm:.1%})")
        print(f"    Other layers: {other_total:.6f} ({other_total/total_grad_norm:.1%})")
        print(f"    Total norm: {total_grad_norm:.6f}")
        
        # Critical checks
        if total_grad_norm < 1e-6:
            print(f"  ❌ CRITICAL: Total gradients near zero - no learning possible!")
        elif var_head_total / total_grad_norm < 0.05:
            print(f"  ❌ CRITICAL: Variable head gets <5% gradients - can't learn selection!")
        elif jnp.std(advantages) < 0.01:
            print(f"  ❌ CRITICAL: Advantage signal too weak - uniform rewards!")
        else:
            print(f"  ✅ Gradient flow looks reasonable")
        
        # Apply gradients manually
        updates, self.optimizer_state = self.optimizer.update(grads, self.optimizer_state, self.policy_params)
        self.policy_params = optax.apply_updates(self.policy_params, updates)
        
        # Create grpo_metrics for compatibility
        grpo_metrics = type('GRPOMetrics', (), {
            'policy_loss': float(loss_info.get('policy_loss', loss_val)),
            'total_loss': float(loss_val),
            'mean_reward': float(jnp.mean(rewards_array)),
            'grad_norm': float(total_grad_norm)
        })()
        
        # POST-UPDATE ANALYSIS
        print(f"\n🔬 POST-UPDATE DIAGNOSTICS:")
        
        # Check new policy output
        new_policy_output = self.policy_fn.apply(self.policy_params, random.PRNGKey(42), tensor, mapper.target_idx)
        new_var_logits = new_policy_output['variable_logits']
        new_var_probs = jax.nn.softmax(new_var_logits)
        
        print(f"  New var logits: {new_var_logits}")
        print(f"  New var probs: {new_var_probs}")
        
        # Policy evolution analysis
        logit_change = new_var_logits - old_var_logits
        prob_change = new_var_probs - old_var_probs
        
        print(f"  Logit changes: {logit_change}")
        print(f"  Prob changes: {prob_change}")
        print(f"  Max logit change: {jnp.max(jnp.abs(logit_change)):.6f}")
        print(f"  Max prob change: {jnp.max(jnp.abs(prob_change)):.6f}")
        
        # Check if changes are meaningful
        if jnp.max(jnp.abs(logit_change)) < 1e-4:
            print(f"  ❌ CRITICAL: Logits barely changing!")
        elif jnp.max(jnp.abs(prob_change)) < 1e-3:
            print(f"  ❌ CRITICAL: Probabilities barely changing!")
        else:
            print(f"  ✅ Policy output is evolving")
        
        # DIAGNOSTIC: Loss component analysis
        print(f"\n📊 LOSS COMPONENT ANALYSIS:")
        total_loss_val = float(grpo_metrics.total_loss)
        policy_loss_val = float(grpo_metrics.policy_loss) 
        entropy_loss_val = getattr(grpo_metrics, 'entropy_loss', 0.0)
        
        print(f"  Total loss: {total_loss_val:.6f}")
        print(f"  Policy loss: {policy_loss_val:.6f}")
        print(f"  Entropy loss: {entropy_loss_val:.6f}")
        
        if abs(total_loss_val) > 1e-8:
            policy_ratio = abs(policy_loss_val) / (abs(total_loss_val) + 1e-8)
            entropy_ratio = abs(entropy_loss_val) / (abs(total_loss_val) + 1e-8)
            print(f"  Loss ratio: policy={policy_ratio:.1%}, entropy={entropy_ratio:.1%}")
            
            if entropy_ratio > 0.8:
                print(f"  ❌ CRITICAL: Entropy dominates ({entropy_ratio:.1%}) - may prevent learning!")
            elif policy_ratio > 0.8:
                print(f"  ✅ Policy loss dominates ({policy_ratio:.1%}) - good for learning")
            else:
                print(f"  ⚠️ Mixed dominance - check balance")
        
        # DIAGNOSTIC: REINFORCE comparison
        print(f"\n🧪 REINFORCE COMPARISON:")
        
        # Compute what REINFORCE loss would be
        new_log_probs_for_reinforce = []
        for i, action in enumerate(grpo_batch_data['actions']):
            selected_var_idx = action['variable']
            log_prob = jnp.log(new_var_probs[selected_var_idx] + 1e-8)
            new_log_probs_for_reinforce.append(log_prob)
        
        new_log_probs_array = jnp.array(new_log_probs_for_reinforce)
        rewards_array = grpo_batch_data['rewards']
        
        # Simple REINFORCE: loss = -mean(rewards * log_probs)
        reinforce_loss = -jnp.mean(rewards_array * new_log_probs_array)
        grpo_policy_loss = grpo_metrics.policy_loss
        
        print(f"  GRPO policy loss: {grpo_policy_loss:.6f}")
        print(f"  REINFORCE loss: {reinforce_loss:.6f}")
        print(f"  Loss difference: {grpo_policy_loss - reinforce_loss:.6f}")
        print(f"  REINFORCE signal strength: {jnp.abs(reinforce_loss):.6f}")
        
        if jnp.abs(reinforce_loss) < 1e-6:
            print(f"  ❌ CRITICAL: Even REINFORCE has no signal!")
        elif jnp.abs(grpo_policy_loss) < jnp.abs(reinforce_loss) * 0.1:
            print(f"  ⚠️ GRPO loss much weaker than REINFORCE - ratio issue?")
        else:
            print(f"  ✅ GRPO and REINFORCE losses similar magnitude")
        
        # KEEP: Parameter change logging
        param_changes = jax.tree.map(lambda old, new: jnp.linalg.norm(new - old), old_params, self.policy_params)
        total_change = sum(jax.tree.leaves(param_changes))
        print(f"Parameter change: {total_change:.6f}")
        
        # EXPERIMENTAL: Use RANDOM candidate selection to eliminate selection bias
        # This tests if GRPO learning is working vs just good candidate selection
        import random as py_random
        best_idx = jnp.argmax(grpo_batch_data['rewards'])  # Track what COULD have been selected
        random_idx = py_random.randint(0, len(grpo_batch_data['rewards']) - 1)  # Actually select random
        selected_intervention_info = grpo_batch_data['intervention_details'][random_idx]
        
        # Log selection comparison
        best_reward = grpo_batch_data['rewards'][best_idx]
        selected_reward = grpo_batch_data['rewards'][random_idx]
        print(f"\n🎲 RANDOM SELECTION TEST:")
        print(f"  Selected: #{random_idx+1} (reward: {selected_reward:.3f})")
        print(f"  Best available: #{best_idx+1} (reward: {best_reward:.3f})")
        print(f"  Selection bias eliminated - testing pure GRPO learning")
        
        return {
            'best_intervention': {
                'intervention': selected_intervention_info['intervention'],
                'outcome': selected_intervention_info['samples'][0] if selected_intervention_info['samples'] else None,
                'posterior': selected_intervention_info.get('posterior')
            },
            'candidate_rewards': [float(r) for r in grpo_batch_data['rewards']],
            'grpo_metrics': grpo_metrics,
            'param_change': total_change,
            'selection_info': {
                'selected_idx': random_idx,
                'selected_reward': float(selected_reward),
                'best_idx': int(best_idx),
                'best_reward': float(best_reward),
                'selection_advantage': float(selected_reward - best_reward)
            }
        }
    
    def _log_candidates_with_rewards(self, grpo_batch_data: Dict, mapper: Any, target_var: str, scm: Any):
        """Log detailed candidate breakdown with reward components."""
        print(f"\n[GRPO CANDIDATES]:")
        
        for i in range(len(grpo_batch_data['actions'])):
            action = grpo_batch_data['actions'][i] 
            reward = grpo_batch_data['rewards'][i]
            var_name = mapper.get_name(int(action['variable']))
            
            # Get target value from outcome
            if i < len(grpo_batch_data['intervention_details']):
                intervention_info = grpo_batch_data['intervention_details'][i]
                if 'samples' in intervention_info and intervention_info['samples']:
                    target_value = get_values(intervention_info['samples'][0]).get(target_var, 0.0)
                    
                    # Compute components
                    target_component = -target_value if self.optimization_direction == "MINIMIZE" else target_value
                    parent_component = 1.0 if var_name in set(get_parents(scm, target_var)) else 0.0
                    
                    # Show weighted breakdown
                    target_weighted = self.reward_config.target_weight * target_component
                    parent_weighted = self.reward_config.parent_weight * parent_component
                    
                    print(f"  Candidate {i+1}: {var_name} = {action['value']:.3f} → TARGET = {target_value:.3f}")
                    print(f"    Target: {target_component:.3f} × {self.reward_config.target_weight:.2f} = {target_weighted:.3f}")
                    print(f"    Parent: {parent_component:.1f} × {self.reward_config.parent_weight:.2f} = {parent_weighted:.3f}")
                    print(f"    TOTAL REWARD: {reward:.3f}")
        
        # KEEP: GRPO advantages
        rewards_array = jnp.array(grpo_batch_data['rewards'])
        baseline = jnp.mean(rewards_array)
        advantages = rewards_array - baseline
        best_idx = jnp.argmax(advantages)
        
        print(f"\n📊 GRPO Advantages:")
        print(f"  Baseline: {baseline:.3f}")
        print(f"  Advantages: {[f'{float(a):+.3f}' for a in advantages]}")
        print(f"  Best: #{best_idx+1} (advantage: {advantages[best_idx]:+.3f})")
        
        # DIAGNOSTIC: Advantage effectiveness analysis
        advantage_std = jnp.std(advantages)
        max_advantage = jnp.max(jnp.abs(advantages))
        non_zero_advantages = jnp.sum(jnp.abs(advantages) > 0.1)
        
        print(f"\n🔍 ADVANTAGE ANALYSIS:")
        print(f"  Advantage std: {advantage_std:.6f}")
        print(f"  Max |advantage|: {max_advantage:.3f}")
        print(f"  Advantage range: [{jnp.min(advantages):.3f}, {jnp.max(advantages):.3f}]")
        print(f"  Meaningful advantages (>0.1): {non_zero_advantages}/{len(advantages)}")
        
        if advantage_std < 0.01:
            print(f"  ❌ CRITICAL: Advantages too uniform - no learning signal!")
        elif max_advantage > 10.0:
            print(f"  ⚠️ WARNING: Very large advantages - may cause instability")
        else:
            print(f"  ✅ Advantage distribution looks reasonable")


def create_unified_grpo_trainer(config: Union[DictConfig, Dict[str, Any], None] = None,
                                pretrained_surrogate: Optional[Dict[str, Any]] = None,
                                **kwargs) -> UnifiedGRPOTrainer:
    """
    Factory function to create unified GRPO trainer.
    
    Can be called with either a config dict/DictConfig or keyword arguments.
    
    Args:
        config: Configuration dictionary or DictConfig
        pretrained_surrogate: Optional dict with 'net' and 'params' for pre-trained surrogate
        **kwargs: Individual parameters if not using config
        
    Returns:
        Initialized UnifiedGRPOTrainer with all enhancements
    """
    # Create unified trainer (now includes all enhancements)
    if config is not None:
        trainer = UnifiedGRPOTrainer(config=config)
    else:
        trainer = UnifiedGRPOTrainer(config=None, **kwargs)
    
    # Override with pre-trained surrogate if provided
    if pretrained_surrogate and trainer.use_surrogate:
        logger.info("Overriding surrogate with pre-trained model")
        trainer._override_surrogate(pretrained_surrogate)
    
    return trainer