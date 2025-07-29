"""
Modular GRPO Training Components

This module provides focused, single-responsibility components for GRPO training
following CLAUDE.md principles. Each component has a clear purpose and can be
tested independently.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import pyrsistent as pyr
import haiku as hk
from omegaconf import DictConfig

from ..acquisition.enriched.policy_heads import EnrichedAcquisitionPolicyNetwork
from ..acquisition.enriched.state_enrichment import EnrichedHistoryBuilder
from ..acquisition.rewards import create_default_reward_config, compute_verifiable_reward
from ..experiments.variable_scm_factory import VariableSCMFactory
from ..data_structures.scm import get_variables

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingMetrics:
    """Immutable training metrics."""
    episode: int
    mean_reward: float
    structure_accuracy: float
    optimization_improvement: float
    policy_loss: float
    value_loss: float
    scm_type: str
    # New structure learning metrics
    f1_score: Optional[float] = None
    true_parent_likelihood: Optional[float] = None
    shd: Optional[int] = None
    marginal_probs: Optional[Dict[str, float]] = None


class PolicyFactory:
    """Factory for creating enriched policy networks."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        
    def create_policy(self) -> Tuple[Any, Dict[str, Any]]:
        """Create enriched policy network."""
        def policy_fn(enriched_history: jnp.ndarray, 
                     target_variable_idx: int = 0,
                     is_training: bool = False) -> Dict[str, jnp.ndarray]:
            
            network = EnrichedAcquisitionPolicyNetwork(
                num_layers=self.config.training.architecture.num_layers,
                num_heads=self.config.training.architecture.num_heads,
                hidden_dim=self.config.training.architecture.hidden_dim,
                key_size=self.config.training.architecture.key_size,
                widening_factor=self.config.training.architecture.widening_factor,
                dropout=self.config.training.architecture.dropout,
                policy_intermediate_dim=self.config.training.architecture.get('policy_intermediate_dim', None),
                use_role_based_projection=self.config.training.architecture.get('use_role_based_projection', True)
            )
            
            return network(
                enriched_history=enriched_history,
                target_variable_idx=target_variable_idx,
                is_training=is_training
            )
        
        # Create architecture config with explicit use_role_based_projection
        arch_config = dict(self.config.training.architecture)
        if 'use_role_based_projection' not in arch_config:
            arch_config['use_role_based_projection'] = True
        
        policy_config = {
            'architecture': arch_config,
            'variable_agnostic': True,
            'enriched_architecture': True
        }
        
        return hk.transform(policy_fn), policy_config


class SCMRotationManager:
    """Manages SCM rotation during training with support for dynamic progression."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.scm_rotation = self._create_scm_rotation()
        self.current_scm_index = 0
        self.episode_count_on_current_scm = 0
        self.dynamic_progression_enabled = config.get('training', {}).get(
            'early_stopping_enabled', False
        )
        
    def _create_scm_rotation(self) -> List[Tuple[str, pyr.PMap]]:
        """Create SCM rotation for training."""
        scm_config = self.config.experiment.scm_generation
        
        if scm_config.use_variable_factory:
            return self._create_variable_scms(scm_config)
        else:
            return self._create_fallback_scms(scm_config)
    
    def _create_variable_scms(self, scm_config) -> List[Tuple[str, pyr.PMap]]:
        """Create SCMs using variable factory."""
        factory = VariableSCMFactory(seed=self.config.seed)
        
        scms = []
        variable_range = scm_config.variable_range
        structure_types = scm_config.structure_types
        
        for num_vars in range(variable_range[0], variable_range[1] + 1):
            for structure_type in structure_types:
                scm = factory.create_variable_scm(
                    num_variables=num_vars,
                    structure_type=structure_type
                )
                name = f"{structure_type}_{num_vars}var"
                scms.append((name, scm))
                
        logger.info(f"Created {len(scms)} variable SCMs for training")
        return scms
    
    def _create_fallback_scms(self, scm_config) -> List[Tuple[str, pyr.PMap]]:
        """Create fallback SCMs if factory is not available."""
        from ..experiments.benchmark_scms import create_fork_scm, create_chain_scm, create_collider_scm
        
        scms = []
        fallback_names = scm_config.get('fallback_scms', ['fork_3var', 'chain_3var', 'collider_3var'])
        
        for scm_name in fallback_names:
            try:
                if scm_name == 'fork_3var':
                    scm = create_fork_scm(noise_scale=1.0, target="Y")
                    scms.append(("fork_3var", scm))
                    
                elif scm_name == 'chain_3var':
                    scm = create_chain_scm(chain_length=3, coefficient=1.5, noise_scale=1.0)
                    scms.append(("chain_3var", scm))
                    
                elif scm_name == 'collider_3var':
                    scm = create_collider_scm(noise_scale=1.0)  # No target parameter
                    scms.append(("collider_3var", scm))
                    
                else:
                    logger.warning(f"Unknown fallback SCM: {scm_name}")
                    
            except Exception as e:
                logger.error(f"Failed to create fallback SCM {scm_name}: {e}")
        
        if not scms:
            # If all else fails, create at least one simple SCM
            logger.warning("Creating minimal fallback SCM")
            scm = create_fork_scm(noise_scale=1.0, target="Y")
            scms.append(("fallback_fork", scm))
            
        logger.info(f"Created {len(scms)} fallback SCMs for training")
        return scms
    
    def get_current_scm(self, episode: int) -> Tuple[str, pyr.PMap]:
        """Get current SCM based on episode and rotation strategy."""
        if self.dynamic_progression_enabled:
            # Use dynamic progression - return current SCM based on internal state
            # NOTE: Episode counting is now handled by increment_episode_count()
            return self.scm_rotation[self.current_scm_index]
        else:
            # Use fixed rotation frequency
            rotation_frequency = self.config.experiment.scm_generation.rotation_frequency
            scm_idx = (episode // rotation_frequency) % len(self.scm_rotation)
            return self.scm_rotation[scm_idx]
    
    def increment_episode_count(self) -> None:
        """Increment the episode count for the current SCM.
        
        This should be called once per episode, not every time get_current_scm is called.
        """
        if self.dynamic_progression_enabled:
            self.episode_count_on_current_scm += 1
    
    def advance_to_next_scm(self) -> bool:
        """
        Advance to the next SCM in rotation.
        
        Returns:
            True if advanced, False if at the end of rotation
        """
        self.episode_count_on_current_scm = 0
        self.current_scm_index = (self.current_scm_index + 1) % len(self.scm_rotation)
        return self.current_scm_index != 0  # False when we wrap around
    
    def should_rotate(self, converged: bool) -> bool:
        """
        Check if we should rotate to the next SCM.
        
        Args:
            converged: Whether current SCM has converged
            
        Returns:
            True if we should rotate
        """
        if not self.dynamic_progression_enabled:
            # Fixed rotation - check episode count
            rotation_frequency = self.config.experiment.scm_generation.rotation_frequency
            return self.episode_count_on_current_scm >= rotation_frequency
        else:
            # Dynamic rotation - rotate if converged
            return converged
    
    def get_current_scm_info(self) -> Dict[str, any]:
        """Get information about current SCM training."""
        return {
            "scm_index": self.current_scm_index,
            "scm_name": self.scm_rotation[self.current_scm_index][0],
            "episodes_on_current": self.episode_count_on_current_scm,
            "total_scms": len(self.scm_rotation),
            "dynamic_progression": self.dynamic_progression_enabled
        }


class StateConverter:
    """Converts states to enriched representation."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.history_builder = EnrichedHistoryBuilder(
            standardize_values=config.training.state_config.get('standardize_values', True),
            include_temporal_features=config.training.state_config.get('include_temporal_features', True),
            max_history_size=config.training.state_config.get('max_history_size', 100),
            support_variable_scms=True,
            num_channels=config.training.state_config.get('num_channels', 5)  # Use configured channels
        )
    
    def convert_state_to_enriched_input(self, state: Any) -> jnp.ndarray:
        """Convert acquisition state to enriched input tensor."""
        try:
            enriched_history, variable_mask = self.history_builder.build_enriched_history(state)
            
            # No padding to max_variables - use actual SCM size for dynamic policy network
            # This enables per-variable encoding without wasted computation
            return enriched_history
            
        except Exception as e:
            logger.error(f"State conversion failed: {e}")
            # Return fallback tensor - use single variable as minimal fallback
            logger.warning("Using single-variable fallback tensor due to state conversion failure")
            return jnp.zeros((
                self.config.training.state_config.get('max_history_size', 100),
                1,  # Single variable fallback instead of max_variables
                self.config.training.state_config.get('num_channels', 5)
            ))
    


class CheckpointManager:
    """Manages model checkpointing."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.checkpoint_dir = Path(config.logging.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       policy_params: Any,
                       policy_config: Dict[str, Any],
                       episode: int,
                       metrics: Optional[TrainingMetrics] = None,
                       is_final: bool = False,
                       surrogate_params: Optional[Any] = None) -> Path:
        """Save training checkpoint."""
        if is_final:
            checkpoint_name = f"enriched_grpo_final"
        else:
            checkpoint_name = f"enriched_grpo_episode_{episode}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        checkpoint_data = {
            'policy_params': policy_params,
            'policy_config': policy_config,
            'training_config': dict(self.config.training),
            'episode': episode,
            'is_final': is_final,
            'enriched_architecture': True
        }
        
        # Include surrogate params if provided
        if surrogate_params is not None:
            checkpoint_data['surrogate_params'] = surrogate_params
            logger.info("Including surrogate parameters in checkpoint")
        
        if metrics:
            checkpoint_data['metrics'] = {
                'mean_reward': metrics.mean_reward,
                'structure_accuracy': metrics.structure_accuracy,
                'optimization_improvement': metrics.optimization_improvement,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'scm_type': metrics.scm_type,
                'f1_score': metrics.f1_score,
                'true_parent_likelihood': metrics.true_parent_likelihood,
                'shd': metrics.shd,
                'marginal_probs': metrics.marginal_probs
            }
        
        checkpoint_file = checkpoint_path / "checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save policy params separately for Phase 2 integration
        policy_only_file = checkpoint_path / "policy_params.pkl"
        policy_only_data = {
            'policy_params': policy_params,
            'policy_config': policy_config,
            'enriched_architecture': True,
            'episode': episode,
            'is_final': is_final
        }
        with open(policy_only_file, 'wb') as f:
            pickle.dump(policy_only_data, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_file}")
        logger.info(f"Saved policy params: {policy_only_file}")
        return checkpoint_path


class MetricsCollector:
    """Collects and manages training metrics."""
    
    def __init__(self):
        self.metrics_history = pyr.v()  # Immutable vector
    
    def add_metrics(self, metrics: TrainingMetrics) -> 'MetricsCollector':
        """Add metrics and return new collector (immutable)."""
        new_collector = MetricsCollector()
        new_collector.metrics_history = self.metrics_history.append(metrics)
        return new_collector
    
    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get most recent metrics."""
        if len(self.metrics_history) > 0:
            return self.metrics_history[-1]
        return None
    
    def analyze_performance(self, total_time: float) -> Dict[str, Any]:
        """Analyze overall training performance."""
        if len(self.metrics_history) == 0:
            return {'status': 'no_metrics'}
        
        rewards = [m.mean_reward for m in self.metrics_history]
        accuracies = [m.structure_accuracy for m in self.metrics_history]
        
        # Collect structure learning metrics
        f1_scores = [m.f1_score for m in self.metrics_history if m.f1_score is not None]
        parent_likelihoods = [m.true_parent_likelihood for m in self.metrics_history if m.true_parent_likelihood is not None]
        shd_values = [m.shd for m in self.metrics_history if m.shd is not None]
        
        analysis = {
            'total_episodes': len(self.metrics_history),
            'training_time': total_time,
            'final_reward': rewards[-1],
            'mean_reward': float(jnp.mean(jnp.array(rewards))),
            'final_accuracy': accuracies[-1],
            'mean_accuracy': float(jnp.mean(jnp.array(accuracies))),
            'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
            'episodes_per_second': len(self.metrics_history) / total_time
        }
        
        # Add structure learning metrics if available
        if f1_scores:
            analysis.update({
                'final_f1_score': f1_scores[-1],
                'max_f1_score': float(jnp.max(jnp.array(f1_scores))),
                'mean_f1_score': float(jnp.mean(jnp.array(f1_scores))),
                'f1_improvement': f1_scores[-1] - f1_scores[0] if len(f1_scores) > 1 else 0.0
            })
        
        if parent_likelihoods:
            analysis.update({
                'final_parent_likelihood': parent_likelihoods[-1],
                'max_parent_likelihood': float(jnp.max(jnp.array(parent_likelihoods))),
                'mean_parent_likelihood': float(jnp.mean(jnp.array(parent_likelihoods))),
                'parent_likelihood_improvement': parent_likelihoods[-1] - parent_likelihoods[0] if len(parent_likelihoods) > 1 else 0.0
            })
        
        if shd_values:
            analysis.update({
                'final_shd': shd_values[-1],
                'min_shd': int(jnp.min(jnp.array(shd_values))),
                'mean_shd': float(jnp.mean(jnp.array(shd_values)))
            })
        
        return analysis