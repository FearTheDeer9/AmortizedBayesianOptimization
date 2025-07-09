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


class PolicyFactory:
    """Factory for creating enriched policy networks."""
    
    def __init__(self, config: DictConfig, max_variables: int):
        self.config = config
        self.max_variables = max_variables
        
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
                policy_intermediate_dim=self.config.training.architecture.get('policy_intermediate_dim', None)
            )
            
            return network(
                enriched_history=enriched_history,
                target_variable_idx=target_variable_idx,
                is_training=is_training
            )
        
        policy_config = {
            'architecture': dict(self.config.training.architecture),
            'num_variables': self.max_variables,
            'variable_agnostic': True,
            'enriched_architecture': True
        }
        
        return hk.transform(policy_fn), policy_config


class SCMRotationManager:
    """Manages SCM rotation during training."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.scm_rotation = self._create_scm_rotation()
        self.max_variables = self._determine_max_variables()
        
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
    
    def _determine_max_variables(self) -> int:
        """Determine maximum variables across all SCMs."""
        max_vars = 0
        for name, scm in self.scm_rotation:
            variables = get_variables(scm)
            max_vars = max(max_vars, len(variables))
        return max_vars
    
    def get_current_scm(self, episode: int) -> Tuple[str, pyr.PMap]:
        """Get current SCM based on episode and rotation frequency."""
        rotation_frequency = self.config.experiment.scm_generation.rotation_frequency
        scm_idx = (episode // rotation_frequency) % len(self.scm_rotation)
        return self.scm_rotation[scm_idx]


class StateConverter:
    """Converts states to enriched representation."""
    
    def __init__(self, config: DictConfig, max_variables: int):
        self.config = config
        self.max_variables = max_variables
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
                       is_final: bool = False) -> Path:
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
        
        if metrics:
            checkpoint_data['metrics'] = {
                'mean_reward': metrics.mean_reward,
                'structure_accuracy': metrics.structure_accuracy,
                'optimization_improvement': metrics.optimization_improvement,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'scm_type': metrics.scm_type
            }
        
        checkpoint_file = checkpoint_path / "checkpoint.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint: {checkpoint_file}")
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
        
        return {
            'total_episodes': len(self.metrics_history),
            'training_time': total_time,
            'final_reward': rewards[-1],
            'mean_reward': float(jnp.mean(jnp.array(rewards))),
            'final_accuracy': accuracies[-1],
            'mean_accuracy': float(jnp.mean(jnp.array(accuracies))),
            'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
            'episodes_per_second': len(self.metrics_history) / total_time
        }