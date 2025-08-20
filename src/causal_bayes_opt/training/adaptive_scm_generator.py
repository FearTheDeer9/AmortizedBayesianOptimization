"""
Adaptive SCM Generator for Joint Training.

This module provides adaptive SCM generation and rotation logic,
integrating with the curriculum factory for progressive difficulty.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass
import random
import numpy as np

from .curriculum_factory import SCMCurriculumFactory
from ..experiments.benchmark_scms import (
    create_dense_scm,
    create_sparse_scm,
    create_chain_scm,
    create_fork_scm,
    create_collider_scm
)

logger = logging.getLogger(__name__)


@dataclass
class SCMPerformanceHistory:
    """Track performance history for an SCM type."""
    scm_type: str
    n_variables: int
    graph_type: str
    episodes_trained: int = 0
    mean_f1_score: float = 0.0
    mean_reward: float = 0.0
    convergence_episodes: Optional[int] = None
    
    def update(self, f1_score: float, reward: float):
        """Update running averages."""
        alpha = 0.1  # Exponential moving average
        self.mean_f1_score = (1 - alpha) * self.mean_f1_score + alpha * f1_score
        self.mean_reward = (1 - alpha) * self.mean_reward + alpha * reward
        self.episodes_trained += 1


class AdaptiveSCMGenerator:
    """
    Manages SCM generation and rotation for joint training.
    
    Features:
    - Integrates with curriculum factory for progressive difficulty
    - Tracks performance per SCM type
    - Implements adaptive rotation logic
    - Supports both fixed and generated SCMs
    """
    
    def __init__(self,
                 curriculum_factory: Optional[SCMCurriculumFactory] = None,
                 fixed_scms: Optional[Union[List[Any], Dict[str, Any]]] = None,
                 rotation_strategy: str = "performance",
                 seed: int = 42):
        """
        Initialize adaptive SCM generator.
        
        Args:
            curriculum_factory: Optional curriculum factory for progressive SCMs
            fixed_scms: Optional fixed set of SCMs to rotate through
            rotation_strategy: Strategy for rotation ("performance", "fixed", "random")
            seed: Random seed for reproducibility
        """
        self.curriculum_factory = curriculum_factory
        self.rotation_strategy = rotation_strategy
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed)
        
        # Initialize SCM source
        if curriculum_factory is not None:
            self.use_curriculum = True
            self.current_level = 1
            logger.info("Using curriculum-based SCM generation")
        elif fixed_scms is not None:
            self.use_curriculum = False
            if isinstance(fixed_scms, dict):
                self.scm_list = list(fixed_scms.values())
                self.scm_names = list(fixed_scms.keys())
            elif isinstance(fixed_scms, list):
                # Check if list contains tuples (name, scm) or just scms
                if fixed_scms and isinstance(fixed_scms[0], tuple) and len(fixed_scms[0]) == 2:
                    # List of (name, scm) tuples
                    self.scm_names = [name for name, _ in fixed_scms]
                    self.scm_list = [scm for _, scm in fixed_scms]
                else:
                    # List of just SCMs
                    self.scm_list = fixed_scms
                    self.scm_names = [f"scm_{i}" for i in range(len(fixed_scms))]
            else:
                # Assume it's a generator function
                self.scm_generator_fn = fixed_scms
                self.scm_list = []
                self.scm_names = []
            self.current_scm_idx = 0
            logger.info(f"Using fixed SCM rotation with {len(self.scm_list)} SCMs")
        else:
            # Default: create a simple generator
            self.use_curriculum = False
            self.scm_generator_fn = self._create_default_generator()
            self.scm_list = []
            self.scm_names = []
            logger.info("Using default SCM generator")
        
        # Performance tracking
        self.performance_history = {}  # (scm_type, n_vars) -> SCMPerformanceHistory
        self.episode_count = 0
        self.current_scm = None
        self.current_scm_metadata = {}
        
        # Rotation parameters
        self.min_episodes_per_scm = 10
        self.max_episodes_per_scm = 30
        self.convergence_threshold = 0.9  # F1 score for considering converged
        
        logger.info(f"Initialized AdaptiveSCMGenerator with strategy: {rotation_strategy}")
    
    def _create_default_generator(self) -> Callable:
        """Create a default SCM generator function."""
        def generator():
            n_vars = self.rng.randint(3, 8)
            graph_type = self.rng.choice(['sparse', 'chain', 'fork', 'collider'])
            
            if graph_type == 'sparse':
                return create_sparse_scm(
                    num_vars=n_vars,
                    edge_prob=0.3,
                    noise_scale=1.0
                )
            elif graph_type == 'chain':
                return create_chain_scm(chain_length=max(3, n_vars))
            elif graph_type == 'fork':
                return create_fork_scm(noise_scale=1.0)
            else:  # collider
                return create_collider_scm(noise_scale=1.0)
        
        return generator
    
    def request_new_scm(self, 
                       performance_metrics: Optional[Dict[str, float]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Request a new SCM based on performance and strategy.
        
        Args:
            performance_metrics: Optional performance on previous SCM
                - 'mean_reward': Average reward achieved
                - 'mean_f1': Average F1 score for structure learning
                
        Returns:
            Tuple of (SCM, metadata dict)
        """
        # Update performance history if provided
        if performance_metrics and self.current_scm_metadata:
            self._update_performance_history(performance_metrics)
        
        # Get new SCM based on source
        if self.use_curriculum:
            scm, metadata = self._get_curriculum_scm(performance_metrics)
        elif self.scm_list:
            scm, metadata = self._get_fixed_scm()
        else:
            scm, metadata = self._generate_scm()
        
        # Update tracking
        self.current_scm = scm
        self.current_scm_metadata = metadata
        self.episode_count += 1
        
        logger.info(f"Generated new SCM: {metadata.get('name', 'unknown')} "
                   f"(type: {metadata.get('graph_type', 'unknown')}, "
                   f"n_vars: {metadata.get('n_variables', 'unknown')})")
        
        return scm, metadata
    
    def _get_curriculum_scm(self, 
                          performance_metrics: Optional[Dict[str, float]]) -> Tuple[Any, Dict]:
        """Get SCM from curriculum factory."""
        # Determine if we should advance level
        if performance_metrics and self.should_advance_curriculum(performance_metrics):
            self.current_level = min(self.current_level + 1, 15)  # Max level 15
            logger.info(f"Advanced curriculum to level {self.current_level}")
        
        # Get SCM from curriculum
        scm, _ = self.curriculum_factory.get_next_scm()
        
        # Extract metadata
        from ..data_structures.scm import get_variables
        n_vars = len(get_variables(scm))
        
        metadata = {
            'name': f"curriculum_L{self.current_level}_ep{self.episode_count}",
            'curriculum_level': self.current_level,
            'n_variables': n_vars,
            'graph_type': self._infer_graph_type(scm),
            'source': 'curriculum'
        }
        
        return scm, metadata
    
    def _get_fixed_scm(self) -> Tuple[Any, Dict]:
        """Get next SCM from fixed list."""
        logger.info(f"_get_fixed_scm called, rotation_strategy={self.rotation_strategy}")
        if self.rotation_strategy == "random":
            idx = self.rng.randint(0, len(self.scm_list) - 1)
        else:
            # Sequential rotation
            idx = self.current_scm_idx
            self.current_scm_idx = (self.current_scm_idx + 1) % len(self.scm_list)
        
        logger.info(f"Selected SCM index: {idx}")
        scm = self.scm_list[idx]
        logger.info(f"Got SCM from list")
        
        # Extract metadata
        logger.info("Importing get_variables...")
        from ..data_structures.scm import get_variables
        logger.info("Calling get_variables...")
        n_vars = len(get_variables(scm))
        logger.info(f"SCM has {n_vars} variables")
        
        metadata = {
            'name': self.scm_names[idx],
            'index': idx,
            'n_variables': n_vars,
            'graph_type': self._infer_graph_type(scm),
            'source': 'fixed'
        }
        
        return scm, metadata
    
    def _generate_scm(self) -> Tuple[Any, Dict]:
        """Generate a new SCM using generator function."""
        scm = self.scm_generator_fn()
        
        # Extract metadata
        from ..data_structures.scm import get_variables
        n_vars = len(get_variables(scm))
        
        metadata = {
            'name': f"generated_ep{self.episode_count}",
            'n_variables': n_vars,
            'graph_type': self._infer_graph_type(scm),
            'source': 'generated'
        }
        
        return scm, metadata
    
    def _infer_graph_type(self, scm) -> str:
        """Infer graph type from SCM structure."""
        from ..data_structures.scm import get_variables, get_parents
        
        variables = list(get_variables(scm))
        n_vars = len(variables)
        
        # Count edges and analyze structure
        total_edges = 0
        max_parents = 0
        max_children = 0
        parent_counts = {}
        child_counts = {v: 0 for v in variables}
        
        for var in variables:
            parents = get_parents(scm, var)
            n_parents = len(parents)
            parent_counts[var] = n_parents
            total_edges += n_parents
            max_parents = max(max_parents, n_parents)
            
            for parent in parents:
                child_counts[parent] += 1
        
        max_children = max(child_counts.values())
        
        # Classify based on structure
        if total_edges == 0:
            return "empty"
        elif total_edges == n_vars - 1 and max_parents == 1 and max_children == 1:
            return "chain"
        elif max_children > 2 and max_parents == 1:
            return "fork"
        elif max_parents > 2 and max_children == 1:
            return "collider"
        elif total_edges < n_vars * 0.5:
            return "sparse"
        else:
            return "dense"
    
    def _update_performance_history(self, metrics: Dict[str, float]):
        """Update performance history for current SCM type."""
        if not self.current_scm_metadata:
            return
        
        # Create key for this SCM type
        graph_type = self.current_scm_metadata.get('graph_type', 'unknown')
        n_vars = self.current_scm_metadata.get('n_variables', 0)
        key = (graph_type, n_vars)
        
        # Update or create history
        if key not in self.performance_history:
            self.performance_history[key] = SCMPerformanceHistory(
                scm_type=graph_type,
                n_variables=n_vars,
                graph_type=graph_type
            )
        
        history = self.performance_history[key]
        history.update(
            f1_score=metrics.get('mean_f1', 0.0),
            reward=metrics.get('mean_reward', 0.0)
        )
        
        # Check for convergence
        if history.mean_f1_score > self.convergence_threshold:
            if history.convergence_episodes is None:
                history.convergence_episodes = history.episodes_trained
                logger.info(f"SCM type {key} converged after {history.convergence_episodes} episodes")
    
    def should_advance_curriculum(self, metrics: Dict[str, float]) -> bool:
        """Determine if curriculum should advance to next level."""
        if not self.use_curriculum:
            return False
        
        # Check performance thresholds
        f1_score = metrics.get('mean_f1', 0.0)
        reward = metrics.get('mean_reward', 0.0)
        
        # Advance if both structure learning and reward are good
        if f1_score > 0.8 and reward > 0.5:
            return True
        
        # Also check if we've spent enough time on current level
        current_level_episodes = sum(
            1 for meta in [self.current_scm_metadata]
            if meta.get('curriculum_level') == self.current_level
        )
        
        if current_level_episodes > 50:  # Max episodes per level
            return True
        
        return False
    
    def should_rotate(self,
                     episodes_on_current: int,
                     current_performance: Optional[Dict[str, float]] = None) -> bool:
        """
        Determine if it's time to rotate to a new SCM.
        
        Args:
            episodes_on_current: Number of episodes on current SCM
            current_performance: Current performance metrics
            
        Returns:
            Boolean indicating whether to rotate
        """
        # Always rotate after max episodes
        if episodes_on_current >= self.max_episodes_per_scm:
            logger.debug(f"Rotating SCM: reached max episodes ({self.max_episodes_per_scm})")
            return True
        
        # Don't rotate too quickly
        if episodes_on_current < self.min_episodes_per_scm:
            return False
        
        # Performance-based rotation
        if self.rotation_strategy == "performance" and current_performance:
            f1_score = current_performance.get('f1_score', 0.0)
            
            # Rotate if converged
            if f1_score > self.convergence_threshold:
                logger.debug(f"Rotating SCM: converged (F1={f1_score:.3f})")
                return True
            
            # Rotate if plateaued (check history)
            if self._is_plateaued(current_performance):
                logger.debug("Rotating SCM: performance plateaued")
                return True
        
        return False
    
    def _is_plateaued(self, metrics: Dict[str, float], window: int = 5) -> bool:
        """Check if performance has plateaued."""
        # Simple check - would need history tracking for proper implementation
        # For now, just return False
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        stats = {
            'episode_count': self.episode_count,
            'use_curriculum': self.use_curriculum,
            'rotation_strategy': self.rotation_strategy,
            'n_scm_types_seen': len(self.performance_history),
            'current_scm': self.current_scm_metadata.get('name', 'none')
        }
        
        if self.use_curriculum:
            stats['current_level'] = self.current_level
        
        if self.performance_history:
            # Aggregate performance across SCM types
            all_f1 = [h.mean_f1_score for h in self.performance_history.values()]
            all_rewards = [h.mean_reward for h in self.performance_history.values()]
            
            stats['mean_f1_across_types'] = np.mean(all_f1)
            stats['mean_reward_across_types'] = np.mean(all_rewards)
            
            # Find best performing SCM type
            best_type = max(
                self.performance_history.items(),
                key=lambda x: x[1].mean_f1_score
            )
            stats['best_scm_type'] = f"{best_type[0][0]}_{best_type[0][1]}vars"
            stats['best_f1'] = best_type[1].mean_f1_score
        
        return stats
    
    def reset(self):
        """Reset generator state."""
        self.episode_count = 0
        self.current_scm = None
        self.current_scm_metadata = {}
        self.performance_history.clear()
        
        if self.use_curriculum:
            self.current_level = 1
        elif self.scm_list:
            self.current_scm_idx = 0
        
        logger.info("Reset AdaptiveSCMGenerator state")
    
    def __repr__(self) -> str:
        """String representation."""
        source = "curriculum" if self.use_curriculum else "fixed" if self.scm_list else "generated"
        return (f"AdaptiveSCMGenerator(source={source}, "
                f"strategy={self.rotation_strategy}, "
                f"episodes={self.episode_count})")