"""
SCM Curriculum Learning Factory

This module provides a curriculum learning factory for progressive training
on SCMs of increasing complexity. The curriculum progresses from simple
structured graphs (chains, forks) to complex random graphs (Erdős-Rényi,
small-world, scale-free networks).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque
import random as py_random
import numpy as np
import pyrsistent as pyr
import yaml
from pathlib import Path

from ..experiments.benchmark_scms import (
    create_chain_scm,
    create_fork_scm,
    create_collider_scm,
    create_diamond_scm,
    create_butterfly_scm,
    create_dense_scm,
    create_sparse_scm,
    create_erdos_renyi_scm,
    create_scale_free_scm,
    create_small_world_scm
)

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    level: int
    name: str
    n_vars_range: Tuple[int, int]
    graph_types: List[str]
    edge_prob: float = 0.3
    episodes: int = 50
    f1_threshold: float = 0.8
    # Graph-specific parameters
    rewiring_prob: float = 0.1  # For small-world
    m_parameter: int = 2  # For scale-free (edges per new node)
    k_neighbors: int = 4  # For small-world (initial neighbors)
    noise_scale: float = 1.0
    coeff_range: Tuple[float, float] = (-2.0, 2.0)


@dataclass
class PerformanceMetrics:
    """Metrics for tracking performance on current stage."""
    episodes_completed: int = 0
    best_f1_score: float = 0.0
    recent_f1_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=10))
    convergence_count: int = 0
    total_scms_seen: int = 0
    
    def update(self, f1_score: float = None, reward: float = None):
        """Update metrics with new episode results."""
        self.episodes_completed += 1
        if f1_score is not None:
            self.recent_f1_scores.append(f1_score)
            self.best_f1_score = max(self.best_f1_score, f1_score)
        if reward is not None:
            self.recent_rewards.append(reward)
    
    def get_mean_f1(self) -> float:
        """Get mean F1 score from recent episodes."""
        if not self.recent_f1_scores:
            return 0.0
        return np.mean(list(self.recent_f1_scores))
    
    def get_mean_reward(self) -> float:
        """Get mean reward from recent episodes."""
        if not self.recent_rewards:
            return 0.0
        return np.mean(list(self.recent_rewards))


class SCMCurriculumFactory:
    """
    Factory for generating SCMs with progressive curriculum learning.
    
    This factory manages progression through stages of increasing complexity,
    from simple structured graphs to complex random networks.
    """
    
    def __init__(self, 
                 curriculum_config: Optional[Dict[str, Any]] = None,
                 config_path: Optional[str] = None,
                 start_level: int = 1,
                 max_level: int = 15,
                 mode: str = "progressive",
                 seed: int = 42):
        """
        Initialize curriculum factory.
        
        Args:
            curriculum_config: Direct curriculum configuration dictionary
            config_path: Path to YAML configuration file
            start_level: Starting curriculum level (1-based)
            max_level: Maximum curriculum level to reach
            mode: Curriculum mode ("progressive", "fixed", "adaptive")
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.seed = seed
        self.start_level = start_level
        self.max_level = max_level
        py_random.seed(seed)
        np.random.seed(seed)
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif curriculum_config:
            self.config = curriculum_config
        else:
            self.config = self._get_default_config()
        
        # Initialize stages
        self.stages = self._create_stages(self.config)
        self.current_level = start_level
        self.current_stage = self.stages[self.current_level - 1]
        
        # Performance tracking
        self.stage_metrics: Dict[int, PerformanceMetrics] = {
            level: PerformanceMetrics() for level in range(1, len(self.stages) + 1)
        }
        self.global_episode_count = 0
        
        # Graph generators mapping
        self._init_generators()
        
        logger.info(f"Initialized curriculum factory with {len(self.stages)} stages")
        logger.info(f"Starting at level {self.current_level}: {self.current_stage.name}")
    
    def _init_generators(self):
        """Initialize mapping of graph type names to generator functions."""
        self.graph_generators = {
            # Simple structures
            'chain': self._generate_chain,
            'fork': self._generate_fork,
            'collider': self._generate_collider,
            'diamond': self._generate_diamond,
            'butterfly': self._generate_butterfly,
            # Density-based
            'dense': self._generate_dense,
            'sparse': self._generate_sparse,
            'sparse_random': self._generate_sparse,
            # Advanced graph types
            'erdos_renyi': self._generate_erdos_renyi,
            'small_world': self._generate_small_world,
            'scale_free': self._generate_scale_free,
            # Mixed/meta types
            'mixed_simple': self._generate_mixed_simple,
            'hierarchical': self._generate_hierarchical,
            'bipartite': self._generate_bipartite,
            'layered': self._generate_layered,
            'all': self._generate_any
        }
    
    def get_next_scm(self, performance_metrics: Optional[Dict[str, Any]] = None) -> Tuple[pyr.PMap, Dict[str, Any]]:
        """
        Get next SCM based on curriculum progression.
        
        Args:
            performance_metrics: Optional metrics from last episode
            
        Returns:
            Tuple of (SCM, metadata dict with curriculum info)
        """
        # Update metrics if provided
        if performance_metrics:
            self._update_metrics(performance_metrics)
            
            # Check for progression in progressive mode
            if self.mode == "progressive":
                if self._should_advance():
                    self._advance_stage()
        
        # Generate SCM for current stage
        scm = self._generate_scm_for_stage(self.current_stage)
        
        # Add curriculum metadata
        metadata = {
            'curriculum_level': self.current_level,
            'curriculum_stage': self.current_stage.name,
            'episodes_at_level': self.stage_metrics[self.current_level].episodes_completed,
            'global_episode': self.global_episode_count
        }
        
        self.global_episode_count += 1
        self.stage_metrics[self.current_level].total_scms_seen += 1
        
        return scm, metadata
    
    def _generate_scm_for_stage(self, stage: CurriculumStage) -> pyr.PMap:
        """Generate an SCM appropriate for the given stage."""
        # Select graph type
        if len(stage.graph_types) == 1:
            graph_type = stage.graph_types[0]
        else:
            graph_type = py_random.choice(stage.graph_types)
        
        # Select number of variables
        n_vars = py_random.randint(stage.n_vars_range[0], stage.n_vars_range[1])
        
        # Generate using appropriate generator
        if graph_type not in self.graph_generators:
            logger.warning(f"Unknown graph type: {graph_type}, using sparse random")
            graph_type = 'sparse_random'
        
        generator = self.graph_generators[graph_type]
        scm = generator(n_vars, stage)
        
        # Log generation
        logger.debug(f"Generated {graph_type} SCM with {n_vars} variables for stage {stage.name}")
        
        return scm
    
    def _should_advance(self) -> bool:
        """Check if we should advance to next stage."""
        metrics = self.stage_metrics[self.current_level]
        stage = self.current_stage
        
        # Don't advance beyond max level
        if self.current_level >= self.max_level:
            return False
        
        # Check minimum episodes
        if metrics.episodes_completed < min(20, stage.episodes // 2):
            return False
        
        # Check performance threshold
        mean_f1 = metrics.get_mean_f1()
        if mean_f1 >= stage.f1_threshold:
            metrics.convergence_count += 1
            if metrics.convergence_count >= 5:  # Consistent performance
                return True
        else:
            metrics.convergence_count = 0
        
        # Check max episodes for this stage
        if metrics.episodes_completed >= stage.episodes:
            return True
        
        return False
    
    def _advance_stage(self):
        """Advance to next curriculum stage."""
        if self.current_level < len(self.stages):
            old_stage = self.current_stage
            self.current_level += 1
            self.current_stage = self.stages[self.current_level - 1]
            
            logger.info(f"Advancing curriculum: Level {self.current_level - 1} ({old_stage.name}) "
                       f"-> Level {self.current_level} ({self.current_stage.name})")
            logger.info(f"  Previous stage metrics: "
                       f"F1={self.stage_metrics[self.current_level - 1].get_mean_f1():.3f}, "
                       f"Episodes={self.stage_metrics[self.current_level - 1].episodes_completed}")
    
    def _update_metrics(self, performance_metrics: Dict[str, Any]):
        """Update metrics for current stage."""
        metrics = self.stage_metrics[self.current_level]
        
        # Extract relevant metrics
        f1_score = None
        if 'structure_metrics' in performance_metrics:
            f1_score = performance_metrics['structure_metrics'].get('f1_score')
        
        reward = performance_metrics.get('mean_reward')
        
        metrics.update(f1_score=f1_score, reward=reward)
    
    # Generator methods for different graph types
    def _generate_chain(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate chain SCM."""
        return create_chain_scm(
            chain_length=n_vars,
            coefficient=py_random.uniform(*stage.coeff_range),
            noise_scale=stage.noise_scale
        )
    
    def _generate_fork(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate fork SCM (fixed 3 variables)."""
        return create_fork_scm(noise_scale=stage.noise_scale)
    
    def _generate_collider(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate collider SCM (fixed 3 variables)."""
        return create_collider_scm(noise_scale=stage.noise_scale)
    
    def _generate_diamond(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate diamond SCM (fixed 4 variables)."""
        return create_diamond_scm(noise_scale=stage.noise_scale)
    
    def _generate_butterfly(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate butterfly SCM (fixed 5 variables)."""
        return create_butterfly_scm(noise_scale=stage.noise_scale)
    
    def _generate_dense(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate dense SCM."""
        edge_prob = min(0.7, stage.edge_prob * 2)  # Dense = higher edge probability
        return create_dense_scm(
            num_vars=n_vars,
            edge_prob=edge_prob,
            noise_scale=stage.noise_scale
        )
    
    def _generate_sparse(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate sparse SCM."""
        return create_sparse_scm(
            num_vars=n_vars,
            edge_prob=stage.edge_prob,
            noise_scale=stage.noise_scale
        )
    
    def _generate_erdos_renyi(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate Erdős-Rényi random graph SCM."""
        return create_erdos_renyi_scm(
            n_nodes=n_vars,
            edge_prob=stage.edge_prob,
            coeff_range=stage.coeff_range,
            noise_scale=stage.noise_scale,
            seed=py_random.randint(0, 10000)
        )
    
    def _generate_small_world(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate small-world (Watts-Strogatz) SCM."""
        k = min(stage.k_neighbors, n_vars - 1)  # Ensure k < n
        return create_small_world_scm(
            n_nodes=n_vars,
            k_neighbors=k,
            rewiring_prob=stage.rewiring_prob,
            coeff_range=stage.coeff_range,
            noise_scale=stage.noise_scale,
            seed=py_random.randint(0, 10000)
        )
    
    def _generate_scale_free(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate scale-free (Barabási-Albert) SCM."""
        # For scale-free, we use alpha/beta/gamma parameters
        return create_scale_free_scm(
            n_nodes=n_vars,
            alpha=0.41,  # Default parameters
            beta=0.54,
            gamma=0.05,
            coeff_range=stage.coeff_range,
            noise_scale=stage.noise_scale,
            seed=py_random.randint(0, 10000)
        )
    
    def _generate_mixed_simple(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate a mix of simple structures."""
        simple_types = ['chain', 'fork', 'collider', 'diamond', 'sparse']
        graph_type = py_random.choice(simple_types)
        return self.graph_generators[graph_type](n_vars, stage)
    
    def _generate_hierarchical(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate hierarchical structure (layers with connections)."""
        # For now, use sparse with lower edge probability
        return create_sparse_scm(
            num_vars=n_vars,
            edge_prob=stage.edge_prob * 0.7,
            noise_scale=stage.noise_scale
        )
    
    def _generate_bipartite(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate bipartite-like structure."""
        # For now, use sparse structure
        return create_sparse_scm(
            num_vars=n_vars,
            edge_prob=stage.edge_prob * 0.8,
            noise_scale=stage.noise_scale
        )
    
    def _generate_layered(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate layered DAG structure."""
        # For now, use chain with some additional edges
        if n_vars <= 3:
            return create_chain_scm(chain_length=n_vars, noise_scale=stage.noise_scale)
        else:
            return create_sparse_scm(
                num_vars=n_vars,
                edge_prob=stage.edge_prob * 0.6,
                noise_scale=stage.noise_scale
            )
    
    def _generate_any(self, n_vars: int, stage: CurriculumStage) -> pyr.PMap:
        """Generate any type of graph."""
        all_types = list(self.graph_generators.keys())
        # Remove meta types to avoid recursion
        exclude = ['all', 'mixed_simple']
        available_types = [t for t in all_types if t not in exclude]
        graph_type = py_random.choice(available_types)
        return self.graph_generators[graph_type](n_vars, stage)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load curriculum configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config.get('curriculum', self._get_default_config())
    
    def _create_stages(self, config: Dict[str, Any]) -> List[CurriculumStage]:
        """Create curriculum stages from configuration."""
        stages = []
        
        for stage_config in config.get('stages', []):
            stage = CurriculumStage(
                level=stage_config['level'],
                name=stage_config['name'],
                n_vars_range=tuple(stage_config['n_vars']),
                graph_types=stage_config['graph_types'],
                edge_prob=stage_config.get('edge_prob', 0.3),
                episodes=stage_config.get('episodes', 50),
                f1_threshold=stage_config.get('f1_threshold', 0.8),
                rewiring_prob=stage_config.get('rewiring_prob', 0.1),
                m_parameter=stage_config.get('m_parameter', 2),
                k_neighbors=stage_config.get('k_neighbors', 4),
                noise_scale=stage_config.get('noise_scale', 1.0),
                coeff_range=tuple(stage_config.get('coeff_range', [-2.0, 2.0]))
            )
            stages.append(stage)
        
        return stages
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default curriculum configuration."""
        return {
            'mode': 'progressive',
            'stages': [
                # Level 1-3: Foundations
                {'level': 1, 'name': 'simple_structures', 'n_vars': [3, 3],
                 'graph_types': ['chain', 'fork', 'collider'],
                 'edge_prob': 0.3, 'episodes': 30, 'f1_threshold': 0.9},
                
                {'level': 2, 'name': 'basic_4node', 'n_vars': [4, 4],
                 'graph_types': ['diamond', 'chain', 'fork'],
                 'edge_prob': 0.3, 'episodes': 40, 'f1_threshold': 0.85},
                
                {'level': 3, 'name': 'mixed_small', 'n_vars': [4, 5],
                 'graph_types': ['mixed_simple'],
                 'edge_prob': 0.35, 'episodes': 50, 'f1_threshold': 0.85},
                
                # Level 4-6: Intermediate
                {'level': 4, 'name': 'medium_structured', 'n_vars': [6, 6],
                 'graph_types': ['butterfly', 'layered'],
                 'edge_prob': 0.3, 'episodes': 60, 'f1_threshold': 0.8},
                
                {'level': 5, 'name': 'medium_mixed', 'n_vars': [6, 7],
                 'graph_types': ['sparse_random'],
                 'edge_prob': 0.35, 'episodes': 70, 'f1_threshold': 0.8},
                
                {'level': 6, 'name': 'larger_structured', 'n_vars': [8, 9],
                 'graph_types': ['hierarchical', 'bipartite'],
                 'edge_prob': 0.25, 'episodes': 80, 'f1_threshold': 0.75},
                
                # Level 7-9: Complex with random graphs
                {'level': 7, 'name': 'erdos_renyi_small', 'n_vars': [10, 12],
                 'graph_types': ['erdos_renyi'],
                 'edge_prob': 0.2, 'episodes': 100, 'f1_threshold': 0.7},
                
                {'level': 8, 'name': 'small_world_intro', 'n_vars': [10, 12],
                 'graph_types': ['small_world'],
                 'rewiring_prob': 0.1, 'episodes': 100, 'f1_threshold': 0.7},
                
                {'level': 9, 'name': 'mixed_complex', 'n_vars': [12, 15],
                 'graph_types': ['erdos_renyi', 'small_world', 'scale_free'],
                 'edge_prob': 0.15, 'episodes': 120, 'f1_threshold': 0.65},
                
                # Level 10-12: Advanced
                {'level': 10, 'name': 'large_erdos_renyi', 'n_vars': [16, 20],
                 'graph_types': ['erdos_renyi'],
                 'edge_prob': 0.1, 'episodes': 150, 'f1_threshold': 0.6},
                
                {'level': 11, 'name': 'large_small_world', 'n_vars': [16, 20],
                 'graph_types': ['small_world'],
                 'rewiring_prob': 0.15, 'episodes': 150, 'f1_threshold': 0.6},
                
                {'level': 12, 'name': 'scale_free_networks', 'n_vars': [18, 22],
                 'graph_types': ['scale_free'],
                 'm_parameter': 2, 'episodes': 150, 'f1_threshold': 0.55},
                
                # Level 13-15: Expert
                {'level': 13, 'name': 'very_large_sparse', 'n_vars': [25, 30],
                 'graph_types': ['erdos_renyi'],
                 'edge_prob': 0.05, 'episodes': 200, 'f1_threshold': 0.5},
                
                {'level': 14, 'name': 'very_large_mixed', 'n_vars': [25, 35],
                 'graph_types': ['erdos_renyi', 'small_world', 'scale_free'],
                 'edge_prob': 0.05, 'episodes': 200, 'f1_threshold': 0.5},
                
                {'level': 15, 'name': 'extreme_complexity', 'n_vars': [35, 50],
                 'graph_types': ['all'],
                 'edge_prob': 0.03, 'episodes': 250, 'f1_threshold': 0.45}
            ]
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of curriculum progress."""
        return {
            'current_level': self.current_level,
            'current_stage': self.current_stage.name,
            'total_episodes': self.global_episode_count,
            'stages_completed': self.current_level - self.start_level,
            'stage_metrics': {
                level: {
                    'episodes': metrics.episodes_completed,
                    'best_f1': metrics.best_f1_score,
                    'mean_f1': metrics.get_mean_f1(),
                    'mean_reward': metrics.get_mean_reward(),
                    'scms_seen': metrics.total_scms_seen
                }
                for level, metrics in self.stage_metrics.items()
                if metrics.episodes_completed > 0
            }
        }
    
    def reset(self, start_level: Optional[int] = None):
        """Reset curriculum to starting level."""
        self.current_level = start_level or self.start_level
        self.current_stage = self.stages[self.current_level - 1]
        self.global_episode_count = 0
        self.stage_metrics = {
            level: PerformanceMetrics() for level in range(1, len(self.stages) + 1)
        }
        logger.info(f"Reset curriculum to level {self.current_level}: {self.current_stage.name}")