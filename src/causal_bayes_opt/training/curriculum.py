#!/usr/bin/env python3
"""
Curriculum Learning Framework for ACBO

Implements progressive difficulty training with clear advancement criteria
and adaptive threshold management for different SCM complexities.

Key Features:
1. Define difficulty levels (difficulty_1, difficulty_2, ...)
2. Progressive SCM complexity (size, density, noise)
3. Advancement criteria based on F1 thresholds
4. Adaptive reward thresholds for each difficulty
5. SCM generation for curriculum stages

Design Principles:
- Pure functions for curriculum logic
- Immutable configuration objects
- Clear progression criteria
- Adaptive difficulty scaling
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union, Literal
from enum import Enum

import jax.numpy as jnp
import pyrsistent as pyr

from .config import TrainingConfig
from ..data_structures.scm import create_scm
from ..mechanisms.linear import create_linear_mechanism, create_root_mechanism

logger = logging.getLogger(__name__)


class DifficultyLevel(Enum):
    """Enumeration of curriculum difficulty levels."""
    DIFFICULTY_1 = 1  # Small graphs, low noise
    DIFFICULTY_2 = 2  # Medium graphs, moderate noise
    DIFFICULTY_3 = 3  # Large graphs, high noise
    DIFFICULTY_4 = 4  # Very large graphs, complex structures
    DIFFICULTY_5 = 5  # Expert level, real-world complexity


@dataclass(frozen=True)
class CurriculumStage:
    """Configuration for a single curriculum stage."""
    level: int
    name: str
    description: str
    
    # SCM characteristics
    n_variables_range: Tuple[int, int]
    n_edges_range: Tuple[int, int] 
    noise_scale_range: Tuple[float, float]
    mechanism_complexity: str  # "linear", "polynomial", "nonlinear"
    
    # Training requirements
    min_f1_score: float
    min_optimization_improvement: float
    max_training_steps: int
    
    # Advancement criteria
    advancement_threshold: float
    stability_window: int  # Number of steps to maintain performance


@dataclass(frozen=True)
class CurriculumConfig:
    """Complete curriculum configuration."""
    stages: List[CurriculumStage]
    adaptive_thresholds: bool = True
    early_termination: bool = True
    max_attempts_per_stage: int = 3


class CurriculumManager:
    """
    Manages curriculum learning progression for ACBO training.
    
    Provides SCM generation, advancement criteria checking,
    and adaptive threshold computation for each difficulty level.
    """
    
    def __init__(self, config: CurriculumConfig, random_seed: Optional[int] = None):
        """
        Initialize curriculum manager.
        
        Args:
            config: Curriculum configuration
            random_seed: Optional seed for reproducible SCM generation
        """
        self.config = config
        self.current_stage = 0
        self.stage_attempts = {}  # Track attempts per stage
        self.performance_history = {}  # Track performance across stages
        self.random_seed = random_seed
        
        # Initialize random state for reproducible generation
        if random_seed is not None:
            random.seed(random_seed)
        
        logger.info(f"Initialized curriculum with {len(config.stages)} stages, seed={random_seed}")
    
    def get_difficulty_sequence(self) -> List[int]:
        """Get sequence of difficulty levels to train on."""
        return [stage.level for stage in self.config.stages]
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        if self.current_stage >= len(self.config.stages):
            return self.config.stages[-1]  # Stay at final stage
        return self.config.stages[self.current_stage]
    
    def get_stage_by_difficulty(self, difficulty_level: int) -> CurriculumStage:
        """Get stage configuration for specific difficulty level."""
        for stage in self.config.stages:
            if stage.level == difficulty_level:
                return stage
        raise ValueError(f"No stage found for difficulty level {difficulty_level}")
    
    def get_scm_config_for_difficulty(self, difficulty_level: int) -> pyr.PMap:
        """
        Generate SCM configuration for given difficulty level.
        
        Args:
            difficulty_level: Curriculum difficulty (1-5)
            
        Returns:
            SCM configuration dict
        """
        stage = self.get_stage_by_difficulty(difficulty_level)
        
        # Select parameters based on difficulty
        n_vars = self._sample_from_range(stage.n_variables_range, return_type='int')
        n_edges = min(
            self._sample_from_range(stage.n_edges_range, return_type='int'),
            n_vars * (n_vars - 1) // 2  # Maximum possible edges
        )
        noise_scale = self._sample_from_range(stage.noise_scale_range, return_type='float')
        
        # Generate variable names
        variables = frozenset(f"X{i}" for i in range(n_vars))
        
        # Generate random DAG structure
        edges = self._generate_random_dag_edges(list(variables), n_edges)
        
        # Create mechanisms for each variable
        mechanisms = self._create_mechanisms_for_variables(
            variables, edges, noise_scale, stage.mechanism_complexity
        )
        
        # Select random target variable (must have at least one parent)
        target_candidates = [v for v in variables if any(e[1] == v for e in edges)]
        target_variable = target_candidates[0] if target_candidates else list(variables)[0]
        
        return pyr.m(
            variables=variables,
            edges=frozenset(edges),
            mechanisms=mechanisms,
            target=target_variable,
            difficulty_level=difficulty_level,
            noise_scale=noise_scale
        )
    
    def should_advance_difficulty(self, training_metrics: Dict[str, Any]) -> bool:
        """
        Check if current difficulty should advance to next level.
        
        Args:
            training_metrics: Training metrics from current stage
            
        Returns:
            True if advancement criteria are met
        """
        current_stage = self.get_current_stage()
        
        # Extract key metrics
        f1_score = training_metrics.get("final_f1_score", 0.0)
        optimization_improvement = training_metrics.get("optimization_improvement", 0.0)
        
        # Check primary advancement criteria
        meets_f1_threshold = f1_score >= current_stage.min_f1_score
        meets_optimization_threshold = optimization_improvement >= current_stage.min_optimization_improvement
        
        # Check stability if required
        stability_met = True
        if current_stage.stability_window > 0:
            recent_scores = training_metrics.get("recent_f1_scores", [])
            if len(recent_scores) >= current_stage.stability_window:
                stability_met = all(
                    score >= current_stage.advancement_threshold 
                    for score in recent_scores[-current_stage.stability_window:]
                )
        
        advancement_decision = meets_f1_threshold and meets_optimization_threshold and stability_met
        
        logger.info(
            f"Advancement check for difficulty {current_stage.level}: "
            f"F1={f1_score:.3f}>={current_stage.min_f1_score:.3f}, "
            f"Opt={optimization_improvement:.3f}>={current_stage.min_optimization_improvement:.3f}, "
            f"Stable={stability_met} → {'ADVANCE' if advancement_decision else 'CONTINUE'}"
        )
        
        if advancement_decision:
            self.current_stage += 1
            self.performance_history[current_stage.level] = training_metrics
        
        return advancement_decision
    
    def get_adaptive_reward_config(self, difficulty_level: int) -> Dict[str, Any]:
        """
        Get adaptive reward configuration for difficulty level.
        
        Args:
            difficulty_level: Curriculum difficulty level
            
        Returns:
            Adaptive reward configuration
        """
        from ..acquisition.rewards import create_adaptive_reward_config
        
        scm_config = self.get_scm_config_for_difficulty(difficulty_level)
        return create_adaptive_reward_config(scm_config, difficulty_level)
    
    def create_scm_for_difficulty(self, difficulty_level: str) -> pyr.PMap:
        """
        Create SCM for given difficulty level.
        
        Args:
            difficulty_level: Difficulty level as string (e.g., "difficulty_1")
            
        Returns:
            SCM as pyrsistent map
        """
        # Parse difficulty level from string
        if difficulty_level.startswith("difficulty_"):
            level_num = int(difficulty_level.split("_")[1])
        else:
            level_num = int(difficulty_level)
        
        # Get SCM configuration
        scm_config = self.get_scm_config_for_difficulty(level_num)
        
        # Convert config to actual SCM
        from ..data_structures.scm import create_scm
        
        return create_scm(
            variables=scm_config['variables'],
            edges=scm_config['edges'],
            mechanisms=scm_config['mechanisms'],
            target=scm_config['target']
        )
    
    def get_next_difficulty(self, current_difficulty: str) -> Optional[str]:
        """
        Get next difficulty level in curriculum.
        
        Args:
            current_difficulty: Current difficulty level string
            
        Returns:
            Next difficulty level string or None if at end
        """
        if current_difficulty.startswith("difficulty_"):
            level_num = int(current_difficulty.split("_")[1])
        else:
            level_num = int(current_difficulty)
        
        next_level = level_num + 1
        
        # Check if next level exists in curriculum
        if next_level <= len(self.config.stages):
            return f"difficulty_{next_level}"
        return None
    
    def _sample_from_range(
        self, 
        range_tuple: Tuple[Union[int, float], Union[int, float]], 
        return_type: Literal['int', 'float'] = 'int'
    ) -> Union[int, float]:
        """
        Sample randomly from range with specified return type.
        
        Args:
            range_tuple: (min, max) values for sampling
            return_type: 'int' for discrete, 'float' for continuous sampling
            
        Returns:
            Random value from range of specified type
            
        Raises:
            ValueError: If range is invalid (min > max)
        """
        min_val, max_val = range_tuple
        
        # Validate range
        if min_val > max_val:
            raise ValueError(f"Invalid range: min ({min_val}) > max ({max_val})")
        
        # Handle single-value ranges
        if min_val == max_val:
            return int(min_val) if return_type == 'int' else float(min_val)
        
        # Sample based on return type
        if return_type == 'int':
            # For integers, use randint (inclusive on both ends)
            return random.randint(int(min_val), int(max_val))
        elif return_type == 'float':
            # For floats, use uniform distribution
            return random.uniform(float(min_val), float(max_val))
        else:
            raise ValueError(f"Unsupported return_type: {return_type}. Use 'int' or 'float'.")
    
    def _generate_random_dag_edges(self, variables: List[str], n_edges: int) -> List[Tuple[str, str]]:
        """
        Generate random DAG edges ensuring no cycles.
        
        Args:
            variables: List of variable names
            n_edges: Number of edges to generate
            
        Returns:
            List of (parent, child) edge tuples
        """
        edges = []
        n_vars = len(variables)
        
        # Create topological ordering to ensure DAG property
        for i in range(min(n_edges, n_vars - 1)):
            parent_idx = i % (n_vars - 1)
            child_idx = (i + 1) % n_vars
            
            # Ensure child comes after parent in ordering
            if child_idx <= parent_idx:
                child_idx = parent_idx + 1
            
            if child_idx < n_vars:
                edges.append((variables[parent_idx], variables[child_idx]))
        
        return edges
    
    def _create_mechanisms_for_variables(
        self, 
        variables: frozenset, 
        edges: List[Tuple[str, str]], 
        noise_scale: float,
        complexity: str
    ) -> Dict[str, Any]:
        """
        Create mechanisms for all variables in SCM.
        
        Args:
            variables: Set of variable names
            edges: List of (parent, child) edges
            noise_scale: Noise scale for mechanisms
            complexity: Mechanism complexity level
            
        Returns:
            Dictionary mapping variables to mechanisms
        """
        mechanisms = {}
        
        # Build parent mapping
        parents_map = {}
        for var in variables:
            parents_map[var] = [parent for parent, child in edges if child == var]
        
        # Create mechanisms based on complexity
        for var in variables:
            parents = parents_map[var]
            
            if not parents:
                # Root variable (no parents)
                mechanisms[var] = create_root_mechanism(
                    mean=0.0, 
                    noise_scale=noise_scale
                )
            else:
                # Variable with parents
                if complexity == "linear":
                    # Simple linear combination
                    coefficients = {parent: 1.0 for parent in parents}
                    mechanisms[var] = create_linear_mechanism(
                        parents=parents,
                        coefficients=coefficients,
                        intercept=0.0,
                        noise_scale=noise_scale
                    )
                else:
                    # For now, fall back to linear (can extend later)
                    coefficients = {parent: 1.0 for parent in parents}
                    mechanisms[var] = create_linear_mechanism(
                        parents=parents,
                        coefficients=coefficients,
                        intercept=0.0,
                        noise_scale=noise_scale
                    )
        
        return mechanisms


def create_default_curriculum_config() -> CurriculumConfig:
    """Create default curriculum configuration for ACBO training."""
    stages = [
        CurriculumStage(
            level=1,
            name="Basic Structure Learning",
            description="Small graphs with clear structure",
            n_variables_range=(3, 5),
            n_edges_range=(2, 4),
            noise_scale_range=(0.1, 0.3),
            mechanism_complexity="linear",
            min_f1_score=0.8,
            min_optimization_improvement=0.2,
            max_training_steps=1000,
            advancement_threshold=0.8,
            stability_window=50
        ),
        CurriculumStage(
            level=2,
            name="Medium Complexity",
            description="Medium graphs with moderate noise",
            n_variables_range=(5, 8),
            n_edges_range=(4, 8),
            noise_scale_range=(0.3, 0.6),
            mechanism_complexity="linear",
            min_f1_score=0.7,
            min_optimization_improvement=0.15,
            max_training_steps=2000,
            advancement_threshold=0.7,
            stability_window=75
        ),
        CurriculumStage(
            level=3,
            name="Large Graphs",
            description="Large graphs with complex structure",
            n_variables_range=(8, 12),
            n_edges_range=(8, 15),
            noise_scale_range=(0.6, 1.0),
            mechanism_complexity="linear",
            min_f1_score=0.6,
            min_optimization_improvement=0.1,
            max_training_steps=3000,
            advancement_threshold=0.6,
            stability_window=100
        ),
        CurriculumStage(
            level=4,
            name="Very Large Graphs",
            description="Very large graphs approaching real-world complexity",
            n_variables_range=(12, 20),
            n_edges_range=(15, 30),
            noise_scale_range=(1.0, 1.5),
            mechanism_complexity="linear",
            min_f1_score=0.5,
            min_optimization_improvement=0.05,
            max_training_steps=4000,
            advancement_threshold=0.5,
            stability_window=150
        ),
        CurriculumStage(
            level=5,
            name="Expert Level",
            description="Real-world complexity with high noise",
            n_variables_range=(20, 30),
            n_edges_range=(30, 60),
            noise_scale_range=(1.5, 2.0),
            mechanism_complexity="linear",
            min_f1_score=0.4,
            min_optimization_improvement=0.02,
            max_training_steps=5000,
            advancement_threshold=0.4,
            stability_window=200
        )
    ]
    
    return CurriculumConfig(
        stages=stages,
        adaptive_thresholds=True,
        early_termination=True,
        max_attempts_per_stage=3
    )


def create_curriculum_manager(
    training_config: TrainingConfig,
    curriculum_config: Optional[CurriculumConfig] = None,
    random_seed: Optional[int] = None
) -> CurriculumManager:
    """
    Create curriculum manager from training configuration.
    
    Args:
        training_config: Overall training configuration
        curriculum_config: Optional curriculum override
        random_seed: Optional seed for reproducible SCM generation
        
    Returns:
        Initialized CurriculumManager
    """
    if curriculum_config is None:
        curriculum_config = create_default_curriculum_config()
    
    # Use training config random seed if not explicitly provided
    if random_seed is None:
        random_seed = getattr(training_config, 'random_seed', None)
    
    return CurriculumManager(curriculum_config, random_seed=random_seed)


def generate_curriculum_scms(difficulty_level: int, n_scms: int = 10) -> List[pyr.PMap]:
    """
    Generate multiple SCMs for a given curriculum difficulty level.
    
    Args:
        difficulty_level: Curriculum difficulty (1-5)
        n_scms: Number of SCMs to generate
        
    Returns:
        List of generated SCM configurations
    """
    config = create_default_curriculum_config()
    manager = CurriculumManager(config)
    
    scms = []
    for _ in range(n_scms):
        scm_config = manager.get_scm_config_for_difficulty(difficulty_level)
        scms.append(scm_config)
    
    return scms