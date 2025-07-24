#!/usr/bin/env python3
"""
Behavioral Cloning Curriculum Manager

Manages curriculum learning for behavioral cloning training, providing
structured progression through difficulty levels with adaptive thresholds
and performance-based advancement.

Key Features:
1. Multi-dimensional difficulty assessment
2. Adaptive curriculum stage management
3. Performance-based advancement criteria
4. Integration with existing curriculum infrastructure

Design Principles (Rich Hickey Approved):
- Pure functions for all assessments
- Immutable curriculum configuration
- Clear progression criteria
- Composable difficulty metrics
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import numpy as onp
import pyrsistent as pyr

from .trajectory_processor import DifficultyLevel
from .expert_collection.data_structures import ExpertDemonstration
from .behavioral_cloning_adapter import compute_demonstration_complexity
from .curriculum import CurriculumStage, CurriculumConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DifficultyMetrics:
    """Multi-dimensional difficulty metrics for demonstrations."""
    graph_complexity: float  # Based on nodes, edges, density
    convergence_difficulty: float  # Based on expert convergence speed
    posterior_uncertainty: float  # Based on posterior entropy
    intervention_complexity: float  # Based on intervention sequence
    overall_complexity: float  # Weighted combination


@dataclass(frozen=True)
class BCCurriculumStage:
    """Curriculum stage specifically for behavioral cloning."""
    level: DifficultyLevel
    demonstrations: List[ExpertDemonstration]
    difficulty_range: Tuple[float, float]
    advancement_criteria: Dict[str, float]
    stage_config: Dict[str, Any]


@dataclass(frozen=True)
class AdvancementCriteria:
    """Criteria for advancing to next curriculum stage."""
    min_surrogate_accuracy: float = 0.8  # KL divergence based
    min_acquisition_accuracy: float = 0.7  # Intervention choice accuracy
    min_epochs: int = 5
    max_epochs: int = 100
    patience: int = 10
    
    # Performance stability requirements
    stability_window: int = 5  # Number of epochs to check stability
    stability_threshold: float = 0.02  # Maximum variation in performance


class BCCurriculumManager:
    """Manager for behavioral cloning curriculum learning."""
    
    def __init__(
        self,
        difficulty_weights: Optional[Dict[str, float]] = None,
        advancement_criteria: Optional[AdvancementCriteria] = None
    ):
        """
        Initialize curriculum manager.
        
        Args:
            difficulty_weights: Weights for combining difficulty metrics
            advancement_criteria: Criteria for curriculum advancement
        """
        self.difficulty_weights = difficulty_weights or {
            'graph_complexity': 0.3,
            'convergence_difficulty': 0.2,
            'posterior_uncertainty': 0.2,
            'intervention_complexity': 0.3
        }
        
        self.advancement_criteria = advancement_criteria or AdvancementCriteria()
    
    def assess_demonstration_difficulty(self, demo: ExpertDemonstration) -> DifficultyMetrics:
        """
        Pure function: Assess multi-dimensional difficulty of demonstration.
        
        Args:
            demo: Expert demonstration to assess
            
        Returns:
            DifficultyMetrics with detailed breakdown
        """
        # Graph complexity based on structure
        graph_complexity = self._compute_graph_complexity(demo)
        
        # Convergence difficulty based on expert performance
        convergence_difficulty = self._compute_convergence_difficulty(demo)
        
        # Posterior uncertainty based on trajectory
        posterior_uncertainty = self._compute_posterior_uncertainty(demo)
        
        # Intervention complexity based on sequence
        intervention_complexity = self._compute_intervention_complexity(demo)
        
        # Overall weighted complexity
        overall_complexity = (
            self.difficulty_weights['graph_complexity'] * graph_complexity +
            self.difficulty_weights['convergence_difficulty'] * convergence_difficulty +
            self.difficulty_weights['posterior_uncertainty'] * posterior_uncertainty +
            self.difficulty_weights['intervention_complexity'] * intervention_complexity
        )
        
        return DifficultyMetrics(
            graph_complexity=graph_complexity,
            convergence_difficulty=convergence_difficulty,
            posterior_uncertainty=posterior_uncertainty,
            intervention_complexity=intervention_complexity,
            overall_complexity=overall_complexity
        )
    
    def create_bc_curriculum_stages(
        self,
        demonstrations: List[ExpertDemonstration]
    ) -> List[BCCurriculumStage]:
        """
        Pure function: Create curriculum stages from demonstrations.
        
        Args:
            demonstrations: List of expert demonstrations
            
        Returns:
            List of curriculum stages ordered by difficulty
        """
        if not demonstrations:
            return []
        
        # Assess difficulty for all demonstrations
        demo_difficulties = []
        for demo in demonstrations:
            difficulty = self.assess_demonstration_difficulty(demo)
            demo_difficulties.append((demo, difficulty))
        
        # Group by difficulty level
        level_groups = {level: [] for level in DifficultyLevel}
        level_complexities = {level: [] for level in DifficultyLevel}
        
        for demo, difficulty in demo_difficulties:
            level = self._classify_difficulty_level(difficulty)
            level_groups[level].append(demo)
            level_complexities[level].append(difficulty.overall_complexity)
        
        # Create curriculum stages
        curriculum_stages = []
        
        for level in DifficultyLevel:
            demos_for_level = level_groups[level]
            complexities = level_complexities[level]
            
            if not demos_for_level:
                continue  # Skip empty levels
            
            # Compute difficulty range for this level
            min_complexity = min(complexities)
            max_complexity = max(complexities)
            
            # Create advancement criteria for this level
            advancement_criteria = self._create_level_advancement_criteria(level)
            
            # Create stage configuration
            stage_config = self._create_stage_config(level, demos_for_level)
            
            stage = BCCurriculumStage(
                level=level,
                demonstrations=demos_for_level,
                difficulty_range=(min_complexity, max_complexity),
                advancement_criteria=advancement_criteria,
                stage_config=stage_config
            )
            
            curriculum_stages.append(stage)
        
        # Sort by difficulty level
        curriculum_stages.sort(key=lambda x: x.level.value)
        
        logger.info(f"Created {len(curriculum_stages)} curriculum stages")
        for stage in curriculum_stages:
            logger.info(f"Level {stage.level}: {len(stage.demonstrations)} demonstrations, "
                       f"complexity range {stage.difficulty_range[0]:.2f}-{stage.difficulty_range[1]:.2f}")
        
        return curriculum_stages
    
    def should_advance_curriculum(
        self,
        current_stage: BCCurriculumStage,
        training_metrics: List[Dict[str, float]],
        validation_metrics: List[Dict[str, float]]
    ) -> bool:
        """
        Pure function: Determine if should advance to next curriculum stage.
        
        Args:
            current_stage: Current curriculum stage
            training_metrics: Recent training metrics
            validation_metrics: Recent validation metrics
            
        Returns:
            True if should advance to next stage
        """
        if not validation_metrics:
            return False
        
        latest_val_metrics = validation_metrics[-1]
        criteria = current_stage.advancement_criteria
        
        # Check minimum accuracy requirements
        surrogate_accuracy = latest_val_metrics.get('surrogate_accuracy', 0.0)
        acquisition_accuracy = latest_val_metrics.get('acquisition_accuracy', 0.0)
        
        accuracy_met = (
            surrogate_accuracy >= criteria.get('min_surrogate_accuracy', 0.8) and
            acquisition_accuracy >= criteria.get('min_acquisition_accuracy', 0.7)
        )
        
        # Check minimum epochs requirement
        epochs_completed = len(training_metrics)
        min_epochs_met = epochs_completed >= criteria.get('min_epochs', 5)
        
        # Check performance stability
        stability_met = self._check_performance_stability(
            validation_metrics,
            criteria.get('stability_window', 5),
            criteria.get('stability_threshold', 0.02)
        )
        
        advancement_decision = accuracy_met and min_epochs_met and stability_met
        
        if advancement_decision:
            logger.info(f"Advancement criteria met for level {current_stage.level}")
            logger.info(f"Surrogate accuracy: {surrogate_accuracy:.3f}, "
                       f"Acquisition accuracy: {acquisition_accuracy:.3f}")
        
        return advancement_decision
    
    def _compute_graph_complexity(self, demo: ExpertDemonstration) -> float:
        """Compute graph structural complexity."""
        n_nodes = demo.n_nodes
        
        # Extract edges from SCM
        from ..data_structures.scm import get_edges
        edges = get_edges(demo.scm)
        n_edges = len(edges)
        
        # Edge density
        max_edges = n_nodes * (n_nodes - 1)
        edge_density = n_edges / max_edges if max_edges > 0 else 0
        
        # Graph type complexity weights
        graph_type_weights = {
            'chain': 1.0,
            'fork': 1.2,
            'collider': 1.5,
            'erdos_renyi': 2.0,
            'scale_free': 2.5,
            'dense': 3.0
        }
        
        type_weight = graph_type_weights.get(demo.graph_type, 1.5)
        
        # Combined graph complexity
        complexity = (
            (n_nodes / 30.0) * 0.4 +  # Node count normalized to max expected
            edge_density * 0.4 +      # Edge density contribution
            (type_weight / 3.0) * 0.2 # Graph type contribution
        )
        
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _compute_convergence_difficulty(self, demo: ExpertDemonstration) -> float:
        """Compute difficulty based on expert convergence behavior."""
        trajectory = demo.parent_posterior.get('trajectory', {})
        
        # Number of iterations needed
        n_iterations = trajectory.get('iterations', 1)
        max_expected_iterations = 20
        iteration_difficulty = min(n_iterations / max_expected_iterations, 1.0)
        
        # Convergence rate (if available)
        convergence_rate = trajectory.get('convergence_rate', 0.5)
        convergence_difficulty = 1.0 - convergence_rate  # Lower rate = higher difficulty
        
        # Expert accuracy (inverse difficulty)
        accuracy_difficulty = 1.0 - demo.accuracy
        
        # Combined convergence difficulty
        difficulty = (
            iteration_difficulty * 0.4 +
            convergence_difficulty * 0.3 +
            accuracy_difficulty * 0.3
        )
        
        return min(difficulty, 1.0)
    
    def _compute_posterior_uncertainty(self, demo: ExpertDemonstration) -> float:
        """Compute difficulty based on posterior uncertainty trajectory."""
        posterior_history = demo.parent_posterior.get('posterior_history', [])
        
        if not posterior_history:
            return 0.5  # Default moderate uncertainty
        
        # Compute entropy for each step
        entropies = []
        for step in posterior_history:
            posterior = step.get('posterior', {})
            if posterior:
                probs = list(posterior.values())
                probs = [p for p in probs if p > 0]  # Remove zero probabilities
                if probs:
                    # Normalize probabilities
                    prob_sum = sum(probs)
                    probs = [p / prob_sum for p in probs]
                    # Compute entropy
                    entropy = -sum(p * onp.log(p) for p in probs)
                    entropies.append(entropy)
        
        if not entropies:
            return 0.5
        
        # Average entropy normalized by log(max_parent_sets)
        avg_entropy = onp.mean(entropies)
        max_entropy = onp.log(len(posterior_history[0].get('posterior', {})))
        normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.5
        
        return min(normalized_entropy, 1.0)
    
    def _compute_intervention_complexity(self, demo: ExpertDemonstration) -> float:
        """Compute difficulty based on intervention sequence complexity."""
        trajectory = demo.parent_posterior.get('trajectory', {})
        
        intervention_sequence = trajectory.get('intervention_sequence', [])
        intervention_values = trajectory.get('intervention_values', [])
        
        if not intervention_sequence:
            return 0.5
        
        # Sequence length complexity
        sequence_length = len(intervention_sequence)
        max_expected_length = 15
        length_complexity = min(sequence_length / max_expected_length, 1.0)
        
        # Intervention diversity (number of unique variables intervened on)
        all_vars = set()
        for intervention in intervention_sequence:
            if isinstance(intervention, (tuple, list)):
                all_vars.update(intervention)
            else:
                all_vars.add(intervention)
        
        diversity = len(all_vars) / demo.n_nodes if demo.n_nodes > 0 else 0
        
        # Value range complexity (standard deviation of intervention values)
        if intervention_values:
            flat_values = []
            for vals in intervention_values:
                if isinstance(vals, (tuple, list)):
                    flat_values.extend(vals)
                else:
                    flat_values.append(vals)
            
            if len(flat_values) > 1:
                value_std = onp.std(flat_values)
                value_complexity = min(value_std / 2.0, 1.0)  # Normalize by typical std
            else:
                value_complexity = 0.1
        else:
            value_complexity = 0.1
        
        # Combined intervention complexity
        complexity = (
            length_complexity * 0.4 +
            diversity * 0.3 +
            value_complexity * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _classify_difficulty_level(self, difficulty: DifficultyMetrics) -> DifficultyLevel:
        """Classify overall difficulty into discrete levels."""
        overall = difficulty.overall_complexity
        
        if overall < 0.25:
            return DifficultyLevel.EASY
        elif overall < 0.5:
            return DifficultyLevel.MEDIUM
        elif overall < 0.75:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT
    
    def _create_level_advancement_criteria(self, level: DifficultyLevel) -> Dict[str, float]:
        """Create advancement criteria specific to difficulty level."""
        base_criteria = {
            'min_surrogate_accuracy': self.advancement_criteria.min_surrogate_accuracy,
            'min_acquisition_accuracy': self.advancement_criteria.min_acquisition_accuracy,
            'min_epochs': self.advancement_criteria.min_epochs,
            'max_epochs': self.advancement_criteria.max_epochs,
            'stability_window': self.advancement_criteria.stability_window,
            'stability_threshold': self.advancement_criteria.stability_threshold
        }
        
        # Adjust criteria based on difficulty level
        if level == DifficultyLevel.EASY:
            base_criteria['min_surrogate_accuracy'] = 0.9
            base_criteria['min_acquisition_accuracy'] = 0.8
            base_criteria['min_epochs'] = 3
        elif level == DifficultyLevel.MEDIUM:
            base_criteria['min_surrogate_accuracy'] = 0.8
            base_criteria['min_acquisition_accuracy'] = 0.7
            base_criteria['min_epochs'] = 5
        elif level == DifficultyLevel.HARD:
            base_criteria['min_surrogate_accuracy'] = 0.7
            base_criteria['min_acquisition_accuracy'] = 0.6
            base_criteria['min_epochs'] = 8
        else:  # EXPERT
            base_criteria['min_surrogate_accuracy'] = 0.6
            base_criteria['min_acquisition_accuracy'] = 0.5
            base_criteria['min_epochs'] = 10
        
        return base_criteria
    
    def _create_stage_config(
        self,
        level: DifficultyLevel,
        demonstrations: List[ExpertDemonstration]
    ) -> Dict[str, Any]:
        """Create configuration for curriculum stage."""
        return {
            'level': level.name,
            'n_demonstrations': len(demonstrations),
            'recommended_batch_size': min(32, max(8, len(demonstrations) // 10)),
            'recommended_learning_rate': {
                DifficultyLevel.EASY: 1e-3,
                DifficultyLevel.MEDIUM: 8e-4,
                DifficultyLevel.HARD: 5e-4,
                DifficultyLevel.EXPERT: 3e-4
            }.get(level, 1e-3)
        }
    
    def _check_performance_stability(
        self,
        metrics: List[Dict[str, float]],
        stability_window: int,
        stability_threshold: float
    ) -> bool:
        """Check if performance is stable over recent window."""
        if len(metrics) < stability_window:
            return False
        
        recent_metrics = metrics[-stability_window:]
        
        # Check stability for key metrics
        key_metrics = ['surrogate_accuracy', 'acquisition_accuracy', 'loss']
        
        for metric_name in key_metrics:
            values = [m.get(metric_name, 0.0) for m in recent_metrics]
            if values and len(set(values)) > 1:  # Not all identical
                variation = onp.std(values) / (onp.mean(values) + 1e-8)
                if variation > stability_threshold:
                    return False
        
        return True


def create_bc_curriculum_manager(
    target_levels: Optional[List[DifficultyLevel]] = None,
    difficulty_weights: Optional[Dict[str, float]] = None,
    advancement_criteria: Optional[AdvancementCriteria] = None
) -> BCCurriculumManager:
    """
    Factory function to create BC curriculum manager.
    
    Args:
        target_levels: Specific difficulty levels to include
        difficulty_weights: Custom weights for difficulty assessment
        advancement_criteria: Custom advancement criteria
        
    Returns:
        Configured BCCurriculumManager
    """
    return BCCurriculumManager(
        difficulty_weights=difficulty_weights,
        advancement_criteria=advancement_criteria
    )