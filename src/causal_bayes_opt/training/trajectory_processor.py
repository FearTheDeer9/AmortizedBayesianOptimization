#!/usr/bin/env python3
"""
Trajectory Processor for Behavioral Cloning

Pure functions for processing expert demonstration trajectories into
training data for behavioral cloning. Handles trajectory segmentation,
curriculum ordering, and data splitting.

Key Features:
1. Extract (state, action) pairs from expert trajectories
2. Curriculum ordering by complexity metrics
3. Train/validation/test splitting
4. Trajectory-level data transformations

Design Principles (Rich Hickey Approved):
- Pure functions with explicit inputs/outputs
- Immutable data transformations
- Simple composition of operations
- No side effects or hidden state
"""

import random
from typing import List, Dict, Any, Tuple, NamedTuple, Optional
from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp
import numpy as onp
import pyrsistent as pyr

from .expert_collection.data_structures import ExpertDemonstration
from .behavioral_cloning_adapter import (
    extract_avici_samples,
    extract_intervention_sequence,
    create_acquisition_state,
    extract_expert_action,
    create_surrogate_training_example,
    compute_demonstration_complexity
)
# Import TrainingExample from shared data structures
from .data_structures import TrainingExample
from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum learning."""
    EASY = 1      # 3-5 nodes, simple graphs
    MEDIUM = 2    # 6-10 nodes, moderate complexity
    HARD = 3      # 11-20 nodes, complex structures
    EXPERT = 4    # 21+ nodes, highly complex


@dataclass(frozen=True)
class TrajectoryStep:
    """Single step in an expert trajectory."""
    state: AcquisitionState
    action: Dict[str, Any]
    step_number: int
    demonstration_id: str


@dataclass(frozen=True)
class TrainValTestSplit:
    """Data split for training/validation/testing."""
    train: List[ExpertDemonstration]
    validation: List[ExpertDemonstration]
    test: List[ExpertDemonstration]


@dataclass(frozen=True)
class SurrogateDataset:
    """Dataset for surrogate model training."""
    training_examples: List[TrainingExample]
    difficulty_levels: List[DifficultyLevel]
    demonstration_ids: List[str]


@dataclass(frozen=True)
class AcquisitionDataset:
    """Dataset for acquisition policy training."""
    trajectory_steps: List[TrajectoryStep]
    difficulty_levels: List[DifficultyLevel]
    demonstration_ids: List[str]


@dataclass(frozen=True)
class CurriculumLevel:
    """Curriculum level with associated demonstrations."""
    level: DifficultyLevel
    demonstrations: List[ExpertDemonstration]
    complexity_range: Tuple[float, float]


def extract_trajectory_steps(demo: ExpertDemonstration, demo_id: str) -> List[TrajectoryStep]:
    """
    Pure function: Extract all trajectory steps from a demonstration.
    
    Args:
        demo: Expert demonstration
        demo_id: Unique identifier for this demonstration
        
    Returns:
        List of trajectory steps (state, action) pairs
    """
    try:
        # Extract basic data
        avici_data = extract_avici_samples(demo)
        intervention_sequence = extract_intervention_sequence(demo)
        
        trajectory_steps = []
        
        # Create trajectory steps for each intervention
        for step in range(len(intervention_sequence)):
            # Create state at this step
            state = create_acquisition_state(
                demo=demo,
                step=step,
                avici_data=avici_data,
                intervention_history=intervention_sequence
            )
            
            # Extract action taken at this step
            action = extract_expert_action(
                demo=demo,
                step=step,
                intervention_sequence=intervention_sequence
            )
            
            trajectory_step = TrajectoryStep(
                state=state,
                action=action,
                step_number=step,
                demonstration_id=demo_id
            )
            
            trajectory_steps.append(trajectory_step)
        
        return trajectory_steps
    
    except Exception as e:
        # Log error but don't fail completely
        print(f"Warning: Failed to extract trajectory steps from {demo_id}: {e}")
        return []


def extract_surrogate_examples(demo: ExpertDemonstration, demo_id: str) -> List[TrainingExample]:
    """
    Pure function: Extract surrogate training examples from demonstration.
    
    Args:
        demo: Expert demonstration
        demo_id: Unique identifier for this demonstration
        
    Returns:
        List of training examples for surrogate model
    """
    try:
        # Extract AVICI data once for efficiency
        avici_data = extract_avici_samples(demo)
        
        # Get posterior history to determine number of steps
        trajectory = demo.parent_posterior.get('trajectory', {})
        posterior_history = trajectory.get('posterior_history', [])
        
        training_examples = []
        
        # Create training example for each step with posterior data
        for step in range(len(posterior_history)):
            training_example = create_surrogate_training_example(
                demo=demo,
                step=step,
                avici_data=avici_data
            )
            training_examples.append(training_example)
        
        return training_examples
    
    except Exception as e:
        print(f"Warning: Failed to extract surrogate examples from {demo_id}: {e}")
        return []


def classify_difficulty(demo: ExpertDemonstration) -> DifficultyLevel:
    """
    Pure function: Classify demonstration difficulty level.
    
    Args:
        demo: Expert demonstration
        
    Returns:
        Difficulty level classification
    """
    complexity = compute_demonstration_complexity(demo)
    n_nodes = demo.n_nodes
    
    # Classification based on both complexity score and node count
    if n_nodes <= 5 and complexity < 5.0:
        return DifficultyLevel.EASY
    elif n_nodes <= 10 and complexity < 15.0:
        return DifficultyLevel.MEDIUM
    elif n_nodes <= 20 and complexity < 30.0:
        return DifficultyLevel.HARD
    else:
        return DifficultyLevel.EXPERT


def create_curriculum_ordering(demos: List[ExpertDemonstration]) -> List[CurriculumLevel]:
    """
    Pure function: Order demonstrations by curriculum difficulty.
    
    Args:
        demos: List of expert demonstrations
        
    Returns:
        List of curriculum levels with associated demonstrations
    """
    # Group demonstrations by difficulty level
    difficulty_groups = {level: [] for level in DifficultyLevel}
    complexity_ranges = {level: [float('inf'), float('-inf')] for level in DifficultyLevel}
    
    for demo in demos:
        difficulty = classify_difficulty(demo)
        complexity = compute_demonstration_complexity(demo)
        
        difficulty_groups[difficulty].append(demo)
        
        # Update complexity range for this level
        min_complexity, max_complexity = complexity_ranges[difficulty]
        complexity_ranges[difficulty] = [
            min(min_complexity, complexity),
            max(max_complexity, complexity)
        ]
    
    # Create curriculum levels
    curriculum_levels = []
    for level in DifficultyLevel:
        if difficulty_groups[level]:  # Only include levels with demonstrations
            min_comp, max_comp = complexity_ranges[level]
            if min_comp == float('inf'):  # No demonstrations for this level
                continue
                
            curriculum_level = CurriculumLevel(
                level=level,
                demonstrations=difficulty_groups[level],
                complexity_range=(min_comp, max_comp)
            )
            curriculum_levels.append(curriculum_level)
    
    return curriculum_levels


def split_demonstrations(
    demos: List[ExpertDemonstration],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_seed: int = 42
) -> TrainValTestSplit:
    """
    Pure function: Split demonstrations into train/validation/test sets.
    
    Args:
        demos: List of expert demonstrations
        ratios: (train, validation, test) split ratios
        random_seed: Seed for reproducible splitting
        
    Returns:
        TrainValTestSplit with data splits
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}")
    
    # Create reproducible random ordering
    rng = random.Random(random_seed)
    shuffled_demos = demos.copy()
    rng.shuffle(shuffled_demos)
    
    n_total = len(shuffled_demos)
    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    
    train_demos = shuffled_demos[:n_train]
    val_demos = shuffled_demos[n_train:n_train + n_val]
    test_demos = shuffled_demos[n_train + n_val:]
    
    return TrainValTestSplit(
        train=train_demos,
        validation=val_demos,
        test=test_demos
    )


def prepare_surrogate_dataset(
    demos: List[ExpertDemonstration],
    max_examples_per_demo: Optional[int] = None
) -> SurrogateDataset:
    """
    Pure function: Prepare dataset for surrogate model training.
    
    Args:
        demos: List of expert demonstrations
        max_examples_per_demo: Maximum examples to extract per demonstration
        
    Returns:
        SurrogateDataset ready for training
    """
    all_examples = []
    difficulty_levels = []
    demonstration_ids = []
    
    for i, demo in enumerate(demos):
        demo_id = f"demo_{i:04d}"
        examples = extract_surrogate_examples(demo, demo_id)
        
        # Limit examples per demo if specified
        if max_examples_per_demo is not None:
            examples = examples[:max_examples_per_demo]
        
        difficulty = classify_difficulty(demo)
        
        all_examples.extend(examples)
        difficulty_levels.extend([difficulty] * len(examples))
        demonstration_ids.extend([demo_id] * len(examples))
    
    return SurrogateDataset(
        training_examples=all_examples,
        difficulty_levels=difficulty_levels,
        demonstration_ids=demonstration_ids
    )


def prepare_acquisition_dataset(
    demos: List[ExpertDemonstration],
    max_steps_per_demo: Optional[int] = None
) -> AcquisitionDataset:
    """
    Pure function: Prepare dataset for acquisition policy training.
    
    Args:
        demos: List of expert demonstrations
        max_steps_per_demo: Maximum trajectory steps per demonstration
        
    Returns:
        AcquisitionDataset ready for training
    """
    all_steps = []
    difficulty_levels = []
    demonstration_ids = []
    
    for i, demo in enumerate(demos):
        demo_id = f"demo_{i:04d}"
        trajectory_steps = extract_trajectory_steps(demo, demo_id)
        
        # Limit steps per demo if specified
        if max_steps_per_demo is not None:
            trajectory_steps = trajectory_steps[:max_steps_per_demo]
        
        difficulty = classify_difficulty(demo)
        
        all_steps.extend(trajectory_steps)
        difficulty_levels.extend([difficulty] * len(trajectory_steps))
        demonstration_ids.extend([demo_id] * len(trajectory_steps))
    
    return AcquisitionDataset(
        trajectory_steps=all_steps,
        difficulty_levels=difficulty_levels,
        demonstration_ids=demonstration_ids
    )


def filter_by_difficulty(
    dataset: SurrogateDataset,
    target_levels: List[DifficultyLevel]
) -> SurrogateDataset:
    """
    Pure function: Filter surrogate dataset by difficulty levels.
    
    Args:
        dataset: Original surrogate dataset
        target_levels: Difficulty levels to include
        
    Returns:
        Filtered dataset with only target difficulty levels
    """
    filtered_examples = []
    filtered_levels = []
    filtered_ids = []
    
    for example, level, demo_id in zip(
        dataset.training_examples,
        dataset.difficulty_levels,
        dataset.demonstration_ids
    ):
        if level in target_levels:
            filtered_examples.append(example)
            filtered_levels.append(level)
            filtered_ids.append(demo_id)
    
    return SurrogateDataset(
        training_examples=filtered_examples,
        difficulty_levels=filtered_levels,
        demonstration_ids=filtered_ids
    )


def filter_acquisition_by_difficulty(
    dataset: AcquisitionDataset,
    target_levels: List[DifficultyLevel]
) -> AcquisitionDataset:
    """
    Pure function: Filter acquisition dataset by difficulty levels.
    
    Args:
        dataset: Original acquisition dataset
        target_levels: Difficulty levels to include
        
    Returns:
        Filtered dataset with only target difficulty levels
    """
    filtered_steps = []
    filtered_levels = []
    filtered_ids = []
    
    for step, level, demo_id in zip(
        dataset.trajectory_steps,
        dataset.difficulty_levels,
        dataset.demonstration_ids
    ):
        if level in target_levels:
            filtered_steps.append(step)
            filtered_levels.append(level)
            filtered_ids.append(demo_id)
    
    return AcquisitionDataset(
        trajectory_steps=filtered_steps,
        difficulty_levels=filtered_levels,
        demonstration_ids=filtered_ids
    )